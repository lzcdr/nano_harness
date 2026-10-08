// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/local_storage.rs
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hnsw_rs::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use tokenizers::Tokenizer;

const STORAGE_BASE: &str = ".local_storage";
const META_FILE: &str = "vector_meta.json";
const RESERVED_NAMES: &[&str] = &[
    ".about",
    ".summary",
    ".skills",
    ".rebukes",
    ".knowledge",
    ".tools",
    META_FILE,
];
pub const SKILLS_PROJECT: &str = "_skills";
pub const KNOWLEDGE_PROJECT: &str = "_knowledge";
pub const TOOLS_PROJECT: &str = "_tools";
const PROJECTS_DIR: &str = "projects";

pub fn is_reserved_name(name: &str) -> bool {
    RESERVED_NAMES.iter().any(|r| name.eq_ignore_ascii_case(r))
}

pub fn valid_project_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[derive(Debug, Clone, Deserialize)]
pub struct VectorDbConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            model_path: ".models/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            tokenizer_path: ".models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"
                .to_string(),
            chunk_size: 512,
            chunk_overlap: 64,
            top_k: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub file_path: PathBuf,
    pub chunk_index: u32,
    pub start_byte: u64,
    pub end_byte: u64,
    pub content: String,
    pub embedding: Vec<f32>,
    pub file_size: u64,
    pub modified: u64,
    #[serde(default)]
    pub project_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub file_path: PathBuf,
    pub chunk_index: u32,
    pub distance: f32,
    pub content_fragment: String,
    pub project_id: String,
}

struct DiskEntry {
    project_id: String,
    rel_path: PathBuf,
    mtime: u64,
    size: u64,
}

fn make_disk_entry(path: &Path, project_id: String, rel_path: PathBuf) -> Option<DiskEntry> {
    let metadata = fs::metadata(path).ok()?;
    if metadata.len() == 0 {
        return None;
    }
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(DiskEntry {
        project_id,
        rel_path,
        mtime,
        size: metadata.len(),
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Entry {
    File {
        name: String,
        path: PathBuf,
        size: u64,
    },
    Dir {
        name: String,
        path: PathBuf,
    },
}

struct VectorDb {
    root: PathBuf,
    config: VectorDbConfig,
    bert: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hnsw: Hnsw<'static, f32, DistCosine>,
    entries: Vec<(String, ChunkMeta)>,
    label_to_id: HashMap<String, usize>,
    id_to_label: HashMap<usize, String>,
    id_to_meta: HashMap<usize, ChunkMeta>,
    deleted_ids: HashSet<usize>,
    next_id: usize,
}

impl VectorDb {
    fn new(config: VectorDbConfig, root: PathBuf) -> io::Result<Self> {
        eprintln!("🧠 Загрузка модели эмбеддингов...");
        let start = std::time::Instant::now();
        let device = Device::Cpu;
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Tokenizer error: {e}"))
        })?;
        let bert_config_path = Path::new(&config.model_path).join("config.json");
        let bert_config_str = std::fs::read_to_string(&bert_config_path)?;
        let bert_config: Config = serde_json::from_str(&bert_config_str).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Config error: {e}"))
        })?;
        let weights_path = Path::new(&config.model_path).join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Weights error: {e}"))
            })?
        };
        let bert = BertModel::load(vb, &bert_config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Model error: {e}")))?;
        let hnsw: Hnsw<'static, f32, DistCosine> = Hnsw::new(16, 100_000, 32, 200, DistCosine);
        eprintln!("✅ Модель загружена за {:?}", start.elapsed());
        Ok(Self {
            root,
            config,
            bert,
            tokenizer,
            device,
            hnsw,
            entries: Vec::new(),
            label_to_id: HashMap::new(),
            id_to_label: HashMap::new(),
            id_to_meta: HashMap::new(),
            deleted_ids: HashSet::new(),
            next_id: 0,
        })
    }

    fn save(&self, root: &Path) -> io::Result<()> {
        let meta_path = root.join(META_FILE);
        fs::write(meta_path, serde_json::to_string(&self.entries)?)?;
        Ok(())
    }

    fn load(root: &Path, config: VectorDbConfig) -> io::Result<Self> {
        eprintln!("📂 Загрузка индекса...");
        let start = std::time::Instant::now();
        let mut db = Self::new(config, root.to_path_buf())?;
        let meta_path = root.join(META_FILE);
        if meta_path.exists() {
            let meta_json = fs::read_to_string(&meta_path)?;
            let entries: Vec<(String, ChunkMeta)> = serde_json::from_str(&meta_json)?;
            let mut valid_entries = Vec::new();
            for (label, meta) in entries {
                let file_path = root.join(&meta.file_path);
                let modified_ok = match fs::metadata(&file_path) {
                    Ok(metadata) => {
                        let modified = metadata
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        modified == meta.modified && metadata.len() == meta.file_size
                    }
                    Err(_) => false,
                };
                if !modified_ok {
                    eprintln!(
                        "⚠️ Файл {} изменён, будет переиндексирован",
                        meta.file_path.display()
                    );
                    continue;
                }
                let emb = meta.embedding.clone();
                let id = db.next_id;
                db.next_id += 1;
                db.hnsw.insert((&emb, id));
                db.label_to_id.insert(label.clone(), id);
                db.id_to_label.insert(id, label.clone());
                valid_entries.push((label, meta));
            }
            db.entries = valid_entries;

            for (label, meta) in &db.entries {
                if let Some(&id) = db.label_to_id.get(label) {
                    db.id_to_meta.insert(id, meta.clone());
                }
            }
        }
        eprintln!(
            "✅ Индекс загружен за {:?}, записей: {}",
            start.elapsed(),
            db.entries.len()
        );
        Ok(db)
    }

    fn embed_batch(&self, texts: &[String]) -> io::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut token_ids_list = Vec::with_capacity(texts.len());
        let mut max_len = 0;
        for text in texts {
            let tokens = self.tokenizer.encode(text.as_str(), true).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Tokenize error: {e}"))
            })?;
            let ids = tokens.get_ids().to_vec();
            max_len = max_len.max(ids.len());
            token_ids_list.push(ids);
        }
        let batch_size = token_ids_list.len();
        let mut padded_ids = vec![0u32; batch_size * max_len];
        let mut mask = vec![0u32; batch_size * max_len];
        for (i, ids) in token_ids_list.iter().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                padded_ids[i * max_len + j] = id;
                mask[i * max_len + j] = 1;
            }
        }
        let input_ids =
            Tensor::from_vec(padded_ids, (batch_size, max_len), &self.device).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Tensor error: {e}"))
            })?;
        let attention_mask =
            Tensor::from_vec(mask, (batch_size, max_len), &self.device).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Tensor error: {e}"))
            })?;
        let token_type_ids =
            Tensor::zeros((batch_size, max_len), candle_core::DType::U32, &self.device).map_err(
                |e| io::Error::new(io::ErrorKind::InvalidData, format!("Tensor error: {e}")),
            )?;

        let outputs = self
            .bert
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Model forward error: {e}"),
                )
            })?;

        let attention_mask_f32 = attention_mask
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("To dtype error: {e}"))
            })?;
        let mask_expanded = attention_mask_f32
            .unsqueeze(2)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Unsqueeze error: {e}"))
            })?
            .broadcast_as(outputs.shape())
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Broadcast error: {e}"))
            })?;
        let masked = outputs
            .mul(&mask_expanded)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Mul error: {e}")))?;
        let sum_hidden = masked
            .sum(1)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Sum error: {e}")))?;
        let sum_mask = attention_mask_f32
            .sum(1)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Sum error: {e}")))?
            .unsqueeze(1)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Unsqueeze error: {e}"))
            })?;
        let sum_mask_b = sum_mask.broadcast_as(sum_hidden.shape()).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Broadcast error: {e}"))
        })?;
        let mean_hidden = sum_hidden
            .div(&sum_mask_b)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Div error: {e}")))?;

        let norm = mean_hidden
            .sqr()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Sqr error: {e}")))?
            .sum(1)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Sum error: {e}")))?
            .sqrt()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Sqrt error: {e}")))?;
        let norm_expanded = norm.unsqueeze(1).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Unsqueeze error: {e}"))
        })?;
        let norm_expanded_b = norm_expanded
            .broadcast_as(mean_hidden.shape())
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Broadcast error: {e}"))
            })?;
        let normalized = mean_hidden
            .div(&norm_expanded_b)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Div error: {e}")))?;

        Ok(normalized.to_vec2::<f32>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("To vec error: {e}"))
        })?)
    }

    fn embed_single(&self, text: &str) -> io::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    fn index_text(&mut self, project_id: &str, rel_path: &Path, content: &str) -> io::Result<()> {
        let storage_rel = match project_id {
            SKILLS_PROJECT => Path::new(".skills").join(rel_path),
            KNOWLEDGE_PROJECT => Path::new(".knowledge").join(rel_path),
            TOOLS_PROJECT => Path::new(".tools").join(rel_path),
            _ => Path::new(PROJECTS_DIR).join(project_id).join(rel_path),
        };
        let label_prefix = format!("{}#", storage_rel.to_string_lossy());
        let mut old_labels = Vec::new();
        for (label, _) in &self.entries {
            if label.starts_with(&label_prefix) {
                old_labels.push(label.clone());
            }
        }
        for label in old_labels {
            if let Some(id) = self.label_to_id.remove(&label) {
                self.deleted_ids.insert(id);
                self.id_to_label.remove(&id);
            }
            self.entries.retain(|(l, _)| l != &label);
        }

        let chunks_with_pos =
            chunk_text(content, self.config.chunk_size, self.config.chunk_overlap);
        if chunks_with_pos.is_empty() {
            return Ok(());
        }
        let chunk_texts: Vec<String> = chunks_with_pos
            .iter()
            .map(|(text, _, _)| text.clone())
            .collect();
        eprintln!(
            "🔄 Индексация файла {} ({} чанков)...",
            storage_rel.display(),
            chunk_texts.len()
        );

        let start = std::time::Instant::now();
        let embeddings = self.embed_batch(&chunk_texts)?;
        let full_path = self.root.join(&storage_rel);
        let (file_size, modified) = match fs::metadata(&full_path) {
            Ok(metadata) => {
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (metadata.len(), modified)
            }
            Err(_) => (0, 0),
        };

        for (i, ((chunk_text, start_byte, end_byte), emb)) in chunks_with_pos
            .into_iter()
            .zip(embeddings.into_iter())
            .enumerate()
        {
            let label = format!("{}#{}", storage_rel.to_string_lossy(), i);
            let id = self.next_id;
            self.next_id += 1;
            self.hnsw.insert((&emb, id));
            self.label_to_id.insert(label.clone(), id);
            self.id_to_label.insert(id, label.clone());
            let meta = ChunkMeta {
                file_path: storage_rel.clone(),
                chunk_index: i as u32,
                start_byte,
                end_byte,
                content: chunk_text,
                embedding: emb,
                file_size,
                modified,
                project_id: project_id.to_string(),
            };
            self.entries.push((label.clone(), meta.clone()));
            self.id_to_meta.insert(id, meta);
        }
        eprintln!("✅ Индексация файла завершена за {:?}", start.elapsed());
        Ok(())
    }

    fn search(
        &self,
        project_id: Option<&str>,
        query: &str,
        top_k: usize,
    ) -> io::Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();
        let query_embedding = self.embed_single(query)?;

        let mut ef = (top_k * 8).max(64);
        let max_ef = self.entries.len().max(64);
        if ef > max_ef {
            ef = max_ef;
        }

        let neighbours = self.hnsw.search(&query_embedding, ef, ef);
        let mut results = Vec::new();
        for neighbour in neighbours {
            let id = neighbour.d_id;
            if self.deleted_ids.contains(&id) {
                continue;
            }
            let Some(meta) = self.id_to_meta.get(&id) else {
                continue;
            };
            if let Some(pid) = project_id {
                if meta.project_id != pid {
                    continue;
                }
            }
            let user_path = if meta.project_id == SKILLS_PROJECT {
                meta.file_path.clone()
            } else {
                let prefix = Path::new(PROJECTS_DIR).join(&meta.project_id);
                meta.file_path
                    .strip_prefix(&prefix)
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|_| meta.file_path.clone())
            };
            results.push(SearchResult {
                file_path: user_path,
                chunk_index: meta.chunk_index,
                distance: neighbour.distance,
                content_fragment: meta.content.clone(),
                project_id: meta.project_id.clone(),
            });
            if results.len() >= top_k {
                break;
            }
        }
        eprintln!(
            "🔍 Поиск завершён за {:?}, найдено {}",
            start.elapsed(),
            results.len()
        );
        Ok(results)
    }

    fn rebuild(&mut self, root: &Path) -> io::Result<()> {
        eprintln!("🔄 Полная переиндексация...");
        let start = std::time::Instant::now();
        self.entries.clear();
        self.label_to_id.clear();
        self.id_to_label.clear();
        self.id_to_meta.clear();
        self.deleted_ids.clear();
        self.next_id = 0;
        self.hnsw = Hnsw::new(16, 100_000, 32, 200, DistCosine);

        let projects_root = root.join(PROJECTS_DIR);
        if projects_root.exists() {
            let mut files = Vec::new();
            collect_files(&projects_root, &mut files)?;
            for file in files {
                let rel_to_root = match file.strip_prefix(root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                let mut comps = rel_to_root.components();
                if comps.next().map(|c| c.as_os_str()) != Some(std::ffi::OsStr::new(PROJECTS_DIR)) {
                    continue;
                }
                let project_id = match comps.next().and_then(|c| c.as_os_str().to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let user_rel: PathBuf = comps.collect();
                if user_rel.as_os_str().is_empty() {
                    continue;
                }
                if let Ok(content) = fs::read_to_string(&file) {
                    self.index_text(&project_id, &user_rel, &content)?;
                }
            }
        }

        let skills_root = root.join(".skills");
        if skills_root.exists() {
            let mut files = Vec::new();
            collect_files(&skills_root, &mut files)?;
            for file in files {
                let rel = match file.strip_prefix(&skills_root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                if rel.as_os_str().is_empty() {
                    continue;
                }
                if let Ok(content) = fs::read_to_string(&file) {
                    self.index_text(SKILLS_PROJECT, &rel, &content)?;
                }
            }
        }

        let knowledge_root = root.join(".knowledge");
        if knowledge_root.exists() {
            for entry in fs::read_dir(&knowledge_root)? {
                let entry = entry?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.ends_with(".md") {
                    continue;
                }
                if let Ok(content) = fs::read_to_string(&path) {
                    self.index_text(KNOWLEDGE_PROJECT, Path::new(&name), &content)?;
                }
            }
        }

        let tools_root = root.join(".tools");
        if tools_root.exists() {
            let mut files = Vec::new();
            collect_files(&tools_root, &mut files)?;
            for file in files {
                let rel = match file.strip_prefix(&tools_root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                if rel.as_os_str().is_empty() {
                    continue;
                }
                if let Ok(content) = fs::read_to_string(&file) {
                    self.index_text(TOOLS_PROJECT, &rel, &content)?;
                }
            }
        }

        eprintln!(
            "✅ Полная переиндексация завершена за {:?}",
            start.elapsed()
        );
        Ok(())
    }
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<(String, u64, u64)> {
    if text.is_empty() {
        return vec![];
    }

    if text.len() <= chunk_size {
        return vec![(text.to_string(), 0, text.len() as u64)];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }

    let mut positions: Vec<u64> = Vec::with_capacity(words.len());
    let mut current_offset = 0;
    for word in &words {
        if let Some(pos) = text[current_offset..].find(word) {
            positions.push((current_offset + pos) as u64);
            current_offset += pos + word.len();
        } else {
            positions.push(current_offset as u64);
            current_offset += word.len();
        }
    }

    let mut chunks = Vec::new();
    let mut start_idx = 0;

    while start_idx < words.len() {
        let mut end_idx = start_idx;
        let mut current_len = 0;

        while end_idx < words.len() {
            let word_len = words[end_idx].len();
            let add_len = if current_len == 0 {
                word_len
            } else {
                current_len + word_len + 1
            };

            if add_len > chunk_size && end_idx > start_idx {
                break;
            }
            current_len = add_len;
            end_idx += 1;
        }

        if end_idx == start_idx {
            end_idx = start_idx + 1;
        }

        let chunk = words[start_idx..end_idx].join(" ");
        let start_byte = positions[start_idx];
        let end_byte = positions[end_idx - 1] + words[end_idx - 1].len() as u64;
        chunks.push((chunk, start_byte, end_byte));

        if end_idx >= words.len() {
            break;
        }

        let mut overlap_len = 0;
        let mut new_start_idx = end_idx;

        while new_start_idx > start_idx {
            new_start_idx -= 1;
            let word_len_with_space = if new_start_idx == end_idx - 1 {
                words[new_start_idx].len()
            } else {
                words[new_start_idx].len() + 1
            };
            overlap_len += word_len_with_space;
            if overlap_len >= overlap {
                break;
            }
        }

        if new_start_idx <= start_idx {
            new_start_idx = start_idx + 1;
        }

        start_idx = new_start_idx;
    }
    chunks
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name == META_FILE {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

pub struct LocalStorage {
    root: PathBuf,
    vector_db: VectorDb,
}

impl LocalStorage {
    pub fn new(storage_name: &str, vector_db_config: VectorDbConfig) -> io::Result<Self> {
        let base = PathBuf::from(STORAGE_BASE);
        if !base.exists() {
            fs::create_dir_all(&base)?;
        }
        if storage_name.is_empty()
            || storage_name.contains('/')
            || storage_name.contains('\\')
            || storage_name.contains("..")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "storage_name must be a simple directory name",
            ));
        }
        let root = base.join(storage_name);
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }
        let projects_root = root.join(PROJECTS_DIR);
        if !projects_root.exists() {
            fs::create_dir_all(&projects_root)?;
        }
        let mut vector_db = VectorDb::load(&root, vector_db_config)?;
        if !root.join(META_FILE).exists() {
            vector_db.rebuild(&root)?;
            vector_db.save(&root)?;
        }
        Ok(Self { root, vector_db })
    }

    pub fn ensure_project(&self, project_id: &str) -> io::Result<()> {
        if !valid_project_id(project_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid project_id",
            ));
        }
        let dir = self.root.join(PROJECTS_DIR).join(project_id);
        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }
        Self::ensure_system_files(&dir)?;
        Ok(())
    }

    fn resolve(&self, project_id: &str, path: &str) -> io::Result<PathBuf> {
        if !valid_project_id(project_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid project_id",
            ));
        }
        let mut out = PathBuf::from(PROJECTS_DIR).join(project_id);
        for c in Path::new(path).components() {
            match c {
                std::path::Component::Normal(p) => {
                    if p.to_str().map(is_reserved_name) == Some(true) {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "path refers to reserved file",
                        ));
                    }
                    out.push(p)
                }
                std::path::Component::CurDir => {}
                std::path::Component::RootDir => {}
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "path escapes storage root",
                    ));
                }
            }
        }
        Ok(self.root.join(out))
    }

    fn ensure_system_files(dir: &Path) -> io::Result<()> {
        let about = dir.join(".about");
        if !about.exists() {
            fs::write(&about, "")?;
        }
        let summary = dir.join(".summary");
        if !summary.exists() {
            fs::write(&summary, "")?;
        }
        Ok(())
    }

    pub fn create_dir(&mut self, project_id: &str, path: &str) -> io::Result<()> {
        self.ensure_project(project_id)?;
        let full = self.resolve(project_id, path)?;
        fs::create_dir_all(&full)?;
        Self::ensure_system_files(&full)
    }

    pub fn create_file(&mut self, project_id: &str, path: &str, content: &str) -> io::Result<()> {
        self.ensure_project(project_id)?;
        let full = self.resolve(project_id, path)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
            Self::ensure_system_files(parent)?;
        }
        fs::write(&full, content)?;
        let rel_path = Path::new(path);
        self.vector_db.index_text(project_id, rel_path, content)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn write_file(&mut self, project_id: &str, path: &str, content: &str) -> io::Result<()> {
        self.create_file(project_id, path, content)
    }

    pub fn create_skill_file(&mut self, rel_path: &str, content: &str) -> io::Result<()> {
        let rel = Path::new(rel_path.trim_start_matches('/'));
        self.vector_db.index_text(SKILLS_PROJECT, rel, content)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    fn remove_index_entries_for_file(&mut self, storage_rel: &Path) {
        let label_prefix = format!("{}#", storage_rel.to_string_lossy());
        let old_labels: Vec<String> = self
            .vector_db
            .entries
            .iter()
            .filter(|(label, _)| label.starts_with(&label_prefix))
            .map(|(label, _)| label.clone())
            .collect();
        for label in old_labels {
            if let Some(id) = self.vector_db.label_to_id.remove(&label) {
                self.vector_db.deleted_ids.insert(id);
                self.vector_db.id_to_label.remove(&id);
                self.vector_db.id_to_meta.remove(&id);
            }
            self.vector_db.entries.retain(|(l, _)| l != &label);
        }
    }

    pub fn delete_skill_index(&mut self, rel_path: &str) -> io::Result<()> {
        let rel = Path::new(rel_path.trim_start_matches('/'));
        self.remove_index_entries_for_file(&Path::new(".skills").join(rel));
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn index_knowledge_file(&mut self, rel_path: &str, content: &str) -> io::Result<()> {
        let rel = Path::new(rel_path.trim_start_matches('/'));
        self.vector_db.index_text(KNOWLEDGE_PROJECT, rel, content)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn delete_knowledge_index(&mut self, rel_path: &str) -> io::Result<()> {
        let rel = Path::new(rel_path.trim_start_matches('/'));
        self.remove_index_entries_for_file(&Path::new(".knowledge").join(rel));
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    /// Пишет `.tools/<name>.md` и индексирует содержимое как TOOLS_PROJECT.
    pub fn write_tool_file(&mut self, name: &str, content: &str) -> io::Result<()> {
        let dir = self.root.join(".tools");
        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }
        let filename = format!("{}.md", name);
        let full_path = dir.join(&filename);
        fs::write(&full_path, content)?;

        let rel = Path::new(&filename);
        self.vector_db.index_text(TOOLS_PROJECT, rel, content)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    /// Полностью сносит `.tools/` и все записи индекса, относящиеся к нему.
    pub fn clear_tools_dir(&mut self) -> io::Result<()> {
        let dir = self.root.join(".tools");
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;

        let prefix_unix = ".tools/";
        let prefix_win = ".tools\\";
        let old_labels: Vec<String> = self
            .vector_db
            .entries
            .iter()
            .filter(|(label, _)| label.starts_with(prefix_unix) || label.starts_with(prefix_win))
            .map(|(label, _)| label.clone())
            .collect();

        for label in old_labels {
            if let Some(id) = self.vector_db.label_to_id.remove(&label) {
                self.vector_db.deleted_ids.insert(id);
                self.vector_db.id_to_label.remove(&id);
                self.vector_db.id_to_meta.remove(&id);
            }
            self.vector_db.entries.retain(|(l, _)| l != &label);
        }

        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn read_file(&self, project_id: &str, path: &str) -> io::Result<String> {
        fs::read_to_string(self.resolve(project_id, path)?)
    }

    pub fn delete_file(&mut self, project_id: &str, path: &str) -> io::Result<()> {
        let full = self.resolve(project_id, path)?;
        let storage_rel = Path::new(PROJECTS_DIR).join(project_id).join(path);
        let label_prefix = format!("{}#", storage_rel.to_string_lossy());
        let mut old_labels = Vec::new();
        for (label, _) in &self.vector_db.entries {
            if label.starts_with(&label_prefix) {
                old_labels.push(label.clone());
            }
        }
        for label in old_labels {
            if let Some(id) = self.vector_db.label_to_id.remove(&label) {
                self.vector_db.deleted_ids.insert(id);
                self.vector_db.id_to_label.remove(&id);
                self.vector_db.id_to_meta.remove(&id);
            }
            self.vector_db.entries.retain(|(l, _)| l != &label);
        }
        fs::remove_file(full)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn list(&self, project_id: &str, path: &str) -> io::Result<Vec<Entry>> {
        let full = self.resolve(project_id, path)?;
        let project_root = self.root.join(PROJECTS_DIR).join(project_id);
        let mut entries = Vec::new();
        for entry in fs::read_dir(full)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if is_reserved_name(&name) {
                continue;
            }
            let rel_path = entry
                .path()
                .strip_prefix(&project_root)
                .unwrap_or(&entry.path())
                .to_path_buf();
            if entry.file_type()?.is_dir() {
                entries.push(Entry::Dir {
                    name,
                    path: rel_path,
                });
            } else {
                entries.push(Entry::File {
                    name,
                    path: rel_path,
                    size: entry.metadata()?.len(),
                });
            }
        }
        Ok(entries)
    }

    pub fn walk(&self, project_id: &str, path: &str) -> io::Result<Vec<Entry>> {
        let full = self.resolve(project_id, path)?;
        let project_root = self.root.join(PROJECTS_DIR).join(project_id);
        let mut result = Vec::new();
        self.recursive_walk(&project_root, &full, &mut result)?;
        Ok(result)
    }

    fn recursive_walk(
        &self,
        project_root: &Path,
        base: &Path,
        out: &mut Vec<Entry>,
    ) -> io::Result<()> {
        for entry in fs::read_dir(base)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if is_reserved_name(&name) {
                continue;
            }
            let rel = path
                .strip_prefix(project_root)
                .unwrap_or(&path)
                .to_path_buf();
            if entry.file_type()?.is_dir() {
                out.push(Entry::Dir { name, path: rel });
                self.recursive_walk(project_root, &path, out)?;
            } else {
                out.push(Entry::File {
                    name,
                    path: rel,
                    size: entry.metadata()?.len(),
                });
            }
        }
        Ok(())
    }

    pub fn search_by_name(&self, project_id: &str, pattern: &str) -> io::Result<Vec<PathBuf>> {
        let all = self.walk(project_id, "")?;
        Ok(all
            .into_iter()
            .filter_map(|e| match e {
                Entry::File { name, path, .. } | Entry::Dir { name, path } => {
                    if name.contains(pattern) {
                        Some(path)
                    } else {
                        None
                    }
                }
            })
            .collect())
    }

    pub fn search_similar(
        &self,
        project_id: Option<&str>,
        query: &str,
        top_k: Option<usize>,
    ) -> io::Result<Vec<SearchResult>> {
        self.vector_db.search(
            project_id,
            query,
            top_k.unwrap_or(self.vector_db.config.top_k),
        )
    }

    pub fn write_about(
        &mut self,
        project_id: &str,
        dir_path: &str,
        content: &str,
    ) -> io::Result<()> {
        self.ensure_project(project_id)?;
        fs::write(self.resolve(project_id, dir_path)?.join(".about"), content)
    }

    pub fn read_about(&self, project_id: &str, dir_path: &str) -> io::Result<String> {
        fs::read_to_string(self.resolve(project_id, dir_path)?.join(".about"))
    }

    pub fn write_summary(
        &mut self,
        project_id: &str,
        dir_path: &str,
        content: &str,
    ) -> io::Result<()> {
        self.ensure_project(project_id)?;
        fs::write(
            self.resolve(project_id, dir_path)?.join(".summary"),
            content,
        )
    }

    pub fn read_summary(&self, project_id: &str, dir_path: &str) -> io::Result<String> {
        fs::read_to_string(self.resolve(project_id, dir_path)?.join(".summary"))
    }

    /// Синхронизирует индекс с файловой системой:
    /// переиндексирует изменённые файлы, убирает удалённые, добавляет новые.
    pub fn sync_index(&mut self) -> io::Result<()> {
        let on_disk = self.scan_disk_state()?;
        let in_index: std::collections::HashMap<PathBuf, (u64, u64)> = self
            .vector_db
            .entries
            .iter()
            .map(|(_, meta)| (meta.file_path.clone(), (meta.modified, meta.file_size)))
            .collect();

        let mut to_remove: Vec<PathBuf> = Vec::new();
        let mut to_index: Vec<(String, PathBuf)> = Vec::new();

        for (storage_rel, (mtime, size)) in &in_index {
            match on_disk.get(storage_rel) {
                None => to_remove.push(storage_rel.clone()),
                Some(entry) if entry.mtime != *mtime || entry.size != *size => {
                    to_remove.push(storage_rel.clone());
                    to_index.push((entry.project_id.clone(), entry.rel_path.clone()));
                }
                Some(_) => {}
            }
        }

        for (storage_rel, entry) in &on_disk {
            if !in_index.contains_key(storage_rel) {
                to_index.push((entry.project_id.clone(), entry.rel_path.clone()));
            }
        }

        if to_remove.is_empty() && to_index.is_empty() {
            return Ok(());
        }

        for storage_rel in to_remove {
            self.remove_index_entries_for_file(&storage_rel);
        }

        for (project_id, rel_path) in to_index {
            let full_path = self.storage_path_for(&project_id, &rel_path);
            if let Ok(content) = fs::read_to_string(&full_path) {
                if let Err(e) = self.vector_db.index_text(&project_id, &rel_path, &content) {
                    eprintln!("⚠️ sync_index: {}: {}", full_path.display(), e);
                }
            }
        }

        self.vector_db.save(&self.root)?;
        Ok(())
    }

    fn storage_path_for(&self, project_id: &str, rel_path: &Path) -> PathBuf {
        match project_id {
            SKILLS_PROJECT => self.root.join(".skills").join(rel_path),
            KNOWLEDGE_PROJECT => self.root.join(".knowledge").join(rel_path),
            TOOLS_PROJECT => self.root.join(".tools").join(rel_path),
            _ => self.root.join(PROJECTS_DIR).join(project_id).join(rel_path),
        }
    }

    fn scan_disk_state(&self) -> io::Result<std::collections::HashMap<PathBuf, DiskEntry>> {
        use std::collections::HashMap;
        let mut out: HashMap<PathBuf, DiskEntry> = HashMap::new();

        let projects_root = self.root.join(PROJECTS_DIR);
        if projects_root.exists() {
            let mut files = Vec::new();
            collect_files(&projects_root, &mut files)?;
            for file in files {
                let storage_rel = match file.strip_prefix(&self.root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                let mut comps = storage_rel.components();
                if comps.next().map(|c| c.as_os_str()) != Some(std::ffi::OsStr::new(PROJECTS_DIR)) {
                    continue;
                }
                let project_id = match comps.next().and_then(|c| c.as_os_str().to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let rel_path: PathBuf = comps.collect();
                if rel_path.as_os_str().is_empty() {
                    continue;
                }
                if let Some(entry) = make_disk_entry(&file, project_id, rel_path) {
                    out.insert(storage_rel, entry);
                }
            }
        }

        let skills_root = self.root.join(".skills");
        if skills_root.exists() {
            let mut files = Vec::new();
            collect_files(&skills_root, &mut files)?;
            for file in files {
                let rel_path = match file.strip_prefix(&skills_root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                if rel_path.as_os_str().is_empty() {
                    continue;
                }
                let storage_rel = Path::new(".skills").join(&rel_path);
                if let Some(entry) = make_disk_entry(&file, SKILLS_PROJECT.to_string(), rel_path) {
                    out.insert(storage_rel, entry);
                }
            }
        }

        let knowledge_root = self.root.join(".knowledge");
        if knowledge_root.exists() {
            for entry in fs::read_dir(&knowledge_root)? {
                let entry = entry?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.ends_with(".md") {
                    continue;
                }
                let storage_rel = Path::new(".knowledge").join(&name);
                if let Some(disk_entry) =
                    make_disk_entry(&path, KNOWLEDGE_PROJECT.to_string(), PathBuf::from(&name))
                {
                    out.insert(storage_rel, disk_entry);
                }
            }
        }

        let tools_root = self.root.join(".tools");
        if tools_root.exists() {
            let mut files = Vec::new();
            collect_files(&tools_root, &mut files)?;
            for file in files {
                let rel_path = match file.strip_prefix(&tools_root) {
                    Ok(p) => p.to_path_buf(),
                    Err(_) => continue,
                };
                if rel_path.as_os_str().is_empty() {
                    continue;
                }
                let storage_rel = Path::new(".tools").join(&rel_path);
                if let Some(entry) = make_disk_entry(&file, TOOLS_PROJECT.to_string(), rel_path) {
                    out.insert(storage_rel, entry);
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn default_config() -> VectorDbConfig {
        VectorDbConfig {
            model_path: ".models/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            tokenizer_path: ".models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"
                .to_string(),
            chunk_size: 256,
            chunk_overlap: 32,
            top_k: 3,
        }
    }

    #[test]
    fn test_index_and_semantic_search() {
        if !Path::new(".models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors").exists() {
            eprintln!("Модель не найдена, пропускаем тест");
            return;
        }
        let storage_name = "test_storage";
        let project = "test_project";
        let mut storage = LocalStorage::new(storage_name, default_config()).unwrap();
        let rust_text = "Rust — это язык системного программирования с безопасной работой с памятью. Он предотвращает гонки данных и утечки памяти благодаря строгой системе владения. Rust отлично подходит для создания надёжного и быстрого программного обеспечения. Многие компании выбирают Rust для системных сервисов, встроенных устройств и веб-разработки. Синтаксис Rust напоминает C++, но семантика гораздо безопаснее.";
        let cooking_text = "Борщ — традиционный украинский суп. Основные ингредиенты: свекла, капуста, картофель, морковь, лук, томатная паста и чеснок. Свеклу обычно тушат отдельно с уксусом для сохранения цвета. Борщ подают со сметаной и зеленью. Это сытное и ароматное блюдо, которое согревает в холодное время года. Рецепт передаётся из поколения в поколение.";
        let travel_text = "Путешествие по Европе может быть незабываемым. Популярные направления: Париж с Эйфелевой башней, Рим с Колизеем, Барселона с собором Саграда Фамилия. Летом лучше всего посетить средиземноморское побережье. Зимой можно отправиться в Альпы кататься на лыжах. Для бюджетных поездок подойдут Прага или Будапешт.";
        let mut rust_content = rust_text.repeat(3);
        let mut cooking_content = cooking_text.repeat(3);
        let mut travel_content = travel_text.repeat(3);
        rust_content.truncate(1024);
        cooking_content.truncate(1024);
        travel_content.truncate(1024);
        storage
            .create_file(project, "docs/rust.txt", &rust_content)
            .unwrap();
        storage
            .create_file(project, "docs/cooking.txt", &cooking_content)
            .unwrap();
        storage
            .create_file(project, "docs/travel.txt", &travel_content)
            .unwrap();

        let queries = [
            (
                "Безопасный язык программирования для системной разработки",
                "rust.txt",
            ),
            ("Как приготовить традиционный борщ", "cooking.txt"),
            ("Куда поехать летом в Европе", "travel.txt"),
        ];
        for (query, expected_file) in queries.iter() {
            let start = Instant::now();
            let results = storage
                .search_similar(Some(project), query, Some(3))
                .unwrap();
            let elapsed = start.elapsed();
            println!("\nЗапрос: '{}'", query);
            println!("Время поиска: {:?}", elapsed);
            for r in &results {
                println!(
                    "  - {} (distance: {:.4}) | фрагмент: {}",
                    r.file_path.display(),
                    r.distance,
                    r.content_fragment.chars().take(80).collect::<String>()
                );
            }
            assert!(
                results
                    .iter()
                    .any(|r| r.file_path.to_string_lossy().contains(expected_file)),
                "Файл {} не найден для запроса '{}'",
                expected_file,
                query
            );
        }
    }

    #[test]
    fn is_reserved_name_matches_known() {
        assert!(is_reserved_name(".about"));
        assert!(is_reserved_name(".summary"));
        assert!(is_reserved_name(".skills"));
        assert!(is_reserved_name(".rebukes"));
        assert!(is_reserved_name(".knowledge"));
        assert!(is_reserved_name("vector_meta.json"));
    }

    #[test]
    fn is_reserved_name_is_case_insensitive() {
        assert!(is_reserved_name(".ABOUT"));
        assert!(is_reserved_name(".Summary"));
    }

    #[test]
    fn is_reserved_name_rejects_regular() {
        assert!(!is_reserved_name("main.rs"));
        assert!(!is_reserved_name("README.md"));
        assert!(!is_reserved_name("about"));
    }

    #[test]
    fn valid_project_id_accepts_alphanumeric_dash_underscore() {
        assert!(valid_project_id("myproject"));
        assert!(valid_project_id("my-project"));
        assert!(valid_project_id("my_project"));
        assert!(valid_project_id("MyProject123"));
    }

    #[test]
    fn valid_project_id_rejects_empty() {
        assert!(!valid_project_id(""));
    }

    #[test]
    fn valid_project_id_rejects_spaces_and_slashes() {
        assert!(!valid_project_id("my project"));
        assert!(!valid_project_id("my/project"));
        assert!(!valid_project_id("my\\project"));
        assert!(!valid_project_id("my.project"));
    }

    #[test]
    fn vector_db_config_default_values() {
        let cfg = VectorDbConfig::default();
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.chunk_overlap, 64);
        assert_eq!(cfg.top_k, 5);
        assert!(cfg.model_path.contains("paraphrase"));
    }

    #[test]
    fn chunk_text_empty_returns_empty() {
        let chunks = chunk_text("", 512, 64);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_text_short_text_single_chunk() {
        let text = "короткий текст";
        let chunks = chunk_text(text, 512, 64);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
        assert_eq!(chunks[0].1, 0);
        assert_eq!(chunks[0].2, text.len() as u64);
    }

    #[test]
    fn chunk_text_long_splits_into_multiple() {
        let text = "word ".repeat(500);
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() > 1);
        for (chunk, start, end) in &chunks {
            assert!(!chunk.is_empty());
            assert!(start <= end);
        }
    }

    #[test]
    fn chunk_text_positions_are_monotonic() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa ".repeat(20);
        let chunks = chunk_text(&text, 50, 10);
        let mut prev_start = 0u64;
        for (_, start, _) in &chunks {
            assert!(*start >= prev_start);
            prev_start = *start;
        }
    }

    #[test]
    fn chunk_text_short_whitespace_is_single_chunk() {
        let text = "   \n\t  ";
        let chunks = chunk_text(text, 512, 64);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, text);
    }

    #[test]
    fn chunk_text_overlap_zero_no_overlap() {
        let text = "a b c d e f g h i j k l m n o p q r s t ".repeat(10);
        let chunks = chunk_text(&text, 20, 0);
        // С нулевым overlap каждая позиция start должна быть больше предыдущей end
        assert!(chunks.len() > 1);
    }
}
