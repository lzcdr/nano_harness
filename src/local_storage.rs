// local_storage.rs

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hnsw_rs::prelude::*;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const STORAGE_BASE: &str = ".local_storage";
const META_FILE: &str = "vector_meta.json";

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
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub file_path: PathBuf,
    pub chunk_index: u32,
    pub score: f32,
    pub content_fragment: String,
}

#[derive(Debug, Clone, Serialize)]
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
            deleted_ids: HashSet::new(),
            next_id: 0,
        })
    }

    fn save(&self, root: &Path) -> io::Result<()> {
        let meta_path = root.join(META_FILE);
        let meta_json = serde_json::to_string(&self.entries)?;
        fs::write(meta_path, meta_json)?;
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
                valid_entries.push((label, meta));
            }

            db.entries = valid_entries;
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

        // Mean pooling с учётом маски
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

        // L2 нормализация
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

        let embeddings = normalized.to_vec2::<f32>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("To vec error: {e}"))
        })?;

        Ok(embeddings)
    }

    fn embed_single(&self, text: &str) -> io::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    fn index_text(&mut self, file_path: &Path, content: &str) -> io::Result<()> {
        let label_prefix = format!("{}#", file_path.to_string_lossy());

        // Удаляем старые записи
        let mut old_labels = Vec::new();
        for (label, _) in &self.entries {
            if label.starts_with(&label_prefix) {
                old_labels.push(label.clone());
            }
        }
        for label in old_labels {
            if let Some(id) = self.label_to_id.remove(&label) {
                self.deleted_ids.insert(id);
            }
            self.entries.retain(|(l, _)| l != &label);
        }

        let chunks_with_pos =
            chunk_text(content, self.config.chunk_size, self.config.chunk_overlap);
        let chunk_texts: Vec<String> = chunks_with_pos
            .iter()
            .map(|(text, _, _)| text.clone())
            .collect();
        eprintln!(
            "🔄 Индексация файла {} ({} чанков)...",
            file_path.display(),
            chunk_texts.len()
        );
        let start = std::time::Instant::now();

        let embeddings = self.embed_batch(&chunk_texts)?;

        let full_path = self.root.join(file_path);
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
            let label = format!("{}#{}", file_path.to_string_lossy(), i);
            let id = self.next_id;
            self.next_id += 1;
            self.hnsw.insert((&emb, id));
            self.label_to_id.insert(label.clone(), id);
            self.entries.push((
                label,
                ChunkMeta {
                    file_path: file_path.to_path_buf(),
                    chunk_index: i as u32,
                    start_byte,
                    end_byte,
                    content: chunk_text,
                    embedding: emb,
                    file_size,
                    modified,
                },
            ));
        }

        eprintln!("✅ Индексация файла завершена за {:?}", start.elapsed());
        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> io::Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();
        let query_embedding = self.embed_single(query)?;
        let neighbours = self.hnsw.search(&query_embedding, top_k * 2, top_k * 2);

        let mut results = Vec::new();
        for neighbour in neighbours {
            let id = neighbour.d_id;
            if self.deleted_ids.contains(&id) {
                continue;
            }
            let label = self.label_to_id.iter().find_map(|(label, &entry_id)| {
                if entry_id == id {
                    Some(label.clone())
                } else {
                    None
                }
            });
            if let Some(label) = label {
                if let Some((_, meta)) = self.entries.iter().find(|(l, _)| l == &label) {
                    results.push(SearchResult {
                        file_path: meta.file_path.clone(),
                        chunk_index: meta.chunk_index,
                        score: neighbour.distance,
                        content_fragment: meta.content.clone(),
                    });
                }
            }
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
        self.deleted_ids.clear();
        self.next_id = 0;
        self.hnsw = Hnsw::new(16, 100_000, 32, 200, DistCosine);

        let mut files = Vec::new();
        collect_files(root, &mut files)?;
        for file in files {
            if let Ok(content) = fs::read_to_string(&file) {
                let rel = file.strip_prefix(root).unwrap_or(&file).to_path_buf();
                self.index_text(&rel, &content)?;
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
    // Получаем слова и их байтовые позиции за один проход
    let mut words: Vec<&str> = Vec::new();
    let mut positions: Vec<u64> = Vec::new();

    let bytes = text.as_bytes();
    let mut i = 0;
    let len = bytes.len();

    while i < len {
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }
        let start = i;
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let end = i;
        let word = &text[start..end];
        words.push(word);
        positions.push(start as u64);
    }

    let mut chunks = Vec::new();
    let mut start_idx = 0;

    while start_idx < words.len() {
        let mut end_idx = start_idx;
        let mut current_len = 0;
        while end_idx < words.len() && current_len < chunk_size {
            current_len += words[end_idx].len() + 1; // +1 за пробел
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

        let overlap_words = (overlap / 5).max(1);
        start_idx = end_idx.saturating_sub(overlap_words);
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

        let mut vector_db = VectorDb::load(&root, vector_db_config)?;
        if !root.join(META_FILE).exists() {
            vector_db.rebuild(&root)?;
            vector_db.save(&root)?;
        }

        Ok(Self { root, vector_db })
    }

    fn resolve(&self, path: &str) -> io::Result<PathBuf> {
        let rel = Path::new(path);
        if rel.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "absolute path not allowed",
            ));
        }
        if rel
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "path escapes storage root",
            ));
        }
        Ok(self.root.join(rel))
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

    pub fn create_dir(&mut self, path: &str) -> io::Result<()> {
        let full = self.resolve(path)?;
        fs::create_dir_all(&full)?;
        Self::ensure_system_files(&full)
    }

    pub fn create_file(&mut self, path: &str, content: &str) -> io::Result<()> {
        let full = self.resolve(path)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
            Self::ensure_system_files(parent)?;
        }
        fs::write(&full, content)?;

        let rel_path = full.strip_prefix(&self.root).unwrap_or(&full).to_path_buf();
        self.vector_db.index_text(&rel_path, content)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn write_file(&mut self, path: &str, content: &str) -> io::Result<()> {
        self.create_file(path, content)
    }

    pub fn read_file(&self, path: &str) -> io::Result<String> {
        let full = self.resolve(path)?;
        fs::read_to_string(full)
    }

    pub fn delete_file(&mut self, path: &str) -> io::Result<()> {
        let full = self.resolve(path)?;
        let rel_path = full.strip_prefix(&self.root).unwrap_or(&full).to_path_buf();
        let label_prefix = format!("{}#", rel_path.to_string_lossy());

        let mut old_labels = Vec::new();
        for (label, _) in &self.vector_db.entries {
            if label.starts_with(&label_prefix) {
                old_labels.push(label.clone());
            }
        }
        for label in old_labels {
            if let Some(id) = self.vector_db.label_to_id.remove(&label) {
                self.vector_db.deleted_ids.insert(id);
            }
            self.vector_db.entries.retain(|(l, _)| l != &label);
        }

        fs::remove_file(full)?;
        self.vector_db.save(&self.root)?;
        Ok(())
    }

    pub fn list(&self, path: &str) -> io::Result<Vec<Entry>> {
        let full = self.resolve(path)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(full)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if name == ".about" || name == ".summary" {
                continue;
            }

            if file_type.is_dir() {
                entries.push(Entry::Dir { name, path });
            } else {
                let size = entry.metadata()?.len();
                entries.push(Entry::File { name, path, size });
            }
        }
        Ok(entries)
    }

    pub fn walk(&self, path: &str) -> io::Result<Vec<Entry>> {
        let full = self.resolve(path)?;
        let mut result = Vec::new();
        self.recursive_walk(&full, &mut result)?;
        Ok(result)
    }

    fn recursive_walk(&self, base: &Path, out: &mut Vec<Entry>) -> io::Result<()> {
        for entry in fs::read_dir(base)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if name == ".about" || name == ".summary" {
                continue;
            }

            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                out.push(Entry::Dir {
                    name,
                    path: self.relativize(&path),
                });
                self.recursive_walk(&path, out)?;
            } else {
                let size = entry.metadata()?.len();
                out.push(Entry::File {
                    name,
                    path: self.relativize(&path),
                    size,
                });
            }
        }
        Ok(())
    }

    fn relativize(&self, path: &Path) -> PathBuf {
        path.strip_prefix(&self.root).unwrap_or(path).to_path_buf()
    }

    pub fn search_by_name(&self, pattern: &str) -> io::Result<Vec<PathBuf>> {
        let all = self.walk("")?;
        let mut results = Vec::new();
        for entry in all {
            match entry {
                Entry::File { name, path, .. } | Entry::Dir { name, path } => {
                    if name.contains(pattern) {
                        results.push(path);
                    }
                }
            }
        }
        Ok(results)
    }

    pub fn search_similar(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> io::Result<Vec<SearchResult>> {
        let k = top_k.unwrap_or(self.vector_db.config.top_k);
        self.vector_db.search(query, k)
    }

    pub fn write_about(&mut self, dir_path: &str, content: &str) -> io::Result<()> {
        let full_dir = self.resolve(dir_path)?;
        fs::write(full_dir.join(".about"), content)
    }

    pub fn read_about(&self, dir_path: &str) -> io::Result<String> {
        let full_dir = self.resolve(dir_path)?;
        fs::read_to_string(full_dir.join(".about"))
    }

    pub fn write_summary(&mut self, dir_path: &str, content: &str) -> io::Result<()> {
        let full_dir = self.resolve(dir_path)?;
        fs::write(full_dir.join(".summary"), content)
    }

    pub fn read_summary(&self, dir_path: &str) -> io::Result<String> {
        let full_dir = self.resolve(dir_path)?;
        fs::read_to_string(full_dir.join(".summary"))
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

        storage.create_file("docs/rust.txt", &rust_content).unwrap();
        storage
            .create_file("docs/cooking.txt", &cooking_content)
            .unwrap();
        storage
            .create_file("docs/travel.txt", &travel_content)
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
            let results = storage.search_similar(query, Some(3)).unwrap();
            let elapsed = start.elapsed();

            println!("\nЗапрос: '{}'", query);
            println!("Время поиска: {:?}", elapsed);
            for r in &results {
                println!(
                    "  - {} (score: {:.4}) | фрагмент: {}",
                    r.file_path.display(),
                    r.score,
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
}
