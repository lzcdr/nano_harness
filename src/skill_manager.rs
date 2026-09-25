// src/skill_manager.rs

use anyhow::Result;
use reqwest::blocking::Client;
use reqwest::Client as AsyncClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::{ChatEngine, EngineConfig, Role};

const SKILLS_DIR: &str = ".skills";

// ----- Структуры данных -----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecord {
    pub skill_file: String,
    pub description_file: String,
    pub prompt_file: String,
    pub agent_name: String,
    pub description: String,
    pub prompt: String,
    pub success_count: u32,
    pub fail_count: u32,
    pub ast_hash: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillCatalog {
    pub skills: Vec<SkillRecord>,
}

// ----- Вспомогательные функции -----

fn normalize_skill_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn skill_path(relative: &str) -> String {
    format!("{}/{}", SKILLS_DIR, relative.trim_start_matches('/'))
}

fn normalize_path(s: &str) -> String {
    s.replace('\\', "/")
}

fn load_catalog(client: &Client, base_url: &str, auth_token: &str) -> Result<SkillCatalog> {
    let url = format!("{}/skills/get", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("path", "skills_catalog.json")])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if resp.status().is_success() {
        let text = resp.text()?;
        if text.trim().is_empty() {
            return Ok(SkillCatalog::default());
        }
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(SkillCatalog::default())
    }
}

fn save_catalog(
    catalog: &SkillCatalog,
    client: &Client,
    base_url: &str,
    auth_token: &str,
) -> Result<()> {
    let json = serde_json::to_string_pretty(catalog)?;
    let url = format!("{}/skills/put", base_url.trim_end_matches('/'));
    client
        .post(&url)
        .query(&[("path", "skills_catalog.json")])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(json)
        .send()?;
    Ok(())
}

fn write_file(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    path: &str,
    content: &str,
) -> Result<()> {
    let url = format!("{}/skills/put", base_url.trim_end_matches('/'));
    client
        .post(&url)
        .query(&[("path", path)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(content.to_string())
        .send()?;
    Ok(())
}

fn read_file(client: &Client, base_url: &str, auth_token: &str, path: &str) -> Result<String> {
    let url = format!("{}/skills/get", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("path", path)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if resp.status().is_success() {
        Ok(resp.text()?)
    } else {
        Err(anyhow::anyhow!("Failed to read file: {}", resp.status()))
    }
}

// ----- Вычисление AST-хэша -----

fn compute_ast_hash(code: &str) -> String {
    let engine = rhai::Engine::new();
    match engine.compile(code) {
        Ok(ast) => {
            let text = format!("{:?}", ast);
            sha256(&text)
        }
        Err(_) => String::new(),
    }
}

fn sha256(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ----- Основные функции -----

pub fn search_best_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    current_agent: &str,
    prompt: &str,
    semantic_threshold: f32,
    top_k: usize,
) -> Result<Option<(SkillRecord, f32)>> {
    let catalog = load_catalog(client, base_url, auth_token)?;
    eprintln!(
        "🔍 search_best_skill: agent={}, catalog_size={}, threshold={}",
        current_agent,
        catalog.skills.len(),
        semantic_threshold
    );
    if catalog.skills.is_empty() {
        eprintln!("🔍 search_best_skill: catalog пуст, возвращаю None");
        return Ok(None);
    }

    let mut prompt_file_to_record: HashMap<String, SkillRecord> = HashMap::new();
    for rec in &catalog.skills {
        let key = normalize_path(&skill_path(&rec.prompt_file));
        prompt_file_to_record.insert(key, rec.clone());
    }

    let search_url = format!("{}/skills/search", base_url.trim_end_matches('/'));
    let top_k_str = (top_k * 2).to_string();
    let resp = client
        .get(&search_url)
        .query(&[("query", prompt), ("top_k", &top_k_str)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    eprintln!("🔍 search_best_skill: HTTP {}", resp.status());
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Search request failed"));
    }
    let results: Vec<serde_json::Value> = resp.json()?;
    eprintln!(
        "🔍 search_best_skill: получено {} результатов",
        results.len()
    );

    let mut own_candidates: Vec<(SkillRecord, f32)> = Vec::new();
    let mut all_candidates: Vec<(SkillRecord, f32)> = Vec::new();

    for item in results {
        let file_path = item.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        let distance = item.get("distance").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        if distance > semantic_threshold {
            eprintln!(
                "🔍   пропуск {}: distance {:.4} > threshold {}",
                file_path, distance, semantic_threshold
            );
            continue;
        }

        let file_path_norm = normalize_path(file_path);
        if let Some(rec) = prompt_file_to_record.get(&file_path_norm) {
            if rec.agent_name == current_agent {
                own_candidates.push((rec.clone(), distance));
            }
            all_candidates.push((rec.clone(), distance));
        }
    }

    eprintln!(
        "🔍 search_best_skill: own_candidates={}, all_candidates={}",
        own_candidates.len(),
        all_candidates.len()
    );

    if let Some(best_own) = select_best_by_success_rate(&own_candidates) {
        return Ok(Some(best_own));
    }

    let best_all = select_best_by_success_rate(&all_candidates);
    Ok(best_all)
}

fn select_best_by_success_rate(candidates: &[(SkillRecord, f32)]) -> Option<(SkillRecord, f32)> {
    candidates
        .iter()
        .max_by(|a, b| {
            let ra = success_rate(&a.0);
            let rb = success_rate(&b.0);
            ra.partial_cmp(&rb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        })
        .cloned()
}

fn success_rate(rec: &SkillRecord) -> f32 {
    rec.success_count as f32 / (rec.success_count + rec.fail_count + 1) as f32
}

pub fn load_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_file: &str,
) -> Result<String> {
    read_file(client, base_url, auth_token, skill_file)
}

#[derive(Debug, Clone)]
pub enum SaveSkillOutcome {
    Saved,
    SkippedTooShort { actual: usize, min: usize },
    SkippedDuplicate,
}

pub fn save_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_name: &str,
    agent_name: &str,
    description: &str,
    prompt: &str,
    rhai_code: &str,
    min_code_length: usize,
) -> Result<SaveSkillOutcome> {
    let trimmed = rhai_code.trim();
    if trimmed.len() < min_code_length {
        return Ok(SaveSkillOutcome::SkippedTooShort {
            actual: trimmed.len(),
            min: min_code_length,
        });
    }

    let ast_hash = compute_ast_hash(rhai_code);
    if ast_hash.is_empty() {
        return Err(anyhow::anyhow!("Invalid Rhai code"));
    }

    let mut catalog = load_catalog(client, base_url, auth_token)?;

    if catalog
        .skills
        .iter()
        .any(|r| r.agent_name == agent_name && r.ast_hash == ast_hash)
    {
        return Ok(SaveSkillOutcome::SkippedDuplicate);
    }

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let normalized_name = if skill_name.trim().is_empty() {
        "task".to_string()
    } else {
        normalize_skill_name(skill_name)
    };

    let skill_file = format!(
        "skill_agent_{}_{}_{}.txt",
        agent_name, normalized_name, timestamp
    );
    let description_file = format!("{}_description.txt", skill_file.trim_end_matches(".txt"));
    let prompt_file = format!("{}_prompt.txt", skill_file.trim_end_matches(".txt"));

    let content = format!(
        "SKILL: {}\nFOR: {}\nDESCRIPTION: {}\n\nINSTRUCTION:\n{}\n\nRHAI_CODE:\n{}\n",
        if skill_name.trim().is_empty() {
            "task"
        } else {
            skill_name
        },
        agent_name,
        description,
        prompt,
        rhai_code
    );
    write_file(client, base_url, auth_token, &skill_file, &content)?;
    write_file(client, base_url, auth_token, &description_file, description)?;
    write_file(client, base_url, auth_token, &prompt_file, prompt)?;

    catalog.skills.push(SkillRecord {
        skill_file,
        description_file,
        prompt_file,
        agent_name: agent_name.to_string(),
        description: description.to_string(),
        prompt: prompt.to_string(),
        success_count: 0,
        fail_count: 0,
        ast_hash,
        created_at: timestamp,
    });
    save_catalog(&catalog, client, base_url, auth_token)?;

    Ok(SaveSkillOutcome::Saved)
}

pub fn record_usage(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_file: &str,
    success: bool,
) -> Result<()> {
    let mut catalog = load_catalog(client, base_url, auth_token)?;
    if let Some(rec) = catalog
        .skills
        .iter_mut()
        .find(|r| r.skill_file == skill_file)
    {
        if success {
            rec.success_count += 1;
        } else {
            rec.fail_count += 1;
        }
        save_catalog(&catalog, client, base_url, auth_token)?;
    }
    Ok(())
}

pub fn list_skills(client: &Client, base_url: &str, auth_token: &str) -> Result<String> {
    let catalog = load_catalog(client, base_url, auth_token)?;
    Ok(serde_json::to_string_pretty(&catalog)?)
}

// ----- Генерация метаданных скилла через LLM -----

pub async fn generate_skill_metadata(
    engine_config: &EngineConfig,
    prompt: &str,
) -> Result<(String, String)> {
    let client = AsyncClient::new();
    let mut temp_engine = ChatEngine::new(engine_config.clone(), client);
    temp_engine.add_message(
        Role::System,
        "Ты - помощник, который генерирует название и описание для скилла (навыка) на основе описания задачи.".to_string(),
    );
    temp_engine.add_message(
        Role::User,
        format!(
            "Сгенерируй название и описание скилла по такому промпту: \"{}\". Ответ верни в виде двух строк, разделенных запятой: \"название\", \"описание\".",
            prompt
        ),
    );
    let response = temp_engine.send().await?;
    let content = response.content;
    let parts: Vec<&str> = content.split(',').collect();
    if parts.len() < 2 {
        return Err(anyhow::anyhow!("Invalid metadata response: {}", content));
    }
    let name = parts[0].trim().trim_matches('"').to_string();
    let description = parts[1..].join(",").trim().trim_matches('"').to_string();
    Ok((name, description))
}
