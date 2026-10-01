// src/skill_manager.rs

use anyhow::Result;
use reqwest::blocking::Client;
use reqwest::Client as AsyncClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::{ChatEngine, EngineConfig, Role};

// ==================== Типы ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecord {
    pub skill_file: String,
    pub agent_name: String,
    pub description: String,
    pub prompt: String,
    pub entities: Vec<String>,
    pub success_count: u32,
    pub fail_count: u32,
    pub fingerprint: String,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub enum SaveSkillOutcome {
    Saved,
    SkippedTooFewCalls { actual: usize, min: usize },
    SkippedDuplicate,
}

// ==================== Служебное ====================

fn ensure_scheme(base_url: &str) -> String {
    if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    }
}

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

fn sha256(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn compute_fingerprint(prompt: &str, tool_calls: &[ToolCallRecord]) -> String {
    let normalized: String = prompt
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let prompt_part: String = normalized.chars().take(2000).collect();
    let calls_part = serde_json::to_string(tool_calls).unwrap_or_default();
    sha256(&format!("{}|{}", prompt_part, calls_part))
}

fn index_file_for(skill_file: &str) -> String {
    let base = skill_file.trim_end_matches(".txt");
    format!("{}_index.txt", base)
}

// ==================== HTTP к хранилищу ====================

pub fn list_skills(client: &Client, base_url: &str, auth_token: &str) -> Result<Vec<SkillRecord>> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/skills/list", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if !resp.status().is_success() {
        return Ok(vec![]);
    }
    Ok(resp.json()?)
}

pub fn load_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_file: &str,
) -> Result<String> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/skills/get", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("path", skill_file)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if resp.status().is_success() {
        Ok(resp.text()?)
    } else {
        Err(anyhow::anyhow!("Failed to read skill: {}", resp.status()))
    }
}

pub fn parse_tool_calls(content: &str) -> Option<Vec<ToolCallRecord>> {
    let marker = "TOOL_CALLS:";
    let pos = content.find(marker)?;
    let json_part = content[pos + marker.len()..].trim();
    serde_json::from_str(json_part).ok()
}

pub fn format_skill_for_injection(record: &SkillRecord, tool_calls: &[ToolCallRecord]) -> String {
    let mut out = String::new();
    out.push_str("[Похожая задача уже решалась так:]\n\n");
    out.push_str(&format!("Промпт: {}\n\n", record.prompt));
    out.push_str("Что делалось:\n");
    for (i, call) in tool_calls.iter().enumerate() {
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        let args_short: String = if args.len() > 200 {
            let truncated: String = args.chars().take(200).collect();
            format!("{}...", truncated)
        } else {
            args
        };
        out.push_str(&format!("{}. {}({})\n", i + 1, call.name, args_short));
    }
    out.push_str("\nИспользуй этот опыт как ориентир, адаптируй под текущую задачу.");
    out
}

// ==================== Сохранение ====================

pub fn save_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_name: &str,
    agent_name: &str,
    description: &str,
    prompt: &str,
    tool_calls: &[ToolCallRecord],
    min_tool_calls: usize,
) -> Result<SaveSkillOutcome> {
    if tool_calls.len() < min_tool_calls {
        return Ok(SaveSkillOutcome::SkippedTooFewCalls {
            actual: tool_calls.len(),
            min: min_tool_calls,
        });
    }

    let fingerprint = compute_fingerprint(prompt, tool_calls);

    let catalog = list_skills(client, base_url, auth_token)?;
    if catalog
        .iter()
        .any(|r| r.agent_name == agent_name && r.fingerprint == fingerprint)
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
    let index_file = index_file_for(&skill_file);

    let (skeleton, entities) =
        crate::lexicon::extract_entities_and_skeleton(prompt, crate::lexicon::get_lexicon());

    let display_name = if skill_name.trim().is_empty() {
        "task"
    } else {
        skill_name
    };
    let entities_line = entities.join(", ");
    let tool_calls_json = serde_json::to_string_pretty(tool_calls)?;

    let content = format!(
        "SKILL: {}\nFOR: {}\nDESCRIPTION: {}\nENTITIES: {}\nSTATS: 0/0\nFINGERPRINT: {}\nCREATED_AT: {}\n\nINSTRUCTION:\n{}\n\nTOOL_CALLS:\n{}\n",
        display_name,
        agent_name,
        description,
        entities_line,
        fingerprint,
        timestamp,
        prompt,
        tool_calls_json
    );

    let record = SkillRecord {
        skill_file: skill_file.clone(),
        agent_name: agent_name.to_string(),
        description: description.to_string(),
        prompt: prompt.to_string(),
        entities,
        success_count: 0,
        fail_count: 0,
        fingerprint,
        created_at: timestamp,
    };

    let body = serde_json::json!({
        "content": content,
        "index": skeleton,
        "record": record,
        "skill_file": skill_file,
        "index_file": index_file,
    });

    let base = ensure_scheme(base_url);
    let url = format!("{}/skills/put", base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", auth_token))
        .json(&body)
        .send()?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "Failed to save skill: HTTP {} — {}",
            resp.status(),
            resp.text().unwrap_or_default()
        );
    }

    Ok(SaveSkillOutcome::Saved)
}

// ==================== Учёт использования ====================

pub fn record_usage(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    skill_file: &str,
    success: bool,
) -> Result<()> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/skills/record_usage", base.trim_end_matches('/'));
    let success_str = if success { "true" } else { "false" };
    let resp = client
        .post(&url)
        .query(&[("file", skill_file), ("success", success_str)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if !resp.status().is_success() {
        anyhow::bail!("Failed to record usage: {}", resp.status());
    }
    Ok(())
}

// ==================== Поиск ====================

pub fn search_best_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    current_agent: &str,
    prompt: &str,
    semantic_threshold: f32,
    top_k: usize,
) -> Result<Option<(SkillRecord, f32)>> {
    let catalog = list_skills(client, base_url, auth_token)?;
    if catalog.is_empty() {
        return Ok(None);
    }

    let (query_skeleton, query_entities) =
        crate::lexicon::extract_entities_and_skeleton(prompt, crate::lexicon::get_lexicon());

    let base = ensure_scheme(base_url);
    let search_url = format!("{}/skills/search", base.trim_end_matches('/'));
    let top_k_str = (top_k * 4).to_string();
    let resp = client
        .get(&search_url)
        .query(&[("query", &query_skeleton), ("top_k", &top_k_str)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Search request failed"));
    }
    let results: Vec<serde_json::Value> = resp.json()?;

    // Карта: нормализованный путь индексного файла → запись каталога.
    let mut by_index: HashMap<String, SkillRecord> = HashMap::new();
    for rec in &catalog {
        let idx = index_file_for(&rec.skill_file);
        by_index.insert(idx, rec.clone());
    }

    let mut own_candidates: Vec<(SkillRecord, f32)> = Vec::new();
    let mut all_candidates: Vec<(SkillRecord, f32)> = Vec::new();

    for item in results {
        let file_path = item.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        let distance = item.get("distance").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        if distance > semantic_threshold {
            continue;
        }

        let normalized = file_path.replace('\\', "/");
        let index_name = normalized
            .strip_prefix(".skills/")
            .unwrap_or(&normalized)
            .to_string();

        let Some(rec) = by_index.get(&index_name) else {
            continue;
        };

        if !crate::lexicon::entities_subset(&query_entities, &rec.entities) {
            eprintln!(
                "🔍 отсеян скилл {}: сущности запроса {:?} не входят в {:?}",
                rec.skill_file, query_entities, rec.entities
            );
            continue;
        }

        if rec.agent_name == current_agent {
            own_candidates.push((rec.clone(), distance));
        }
        all_candidates.push((rec.clone(), distance));
    }

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

// ==================== Метаданные через LLM ====================

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

