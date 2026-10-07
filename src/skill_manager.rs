// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/skill_manager.rs

use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::yake::{extract as yake_extract, YakeConfig};

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

fn compute_fingerprint(prompt: &str) -> String {
    let (skeleton, _entities) =
        crate::lexicon::extract_entities_and_skeleton(prompt, crate::lexicon::get_lexicon());
    let normalized: String = skeleton
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    sha256(&normalized)
}

/// Из имени индексного файла `.skills/<stem>_index_<N>.idx`
/// достаёт имя основного файла скилла `<stem>.md`.
///
/// Универсально для unix- и windows-разделителей пути.
fn skill_file_from_index_path(path: &str) -> Option<String> {
    let name = path.rsplit(|c| c == '/' || c == '\\').next()?;
    let stem = name.strip_suffix(".idx")?;
    let without_n = stem.rsplit_once('_')?.0;
    let base = without_n.strip_suffix("_index")?;
    Some(format!("{}.md", base))
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

/// Извлекает многословные фразы YAKE из промпта.
fn extract_index_phrases(prompt: &str, top_n: usize, min_words: usize) -> Vec<String> {
    let phrases = yake_extract(
        prompt,
        &YakeConfig {
            top_n,
            ..Default::default()
        },
    );
    phrases
        .into_iter()
        .map(|(p, _)| p)
        .filter(|p| p.split_whitespace().count() >= min_words)
        .collect()
}

#[allow(clippy::too_many_arguments)]
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
    phrases_top_n: usize,
    phrase_min_words: usize,
) -> Result<SaveSkillOutcome> {
    if tool_calls.len() < min_tool_calls {
        return Ok(SaveSkillOutcome::SkippedTooFewCalls {
            actual: tool_calls.len(),
            min: min_tool_calls,
        });
    }

    let fingerprint = compute_fingerprint(prompt);

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
        "skill_agent_{}_{}_{}.md",
        agent_name, normalized_name, timestamp
    );

    let (_, entities) =
        crate::lexicon::extract_entities_and_skeleton(prompt, crate::lexicon::get_lexicon());

    let index_phrases = extract_index_phrases(prompt, phrases_top_n, phrase_min_words);

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
        "index_files": index_phrases,
        "record": record,
        "skill_file": skill_file,
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

fn success_rate(rec: &SkillRecord) -> f32 {
    rec.success_count as f32 / (rec.success_count + rec.fail_count + 1) as f32
}

/// Ищет лучший скилл под заданный промпт.
///
/// Алгоритм:
/// 1. YAKE по промпту → фразы запроса (>= 2 слов).
/// 2. По каждой фразе — векторный поиск по фразам скиллов.
/// 3. Порог distance <= threshold.
/// 4. Агрегация hits по файлу скилла.
/// 5. Фильтр hits >= min_hits.
/// 6. Top-1 с приоритетом: свой агент → success_rate → min distance.
#[allow(clippy::too_many_arguments)]
pub fn search_best_skill(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    current_agent: &str,
    prompt: &str,
    phrases_top_n: usize,
    phrase_min_words: usize,
    threshold: f32,
    min_hits: usize,
) -> Result<Option<(SkillRecord, f32)>> {
    let catalog = list_skills(client, base_url, auth_token)?;
    if catalog.is_empty() {
        return Ok(None);
    }

    let query_phrases = extract_index_phrases(prompt, phrases_top_n, phrase_min_words);
    if query_phrases.is_empty() {
        return Ok(None);
    }

    let base = ensure_scheme(base_url);
    let search_url = format!("{}/skills/search", base.trim_end_matches('/'));

    let mut hits: HashMap<String, HashSet<String>> = HashMap::new();
    let mut min_dist: HashMap<String, f32> = HashMap::new();

    // top_k для одного запроса: с запасом покрыть все индексные чанки.
    let per_query_top_k = 100usize;

    for phrase in &query_phrases {
        let resp = client
            .get(&search_url)
            .query(&[
                ("query", phrase.as_str()),
                ("top_k", per_query_top_k.to_string().as_str()),
            ])
            .header("Authorization", format!("Bearer {}", auth_token))
            .send()?;
        if !resp.status().is_success() {
            continue;
        }
        let results: Vec<serde_json::Value> = resp.json()?;

        for r in results {
            let dist = r.get("distance").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            if dist > threshold {
                continue;
            }
            let file_path = r.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let Some(skill_file) = skill_file_from_index_path(file_path) else {
                continue;
            };
            hits.entry(skill_file.clone())
                .or_default()
                .insert(phrase.clone());
            let e = min_dist.entry(skill_file).or_insert(f32::MAX);
            if dist < *e {
                *e = dist;
            }
        }
    }

    if hits.is_empty() {
        return Ok(None);
    }

    let by_file: HashMap<String, SkillRecord> = catalog
        .into_iter()
        .map(|r| (r.skill_file.clone(), r))
        .collect();

    let mut candidates: Vec<(String, usize, f32)> = hits
        .into_iter()
        .map(|(f, set)| {
            let d = min_dist.get(&f).copied().unwrap_or(1.0);
            (f, set.len(), d)
        })
        .filter(|(f, h, _)| *h >= min_hits && by_file.contains_key(f))
        .collect();

    if candidates.is_empty() {
        return Ok(None);
    }

    candidates.sort_by(|a, b| {
        let ra = by_file.get(&a.0);
        let rb = by_file.get(&b.0);
        let own_a = ra.map(|r| r.agent_name == current_agent).unwrap_or(false);
        let own_b = rb.map(|r| r.agent_name == current_agent).unwrap_or(false);
        own_b
            .cmp(&own_a)
            .then_with(|| {
                let sa = ra.map(success_rate).unwrap_or(0.0);
                let sb = rb.map(success_rate).unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let (skill_file, _hits, dist) = candidates.into_iter().next().unwrap();
    let Some(record) = by_file.get(&skill_file).cloned() else {
        return Ok(None);
    };
    Ok(Some((record, dist)))
}

// ==================== Метаданные через LLM ====================

pub async fn generate_skill_metadata(
    engine_config: &crate::engine::EngineConfig,
    prompt: &str,
) -> Result<(String, String)> {
    use crate::engine::{ChatEngine, Role};

    let client = reqwest::Client::new();
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

// ==================== Тесты ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_replaces_non_alnum_with_underscore() {
        assert_eq!(normalize_skill_name("My Task!"), "my_task_");
        assert_eq!(normalize_skill_name("hello-world"), "hello_world");
        assert_eq!(normalize_skill_name("café"), "café");
    }

    #[test]
    fn fingerprint_stable_for_same_input() {
        let a = compute_fingerprint("read the file");
        let b = compute_fingerprint("read the file");
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_ignores_whitespace_and_case() {
        let a = compute_fingerprint("  Read  THE File  ");
        let b = compute_fingerprint("read the file");
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_same_skeleton_different_params() {
        let a = compute_fingerprint(
            "создай файл config.txt с текстом \"hello\", потом создай файл notes.txt с текстом \"world\"",
        );
        let b = compute_fingerprint(
            "создай файл settings.json с текстом \"{}\", потом создай файл log.txt с текстом \"start\"",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_differs_for_different_procedures() {
        let a = compute_fingerprint("создай файл config.txt с текстом hello");
        let b = compute_fingerprint("удали файл config.txt");
        assert_ne!(a, b);
    }

    #[test]
    fn parse_tool_calls_extracts_json() {
        let content = "SKILL: x\nFOR: agent\n\nTOOL_CALLS:\n[{\"name\":\"a\",\"arguments\":{}}]";
        let calls = parse_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn parse_tool_calls_missing_marker_returns_none() {
        assert!(parse_tool_calls("no marker here").is_none());
    }

    #[test]
    fn parse_tool_calls_invalid_json_returns_none() {
        assert!(parse_tool_calls("TOOL_CALLS:\n{not json").is_none());
    }

    #[test]
    fn success_rate_zero_when_no_stats() {
        let rec = SkillRecord {
            skill_file: "s".into(),
            agent_name: "a".into(),
            description: String::new(),
            prompt: String::new(),
            entities: vec![],
            success_count: 0,
            fail_count: 0,
            fingerprint: String::new(),
            created_at: 0,
        };
        assert_eq!(success_rate(&rec), 0.0);
    }

    #[test]
    fn success_rate_high_when_all_success() {
        let rec = SkillRecord {
            skill_file: "s".into(),
            agent_name: "a".into(),
            description: String::new(),
            prompt: String::new(),
            entities: vec![],
            success_count: 10,
            fail_count: 0,
            fingerprint: String::new(),
            created_at: 0,
        };
        assert!(success_rate(&rec) > 0.9);
    }

    #[test]
    fn format_skill_truncates_long_arguments() {
        let long = "x".repeat(500);
        let rec = SkillRecord {
            skill_file: "s".into(),
            agent_name: "a".into(),
            description: String::new(),
            prompt: "задача".into(),
            entities: vec![],
            success_count: 0,
            fail_count: 0,
            fingerprint: String::new(),
            created_at: 0,
        };
        let calls = vec![ToolCallRecord {
            name: "t".into(),
            arguments: serde_json::json!({ "data": long }),
        }];
        let out = format_skill_for_injection(&rec, &calls);
        assert!(out.contains("..."));
    }

    #[test]
    fn skill_file_from_index_path_unix() {
        assert_eq!(
            skill_file_from_index_path("skill_agent_x_build_index_123_index_0.idx"),
            Some("skill_agent_x_build_index_123.md".to_string())
        );
    }

    #[test]
    fn skill_file_from_index_path_windows() {
        assert_eq!(
            skill_file_from_index_path(".skills\\skill_agent_x_task_99_index_5.idx"),
            Some("skill_agent_x_task_99.md".to_string())
        );
    }

    #[test]
    fn skill_file_from_index_path_rejects_non_idx() {
        assert_eq!(skill_file_from_index_path("skill.md"), None);
        assert_eq!(skill_file_from_index_path("skill_index.txt"), None);
    }
}
