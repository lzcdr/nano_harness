// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/knowledge_auto.rs

//! Авто-инжект знаний в контекст.
//!
//! На вход — текст текущего хода (сообщения assistant + user).
//! На выход — готовое system-сообщение со склеенными телами top-K знаний,
//! либо None, если ничего релевантного не нашлось.

use anyhow::Result;
use reqwest::Client;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::yake::{extract as yake_extract, YakeConfig};

#[derive(Debug, Clone)]
pub struct KnowledgeAutoConfig {
    /// Сколько ключевых фраз брать из YAKE.
    pub top_n: usize,
    /// Сколько знаний в итоге инжектить.
    pub top_k: usize,
    /// Порог distance: результаты с большим distance отбрасываются.
    pub threshold: f32,
    /// Сколько результатов запрашивать на одну фразу (top_k хранилища).
    pub per_phrase: usize,
}

impl Default for KnowledgeAutoConfig {
    fn default() -> Self {
        Self {
            top_n: 7,
            top_k: 3,
            threshold: 0.70,
            per_phrase: 5,
        }
    }
}

fn ensure_scheme(base_url: &str) -> String {
    if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    }
}

/// Из `.knowledge/rust_ownership.md` достаёт `rust_ownership`.
fn knowledge_name_from_path(path: &str) -> Option<String> {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
}

/// Убирает frontmatter (`---\n...\n---\n`) из начала markdown-файла.
/// Если frontmatter нет — возвращает как есть.
fn strip_frontmatter(content: &str) -> String {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut lines = content.lines();

    let first = lines.next().unwrap_or("").trim();
    if first != "---" {
        return content.to_string();
    }

    for line in lines.by_ref() {
        if line.trim() == "---" {
            let rest: Vec<&str> = lines.collect();
            let body = rest.join("\n");
            return body.trim_start_matches('\n').to_string();
        }
    }

    content.to_string()
}

/// Строит множество слов из текста: split_whitespace + lowercase.
fn word_set(text: &str) -> HashSet<String> {
    text.split_whitespace().map(|w| w.to_lowercase()).collect()
}

/// Собирает system-сообщение с автоматически подгруженными знаниями.
///
/// Возвращает `Ok(Some(text))`, если хотя бы одно знание прошло порог,
/// и `Ok(None)` иначе.
pub async fn build_knowledge_context(
    client: &Client,
    storage_base_url: &str,
    storage_auth_token: &str,
    turn_text: &str,
    config: &KnowledgeAutoConfig,
) -> Result<Option<String>> {
    if turn_text.trim().is_empty() {
        return Ok(None);
    }

    // 1. YAKE
    let yake_cfg = YakeConfig {
        top_n: config.top_n,
        ..Default::default()
    };
    let phrases = yake_extract(turn_text, &yake_cfg);
    eprintln!(
        "🔬 knowledge_auto: turn_text={:?}, yake_phrases={:?}",
        turn_text,
        phrases.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>()
    );
    if phrases.is_empty() {
        return Ok(None);
    }

    // Множество слов из всех фраз YAKE текущего хода.
    let phrases_joined: String = phrases
        .iter()
        .map(|(p, _)| p.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let turn_words = word_set(&phrases_joined);

    let base = ensure_scheme(storage_base_url);
    let base = base.trim_end_matches('/');
    let search_url = format!("{}/knowledge/search", base);

    // 2. Поиск по каждой фразе, агрегация по file_path.
    //    file_phrases[path] — множество РАЗНЫХ фраз, попавших в файл.
    //    file_min_dist[path] — минимальный distance.
    //    Лексический фильтр: чанк учитывается только если его content_fragment
    //    содержит хотя бы одно слово из фраз YAKE.
    let mut file_phrases: HashMap<String, HashSet<String>> = HashMap::new();
    let mut file_min_dist: HashMap<String, f32> = HashMap::new();

    for (phrase, _score) in &phrases {
        let resp = client
            .get(&search_url)
            .query(&[
                ("query", phrase.as_str()),
                ("top_k", &config.per_phrase.to_string()),
            ])
            .header("Authorization", format!("Bearer {}", storage_auth_token))
            .send()
            .await;

        let Ok(resp) = resp else { continue };
        if !resp.status().is_success() {
            continue;
        }

        let results: Vec<serde_json::Value> = match resp.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };

        for r in results {
            let dist = r.get("distance").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            if dist > config.threshold {
                continue;
            }

            // Лексический фильтр: пересечение множеств слов.
            let fragment = r
                .get("content_fragment")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let fragment_words = word_set(fragment);
            if fragment_words.intersection(&turn_words).next().is_none() {
                continue;
            }

            let path = match r.get("file_path").and_then(|v| v.as_str()) {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => continue,
            };
            file_phrases
                .entry(path.clone())
                .or_default()
                .insert(phrase.clone());
            let entry = file_min_dist.entry(path).or_insert(f32::MAX);
            if dist < *entry {
                *entry = dist;
            }
        }
    }

    // 3. Ранжирование: по hits (уникальные фразы) убыв., тай-брейк — по min distance
    let mut ranked: Vec<(String, usize, f32)> = file_phrases
        .into_iter()
        .map(|(path, phrases)| {
            let hits = phrases.len();
            let dist = file_min_dist.get(&path).copied().unwrap_or(1.0);
            (path, hits, dist)
        })
        .collect();

    eprintln!(
        "🔬 knowledge_auto: ranked={:?}",
        ranked
            .iter()
            .map(|(p, h, d)| format!("{} h={} d={:.3}", p, h, d))
            .collect::<Vec<_>>()
    );

    if ranked.is_empty() {
        return Ok(None);
    }

    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    ranked.truncate(config.top_k);

    // 4. Загрузка тел знаний
    let get_url = format!("{}/knowledge/get", base);
    let mut blocks: Vec<String> = Vec::new();

    for (path, _hits, _dist) in &ranked {
        let Some(name) = knowledge_name_from_path(path) else {
            continue;
        };

        let resp = client
            .get(&get_url)
            .query(&[("name", name.as_str())])
            .header("Authorization", format!("Bearer {}", storage_auth_token))
            .send()
            .await;

        let Ok(resp) = resp else { continue };
        if !resp.status().is_success() {
            continue;
        }
        let raw = match resp.text().await {
            Ok(t) => t,
            Err(_) => continue,
        };
        let body = strip_frontmatter(&raw);
        if body.trim().is_empty() {
            continue;
        }
        blocks.push(format!("=== {} ===\n{}", name, body.trim_end()));
    }

    if blocks.is_empty() {
        return Ok(None);
    }

    Ok(Some(format!(
        "[Автоматически подгруженные знания по текущему запросу:]\n\n{}",
        blocks.join("\n\n")
    )))
}

// ==================== Тесты ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_frontmatter_removes_block() {
        let s = "---\nname: x\ndescription: y\n---\n\nbody";
        assert_eq!(strip_frontmatter(s), "body");
    }

    #[test]
    fn strip_frontmatter_without_block_returns_as_is() {
        let s = "just body";
        assert_eq!(strip_frontmatter(s), "just body");
    }

    #[test]
    fn strip_frontmatter_missing_closer_returns_as_is() {
        let s = "---\nname: x\nbody";
        assert_eq!(strip_frontmatter(s), s);
    }

    #[test]
    fn strip_frontmatter_strips_bom() {
        let s = "\u{feff}---\nname: x\n---\nbody";
        assert_eq!(strip_frontmatter(s), "body");
    }

    #[test]
    fn knowledge_name_from_path_unix() {
        assert_eq!(
            knowledge_name_from_path(".knowledge/rust_ownership.md"),
            Some("rust_ownership".to_string())
        );
    }

    #[test]
    fn knowledge_name_from_path_windows() {
        assert_eq!(
            knowledge_name_from_path(".knowledge\\rust_ownership.md"),
            Some("rust_ownership".to_string())
        );
    }

    #[test]
    fn knowledge_name_from_path_without_ext() {
        assert_eq!(
            knowledge_name_from_path(".knowledge/foo"),
            Some("foo".to_string())
        );
    }

    #[test]
    fn ensure_scheme_adds_http() {
        assert_eq!(ensure_scheme("localhost:8080"), "http://localhost:8080");
        assert_eq!(ensure_scheme("http://x"), "http://x");
        assert_eq!(ensure_scheme("https://x"), "https://x");
    }

    #[test]
    fn word_set_lowercases_and_splits() {
        let s = word_set("Work-Stealing в Tokio");
        assert!(s.contains("work-stealing"));
        assert!(s.contains("в"));
        assert!(s.contains("tokio"));
    }

    #[test]
    fn word_set_dedups() {
        let s = word_set("tokio tokio TOKIO");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn word_set_empty() {
        let s = word_set("");
        assert!(s.is_empty());
    }
}
