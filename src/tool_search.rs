// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_search.rs

use anyhow::Result;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use crate::yake::{extract as yake_extract, YakeConfig};

// ==================== Конфиг ====================

#[derive(Debug, Clone)]
pub struct ToolsSearchConfig {
    pub top_n: usize,
    pub phrase_min_words: usize,
    pub threshold: f32,
    pub slots: usize,
    pub min_score: f32,
    pub per_query_top_k: usize,
}

impl Default for ToolsSearchConfig {
    fn default() -> Self {
        Self {
            top_n: 7,
            phrase_min_words: 2,
            threshold: 0.45,
            slots: 4,
            min_score: 0.20,
            per_query_top_k: 5,
        }
    }
}

// ==================== Служебное ====================

fn ensure_scheme(base_url: &str) -> String {
    if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    }
}

/// Из `.tools/storage_read_file.md` достаёт `storage_read_file`.
/// Универсально для unix- и windows-разделителей.
fn tool_name_from_path(path: &str) -> Option<String> {
    let name = path.rsplit(|c| c == '/' || c == '\\').next()?;
    name.strip_suffix(".md").map(|s| s.to_string())
}

// ==================== Чистые хелперы ====================

/// YAKE по промпту, фильтр по phrase_min_words.
fn extract_phrases(prompt: &str, config: &ToolsSearchConfig) -> Vec<String> {
    let phrases = yake_extract(
        prompt,
        &YakeConfig {
            top_n: config.top_n,
            ..Default::default()
        },
    );
    phrases
        .into_iter()
        .map(|(p, _)| p)
        .filter(|p| p.split_whitespace().count() >= config.phrase_min_words)
        .collect()
}

/// Скоринг по (hits, min_distance). Формула из config.toml.
fn score_for(hits: usize, min_dist: f32) -> f32 {
    if hits == 0 {
        return 0.0;
    }
    let h = hits as f32;
    (1.0 - min_dist) * h / (h + 0.5)
}

/// Агрегация hits/min_dist по имени тулза.
fn aggregate(per_phrase_hits: &[Vec<(String, f32)>]) -> HashMap<String, (HashSet<usize>, f32)> {
    let mut acc: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
    for (phrase_idx, hits) in per_phrase_hits.iter().enumerate() {
        for (name, dist) in hits {
            let entry = acc
                .entry(name.clone())
                .or_insert((HashSet::new(), f32::MAX));
            entry.0.insert(phrase_idx);
            if *dist < entry.1 {
                entry.1 = *dist;
            }
        }
    }
    acc
}

/// Пул = кандидаты (hits > 0) ∪ current_slots, дедуп по имени.
/// Сортировка по score ↓, отсев по min_score, top-N = slots.
fn rank_pool(
    agg: &HashMap<String, (HashSet<usize>, f32)>,
    current_slots: &[String],
    allowed: &HashSet<String>,
    slots_n: usize,
    min_score: f32,
) -> Vec<String> {
    let mut pool: HashMap<String, f32> = HashMap::new();
    for (name, (phrase_set, min_dist)) in agg {
        pool.insert(name.clone(), score_for(phrase_set.len(), *min_dist));
    }
    for slot in current_slots {
        if !allowed.contains(slot) {
            continue;
        }
        pool.entry(slot.clone()).or_insert(0.0);
    }
    let mut ranked: Vec<(String, f32)> =
        pool.into_iter().filter(|(_, s)| *s >= min_score).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(slots_n);
    ranked.into_iter().map(|(n, _)| n).collect()
}

// ==================== HTTP-ответ /tools/search ====================

#[derive(Debug, Deserialize)]
struct ToolSearchHit {
    file_path: String,
    distance: f32,
}

// ==================== Публичная функция ====================

/// Отбирает тулзы под текущий промпт.
///
/// Возвращает список имён из `allowed`. Пустой результат означает
/// «ничего релевантного не нашлось, слоты не трогаем» — вызывающий код
/// должен в этом случае оставить `visible_tools` без изменений.
pub fn select_tools(
    client: &Client,
    storage_base_url: &str,
    storage_auth_token: &str,
    prompt: &str,
    current_slots: &[String],
    allowed: &[String],
    config: &ToolsSearchConfig,
) -> Result<Vec<String>> {
    if allowed.is_empty() {
        return Ok(Vec::new());
    }

    let allowed_set: HashSet<String> = allowed.iter().cloned().collect();

    // Два особых тулза, всегда доступные LLM (если в allowed).
    // Не участвуют в слотах, добавляются в конец результата.
    const ALWAYS_VISIBLE: &[&str] = &["call_agent", "post_task"];

    let always: Vec<String> = ALWAYS_VISIBLE
        .iter()
        .filter(|n| allowed_set.contains(**n))
        .map(|s| s.to_string())
        .collect();

    let phrases = extract_phrases(prompt, config);

    let mut result = if phrases.is_empty() {
        // YAKE не дал ни одной фразы — слоты сохраняются, поиск не идёт.
        current_slots.to_vec()
    } else {
        let base = ensure_scheme(storage_base_url);
        let url = format!("{}/tools/search", base.trim_end_matches('/'));

        let top_k_str = config.per_query_top_k.to_string();
        let mut per_phrase_hits: Vec<Vec<(String, f32)>> = Vec::with_capacity(phrases.len());

        for phrase in &phrases {
            let resp = client
                .get(&url)
                .query(&[("query", phrase.as_str()), ("top_k", top_k_str.as_str())])
                .header("Authorization", format!("Bearer {}", storage_auth_token))
                .send()?
                .error_for_status()?;
            let raw: Vec<ToolSearchHit> = resp.json()?;
            let mut filtered: Vec<(String, f32)> = Vec::new();
            for hit in raw {
                if hit.distance > config.threshold {
                    continue;
                }
                let Some(name) = tool_name_from_path(&hit.file_path) else {
                    continue;
                };
                if !allowed_set.contains(&name) {
                    continue;
                }
                filtered.push((name, hit.distance));
            }
            per_phrase_hits.push(filtered);
        }

        let agg = aggregate(&per_phrase_hits);

        if agg.is_empty() {
            // Правило «поиск пуст — слоты не трогаем».
            current_slots.to_vec()
        } else {
            rank_pool(
                &agg,
                current_slots,
                &allowed_set,
                config.slots,
                config.min_score,
            )
        }
    };

    // always_visible добавляются всегда — независимо от результата поиска.
    for name in always {
        if !result.contains(&name) {
            result.push(name);
        }
    }

    Ok(result)
}

// ==================== Тесты ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name_from_path_unix() {
        assert_eq!(
            tool_name_from_path(".tools/storage_read_file.md"),
            Some("storage_read_file".to_string())
        );
    }

    #[test]
    fn tool_name_from_path_windows() {
        assert_eq!(
            tool_name_from_path(".tools\\storage_walk.md"),
            Some("storage_walk".to_string())
        );
    }

    #[test]
    fn tool_name_from_path_no_dir() {
        assert_eq!(
            tool_name_from_path("ripgrep.md"),
            Some("ripgrep".to_string())
        );
    }

    #[test]
    fn tool_name_from_path_not_md() {
        assert_eq!(tool_name_from_path(".tools/foo.txt"), None);
        assert_eq!(tool_name_from_path(".tools/foo"), None);
    }

    #[test]
    fn score_zero_hits() {
        assert_eq!(score_for(0, 0.1), 0.0);
    }

    #[test]
    fn score_formula() {
        let s = score_for(1, 0.2);
        assert!((s - 0.53333).abs() < 1e-4);

        let s = score_for(4, 0.2);
        assert!((s - 0.71111).abs() < 1e-4);
    }

    #[test]
    fn score_high_hits_dominates_low_hits() {
        assert!(score_for(4, 0.3) > score_for(1, 0.3));
    }

    #[test]
    fn score_lower_distance_dominates() {
        assert!(score_for(2, 0.1) > score_for(2, 0.4));
    }

    #[test]
    fn aggregate_merges_by_name() {
        let per_phrase = vec![
            vec![("tool_a".into(), 0.2), ("tool_b".into(), 0.3)],
            vec![("tool_a".into(), 0.15)],
        ];
        let agg = aggregate(&per_phrase);
        let a = agg.get("tool_a").unwrap();
        assert_eq!(a.0.len(), 2);
        assert!((a.1 - 0.15).abs() < 1e-6);

        let b = agg.get("tool_b").unwrap();
        assert_eq!(b.0.len(), 1);
        assert!((b.1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn aggregate_empty() {
        assert!(aggregate(&[]).is_empty());
    }

    #[test]
    fn rank_candidates_beat_slots() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        agg.insert("cand".into(), (vec![0, 1].into_iter().collect(), 0.2));
        let slots = vec!["slot".to_string()];
        let allowed: HashSet<String> = ["cand".to_string(), "slot".to_string()]
            .into_iter()
            .collect();
        let ranked = rank_pool(&agg, &slots, &allowed, 4, 0.20);
        assert_eq!(ranked, vec!["cand"]);
    }

    #[test]
    fn rank_slot_survives_when_min_score_zero() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        agg.insert("cand".into(), (vec![0].into_iter().collect(), 0.2));
        let slots = vec!["slot".to_string()];
        let allowed: HashSet<String> = ["cand".to_string(), "slot".to_string()]
            .into_iter()
            .collect();
        let ranked = rank_pool(&agg, &slots, &allowed, 4, 0.0);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn rank_empty_candidates_returns_slots() {
        let agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        let slots = vec!["a".to_string(), "b".to_string()];
        let allowed: HashSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
        let ranked = rank_pool(&agg, &slots, &allowed, 4, 0.0);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn rank_truncates_to_slots_n() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        for i in 0..10 {
            agg.insert(format!("t{}", i), (vec![0].into_iter().collect(), 0.2));
        }
        let ranked = rank_pool(&agg, &[], &HashSet::new(), 4, 0.20);
        assert_eq!(ranked.len(), 4);
    }

    #[test]
    fn rank_slot_not_in_allowed_is_dropped() {
        let agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        let slots = vec!["bad".to_string()];
        let allowed: HashSet<String> = ["good".to_string()].into_iter().collect();
        let ranked = rank_pool(&agg, &slots, &allowed, 4, 0.0);
        assert!(ranked.is_empty());
    }

    #[test]
    fn rank_dedup_by_name() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        agg.insert("same".into(), (vec![0].into_iter().collect(), 0.2));
        let slots = vec!["same".to_string()];
        let allowed: HashSet<String> = ["same".to_string()].into_iter().collect();
        let ranked = rank_pool(&agg, &slots, &allowed, 4, 0.20);
        assert_eq!(ranked, vec!["same"]);
    }

    #[test]
    fn rank_min_score_filters_low_scores() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        agg.insert("low".into(), (vec![0].into_iter().collect(), 0.9));
        agg.insert("high".into(), (vec![0].into_iter().collect(), 0.3));
        let ranked = rank_pool(&agg, &[], &HashSet::new(), 4, 0.20);
        assert_eq!(ranked, vec!["high"]);
    }

    #[test]
    fn rank_all_below_min_score_returns_empty() {
        let mut agg: HashMap<String, (HashSet<usize>, f32)> = HashMap::new();
        agg.insert("a".into(), (vec![0].into_iter().collect(), 0.9));
        agg.insert("b".into(), (vec![0].into_iter().collect(), 0.95));
        let ranked = rank_pool(&agg, &[], &HashSet::new(), 4, 0.20);
        assert!(ranked.is_empty());
    }
}
