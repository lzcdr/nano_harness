// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use nano_harness::local_storage::{LocalStorage, VectorDbConfig, TOOLS_PROJECT};
use nano_harness::yake::{extract as yake_extract, YakeConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const TOP_N: usize = 3;
const THRESHOLD: f32 = 0.45;

fn model_exists() -> bool {
    Path::new(".models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors").exists()
}

fn seed_tools(storage: &mut LocalStorage) {
    storage.clear_tools_dir().unwrap();
    for tool in nano_harness::tool_runtime::all() {
        let name = tool.name();
        let content = format!("{} {}", name, tool.description());
        storage.write_tool_file(name, &content).unwrap();
    }
}

fn score_for(hits: usize, min_dist: f32) -> f32 {
    if hits == 0 {
        return 0.0;
    }
    let h = hits as f32;
    (1.0 - min_dist) * h / (h + 0.5)
}

fn run_query(storage: &LocalStorage, query: &str) -> Vec<(String, usize, f32, f32)> {
    let phrases_all = yake_extract(
        query,
        &YakeConfig {
            top_n: 7,
            ..Default::default()
        },
    );
    let phrases: Vec<String> = phrases_all
        .into_iter()
        .map(|(p, _)| p)
        .filter(|p| p.split_whitespace().count() >= 2)
        .collect();

    let mut hits: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut min_dist: HashMap<String, f32> = HashMap::new();

    for (idx, phrase) in phrases.iter().enumerate() {
        let results = storage
            .search_similar(Some(TOOLS_PROJECT), phrase, Some(100))
            .unwrap();
        for r in &results {
            let name = r
                .file_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            hits.entry(name.clone()).or_default().insert(idx);
            let e = min_dist.entry(name).or_insert(f32::MAX);
            if r.distance < *e {
                *e = r.distance;
            }
        }
    }

    let mut ranked: Vec<(String, usize, f32, f32)> = hits
        .into_iter()
        .map(|(name, set)| {
            let h = set.len();
            let d = min_dist.get(&name).copied().unwrap_or(1.0);
            (name, h, d, score_for(h, d))
        })
        .collect();
    ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

#[test]
fn tool_search_top_n_probe() {
    if !model_exists() {
        eprintln!("Модель не найдена, пропускаем");
        return;
    }

    let storage_name = "test_tool_search_topn";
    let base = std::path::PathBuf::from(".local_storage").join(storage_name);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }

    let mut storage = LocalStorage::new(storage_name, VectorDbConfig::default()).unwrap();
    seed_tools(&mut storage);

    let queries: &[&str] = &[
        "прочитай файл README.md",
        "запиши файл config.txt с текстом hello",
        "удали файл foo.txt",
        "создай директорию src/new",
        "покажи список файлов в src",
        "рекурсивно обойди весь проект",
        "найди файл с именем Cargo",
        "найди что-нибудь похожее на \"асинхронный рантайм\"",
        "найди все упоминания tokio в проекте",
        "посчитай строки кода",
        "проиндексируй символы проекта",
        "составь индекс проекта",
        "вызови агента analyzer для анализа кода",
        "отправь асинхронную задачу агенту coder",
        "какая сегодня погода в москве",
        "привет, как дела",
        "расскажи анекдот",
    ];

    for query in queries {
        let ranked = run_query(&storage, query);
        eprintln!("\nQUERY: {:?}", query);
        if ranked.is_empty() {
            eprintln!("  (нет кандидатов)");
            continue;
        }
        for (i, (name, h, d, score)) in ranked.iter().take(TOP_N).enumerate() {
            eprintln!(
                "  {}. {:>28}  score={:.4}  (hits={}, min_dist={:.4})",
                i + 1,
                name,
                score,
                h,
                d
            );
        }
    }

    drop(storage);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }
}
