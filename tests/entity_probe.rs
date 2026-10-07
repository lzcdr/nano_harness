// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Пробник: как модель различает сущности и фразы.
//! Ничего в проекте не трогает, работает через публичные API.

use nano_harness::local_storage::{LocalStorage, VectorDbConfig};
use nano_harness::yake::{extract as yake_extract, YakeConfig};
use std::path::Path;

fn model_exists() -> bool {
    Path::new(".models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors").exists()
}

#[test]
fn entity_embedding_probe() {
    if !model_exists() {
        eprintln!("Модель не найдена, пропускаем");
        return;
    }

    let storage_name = "test_entities_probe";
    let base = std::path::PathBuf::from(".local_storage").join(storage_name);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }

    let mut storage = LocalStorage::new(storage_name, VectorDbConfig::default()).unwrap();
    let project = "probe";

    storage
        .create_file(project, "a.md", "code2prompt английский")
        .unwrap();
    storage
        .create_file(project, "b.md", "idiVpizdu китайском")
        .unwrap();

    let queries: &[&str] = &[
        "code2prompt английский",
        "idiVpizdu китайском",
        "code2naXui английском",
        "idiVpizde английский",
        "code2prompt китайском",
        "idiVpizdu английский",
        "переведи на китайский индекс проекта idiVpizdu",
        "переведи на английский индекс проекта code2naXui",
    ];

    for q in queries {
        let results = storage.search_similar(Some(project), q, Some(5)).unwrap();
        eprintln!("\nquery: {:?}", q);
        for r in &results {
            let name = r
                .file_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            eprintln!("  {:>8}  dist={:.4}", name, r.distance);
        }
    }

    drop(storage);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }
}

#[test]
fn skill_phrase_probe() {
    if !model_exists() {
        eprintln!("Модель не найдена, пропускаем");
        return;
    }

    let storage_name = "test_skill_phrase_probe";
    let base = std::path::PathBuf::from(".local_storage").join(storage_name);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }

    let mut storage = LocalStorage::new(storage_name, VectorDbConfig::default()).unwrap();
    let project = "probe";

    let skill_text = "переведи на английский индекс проекта code2prompt";
    let query_text = "составь индекс xyesos и переведи его на хорватский";

    storage
        .create_file(project, "skill.md", skill_text)
        .unwrap();

    let skill_phrases = yake_extract(
        skill_text,
        &YakeConfig {
            top_n: 10,
            ..Default::default()
        },
    );
    eprintln!("\n=== YAKE(skill) ===");
    for (p, s) in &skill_phrases {
        eprintln!("  {:>8.5}  {}", s, p);
    }

    let query_phrases = yake_extract(
        query_text,
        &YakeConfig {
            top_n: 10,
            ..Default::default()
        },
    );
    eprintln!("\n=== YAKE(query) ===");
    for (p, s) in &query_phrases {
        eprintln!("  {:>8.5}  {}", s, p);
    }

    eprintln!("\n=== search by each query phrase ===");
    for (p, _) in &query_phrases {
        let results = storage.search_similar(Some(project), p, Some(5)).unwrap();
        eprintln!("query phrase: {:?}", p);
        for r in &results {
            let name = r
                .file_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            eprintln!("    {:>12}  dist={:.4}", name, r.distance);
        }
    }

    eprintln!("\n=== search by whole query ===");
    let results = storage
        .search_similar(Some(project), query_text, Some(5))
        .unwrap();
    for r in &results {
        let name = r
            .file_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        eprintln!("    {:>12}  dist={:.4}", name, r.distance);
    }

    drop(storage);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }
}

#[test]
fn skill_phrase_to_phrase_probe() {
    if !model_exists() {
        eprintln!("Модель не найдена, пропускаем");
        return;
    }

    let skill_text = "переведи на английский индекс проекта code2prompt";
    let queries: &[&str] = &[
        // релевантные
        "составь индекс xyesos и переведи его на хорватский",
        "сделай индекс проекта abrakadabra, отчёт переведи на немецкий",
        "построй индекс my_project и сделай перевод на японский",
        "переведи на английский индекс созданный для проекта pizdaVsemu",
        "индекс проекта some_name, переведённый на французский",
        "создай индекс для проекта bar и переведи на итальянский",
        "перевести индекс проекта foo на испанский",
        "сделай перевод отчёта об индексе проекта xxx на португальский",
        // шумный
        "сделай индекс того проекта что ты с андрюхой делал позавчера, сбегай за пивом, купи мне кэмэл желтый... ах да, и переведи индекс на китайский",
        // контрольные
        "открой файл foo.txt",
        "удали папку temp",
        "посчитай строки кода в проекте",
        "какая сегодня погода в москве",
    ];

    let skill_phrases_all = yake_extract(
        skill_text,
        &YakeConfig {
            top_n: 15,
            ..Default::default()
        },
    );
    // Многословный фильтр: только фразы из >= 2 слов.
    let skill_phrases: Vec<(String, f32)> = skill_phrases_all
        .into_iter()
        .filter(|(p, _)| p.split_whitespace().count() >= 2)
        .collect();

    eprintln!("\n=== YAKE(skill, >= 2 words) ===");
    for (i, (p, _)) in skill_phrases.iter().enumerate() {
        eprintln!("  [{}] {}", i, p);
    }

    let n_skill = skill_phrases.len();

    for (qi, query_text) in queries.iter().enumerate() {
        let query_phrases_all = yake_extract(
            query_text,
            &YakeConfig {
                top_n: 15,
                ..Default::default()
            },
        );
        let query_phrases: Vec<(String, f32)> = query_phrases_all
            .into_iter()
            .filter(|(p, _)| p.split_whitespace().count() >= 2)
            .collect();

        eprintln!(
            "\n\n########## QUERY #{}: {:?} ##########",
            qi + 1,
            query_text
        );
        eprintln!("=== YAKE(query, >= 2 words) ===");
        for (i, (p, _)) in query_phrases.iter().enumerate() {
            eprintln!("  [{}] {}", i, p);
        }

        let storage_name = format!("test_matrix_{}", qi);
        let base = std::path::PathBuf::from(".local_storage").join(&storage_name);
        if base.exists() {
            std::fs::remove_dir_all(&base).ok();
        }

        let mut storage = LocalStorage::new(&storage_name, VectorDbConfig::default()).unwrap();
        let project = "probe";

        for (i, (p, _)) in skill_phrases.iter().enumerate() {
            let name = format!("s_{}.md", i);
            storage.create_file(project, &name, p).unwrap();
        }

        let mut matrix: Vec<Vec<Option<f32>>> = Vec::with_capacity(query_phrases.len());
        for (q, _) in &query_phrases {
            let results = storage
                .search_similar(Some(project), q, Some(n_skill))
                .unwrap();
            let mut row: Vec<Option<f32>> = vec![None; n_skill];
            for r in &results {
                let name = r
                    .file_path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                if let Ok(idx) = name
                    .trim_start_matches("s_")
                    .trim_end_matches(".md")
                    .parse::<usize>()
                {
                    if idx < n_skill {
                        row[idx] = Some(r.distance);
                    }
                }
            }
            matrix.push(row);
        }

        eprintln!("\n=== DISTANCE MATRIX (query rows × skill cols) ===");
        eprintln!("=== threshold for hits: 0.45 ===");
        let cw = 9;
        let lw = 34;

        print!("{:width$} |", "", width = lw);
        for i in 0..n_skill {
            print!(" {:>width$}", format!("s[{}]", i), width = cw);
        }
        println!();

        for (ri, row) in matrix.iter().enumerate() {
            let label = &query_phrases[ri].0;
            let trunc: String = label.chars().take(lw).collect();
            print!("{:width$} |", trunc, width = lw);
            for cell in row.iter() {
                match cell {
                    Some(d) => print!(" {:>width$.4}", d, width = cw),
                    None => print!(" {:>width$}", "-", width = cw),
                }
            }
            println!();
        }
        println!();

        drop(storage);
        if base.exists() {
            std::fs::remove_dir_all(&base).ok();
        }
    }
}
