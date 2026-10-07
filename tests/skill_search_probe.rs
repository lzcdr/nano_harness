// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Интеграционный тест: skill_manager::search_best_skill против живого хранилища.

use nano_harness::local_storage::{LocalStorage, VectorDbConfig, SKILLS_PROJECT};
use std::path::Path;

fn model_exists() -> bool {
    Path::new(".models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors").exists()
}

fn seed_skill_index(storage: &mut LocalStorage, stem: &str, phrases: &[&str]) {
    for (i, phrase) in phrases.iter().enumerate() {
        let name = format!("{}_index_{}.idx", stem, i);
        storage.create_skill_file(&name, phrase).unwrap();
    }
}

#[test]
fn search_best_skill_probe() {
    if !model_exists() {
        eprintln!("Модель не найдена, пропускаем");
        return;
    }

    let storage_name = "test_skill_search_probe";
    let base = std::path::PathBuf::from(".local_storage").join(storage_name);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }

    let mut storage = LocalStorage::new(storage_name, VectorDbConfig::default()).unwrap();

    // Кладём в индекс две фразы одного скилла.
    // Идентификатор stem — без .md (LocalStorage работает от относительных путей).
    let stem_a = "skill_agent_analyzer_build_index_123";
    let phrases_a = [
        "индекс проекта",
        "переведи на английский",
        "английский индекс",
        "индекс проекта code2prompt",
        "создать индекс для проекта",
        "перевести отчёт на английский",
    ];
    seed_skill_index(&mut storage, stem_a, &phrases_a);

    // Второй скилл — про другое, должен не находиться на запрос про индекс.
    let stem_b = "skill_agent_analyzer_count_lines_456";
    let phrases_b = [
        "посчитай строки кода",
        "количество строк в проекте",
        "статистика по коду",
    ];
    seed_skill_index(&mut storage, stem_b, &phrases_b);

    // Проверяем через живой HTTP API через локальный client.
    // Но для простоты — используем LocalStorage::search_similar напрямую
    // и агрегируем hits вручную как это делает skill_manager.

    let queries = [
        ("переведи на немецкий индекс проекта some_name", stem_a),
        ("создай индекс для нового проекта bar и переведи на испанский", stem_a),
        ("посчитай строки кода в проекте", stem_b),
        ("какая сегодня погода в москве", "none"),
    ];

    for (query, expected) in &queries {
        let phrases = nano_harness::yake::extract(
            query,
            &nano_harness::yake::YakeConfig { top_n: 15, ..Default::default() },
        );
        let phrases: Vec<String> = phrases
            .into_iter()
            .map(|(p, _)| p)
            .filter(|p| p.split_whitespace().count() >= 2)
            .collect();

        eprintln!("\n=== query: {:?} (expected: {}) ===", query, expected);
        eprintln!("yake phrases: {:?}", phrases);

        // hits по stem
        let mut hits: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        let mut min_dist: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();

        for phrase in &phrases {
            let results = storage
                .search_similar(Some(SKILLS_PROJECT), phrase, Some(100))
                .unwrap();
            for r in results {
                if r.distance > 0.45 {
                    continue;
                }
                let path_str = r.file_path.to_string_lossy().to_string();
                // Из имени `stem_index_N.idx` получаем stem.
                let name = path_str
                    .rsplit(|c| c == '/' || c == '\\')
                    .next()
                    .unwrap_or("")
                    .to_string();
                let stem = match name.strip_suffix(".idx") {
                    Some(s) => match s.rsplit_once('_') {
                        Some((without_n, _)) => match without_n.strip_suffix("_index") {
                            Some(base) => base.to_string(),
                            None => continue,
                        },
                        None => continue,
                    },
                    None => continue,
                };
                hits.entry(stem.clone())
                    .or_default()
                    .insert(phrase.clone());
                let e = min_dist.entry(stem).or_insert(f32::MAX);
                if r.distance < *e {
                    *e = r.distance;
                }
            }
        }

        for (stem, set) in &hits {
            eprintln!(
                "  {} hits={} min_dist={:.3}",
                stem,
                set.len(),
                min_dist.get(stem).copied().unwrap_or(1.0)
            );
        }

        let best = hits
            .iter()
            .filter(|(_, set)| set.len() >= 3)
            .max_by_key(|(_, set)| set.len());

        match best {
            Some((stem, _)) => eprintln!("  >>> winner: {}", stem),
            None => eprintln!("  >>> winner: none"),
        }
    }

    drop(storage);
    if base.exists() {
        std::fs::remove_dir_all(&base).ok();
    }
}
