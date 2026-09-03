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
    pub prompt_file: String, // новое поле: файл с исходным промптом
    pub agent_name: String,
    pub description: String,
    pub prompt: String, // новое поле: исходный промпт
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

fn load_catalog(client: &Client, base_url: &str, auth_token: &str) -> Result<SkillCatalog> {
    let url = format!("{}/files", base_url.trim_end_matches('/'));
    let catalog_path = skill_path("skills_catalog.json");
    let resp = client
        .get(&url)
        .query(&[("path", &catalog_path)])
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
    let url = format!("{}/files", base_url.trim_end_matches('/'));
    let catalog_path = skill_path("skills_catalog.json");
    client
        .post(&url)
        .query(&[("path", &catalog_path)])
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
    let url = format!("{}/files", base_url.trim_end_matches('/'));
    let full_path = skill_path(path);
    client
        .post(&url)
        .query(&[("path", &full_path)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(content.to_string())
        .send()?;
    Ok(())
}

fn read_file(client: &Client, base_url: &str, auth_token: &str, path: &str) -> Result<String> {
    let url = format!("{}/files", base_url.trim_end_matches('/'));
    let full_path = skill_path(path);
    let resp = client
        .get(&url)
        .query(&[("path", &full_path)])
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
    // 1. Загружаем каталог
    let catalog = load_catalog(client, base_url, auth_token)?;
    if catalog.skills.is_empty() {
        eprintln!("DEBUG: каталог скиллов пуст");
        return Ok(None);
    }

    // 2. Строим карту: ключ = нормализованный путь к файлу промпта (без префикса хранилища)
    //    Будем хранить как путь с ".skills/" (как возвращает поиск после нормализации)
    let mut prompt_file_to_record: HashMap<String, SkillRecord> = HashMap::new();
    for rec in &catalog.skills {
        // rec.prompt_file хранится как "skill_agent_..._prompt.txt"
        // Добавляем ".skills/" для соответствия путям из поиска
        let key = format!("{}/{}", SKILLS_DIR, rec.prompt_file.trim_start_matches('/'));
        eprintln!("DEBUG: добавляем в карту ключ: {}", key);
        prompt_file_to_record.insert(key, rec.clone());
    }

    // 3. Семантический поиск по промптам
    let search_url = format!("{}/search", base_url.trim_end_matches('/'));
    let top_k_str = (top_k * 2).to_string(); // берём с запасом
    let resp = client
        .get(&search_url)
        .query(&[("query", prompt), ("top_k", &top_k_str)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Search request failed"));
    }
    let results: Vec<serde_json::Value> = resp.json()?;

    let mut own_candidates: Vec<(SkillRecord, f32)> = Vec::new();
    let mut all_candidates: Vec<(SkillRecord, f32)> = Vec::new();

    for item in results {
        let file_path_raw = item.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        let distance = item.get("distance").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        eprintln!(
            "DEBUG: результат поиска: file_path_raw='{}', distance={:.4}",
            file_path_raw, distance
        );

        // Нормализуем путь: убираем возможные префиксы хранилища, оставляем только путь от корня .skills/
        // Пример: ".local_storage/default_storage/.skills/skill_..._prompt.txt" -> ".skills/skill_..._prompt.txt"
        let file_path = file_path_raw
            .split('/')
            .skip_while(|seg| *seg != SKILLS_DIR && !seg.ends_with(".txt")) // грубо, но для отладки
            .collect::<Vec<_>>()
            .join("/");

        // Более надёжно: найти позицию ".skills/" и взять оттуда
        let file_path = if let Some(pos) = file_path_raw.find(SKILLS_DIR) {
            &file_path_raw[pos..]
        } else {
            file_path_raw
        };
        eprintln!("DEBUG: нормализованный путь: {}", file_path);

        if distance > semantic_threshold {
            eprintln!(
                "DEBUG: расстояние {} > порога {} — пропускаем",
                distance, semantic_threshold
            );
            continue;
        }

        if let Some(rec) = prompt_file_to_record.get(file_path) {
            eprintln!(
                "DEBUG: НАЙДЕНО совпадение для path='{}', rec.agent={}",
                file_path, rec.agent_name
            );
            if rec.agent_name == current_agent {
                own_candidates.push((rec.clone(), distance));
            }
            all_candidates.push((rec.clone(), distance));
        } else {
            eprintln!("DEBUG: совпадений в каталоге нет для path='{}'", file_path);
        }
    }

    // 4. Выбираем лучшего: сначала свои, потом все
    if let Some(best_own) = select_best_by_success_rate(&own_candidates) {
        eprintln!("DEBUG: выбран свой скилл: {}", best_own.0.skill_file);
        return Ok(Some(best_own));
    }
    if let Some(best_all) = select_best_by_success_rate(&all_candidates) {
        eprintln!("DEBUG: выбран чужой скилл: {}", best_all.0.skill_file);
        return Ok(Some(best_all));
    }

    eprintln!("DEBUG: подходящий скилл не найден");
    Ok(None)
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
) -> Result<()> {
    if rhai_code.trim().len() < min_code_length {
        return Ok(());
    }

    let ast_hash = compute_ast_hash(rhai_code);
    if ast_hash.is_empty() {
        return Err(anyhow::anyhow!("Invalid Rhai code"));
    }

    // В автоматическом режиме поиск уже был выполнен перед вызовом этой функции,
    // поэтому если мы здесь, значит подходящего скилла нет — сохраняем без дополнительных проверок.

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

    // Основной файл скилла
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

    // Файл описания (для совместимости или ручного режима)
    write_file(client, base_url, auth_token, &description_file, description)?;

    // Файл промпта (для семантического поиска)
    write_file(client, base_url, auth_token, &prompt_file, prompt)?;

    // Обновляем каталог
    let mut catalog = load_catalog(client, base_url, auth_token)?;
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

    Ok(())
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
