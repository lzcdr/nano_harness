// src/knowledge_manager.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

const KNOWLEDGE_DIR: &str = ".knowledge";

fn ensure_scheme(base_url: &str) -> String {
    if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    }
}

pub fn validate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub name: String,
    pub description: String,
}

/// Разбирает файл знания. Формат:
///
/// ```text
/// ---
/// name: <имя>
/// description: <описание>
/// ---
///
/// <тело>
/// ```
///
/// Возвращает (name, description, body).
pub fn parse_frontmatter(content: &str) -> Result<(String, String, String)> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut lines = content.lines();

    let first = lines.next().unwrap_or("").trim();
    if first != "---" {
        anyhow::bail!("frontmatter должен начинаться с '---'");
    }

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;

    loop {
        let line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("frontmatter не закрыт: нет второго '---'"))?;
        let trimmed = line.trim();
        if trimmed == "---" {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("name:") {
            name = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("description:") {
            description = Some(rest.trim().to_string());
        }
    }

    let name = name.ok_or_else(|| anyhow::anyhow!("в frontmatter нет поля 'name'"))?;
    let description =
        description.ok_or_else(|| anyhow::anyhow!("в frontmatter нет поля 'description'"))?;

    if name.is_empty() {
        anyhow::bail!("поле 'name' пустое");
    }
    if description.is_empty() {
        anyhow::bail!("поле 'description' пустое");
    }

    let body: String = lines.collect::<Vec<_>>().join("\n");
    let body = body.trim_start_matches('\n').to_string();

    Ok((name, description, body))
}

pub fn build_catalog_prompt(entries: &[KnowledgeEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("У тебя есть база знаний. Каталог:\n\n");
    for e in entries {
        out.push_str(&format!("- {} — {}\n", e.name, e.description));
    }
    out.push_str(
        "\nЕсли знание кажется релевантным текущей задаче, вызови knowledge_load(name). \
         Загруженное знание попадёт в контекст и будет доступно в следующих ходах. \
         Чтобы снять знание, вызови knowledge_unload().",
    );
    out
}

pub async fn list_knowledge(
    client: &Client,
    base_url: &str,
    auth_token: &str,
) -> Result<Vec<KnowledgeEntry>> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/knowledge/list", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()
        .await
        .context("ошибка запроса списка знаний")?;
    if !resp.status().is_success() {
        return Ok(vec![]);
    }
    let entries: Vec<KnowledgeEntry> = resp.json().await?;
    Ok(entries)
}

pub async fn load_knowledge_raw(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    name: &str,
) -> Result<String> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/knowledge/get", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("name", name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()
        .await
        .context("ошибка запроса знания")?;
    if !resp.status().is_success() {
        anyhow::bail!("знание '{}' не найдено (HTTP {})", name, resp.status());
    }
    Ok(resp.text().await?)
}

pub async fn load_knowledge(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    name: &str,
) -> Result<String> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/knowledge/get", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("name", name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()
        .await
        .context("ошибка запроса знания")?;
    if !resp.status().is_success() {
        anyhow::bail!("знание '{}' не найдено (HTTP {})", name, resp.status());
    }
    let raw = resp.text().await?;
    let (_, _, body) = parse_frontmatter(&raw)?;
    Ok(body)
}

pub async fn save_knowledge(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    name: &str,
    content: &str,
) -> Result<()> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/knowledge/put", base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .query(&[("name", name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(content.to_string())
        .send()
        .await
        .context("ошибка сохранения знания")?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "не удалось сохранить знание '{}': HTTP {} — {}",
            name,
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }
    Ok(())
}

pub async fn delete_knowledge(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    name: &str,
) -> Result<()> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/knowledge/delete", base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .query(&[("name", name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()
        .await
        .context("ошибка удаления знания")?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "не удалось удалить знание '{}': HTTP {}",
            name,
            resp.status()
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub const DIR: &str = KNOWLEDGE_DIR;
