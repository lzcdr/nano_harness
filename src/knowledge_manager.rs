// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_name_accepts_alphanumeric() {
        assert!(validate_name("rust"));
        assert!(validate_name("my_knowledge"));
        assert!(validate_name("my-knowledge"));
        assert!(validate_name("rust2026"));
    }

    #[test]
    fn validate_name_rejects_empty() {
        assert!(!validate_name(""));
    }

    #[test]
    fn validate_name_rejects_spaces_and_punct() {
        assert!(!validate_name("my knowledge"));
        assert!(!validate_name("my/knowledge"));
        assert!(!validate_name("my.knowledge"));
    }

    #[test]
    fn validate_name_rejects_cyrillic() {
        assert!(!validate_name("знание"));
    }

    #[test]
    fn parse_frontmatter_ok() {
        let content = "---\nname: rust\ndescription: язык\n---\n\nтело";
        let (name, desc, body) = parse_frontmatter(content).unwrap();
        assert_eq!(name, "rust");
        assert_eq!(desc, "язык");
        assert_eq!(body, "тело");
    }

    #[test]
    fn parse_frontmatter_missing_opener_errors() {
        assert!(parse_frontmatter("name: rust\n---\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_missing_closer_errors() {
        assert!(parse_frontmatter("---\nname: rust\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_missing_name_errors() {
        assert!(parse_frontmatter("---\ndescription: x\n---\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_missing_description_errors() {
        assert!(parse_frontmatter("---\nname: x\n---\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_empty_name_errors() {
        assert!(parse_frontmatter("---\nname: \ndescription: x\n---\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_empty_description_errors() {
        assert!(parse_frontmatter("---\nname: x\ndescription: \n---\nbody").is_err());
    }

    #[test]
    fn parse_frontmatter_strips_bom() {
        let content = "\u{feff}---\nname: x\ndescription: y\n---\nbody";
        assert!(parse_frontmatter(content).is_ok());
    }

    #[test]
    fn parse_frontmatter_body_strips_leading_newlines() {
        let content = "---\nname: x\ndescription: y\n---\n\n\n\nbody";
        let (_, _, body) = parse_frontmatter(content).unwrap();
        assert_eq!(body, "body");
    }

    #[test]
    fn build_catalog_prompt_empty() {
        assert_eq!(build_catalog_prompt(&[]), "");
    }

    #[test]
    fn build_catalog_prompt_lists_entries() {
        let entries = vec![
            KnowledgeEntry {
                name: "a".into(),
                description: "first".into(),
            },
            KnowledgeEntry {
                name: "b".into(),
                description: "second".into(),
            },
        ];
        let prompt = build_catalog_prompt(&entries);
        assert!(prompt.contains("- a — first"));
        assert!(prompt.contains("- b — second"));
        assert!(prompt.contains("knowledge_load"));
        assert!(prompt.contains("knowledge_unload"));
    }

    #[test]
    fn ensure_scheme_adds_http() {
        assert_eq!(ensure_scheme("localhost:8080"), "http://localhost:8080");
    }

    #[test]
    fn ensure_scheme_keeps_http() {
        assert_eq!(ensure_scheme("http://x"), "http://x");
    }

    #[test]
    fn ensure_scheme_keeps_https() {
        assert_eq!(ensure_scheme("https://x"), "https://x");
    }
}
