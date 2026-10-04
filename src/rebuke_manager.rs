// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/rebuke_manager.rs

use anyhow::Result;
use reqwest::Client;

const REBUKES_DIR: &str = ".rebukes";

fn ensure_scheme(base_url: &str) -> String {
    if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    }
}

pub async fn load_rebuke(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    agent_name: &str,
) -> Result<String> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/rebukes/get", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[("agent", agent_name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok(resp.text().await?)
    } else {
        Ok(String::new())
    }
}

pub async fn save_rebuke(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    agent_name: &str,
    content: &str,
) -> Result<()> {
    let base = ensure_scheme(base_url);
    let url = format!("{}/rebukes/put", base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .query(&[("agent", agent_name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(content.to_string())
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "не удалось сохранить rebuke для '{}': HTTP {}",
            agent_name,
            resp.status()
        );
    }
    Ok(())
}

pub async fn append_rebuke(
    client: &Client,
    base_url: &str,
    auth_token: &str,
    agent_name: &str,
    text: &str,
) -> Result<()> {
    let existing = load_rebuke(client, base_url, auth_token, agent_name).await?;

    let now = chrono::Local::now().format("%Y-%m-%d %H:%M");
    let separator = format!("---[{}]--------", now);
    let trimmed_text = text.trim();

    let new_content = if existing.trim().is_empty() {
        format!("{}\n{}\n", separator, trimmed_text)
    } else {
        format!("{}\n{}\n{}\n", existing.trim_end(), separator, trimmed_text)
    };

    let base = ensure_scheme(base_url);
    let url = format!("{}/rebukes/put", base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .query(&[("agent", agent_name)])
        .header("Authorization", format!("Bearer {}", auth_token))
        .body(new_content)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }
    Ok(())
}

/// Разбирает файл ребуков на отдельные блоки и возвращает только тексты
/// замечаний, без служебных заголовков `---[дата]---`.
fn extract_rebuke_texts(content: &str) -> Vec<String> {
    let mut texts = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("---[") && trimmed.ends_with("--------") {
            let text = current.trim();
            if !text.is_empty() {
                texts.push(text.to_string());
            }
            current.clear();
            continue;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    let text = current.trim();
    if !text.is_empty() {
        texts.push(text.to_string());
    }

    texts
}

pub fn build_system_with_rebuke(base_prompt: &str, rebuke: &str) -> String {
    let texts = extract_rebuke_texts(rebuke);
    if texts.is_empty() {
        return base_prompt.to_string();
    }

    let joined = texts
        .iter()
        .map(|t| {
            let t = t.trim();
            if t.ends_with('.') || t.ends_with('!') || t.ends_with('?') {
                t.to_string()
            } else {
                format!("{}.", t)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "{}\n\n[Замечания пользователя, учитывай в работе:]\n\n{}",
        base_prompt.trim_end(),
        joined
    )
}

#[allow(dead_code)]
pub const DIR: &str = REBUKES_DIR;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_texts_empty() {
        assert!(extract_rebuke_texts("").is_empty());
    }

    #[test]
    fn extract_texts_single_block() {
        let content = "---[2026-01-15 14:32]--------\nНе выдумывай факты.";
        let texts = extract_rebuke_texts(content);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "Не выдумывай факты.");
    }

    #[test]
    fn extract_texts_multiple_blocks() {
        let content =
            "---[2026-01-15 14:32]--------\nПервое.\n\n---[2026-01-15 15:01]--------\nВторое.";
        let texts = extract_rebuke_texts(content);
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "Первое.");
        assert_eq!(texts[1], "Второе.");
    }

    #[test]
    fn extract_texts_without_separator() {
        let content = "Просто текст без разделителя.";
        let texts = extract_rebuke_texts(content);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "Просто текст без разделителя.");
    }

    #[test]
    fn build_system_empty_rebuke_returns_base() {
        let base = "Ты — ассистент.";
        assert_eq!(build_system_with_rebuke(base, ""), base);
    }

    #[test]
    fn build_system_appends_rebuke_block() {
        let base = "Ты — ассистент.";
        let rebuke = "---[2026-01-15 14:32]--------\nНе выдумывай факты.";
        let result = build_system_with_rebuke(base, rebuke);
        assert!(result.starts_with(base));
        assert!(result.contains("[Замечания пользователя"));
        assert!(result.contains("Не выдумывай факты."));
    }

    #[test]
    fn build_system_adds_period_if_missing() {
        let rebuke = "---[x]--------\nНе выдумывай";
        let result = build_system_with_rebuke("base", rebuke);
        assert!(result.contains("Не выдумывай."));
    }

    #[test]
    fn build_system_keeps_existing_period() {
        let rebuke = "---[x]--------\nНе выдумывай.";
        let result = build_system_with_rebuke("base", rebuke);
        assert!(result.contains("Не выдумывай."));
        assert!(!result.contains("Не выдумывай.."));
    }

    #[test]
    fn build_system_joins_multiple_rebukes_with_space() {
        let rebuke = "---[x]--------\nПервое.\n\n---[y]--------\nВторое.";
        let result = build_system_with_rebuke("base", rebuke);
        assert!(result.contains("Первое. Второе."));
    }

    #[test]
    fn ensure_scheme_works() {
        assert_eq!(ensure_scheme("localhost:8080"), "http://localhost:8080");
        assert_eq!(ensure_scheme("http://x"), "http://x");
        assert_eq!(ensure_scheme("https://x"), "https://x");
    }
}
