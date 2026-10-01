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
