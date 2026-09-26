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
        format!(
            "{}\n{}\n{}\n",
            existing.trim_end(),
            separator,
            trimmed_text
        )
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

pub fn build_system_with_rebuke(base_prompt: &str, rebuke: &str) -> String {
    if rebuke.trim().is_empty() {
        base_prompt.to_string()
    } else {
        format!(
            "{}\n\n[Замечания пользователя, учитывай в работе:]\n\n{}",
            base_prompt.trim_end(),
            rebuke.trim_end()
        )
    }
}

#[allow(dead_code)]
pub const DIR: &str = REBUKES_DIR;
