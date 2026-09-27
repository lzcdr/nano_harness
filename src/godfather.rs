// src/godfather.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

use crate::engine::{Message, Role, Turn};

#[derive(Debug, Clone, Deserialize)]
pub struct GodfatherConfig {
    pub model: String,
    #[serde(default)]
    pub api_key: String,
    pub base_url: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    pub prompt: String,
}

impl GodfatherConfig {
    /// Проверка, что конфиг пригоден для работы.
    /// Если нет ключа — считаем, что godfather отключён.
    pub fn is_usable(&self) -> bool {
        !self.api_key.is_empty()
            && !self.model.is_empty()
            && !self.base_url.is_empty()
            && !self.prompt.is_empty()
    }
}

/// Сжать набор ходов через LLM. Возвращает новый набор сообщений.
///
/// Гарантии:
/// - Если LLM вернул мусор — Err.
/// - Если результат не короче оригинала — Err.
/// - Если LLM вернул пустой массив — Err.
///
/// Отмену решения применяет вызывающий код: полученный Err означает
/// «не сжимать, оставить как было».
pub async fn compact_turns(
    config: &GodfatherConfig,
    turns: &[Turn],
    timeout_sec: u64,
) -> Result<Vec<Message>> {
    if turns.is_empty() {
        return Err(anyhow::anyhow!("godfather: пустой набор ходов"));
    }

    let serialized = serialize_turns(turns);
    let original_bytes = serialized.len();

    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_sec))
        .build()
        .context("godfather: не удалось создать HTTP-клиент")?;

    let request = serde_json::json!({
        "model": config.model,
        "messages": [
            { "role": "system", "content": config.prompt },
            { "role": "user", "content": format!("История для сжатия:\n\n{}", serialized) }
        ],
        "stream": false,
        "temperature": config.temperature.unwrap_or(0.3),
        "max_tokens": config.max_tokens.unwrap_or(8192),
    });

    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .context("godfather: ошибка HTTP-запроса")?
        .error_for_status()
        .context("godfather: HTTP-ошибка")?;

    let parsed: GodfatherResponse = resp
        .json()
        .await
        .context("godfather: не удалось распарсить ответ API")?;

    let usage_info = parsed
        .usage
        .as_ref()
        .map(|u| {
            format!(
                "prompt={}, completion={}, cost={:.6}",
                u.prompt_tokens,
                u.completion_tokens,
                u.cost_rub.unwrap_or(0.0)
            )
        })
        .unwrap_or_else(|| "usage отсутствует".to_string());

    let raw_content = parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow::anyhow!("godfather: пустой ответ модели"))?;

    let cleaned = strip_markdown_fences(&raw_content);

    let raw_messages: Vec<RawMessage> = serde_json::from_str(cleaned).map_err(|e| {
        anyhow::anyhow!(
            "godfather: не удалось распарсить JSON-ответ: {}. Ответ был: {}",
            e,
            truncate_for_log(cleaned, 500)
        )
    })?;

    if raw_messages.is_empty() {
        return Err(anyhow::anyhow!("godfather: пустой массив в ответе"));
    }

    let mut result: Vec<Message> = Vec::with_capacity(raw_messages.len());
    for (idx, rm) in raw_messages.into_iter().enumerate() {
        let role = match rm.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(anyhow::anyhow!(
                    "godfather: недопустимая роль '{}' в позиции {}",
                    other,
                    idx
                ));
            }
        };
        result.push(Message {
            role,
            content: Some(rm.content),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    let result_bytes: usize = result
        .iter()
        .map(|m| m.content.as_ref().map(|c| c.len()).unwrap_or(0))
        .sum();

    if result_bytes >= original_bytes {
        return Err(anyhow::anyhow!(
            "godfather: сжатие не уменьшило контекст ({} → {} байт)",
            original_bytes,
            result_bytes
        ));
    }

    eprintln!(
        "👑 godfather: {} ходов, {} → {} байт ({:.0}% сжатия), {}",
        turns.len(),
        original_bytes,
        result_bytes,
        100.0 * (1.0 - result_bytes as f64 / original_bytes as f64),
        usage_info
    );

    Ok(result)
}

// ==================== Сериализация turns в текст ====================

fn serialize_turns(turns: &[Turn]) -> String {
    let mut out = String::new();

    for (turn_idx, turn) in turns.iter().enumerate() {
        if turn_idx > 0 {
            out.push('\n');
        }
        for msg in &turn.messages {
            match msg.role {
                Role::User => {
                    out.push_str("Пользователь: ");
                    out.push_str(msg.content.as_deref().unwrap_or(""));
                    out.push('\n');
                }
                Role::Assistant => {
                    if let Some(tc) = &msg.tool_calls {
                        for call in tc {
                            out.push_str(&format!(
                                "[инструмент: {}({})]\n",
                                call.function.name, call.function.arguments
                            ));
                        }
                    }
                    if let Some(c) = &msg.content {
                        if !c.trim().is_empty() {
                            out.push_str("Ассистент: ");
                            out.push_str(c);
                            out.push('\n');
                        }
                    }
                }
                Role::Tool => {
                    out.push_str("[результат инструмента: ");
                    out.push_str(msg.content.as_deref().unwrap_or(""));
                    out.push_str("]\n");
                }
                Role::System | Role::Developer => {
                    // system-сообщения в turns не должны встречаться,
                    // но на всякий случай пропускаем.
                }
            }
        }
    }

    out
}

fn strip_markdown_fences(s: &str) -> &str {
    let t = s.trim();

    if !t.starts_with("```") {
        return t;
    }

    let after_open = if let Some(rest) = t.strip_prefix("```json") {
        rest
    } else if let Some(rest) = t.strip_prefix("```JSON") {
        rest
    } else if let Some(rest) = t.strip_prefix("```") {
        rest
    } else {
        return t;
    };

    let after_open = after_open.trim_start_matches(['\r', '\n']);

    if let Some(end) = after_open.rfind("```") {
        return after_open[..end].trim();
    }

    after_open.trim()
}

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push_str("...");
        out
    }
}

// ==================== Внутренние структуры для парсинга ====================

#[derive(Debug, Deserialize)]
struct GodfatherResponse {
    choices: Vec<GodfatherChoice>,
    #[serde(default)]
    usage: Option<GodfatherUsage>,
}

#[derive(Debug, Deserialize)]
struct GodfatherChoice {
    message: GodfatherMessage,
}

#[derive(Debug, Deserialize)]
struct GodfatherMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GodfatherUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    cost_rub: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawMessage {
    role: String,
    content: String,
}
