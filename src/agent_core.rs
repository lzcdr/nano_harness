// src/agent_core.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;

use crate::engine::{ChatEngine, EngineConfig, Role, ToolCall};

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub bind_addr: String,
    pub auth_token: String,
    pub timeout_sec: u64,
    pub max_iterations: usize,
    pub tools: Vec<String>,
    pub system_prompt: Option<String>,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub prefix_message_count: Option<usize>,
    pub tail_message_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    pub prompt: String,
    pub max_iterations: Option<usize>,
    pub tools: Option<Vec<String>>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub status: String,
    pub result: String,
    pub reasoning: Option<String>,
    pub tool_calls_log: Vec<ToolCallLogEntry>,
    pub metrics: AgentMetrics,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolCallLogEntry {
    pub name: String,
    pub arguments: String,
    pub result: String,
}

#[derive(Debug, Serialize)]
pub struct AgentMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cost_rub: f64,
    pub api_calls_count: u32,
}

pub struct AgentContext {
    pub http_client: Client,
    pub storage_base_url: String,
    pub storage_auth_token: String,
}

impl AgentContext {
    pub fn new(storage_base_url: String, storage_auth_token: String, timeout_sec: u64) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(timeout_sec))
            .build()
            .expect("Failed to create HTTP client");
        Self {
            http_client,
            storage_base_url,
            storage_auth_token,
        }
    }
}

pub async fn run_agent(
    config: &AgentConfig,
    context: &AgentContext,
    request: AgentRequest,
) -> Result<AgentResponse> {
    let max_iterations = request.max_iterations.unwrap_or(config.max_iterations);
    let allowed_tools = request.tools.unwrap_or_else(|| config.tools.clone());
    let system_prompt = request
        .system_prompt
        .or_else(|| config.system_prompt.clone())
        .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());

    eprintln!(
        "🚀 Агент '{}' получил задачу: {}",
        config.name, request.prompt
    );

    let engine_config = EngineConfig {
        api_key: config.api_key.clone(),
        base_url: config.base_url.clone(),
        model: config.model.clone(),
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        top_p: config.top_p,
        stop: None,
        stream: config.stream,
        allowed_tools: Some(
            allowed_tools
                .iter()
                .map(|name| crate::engine::ToolExecutionConfig {
                    name: name.clone(),
                    mode: "auto".to_string(),
                })
                .collect(),
        ),
        tool_choice: None,
        reasoning_effort: None,
        max_cost_rub: None,
        prefix_message_count: config.prefix_message_count,
        tail_message_count: config.tail_message_count,
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(config.timeout_sec))
        .build()
        .context("Failed to create HTTP client")?;

    let mut engine = ChatEngine::new(engine_config.clone(), client);
    engine.add_message(Role::System, system_prompt);
    engine.add_message(Role::User, request.prompt.clone());

    let mut tool_calls_log = Vec::new();
    let mut final_response = None;

    for _ in 0..=max_iterations {
        let response = timeout(Duration::from_secs(config.timeout_sec), engine.send())
            .await
            .map_err(|_| anyhow::anyhow!("Agent timed out"))??;

        let has_tool_calls = response.tool_calls.is_some();
        if has_tool_calls {
            let tool_calls = response.tool_calls.clone().unwrap();

            eprintln!("⚠️ Модель запросила инструменты:");
            for tc in &tool_calls {
                eprintln!("   - {} ({})", tc.function.name, tc.function.arguments);
            }

            for tc in &tool_calls {
                let result = execute_agent_tool(context, tc).await?;
                eprintln!("✅ Результат '{}': {}", tc.function.name, result);

                tool_calls_log.push(ToolCallLogEntry {
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                    result: result.clone(),
                });
                engine.add_tool_result(tc.id.clone(), result);
            }
            final_response = Some(response);
        } else {
            final_response = Some(response);
            break;
        }
    }

    let final_response = final_response.ok_or_else(|| anyhow::anyhow!("No response from agent"))?;

    let metrics = AgentMetrics {
        prompt_tokens: engine.metrics.total_prompt_tokens,
        completion_tokens: engine.metrics.total_completion_tokens,
        cost_rub: engine.metrics.total_cost_rub,
        api_calls_count: engine.metrics.api_calls_count,
    };

    eprintln!("🎯 Финальный ответ агента: {}", final_response.content);

    Ok(AgentResponse {
        status: "completed".to_string(),
        result: final_response.content.clone(),
        reasoning: if final_response.reasoning.is_empty() {
            None
        } else {
            Some(final_response.reasoning.clone())
        },
        tool_calls_log,
        metrics,
        error: None,
    })
}

async fn execute_agent_tool(context: &AgentContext, tool_call: &ToolCall) -> Result<String> {
    match tool_call.function.name.as_str() {
        "run_code" => {
            // Извлекаем код из JSON-аргумента
            let parsed: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or(serde_json::json!({"code": ""}));
            let code = parsed
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let storage_base_url = context.storage_base_url.clone();
            let storage_auth_token = context.storage_auth_token.clone();
            // Выполняем Rhai в отдельном блокирующем потоке
            let result = tokio::task::spawn_blocking(move || {
                run_code_with_storage(storage_base_url, storage_auth_token, &code)
            })
            .await
            .map_err(|e| anyhow::anyhow!("Blocking task failed: {}", e))??;
            Ok(result)
        }
        "local_storage" => {
            let result = crate::tools::execute_tool(
                "local_storage",
                &tool_call.function.arguments,
                Some(&context.http_client),
                &context.storage_base_url,
                &context.storage_auth_token,
            )
            .await;
            Ok(result)
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_call.function.name)),
    }
}

/// Синхронная функция выполнения Rhai-кода с функциями доступа к хранилищу через HTTP API.
/// Вызывается в отдельном потоке через `spawn_blocking`.
fn run_code_with_storage(
    storage_base_url: String,
    storage_auth_token: String,
    code: &str,
) -> Result<String> {
    // Нормализуем URL
    let base_url =
        if storage_base_url.starts_with("http://") || storage_base_url.starts_with("https://") {
            storage_base_url
        } else {
            format!("http://{}", storage_base_url)
        };

    // Создаём блокирующий HTTP-клиент
    let blocking_client = reqwest::blocking::Client::new();

    let mut engine = rhai::Engine::new();
    engine.set_max_operations(10_000);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1024 * 10);

    engine.register_fn("get_time", || -> String {
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    });
    engine.register_fn("get_weather", |city: String| -> String {
        format!("Погода в городе {}: солнечно, +22°C (заглушка)", city)
    });

    // Регистрируем функции для работы с хранилищем через HTTP API
    {
        let client = blocking_client.clone();
        let base = base_url.clone();
        let token = storage_auth_token.clone();
        engine.register_fn("storage_read_file", move |path: String| -> String {
            let url = format!("{}/files", base.trim_end_matches('/'));
            match client
                .get(&url)
                .header("Authorization", format!("Bearer {}", token))
                .query(&[("path", &path)])
                .send()
            {
                Ok(resp) if resp.status().is_success() => resp.text().unwrap_or_default(),
                Ok(resp) => format!(
                    "HTTP ошибка {}: {}",
                    resp.status(),
                    resp.text().unwrap_or_default()
                ),
                Err(e) => format!("Ошибка запроса: {}", e),
            }
        });
    }

    // При необходимости добавьте другие функции хранилища (storage_write_file, storage_list_dir и т.д.)

    match engine.eval::<rhai::Dynamic>(code) {
        Ok(result) => Ok(result.to_string()),
        Err(e) => Err(anyhow::anyhow!("Rhai execution error: {}", e)),
    }
}
