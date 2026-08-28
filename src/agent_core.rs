// src/agent_core.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::rc::Rc;
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
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub rhai_timeout_sec: Option<u64>,
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

fn write_log(file: &mut fs::File, role: &str, content: &str) -> Result<()> {
    writeln!(
        file,
        "[{}] {}: {}",
        chrono::Local::now().format("%H:%M:%S"),
        role,
        content
    )?;
    Ok(())
}

pub async fn run_agent(
    config: &AgentConfig,
    context: &AgentContext,
    request: AgentRequest,
) -> Result<AgentResponse> {
    let log_dir = format!("chats/{}", config.name);
    fs::create_dir_all(&log_dir)?;
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let log_path = format!("{}/{}.txt", log_dir, timestamp);
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let max_iterations = request.max_iterations.unwrap_or(config.max_iterations);
    let allowed_tools = request.tools.unwrap_or_else(|| config.tools.clone());
    let system_prompt = request
        .system_prompt
        .or_else(|| config.system_prompt.clone())
        .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());

    write_log(&mut log_file, "system", &system_prompt)?;
    write_log(&mut log_file, "task", &request.prompt)?;

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
        reasoning_effort: config.reasoning_effort.clone(),
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

    let rhai_timeout = config.rhai_timeout_sec.unwrap_or(30);

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
                write_log(
                    &mut log_file,
                    "tool_request",
                    &format!("{} ({})", tc.function.name, tc.function.arguments),
                )?;
            }

            for tc in &tool_calls {
                let result = execute_agent_tool(context, tc, rhai_timeout).await?;
                eprintln!("✅ Результат '{}': {}", tc.function.name, result);

                write_log(
                    &mut log_file,
                    "tool_result",
                    &format!(
                        "{} ({}) -> {}",
                        tc.function.name, tc.function.arguments, result
                    ),
                )?;

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
    write_log(&mut log_file, "assistant", &final_response.content)?;
    if !final_response.reasoning.is_empty() {
        write_log(&mut log_file, "reasoning", &final_response.reasoning)?;
    }
    write_log(
        &mut log_file,
        "metrics",
        &format!(
            "Prompt: {}, Completion: {}, Cost: {:.6} RUB, Calls: {}",
            metrics.prompt_tokens,
            metrics.completion_tokens,
            metrics.cost_rub,
            metrics.api_calls_count
        ),
    )?;

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

async fn execute_agent_tool(
    context: &AgentContext,
    tool_call: &ToolCall,
    rhai_timeout_sec: u64,
) -> Result<String> {
    match tool_call.function.name.as_str() {
        "run_code" => {
            let parsed: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or(serde_json::json!({"code": ""}));
            let code = parsed
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let storage_base_url = context.storage_base_url.clone();
            let storage_auth_token = context.storage_auth_token.clone();
            let timeout_duration = Duration::from_secs(rhai_timeout_sec);

            let result = tokio::time::timeout(
                timeout_duration,
                tokio::task::spawn_blocking(move || {
                    run_code_with_storage(storage_base_url, storage_auth_token, &code)
                }),
            )
            .await
            .map_err(|_| anyhow::anyhow!("Rhai execution timed out"))?;

            let result = result.map_err(|e| anyhow::anyhow!("JoinError: {}", e))?;

            let result = result.map_err(|e| anyhow::anyhow!("Rhai execution error: {}", e))?;

            Ok(result)
        }
        "local_storage" => {
            let result = crate::tools::execute_tool(
                "local_storage",
                &tool_call.function.arguments,
                Some(&context.http_client),
                &context.storage_base_url,
                &context.storage_auth_token,
                rhai_timeout_sec,
            )
            .await;
            Ok(result)
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_call.function.name)),
    }
}

fn run_code_with_storage(
    storage_base_url: String,
    storage_auth_token: String,
    code: &str,
) -> Result<String> {
    let mut engine = rhai::Engine::new();
    engine.set_max_operations(10_000);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1024 * 10);

    // Перехват вывода print
    let output = Rc::new(RefCell::new(String::new()));
    let output_clone = output.clone();
    engine.on_print(move |s| output_clone.borrow_mut().push_str(s));

    crate::tools::register_basic_functions(&mut engine);
    crate::tools::register_storage_functions(&mut engine, &storage_base_url, &storage_auth_token);

    match engine.eval::<rhai::Dynamic>(code) {
        Ok(result) => {
            let printed = output.borrow().clone();
            if !printed.is_empty() {
                Ok(printed)
            } else {
                Ok(result.to_string())
            }
        }
        Err(e) => Err(anyhow::anyhow!("Rhai execution error: {}", e)),
    }
}
