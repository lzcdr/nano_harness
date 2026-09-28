// src/agent_core.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use crate::engine::{ChatEngine, EngineConfig, Role, ToolCall};

pub struct PendingCall {
    pub tx: tokio::sync::oneshot::Sender<String>,
    pub created_at: std::time::Instant,
}

pub type PendingCalls = Arc<tokio::sync::Mutex<HashMap<String, PendingCall>>>;

#[derive(Debug, Clone)]
pub struct OutgoingTask {
    pub task_id: String,
    pub session_id: String,
    pub to_agent_name: String,
    pub to_session_id: String,
    pub project_id: String,
    pub chain: Vec<String>,
}

pub type OutgoingTasks = Arc<tokio::sync::Mutex<HashMap<String, OutgoingTask>>>;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AgentType {
    Stateless,
    Stateful,
}

impl Default for AgentType {
    fn default() -> Self {
        AgentType::Stateless
    }
}

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
    #[serde(default)]
    pub agent_type: AgentType,
    #[serde(default)]
    pub session_ttl_secs: Option<u64>,
    #[serde(default = "default_skill_mode")]
    pub skill_mode: String,
    #[serde(default = "default_skill_semantic_threshold")]
    pub skill_semantic_threshold: f32,
    #[serde(default = "default_skill_min_code_length")]
    pub skill_min_code_length: usize,
    #[serde(default)]
    pub agent_call_timeout_sec: Option<u64>,
    #[serde(default = "default_skill_auto_execute_threshold")]
    pub skill_auto_execute_threshold: f32,
    #[serde(default)]
    pub max_cost_rub: Option<f64>,
    #[serde(default)]
    pub compact_threshold_bytes: Option<usize>,
    #[serde(default)]
    pub tail_byte_budget: Option<usize>,
}

fn default_skill_mode() -> String {
    "auto".to_string()
}
fn default_skill_semantic_threshold() -> f32 {
    0.5
}
fn default_skill_min_code_length() -> usize {
    100
}
fn default_skill_auto_execute_threshold() -> f32 {
    0.15
}

#[derive(Debug, Deserialize)]
pub struct AgentRequest {
    pub prompt: String,
    pub max_iterations: Option<usize>,
    pub tools: Option<Vec<String>>,
    pub system_prompt: Option<String>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub status: String,
    pub result: String,
    pub reasoning: Option<String>,
    pub tool_calls_log: Vec<ToolCallLogEntry>,
    pub metrics: AgentMetrics,
    pub error: Option<String>,
    pub session_id: Option<String>,
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
    pub self_name: Option<String>,
    pub agent_call_timeout_sec: u64,
    pub board_base_url: String,
    pub board_auth_token: String,
    pub pending_calls: PendingCalls,
    pub outgoing_tasks: OutgoingTasks,
}

impl AgentContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        storage_base_url: String,
        storage_auth_token: String,
        timeout_sec: u64,
        self_name: Option<String>,
        agent_call_timeout_sec: u64,
        board_base_url: String,
        board_auth_token: String,
        pending_calls: PendingCalls,
        outgoing_tasks: OutgoingTasks,
    ) -> Result<Self> {
        let storage_base_url = if storage_base_url.starts_with("http://")
            || storage_base_url.starts_with("https://")
        {
            storage_base_url
        } else {
            format!("http://{}", storage_base_url)
        };

        let board_base_url =
            if board_base_url.starts_with("http://") || board_base_url.starts_with("https://") {
                board_base_url
            } else {
                format!("http://{}", board_base_url)
            };

        let http_client = Client::builder()
            .timeout(Duration::from_secs(timeout_sec))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            http_client,
            storage_base_url,
            storage_auth_token,
            self_name,
            agent_call_timeout_sec,
            board_base_url,
            board_auth_token,
            pending_calls,
            outgoing_tasks,
        })
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

pub fn build_engine_config(config: &AgentConfig) -> EngineConfig {
    let mut allowed_tools = config.tools.clone();

    // knowledge_load и knowledge_unload разрешены всегда.
    // Каталог знаний вшивается в системный промпт безусловно, и если
    // инструменты не будут доступны — модель попробует их вызвать и получит
    // "unknown tool". Чтобы этого не происходило, добавляем их принудительно.
    for extra in ["knowledge_load", "knowledge_unload"] {
        if !allowed_tools.iter().any(|n| n == extra) {
            allowed_tools.push(extra.to_string());
        }
    }

    EngineConfig {
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
        max_cost_rub: config.max_cost_rub,
        prefix_message_count: config.prefix_message_count,
        tail_message_count: config.tail_message_count,
        compact_threshold_bytes: config.compact_threshold_bytes.unwrap_or(512),
        tail_byte_budget: config.tail_byte_budget.unwrap_or(100 * 1024),
    }
}

pub fn create_agent_engine(config: &AgentConfig) -> Result<ChatEngine> {
    let engine_config = build_engine_config(config);
    let client = Client::builder()
        .timeout(Duration::from_secs(config.timeout_sec))
        .build()
        .context("Failed to create HTTP client")?;
    Ok(ChatEngine::new(engine_config, client))
}

#[allow(clippy::too_many_arguments)]
pub async fn process_agent_turns(
    engine: &mut ChatEngine,
    config: &AgentConfig,
    context: &AgentContext,
    request: &AgentRequest,
    log_file: &mut fs::File,
    session_id: Option<String>,
    project_id: String,
    parent_chain: Vec<String>,
) -> Result<AgentResponse> {
    let max_iterations = request.max_iterations.unwrap_or(config.max_iterations);
    let mut tool_calls_log = Vec::new();
    let mut final_response = None;
    let rhai_timeout = config.rhai_timeout_sec.unwrap_or(30);
    let mut posted_any = false;

    let mut skill_found = false;

    // Скилл-контекст живёт только текущий ход. Сбрасываем перед новым поиском.
    engine.clear_skill_context();

    if config.skill_mode == "auto" && !request.prompt.trim().is_empty() {
        let storage_base_url = context.storage_base_url.clone();
        let storage_auth_token = context.storage_auth_token.clone();
        let current_agent = config.name.clone();
        let prompt = request.prompt.clone();
        let threshold = config.skill_semantic_threshold;

        let skill_result = tokio::task::spawn_blocking(
            move || -> Result<Option<(crate::skill_manager::SkillRecord, f32)>> {
                let client = reqwest::blocking::Client::new();
                crate::skill_manager::search_best_skill(
                    &client,
                    &storage_base_url,
                    &storage_auth_token,
                    &current_agent,
                    &prompt,
                    threshold,
                    10,
                )
            },
        )
        .await
        .map_err(|e| anyhow::anyhow!("Join error: {}", e))??;

        if let Some((skill_record, distance)) = skill_result {
            let storage_base_url = context.storage_base_url.clone();
            let storage_auth_token = context.storage_auth_token.clone();
            let skill_file = skill_record.skill_file.clone();

            let content = tokio::task::spawn_blocking(move || {
                let client = reqwest::blocking::Client::new();
                crate::skill_manager::load_skill(
                    &client,
                    &storage_base_url,
                    &storage_auth_token,
                    &skill_file,
                )
            })
            .await
            .map_err(|e| anyhow::anyhow!("Join error: {}", e))??;

            write_log(log_file, "skill_found", &skill_record.skill_file)?;
            eprintln!("🔍 Найден скилл: {}", skill_record.skill_file);

            if distance <= config.skill_auto_execute_threshold {
                let code = extract_rhai_code(&content)
                    .ok_or_else(|| anyhow::anyhow!("RHAI_CODE not found in skill"))?;

                write_log(
                    log_file,
                    "skill_auto_execute",
                    &format!("{} (distance: {:.4})", skill_record.skill_file, distance),
                )?;

                let result = execute_skill_code_directly(
                    context,
                    config,
                    code.clone(),
                    session_id.clone(),
                    project_id.clone(),
                    parent_chain.clone(),
                )
                .await;

                let success = result.is_ok();
                let storage_base_url = context.storage_base_url.clone();
                let storage_auth_token = context.storage_auth_token.clone();
                let skill_file = skill_record.skill_file.clone();
                tokio::task::spawn_blocking(move || {
                    let client = reqwest::blocking::Client::new();
                    if let Err(e) = crate::skill_manager::record_usage(
                        &client,
                        &storage_base_url,
                        &storage_auth_token,
                        &skill_file,
                        success,
                    ) {
                        eprintln!("⚠️ Не удалось обновить счётчик скилла: {}", e);
                    }
                });

                let result = result?;

                write_log(
                    log_file,
                    "tool_result",
                    &format!("run_code (auto from skill) -> {}", result),
                )?;

                engine.add_message(Role::Assistant, result.clone());
                write_log(log_file, "assistant", &result)?;

                let tool_log = ToolCallLogEntry {
                    name: "run_code".to_string(),
                    arguments: serde_json::json!({ "code": code }).to_string(),
                    result: result.clone(),
                };

                return Ok(AgentResponse {
                    status: "completed".to_string(),
                    result,
                    reasoning: None,
                    tool_calls_log: vec![tool_log],
                    metrics: AgentMetrics {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        cost_rub: 0.0,
                        api_calls_count: 0,
                    },
                    error: None,
                    session_id: None,
                });
            } else {
                engine.set_skill_context(format!("Найден подходящий скилл:\n{}", content));
                write_log(log_file, "skill_injected", &skill_record.skill_file)?;
                skill_found = true;
            }
        }
    }

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
                    log_file,
                    "tool_request",
                    &format!("{} ({})", tc.function.name, tc.function.arguments),
                )?;
            }

            for tc in &tool_calls {
                let (result, posted) = if tc.function.name == "knowledge_load" {
                    let r = handle_knowledge_load(context, engine, tc).await;
                    eprintln!("✅ Результат '{}': {}", tc.function.name, r);
                    (r, false)
                } else if tc.function.name == "knowledge_unload" {
                    let r = handle_knowledge_unload(engine);
                    eprintln!("✅ Результат '{}': {}", tc.function.name, r);
                    (r, false)
                } else {
                    let tool_outcome = execute_agent_tool(
                        context,
                        config,
                        tc,
                        rhai_timeout,
                        session_id.clone(),
                        project_id.clone(),
                        parent_chain.clone(),
                    )
                    .await;

                    match tool_outcome {
                        Ok(pair) => {
                            eprintln!("✅ Результат '{}': {}", tc.function.name, pair.0);
                            pair
                        }
                        Err(e) => {
                            eprintln!("❌ Ошибка '{}': {:#}", tc.function.name, e);
                            write_log(
                                log_file,
                                "tool_error",
                                &format!(
                                    "{} ({}) -> {:#}",
                                    tc.function.name, tc.function.arguments, e
                                ),
                            )?;
                            return Err(e);
                        }
                    }
                };

                write_log(
                    log_file,
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

                engine.add_tool_result(tc.id.clone(), result.clone());

                if posted || result.starts_with("posted:") {
                    posted_any = true;
                }
            }
            final_response = Some(response);

            if posted_any {
                break;
            }
        } else {
            final_response = Some(response);
            break;
        }
    }

    if posted_any {
        let metrics = AgentMetrics {
            prompt_tokens: engine.metrics.total_prompt_tokens,
            completion_tokens: engine.metrics.total_completion_tokens,
            cost_rub: engine.metrics.total_cost_rub,
            api_calls_count: engine.metrics.api_calls_count,
        };
        return Ok(AgentResponse {
            status: "waiting".to_string(),
            result: String::new(),
            reasoning: None,
            tool_calls_log,
            metrics,
            error: None,
            session_id: None,
        });
    }

    let final_response = final_response.ok_or_else(|| anyhow::anyhow!("No response from agent"))?;

    let final_content =
        if final_response.content.trim().is_empty() && final_response.tool_calls.is_some() {
            tool_calls_log
                .last()
                .map(|t| t.result.clone())
                .unwrap_or_else(|| "(пустой результат: исчерпан max_iterations)".to_string())
        } else {
            final_response.content.clone()
        };

    if config.skill_mode == "auto" && !skill_found {
        let rhai_calls: Vec<&ToolCallLogEntry> = tool_calls_log
            .iter()
            .filter(|t| t.name == "run_code" && !t.result.starts_with("Ошибка"))
            .collect();
        for call in rhai_calls {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&call.arguments) {
                if let Some(code_ref) = parsed.get("code").and_then(|v| v.as_str()) {
                    let code = code_ref.to_string();

                    let min_len = config.skill_min_code_length;
                    let trimmed_len = code.trim().len();
                    if trimmed_len < min_len {
                        write_log(
                            log_file,
                            "skill_skipped",
                            &format!("(too short) ({} < {})", trimmed_len, min_len),
                        )?;
                        eprintln!(
                            "⏭️ Скилл не сохранён: код {} символов < {}",
                            trimmed_len, min_len
                        );
                        continue;
                    }

                    let engine_config = build_engine_config(config);
                    let (skill_name, skill_description) =
                        match crate::skill_manager::generate_skill_metadata(
                            &engine_config,
                            &request.prompt,
                        )
                        .await
                        {
                            Ok(meta) => meta,
                            Err(e) => {
                                write_log(log_file, "skill_metadata_failed", &format!("{}", e))?;
                                continue;
                            }
                        };

                    let storage_base_url = context.storage_base_url.clone();
                    let storage_auth_token = context.storage_auth_token.clone();
                    let agent_name = config.name.clone();
                    let prompt = request.prompt.clone();
                    let skill_name_for_log = skill_name.clone();

                    let outcome = match tokio::task::spawn_blocking(move || {
                        let client = reqwest::blocking::Client::new();
                        crate::skill_manager::save_skill(
                            &client,
                            &storage_base_url,
                            &storage_auth_token,
                            &skill_name,
                            &agent_name,
                            &skill_description,
                            &prompt,
                            &code,
                            min_len,
                        )
                    })
                    .await
                    {
                        Ok(Ok(o)) => o,
                        Ok(Err(e)) => {
                            write_log(log_file, "skill_save_error", &format!("{}", e))?;
                            eprintln!("⚠️ Ошибка сохранения скилла: {}", e);
                            continue;
                        }
                        Err(e) => {
                            write_log(log_file, "skill_save_error", &format!("Join error: {}", e))?;
                            eprintln!("⚠️ Join error при сохранении скилла: {}", e);
                            continue;
                        }
                    };

                    match outcome {
                        crate::skill_manager::SaveSkillOutcome::Saved => {
                            write_log(log_file, "skill_saved", &skill_name_for_log)?;
                            eprintln!("💾 Скилл сохранён: {}", skill_name_for_log);
                        }
                        crate::skill_manager::SaveSkillOutcome::SkippedTooShort { actual, min } => {
                            write_log(
                                log_file,
                                "skill_skipped",
                                &format!("{} ({} < {})", skill_name_for_log, actual, min),
                            )?;
                            eprintln!(
                                "⏭️ Скилл не сохранён: {} ({} < {})",
                                skill_name_for_log, actual, min
                            );
                        }
                        crate::skill_manager::SaveSkillOutcome::SkippedDuplicate => {
                            write_log(log_file, "skill_skipped_duplicate", &skill_name_for_log)?;
                            eprintln!("♻️ Скилл пропущен (дубликат): {}", skill_name_for_log);
                        }
                    }
                }
            }
        }
    }

    let metrics = AgentMetrics {
        prompt_tokens: engine.metrics.total_prompt_tokens,
        completion_tokens: engine.metrics.total_completion_tokens,
        cost_rub: engine.metrics.total_cost_rub,
        api_calls_count: engine.metrics.api_calls_count,
    };

    eprintln!("🎯 Финальный ответ агента: {}", final_content);
    write_log(log_file, "assistant", &final_content)?;
    if !final_response.reasoning.is_empty() {
        write_log(log_file, "reasoning", &final_response.reasoning)?;
    }

    Ok(AgentResponse {
        status: "completed".to_string(),
        result: final_content,
        reasoning: if final_response.reasoning.is_empty() {
            None
        } else {
            Some(final_response.reasoning.clone())
        },
        tool_calls_log,
        metrics,
        error: None,
        session_id: None,
    })
}

pub async fn run_agent(
    config: &AgentConfig,
    context: &AgentContext,
    request: AgentRequest,
    project_id: String,
) -> Result<AgentResponse> {
    let log_session_id = request
        .session_id
        .clone()
        .unwrap_or_else(|| format!("stateless-{}", uuid::Uuid::new_v4()));
    let mut log_file =
        crate::session_store::open_log(&log_session_id, "agent", Some(&config.name))?;

    let system_prompt = request
        .system_prompt
        .clone()
        .or_else(|| config.system_prompt.clone())
        .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());

    write_log(&mut log_file, "system", &system_prompt)?;
    write_log(&mut log_file, "task", &request.prompt)?;
    eprintln!(
        "🚀 Агент '{}' получил задачу: {}",
        config.name, request.prompt
    );

    let mut engine = create_agent_engine(config)?;
    engine.add_message(Role::System, system_prompt);
    engine.add_message(Role::User, request.prompt.clone());

    let mut response = process_agent_turns(
        &mut engine,
        config,
        context,
        &request,
        &mut log_file,
        request.session_id.clone(),
        project_id,
        vec![],
    )
    .await?;
    response.session_id = request.session_id;
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
async fn execute_agent_tool(
    context: &AgentContext,
    config: &AgentConfig,
    tool_call: &ToolCall,
    rhai_timeout_sec: u64,
    session_id: Option<String>,
    project_id: String,
    parent_chain: Vec<String>,
) -> Result<(String, bool)> {
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
            let self_name = context.self_name.clone();
            let agent_call_timeout = context.agent_call_timeout_sec;
            let board_url = context.board_base_url.clone();
            let board_token = context.board_auth_token.clone();
            let pending_calls = context.pending_calls.clone();
            let outgoing_tasks = context.outgoing_tasks.clone();
            let timeout_duration = Duration::from_secs(rhai_timeout_sec);
            let config_owned = config.clone();
            let pid = project_id.clone();

            let posted_flag = Arc::new(AtomicBool::new(false));
            let posted_flag_clone = posted_flag.clone();

            let result = tokio::time::timeout(
                timeout_duration,
                tokio::task::spawn_blocking(move || {
                    run_code_with_storage(
                        storage_base_url,
                        storage_auth_token,
                        self_name,
                        agent_call_timeout,
                        board_url,
                        board_token,
                        session_id,
                        pid,
                        parent_chain,
                        pending_calls,
                        outgoing_tasks,
                        posted_flag_clone,
                        &config_owned,
                        &code,
                    )
                }),
            )
            .await
            .map_err(|_| anyhow::anyhow!("Rhai execution timed out"))?;

            let result = result.map_err(|e| anyhow::anyhow!("JoinError: {}", e))?;
            let result = result.map_err(|e| anyhow::anyhow!("Rhai execution error: {}", e))?;

            let posted = posted_flag.load(Ordering::Relaxed);
            Ok((result, posted))
        }
        "local_storage" => {
            let result = crate::tools::execute_tool(
                "local_storage",
                &tool_call.function.arguments,
                Some(&context.http_client),
                &context.storage_base_url,
                &context.storage_auth_token,
                &project_id,
                rhai_timeout_sec,
                None,
            )
            .await;
            Ok((result, false))
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_call.function.name)),
    }
}

async fn handle_knowledge_load(
    context: &AgentContext,
    engine: &mut ChatEngine,
    tool_call: &ToolCall,
) -> String {
    let parsed: serde_json::Value =
        serde_json::from_str(&tool_call.function.arguments).unwrap_or(serde_json::json!({}));
    let name = match parsed.get("name").and_then(|v| v.as_str()) {
        Some(n) if !n.trim().is_empty() => n.trim().to_string(),
        _ => return "Ошибка: не указано имя знания".to_string(),
    };

    match crate::knowledge_manager::load_knowledge(
        &context.http_client,
        &context.storage_base_url,
        &context.storage_auth_token,
        &name,
    )
    .await
    {
        Ok(content) => {
            let bytes = content.len();
            engine.set_knowledge_context(&name, content);
            format!(
                "Знание '{}' загружено в контекст ({} байт). Оно будет доступно в последующих ходах.",
                name, bytes
            )
        }
        Err(e) => format!("Ошибка загрузки знания '{}': {:#}", name, e),
    }
}

fn handle_knowledge_unload(engine: &mut ChatEngine) -> String {
    if engine.has_knowledge_context() {
        engine.clear_knowledge_context();
        "Знание снято из контекста".to_string()
    } else {
        "Активного знания не было".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_code_with_storage(
    storage_base_url: String,
    storage_auth_token: String,
    self_name: Option<String>,
    agent_call_timeout_sec: u64,
    board_url: String,
    board_token: String,
    self_session_id: Option<String>,
    project_id: String,
    parent_chain: Vec<String>,
    pending_calls: PendingCalls,
    outgoing_tasks: OutgoingTasks,
    posted_flag: Arc<AtomicBool>,
    config: &AgentConfig,
    code: &str,
) -> Result<String> {
    let mut engine = rhai::Engine::new();
    engine.set_max_operations(crate::tools::RHAI_MAX_OPERATIONS);
    engine.set_max_call_levels(crate::tools::RHAI_MAX_CALL_LEVELS);
    engine.set_max_string_size(crate::tools::RHAI_MAX_STRING_SIZE);

    let output = Rc::new(RefCell::new(String::new()));
    let output_clone = output.clone();
    engine.on_print(move |s| output_clone.borrow_mut().push_str(s));

    crate::tools::register_basic_functions(&mut engine);
    crate::tools::register_storage_functions(
        &mut engine,
        &storage_base_url,
        &storage_auth_token,
        &project_id,
    );
    crate::tools::register_board_functions(
        &mut engine,
        board_url,
        board_token,
        self_name.unwrap_or_default(),
        self_session_id,
        project_id,
        parent_chain,
        pending_calls,
        agent_call_timeout_sec,
        posted_flag,
        outgoing_tasks,
    );

    if config.skill_mode == "manual" {
        crate::tools::register_skill_functions(
            &mut engine,
            &storage_base_url,
            &storage_auth_token,
            config,
        );
    }

    let normalized = crate::tools::normalize_multiline_strings(code);
    match engine.eval::<rhai::Dynamic>(&normalized) {
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

fn extract_rhai_code(skill_content: &str) -> Option<String> {
    let marker = "RHAI_CODE:";
    let pos = skill_content.find(marker)?;
    Some(skill_content[pos + marker.len()..].trim().to_string())
}

async fn execute_skill_code_directly(
    context: &AgentContext,
    config: &AgentConfig,
    code: String,
    session_id: Option<String>,
    project_id: String,
    parent_chain: Vec<String>,
) -> Result<String> {
    let storage_base_url = context.storage_base_url.clone();
    let storage_auth_token = context.storage_auth_token.clone();
    let self_name = context.self_name.clone();
    let agent_call_timeout = context.agent_call_timeout_sec;
    let board_url = context.board_base_url.clone();
    let board_token = context.board_auth_token.clone();
    let pending_calls = context.pending_calls.clone();
    let outgoing_tasks = context.outgoing_tasks.clone();
    let config_owned = config.clone();
    let rhai_timeout = config.rhai_timeout_sec.unwrap_or(30);

    let posted_flag = Arc::new(AtomicBool::new(false));

    let result = tokio::time::timeout(
        Duration::from_secs(rhai_timeout),
        tokio::task::spawn_blocking(move || {
            run_code_with_storage(
                storage_base_url,
                storage_auth_token,
                self_name,
                agent_call_timeout,
                board_url,
                board_token,
                session_id,
                project_id,
                parent_chain,
                pending_calls,
                outgoing_tasks,
                posted_flag,
                &config_owned,
                &code,
            )
        }),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Rhai execution timed out"))??;

    result.map_err(|e| anyhow::anyhow!("Rhai execution error: {}", e))
}
