// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/agent_core.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use crate::engine::{ChatEngine, EngineConfig, Role};
use crate::knowledge_auto::{build_knowledge_context, KnowledgeAutoConfig};

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
    pub agent_type: AgentType,
    #[serde(default)]
    pub session_ttl_secs: Option<u64>,
    #[serde(default)]
    pub skill_mode: Option<String>,
    #[serde(default)]
    pub skill_min_tool_calls: Option<usize>,
    #[serde(default)]
    pub agent_call_timeout_sec: Option<u64>,
    #[serde(default)]
    pub max_cost_rub: Option<f64>,
    #[serde(default)]
    pub compact_threshold_bytes: Option<usize>,
    #[serde(default)]
    pub tail_byte_budget: Option<usize>,
    #[serde(default)]
    pub knowledge_auto_top_n: Option<usize>,
    #[serde(default)]
    pub knowledge_auto_top_k: Option<usize>,
    #[serde(default)]
    pub knowledge_auto_threshold: Option<f32>,
    #[serde(default)]
    pub skill_phrases_top_n: Option<usize>,
    #[serde(default)]
    pub skill_phrase_min_words: Option<usize>,
    #[serde(default)]
    pub skill_search_threshold: Option<f32>,
    #[serde(default)]
    pub skill_min_hits: Option<usize>,
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

#[derive(Debug, Serialize, Clone)]
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
    pub storage_root_path: std::path::PathBuf,
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
        storage_root_path: std::path::PathBuf,
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
            storage_root_path,
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
    let allowed_tools = config.tools.clone();

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

fn tool_context_for_agent(
    context: &AgentContext,
    project_id: &str,
    session_id: Option<String>,
    parent_chain: Vec<String>,
) -> crate::tool_runtime::ToolContext {
    crate::tool_runtime::ToolContext {
        http_client: context.http_client.clone(),
        storage_root_path: context.storage_root_path.clone(),
        storage_base_url: context.storage_base_url.clone(),
        storage_auth_token: context.storage_auth_token.clone(),
        board_base_url: context.board_base_url.clone(),
        board_auth_token: context.board_auth_token.clone(),
        project_id: project_id.to_string(),
        session_id,
        parent_chain,
        self_agent_name: context.self_name.clone().unwrap_or_default(),
        pending_calls: context.pending_calls.clone(),
        outgoing_tasks: context.outgoing_tasks.clone(),
        agent_call_timeout_sec: context.agent_call_timeout_sec,
    }
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
    let mut tool_calls_log: Vec<ToolCallLogEntry> = Vec::new();
    let mut final_response = None;
    let mut posted_any = false;

    let mut injected_skill: Option<crate::skill_manager::SkillRecord> = None;

    engine.clear_skill_context();

    let skill_mode = config.skill_mode.as_deref().unwrap_or("auto");
    let skill_min_tool_calls = config.skill_min_tool_calls.unwrap_or(2);

    // ==================== Поиск и инжект скилла ====================
    if skill_mode == "auto" && !request.prompt.trim().is_empty() {
        let storage_base_url = context.storage_base_url.clone();
        let storage_auth_token = context.storage_auth_token.clone();
        let current_agent = config.name.clone();
        let prompt = request.prompt.clone();
        let phrases_top_n = config.skill_phrases_top_n.unwrap_or(15);
        let phrase_min_words = config.skill_phrase_min_words.unwrap_or(2);
        let threshold = config.skill_search_threshold.unwrap_or(0.45);
        let min_hits = config.skill_min_hits.unwrap_or(3);

        let skill_result = tokio::task::spawn_blocking(
            move || -> Result<Option<(crate::skill_manager::SkillRecord, f32)>> {
                let client = reqwest::blocking::Client::new();
                crate::skill_manager::search_best_skill(
                    &client,
                    &storage_base_url,
                    &storage_auth_token,
                    &current_agent,
                    &prompt,
                    phrases_top_n,
                    phrase_min_words,
                    threshold,
                    min_hits,
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
            eprintln!(
                "🔍 Найден скилл: {} (distance: {:.4})",
                skill_record.skill_file, distance
            );

            if let Some(calls) = crate::skill_manager::parse_tool_calls(&content) {
                let injection =
                    crate::skill_manager::format_skill_for_injection(&skill_record, &calls);
                engine.set_skill_context(injection);
                write_log(log_file, "skill_injected", &skill_record.skill_file)?;
                injected_skill = Some(skill_record);
            } else {
                write_log(log_file, "skill_parse_failed", &skill_record.skill_file)?;
            }
        }
    }

    // ==================== Авто-инжект знаний ====================
    {
        let auto_cfg = KnowledgeAutoConfig {
            top_n: config.knowledge_auto_top_n.unwrap_or(7),
            top_k: config.knowledge_auto_top_k.unwrap_or(3),
            threshold: config.knowledge_auto_threshold.unwrap_or(0.70),
            per_phrase: 5,
        };
        let turn_text = if !request.prompt.trim().is_empty() {
            Some(request.prompt.clone())
        } else {
            engine.pending_turn_text()
        };

        if let Some(text) = turn_text {
            match build_knowledge_context(
                &context.http_client,
                &context.storage_base_url,
                &context.storage_auth_token,
                &text,
                &auto_cfg,
            )
            .await
            {
                Ok(Some(ctx_str)) => {
                    let names: Vec<String> = ctx_str
                        .lines()
                        .filter_map(|l| {
                            let t = l.trim();
                            t.strip_prefix("=== ")
                                .and_then(|r| r.strip_suffix(" ==="))
                                .map(|s| s.to_string())
                        })
                        .collect();
                    engine.set_knowledge_context("auto", ctx_str);
                    eprintln!(
                        "📚 knowledge auto-inject (agent {}): injected [{}]",
                        config.name,
                        names.join(", ")
                    );
                }
                Ok(None) => {
                    engine.clear_knowledge_context();
                    eprintln!(
                        "📚 knowledge auto-inject (agent {}): nothing relevant",
                        config.name
                    );
                }
                Err(e) => {
                    eprintln!(
                        "⚠️ knowledge auto-inject (agent {}) failed: {:#}",
                        config.name, e
                    );
                }
            }
        } else {
            engine.clear_knowledge_context();
        }
    }

    // ==================== Цикл LLM ====================
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
                let (result, posted) = if crate::tool_runtime::find(&tc.function.name).is_some() {
                    let tool_ctx = tool_context_for_agent(
                        context,
                        &project_id,
                        session_id.clone(),
                        parent_chain.clone(),
                    );
                    let r = crate::tool_runtime::execute(
                        &tc.function.name,
                        &tc.function.arguments,
                        &tool_ctx,
                    )
                    .await;
                    let posted = r.contains("\"content\":\"posted:");
                    eprintln!("✅ Результат '{}': {}", tc.function.name, r);
                    (r, posted)
                } else {
                    let r = format!("Ошибка: неизвестный инструмент '{}'", tc.function.name);
                    eprintln!("❌ {}", r);
                    (r, false)
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

                if posted {
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

    let final_content = if !final_response.content.trim().is_empty() {
        final_response.content.clone()
    } else if final_response.tool_calls.is_some() {
        tool_calls_log
            .last()
            .map(|t| t.result.clone())
            .unwrap_or_else(|| "(пустой результат: исчерпан max_iterations)".to_string())
    } else if !final_response.reasoning.trim().is_empty() {
        final_response.reasoning.clone()
    } else {
        "(пустой ответ модели)".to_string()
    };

    // ==================== Auto-save скилла ====================
    if skill_mode == "auto" {
        let successful: Vec<&ToolCallLogEntry> = tool_calls_log
            .iter()
            .filter(|t| !t.result.contains("\"ok\":false") && !t.result.starts_with("Ошибка"))
            .collect();

        let min_calls = skill_min_tool_calls;

        if successful.len() < min_calls {
            write_log(
                log_file,
                "skill_skipped",
                &format!("(too few calls) ({} < {})", successful.len(), min_calls),
            )?;
            eprintln!(
                "⏭️ Скилл не сохранён: успешных вызовов {} < {}",
                successful.len(),
                min_calls
            );
        } else {
            let engine_config = build_engine_config(config);
            let (skill_name, skill_description) =
                match crate::skill_manager::generate_skill_metadata(&engine_config, &request.prompt)
                    .await
                {
                    Ok(meta) => meta,
                    Err(e) => {
                        write_log(log_file, "skill_metadata_failed", &format!("{}", e))?;
                        (String::new(), String::new())
                    }
                };

            if !skill_name.is_empty() || !skill_description.is_empty() {
                let tool_records: Vec<crate::skill_manager::ToolCallRecord> = successful
                    .iter()
                    .filter_map(|t| {
                        let args: serde_json::Value =
                            serde_json::from_str(&t.arguments).unwrap_or(serde_json::json!({}));
                        Some(crate::skill_manager::ToolCallRecord {
                            name: t.name.clone(),
                            arguments: args,
                        })
                    })
                    .collect();

                let storage_base_url = context.storage_base_url.clone();
                let storage_auth_token = context.storage_auth_token.clone();
                let agent_name = config.name.clone();
                let prompt = request.prompt.clone();
                let skill_name_for_log = skill_name.clone();

                let skill_phrases_top_n = config.skill_phrases_top_n.unwrap_or(15);
                let skill_phrase_min_words = config.skill_phrase_min_words.unwrap_or(2);

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
                        &tool_records,
                        min_calls,
                        skill_phrases_top_n,
                        skill_phrase_min_words,
                    )
                })
                .await
                {
                    Ok(Ok(o)) => o,
                    Ok(Err(e)) => {
                        write_log(log_file, "skill_save_error", &format!("{}", e))?;
                        eprintln!("⚠️ Ошибка сохранения скилла: {}", e);
                        crate::skill_manager::SaveSkillOutcome::SkippedTooFewCalls {
                            actual: 0,
                            min: 0,
                        }
                    }
                    Err(e) => {
                        write_log(log_file, "skill_save_error", &format!("Join error: {}", e))?;
                        eprintln!("⚠️ Join error при сохранении скилла: {}", e);
                        crate::skill_manager::SaveSkillOutcome::SkippedTooFewCalls {
                            actual: 0,
                            min: 0,
                        }
                    }
                };

                match outcome {
                    crate::skill_manager::SaveSkillOutcome::Saved => {
                        write_log(log_file, "skill_saved", &skill_name_for_log)?;
                        eprintln!("💾 Скилл сохранён: {}", skill_name_for_log);
                    }
                    crate::skill_manager::SaveSkillOutcome::SkippedTooFewCalls { actual, min } => {
                        write_log(
                            log_file,
                            "skill_skipped",
                            &format!("{} ({} < {})", skill_name_for_log, actual, min),
                        )?;
                    }
                    crate::skill_manager::SaveSkillOutcome::SkippedDuplicate => {
                        write_log(log_file, "skill_skipped_duplicate", &skill_name_for_log)?;
                        eprintln!("♻️ Скилл пропущен (дубликат): {}", skill_name_for_log);
                    }
                }
            }
        }
    }

    // ==================== Запись об использовании скилла ====================
    if let Some(rec) = injected_skill {
        let storage_base_url = context.storage_base_url.clone();
        let storage_auth_token = context.storage_auth_token.clone();
        let skill_file = rec.skill_file.clone();
        let success = !final_content.starts_with("(пустой");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_agent_config() -> AgentConfig {
        AgentConfig {
            name: "test".into(),
            bind_addr: "127.0.0.1:0".into(),
            auth_token: String::new(),
            timeout_sec: 60,
            max_iterations: 3,
            tools: vec!["storage_read_file".into()],
            system_prompt: None,
            api_key: "k".into(),
            base_url: "http://localhost".into(),
            model: "m".into(),
            temperature: Some(0.5),
            max_tokens: Some(1000),
            top_p: Some(1.0),
            stream: false,
            prefix_message_count: Some(2),
            tail_message_count: Some(3),
            reasoning_effort: None,
            agent_type: AgentType::default(),
            session_ttl_secs: None,
            skill_mode: None,
            skill_min_tool_calls: None,
            agent_call_timeout_sec: None,
            max_cost_rub: None,
            compact_threshold_bytes: None,
            tail_byte_budget: None,
            knowledge_auto_top_n: None,
            knowledge_auto_top_k: None,
            knowledge_auto_threshold: None,
            skill_phrases_top_n: None,
            skill_phrase_min_words: None,
            skill_search_threshold: None,
            skill_min_hits: None,
        }
    }

    #[test]
    fn agent_type_default_is_stateless() {
        assert_eq!(AgentType::default(), AgentType::Stateless);
    }

    #[test]
    fn build_engine_config_propagates_fields() {
        let cfg = dummy_agent_config();
        let engine = build_engine_config(&cfg);
        assert_eq!(engine.api_key, "k");
        assert_eq!(engine.model, "m");
        assert_eq!(engine.temperature, Some(0.5));
        assert_eq!(engine.max_tokens, Some(1000));
        assert!(!engine.stream);
        assert_eq!(engine.prefix_message_count, Some(2));
        assert_eq!(engine.tail_message_count, Some(3));
    }

    #[test]
    fn build_engine_config_default_compact_threshold() {
        let cfg = dummy_agent_config();
        let engine = build_engine_config(&cfg);
        assert_eq!(engine.compact_threshold_bytes, 512);
        assert_eq!(engine.tail_byte_budget, 100 * 1024);
    }

    #[test]
    fn build_engine_config_custom_compact_threshold() {
        let mut cfg = dummy_agent_config();
        cfg.compact_threshold_bytes = Some(1024);
        cfg.tail_byte_budget = Some(4096);
        let engine = build_engine_config(&cfg);
        assert_eq!(engine.compact_threshold_bytes, 1024);
        assert_eq!(engine.tail_byte_budget, 4096);
    }
}
