// src/config.rs

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

use crate::agent_core::AgentConfig;
use crate::engine::{EngineConfig, ToolExecutionConfig};
use crate::local_storage::VectorDbConfig;
use crate::local_storage_http_api::LocalStorageServerConfig;

#[derive(Debug, Deserialize, Default)]
pub struct TomlConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<ToolExecutionConfig>>,
    pub tool_choice: Option<serde_json::Value>,
    pub reasoning_effort: Option<String>,
    pub max_cost_rub: Option<f64>,
    pub prefix_message_count: Option<usize>,
    pub tail_message_count: Option<usize>,
    pub system_prompt: Option<String>,

    // Глобальный таймаут Rhai для CLI (чата), секунды
    pub rhai_timeout_sec: Option<u64>,

    pub vector_db: Option<VectorDbConfig>,
    pub local_storage_http_server: Option<LocalStorageServerConfig>,

    #[serde(default)]
    pub agents: Vec<AgentConfig>,
}

impl TomlConfig {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let candidates = vec![
            path.clone(),
            PathBuf::from("config.toml"),
            dirs::config_dir()
                .unwrap_or_default()
                .join("polza_chat")
                .join("config.toml"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                let content = std::fs::read_to_string(&candidate)
                    .with_context(|| format!("Не удалось прочитать {:?}", candidate))?;
                return toml::from_str(&content)
                    .with_context(|| format!("Ошибка разбора {:?}", candidate));
            }
        }

        Ok(TomlConfig::default())
    }
}

pub fn build_engine_config(
    cli_api_key: Option<String>,
    cli_base_url: Option<String>,
    cli_model: Option<String>,
    cli_temperature: Option<f32>,
    cli_max_tokens: Option<u32>,
    cli_stream: Option<bool>,
    cli_reasoning_effort: Option<String>,
    cli_max_cost_rub: Option<f64>,
    cli_prefix_message_count: Option<usize>,
    cli_tail_message_count: Option<usize>,
    toml_config: &TomlConfig,
) -> Result<EngineConfig> {
    let api_key = cli_api_key
        .or_else(|| std::env::var("POLZA_API_KEY").ok())
        .or_else(|| toml_config.api_key.clone())
        .context("API-ключ не задан. Используйте --api-key, POLZA_API_KEY или config.toml")?;

    let base_url = cli_base_url
        .or_else(|| toml_config.base_url.clone())
        .unwrap_or_else(|| "https://polza.ai/api/v1".to_string());

    let model = cli_model
        .or_else(|| toml_config.model.clone())
        .unwrap_or_else(|| "openai/gpt-4o".to_string());

    let temperature = cli_temperature.or(toml_config.temperature);
    let max_tokens = cli_max_tokens.or(toml_config.max_tokens);
    let stream = cli_stream.or(toml_config.stream).unwrap_or(true);
    let reasoning_effort = cli_reasoning_effort.or_else(|| toml_config.reasoning_effort.clone());
    let max_cost_rub = cli_max_cost_rub.or(toml_config.max_cost_rub);
    let prefix_message_count = cli_prefix_message_count.or(toml_config.prefix_message_count);
    let tail_message_count = cli_tail_message_count.or(toml_config.tail_message_count);
    let allowed_tools = toml_config.tools.clone();

    Ok(EngineConfig {
        api_key,
        base_url,
        model,
        temperature,
        max_tokens,
        top_p: toml_config.top_p,
        stop: toml_config.stop.clone(),
        stream,
        allowed_tools,
        tool_choice: toml_config.tool_choice.clone(),
        reasoning_effort,
        max_cost_rub,
        prefix_message_count,
        tail_message_count,
    })
}
