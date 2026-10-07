// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/config.rs

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

use crate::agent_core::AgentConfig;
use crate::engine::{EngineConfig, ToolExecutionConfig};
use crate::godfather::GodfatherConfig;
use crate::local_storage::VectorDbConfig;
use crate::local_storage_http_api::LocalStorageServerConfig;
use crate::message_board::MessageBoardConfig;

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
    pub max_iterations: Option<usize>,
    pub system_prompt: Option<String>,

    // Авто-инжект знаний в контекст (чат и агенты).
    pub knowledge_auto_top_n: Option<usize>,
    pub knowledge_auto_top_k: Option<usize>,
    pub knowledge_auto_threshold: Option<f32>,

    pub skill_mode: Option<String>,
    pub skill_min_tool_calls: Option<usize>,
    pub skill_phrases_top_n: Option<usize>,
    pub skill_phrase_min_words: Option<usize>,
    pub skill_search_threshold: Option<f32>,
    pub skill_min_hits: Option<usize>,

    // Глобальный таймаут Rhai для CLI (чата), секунды
    pub rhai_timeout_sec: Option<u64>,

    // Редактор для /rebuke_edit. Если не задан — $EDITOR, иначе notepad (Windows) / vi.
    pub editor: Option<String>,

    // Порог компактизации tool-результатов в байтах.
    pub compact_threshold_bytes: Option<usize>,

    // Бюджет хвоста в байтах.
    pub tail_byte_budget: Option<usize>,

    pub vector_db: Option<VectorDbConfig>,
    pub local_storage_http_server: Option<LocalStorageServerConfig>,
    pub message_board: Option<MessageBoardConfig>,
    pub godfather: Option<GodfatherConfig>,

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
        compact_threshold_bytes: toml_config.compact_threshold_bytes.unwrap_or(512),
        tail_byte_budget: toml_config.tail_byte_budget.unwrap_or(100 * 1024),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn toml_config_load_parses_minimal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "api_key = \"k\"").unwrap();
        writeln!(f, "model = \"m\"").unwrap();
        drop(f);
        let cfg = TomlConfig::load(&path).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("k"));
        assert_eq!(cfg.model.as_deref(), Some("m"));
    }

    #[test]
    fn toml_config_load_parses_agents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"
api_key = "k"
[[agents]]
name = "analyzer"
bind_addr = "127.0.0.1:8081"
auth_token = ""
timeout_sec = 60
max_iterations = 3
tools = ["storage_read_file"]
api_key = ""
base_url = "http://x"
model = "m"
stream = true
"#
        )
        .unwrap();
        drop(f);
        let cfg = TomlConfig::load(&path).unwrap();
        assert_eq!(cfg.agents.len(), 1);
        assert_eq!(cfg.agents[0].name, "analyzer");
    }

    #[test]
    fn build_engine_config_uses_cli_api_key_first() {
        let toml = TomlConfig::default();
        let cfg = build_engine_config(
            Some("cli-key".into()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &toml,
        )
        .unwrap();
        assert_eq!(cfg.api_key, "cli-key");
    }

    #[test]
    fn build_engine_config_falls_back_to_toml_api_key() {
        let mut toml = TomlConfig::default();
        toml.api_key = Some("toml-key".into());
        let cfg = build_engine_config(
            None, None, None, None, None, None, None, None, None, None, &toml,
        )
        .unwrap();
        assert_eq!(cfg.api_key, "toml-key");
    }

    #[test]
    fn build_engine_config_default_base_url() {
        let toml = TomlConfig {
            api_key: Some("k".into()),
            ..Default::default()
        };
        let cfg = build_engine_config(
            None, None, None, None, None, None, None, None, None, None, &toml,
        )
        .unwrap();
        assert_eq!(cfg.base_url, "https://polza.ai/api/v1");
    }

    #[test]
    fn build_engine_config_default_model() {
        let toml = TomlConfig {
            api_key: Some("k".into()),
            ..Default::default()
        };
        let cfg = build_engine_config(
            None, None, None, None, None, None, None, None, None, None, &toml,
        )
        .unwrap();
        assert_eq!(cfg.model, "openai/gpt-4o");
    }

    #[test]
    fn build_engine_config_default_stream_true() {
        let toml = TomlConfig {
            api_key: Some("k".into()),
            ..Default::default()
        };
        let cfg = build_engine_config(
            None, None, None, None, None, None, None, None, None, None, &toml,
        )
        .unwrap();
        assert!(cfg.stream);
    }

    #[test]
    fn build_engine_config_cli_overrides_toml() {
        let toml = TomlConfig {
            api_key: Some("toml-key".into()),
            model: Some("toml-model".into()),
            temperature: Some(0.1),
            ..Default::default()
        };
        let cfg = build_engine_config(
            None,
            None,
            Some("cli-model".into()),
            Some(0.9),
            None,
            None,
            None,
            None,
            None,
            None,
            &toml,
        )
        .unwrap();
        assert_eq!(cfg.model, "cli-model");
        assert_eq!(cfg.temperature, Some(0.9));
    }

    #[test]
    fn build_engine_config_default_compact_thresholds() {
        let toml = TomlConfig {
            api_key: Some("k".into()),
            ..Default::default()
        };
        let cfg = build_engine_config(
            None, None, None, None, None, None, None, None, None, None, &toml,
        )
        .unwrap();
        assert_eq!(cfg.compact_threshold_bytes, 512);
        assert_eq!(cfg.tail_byte_budget, 100 * 1024);
    }
}
