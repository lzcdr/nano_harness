// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/bin/agent_server.rs

use clap::Parser;
use dotenvy::{self};
use nano_harness::agent_core::{AgentContext, OutgoingTasks, PendingCalls};
use nano_harness::agent_http_api;
use nano_harness::config::TomlConfig;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(name = "agent_server")]
struct Args {
    #[arg(long)]
    agent_name: String,

    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let _ = nano_harness::lexicon::init_lexicon(std::path::Path::new(".dict"));
    let args = Args::parse();
    let config = TomlConfig::load(&args.config)?;

    let mut agent_config = config
        .agents
        .iter()
        .find(|a| a.name == args.agent_name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Агент '{}' не найден в конфиге", args.agent_name))?;

    // Если у агента не задан max_cost_rub — берём глобальный.
    if agent_config.max_cost_rub.is_none() {
        agent_config.max_cost_rub = config.max_cost_rub;
    }

    if agent_config.vector_db_top_k.is_none() {
        agent_config.vector_db_top_k = config.vector_db.as_ref().map(|v| v.top_k);
    }

    if agent_config.auth_token.is_empty() {
        let key = format!(
            "AGENT_{}_AUTH_TOKEN",
            agent_config.name.to_ascii_uppercase()
        );
        agent_config.auth_token = std::env::var(&key).unwrap_or_default();
    }

    if agent_config.api_key.is_empty() {
        agent_config.api_key = std::env::var("POLZA_API_KEY")
            .ok()
            .or_else(|| config.api_key.clone())
            .unwrap_or_default();
    }

    if agent_config.base_url.is_empty() {
        agent_config.base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://polza.ai/api/v1".to_string());
    }

    if agent_config.model.is_empty() {
        agent_config.model = config
            .model
            .clone()
            .unwrap_or_else(|| "openai/gpt-4o".to_string());
    }

    let mut storage_config = config
        .local_storage_http_server
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Секция [local_storage_http_server] не найдена"))?
        .clone();

    if storage_config.auth_token.is_empty() {
        storage_config.auth_token = std::env::var("LOCAL_STORAGE_AUTH_TOKEN").unwrap_or_default();
    }

    let board_url = config
        .message_board
        .as_ref()
        .map(|c| c.bind_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8090".to_string());

    let mut board_token = config
        .message_board
        .as_ref()
        .map(|c| c.auth_token.clone())
        .unwrap_or_default();

    if board_token.is_empty() {
        board_token = std::env::var("MESSAGE_BOARD_AUTH_TOKEN").unwrap_or_default();
    }

    let pending_calls: PendingCalls = Arc::new(Mutex::new(HashMap::new()));
    let outgoing_tasks: OutgoingTasks = Arc::new(Mutex::new(HashMap::new()));

    let agent_call_timeout = agent_config.agent_call_timeout_sec.unwrap_or(120);
    let storage_root_path =
        std::path::PathBuf::from(".local_storage").join(&storage_config.storage_name);

    let context = AgentContext::new(
        storage_config.bind_addr.clone(),
        storage_config.auth_token.clone(),
        storage_root_path,
        agent_config.timeout_sec,
        Some(agent_config.name.clone()),
        agent_call_timeout,
        board_url,
        board_token,
        pending_calls,
        outgoing_tasks,
    )?;

    agent_http_api::run_server(agent_config, context).await
}
