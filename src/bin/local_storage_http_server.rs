// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use clap::Parser;
use dotenvy;
use nano_harness::config::TomlConfig;
use nano_harness::local_storage_http_api;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "local_storage_http_server")]
struct Args {
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();
    let config = TomlConfig::load(&args.config)?;

    let mut server_cfg = config.local_storage_http_server.ok_or_else(|| {
        anyhow::anyhow!("Секция [local_storage_http_server] не найдена в конфиге")
    })?;

    if server_cfg.auth_token.is_empty() {
        server_cfg.auth_token = std::env::var("LOCAL_STORAGE_AUTH_TOKEN").unwrap_or_default();
    }

    let vector_db_config = config.vector_db.unwrap_or_default();

    local_storage_http_api::run_server(server_cfg, vector_db_config).await
}
