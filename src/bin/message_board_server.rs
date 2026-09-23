// src/bin/message_board_server.rs

use clap::Parser;
use dotenvy;
use nano_harness::config::TomlConfig;
use nano_harness::message_board::{build_router, MessageBoard};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "message_board_server")]
struct Args {
    #[arg(long)]
    bind_addr: Option<String>,

    #[arg(long)]
    auth_token: Option<String>,

    #[arg(long)]
    tasks_dir: Option<PathBuf>,

    #[arg(long)]
    task_timeout_sec: Option<u64>,

    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("⏹️ Получен сигнал завершения, останавливаю...");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();
    let toml_config = TomlConfig::load(&args.config)?;
    let board_config = toml_config.message_board.as_ref();

    let bind_addr = args
        .bind_addr
        .or_else(|| board_config.map(|c| c.bind_addr.clone()))
        .unwrap_or_else(|| "127.0.0.1:8090".to_string());

    let mut auth_token = args
        .auth_token
        .or_else(|| board_config.map(|c| c.auth_token.clone()))
        .unwrap_or_default();

    if auth_token.is_empty() {
        auth_token = std::env::var("MESSAGE_BOARD_AUTH_TOKEN").unwrap_or_default();
    }

    let tasks_dir = args
        .tasks_dir
        .or_else(|| board_config.map(|c| PathBuf::from(&c.tasks_dir)))
        .unwrap_or_else(|| PathBuf::from(".board/tasks"));

    let task_timeout_sec = args
        .task_timeout_sec
        .or_else(|| board_config.map(|c| c.task_timeout_sec))
        .unwrap_or(300);

    let board = Arc::new(MessageBoard::new(tasks_dir.clone(), task_timeout_sec).await?);
    let app = build_router(board, auth_token);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("📋 Доска сообщений запущена на http://{}", bind_addr);
    println!("   Папка задач: {}", tasks_dir.display());
    println!("   Таймаут задачи: {} сек", task_timeout_sec);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    println!("✅ Доска сообщений остановлена корректно");
    Ok(())
}
