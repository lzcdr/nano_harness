use clap::Parser;
use nano_harness::agent_core::AgentContext;
use nano_harness::agent_http_api;
use nano_harness::config::TomlConfig;
use std::path::PathBuf;

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
    let args = Args::parse();
    let config = TomlConfig::load(&args.config)?;

    let agent_config = config
        .agents
        .iter()
        .find(|a| a.name == args.agent_name)
        .ok_or_else(|| anyhow::anyhow!("Агент '{}' не найден в конфиге", args.agent_name))?;

    // Получаем параметры хранилища из конфига
    let storage_config = config
        .local_storage_http_server
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Секция [local_storage_http_server] не найдена"))?;

    let context = AgentContext::new(
        storage_config.bind_addr.clone(),
        storage_config.auth_token.clone(),
        agent_config.timeout_sec,
    );

    agent_http_api::run_server(agent_config.clone(), context).await
}
