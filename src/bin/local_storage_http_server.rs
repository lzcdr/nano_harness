use clap::Parser;
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
    let args = Args::parse();
    let config = TomlConfig::load(&args.config)?;

    let http_cfg = config
        .http_server
        .ok_or_else(|| anyhow::anyhow!("Секция [http_server] не найдена в конфиге"))?;

    let vector_db_config = config.vector_db.unwrap_or_default();

    local_storage_http_api::run_server(http_cfg, vector_db_config).await
}
