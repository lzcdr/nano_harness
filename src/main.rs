// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/main.rs

use anyhow::{Context, Result};
use clap::Parser;
use dotenvy;
use reqwest::Client;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::StreamExt;

use nano_harness::agent_core::{OutgoingTasks, PendingCalls, ToolCallLogEntry};
use nano_harness::config::{build_engine_config, TomlConfig};
use nano_harness::engine::{ChatEngine, EngineConfig, Message, Role};
use nano_harness::godfather::GodfatherConfig;
use nano_harness::knowledge_auto::{build_knowledge_context, KnowledgeAutoConfig};
use nano_harness::knowledge_manager::KnowledgeEntry;
use nano_harness::local_storage_http_api::LocalStorageServerConfig;
use nano_harness::message_board::BoardEvent;
use nano_harness::rebuke_manager;
use nano_harness::session_store::{self, ContextBlock, Session};
use nano_harness::tool_search::{select_tools, ToolsSearchConfig};

#[derive(Parser, Debug)]
#[command(author, version, about = "CLI чат с Polza AI")]
struct Args {
    #[arg(short = 'k', long, help = "API ключ")]
    api_key: Option<String>,

    #[arg(short = 'u', long, help = "Base URL API")]
    base_url: Option<String>,

    #[arg(short = 'm', long, help = "Модель")]
    model: Option<String>,

    #[arg(long, help = "Температура (0-2)")]
    temperature: Option<f32>,

    #[arg(long, help = "Максимум токенов")]
    max_tokens: Option<u32>,

    #[arg(long, help = "Включить/выключить стриминг")]
    stream: Option<bool>,

    #[arg(long, help = "Уровень reasoning: low, medium, high")]
    reasoning_effort: Option<String>,

    #[arg(long, help = "Максимальный бюджет в рублях")]
    max_cost_rub: Option<f64>,

    #[arg(long, help = "Количество первых пар (запрос-ответ) для префикса")]
    prefix_message_count: Option<usize>,

    #[arg(long, help = "Количество последних пар для хвоста")]
    tail_message_count: Option<usize>,

    #[arg(long, help = "Системный промпт")]
    system_prompt: Option<String>,

    #[arg(
        long = "config",
        default_value = "config.toml",
        help = "Путь к конфигу"
    )]
    config_path: PathBuf,

    #[arg(long, default_value = "60", help = "Таймаут HTTP в секундах")]
    timeout_sec: u64,

    #[arg(
        long,
        help = "Имя проекта для запуска. Если не задано — последний проект."
    )]
    project: Option<String>,

    #[arg(
        long,
        default_value = "120",
        help = "Таймаут синхронного call_agent в секундах"
    )]
    agent_call_timeout_sec: u64,

    #[arg(long, default_value = "300", help = "Таймаут godfather в секундах")]
    godfather_timeout_sec: u64,
}

// ==================== Разделяемое состояние чата ====================

struct ChatRuntime {
    session: Session,
    engine: ChatEngine,
    log_file: std::fs::File,
    project_id: String,
    outgoing_tasks: OutgoingTasks,
    pending_calls: PendingCalls,
    tool_calls_log: Vec<ToolCallLogEntry>,
    injected_skill: Option<nano_harness::skill_manager::SkillRecord>,
}

type SharedChat = Arc<AsyncMutex<ChatRuntime>>;

#[derive(Clone)]
struct ReplyContext {
    engine_config: Arc<EngineConfig>,
    client: Client,
    storage_http_config: Arc<LocalStorageServerConfig>,
    agent_call_timeout_sec: u64,
    board_url: String,
    board_token: String,
    knowledge_auto: Arc<KnowledgeAutoConfig>,
    tools_search: Arc<ToolsSearchConfig>,
    allowed_tools_chat: Arc<Vec<String>>,
}

// ==================== Утилиты ====================

fn write_log(file: &mut std::fs::File, role: &str, content: &str) -> Result<()> {
    writeln!(
        file,
        "[{}] {}: {}",
        chrono::Local::now().format("%H:%M:%S"),
        role,
        content
    )?;
    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn sanitize_project_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "default".to_string()
    } else {
        s
    }
}

fn make_project_id(name: &str) -> String {
    let sanitized = sanitize_project_name(name);
    let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    format!("{}_{}", sanitized, ts)
}

fn editor_name(toml_config: &TomlConfig) -> String {
    if let Some(e) = &toml_config.editor {
        e.clone()
    } else if let Ok(e) = std::env::var("EDITOR") {
        e
    } else if cfg!(target_os = "windows") {
        "notepad".to_string()
    } else {
        "vi".to_string()
    }
}

fn run_editor_sync(editor: &str, path: &std::path::Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("не удалось создать каталог {}", parent.display()))?;
    }
    let status = std::process::Command::new(editor)
        .arg(path)
        .status()
        .with_context(|| format!("не удалось запустить редактор '{}'", editor))?;
    if !status.success() {
        anyhow::bail!("редактор '{}' завершился с ошибкой", editor);
    }
    Ok(())
}

fn edit_buffer_path(kind: &str, name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(".sessions")
        .join(kind)
        .join(format!("{}.txt", name))
}

async fn edit_via_buffer(
    kind: &str,
    name: &str,
    initial_content: &str,
    editor: &str,
) -> Result<String> {
    let buffer = edit_buffer_path(kind, name);
    if let Some(parent) = buffer.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&buffer, initial_content)?;

    if let Err(e) = run_editor_sync(editor, &buffer) {
        let _ = std::fs::remove_file(&buffer);
        return Err(e);
    }

    let edited = std::fs::read_to_string(&buffer)?;
    let _ = std::fs::remove_file(&buffer);
    Ok(edited)
}

async fn refresh_knowledge_catalog(
    client: &Client,
    storage_base_url: &str,
    storage_auth_token: &str,
) -> Result<Vec<KnowledgeEntry>> {
    nano_harness::knowledge_manager::list_knowledge(client, storage_base_url, storage_auth_token)
        .await
}

fn build_tool_context(
    ctx: &ReplyContext,
    project_id: String,
    session_id: Option<String>,
    self_agent_name: String,
    pending_calls: PendingCalls,
    outgoing_tasks: OutgoingTasks,
) -> nano_harness::tool_runtime::ToolContext {
    let storage_url = if ctx.storage_http_config.bind_addr.starts_with("http") {
        ctx.storage_http_config.bind_addr.clone()
    } else {
        format!("http://{}", ctx.storage_http_config.bind_addr)
    };
    let board_url = if ctx.board_url.starts_with("http") {
        ctx.board_url.clone()
    } else {
        format!("http://{}", ctx.board_url)
    };
    let storage_root_path =
        std::path::PathBuf::from(".local_storage").join(&ctx.storage_http_config.storage_name);
    nano_harness::tool_runtime::ToolContext {
        http_client: ctx.client.clone(),
        storage_root_path,
        storage_base_url: storage_url,
        storage_auth_token: ctx.storage_http_config.auth_token.clone(),
        board_base_url: board_url,
        board_auth_token: ctx.board_token.clone(),
        project_id,
        session_id,
        parent_chain: vec![],
        self_agent_name,
        pending_calls,
        outgoing_tasks,
        agent_call_timeout_sec: ctx.agent_call_timeout_sec,
    }
}

fn open_project(
    id: &str,
    display_name: &str,
    engine_config: &EngineConfig,
    client: &Client,
    default_system_prompt: &str,
) -> Result<(Session, ChatEngine, std::fs::File)> {
    let session = match session_store::load_session_with_key(id, &engine_config.api_key, None) {
        Ok(s) => s,
        Err(_) => {
            let mut s = session_store::create_session_with_id(id, display_name, None)?;
            s.context = Some(ContextBlock {
                engine_config: engine_config.clone(),
                engine_state: nano_harness::session_store::EngineState {
                    system_messages: vec![Message {
                        role: Role::System,
                        content: Some(default_system_prompt.to_string()),
                        reasoning: None,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    }],
                    skill_context: None,
                    knowledge_context: None,
                    prefix_turns: vec![],
                    tail_turns: vec![],
                    pending_turn: None,
                    metrics: Default::default(),
                },
            });
            if let Err(e) = session_store::save_session(&s) {
                eprintln!("Ошибка сохранения нового проекта: {}", e);
            }
            s
        }
    };

    let mut engine = ChatEngine::new(engine_config.clone(), client.clone());
    engine.on_token = Some(Box::new(|token| {
        print!("{}", token);
        let _ = io::stdout().flush();
    }));
    engine.on_reasoning_token = Some(Box::new(|token| {
        print!("\x1b[90m{}\x1b[0m", token);
        let _ = io::stdout().flush();
    }));
    engine.on_tool_call_token = Some(Box::new(|s| {
        print!("\x1b[36m{}\x1b[0m", s);
        let _ = io::stdout().flush();
    }));

    if let Some(ctx) = session.context.clone() {
        engine.set_state(ctx.engine_state);
    } else {
        engine.add_message(Role::System, default_system_prompt.to_string());
    }

    let log_file = session_store::open_log(id, "chat", None)?;

    Ok((session, engine, log_file))
}

fn save_current(rt: &mut ChatRuntime) -> Result<()> {
    rt.session.context = Some(ContextBlock {
        engine_config: rt.engine.get_config().clone(),
        engine_state: rt.engine.get_state(),
    });
    rt.session.updated_at = session_store::now_ts();
    session_store::save_session(&rt.session)
}

async fn check_active_tasks(
    client: &Client,
    board_url: &str,
    board_token: &str,
    session_id: &str,
) -> Result<Vec<String>> {
    let url = format!(
        "{}/tasks?session_id={}&status=in_progress,pending",
        board_url.trim_end_matches('/'),
        urlencoding::encode(session_id)
    );
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", board_token))
        .send()
        .await
        .context("Не удалось подключиться к доске")?;
    if !resp.status().is_success() {
        anyhow::bail!("Доска вернула HTTP {}", resp.status());
    }
    let tasks: Vec<serde_json::Value> =
        resp.json().await.context("Ошибка парсинга списка задач")?;
    Ok(tasks
        .into_iter()
        .filter_map(|t| t.get("id").and_then(|v| v.as_str()).map(String::from))
        .collect())
}

// ==================== SSE-слушатель чата ====================

async fn chat_sse_loop(
    shared: SharedChat,
    board_url: String,
    board_token: String,
    reply_ctx: Arc<ReplyContext>,
) {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);
    let mut errors = 0u32;

    loop {
        match connect_and_listen_chat(&shared, &board_url, &board_token, &reply_ctx).await {
            Ok(()) => {
                errors = 0;
                backoff = Duration::from_secs(1);
            }
            Err(e) => {
                errors += 1;
                if errors <= 3 || errors % 10 == 0 {
                    eprintln!("❌ SSE чата ошибка #{}: {}", errors, e);
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(max_backoff);
                continue;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn connect_and_listen_chat(
    shared: &SharedChat,
    board_url: &str,
    board_token: &str,
    reply_ctx: &ReplyContext,
) -> anyhow::Result<()> {
    let url = format!("{}/events?agent_name=chat", board_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", board_token))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].to_string();
            buffer.drain(..=pos);
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data: ") {
                match serde_json::from_str::<BoardEvent>(data) {
                    Ok(event) => handle_chat_board_event(shared, event, reply_ctx).await,
                    Err(e) => eprintln!("Ошибка парсинга SSE чата: {}", e),
                }
            }
        }
    }
    Ok(())
}

async fn handle_chat_board_event(shared: &SharedChat, event: BoardEvent, reply_ctx: &ReplyContext) {
    match event {
        BoardEvent::TaskCreated { .. } => {}
        BoardEvent::TaskCompleted {
            task_id,
            result,
            from_agent,
            from_session_id,
        } => {
            handle_chat_result(
                shared,
                task_id,
                result,
                from_agent,
                from_session_id,
                false,
                reply_ctx,
            )
            .await;
        }
        BoardEvent::TaskFailed {
            task_id,
            error,
            from_agent,
            from_session_id,
        } => {
            handle_chat_result(
                shared,
                task_id,
                error,
                from_agent,
                from_session_id,
                true,
                reply_ctx,
            )
            .await;
        }
    }
}

async fn handle_chat_result(
    shared: &SharedChat,
    task_id: String,
    result: String,
    from_agent: String,
    from_session_id: String,
    is_error: bool,
    reply_ctx: &ReplyContext,
) {
    {
        let rt = shared.lock().await;
        let mut pc = rt.pending_calls.lock().await;
        if let Some(entry) = pc.remove(&task_id) {
            let _ = entry.tx.send(result.clone());
            return;
        }
    }

    let mut rt = shared.lock().await;
    let outgoing = {
        let mut map = rt.outgoing_tasks.lock().await;
        map.remove(&task_id)
    };
    let Some(outgoing) = outgoing else {
        eprintln!("📬 Ответ для неизвестной задачи {}, игнорирую", task_id);
        return;
    };

    if outgoing.session_id != rt.project_id {
        eprintln!(
            "📬 Ответ для проекта {} (текущий {}), игнорирую",
            outgoing.session_id, rt.project_id
        );
        return;
    }

    let header = if is_error {
        format!("[Ошибка от агента {}]: {}", from_agent, result)
    } else {
        format!("[Ответ от агента {}]: {}", from_agent, result)
    };

    rt.engine.add_message(Role::User, header.clone());
    if let Err(e) = write_log(&mut rt.log_file, "task_result", &header) {
        eprintln!("Ошибка записи в лог: {}", e);
    }

    if let Err(e) = save_current(&mut *rt) {
        eprintln!("Ошибка сохранения сессии: {}", e);
    }

    eprintln!(
        "📬 Ответ от {} добавлен в контекст проекта {}",
        from_agent, rt.project_id
    );
    let _ = from_session_id;

    drop(rt);

    continue_chat_turn(shared, reply_ctx).await;
}

async fn continue_chat_turn(shared: &SharedChat, ctx: &ReplyContext) {
    const MAX_ITER: usize = 5;

    // Авто-инжект знаний.
    {
        let turn_text = {
            let rt = shared.lock().await;
            rt.engine.pending_turn_text()
        };
        if let Some(text) = turn_text {
            let storage_url = &ctx.storage_http_config.bind_addr;
            let token = &ctx.storage_http_config.auth_token;
            match build_knowledge_context(
                &ctx.client,
                storage_url,
                token,
                &text,
                &ctx.knowledge_auto,
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
                    let mut rt = shared.lock().await;
                    rt.engine.set_knowledge_context("auto", ctx_str);
                    eprintln!("📚 knowledge auto-inject: injected [{}]", names.join(", "));
                }
                Ok(None) => {
                    let mut rt = shared.lock().await;
                    rt.engine.clear_knowledge_context();
                    eprintln!("📚 knowledge auto-inject: nothing relevant");
                }
                Err(e) => eprintln!("⚠️ knowledge auto-inject failed: {:#}", e),
            }
        }
    }

    // ==================== Выбор тулзов ====================
    {
        let (current_slots, prompt) = {
            let rt = shared.lock().await;
            let slots = rt
                .engine
                .visible_tools()
                .map(|s| s.to_vec())
                .unwrap_or_default();
            let prompt = rt.engine.pending_turn_text().unwrap_or_default();
            (slots, prompt)
        };
        let allowed = ctx.allowed_tools_chat.clone();
        let storage_url = ctx.storage_http_config.bind_addr.clone();
        let token = ctx.storage_http_config.auth_token.clone();
        let cfg = ctx.tools_search.clone();

        let selection = tokio::task::spawn_blocking(move || {
            let client = reqwest::blocking::Client::new();
            select_tools(
                &client,
                &storage_url,
                &token,
                &prompt,
                &current_slots,
                &allowed,
                &cfg,
            )
        })
        .await;

        match selection {
            Ok(Ok(names)) if !names.is_empty() => {
                let mut rt = shared.lock().await;
                rt.engine.set_visible_tools(names);
            }
            _ => {}
        }
    }

    for _ in 0..=MAX_ITER {
        print!("💭 ");
        let _ = io::stdout().flush();
        let response = {
            let mut rt = shared.lock().await;
            match rt.engine.send().await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("\n❌ Ошибка продолжения после async-ответа: {:#}", e);
                    let _ = write_log(&mut rt.log_file, "error", &format!("{:#}", e));
                    rt.engine.rollback_pending_turn();
                    return;
                }
            }
        };

        println!();
        let has_content = !response.content.trim().is_empty();
        {
            let mut rt = shared.lock().await;
            let answer = if has_content {
                response.content.clone()
            } else {
                response.reasoning.clone()
            };
            if !answer.is_empty() {
                let _ = write_log(&mut rt.log_file, "assistant", &answer);
            }
            if !response.reasoning.is_empty() && has_content {
                let _ = write_log(&mut rt.log_file, "reasoning", &response.reasoning);
            }
        }
        if !response.reasoning.is_empty() && has_content {
            println!("\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n", response.reasoning);
        }

        let tool_calls = match response.tool_calls {
            Some(tc) => tc,
            None => break,
        };

        let mut any_posted = false;
        for tc in &tool_calls {
            let name = &tc.function.name;
            let mode = ctx
                .engine_config
                .allowed_tools
                .as_ref()
                .and_then(|l| l.iter().find(|a| a.name == *name).map(|a| a.mode.clone()))
                .unwrap_or_else(|| "manual".to_string());

            if mode != "auto" {
                let mut rt = shared.lock().await;
                let _ = write_log(
                    &mut rt.log_file,
                    "tool_skipped",
                    &format!(
                        "{} ({}) - manual skipped in async",
                        name, tc.function.arguments
                    ),
                );
                rt.engine.add_tool_result(
                    tc.id.clone(),
                    "Пропущено: ручной режим недоступен в async-продолжении".to_string(),
                );
                continue;
            }

            let tool_ctx = {
                let rt = shared.lock().await;
                let pid = rt.project_id.clone();
                build_tool_context(
                    ctx,
                    pid.clone(),
                    Some(pid),
                    "chat".to_string(),
                    rt.pending_calls.clone(),
                    rt.outgoing_tasks.clone(),
                )
            };

            let result =
                nano_harness::tool_runtime::execute(name, &tc.function.arguments, &tool_ctx).await;

            let posted = result.contains("\"content\":\"posted:");

            {
                let mut rt = shared.lock().await;
                let _ = write_log(
                    &mut rt.log_file,
                    "tool_result",
                    &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                );
                rt.engine.add_tool_result(tc.id.clone(), result.clone());
                rt.tool_calls_log.push(ToolCallLogEntry {
                    name: name.clone(),
                    arguments: tc.function.arguments.clone(),
                    result: result.clone(),
                });
            }

            if posted {
                any_posted = true;
            }
        }

        if any_posted {
            break;
        }
    }

    {
        let mut rt = shared.lock().await;
        if let Err(e) = save_current(&mut *rt) {
            eprintln!("Ошибка сохранения сессии: {}", e);
        }
        println!(
            "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB | Контекст: {}]\x1b[0m\n",
            rt.engine.metrics.total_prompt_tokens,
            rt.engine.metrics.total_completion_tokens,
            rt.engine.metrics.total_cost_rub,
            format_bytes(rt.engine.total_context_bytes())
        );
    }
}

// ==================== main ====================

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let _ = nano_harness::lexicon::init_lexicon(std::path::Path::new(".dict"));
    let args = Args::parse();

    let toml_config = TomlConfig::load(&args.config_path)?;

    let engine_config = build_engine_config(
        args.api_key,
        args.base_url,
        args.model,
        args.temperature,
        args.max_tokens,
        args.stream,
        args.reasoning_effort,
        args.max_cost_rub,
        args.prefix_message_count,
        args.tail_message_count,
        &toml_config,
    )?;

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()?;

    let base_system_prompt = args
        .system_prompt
        .clone()
        .or(toml_config.system_prompt.clone())
        .unwrap_or_else(|| "Ты полезный ассистент. Отвечай кратко и по делу.".to_string());

    let mut storage_http_config = toml_config
        .local_storage_http_server
        .clone()
        .unwrap_or_else(|| LocalStorageServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            storage_name: "default_storage".to_string(),
            auth_token: String::new(),
        });
    if storage_http_config.auth_token.is_empty() {
        storage_http_config.auth_token =
            std::env::var("LOCAL_STORAGE_AUTH_TOKEN").unwrap_or_default();
    }

    let chat_rebuke = rebuke_manager::load_rebuke(
        &client,
        &storage_http_config.bind_addr,
        &storage_http_config.auth_token,
        "chat",
    )
    .await
    .unwrap_or_default();

    let system_prompt = rebuke_manager::build_system_with_rebuke(&base_system_prompt, &chat_rebuke);

    let board_url = toml_config
        .message_board
        .as_ref()
        .map(|c| c.bind_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8090".to_string());
    let mut board_token = toml_config
        .message_board
        .as_ref()
        .map(|c| c.auth_token.clone())
        .unwrap_or_default();
    if board_token.is_empty() {
        board_token = std::env::var("MESSAGE_BOARD_AUTH_TOKEN").unwrap_or_default();
    }
    let board_url = if board_url.starts_with("http") {
        board_url
    } else {
        format!("http://{}", board_url)
    };

    let godfather_config: Option<Arc<GodfatherConfig>> = toml_config
        .godfather
        .as_ref()
        .map(|cfg| {
            let mut c = cfg.clone();
            if c.api_key.is_empty() {
                c.api_key = std::env::var("GODFATHER_API_KEY")
                    .ok()
                    .or_else(|| std::env::var("POLZA_API_KEY").ok())
                    .unwrap_or_else(|| engine_config.api_key.clone());
            }
            c
        })
        .filter(|c| c.is_usable())
        .map(Arc::new);

    let (session, engine, log_file) = match args.project.clone() {
        Some(name) => {
            let candidates = session_store::list_chat_projects()?;
            let found = candidates
                .into_iter()
                .filter(|s| s.display_name == name)
                .max_by_key(|s| s.updated_at);
            match found {
                Some(s) => open_project(
                    &s.session_id,
                    &s.display_name,
                    &engine_config,
                    &client,
                    &system_prompt,
                )?,
                None => {
                    let id = make_project_id(&name);
                    open_project(&id, &name, &engine_config, &client, &system_prompt)?
                }
            }
        }
        None => {
            let candidates = session_store::list_chat_projects()?;
            let latest = candidates.into_iter().max_by_key(|s| s.updated_at);
            match latest {
                Some(s) => open_project(
                    &s.session_id,
                    &s.display_name,
                    &engine_config,
                    &client,
                    &system_prompt,
                )?,
                None => {
                    let name = "default";
                    let id = make_project_id(name);
                    open_project(&id, name, &engine_config, &client, &system_prompt)?
                }
            }
        }
    };

    let mut current_session = session;
    let engine = engine;
    let log_file = log_file;

    session_store::add_or_update_log(&mut current_session, "chat", None)?;
    if let Err(e) = session_store::save_session(&current_session) {
        eprintln!("Ошибка сохранения сессии: {}", e);
    }

    let project_id = current_session.session_id.clone();

    let shared: SharedChat = Arc::new(AsyncMutex::new(ChatRuntime {
        session: current_session,
        engine,
        log_file,
        project_id: project_id.clone(),
        outgoing_tasks: Arc::new(AsyncMutex::new(std::collections::HashMap::new())),
        pending_calls: Arc::new(AsyncMutex::new(std::collections::HashMap::new())),
        tool_calls_log: Vec::new(),
        injected_skill: None,
    }));

    {
        let rt = shared.lock().await;
        println!("📁 Проект: {}", rt.session.display_name);
        println!("   session_id: {}", rt.project_id);
    }

    {
        let mut rt = shared.lock().await;
        write_log(&mut rt.log_file, "system", &system_prompt)?;
    }

    println!("🤖 Чат запущен");
    println!("   Модель: {}", engine_config.model);
    println!(
        "   Стриминг: {}",
        if engine_config.stream {
            "включен"
        } else {
            "выключен"
        }
    );
    if let Some(cost) = engine_config.max_cost_rub {
        println!("   Лимит бюджета: {:.2} RUB", cost);
    }
    if let Some(ref gf) = godfather_config {
        println!("   Godfather: модель {}", gf.model);
    } else {
        println!("   Godfather: отключён (нет api_key в [godfather])");
    }
    if let Some(allowed_tools) = &engine_config.allowed_tools {
        println!("   Разрешённые инструменты:");
        for tool in allowed_tools {
            println!("     - {} (режим: {})", tool.name, tool.mode);
        }
    }
    println!("   Команды:");
    println!("     /exit | /quit          — выход");
    println!("     /clear                 — очистить контекст");
    println!("     /fix                   — закрепить последнюю пару в префиксе");
    println!("     /system <промпт>       — сменить системный промпт");
    println!("     /metrics               — метрики сессии");
    println!("     /godfather             — сжать контекст (механически + семантически)");
    println!("     /unfreeze              — вернуть всё из архива godfather");
    println!("     /unfreeze last         — вернуть последний сжатый ход");
    println!("     /knowledge             — управление базой знаний");
    println!("     /rebuke <агент> <текст> — добавить замечание агенту или чату");
    println!("     /rebuke_edit <агент>   — редактировать замечания в редакторе");
    println!("     /project               — управление проектами");
    println!("       list                 — список проектов");
    println!("       new <name>           — создать проект");
    println!("       switch <session_id>  — переключиться на проект");
    println!("       last                 — переключиться на предыдущий");
    println!("       delete <id> [--force]— пометить удалённым");
    println!("       purge <id> [--force] — удалить файлы");
    println!();

    let agent_call_timeout_sec = args.agent_call_timeout_sec;
    let godfather_timeout_sec = args.godfather_timeout_sec;
    let max_iterations = toml_config.max_iterations.unwrap_or(5);

    let knowledge_auto_cfg = KnowledgeAutoConfig {
        top_n: toml_config.knowledge_auto_top_n.unwrap_or(7),
        top_k: toml_config.knowledge_auto_top_k.unwrap_or(3),
        threshold: toml_config.knowledge_auto_threshold.unwrap_or(0.70),
        per_phrase: toml_config.vector_db.as_ref().map(|v| v.top_k).unwrap_or(5),
    };

    let tools_search_cfg = ToolsSearchConfig {
        top_n: toml_config.tools_search_top_n.unwrap_or(7),
        phrase_min_words: toml_config.tools_search_phrase_min_words.unwrap_or(2),
        threshold: toml_config.tools_search_threshold.unwrap_or(0.45),
        slots: toml_config.tools_search_slots.unwrap_or(4),
        min_score: toml_config.tools_search_min_score.unwrap_or(0.20),
        per_query_top_k: toml_config.vector_db.as_ref().map(|v| v.top_k).unwrap_or(5),
    };

    let allowed_tools_chat: Vec<String> = toml_config
        .tools
        .as_ref()
        .map(|v| v.iter().map(|t| t.name.clone()).collect())
        .unwrap_or_default();

    let reply_ctx = Arc::new(ReplyContext {
        engine_config: Arc::new(engine_config.clone()),
        client: client.clone(),
        storage_http_config: Arc::new(storage_http_config.clone()),
        agent_call_timeout_sec,
        board_url: board_url.clone(),
        board_token: board_token.clone(),
        knowledge_auto: Arc::new(knowledge_auto_cfg),
        tools_search: Arc::new(tools_search_cfg),
        allowed_tools_chat: Arc::new(allowed_tools_chat),
    });

    {
        let shared_clone = shared.clone();
        let board_url_clone = board_url.clone();
        let board_token_clone = board_token.clone();
        let reply_ctx_clone = reply_ctx.clone();
        tokio::spawn(async move {
            chat_sse_loop(
                shared_clone,
                board_url_clone,
                board_token_clone,
                reply_ctx_clone,
            )
            .await;
        });
    }

    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut input = String::new();

    loop {
        let display_name = {
            let rt = shared.lock().await;
            rt.session.display_name.clone()
        };
        print!("Вы [{}] > ", display_name);
        let _ = io::stdout().flush();
        input.clear();

        if tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut input).await? == 0 {
            println!("\nДо свидания!");
            let mut rt = shared.lock().await;
            save_current(&mut *rt)?;
            break;
        }

        let user_input = input.trim();
        if user_input.is_empty() {
            continue;
        }

        match user_input {
            "/exit" | "/quit" => {
                let mut rt = shared.lock().await;
                save_current(&mut *rt)?;
                println!("До свидания!");
                break;
            }
            "/clear" => {
                let mut rt = shared.lock().await;
                rt.engine.clear_context();
                println!("🗑️ Контекст очищен.\n");
                continue;
            }
            "/fix" => {
                let mut rt = shared.lock().await;
                rt.engine.fix_last_turn();
                println!("📌 Последняя пара закреплена в префиксе.\n");
                continue;
            }
            "/system" if user_input.len() > 8 => {
                let new_prompt = user_input[8..].trim().to_string();
                let mut rt = shared.lock().await;
                rt.engine.set_system_prompt(new_prompt.clone());
                write_log(&mut rt.log_file, "system", &new_prompt)?;
                println!("✅ Системный промпт обновлён.\n");
                continue;
            }
            "/rebuke" => {
                println!("Использование: /rebuke <agent|chat> <TEXT>");
                continue;
            }
            "/rebuke_edit" => {
                println!("Использование: /rebuke_edit <agent|chat>");
                continue;
            }
            _ if user_input.starts_with("/rebuke ") => {
                let rest = &user_input[8..];
                let mut parts = rest.splitn(2, ' ');
                let agent = parts.next().unwrap_or("").trim();
                let text = parts.next().unwrap_or("").trim();
                if agent.is_empty() {
                    println!("Использование: /rebuke <agent|chat> <TEXT>");
                    continue;
                }
                if text.is_empty() {
                    println!("Нужен текст замечания.");
                    continue;
                }
                if agent != "chat" && !toml_config.agents.iter().any(|a| a.name == agent) {
                    println!("Агент '{}' не найден в конфиге.", agent);
                    continue;
                }
                match nano_harness::rebuke_manager::append_rebuke(
                    &client,
                    &storage_http_config.bind_addr,
                    &storage_http_config.auth_token,
                    agent,
                    text,
                )
                .await
                {
                    Ok(_) => {
                        if agent == "chat" {
                            println!("Rebuke добавлен чату. Перезапусти чат, чтобы он его увидел.");
                        } else {
                            println!(
                                "Rebuke добавлен агенту '{}'. Перезапусти агента, чтобы он его увидел.",
                                agent
                            );
                        }
                    }
                    Err(e) => println!("Ошибка: {:#}", e),
                }
                continue;
            }
            _ if user_input.starts_with("/rebuke_edit ") => {
                let agent = user_input[13..].trim();
                if agent.is_empty() {
                    println!("Использование: /rebuke_edit <agent|chat>");
                    continue;
                }
                if agent != "chat" && !toml_config.agents.iter().any(|a| a.name == agent) {
                    println!("Агент '{}' не найден в конфиге.", agent);
                    continue;
                }

                let existing = nano_harness::rebuke_manager::load_rebuke(
                    &client,
                    &storage_http_config.bind_addr,
                    &storage_http_config.auth_token,
                    agent,
                )
                .await
                .unwrap_or_default();

                let editor = editor_name(&toml_config);

                let edited = match edit_via_buffer("rebuke_edit", agent, &existing, &editor).await {
                    Ok(content) => content,
                    Err(e) => {
                        println!("Ошибка редактирования: {:#}", e);
                        continue;
                    }
                };

                match nano_harness::rebuke_manager::save_rebuke(
                    &client,
                    &storage_http_config.bind_addr,
                    &storage_http_config.auth_token,
                    agent,
                    &edited,
                )
                .await
                {
                    Ok(_) => {
                        if agent == "chat" {
                            println!("Rebuke для чата сохранён. Перезапусти чат.");
                        } else {
                            println!(
                                "Rebuke для '{}' сохранён. Перезапусти агента, чтобы он увидел изменения.",
                                agent
                            );
                        }
                    }
                    Err(e) => println!("Ошибка сохранения: {:#}", e),
                }
                continue;
            }
            "/knowledge" => {
                println!("Использование:");
                println!("  /knowledge list              — список знаний");
                println!("  /knowledge show <name>       — показать содержимое");
                println!("  /knowledge new <name>        — создать знание");
                println!("  /knowledge edit <name>       — редактировать знание");
                println!("  /knowledge delete <name>     — удалить знание");
                continue;
            }
            _ if user_input.starts_with("/knowledge ") => {
                let rest = &user_input[11..];
                let mut parts = rest.splitn(2, ' ');
                let sub = parts.next().unwrap_or("").trim();
                let arg = parts.next().unwrap_or("").trim();

                match sub {
                    "list" => {
                        match refresh_knowledge_catalog(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                        )
                        .await
                        {
                            Ok(entries) if entries.is_empty() => {
                                println!("База знаний пуста.");
                            }
                            Ok(entries) => {
                                println!("Знания ({}):", entries.len());
                                for e in entries {
                                    println!("  {} — {}", e.name, e.description);
                                }
                            }
                            Err(e) => println!("Ошибка: {:#}", e),
                        }
                    }
                    "show" => {
                        if arg.is_empty() {
                            println!("Использование: /knowledge show <name>");
                            continue;
                        }
                        match nano_harness::knowledge_manager::load_knowledge_raw(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                        )
                        .await
                        {
                            Ok(raw) => {
                                println!("--- {} (полный файл) ---", arg);
                                println!("{}", raw);
                                println!("--- конец ---");
                            }
                            Err(e) => println!("Ошибка: {:#}", e),
                        }
                    }
                    "new" => {
                        if arg.is_empty() {
                            println!("Использование: /knowledge new <name>");
                            continue;
                        }
                        if !nano_harness::knowledge_manager::validate_name(arg) {
                            println!(
                                "Недопустимое имя '{}'. Разрешены латиница, цифры, '_', '-'.",
                                arg
                            );
                            continue;
                        }

                        let existing = nano_harness::knowledge_manager::load_knowledge_raw(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                        )
                        .await;
                        if existing.is_ok() {
                            println!(
                                "Знание '{}' уже существует. Используй /knowledge edit.",
                                arg
                            );
                            continue;
                        }

                        let template = format!(
                            "---\nname: {}\ndescription: краткое описание для каталога\n---\n\nТекст знания.\n",
                            arg
                        );
                        let editor = editor_name(&toml_config);

                        let edited = match edit_via_buffer(
                            "knowledge_edit",
                            arg,
                            &template,
                            &editor,
                        )
                        .await
                        {
                            Ok(content) => content,
                            Err(e) => {
                                println!("Ошибка редактирования: {:#}", e);
                                continue;
                            }
                        };

                        match nano_harness::knowledge_manager::save_knowledge(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                            &edited,
                        )
                        .await
                        {
                            Ok(_) => {
                                println!("Знание '{}' сохранено и попало в каталог.", arg)
                            }
                            Err(e) => println!("Ошибка сохранения: {:#}", e),
                        }
                    }
                    "edit" => {
                        if arg.is_empty() {
                            println!("Использование: /knowledge edit <name>");
                            continue;
                        }
                        if !nano_harness::knowledge_manager::validate_name(arg) {
                            println!("Недопустимое имя '{}'.", arg);
                            continue;
                        }

                        let existing = match nano_harness::knowledge_manager::load_knowledge_raw(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                        )
                        .await
                        {
                            Ok(c) => c,
                            Err(_) => {
                                println!("Знание '{}' не найдено. Используй /knowledge new.", arg);
                                continue;
                            }
                        };

                        let editor = editor_name(&toml_config);

                        let edited = match edit_via_buffer(
                            "knowledge_edit",
                            arg,
                            &existing,
                            &editor,
                        )
                        .await
                        {
                            Ok(content) => content,
                            Err(e) => {
                                println!("Ошибка редактирования: {:#}", e);
                                continue;
                            }
                        };

                        match nano_harness::knowledge_manager::save_knowledge(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                            &edited,
                        )
                        .await
                        {
                            Ok(_) => {
                                println!("Знание '{}' сохранено и попало в каталог.", arg)
                            }
                            Err(e) => println!("Ошибка сохранения: {:#}", e),
                        }
                    }
                    "delete" => {
                        if arg.is_empty() {
                            println!("Использование: /knowledge delete <name>");
                            continue;
                        }
                        match nano_harness::knowledge_manager::delete_knowledge(
                            &client,
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            arg,
                        )
                        .await
                        {
                            Ok(_) => println!("Знание '{}' удалено.", arg),
                            Err(e) => println!("Ошибка: {:#}", e),
                        }
                    }
                    other => {
                        println!("Неизвестная подкоманда '{}'.", other);
                        println!("Доступно: list | show | new | edit | delete");
                    }
                }
                continue;
            }
            "/unfreeze" => {
                let mut rt = shared.lock().await;
                let count = rt.engine.unfreeze_all();
                if count == 0 {
                    println!("Архив пуст.");
                } else {
                    println!("🔄 Развёрнуто ходов из архива: {}", count);
                }
                save_current(&mut *rt)?;
                continue;
            }
            _ if user_input.starts_with("/unfreeze ") => {
                let rest = user_input[10..].trim();
                match rest {
                    "last" => {
                        let mut rt = shared.lock().await;
                        let count = rt.engine.unfreeze_last();
                        if count == 0 {
                            println!("Архив пуст.");
                        } else {
                            println!("🔄 Развёрнут последний сжатый ход ({} turns)", count);
                        }
                        save_current(&mut *rt)?;
                    }
                    _ => {
                        println!("Использование: /unfreeze | /unfreeze last");
                    }
                }
                continue;
            }
            "/godfather" => {
                let before_bytes = {
                    let rt = shared.lock().await;
                    rt.engine.total_context_bytes()
                };

                let min_reasonable_bytes = 20 * 1024;
                if before_bytes < min_reasonable_bytes && godfather_config.is_some() {
                    println!(
                        "⚠️ Контекст мал ({}). Семантическая компактизация может стоить дороже, чем выигрыш.",
                        format_bytes(before_bytes)
                    );
                    print!("   Продолжить? (y/n): ");
                    let _ = io::stdout().flush();
                    let mut answer = String::new();
                    tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut answer).await?;
                    if !answer.trim().eq_ignore_ascii_case("y") {
                        println!("Отменено.");
                        continue;
                    }
                }

                println!(
                    "👑 Godfather запущен. Контекст до: {}",
                    format_bytes(before_bytes)
                );

                let (mech_before, mech_after) = {
                    let mut rt = shared.lock().await;
                    rt.engine.compact_all_turns()
                };
                println!(
                    "   Механическая: {} → {} (освобождено {})",
                    format_bytes(mech_before),
                    format_bytes(mech_after),
                    format_bytes(mech_before.saturating_sub(mech_after))
                );

                let tail_count = {
                    let rt = shared.lock().await;
                    rt.engine.tail_turn_count()
                };

                if let Some(ref gf) = godfather_config {
                    if tail_count == 0 {
                        println!("   Семантическая: нечего сжимать (хвост пуст)");
                    } else {
                        let snapshot = {
                            let rt = shared.lock().await;
                            rt.engine.tail_turns_snapshot()
                        };

                        print!("   Семантическая: сжимаю {} ходов...", snapshot.len());
                        let _ = io::stdout().flush();

                        match nano_harness::godfather::compact_turns(
                            gf,
                            &snapshot,
                            godfather_timeout_sec,
                        )
                        .await
                        {
                            Ok(new_messages) => {
                                let msg_count = new_messages.len();
                                let mut rt = shared.lock().await;
                                rt.engine.apply_godfather_result(new_messages);
                                println!(
                                    " готово ({} ходов → {} сообщений, архив: {} ходов)",
                                    snapshot.len(),
                                    msg_count,
                                    rt.engine.archive_len()
                                );
                            }
                            Err(e) => {
                                println!(" откат ({:#})", e);
                            }
                        }
                    }
                } else {
                    println!("   Семантическая: godfather не настроен, пропускаю");
                }

                let after_bytes = {
                    let mut rt = shared.lock().await;
                    save_current(&mut *rt)?;
                    rt.engine.total_context_bytes()
                };
                println!(
                    "📊 Итог: {} → {} (освобождено {})\n",
                    format_bytes(before_bytes),
                    format_bytes(after_bytes),
                    format_bytes(before_bytes.saturating_sub(after_bytes))
                );
                continue;
            }
            "/metrics" => {
                let rt = shared.lock().await;
                println!("\n📊 Метрики сессии:");
                println!("   Запросов к API: {}", rt.engine.metrics.api_calls_count);
                println!(
                    "   Prompt токенов: {}",
                    rt.engine.metrics.total_prompt_tokens
                );
                println!(
                    "   Completion токенов: {}",
                    rt.engine.metrics.total_completion_tokens
                );
                println!(
                    "   Reasoning токенов: {}",
                    rt.engine.metrics.total_reasoning_tokens
                );
                println!(
                    "   💰 Потрачено: {:.6} RUB",
                    rt.engine.metrics.total_cost_rub
                );
                println!(
                    "   Префикс: {} пар, {} слов",
                    rt.engine.prefix_turn_count(),
                    rt.engine.prefix_word_count()
                );
                println!(
                    "   Хвост: {} пар, {} слов",
                    rt.engine.tail_turn_count(),
                    rt.engine.tail_word_count()
                );
                let bytes = rt.engine.total_context_bytes();
                println!(
                    "   Контекст: {} (~{} токенов)",
                    format_bytes(bytes),
                    bytes / 4
                );
                println!(
                    "   Слот скилла: {}",
                    if rt.engine.has_skill_context() {
                        "занят"
                    } else {
                        "—"
                    }
                );
                println!(
                    "   Слот знания: {}",
                    if rt.engine.has_knowledge_context() {
                        "занят"
                    } else {
                        "—"
                    }
                );
                println!();
                continue;
            }
            _ => {}
        }

        if user_input == "/project" || user_input.starts_with("/project ") {
            let parts: Vec<&str> = user_input.splitn(3, ' ').collect();
            match parts.get(1).copied() {
                None => {
                    let rt = shared.lock().await;
                    println!("📁 Текущий проект: {}", rt.session.display_name);
                    println!("   session_id: {}", rt.project_id);
                }
                Some("list") => {
                    let projects = session_store::list_chat_projects()?;
                    let current_id = {
                        let rt = shared.lock().await;
                        rt.project_id.clone()
                    };
                    if projects.is_empty() {
                        println!("Проектов нет.");
                    } else {
                        println!("Проекты:");
                        for p in projects {
                            let marker = if p.session_id == current_id { " *" } else { "" };
                            let dt = chrono::DateTime::<chrono::Local>::from(
                                std::time::UNIX_EPOCH
                                    + std::time::Duration::from_secs(p.updated_at),
                            )
                            .format("%Y-%m-%d %H:%M");
                            println!(
                                "  {} | {} | updated {} {}",
                                p.session_id, p.display_name, dt, marker
                            );
                        }
                    }
                }
                Some("new") => {
                    let name = parts.get(2).copied().unwrap_or("default");
                    let mut rt = shared.lock().await;
                    save_current(&mut *rt)?;
                    let id = make_project_id(name);
                    let (session, new_engine, new_log) =
                        open_project(&id, name, &engine_config, &client, &system_prompt)?;
                    rt.session = session;
                    rt.engine = new_engine;
                    rt.log_file = new_log;
                    rt.project_id = id.clone();
                    rt.tool_calls_log.clear();
                    rt.injected_skill = None;
                    session_store::add_or_update_log(&mut rt.session, "chat", None)?;
                    session_store::save_session(&rt.session)?;
                    println!("🆕 Создан проект {} ({})", name, id);
                }
                Some("switch") => {
                    if let Some(id) = parts.get(2) {
                        let mut rt = shared.lock().await;
                        save_current(&mut *rt)?;
                        match session_store::load_session_with_key(id, &engine_config.api_key, None)
                        {
                            Ok(s) if s.owner_agent.is_none() && !s.deleted => {
                                let (session, new_engine, new_log) = open_project(
                                    &s.session_id,
                                    &s.display_name,
                                    &engine_config,
                                    &client,
                                    &system_prompt,
                                )?;
                                rt.session = session;
                                rt.engine = new_engine;
                                rt.log_file = new_log;
                                rt.project_id = s.session_id.clone();
                                rt.tool_calls_log.clear();
                                rt.injected_skill = None;
                                println!(
                                    "🔄 Переключено на проект {} ({})",
                                    rt.session.display_name, rt.project_id
                                );
                            }
                            Ok(s) if s.deleted => {
                                println!(
                                    "Проект '{}' помечен удалённым. Сначала восстанови его.",
                                    id
                                );
                            }
                            Ok(_) => println!("Это не проект чата."),
                            Err(_) => println!("Проект '{}' не найден.", id),
                        }
                    } else {
                        println!("Использование: /project switch <session_id>");
                    }
                }
                Some("last") => {
                    let projects = session_store::list_chat_projects()?;
                    let current_id = {
                        let rt = shared.lock().await;
                        rt.project_id.clone()
                    };
                    let target = projects
                        .into_iter()
                        .filter(|s| s.session_id != current_id)
                        .max_by_key(|s| s.updated_at);
                    match target {
                        Some(s) => {
                            let mut rt = shared.lock().await;
                            save_current(&mut *rt)?;
                            let (session, new_engine, new_log) = open_project(
                                &s.session_id,
                                &s.display_name,
                                &engine_config,
                                &client,
                                &system_prompt,
                            )?;
                            rt.session = session;
                            rt.engine = new_engine;
                            rt.log_file = new_log;
                            rt.project_id = s.session_id.clone();
                            rt.tool_calls_log.clear();
                            rt.injected_skill = None;
                            println!(
                                "🔄 Переключено на {} ({})",
                                rt.session.display_name, rt.project_id
                            );
                        }
                        None => println!("Других проектов нет."),
                    }
                }
                Some("delete") => {
                    let id = match parts.get(2) {
                        Some(s) => s.split_whitespace().next().unwrap_or("").to_string(),
                        None => {
                            println!("Использование: /project delete <session_id> [--force]");
                            continue;
                        }
                    };
                    let force = user_input.ends_with("--force");

                    if !force {
                        match check_active_tasks(&client, &board_url, &board_token, &id).await {
                            Ok(tasks) if !tasks.is_empty() => {
                                println!(
                                    "⚠️ У проекта {} {} активных задач: {}",
                                    id,
                                    tasks.len(),
                                    tasks.join(", ")
                                );
                                println!("Дождись завершения или используй --force.");
                                continue;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                println!(
                                    "Не могу проверить активные задачи: {}. \
                                     Используй --force для удаления вслепую.",
                                    e
                                );
                                continue;
                            }
                        }
                    }

                    {
                        let current_id = {
                            let rt = shared.lock().await;
                            rt.project_id.clone()
                        };
                        if id == current_id {
                            let mut rt = shared.lock().await;
                            let projects = session_store::list_chat_projects()?;
                            let target = projects
                                .into_iter()
                                .filter(|s| s.session_id != id)
                                .max_by_key(|s| s.updated_at);
                            let new_name = target
                                .as_ref()
                                .map(|s| s.display_name.clone())
                                .unwrap_or_else(|| "default".to_string());
                            let new_id = target
                                .as_ref()
                                .map(|s| s.session_id.clone())
                                .unwrap_or_else(|| make_project_id(&new_name));
                            let (session, new_engine, new_log) = open_project(
                                &new_id,
                                &new_name,
                                &engine_config,
                                &client,
                                &system_prompt,
                            )?;
                            rt.session = session;
                            rt.engine = new_engine;
                            rt.log_file = new_log;
                            rt.project_id = new_id;
                            rt.tool_calls_log.clear();
                            rt.injected_skill = None;
                            session_store::add_or_update_log(&mut rt.session, "chat", None)?;
                            session_store::save_session(&rt.session)?;
                        }
                    }

                    match session_store::mark_project_deleted(&id) {
                        Ok(marked) => {
                            println!("🗑️ Помечено сессий: {}", marked.len());
                            println!("   Файлы останутся на диске до /project purge {}", id);
                        }
                        Err(e) => println!("Ошибка: {}", e),
                    }
                }
                Some("purge") => {
                    let id = match parts.get(2) {
                        Some(s) => s.split_whitespace().next().unwrap_or("").to_string(),
                        None => {
                            println!("Использование: /project purge <session_id> [--force]");
                            continue;
                        }
                    };
                    let force = user_input.ends_with("--force");

                    if !force {
                        match check_active_tasks(&client, &board_url, &board_token, &id).await {
                            Ok(tasks) if !tasks.is_empty() => {
                                println!(
                                    "⚠️ У проекта {} {} активных задач: {}",
                                    id,
                                    tasks.len(),
                                    tasks.join(", ")
                                );
                                println!("Дождись завершения или используй --force.");
                                continue;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                println!(
                                    "Не могу проверить активные задачи: {}. \
                                     Используй --force для удаления вслепую.",
                                    e
                                );
                                continue;
                            }
                        }
                    }

                    let current_id = {
                        let rt = shared.lock().await;
                        rt.project_id.clone()
                    };
                    if id == current_id && !force {
                        println!(
                            "Нельзя удалить текущий проект без --force. \
                             Сначала переключись на другой."
                        );
                        continue;
                    }

                    match session_store::purge_project(&id) {
                        Ok(deleted) => {
                            println!("🗑️ Удалено файлов: {}", deleted.len());
                            for p in &deleted {
                                println!("   {}", p.display());
                            }
                        }
                        Err(e) => println!("Ошибка удаления: {}", e),
                    }
                }
                Some(other) => {
                    println!("Неизвестная подкоманда '{}'.", other);
                    println!("Команды: /project [list|new <name>|switch <id>|last|delete <id> [--force]|purge <id> [--force]]");
                }
            }
            continue;
        }

        // ==================== Обычный ход ====================

        {
            let mut rt = shared.lock().await;
            rt.engine.add_message(Role::User, user_input.to_string());
            write_log(&mut rt.log_file, "user", user_input)?;
            rt.tool_calls_log.clear();
            rt.injected_skill = None;
        }

        // Авто-инжект знаний по текущему ходу.
        {
            let turn_text = {
                let rt = shared.lock().await;
                rt.engine.pending_turn_text()
            };
            if let Some(text) = turn_text {
                let storage_url = &reply_ctx.storage_http_config.bind_addr;
                let token = &reply_ctx.storage_http_config.auth_token;
                match build_knowledge_context(
                    &reply_ctx.client,
                    storage_url,
                    token,
                    &text,
                    &reply_ctx.knowledge_auto,
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
                        let mut rt = shared.lock().await;
                        rt.engine.set_knowledge_context("auto", ctx_str);
                        eprintln!("📚 knowledge auto-inject: injected [{}]", names.join(", "));
                    }
                    Ok(None) => {
                        let mut rt = shared.lock().await;
                        rt.engine.clear_knowledge_context();
                        eprintln!("📚 knowledge auto-inject: nothing relevant");
                    }
                    Err(e) => {
                        eprintln!("⚠️ knowledge auto-inject failed: {:#}", e);
                    }
                }
            }
        }

        // Авто-инжект скилла по текущему ходу.
        {
            let storage_url = reply_ctx.storage_http_config.bind_addr.clone();
            let token = reply_ctx.storage_http_config.auth_token.clone();
            let prompt_for_skill = user_input.to_string();
            let phrases_top_n = toml_config.skill_phrases_top_n.unwrap_or(15);
            let phrase_min_words = toml_config.skill_phrase_min_words.unwrap_or(2);
            let threshold = toml_config.skill_search_threshold.unwrap_or(0.45);
            let min_hits = toml_config.skill_min_hits.unwrap_or(3);
            let per_query_top_k = toml_config.vector_db.as_ref().map(|v| v.top_k).unwrap_or(5);

            let skill_result = tokio::task::spawn_blocking(
                move || -> anyhow::Result<Option<(nano_harness::skill_manager::SkillRecord, f32)>> {
                    let client = reqwest::blocking::Client::new();
                    nano_harness::skill_manager::search_best_skill(
                        &client,
                        &storage_url,
                        &token,
                        "chat",
                        &prompt_for_skill,
                        phrases_top_n,
                        phrase_min_words,
                        threshold,
                        min_hits,
                        per_query_top_k,
                    )
                },
            )
            .await;

            let record_opt = match skill_result {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    eprintln!("⚠️ skill search failed: {:#}", e);
                    None
                }
                Err(e) => {
                    eprintln!("⚠️ skill search join error: {:#}", e);
                    None
                }
            };

            if let Some((record, distance)) = record_opt {
                let storage_url = reply_ctx.storage_http_config.bind_addr.clone();
                let token = reply_ctx.storage_http_config.auth_token.clone();
                let skill_file = record.skill_file.clone();

                let content_result = tokio::task::spawn_blocking(move || {
                    let client = reqwest::blocking::Client::new();
                    nano_harness::skill_manager::load_skill(
                        &client,
                        &storage_url,
                        &token,
                        &skill_file,
                    )
                })
                .await;

                match content_result {
                    Ok(Ok(content)) => {
                        if let Some(calls) = nano_harness::skill_manager::parse_tool_calls(&content)
                        {
                            let injection = nano_harness::skill_manager::format_skill_for_injection(
                                &record, &calls,
                            );
                            let mut rt = shared.lock().await;
                            rt.engine.set_skill_context(injection);
                            rt.injected_skill = Some(record.clone());
                            eprintln!(
                                "🔍 skill auto-inject: {} (distance: {:.4})",
                                record.skill_file, distance
                            );
                        } else {
                            eprintln!("⚠️ skill parse failed for {}", record.skill_file);
                        }
                    }
                    Ok(Err(e)) => eprintln!("⚠️ skill load failed: {:#}", e),
                    Err(e) => eprintln!("⚠️ skill load join error: {:#}", e),
                }
            } else {
                let mut rt = shared.lock().await;
                rt.engine.clear_skill_context();
                rt.injected_skill = None;
            }
        }

        // ==================== Выбор тулзов под текущий ход ====================
        {
            let current_slots: Vec<String> = {
                let rt = shared.lock().await;
                rt.engine
                    .visible_tools()
                    .map(|s| s.to_vec())
                    .unwrap_or_default()
            };
            let allowed = reply_ctx.allowed_tools_chat.clone();
            let prompt = user_input.to_string();
            let storage_url = reply_ctx.storage_http_config.bind_addr.clone();
            let token = reply_ctx.storage_http_config.auth_token.clone();
            let cfg = reply_ctx.tools_search.clone();

            let selection = tokio::task::spawn_blocking(move || {
                let client = reqwest::blocking::Client::new();
                select_tools(
                    &client,
                    &storage_url,
                    &token,
                    &prompt,
                    &current_slots,
                    &allowed,
                    &cfg,
                )
            })
            .await;

            match selection {
                Ok(Ok(names)) if !names.is_empty() => {
                    eprintln!("🔧 visible_tools: {:?}", names);
                    let mut rt = shared.lock().await;
                    rt.engine.set_visible_tools(names);
                }
                Ok(Ok(_)) => {
                    eprintln!("🔧 visible_tools: пусто — слоты не трогаем");
                }
                Ok(Err(e)) => {
                    eprintln!("⚠️ select_tools failed: {:#}", e);
                }
                Err(e) => {
                    eprintln!("⚠️ select_tools join error: {:#}", e);
                }
            }
        }

        let allowed = engine_config.allowed_tools.clone();

        for _iter in 0..=max_iterations {
            print!("💭 ");
            let _ = io::stdout().flush();
            let response = {
                let mut rt = shared.lock().await;
                match rt.engine.send().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        println!("\n❌ Ошибка: {:#}", e);
                        write_log(&mut rt.log_file, "error", &format!("{:#}", e))?;
                        rt.engine.rollback_pending_turn();
                        break;
                    }
                }
            };

            println!();

            let has_content = !response.content.trim().is_empty();
            {
                let mut rt = shared.lock().await;
                let answer = if has_content {
                    response.content.clone()
                } else {
                    response.reasoning.clone()
                };
                if !answer.is_empty() {
                    write_log(&mut rt.log_file, "assistant", &answer)?;
                }
                if !response.reasoning.is_empty() && has_content {
                    write_log(&mut rt.log_file, "reasoning", &response.reasoning)?;
                }
            }

            if !response.reasoning.is_empty() && has_content {
                println!("\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n", response.reasoning);
            }

            let tool_calls = match response.tool_calls {
                Some(tc) if !tc.is_empty() => tc,
                _ => break,
            };

            {
                let mut rt = shared.lock().await;
                for tc in &tool_calls {
                    write_log(
                        &mut rt.log_file,
                        "tool_request",
                        &format!("{} ({})", tc.function.name, tc.function.arguments),
                    )?;
                }
            }

            let get_mode = |name: &str| -> Option<String> {
                allowed
                    .as_ref()
                    .and_then(|list| list.iter().find(|a| a.name == name).map(|a| a.mode.clone()))
            };

            let mut posted_async = false;

            for tc in &tool_calls {
                let name = &tc.function.name;
                let mode = get_mode(name).unwrap_or_else(|| "manual".to_string());

                if !allowed
                    .as_ref()
                    .map(|l| l.iter().any(|a| a.name == *name))
                    .unwrap_or(false)
                {
                    println!("⛔ Инструмент '{}' не разрешён, пропускаем.", name);
                    let mut rt = shared.lock().await;
                    write_log(
                        &mut rt.log_file,
                        "tool_denied",
                        &format!("{} ({})", name, tc.function.arguments),
                    )?;
                    continue;
                }

                let should_run = if mode == "auto" {
                    true
                } else {
                    println!("❓ Выполнить инструмент '{}'? (y/n)", name);
                    print!("> ");
                    let _ = io::stdout().flush();
                    let mut answer = String::new();
                    tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut answer).await?;
                    answer.trim().eq_ignore_ascii_case("y")
                };

                if !should_run {
                    println!("Пропущено.");
                    let mut rt = shared.lock().await;
                    write_log(
                        &mut rt.log_file,
                        "tool_skipped",
                        &format!("{} ({})", name, tc.function.arguments),
                    )?;
                    continue;
                }

                let tool_ctx = {
                    let rt = shared.lock().await;
                    let pid = rt.project_id.clone();
                    build_tool_context(
                        &reply_ctx,
                        pid.clone(),
                        Some(pid),
                        "chat".to_string(),
                        rt.pending_calls.clone(),
                        rt.outgoing_tasks.clone(),
                    )
                };

                let result =
                    nano_harness::tool_runtime::execute(name, &tc.function.arguments, &tool_ctx)
                        .await;

                let posted = result.contains("\"content\":\"posted:");

                {
                    let mut rt = shared.lock().await;
                    write_log(
                        &mut rt.log_file,
                        "tool_result",
                        &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                    )?;
                    rt.engine.add_tool_result(tc.id.clone(), result.clone());
                    rt.tool_calls_log.push(ToolCallLogEntry {
                        name: name.clone(),
                        arguments: tc.function.arguments.clone(),
                        result: result.clone(),
                    });
                }

                if posted {
                    println!("✅ Асинхронная задача опубликована. Ответ придёт позже.");
                    posted_async = true;
                    break;
                } else {
                    println!("✅ Автовыполнение: {}", result);
                }
            }

            if posted_async {
                let mut rt = shared.lock().await;
                save_current(&mut *rt)?;
                break;
            }
        }

        // Auto-save скилла и record_usage.
        {
            let (successful, injected, prompt_for_skill) = {
                let rt = shared.lock().await;
                let successful: Vec<ToolCallLogEntry> = rt
                    .tool_calls_log
                    .iter()
                    .filter(|t| {
                        !t.result.contains("\"ok\":false") && !t.result.starts_with("Ошибка")
                    })
                    .cloned()
                    .collect();
                (
                    successful,
                    rt.injected_skill.clone(),
                    user_input.to_string(),
                )
            };

            let min_calls = toml_config.skill_min_tool_calls.unwrap_or(2);

            if let Some(rec) = injected {
                let storage_url = reply_ctx.storage_http_config.bind_addr.clone();
                let token = reply_ctx.storage_http_config.auth_token.clone();
                let skill_file = rec.skill_file.clone();
                let success = !successful.is_empty();
                tokio::task::spawn_blocking(move || {
                    let client = reqwest::blocking::Client::new();
                    if let Err(e) = nano_harness::skill_manager::record_usage(
                        &client,
                        &storage_url,
                        &token,
                        &skill_file,
                        success,
                    ) {
                        eprintln!("⚠️ record_usage failed: {:#}", e);
                    }
                });
            }

            if successful.len() >= min_calls {
                let engine_config = reply_ctx.engine_config.clone();
                let prompt_for_metadata = prompt_for_skill.clone();

                let (skill_name, skill_description) =
                    match nano_harness::skill_manager::generate_skill_metadata(
                        &engine_config,
                        &prompt_for_metadata,
                    )
                    .await
                    {
                        Ok(meta) => meta,
                        Err(e) => {
                            eprintln!("⚠️ skill metadata failed: {:#}", e);
                            (String::new(), String::new())
                        }
                    };

                if !skill_name.is_empty() || !skill_description.is_empty() {
                    let tool_records: Vec<nano_harness::skill_manager::ToolCallRecord> = successful
                        .iter()
                        .filter_map(|t| {
                            let args: serde_json::Value =
                                serde_json::from_str(&t.arguments).unwrap_or(serde_json::json!({}));
                            Some(nano_harness::skill_manager::ToolCallRecord {
                                name: t.name.clone(),
                                arguments: args,
                            })
                        })
                        .collect();

                    let storage_url = reply_ctx.storage_http_config.bind_addr.clone();
                    let token = reply_ctx.storage_http_config.auth_token.clone();
                    let agent_name = "chat".to_string();
                    let prompt_for_save = prompt_for_skill.clone();
                    let skill_name_for_log = skill_name.clone();
                    let phrases_top_n = toml_config.skill_phrases_top_n.unwrap_or(15);
                    let phrase_min_words = toml_config.skill_phrase_min_words.unwrap_or(2);

                    let outcome = tokio::task::spawn_blocking(move || {
                        let client = reqwest::blocking::Client::new();
                        nano_harness::skill_manager::save_skill(
                            &client,
                            &storage_url,
                            &token,
                            &skill_name,
                            &agent_name,
                            &skill_description,
                            &prompt_for_save,
                            &tool_records,
                            min_calls,
                            phrases_top_n,
                            phrase_min_words,
                        )
                    })
                    .await;

                    match outcome {
                        Ok(Ok(nano_harness::skill_manager::SaveSkillOutcome::Saved)) => {
                            eprintln!("💾 Скилл сохранён: {}", skill_name_for_log);
                        }
                        Ok(Ok(nano_harness::skill_manager::SaveSkillOutcome::SkippedDuplicate)) => {
                            eprintln!("♻️ Скилл пропущен (дубликат): {}", skill_name_for_log);
                        }
                        Ok(Ok(
                            nano_harness::skill_manager::SaveSkillOutcome::SkippedTooFewCalls {
                                actual,
                                min,
                            },
                        )) => {
                            eprintln!(
                                "⏭️ Скилл не сохранён: {} ({} < {})",
                                skill_name_for_log, actual, min
                            );
                        }
                        Ok(Err(e)) => eprintln!("⚠️ skill save failed: {:#}", e),
                        Err(e) => eprintln!("⚠️ skill save join error: {:#}", e),
                    }
                }
            }
        }

        {
            let mut rt = shared.lock().await;
            save_current(&mut *rt)?;
            println!(
                "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB | Контекст: {}]\x1b[0m\n",
                rt.engine.metrics.total_prompt_tokens,
                rt.engine.metrics.total_completion_tokens,
                rt.engine.metrics.total_cost_rub,
                format_bytes(rt.engine.total_context_bytes())
            );
        }
    }

    Ok(())
}
