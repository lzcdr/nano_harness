// src/main.rs

use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::StreamExt;

use nano_harness::agent_core::{OutgoingTasks, PendingCalls};
use nano_harness::config::{build_engine_config, TomlConfig};
use nano_harness::engine::{ChatEngine, EngineConfig, Role};
use nano_harness::local_storage_http_api::LocalStorageServerConfig;
use nano_harness::message_board::BoardEvent;
use nano_harness::session_store::{self, ContextBlock, Session};
use nano_harness::tools::{execute_tool, BoardContext};

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
}

// ==================== Разделяемое состояние чата ====================

struct ChatRuntime {
    session: Session,
    engine: ChatEngine,
    log_file: std::fs::File,
    project_id: String,
    outgoing_tasks: OutgoingTasks,
    pending_calls: PendingCalls,
}

type SharedChat = Arc<AsyncMutex<ChatRuntime>>;

#[derive(Clone)]
struct ReplyContext {
    engine_config: Arc<EngineConfig>,
    client: Client,
    storage_http_config: Arc<LocalStorageServerConfig>,
    rhai_timeout_sec: u64,
    agent_call_timeout_sec: u64,
    board_url: String,
    board_token: String,
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
                    system_messages: vec![],
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
    //eprintln!("⏳ Подключаю SSE чата -> {}", url);
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", board_token))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    //eprintln!("🔌 SSE чата подключён");

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
        BoardEvent::TaskCreated { .. } => {
            // чат не принимает задачи
        }
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
    // 1. Синхронный call_agent — снимаем oneshot.
    {
        let rt = shared.lock().await;
        let mut pc = rt.pending_calls.lock().await;
        if let Some(entry) = pc.remove(&task_id) {
            let _ = entry.tx.send(result.clone());
            return;
        }
    }

    // 2. Асинхронный post_task — пишем в контекст текущего проекта.
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

    // Освобождаем lock перед продолжением диалога.
    drop(rt);

    continue_chat_turn(shared, reply_ctx).await;
}

async fn continue_chat_turn(shared: &SharedChat, ctx: &ReplyContext) {
    const MAX_ITER: usize = 5;
    for _ in 0..=MAX_ITER {
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
        {
            let mut rt = shared.lock().await;
            if !response.content.is_empty() {
                let _ = write_log(&mut rt.log_file, "assistant", &response.content);
            }
            if !response.reasoning.is_empty() {
                let _ = write_log(&mut rt.log_file, "reasoning", &response.reasoning);
            }
        }
        if !response.reasoning.is_empty() {
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

            let (board_ctx, posted_flag) = {
                let rt = shared.lock().await;
                let pf = Arc::new(AtomicBool::new(false));
                let bctx = BoardContext {
                    board_url: ctx.board_url.clone(),
                    board_token: ctx.board_token.clone(),
                    self_agent_name: "chat".to_string(),
                    self_session_id: Some(rt.project_id.clone()),
                    parent_chain: vec![],
                    pending_calls: rt.pending_calls.clone(),
                    agent_call_timeout_sec: ctx.agent_call_timeout_sec,
                    posted_flag: pf.clone(),
                    outgoing_tasks: rt.outgoing_tasks.clone(),
                };
                (bctx, pf)
            };

            let result = execute_tool(
                name,
                &tc.function.arguments,
                Some(&ctx.client),
                &ctx.storage_http_config.bind_addr,
                &ctx.storage_http_config.auth_token,
                ctx.rhai_timeout_sec,
                Some(board_ctx),
            )
            .await;

            let posted = posted_flag.load(Ordering::Relaxed);

            {
                let mut rt = shared.lock().await;
                let _ = write_log(
                    &mut rt.log_file,
                    "tool_result",
                    &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                );
                rt.engine.add_tool_result(tc.id.clone(), result.clone());
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
            "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB]\x1b[0m\n",
            rt.engine.metrics.total_prompt_tokens,
            rt.engine.metrics.total_completion_tokens,
            rt.engine.metrics.total_cost_rub
        );
    }
}

// ==================== main ====================

#[tokio::main]
async fn main() -> Result<()> {
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

    let system_prompt = args
        .system_prompt
        .clone()
        .or(toml_config.system_prompt.clone())
        .unwrap_or_else(|| "Ты полезный ассистент. Отвечай кратко и по делу.".to_string());

    let board_url = toml_config
        .message_board
        .as_ref()
        .map(|c| c.bind_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8090".to_string());
    let board_token = toml_config
        .message_board
        .as_ref()
        .map(|c| c.auth_token.clone())
        .unwrap_or_else(|| "board_token".to_string());
    let board_url = if board_url.starts_with("http") {
        board_url
    } else {
        format!("http://{}", board_url)
    };

    // Выбор проекта.
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
    }));

    // Заголовок.
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
    println!("     /project               — управление проектами");
    println!("       list                 — список проектов");
    println!("       new <name>           — создать проект");
    println!("       switch <session_id>  — переключиться на проект");
    println!("       last                 — переключиться на предыдущий");
    println!("       delete <id> [--force]— пометить удалённым");
    println!("       purge <id> [--force] — удалить файлы");
    println!();

    let storage_http_config = toml_config
        .local_storage_http_server
        .clone()
        .unwrap_or_else(|| LocalStorageServerConfig {
            bind_addr: "127.0.0.1:8080".to_string(),
            storage_name: "default_storage".to_string(),
            auth_token: String::new(),
        });

    let rhai_timeout_sec = toml_config.rhai_timeout_sec.unwrap_or(30);
    let agent_call_timeout_sec = args.agent_call_timeout_sec;

    let reply_ctx = Arc::new(ReplyContext {
        engine_config: Arc::new(engine_config.clone()),
        client: client.clone(),
        storage_http_config: Arc::new(storage_http_config.clone()),
        rhai_timeout_sec,
        agent_call_timeout_sec,
        board_url: board_url.clone(),
        board_token: board_token.clone(),
    });

    // Поднимаем SSE чата.
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
        // Печатаем приглашение с текущим проектом.
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
                println!();
                continue;
            }
            _ => {}
        }

        // Команды /project ...
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

        // Добавляем user-message и логируем.
        {
            let mut rt = shared.lock().await;
            rt.engine.add_message(Role::User, user_input.to_string());
            write_log(&mut rt.log_file, "user", user_input)?;
        }

        // Первый send.
        let response = {
            let mut rt = shared.lock().await;
            match rt.engine.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    println!("\n❌ Ошибка: {:#}", e);
                    write_log(&mut rt.log_file, "error", &format!("{:#}", e))?;
                    rt.engine.rollback_pending_turn();
                    continue;
                }
            }
        };

        println!();

        {
            let mut rt = shared.lock().await;
            if !response.content.is_empty() {
                write_log(&mut rt.log_file, "assistant", &response.content)?;
            }
            if !response.reasoning.is_empty() {
                write_log(&mut rt.log_file, "reasoning", &response.reasoning)?;
            }
        }

        if !response.reasoning.is_empty() {
            println!("\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n", response.reasoning);
        }

        if let Some(tool_calls) = response.tool_calls {
            println!("\n⚠️ Модель запросила инструменты:");
            {
                let mut rt = shared.lock().await;
                for tc in &tool_calls {
                    println!("   - {} ({})", tc.function.name, tc.function.arguments);
                    write_log(
                        &mut rt.log_file,
                        "tool_request",
                        &format!("{} ({})", tc.function.name, tc.function.arguments),
                    )?;
                }
            }

            let allowed = &engine_config.allowed_tools;
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

                // Собираем BoardContext из текущего чата.
                let (board_ctx, posted_flag_check) = {
                    let rt = shared.lock().await;
                    let posted_flag = Arc::new(AtomicBool::new(false));
                    let bctx = BoardContext {
                        board_url: board_url.clone(),
                        board_token: board_token.clone(),
                        self_agent_name: "chat".to_string(),
                        self_session_id: Some(rt.project_id.clone()),
                        parent_chain: vec![],
                        pending_calls: rt.pending_calls.clone(),
                        agent_call_timeout_sec,
                        posted_flag: posted_flag.clone(),
                        outgoing_tasks: rt.outgoing_tasks.clone(),
                    };
                    (bctx, posted_flag)
                };

                let result = execute_tool(
                    name,
                    &tc.function.arguments,
                    Some(&client),
                    &storage_http_config.bind_addr,
                    &storage_http_config.auth_token,
                    rhai_timeout_sec,
                    Some(board_ctx),
                )
                .await;

                let posted = posted_flag_check.load(Ordering::Relaxed);

                {
                    let mut rt = shared.lock().await;
                    write_log(
                        &mut rt.log_file,
                        "tool_result",
                        &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                    )?;
                    rt.engine.add_tool_result(tc.id.clone(), result.clone());
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
                // Сохраняем сессию и продолжаем цикл. Ответ придёт асинхронно через SSE,
                // и continue_chat_turn доиграет диалог после получения ответа.
                let mut rt = shared.lock().await;
                save_current(&mut *rt)?;
                continue;
            }

            println!("\nЗапрашиваю финальный ответ...");
            let final_response = {
                let mut rt = shared.lock().await;
                match rt.engine.send().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        println!("\n❌ Ошибка: {:#}", e);
                        write_log(&mut rt.log_file, "error", &format!("{:#}", e))?;
                        rt.engine.rollback_pending_turn();
                        continue;
                    }
                }
            };
            println!();
            {
                let mut rt = shared.lock().await;
                if !final_response.content.is_empty() {
                    write_log(&mut rt.log_file, "assistant", &final_response.content)?;
                }
                if !final_response.reasoning.is_empty() {
                    write_log(&mut rt.log_file, "reasoning", &final_response.reasoning)?;
                }
            }
            if !final_response.reasoning.is_empty() {
                println!(
                    "\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n",
                    final_response.reasoning
                );
            }
        }

        // Сохраняем сессию после хода.
        {
            let mut rt = shared.lock().await;
            save_current(&mut *rt)?;
            println!(
                "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB]\x1b[0m\n",
                rt.engine.metrics.total_prompt_tokens,
                rt.engine.metrics.total_completion_tokens,
                rt.engine.metrics.total_cost_rub
            );
        }
    }

    Ok(())
}
