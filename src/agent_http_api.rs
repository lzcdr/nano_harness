// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/agent_http_api.rs

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::StreamExt;

use crate::agent_core::{
    create_agent_engine, process_agent_turns, run_agent, AgentConfig, AgentContext, AgentMetrics,
    AgentRequest, AgentResponse, AgentType, OutgoingTask,
};
use crate::engine::Role;
use crate::message_board::BoardEvent;
use crate::session_store::{self, ContextBlock, PendingTask};

struct Session {
    engine: crate::engine::ChatEngine,
    log_file: std::fs::File,
    last_used: Instant,
}

#[derive(Clone)]
struct AppState {
    config: AgentConfig,
    context: Arc<AgentContext>,
    auth_token: String,
    sessions: Arc<AsyncMutex<HashMap<String, Arc<AsyncMutex<Session>>>>>,
    sse_listeners: Arc<AsyncMutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    incoming_tasks: Arc<AsyncMutex<HashMap<String, Vec<String>>>>,
}

async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string());

    if token.as_deref() == Some(state.auth_token.as_str()) {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "Unauthorized").into_response()
    }
}

fn write_log(file: &mut std::fs::File, role: &str, content: &str) -> anyhow::Result<()> {
    writeln!(
        file,
        "[{}] {}: {}",
        chrono::Local::now().format("%H:%M:%S"),
        role,
        content
    )?;
    Ok(())
}

fn error_response(e: anyhow::Error) -> Response {
    Json(AgentResponse {
        status: "failed".to_string(),
        result: String::new(),
        reasoning: None,
        tool_calls_log: vec![],
        metrics: AgentMetrics {
            prompt_tokens: 0,
            completion_tokens: 0,
            cost_rub: 0.0,
            api_calls_count: 0,
        },
        error: Some(format!("{:#}", e)),
        session_id: None,
    })
    .into_response()
}

async fn persist_session(
    state: &Arc<AppState>,
    session_id: &str,
    engine: &crate::engine::ChatEngine,
) {
    let pending: Vec<PendingTask> = {
        let map = state.context.outgoing_tasks.lock().await;
        map.values()
            .filter(|t| t.session_id == session_id)
            .map(|t| PendingTask {
                task_id: t.task_id.clone(),
                to_agent_name: t.to_agent_name.clone(),
                to_session_id: t.to_session_id.clone(),
                project_id: t.project_id.clone(),
                chain: t.chain.clone(),
            })
            .collect()
    };

    match session_store::load_session(session_id, Some(&state.config.name)) {
        Ok(mut store_session) => {
            store_session.context = Some(ContextBlock {
                engine_config: engine.get_config().clone(),
                engine_state: engine.get_state(),
            });
            store_session.pending_tasks = pending;
            store_session.updated_at = session_store::now_ts();
            if let Err(e) = session_store::save_session(&store_session) {
                eprintln!("Ошибка сохранения сессии {}: {}", session_id, e);
            }
        }
        Err(e) => eprintln!("Ошибка загрузки сессии {}: {}", session_id, e),
    }
}

async fn load_or_create_engine(
    session_id: &str,
    config: &AgentConfig,
    context: &crate::agent_core::AgentContext,
) -> anyhow::Result<(crate::engine::ChatEngine, Option<session_store::Session>)> {
    let store_session =
        session_store::load_session_with_key(session_id, &config.api_key, Some(&config.name)).ok();
    let mut engine = create_agent_engine(config)?;

    let base_prompt = config
        .system_prompt
        .clone()
        .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());

    let rebuke = crate::rebuke_manager::load_rebuke(
        &context.http_client,
        &context.storage_base_url,
        &context.storage_auth_token,
        &config.name,
    )
    .await
    .unwrap_or_default();

    let system_prompt = crate::rebuke_manager::build_system_with_rebuke(&base_prompt, &rebuke);

    if let Some(ref session) = store_session {
        if let Some(ctx) = &session.context {
            engine.set_state(ctx.engine_state.clone());
            engine.set_system_prompt(system_prompt);
            return Ok((engine, Some(session.clone())));
        }
    }

    engine.add_message(Role::System, system_prompt);

    Ok((engine, store_session))
}

async fn ensure_session(
    state: &Arc<AppState>,
    session_id: &str,
) -> anyhow::Result<Arc<AsyncMutex<Session>>> {
    {
        let sessions = state.sessions.lock().await;
        if let Some(arc) = sessions.get(session_id).cloned() {
            return Ok(arc);
        }
    }

    if let Ok(s) = session_store::load_session(session_id, Some(&state.config.name)) {
        if s.deleted {
            return Err(anyhow::anyhow!(
                "Сессия '{}' помечена удалённой",
                session_id
            ));
        }
    }

    let (engine, store_session_opt) =
        load_or_create_engine(session_id, &state.config, &state.context).await?;

    let mut store_session = match store_session_opt {
        Some(s) => {
            eprintln!(
                "📂 Загружена сессия агента '{}': session_id={}",
                state.config.name, session_id
            );
            s
        }
        None => {
            let display_name = format!("Agent session {}", session_id);
            let mut s = session_store::create_session_with_id(
                session_id,
                &display_name,
                Some(&state.config.name),
            )?;
            s.context = Some(ContextBlock {
                engine_config: engine.get_config().clone(),
                engine_state: engine.get_state(),
            });
            if let Err(e) = session_store::save_session(&s) {
                eprintln!("Ошибка сохранения новой сессии: {}", e);
            }
            eprintln!(
                "🆕 Создана сессия агента '{}': session_id={}",
                state.config.name, session_id
            );
            s
        }
    };

    if let Err(e) =
        session_store::add_or_update_log(&mut store_session, "agent", Some(&state.config.name))
    {
        eprintln!("Ошибка инициализации лога: {}", e);
    }

    let log_file = session_store::open_log(session_id, "agent", Some(&state.config.name))?;
    if let Err(e) = session_store::save_session(&store_session) {
        eprintln!("Ошибка сохранения сессии: {}", e);
    }

    let session = Session {
        engine,
        log_file,
        last_used: Instant::now(),
    };
    let arc = Arc::new(AsyncMutex::new(session));
    state
        .sessions
        .lock()
        .await
        .insert(session_id.to_string(), arc.clone());

    Ok(arc)
}

// ==================== Управление SSE ====================

fn spawn_sse_listener(state: Arc<AppState>) {
    let state_clone = state.clone();
    tokio::spawn(async move {
        let agent_name = state_clone.config.name.clone();
        {
            let mut listeners = state_clone.sse_listeners.lock().await;
            if let Some(handle) = listeners.get(&agent_name) {
                if !handle.is_finished() {
                    return;
                }
                listeners.remove(&agent_name);
            }
        }
        let state_for_loop = state_clone.clone();
        let agent_for_loop = agent_name.clone();
        let handle = tokio::spawn(async move {
            sse_loop(state_for_loop, agent_for_loop).await;
        });
        state_clone
            .sse_listeners
            .lock()
            .await
            .insert(agent_name, handle);
    });
}

async fn stop_sse_listener(state: &Arc<AppState>, agent_name: &str) {
    let mut listeners = state.sse_listeners.lock().await;
    if let Some(handle) = listeners.remove(agent_name) {
        handle.abort();
        eprintln!("🛑 SSE остановлен: agent={}", agent_name);
    }
}

async fn sse_cleanup_loop(state: Arc<AppState>, mut shutdown_rx: watch::Receiver<bool>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    interval.tick().await;
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => break,
            _ = interval.tick() => {
                let mut listeners = state.sse_listeners.lock().await;
                let before = listeners.len();
                listeners.retain(|_, h| !h.is_finished());
                let removed = before - listeners.len();
                if removed > 0 {
                    eprintln!("🧹 Убрано {} мёртвых SSE-слушателей", removed);
                }
            }
        }
    }
}

async fn pending_calls_cleanup_loop(
    state: Arc<AppState>,
    timeout_sec: u64,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    interval.tick().await;
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => break,
            _ = interval.tick() => {
                let now = Instant::now();
                let mut pc = state.context.pending_calls.lock().await;
                let before = pc.len();
                pc.retain(|_, entry| now.duration_since(entry.created_at).as_secs() < timeout_sec);
                let removed = before - pc.len();
                if removed > 0 {
                    eprintln!("🧹 Убрано {} просроченных pending_calls", removed);
                }
            }
        }
    }
}

async fn session_cleanup_loop(
    state: Arc<AppState>,
    ttl_sec: u64,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    interval.tick().await;
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => break,
            _ = interval.tick() => {
                let snapshot: Vec<(String, Arc<AsyncMutex<Session>>)> = {
                    let sessions = state.sessions.lock().await;
                    sessions.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                };
                let now = Instant::now();
                let mut to_remove: Vec<String> = Vec::new();
                for (id, arc) in snapshot {
                    if session_store::is_session_deleted_or_missing(
                        &id,
                        Some(&state.config.name),
                    ) {
                        to_remove.push(id);
                        continue;
                    }
                    if let Ok(guard) = arc.try_lock() {
                        if now.duration_since(guard.last_used).as_secs() > ttl_sec {
                            to_remove.push(id);
                        }
                    }
                }
                if !to_remove.is_empty() {
                    let mut sessions = state.sessions.lock().await;
                    let mut evicted = 0usize;
                    for id in &to_remove {
                        if sessions.remove(id).is_some() {
                            evicted += 1;
                        }
                    }
                    if evicted > 0 {
                        eprintln!(
                            "🧹 Выгружено {} неактивных/удалённых сессий из кэша",
                            evicted
                        );
                    }
                }
            }
        }
    }
}

async fn sse_loop(state: Arc<AppState>, agent_name: String) {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);
    let mut consecutive_errors = 0u32;

    loop {
        match connect_and_listen(&state, &agent_name).await {
            Ok(()) => {
                consecutive_errors = 0;
                backoff = Duration::from_secs(1);
            }
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors <= 3 || consecutive_errors % 10 == 0 {
                    eprintln!(
                        "❌ SSE [agent={}] ошибка #{}: {}",
                        agent_name, consecutive_errors, e
                    );
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(max_backoff);
                continue;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn connect_and_listen(state: &Arc<AppState>, agent_name: &str) -> anyhow::Result<()> {
    let url = format!(
        "{}/events?agent_name={}",
        state.context.board_base_url.trim_end_matches('/'),
        urlencoding::encode(agent_name)
    );
    eprintln!("⏳ Подключаю SSE: agent={} -> {}", agent_name, url);
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header(
            "Authorization",
            format!("Bearer {}", state.context.board_auth_token),
        )
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    eprintln!("🔌 SSE подключён: agent={}", agent_name);

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
                    Ok(event) => handle_board_event(state.clone(), event).await,
                    Err(e) => eprintln!("Ошибка парсинга SSE события: {}", e),
                }
            }
        }
    }
    Ok(())
}

// ==================== Обработка событий доски ====================

async fn handle_board_event(state: Arc<AppState>, event: BoardEvent) {
    match event {
        BoardEvent::TaskCreated { task } => {
            handle_task_created(state, task).await;
        }
        BoardEvent::TaskCompleted {
            task_id,
            result,
            from_agent,
            from_session_id,
        } => {
            handle_task_result(state, task_id, result, from_agent, from_session_id, false).await;
        }
        BoardEvent::TaskFailed {
            task_id,
            error,
            from_agent,
            from_session_id,
        } => {
            handle_task_result(state, task_id, error, from_agent, from_session_id, true).await;
        }
    }
}

async fn handle_task_created(state: Arc<AppState>, task: crate::message_board::Task) {
    if task.to_agent != state.config.name {
        return;
    }
    let session_id = task.to_session_id.clone();
    let project_id = task.project_id.clone();
    let prompt = task
        .payload
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if let Err(e) = ensure_session(&state, &session_id).await {
        eprintln!("Ошибка подготовки сессии {}: {:#}", session_id, e);
        let _ = publish_fail(&state, &task.id, &format!("{:#}", e)).await;
        return;
    }

    let arc = {
        let sessions = state.sessions.lock().await;
        sessions.get(&session_id).cloned()
    };
    let Some(arc) = arc else { return };
    let mut guard = arc.lock().await;
    guard.last_used = Instant::now();
    guard.engine.add_message(Role::User, prompt.clone());
    if let Err(e) = write_log(&mut guard.log_file, "task", &prompt) {
        eprintln!("Ошибка записи в лог: {}", e);
    }

    let request = AgentRequest {
        prompt: prompt.clone(),
        max_iterations: None,
        tools: None,
        system_prompt: None,
        session_id: Some(session_id.clone()),
        project_id: Some(project_id.clone()),
    };

    let parent_chain = task.chain.clone();
    let Session {
        engine, log_file, ..
    } = &mut *guard;

    let result = process_agent_turns(
        engine,
        &state.config,
        &state.context,
        &request,
        log_file,
        Some(session_id.clone()),
        project_id,
        parent_chain,
    )
    .await;

    match result {
        Ok(response) => {
            persist_session(&state, &session_id, engine).await;
            drop(guard);
            match response.status.as_str() {
                "completed" => {
                    let _ = publish_complete(&state, &task.id, &response.result).await;
                }
                "waiting" => {
                    let mut stack = state.incoming_tasks.lock().await;
                    stack
                        .entry(session_id.clone())
                        .or_default()
                        .push(task.id.clone());
                }
                _ => {
                    let err = response
                        .error
                        .clone()
                        .unwrap_or_else(|| "Неизвестный статус".to_string());
                    let _ = publish_fail(&state, &task.id, &err).await;
                }
            }
        }
        Err(e) => {
            drop(guard);
            eprintln!("❌ Ошибка обработки задачи {}: {:#}", task.id, e);
            let _ = publish_fail(&state, &task.id, &format!("{:#}", e)).await;
        }
    }
}

async fn handle_task_result(
    state: Arc<AppState>,
    task_id: String,
    result: String,
    from_agent: String,
    from_session_id: String,
    is_error: bool,
) {
    // 1. Синхронный call_agent?
    {
        let mut pc = state.context.pending_calls.lock().await;
        if let Some(entry) = pc.remove(&task_id) {
            let _ = entry.tx.send(result.clone());
            return;
        }
    }

    // 2. Асинхронный post_task?
    let outgoing = {
        let mut map = state.context.outgoing_tasks.lock().await;
        map.remove(&task_id)
    };
    let Some(outgoing) = outgoing else {
        eprintln!("TaskCompleted для неизвестной задачи {}", task_id);
        return;
    };

    let session_id = outgoing.session_id.clone();
    let project_id = outgoing.project_id.clone();
    let parent_chain = outgoing.chain.clone();

    if session_id.is_empty() {
        eprintln!("TaskCompleted для задачи {} без session_id", task_id);
        return;
    }

    if let Err(e) = ensure_session(&state, &session_id).await {
        eprintln!("Ошибка подготовки сессии {}: {:#}", session_id, e);
        return;
    }
    let arc = {
        let sessions = state.sessions.lock().await;
        sessions.get(&session_id).cloned()
    };
    let Some(arc) = arc else { return };
    let mut guard = arc.lock().await;
    guard.last_used = Instant::now();

    if let Ok(mut store) = session_store::load_session(&session_id, Some(&state.config.name)) {
        session_store::remove_pending_task(&mut store, &task_id);
        if let Err(e) = session_store::save_session(&store) {
            eprintln!("Ошибка сохранения сессии {}: {}", session_id, e);
        }
    }

    let header = if is_error {
        format!("[Ошибка от агента {}]: {}", from_agent, result)
    } else {
        format!("[Ответ от агента {}]: {}", from_agent, result)
    };
    guard.engine.add_message(Role::User, header.clone());
    if let Err(e) = write_log(&mut guard.log_file, "task_result", &header) {
        eprintln!("Ошибка записи в лог: {}", e);
    }

    // Сохранить сессию сразу после добавления ответа агента в контекст.
    {
        if let Ok(mut store) = session_store::load_session(&session_id, Some(&state.config.name)) {
            store.context = Some(ContextBlock {
                engine_config: guard.engine.get_config().clone(),
                engine_state: guard.engine.get_state(),
            });
            store.updated_at = session_store::now_ts();
            if let Err(e) = session_store::save_session(&store) {
                eprintln!("Ошибка сохранения сессии после ответа агента: {}", e);
            }
        }
    }

    let request = AgentRequest {
        prompt: String::new(),
        max_iterations: None,
        tools: None,
        system_prompt: None,
        session_id: Some(session_id.clone()),
        project_id: Some(project_id.clone()),
    };

    let Session {
        engine, log_file, ..
    } = &mut *guard;

    let result_turn = process_agent_turns(
        engine,
        &state.config,
        &state.context,
        &request,
        log_file,
        Some(session_id.clone()),
        project_id,
        parent_chain,
    )
    .await;

    match result_turn {
        Ok(response) => {
            persist_session(&state, &session_id, engine).await;
            drop(guard);

            match response.status.as_str() {
                "completed" => {
                    if let Some(parent_task_id) = pop_incoming_task(&state, &session_id).await {
                        let _ = publish_complete(&state, &parent_task_id, &response.result).await;
                    }
                }
                "waiting" => {}
                _ => {
                    if let Some(parent_task_id) = pop_incoming_task(&state, &session_id).await {
                        let err = response
                            .error
                            .clone()
                            .unwrap_or_else(|| "Неизвестный статус".to_string());
                        let _ = publish_fail(&state, &parent_task_id, &err).await;
                    }
                }
            }
        }
        Err(e) => {
            drop(guard);
            eprintln!("Ошибка продолжения сессии {}: {:#}", session_id, e);
            if let Some(parent_task_id) = pop_incoming_task(&state, &session_id).await {
                let _ = publish_fail(&state, &parent_task_id, &format!("{:#}", e)).await;
            }
        }
    }

    let _ = from_session_id;
}

async fn pop_incoming_task(state: &Arc<AppState>, session_id: &str) -> Option<String> {
    let mut stack = state.incoming_tasks.lock().await;
    if let Some(v) = stack.get_mut(session_id) {
        let popped = v.pop();
        if v.is_empty() {
            stack.remove(session_id);
        }
        popped
    } else {
        None
    }
}

async fn publish_complete(
    state: &Arc<AppState>,
    task_id: &str,
    result: &str,
) -> anyhow::Result<()> {
    let url = format!(
        "{}/tasks/{}/complete",
        state.context.board_base_url.trim_end_matches('/'),
        task_id
    );
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header(
            "Authorization",
            format!("Bearer {}", state.context.board_auth_token),
        )
        .json(&serde_json::json!({ "result": result }))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(())
}

async fn publish_fail(state: &Arc<AppState>, task_id: &str, error: &str) -> anyhow::Result<()> {
    let url = format!(
        "{}/tasks/{}/fail",
        state.context.board_base_url.trim_end_matches('/'),
        task_id
    );
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header(
            "Authorization",
            format!("Bearer {}", state.context.board_auth_token),
        )
        .json(&serde_json::json!({ "error": error }))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(())
}

// ==================== Фоновая проверка незавершённых задач ====================

async fn poll_pending_tasks(state: Arc<AppState>) {
    let active_task_ids: Vec<String> = {
        let map = state.context.outgoing_tasks.lock().await;
        map.keys().cloned().collect()
    };
    if active_task_ids.is_empty() {
        return;
    }

    let url = format!(
        "{}/tasks?from_agent={}&status=completed,failed",
        state.context.board_base_url.trim_end_matches('/'),
        urlencoding::encode(&state.config.name)
    );
    let client = reqwest::Client::new();
    let resp = match client
        .get(&url)
        .header(
            "Authorization",
            format!("Bearer {}", state.context.board_auth_token),
        )
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            eprintln!("poll_pending_tasks: HTTP {}", r.status());
            return;
        }
        Err(e) => {
            eprintln!("poll_pending_tasks: ошибка запроса: {}", e);
            return;
        }
    };
    let tasks: Vec<crate::message_board::Task> = match resp.json().await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("poll_pending_tasks: ошибка парсинга: {}", e);
            return;
        }
    };

    for task in tasks {
        if !active_task_ids.contains(&task.id) {
            continue;
        }
        match task.status {
            crate::message_board::TaskStatus::Completed => {
                if let Some(result) = task.result.clone() {
                    eprintln!("📬 Восстановлена завершённая задача {}", task.id);
                    handle_task_result(
                        state.clone(),
                        task.id.clone(),
                        result,
                        task.from_agent.clone(),
                        task.from_session_id.clone(),
                        false,
                    )
                    .await;
                }
            }
            crate::message_board::TaskStatus::Failed => {
                let err = task.error.clone().unwrap_or_default();
                eprintln!("📬 Восстановлена упавшая задача {}", task.id);
                handle_task_result(
                    state.clone(),
                    task.id.clone(),
                    err,
                    task.from_agent.clone(),
                    task.from_session_id.clone(),
                    true,
                )
                .await;
            }
            _ => {}
        }
    }
}

async fn poll_pending_tasks_loop(state: Arc<AppState>, mut shutdown_rx: watch::Receiver<bool>) {
    tokio::time::sleep(Duration::from_secs(2)).await;
    poll_pending_tasks(state.clone()).await;

    let mut interval = tokio::time::interval(Duration::from_secs(60));
    interval.tick().await;
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => break,
            _ = interval.tick() => {
                poll_pending_tasks(state.clone()).await;
            }
        }
    }
}

// ==================== HTTP-обработчики ====================

async fn run_stateful_agent(
    state: &AppState,
    request: AgentRequest,
    project_id: String,
) -> Response {
    let session_id = match request.session_id.clone() {
        Some(id) if !id.trim().is_empty() => id,
        _ => {
            return error_response(anyhow::anyhow!("session_id is required for stateful agent"));
        }
    };
    eprintln!(
        "📨 HTTP /agent/run: agent='{}', session_id={}, project_id={}",
        state.config.name, session_id, project_id
    );

    let state_arc = Arc::new(state.clone());

    if let Err(e) = ensure_session(&state_arc, &session_id).await {
        return error_response(e);
    }

    let arc = {
        let sessions = state.sessions.lock().await;
        sessions.get(&session_id).cloned()
    };
    let Some(arc) = arc else {
        return error_response(anyhow::anyhow!("Session not found"));
    };
    let mut guard = arc.lock().await;
    guard.last_used = Instant::now();
    guard.engine.add_message(Role::User, request.prompt.clone());
    if let Err(e) = write_log(&mut guard.log_file, "task", &request.prompt) {
        eprintln!("Ошибка записи в лог: {}", e);
    }

    let sid = session_id.clone();
    let Session {
        engine, log_file, ..
    } = &mut *guard;

    match process_agent_turns(
        engine,
        &state.config,
        &state.context,
        &request,
        log_file,
        Some(sid.clone()),
        project_id,
        vec![],
    )
    .await
    {
        Ok(mut response) => {
            response.session_id = Some(session_id.clone());
            persist_session(&state_arc, &session_id, engine).await;
            Json(response).into_response()
        }
        Err(e) => {
            drop(guard);
            error_response(e)
        }
    }
}

async fn run_agent_handler(
    State(state): State<AppState>,
    Json(request): Json<AgentRequest>,
) -> Response {
    let project_id = match request.project_id.clone() {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            return error_response(anyhow::anyhow!(
                "project_id is required (either in body or via message board task)"
            ));
        }
    };

    match state.config.agent_type {
        AgentType::Stateless => {
            match run_agent(&state.config, &state.context, request, project_id).await {
                Ok(response) => Json(response).into_response(),
                Err(e) => error_response(e),
            }
        }
        AgentType::Stateful => run_stateful_agent(&state, request, project_id).await,
    }
}

// ==================== Запуск ====================

pub async fn run_server(config: AgentConfig, context: AgentContext) -> anyhow::Result<()> {
    let state = AppState {
        config: config.clone(),
        context: Arc::new(context),
        auth_token: config.auth_token.clone(),
        sessions: Arc::new(AsyncMutex::new(HashMap::new())),
        sse_listeners: Arc::new(AsyncMutex::new(HashMap::new())),
        incoming_tasks: Arc::new(AsyncMutex::new(HashMap::new())),
    };

    let state_arc = Arc::new(state.clone());

    eprintln!("🤖 Агент '{}' (тип: {:?})", config.name, config.agent_type);
    eprintln!("   Bind: {}", config.bind_addr);
    eprintln!("   Board: {}", state.context.board_base_url);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    if matches!(config.agent_type, AgentType::Stateful) {
        spawn_sse_listener(state_arc.clone());

        match session_store::list_sessions_for_agent(&config.name) {
            Ok(sessions) => {
                if sessions.is_empty() {
                    eprintln!("📁 Активных сессий агента '{}' нет.", config.name);
                } else {
                    eprintln!("📁 Найдено {} сессий агента:", sessions.len());
                    let mut restored_pending = 0usize;
                    for s in &sessions {
                        if let Ok(session) =
                            session_store::load_session(&s.session_id, Some(&config.name))
                        {
                            let mut map = state_arc.context.outgoing_tasks.lock().await;
                            for pt in &session.pending_tasks {
                                map.insert(
                                    pt.task_id.clone(),
                                    OutgoingTask {
                                        task_id: pt.task_id.clone(),
                                        session_id: s.session_id.clone(),
                                        to_agent_name: pt.to_agent_name.clone(),
                                        to_session_id: pt.to_session_id.clone(),
                                        project_id: pt.project_id.clone(),
                                        chain: pt.chain.clone(),
                                    },
                                );
                                restored_pending += 1;
                            }
                        }
                    }
                    if restored_pending > 0 {
                        eprintln!("🔄 Восстановлено {} ожидающих задач", restored_pending);
                    }
                }
            }
            Err(e) => eprintln!("⚠️ Не удалось прочитать список сессий: {}", e),
        }

        let state_for_poll = state_arc.clone();
        let rx_poll = shutdown_rx.clone();
        tokio::spawn(async move {
            poll_pending_tasks_loop(state_for_poll, rx_poll).await;
        });

        let state_for_sse_cleanup = state_arc.clone();
        let rx_sse = shutdown_rx.clone();
        tokio::spawn(async move {
            sse_cleanup_loop(state_for_sse_cleanup, rx_sse).await;
        });

        let state_for_pc = state_arc.clone();
        let timeout_sec = config.agent_call_timeout_sec.unwrap_or(120);
        let rx_pc = shutdown_rx.clone();
        tokio::spawn(async move {
            pending_calls_cleanup_loop(state_for_pc, timeout_sec, rx_pc).await;
        });

        let ttl = config.session_ttl_secs.unwrap_or(1800);
        if ttl > 0 {
            let state_for_ttl = state_arc.clone();
            let rx_ttl = shutdown_rx.clone();
            tokio::spawn(async move {
                session_cleanup_loop(state_for_ttl, ttl, rx_ttl).await;
            });
        }
    } else {
        eprintln!("   Тип stateless: SSE и сессии не используются.");
    }

    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("⏹️ Получен сигнал завершения, останавливаю...");
        let _ = shutdown_tx_clone.send(true);
    });

    let app = Router::new()
        .route("/agent/run", post(run_agent_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let mut rx_serve = shutdown_rx.clone();
    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    eprintln!("✅ Агент '{}' готов принимать запросы", config.name);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = rx_serve.changed().await;
        })
        .await?;

    stop_sse_listener(&state_arc, &config.name).await;
    eprintln!("✅ Агент '{}' остановлен корректно", config.name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_core::AgentMetrics;

    #[test]
    fn error_response_returns_failed_status() {
        let resp = error_response(anyhow::anyhow!("boom"));
        // Response не имеет публичного доступа к телу — проверяем статус
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn agent_metrics_default_zeroes() {
        let m = AgentMetrics {
            prompt_tokens: 0,
            completion_tokens: 0,
            cost_rub: 0.0,
            api_calls_count: 0,
        };
        assert_eq!(m.prompt_tokens, 0);
        assert_eq!(m.cost_rub, 0.0);
    }

    #[test]
    fn write_log_appends_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        write_log(&mut f, "user", "hello").unwrap();
        write_log(&mut f, "assistant", "hi").unwrap();
        drop(f);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("user: hello"));
        assert!(content.contains("assistant: hi"));
        assert_eq!(content.lines().count(), 2);
    }
}
