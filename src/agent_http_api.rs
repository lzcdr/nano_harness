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
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;

use crate::agent_core::{
    create_agent_engine, process_agent_turns, run_agent, AgentConfig, AgentContext, AgentMetrics,
    AgentRequest, AgentResponse, AgentType,
};
use crate::engine::Role;

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

fn generate_session_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", nanos)
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

async fn run_stateful_agent(state: &AppState, request: AgentRequest) -> Response {
    let session_id = request
        .session_id
        .clone()
        .unwrap_or_else(generate_session_id);
    let mut sessions = state.sessions.lock().await;

    // Удаляем просроченные сессии (по TTL)
    if let Some(ttl) = state.config.session_ttl_secs {
        let now = Instant::now();
        let mut to_remove = Vec::new();
        for (id, session_arc) in sessions.iter() {
            if let Ok(session) = session_arc.try_lock() {
                if now.duration_since(session.last_used) >= Duration::from_secs(ttl) {
                    to_remove.push(id.clone());
                }
            }
        }
        for id in to_remove {
            sessions.remove(&id);
        }
    }

    let session_arc = match sessions.get(&session_id) {
        Some(s) => s.clone(),
        None => {
            // Создаём новую сессию
            let mut engine = match create_agent_engine(&state.config) {
                Ok(e) => e,
                Err(e) => {
                    return Json(AgentResponse {
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
                    .into_response();
                }
            };

            let system_prompt = state
                .config
                .system_prompt
                .clone()
                .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());
            engine.add_message(Role::System, system_prompt.clone());

            // Создаём лог-файл
            let log_dir = format!("chats/{}", state.config.name);
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                return Json(AgentResponse {
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
                    error: Some(format!("Failed to create log dir: {}", e)),
                    session_id: None,
                })
                .into_response();
            }
            let log_path = format!("{}/{}.txt", log_dir, session_id);
            let mut log_file = match OpenOptions::new().create(true).append(true).open(&log_path) {
                Ok(f) => f,
                Err(e) => {
                    return Json(AgentResponse {
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
                        error: Some(format!("Failed to open log file: {}", e)),
                        session_id: None,
                    })
                    .into_response();
                }
            };
            if let Err(e) = write_log(&mut log_file, "system", &system_prompt) {
                return Json(AgentResponse {
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
                    error: Some(format!("Failed to write system log: {}", e)),
                    session_id: None,
                })
                .into_response();
            }

            let session = Session {
                engine,
                log_file,
                last_used: Instant::now(),
            };
            let arc = Arc::new(AsyncMutex::new(session));
            sessions.insert(session_id.clone(), arc.clone());
            arc
        }
    };
    drop(sessions);

    // Блокируем сессию
    let mut session_guard = session_arc.lock().await;
    session_guard.last_used = Instant::now();

    // Добавляем сообщение пользователя
    session_guard
        .engine
        .add_message(Role::User, request.prompt.clone());
    if let Err(e) = write_log(&mut session_guard.log_file, "task", &request.prompt) {
        return Json(AgentResponse {
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
            error: Some(format!("Failed to write task log: {}", e)),
            session_id: None,
        })
        .into_response();
    }

    // Разделяем заимствования: берём отдельные ссылки на engine и log_file
    let Session {
        engine, log_file, ..
    } = &mut *session_guard;

    // Выполняем цикл обработки
    match process_agent_turns(engine, &state.config, &state.context, &request, log_file).await {
        Ok(mut response) => {
            response.session_id = Some(session_id);
            Json(response).into_response()
        }
        Err(e) => Json(AgentResponse {
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
            session_id: Some(session_id),
        })
        .into_response(),
    }
}

async fn run_agent_handler(
    State(state): State<AppState>,
    Json(request): Json<AgentRequest>,
) -> Response {
    match state.config.agent_type {
        AgentType::Stateless => match run_agent(&state.config, &state.context, request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => Json(AgentResponse {
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
            .into_response(),
        },
        AgentType::Stateful => run_stateful_agent(&state, request).await,
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

pub async fn run_server(config: AgentConfig, context: AgentContext) -> anyhow::Result<()> {
    let state = AppState {
        config: config.clone(),
        context: Arc::new(context),
        auth_token: config.auth_token.clone(),
        sessions: Arc::new(AsyncMutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/agent/run", post(run_agent_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    eprintln!(
        "🤖 Агент '{}' запущен на http://{}",
        config.name, config.bind_addr
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
