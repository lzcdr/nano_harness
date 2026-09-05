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
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;

use crate::agent_core::{
    create_agent_engine, process_agent_turns, run_agent, AgentConfig, AgentContext, AgentMetrics,
    AgentRequest, AgentResponse, AgentType,
};
use crate::engine::Role;
use crate::session_store;

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

async fn run_stateful_agent(state: &AppState, request: AgentRequest) -> Response {
    let session_id = request
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Попытка получить из кэша
    {
        let sessions = state.sessions.lock().await;
        if let Some(cached) = sessions.get(&session_id).cloned() {
            drop(sessions);
            let mut session_guard = cached.lock().await;
            session_guard.last_used = Instant::now();

            session_guard
                .engine
                .add_message(Role::User, request.prompt.clone());
            if let Err(e) = write_log(&mut session_guard.log_file, "task", &request.prompt) {
                eprintln!("Ошибка записи в лог: {}", e);
            }

            let Session {
                engine, log_file, ..
            } = &mut *session_guard;

            match process_agent_turns(engine, &state.config, &state.context, &request, log_file)
                .await
            {
                Ok(mut response) => {
                    response.session_id = Some(session_id.clone());
                    // Обновить и сохранить сессию
                    if let Ok(mut store_session) = session_store::load_session(&session_id) {
                        if let Some(ctx) = store_session
                            .contexts
                            .agents
                            .iter_mut()
                            .find(|a| a.name == state.config.name)
                        {
                            ctx.engine_state = engine.get_state();
                            ctx.engine_config = engine.get_config().clone();
                        } else {
                            store_session
                                .contexts
                                .agents
                                .push(session_store::AgentContextBlock {
                                    name: state.config.name.clone(),
                                    engine_config: engine.get_config().clone(),
                                    engine_state: engine.get_state(),
                                });
                        }
                        store_session.updated_at = session_store::now_ts();
                        if let Err(e) = session_store::save_session(&store_session) {
                            eprintln!("Ошибка сохранения сессии: {}", e);
                        }
                    }
                    return Json(response).into_response();
                }
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
                        session_id: Some(session_id),
                    })
                    .into_response();
                }
            };
        }
    }

    // Загрузка или создание сессии
    let mut store_session = match session_store::load_session_with_key(
        &session_id,
        &state.config.api_key,
        Some(&state.config.name),
    ) {
        Ok(s) => {
            // restore_api_key_for_agent уже применён внутри load_session_with_key
            s
        }
        Err(_) => {
            let display_name = format!("Agent session {}", session_id);
            match session_store::create_session_with_id(&session_id, &display_name) {
                Ok(s) => s,
                Err(e) => return error_response(e),
            }
        }
    };

    // Инициализация лога для агента
    if let Err(e) =
        session_store::add_or_update_log(&mut store_session, "agent", Some(&state.config.name))
    {
        eprintln!("Ошибка инициализации лога: {}", e);
    }

    // Получение или создание контекста агента
    let agent_ctx = store_session
        .contexts
        .agents
        .iter_mut()
        .find(|a| a.name == state.config.name);

    let engine = if let Some(ctx) = agent_ctx {
        let mut engine = match create_agent_engine(&state.config) {
            Ok(e) => e,
            Err(e) => return error_response(e),
        };
        engine.set_state(ctx.engine_state.clone());
        engine
    } else {
        let mut engine = match create_agent_engine(&state.config) {
            Ok(e) => e,
            Err(e) => return error_response(e),
        };
        let system_prompt = state
            .config
            .system_prompt
            .clone()
            .unwrap_or_else(|| "Вы - полезный ассистент.".to_string());
        engine.add_message(Role::System, system_prompt);
        store_session
            .contexts
            .agents
            .push(session_store::AgentContextBlock {
                name: state.config.name.clone(),
                engine_config: engine.get_config().clone(),
                engine_state: engine.get_state(),
            });
        engine
    };

    // Открытие лог-файла
    let log_file = match session_store::open_log(&session_id, "agent", Some(&state.config.name)) {
        Ok(f) => f,
        Err(e) => return error_response(e),
    };

    // Сохранение сессии после возможного создания контекста
    if let Err(e) = session_store::save_session(&store_session) {
        eprintln!("Ошибка сохранения сессии: {}", e);
    }

    // Кэширование сессии
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
        .insert(session_id.clone(), arc.clone());

    let mut session_guard = arc.lock().await;
    session_guard.last_used = Instant::now();
    session_guard
        .engine
        .add_message(Role::User, request.prompt.clone());
    if let Err(e) = write_log(&mut session_guard.log_file, "task", &request.prompt) {
        eprintln!("Ошибка записи в лог: {}", e);
    }

    let Session {
        engine, log_file, ..
    } = &mut *session_guard;

    match process_agent_turns(engine, &state.config, &state.context, &request, log_file).await {
        Ok(mut response) => {
            response.session_id = Some(session_id.clone());
            // Обновить и сохранить сессию
            if let Ok(mut store_session) = session_store::load_session(&session_id) {
                if let Some(ctx) = store_session
                    .contexts
                    .agents
                    .iter_mut()
                    .find(|a| a.name == state.config.name)
                {
                    ctx.engine_state = engine.get_state();
                    ctx.engine_config = engine.get_config().clone();
                } else {
                    store_session
                        .contexts
                        .agents
                        .push(session_store::AgentContextBlock {
                            name: state.config.name.clone(),
                            engine_config: engine.get_config().clone(),
                            engine_state: engine.get_state(),
                        });
                }
                store_session.updated_at = session_store::now_ts();
                if let Err(e) = session_store::save_session(&store_session) {
                    eprintln!("Ошибка сохранения сессии: {}", e);
                }
            }
            return Json(response).into_response();
        }
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
                session_id: Some(session_id),
            })
            .into_response();
        }
    };
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
