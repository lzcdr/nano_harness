
// src/agent_http_api.rs

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use std::sync::Arc;

use crate::agent_core::{
    run_agent, AgentConfig, AgentContext, AgentMetrics, AgentRequest, AgentResponse,
};

#[derive(Clone)]
struct AppState {
    config: AgentConfig,
    context: Arc<AgentContext>,
    auth_token: String,
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

async fn run_agent_handler(
    State(state): State<AppState>,
    Json(request): Json<AgentRequest>,
) -> Response {
    match run_agent(&state.config, &state.context, request).await {
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
        })
        .into_response(),
    }
}

pub async fn run_server(
    config: AgentConfig,
    context: AgentContext,
) -> anyhow::Result<()> {
    let state = AppState {
        config: config.clone(),
        context: Arc::new(context),
        auth_token: config.auth_token.clone(),
    };

    let app = Router::new()
        .route("/agent/run", post(run_agent_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    eprintln!("🤖 Агент '{}' запущен на http://{}", config.name, config.bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}
