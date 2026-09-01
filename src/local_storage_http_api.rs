// src/local_storage_http_api.rs

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use tokio::task; // добавлено для spawn_blocking

use crate::local_storage::{LocalStorage, VectorDbConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct LocalStorageServerConfig {
    pub bind_addr: String,
    pub storage_name: String,
    pub auth_token: String,
}

#[derive(Clone)]
struct AppState {
    storage: Arc<Mutex<LocalStorage>>,
    auth_token: String,
}

#[derive(Deserialize)]
struct PathQuery {
    path: String,
}

#[derive(Deserialize)]
struct SearchQuery {
    query: String,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Deserialize)]
struct NameSearchQuery {
    pattern: String,
}

#[derive(Deserialize)]
struct WriteMetaQuery {
    path: String,
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

fn io_error_response(err: std::io::Error) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response()
}

fn join_error_response(err: tokio::task::JoinError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("Join error: {}", err),
    )
        .into_response()
}

// ---------- File handlers ----------

async fn create_file(
    State(state): State<AppState>,
    Query(query): Query<PathQuery>,
    body: String,
) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();
    let content = body.clone();

    let result = task::spawn_blocking(move || {
        let mut storage = storage.lock().unwrap();
        storage.write_file(&path, &content)
    })
    .await;

    match result {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn read_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.read_file(&path)
    })
    .await;

    match result {
        Ok(Ok(content)) => (StatusCode::OK, content).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn delete_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let mut storage = storage.lock().unwrap();
        storage.delete_file(&path)
    })
    .await;

    match result {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

// ---------- Directory handlers ----------

async fn create_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let mut storage = storage.lock().unwrap();
        storage.create_dir(&path)
    })
    .await;

    match result {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn list_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.list(&path)
    })
    .await;

    match result {
        Ok(Ok(entries)) => Json(entries).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn walk_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.walk(&path)
    })
    .await;

    match result {
        Ok(Ok(entries)) => Json(entries).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

// ---------- Search handlers ----------

async fn search_similar(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Response {
    let storage = state.storage.clone();
    let query_text = query.query.clone();
    let top_k = query.top_k;

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.search_similar(&query_text, top_k)
    })
    .await;

    match result {
        Ok(Ok(results)) => Json(results).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn search_by_name(
    State(state): State<AppState>,
    Query(query): Query<NameSearchQuery>,
) -> Response {
    let storage = state.storage.clone();
    let pattern = query.pattern.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.search_by_name(&pattern)
    })
    .await;

    match result {
        Ok(Ok(paths)) => Json(paths).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

// ---------- Meta handlers (.about, .summary) ----------

async fn read_about(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.read_about(&path)
    })
    .await;

    match result {
        Ok(Ok(content)) => (StatusCode::OK, content).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn write_about(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();
    let content = body.clone();

    let result = task::spawn_blocking(move || {
        let mut storage = storage.lock().unwrap();
        storage.write_about(&path, &content)
    })
    .await;

    match result {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn read_summary(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();

    let result = task::spawn_blocking(move || {
        let storage = storage.lock().unwrap();
        storage.read_summary(&path)
    })
    .await;

    match result {
        Ok(Ok(content)) => (StatusCode::OK, content).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn write_summary(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    let storage = state.storage.clone();
    let path = query.path.clone();
    let content = body.clone();

    let result = task::spawn_blocking(move || {
        let mut storage = storage.lock().unwrap();
        storage.write_summary(&path, &content)
    })
    .await;

    match result {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(e) => join_error_response(e),
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

pub async fn run_server(
    config: LocalStorageServerConfig,
    vector_db_config: VectorDbConfig,
) -> anyhow::Result<()> {
    let storage = LocalStorage::new(&config.storage_name, vector_db_config)?;

    let state = AppState {
        storage: Arc::new(Mutex::new(storage)),
        auth_token: config.auth_token,
    };

    let app = Router::new()
        .route(
            "/files",
            post(create_file).get(read_file).delete(delete_file),
        )
        .route("/dirs", post(create_dir))
        .route("/list", get(list_dir))
        .route("/walk", get(walk_dir))
        .route("/search", get(search_similar))
        .route("/search_name", get(search_by_name))
        .route("/about", get(read_about).post(write_about))
        .route("/summary", get(read_summary).post(write_summary))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    eprintln!("🌐 HTTP API запущен на http://{}", config.bind_addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
