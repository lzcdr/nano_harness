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

// ---------- File handlers ----------

async fn create_file(
    State(state): State<AppState>,
    Query(query): Query<PathQuery>,
    body: String,
) -> Response {
    let mut storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.write_file(&query.path, &body) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn read_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.read_file(&query.path) {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn delete_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let mut storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.delete_file(&query.path) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

// ---------- Directory handlers ----------

async fn create_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let mut storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.create_dir(&query.path) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn list_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.list(&query.path) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn walk_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.walk(&query.path) {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

// ---------- Search handlers ----------

async fn search_similar(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.search_similar(&query.query, query.top_k) {
        Ok(results) => Json(results).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn search_by_name(
    State(state): State<AppState>,
    Query(query): Query<NameSearchQuery>,
) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.search_by_name(&query.pattern) {
        Ok(paths) => Json(paths).into_response(),
        Err(e) => io_error_response(e),
    }
}

// ---------- Meta handlers (.about, .summary) ----------

async fn read_about(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.read_about(&query.path) {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_about(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    let mut storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.write_about(&query.path, &body) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn read_summary(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.read_summary(&query.path) {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_summary(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    let mut storage = match state.storage.lock() {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Lock poisoned").into_response(),
    };
    match storage.write_summary(&query.path, &body) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

/// Запускает HTTP-сервер для работы с локальным хранилищем.
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
    axum::serve(listener, app).await?;
    Ok(())
}
