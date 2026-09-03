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
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use crate::local_storage::{LocalStorage, VectorDbConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct LocalStorageServerConfig {
    pub bind_addr: String,
    pub storage_name: String,
    pub auth_token: String,
}

#[derive(Clone)]
struct AppState {
    files: Arc<RwLock<FileStorage>>,
    search_tx: mpsc::Sender<SearchTask>,
    index_tx: mpsc::Sender<IndexTask>,
    auth_token: String,
}

struct FileStorage {
    root: std::path::PathBuf,
}

impl FileStorage {
    fn resolve(&self, path: &str) -> std::io::Result<std::path::PathBuf> {
        let rel = std::path::Path::new(path);
        if rel.is_absolute() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "absolute path not allowed",
            ));
        }
        if rel
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "path escapes storage root",
            ));
        }
        Ok(self.root.join(rel))
    }

    fn ensure_system_files(dir: &std::path::Path) -> std::io::Result<()> {
        let about = dir.join(".about");
        if !about.exists() {
            std::fs::write(&about, "")?;
        }
        let summary = dir.join(".summary");
        if !summary.exists() {
            std::fs::write(&summary, "")?;
        }
        Ok(())
    }

    async fn create_dir(&self, path: &str) -> std::io::Result<()> {
        let full = self.resolve(path)?;
        tokio::fs::create_dir_all(&full).await?;
        Self::ensure_system_files(&full)
    }

    async fn create_file(&self, path: &str, content: &str) -> std::io::Result<()> {
        let full = self.resolve(path)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await?;
            }
            Self::ensure_system_files(parent)?;
        }
        tokio::fs::write(&full, content).await?;
        Ok(())
    }

    async fn read_file(&self, path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(path)?).await
    }
    async fn delete_file(&self, path: &str) -> std::io::Result<()> {
        tokio::fs::remove_file(self.resolve(path)?).await
    }

    async fn list(&self, path: &str) -> std::io::Result<Vec<crate::local_storage::Entry>> {
        let full = self.resolve(path)?;
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(full).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".about" || name == ".summary" {
                continue;
            }
            if entry.file_type().await?.is_dir() {
                entries.push(crate::local_storage::Entry::Dir {
                    name,
                    path: entry.path(),
                });
            } else {
                entries.push(crate::local_storage::Entry::File {
                    name,
                    path: entry.path(),
                    size: entry.metadata().await?.len(),
                });
            }
        }
        Ok(entries)
    }

    // Итеративный обход вместо рекурсивного async (избегает ошибки E0733)
    async fn walk(&self, path: &str) -> std::io::Result<Vec<crate::local_storage::Entry>> {
        let full = self.resolve(path)?;
        let mut result = Vec::new();
        let mut stack = vec![full];

        while let Some(current_dir) = stack.pop() {
            let mut dir = tokio::fs::read_dir(&current_dir).await?;
            while let Some(entry) = dir.next_entry().await? {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if name == ".about" || name == ".summary" {
                    continue;
                }
                if entry.file_type().await?.is_dir() {
                    result.push(crate::local_storage::Entry::Dir {
                        name: name.clone(),
                        path: self.relativize(&path),
                    });
                    stack.push(path);
                } else {
                    result.push(crate::local_storage::Entry::File {
                        name,
                        path: self.relativize(&path),
                        size: entry.metadata().await?.len(),
                    });
                }
            }
        }
        Ok(result)
    }

    fn relativize(&self, path: &std::path::Path) -> std::path::PathBuf {
        path.strip_prefix(&self.root).unwrap_or(path).to_path_buf()
    }

    async fn read_about(&self, dir_path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(dir_path)?.join(".about")).await
    }
    async fn write_about(&self, dir_path: &str, content: &str) -> std::io::Result<()> {
        tokio::fs::write(self.resolve(dir_path)?.join(".about"), content).await
    }
    async fn read_summary(&self, dir_path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(dir_path)?.join(".summary")).await
    }
    async fn write_summary(&self, dir_path: &str, content: &str) -> std::io::Result<()> {
        tokio::fs::write(self.resolve(dir_path)?.join(".summary"), content).await
    }
}

#[derive(Debug)]
struct SearchTask {
    query: String,
    top_k: Option<usize>,
    response_tx:
        tokio::sync::oneshot::Sender<std::io::Result<Vec<crate::local_storage::SearchResult>>>,
}

#[derive(Debug)]
struct IndexTask {
    path: std::path::PathBuf,
    content: String,
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

async fn create_file(
    State(state): State<AppState>,
    Query(query): Query<PathQuery>,
    body: String,
) -> Response {
    let path = query.path.clone();
    let content = body.clone();
    match state.files.read().await.create_file(&path, &content).await {
        Ok(_) => {
            let _ = state
                .index_tx
                .send(IndexTask {
                    path: std::path::PathBuf::from(path),
                    content,
                })
                .await;
            StatusCode::OK.into_response()
        }
        Err(e) => io_error_response(e),
    }
}

async fn read_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.read_file(&query.path).await {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn delete_file(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.delete_file(&query.path).await {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn create_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.create_dir(&query.path).await {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn list_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.list(&query.path).await {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn walk_dir(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.walk(&query.path).await {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn search_similar(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            query: query.query.clone(),
            top_k: query.top_k,
            response_tx: tx,
        })
        .await
        .is_err()
    {
        return (StatusCode::SERVICE_UNAVAILABLE, "Search queue closed").into_response();
    }
    match rx.await {
        Ok(Ok(results)) => Json(results).into_response(),
        Ok(Err(e)) => io_error_response(e),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Search task cancelled").into_response(),
    }
}

async fn search_by_name(
    State(state): State<AppState>,
    Query(query): Query<NameSearchQuery>,
) -> Response {
    match state.files.read().await.walk("").await {
        Ok(all) => {
            let results: Vec<_> = all
                .into_iter()
                .filter_map(|entry| match entry {
                    crate::local_storage::Entry::File { name, path, .. }
                    | crate::local_storage::Entry::Dir { name, path } => {
                        if name.contains(&query.pattern) {
                            Some(path)
                        } else {
                            None
                        }
                    }
                })
                .collect();
            Json(results).into_response()
        }
        Err(e) => io_error_response(e),
    }
}

async fn read_about(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.read_about(&query.path).await {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_about(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    match state
        .files
        .read()
        .await
        .write_about(&query.path, &body)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn read_summary(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.read_summary(&query.path).await {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_summary(
    State(state): State<AppState>,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    match state
        .files
        .read()
        .await
        .write_summary(&query.path, &body)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

pub async fn run_server(
    config: LocalStorageServerConfig,
    vector_db_config: VectorDbConfig,
) -> anyhow::Result<()> {
    let base = std::path::PathBuf::from(".local_storage");
    let root = base.join(&config.storage_name);
    let files = Arc::new(RwLock::new(FileStorage { root }));

    let (search_tx, mut search_rx) = mpsc::channel::<SearchTask>(100);
    let (index_tx, mut index_rx) = mpsc::channel::<IndexTask>(100);

    // ГАРАНТИРОВАННОЕ РЕШЕНИЕ: отдельный поток для VectorDB, чтобы избежать паники "runtime within runtime"
    let vector_db_config_clone = vector_db_config.clone();
    let storage_name = config.storage_name.clone();
    std::thread::spawn(move || {
        let mut storage = match LocalStorage::new(&storage_name, vector_db_config_clone) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Failed to initialize LocalStorage: {}", e);
                return;
            }
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build runtime");
        rt.block_on(async move {
            loop {
                tokio::select! {
                    Some(task) = search_rx.recv() => {
                        let result = storage.search_similar(&task.query, task.top_k);
                        let _ = task.response_tx.send(result);
                    }
                    Some(task) = index_rx.recv() => {
                        if let Err(e) = storage.create_file(&task.path.to_string_lossy(), &task.content) {
                            eprintln!("⚠️ Indexing failed for {:?}: {}", task.path, e);
                        }
                    }
                    else => break,
                }
            }
        });
    });

    let state = AppState {
        files,
        search_tx,
        index_tx,
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
