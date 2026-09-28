// src/local_storage_http_api.rs
use axum::{
    extract::{FromRequestParts, Query, State},
    http::{request::Parts, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use crate::knowledge_manager::KnowledgeEntry;
use crate::local_storage::{valid_project_id, LocalStorage, VectorDbConfig, SKILLS_PROJECT};

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
    skill_index_tx: mpsc::Sender<SkillIndexTask>,
    skill_delete_tx: mpsc::Sender<SkillDeleteTask>,
    knowledge_catalog: Arc<RwLock<Vec<KnowledgeEntry>>>,
    auth_token: String,
}

struct FileStorage {
    root: std::path::PathBuf,
}

impl FileStorage {
    fn resolve(&self, project_id: &str, path: &str) -> std::io::Result<std::path::PathBuf> {
        if !valid_project_id(project_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid project_id",
            ));
        }
        let mut out = std::path::PathBuf::from("projects").join(project_id);
        for c in std::path::Path::new(path).components() {
            match c {
                std::path::Component::Normal(p) => {
                    if p.to_str().map(crate::local_storage::is_reserved_name) == Some(true) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "path refers to reserved file",
                        ));
                    }
                    out.push(p)
                }
                std::path::Component::CurDir => {}
                std::path::Component::RootDir => {}
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "path escapes storage root",
                    ));
                }
            }
        }
        Ok(self.root.join(out))
    }

    fn resolve_skill(&self, path: &str) -> std::io::Result<std::path::PathBuf> {
        let mut out = std::path::PathBuf::from(".skills");
        for c in std::path::Path::new(path).components() {
            match c {
                std::path::Component::Normal(p) => {
                    let s = p.to_str().unwrap_or("");
                    if crate::local_storage::is_reserved_name(s) {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "path refers to reserved file",
                        ));
                    }
                    out.push(p)
                }
                std::path::Component::CurDir => {}
                std::path::Component::RootDir => {}
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "path escapes skills root",
                    ));
                }
            }
        }
        Ok(self.root.join(out))
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

    async fn ensure_project(&self, project_id: &str) -> std::io::Result<()> {
        if !valid_project_id(project_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid project_id",
            ));
        }
        let dir = self.root.join("projects").join(project_id);
        if !dir.exists() {
            tokio::fs::create_dir_all(&dir).await?;
        }
        Self::ensure_system_files(&dir)?;
        Ok(())
    }

    async fn create_dir(&self, project_id: &str, path: &str) -> std::io::Result<()> {
        self.ensure_project(project_id).await?;
        let full = self.resolve(project_id, path)?;
        tokio::fs::create_dir_all(&full).await?;
        Self::ensure_system_files(&full)
    }

    async fn create_file(
        &self,
        project_id: &str,
        path: &str,
        content: &str,
    ) -> std::io::Result<()> {
        self.ensure_project(project_id).await?;
        let full = self.resolve(project_id, path)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await?;
            }
            Self::ensure_system_files(parent)?;
        }
        tokio::fs::write(&full, content).await?;
        Ok(())
    }

    async fn read_file(&self, project_id: &str, path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(project_id, path)?).await
    }

    async fn delete_file(&self, project_id: &str, path: &str) -> std::io::Result<()> {
        tokio::fs::remove_file(self.resolve(project_id, path)?).await
    }

    async fn list(
        &self,
        project_id: &str,
        path: &str,
    ) -> std::io::Result<Vec<crate::local_storage::Entry>> {
        let full = self.resolve(project_id, path)?;
        let project_root = self.root.join("projects").join(project_id);
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(full).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            if crate::local_storage::is_reserved_name(&name) {
                continue;
            }
            let rel_path = entry
                .path()
                .strip_prefix(&project_root)
                .unwrap_or(&entry.path())
                .to_path_buf();
            if entry.file_type().await?.is_dir() {
                entries.push(crate::local_storage::Entry::Dir {
                    name,
                    path: rel_path,
                });
            } else {
                entries.push(crate::local_storage::Entry::File {
                    name,
                    path: rel_path,
                    size: entry.metadata().await?.len(),
                });
            }
        }
        Ok(entries)
    }

    async fn walk(
        &self,
        project_id: &str,
        path: &str,
    ) -> std::io::Result<Vec<crate::local_storage::Entry>> {
        let full = self.resolve(project_id, path)?;
        let project_root = self.root.join("projects").join(project_id);
        let mut result = Vec::new();
        let mut stack = vec![full];

        while let Some(current_dir) = stack.pop() {
            let mut dir = tokio::fs::read_dir(&current_dir).await?;
            while let Some(entry) = dir.next_entry().await? {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if crate::local_storage::is_reserved_name(&name) {
                    continue;
                }
                let rel = path
                    .strip_prefix(&project_root)
                    .unwrap_or(&path)
                    .to_path_buf();
                if entry.file_type().await?.is_dir() {
                    result.push(crate::local_storage::Entry::Dir {
                        name: name.clone(),
                        path: rel,
                    });
                    stack.push(path);
                } else {
                    result.push(crate::local_storage::Entry::File {
                        name,
                        path: rel,
                        size: entry.metadata().await?.len(),
                    });
                }
            }
        }
        Ok(result)
    }

    async fn read_about(&self, project_id: &str, dir_path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(project_id, dir_path)?.join(".about")).await
    }

    async fn write_about(
        &self,
        project_id: &str,
        dir_path: &str,
        content: &str,
    ) -> std::io::Result<()> {
        self.ensure_project(project_id).await?;
        tokio::fs::write(self.resolve(project_id, dir_path)?.join(".about"), content).await
    }

    async fn read_summary(&self, project_id: &str, dir_path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve(project_id, dir_path)?.join(".summary")).await
    }

    async fn write_summary(
        &self,
        project_id: &str,
        dir_path: &str,
        content: &str,
    ) -> std::io::Result<()> {
        self.ensure_project(project_id).await?;
        tokio::fs::write(
            self.resolve(project_id, dir_path)?.join(".summary"),
            content,
        )
        .await
    }

    async fn read_skill_file(&self, path: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve_skill(path)?).await
    }

    async fn write_skill_file(&self, path: &str, content: &str) -> std::io::Result<()> {
        let full = self.resolve_skill(path)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(&full, content).await
    }

    async fn delete_skill_file(&self, path: &str) -> std::io::Result<()> {
        tokio::fs::remove_file(self.resolve_skill(path)?).await
    }

    fn resolve_knowledge(&self, name: &str) -> std::io::Result<std::path::PathBuf> {
        if !crate::knowledge_manager::validate_name(name) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid knowledge name",
            ));
        }
        Ok(self.root.join(".knowledge").join(format!("{}.md", name)))
    }

    async fn read_knowledge_file(&self, name: &str) -> std::io::Result<String> {
        tokio::fs::read_to_string(self.resolve_knowledge(name)?).await
    }

    async fn write_knowledge_file(&self, name: &str, content: &str) -> std::io::Result<()> {
        let full = self.resolve_knowledge(name)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(&full, content).await
    }

    async fn delete_knowledge_file(&self, name: &str) -> std::io::Result<()> {
        tokio::fs::remove_file(self.resolve_knowledge(name)?).await
    }

    fn resolve_rebuke(&self, agent: &str) -> std::io::Result<std::path::PathBuf> {
        if agent.is_empty() || agent.contains('/') || agent.contains('\\') || agent.contains("..") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid agent name",
            ));
        }
        Ok(self.root.join(".rebukes").join(format!("{}.txt", agent)))
    }

    async fn read_rebuke_file(&self, agent: &str) -> std::io::Result<String> {
        match tokio::fs::read_to_string(self.resolve_rebuke(agent)?).await {
            Ok(s) => Ok(s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
            Err(e) => Err(e),
        }
    }

    async fn write_rebuke_file(&self, agent: &str, content: &str) -> std::io::Result<()> {
        let full = self.resolve_rebuke(agent)?;
        if let Some(parent) = full.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(&full, content).await
    }

    async fn list_skill_dir(
        &self,
        path: &str,
    ) -> std::io::Result<Vec<crate::local_storage::Entry>> {
        let full = self.resolve_skill(path)?;
        let skills_root = self.root.join(".skills");
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(full).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            let rel_path = entry
                .path()
                .strip_prefix(&skills_root)
                .unwrap_or(&entry.path())
                .to_path_buf();
            if entry.file_type().await?.is_dir() {
                entries.push(crate::local_storage::Entry::Dir {
                    name,
                    path: rel_path,
                });
            } else {
                entries.push(crate::local_storage::Entry::File {
                    name,
                    path: rel_path,
                    size: entry.metadata().await?.len(),
                });
            }
        }
        Ok(entries)
    }
}

#[derive(Debug)]
struct SearchTask {
    project_id: Option<String>,
    query: String,
    top_k: Option<usize>,
    response_tx:
        tokio::sync::oneshot::Sender<std::io::Result<Vec<crate::local_storage::SearchResult>>>,
}

#[derive(Debug)]
struct IndexTask {
    project_id: String,
    path: std::path::PathBuf,
    content: String,
}

#[derive(Debug)]
struct SkillIndexTask {
    path: std::path::PathBuf,
    content: String,
}

#[derive(Debug)]
struct SkillDeleteTask {
    path: std::path::PathBuf,
}

struct ProjectId(String);

impl<S> FromRequestParts<S> for ProjectId
where
    S: Send + Sync,
{
    type Rejection = Response;
    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let pid = parts
            .headers
            .get("x-nh-project")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !valid_project_id(pid) {
            return Err((
                StatusCode::BAD_REQUEST,
                "missing or invalid X-NH-Project header",
            )
                .into_response());
        }
        Ok(ProjectId(pid.to_string()))
    }
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
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
    body: String,
) -> Response {
    let path = query.path.clone();
    let content = body.clone();
    match state
        .files
        .read()
        .await
        .create_file(&project_id, &path, &content)
        .await
    {
        Ok(_) => {
            let _ = state
                .index_tx
                .send(IndexTask {
                    project_id,
                    path: std::path::PathBuf::from(path),
                    content,
                })
                .await;
            StatusCode::OK.into_response()
        }
        Err(e) => io_error_response(e),
    }
}

async fn read_file(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .read_file(&project_id, &query.path)
        .await
    {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn delete_file(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .delete_file(&project_id, &query.path)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn create_dir(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .create_dir(&project_id, &query.path)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn list_dir(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .list(&project_id, &query.path)
        .await
    {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn walk_dir(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .walk(&project_id, &query.path)
        .await
    {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn search_similar(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<SearchQuery>,
) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            project_id: Some(project_id),
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
    ProjectId(project_id): ProjectId,
    Query(query): Query<NameSearchQuery>,
) -> Response {
    match state.files.read().await.walk(&project_id, "").await {
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

async fn read_about(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .read_about(&project_id, &query.path)
        .await
    {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_about(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    match state
        .files
        .read()
        .await
        .write_about(&project_id, &query.path, &body)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn read_summary(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<PathQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .read_summary(&project_id, &query.path)
        .await
    {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn write_summary(
    State(state): State<AppState>,
    ProjectId(project_id): ProjectId,
    Query(query): Query<WriteMetaQuery>,
    body: String,
) -> Response {
    match state
        .files
        .read()
        .await
        .write_summary(&project_id, &query.path, &body)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn skill_get(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.read_skill_file(&query.path).await {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn skill_put(
    State(state): State<AppState>,
    Query(query): Query<PathQuery>,
    body: String,
) -> Response {
    let path = query.path.clone();
    let content = body.clone();
    match state
        .files
        .read()
        .await
        .write_skill_file(&path, &content)
        .await
    {
        Ok(_) => {
            let _ = state
                .skill_index_tx
                .send(SkillIndexTask {
                    path: std::path::PathBuf::from(path),
                    content,
                })
                .await;
            StatusCode::OK.into_response()
        }
        Err(e) => io_error_response(e),
    }
}

async fn skill_delete(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let path = query.path.clone();
    match state.files.read().await.delete_skill_file(&path).await {
        Ok(_) => {
            let _ = state
                .skill_delete_tx
                .send(SkillDeleteTask {
                    path: std::path::PathBuf::from(path),
                })
                .await;
            StatusCode::OK.into_response()
        }
        Err(e) => io_error_response(e),
    }
}

async fn skill_list(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    match state.files.read().await.list_skill_dir(&query.path).await {
        Ok(entries) => Json(entries).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn skill_search(State(state): State<AppState>, Query(query): Query<SearchQuery>) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            project_id: Some(SKILLS_PROJECT.to_string()),
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

#[derive(Deserialize)]
struct AgentQuery {
    agent: String,
}

async fn rebuke_get(State(state): State<AppState>, Query(query): Query<AgentQuery>) -> Response {
    match state
        .files
        .read()
        .await
        .read_rebuke_file(&query.agent)
        .await
    {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn rebuke_put(
    State(state): State<AppState>,
    Query(query): Query<AgentQuery>,
    body: String,
) -> Response {
    match state
        .files
        .read()
        .await
        .write_rebuke_file(&query.agent, &body)
        .await
    {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => io_error_response(e),
    }
}

#[derive(Deserialize)]
struct KnowledgeNameQuery {
    name: String,
}

async fn scan_knowledge_catalog(root: &std::path::Path) -> Vec<KnowledgeEntry> {
    let dir = root.join(".knowledge");
    let mut entries = Vec::new();
    if let Ok(mut d) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = d.next_entry().await {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".md") {
                continue;
            }
            if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                if let Ok((name, description, _)) =
                    crate::knowledge_manager::parse_frontmatter(&content)
                {
                    entries.push(KnowledgeEntry { name, description });
                }
            }
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

async fn knowledge_list(State(state): State<AppState>) -> Response {
    let catalog = state.knowledge_catalog.read().await;
    Json(catalog.clone()).into_response()
}

async fn knowledge_get(
    State(state): State<AppState>,
    Query(query): Query<KnowledgeNameQuery>,
) -> Response {
    match state
        .files
        .read()
        .await
        .read_knowledge_file(&query.name)
        .await
    {
        Ok(content) => (StatusCode::OK, content).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn knowledge_put(
    State(state): State<AppState>,
    Query(query): Query<KnowledgeNameQuery>,
    body: String,
) -> Response {
    let (inner_name, _, _) = match crate::knowledge_manager::parse_frontmatter(&body) {
        Ok(triple) => triple,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid frontmatter: {}", e),
            )
                .into_response();
        }
    };

    if inner_name != query.name {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "name в frontmatter ('{}') не совпадает с именем файла ('{}')",
                inner_name, query.name
            ),
        )
            .into_response();
    }

    {
        let files = state.files.read().await;
        if let Err(e) = files.write_knowledge_file(&query.name, &body).await {
            return io_error_response(e);
        }
    }

    let new_catalog = {
        let files = state.files.read().await;
        scan_knowledge_catalog(&files.root).await
    };
    *state.knowledge_catalog.write().await = new_catalog;

    StatusCode::OK.into_response()
}

async fn knowledge_delete(
    State(state): State<AppState>,
    Query(query): Query<KnowledgeNameQuery>,
) -> Response {
    {
        let files = state.files.read().await;
        match files.delete_knowledge_file(&query.name).await {
            Ok(_) => {}
            Err(e) => return io_error_response(e),
        }
    }

    let new_catalog = {
        let files = state.files.read().await;
        scan_knowledge_catalog(&files.root).await
    };
    *state.knowledge_catalog.write().await = new_catalog;

    StatusCode::OK.into_response()
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
    let knowledge_root = root.clone();
    let files = Arc::new(RwLock::new(FileStorage { root }));
    let knowledge_catalog = Arc::new(RwLock::new(scan_knowledge_catalog(&knowledge_root).await));

    let (search_tx, mut search_rx) = mpsc::channel::<SearchTask>(100);
    let (index_tx, mut index_rx) = mpsc::channel::<IndexTask>(100);
    let (skill_index_tx, mut skill_index_rx) = mpsc::channel::<SkillIndexTask>(100);
    let (skill_delete_tx, mut skill_delete_rx) = mpsc::channel::<SkillDeleteTask>(100);

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
                        let result = storage.search_similar(
                            task.project_id.as_deref(),
                            &task.query,
                            task.top_k,
                        );
                        let _ = task.response_tx.send(result);
                    }
                    Some(task) = index_rx.recv() => {
                        if let Err(e) = storage.create_file(
                            &task.project_id,
                            &task.path.to_string_lossy(),
                            &task.content,
                        ) {
                            eprintln!("⚠️ Indexing failed for {:?}: {}", task.path, e);
                        }
                    }
                    Some(task) = skill_index_rx.recv() => {
                        if let Err(e) = storage.create_skill_file(
                            &task.path.to_string_lossy(),
                            &task.content,
                        ) {
                            eprintln!("⚠️ Skill indexing failed for {:?}: {}", task.path, e);
                        }
                    }
                    Some(task) = skill_delete_rx.recv() => {
                        if let Err(e) = storage.delete_skill_index(
                            &task.path.to_string_lossy(),
                        ) {
                            eprintln!("⚠️ Skill deindex failed for {:?}: {}", task.path, e);
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
        skill_index_tx,
        skill_delete_tx,
        knowledge_catalog,
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
        .route("/skills/get", get(skill_get))
        .route("/skills/put", post(skill_put))
        .route("/skills/delete", post(skill_delete))
        .route("/skills/list", get(skill_list))
        .route("/skills/search", get(skill_search))
        .route("/rebukes/get", get(rebuke_get))
        .route("/rebukes/put", post(rebuke_put))
        .route("/knowledge/list", get(knowledge_list))
        .route("/knowledge/get", get(knowledge_get))
        .route("/knowledge/put", post(knowledge_put))
        .route("/knowledge/delete", post(knowledge_delete))
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
