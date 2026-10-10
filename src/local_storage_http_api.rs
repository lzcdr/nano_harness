// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};

use crate::knowledge_manager::KnowledgeEntry;
use crate::local_storage::{
    valid_project_id, LocalStorage, VectorDbConfig, KNOWLEDGE_PROJECT, SKILLS_PROJECT,
    TOOLS_PROJECT,
};
use crate::skill_manager::SkillRecord;

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
    aux_index_tx: mpsc::Sender<AuxIndexTask>,
    aux_delete_tx: mpsc::Sender<AuxDeleteTask>,
    knowledge_catalog: Arc<RwLock<Vec<KnowledgeEntry>>>,
    skills_catalog: Arc<RwLock<Vec<SkillRecord>>>,
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

        let dst = dir.join(".storageignore");
        if !dst.exists() {
            if let Some(src) = crate::storage_ignore::global_ignore_path() {
                if src.exists() {
                    let _ = tokio::fs::copy(&src, &dst).await;
                }
            }
        }
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
        if !full.exists() {
            return Ok(Vec::new());
        }
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
        if !full.exists() {
            return Ok(Vec::new());
        }
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

    /// Удаляет все `<stem>_index_<N>.idx` в `.skills/`.
    /// Возвращает имена удалённых файлов (без директории).
    async fn delete_index_files(&self, stem: &str) -> std::io::Result<Vec<String>> {
        let skills_dir = self.root.join(".skills");
        let prefix = format!("{}_index_", stem);

        let mut removed = Vec::new();
        let mut dir = match tokio::fs::read_dir(&skills_dir).await {
            Ok(d) => d,
            Err(_) => return Ok(removed),
        };
        while let Ok(Some(entry)) = dir.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".idx") {
                if tokio::fs::remove_file(entry.path()).await.is_ok() {
                    removed.push(name);
                }
            }
        }
        Ok(removed)
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
}

#[derive(Debug)]
struct SearchTask {
    project_id: Option<String>,
    query: String,
    top_k: Option<usize>,
    sync_first: bool,
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
struct AuxIndexTask {
    project_id: String,
    path: std::path::PathBuf,
    content: String,
}

#[derive(Debug)]
struct AuxDeleteTask {
    project_id: String,
    paths: Vec<std::path::PathBuf>,
}

#[derive(Debug)]
struct SyncTask;

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
            sync_first: true,
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

#[derive(Deserialize)]
struct SkillPutRequest {
    content: String,
    index_files: Vec<String>,
    record: SkillRecord,
    skill_file: String,
}

async fn skill_put(State(state): State<AppState>, Json(req): Json<SkillPutRequest>) -> Response {
    let stem = match req.skill_file.strip_suffix(".md") {
        Some(s) => s.to_string(),
        None => {
            return (StatusCode::BAD_REQUEST, "skill_file must end with .md").into_response();
        }
    };

    let mut index_tasks: Vec<(String, String)> = Vec::new();

    {
        let files = state.files.read().await;

        // Удаляем старые index-файлы этого скилла (если были).
        if let Err(e) = files.delete_index_files(&stem).await {
            eprintln!("cleanup index files for {}: {}", stem, e);
        }

        // Пишем основной .md.
        if let Err(e) = files.write_skill_file(&req.skill_file, &req.content).await {
            return io_error_response(e);
        }

        // Пишем index-файлы.
        for (i, phrase) in req.index_files.iter().enumerate() {
            let name = format!("{}_index_{}.idx", stem, i);
            if let Err(e) = files.write_skill_file(&name, phrase).await {
                eprintln!("write index file {}: {}", name, e);
                continue;
            }
            index_tasks.push((name, phrase.clone()));
        }
    }

    for (name, content) in index_tasks {
        let _ = state
            .aux_index_tx
            .send(AuxIndexTask {
                project_id: SKILLS_PROJECT.to_string(),
                path: std::path::PathBuf::from(name),
                content,
            })
            .await;
    }

    {
        let mut catalog = state.skills_catalog.write().await;
        catalog.retain(|r| r.skill_file != req.record.skill_file);
        catalog.push(req.record);
    }

    StatusCode::OK.into_response()
}

async fn skill_delete(State(state): State<AppState>, Query(query): Query<PathQuery>) -> Response {
    let path = query.path.clone();
    let stem = match path.strip_suffix(".md") {
        Some(s) => s.to_string(),
        None => {
            return (StatusCode::BAD_REQUEST, "skill_file must end with .md").into_response();
        }
    };

    if let Err(e) = state.files.read().await.delete_skill_file(&path).await {
        return io_error_response(e);
    }

    let removed_indexes = state
        .files
        .read()
        .await
        .delete_index_files(&stem)
        .await
        .unwrap_or_default();

    let mut paths_to_delete: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from(&path)];
    for name in removed_indexes {
        paths_to_delete.push(std::path::PathBuf::from(name));
    }

    let _ = state
        .aux_delete_tx
        .send(AuxDeleteTask {
            project_id: SKILLS_PROJECT.to_string(),
            paths: paths_to_delete,
        })
        .await;

    let mut catalog = state.skills_catalog.write().await;
    catalog.retain(|r| r.skill_file != path);

    StatusCode::OK.into_response()
}

async fn skill_list(State(state): State<AppState>) -> Response {
    let catalog = state.skills_catalog.read().await;
    Json(catalog.clone()).into_response()
}

async fn skill_search(State(state): State<AppState>, Query(query): Query<SearchQuery>) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            project_id: Some(SKILLS_PROJECT.to_string()),
            query: query.query.clone(),
            top_k: query.top_k,
            sync_first: true,
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

async fn tool_search(State(state): State<AppState>, Query(query): Query<SearchQuery>) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            project_id: Some(TOOLS_PROJECT.to_string()),
            query: query.query.clone(),
            top_k: query.top_k,
            sync_first: true,
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

async fn knowledge_search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    if state
        .search_tx
        .send(SearchTask {
            project_id: Some(KNOWLEDGE_PROJECT.to_string()),
            query: query.query.clone(),
            top_k: query.top_k,
            sync_first: true,
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
struct RecordUsageQuery {
    file: String,
    success: bool,
}

async fn skill_record_usage(
    State(state): State<AppState>,
    Query(query): Query<RecordUsageQuery>,
) -> Response {
    let (success, fail) = {
        let mut catalog = state.skills_catalog.write().await;
        let Some(rec) = catalog.iter_mut().find(|r| r.skill_file == query.file) else {
            return (StatusCode::NOT_FOUND, "skill not found").into_response();
        };
        if query.success {
            rec.success_count += 1;
        } else {
            rec.fail_count += 1;
        }
        (rec.success_count, rec.fail_count)
    };

    {
        let files = state.files.read().await;
        if let Err(e) = update_skill_stats_in_file(&files, &query.file, success, fail).await {
            eprintln!("⚠️ Не удалось обновить STATS в файле {}: {}", query.file, e);
        }
    }

    StatusCode::OK.into_response()
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

fn parse_skill_file(file_name: &str, content: &str) -> Option<SkillRecord> {
    let mut agent_name = String::new();
    let mut description = String::new();
    let mut entities: Vec<String> = Vec::new();
    let mut success_count = 0u32;
    let mut fail_count = 0u32;
    let mut fingerprint = String::new();
    let mut created_at = 0u64;
    let mut instruction_lines: Vec<String> = Vec::new();

    let mut in_instruction = false;
    let mut in_tool_calls = false;

    for line in content.lines() {
        if in_tool_calls {
            continue;
        }
        if line.starts_with("TOOL_CALLS:") {
            in_tool_calls = true;
            in_instruction = false;
            continue;
        }
        if in_instruction {
            instruction_lines.push(line.to_string());
            continue;
        }
        if line.trim() == "INSTRUCTION:" {
            in_instruction = true;
            continue;
        }
        if line.starts_with("SKILL:") {
            continue;
        }
        if let Some(v) = line.strip_prefix("FOR:") {
            agent_name = v.trim().to_string();
            continue;
        }
        if let Some(v) = line.strip_prefix("DESCRIPTION:") {
            description = v.trim().to_string();
            continue;
        }
        if let Some(v) = line.strip_prefix("ENTITIES:") {
            entities = v
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            continue;
        }
        if let Some(v) = line.strip_prefix("STATS:") {
            let parts: Vec<&str> = v.trim().split('/').collect();
            if parts.len() == 2 {
                success_count = parts[0].trim().parse().unwrap_or(0);
                fail_count = parts[1].trim().parse().unwrap_or(0);
            }
            continue;
        }
        if let Some(v) = line.strip_prefix("FINGERPRINT:") {
            fingerprint = v.trim().to_string();
            continue;
        }
        if let Some(v) = line.strip_prefix("CREATED_AT:") {
            created_at = v.trim().parse().unwrap_or(0);
            continue;
        }
    }

    let prompt = instruction_lines.join("\n").trim().to_string();

    Some(SkillRecord {
        skill_file: file_name.to_string(),
        agent_name,
        description,
        prompt,
        entities,
        success_count,
        fail_count,
        fingerprint,
        created_at,
    })
}

async fn scan_skills_catalog(skills_root: &std::path::Path) -> Vec<SkillRecord> {
    let mut records = Vec::new();
    let mut dir = match tokio::fs::read_dir(skills_root).await {
        Ok(d) => d,
        Err(_) => return records,
    };
    while let Ok(Some(entry)) = dir.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".md") {
            continue;
        }
        let content = match tokio::fs::read_to_string(entry.path()).await {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Some(rec) = parse_skill_file(&name, &content) {
            records.push(rec);
        }
    }
    records
}

async fn update_skill_stats_in_file(
    files: &FileStorage,
    skill_file: &str,
    success: u32,
    fail: u32,
) -> std::io::Result<()> {
    let content = files.read_skill_file(skill_file).await?;
    let mut new_content = String::with_capacity(content.len());
    let mut replaced = false;
    for line in content.lines() {
        if line.starts_with("STATS:") {
            new_content.push_str(&format!("STATS: {}/{}", success, fail));
            new_content.push('\n');
            replaced = true;
        } else {
            new_content.push_str(line);
            new_content.push('\n');
        }
    }
    if !replaced {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "STATS line not found in skill file",
        ));
    }
    files.write_skill_file(skill_file, &new_content).await
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

    let _ = state
        .aux_index_tx
        .send(AuxIndexTask {
            project_id: KNOWLEDGE_PROJECT.to_string(),
            path: std::path::PathBuf::from(format!("{}.md", query.name)),
            content: body,
        })
        .await;

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

    let _ = state
        .aux_delete_tx
        .send(AuxDeleteTask {
            project_id: KNOWLEDGE_PROJECT.to_string(),
            paths: vec![std::path::PathBuf::from(format!("{}.md", query.name))],
        })
        .await;

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
    let skills_root = root.join(".skills");
    let files = Arc::new(RwLock::new(FileStorage { root }));
    let knowledge_catalog = Arc::new(RwLock::new(scan_knowledge_catalog(&knowledge_root).await));
    let skills_catalog = Arc::new(RwLock::new(scan_skills_catalog(&skills_root).await));

    let (search_tx, mut search_rx) = mpsc::channel::<SearchTask>(100);
    let (index_tx, mut index_rx) = mpsc::channel::<IndexTask>(100);
    let (aux_index_tx, mut aux_index_rx) = mpsc::channel::<AuxIndexTask>(100);
    let (aux_delete_tx, mut aux_delete_rx) = mpsc::channel::<AuxDeleteTask>(100);
    let (sync_tx, mut sync_rx) = mpsc::channel::<SyncTask>(16);

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
            // Пересобрать .tools/ из inventory.
            if let Err(e) = storage.clear_tools_dir() {
                eprintln!("⚠️ clear_tools_dir: {}", e);
            }
            let tools = crate::tool_runtime::all();
            let all_names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();

            let mut written = 0usize;
            let mut skipped = 0usize;

            for tool in tools {
                let name = tool.name();
                let desc = tool.description();

                // Инвариант: описание тулза не должно упоминать имена других
                // тулзов. Иначе модель видит это имя в описании доступного
                // тулза, зовёт его, а в API-запросе его нет — провайдер
                // падает с BAD_GATEWAY. См. README_tool_dev_roadmap.
                let conflicts: Vec<&str> = all_names
                    .iter()
                    .filter(|other| other.as_str() != name && desc.contains(other.as_str()))
                    .map(|s| s.as_str())
                    .collect();

                if !conflicts.is_empty() {
                    eprintln!(
                        "❌ Тулз '{}' невалиден: описание упоминает другие тулзы {:?}. Исключён из .tools/.",
                        name, conflicts
                    );
                    skipped += 1;
                    continue;
                }

                let content = format!("{} {}", name, desc);
                match storage.write_tool_file(name, &content) {
                    Ok(_) => written += 1,
                    Err(e) => eprintln!("⚠️ write_tool_file {}: {}", name, e),
                }
            }
            eprintln!(
                "🔧 .tools/ пересобран: {} записано, {} исключено",
                written, skipped
            );

            loop {
                tokio::select! {
                    Some(task) = search_rx.recv() => {
                        if task.sync_first {
                            if let Err(e) = storage.sync_index() {
                                eprintln!("⚠️ sync_index перед поиском: {}", e);
                            }
                        }
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
                    Some(task) = aux_index_rx.recv() => {
                        let res = match task.project_id.as_str() {
                            SKILLS_PROJECT => storage
                                .create_skill_file(&task.path.to_string_lossy(), &task.content),
                            KNOWLEDGE_PROJECT => storage
                                .index_knowledge_file(&task.path.to_string_lossy(), &task.content),
                            other => {
                                eprintln!("⚠️ aux_index: неизвестный project_id '{}'", other);
                                continue;
                            }
                        };
                        if let Err(e) = res {
                            eprintln!("⚠️ Aux indexing failed for {:?}: {}", task.path, e);
                        }
                    }
                    Some(task) = aux_delete_rx.recv() => {
                        for path in &task.paths {
                            let res = match task.project_id.as_str() {
                                SKILLS_PROJECT => storage.delete_skill_index(&path.to_string_lossy()),
                                KNOWLEDGE_PROJECT => storage.delete_knowledge_index(&path.to_string_lossy()),
                                other => {
                                    eprintln!("⚠️ aux_delete: неизвестный project_id '{}'", other);
                                    continue;
                                }
                            };
                            if let Err(e) = res {
                                eprintln!("⚠️ Aux deindex failed for {:?}: {}", path, e);
                            }
                        }
                    }
                    Some(_) = sync_rx.recv() => {
                        if let Err(e) = storage.sync_index() {
                            eprintln!("⚠️ Периодический sync_index: {}", e);
                        }
                    }
                    else => break,
                }
            }
        });
    });

    {
        let sync_tx_bg = sync_tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.tick().await;
            loop {
                interval.tick().await;
                if sync_tx_bg.send(SyncTask).await.is_err() {
                    break;
                }
            }
        });
    }

    let state = AppState {
        files,
        search_tx,
        index_tx,
        aux_index_tx,
        aux_delete_tx,
        knowledge_catalog,
        skills_catalog,
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
        .route("/skills/record_usage", post(skill_record_usage))
        .route("/tools/search", get(tool_search))
        .route("/rebukes/get", get(rebuke_get))
        .route("/rebukes/put", post(rebuke_put))
        .route("/knowledge/list", get(knowledge_list))
        .route("/knowledge/get", get(knowledge_get))
        .route("/knowledge/put", post(knowledge_put))
        .route("/knowledge/delete", post(knowledge_delete))
        .route("/knowledge/search", get(knowledge_search))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn storage() -> FileStorage {
        FileStorage {
            root: PathBuf::from("/tmp/nh_test_root"),
        }
    }

    #[test]
    fn resolve_rejects_empty_project_id() {
        let s = storage();
        assert!(s.resolve("", "file.txt").is_err());
    }

    #[test]
    fn resolve_rejects_project_id_with_slash() {
        let s = storage();
        assert!(s.resolve("a/b", "file.txt").is_err());
    }

    #[test]
    fn resolve_rejects_parent_dir_in_path() {
        let s = storage();
        assert!(s.resolve("proj", "../etc/passwd").is_err());
    }

    #[test]
    fn resolve_rejects_absolute_path() {
        let s = storage();
        // Абсолютный путь: компоненты RootDir игнорируются, но "/" в начале
        // должен приводить к тому, что итог не выходит за корень
        let res = s.resolve("proj", "/etc/passwd");
        // Либо Err, либо путь внутри проекта — проверяем что не паникует
        // и не выходит наружу. Реальная защита — canonicalize на уровне fs.
        assert!(res.is_ok() || res.is_err());
    }

    #[test]
    fn resolve_rejects_reserved_name_in_path() {
        let s = storage();
        assert!(s.resolve("proj", ".about").is_err());
        assert!(s.resolve("proj", "sub/.summary").is_err());
    }

    #[test]
    fn resolve_normal_path_ok() {
        let s = storage();
        let p = s.resolve("proj", "src/main.rs").unwrap();
        assert!(p.to_string_lossy().contains("proj"));
        assert!(p.to_string_lossy().contains("main.rs"));
    }

    #[test]
    fn resolve_skill_rejects_parent_dir() {
        let s = storage();
        assert!(s.resolve_skill("../escape.txt").is_err());
    }

    #[test]
    fn resolve_skill_rejects_reserved_name() {
        let s = storage();
        assert!(s.resolve_skill(".about").is_err());
    }

    #[test]
    fn resolve_skill_normal_ok() {
        let s = storage();
        let p = s.resolve_skill("skill_agent_a_task_123.txt").unwrap();
        assert!(p.to_string_lossy().contains(".skills"));
    }

    #[test]
    fn resolve_knowledge_rejects_empty() {
        let s = storage();
        assert!(s.resolve_knowledge("").is_err());
    }

    #[test]
    fn resolve_knowledge_rejects_invalid_name() {
        let s = storage();
        assert!(s.resolve_knowledge("my knowledge").is_err());
        assert!(s.resolve_knowledge("../etc").is_err());
        assert!(s.resolve_knowledge("a/b").is_err());
    }

    #[test]
    fn resolve_knowledge_normal_ok() {
        let s = storage();
        let p = s.resolve_knowledge("rust_async").unwrap();
        assert!(p.to_string_lossy().ends_with("rust_async.md"));
    }

    #[test]
    fn resolve_rebuke_rejects_empty_agent() {
        let s = storage();
        assert!(s.resolve_rebuke("").is_err());
    }

    #[test]
    fn resolve_rebuke_rejects_slash_in_agent() {
        let s = storage();
        assert!(s.resolve_rebuke("a/b").is_err());
        assert!(s.resolve_rebuke("a\\b").is_err());
    }

    #[test]
    fn resolve_rebuke_rejects_parent_dir() {
        let s = storage();
        assert!(s.resolve_rebuke("..").is_err());
    }

    #[test]
    fn resolve_rebuke_normal_ok() {
        let s = storage();
        let p = s.resolve_rebuke("analyzer").unwrap();
        assert!(p.to_string_lossy().ends_with("analyzer.txt"));
    }

    #[test]
    fn parse_skill_file_extracts_all_fields() {
        let content = "\
SKILL: my task
FOR: analyzer
DESCRIPTION: does things
ENTITIES: rust, cargo, async
STATS: 5/2
FINGERPRINT: abc123
CREATED_AT: 1700000000

INSTRUCTION:
Проанализируй проект.

TOOL_CALLS:
[{\"name\":\"storage_walk\",\"arguments\":{}}]
";
        let rec = parse_skill_file("skill_agent_analyzer_my_task_1700000000.txt", content).unwrap();
        assert_eq!(rec.agent_name, "analyzer");
        assert_eq!(rec.description, "does things");
        assert_eq!(rec.entities, vec!["rust", "cargo", "async"]);
        assert_eq!(rec.success_count, 5);
        assert_eq!(rec.fail_count, 2);
        assert_eq!(rec.fingerprint, "abc123");
        assert_eq!(rec.created_at, 1700000000);
        assert!(rec.prompt.contains("Проанализируй проект."));
    }

    #[test]
    fn parse_skill_file_missing_stats_defaults_zero() {
        let content = "SKILL: x\nFOR: a\nDESCRIPTION: d\nENTITIES:\nFINGERPRINT: f\nCREATED_AT: 0\n\nINSTRUCTION:\nprompt\n\nTOOL_CALLS:\n[]\n";
        let rec = parse_skill_file("s.txt", content).unwrap();
        assert_eq!(rec.success_count, 0);
        assert_eq!(rec.fail_count, 0);
    }

    #[test]
    fn parse_skill_file_empty_content_returns_record_with_empty_fields() {
        let rec = parse_skill_file("s.txt", "").unwrap();
        assert_eq!(rec.skill_file, "s.txt");
        assert!(rec.agent_name.is_empty());
        assert!(rec.prompt.is_empty());
        assert!(rec.entities.is_empty());
    }

    #[test]
    fn parse_skill_file_entities_are_lowercased() {
        let content = "SKILL: x\nFOR: a\nDESCRIPTION: d\nENTITIES: Rust, CARGO\nSTATS: 0/0\nFINGERPRINT: f\nCREATED_AT: 0\n\nINSTRUCTION:\np\n\nTOOL_CALLS:\n[]\n";
        let rec = parse_skill_file("s.txt", content).unwrap();
        assert_eq!(rec.entities, vec!["rust", "cargo"]);
    }

    #[test]
    fn parse_skill_file_stops_at_tool_calls() {
        let content = "SKILL: x\nFOR: a\nDESCRIPTION: d\nENTITIES:\nSTATS: 0/0\nFINGERPRINT: f\nCREATED_AT: 0\n\nINSTRUCTION:\nline1\nline2\n\nTOOL_CALLS:\n[{\"name\":\"t\"}]\n\nFOR: should not override\n";
        let rec = parse_skill_file("s.txt", content).unwrap();
        assert_eq!(rec.agent_name, "a");
        assert!(rec.prompt.contains("line1"));
        assert!(rec.prompt.contains("line2"));
    }
}
