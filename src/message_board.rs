// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/message_board.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::sse::{Event as SseEvent, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

// ==================== Конфиг ====================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MessageBoardConfig {
    pub bind_addr: String,
    pub auth_token: String,
    pub tasks_dir: String,
    #[serde(default = "default_task_timeout_sec")]
    pub task_timeout_sec: u64,
}

fn default_task_timeout_sec() -> u64 {
    300
}

// ==================== Типы данных ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub from_agent: String,
    pub from_session_id: String,
    pub to_agent: String,
    pub to_session_id: String,
    #[serde(default)]
    pub project_id: String,
    pub payload: Value,
    pub status: TaskStatus,
    pub parent_task_id: Option<String>,
    /// Цепочка агентов, участвовавших в вызове (от корня до текущего).
    #[serde(default)]
    pub chain: Vec<String>,
    pub result: Option<String>,
    pub error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BoardEvent {
    TaskCreated {
        task: Task,
    },
    TaskCompleted {
        task_id: String,
        result: String,
        from_agent: String,
        from_session_id: String,
    },
    TaskFailed {
        task_id: String,
        error: String,
        from_agent: String,
        from_session_id: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct RegisterSessionRequest {
    pub agent_name: String,
}

#[derive(Debug, Deserialize)]
pub struct UnregisterSessionRequest {
    pub agent_name: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub from_agent: String,
    pub from_session_id: String,
    pub to_agent: String,
    pub to_session_id: String,
    pub project_id: String,
    pub payload: Value,
    pub parent_task_id: Option<String>,
    #[serde(default)]
    pub chain: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateTaskResponse {
    pub task_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CompleteTaskRequest {
    pub result: String,
}

#[derive(Debug, Deserialize)]
pub struct FailTaskRequest {
    pub error: String,
}

// ==================== Команды актору ====================

enum BoardCommand {
    RegisterSession {
        agent_name: String,
        sender: mpsc::UnboundedSender<BoardEvent>,
    },
    UnregisterSession {
        agent_name: String,
    },
    CreateTask {
        request: CreateTaskRequest,
        response_tx: oneshot::Sender<anyhow::Result<CreateTaskResponse>>,
    },
    CompleteTask {
        task_id: String,
        result: String,
        response_tx: oneshot::Sender<anyhow::Result<()>>,
    },
    FailTask {
        task_id: String,
        error: String,
        response_tx: oneshot::Sender<anyhow::Result<()>>,
    },
    GetTask {
        task_id: String,
        response_tx: oneshot::Sender<Option<Task>>,
    },
    ListTasksFiltered {
        session_id: Option<String>,
        from_agent: Option<String>,
        statuses: Option<Vec<TaskStatus>>,
        response_tx: oneshot::Sender<Vec<Task>>,
    },
}

// ==================== Актор доски ====================

#[allow(dead_code)]
pub struct MessageBoard {
    command_tx: mpsc::UnboundedSender<BoardCommand>,
    tasks_dir: PathBuf,
}

impl MessageBoard {
    pub async fn new(tasks_dir: PathBuf, task_timeout_sec: u64) -> Result<Self> {
        tokio::fs::create_dir_all(&tasks_dir).await?;
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<BoardCommand>();

        let tasks_dir_actor = tasks_dir.clone();

        let mut tasks: HashMap<String, Task> = HashMap::new();
        let mut connections: HashMap<String, mpsc::UnboundedSender<BoardEvent>> = HashMap::new();

        let mut entries = tokio::fs::read_dir(&tasks_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let content = match tokio::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let mut task: Task = match serde_json::from_str(&content) {
                Ok(t) => t,
                Err(_) => continue,
            };
            match task.status {
                TaskStatus::Pending => {
                    let _ = tokio::fs::remove_file(&path).await;
                    continue;
                }
                TaskStatus::InProgress => {
                    task.status = TaskStatus::Failed;
                    task.error = Some("executor lost (server restart)".to_string());
                    task.updated_at = now_ts();
                    if let Err(e) = save_task(&tasks_dir, &task).await {
                        eprintln!("Ошибка сохранения задачи {}: {}", task.id, e);
                    }
                }
                TaskStatus::Completed | TaskStatus::Failed => {
                    let age = now_ts().saturating_sub(task.updated_at);
                    if age > task_timeout_sec {
                        let _ = tokio::fs::remove_file(&path).await;
                        continue;
                    }
                }
            }
            tasks.insert(task.id.clone(), task);
        }

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.tick().await;

            loop {
                tokio::select! {
                    maybe_cmd = command_rx.recv() => {
                        let Some(cmd) = maybe_cmd else { break };
                        match cmd {
                            BoardCommand::RegisterSession { agent_name, sender } => {
                                connections.insert(agent_name, sender);
                            }
                            BoardCommand::UnregisterSession { agent_name } => {
                                connections.remove(&agent_name);
                            }
                            BoardCommand::CreateTask { request, response_tx } => {
                                if !crate::local_storage::valid_project_id(&request.project_id) {
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "project_id is required and must be non-empty"
                                    )));
                                    continue;
                                }
                                match request.payload.get("prompt").and_then(|v| v.as_str()) {
                                    Some(s) if !s.trim().is_empty() => {}
                                    _ => {
                                        let _ = response_tx.send(Err(anyhow::anyhow!(
                                            "payload.prompt is required and must be non-empty"
                                        )));
                                        continue;
                                    }
                                }

                                let Some(conn) = connections.get(&request.to_agent).cloned() else {
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "Agent '{}' is not connected",
                                        request.to_agent
                                    )));
                                    continue;
                                };

                                let task_id = uuid::Uuid::new_v4().to_string();
                                let now = now_ts();
                                let task = Task {
                                    id: task_id.clone(),
                                    from_agent: request.from_agent.clone(),
                                    from_session_id: request.from_session_id.clone(),
                                    to_agent: request.to_agent.clone(),
                                    to_session_id: request.to_session_id.clone(),
                                    project_id: request.project_id.clone(),
                                    payload: request.payload.clone(),
                                    status: TaskStatus::InProgress,
                                    parent_task_id: request.parent_task_id.clone(),
                                    chain: request.chain.clone(),
                                    result: None,
                                    error: None,
                                    created_at: now,
                                    updated_at: now,
                                    attempts: 0,
                                };

                                if let Err(e) = save_task(&tasks_dir_actor, &task).await {
                                    eprintln!("Ошибка сохранения задачи {}: {}", task.id, e);
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "Failed to save task: {}", e
                                    )));
                                    continue;
                                }

                                if let Err(e) = conn.send(BoardEvent::TaskCreated { task: task.clone() }) {
                                    eprintln!("Ошибка отправки TaskCreated для {}: {}", task.id, e);
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "Failed to deliver task: {}", e
                                    )));
                                    continue;
                                }

                                tasks.insert(task_id.clone(), task);
                                let _ = response_tx.send(Ok(CreateTaskResponse { task_id }));
                            }
                            BoardCommand::CompleteTask { task_id, result, response_tx } => {
                                if let Some(task) = tasks.get_mut(&task_id) {
                                    task.status = TaskStatus::Completed;
                                    task.result = Some(result.clone());
                                    task.updated_at = now_ts();
                                    if let Err(e) = save_task(&tasks_dir_actor, task).await {
                                        eprintln!("Ошибка сохранения задачи {}: {}", task.id, e);
                                    }
                                    let key = task.from_agent.clone();
                                    if let Some(conn) = connections.get(&key) {
                                        if let Err(e) = conn.send(BoardEvent::TaskCompleted {
                                            task_id: task_id.clone(),
                                            result,
                                            from_agent: task.to_agent.clone(),
                                            from_session_id: task.to_session_id.clone(),
                                        }) {
                                            eprintln!("Ошибка отправки TaskCompleted для {}: {}", task_id, e);
                                        }
                                    }
                                    let _ = response_tx.send(Ok(()));
                                } else {
                                    eprintln!("Задача {} не найдена для завершения", task_id);
                                    let _ = response_tx.send(Err(anyhow::anyhow!("Task not found")));
                                }
                            }
                            BoardCommand::FailTask { task_id, error, response_tx } => {
                                if let Some(task) = tasks.get_mut(&task_id) {
                                    task.status = TaskStatus::Failed;
                                    task.error = Some(error.clone());
                                    task.updated_at = now_ts();
                                    if let Err(e) = save_task(&tasks_dir_actor, task).await {
                                        eprintln!("Ошибка сохранения задачи {}: {}", task.id, e);
                                    }
                                    let key = task.from_agent.clone();
                                    if let Some(conn) = connections.get(&key) {
                                        if let Err(e) = conn.send(BoardEvent::TaskFailed {
                                            task_id: task_id.clone(),
                                            error,
                                            from_agent: task.to_agent.clone(),
                                            from_session_id: task.to_session_id.clone(),
                                        }) {
                                            eprintln!("Ошибка отправки TaskFailed для {}: {}", task_id, e);
                                        }
                                    }
                                    let _ = response_tx.send(Ok(()));
                                } else {
                                    eprintln!("Задача {} не найдена для ошибки", task_id);
                                    let _ = response_tx.send(Err(anyhow::anyhow!("Task not found")));
                                }
                            }
                            BoardCommand::GetTask { task_id, response_tx } => {
                                if let Err(e) = response_tx.send(tasks.get(&task_id).cloned()) {
                                    eprintln!("Ошибка отправки ответа get_task для {}: {:?}", task_id, e);
                                }
                            }
                            BoardCommand::ListTasksFiltered {
                                session_id,
                                from_agent,
                                statuses,
                                response_tx,
                            } => {
                                let filtered: Vec<Task> = tasks
                                    .values()
                                    .filter(|t| {
                                        let session_match = match &session_id {
                                            Some(sid) => {
                                                t.from_session_id == *sid || t.to_session_id == *sid
                                            }
                                            None => true,
                                        };
                                        let agent_match = match &from_agent {
                                            Some(a) => t.from_agent == *a,
                                            None => true,
                                        };
                                        let status_match = match &statuses {
                                            Some(sts) => sts.contains(&t.status),
                                            None => true,
                                        };
                                        session_match && agent_match && status_match
                                    })
                                    .cloned()
                                    .collect();
                                if let Err(e) = response_tx.send(filtered) {
                                    eprintln!("Ошибка отправки ответа list_tasks_filtered: {:?}", e);
                                }
                            }
                        }
                    }
                    _ = interval.tick() => {
                        let now = now_ts();
                        let mut to_fail: Vec<String> = Vec::new();
                        for (id, task) in tasks.iter() {
                            if task.status == TaskStatus::InProgress
                                && now.saturating_sub(task.updated_at) > task_timeout_sec
                            {
                                to_fail.push(id.clone());
                            }
                        }
                        for id in to_fail {
                            if let Some(task) = tasks.get_mut(&id) {
                                task.status = TaskStatus::Failed;
                                task.error = Some(format!(
                                    "task timeout: no update for {}s", task_timeout_sec
                                ));
                                task.updated_at = now_ts();
                                if let Err(e) = save_task(&tasks_dir_actor, task).await {
                                    eprintln!("Ошибка сохранения задачи {}: {}", task.id, e);
                                }
                                let key = task.from_agent.clone();
                                if let Some(conn) = connections.get(&key) {
                                    if let Err(e) = conn.send(BoardEvent::TaskFailed {
                                        task_id: id.clone(),
                                        error: task.error.clone().unwrap_or_default(),
                                        from_agent: task.to_agent.clone(),
                                        from_session_id: task.to_session_id.clone(),
                                    }) {
                                        eprintln!("Ошибка отправки TaskFailed для {}: {}", id, e);
                                    }
                                }
                                eprintln!("⏱️ Задача {} помечена Failed (таймаут)", id);
                            }
                        }

                        let mut to_evict: Vec<String> = Vec::new();
                        for (id, task) in tasks.iter() {
                            let terminal = matches!(
                                task.status,
                                TaskStatus::Completed | TaskStatus::Failed
                            );
                            if terminal
                                && now.saturating_sub(task.updated_at) > task_timeout_sec
                            {
                                to_evict.push(id.clone());
                            }
                        }
                        for id in &to_evict {
                            tasks.remove(id);
                            let file = tasks_dir_actor.join(format!("{}.json", id));
                            if let Err(e) = tokio::fs::remove_file(&file).await {
                                if e.kind() != std::io::ErrorKind::NotFound {
                                    eprintln!("Очистка файла задачи {}: {}", id, e);
                                }
                            }
                        }
                        if !to_evict.is_empty() {
                            eprintln!("🧹 Очищено {} завершённых задач", to_evict.len());
                        }
                    }
                }
            }
        });

        Ok(Self {
            command_tx,
            tasks_dir,
        })
    }

    pub async fn register_session(
        &self,
        agent_name: String,
        sender: mpsc::UnboundedSender<BoardEvent>,
    ) {
        if let Err(e) = self
            .command_tx
            .send(BoardCommand::RegisterSession { agent_name, sender })
        {
            eprintln!("Ошибка отправки команды RegisterSession: {}", e);
        }
    }

    pub async fn unregister_session(&self, agent_name: String) {
        if let Err(e) = self
            .command_tx
            .send(BoardCommand::UnregisterSession { agent_name })
        {
            eprintln!("Ошибка отправки команды UnregisterSession: {}", e);
        }
    }

    pub async fn create_task(&self, request: CreateTaskRequest) -> Result<CreateTaskResponse> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self.command_tx.send(BoardCommand::CreateTask {
            request,
            response_tx: tx,
        }) {
            eprintln!("Ошибка отправки команды CreateTask: {}", e);
            return Err(anyhow::anyhow!("Failed to send CreateTask command: {}", e));
        }
        match rx.await {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Ошибка получения ответа CreateTask: {}", e);
                Err(anyhow::anyhow!("CreateTask channel closed: {}", e))
            }
        }
    }

    pub async fn complete_task(&self, task_id: String, result: String) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self.command_tx.send(BoardCommand::CompleteTask {
            task_id,
            result,
            response_tx: tx,
        }) {
            eprintln!("Ошибка отправки команды CompleteTask: {}", e);
            return Err(anyhow::anyhow!(
                "Failed to send CompleteTask command: {}",
                e
            ));
        }
        match rx.await {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Ошибка получения ответа CompleteTask: {}", e);
                Err(anyhow::anyhow!("CompleteTask channel closed: {}", e))
            }
        }
    }

    pub async fn fail_task(&self, task_id: String, error: String) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self.command_tx.send(BoardCommand::FailTask {
            task_id,
            error,
            response_tx: tx,
        }) {
            eprintln!("Ошибка отправки команды FailTask: {}", e);
            return Err(anyhow::anyhow!("Failed to send FailTask command: {}", e));
        }
        match rx.await {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Ошибка получения ответа FailTask: {}", e);
                Err(anyhow::anyhow!("FailTask channel closed: {}", e))
            }
        }
    }

    pub async fn get_task(&self, task_id: String) -> Option<Task> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self.command_tx.send(BoardCommand::GetTask {
            task_id,
            response_tx: tx,
        }) {
            eprintln!("Ошибка отправки команды GetTask: {}", e);
            return None;
        }
        match rx.await {
            Ok(task) => task,
            Err(e) => {
                eprintln!("Ошибка получения ответа GetTask: {}", e);
                None
            }
        }
    }

    pub async fn list_tasks_filtered(
        &self,
        session_id: Option<String>,
        from_agent: Option<String>,
        statuses: Option<Vec<TaskStatus>>,
    ) -> Vec<Task> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self.command_tx.send(BoardCommand::ListTasksFiltered {
            session_id,
            from_agent,
            statuses,
            response_tx: tx,
        }) {
            eprintln!("Ошибка отправки команды ListTasksFiltered: {}", e);
            return vec![];
        }
        match rx.await {
            Ok(tasks) => tasks,
            Err(e) => {
                eprintln!("Ошибка получения ответа ListTasksFiltered: {}", e);
                vec![]
            }
        }
    }
}

// ==================== Утилиты ====================

fn now_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

async fn save_task(tasks_dir: &PathBuf, task: &Task) -> Result<()> {
    let path = tasks_dir.join(format!("{}.json", task.id));
    let content = serde_json::to_string_pretty(task)?;
    tokio::fs::write(path, content).await?;
    Ok(())
}

// ==================== HTTP API и SSE ====================

#[derive(Clone)]
#[allow(dead_code)]
struct AppState {
    board: std::sync::Arc<MessageBoard>,
    auth_token: String,
}

#[derive(Deserialize)]
struct EventsQuery {
    agent_name: String,
}

#[derive(Deserialize)]
struct UnregisterQuery {
    agent_name: String,
}

#[derive(Deserialize)]
struct ListTasksQuery {
    session_id: Option<String>,
    from_agent: Option<String>,
    status: Option<String>,
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

pub fn build_router(board: std::sync::Arc<MessageBoard>, auth_token: String) -> Router {
    let state = AppState { board, auth_token };
    Router::new()
        .route("/register_session", post(register_session))
        .route("/unregister_session", post(unregister_session))
        .route("/tasks", post(create_task).get(list_tasks))
        .route("/tasks/{id}", get(get_task))
        .route("/tasks/{id}/complete", post(complete_task))
        .route("/tasks/{id}/fail", post(fail_task))
        .route("/events", get(sse_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}

async fn register_session(
    State(_state): State<AppState>,
    Json(_req): Json<RegisterSessionRequest>,
) -> impl IntoResponse {
    eprintln!("Получен запрос на /register_session");
    (axum::http::StatusCode::OK, "Registered")
}

async fn unregister_session(
    State(state): State<AppState>,
    Query(query): Query<UnregisterQuery>,
) -> impl IntoResponse {
    eprintln!("Запрос на /unregister_session: agent={}", query.agent_name);
    state.board.unregister_session(query.agent_name).await;
    (axum::http::StatusCode::OK, "Unregistered")
}

async fn create_task(
    State(state): State<AppState>,
    Json(req): Json<CreateTaskRequest>,
) -> axum::response::Response {
    match state.board.create_task(req).await {
        Ok(resp) => {
            eprintln!("Задача создана: {}", resp.task_id);
            (axum::http::StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            eprintln!("Ошибка создания задачи: {}", e);
            (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn list_tasks(
    State(state): State<AppState>,
    Query(query): Query<ListTasksQuery>,
) -> axum::response::Response {
    let statuses = query.status.as_ref().map(|s| {
        s.split(',')
            .filter_map(|p| match p.trim() {
                "pending" => Some(TaskStatus::Pending),
                "in_progress" => Some(TaskStatus::InProgress),
                "completed" => Some(TaskStatus::Completed),
                "failed" => Some(TaskStatus::Failed),
                _ => None,
            })
            .collect::<Vec<_>>()
    });
    let tasks = state
        .board
        .list_tasks_filtered(query.session_id, query.from_agent, statuses)
        .await;
    (axum::http::StatusCode::OK, Json(tasks)).into_response()
}

async fn get_task(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
) -> axum::response::Response {
    match state.board.get_task(task_id).await {
        Some(task) => (axum::http::StatusCode::OK, Json(task)).into_response(),
        None => (axum::http::StatusCode::NOT_FOUND, "Task not found").into_response(),
    }
}

async fn complete_task(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
    Json(req): Json<CompleteTaskRequest>,
) -> axum::response::Response {
    match state.board.complete_task(task_id.clone(), req.result).await {
        Ok(()) => {
            eprintln!("Задача {} завершена", task_id);
            (axum::http::StatusCode::OK, "Completed").into_response()
        }
        Err(e) => {
            eprintln!("Ошибка завершения задачи {}: {}", task_id, e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn fail_task(
    State(state): State<AppState>,
    Path(task_id): Path<String>,
    Json(req): Json<FailTaskRequest>,
) -> axum::response::Response {
    match state.board.fail_task(task_id.clone(), req.error).await {
        Ok(()) => {
            eprintln!("Задача {} помечена как ошибочная", task_id);
            (axum::http::StatusCode::OK, "Failed").into_response()
        }
        Err(e) => {
            eprintln!("Ошибка обработки fail_task {}: {}", task_id, e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn sse_handler(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Sse<impl tokio_stream::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<BoardEvent>();
    state
        .board
        .register_session(query.agent_name.clone(), tx)
        .await;
    eprintln!("Новое SSE-подключение: agent={}", query.agent_name);

    let stream = UnboundedReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event).unwrap_or_default();
        Ok(SseEvent::default().data(data))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ==================== Тесты ====================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_create_task_rejected_for_offline_agent() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            project_id: "test_project".into(),
            payload: serde_json::json!({"prompt": "hi"}),
            parent_task_id: None,
            chain: vec!["b".into()],
        };
        assert!(board.create_task(req).await.is_err());
    }

    #[tokio::test]
    async fn test_create_task_delivered_when_online() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx).await;

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            project_id: "test_project".into(),
            payload: serde_json::json!({"prompt": "hi"}),
            parent_task_id: None,
            chain: vec!["b".into()],
        };
        let resp = board.create_task(req).await.unwrap();
        let event = rx.recv().await.unwrap();
        match event {
            BoardEvent::TaskCreated { task } => {
                assert_eq!(task.id, resp.task_id);
                assert_eq!(task.chain, vec!["b".to_string()]);
                assert_eq!(task.project_id, "test_project");
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_create_task_rejected_for_missing_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx).await;

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            project_id: "test_project".into(),
            payload: serde_json::json!({}),
            parent_task_id: None,
            chain: vec![],
        };
        assert!(board.create_task(req).await.is_err());
    }

    #[tokio::test]
    async fn test_create_task_rejected_for_invalid_project_id() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx).await;

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            project_id: "".into(),
            payload: serde_json::json!({"prompt": "hi"}),
            parent_task_id: None,
            chain: vec![],
        };
        assert!(board.create_task(req).await.is_err());
    }

    #[tokio::test]
    async fn test_complete_task_sends_event() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx_sender, mut rx_sender) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("a".to_string(), tx_sender).await;
        let (tx_receiver, mut rx_receiver) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx_receiver).await;

        let resp = board
            .create_task(CreateTaskRequest {
                from_agent: "a".into(),
                from_session_id: "s".into(),
                to_agent: "b".into(),
                to_session_id: "s2".into(),
                project_id: "test_project".into(),
                payload: serde_json::json!({"prompt": "hi"}),
                parent_task_id: None,
                chain: vec!["b".into()],
            })
            .await
            .unwrap();

        let _ = rx_receiver.recv().await.unwrap();
        board
            .complete_task(resp.task_id.clone(), "ok".into())
            .await
            .unwrap();

        let event = rx_sender.recv().await.unwrap();
        match event {
            BoardEvent::TaskCompleted {
                task_id,
                result,
                from_agent,
                ..
            } => {
                assert_eq!(task_id, resp.task_id);
                assert_eq!(result, "ok");
                assert_eq!(from_agent, "b");
            }
            _ => panic!("expected TaskCompleted"),
        }
    }

    #[tokio::test]
    async fn test_fail_task_sends_event() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx_sender, mut rx_sender) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("a".to_string(), tx_sender).await;
        let (tx_receiver, mut rx_receiver) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx_receiver).await;

        let resp = board
            .create_task(CreateTaskRequest {
                from_agent: "a".into(),
                from_session_id: "s".into(),
                to_agent: "b".into(),
                to_session_id: "s2".into(),
                project_id: "test_project".into(),
                payload: serde_json::json!({"prompt": "hi"}),
                parent_task_id: None,
                chain: vec!["b".into()],
            })
            .await
            .unwrap();

        let _ = rx_receiver.recv().await.unwrap();
        board
            .fail_task(resp.task_id.clone(), "err".into())
            .await
            .unwrap();

        let event = rx_sender.recv().await.unwrap();
        match event {
            BoardEvent::TaskFailed {
                task_id,
                error,
                from_agent,
                ..
            } => {
                assert_eq!(task_id, resp.task_id);
                assert_eq!(error, "err");
                assert_eq!(from_agent, "b");
            }
            _ => panic!("expected TaskFailed"),
        }
    }

    #[tokio::test]
    async fn test_get_task() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel::<BoardEvent>();
        board.register_session("b".to_string(), tx).await;

        let resp = board
            .create_task(CreateTaskRequest {
                from_agent: "a".into(),
                from_session_id: "s".into(),
                to_agent: "b".into(),
                to_session_id: "s2".into(),
                project_id: "test_project".into(),
                payload: serde_json::json!({"prompt": "hi"}),
                parent_task_id: None,
                chain: vec!["b".into()],
            })
            .await
            .unwrap();

        let task = board.get_task(resp.task_id.clone()).await.unwrap();
        assert_eq!(task.id, resp.task_id);
        assert_eq!(task.status, TaskStatus::InProgress);
        assert_eq!(task.project_id, "test_project");
    }
}
