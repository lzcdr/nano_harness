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
    response::sse::{Event as SseEvent, KeepAlive, Sse},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};

// ==================== Конфиг ====================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MessageBoardConfig {
    pub bind_addr: String,
    pub auth_token: String,
    pub tasks_dir: String,
    /// Через сколько секунд без обновления задача InProgress помечается Failed.
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
    pub payload: Value,
    pub status: TaskStatus,
    pub parent_task_id: Option<String>,
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
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct UnregisterSessionRequest {
    pub agent_name: String,
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub from_agent: String,
    pub from_session_id: String,
    pub to_agent: String,
    pub to_session_id: String,
    pub payload: Value,
    pub parent_task_id: Option<String>,
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
        session_id: String,
        sender: mpsc::UnboundedSender<BoardEvent>,
    },
    UnregisterSession {
        agent_name: String,
        session_id: String,
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
    ListTasks {
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
        let mut connections: HashMap<(String, String), mpsc::UnboundedSender<BoardEvent>> =
            HashMap::new();

        // Загрузка задач с диска.
        // - Pending: не поддерживается новой моделью — удаляем файл.
        // - InProgress: исполнитель был потерян при рестарте — помечаем Failed.
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
                _ => {}
            }
            tasks.insert(task.id.clone(), task);
        }

        // Актор
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.tick().await; // пропустить немедленный первый тик

            loop {
                tokio::select! {
                    maybe_cmd = command_rx.recv() => {
                        let Some(cmd) = maybe_cmd else { break };
                        match cmd {
                            BoardCommand::RegisterSession { agent_name, session_id, sender } => {
                                let key = (agent_name, session_id);
                                connections.insert(key, sender);
                            }
                            BoardCommand::UnregisterSession { agent_name, session_id } => {
                                connections.remove(&(agent_name, session_id));
                            }
                            BoardCommand::CreateTask { request, response_tx } => {
                                match request.payload.get("prompt").and_then(|v| v.as_str()) {
                                    Some(s) if !s.trim().is_empty() => {}
                                    _ => {
                                        let _ = response_tx.send(Err(anyhow::anyhow!(
                                            "payload.prompt is required and must be non-empty"
                                        )));
                                        continue;
                                    }
                                }

                                let key = (request.to_agent.clone(), request.to_session_id.clone());

                                // Отклоняем, если получатель не подключён.
                                let Some(conn) = connections.get(&key).cloned() else {
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "Session (agent='{}', session_id='{}') is not connected",
                                        request.to_agent,
                                        request.to_session_id
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
                                    payload: request.payload.clone(),
                                    status: TaskStatus::InProgress,
                                    parent_task_id: request.parent_task_id.clone(),
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
                                    let key = (task.from_agent.clone(), task.from_session_id.clone());
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
                                    let key = (task.from_agent.clone(), task.from_session_id.clone());
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
                            BoardCommand::ListTasks { response_tx } => {
                                if let Err(e) = response_tx.send(tasks.values().cloned().collect()) {
                                    eprintln!("Ошибка отправки ответа list_tasks: {:?}", e);
                                }
                            }
                        }
                    }
                    _ = interval.tick() => {
                        // Watchdog: помечаем зависшие InProgress как Failed.
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
                                let key = (task.from_agent.clone(), task.from_session_id.clone());
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
                    }
                }
            }
        });

        Ok(Self {
            command_tx,
            tasks_dir,
        })
    }

    // ----- Клиентские методы -----

    pub async fn register_session(
        &self,
        agent_name: String,
        session_id: String,
        sender: mpsc::UnboundedSender<BoardEvent>,
    ) {
        if let Err(e) = self.command_tx.send(BoardCommand::RegisterSession {
            agent_name,
            session_id,
            sender,
        }) {
            eprintln!("Ошибка отправки команды RegisterSession: {}", e);
        }
    }

    pub async fn unregister_session(&self, agent_name: String, session_id: String) {
        if let Err(e) = self.command_tx.send(BoardCommand::UnregisterSession {
            agent_name,
            session_id,
        }) {
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

    pub async fn list_tasks(&self) -> Vec<Task> {
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self
            .command_tx
            .send(BoardCommand::ListTasks { response_tx: tx })
        {
            eprintln!("Ошибка отправки команды ListTasks: {}", e);
            return vec![];
        }
        match rx.await {
            Ok(tasks) => tasks,
            Err(e) => {
                eprintln!("Ошибка получения ответа ListTasks: {}", e);
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
    session_id: String,
}

#[derive(Deserialize)]
struct UnregisterQuery {
    agent_name: String,
    session_id: String,
}

pub fn build_router(board: std::sync::Arc<MessageBoard>, auth_token: String) -> Router {
    let state = AppState { board, auth_token };
    Router::new()
        .route("/register_session", post(register_session))
        .route("/unregister_session", post(unregister_session))
        .route("/tasks", post(create_task))
        .route("/tasks/{id}", get(get_task))
        .route("/tasks/{id}/complete", post(complete_task))
        .route("/tasks/{id}/fail", post(fail_task))
        .route("/events", get(sse_handler))
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
    eprintln!(
        "Запрос на /unregister_session: agent={}, session={}",
        query.agent_name, query.session_id
    );
    state
        .board
        .unregister_session(query.agent_name, query.session_id)
        .await;
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
        .register_session(query.agent_name.clone(), query.session_id.clone(), tx)
        .await;
    eprintln!(
        "Новое SSE-подключение: agent={}, session={}",
        query.agent_name, query.session_id
    );

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
    async fn test_create_task_rejected_for_offline_session() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            payload: serde_json::json!({"prompt": "hi"}),
            parent_task_id: None,
        };
        let res = board.create_task(req).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_create_task_delivered_when_online() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel::<BoardEvent>();
        board
            .register_session("b".to_string(), "s2".to_string(), tx)
            .await;

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            payload: serde_json::json!({"prompt": "hi"}),
            parent_task_id: None,
        };
        let resp = board.create_task(req).await.unwrap();
        let event = rx.recv().await.unwrap();
        match event {
            BoardEvent::TaskCreated { task } => assert_eq!(task.id, resp.task_id),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_get_task() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel::<BoardEvent>();
        board
            .register_session("b".to_string(), "s2".to_string(), tx)
            .await;

        let resp = board
            .create_task(CreateTaskRequest {
                from_agent: "a".into(),
                from_session_id: "s".into(),
                to_agent: "b".into(),
                to_session_id: "s2".into(),
                payload: serde_json::json!({"prompt": "hi"}),
                parent_task_id: None,
            })
            .await
            .unwrap();

        let task = board.get_task(resp.task_id.clone()).await.unwrap();
        assert_eq!(task.id, resp.task_id);
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[tokio::test]
    async fn test_create_task_rejected_for_missing_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let board = MessageBoard::new(dir.path().to_path_buf(), 300)
            .await
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel::<BoardEvent>();
        board
            .register_session("b".to_string(), "s2".to_string(), tx)
            .await;

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            payload: serde_json::json!({}),
            parent_task_id: None,
        };
        assert!(board.create_task(req).await.is_err());

        let req = CreateTaskRequest {
            from_agent: "a".into(),
            from_session_id: "s".into(),
            to_agent: "b".into(),
            to_session_id: "s2".into(),
            payload: serde_json::json!({"prompt": "   "}),
            parent_task_id: None,
        };
        assert!(board.create_task(req).await.is_err());
    }
}
