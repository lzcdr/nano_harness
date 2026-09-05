// src/session_store.rs

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::{EngineConfig, Message, SessionMetrics, Turn};

const SESSIONS_DIR: &str = ".sessions";
const LOGS_SUBDIR: &str = "logs";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    pub system_messages: Vec<Message>,
    pub prefix_turns: Vec<Turn>,
    pub tail_turns: Vec<Turn>,
    pub pending_turn: Option<Turn>,
    pub metrics: SessionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub engine_config: EngineConfig,
    pub engine_state: EngineState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContextBlock {
    pub name: String,
    pub engine_config: EngineConfig,
    pub engine_state: EngineState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContexts {
    pub chat: Option<ContextBlock>,
    pub agents: Vec<AgentContextBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub entity: String,
    pub agent_name: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub display_name: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub contexts: SessionContexts,
    pub logs: Vec<LogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub display_name: String,
    pub created_at: u64,
    pub updated_at: u64,
}

fn sessions_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(SESSIONS_DIR)
}

fn logs_dir() -> PathBuf {
    sessions_dir().join(LOGS_SUBDIR)
}

fn session_file(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{}.json", session_id))
}

fn lock_file(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{}.lock", session_id))
}

pub fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn new_session(display_name: &str) -> Result<Session> {
    fs::create_dir_all(sessions_dir())?;
    fs::create_dir_all(logs_dir())?;

    let session_id = uuid::Uuid::new_v4().to_string();
    let ts = now_ts();

    let session = Session {
        session_id: session_id.clone(),
        display_name: display_name.to_string(),
        created_at: ts,
        updated_at: ts,
        contexts: SessionContexts {
            chat: None,
            agents: Vec::new(),
        },
        logs: Vec::new(),
    };

    save_session(&session)?;
    Ok(session)
}

pub fn create_session_with_id(session_id: &str, display_name: &str) -> Result<Session> {
    fs::create_dir_all(sessions_dir())?;
    fs::create_dir_all(logs_dir())?;

    let ts = now_ts();
    let session = Session {
        session_id: session_id.to_string(),
        display_name: display_name.to_string(),
        created_at: ts,
        updated_at: ts,
        contexts: SessionContexts {
            chat: None,
            agents: Vec::new(),
        },
        logs: Vec::new(),
    };

    save_session(&session)?;
    Ok(session)
}

pub fn load_session(session_id: &str) -> Result<Session> {
    let path = session_file(session_id);
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Не удалось прочитать файл сессии {}", path.display()))?;
    let session: Session = serde_json::from_str(&content)?;
    Ok(session)
}

pub fn save_session(session: &Session) -> Result<()> {
    fs::create_dir_all(sessions_dir())?;

    let lock_path = lock_file(&session.session_id);
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("Не удалось открыть lock-файл {}", lock_path.display()))?;
    lock.lock_exclusive()?;

    let tmp_path = sessions_dir().join(format!("{}.tmp", session.session_id));
    let final_path = session_file(&session.session_id);

    // Маскируем API-ключи перед сохранением
    let mut session_clone = session.clone();
    if let Some(chat_ctx) = session_clone.contexts.chat.as_mut() {
        chat_ctx.engine_config.api_key = "***".to_string();
    }
    for agent_ctx in session_clone.contexts.agents.iter_mut() {
        agent_ctx.engine_config.api_key = "***".to_string();
    }
    let json = serde_json::to_string_pretty(&session_clone)?;

    {
        let mut tmp_file = File::create(&tmp_path)
            .with_context(|| format!("Не удалось создать временный файл {}", tmp_path.display()))?;
        tmp_file.write_all(json.as_bytes())?;
        tmp_file.flush()?;
    }
    if final_path.exists() {
        fs::remove_file(&final_path)
            .with_context(|| format!("Не удалось удалить старый файл {}", final_path.display()))?;
    }
    fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "Не удалось переименовать {} в {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    lock.unlock()?;

    // Удаляем lock-файл после сохранения
    if let Err(e) = fs::remove_file(&lock_path) {
        eprintln!(
            "Предупреждение: не удалось удалить lock-файл {}: {}",
            lock_path.display(),
            e
        );
    }

    Ok(())
}

pub fn list_sessions() -> Result<Vec<SessionSummary>> {
    fs::create_dir_all(sessions_dir())?;
    let mut sessions = Vec::new();
    for entry in fs::read_dir(sessions_dir())? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(session) = serde_json::from_str::<Session>(&content) {
                    sessions.push(SessionSummary {
                        session_id: session.session_id,
                        display_name: session.display_name,
                        created_at: session.created_at,
                        updated_at: session.updated_at,
                    });
                }
            }
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(sessions)
}

pub fn delete_session(session_id: &str) -> Result<()> {
    let json_path = session_file(session_id);
    if json_path.exists() {
        fs::remove_file(&json_path)?;
    }
    let lock_path = lock_file(session_id);
    if lock_path.exists() {
        fs::remove_file(&lock_path)?;
    }
    Ok(())
}

pub fn init_log(session_id: &str, entity: &str, agent_name: Option<&str>) -> Result<String> {
    fs::create_dir_all(logs_dir())?;
    let filename = match entity {
        "chat" => format!("session_{}_chat.txt", session_id),
        "agent" => {
            let name = agent_name.unwrap_or("unknown");
            format!("session_{}_{}.txt", session_id, name)
        }
        _ => return Err(anyhow::anyhow!("Unknown entity type")),
    };
    let rel_path = Path::new(LOGS_SUBDIR).join(&filename);
    let full_path = sessions_dir().join(&rel_path);
    if !full_path.exists() {
        File::create(&full_path)
            .with_context(|| format!("Не удалось создать лог-файл {}", full_path.display()))?;
    }
    Ok(rel_path.to_string_lossy().to_string())
}

pub fn open_log(session_id: &str, entity: &str, agent_name: Option<&str>) -> Result<File> {
    let rel_path = init_log(session_id, entity, agent_name)?;
    let full_path = sessions_dir().join(&rel_path);
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&full_path)
        .with_context(|| format!("Не удалось открыть лог-файл {}", full_path.display()))
}

pub fn add_or_update_log(
    session: &mut Session,
    entity: &str,
    agent_name: Option<&str>,
) -> Result<()> {
    let path = init_log(&session.session_id, entity, agent_name)?;
    if let Some(log) = session
        .logs
        .iter_mut()
        .find(|l| l.entity == entity && l.agent_name.as_deref() == agent_name)
    {
        log.path = path;
    } else {
        session.logs.push(LogEntry {
            entity: entity.to_string(),
            agent_name: agent_name.map(String::from),
            path,
        });
    }
    Ok(())
}

pub fn restore_api_key_for_chat(session: &mut Session, api_key: &str) {
    if let Some(chat_ctx) = session.contexts.chat.as_mut() {
        if chat_ctx.engine_config.api_key == "***" {
            chat_ctx.engine_config.api_key = api_key.to_string();
        }
    }
}

pub fn restore_api_key_for_agent(session: &mut Session, agent_name: &str, api_key: &str) {
    if let Some(agent_ctx) = session
        .contexts
        .agents
        .iter_mut()
        .find(|a| a.name == agent_name)
    {
        if agent_ctx.engine_config.api_key == "***" {
            agent_ctx.engine_config.api_key = api_key.to_string();
        }
    }
}

pub fn load_session_with_key(
    session_id: &str,
    api_key: &str,
    agent_name: Option<&str>,
) -> Result<Session> {
    let mut session = load_session(session_id)?;
    match agent_name {
        Some(name) => restore_api_key_for_agent(&mut session, name, api_key),
        None => restore_api_key_for_chat(&mut session, api_key),
    }
    Ok(session)
}
