// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/session_store.rs

use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
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
    #[serde(default)]
    pub skill_context: Option<Message>,
    #[serde(default)]
    pub knowledge_context: Option<Message>,
    #[serde(default)]
    pub visible_tools: Vec<String>,
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
pub struct PendingTask {
    pub task_id: String,
    pub to_agent_name: String,
    pub to_session_id: String,
    #[serde(default)]
    pub project_id: String,
    #[serde(default)]
    pub chain: Vec<String>,
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
    #[serde(default)]
    pub owner_agent: Option<String>,
    #[serde(default)]
    pub context: Option<ContextBlock>,
    pub logs: Vec<LogEntry>,
    #[serde(default)]
    pub pending_tasks: Vec<PendingTask>,
    #[serde(default)]
    pub incoming_stack: Vec<String>,
    #[serde(default)]
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub display_name: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub owner_agent: Option<String>,
}

fn sessions_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(SESSIONS_DIR)
}

fn logs_dir() -> PathBuf {
    sessions_dir().join(LOGS_SUBDIR)
}

fn session_file(session_id: &str, owner_agent: Option<&str>) -> PathBuf {
    match owner_agent {
        Some(agent) => sessions_dir().join(format!("agent_{}_{}.json", agent, session_id)),
        None => sessions_dir().join(format!("chat_{}.json", session_id)),
    }
}

fn find_session_file(session_id: &str, owner_agent: Option<&str>) -> Option<PathBuf> {
    let path = session_file(session_id, owner_agent);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn lock_file(session_id: &str, owner_agent: Option<&str>) -> PathBuf {
    let base = session_file(session_id, owner_agent);
    let mut new_name = OsString::from(base.file_name().unwrap_or_default());
    new_name.push(".lock");
    let mut p = base.clone();
    p.set_file_name(new_name);
    p
}

pub fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn new_session(display_name: &str, owner_agent: Option<&str>) -> Result<Session> {
    fs::create_dir_all(sessions_dir())?;
    fs::create_dir_all(logs_dir())?;

    let session_id = uuid::Uuid::new_v4().to_string();
    let ts = now_ts();

    let session = Session {
        session_id,
        display_name: display_name.to_string(),
        created_at: ts,
        updated_at: ts,
        owner_agent: owner_agent.map(String::from),
        context: None,
        logs: Vec::new(),
        pending_tasks: Vec::new(),
        incoming_stack: Vec::new(),
        deleted: false,
    };

    save_session(&session)?;
    Ok(session)
}

pub fn create_session_with_id(
    session_id: &str,
    display_name: &str,
    owner_agent: Option<&str>,
) -> Result<Session> {
    fs::create_dir_all(sessions_dir())?;
    fs::create_dir_all(logs_dir())?;

    let ts = now_ts();
    let session = Session {
        session_id: session_id.to_string(),
        display_name: display_name.to_string(),
        created_at: ts,
        updated_at: ts,
        owner_agent: owner_agent.map(String::from),
        context: None,
        logs: Vec::new(),
        pending_tasks: Vec::new(),
        incoming_stack: Vec::new(),
        deleted: false,
    };

    save_session(&session)?;
    Ok(session)
}

pub fn load_session(session_id: &str, owner_agent: Option<&str>) -> Result<Session> {
    let path = find_session_file(session_id, owner_agent).ok_or_else(|| {
        anyhow::anyhow!(
            "Файл сессии '{}' не найден (owner: {:?})",
            session_id,
            owner_agent
        )
    })?;
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Не удалось прочитать файл сессии {}", path.display()))?;
    let session: Session = serde_json::from_str(&content)?;
    Ok(session)
}

pub fn save_session(session: &Session) -> Result<()> {
    fs::create_dir_all(sessions_dir())?;
    let owner = session.owner_agent.as_deref();

    let lock_path = lock_file(&session.session_id, owner);
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("Не удалось открыть lock-файл {}", lock_path.display()))?;
    lock.lock_exclusive()?;

    let tmp_path = sessions_dir().join(format!(".{}.tmp", session.session_id));
    let final_path = session_file(&session.session_id, owner);

    let mut session_clone = session.clone();
    if let Some(ctx) = session_clone.context.as_mut() {
        ctx.engine_config.api_key = "***".to_string();
    }

    if !session_clone.deleted {
        if let Some(old_path) = find_session_file(&session.session_id, owner) {
            if let Ok(content) = fs::read_to_string(&old_path) {
                if let Ok(old) = serde_json::from_str::<Session>(&content) {
                    if old.deleted {
                        session_clone.deleted = true;
                    }
                }
            }
        }
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
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".json") || name.ends_with(".tmp") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(session) = serde_json::from_str::<Session>(&content) {
                if session.deleted {
                    continue;
                }
                sessions.push(SessionSummary {
                    session_id: session.session_id,
                    display_name: session.display_name,
                    created_at: session.created_at,
                    updated_at: session.updated_at,
                    owner_agent: session.owner_agent,
                });
            }
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(sessions)
}

pub fn list_sessions_for_agent(agent_name: &str) -> Result<Vec<SessionSummary>> {
    let all = list_sessions()?;
    Ok(all
        .into_iter()
        .filter(|s| s.owner_agent.as_deref() == Some(agent_name))
        .collect())
}

pub fn list_chat_projects() -> Result<Vec<SessionSummary>> {
    let all = list_sessions()?;
    Ok(all
        .into_iter()
        .filter(|s| s.owner_agent.is_none())
        .collect())
}

pub fn is_session_deleted_or_missing(session_id: &str, owner_agent: Option<&str>) -> bool {
    match load_session(session_id, owner_agent) {
        Ok(s) => s.deleted,
        Err(_) => true,
    }
}

pub fn delete_session(session_id: &str, owner_agent: Option<&str>) -> Result<()> {
    if let Some(json_path) = find_session_file(session_id, owner_agent) {
        fs::remove_file(&json_path)?;
        let mut lock = json_path.clone();
        let mut new_name = OsString::from(json_path.file_name().unwrap_or_default());
        new_name.push(".lock");
        lock.set_file_name(new_name);
        if lock.exists() {
            let _ = fs::remove_file(&lock);
        }
    }
    Ok(())
}

pub fn init_log(session_id: &str, entity: &str, agent_name: Option<&str>) -> Result<String> {
    fs::create_dir_all(logs_dir())?;
    let filename = match entity {
        "chat" => format!("chat_{}.txt", session_id),
        "agent" => {
            let name = agent_name.unwrap_or("unknown");
            format!("agent_{}_{}.txt", name, session_id)
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

pub fn restore_api_key(session: &mut Session, api_key: &str) {
    if let Some(ctx) = session.context.as_mut() {
        if ctx.engine_config.api_key == "***" {
            ctx.engine_config.api_key = api_key.to_string();
        }
    }
}

pub fn load_session_with_key(
    session_id: &str,
    api_key: &str,
    owner_agent: Option<&str>,
) -> Result<Session> {
    let mut session = load_session(session_id, owner_agent)?;
    restore_api_key(&mut session, api_key);
    Ok(session)
}

pub fn add_pending_task(session: &mut Session, task: PendingTask) {
    session.pending_tasks.push(task);
}

pub fn remove_pending_task(session: &mut Session, task_id: &str) {
    session.pending_tasks.retain(|t| t.task_id != task_id);
}

pub fn find_pending_task<'a>(session: &'a Session, task_id: &str) -> Option<&'a PendingTask> {
    session.pending_tasks.iter().find(|t| t.task_id == task_id)
}

// ==================== Работа с проектами ====================

pub fn mark_project_deleted(session_id: &str) -> Result<Vec<String>> {
    let mut marked = Vec::new();
    let dir = sessions_dir();
    if !dir.exists() {
        return Ok(marked);
    }

    let chat_json = format!("chat_{}.json", session_id);
    let agent_json_suffix = format!("_{}.json", session_id);

    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let is_target =
            name == chat_json || (name.starts_with("agent_") && name.ends_with(&agent_json_suffix));
        if !is_target {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let mut session: Session = match serde_json::from_str(&content) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Не удалось распарсить {}: {}", path.display(), e);
                continue;
            }
        };
        if session.deleted {
            continue;
        }
        session.deleted = true;
        session.updated_at = now_ts();
        if let Err(e) = save_session(&session) {
            eprintln!("Не удалось пометить {}: {}", path.display(), e);
            continue;
        }
        marked.push(session.session_id);
    }

    Ok(marked)
}

pub fn purge_project(session_id: &str) -> Result<Vec<PathBuf>> {
    let mut deleted = Vec::new();

    let dir = sessions_dir();
    let chat_json = format!("chat_{}.json", session_id);
    let chat_lock = format!("chat_{}.json.lock", session_id);
    let agent_json_suffix = format!("_{}.json", session_id);
    let agent_lock_suffix = format!("_{}.json.lock", session_id);

    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let is_target = name == chat_json
                || name == chat_lock
                || (name.starts_with("agent_") && name.ends_with(&agent_json_suffix))
                || (name.starts_with("agent_") && name.ends_with(&agent_lock_suffix));
            if is_target {
                if let Err(e) = fs::remove_file(&path) {
                    eprintln!("Не удалось удалить {}: {}", path.display(), e);
                } else {
                    deleted.push(path);
                }
            }
        }
    }

    let logs = logs_dir();
    let chat_log = format!("chat_{}.txt", session_id);
    let agent_log_suffix = format!("_{}.txt", session_id);
    if logs.exists() {
        for entry in fs::read_dir(&logs)? {
            let entry = entry?;
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let is_target = name == chat_log
                || (name.starts_with("agent_") && name.ends_with(&agent_log_suffix));
            if is_target {
                if let Err(e) = fs::remove_file(&path) {
                    eprintln!("Не удалось удалить {}: {}", path.display(), e);
                } else {
                    deleted.push(path);
                }
            }
        }
    }

    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, Message, Role};

    fn unique_id(prefix: &str) -> String {
        format!("_test_{}_{}", prefix, uuid::Uuid::new_v4())
    }

    fn cleanup(id: &str) {
        let _ = purge_project(id);
    }

    #[test]
    fn create_session_with_id_persists() {
        let id = unique_id("create");
        let s = create_session_with_id(&id, "Test Project", None).unwrap();
        assert_eq!(s.session_id, id);
        assert_eq!(s.display_name, "Test Project");
        assert!(!s.deleted);
        assert!(s.owner_agent.is_none());

        let loaded = load_session(&id, None).unwrap();
        assert_eq!(loaded.session_id, id);
        cleanup(&id);
    }

    #[test]
    fn load_nonexistent_errors() {
        let res = load_session("_definitely_does_not_exist_xyz", None);
        assert!(res.is_err());
    }

    #[test]
    fn save_and_load_roundtrip_preserves_fields() {
        let id = unique_id("roundtrip");
        let mut s = create_session_with_id(&id, "Roundtrip", None).unwrap();
        s.display_name = "Changed".into();
        s.pending_tasks.push(PendingTask {
            task_id: "t1".into(),
            to_agent_name: "analyzer".into(),
            to_session_id: "s2".into(),
            project_id: "proj".into(),
            chain: vec!["analyzer".into()],
        });
        save_session(&s).unwrap();

        let loaded = load_session(&id, None).unwrap();
        assert_eq!(loaded.display_name, "Changed");
        assert_eq!(loaded.pending_tasks.len(), 1);
        assert_eq!(loaded.pending_tasks[0].to_agent_name, "analyzer");
        cleanup(&id);
    }

    #[test]
    fn save_masks_api_key_in_context() {
        let id = unique_id("mask");
        let mut s = create_session_with_id(&id, "Mask", None).unwrap();
        s.context = Some(ContextBlock {
            engine_config: EngineConfig {
                api_key: "supersecret".into(),
                ..Default::default()
            },
            engine_state: EngineState {
                system_messages: vec![],
                skill_context: None,
                knowledge_context: None,
                prefix_turns: vec![],
                tail_turns: vec![],
                pending_turn: None,
                metrics: Default::default(),
            },
        });
        save_session(&s).unwrap();

        // Читаем сырой JSON, чтобы убедиться, что ключ замаскирован
        let path = find_session_file(&id, None).unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("supersecret"));
        assert!(raw.contains("***"));

        // load_session_with_key восстанавливает
        let loaded = load_session_with_key(&id, "realkey", None).unwrap();
        let ctx = loaded.context.unwrap();
        assert_eq!(ctx.engine_config.api_key, "realkey");
        cleanup(&id);
    }

    #[test]
    fn soft_delete_survives_resave() {
        let id = unique_id("softdel");
        let s = create_session_with_id(&id, "SoftDel", None).unwrap();

        // Помечаем удалённым
        mark_project_deleted(&id).unwrap();

        // Пытаемся перезаписать (эмулируем «забыли про deleted»)
        let mut s2 = s.clone();
        s2.display_name = "Trying to resurrect".into();
        save_session(&s2).unwrap();

        // deleted должен остаться true (защита в save_session)
        let loaded = load_session(&id, None).unwrap();
        assert!(loaded.deleted);
        cleanup(&id);
    }

    #[test]
    fn list_sessions_excludes_deleted() {
        let id_live = unique_id("live");
        let id_dead = unique_id("dead");
        create_session_with_id(&id_live, "Live", None).unwrap();
        create_session_with_id(&id_dead, "Dead", None).unwrap();
        mark_project_deleted(&id_dead).unwrap();

        let all = list_sessions().unwrap();
        let ids: Vec<&str> = all.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&id_live.as_str()));
        assert!(!ids.contains(&id_dead.as_str()));
        cleanup(&id_live);
        cleanup(&id_dead);
    }

    #[test]
    fn list_chat_projects_excludes_agent_sessions() {
        let chat_id = unique_id("chat");
        let agent_id = unique_id("agent");
        create_session_with_id(&chat_id, "Chat", None).unwrap();
        create_session_with_id(&agent_id, "Agent", Some("analyzer")).unwrap();

        let chat_projects = list_chat_projects().unwrap();
        let ids: Vec<&str> = chat_projects
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        assert!(ids.contains(&chat_id.as_str()));
        assert!(!ids.contains(&agent_id.as_str()));
        cleanup(&chat_id);
        cleanup(&agent_id);
    }

    #[test]
    fn list_sessions_for_agent_returns_only_that_agent() {
        let a1 = unique_id("a1");
        let a2 = unique_id("a2");
        let b = unique_id("b");
        create_session_with_id(&a1, "A1", Some("analyzer")).unwrap();
        create_session_with_id(&a2, "A2", Some("analyzer")).unwrap();
        create_session_with_id(&b, "B", Some("coder")).unwrap();

        let analyzer_sessions = list_sessions_for_agent("analyzer").unwrap();
        let ids: Vec<&str> = analyzer_sessions
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        assert!(ids.contains(&a1.as_str()));
        assert!(ids.contains(&a2.as_str()));
        assert!(!ids.contains(&b.as_str()));
        cleanup(&a1);
        cleanup(&a2);
        cleanup(&b);
    }

    #[test]
    fn purge_project_removes_session_file() {
        let id = unique_id("purge");
        create_session_with_id(&id, "Purge", None).unwrap();
        assert!(find_session_file(&id, None).is_some());

        purge_project(&id).unwrap();
        assert!(find_session_file(&id, None).is_none());
    }

    #[test]
    fn is_session_deleted_or_missing_true_for_missing() {
        assert!(is_session_deleted_or_missing("_no_such_session_zzz", None));
    }

    #[test]
    fn is_session_deleted_or_missing_true_after_mark() {
        let id = unique_id("isdel");
        create_session_with_id(&id, "IsDel", None).unwrap();
        mark_project_deleted(&id).unwrap();
        assert!(is_session_deleted_or_missing(&id, None));
        cleanup(&id);
    }

    #[test]
    fn is_session_deleted_or_missing_false_for_live() {
        let id = unique_id("islive");
        create_session_with_id(&id, "IsLive", None).unwrap();
        assert!(!is_session_deleted_or_missing(&id, None));
        cleanup(&id);
    }

    #[test]
    fn pending_task_add_remove_find() {
        let mut s = create_session_with_id(&unique_id("pend"), "P", None).unwrap();
        let task = PendingTask {
            task_id: "t1".into(),
            to_agent_name: "a".into(),
            to_session_id: "s".into(),
            project_id: "p".into(),
            chain: vec![],
        };
        add_pending_task(&mut s, task);
        assert_eq!(s.pending_tasks.len(), 1);
        assert!(find_pending_task(&s, "t1").is_some());

        remove_pending_task(&mut s, "t1");
        assert!(s.pending_tasks.is_empty());
        assert!(find_pending_task(&s, "t1").is_none());
        cleanup(&s.session_id);
    }

    #[test]
    fn init_log_creates_file_and_returns_rel_path() {
        let id = unique_id("log");
        let rel = init_log(&id, "chat", None).unwrap();
        assert!(rel.contains("chat_"));
        assert!(rel.ends_with(".txt"));

        let full = sessions_dir().join(&rel);
        assert!(full.exists());
        cleanup(&id);
    }

    #[test]
    fn add_or_update_log_populates_session() {
        let id = unique_id("logup");
        let mut s = create_session_with_id(&id, "LogUp", None).unwrap();
        add_or_update_log(&mut s, "chat", None).unwrap();
        assert_eq!(s.logs.len(), 1);
        assert_eq!(s.logs[0].entity, "chat");

        // Повторный вызов не должен добавить вторую запись
        add_or_update_log(&mut s, "chat", None).unwrap();
        assert_eq!(s.logs.len(), 1);
        cleanup(&id);
    }

    #[test]
    fn now_ts_is_monotonic_nonzero() {
        let a = now_ts();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let b = now_ts();
        assert!(a > 0);
        assert!(b >= a);
    }

    #[test]
    fn restore_api_key_replaces_placeholder() {
        let mut s = Session {
            session_id: "x".into(),
            display_name: "x".into(),
            created_at: 0,
            updated_at: 0,
            owner_agent: None,
            context: Some(ContextBlock {
                engine_config: EngineConfig {
                    api_key: "***".into(),
                    ..Default::default()
                },
                engine_state: EngineState {
                    system_messages: vec![],
                    skill_context: None,
                    knowledge_context: None,
                    prefix_turns: vec![],
                    tail_turns: vec![],
                    pending_turn: None,
                    metrics: Default::default(),
                },
            }),
            logs: vec![],
            pending_tasks: vec![],
            incoming_stack: vec![],
            deleted: false,
        };
        restore_api_key(&mut s, "realkey");
        let ctx = s.context.unwrap();
        assert_eq!(ctx.engine_config.api_key, "realkey");
    }

    #[test]
    fn restore_api_key_does_not_overwrite_real_key() {
        let mut s = Session {
            session_id: "x".into(),
            display_name: "x".into(),
            created_at: 0,
            updated_at: 0,
            owner_agent: None,
            context: Some(ContextBlock {
                engine_config: EngineConfig {
                    api_key: "real".into(),
                    ..Default::default()
                },
                engine_state: EngineState {
                    system_messages: vec![],
                    skill_context: None,
                    knowledge_context: None,
                    prefix_turns: vec![],
                    tail_turns: vec![],
                    pending_turn: None,
                    metrics: Default::default(),
                },
            }),
            logs: vec![],
            pending_tasks: vec![],
            incoming_stack: vec![],
            deleted: false,
        };
        restore_api_key(&mut s, "newkey");
        let ctx = s.context.unwrap();
        assert_eq!(ctx.engine_config.api_key, "real");
    }

    #[test]
    fn load_session_with_key_missing_errors() {
        assert!(load_session_with_key("_no_such_zzz", "k", None).is_err());
    }

    #[test]
    fn delete_session_removes_file() {
        let id = unique_id("del");
        create_session_with_id(&id, "Del", None).unwrap();
        assert!(find_session_file(&id, None).is_some());
        delete_session(&id, None).unwrap();
        assert!(find_session_file(&id, None).is_none());
    }

    // Suppress unused-import warnings for Role/Message in case
    // tests are later refactored.
    #[allow(dead_code)]
    fn _touch_types() {
        let _ = Role::User;
        let _: Option<Message> = None;
    }
}
