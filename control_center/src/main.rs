#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::State;

use nano_harness::config::TomlConfig;

// ============================================================================
// spawn / kill
// ============================================================================

#[cfg(target_os = "windows")]
fn spawn_service(svc: &ServiceDef) -> Result<Child, String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_CONSOLE: u32 = 0x00000010;

    let mut parts: Vec<String> = vec![svc.cmd.clone()];
    for a in &svc.args {
        parts.push(a.clone());
    }
    let svc_line = parts.join(" ");
    let full = format!("title \"{}\"&& {}", svc.title, svc_line);

    let mut cmd = Command::new("cmd.exe");
    cmd.raw_arg("/c");
    cmd.raw_arg(&full);
    cmd.creation_flags(CREATE_NEW_CONSOLE);

    eprintln!("[cmd] spawn cmd.exe /c {}", full);
    cmd.spawn().map_err(|e| format!("spawn cmd.exe: {}", e))
}

#[cfg(not(target_os = "windows"))]
fn spawn_service(svc: &ServiceDef) -> Result<Child, String> {
    eprintln!("[cmd] spawn '{}' {:?}", svc.cmd, svc.args);
    let mut cmd = Command::new(&svc.cmd);
    cmd.args(&svc.args);
    cmd.spawn()
        .map_err(|e| format!("spawn '{}': {}", svc.cmd, e))
}

fn kill_tree(child: &mut Child) -> Result<(), String> {
    if let Ok(Some(_)) = child.try_wait() {
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let pid = child.id();

        // 1. Graceful: WM_CLOSE верхнеуровневому окну → CTRL_CLOSE_EVENT всем в группе.
        let _ = Command::new("taskkill")
            .args(["/T", "/PID", &pid.to_string()])
            .output();

        // 2. Дать сервису шанс погаситься.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return Ok(()),
                Ok(None) => {}
                Err(_) => break,
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        // 3. Fallback: hard kill.
        let out = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output()
            .map_err(|e| format!("taskkill: {}", e))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "[warn] taskkill /F /T /PID {} exited non-zero: {}",
                pid,
                stderr.trim()
            );
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let pid = child.id();
        let _ = Command::new("kill")
            .args(["-TERM", &format!("-{}", pid)])
            .status();
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }

    let _ = child.wait();
    Ok(())
}

// ============================================================================
// Service definitions
// ============================================================================

#[derive(Debug, Clone)]
struct ServiceDef {
    key: String,
    label: String,
    bind: String,
    title: String,
    cmd: String,
    args: Vec<String>,
}

fn cargo_run(bin: &str, extra: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = vec!["run".into()];
    if !cfg!(debug_assertions) {
        args.push("--release".into());
    }
    args.push("--bin".into());
    args.push(bin.into());
    for e in extra {
        args.push((*e).into());
    }
    args
}

fn build_services(cfg: &TomlConfig) -> Vec<ServiceDef> {
    let mut out = Vec::new();

    let board_bind = cfg
        .message_board
        .as_ref()
        .map(|b| b.bind_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8090".into());
    out.push(ServiceDef {
        key: "board".into(),
        label: "Message Board".into(),
        bind: board_bind,
        title: "Message Board".into(),
        cmd: "cargo".into(),
        args: cargo_run("message_board_server", &[]),
    });

    let storage_bind = cfg
        .local_storage_http_server
        .as_ref()
        .map(|s| s.bind_addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8080".into());
    out.push(ServiceDef {
        key: "storage".into(),
        label: "Local Storage".into(),
        bind: storage_bind,
        title: "Local Storage".into(),
        cmd: "cargo".into(),
        args: cargo_run("local_storage_http_server", &[]),
    });

    out.push(ServiceDef {
        key: "chat".into(),
        label: "Chat".into(),
        bind: "—".into(),
        title: "Chat".into(),
        cmd: "cargo".into(),
        args: cargo_run("nano_harness", &[]),
    });

    for a in &cfg.agents {
        let display = format!("Agent {}", a.name);
        out.push(ServiceDef {
            key: format!("agent:{}", a.name),
            label: display.clone(),
            bind: a.bind_addr.clone(),
            title: display,
            cmd: "cargo".into(),
            args: cargo_run("agent_server", &["--", "--agent-name", a.name.as_str()]),
        });
    }

    out
}

// ============================================================================
// AppState
// ============================================================================

struct RunningService {
    child: Child,
    started_at: Instant,
}

#[derive(Clone)]
struct ServiceError {
    message: String,
    at: Instant,
}

const ERROR_TTL: Duration = Duration::from_secs(30);

struct AppStateInner {
    services: Mutex<Vec<ServiceDef>>,
    running: Mutex<HashMap<String, RunningService>>,
    errors: Mutex<HashMap<String, ServiceError>>,
    config_error: Mutex<Option<String>>,
}

type AppState = Arc<AppStateInner>;

fn load_config() -> Result<TomlConfig, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(p) = std::env::var("NH_CONFIG") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(PathBuf::from("config.toml"));
    candidates.push(PathBuf::from("../config.toml"));

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("config.toml"));
            candidates.push(dir.join("../../config.toml"));
            candidates.push(dir.join("../../../config.toml"));
        }
    }

    for p in candidates {
        if p.exists() {
            eprintln!("[cfg] loaded {}", p.display());
            return TomlConfig::load(&p).map_err(|e| format!("{:#}", e));
        }
    }

    Err("config.toml not found".to_string())
}

#[derive(Serialize)]
struct ServiceInfo {
    key: String,
    label: String,
    bind: String,
    state: String,
    uptime_sec: Option<u64>,
    title: String,
    error: Option<String>,
}

fn find_service_def(state: &AppStateInner, key: &str) -> Result<ServiceDef, String> {
    state
        .services
        .lock()
        .unwrap()
        .iter()
        .find(|s| s.key == key)
        .cloned()
        .ok_or_else(|| format!("unknown service: {}", key))
}

// ============================================================================
// Sync-реализации команд
// ============================================================================

fn do_start_blocking(key: &str, st: &AppStateInner) -> Result<(), String> {
    let svc = find_service_def(st, key)?;

    {
        let running = st.running.lock().unwrap();
        if running.contains_key(key) {
            return Err(format!("already running: {}", svc.title));
        }
    }

    eprintln!("[cmd] start {}", key);

    match spawn_service(&svc) {
        Ok(child) => {
            st.running.lock().unwrap().insert(
                key.to_string(),
                RunningService {
                    child,
                    started_at: Instant::now(),
                },
            );
            st.errors.lock().unwrap().remove(key);
            eprintln!("[ok] started {}", key);
            Ok(())
        }
        Err(e) => {
            eprintln!("[err] start {}: {}", key, e);
            st.errors.lock().unwrap().insert(
                key.to_string(),
                ServiceError {
                    message: e.clone(),
                    at: Instant::now(),
                },
            );
            Err(e)
        }
    }
}

fn do_stop_blocking(key: &str, st: &AppStateInner) -> Result<(), String> {
    eprintln!("[cmd] stop {}", key);
    let svc = find_service_def(st, key)?;

    let mut rs_opt = {
        let mut running = st.running.lock().unwrap();
        running.remove(key)
    };

    match rs_opt.as_mut() {
        Some(rs) => {
            kill_tree(&mut rs.child)?;
            st.errors.lock().unwrap().remove(key);
            eprintln!("[ok] stopped {}", key);
            Ok(())
        }
        None => {
            let msg = format!("not running: {}", svc.title);
            eprintln!("[err] stop {}: {}", key, msg);
            st.errors.lock().unwrap().insert(
                key.to_string(),
                ServiceError {
                    message: msg.clone(),
                    at: Instant::now(),
                },
            );
            Err(msg)
        }
    }
}

// ============================================================================
// Commands
// ============================================================================

#[tauri::command]
async fn list_services(state: State<'_, AppState>) -> Result<Vec<ServiceInfo>, String> {
    let st = state.inner().clone();

    tokio::task::spawn_blocking(move || {
        let services_snapshot: Vec<ServiceDef> = {
            let services = st.services.lock().unwrap();
            services.clone()
        };

        let mut new_errors: Vec<(String, String)> = Vec::new();
        let mut to_remove: Vec<String> = Vec::new();
        {
            let mut running = st.running.lock().unwrap();
            for (key, rs) in running.iter_mut() {
                match rs.child.try_wait() {
                    Ok(None) => {}
                    Ok(Some(status)) => {
                        if !status.success() {
                            let code = status.code().unwrap_or(-1);
                            new_errors.push((key.clone(), format!("exited with code {}", code)));
                        }
                        to_remove.push(key.clone());
                    }
                    Err(e) => {
                        new_errors.push((key.clone(), format!("try_wait failed: {}", e)));
                        to_remove.push(key.clone());
                    }
                }
            }
            for k in &to_remove {
                running.remove(k);
            }
        }
        {
            let mut errors = st.errors.lock().unwrap();
            for (k, msg) in new_errors {
                errors.insert(
                    k,
                    ServiceError {
                        message: msg,
                        at: Instant::now(),
                    },
                );
            }
        }

        let running_snapshot: HashMap<String, u64> = {
            let running = st.running.lock().unwrap();
            running
                .iter()
                .map(|(k, rs)| (k.clone(), rs.started_at.elapsed().as_secs()))
                .collect()
        };
        let errors_snapshot: HashMap<String, String> = {
            let mut errors = st.errors.lock().unwrap();
            let now = Instant::now();
            errors.retain(|_, e| now.duration_since(e.at) < ERROR_TTL);
            errors
                .iter()
                .map(|(k, e)| (k.clone(), e.message.clone()))
                .collect()
        };

        let mut out = Vec::with_capacity(services_snapshot.len());
        for s in &services_snapshot {
            let (state_str, uptime, err): (&str, Option<u64>, Option<String>) =
                if let Some(up) = running_snapshot.get(&s.key) {
                    ("running", Some(*up), None)
                } else if let Some(msg) = errors_snapshot.get(&s.key) {
                    ("error", None, Some(msg.clone()))
                } else {
                    ("stopped", None, None)
                };

            out.push(ServiceInfo {
                key: s.key.clone(),
                label: s.label.clone(),
                bind: s.bind.clone(),
                state: state_str.to_string(),
                uptime_sec: uptime,
                title: s.title.clone(),
                error: err,
            });
        }
        Ok::<_, String>(out)
    })
    .await
    .map_err(|e| format!("join error: {}", e))?
}

#[tauri::command]
async fn get_config_error(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let st = state.inner().clone();
    let guard = st.config_error.lock().unwrap();
    Ok(guard.clone())
}

#[tauri::command]
async fn start_service(key: String, state: State<'_, AppState>) -> Result<(), String> {
    let st = state.inner().clone();
    tokio::task::spawn_blocking(move || do_start_blocking(&key, st.as_ref()))
        .await
        .map_err(|e| format!("join error: {}", e))?
}

#[tauri::command]
async fn stop_service(key: String, state: State<'_, AppState>) -> Result<(), String> {
    let st = state.inner().clone();
    tokio::task::spawn_blocking(move || do_stop_blocking(&key, st.as_ref()))
        .await
        .map_err(|e| format!("join error: {}", e))?
}

#[tauri::command]
async fn restart_service(key: String, state: State<'_, AppState>) -> Result<(), String> {
    eprintln!("[cmd] restart {}", key);
    let st = state.inner().clone();

    find_service_def(st.as_ref(), &key)?;

    {
        let st_stop = st.clone();
        let key_stop = key.clone();
        let stop_res = tokio::task::spawn_blocking(move || {
            let mut rs_opt = {
                let mut running = st_stop.running.lock().unwrap();
                running.remove(&key_stop)
            };
            if let Some(rs) = rs_opt.as_mut() {
                kill_tree(&mut rs.child)?;
            }
            Ok::<(), String>(())
        })
        .await
        .map_err(|e| format!("join error: {}", e))?;
        stop_res?;
    }

    tokio::time::sleep(Duration::from_millis(500)).await;

    let st_start = st.clone();
    let key_start = key.clone();
    tokio::task::spawn_blocking(move || do_start_blocking(&key_start, st_start.as_ref()))
        .await
        .map_err(|e| format!("join error: {}", e))?
}

#[tauri::command]
async fn start_all(state: State<'_, AppState>) -> Result<(), String> {
    eprintln!("[cmd] start_all");
    let st = state.inner().clone();

    let keys: Vec<String> = {
        let services = st.services.lock().unwrap();
        services.iter().map(|s| s.key.clone()).collect()
    };

    let mut ordered: Vec<String> = Vec::new();
    for k in &keys {
        if k == "board" {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k == "storage" {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k.starts_with("agent:") {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k == "chat" {
            ordered.push(k.clone());
        }
    }

    let mut errors: Vec<String> = Vec::new();
    for k in ordered {
        let already_running = {
            let running = st.running.lock().unwrap();
            running.contains_key(&k)
        };
        if already_running {
            continue;
        }

        let st_run = st.clone();
        let k_run = k.clone();
        let res = tokio::task::spawn_blocking(move || do_start_blocking(&k_run, st_run.as_ref()))
            .await
            .map_err(|e| format!("join error: {}", e))?;
        if let Err(e) = res {
            errors.push(format!("{}: {}", k, e));
        }
        tokio::time::sleep(Duration::from_millis(900)).await;
    }

    if errors.is_empty() {
        eprintln!("[ok] start_all done");
        Ok(())
    } else {
        let msg = errors.join("; ");
        eprintln!("[err] start_all: {}", msg);
        Err(msg)
    }
}

#[tauri::command]
async fn stop_all(state: State<'_, AppState>) -> Result<(), String> {
    eprintln!("[cmd] stop_all");
    let st = state.inner().clone();

    let keys: Vec<String> = {
        let services = st.services.lock().unwrap();
        services.iter().map(|s| s.key.clone()).collect()
    };

    let mut ordered: Vec<String> = Vec::new();
    for k in &keys {
        if k == "chat" {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k.starts_with("agent:") {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k == "storage" {
            ordered.push(k.clone());
        }
    }
    for k in &keys {
        if k == "board" {
            ordered.push(k.clone());
        }
    }

    let mut errors: Vec<String> = Vec::new();
    for k in ordered {
        let running = {
            let map = st.running.lock().unwrap();
            map.contains_key(&k)
        };
        if !running {
            continue;
        }

        let st_run = st.clone();
        let k_run = k.clone();
        let res = tokio::task::spawn_blocking(move || do_stop_blocking(&k_run, st_run.as_ref()))
            .await
            .map_err(|e| format!("join error: {}", e))?;
        if let Err(e) = res {
            errors.push(format!("{}: {}", k, e));
        }
    }

    if errors.is_empty() {
        eprintln!("[ok] stop_all done");
        Ok(())
    } else {
        let msg = errors.join("; ");
        eprintln!("[err] stop_all: {}", msg);
        Err(msg)
    }
}

#[tauri::command]
async fn reload_config(state: State<'_, AppState>) -> Result<(), String> {
    eprintln!("[cmd] reload_config");
    let st = state.inner().clone();

    let cfg = match load_config() {
        Ok(c) => c,
        Err(e) => {
            *st.config_error.lock().unwrap() = Some(e.clone());
            return Err(e);
        }
    };

    let services = build_services(&cfg);
    let valid_keys: HashSet<String> = services.iter().map(|s| s.key.clone()).collect();

    {
        *st.services.lock().unwrap() = services;

        st.errors
            .lock()
            .unwrap()
            .retain(|k, _| valid_keys.contains(k));
        st.running
            .lock()
            .unwrap()
            .retain(|k, _| valid_keys.contains(k));
        *st.config_error.lock().unwrap() = None;
    }

    eprintln!("[ok] config reloaded");
    Ok(())
}

// ============================================================================
// main
// ============================================================================

fn main() {
    eprintln!("[boot] control_center");

    let (services, config_error) = match load_config() {
        Ok(cfg) => (build_services(&cfg), None),
        Err(e) => {
            eprintln!("[err] config load failed: {}", e);
            (Vec::new(), Some(e))
        }
    };

    eprintln!("[boot] services={}", services.len());

    let state: AppState = Arc::new(AppStateInner {
        services: Mutex::new(services),
        running: Mutex::new(HashMap::new()),
        errors: Mutex::new(HashMap::new()),
        config_error: Mutex::new(config_error),
    });

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            list_services,
            get_config_error,
            start_service,
            stop_service,
            restart_service,
            start_all,
            stop_all,
            reload_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
