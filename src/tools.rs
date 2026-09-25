// src/tools.rs

use crate::agent_core::{AgentConfig, OutgoingTask, OutgoingTasks, PendingCalls};
use crate::engine::{FunctionDefinition, ToolDefinition};
use reqwest::blocking::Client;
use reqwest::Method;
use rhai::{Dynamic, Engine};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Handle;

pub const RHAI_MAX_OPERATIONS: u64 = 1_000_000;
pub const RHAI_MAX_CALL_LEVELS: usize = 64;
pub const RHAI_MAX_STRING_SIZE: usize = 1024 * 256;

/// Контекст для регистрации board-функций в Rhai.
pub struct BoardContext {
    pub board_url: String,
    pub board_token: String,
    pub self_agent_name: String,
    pub self_session_id: Option<String>,
    pub project_id: String,
    pub parent_chain: Vec<String>,
    pub pending_calls: PendingCalls,
    pub agent_call_timeout_sec: u64,
    pub posted_flag: Arc<AtomicBool>,
    pub outgoing_tasks: OutgoingTasks,
}

pub fn available_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "run_code".to_string(),
                description: "Выполняет код на языке Rhai и возвращает результат. \
                              В коде доступны функции: get_time(), get_weather(city), \
                              функции хранилища: storage_read_file, storage_write_file, \
                              storage_delete_file, storage_create_dir, storage_list_dir, \
                              storage_walk, storage_search_by_name, storage_search_similar, \
                              storage_read_about, storage_write_about, storage_read_summary, \
                              storage_write_summary, а также функции работы с доской: \
                              call_agent(to_agent, prompt) — синхронный вызов, \
                              post_task(to_agent, prompt) — асинхронный. \
                              session_id и project_id передаются автоматически из текущей сессии."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Скрипт на Rhai для выполнения"
                        }
                    },
                    "required": ["code"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "local_storage".to_string(),
                description: "Работа с локальным хранилищем файлов через HTTP API: чтение, запись, поиск, список файлов".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "query": {"type": "string"},
                        "pattern": {"type": "string"},
                        "top_k": {"type": "integer"}
                    },
                    "required": ["action"]
                }),
            },
        },
    ]
}

pub fn register_basic_functions(engine: &mut Engine) {
    engine.register_fn("get_time", || -> String {
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    });
    engine.register_fn("get_weather", |city: String| -> String {
        format!("Погода в городе {}: солнечно, +22°C (заглушка)", city)
    });
    engine.register_fn("slice", |s: String, start: i64, len: i64| -> String {
        let chars: Vec<char> = s.chars().collect();
        let start_idx = if start < 0 {
            (chars.len() as i64 + start).max(0) as usize
        } else {
            start as usize
        };
        chars
            .into_iter()
            .skip(start_idx)
            .take(len.max(0) as usize)
            .collect()
    });
    engine.register_fn("join", |arr: rhai::Array, sep: String| -> String {
        arr.iter()
            .map(|v| {
                v.clone()
                    .try_cast::<rhai::ImmutableString>()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| v.to_string())
            })
            .collect::<Vec<_>>()
            .join(&sep)
    });
}

#[allow(clippy::too_many_arguments)]
pub fn register_board_functions(
    engine: &mut Engine,
    board_url: String,
    board_token: String,
    self_agent_name: String,
    self_session_id: Option<String>,
    project_id: String,
    parent_chain: Vec<String>,
    pending_calls: PendingCalls,
    agent_call_timeout_sec: u64,
    posted_flag: Arc<AtomicBool>,
    outgoing_tasks: OutgoingTasks,
) {
    // post_task — асинхронный
    {
        let url = board_url.clone();
        let token = board_token.clone();
        let agent = self_agent_name.clone();
        let sess = self_session_id.clone();
        let pid = project_id.clone();
        let flag = posted_flag.clone();
        let outgoing = outgoing_tasks.clone();
        let chain = parent_chain.clone();
        engine.register_fn(
            "post_task",
            move |to_agent: String, prompt: String| -> String {
                if to_agent == agent {
                    return format!("Ошибка: агент '{}' не может вызывать сам себя", to_agent);
                }
                if chain.contains(&to_agent) {
                    return format!(
                        "Ошибка: циклический вызов. Агент '{}' уже в цепочке: {:?}",
                        to_agent, chain
                    );
                }
                let mut new_chain = chain.clone();
                new_chain.push(to_agent.clone());

                let to_session = match sess.as_deref() {
                    Some(s) if !s.is_empty() => s.to_string(),
                    _ => return "Ошибка: агент не привязан к сессии".to_string(),
                };
                let result = post_task_request(
                    &url,
                    &token,
                    &agent,
                    sess.as_deref(),
                    &to_agent,
                    &to_session,
                    &pid,
                    &prompt,
                    &new_chain,
                );
                if let Some(task_id) = result.strip_prefix("posted:") {
                    let handle = Handle::current();
                    let tid = task_id.to_string();
                    let to_a = to_agent.clone();
                    let to_s = to_session.clone();
                    let from_s = sess.clone().unwrap_or_default();
                    let pid_for_outgoing = pid.clone();
                    let outgoing = outgoing.clone();
                    let chain_for_outgoing = new_chain.clone();
                    handle.block_on(async move {
                        let mut map = outgoing.lock().await;
                        map.insert(
                            tid.clone(),
                            OutgoingTask {
                                task_id: tid,
                                session_id: from_s,
                                to_agent_name: to_a,
                                to_session_id: to_s,
                                project_id: pid_for_outgoing,
                                chain: chain_for_outgoing,
                            },
                        );
                    });
                    flag.store(true, Ordering::Relaxed);
                }
                result
            },
        );
    }

    // call_agent — синхронный
    {
        let url = board_url.clone();
        let token = board_token.clone();
        let agent = self_agent_name.clone();
        let sess = self_session_id.clone();
        let pid = project_id.clone();
        let pc = pending_calls.clone();
        let timeout_sec = agent_call_timeout_sec;
        let chain = parent_chain.clone();
        engine.register_fn(
            "call_agent",
            move |to_agent: String, prompt: String| -> String {
                if to_agent == agent {
                    return format!("Ошибка: агент '{}' не может вызывать сам себя", to_agent);
                }
                if chain.contains(&to_agent) {
                    return format!(
                        "Ошибка: циклический вызов. Агент '{}' уже в цепочке: {:?}",
                        to_agent, chain
                    );
                }
                let mut new_chain = chain.clone();
                new_chain.push(to_agent.clone());

                let to_session = match sess.as_deref() {
                    Some(s) if !s.is_empty() => s.to_string(),
                    _ => return "Ошибка: агент не привязан к сессии".to_string(),
                };
                let result = post_task_request(
                    &url,
                    &token,
                    &agent,
                    sess.as_deref(),
                    &to_agent,
                    &to_session,
                    &pid,
                    &prompt,
                    &new_chain,
                );
                let task_id = match result.strip_prefix("posted:") {
                    Some(id) => id.to_string(),
                    None => return result,
                };

                let (tx, rx) = tokio::sync::oneshot::channel::<String>();
                {
                    let handle = Handle::current();
                    let created_at = std::time::Instant::now();
                    let tid = task_id.clone();
                    handle.block_on(async {
                        let mut map = pc.lock().await;
                        map.insert(tid, crate::agent_core::PendingCall { tx, created_at });
                    });
                }

                let url_for_fail = url.clone();
                let token_for_fail = token.clone();
                let outcome = {
                    let handle = Handle::current();
                    let pc_cleanup = pc.clone();
                    let tid = task_id.clone();
                    handle.block_on(async move {
                        match tokio::time::timeout(Duration::from_secs(timeout_sec), rx).await {
                            Ok(Ok(s)) => Ok(s),
                            Ok(Err(_)) => Err("канал ожидания закрыт".to_string()),
                            Err(_) => {
                                {
                                    let mut map = pc_cleanup.lock().await;
                                    map.remove(&tid);
                                }
                                let body = serde_json::json!({
                                    "error": format!(
                                        "call_agent timeout ({} sec)",
                                        timeout_sec
                                    )
                                });
                                let _ = reqwest::blocking::Client::new()
                                    .post(format!(
                                        "{}/tasks/{}/fail",
                                        url_for_fail.trim_end_matches('/'),
                                        tid
                                    ))
                                    .header("Authorization", format!("Bearer {}", token_for_fail))
                                    .json(&body)
                                    .send();
                                Err(format!("таймаут ожидания ответа ({} сек)", timeout_sec))
                            }
                        }
                    })
                };

                match outcome {
                    Ok(s) => s,
                    Err(e) => format!("Ошибка: {}", e),
                }
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn post_task_request(
    url: &str,
    token: &str,
    from_agent: &str,
    from_session: Option<&str>,
    to_agent: &str,
    to_session: &str,
    project_id: &str,
    prompt: &str,
    chain: &[String],
) -> String {
    let client = reqwest::blocking::Client::new();
    let body = serde_json::json!({
        "from_agent": from_agent,
        "from_session_id": from_session.unwrap_or(""),
        "to_agent": to_agent,
        "to_session_id": to_session,
        "project_id": project_id,
        "payload": { "prompt": prompt },
        "parent_task_id": null,
        "chain": chain
    });
    match client
        .post(format!("{}/tasks", url.trim_end_matches('/')))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
    {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>() {
            Ok(v) => {
                if let Some(id) = v.get("task_id").and_then(|x| x.as_str()) {
                    format!("posted:{}", id)
                } else {
                    "Ошибка: нет task_id в ответе".to_string()
                }
            }
            Err(e) => format!("Ошибка парсинга ответа: {}", e),
        },
        Ok(r) => format!(
            "HTTP ошибка {}: {}",
            r.status(),
            r.text().unwrap_or_default()
        ),
        Err(e) => format!("Ошибка запроса к доске: {}", e),
    }
}

pub fn register_skill_functions(
    engine: &mut Engine,
    storage_base_url: &str,
    storage_auth_token: &str,
    config: &AgentConfig,
) {
    let base_url = storage_base_url.to_string();
    let auth_token = storage_auth_token.to_string();
    let agent_name = config.name.clone();
    let min_code_len = config.skill_min_code_length;
    let semantic_threshold = config.skill_semantic_threshold;
    let client = reqwest::blocking::Client::new();

    {
        let c = client.clone();
        let b = base_url.clone();
        let t = auth_token.clone();
        let a = agent_name.clone();
        let th = semantic_threshold;
        engine.register_fn("skill_search", move |query: String| -> String {
            match crate::skill_manager::search_best_skill(&c, &b, &t, &a, &query, th, 5) {
                Ok(Some((rec, score))) => format!("{} (score: {:.2})", rec.description, score),
                Ok(None) => "Скилл не найден".to_string(),
                Err(e) => format!("Ошибка поиска: {}", e),
            }
        });
    }

    {
        let c = client.clone();
        let b = base_url.clone();
        let t = auth_token.clone();
        engine.register_fn("skill_load", move |name: String| -> String {
            let catalog_json = crate::skill_manager::list_skills(&c, &b, &t).unwrap_or_default();
            let catalog: crate::skill_manager::SkillCatalog =
                serde_json::from_str(&catalog_json).unwrap_or_default();
            if let Some(rec) = catalog.skills.iter().find(|r| r.skill_file.contains(&name)) {
                crate::skill_manager::load_skill(&c, &b, &t, &rec.skill_file)
                    .unwrap_or_else(|e| format!("Ошибка: {}", e))
            } else {
                format!("Скилл '{}' не найден", name)
            }
        });
    }

    {
        let c = client.clone();
        let b = base_url.clone();
        let t = auth_token.clone();
        let a = agent_name.clone();
        let min_len = min_code_len;
        engine.register_fn(
            "skill_save",
            move |skill_name: String,
                  description: String,
                  prompt: String,
                  rhai_code: String|
                  -> String {
                match crate::skill_manager::save_skill(
                    &c,
                    &b,
                    &t,
                    &skill_name,
                    &a,
                    &description,
                    &prompt,
                    &rhai_code,
                    min_len,
                ) {
                    Ok(_) => "OK".to_string(),
                    Err(e) => format!("Ошибка сохранения: {}", e),
                }
            },
        );
    }

    {
        let c = client.clone();
        let b = base_url.clone();
        let t = auth_token.clone();
        engine.register_fn(
            "skill_record_usage",
            move |skill_name: String, success: bool| -> String {
                let catalog_json =
                    crate::skill_manager::list_skills(&c, &b, &t).unwrap_or_default();
                let catalog: crate::skill_manager::SkillCatalog =
                    serde_json::from_str(&catalog_json).unwrap_or_default();
                if let Some(rec) = catalog
                    .skills
                    .iter()
                    .find(|r| r.skill_file.contains(&skill_name))
                {
                    match crate::skill_manager::record_usage(&c, &b, &t, &rec.skill_file, success) {
                        Ok(_) => "OK".to_string(),
                        Err(e) => format!("Ошибка: {}", e),
                    }
                } else {
                    "Скилл не найден".to_string()
                }
            },
        );
    }

    {
        let c = client.clone();
        let b = base_url.clone();
        let t = auth_token.clone();
        engine.register_fn("skill_list", move || -> String {
            crate::skill_manager::list_skills(&c, &b, &t)
                .unwrap_or_else(|e| format!("Ошибка: {}", e))
        });
    }
}

pub async fn execute_tool(
    name: &str,
    args: &str,
    http_client: Option<&reqwest::Client>,
    storage_base_url: &str,
    storage_auth_token: &str,
    project_id: &str,
    rhai_timeout_sec: u64,
    board_ctx: Option<BoardContext>,
) -> String {
    match name {
        "run_code" => {
            let parsed: serde_json::Value =
                serde_json::from_str(args).unwrap_or(json!({"code": ""}));
            let code = parsed
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let base_url = storage_base_url.to_string();
            let token = storage_auth_token.to_string();
            let pid = project_id.to_string();
            let timeout_duration = Duration::from_secs(rhai_timeout_sec);

            match tokio::time::timeout(
                timeout_duration,
                tokio::task::spawn_blocking(move || {
                    run_rhai_code(&code, &base_url, &token, &pid, board_ctx)
                }),
            )
            .await
            {
                Ok(Ok(result)) => result,
                Ok(Err(e)) => format!("Ошибка выполнения Rhai-кода: {}", e),
                Err(_) => "Ошибка: превышено время выполнения Rhai-кода".to_string(),
            }
        }
        "local_storage" => {
            if let Some(client) = http_client {
                execute_local_storage_tool(
                    client,
                    storage_base_url,
                    storage_auth_token,
                    project_id,
                    args,
                )
                .await
            } else {
                "Ошибка: HTTP-клиент не инициализирован".to_string()
            }
        }
        _ => format!("Ошибка: неизвестный инструмент {}", name),
    }
}

fn run_rhai_code(
    code: &str,
    storage_base_url: &str,
    storage_auth_token: &str,
    project_id: &str,
    board_ctx: Option<BoardContext>,
) -> String {
    let mut engine = Engine::new();
    engine.set_max_operations(1_000_000);
    engine.set_max_call_levels(64);
    engine.set_max_string_size(1024 * 256);

    let output = Rc::new(RefCell::new(String::new()));
    let output_clone = output.clone();
    engine.on_print(move |s| output_clone.borrow_mut().push_str(s));

    register_basic_functions(&mut engine);
    register_storage_functions(
        &mut engine,
        storage_base_url,
        storage_auth_token,
        project_id,
    );

    if let Some(bc) = board_ctx {
        register_board_functions(
            &mut engine,
            bc.board_url,
            bc.board_token,
            bc.self_agent_name,
            bc.self_session_id,
            bc.project_id,
            bc.parent_chain,
            bc.pending_calls,
            bc.agent_call_timeout_sec,
            bc.posted_flag,
            bc.outgoing_tasks,
        );
    }

    match engine.eval::<Dynamic>(code) {
        Ok(result) => {
            let printed = output.borrow().clone();
            if !printed.is_empty() {
                printed
            } else {
                result.to_string()
            }
        }
        Err(e) => format!("Ошибка выполнения Rhai-кода: {}", e),
    }
}

fn storage_request(
    client: &Client,
    method: Method,
    url: String,
    token: &str,
    project_id: &str,
    query: Vec<(&str, String)>,
    body: Option<String>,
) -> String {
    let mut req = client
        .request(method, &url)
        .header("Authorization", format!("Bearer {}", token))
        .header("X-NH-Project", project_id);
    if !query.is_empty() {
        req = req.query(&query);
    }
    if let Some(b) = body {
        req = req.body(b);
    }
    match req.send() {
        Ok(resp) if resp.status().is_success() => {
            let text = resp.text().unwrap_or_default();
            if text.is_empty() {
                "OK".to_string()
            } else {
                text
            }
        }
        Ok(resp) => format!(
            "HTTP ошибка {}: {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ),
        Err(e) => format!("Ошибка запроса: {}", e),
    }
}

macro_rules! register_storage_fn {
    ($engine:expr, $client:expr, $base:expr, $token:expr, $project:expr,
     $name:expr, $method:expr, $path:expr,
     $($arg:ident : $ty:ty => $q:expr),* $(,)?) => {
        {
            let c = $client.clone();
            let b = $base.clone();
            let t = $token.clone();
            let p = $project.clone();
            $engine.register_fn($name, move |$($arg: $ty),*| -> String {
                let mut query = Vec::new();
                $( query.push($q); )*
                storage_request(&c, $method, format!("{}{}", b.trim_end_matches('/'), $path), &t, &p, query, None)
            });
        }
    };
}

pub fn register_storage_functions(
    engine: &mut Engine,
    base_url: &str,
    auth_token: &str,
    project_id: &str,
) {
    let base = if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    };
    let token = auth_token.to_string();
    let project = project_id.to_string();
    let client = Client::new();

    register_storage_fn!(engine, client, base, token, project,
        "storage_read_file", Method::GET, "/files",
        path: String => ("path", path));

    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        let p = project.clone();
        engine.register_fn(
            "storage_write_file",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/files", b.trim_end_matches('/')),
                    &t,
                    &p,
                    vec![("path", path)],
                    Some(content),
                )
            },
        );
    }

    register_storage_fn!(engine, client, base, token, project,
        "storage_delete_file", Method::DELETE, "/files",
        path: String => ("path", path));

    register_storage_fn!(engine, client, base, token, project,
        "storage_create_dir", Method::POST, "/dirs",
        path: String => ("path", path));

    register_storage_fn!(engine, client, base, token, project,
        "storage_list_dir", Method::GET, "/list",
        path: String => ("path", path));

    register_storage_fn!(engine, client, base, token, project,
        "storage_walk", Method::GET, "/walk",
        path: String => ("path", path));

    register_storage_fn!(engine, client, base, token, project,
        "storage_search_by_name", Method::GET, "/search_name",
        pattern: String => ("pattern", pattern));

    register_storage_fn!(engine, client, base, token, project,
        "storage_search_similar", Method::GET, "/search",
        query: String => ("query", query),
        top_k: i64 => ("top_k", top_k.to_string()));

    register_storage_fn!(engine, client, base, token, project,
        "storage_read_about", Method::GET, "/about",
        path: String => ("path", path));

    register_storage_fn!(engine, client, base, token, project,
        "storage_read_summary", Method::GET, "/summary",
        path: String => ("path", path));

    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        let p = project.clone();
        engine.register_fn(
            "storage_write_about",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/about", b.trim_end_matches('/')),
                    &t,
                    &p,
                    vec![("path", path)],
                    Some(content),
                )
            },
        );
    }

    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        let p = project.clone();
        engine.register_fn(
            "storage_write_summary",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/summary", b.trim_end_matches('/')),
                    &t,
                    &p,
                    vec![("path", path)],
                    Some(content),
                )
            },
        );
    }
}

async fn execute_local_storage_tool(
    client: &reqwest::Client,
    base_url: &str,
    auth_token: &str,
    project_id: &str,
    args: &str,
) -> String {
    let base_url = if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    };

    let params: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(e) => return format!("Ошибка парсинга JSON: {}", e),
    };

    let action = match params.get("action").and_then(|v| v.as_str()) {
        Some(a) => a,
        None => return "Ошибка: не указано поле action".to_string(),
    };

    let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let pattern = params.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
    let top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5);

    let endpoint = endpoint_for_action(action);
    let url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);

    let mut request_builder = match action {
        "write_file" | "create_dir" | "write_about" | "write_summary" => client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth_token))
            .header("X-NH-Project", project_id),
        "delete_file" => client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", auth_token))
            .header("X-NH-Project", project_id),
        _ => client
            .get(&url)
            .header("Authorization", format!("Bearer {}", auth_token))
            .header("X-NH-Project", project_id),
    };

    let mut query_params = Vec::new();
    match action {
        "read_file" | "write_file" | "delete_file" | "create_dir" | "walk" | "list_dir" => {
            query_params.push(("path", path.to_string()));
        }
        "search_by_name" => query_params.push(("pattern", pattern.to_string())),
        "search_similar" => {
            query_params.push(("query", query.to_string()));
            query_params.push(("top_k", top_k.to_string()));
        }
        "read_about" | "read_summary" | "write_about" | "write_summary" => {
            query_params.push(("path", path.to_string()))
        }
        _ => {}
    }
    if !query_params.is_empty() {
        request_builder = request_builder.query(&query_params);
    }

    if matches!(action, "write_file" | "write_about" | "write_summary") {
        request_builder = request_builder.body(content.to_string());
    }

    match request_builder.send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.text().await {
                    Ok(text) => text,
                    Err(e) => format!("Ошибка чтения ответа: {}", e),
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                format!("HTTP ошибка {}: {}", status, text)
            }
        }
        Err(e) => format!("Ошибка запроса к хранилищу: {}", e),
    }
}

fn endpoint_for_action(action: &str) -> &'static str {
    match action {
        "read_file" | "write_file" | "delete_file" => "/files",
        "create_dir" => "/dirs",
        "list_dir" => "/list",
        "walk" => "/walk",
        "search_by_name" => "/search_name",
        "search_similar" => "/search",
        "read_about" | "write_about" => "/about",
        "read_summary" | "write_summary" => "/summary",
        _ => "/",
    }
}
