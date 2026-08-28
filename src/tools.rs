// src/tools.rs

use crate::engine::{FunctionDefinition, ToolDefinition};
use reqwest::blocking::Client;
use reqwest::Method;
use rhai::{Dynamic, Engine};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

pub fn available_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "run_code".to_string(),
                description: "Выполняет код на языке Rhai и возвращает результат. \
                              В коде доступны функции: get_time() и get_weather(city), \
                              а также функции хранилища: storage_read_file, storage_write_file, \
                              storage_delete_file, storage_create_dir, storage_list_dir, \
                              storage_walk, storage_search_by_name, storage_search_similar, \
                              storage_read_about, storage_write_about, storage_read_summary, \
                              storage_write_summary."
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
                        "action": {
                            "type": "string",
                            "enum": [
                                "read_file", "write_file", "delete_file", "create_dir",
                                "list_dir", "walk", "search_by_name", "search_similar",
                                "write_about", "read_about", "write_summary", "read_summary"
                            ],
                            "description": "Действие"
                        },
                        "path": {"type": "string", "description": "Путь (если требуется)"},
                        "content": {"type": "string", "description": "Содержимое (для записи)"},
                        "query": {"type": "string", "description": "Поисковый запрос"},
                        "pattern": {"type": "string", "description": "Шаблон имени"},
                        "top_k": {"type": "integer", "description": "Количество результатов поиска"}
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
}

pub async fn execute_tool(
    name: &str,
    args: &str,
    http_client: Option<&reqwest::Client>,
    storage_base_url: &str,
    storage_auth_token: &str,
    rhai_timeout_sec: u64,
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
            let timeout_duration = Duration::from_secs(rhai_timeout_sec);

            match tokio::time::timeout(
                timeout_duration,
                tokio::task::spawn_blocking(move || run_rhai_code(&code, &base_url, &token)),
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
                execute_local_storage_tool(client, storage_base_url, storage_auth_token, args).await
            } else {
                "Ошибка: HTTP-клиент не инициализирован".to_string()
            }
        }
        _ => format!("Ошибка: неизвестный инструмент {}", name),
    }
}

fn run_rhai_code(code: &str, storage_base_url: &str, storage_auth_token: &str) -> String {
    let mut engine = Engine::new();
    engine.set_max_operations(10_000);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1024 * 10);

    // Перехват вывода print
    let output = Rc::new(RefCell::new(String::new()));
    let output_clone = output.clone();
    engine.on_print(move |s| output_clone.borrow_mut().push_str(s));

    register_basic_functions(&mut engine);
    register_storage_functions(&mut engine, storage_base_url, storage_auth_token);

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
    query: Vec<(&str, String)>,
    body: Option<String>,
) -> String {
    let mut req = client
        .request(method, &url)
        .header("Authorization", format!("Bearer {}", token));
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
    ($engine:expr, $client:expr, $base:expr, $token:expr,
     $name:expr, $method:expr, $path:expr,
     $($arg:ident : $ty:ty => $q:expr),* $(,)?) => {
        {
            let c = $client.clone();
            let b = $base.clone();
            let t = $token.clone();
            $engine.register_fn($name, move |$($arg: $ty),*| -> String {
                let mut query = Vec::new();
                $( query.push($q); )*
                storage_request(&c, $method, format!("{}{}", b.trim_end_matches('/'), $path), &t, query, None)
            });
        }
    };
}

pub fn register_storage_functions(engine: &mut Engine, base_url: &str, auth_token: &str) {
    let base = if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("http://{}", base_url)
    };
    let token = auth_token.to_string();
    let client = Client::new();

    // GET /files
    register_storage_fn!(engine, client, base, token,
        "storage_read_file", Method::GET, "/files",
        path: String => ("path", path));

    // POST /files (с телом)
    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        engine.register_fn(
            "storage_write_file",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/files", b.trim_end_matches('/')),
                    &t,
                    vec![("path", path)],
                    Some(content),
                )
            },
        );
    }

    // DELETE /files
    register_storage_fn!(engine, client, base, token,
        "storage_delete_file", Method::DELETE, "/files",
        path: String => ("path", path));

    // POST /dirs
    register_storage_fn!(engine, client, base, token,
        "storage_create_dir", Method::POST, "/dirs",
        path: String => ("path", path));

    // GET /list
    register_storage_fn!(engine, client, base, token,
        "storage_list_dir", Method::GET, "/list",
        path: String => ("path", path));

    // GET /walk
    register_storage_fn!(engine, client, base, token,
        "storage_walk", Method::GET, "/walk",
        path: String => ("path", path));

    // GET /search_name
    register_storage_fn!(engine, client, base, token,
        "storage_search_by_name", Method::GET, "/search_name",
        pattern: String => ("pattern", pattern));

    // GET /search
    register_storage_fn!(engine, client, base, token,
        "storage_search_similar", Method::GET, "/search",
        query: String => ("query", query),
        top_k: i64 => ("top_k", top_k.to_string()));

    // GET /about
    register_storage_fn!(engine, client, base, token,
        "storage_read_about", Method::GET, "/about",
        path: String => ("path", path));

    // POST /about
    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        engine.register_fn(
            "storage_write_about",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/about", b.trim_end_matches('/')),
                    &t,
                    vec![("path", path)],
                    Some(content),
                )
            },
        );
    }

    // GET /summary
    register_storage_fn!(engine, client, base, token,
        "storage_read_summary", Method::GET, "/summary",
        path: String => ("path", path));

    // POST /summary
    {
        let c = client.clone();
        let b = base.clone();
        let t = token.clone();
        engine.register_fn(
            "storage_write_summary",
            move |path: String, content: String| -> String {
                storage_request(
                    &c,
                    Method::POST,
                    format!("{}/summary", b.trim_end_matches('/')),
                    &t,
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
            .header("Authorization", format!("Bearer {}", auth_token)),
        "delete_file" => client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", auth_token)),
        _ => client
            .get(&url)
            .header("Authorization", format!("Bearer {}", auth_token)),
    };

    let mut query_params = Vec::new();
    match action {
        "read_file" | "write_file" | "delete_file" | "create_dir" | "walk" => {
            query_params.push(("path", path.to_string()));
        }
        "search_by_name" => {
            query_params.push(("pattern", pattern.to_string()));
        }
        "search_similar" => {
            query_params.push(("query", query.to_string()));
            query_params.push(("top_k", top_k.to_string()));
        }
        "read_about" | "read_summary" => {
            query_params.push(("path", path.to_string()));
        }
        "write_about" | "write_summary" => {
            query_params.push(("path", path.to_string()));
        }
        "list_dir" => {
            query_params.push(("path", path.to_string()));
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
