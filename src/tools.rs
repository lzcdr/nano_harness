use crate::engine::{FunctionDefinition, ToolDefinition};
use reqwest::Client;
use rhai::{Dynamic, Engine};
use serde_json::json;

pub fn available_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "run_code".to_string(),
                description: "Выполняет код на языке Rhai и возвращает результат. \
                              В коде доступны функции: get_time() и get_weather(city). \
                              Код должен быть корректным выражением Rhai. \
                              Код должен возвращать строку."
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

pub async fn execute_tool(
    name: &str,
    args: &str,
    http_client: Option<&Client>,
    storage_base_url: &str,
    storage_auth_token: &str,
) -> String {
    match name {
        "run_code" => {
            let parsed: serde_json::Value =
                serde_json::from_str(args).unwrap_or(json!({"code": ""}));
            let code = parsed.get("code").and_then(|v| v.as_str()).unwrap_or("");
            run_rhai_code(code)
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

async fn execute_local_storage_tool(
    client: &Client,
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

    // Добавляем query-параметры
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

    // Тело для записей
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

fn run_rhai_code(code: &str) -> String {
    let mut engine = Engine::new();

    engine.set_max_operations(10_000);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1024 * 10);

    engine.register_fn("get_time", || -> String {
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    });

    engine.register_fn("get_weather", |city: String| -> String {
        format!("Погода в городе {}: солнечно, +22°C (заглушка)", city)
    });

    match engine.eval::<Dynamic>(code) {
        Ok(result) => result.to_string(),
        Err(e) => format!("Ошибка выполнения Rhai-кода: {}", e),
    }
}
