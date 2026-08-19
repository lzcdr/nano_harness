use crate::engine::{FunctionDefinition, ToolDefinition};
use rhai::{Dynamic, Engine};

// Единственный доступный инструмент: run_code (выполняет Rhai-скрипт)
pub fn available_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "run_code".to_string(),
            description: "Выполняет код на языке Rhai и возвращает результат. \
                          В коде доступны функции: get_time() и get_weather(city). \
                          Код должен быть корректным выражением Rhai. \
                          Код должен возвращать строку."
                .to_string(),
            parameters: serde_json::json!({
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
    }]
}

// Обработчик инструмента
pub fn execute_tool(name: &str, args: &str) -> String {
    match name {
        "run_code" => {
            let parsed: serde_json::Value =
                serde_json::from_str(args).unwrap_or(serde_json::json!({"code": ""}));
            let code = parsed.get("code").and_then(|v| v.as_str()).unwrap_or("");
            run_rhai_code(code)
        }
        _ => format!("Ошибка: неизвестный инструмент {}", name),
    }
}

// Выполняет Rhai-код с зарегистрированными внешними функциями
fn run_rhai_code(code: &str) -> String {
    let mut engine = Engine::new();

    engine.set_max_operations(10_000);
    engine.set_max_call_levels(32);
    engine.set_max_string_size(1024 * 10);

    // Регистрируем функцию get_time (без аргументов)
    engine.register_fn("get_time", || -> String {
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    });

    // Регистрируем функцию get_weather (принимает строку с городом)
    engine.register_fn("get_weather", |city: String| -> String {
        format_weather(&city)
    });

    // Выполняем код
    match engine.eval::<Dynamic>(code) {
        Ok(result) => result.to_string(),
        Err(e) => format!("Ошибка выполнения Rhai-кода: {}", e),
    }
}

// Заглушка для получения погоды (замените на реальный API при необходимости)
fn format_weather(city: &str) -> String {
    format!("Погода в городе {}: солнечно, +22°C (заглушка)", city)
}
