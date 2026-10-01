// src/tool_runtime/primitives.rs

use serde_json::{json, Value};

/// Извлечь обязательный строковый параметр.
pub fn require_string(args: &Value, name: &str) -> Result<String, String> {
    match args.get(name).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => Ok(s.to_string()),
        Some(_) => Err(format!("параметр '{}' пуст", name)),
        None => Err(format!("не указан обязательный параметр '{}'", name)),
    }
}

/// Извлечь опциональный строковый параметр.
pub fn optional_string(args: &Value, name: &str) -> Option<String> {
    args.get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Извлечь опциональный целочисленный параметр.
pub fn optional_int(args: &Value, name: &str) -> Option<i64> {
    args.get(name).and_then(|v| v.as_i64())
}

/// Распарсить входную JSON-строку в Value. Пустая строка → пустой объект.
pub fn parse_input(input: &str) -> Result<Value, String> {
    if input.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(input).map_err(|e| format!("некорректный JSON: {}", e))
}

/// Успешный ответ инструмента. Оборачивает произвольный текст в JSON.
pub fn ok_text(text: impl Into<String>) -> String {
    json!({ "ok": true, "content": text.into() }).to_string()
}

/// Ошибка инструмента. Модель увидит этот JSON и поймёт, что пошло не так.
pub fn err(message: impl Into<String>) -> String {
    json!({ "ok": false, "error": message.into() }).to_string()
}

/// Ответ с произвольным JSON-объектом внутри.
pub fn ok_json(value: Value) -> String {
    json!({ "ok": true, "data": value }).to_string()
}
