// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_string_ok() {
        let args = json!({"name": "hello"});
        assert_eq!(require_string(&args, "name").unwrap(), "hello");
    }

    #[test]
    fn require_string_missing_errors() {
        let args = json!({});
        assert!(require_string(&args, "name").is_err());
    }

    #[test]
    fn require_string_empty_errors() {
        let args = json!({"name": ""});
        assert!(require_string(&args, "name").is_err());
    }

    #[test]
    fn require_string_non_string_errors() {
        let args = json!({"name": 42});
        assert!(require_string(&args, "name").is_err());
    }

    #[test]
    fn optional_string_some() {
        let args = json!({"name": "x"});
        assert_eq!(optional_string(&args, "name"), Some("x".to_string()));
    }

    #[test]
    fn optional_string_none_when_missing() {
        let args = json!({});
        assert_eq!(optional_string(&args, "name"), None);
    }

    #[test]
    fn optional_string_none_when_wrong_type() {
        let args = json!({"name": 42});
        assert_eq!(optional_string(&args, "name"), None);
    }

    #[test]
    fn optional_int_some() {
        let args = json!({"n": 5});
        assert_eq!(optional_int(&args, "n"), Some(5));
    }

    #[test]
    fn optional_int_none_when_missing() {
        let args = json!({});
        assert_eq!(optional_int(&args, "n"), None);
    }

    #[test]
    fn optional_int_none_when_wrong_type() {
        let args = json!({"n": "5"});
        assert_eq!(optional_int(&args, "n"), None);
    }

    #[test]
    fn parse_input_empty_returns_empty_object() {
        let v = parse_input("").unwrap();
        assert!(v.is_object());
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn parse_input_whitespace_returns_empty_object() {
        let v = parse_input("   \n\t  ").unwrap();
        assert!(v.is_object());
    }

    #[test]
    fn parse_input_valid_json() {
        let v = parse_input(r#"{"a":1}"#).unwrap();
        assert_eq!(v.get("a").and_then(|x| x.as_i64()), Some(1));
    }

    #[test]
    fn parse_input_invalid_json_errors() {
        assert!(parse_input("{not json").is_err());
    }

    #[test]
    fn ok_text_wraps_string() {
        let s = ok_text("hello");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["content"], "hello");
    }

    #[test]
    fn err_wraps_message() {
        let s = err("bad");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"], "bad");
    }

    #[test]
    fn ok_json_wraps_value() {
        let s = ok_json(json!({"x": 1}));
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["data"]["x"], 1);
    }
}
