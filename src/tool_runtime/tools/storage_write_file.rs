// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/storage_write_file.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageWriteFile;

#[async_trait::async_trait]
impl ToolImpl for StorageWriteFile {
    fn name(&self) -> &'static str {
        "storage_write_file"
    }

    fn description(&self) -> &'static str {
        "Записывает, создаёт или сохраняет файл (текст, документ) \
         в локальном хранилище. Запиши файл, сохрани в файл, создай файл \
         с содержимым, положи текст в файл. \
         Создаёт промежуточные директории автоматически. \
         Перезаписывает существующий файл. Возвращает 'OK' при успехе."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу относительно корня проекта"
                },
                "content": {
                    "type": "string",
                    "description": "Содержимое файла"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(e),
        };
        let path = match require_string(&args, "path") {
            Ok(p) => p,
            Err(e) => return err(e),
        };
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let url = format!("{}/files", ctx.storage_base());
        let resp = ctx
            .http_client
            .post(&url)
            .query(&[("path", &path)])
            .header(
                "Authorization",
                format!("Bearer {}", ctx.storage_auth_token),
            )
            .header("X-NH-Project", &ctx.project_id)
            .body(content)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => ok_text("OK"),
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                err(format!("HTTP {} — {}", status, body))
            }
            Err(e) => err(format!("ошибка запроса: {:#}", e)),
        }
    }
}

inventory::submit! {
    &StorageWriteFile as &'static dyn crate::tool_runtime::ToolImpl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_runtime::context::ToolContext;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn ctx() -> ToolContext {
        ToolContext {
            http_client: reqwest::Client::new(),
            storage_root_path: std::path::PathBuf::from("/tmp/nh_test"),
            storage_base_url: "http://127.0.0.1:1".to_string(),
            storage_auth_token: String::new(),
            board_base_url: "http://127.0.0.1:1".to_string(),
            board_auth_token: String::new(),
            project_id: "test".to_string(),
            session_id: None,
            parent_chain: vec![],
            self_agent_name: "test".to_string(),
            pending_calls: Arc::new(Mutex::new(HashMap::new())),
            outgoing_tasks: Arc::new(Mutex::new(HashMap::new())),
            agent_call_timeout_sec: 1,
        }
    }

    #[tokio::test]
    async fn rejects_invalid_json() {
        let r = StorageWriteFile.run("{not json", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn rejects_missing_path() {
        let r = StorageWriteFile.run("{}", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
        assert!(r.contains("path"));
    }

    #[tokio::test]
    async fn rejects_empty_path() {
        let r = StorageWriteFile
            .run(r#"{"path":"","content":"x"}"#, &ctx())
            .await;
        assert!(r.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn accepts_missing_content_as_empty() {
        // content не обязателен — пустая строка допустима
        let r = StorageWriteFile.run(r#"{"path":"x"}"#, &ctx()).await;
        assert!(r.contains("\"ok\":false"));
        assert!(!r.contains("content"));
    }

    #[tokio::test]
    async fn unreachable_storage_returns_error() {
        let r = StorageWriteFile
            .run(r#"{"path":"x","content":"y"}"#, &ctx())
            .await;
        assert!(r.contains("\"ok\":false"));
    }
}
