// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/storage_list_dir.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, optional_string, parse_input};
use crate::tool_runtime::ToolImpl;

pub struct StorageListDir;

#[async_trait::async_trait]
impl ToolImpl for StorageListDir {
    fn name(&self) -> &'static str {
        "storage_list_dir"
    }

    fn description(&self) -> &'static str {
        "Возвращает список файлов и поддиректорий в указанной директории (папке, каталоге). \
         Показывает, что находится в папке. \
         Результат — JSON-массив объектов вида {kind: 'file'|'dir', name, path, size?}."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к директории. Пустая строка или '/' — корень проекта."
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(e),
        };
        let path = optional_string(&args, "path").unwrap_or_default();

        let url = format!("{}/list", ctx.storage_base());
        let resp = ctx
            .http_client
            .get(&url)
            .query(&[("path", &path)])
            .header(
                "Authorization",
                format!("Bearer {}", ctx.storage_auth_token),
            )
            .header("X-NH-Project", &ctx.project_id)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => match r.text().await {
                Ok(text) => ok_text(text),
                Err(e) => err(format!("ошибка чтения ответа: {:#}", e)),
            },
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
    &StorageListDir as &'static dyn crate::tool_runtime::ToolImpl
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
        let r = StorageListDir.run("{not json", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn accepts_empty_object() {
        let r = StorageListDir.run("{}", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
        assert!(!r.contains("не указан"));
    }

    #[tokio::test]
    async fn unreachable_storage_returns_error() {
        let r = StorageListDir.run(r#"{"path":"src"}"#, &ctx()).await;
        assert!(r.contains("\"ok\":false"));
    }
}
