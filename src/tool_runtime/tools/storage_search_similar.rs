// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/storage_search_similar.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, optional_int, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageSearchSimilar;

#[async_trait::async_trait]
impl ToolImpl for StorageSearchSimilar {
    fn name(&self) -> &'static str {
        "storage_search_similar"
    }

    fn description(&self) -> &'static str {
        "Семантический поиск файлов. \
         Находит похожие файлы, похожие по смыслу на запрос. \
         Возвращает JSON-массив \
         объектов {file_path, chunk_index, distance, content_fragment, project_id}, \
         отсортированный по возрастанию distance (меньше — ближе)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"}
            },
            "required": ["query"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(e),
        };
        let query = match require_string(&args, "query") {
            Ok(q) => q,
            Err(e) => return err(e),
        };
        let top_k = optional_int(&args, "top_k").unwrap_or(5).max(1);

        let url = format!("{}/search", ctx.storage_base());
        let resp = ctx
            .http_client
            .get(&url)
            .query(&[("query", &query), ("top_k", &top_k.to_string())])
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
    &StorageSearchSimilar as &'static dyn crate::tool_runtime::ToolImpl
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
        let r = StorageSearchSimilar.run("{not json", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn rejects_missing_query() {
        let r = StorageSearchSimilar.run("{}", &ctx()).await;
        assert!(r.contains("\"ok\":false"));
        assert!(r.contains("query"));
    }

    #[tokio::test]
    async fn rejects_empty_query() {
        let r = StorageSearchSimilar.run(r#"{"query":""}"#, &ctx()).await;
        assert!(r.contains("\"ok\":false"));
    }

    #[tokio::test]
    async fn accepts_top_k_optional() {
        let r = StorageSearchSimilar
            .run(r#"{"query":"rust"}"#, &ctx())
            .await;
        // top_k не указан — не должно быть ошибки валидации параметра.
        // HTTP-ошибка к недостижимому порту ожидаема.
        assert!(r.contains("\"ok\":false"));
        assert!(!r.contains("не указан обязательный параметр"));
    }

    #[tokio::test]
    async fn unreachable_storage_returns_error() {
        let r = StorageSearchSimilar
            .run(r#"{"query":"rust"}"#, &ctx())
            .await;
        assert!(r.contains("\"ok\":false"));
    }
}
