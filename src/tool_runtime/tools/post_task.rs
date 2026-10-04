// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::agent_core::OutgoingTask;
use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct PostTask;

#[async_trait::async_trait]
impl ToolImpl for PostTask {
    fn name(&self) -> &'static str {
        "post_task"
    }

    fn description(&self) -> &'static str {
        "Асинхронно публикует задачу другому агенту. Возвращает 'posted:<task_id>'. \
         После вызова заверши ход — продолжение произойдёт, когда агент ответит."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Имя агента из конфига"},
                "prompt": {"type": "string", "description": "Текст задачи"}
            },
            "required": ["to_agent", "prompt"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(e),
        };
        let to_agent = match require_string(&args, "to_agent") {
            Ok(a) => a,
            Err(e) => return err(e),
        };
        let prompt = match require_string(&args, "prompt") {
            Ok(p) => p,
            Err(e) => return err(e),
        };

        if to_agent == ctx.self_agent_name {
            return err(format!("агент '{}' не может вызывать сам себя", to_agent));
        }
        if ctx.parent_chain.contains(&to_agent) {
            return err(format!(
                "циклический вызов: агент '{}' уже в цепочке {:?}",
                to_agent, ctx.parent_chain
            ));
        }

        let mut new_chain = ctx.parent_chain.clone();
        new_chain.push(to_agent.clone());

        let url = format!("{}/tasks", ctx.board_base());
        let body = serde_json::json!({
            "from_agent": ctx.self_agent_name,
            "from_session_id": ctx.session_id.clone().unwrap_or_default(),
            "to_agent": &to_agent,
            "to_session_id": ctx.session_id.clone().unwrap_or_default(),
            "project_id": ctx.project_id,
            "payload": { "prompt": &prompt },
            "parent_task_id": null,
            "chain": &new_chain
        });

        let resp = match ctx
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", ctx.board_auth_token))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return err(format!("ошибка запроса к доске: {:#}", e)),
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return err(format!("HTTP {} — {}", status, text));
        }

        let v: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return err(format!("ошибка парсинга: {:#}", e)),
        };
        let task_id = match v.get("task_id").and_then(|x| x.as_str()) {
            Some(s) => s.to_string(),
            None => return err("в ответе нет task_id"),
        };

        {
            let mut map = ctx.outgoing_tasks.lock().await;
            map.insert(
                task_id.clone(),
                OutgoingTask {
                    task_id: task_id.clone(),
                    session_id: ctx.session_id.clone().unwrap_or_default(),
                    to_agent_name: to_agent,
                    to_session_id: ctx.session_id.clone().unwrap_or_default(),
                    project_id: ctx.project_id.clone(),
                    chain: new_chain,
                },
            );
        }

        ok_text(format!("posted:{}", task_id))
    }
}

inventory::submit! { &PostTask as &'static dyn crate::tool_runtime::ToolImpl }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_runtime::context::ToolContext;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn ctx_with(self_name: &str, chain: Vec<String>) -> ToolContext {
        ToolContext {
            http_client: reqwest::Client::new(),
            storage_root_path: std::path::PathBuf::from("/tmp/nh_test"),
            storage_base_url: "http://127.0.0.1:1".to_string(),
            storage_auth_token: String::new(),
            board_base_url: "http://127.0.0.1:1".to_string(),
            board_auth_token: String::new(),
            project_id: "test".to_string(),
            session_id: Some("s1".to_string()),
            parent_chain: chain,
            self_agent_name: self_name.to_string(),
            pending_calls: Arc::new(Mutex::new(HashMap::new())),
            outgoing_tasks: Arc::new(Mutex::new(HashMap::new())),
            agent_call_timeout_sec: 1,
        }
    }

    #[tokio::test]
    async fn rejects_invalid_json() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec![]);
        let result = tool.run("{not json", &ctx).await;
        assert!(result.contains("\"ok\":false"));
        assert!(result.contains("некорректный JSON"));
    }

    #[tokio::test]
    async fn rejects_missing_to_agent() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec![]);
        let result = tool.run(r#"{"prompt":"hi"}"#, &ctx).await;
        assert!(result.contains("\"ok\":false"));
        assert!(result.contains("to_agent"));
    }

    #[tokio::test]
    async fn rejects_missing_prompt() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec![]);
        let result = tool.run(r#"{"to_agent":"b"}"#, &ctx).await;
        assert!(result.contains("\"ok\":false"));
        assert!(result.contains("prompt"));
    }

    #[tokio::test]
    async fn rejects_self_call() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec![]);
        let result = tool.run(r#"{"to_agent":"a","prompt":"hi"}"#, &ctx).await;
        assert!(result.contains("\"ok\":false"));
        assert!(result.contains("не может вызывать сам себя"));
    }

    #[tokio::test]
    async fn rejects_cyclic_call() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec!["b".to_string(), "c".to_string()]);
        let result = tool.run(r#"{"to_agent":"c","prompt":"hi"}"#, &ctx).await;
        assert!(result.contains("\"ok\":false"));
        assert!(result.contains("циклический вызов"));
    }

    #[tokio::test]
    async fn board_unreachable_returns_error() {
        let tool = PostTask;
        let ctx = ctx_with("a", vec![]);
        let result = tool.run(r#"{"to_agent":"b","prompt":"hi"}"#, &ctx).await;
        assert!(result.contains("\"ok\":false"));
    }
}
