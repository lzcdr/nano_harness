// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/mod.rs

pub mod context;
pub mod primitives;

pub mod tools;

pub use context::ToolContext;

use crate::engine::{FunctionDefinition, ToolDefinition};

#[async_trait::async_trait]
pub trait ToolImpl: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters(&self) -> serde_json::Value;
    async fn run(&self, input: &str, ctx: &ToolContext) -> String;
}

inventory::collect!(&'static dyn ToolImpl);

/// Все зарегистрированные инструменты.
pub fn all() -> Vec<&'static dyn ToolImpl> {
    inventory::iter::<&'static dyn ToolImpl>
        .into_iter()
        .copied()
        .collect()
}

/// Найти инструмент по имени.
pub fn find(name: &str) -> Option<&'static dyn ToolImpl> {
    inventory::iter::<&'static dyn ToolImpl>
        .into_iter()
        .find(|t| t.name() == name)
        .copied()
}

/// Описания инструментов для LLM. Фильтруются по списку разрешённых.
pub fn available_tools(allowed: Option<&[String]>) -> Vec<ToolDefinition> {
    all()
        .into_iter()
        .filter(|t| match allowed {
            Some(list) => list.iter().any(|n| n == t.name()),
            None => true,
        })
        .map(|t| ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            },
        })
        .collect()
}

/// Выполнить инструмент по имени.
pub async fn execute(name: &str, input: &str, ctx: &ToolContext) -> String {
    match find(name) {
        Some(tool) => tool.run(input, ctx).await,
        None => format!("Ошибка: неизвестный инструмент '{}'", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_registered_tools() {
        let tools = all();
        assert!(!tools.is_empty(), "должен быть хотя бы один инструмент");
    }

    #[test]
    fn all_contains_known_tools() {
        let names: Vec<&str> = all().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"storage_read_file"));
        assert!(names.contains(&"storage_write_file"));
        assert!(names.contains(&"storage_walk"));
        assert!(names.contains(&"ripgrep"));
        assert!(names.contains(&"scc"));
        assert!(names.contains(&"ctags"));
        assert!(names.contains(&"code_index"));
        assert!(names.contains(&"call_agent"));
        assert!(names.contains(&"post_task"));
    }

    #[test]
    fn all_names_are_unique() {
        let names: Vec<&str> = all().iter().map(|t| t.name()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "имена инструментов должны быть уникальны"
        );
    }

    #[test]
    fn find_existing_tool() {
        assert!(find("storage_read_file").is_some());
        assert!(find("call_agent").is_some());
    }

    #[test]
    fn find_unknown_tool_returns_none() {
        assert!(find("no_such_tool_xyz").is_none());
    }

    #[test]
    fn available_tools_with_none_returns_all() {
        let all_tools = available_tools(None);
        assert_eq!(all_tools.len(), all().len());
    }

    #[test]
    fn available_tools_with_empty_filter_returns_empty() {
        let none: Vec<String> = vec![];
        let result = available_tools(Some(&none));
        assert!(result.is_empty());
    }

    #[test]
    fn available_tools_filters_by_name() {
        let filter = vec![
            "storage_read_file".to_string(),
            "storage_write_file".to_string(),
        ];
        let result = available_tools(Some(&filter));
        assert_eq!(result.len(), 2);
        let names: Vec<&str> = result.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"storage_read_file"));
        assert!(names.contains(&"storage_write_file"));
    }

    #[test]
    fn available_tools_ignores_unknown_names() {
        let filter = vec!["storage_read_file".to_string(), "no_such_tool".to_string()];
        let result = available_tools(Some(&filter));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "storage_read_file");
    }

    #[test]
    fn available_tools_returns_function_type() {
        let filter = vec!["storage_read_file".to_string()];
        let result = available_tools(Some(&filter));
        assert_eq!(result[0].tool_type, "function");
    }

    #[test]
    fn available_tools_populates_description_and_parameters() {
        let filter = vec!["storage_read_file".to_string()];
        let result = available_tools(Some(&filter));
        assert!(!result[0].function.description.is_empty());
        assert!(result[0].function.parameters.is_object());
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error_string() {
        let ctx = dummy_context();
        let result = execute("no_such_tool", "{}", &ctx).await;
        assert!(result.contains("неизвестный инструмент"));
        assert!(result.contains("no_such_tool"));
    }

    fn dummy_context() -> ToolContext {
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
            pending_calls: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            outgoing_tasks: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            agent_call_timeout_sec: 5,
        }
    }
}
