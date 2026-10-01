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
