// src/tool_runtime/tools/storage_search_similar.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{
    err, ok_text, optional_int, parse_input, require_string,
};
use crate::tool_runtime::ToolImpl;

pub struct StorageSearchSimilar;

#[async_trait::async_trait]
impl ToolImpl for StorageSearchSimilar {
    fn name(&self) -> &'static str {
        "storage_search_similar"
    }

    fn description(&self) -> &'static str {
        "Семантический поиск файлов по смыслу запроса. Возвращает JSON-массив \
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
            .header("Authorization", format!("Bearer {}", ctx.storage_auth_token))
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
