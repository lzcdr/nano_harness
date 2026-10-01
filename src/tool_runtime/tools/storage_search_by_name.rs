// src/tool_runtime/tools/storage_search_by_name.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageSearchByName;

#[async_trait::async_trait]
impl ToolImpl for StorageSearchByName {
    fn name(&self) -> &'static str {
        "storage_search_by_name"
    }

    fn description(&self) -> &'static str {
        "Ищет файлы и директории по подстроке в имени. \
         Результат — JSON-массив путей."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string"}
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(e),
        };
        let pattern = match require_string(&args, "pattern") {
            Ok(p) => p,
            Err(e) => return err(e),
        };

        let url = format!("{}/search_name", ctx.storage_base());
        let resp = ctx
            .http_client
            .get(&url)
            .query(&[("pattern", &pattern)])
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
    &StorageSearchByName as &'static dyn crate::tool_runtime::ToolImpl
}
