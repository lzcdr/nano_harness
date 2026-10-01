// src/tool_runtime/tools/storage_create_dir.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageCreateDir;

#[async_trait::async_trait]
impl ToolImpl for StorageCreateDir {
    fn name(&self) -> &'static str {
        "storage_create_dir"
    }

    fn description(&self) -> &'static str {
        "Создаёт директорию в локальном хранилище, включая все промежуточные. \
         Возвращает 'OK' при успехе."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
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

        let url = format!("{}/dirs", ctx.storage_base());
        let resp = ctx
            .http_client
            .post(&url)
            .query(&[("path", &path)])
            .header("Authorization", format!("Bearer {}", ctx.storage_auth_token))
            .header("X-NH-Project", &ctx.project_id)
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
    &StorageCreateDir as &'static dyn crate::tool_runtime::ToolImpl
}
