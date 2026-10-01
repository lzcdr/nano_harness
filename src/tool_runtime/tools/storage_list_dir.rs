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
        "Возвращает список файлов и поддиректорий в указанной директории. \
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
    &StorageListDir as &'static dyn crate::tool_runtime::ToolImpl
}
