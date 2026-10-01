// src/tool_runtime/tools/storage_read_file.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageReadFile;

#[async_trait::async_trait]
impl ToolImpl for StorageReadFile {
    fn name(&self) -> &'static str {
        "storage_read_file"
    }

    fn description(&self) -> &'static str {
        "Читает содержимое файла в локальном хранилище. Возвращает текст файла. \
         При ошибке (файл не найден, нет доступа) возвращает JSON с полем 'error'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу относительно корня проекта"
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

        let path = match require_string(&args, "path") {
            Ok(p) => p,
            Err(e) => return err(e),
        };

        let url = format!("{}/files", ctx.storage_base());
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
    &StorageReadFile as &'static dyn crate::tool_runtime::ToolImpl
}
