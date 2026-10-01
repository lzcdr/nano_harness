// src/tool_runtime/tools/storage_write_file.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageWriteFile;

#[async_trait::async_trait]
impl ToolImpl for StorageWriteFile {
    fn name(&self) -> &'static str {
        "storage_write_file"
    }

    fn description(&self) -> &'static str {
        "Записывает файл в локальное хранилище. Создаёт промежуточные директории \
         автоматически. Перезаписывает существующий файл. Возвращает 'OK' при успехе."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу относительно корня проекта"
                },
                "content": {
                    "type": "string",
                    "description": "Содержимое файла"
                }
            },
            "required": ["path", "content"]
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
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let url = format!("{}/files", ctx.storage_base());
        let resp = ctx
            .http_client
            .post(&url)
            .query(&[("path", &path)])
            .header("Authorization", format!("Bearer {}", ctx.storage_auth_token))
            .header("X-NH-Project", &ctx.project_id)
            .body(content)
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
    &StorageWriteFile as &'static dyn crate::tool_runtime::ToolImpl
}
