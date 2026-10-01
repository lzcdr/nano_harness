use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageWriteAbout;

#[async_trait::async_trait]
impl ToolImpl for StorageWriteAbout {
    fn name(&self) -> &'static str { "storage_write_about" }

    fn description(&self) -> &'static str {
        "Записывает файл .about указанной директории. Возвращает 'OK' или JSON с 'error'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к директории"},
                "content": {"type": "string", "description": "Текст описания"}
            },
            "required": ["path", "content"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) { Ok(v) => v, Err(e) => return err(e) };
        let path = match require_string(&args, "path") { Ok(p) => p, Err(e) => return err(e) };
        let content = match require_string(&args, "content") { Ok(c) => c, Err(e) => return err(e) };

        let url = format!("{}/about", ctx.storage_base());
        let resp = ctx.http_client
            .post(&url)
            .query(&[("path", &path)])
            .header("Authorization", format!("Bearer {}", ctx.storage_auth_token))
            .header("X-NH-Project", &ctx.project_id)
            .body(content)
            .send().await;

        match resp {
            Ok(r) if r.status().is_success() => ok_text("OK"),
            Ok(r) => { let s = r.status(); err(format!("HTTP {} — {}", s, r.text().await.unwrap_or_default())) }
            Err(e) => err(format!("ошибка запроса: {:#}", e)),
        }
    }
}

inventory::submit! { &StorageWriteAbout as &'static dyn crate::tool_runtime::ToolImpl }
