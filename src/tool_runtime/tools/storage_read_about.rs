use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;

pub struct StorageReadAbout;

#[async_trait::async_trait]
impl ToolImpl for StorageReadAbout {
    fn name(&self) -> &'static str { "storage_read_about" }

    fn description(&self) -> &'static str {
        "Читает файл .about указанной директории — краткое описание её назначения."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к директории"}
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) { Ok(v) => v, Err(e) => return err(e) };
        let path = match require_string(&args, "path") { Ok(p) => p, Err(e) => return err(e) };

        let url = format!("{}/about", ctx.storage_base());
        let resp = ctx.http_client
            .get(&url)
            .query(&[("path", &path)])
            .header("Authorization", format!("Bearer {}", ctx.storage_auth_token))
            .header("X-NH-Project", &ctx.project_id)
            .send().await;

        match resp {
            Ok(r) if r.status().is_success() => match r.text().await {
                Ok(t) => ok_text(t),
                Err(e) => err(format!("ошибка чтения: {:#}", e)),
            },
            Ok(r) => { let s = r.status(); err(format!("HTTP {} — {}", s, r.text().await.unwrap_or_default())) }
            Err(e) => err(format!("ошибка запроса: {:#}", e)),
        }
    }
}

inventory::submit! { &StorageReadAbout as &'static dyn crate::tool_runtime::ToolImpl }
