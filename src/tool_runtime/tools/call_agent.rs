use crate::agent_core::PendingCall;
use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
use crate::tool_runtime::ToolImpl;
use std::time::Duration;
use tokio::time::timeout;

pub struct CallAgent;

#[async_trait::async_trait]
impl ToolImpl for CallAgent {
    fn name(&self) -> &'static str { "call_agent" }

    fn description(&self) -> &'static str {
        "Синхронно вызывает другого агента и ждёт его ответа. Возвращает текст \
         ответа. При ошибке — JSON с полем 'error'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Имя агента из конфига"},
                "prompt": {"type": "string", "description": "Текст задачи"}
            },
            "required": ["to_agent", "prompt"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) { Ok(v) => v, Err(e) => return err(e) };
        let to_agent = match require_string(&args, "to_agent") { Ok(a) => a, Err(e) => return err(e) };
        let prompt = match require_string(&args, "prompt") { Ok(p) => p, Err(e) => return err(e) };

        if to_agent == ctx.self_agent_name {
            return err(format!("агент '{}' не может вызывать сам себя", to_agent));
        }
        if ctx.parent_chain.contains(&to_agent) {
            return err(format!(
                "циклический вызов: агент '{}' уже в цепочке {:?}",
                to_agent, ctx.parent_chain
            ));
        }

        let mut new_chain = ctx.parent_chain.clone();
        new_chain.push(to_agent.clone());

        let task_id = match post_task_internal(ctx, &to_agent, &prompt, &new_chain).await {
            Ok(id) => id,
            Err(e) => return err(e),
        };

        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        {
            let mut pc = ctx.pending_calls.lock().await;
            pc.insert(
                task_id.clone(),
                PendingCall { tx, created_at: std::time::Instant::now() },
            );
        }

        let timeout_sec = ctx.agent_call_timeout_sec;
        match timeout(Duration::from_secs(timeout_sec), rx).await {
            Ok(Ok(answer)) => ok_text(answer),
            Ok(Err(_)) => {
                let mut pc = ctx.pending_calls.lock().await;
                pc.remove(&task_id);
                err("канал ожидания закрыт")
            }
            Err(_) => {
                {
                    let mut pc = ctx.pending_calls.lock().await;
                    pc.remove(&task_id);
                }
                let fail_url = format!("{}/tasks/{}/fail", ctx.board_base(), task_id);
                let _ = ctx.http_client
                    .post(&fail_url)
                    .header("Authorization", format!("Bearer {}", ctx.board_auth_token))
                    .json(&serde_json::json!({
                        "error": format!("call_agent timeout ({} sec)", timeout_sec)
                    }))
                    .send()
                    .await;
                err(format!("таймаут ожидания ответа ({} сек)", timeout_sec))
            }
        }
    }
}

async fn post_task_internal(
    ctx: &ToolContext,
    to_agent: &str,
    prompt: &str,
    chain: &[String],
) -> Result<String, String> {
    let url = format!("{}/tasks", ctx.board_base());
    let body = serde_json::json!({
        "from_agent": ctx.self_agent_name,
        "from_session_id": ctx.session_id.clone().unwrap_or_default(),
        "to_agent": to_agent,
        "to_session_id": ctx.session_id.clone().unwrap_or_default(),
        "project_id": ctx.project_id,
        "payload": { "prompt": prompt },
        "parent_task_id": null,
        "chain": chain
    });
    let resp = ctx.http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", ctx.board_auth_token))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("ошибка запроса к доске: {:#}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {} — {}", status, text));
    }

    let v: serde_json::Value = resp.json().await
        .map_err(|e| format!("ошибка парсинга ответа: {:#}", e))?;
    v.get("task_id").and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "в ответе нет task_id".to_string())
}

inventory::submit! { &CallAgent as &'static dyn crate::tool_runtime::ToolImpl }
