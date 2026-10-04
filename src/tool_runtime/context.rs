// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/context.rs

use crate::agent_core::{OutgoingTasks, PendingCalls};
use reqwest::Client;

/// Всё, что нужно инструменту от мира. Контекст формирует обёртка
/// (чат или агент) и передаёт в tool_runtime::execute.
#[derive(Clone)]
pub struct ToolContext {
    /// HTTP-клиент для обращения к хранилищу и доске.
    pub http_client: Client,

    /// Локальный путь к корню хранилища. Нужен инструментам, которые
    /// работают с файлами напрямую (не через HTTP), например ripgrep.
    pub storage_root_path: std::path::PathBuf,

    /// Адрес хранилища. С гарантией http://.
    pub storage_base_url: String,

    /// Токен хранилища.
    pub storage_auth_token: String,

    /// Адрес доски. С гарантией http://.
    pub board_base_url: String,

    /// Токен доски.
    pub board_auth_token: String,

    /// Проект, в котором работает вызывающий. Идёт в X-NH-Project.
    pub project_id: String,

    /// Сессия вызывающего (для агента — своя stateful-сессия, для чата — None).
    pub session_id: Option<String>,

    /// Цепочка вызовов агентов, для защиты от циклов.
    pub parent_chain: Vec<String>,

    /// Имя вызывающего агента. Для чата — "chat".
    pub self_agent_name: String,

    /// Общие с агентом/чатом структуры для синхронных и асинхронных вызовов.
    pub pending_calls: PendingCalls,
    pub outgoing_tasks: OutgoingTasks,

    /// Таймаут вызова другого агента через call_agent.
    pub agent_call_timeout_sec: u64,
}

impl ToolContext {
    /// Хранилище без завершающего слэша.
    pub fn storage_base(&self) -> &str {
        self.storage_base_url.trim_end_matches('/')
    }

    /// Доска без завершающего слэша.
    pub fn board_base(&self) -> &str {
        self.board_base_url.trim_end_matches('/')
    }
}
