// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/engine.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio_stream::StreamExt;

// ============================================================================
// 1. ТИПЫ ДАННЫХ API POLZA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Developer,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ============================================================================
// 2. МЕТРИКИ СЕССИИ
// ============================================================================

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub total_prompt_tokens: u32,
    pub total_completion_tokens: u32,
    pub total_reasoning_tokens: u32,
    pub total_cost_rub: f64,
    pub api_calls_count: u32,
}

impl SessionMetrics {
    pub fn accumulate(&mut self, usage: &Usage) {
        self.total_prompt_tokens += usage.prompt_tokens;
        self.total_completion_tokens += usage.completion_tokens;
        self.total_cost_rub += usage.cost_rub.unwrap_or(0.0);
        self.api_calls_count += 1;
        if let Some(details) = &usage.completion_tokens_details {
            self.total_reasoning_tokens += details.reasoning_tokens.unwrap_or(0);
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub cost_rub: Option<f64>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CompletionDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

// ============================================================================
// 3. ОТВЕТ ДВИЖКА
// ============================================================================

#[allow(dead_code)]
pub struct EngineResponse {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Option<Usage>,
}

// ============================================================================
// 4. КОНФИГ ДВИЖКА
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolExecutionConfig {
    pub name: String,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: bool,
    pub allowed_tools: Option<Vec<ToolExecutionConfig>>,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub max_cost_rub: Option<f64>,
    pub prefix_message_count: Option<usize>,
    pub tail_message_count: Option<usize>,
    #[serde(default = "default_compact_threshold_bytes")]
    pub compact_threshold_bytes: usize,
    #[serde(default = "default_tail_byte_budget")]
    pub tail_byte_budget: usize,
}

fn default_compact_threshold_bytes() -> usize {
    512
}

fn default_tail_byte_budget() -> usize {
    100 * 1024
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://polza.ai/api/v1".to_string(),
            model: "openai/gpt-4o".to_string(),
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            stop: None,
            stream: true,
            allowed_tools: None,
            tool_choice: None,
            reasoning_effort: None,
            max_cost_rub: None,
            prefix_message_count: None,
            tail_message_count: None,
            compact_threshold_bytes: default_compact_threshold_bytes(),
            tail_byte_budget: default_tail_byte_budget(),
        }
    }
}

// ============================================================================
// 5. ДВИЖОК
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_origin: Option<Vec<Turn>>,
}

pub struct ChatEngine {
    config: EngineConfig,
    system_messages: Vec<Message>,
    /// Отдельный слот под инжектнутый скилл. Не копится, перезаписывается.
    skill_context: Option<Message>,
    /// Отдельный слот под загруженное знание. Не копится, перезаписывается.
    knowledge_context: Option<Message>,
    /// Набор имён тулзов, отдаваемых LLM в текущем ходу. Не сохраняется
    /// в EngineState — заполняется снаружи (select_tools) перед каждым send.
    visible_tools: Option<Vec<String>>,
    prefix_turns: Vec<Turn>,
    tail_turns: Vec<Turn>,
    pending_turn: Option<Turn>,
    client: Client,
    pub metrics: SessionMetrics,
    pub on_token: Option<Box<dyn Fn(&str) + Send + Sync>>,
    pub on_reasoning_token: Option<Box<dyn Fn(&str) + Send + Sync>>,
    pub on_tool_call_token: Option<Box<dyn Fn(&str) + Send + Sync>>,
}

impl ChatEngine {
    pub fn new(config: EngineConfig, client: Client) -> Self {
        Self {
            config,
            system_messages: Vec::new(),
            skill_context: None,
            knowledge_context: None,
            visible_tools: None,
            prefix_turns: Vec::new(),
            tail_turns: Vec::new(),
            pending_turn: None,
            client,
            metrics: SessionMetrics::default(),
            on_token: None,
            on_reasoning_token: None,
            on_tool_call_token: None,
        }
    }

    pub fn add_message(&mut self, role: Role, content: String) {
        self.add_message_internal(Message {
            role,
            content: Some(content),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    pub fn add_tool_result(&mut self, tool_call_id: String, result: String) {
        self.add_message_internal(Message {
            role: Role::Tool,
            content: Some(result),
            reasoning: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            name: None,
        });
    }

    /// Установить скилл-контекст. Перезаписывает предыдущий.
    pub fn set_skill_context(&mut self, content: String) {
        self.skill_context = Some(Message {
            role: Role::System,
            content: Some(content),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    pub fn clear_skill_context(&mut self) {
        self.skill_context = None;
    }

    pub fn has_skill_context(&self) -> bool {
        self.skill_context.is_some()
    }

    /// Установить контекст знания. Перезаписывает предыдущий.
    pub fn set_knowledge_context(&mut self, name: &str, content: String) {
        self.knowledge_context = Some(Message {
            role: Role::System,
            content: Some(format!("[Загруженное знание: {}]\n\n{}", name, content)),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    pub fn clear_knowledge_context(&mut self) {
        self.knowledge_context = None;
    }

    pub fn has_knowledge_context(&self) -> bool {
        self.knowledge_context.is_some()
    }

    pub fn set_visible_tools(&mut self, names: Vec<String>) {
        self.visible_tools = Some(names);
    }

    pub fn visible_tools(&self) -> Option<&[String]> {
        self.visible_tools.as_deref()
    }

    /// Текст текущего хода: сообщения `User` и `Assistant` из `pending_turn`.
    /// Возвращает None, если ход пуст или содержит только tool-сообщения.
    pub fn pending_turn_text(&self) -> Option<String> {
        let turn = self.pending_turn.as_ref()?;
        let mut parts: Vec<String> = Vec::new();
        for msg in &turn.messages {
            match msg.role {
                Role::User | Role::Assistant => {
                    if let Some(c) = &msg.content {
                        if !c.trim().is_empty() {
                            parts.push(c.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    pub fn clear_context(&mut self) {
        self.finalize_pending_turn();
        self.prefix_turns.clear();
        self.tail_turns.clear();
        self.pending_turn = None;
        self.skill_context = None;
        self.knowledge_context = None;
        self.visible_tools = None;
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        if let Some(first) = self.system_messages.first_mut() {
            first.content = Some(prompt);
        } else {
            self.system_messages.push(Message {
                role: Role::System,
                content: Some(prompt),
                reasoning: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    // ==================== Компактизация ====================

    /// Применить компактизацию ко всем сохранённым ходам.
    /// Возвращает (было_байт, стало_байт).
    pub fn compact_all_turns(&mut self) -> (usize, usize) {
        let before = self.total_context_bytes();
        let threshold = self.config.compact_threshold_bytes;
        for turn in self.prefix_turns.iter_mut() {
            compact_turn(threshold, turn);
        }
        for turn in self.tail_turns.iter_mut() {
            compact_turn(threshold, turn);
        }
        let after = self.total_context_bytes();
        (before, after)
    }

    /// Заменить указанный диапазон ходов в tail_turns на новый набор.
    /// Используется godfather'ом. Оригиналы сохраняются внутри нового turn'а
    /// в поле `archived_origin`.
    pub fn replace_tail_range_with_summary(
        &mut self,
        start: usize,
        end: usize,
        new_messages: Vec<Message>,
    ) {
        if start >= end || end > self.tail_turns.len() {
            return;
        }

        let originals: Vec<Turn> = self.tail_turns[start..end].to_vec();

        let new_turn = Turn {
            messages: new_messages,
            archived_origin: Some(originals),
        };
        self.tail_turns
            .splice(start..end, std::iter::once(new_turn));
    }

    /// Суммарное количество ходов, скрытых в архивных частях tail_turns.
    pub fn archive_len(&self) -> usize {
        fn count(turn: &Turn) -> usize {
            match &turn.archived_origin {
                None => 0,
                Some(orig) => orig.len() + orig.iter().map(count).sum::<usize>(),
            }
        }
        self.tail_turns.iter().map(count).sum()
    }

    /// Развернуть всё содержимое архива обратно в tail_turns.
    /// Возвращает количество восстановленных ходов.
    pub fn unfreeze_all(&mut self) -> usize {
        fn flatten(turn: Turn) -> Vec<Turn> {
            match turn.archived_origin {
                None => vec![turn],
                Some(orig) => orig.into_iter().flat_map(flatten).collect(),
            }
        }

        let mut count = 0;
        let mut new_tail = Vec::new();
        for turn in self.tail_turns.drain(..) {
            if turn.archived_origin.is_some() {
                let flat = flatten(turn);
                count += flat.len();
                new_tail.extend(flat);
            } else {
                new_tail.push(turn);
            }
        }
        self.tail_turns = new_tail;
        count
    }

    /// Развернуть последний сжатый turn обратно.
    /// Возвращает количество восстановленных ходов, или 0, если архив пуст.
    pub fn unfreeze_last(&mut self) -> usize {
        for i in (0..self.tail_turns.len()).rev() {
            if self.tail_turns[i].archived_origin.is_some() {
                let turn = self.tail_turns.remove(i);
                let originals = turn.archived_origin.unwrap();
                let count = originals.len();
                for (j, t) in originals.into_iter().enumerate() {
                    self.tail_turns.insert(i + j, t);
                }
                return count;
            }
        }
        0
    }

    /// Публичная функция для godfather'а: вернуть срез tail_turns для сжатия.
    pub fn tail_turns_snapshot(&self) -> Vec<Turn> {
        self.tail_turns.clone()
    }

    /// Публичная функция для godfather'а: применить результат.
    pub fn apply_godfather_result(&mut self, new_messages: Vec<Message>) {
        if self.tail_turns.is_empty() {
            return;
        }
        let start = 0;
        let end = self.tail_turns.len();
        self.replace_tail_range_with_summary(start, end, new_messages);
    }

    // ==================== Служебное ====================

    fn finalize_pending_turn(&mut self) {
        if let Some(mut turn) = self.pending_turn.take() {
            let threshold = self.config.compact_threshold_bytes;
            compact_turn(threshold, &mut turn);
            let prefix_limit = self.config.prefix_message_count.unwrap_or(0);
            if self.prefix_turns.len() < prefix_limit {
                self.prefix_turns.push(turn);
            } else {
                self.tail_turns.push(turn);
                self.trim_tail();
            }
        }
    }

    pub fn fix_last_turn(&mut self) {
        self.finalize_pending_turn();

        if let Some(last_turn) = self.tail_turns.pop() {
            self.prefix_turns.push(last_turn);
        }
    }

    pub fn prefix_turn_count(&self) -> usize {
        self.prefix_turns.len()
    }

    pub fn tail_turn_count(&self) -> usize {
        self.tail_turns.len()
    }

    pub fn prefix_word_count(&self) -> usize {
        self.count_words_in_turns(&self.prefix_turns)
    }

    pub fn tail_word_count(&self) -> usize {
        self.count_words_in_turns(&self.tail_turns)
    }

    pub fn rollback_pending_turn(&mut self) {
        self.pending_turn = None;
    }

    fn count_words_in_turns(&self, turns: &[Turn]) -> usize {
        turns
            .iter()
            .flat_map(|turn| turn.messages.iter())
            .map(|m| {
                let mut words = 0;
                if let Some(content) = &m.content {
                    words += content.split_whitespace().count();
                }
                if let Some(reasoning) = &m.reasoning {
                    words += reasoning.split_whitespace().count();
                }
                words
            })
            .sum()
    }

    /// Приблизительный размер всего контекста в байтах.
    pub fn total_context_bytes(&self) -> usize {
        let mut total = 0usize;
        for m in &self.system_messages {
            total += message_bytes(m);
        }
        if let Some(m) = &self.skill_context {
            total += message_bytes(m);
        }
        if let Some(m) = &self.knowledge_context {
            total += message_bytes(m);
        }
        for t in &self.prefix_turns {
            for m in &t.messages {
                total += message_bytes(m);
            }
        }
        for t in &self.tail_turns {
            for m in &t.messages {
                total += message_bytes(m);
            }
        }
        if let Some(t) = &self.pending_turn {
            for m in &t.messages {
                total += message_bytes(m);
            }
        }
        total
    }

    #[allow(dead_code)]
    pub fn get_messages(&self) -> Vec<Message> {
        let mut result = Vec::new();
        result.extend(self.system_messages.iter().cloned());
        if let Some(skill) = &self.skill_context {
            result.push(skill.clone());
        }
        if let Some(knowledge) = &self.knowledge_context {
            result.push(knowledge.clone());
        }
        for turn in &self.prefix_turns {
            result.extend(turn.messages.iter().cloned());
        }
        for turn in &self.tail_turns {
            result.extend(turn.messages.iter().cloned());
        }
        if let Some(ref pending) = self.pending_turn {
            result.extend(pending.messages.iter().cloned());
        }
        result
    }

    fn add_message_internal(&mut self, msg: Message) {
        match msg.role {
            Role::System => {
                self.system_messages.push(msg);
            }
            Role::User => {
                self.finalize_pending_turn();

                let turn = Turn {
                    messages: vec![msg],
                    archived_origin: None,
                };
                self.pending_turn = Some(turn);
            }
            _ => {
                if let Some(ref mut pending) = self.pending_turn {
                    pending.messages.push(msg);
                } else {
                    let turn = Turn {
                        messages: vec![msg],
                        archived_origin: None,
                    };
                    self.tail_turns.push(turn);
                    self.trim_tail();
                }
            }
        }
    }

    fn add_assistant_message(
        &mut self,
        content: String,
        reasoning: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        self.add_message_internal(Message {
            role: Role::Assistant,
            content: Some(content),
            reasoning,
            tool_calls,
            tool_call_id: None,
            name: None,
        });
    }

    /// Трим хвоста по двум лимитам одновременно: число ходов и байты.
    /// Ход выкидывается, если его выкидывает хотя бы один лимит.
    /// Новый ход никогда не выкидывается.
    fn trim_tail(&mut self) {
        let count_limit = self.config.tail_message_count.unwrap_or(usize::MAX);
        let byte_limit = self.config.tail_byte_budget;

        while self.tail_turns.len() > 1 {
            let count_ok = self.tail_turns.len() <= count_limit;
            let bytes: usize = self
                .tail_turns
                .iter()
                .flat_map(|t| t.messages.iter())
                .map(message_bytes)
                .sum();
            let bytes_ok = bytes <= byte_limit;

            if count_ok && bytes_ok {
                break;
            }

            self.tail_turns.remove(0);
        }
    }

    // ==================== Запрос ====================

    fn build_request(&self) -> Value {
        let all_messages = self.get_messages();
        let mut req = serde_json::json!({
            "model": self.config.model,
            "messages": all_messages,
            "stream": self.config.stream,
        });

        if let Some(temp) = self.config.temperature {
            req["temperature"] = serde_json::json!(temp);
        }
        if let Some(max_tok) = self.config.max_tokens {
            req["max_tokens"] = serde_json::json!(max_tok);
        }
        if let Some(top_p) = self.config.top_p {
            req["top_p"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &self.config.stop {
            req["stop"] = serde_json::json!(stop);
        }

        // visible_tools задаёт снаружи точный набор тулзов на текущий ход.
        // None — тулзы не отправляются вовсе (пустой tools: []).
        // Some(names) — отправляются только они, пересечённые с реально
        // зарегистрированными в inventory.
        match &self.visible_tools {
            Some(names) => {
                let filtered: Vec<ToolDefinition> =
                    crate::tool_runtime::available_tools(Some(names));
                req["tools"] = serde_json::to_value(filtered).unwrap();
                if let Some(choice) = &self.config.tool_choice {
                    req["tool_choice"] = choice.clone();
                } else {
                    req["tool_choice"] = serde_json::json!("auto");
                }
            }
            None => {
                req["tools"] = serde_json::json!([]);
            }
        }

        if let Some(effort) = &self.config.reasoning_effort {
            req["reasoning"] = serde_json::json!({ "effort": effort });
        }

        req
    }

    pub async fn send(&mut self) -> Result<EngineResponse> {
        if let Some(max_cost) = self.config.max_cost_rub {
            if self.metrics.total_cost_rub >= max_cost {
                return Err(anyhow::anyhow!(
                    "Превышен лимит бюджета: {:.4} RUB",
                    max_cost
                ));
            }
        }

        let request = self.build_request();

        let response = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?
            .error_for_status()
            .context("Ошибка HTTP-запроса к Polza API")?;

        if self.config.stream {
            self.handle_streaming_response(response).await
        } else {
            self.handle_regular_response(response).await
        }
    }

    async fn handle_regular_response(
        &mut self,
        response: reqwest::Response,
    ) -> Result<EngineResponse> {
        let chat_resp: ChatResponse = response.json().await?;

        if let Some(usage) = &chat_resp.usage {
            self.metrics.accumulate(usage);
        }

        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .context("API вернул пустой ответ")?;

        let content = choice.message.content.clone().unwrap_or_default();
        let reasoning = choice.message.reasoning.clone().unwrap_or_default();
        let tool_calls = choice.message.tool_calls.clone();

        if let Some(ref cb) = self.on_token {
            if !content.is_empty() {
                cb(&content);
            }
        }
        if let Some(ref cb) = self.on_reasoning_token {
            if !reasoning.is_empty() {
                cb(&reasoning);
            }
        }

        self.add_assistant_message(
            content.clone(),
            if reasoning.is_empty() {
                None
            } else {
                Some(reasoning.clone())
            },
            tool_calls.clone(),
        );

        Ok(EngineResponse {
            content,
            reasoning,
            tool_calls,
            usage: chat_resp.usage,
        })
    }

    async fn handle_streaming_response(
        &mut self,
        response: reqwest::Response,
    ) -> Result<EngineResponse> {
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut done = false;

        let mut current_content = String::new();
        let mut current_reasoning = String::new();
        let mut tool_calls_accumulator: HashMap<u32, StreamingToolCall> = HashMap::new();
        let mut final_usage: Option<Usage> = None;

        let mut tool_header_printed = false;
        let mut tool_open_index: Option<u32> = None;
        let mut tool_printed_indexes: std::collections::HashSet<u32> =
            std::collections::HashSet::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].to_string();
                buffer.drain(..=pos);

                let line = line.trim_end_matches('\r');
                if !line.starts_with("data: ") {
                    continue;
                }

                let data = line[6..].trim();
                if data == "[DONE]" {
                    done = true;
                    break;
                }
                if data.is_empty() {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    if let Some(error) = chunk.error {
                        return Err(anyhow::anyhow!("Ошибка API в потоке: {:?}", error));
                    }

                    if let Some(usage) = chunk.usage {
                        final_usage = Some(usage);
                    }

                    for choice in chunk.choices {
                        if let Some(delta) = choice.delta {
                            if let Some(text) = delta.content {
                                current_content.push_str(&text);
                                if let Some(ref cb) = self.on_token {
                                    cb(&text);
                                }
                            }
                            if let Some(reasoning) = delta.reasoning {
                                current_reasoning.push_str(&reasoning);
                                if let Some(ref cb) = self.on_reasoning_token {
                                    cb(&reasoning);
                                }
                            }
                            if let Some(tc_stream) = delta.tool_calls {
                                if !tool_header_printed {
                                    if let Some(ref cb) = self.on_tool_call_token {
                                        cb("⚠️ Модель запросила инструменты:\n");
                                    }
                                    tool_header_printed = true;
                                }

                                for tc in tc_stream {
                                    let entry = tool_calls_accumulator
                                        .entry(tc.index)
                                        .or_insert_with(|| StreamingToolCall {
                                            id: tc.id.clone().unwrap_or_default(),
                                            name: String::new(),
                                            arguments: String::new(),
                                        });

                                    if let Some(id) = &tc.id {
                                        if !id.is_empty() {
                                            entry.id = id.clone();
                                        }
                                    }

                                    if let Some(name) =
                                        tc.function.as_ref().and_then(|f| f.name.clone())
                                    {
                                        if !name.is_empty() {
                                            entry.name.push_str(&name);
                                        }
                                    }

                                    if let Some(args) =
                                        tc.function.as_ref().and_then(|f| f.arguments.clone())
                                    {
                                        if !args.is_empty() {
                                            // Открываем строку для нового index — только
                                            // когда у нас уже есть имя и пошли аргументы.
                                            if !tool_printed_indexes.contains(&tc.index) {
                                                if let Some(prev) = tool_open_index {
                                                    if prev != tc.index {
                                                        if let Some(ref cb) =
                                                            self.on_tool_call_token
                                                        {
                                                            cb(")\n");
                                                        }
                                                    }
                                                }
                                                tool_open_index = Some(tc.index);
                                                if let Some(ref cb) = self.on_tool_call_token {
                                                    cb(&format!("   - {} (", entry.name));
                                                }
                                                tool_printed_indexes.insert(tc.index);
                                            }
                                            entry.arguments.push_str(&args);
                                            if let Some(ref cb) = self.on_tool_call_token {
                                                cb(&args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if done {
                break;
            }
        }

        if tool_open_index.is_some() {
            if let Some(ref cb) = self.on_tool_call_token {
                cb(")\n");
            }
        }

        if let Some(usage) = &final_usage {
            self.metrics.accumulate(usage);
        }

        let final_tool_calls: Option<Vec<ToolCall>> = if tool_calls_accumulator.is_empty() {
            None
        } else {
            Some(
                tool_calls_accumulator
                    .into_values()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.name,
                            arguments: tc.arguments,
                        },
                    })
                    .collect(),
            )
        };

        self.add_assistant_message(
            current_content.clone(),
            if current_reasoning.is_empty() {
                None
            } else {
                Some(current_reasoning.clone())
            },
            final_tool_calls.clone(),
        );

        Ok(EngineResponse {
            content: current_content,
            reasoning: current_reasoning,
            tool_calls: final_tool_calls,
            usage: final_usage,
        })
    }

    pub fn get_state(&self) -> crate::session_store::EngineState {
        crate::session_store::EngineState {
            system_messages: self.system_messages.clone(),
            skill_context: self.skill_context.clone(),
            knowledge_context: self.knowledge_context.clone(),
            prefix_turns: self.prefix_turns.clone(),
            tail_turns: self.tail_turns.clone(),
            pending_turn: self.pending_turn.clone(),
            metrics: self.metrics.clone(),
        }
    }

    pub fn set_state(&mut self, state: crate::session_store::EngineState) {
        self.system_messages = state.system_messages;
        self.skill_context = state.skill_context;
        self.knowledge_context = state.knowledge_context;
        self.prefix_turns = state.prefix_turns;
        self.tail_turns = state.tail_turns;
        self.pending_turn = state.pending_turn;
        self.metrics = state.metrics;
    }

    pub fn get_config(&self) -> &EngineConfig {
        &self.config
    }
}

fn message_bytes(m: &Message) -> usize {
    let mut total = 0usize;
    if let Some(c) = &m.content {
        total += c.len();
    }
    if let Some(r) = &m.reasoning {
        total += r.len();
    }
    if let Some(tc) = &m.tool_calls {
        for t in tc {
            total += t.function.name.len() + t.function.arguments.len();
        }
    }
    total
}

/// Заменяет тело tool-сообщения на плейсхолдер, если оно больше порога.
/// Обнуляет reasoning у assistant-сообщений.
fn compact_turn(threshold: usize, turn: &mut Turn) {
    for msg in turn.messages.iter_mut() {
        if msg.role == Role::Tool {
            if let Some(ref content) = msg.content {
                if content.len() > threshold {
                    let bytes = content.len();
                    msg.content = Some(format!(
                        "[результат опущен при компактизации, {} байт; перезапроси инструмент, если нужно]",
                        bytes
                    ));
                }
            }
        }
        // Reasoning для будущих ходов не нужен — модель его не перечитывает.
        if msg.role == Role::Assistant && msg.reasoning.is_some() {
            msg.reasoning = None;
        }
    }
}

// ============================================================================
// 6. ВНУТРЕННИЕ СТРУКТУРЫ ДЛЯ ПАРСИНГА
// ============================================================================

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    model: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Option<StreamDelta>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    index: u32,
    id: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

struct StreamingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> EngineConfig {
        EngineConfig {
            api_key: "test".into(),
            base_url: "http://127.0.0.1:1".into(), // недостижимый, но не зовётся
            model: "test".into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            stream: false,
            allowed_tools: None,
            tool_choice: None,
            reasoning_effort: None,
            max_cost_rub: None,
            prefix_message_count: Some(1),
            tail_message_count: Some(2),
            compact_threshold_bytes: 512,
            tail_byte_budget: 100 * 1024,
        }
    }

    fn new_engine() -> ChatEngine {
        ChatEngine::new(dummy_config(), Client::new())
    }

    fn long_string(n: usize) -> String {
        "x".repeat(n)
    }

    #[test]
    fn add_user_message_creates_pending_turn() {
        let mut e = new_engine();
        e.add_message(Role::User, "hi".into());
        assert_eq!(e.tail_turn_count(), 0);
        assert_eq!(e.total_context_bytes(), 2);
    }

    #[test]
    fn user_then_assistant_finalizes_turn_into_prefix() {
        let mut e = new_engine();
        e.add_message(Role::User, "hi".into());
        e.add_message(Role::Assistant, "hello".into());
        e.add_message(Role::User, "next".into()); // триггерит finalize
        assert_eq!(e.prefix_turn_count(), 1);
        assert_eq!(e.prefix_word_count(), 2); // "hi" + "hello"
    }

    #[test]
    fn excess_turns_go_to_tail() {
        let mut e = new_engine();
        // prefix_message_count = 1
        e.add_message(Role::User, "u1".into());
        e.add_message(Role::Assistant, "a1".into());
        e.add_message(Role::User, "u2".into());
        e.add_message(Role::Assistant, "a2".into());
        e.add_message(Role::User, "u3".into()); // finalize u2a2
        assert_eq!(e.prefix_turn_count(), 1);
        assert_eq!(e.tail_turn_count(), 1);
    }

    #[test]
    fn tail_trims_by_count() {
        let mut e = new_engine();
        e.add_message(Role::User, "u1".into());
        e.add_message(Role::Assistant, "a1".into());
        for i in 2..=5 {
            e.add_message(Role::User, format!("u{}", i));
            e.add_message(Role::Assistant, format!("a{}", i));
        }
        // tail_message_count = 2
        assert!(e.tail_turn_count() <= 2);
    }

    #[test]
    fn compact_turn_replaces_long_tool_result() {
        let mut turn = Turn {
            messages: vec![Message {
                role: Role::Tool,
                content: Some(long_string(1000)),
                reasoning: None,
                tool_calls: None,
                tool_call_id: Some("1".into()),
                name: None,
            }],
            archived_origin: None,
        };
        compact_turn(512, &mut turn);
        let content = turn.messages[0].content.as_ref().unwrap();
        assert!(content.contains("результат опущен"));
        assert!(content.len() < 200);
    }

    #[test]
    fn compact_turn_keeps_short_tool_result() {
        let mut turn = Turn {
            messages: vec![Message {
                role: Role::Tool,
                content: Some("short".into()),
                reasoning: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            archived_origin: None,
        };
        compact_turn(512, &mut turn);
        assert_eq!(turn.messages[0].content.as_deref(), Some("short"));
    }

    #[test]
    fn compact_turn_strips_assistant_reasoning() {
        let mut turn = Turn {
            messages: vec![Message {
                role: Role::Assistant,
                content: Some("answer".into()),
                reasoning: Some("thinking...".into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            archived_origin: None,
        };
        compact_turn(512, &mut turn);
        assert!(turn.messages[0].reasoning.is_none());
        assert_eq!(turn.messages[0].content.as_deref(), Some("answer"));
    }

    #[test]
    fn message_bytes_sums_content_and_reasoning() {
        let m = Message {
            role: Role::Assistant,
            content: Some("abc".into()),
            reasoning: Some("de".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(message_bytes(&m), 5);
    }

    #[test]
    fn message_bytes_counts_tool_calls() {
        let m = Message {
            role: Role::Assistant,
            content: None,
            reasoning: None,
            tool_calls: Some(vec![ToolCall {
                id: "1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "fn".into(),
                    arguments: "{}".into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        assert_eq!(message_bytes(&m), 4); // "fn" + "{}"
    }

    #[test]
    fn skill_context_set_and_clear() {
        let mut e = new_engine();
        assert!(!e.has_skill_context());
        e.set_skill_context("skill body".into());
        assert!(e.has_skill_context());
        e.clear_skill_context();
        assert!(!e.has_skill_context());
    }

    #[test]
    fn knowledge_context_set_and_clear() {
        let mut e = new_engine();
        assert!(!e.has_knowledge_context());
        e.set_knowledge_context("rust", "async".into());
        assert!(e.has_knowledge_context());
        e.clear_knowledge_context();
        assert!(!e.has_knowledge_context());
    }

    #[test]
    fn clear_context_removes_everything() {
        let mut e = new_engine();
        e.add_message(Role::System, "sys".into());
        e.add_message(Role::User, "u".into());
        e.add_message(Role::Assistant, "a".into());
        e.set_skill_context("s".into());
        e.set_knowledge_context("k", "body".into());
        e.clear_context();
        assert_eq!(e.tail_turn_count(), 0);
        assert_eq!(e.prefix_turn_count(), 0);
        assert!(!e.has_skill_context());
        assert!(!e.has_knowledge_context());
    }

    #[test]
    fn unfreeze_last_restores_archived_turn() {
        let mut e = new_engine();
        e.add_message(Role::User, "u1".into());
        e.add_message(Role::Assistant, "a1".into());
        e.add_message(Role::User, "u2".into());
        e.add_message(Role::Assistant, "a2".into());
        e.add_message(Role::User, "u3".into());
        // tail_turns = [u2a2] (prefix занят u1a1)
        let snapshot = e.tail_turns_snapshot();
        assert!(!snapshot.is_empty());

        e.apply_godfather_result(vec![Message {
            role: Role::User,
            content: Some("сжато".into()),
            reasoning: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]);
        assert_eq!(e.archive_len(), 1);

        let restored = e.unfreeze_last();
        assert!(restored > 0);
        assert_eq!(e.archive_len(), 0);
    }

    #[test]
    fn unfreeze_all_on_empty_returns_zero() {
        let mut e = new_engine();
        assert_eq!(e.unfreeze_all(), 0);
    }

    #[test]
    fn unfreeze_last_on_empty_returns_zero() {
        let mut e = new_engine();
        assert_eq!(e.unfreeze_last(), 0);
    }

    #[test]
    fn set_system_prompt_replaces_first() {
        let mut e = new_engine();
        e.add_message(Role::System, "first".into());
        e.set_system_prompt("second".into());
        let msgs = e.get_messages();
        assert_eq!(msgs[0].content.as_deref(), Some("second"));
    }

    #[test]
    fn set_system_prompt_creates_if_absent() {
        let mut e = new_engine();
        e.set_system_prompt("only".into());
        let msgs = e.get_messages();
        assert_eq!(msgs[0].content.as_deref(), Some("only"));
    }
}
