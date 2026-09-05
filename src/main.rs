// src/main.rs

use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use nano_harness::config::{build_engine_config, TomlConfig};
use nano_harness::engine::{ChatEngine, Role};
use nano_harness::session_store;
use nano_harness::tools::execute_tool;

#[derive(Parser, Debug)]
#[command(author, version, about = "CLI чат с Polza AI")]
struct Args {
    #[arg(short = 'k', long, help = "API ключ")]
    api_key: Option<String>,

    #[arg(short = 'u', long, help = "Base URL API")]
    base_url: Option<String>,

    #[arg(short = 'm', long, help = "Модель")]
    model: Option<String>,

    #[arg(long, help = "Температура (0-2)")]
    temperature: Option<f32>,

    #[arg(long, help = "Максимум токенов")]
    max_tokens: Option<u32>,

    #[arg(long, help = "Включить/выключить стриминг")]
    stream: Option<bool>,

    #[arg(long, help = "Уровень reasoning: low, medium, high")]
    reasoning_effort: Option<String>,

    #[arg(long, help = "Максимальный бюджет в рублях")]
    max_cost_rub: Option<f64>,

    #[arg(long, help = "Количество первых пар (запрос-ответ) для префикса")]
    prefix_message_count: Option<usize>,

    #[arg(long, help = "Количество последних пар для хвоста")]
    tail_message_count: Option<usize>,

    #[arg(long, help = "Системный промпт")]
    system_prompt: Option<String>,

    #[arg(
        long = "config",
        default_value = "config.toml",
        help = "Путь к конфигу"
    )]
    config_path: PathBuf,

    #[arg(long, default_value = "60", help = "Таймаут HTTP в секундах")]
    timeout_sec: u64,

    #[arg(long, help = "ID существующей сессии для продолжения")]
    session: Option<String>,
}

fn write_log(file: &mut std::fs::File, role: &str, content: &str) -> Result<()> {
    writeln!(
        file,
        "[{}] {}: {}",
        chrono::Local::now().format("%H:%M:%S"),
        role,
        content
    )?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let toml_config = TomlConfig::load(&args.config_path)?;

    let engine_config = build_engine_config(
        args.api_key,
        args.base_url,
        args.model,
        args.temperature,
        args.max_tokens,
        args.stream,
        args.reasoning_effort,
        args.max_cost_rub,
        args.prefix_message_count,
        args.tail_message_count,
        &toml_config,
    )?;

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()?;

    let mut engine = ChatEngine::new(engine_config.clone(), client.clone());

    engine.on_token = Some(Box::new(|token| {
        print!("{}", token);
        let _ = io::stdout().flush();
    }));

    engine.on_reasoning_token = Some(Box::new(|token| {
        print!("\x1b[90m{}\x1b[0m", token);
        let _ = io::stdout().flush();
    }));

    let system_prompt = args
        .system_prompt
        .or(toml_config.system_prompt)
        .unwrap_or_else(|| "Ты полезный ассистент. Отвечай кратко и по делу.".to_string());
    engine.add_message(Role::System, system_prompt.clone());

    // Работа с сессией
    let mut current_session = if let Some(sid) = args.session.clone() {
        session_store::load_session_with_key(&sid, &engine_config.api_key, None)
            .with_context(|| format!("Не удалось загрузить сессию '{}'", sid))?
    } else {
        let auto_name = format!("Сессия {}", chrono::Local::now().format("%Y-%m-%d %H:%M"));
        session_store::new_session(&auto_name).with_context(|| "Не удалось создать новую сессию")?
    };

    // Восстановление контекста чата, если есть
    if let Some(chat_ctx) = current_session.contexts.chat.clone() {
        engine.set_state(chat_ctx.engine_state);
    } else {
        // Для новой сессии заполняем контекст текущим состоянием движка (system_prompt уже добавлен)
        current_session.contexts.chat = Some(session_store::ContextBlock {
            engine_config: engine.get_config().clone(),
            engine_state: engine.get_state(),
        });
    }

    // Открытие лога
    let mut log_file = session_store::open_log(&current_session.session_id, "chat", None)
        .with_context(|| {
            format!(
                "Не удалось открыть лог-файл для сессии '{}'",
                current_session.session_id
            )
        })?;
    session_store::add_or_update_log(&mut current_session, "chat", None)
        .with_context(|| "Не удалось добавить запись о логе в сессию")?;
    if let Err(e) = session_store::save_session(&current_session) {
        eprintln!("Ошибка сохранения сессии: {}", e);
    }

    println!("📝 Сессия: {}", current_session.session_id);
    write_log(&mut log_file, "system", &system_prompt)?;

    println!("🤖 Чат запущен");
    println!("   Модель: {}", engine_config.model);
    println!(
        "   Стриминг: {}",
        if engine_config.stream {
            "включен"
        } else {
            "выключен"
        }
    );
    if let Some(cost) = engine_config.max_cost_rub {
        println!("   Лимит бюджета: {:.2} RUB", cost);
    }
    if let Some(allowed_tools) = &engine_config.allowed_tools {
        println!("   Разрешённые инструменты:");
        for tool in allowed_tools {
            println!("     - {} (режим: {})", tool.name, tool.mode);
        }
    } else {
        println!("   Инструменты: не заданы");
    }
    if let Some(prefix) = engine_config.prefix_message_count {
        println!("   Префикс: {} пар", prefix);
    }
    if let Some(tail) = engine_config.tail_message_count {
        println!("   Хвост: {} пар", tail);
    }
    println!("   Команды: /clear, /metrics, /system <текст>, /fix, /session, /exit");
    println!();

    let storage_http_config = toml_config
        .local_storage_http_server
        .clone()
        .unwrap_or_else(
            || nano_harness::local_storage_http_api::LocalStorageServerConfig {
                bind_addr: "127.0.0.1:8080".to_string(),
                storage_name: "default_storage".to_string(),
                auth_token: String::new(),
            },
        );

    let rhai_timeout_sec = toml_config.rhai_timeout_sec.unwrap_or(30);

    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut input = String::new();

    loop {
        print!("Вы > ");
        let _ = io::stdout().flush();
        input.clear();

        if tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut input).await? == 0 {
            println!("\nДо свидания!");
            break;
        }

        let user_input = input.trim();
        if user_input.is_empty() {
            continue;
        }

        match user_input {
            "/exit" | "/quit" => {
                println!("До свидания!");
                break;
            }
            "/clear" => {
                engine.clear_context();
                println!("🗑️ Контекст очищен.\n");
                continue;
            }
            "/fix" => {
                engine.fix_last_turn();
                println!("📌 Последняя пара закреплена в префиксе.\n");
                continue;
            }
            "/system" if user_input.len() > 8 => {
                let new_prompt = user_input[8..].trim().to_string();
                engine.set_system_prompt(new_prompt.clone());
                write_log(&mut log_file, "system", &new_prompt)?;
                println!("✅ Системный промпт обновлён.\n");
                continue;
            }
            "/metrics" => {
                println!("\n📊 Метрики сессии:");
                println!("   Запросов к API: {}", engine.metrics.api_calls_count);
                println!("   Prompt токенов: {}", engine.metrics.total_prompt_tokens);
                println!(
                    "   Completion токенов: {}",
                    engine.metrics.total_completion_tokens
                );
                println!(
                    "   Reasoning токенов: {}",
                    engine.metrics.total_reasoning_tokens
                );
                println!("   💰 Потрачено: {:.6} RUB", engine.metrics.total_cost_rub);
                println!(
                    "   Префикс: {} пар, {} слов",
                    engine.prefix_turn_count(),
                    engine.prefix_word_count()
                );
                println!(
                    "   Хвост: {} пар, {} слов",
                    engine.tail_turn_count(),
                    engine.tail_word_count()
                );
                println!();
                continue;
            }
            "/session" => {
                let parts: Vec<&str> = user_input.splitn(3, ' ').collect();
                match parts.get(1) {
                    Some(&"list") => {
                        let sessions = session_store::list_sessions()?;
                        println!("Список сессий:");
                        for s in sessions {
                            println!(
                                "  {} - {} (обновлена: {})",
                                s.session_id, s.display_name, s.updated_at
                            );
                        }
                    }
                    Some(&"new") => {
                        let name = parts.get(2).copied().unwrap_or("Новая сессия");
                        current_session = session_store::new_session(name)
                            .with_context(|| "Не удалось создать новую сессию")?;
                        engine = ChatEngine::new(engine_config.clone(), client.clone());
                        engine.add_message(Role::System, system_prompt.clone());
                        // Сразу заполняем контекст в новой сессии
                        current_session.contexts.chat = Some(session_store::ContextBlock {
                            engine_config: engine.get_config().clone(),
                            engine_state: engine.get_state(),
                        });
                        log_file =
                            session_store::open_log(&current_session.session_id, "chat", None)
                                .with_context(|| {
                                    format!(
                                        "Не удалось открыть лог для сессии '{}'",
                                        current_session.session_id
                                    )
                                })?;
                        session_store::add_or_update_log(&mut current_session, "chat", None)?;
                        if let Err(e) = session_store::save_session(&current_session) {
                            eprintln!("Ошибка сохранения сессии: {}", e);
                        }
                        println!("Создана сессия {}", current_session.session_id);
                    }
                    Some(&"switch") => {
                        if let Some(sid) = parts.get(2) {
                            current_session = session_store::load_session_with_key(
                                sid,
                                &engine_config.api_key,
                                None,
                            )
                            .with_context(|| format!("Не удалось загрузить сессию '{}'", sid))?;
                            engine = ChatEngine::new(engine_config.clone(), client.clone());
                            if let Some(chat_ctx) = current_session.contexts.chat.clone() {
                                engine.set_state(chat_ctx.engine_state);
                            } else {
                                engine.add_message(Role::System, system_prompt.clone());
                                current_session.contexts.chat = Some(session_store::ContextBlock {
                                    engine_config: engine.get_config().clone(),
                                    engine_state: engine.get_state(),
                                });
                            }
                            log_file =
                                session_store::open_log(&current_session.session_id, "chat", None)
                                    .with_context(|| {
                                        format!(
                                            "Не удалось открыть лог для сессии '{}'",
                                            current_session.session_id
                                        )
                                    })?;
                            session_store::add_or_update_log(&mut current_session, "chat", None)?;
                            if let Err(e) = session_store::save_session(&current_session) {
                                eprintln!("Ошибка сохранения сессии: {}", e);
                            }
                            println!("Переключено на сессию {}", sid);
                        } else {
                            println!("Укажите ID сессии");
                        }
                    }
                    Some(&"rename") => {
                        if let Some(name) = parts.get(2) {
                            current_session.display_name = name.to_string();
                            current_session.updated_at = session_store::now_ts();
                            session_store::save_session(&current_session)?;
                            println!("Имя сессии изменено");
                        }
                    }
                    _ => println!("Команды: /session list|new|switch <id>|rename <имя>"),
                }
                continue;
            }
            _ => {}
        }

        engine.add_message(Role::User, user_input.to_string());
        write_log(&mut log_file, "user", user_input)?;

        let response = match engine.send().await {
            Ok(resp) => resp,
            Err(e) => {
                println!("\n❌ Ошибка: {:#}", e);
                write_log(&mut log_file, "error", &format!("{:#}", e))?;
                engine.rollback_pending_turn();
                continue;
            }
        };

        println!();

        if !response.content.is_empty() {
            write_log(&mut log_file, "assistant", &response.content)?;
        }
        if !response.reasoning.is_empty() {
            write_log(&mut log_file, "reasoning", &response.reasoning)?;
        }

        if !response.reasoning.is_empty() {
            println!("\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n", response.reasoning);
        }

        if let Some(tool_calls) = response.tool_calls {
            println!("\n⚠️ Модель запросила инструменты:");
            for tc in &tool_calls {
                println!("   - {} ({})", tc.function.name, tc.function.arguments);
                write_log(
                    &mut log_file,
                    "tool_request",
                    &format!("{} ({})", tc.function.name, tc.function.arguments),
                )?;
            }

            let allowed = &engine_config.allowed_tools;
            let get_mode = |name: &str| -> Option<String> {
                allowed
                    .as_ref()
                    .and_then(|list| list.iter().find(|a| a.name == name).map(|a| a.mode.clone()))
            };

            for tc in &tool_calls {
                let name = &tc.function.name;
                let mode = get_mode(name).unwrap_or_else(|| "manual".to_string());

                if !allowed
                    .as_ref()
                    .map(|l| l.iter().any(|a| a.name == *name))
                    .unwrap_or(false)
                {
                    println!("⛔ Инструмент '{}' не разрешён, пропускаем.", name);
                    write_log(
                        &mut log_file,
                        "tool_denied",
                        &format!("{} ({})", name, tc.function.arguments),
                    )?;
                    continue;
                }

                if mode == "auto" {
                    let result = execute_tool(
                        name,
                        &tc.function.arguments,
                        Some(&client),
                        &storage_http_config.bind_addr,
                        &storage_http_config.auth_token,
                        rhai_timeout_sec,
                    )
                    .await;
                    println!("✅ Автовыполнение: {}", result);
                    write_log(
                        &mut log_file,
                        "tool_result",
                        &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                    )?;
                    engine.add_tool_result(tc.id.clone(), result);
                } else {
                    println!("❓ Выполнить инструмент '{}'? (y/n)", name);
                    print!("> ");
                    let _ = io::stdout().flush();
                    let mut answer = String::new();
                    tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut answer).await?;
                    if answer.trim().eq_ignore_ascii_case("y") {
                        let result = execute_tool(
                            name,
                            &tc.function.arguments,
                            Some(&client),
                            &storage_http_config.bind_addr,
                            &storage_http_config.auth_token,
                            rhai_timeout_sec,
                        )
                        .await;
                        println!("✅ Выполнено: {}", result);
                        write_log(
                            &mut log_file,
                            "tool_result",
                            &format!("{} ({}) -> {}", name, tc.function.arguments, result),
                        )?;
                        engine.add_tool_result(tc.id.clone(), result);
                    } else {
                        println!("Пропущено.");
                        write_log(
                            &mut log_file,
                            "tool_skipped",
                            &format!("{} ({})", name, tc.function.arguments),
                        )?;
                    }
                }
            }

            println!("\nЗапрашиваю финальный ответ...");
            let final_response = match engine.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    println!("\n❌ Ошибка: {:#}", e);
                    write_log(&mut log_file, "error", &format!("{:#}", e))?;
                    engine.rollback_pending_turn();
                    continue;
                }
            };
            println!();
            if !final_response.content.is_empty() {
                write_log(&mut log_file, "assistant", &final_response.content)?;
            }
            if !final_response.reasoning.is_empty() {
                write_log(&mut log_file, "reasoning", &final_response.reasoning)?;
                println!(
                    "\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n",
                    final_response.reasoning
                );
            }
        }

        // Сохранение сессии после каждого обмена
        current_session.contexts.chat = Some(session_store::ContextBlock {
            engine_config: engine.get_config().clone(),
            engine_state: engine.get_state(),
        });
        current_session.updated_at = session_store::now_ts();
        if let Err(e) = session_store::save_session(&current_session) {
            eprintln!("Ошибка сохранения сессии: {}", e);
        }

        println!(
            "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB]\x1b[0m\n",
            engine.metrics.total_prompt_tokens,
            engine.metrics.total_completion_tokens,
            engine.metrics.total_cost_rub
        );
    }

    Ok(())
}
