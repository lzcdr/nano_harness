// src/main.rs

mod config;
mod engine;
mod local_storage;
mod tools;

use crate::config::{build_engine_config, TomlConfig};
use crate::engine::{ChatEngine, Role};
use crate::tools::execute_tool;
use anyhow::Result;
use clap::Parser;
use reqwest::Client;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

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
}

fn sanitize_model_name(model: &str) -> String {
    model
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

fn write_log(file: &mut fs::File, role: &str, content: &str) -> Result<()> {
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

    // 1. Загружаем TOML конфиг
    let toml_config = TomlConfig::load(&args.config_path)?;

    // 2. Собираем EngineConfig
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

    // 3. Создаем HTTP-клиент
    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()?;

    // 4. Создаем движок
    let mut engine = ChatEngine::new(engine_config.clone(), client);

    // 5. Устанавливаем callbacks для стриминга
    engine.on_token = Some(Box::new(|token| {
        print!("{}", token);
        let _ = io::stdout().flush();
    }));

    engine.on_reasoning_token = Some(Box::new(|token| {
        print!("\x1b[90m{}\x1b[0m", token);
        let _ = io::stdout().flush();
    }));

    // 6. Добавляем системный промпт
    let system_prompt = args
        .system_prompt
        .or(toml_config.system_prompt)
        .unwrap_or_else(|| "Ты полезный ассистент. Отвечай кратко и по делу.".to_string());
    engine.add_message(Role::System, system_prompt.clone());

    // 7. Создаём папку для логов
    fs::create_dir_all("chats")?;

    // 8. Формируем имя файла лога
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let safe_model = sanitize_model_name(&engine_config.model);
    let log_path = format!("chats/{}_{}.txt", timestamp, safe_model);

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    println!("📝 Лог чата: {}", log_path);

    // 9. Логируем системный промпт
    write_log(&mut log_file, "system", &system_prompt)?;

    // 10. Выводим информацию о запуске
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
    println!("   Команды: /clear, /metrics, /system <текст>, /fix, /exit");
    println!();

    // 11. Основной цикл
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
            _ => {}
        }

        engine.add_message(Role::User, user_input.to_string());
        write_log(&mut log_file, "user", user_input)?;

        // Отправляем запрос с обработкой ошибок
        let response = match engine.send().await {
            Ok(resp) => resp,
            Err(e) => {
                println!("\n❌ Ошибка: {:#}", e);
                write_log(&mut log_file, "error", &format!("{:#}", e))?;
                engine.rollback_pending_turn();
                continue;
            }
        };

        println!(); // Перенос строки после стриминга

        // Логируем ответ ассистента
        if !response.content.is_empty() {
            write_log(&mut log_file, "assistant", &response.content)?;
        }
        if !response.reasoning.is_empty() {
            write_log(&mut log_file, "reasoning", &response.reasoning)?;
        }

        // Показываем reasoning, если он есть
        if !response.reasoning.is_empty() {
            println!("\n\x1b[90m🧠 Reasoning:\n{}\x1b[0m\n", response.reasoning);
        }

        // Обрабатываем tool calls
        if let Some(tool_calls) = response.tool_calls {
            println!("\n⚠️ Модель запросила инструменты:");
            for tc in &tool_calls {
                println!("   - {} ({})", tc.function.name, tc.function.arguments);
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
                    continue;
                }

                if mode == "auto" {
                    let result = execute_tool(name, &tc.function.arguments);
                    println!("✅ Автовыполнение: {}", result);
                    write_log(&mut log_file, "tool", &format!("{} -> {}", name, result))?;
                    engine.add_tool_result(tc.id.clone(), result);
                } else {
                    println!("❓ Выполнить инструмент '{}'? (y/n)", name);
                    print!("> ");
                    let _ = io::stdout().flush();
                    let mut answer = String::new();
                    tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut answer).await?;
                    if answer.trim().eq_ignore_ascii_case("y") {
                        let result = execute_tool(name, &tc.function.arguments);
                        println!("✅ Выполнено: {}", result);
                        write_log(&mut log_file, "tool", &format!("{} -> {}", name, result))?;
                        engine.add_tool_result(tc.id.clone(), result);
                    } else {
                        println!("Пропущено.");
                        write_log(&mut log_file, "tool", &format!("{} - пропущено", name))?;
                    }
                }
            }

            // Повторный запрос с результатами инструментов
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

        // Показываем метрики после каждого запроса
        println!(
            "\n\x1b[90m[Токены: {} prompt + {} completion | 💰 {:.6} RUB]\x1b[0m\n",
            engine.metrics.total_prompt_tokens,
            engine.metrics.total_completion_tokens,
            engine.metrics.total_cost_rub
        );
    }

    Ok(())
}
