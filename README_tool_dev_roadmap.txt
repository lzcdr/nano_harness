================================================================================
                    NANO_HARNESS — КАК ПИСАТЬ НОВЫЕ ИНСТРУМЕНТЫ
================================================================================

Файл для разработчика.

--------------------------------------------------------------------------------
ЧТО ТАКОЕ ИНСТРУМЕНТ
--------------------------------------------------------------------------------

Инструмент — это функция, которую LLM может вызвать через tool_calls.
Модель видит описание и JSON-схему параметров в API-запросе (поле tools).
Harness исполняет функцию и возвращает JSON-строку с результатом.

Три части:

1. Определение (name, description, parameters) — то, что видит модель.
2. Обработчик (async fn run) — код, который исполняется при вызове.
3. Регистрация — автоматическая через inventory.

Всё живёт в src/tool_runtime/tools/.

--------------------------------------------------------------------------------
КАКИЕ БЫВАЮТ ИНСТРУМЕНТЫ
--------------------------------------------------------------------------------

По природе:

- Читающие. Ничего не меняют. storage_read_file, storage_walk, storage_list_dir.
- Пишущие. Меняют состояние. storage_write_file, storage_delete_file.
- Сетевые. Обращаются к другим сервисам. call_agent, post_task.
- Вычислительные. Обрабатывают данные без побочных эффектов.

--------------------------------------------------------------------------------
ШАГ 1. СОЗДАТЬ ФАЙЛ
--------------------------------------------------------------------------------

Один инструмент — один файл в src/tool_runtime/tools/.
Имя файла = имя инструмента в snake_case.

    src/tool_runtime/tools/storage_read_file.rs

--------------------------------------------------------------------------------
ШАГ 2. РЕАЛИЗОВАТЬ ТРЕЙТ
--------------------------------------------------------------------------------

Каждый инструмент — это структура (unit struct, без полей) и impl трейта
ToolImpl. Трейт определён в src/tool_runtime/mod.rs.

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
            "Читает содержимое файла в локальном хранилище. \
             Возвращает текст файла. При ошибке — JSON с полем 'error'."
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
            // ... сама логика ...
        }
    }

    inventory::submit! {
        &StorageReadFile as &'static dyn crate::tool_runtime::ToolImpl
    }

Четыре метода:

- name         — имя инструмента. Оно же указывается в config.toml.
- description  — что делает. Модель читает и решает, звать ли.
- parameters   — JSON Schema параметров.
- run          — сама логика. Принимает JSON-строку, возвращает JSON-строку.

--------------------------------------------------------------------------------
ШАГ 3. ДОБАВИТЬ В mod.rs
--------------------------------------------------------------------------------

Файл: src/tool_runtime/tools/mod.rs.

    mod storage_read_file;
    pub use storage_read_file::StorageReadFile;

Это единственное место, где надо прописывать что-то руками. Одна строка
mod, одна строка pub use на каждый инструмент. Всё остальное (списки для
модели, диспетчер вызовов) собирается автоматически из inventory.

--------------------------------------------------------------------------------
ШАГ 4. РАЗРЕШИТЬ В КОНФИГЕ
--------------------------------------------------------------------------------

Инструменты разрешаются в config.toml. У чата — в [[tools]], у каждого
агента — в его секции tools = [...].

    [[agents]]
    name = "assistant"
    tools = ["storage_read_file", "storage_write_file", "storage_walk"]

Или для чата:

    [[tools]]
    name = "storage_read_file"
    mode = "auto"

Для инструментов, которые нужны всем и всегда (knowledge_load,
knowledge_unload), можно добавить принудительно в build_engine_config.
Но это исключение, не правило. Обычно — явный список в конфиге.

--------------------------------------------------------------------------------
ШАГ 5. ПРОВЕРИТЬ
--------------------------------------------------------------------------------

1. Собрать: cargo build --release
2. Запустить хранилище, доску, агента.
3. Дать агенту задачу, которая использует новый инструмент.
4. Смотреть лог агента: там будет строка tool_request с именем инструмента
   и аргументами, потом tool_result с результатом.
5. Если инструмент не вызывается — модель его не видит. Проверь, что имя
   есть в config.toml, что агент перезапущен, что описание не пустое.
6. Если вызывается, но падает — смотри текст ошибки в tool_result.

--------------------------------------------------------------------------------
ДОСТУПНЫЙ КОНТЕКСТ
--------------------------------------------------------------------------------

Обработчик получает &ToolContext. В нём:

    http_client            reqwest::Client        клиент для HTTP-запросов
    storage_base_url       String                 адрес хранилища (с http://)
    storage_auth_token     String                 токен хранилища
    board_base_url         String                 адрес доски (с http://)
    board_auth_token       String                 токен доски
    project_id             String                 проект, в котором идёт работа
    session_id             Option<String>         сессия вызывающего
    parent_chain           Vec<String>            цепочка вызовов агентов
    self_agent_name        String                 имя вызывающего ("chat" для чата)
    pending_calls          PendingCalls           ожидающие синхронных вызовов
    outgoing_tasks         OutgoingTasks          исходящие асинхронные задачи
    agent_call_timeout_sec u64                    таймаут call_agent

project_id нужен для запросов к хранилищу — идёт в заголовок X-NH-Project.
parent_chain нужен для call_agent / post_task — защита от циклов.

Хелперы (в ctx):

    ctx.storage_base()  -> &str  адрес хранилища без завершающего слэша
    ctx.board_base()    -> &str  адрес доски без завершающего слэша

--------------------------------------------------------------------------------
ПРИМИТИВЫ (src/tool_runtime/primitives.rs)
--------------------------------------------------------------------------------

Готовые функции для типовых операций внутри run.

    parse_input(input)                  -> Result<Value, String>
        Парсит входную JSON-строку. Пустая строка → пустой объект.

    require_string(&args, "path")       -> Result<String, String>
        Извлекает обязательный строковый параметр. Возвращает Err с понятным
        текстом, если параметр отсутствует или пуст.

    optional_string(&args, "name")      -> Option<String>
        Опциональный строковый параметр.

    optional_int(&args, "top_k")        -> Option<i64>
        Опциональный целочисленный параметр.

    ok_text("результат")                -> String
        Успешный ответ. Оборачивает произвольный текст в JSON:
        {"ok": true, "content": "результат"}

    ok_json(value)                      -> String
        Успешный ответ с произвольным JSON-объектом:
        {"ok": true, "data": {...}}

    err("сообщение")                    -> String
        Ошибка:
        {"ok": false, "error": "сообщение"}

Всегда возвращай через ok_text или err. Модель видит поле ok и понимает,
успех это или провал.

--------------------------------------------------------------------------------
ПРИМЕР. ПРОСТОЙ ЧИТАЮЩИЙ ИНСТРУМЕНТ
--------------------------------------------------------------------------------

Задача: добавить storage_file_size, возвращающий размер файла в байтах.

src/tool_runtime/tools/storage_file_size.rs:

    use crate::tool_runtime::context::ToolContext;
    use crate::tool_runtime::primitives::{err, ok_text, parse_input, require_string};
    use crate::tool_runtime::ToolImpl;

    pub struct StorageFileSize;

    #[async_trait::async_trait]
    impl ToolImpl for StorageFileSize {
        fn name(&self) -> &'static str {
            "storage_file_size"
        }

        fn description(&self) -> &'static str {
            "Возвращает размер файла в байтах."
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
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

            let url = format!("{}/list", ctx.storage_base());
            // ... запрос списка директории, поиск файла, размер ...
            // Для краткости опущено. В реальности — HTTP-запрос через
            // ctx.http_client с заголовками Authorization и X-NH-Project.
            ok_text("0")
        }
    }

    inventory::submit! {
        &StorageFileSize as &'static dyn crate::tool_runtime::ToolImpl
    }

Добавить в mod.rs:

    mod storage_file_size;
    pub use storage_file_size::StorageFileSize;

Разрешить в config.toml у нужных агентов.

--------------------------------------------------------------------------------
ПРИМЕР. ИНСТРУМЕНТ, ВЫЗЫВАЮЩИЙ ДРУГОЙ СЕРВИС (call_agent)
--------------------------------------------------------------------------------

Сложнее. Публикует задачу на доске, ждёт ответа через pending_calls.

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) { Ok(v) => v, Err(e) => return err(e) };
        let to_agent = match require_string(&args, "to_agent") { Ok(a) => a, Err(e) => return err(e) };
        let prompt = match require_string(&args, "prompt") { Ok(p) => p, Err(e) => return err(e) };

        // Проверки на самовызов и цикл.
        if to_agent == ctx.self_agent_name { ... }
        if ctx.parent_chain.contains(&to_agent) { ... }

        // 1. POST /tasks на доске.
        // 2. Получаем task_id.
        // 3. Кладём oneshot-канал в ctx.pending_calls под этим task_id.
        // 4. tokio::time::timeout(rx).await с ctx.agent_call_timeout_sec.
        // 5. На таймауте — POST /tasks/{id}/fail, вернуть err.

        // ... полностью расписано в src/tool_runtime/tools/call_agent.rs ...
        ok_text("ответ")
    }

Ключевое: обработчик асинхронный, работает в tokio-контексте, никаких
block_on, никакого Handle::current. reqwest::Client из ctx.http_client.

--------------------------------------------------------------------------------
ЧЕГО НЕ ДЕЛАТЬ
--------------------------------------------------------------------------------

1. Не создавать reqwest::blocking::Client внутри async-обработчика. Это
   дропает runtime в неправильном месте и роняет процесс. Только async
   reqwest::Client из ctx.http_client.

2. Не использовать std::sync::Mutex в горячем пути. Только tokio::sync::Mutex
   и .lock().await.

3. Не паниковать при ошибках. unwrap и expect — только там, где сбой
   действительно невозможен. Всё остальное — return err(...).

4. Не смешивать sync и async. Если обработчик async, всё внутри async.
   Если нужна блокирующая операция — spawn_blocking с явным замыканием.

5. Не добавлять инструмент с неясным описанием. Модель его не вызовет.
   Описание — часть контракта.

6. Не прописывать инструмент вручную в списках для модели. available_tools
   собирается автоматически из inventory. Единственное место с ручной
   регистрацией — mod.rs в tool_runtime/tools.

--------------------------------------------------------------------------------
ШПАРГАЛКА
--------------------------------------------------------------------------------

Новый инструмент за 4 шага:

    1. src/tool_runtime/tools/<имя>.rs   — структура + impl ToolImpl + inventory::submit!
    2. src/tool_runtime/tools/mod.rs     — mod <имя>; pub use ...;
    3. config.toml                       — вписать имя в tools нужных агентов
    4. cargo build --release

Проверить:

    - перезапустить агента
    - дать задачу, которая требует инструмент
    - смотреть tool_request и tool_result в логе

Признаки проблем:

    - Модель не зовёт инструмент — его нет в config.toml или плохое описание.
    - Инструмент падает — смотри текст ошибки в tool_result, чаще всего
      неверные аргументы или недоступный сервис.
    - Паника процесса — где-то блокирующий вызов в async-контексте.

--------------------------------------------------------------------------------
ФИЛОСОФИЯ
--------------------------------------------------------------------------------

Инструмент — это контракт между тобой и моделью. Ты описываешь, что функция
делает и какие у неё параметры. Модель решает, когда её вызывать. Harness
следит за тем, чтобы вызов был корректным и результат вернулся.

Чем проще инструмент, тем надёжнее. Один инструмент — одно действие.
Составные инструменты (read_many, walk_with_content) — когда нужно закрыть
частый сценарий без N round-trip. Но не увлекаться: каждый новый инструмент
это ещё одна вещь, о которой модель должна помнить.

Инструментов должно быть столько, сколько нужно, и ни одним больше.

================================================================================
