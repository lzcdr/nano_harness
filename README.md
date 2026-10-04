# nano_harness

Мультиагентная платформа на Rust: CLI-чат с LLM, автономные агенты с сессиями, локальное файловое хранилище с семантическим поиском, доска задач для асинхронного общения агентов, самообучение через скиллы и десктопная панель управления.

---

## Содержание

- [Что это](#что-это)
- [Архитектура](#архитектура)
- [Компоненты](#компоненты)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [CLI-чат](#cli-чат)
- [Агенты](#агенты)
- [Инструменты](#инструменты)
- [Локальное хранилище](#локальное-хранилище)
- [Доска задач](#доска-задач)
- [Скиллы](#скиллы)
- [База знаний](#база-знаний)
- [Rebuke](#rebuke)
- [Godfather](#godfather)
- [Control Center](#control-center)
- [Структура каталогов](#структура-каталогов)
- [Сборка](#сборка)

---

## Что это

`nano_harness` — это харнесс для работы с LLM через OpenAI-совместимый API. Поверх базового чата построены:

- **Мультиагентность.** Несколько агентов, каждый — отдельный HTTP-сервис со своим системным промптом, набором инструментов и параметрами модели. Агенты вызывают друг друга синхронно (`call_agent`) или асинхронно (`post_task`).
- **Локальное хранилище.** Файлы проектов лежат на диске, поверх них — векторный индекс на HNSW с эмбеддингами от локальной BERT-модели (`candle`, без внешних сервисов).
- **Скиллы.** Если агент успешно решил задачу через инструменты — опыт сохраняется как «скилл» и потом инжектится в похожие задачи.
- **База знаний.** Markdown-файлы с frontmatter, которые агент может подгружать в контекст по имени.
- **Rebuke.** Замечания пользователя агенту, встраиваемые в системный промпт.
- **Godfather.** Семантическая компактизация истории диалога через LLM.
- **Control Center.** Tauri-приложение для запуска/остановки всех сервисов и наблюдения за ними.

---

## Архитектура

Четыре независимых процесса, общающихся по HTTP и SSE:

```
┌──────────────────┐       HTTP/SSE       ┌────────────────────┐
│  nano_harness    │◄────────────────────►│ message_board_     │
│  (CLI-чат)       │                      │ server             │
└────────┬─────────┘                      └─────────┬──────────┘
         │                                          │
         │ HTTP                                     │ SSE
         │                                          │
         ▼                                          ▼
┌──────────────────┐                      ┌────────────────────┐
│ local_storage_   │◄────────────────────►│  agent_server      │
│ http_server      │       HTTP           │  (analyzer, ...)   │
└──────────────────┘                      └────────────────────┘

┌────────┴─────────┐
│  control_center  │  (Tauri GUI)
└──────────────────┘
```

Все четыре сервиса запускаются как отдельные процессы. `control_center` управляет их жизненным циклом.

---

## Компоненты

| Бинарник | Роль | Порт по умолчанию |
|---|---|---|
| `nano_harness` | CLI-чат + проект-менеджер | — |
| `agent_server` | Один агент (stateful или stateless) | `127.0.0.1:8081` (per agent) |
| `local_storage_http_server` | REST API хранилища, векторный поиск, скиллы, знания, ребуки | `127.0.0.1:8080` |
| `message_board_server` | Доска задач + SSE-события | `127.0.0.1:8090` |

---

## Быстрый старт

Семантический поиск работает на модели `paraphrase-multilingual-MiniLM-L12-v2`. Её нужно скачать с Hugging Face и положить в `.models/paraphrase-multilingual-MiniLM-L12-v2/`. Нужны три файла: `config.json`, `tokenizer.json`, `model.safetensors`.

```bash
# 1. Клонировать и собрать
cargo build --release

# 2. Положить модель эмбеддингов
#    .models/paraphrase-multilingual-MiniLM-L12-v2/{config.json, model.safetensors, tokenizer.json}

# 3. Положить словари в .dict/*.dic (hunspell, опционально)

# 4. Задать API-ключ
export POLZA_API_KEY="..."
(или положить в .env-файл)

# 5a. Запустить всё через GUI
cargo run -p control_center
# нажать "Start all"

# 5b. Или вручную, в отдельных терминалах
cargo run --bin message_board_server
cargo run --bin local_storage_http_server
cargo run --bin agent_server -- --agent-name analyzer
cargo run --bin nano_harness
```

---

## Конфигурация

Единый файл `config.toml` в корне. Полный пример — в репозитории. Ключевые секции:

### Глобальные параметры LLM

```toml
base_url = "https://polza.ai/api/v1"
model = "google/gemini-3.1-flash-lite"
temperature = 0.7
max_tokens = 4096
stream = true
reasoning_effort = "medium"
max_cost_rub = 10.0

# Ограничения контекста
prefix_message_count = 4   # сколько первых пар сообщений закрепить навсегда
tail_message_count = 6     # сколько последних сообщений держать в хвосте

system_prompt = """..."""
editor = "notepad"         # для редактирования выговоров и знаний
```

### Инструменты верхнего уровня

```toml
[[tools]]
name = "storage_read_file"
mode = "auto"     # auto | manual
```

### Векторная база

```toml
[vector_db]
model_path = ".models/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer_path = ".models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json"
chunk_size = 512
chunk_overlap = 64
top_k = 5
```

### Хранилище

```toml
[local_storage_http_server]
bind_addr = "127.0.0.1:8080"
storage_name = "default_storage"
auth_token = ""   # если пусто — берётся LOCAL_STORAGE_AUTH_TOKEN из .env или переменных среды
```

### Доска задач

```toml
[message_board]
bind_addr = "127.0.0.1:8090"
auth_token = ""
tasks_dir = ".board/tasks"
task_timeout_sec = 300
```

### Агенты

```toml
[[agents]]
name = "analyzer"
bind_addr = "127.0.0.1:8081"
auth_token = ""
timeout_sec = 120
max_iterations = 3

tools = ["storage_read_file", "storage_walk", "ripgrep", "scc", "ctags", "code_index"]

system_prompt = """..."""

base_url = "https://polza.ai/api/v1"
model = "google/gemini-3.1-flash-lite"
temperature = 0.7
max_tokens = 4096

agent_type = "stateful"   # stateful | stateless
session_ttl_secs = 1800
skill_mode = "auto"
skill_semantic_threshold = 0.5
skill_min_tool_calls = 2
```

---

## CLI-чат

```
cargo run --bin nano_harness [-- --project <name>] [--api-key ...] ...
```

### Команды в чате

| Команда | Действие |
|---|---|
| `/exit`, `/quit` | выход |
| `/clear` | очистить контекст |
| `/fix` | закрепить последнюю пару в префиксе |
| `/system <промпт>` | сменить системный промпт |
| `/metrics` | метрики сессии (токены, рубли, размер контекста) |
| `/godfather` | механическая + семантическая компактизация |
| `/unfreeze` | вернуть всё из архива godfather |
| `/unfreeze last` | вернуть последний сжатый ход |
| `/knowledge ...` | управление базой знаний |
| `/rebuke <agent> <текст>` | добавить замечание агенту |
| `/rebuke_edit <agent>` | редактировать замечания в редакторе |
| `/project ...` | управление проектами (list/new/switch/last/delete/purge) |

### Проекты

Проект — это `session_id` чата. Хранится в `.sessions/chat_<id>.json`. Команды:

```
/project list
/project new <name>
/project switch <session_id>
/project last
/project delete <id> [--force]
/project purge <id> [--force]
```

`delete` — мягкое удаление (флаг в JSON), `purge` — физическое удаление файлов.

---

## Агенты

Агент — это отдельный сервис. Два типа:

- **`stateless`** — каждый запрос независим, история не сохраняется.
- **`stateful`** — сессии сохраняются на диск, восстановление после рестарта, SSE-слушатель для асинхронных задач.

### Запуск

```bash
cargo run --bin agent_server -- --agent-name analyzer
```

Агент читает свою секцию `[[agents]]` из `config.toml`. Если `api_key`/`auth_token` пусты — берутся из переменных окружения:
- `AGENT_<NAME>_AUTH_TOKEN`
- `POLZA_API_KEY`
- `LOCAL_STORAGE_AUTH_TOKEN`
- `MESSAGE_BOARD_AUTH_TOKEN`

### Вызовы между агентами

- **`call_agent(to_agent, prompt)`** — синхронный. Ждёт ответа через `oneshot` (таймаут `agent_call_timeout_sec`).
- **`post_task(to_agent, prompt)`** — асинхронный. Возвращает `posted:<task_id>`, агент завершает ход со статусом `waiting`. Продолжение — когда придёт `TaskCompleted` от доски.

Защита от циклов: `parent_chain` прокидывается в payload задачи.

### HTTP API агента

```
POST /agent/run
Authorization: Bearer <token>
Content-Type: application/json

{
  "prompt": "...",
  "session_id": "..." (для stateful),
  "project_id": "...",
  "max_iterations": 3,
  "tools": [...],
  "system_prompt": "..."
}
```

---

## Инструменты

Регистрируются через `inventory`. Список — в `src/tool_runtime/tools/`.

### Работа с хранилищем

| Инструмент | Назначение |
|---|---|
| `storage_read_file` | прочитать файл |
| `storage_write_file` | записать файл |
| `storage_delete_file` | удалить файл |
| `storage_create_dir` | создать директорию |
| `storage_list_dir` | список в директории |
| `storage_walk` | рекурсивный обход |
| `storage_search_by_name` | поиск по имени |
| `storage_search_similar` | семантический поиск |
| `storage_read_about` / `storage_write_about` | файл `.about` в директории |
| `storage_read_summary` / `storage_write_summary` | файл `.summary` в директории |

### Анализ кода

| Инструмент | Назначение |
|---|---|
| `ripgrep` | пакетный поиск по содержимому файлов |
| `scc` | подсчёт строк кода по языкам |
| `ctags` | индексация символов (universal-ctags) |
| `code_index` | комплексный индекс проекта: модули, экспорты, импорты, doc-комментарии, конфиг-файлы |

`code_index` пишет `CODE_INDEX.json` в корень проекта и возвращает краткую сводку.

### Агентские

| Инструмент | Назначение |
|---|---|
| `call_agent` | синхронный вызов другого агента |
| `post_task` | асинхронная задача другому агенту |

Все инструменты возвращают JSON вида `{"ok": true, "content": "..."}` или `{"ok": false, "error": "..."}`.

Требования к внешним бинарникам: `rg`, `scc`, `ctags` должны быть в `PATH`.

---

## Локальное хранилище

Файлы лежат в `.local_storage/<storage_name>/projects/<project_id>/...`.

- **Зарезервированные имена:** `.about`, `.summary`, `.skills`, `.rebukes`, `.knowledge`, `vector_meta.json`.
- **Векторный индекс:** `vector_meta.json` + HNSW в памяти. Загружается при старте, автоматически перестраивается при обнаружении изменённых файлов.
- **Эмбеддинги:** `candle` + BERT (`paraphrase-multilingual-MiniLM-L12-v2`), выполняется в отдельном потоке (модель не `Send`/`Sync`).

### HTTP API

Все запросы требуют заголовок `Authorization: Bearer <token>` и `X-NH-Project: <project_id>` (кроме скиллов/знаний/ребуков, где project не нужен).

```
POST   /files?path=...        — записать файл (body = содержимое)
GET    /files?path=...        — прочитать файл
DELETE /files?path=...        — удалить файл
POST   /dirs?path=...         — создать директорию
GET    /list?path=...         — список в директории
GET    /walk?path=...         — рекурсивный обход
GET    /search?query=...&top_k=...  — семантический поиск
GET    /search_name?pattern=...     — поиск по имени
GET    /about?path=...  POST /about?path=...
GET    /summary?path=... POST /summary?path=...
```

### Скиллы

```
GET  /skills/get?path=...
POST /skills/put                     — тело: {content, index, record, skill_file, index_file}
POST /skills/delete?path=...
GET  /skills/list
GET  /skills/search?query=...&top_k=...
POST /skills/record_usage?file=...&success=...
```

### Знания

```
GET  /knowledge/list
GET  /knowledge/get?name=...
POST /knowledge/put?name=...         — тело: markdown с frontmatter
POST /knowledge/delete?name=...
```

### Rebuke

```
GET  /rebukes/get?agent=...
POST /rebukes/put?agent=...          — тело: текст
```

---

## Доска задач

`message_board_server` — актор, хранящий задачи в `.board/tasks/*.json`. Подписчики получают события через SSE.

### HTTP API

```
POST /tasks                         — создать задачу
GET  /tasks?session_id=...&from_agent=...&status=...
GET  /tasks/{id}
POST /tasks/{id}/complete           — {result}
POST /tasks/{id}/fail               — {error}
GET  /events?agent_name=...         — SSE-поток
```

### События SSE

```json
{"type": "taskcreated", "task": {}}
{"type": "taskcompleted", "task_id": "...", "result": "...", "from_agent": "...", "from_session_id": "..."}
{"type": "taskfailed", "task_id": "...", "error": "...", "from_agent": "...", "from_session_id": "..."}
```

При старте доска восстанавливает задачи из `.board/tasks/`: `pending` удаляются, `in_progress` помечаются `failed` («executor lost»), завершённые старше `task_timeout_sec` удаляются.

---

## Скиллы

Механизм самообучения агентов.

**Как работает:**

1. Агент выполняет задачу, делает ≥ `skill_min_tool_calls` успешных вызовов инструментов.
2. LLM генерирует название и описание скилла.
3. Скилл сохраняется в `.local_storage/<name>/.skills/`.
4. Дополнительно строится «скелет» промпта через лексикон: известные слова остаются, неизвестные заменяются на `<ENT>`. Это абстрагирует задачу от конкретных имён.
5. Fingerprint (SHA-256) защищает от дублей.
6. При новой задаче — векторный поиск по скелету + фильтр по сущностям + приоритет скиллам того же агента + сортировка по success_rate.
7. Найденный скилл инжектится в контекст как системное сообщение.

**Формат файла скилла:**

```
SKILL: <название>
FOR: <агент>
DESCRIPTION: <описание>
ENTITIES: <сущности через запятую>
STATS: <успехи>/<провалы>
FINGERPRINT: <sha256>
CREATED_AT: <unix>

INSTRUCTION:
<промпт задачи>

TOOL_CALLS:
<json массив вызовов>
```

---

## База знаний

Markdown-файлы в `.local_storage/<name>/.knowledge/<name>.md`:

```markdown
---
name: rust_async
description: Основы async/await в Rust
---

Текст знания...
```

Каталог знаний инжектится в системный промпт. Агент сам вызывает `knowledge_load(name)` / `knowledge_unload()`.

Управление — через CLI-команды `/knowledge new|edit|show|delete|load|unload|list`.

---

## Rebuke

Замечания пользователя агенту. Хранятся в `.local_storage/<name>/.rebukes/<agent>.txt`:

```
---[2026-01-15 14:32]--------
Не выдумывай факты, которых нет в коде.

---[2026-01-15 15:01]--------
Всегда проверяй, что путь не выходит за пределы проекта.
```

При старте агента файл читается, тексты замечаний встраиваются в системный промпт. Перезапуск агента подхватывает изменения.

---

## Godfather

Семантическая компактизация истории через LLM. Настраивается в `[godfather]`:

```toml
[godfather]
model = "google/gemini-3.1-flash-lite"
api_key = ""              # если пусто — отключён
base_url = "https://polza.ai/api/v1"
temperature = 0.3
max_tokens = 8192
prompt = """..."""
```

**Двухэтапный процесс** (`/godfather`):

1. **Механическая компактизация.** Длинные tool-результаты заменяются плейсхолдером, reasoning ассистента обнуляется.
2. **Семантическая.** Хвост диалога отправляется LLM с промптом-редактором. Ответ парсится как JSON-массив сообщений. Если короче оригинала — применяется, оригиналы сохраняются в `archived_origin`.

Откат: `/unfreeze` (всё) или `/unfreeze last` (последний сжатый ход).

---

## Control Center

Tauri 2 приложение. Управляет жизненным циклом сервисов.

**Функции:**

- `Start all` / `Stop all` — запуск/остановка в правильном порядке (доска → хранилище → агенты → чат).
- Индивидуальные `Start` / `Stop` / `Restart` для каждого сервиса.
- `Reload config` — перечитать `config.toml` без перезапуска GUI.
- Список сервисов с состоянием, uptime, bind-адресом.
- Кэш ошибок (TTL 30 секунд).

**Запуск сервисов на Windows** — через `cmd.exe /c` с `CREATE_NEW_CONSOLE` (чтобы Ctrl+C работал в консоли сервиса). Остановка — `taskkill /T` (graceful) → ожидание 5 секунд → `taskkill /F /T` (hard).

**Тема** — `control_center/theme.toml` (цвета, шрифты, отступы). CSS-переменные применяются через JS.

---

## Структура каталогов

```
.
├── config.toml                  # конфигурация
├── Cargo.toml
├── .dict/                       # словари hunspell (*.dic)
├── .models/                     # модель эмбеддингов
├── .local_storage/              # данные хранилища
│   └── <storage_name>/
│       ├── projects/<project_id>/
│       ├── .skills/
│       ├── .knowledge/
│       ├── .rebukes/
│       └── vector_meta.json
├── .sessions/                   # сессии и логи
│   ├── chat_<id>.json
│   ├── agent_<name>_<id>.json
│   └── logs/
├── .board/tasks/                # задачи доски
├── src/
│   ├── agent_core.rs
│   ├── agent_http_api.rs
│   ├── config.rs
│   ├── engine.rs
│   ├── godfather.rs
│   ├── knowledge_manager.rs
│   ├── lexicon.rs
│   ├── local_storage.rs
│   ├── local_storage_http_api.rs
│   ├── message_board.rs
│   ├── rebuke_manager.rs
│   ├── session_store.rs
│   ├── skill_manager.rs
│   ├── tool_runtime/
│   │   ├── context.rs
│   │   ├── mod.rs
│   │   ├── primitives.rs
│   │   └── tools/
│   └── bin/
│       ├── agent_server.rs
│       ├── local_storage_http_server.rs
│       └── message_board_server.rs
└── control_center/              # Tauri GUI
    ├── src/main.rs
    ├── ui/index.html
    ├── theme.toml
    └── tauri.conf.json
```

---

## Сборка

**Требования:**

- Rust (edition 2021, stable)
- `ripgrep`, `scc`, `universal-ctags` в `PATH` (для инструментов анализа)
- На Windows — WebView2 (для Tauri)
- Модель эмбеддингов в `.models/paraphrase-multilingual-MiniLM-L12-v2/`

**Сборка:**

```bash
cargo build --release
```

**Переменные окружения:**

| Переменная | Назначение |
|---|---|
| `POLZA_API_KEY` | API-ключ LLM |
| `GODFATHER_API_KEY` | API-ключ для godfather (fallback на `POLZA_API_KEY`) |
| `LOCAL_STORAGE_AUTH_TOKEN` | Токен хранилища |
| `MESSAGE_BOARD_AUTH_TOKEN` | Токен доски |
| `AGENT_<NAME>_AUTH_TOKEN` | Токен конкретного агента |
| `NH_CONFIG` | Путь к `config.toml` для control_center |
| `EDITOR` | Редактор для `/rebuke_edit` и `/knowledge edit` |

---

## Лицензия

Лицензировано на условиях двойной лицензии:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

на ваш выбор.

### Вклад

Если вы явно не укажете иное, любой вклад, отправленный вами
для включения в эту работу, будет лицензирован под указанной
двойной лицензией без каких-либо дополнительных условий.
