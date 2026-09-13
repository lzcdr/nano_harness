# Проверка функционала nano_harness

В этом документе собраны curl-команды для проверки работы HTTP API хранилища и агентов. Все команды предназначены для PowerShell (используется `curl.exe`).

## Предварительные условия

- Все серверы запущены локально.
- Сервер хранилища: `http://127.0.0.1:8080`, токен `secret`.
- Агентский сервер (assistant): `http://127.0.0.1:8081`, токен `agent_token_1`.
- Для семантического поиска модель эмбеддингов должна быть загружена (при первом старте сервера хранилища).

### Запуск серверов (если ещё не запущены)

В отдельных терминалах выполните:

```powershell
# Хранилище
cargo run --bin local_storage_http_server

# Агент (stateless или stateful в зависимости от конфига)
cargo run --bin agent_server -- --agent-name assistant
```

## Хранилище (http://127.0.0.1:8080, токен: secret)

### Запись файла
```powershell
curl.exe -X POST "http://127.0.0.1:8080/files?path=example.txt" -H "Authorization: Bearer secret" --data "Hello, storage!"
```

### Чтение файла
```powershell
curl.exe "http://127.0.0.1:8080/files?path=example.txt" -H "Authorization: Bearer secret"
```

### Список файлов
```powershell
curl.exe "http://127.0.0.1:8080/list?path=." -H "Authorization: Bearer secret"
```

### Рекурсивный обход
```powershell
curl.exe "http://127.0.0.1:8080/walk?path=." -H "Authorization: Bearer secret"
```

### Поиск по имени
```powershell
curl.exe "http://127.0.0.1:8080/search_name?pattern=example" -H "Authorization: Bearer secret"
```

### Семантический поиск
```powershell
curl.exe "http://127.0.0.1:8080/search?query=hello&top_k=3" -H "Authorization: Bearer secret"
```

### Запись .about
```powershell
curl.exe -X POST "http://127.0.0.1:8080/about?path=." -H "Authorization: Bearer secret" --data "About root"
```

### Чтение .about
```powershell
curl.exe "http://127.0.0.1:8080/about?path=." -H "Authorization: Bearer secret"
```

### Запись .summary
```powershell
curl.exe -X POST "http://127.0.0.1:8080/summary?path=." -H "Authorization: Bearer secret" --data "Summary root"
```

### Чтение .summary
```powershell
curl.exe "http://127.0.0.1:8080/summary?path=." -H "Authorization: Bearer secret"
```

## Агент (http://127.0.0.1:8081, токен: agent_token_1)

### Stateless: время через run_code
```powershell
curl.exe -X POST "http://127.0.0.1:8081/agent/run" -H "Authorization: Bearer agent_token_1" -H "Content-Type: application/json" -d '{"prompt":"Узнай текущее время"}'
```

### Stateless: чтение файла через run_code
```powershell
curl.exe -X POST "http://127.0.0.1:8081/agent/run" -H "Authorization: Bearer agent_token_1" -H "Content-Type: application/json" -d '{"prompt":"Прочитай example.txt"}'
```

### Stateful: первый запрос (получить session_id)
```powershell
curl.exe -X POST "http://127.0.0.1:8081/agent/run" -H "Authorization: Bearer agent_token_1" -H "Content-Type: application/json" -d '{"prompt":"Запомни: цвет синий"}'
```

### Stateful: следующий запрос с session_id
```powershell
curl.exe -X POST "http://127.0.0.1:8081/agent/run" -H "Authorization: Bearer agent_token_1" -H "Content-Type: application/json" -d '{"prompt":"Какой цвет?","session_id":"ПОДСТАВЬТЕ_ID"}'
```

## Проверка ошибок
### Неверный токен хранилища
```powershell
curl.exe "http://127.0.0.1:8080/files?path=example.txt" -H "Authorization: Bearer wrong"
```

## Использование скилов
```powershell
curl.exe -X POST http://127.0.0.1:8081/agent/run -H "Authorization: Bearer agent_token_1" -H "Content-Type: application/json" -d '{"prompt":"Напиши и выполни Rhai-код, который создаёт директорию test_skill_dir и записывает в неё файл example.txt с текстом Привет от агента. Код должен быть не короче 150 символов."}'
```
## Использование мессаджборда
### Создание задачи (замените board_token на ваш токен из config.toml)
```powershell
curl.exe -X POST "http://127.0.0.1:8090/tasks" -H "Authorization: Bearer board_token" -H "Content-Type: application/json" -d '{"from_agent":"agentA","from_session_id":"sessA","to_agent":"agentB","to_session_id":"sessB","payload":{"prompt":"hello"},"parent_task_id":null}'
```

### Завершение задачи (подставьте task_id из ответа предыдущей команды)
```powershell
curl.exe -X POST "http://127.0.0.1:8090/tasks/{task_id}/complete" -H "Authorization: Bearer board_token" -H "Content-Type: application/json" -d '{"result":"done"}'
```
