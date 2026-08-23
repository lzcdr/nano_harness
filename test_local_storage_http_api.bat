@echo off
chcp 65001 >nul
setlocal

REM === НАСТРОЙКИ ===
set BASE_URL=http://127.0.0.1:8080
set TOKEN=secret

REM ============================================
REM 1. Создание директории docs
echo [1] Создание директории docs
curl -s -i -X POST "%BASE_URL%/dirs?path=docs" -H "Authorization: Bearer %TOKEN%"
echo.

REM 2. Создание файла docs/hello.txt
echo [2] Создание файла docs/hello.txt
curl -s -i -X POST "%BASE_URL%/files?path=docs/hello.txt" -H "Authorization: Bearer %TOKEN%" -d "Привет, мир! Это тестовый файл."
echo.

REM 3. Чтение файла docs/hello.txt
echo [3] Чтение файла docs/hello.txt
curl -s "%BASE_URL%/files?path=docs/hello.txt" -H "Authorization: Bearer %TOKEN%"
echo.

REM 4. Листинг директории docs
echo [4] Листинг директории docs
curl -s "%BASE_URL%/list?path=docs" -H "Authorization: Bearer %TOKEN%"
echo.

REM 5. Рекурсивный обход docs
echo [5] Рекурсивный обход docs
curl -s "%BASE_URL%/walk?path=docs" -H "Authorization: Bearer %TOKEN%"
echo.

REM 6. Поиск файлов по имени "hello"
echo [6] Поиск по имени "hello"
curl -s "%BASE_URL%/search_name?pattern=hello" -H "Authorization: Bearer %TOKEN%"
echo.

REM 7. Семантический поиск (требует модель, при желании раскомментируйте)
echo [7] Семантический поиск "тестовый файл"
curl -s "%BASE_URL%/search?query=тестовый%%20файл&top_k=3" -H "Authorization: Bearer %TOKEN%"
echo.

REM 8. Запись .about для docs
echo [8] Запись .about
curl -s -i -X POST "%BASE_URL%/about?path=docs" -H "Authorization: Bearer %TOKEN%" -d "Эта директория содержит тестовые файлы."
echo.

REM 9. Чтение .about
echo [9] Чтение .about
curl -s "%BASE_URL%/about?path=docs" -H "Authorization: Bearer %TOKEN%"
echo.

REM 10. Запись .summary
echo [10] Запись .summary
curl -s -i -X POST "%BASE_URL%/summary?path=docs" -H "Authorization: Bearer %TOKEN%" -d "Краткое описание."
echo.

REM 11. Чтение .summary
echo [11] Чтение .summary
curl -s "%BASE_URL%/summary?path=docs" -H "Authorization: Bearer %TOKEN%"
echo.

REM 12. Конфликт: создание папки по пути существующего файла
echo [12] Попытка создать папку docs/hello.txt
curl -s -i -X POST "%BASE_URL%/dirs?path=docs/hello.txt" -H "Authorization: Bearer %TOKEN%"
echo.

REM 13. Конфликт: создание файла по пути существующей папки
echo [13] Попытка создать файл docs (папка)
curl -s -i -X POST "%BASE_URL%/files?path=docs" -H "Authorization: Bearer %TOKEN%" -d "test"
echo.

REM 14. Удаление файла docs/hello.txt
echo [14] Удаление файла docs/hello.txt
curl -s -i -X DELETE "%BASE_URL%/files?path=docs/hello.txt" -H "Authorization: Bearer %TOKEN%"
echo.

REM 15. Проверка авторизации (без токена)
echo [15] Запрос без авторизации
curl -s -i "%BASE_URL%/list?path=docs"
echo.

echo.
echo Все тесты выполнены.
pause
