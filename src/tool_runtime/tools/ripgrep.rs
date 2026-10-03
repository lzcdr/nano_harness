// src/tool_runtime/tools/ripgrep.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_json, parse_input};
use crate::tool_runtime::ToolImpl;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

pub struct Ripgrep;

const MAX_TOTAL_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_MATCHES_PER_QUERY: usize = 500;
const TIMEOUT_SEC_PER_QUERY: u64 = 20;

fn validate_relative(path_str: &str) -> Result<PathBuf, String> {
    if path_str.is_empty() {
        return Err("пустой путь".to_string());
    }
    let p = Path::new(path_str);
    if p.is_absolute() {
        return Err("абсолютный путь запрещён".to_string());
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir => return Err("'..' в пути запрещён".to_string()),
            Component::RootDir => return Err("корневой путь запрещён".to_string()),
            Component::Prefix(_) => return Err("префикс пути запрещён".to_string()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    Ok(out)
}

fn validate_glob(g: &str) -> Result<(), String> {
    if g.is_empty() {
        return Err("пустой glob".to_string());
    }
    for c in g.chars() {
        if c.is_ascii_alphanumeric()
            || c == '*'
            || c == '?'
            || c == '.'
            || c == '_'
            || c == '-'
            || c == '/'
        {
            continue;
        }
        return Err(format!("недопустимый символ в glob: '{}'", c));
    }
    if g.contains("..") {
        return Err("'..' в glob запрещён".to_string());
    }
    if g.starts_with('/') {
        return Err("glob не может начинаться с '/'".to_string());
    }
    Ok(())
}

struct Query {
    path: String,
    pattern: String,
    glob: Option<String>,
}

fn result_object(
    path: &str,
    pattern: &str,
    glob: Option<&str>,
    matches: Vec<Value>,
    total: usize,
    error: Option<String>,
    truncated: bool,
) -> Value {
    let shown = matches.len();
    json!({
        "path": path,
        "pattern": pattern,
        "glob": glob,
        "matches": matches,
        "total": total,
        "shown": shown,
        "truncated": truncated,
        "error": error,
    })
}

fn result_error(path: &str, pattern: &str, glob: Option<&str>, error: String) -> Value {
    result_object(path, pattern, glob, vec![], 0, Some(error), false)
}

fn parse_query(v: &Value) -> Result<Query, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "запрос должен быть объектом".to_string())?;
    let path = obj
        .get("path")
        .and_then(|x| x.as_str())
        .ok_or_else(|| "поле 'path' обязательно".to_string())?
        .to_string();
    if path.is_empty() {
        return Err("поле 'path' пустое".to_string());
    }
    let pattern = obj
        .get("pattern")
        .and_then(|x| x.as_str())
        .ok_or_else(|| "поле 'pattern' обязательно".to_string())?
        .to_string();
    if pattern.is_empty() {
        return Err("поле 'pattern' пустое".to_string());
    }
    let glob = match obj.get("glob") {
        Some(Value::Null) | None => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err("поле 'glob' должно быть строкой".to_string()),
    };
    Ok(Query {
        path,
        pattern,
        glob,
    })
}

async fn execute_one(project_dir: &Path, q: &Query) -> (Vec<Value>, usize, Option<String>) {
    let rel = match validate_relative(&q.path) {
        Ok(p) => p,
        Err(e) => return (vec![], 0, Some(format!("path: {}", e))),
    };

    if let Some(ref g) = q.glob {
        if let Err(e) = validate_glob(g) {
            return (vec![], 0, Some(format!("glob: {}", e)));
        }
    }

    let target = project_dir.join(&rel);
    let canonical_target = match target.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return (
                vec![],
                0,
                Some(format!("путь '{}' не существует", rel.display())),
            );
        }
    };
    let canonical_root = match project_dir.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            return (
                vec![],
                0,
                Some(format!("не удалось развернуть корень проекта: {}", e)),
            );
        }
    };
    if !canonical_target.starts_with(&canonical_root) {
        return (
            vec![],
            0,
            Some("путь выходит за пределы проекта".to_string()),
        );
    }

    let mut cmd = Command::new("rg");
    cmd.current_dir(project_dir);
    cmd.kill_on_drop(true);
    cmd.stdin(std::process::Stdio::null());
    cmd.arg("--json");
    cmd.arg("--no-follow");
    cmd.arg("--max-count")
        .arg(MAX_MATCHES_PER_QUERY.to_string());
    if let Some(ref g) = q.glob {
        cmd.arg("-g").arg(g);
    }
    cmd.arg(&q.pattern);
    cmd.arg(&rel);

    let output = match tokio::time::timeout(
        Duration::from_secs(TIMEOUT_SEC_PER_QUERY),
        cmd.output(),
    )
    .await
    {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return (
                    vec![],
                    0,
                    Some("ripgrep ('rg') не найден в PATH".to_string()),
                );
            }
            return (vec![], 0, Some(format!("ошибка запуска rg: {}", e)));
        }
        Err(_) => {
            return (
                vec![],
                0,
                Some(format!(
                    "rg: превышен таймаут {} сек",
                    TIMEOUT_SEC_PER_QUERY
                )),
            );
        }
    };

    let status_code = output.status.code().unwrap_or(-1);
    if status_code != 0 && status_code != 1 {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return (
            vec![],
            0,
            Some(format!(
                "rg завершился с кодом {}: {}",
                status_code,
                stderr.trim()
            )),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut matches = Vec::new();
    let mut total = 0usize;

    for line in stdout.lines() {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("type").and_then(|t| t.as_str()) != Some("match") {
            continue;
        }
        total += 1;
        let data = v.get("data").cloned().unwrap_or_default();
        let file = data
            .get("path")
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        let line_no = data
            .get("line_number")
            .and_then(|n| n.as_u64())
            .unwrap_or(0);
        let text = data
            .get("lines")
            .and_then(|l| l.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .trim_end_matches('\n');
        matches.push(json!({
            "file": file,
            "line": line_no,
            "text": text,
        }));
    }

    (matches, total, None)
}

#[async_trait::async_trait]
impl ToolImpl for Ripgrep {
    fn name(&self) -> &'static str {
        "ripgrep"
    }

    fn description(&self) -> &'static str {
        "Поиск по содержимому файлов проекта через ripgrep. \
         Принимает несколько поисковых запросов и выполняет их последовательно. \
         \n\n\
         ВХОД: массив запросов. Каждый запрос — объект с полями path (путь внутри \
         проекта), pattern (строка или регулярное выражение), glob (необязательный \
         фильтр имён файлов). \
         \n\n\
         ВЫХОД: массив результатов в том же порядке, что и запросы. Каждый результат — \
         объект с полями: \
         path, pattern, glob — эхо запроса; \
         matches — массив найденных совпадений, каждое из file, line, text; \
         total — сколько всего найдено; \
         shown — сколько в matches; \
         truncated — true, если часть не поместилась; \
         error — текст ошибки или null. \
         \n\n\
         Если один запрос завершился ошибкой, остальные всё равно выполняются."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "description": "Массив поисковых запросов",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Путь к файлу или папке внутри проекта, относительно корня проекта"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Строка или регулярное выражение для поиска"
                            },
                            "glob": {
                                "type": "string",
                                "description": "Фильтр имён файлов, например '*.rs'. Необязательно"
                            }
                        },
                        "required": ["path", "pattern"]
                    }
                }
            },
            "required": ["queries"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let root = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(format!("некорректный JSON: {}", e)),
        };

        let queries_arr = match root.get("queries").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return err("не указано поле 'queries' или оно не массив"),
        };

        if queries_arr.is_empty() {
            return err("'queries' — пустой массив");
        }

        let mut queries = Vec::with_capacity(queries_arr.len());
        for (i, item) in queries_arr.iter().enumerate() {
            match parse_query(item) {
                Ok(q) => queries.push(q),
                Err(e) => return err(format!("запрос #{}: {}", i, e)),
            }
        }

        let project_dir = ctx.storage_root_path.join("projects").join(&ctx.project_id);

        if !project_dir.exists() {
            return err(format!("проект '{}' не найден на диске", ctx.project_id));
        }

        let mut results: Vec<Value> = Vec::with_capacity(queries.len());
        let mut used_bytes = 0usize;
        let mut budget_exhausted = false;

        for q in &queries {
            if budget_exhausted {
                results.push(result_object(
                    &q.path,
                    &q.pattern,
                    q.glob.as_deref(),
                    vec![],
                    0,
                    None,
                    true,
                ));
                continue;
            }

            let (matches, total, error) = execute_one(&project_dir, q).await;

            if let Some(e) = error {
                results.push(result_error(&q.path, &q.pattern, q.glob.as_deref(), e));
                continue;
            }

            let mut fitted = Vec::new();
            let mut local_truncated = false;
            for m in matches.into_iter() {
                let sz = serde_json::to_string(&m).map(|s| s.len()).unwrap_or(0);
                if used_bytes + sz > MAX_TOTAL_OUTPUT_BYTES {
                    local_truncated = true;
                    break;
                }
                used_bytes += sz;
                fitted.push(m);
            }

            if local_truncated {
                budget_exhausted = true;
            }

            results.push(result_object(
                &q.path,
                &q.pattern,
                q.glob.as_deref(),
                fitted,
                total,
                None,
                local_truncated,
            ));
        }

        ok_json(Value::Array(results))
    }
}

inventory::submit! { &Ripgrep as &'static dyn crate::tool_runtime::ToolImpl }
