// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/scc.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_json, parse_input, require_string};
use crate::tool_runtime::ToolImpl;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

pub struct Scc;

const TIMEOUT_SEC: u64 = 30;

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

fn get_u64(v: &Value, keys: &[&str]) -> u64 {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(|x| x.as_u64()) {
            return n;
        }
    }
    0
}

fn get_str<'a>(v: &'a Value, keys: &[&str]) -> &'a str {
    for k in keys {
        if let Some(s) = v.get(*k).and_then(|x| x.as_str()) {
            return s;
        }
    }
    ""
}

#[async_trait::async_trait]
impl ToolImpl for Scc {
    fn name(&self) -> &'static str {
        "scc"
    }

    fn description(&self) -> &'static str {
        "Подсчёт строк кода по языкам программирования через scc. \
         Принимает путь внутри проекта. Возвращает статистику по каждому языку \
         и суммарно: сколько файлов, всего строк, строк кода, комментариев, \
         пустых строк, оценка сложности и размер в байтах. \
         Учитывает .gitignore."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь внутри проекта. '.' — весь проект, 'src' — только эта папка, 'src/main.rs' — конкретный файл."
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &str, ctx: &ToolContext) -> String {
        let args = match parse_input(input) {
            Ok(v) => v,
            Err(e) => return err(format!("некорректный JSON: {}", e)),
        };
        let path = match require_string(&args, "path") {
            Ok(p) => p,
            Err(e) => return err(e),
        };

        let rel = match validate_relative(&path) {
            Ok(p) => p,
            Err(e) => return err(format!("path: {}", e)),
        };

        let project_dir = ctx.storage_root_path.join("projects").join(&ctx.project_id);

        if !project_dir.exists() {
            return err(format!("проект '{}' не найден на диске", ctx.project_id));
        }

        let target = project_dir.join(&rel);
        let canonical_target = match target.canonicalize() {
            Ok(p) => p,
            Err(_) => return err(format!("путь '{}' не существует", rel.display())),
        };
        let canonical_root = match project_dir.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return err(format!("не удалось развернуть корень проекта: {}", e));
            }
        };
        if !canonical_target.starts_with(&canonical_root) {
            return err("путь выходит за пределы проекта".to_string());
        }

        let mut cmd = Command::new("scc");
        cmd.current_dir(&project_dir);
        cmd.kill_on_drop(true);
        cmd.stdin(std::process::Stdio::null());
        cmd.arg("--format").arg("json");
        cmd.arg(&rel);

        let output =
            match tokio::time::timeout(Duration::from_secs(TIMEOUT_SEC), cmd.output()).await {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        return err("scc не найден в PATH");
                    }
                    return err(format!("ошибка запуска scc: {}", e));
                }
                Err(_) => return err(format!("scc: превышен таймаут {} сек", TIMEOUT_SEC)),
            };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return err(format!(
                "scc завершился с кодом {}: {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ));
        }

        let parsed: Value = match serde_json::from_slice(&output.stdout) {
            Ok(v) => v,
            Err(e) => return err(format!("не удалось распарсить JSON от scc: {}", e)),
        };

        let languages_arr = match parsed.as_array() {
            Some(a) => a,
            None => return err("scc вернул не массив"),
        };

        let mut languages = Vec::with_capacity(languages_arr.len());
        let mut total_files: u64 = 0;
        let mut total_lines: u64 = 0;
        let mut total_code: u64 = 0;
        let mut total_comments: u64 = 0;
        let mut total_blanks: u64 = 0;
        let mut total_complexity: u64 = 0;
        let mut total_bytes: u64 = 0;

        for item in languages_arr {
            let name = get_str(item, &["Name", "name"]).to_string();
            let files = get_u64(item, &["Count", "count"]);
            let lines = get_u64(item, &["Lines", "lines"]);
            let code = get_u64(item, &["Code", "code"]);
            let comments = get_u64(item, &["Comment", "comment", "Comments", "comments"]);
            let blanks = get_u64(item, &["Blank", "blank", "Blanks", "blanks"]);
            let complexity = get_u64(item, &["Complexity", "complexity"]);
            let bytes = get_u64(item, &["Bytes", "bytes"]);

            total_files += files;
            total_lines += lines;
            total_code += code;
            total_comments += comments;
            total_blanks += blanks;
            total_complexity += complexity;
            total_bytes += bytes;

            languages.push(json!({
                "name": name,
                "files": files,
                "lines": lines,
                "code": code,
                "comments": comments,
                "blanks": blanks,
                "complexity": complexity,
                "bytes": bytes,
            }));
        }

        languages.sort_by(|a, b| {
            let ca = a.get("code").and_then(|v| v.as_u64()).unwrap_or(0);
            let cb = b.get("code").and_then(|v| v.as_u64()).unwrap_or(0);
            cb.cmp(&ca)
        });

        ok_json(json!({
            "path": path,
            "totals": {
                "files": total_files,
                "lines": total_lines,
                "code": total_code,
                "comments": total_comments,
                "blanks": total_blanks,
                "complexity": total_complexity,
                "bytes": total_bytes,
            },
            "languages": languages,
            "error": null,
        }))
    }
}

inventory::submit! { &Scc as &'static dyn crate::tool_runtime::ToolImpl }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_relative_rejects_empty() {
        assert!(validate_relative("").is_err());
    }

    #[test]
    fn validate_relative_rejects_absolute() {
        assert!(validate_relative("/etc").is_err());
    }

    #[test]
    fn validate_relative_rejects_parent_dir() {
        assert!(validate_relative("../x").is_err());
        assert!(validate_relative("a/../b").is_err());
    }

    #[test]
    fn validate_relative_normal_ok() {
        let p = validate_relative("src").unwrap();
        assert!(p.to_string_lossy().contains("src"));
    }

    #[test]
    fn validate_relative_dot_ok() {
        let p = validate_relative(".").unwrap();
        assert_eq!(p.to_string_lossy(), ".");
    }

    #[test]
    fn get_u64_returns_first_matching_key() {
        let v = json!({"Count": 5, "count": 7});
        assert_eq!(get_u64(&v, &["Count", "count"]), 5);
    }

    #[test]
    fn get_u64_falls_back_to_second_key() {
        let v = json!({"count": 7});
        assert_eq!(get_u64(&v, &["Count", "count"]), 7);
    }

    #[test]
    fn get_u64_returns_zero_when_missing() {
        let v = json!({});
        assert_eq!(get_u64(&v, &["Count", "count"]), 0);
    }

    #[test]
    fn get_u64_returns_zero_on_wrong_type() {
        let v = json!({"Count": "5"});
        assert_eq!(get_u64(&v, &["Count"]), 0);
    }

    #[test]
    fn get_str_returns_first_matching_key() {
        let v = json!({"Name": "Rust", "name": "C"});
        assert_eq!(get_str(&v, &["Name", "name"]), "Rust");
    }

    #[test]
    fn get_str_falls_back_to_second_key() {
        let v = json!({"name": "C"});
        assert_eq!(get_str(&v, &["Name", "name"]), "C");
    }

    #[test]
    fn get_str_returns_empty_when_missing() {
        let v = json!({});
        assert_eq!(get_str(&v, &["Name", "name"]), "");
    }

    #[test]
    fn get_str_returns_empty_on_wrong_type() {
        let v = json!({"Name": 42});
        assert_eq!(get_str(&v, &["Name"]), "");
    }
}
