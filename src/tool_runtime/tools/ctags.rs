// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/ctags.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_json, parse_input, require_string};
use crate::tool_runtime::ToolImpl;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use tokio::process::Command;

pub struct Ctags;

const DEFAULT_EXCLUDES: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".gradle",
    ".idea",
    ".vscode",
];

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

/// Читает .gitignore из корня проекта и возвращает список паттернов,
/// пригодных для передачи в ctags --exclude.
///
/// Игнорирует: пустые строки, комментарии, негативные паттерны (начало с '!'),
/// завершающий '/' (директории), ведущий '/' (якорь от корня).
fn read_gitignore_patterns(project_dir: &Path) -> Vec<String> {
    let gitignore_path = project_dir.join(".gitignore");
    let content = match std::fs::read_to_string(&gitignore_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        let pat = line.trim_start_matches('/').trim_end_matches('/');
        if pat.is_empty() {
            continue;
        }
        if !out.iter().any(|p: &String| p == pat) {
            out.push(pat.to_string());
        }
    }
    out
}

#[async_trait::async_trait]
impl ToolImpl for Ctags {
    fn name(&self) -> &'static str {
        "ctags"
    }

    fn description(&self) -> &'static str {
        "Индексирует символы в исходном коде проекта через universal-ctags. \
         Показывает, где определён каждый символ (функция, класс, метод, переменная, \
         структура): имя, тип, файл, номер строки, язык, область видимости, сигнатуру. \
         Используй, чтобы найти определение функции или класса, не читая файл целиком. \
         Автоматически учитывает .gitignore и пропускает служебные директории \
         (target, node_modules, .git и другие)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу или папке внутри проекта, относительно корня проекта"
                },
                "languages": {
                    "type": "string",
                    "description": "Опциональный фильтр языков через запятую, например 'Rust' или 'Python,JavaScript'. Если не указан, обрабатываются все известные ctags языки."
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
        let languages = args
            .get("languages")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

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

        // Собираем исключения: дефолтные + из .gitignore.
        let mut excludes: Vec<String> = DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();
        for p in read_gitignore_patterns(&project_dir) {
            if !excludes.iter().any(|e| e == &p) {
                excludes.push(p);
            }
        }

        let mut cmd = Command::new("ctags");
        cmd.current_dir(&project_dir);
        cmd.kill_on_drop(true);
        cmd.stdin(std::process::Stdio::null());
        cmd.arg("--output-format=json");
        cmd.arg("--recurse=yes");
        cmd.arg("--sort=no");
        cmd.arg("--fields=+nKSl");
        cmd.arg("-o");
        cmd.arg("-");
        if let Some(ref lang) = languages {
            cmd.arg(format!("--languages={}", lang));
        }
        for ex in &excludes {
            cmd.arg(format!("--exclude={}", ex));
        }
        cmd.arg(&rel);

        let output = match cmd.output().await {
            Ok(o) => o,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return err("ctags не найден в PATH");
                }
                return err(format!("ошибка запуска ctags: {}", e));
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return err(format!(
                "ctags завершился с кодом {}: {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut symbols = Vec::new();
        let mut total = 0usize;

        for line in stdout.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("_type").and_then(|t| t.as_str()) != Some("tag") {
                continue;
            }
            total += 1;
            let name = v
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let kind = v
                .get("kind")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let file = v
                .get("path")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let line_no = v.get("line").and_then(|x| x.as_u64()).unwrap_or(0);
            let language = v
                .get("language")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let scope = v
                .get("scope")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());
            let signature = v
                .get("signature")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());

            symbols.push(json!({
                "name": name,
                "kind": kind,
                "file": file,
                "line": line_no,
                "language": language,
                "scope": scope,
                "signature": signature,
            }));
        }

        ok_json(json!({
            "path": path,
            "total": total,
            "symbols": symbols,
            "error": null,
        }))
    }
}

inventory::submit! { &Ctags as &'static dyn crate::tool_runtime::ToolImpl }

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
        let p = validate_relative("src/main.rs").unwrap();
        assert!(p.to_string_lossy().contains("main.rs"));
    }

    #[test]
    fn validate_relative_dot_ok() {
        let p = validate_relative(".").unwrap();
        assert_eq!(p.to_string_lossy(), ".");
    }

    #[test]
    fn read_gitignore_missing_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_gitignore_patterns(dir.path()).is_empty());
    }

    #[test]
    fn read_gitignore_skips_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let gi = dir.path().join(".gitignore");
        fs::write(&gi, "# comment\n\n  \ntarget\nnode_modules\n").unwrap();
        let pats = read_gitignore_patterns(dir.path());
        assert_eq!(pats, vec!["target", "node_modules"]);
    }

    #[test]
    fn read_gitignore_skips_negations() {
        let dir = tempfile::tempdir().unwrap();
        let gi = dir.path().join(".gitignore");
        fs::write(&gi, "target\n!keep_me\nbuild\n").unwrap();
        let pats = read_gitignore_patterns(dir.path());
        assert_eq!(pats, vec!["target", "build"]);
    }

    #[test]
    fn read_gitignore_strips_leading_and_trailing_slashes() {
        let dir = tempfile::tempdir().unwrap();
        let gi = dir.path().join(".gitignore");
        fs::write(&gi, "/target/\n/dist\n").unwrap();
        let pats = read_gitignore_patterns(dir.path());
        assert_eq!(pats, vec!["target", "dist"]);
    }

    #[test]
    fn read_gitignore_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let gi = dir.path().join(".gitignore");
        fs::write(&gi, "target\ntarget\nnode_modules\n").unwrap();
        let pats = read_gitignore_patterns(dir.path());
        assert_eq!(pats, vec!["target", "node_modules"]);
    }
}
