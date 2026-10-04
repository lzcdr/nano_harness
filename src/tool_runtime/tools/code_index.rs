// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/tool_runtime/tools/code_index.rs

use crate::tool_runtime::context::ToolContext;
use crate::tool_runtime::primitives::{err, ok_json, parse_input, require_string};
use crate::tool_runtime::ToolImpl;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tokio::process::Command;

pub struct CodeIndex;

const SCC_TIMEOUT_SEC: u64 = 120;
const CTAGS_TIMEOUT_SEC: u64 = 300;

const EXCLUDED_DIRS: &[&str] = &[
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
    "bin",
    "obj",
    ".cache",
    ".next",
    ".nuxt",
    ".parcel-cache",
    "coverage",
    ".nyc_output",
];

const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "go", "js", "ts", "jsx", "tsx", "java", "kt", "kts", "c", "h", "cpp", "hpp", "cc",
    "cxx", "cs", "rb", "php", "swift", "scala", "clj", "ex", "exs", "erl", "hs", "ml", "lua", "sh",
    "bash", "ps1", "pl", "r", "jl", "dart", "vue", "svelte", "m", "mm",
];

const CONFIG_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain",
    "rust-toolchain.toml",
    "rustfmt.toml",
    ".rustfmt.toml",
    "clippy.toml",
    "CMakeLists.txt",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "configure",
    "configure.ac",
    "meson.build",
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "tox.ini",
    "pytest.ini",
    "mypy.ini",
    "ruff.toml",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "tsconfig.json",
    ".eslintrc",
    ".eslintrc.json",
    ".prettierrc",
    "webpack.config.js",
    "vite.config.js",
    "rollup.config.js",
    "next.config.js",
    "nuxt.config.js",
    "packages.config",
    "Directory.Build.props",
    "global.json",
    "nuget.config",
    "go.mod",
    "go.sum",
    "Gemfile",
    "Gemfile.lock",
    "Rakefile",
    ".ruby-version",
    "composer.json",
    "composer.lock",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".dockerignore",
    ".gitignore",
    ".gitattributes",
    "README.md",
    "README",
    "LICENSE",
    "LICENSE.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
];

const CONFIG_EXTENSIONS: &[&str] = &["csproj", "fsproj", "vbproj", "sln"];

const EXPORT_KINDS: &[&str] = &[
    "function",
    "func",
    "fn",
    "struct",
    "class",
    "interface",
    "trait",
    "enum",
    "type",
    "typedef",
    "constant",
    "const",
    "static",
    "module",
    "namespace",
    "union",
    "protocol",
];

const FUNCTION_KINDS: &[&str] = &["function", "func", "fn", "method"];

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

fn to_forward_slashes(s: &str) -> String {
    s.replace('\\', "/")
}

fn is_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SOURCE_EXTENSIONS.iter().any(|s| s.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

fn is_config_file(name: &str) -> bool {
    if CONFIG_FILES.iter().any(|c| c == &name) {
        return true;
    }
    if let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str()) {
        if CONFIG_EXTENSIONS
            .iter()
            .any(|c| c.eq_ignore_ascii_case(ext))
        {
            return true;
        }
    }
    false
}

fn is_export_kind(kind: &str) -> bool {
    EXPORT_KINDS.iter().any(|k| k.eq_ignore_ascii_case(kind))
}

fn is_function_kind(kind: &str) -> bool {
    FUNCTION_KINDS.iter().any(|k| k.eq_ignore_ascii_case(kind))
}

fn is_comment_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//")
        || t.starts_with("/*")
        || t.starts_with('*')
        || t.starts_with('#')
        || t.starts_with("--")
        || t.starts_with(";;")
        || t.starts_with("%%")
}

/// Очищает комментарий от языковых префиксов и схлопывает в одну строку.
fn clean_comment(raw: &str) -> String {
    let mut pieces: Vec<String> = Vec::new();
    for line in raw.lines() {
        let mut t = line.trim();
        // Блочные обёртки.
        if t.starts_with("/*") {
            t = t.trim_start_matches("/*");
        }
        if t.starts_with("/**") {
            t = t.trim_start_matches("/**");
        }
        if t.ends_with("*/") {
            t = t.trim_end_matches("*/");
        }
        // Line-префиксы.
        if t.starts_with("///") {
            t = t.trim_start_matches("///");
        } else if t.starts_with("//!") {
            t = t.trim_start_matches("//!");
        } else if t.starts_with("//") {
            t = t.trim_start_matches("//");
        }
        if t.starts_with('#') {
            t = t.trim_start_matches('#');
        }
        if t.starts_with("--") {
            t = t.trim_start_matches("--");
        }
        if t.starts_with(";;") {
            t = t.trim_start_matches(";;");
        }
        if t.starts_with("%%") {
            t = t.trim_start_matches("%%");
        }
        if t.starts_with('*') {
            t = t.trim_start_matches('*');
        }
        let t = t.trim();
        if !t.is_empty() {
            pieces.push(t.to_string());
        }
    }
    pieces.join(" ")
}

// ==================== Комментарии символов ====================

fn extract_comment(lines: &[&str], line_no: usize, language: &str) -> Option<String> {
    if language == "Python" {
        extract_python_docstring(lines, line_no)
    } else {
        extract_above(lines, line_no)
    }
}

fn extract_above(lines: &[&str], line_no: usize) -> Option<String> {
    let mut idx = match line_no.checked_sub(2) {
        Some(i) => i,
        None => return None,
    };

    let mut collected: Vec<String> = Vec::new();

    loop {
        let trimmed = lines[idx].trim();

        if trimmed.starts_with("#[") || trimmed.starts_with("#![") || trimmed.starts_with('@') {
            if idx == 0 {
                break;
            }
            idx -= 1;
            continue;
        }

        if trimmed.is_empty() {
            break;
        }

        if is_comment_line(trimmed) {
            collected.push(trimmed.to_string());
            if idx == 0 {
                break;
            }
            idx -= 1;
            continue;
        }

        break;
    }

    if collected.is_empty() {
        return None;
    }
    collected.reverse();
    let joined = collected.join("\n");
    let cleaned = clean_comment(&joined);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn extract_python_docstring(lines: &[&str], line_no: usize) -> Option<String> {
    let mut idx = line_no;

    while idx < lines.len() && lines[idx].trim().is_empty() {
        idx += 1;
    }
    if idx >= lines.len() {
        return None;
    }

    let first = lines[idx].trim();
    let (opener, same_line_close) = if first.starts_with("\"\"\"") {
        ("\"\"\"", first.len() > 6 && first.ends_with("\"\"\""))
    } else if first.starts_with("'''") {
        ("'''", first.len() > 6 && first.ends_with("'''"))
    } else {
        return None;
    };

    if same_line_close {
        let inner = first
            .trim_start_matches(opener)
            .trim_end_matches(opener)
            .trim();
        if inner.is_empty() {
            return None;
        }
        return Some(inner.to_string());
    }

    let mut collected: Vec<String> = Vec::new();
    let first_after_opener = first.trim_start_matches(opener).trim();
    if !first_after_opener.is_empty() {
        collected.push(first_after_opener.to_string());
    }

    idx += 1;
    while idx < lines.len() {
        let line = lines[idx];
        let trimmed = line.trim();
        if trimmed.ends_with(opener) {
            let before_close = trimmed.trim_end_matches(opener).trim();
            if !before_close.is_empty() {
                collected.push(before_close.to_string());
            }
            break;
        }
        collected.push(trimmed.to_string());
        idx += 1;
    }

    while collected.first().map(|s| s.is_empty()).unwrap_or(false) {
        collected.remove(0);
    }
    while collected.last().map(|s| s.is_empty()).unwrap_or(false) {
        collected.pop();
    }
    if collected.is_empty() {
        return None;
    }
    let joined = collected.join(" ");
    let cleaned = joined.trim();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn extract_file_doc(lines: &[&str]) -> Option<String> {
    let mut collected: Vec<String> = Vec::new();
    for raw in lines.iter().take(30) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if collected.is_empty() {
                continue;
            }
            collected.push(String::new());
            continue;
        }
        if is_comment_line(trimmed) {
            collected.push(trimmed.to_string());
            continue;
        }
        break;
    }
    if collected.is_empty() {
        return None;
    }
    while collected.last().map(|s| s.is_empty()).unwrap_or(false) {
        collected.pop();
    }
    let joined = collected.join("\n");
    let cleaned = clean_comment(&joined);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

// ==================== Публичность экспорта ====================

fn line_text<'a>(lines: &'a [String], line_no: u64) -> Option<&'a str> {
    if line_no == 0 {
        return None;
    }
    let idx = (line_no - 1) as usize;
    lines.get(idx).map(|s| s.as_str())
}

fn is_public_export(name: &str, _kind: &str, file_ext: &str, decl_line: Option<&str>) -> bool {
    let ext = file_ext.to_ascii_lowercase();
    let decl = decl_line.unwrap_or("").trim_start();

    match ext.as_str() {
        "rs" => decl.starts_with("pub ") || decl.starts_with("pub("),
        "go" => name
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false),
        "py" => !name.starts_with('_'),
        "js" | "jsx" | "ts" | "tsx" | "vue" | "svelte" => decl.contains("export "),
        "java" | "cs" | "scala" => decl.contains("public "),
        "kt" | "kts" => {
            !decl.contains("private ")
                && !decl.contains("internal ")
                && !decl.contains("protected ")
        }
        "c" | "cpp" | "cc" | "cxx" => false,
        "h" | "hpp" => true,
        "rb" | "php" | "swift" => {
            !decl.contains("private ")
                && !decl.contains("protected ")
                && !decl.contains("internal ")
        }
        "sh" | "bash" | "ps1" | "lua" | "pl" => true,
        _ => true,
    }
}

// ==================== Импорты ====================

fn extract_imports(rel_path: &Path, content: &str) -> (Vec<String>, Vec<String>) {
    let ext = rel_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut internal: Vec<String> = Vec::new();
    let mut external: Vec<String> = Vec::new();
    let mut in_block_comment = false;

    let push = |from: String,
                is_external: bool,
                internal: &mut Vec<String>,
                external: &mut Vec<String>| {
        if is_external {
            if !external.contains(&from) {
                external.push(from);
            }
        } else if !internal.contains(&from) {
            internal.push(from);
        }
    };

    for raw in content.lines() {
        let line = raw.trim();

        if in_block_comment {
            if line.contains("*/") {
                in_block_comment = false;
            }
            continue;
        }
        if line.starts_with("/*") && !line.contains("*/") {
            in_block_comment = true;
            continue;
        }
        if line.starts_with("//") || line.starts_with('*') {
            continue;
        }
        if line.is_empty() {
            continue;
        }

        match ext.as_str() {
            "rs" => {
                if let Some(rest) = line.strip_prefix("use ") {
                    if let Some(end) = rest.find(';') {
                        let path = rest[..end].trim();
                        let is_ext = !(path.starts_with("crate::")
                            || path.starts_with("self::")
                            || path.starts_with("super::"));
                        push(path.to_string(), is_ext, &mut internal, &mut external);
                    }
                } else if let Some(rest) = line.strip_prefix("extern crate ") {
                    if let Some(end) = rest.find(';') {
                        push(
                            rest[..end].trim().to_string(),
                            true,
                            &mut internal,
                            &mut external,
                        );
                    }
                }
            }
            "py" => {
                if let Some(rest) = line.strip_prefix("from ") {
                    if let Some(idx) = rest.find(" import") {
                        let module = rest[..idx].trim();
                        if !module.is_empty() {
                            let is_ext = !module.starts_with('.');
                            push(module.to_string(), is_ext, &mut internal, &mut external);
                        }
                    }
                } else if let Some(rest) = line.strip_prefix("import ") {
                    for part in rest.split(',') {
                        let name = part.split_whitespace().next().unwrap_or("");
                        if !name.is_empty() {
                            push(name.to_string(), true, &mut internal, &mut external);
                        }
                    }
                }
            }
            "js" | "jsx" | "ts" | "tsx" | "vue" | "svelte" => {
                if line.starts_with("import ") || line.starts_with("export ") {
                    if let Some(idx) = line.rfind("from ") {
                        let rest = line[idx + 5..].trim();
                        let path = rest
                            .trim_end_matches(';')
                            .trim()
                            .trim_matches(|c: char| c == '"' || c == '\'' || c == '`');
                        if !path.is_empty() {
                            let is_ext = !path.starts_with('.') && !path.starts_with('/');
                            push(path.to_string(), is_ext, &mut internal, &mut external);
                        }
                    }
                }
                if let Some(idx) = line.find("require(") {
                    let rest = &line[idx + 8..];
                    if let Some(end) = rest.find(')') {
                        let inner = rest[..end].trim();
                        let path = inner.trim_matches(|c: char| c == '"' || c == '\'' || c == '`');
                        if !path.is_empty() {
                            let is_ext = !path.starts_with('.') && !path.starts_with('/');
                            push(path.to_string(), is_ext, &mut internal, &mut external);
                        }
                    }
                }
            }
            "java" | "kt" | "kts" | "scala" => {
                if let Some(rest) = line.strip_prefix("import ") {
                    if let Some(end) = rest.find(';') {
                        let path = rest[..end].trim();
                        push(path.to_string(), true, &mut internal, &mut external);
                    }
                }
            }
            "c" | "h" | "cpp" | "hpp" | "cc" | "cxx" => {
                if let Some(rest) = line.strip_prefix("#include") {
                    let rest = rest.trim();
                    if let Some(inner) = rest.strip_prefix('<') {
                        if let Some(end) = inner.find('>') {
                            push(inner[..end].to_string(), true, &mut internal, &mut external);
                        }
                    } else if let Some(inner) = rest.strip_prefix('"') {
                        if let Some(end) = inner.find('"') {
                            push(
                                inner[..end].to_string(),
                                false,
                                &mut internal,
                                &mut external,
                            );
                        }
                    }
                }
            }
            "cs" => {
                if let Some(rest) = line.strip_prefix("using ") {
                    if let Some(end) = rest.find(';') {
                        let path = rest[..end].trim();
                        push(path.to_string(), true, &mut internal, &mut external);
                    }
                }
            }
            "go" => {
                if let Some(rest) = line.strip_prefix("import ") {
                    let rest = rest.trim();
                    let inner = if rest.starts_with('(') {
                        rest.trim_start_matches('(').trim()
                    } else {
                        rest
                    };
                    let trimmed = inner.trim_matches(|c: char| c == '"' || c == '`' || c == ')');
                    if !trimmed.is_empty() {
                        let first_seg = trimmed.split('/').next().unwrap_or("");
                        let is_ext = first_seg.contains('.');
                        push(trimmed.to_string(), is_ext, &mut internal, &mut external);
                    }
                }
            }
            "rb" => {
                if let Some(rest) = line.strip_prefix("require ") {
                    let path = rest.trim().trim_matches(|c: char| c == '"' || c == '\'');
                    if !path.is_empty() {
                        let is_ext = !path.starts_with('.');
                        push(path.to_string(), is_ext, &mut internal, &mut external);
                    }
                }
            }
            _ => {}
        }
    }

    (internal, external)
}

// ==================== Внешние процессы ====================

async fn run_scc(project_dir: &Path, rel: &Path) -> Result<Value, String> {
    let mut cmd = Command::new("scc");
    cmd.current_dir(project_dir);
    cmd.kill_on_drop(true);
    cmd.stdin(std::process::Stdio::null());
    cmd.arg("--format").arg("json");
    cmd.arg("--by-file");
    cmd.arg(rel);

    let output =
        match tokio::time::timeout(Duration::from_secs(SCC_TIMEOUT_SEC), cmd.output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Err("scc не найден в PATH".to_string());
                }
                return Err(format!("ошибка запуска scc: {}", e));
            }
            Err(_) => return Err(format!("scc: превышен таймаут {} сек", SCC_TIMEOUT_SEC)),
        };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "scc завершился с кодом {}: {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    serde_json::from_slice(&output.stdout).map_err(|e| format!("scc JSON parse: {}", e))
}

struct CtagsSymbol {
    name: String,
    kind: String,
    file: String,
    line: u64,
    scope: Option<String>,
    signature: Option<String>,
}

async fn run_ctags(project_dir: &Path, rel: &Path) -> Result<Vec<CtagsSymbol>, String> {
    let mut cmd = Command::new("ctags");
    cmd.current_dir(project_dir);
    cmd.kill_on_drop(true);
    cmd.stdin(std::process::Stdio::null());
    cmd.arg("--output-format=json");
    cmd.arg("--recurse=yes");
    cmd.arg("--sort=no");
    cmd.arg("--fields=+nKSl");
    cmd.arg("-o");
    cmd.arg("-");
    cmd.arg(rel);

    let output =
        match tokio::time::timeout(Duration::from_secs(CTAGS_TIMEOUT_SEC), cmd.output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Err("ctags не найден в PATH".to_string());
                }
                return Err(format!("ошибка запуска ctags: {}", e));
            }
            Err(_) => {
                return Err(format!("ctags: превышен таймаут {} сек", CTAGS_TIMEOUT_SEC));
            }
        };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ctags завершился с кодом {}: {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut out = Vec::new();
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
        let file = to_forward_slashes(v.get("path").and_then(|x| x.as_str()).unwrap_or(""));
        let line = v.get("line").and_then(|x| x.as_u64()).unwrap_or(0);
        let scope = v
            .get("scope")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let signature = v
            .get("signature")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        out.push(CtagsSymbol {
            name,
            kind,
            file,
            line,
            scope,
            signature,
        });
    }
    Ok(out)
}

async fn walk_directories(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        dirs.push(dir.clone());
        let mut rd = fs::read_dir(&dir).await?;
        while let Some(entry) = rd.next_entry().await? {
            let ft = entry.file_type().await?;
            if ft.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if EXCLUDED_DIRS.iter().any(|d| d == &name) {
                    continue;
                }
                stack.push(entry.path());
            }
        }
    }
    Ok(dirs)
}

async fn read_file_lines(path: &Path) -> Option<Vec<String>> {
    fs::read_to_string(path)
        .await
        .ok()
        .map(|s| s.lines().map(|l| l.to_string()).collect())
}

fn lang_name_for_ext(ext: &str) -> &'static str {
    match ext {
        "rs" => "Rust",
        "py" => "Python",
        "go" => "Go",
        "js" => "JavaScript",
        "jsx" => "JavaScript",
        "ts" => "TypeScript",
        "tsx" => "TypeScript",
        "java" => "Java",
        "kt" | "kts" => "Kotlin",
        "c" => "C",
        "h" => "C",
        "cpp" | "hpp" | "cc" | "cxx" => "C++",
        "cs" => "C#",
        "rb" => "Ruby",
        "php" => "PHP",
        "swift" => "Swift",
        "scala" => "Scala",
        "sh" | "bash" => "Shell",
        "ps1" => "PowerShell",
        "lua" => "Lua",
        "r" => "R",
        "jl" => "Julia",
        "dart" => "Dart",
        "vue" => "Vue",
        "svelte" => "Svelte",
        "clj" => "Clojure",
        "ex" | "exs" => "Elixir",
        "erl" => "Erlang",
        "hs" => "Haskell",
        "ml" => "OCaml",
        _ => "Other",
    }
}

// ==================== Инструмент ====================

#[async_trait::async_trait]
impl ToolImpl for CodeIndex {
    fn name(&self) -> &'static str {
        "code_index"
    }

    fn description(&self) -> &'static str {
        "Строит индекс структуры проекта: список модулей и файлов с метриками, \
         экспорты с сигнатурами и комментариями, импорты с разделением на внешние \
         и внутренние, конфигурационные файлы. Полный индекс сохраняется в \
         CODE_INDEX.json в корне проекта. В ответ возвращается краткая сводка. \
         Все пути в индексе относительны project_root. Для чтения файла используй \
         storage_read_file('{project_root}/{path_from_index}')."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к проекту внутри хранилища, например 'code2prompt' или '.'"
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

        let canonical_project_dir = match project_dir.canonicalize() {
            Ok(p) => p,
            Err(e) => return err(format!("canonicalize project_dir: {}", e)),
        };

        let target = canonical_project_dir.join(&rel);
        let canonical_target = match target.canonicalize() {
            Ok(p) => p,
            Err(_) => return err(format!("путь '{}' не существует", rel.display())),
        };
        if !canonical_target.starts_with(&canonical_project_dir) {
            return err("путь выходит за пределы проекта".to_string());
        }

        let scc_value = match run_scc(&project_dir, &rel).await {
            Ok(v) => v,
            Err(e) => return err(format!("scc: {}", e)),
        };

        let ctags_symbols = match run_ctags(&project_dir, &rel).await {
            Ok(v) => v,
            Err(e) => return err(format!("ctags: {}", e)),
        };

        let dirs = match walk_directories(&canonical_target).await {
            Ok(d) => d,
            Err(e) => return err(format!("обход директорий: {}", e)),
        };

        let mut dir_files: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
        let mut all_files: Vec<PathBuf> = Vec::new();
        for dir in &dirs {
            let mut rd = match fs::read_dir(dir).await {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            let mut files_here: Vec<PathBuf> = Vec::new();
            while let Ok(Some(entry)) = rd.next_entry().await {
                if let Ok(ft) = entry.file_type().await {
                    if ft.is_file() {
                        files_here.push(entry.path());
                        all_files.push(entry.path());
                    }
                }
            }
            files_here.sort();
            dir_files.insert(dir.clone(), files_here);
        }

        let mut modules: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();
        for (dir, files) in &dir_files {
            let src_files: Vec<PathBuf> = files
                .iter()
                .filter(|f| is_source_file(f))
                .cloned()
                .collect();
            if !src_files.is_empty() {
                modules.push((dir.clone(), src_files));
            }
        }

        let mut file_cache: HashMap<PathBuf, Vec<String>> = HashMap::new();
        for f in &all_files {
            if let Some(lines) = read_file_lines(f).await {
                file_cache.insert(f.clone(), lines);
            }
        }

        let mut scc_by_file: HashMap<String, (u64, u64)> = HashMap::new();
        let mut scc_langs: HashMap<String, (u64, u64, u64)> = HashMap::new();
        if let Some(arr) = scc_value.as_array() {
            for item in arr {
                let lang = item
                    .get("Name")
                    .or_else(|| item.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let files_count = item
                    .get("Count")
                    .or_else(|| item.get("count"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let code = item
                    .get("Code")
                    .or_else(|| item.get("code"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let lines = item
                    .get("Lines")
                    .or_else(|| item.get("lines"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                if !lang.is_empty() {
                    let e = scc_langs.entry(lang).or_insert((0, 0, 0));
                    e.0 += files_count;
                    e.1 += code;
                    e.2 += lines;
                }
                if let Some(files_arr) = item.get("Files").and_then(|v| v.as_array()) {
                    for f in files_arr {
                        let fname_raw = f
                            .get("Location")
                            .or_else(|| f.get("location"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let fname = to_forward_slashes(fname_raw);
                        let fcode = f.get("Code").and_then(|v| v.as_u64()).unwrap_or(0);
                        let flines = f.get("Lines").and_then(|v| v.as_u64()).unwrap_or(0);
                        if !fname.is_empty() {
                            scc_by_file.insert(fname, (fcode, flines));
                        }
                    }
                }
            }
        }

        let mut symbols_by_file: HashMap<String, Vec<&CtagsSymbol>> = HashMap::new();
        for s in &ctags_symbols {
            symbols_by_file.entry(s.file.clone()).or_default().push(s);
        }

        let mut config_files: Vec<String> = Vec::new();
        for f in &all_files {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if is_config_file(name) {
                if let Ok(rel_to_project) = f.strip_prefix(&canonical_project_dir) {
                    config_files.push(to_forward_slashes(&rel_to_project.to_string_lossy()));
                }
            }
        }
        config_files.sort();

        let mut modules_json: Vec<Value> = Vec::new();
        let mut total_files: u64 = 0;
        let mut total_code_lines: u64 = 0;
        let mut total_lines: u64 = 0;

        for (dir, src_files) in &modules {
            let module_rel = match dir.strip_prefix(&canonical_project_dir) {
                Ok(p) => to_forward_slashes(&p.to_string_lossy()),
                Err(_) => continue,
            };

            let mut files_json: Vec<Value> = Vec::new();
            let mut module_langs: HashSet<String> = HashSet::new();
            let mut module_code_lines: u64 = 0;
            let mut module_total_lines: u64 = 0;

            for f in src_files {
                let file_rel = match f.strip_prefix(&canonical_project_dir) {
                    Ok(p) => to_forward_slashes(&p.to_string_lossy()),
                    Err(_) => continue,
                };

                total_files += 1;

                let ext = f
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let lang = lang_name_for_ext(&ext).to_string();
                module_langs.insert(lang.clone());

                let (file_code_lines, file_total_lines) =
                    scc_by_file.get(&file_rel).cloned().unwrap_or((0, 0));
                module_code_lines += file_code_lines;
                module_total_lines += file_total_lines;
                total_code_lines += file_code_lines;
                total_lines += file_total_lines;

                let empty: Vec<String> = Vec::new();
                let lines = file_cache.get(f).unwrap_or(&empty);
                let lines_str: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

                // Файл. Doc + экспорты + импорты.
                let doc = extract_file_doc(&lines_str);

                let (internal, external) = extract_imports(f, &lines.join("\n"));

                let mut exports_json: Vec<Value> = Vec::new();
                let symbols = symbols_by_file.get(&file_rel).cloned().unwrap_or_default();
                for s in symbols {
                    if s.scope.is_some() {
                        continue;
                    }
                    if !is_export_kind(&s.kind) {
                        continue;
                    }
                    let decl_line = line_text(lines, s.line);
                    if !is_public_export(&s.name, &s.kind, &ext, decl_line) {
                        continue;
                    }
                    let mut obj = json!({
                        "type": s.kind,
                        "name": s.name,
                        "line": s.line,
                    });
                    if is_function_kind(&s.kind) {
                        obj["signature"] = json!(s.signature);
                    }
                    if let Some(c) = extract_comment(&lines_str, s.line as usize, &lang) {
                        obj["comment"] = json!(c);
                    }
                    exports_json.push(obj);
                }
                exports_json.sort_by(|a, b| {
                    a.get("line")
                        .and_then(|v| v.as_u64())
                        .cmp(&b.get("line").and_then(|v| v.as_u64()))
                });

                files_json.push(json!({
                    "path": file_rel,
                    "code_lines": file_code_lines,
                    "doc": doc,
                    "exports": exports_json,
                    "imports": {
                        "internal": internal,
                        "external": external,
                    }
                }));
            }

            files_json.sort_by(|a, b| {
                let pa = a.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let pb = b.get("path").and_then(|v| v.as_str()).unwrap_or("");
                pa.cmp(pb)
            });

            let mut langs_sorted: Vec<String> = module_langs.into_iter().collect();
            langs_sorted.sort();

            modules_json.push(json!({
                "path": module_rel,
                "file_count": src_files.len(),
                "code_lines": module_code_lines,
                "total_lines": module_total_lines,
                "languages": langs_sorted,
                "files": files_json,
            }));
        }

        modules_json.sort_by(|a, b| {
            let pa = a.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let pb = b.get("path").and_then(|v| v.as_str()).unwrap_or("");
            pa.cmp(pb)
        });

        let mut languages_json: Vec<Value> = Vec::new();
        let mut langs_sorted: Vec<(String, (u64, u64, u64))> = scc_langs.into_iter().collect();
        langs_sorted.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        for (name, (files, code, lines)) in &langs_sorted {
            languages_json.push(json!({
                "name": name,
                "files": files,
                "code_lines": code,
                "total_lines": lines,
            }));
        }

        let generated_at = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();

        let full_index = json!({
            "_description": "Индекс структуры проекта. Сгенерирован инструментом code_index. \
                             Все пути относительны project_root. Для чтения файла используй \
                             storage_read_file('{project_root}/{path}'). Содержит модули и файлы \
                             с метриками, экспорты с сигнатурами (для функций) и комментариями, \
                             импорты с разделением на внешние и внутренние, конфигурационные файлы.",
            "_generated_at": generated_at,
            "project_root": path,
            "config_files": config_files,
            "languages": languages_json,
            "totals": {
                "modules": modules_json.len(),
                "files": total_files,
                "code_lines": total_code_lines,
                "total_lines": total_lines,
            },
            "modules": modules_json,
        });

        let full_str = match serde_json::to_string_pretty(&full_index) {
            Ok(s) => s,
            Err(e) => return err(format!("не удалось сериализовать индекс: {}", e)),
        };

        let code_index_path = if rel == PathBuf::from(".") {
            "CODE_INDEX.json".to_string()
        } else {
            format!(
                "{}/CODE_INDEX.json",
                to_forward_slashes(&rel.to_string_lossy())
            )
        };

        let url = format!("{}/files", ctx.storage_base());
        let write_resp = ctx
            .http_client
            .post(&url)
            .query(&[("path", &code_index_path)])
            .header(
                "Authorization",
                format!("Bearer {}", ctx.storage_auth_token),
            )
            .header("X-NH-Project", &ctx.project_id)
            .body(full_str.clone())
            .send()
            .await;

        match write_resp {
            Ok(r) if r.status().is_success() => {}
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                return err(format!(
                    "не удалось записать {}: HTTP {} — {}",
                    code_index_path, status, body
                ));
            }
            Err(e) => {
                return err(format!("не удалось записать {}: {}", code_index_path, e));
            }
        }

        // Краткая сводка — та же структура, но без files.
        let mut summary_modules: Vec<Value> = Vec::new();
        for m in &modules_json {
            summary_modules.push(json!({
                "path": m.get("path"),
                "file_count": m.get("file_count"),
                "code_lines": m.get("code_lines"),
                "total_lines": m.get("total_lines"),
                "languages": m.get("languages"),
            }));
        }

        let summary = json!({
            "info": "Полный индекс записан в CODE_INDEX.json. Все пути в индексе \
                     относительны project_root. Для чтения файла используй \
                     storage_read_file('{project_root}/{path}').",
            "project_root": path,
            "generated_at": generated_at,
            "config_files": config_files,
            "languages": languages_json,
            "totals": {
                "modules": modules_json.len(),
                "files": total_files,
                "code_lines": total_code_lines,
                "total_lines": total_lines,
            },
            "modules": summary_modules,
            "full_index_file": code_index_path,
        });

        ok_json(summary)
    }
}

inventory::submit! { &CodeIndex as &'static dyn crate::tool_runtime::ToolImpl }

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- validate_relative ----------

    #[test]
    fn validate_relative_rejects_empty() {
        assert!(validate_relative("").is_err());
    }

    #[test]
    fn validate_relative_rejects_absolute() {
        assert!(validate_relative("/etc/passwd").is_err());
    }

    #[test]
    fn validate_relative_rejects_parent_dir() {
        assert!(validate_relative("../etc").is_err());
        assert!(validate_relative("src/../etc").is_err());
    }

    #[test]
    fn validate_relative_normal() {
        let p = validate_relative("src/main.rs").unwrap();
        assert_eq!(to_forward_slashes(&p.to_string_lossy()), "src/main.rs");
    }

    #[test]
    fn validate_relative_cur_dir_skipped() {
        let p = validate_relative("./src").unwrap();
        assert_eq!(p.to_string_lossy(), "src");
    }

    #[test]
    fn validate_relative_empty_components_becomes_dot() {
        let p = validate_relative(".").unwrap();
        assert_eq!(p.to_string_lossy(), ".");
    }

    // ---------- to_forward_slashes ----------

    #[test]
    fn to_forward_slashes_converts_backslashes() {
        assert_eq!(to_forward_slashes("a\\b\\c"), "a/b/c");
        assert_eq!(to_forward_slashes("a/b/c"), "a/b/c");
    }

    // ---------- is_source_file ----------

    #[test]
    fn is_source_file_known_extensions() {
        assert!(is_source_file(Path::new("main.rs")));
        assert!(is_source_file(Path::new("app.py")));
        assert!(is_source_file(Path::new("x.go")));
        assert!(is_source_file(Path::new("y.tsx")));
        assert!(is_source_file(Path::new("z.JS"))); // регистронезависимо
    }

    #[test]
    fn is_source_file_rejects_others() {
        assert!(!is_source_file(Path::new("README.md")));
        assert!(!is_source_file(Path::new("data.json")));
        assert!(!is_source_file(Path::new("no_extension")));
    }

    // ---------- is_config_file ----------

    #[test]
    fn is_config_file_known_names() {
        assert!(is_config_file("Cargo.toml"));
        assert!(is_config_file("package.json"));
        assert!(is_config_file("Dockerfile"));
        assert!(is_config_file("Makefile"));
    }

    #[test]
    fn is_config_file_known_extensions() {
        assert!(is_config_file("proj.csproj"));
        assert!(is_config_file("app.sln"));
    }

    #[test]
    fn is_config_file_rejects_regular() {
        assert!(!is_config_file("main.rs"));
        assert!(!is_config_file("foo.txt"));
    }

    // ---------- is_export_kind / is_function_kind ----------

    #[test]
    fn is_export_kind_recognizes() {
        assert!(is_export_kind("function"));
        assert!(is_export_kind("fn"));
        assert!(is_export_kind("struct"));
        assert!(is_export_kind("CLASS"));
    }

    #[test]
    fn is_export_kind_rejects_unknown() {
        assert!(!is_export_kind("variable"));
        assert!(!is_export_kind(""));
    }

    #[test]
    fn is_function_kind_recognizes() {
        assert!(is_function_kind("function"));
        assert!(is_function_kind("fn"));
        assert!(is_function_kind("method"));
        assert!(is_function_kind("func"));
    }

    #[test]
    fn is_function_kind_rejects_struct() {
        assert!(!is_function_kind("struct"));
    }

    // ---------- is_comment_line ----------

    #[test]
    fn is_comment_line_recognizes_styles() {
        assert!(is_comment_line("// rust"));
        assert!(is_comment_line("  // indented"));
        assert!(is_comment_line("/* c */"));
        assert!(is_comment_line("* inside block"));
        assert!(is_comment_line("# python"));
        assert!(is_comment_line("-- sql"));
        assert!(is_comment_line(";; lisp"));
        assert!(is_comment_line("%% latex"));
    }

    #[test]
    fn is_comment_line_rejects_code() {
        assert!(!is_comment_line("let x = 1;"));
        assert!(!is_comment_line(""));
    }

    // ---------- clean_comment ----------

    #[test]
    fn clean_comment_strips_line_prefixes() {
        assert_eq!(clean_comment("// hello"), "hello");
        assert_eq!(clean_comment("/// doc"), "doc");
        assert_eq!(clean_comment("//! inner"), "inner");
        assert_eq!(clean_comment("# python"), "python");
        assert_eq!(clean_comment("-- sql"), "sql");
    }

    #[test]
    fn clean_comment_strips_block_wrappers() {
        assert_eq!(clean_comment("/* block */"), "block");
        assert_eq!(clean_comment("/** doc */"), "doc");
    }

    #[test]
    fn clean_comment_joins_multiline() {
        let raw = "// line one\n// line two";
        assert_eq!(clean_comment(raw), "line one line two");
    }

    #[test]
    fn clean_comment_skips_empty_lines() {
        let raw = "// a\n//\n// b";
        assert_eq!(clean_comment(raw), "a b");
    }

    #[test]
    fn clean_comment_handles_star_prefixed() {
        let raw = "/*\n * line one\n * line two\n */";
        assert_eq!(clean_comment(raw), "line one line two");
    }

    #[test]
    fn clean_comment_empty_returns_empty() {
        assert_eq!(clean_comment(""), "");
    }

    // ---------- lang_name_for_ext ----------

    #[test]
    fn lang_name_for_common_exts() {
        assert_eq!(lang_name_for_ext("rs"), "Rust");
        assert_eq!(lang_name_for_ext("py"), "Python");
        assert_eq!(lang_name_for_ext("go"), "Go");
        assert_eq!(lang_name_for_ext("ts"), "TypeScript");
        assert_eq!(lang_name_for_ext("tsx"), "TypeScript");
        assert_eq!(lang_name_for_ext("cpp"), "C++");
    }

    #[test]
    fn lang_name_for_unknown() {
        assert_eq!(lang_name_for_ext("xyz"), "Other");
        assert_eq!(lang_name_for_ext(""), "Other");
    }

    // ---------- extract_above ----------

    #[test]
    fn extract_above_simple() {
        let lines: Vec<&str> = vec!["// comment", "fn foo() {}"];
        let c = extract_above(&lines, 2);
        assert_eq!(c.as_deref(), Some("comment"));
    }

    #[test]
    fn extract_above_multiline() {
        let lines: Vec<&str> = vec!["// line one", "// line two", "fn foo() {}"];
        let c = extract_above(&lines, 3);
        assert_eq!(c.as_deref(), Some("line one line two"));
    }

    #[test]
    fn extract_above_skips_attributes() {
        let lines: Vec<&str> = vec!["// doc", "#[derive(Debug)]", "fn foo() {}"];
        let c = extract_above(&lines, 3);
        assert_eq!(c.as_deref(), Some("doc"));
    }

    #[test]
    fn extract_above_stops_on_code() {
        let lines: Vec<&str> = vec!["// doc", "let x = 1;", "fn foo() {}"];
        assert!(extract_above(&lines, 3).is_none());
    }

    #[test]
    fn extract_above_blank_line_breaks() {
        let lines: Vec<&str> = vec!["// doc", "", "fn foo() {}"];
        assert!(extract_above(&lines, 3).is_none());
    }

    #[test]
    fn extract_above_none_when_first_line() {
        let lines: Vec<&str> = vec!["fn foo() {}"];
        assert!(extract_above(&lines, 1).is_none());
    }

    // ---------- extract_file_doc ----------

    #[test]
    fn extract_file_doc_simple() {
        let lines: Vec<&str> = vec!["// file description", "// second line", "", "fn main() {}"];
        let doc = extract_file_doc(&lines);
        assert_eq!(doc.as_deref(), Some("file description second line"));
    }

    #[test]
    fn extract_file_doc_empty_when_starts_with_code() {
        let lines: Vec<&str> = vec!["fn main() {}", "// comment"];
        assert!(extract_file_doc(&lines).is_none());
    }

    // ---------- extract_imports ----------

    #[test]
    fn extract_imports_rust_internal_vs_external() {
        let content = "use std::io;\nuse serde::Serialize;\nuse crate::foo;\nuse super::bar;";
        let (internal, external) = extract_imports(Path::new("x.rs"), content);
        assert!(internal.contains(&"crate::foo".to_string()));
        assert!(internal.contains(&"super::bar".to_string()));
        assert!(external.contains(&"std::io".to_string()));
        assert!(external.contains(&"serde::Serialize".to_string()));
    }

    #[test]
    fn extract_imports_python() {
        let content = "import os\nfrom .foo import bar\nfrom sys import path";
        let (internal, external) = extract_imports(Path::new("x.py"), content);
        assert!(internal.contains(&".foo".to_string()));
        assert!(external.contains(&"os".to_string()));
        assert!(external.contains(&"sys".to_string()));
    }

    #[test]
    fn extract_imports_js() {
        let content = "import x from './local';\nimport y from 'lodash';";
        let (internal, external) = extract_imports(Path::new("x.js"), content);
        assert!(internal.contains(&"./local".to_string()));
        assert!(external.contains(&"lodash".to_string()));
    }

    #[test]
    fn extract_imports_c_include() {
        let content = "#include <stdio.h>\n#include \"myheader.h\"";
        let (internal, external) = extract_imports(Path::new("x.c"), content);
        assert!(internal.contains(&"myheader.h".to_string()));
        assert!(external.contains(&"stdio.h".to_string()));
    }

    #[test]
    fn extract_imports_skips_line_comments() {
        let content = "// use foo::bar;\nuse real::thing;";
        let (_, external) = extract_imports(Path::new("x.rs"), content);
        assert!(!external.contains(&"foo::bar".to_string()));
        assert!(external.contains(&"real::thing".to_string()));
    }

    #[test]
    fn extract_imports_deduplicates() {
        let content = "use std::io;\nuse std::io;";
        let (_, external) = extract_imports(Path::new("x.rs"), content);
        assert_eq!(external.iter().filter(|s| *s == "std::io").count(), 1);
    }

    // ---------- is_public_export ----------

    #[test]
    fn is_public_export_rust_pub() {
        assert!(is_public_export("foo", "fn", "rs", Some("pub fn foo() {}")));
        assert!(is_public_export(
            "foo",
            "fn",
            "rs",
            Some("pub(crate) fn foo() {}")
        ));
    }

    #[test]
    fn is_public_export_rust_private() {
        assert!(!is_public_export("foo", "fn", "rs", Some("fn foo() {}")));
    }

    #[test]
    fn is_public_export_go_capitalized() {
        assert!(is_public_export("Foo", "func", "go", None));
        assert!(!is_public_export("foo", "func", "go", None));
    }

    #[test]
    fn is_public_export_python_underscore() {
        assert!(is_public_export("foo", "function", "py", None));
        assert!(!is_public_export("_foo", "function", "py", None));
    }

    #[test]
    fn is_public_export_js_export() {
        assert!(is_public_export(
            "foo",
            "function",
            "js",
            Some("export function foo()")
        ));
        assert!(!is_public_export(
            "foo",
            "function",
            "js",
            Some("function foo()")
        ));
    }

    // ---------- extract_python_docstring ----------

    #[test]
    fn extract_python_docstring_single_line() {
        let lines: Vec<&str> = vec!["def foo():", "    \"\"\"Short doc.\"\"\"", "    pass"];
        let doc = extract_python_docstring(&lines, 1);
        assert_eq!(doc.as_deref(), Some("Short doc."));
    }

    #[test]
    fn extract_python_docstring_multiline() {
        let lines: Vec<&str> = vec![
            "def foo():",
            "    \"\"\"",
            "    Line one.",
            "    Line two.",
            "    \"\"\"",
            "    pass",
        ];
        let doc = extract_python_docstring(&lines, 1);
        assert_eq!(doc.as_deref(), Some("Line one. Line two."));
    }

    #[test]
    fn extract_python_docstring_none_when_no_docstring() {
        let lines: Vec<&str> = vec!["def foo():", "    pass"];
        assert!(extract_python_docstring(&lines, 1).is_none());
    }

    #[test]
    fn extract_python_docstring_triple_single() {
        let lines: Vec<&str> = vec!["def foo():", "    '''Single quotes.'''", "    pass"];
        let doc = extract_python_docstring(&lines, 1);
        assert_eq!(doc.as_deref(), Some("Single quotes."));
    }
}
