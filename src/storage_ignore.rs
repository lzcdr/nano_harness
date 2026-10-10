// SPDX-FileCopyrightText: 2026 lzcdr
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/storage_ignore.rs

use std::io;
use std::path::{Component, Path, PathBuf};

const DEFAULT_CONTENT: &str = "\
.git
.svn
.hg
target
node_modules
dist
build
out
vendor
.venv
venv
__pycache__
.pytest_cache
.mypy_cache
.tox
.gradle
.idea
.vscode
bin
obj
.cache
.next
.nuxt
.parcel-cache
coverage
.nyc_output
";

#[derive(Debug, Clone)]
enum Pattern {
    /// Glob по имени сегмента, в любом месте дерева.
    Name(String),
    /// Glob по пути от корня проекта.
    Path(String),
}

#[derive(Debug, Clone, Default)]
pub struct StorageIgnore {
    patterns: Vec<Pattern>,
}

impl StorageIgnore {
    pub fn empty() -> Self {
        Self { patterns: Vec::new() }
    }

    /// Загружает файл. Если файла нет — пустой список (игнорируется всё подряд).
    pub fn load(path: &Path) -> io::Result<Self> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Self::empty()),
            Err(e) => return Err(e),
        };
        Ok(Self::parse(&content))
    }

    pub fn parse(content: &str) -> Self {
        let mut patterns = Vec::new();
        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('/') {
                patterns.push(Pattern::Path(rest.to_string()));
            } else {
                patterns.push(Pattern::Name(line.to_string()));
            }
        }
        Self { patterns }
    }

    pub fn is_ignored(&self, rel_path: &Path) -> bool {
        for pat in &self.patterns {
            match pat {
                Pattern::Name(glob) => {
                    for comp in rel_path.components() {
                        if let Component::Normal(s) = comp {
                            if let Some(s) = s.to_str() {
                                if glob_match(glob, s) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                Pattern::Path(glob) => {
                    let p: String = rel_path
                        .components()
                        .filter_map(|c| match c {
                            Component::Normal(s) => s.to_str(),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("/");
                    if path_glob_match(glob, &p) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Простой glob: `*`, `?`. Без `**`, без escapes.
fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    glob_match_rec(&p, &n)
}

fn glob_match_rec(p: &[char], n: &[char]) -> bool {
    if p.is_empty() {
        return n.is_empty();
    }
    match p[0] {
        '*' => {
            // * матчит ноль или больше символов
            for i in 0..=n.len() {
                if glob_match_rec(&p[1..], &n[i..]) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if n.is_empty() {
                false
            } else {
                glob_match_rec(&p[1..], &n[1..])
            }
        }
        c => {
            if n.is_empty() || n[0] != c {
                false
            } else {
                glob_match_rec(&p[1..], &n[1..])
            }
        }
    }
}

/// Матч пути по сегментам. `*` внутри сегмента не переходит через `/`.
fn path_glob_match(pattern: &str, path: &str) -> bool {
    let p_segs: Vec<&str> = pattern.split('/').collect();
    let n_segs: Vec<&str> = path.split('/').collect();
    if p_segs.len() != n_segs.len() {
        return false;
    }
    for (p, n) in p_segs.iter().zip(n_segs.iter()) {
        if !glob_match(p, n) {
            return false;
        }
    }
    true
}

/// Путь к глобальному .storageignore рядом с текущим exe.
pub fn global_ignore_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    Some(dir.join(".storageignore"))
}

/// Гарантирует, что глобальный .storageignore существует.
/// Если нет — создаёт с дефолтным содержимым.
pub fn ensure_global_ignore() -> io::Result<PathBuf> {
    let path = global_ignore_path()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "cannot locate exe dir"))?;
    if !path.exists() {
        std::fs::write(&path, DEFAULT_CONTENT)?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_comments_and_blanks() {
        let s = "# comment\n\nfoo\ntarget\n";
        let si = StorageIgnore::parse(s);
        assert_eq!(si.patterns.len(), 2);
    }

    #[test]
    fn parse_path_vs_name() {
        let s = "foo\n/bar/baz.txt\n";
        let si = StorageIgnore::parse(s);
        match &si.patterns[0] {
            Pattern::Name(n) => assert_eq!(n, "foo"),
            _ => panic!(),
        }
        match &si.patterns[1] {
            Pattern::Path(p) => assert_eq!(p, "bar/baz.txt"),
            _ => panic!(),
        }
    }

    #[test]
    fn name_matches_in_any_segment() {
        let si = StorageIgnore::parse("target\n");
        assert!(si.is_ignored(Path::new("target")));
        assert!(si.is_ignored(Path::new("a/target")));
        assert!(si.is_ignored(Path::new("a/b/target")));
        assert!(si.is_ignored(Path::new("target/debug/foo.d")));
        assert!(!si.is_ignored(Path::new("src/main.rs")));
    }

    #[test]
    fn name_glob() {
        let si = StorageIgnore::parse("foo*\n");
        assert!(si.is_ignored(Path::new("foobar")));
        assert!(si.is_ignored(Path::new("a/foo")));
        assert!(si.is_ignored(Path::new("a/foo123/x.txt")));
        assert!(!si.is_ignored(Path::new("barfoo")));
    }

    #[test]
    fn path_matches_from_root() {
        let si = StorageIgnore::parse("/target\n");
        assert!(si.is_ignored(Path::new("target")));
        assert!(si.is_ignored(Path::new("target/debug")));
        assert!(!si.is_ignored(Path::new("a/target")));
    }

    #[test]
    fn path_glob_segments() {
        let si = StorageIgnore::parse("/foo*/bar/baz.txt\n");
        assert!(si.is_ignored(Path::new("foo123/bar/baz.txt")));
        assert!(!si.is_ignored(Path::new("foo123/x/baz.txt")));
        assert!(!si.is_ignored(Path::new("a/foo123/bar/baz.txt")));
    }

    #[test]
    fn glob_star_and_question() {
        assert!(glob_match("*.log", "a.log"));
        assert!(glob_match("*.log", ".log"));
        assert!(!glob_match("*.log", "a.txt"));
        assert!(glob_match("f?o", "foo"));
        assert!(!glob_match("f?o", "fooo"));
    }
}
