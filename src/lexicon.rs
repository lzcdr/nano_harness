// src/lexicon.rs

use anyhow::{Context, Result};
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

pub struct Lexicon {
    words: HashSet<String>,
    ru_stemmer: Stemmer,
    en_stemmer: Stemmer,
}

fn is_cyrillic(c: char) -> bool {
    let lower = c.to_lowercase().next().unwrap_or(c);
    matches!(lower, 'а'..='я' | 'ё')
}

fn stem(word: &str, ru: &Stemmer, en: &Stemmer) -> String {
    let has_cyrillic = word.chars().any(is_cyrillic);
    let stemmer = if has_cyrillic { ru } else { en };
    stemmer.stem(&word.to_lowercase()).to_string()
}

impl Lexicon {
    pub fn load(dir: &Path) -> Result<Self> {
        let entries = fs::read_dir(dir)
            .with_context(|| format!("не удалось прочитать директорию {}", dir.display()))?;

        let mut words: HashSet<String> = HashSet::new();
        let mut loaded_files = 0usize;
        let ru = Stemmer::create(Algorithm::Russian);
        let en = Stemmer::create(Algorithm::English);

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("dic") {
                continue;
            }
            match load_dic(&path) {
                Ok(raw_words) => {
                    loaded_files += 1;
                    let before = words.len();
                    for w in raw_words {
                        words.insert(stem(&w, &ru, &en));
                    }
                    eprintln!(
                        "📖 Загружен словарь {}: +{} стемов",
                        path.display(),
                        words.len() - before
                    );
                }
                Err(e) => {
                    eprintln!("⚠️ Не удалось загрузить {}: {:#}", path.display(), e);
                }
            }
        }

        if loaded_files == 0 {
            anyhow::bail!(
                "в директории {} не найдено ни одного .dic файла",
                dir.display()
            );
        }

        eprintln!("📖 Всего стемов в лексиконе: {}", words.len());

        // сохраняем стеммеры в структуре, чтобы contains мог их использовать
        Ok(Self {
            words,
            ru_stemmer: ru,
            en_stemmer: en,
        })
    }

    pub fn contains(&self, word: &str) -> bool {
        let s = stem(word, &self.ru_stemmer, &self.en_stemmer);
        self.words.contains(&s)
    }
}

fn load_dic(path: &Path) -> Result<HashSet<String>> {
    let content = fs::read_to_string(path)?;
    let mut set = HashSet::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let word = match line.find('/') {
            Some(idx) => &line[..idx],
            None => line,
        };
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        set.insert(word.to_lowercase());
    }
    Ok(set)
}

static LEXICON: OnceLock<Lexicon> = OnceLock::new();

pub fn init_lexicon(dir: &Path) -> Result<()> {
    if !dir.exists() {
        eprintln!(
            "⚠️ Директория словарей {} не найдена — работаю без лексикона",
            dir.display()
        );
        return Ok(());
    }
    match Lexicon::load(dir) {
        Ok(lex) => {
            let _ = LEXICON.set(lex);
            Ok(())
        }
        Err(e) => {
            eprintln!("⚠️ Лексикон не загружен: {:#}", e);
            Ok(())
        }
    }
}

pub fn get_lexicon() -> Option<&'static Lexicon> {
    LEXICON.get()
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Quoted(String),
    Word(String),
}

pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut current_word = String::new();

    while i < chars.len() {
        let c = chars[i];

        if c == '"' {
            if !current_word.is_empty() {
                tokens.push(Token::Word(std::mem::take(&mut current_word)));
            }
            let mut j = i + 1;
            let mut quoted = String::new();
            while j < chars.len() && chars[j] != '"' {
                quoted.push(chars[j]);
                j += 1;
            }
            if j < chars.len() {
                tokens.push(Token::Quoted(quoted));
                i = j + 1;
            } else {
                current_word.push('"');
                current_word.push_str(&quoted);
                i = j;
            }
            continue;
        }

        if is_word_char(c) {
            current_word.push(c);
            i += 1;
            continue;
        }

        if c == '.' && !current_word.is_empty() && i + 1 < chars.len() && is_word_char(chars[i + 1])
        {
            current_word.push('.');
            i += 1;
            continue;
        }

        if !current_word.is_empty() {
            tokens.push(Token::Word(std::mem::take(&mut current_word)));
        }
        i += 1;
    }

    if !current_word.is_empty() {
        tokens.push(Token::Word(current_word));
    }

    tokens
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

pub fn extract_entities_and_skeleton(
    text: &str,
    lexicon: Option<&Lexicon>,
) -> (String, Vec<String>) {
    let tokens = tokenize(text);
    let mut skeleton = String::new();
    let mut entities = Vec::new();
    let mut first = true;

    for token in tokens {
        if !first {
            skeleton.push(' ');
        }
        first = false;

        match token {
            Token::Quoted(q) => {
                skeleton.push_str("<ENT>");
                entities.push(q.to_lowercase());
            }
            Token::Word(w) => {
                let known = lexicon.map(|l| l.contains(&w)).unwrap_or(false);
                if known {
                    skeleton.push_str(&w);
                } else {
                    skeleton.push_str("<ENT>");
                    entities.push(w.to_lowercase());
                }
            }
        }
    }

    (skeleton, entities)
}

pub fn entities_subset(query: &[String], skill: &[String]) -> bool {
    if skill.is_empty() {
        return true;
    }
    let set: HashSet<&String> = skill.iter().collect();
    query.iter().all(|q| set.contains(q))
}
