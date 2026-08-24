//! Heuristic entity extraction. The paper uses spaCy; anything non-generative
//! qualifies, swap in an ONNX NER via the trait if quality matters.

use crate::text::{is_stopword, split_sentences};
use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EntityKind {
    Named,
    Date,
    Number,
    Quote,
    Url,
    Email,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub display: String,
    pub canon: String,
    pub kind: EntityKind,
}

impl Entity {
    fn new(display: &str, kind: EntityKind) -> Self {
        Self {
            display: display.trim().to_string(),
            canon: canon(display),
            kind,
        }
    }
}

pub fn canon(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let w = w.to_lowercase();
            w.strip_suffix("'s").map(str::to_string).unwrap_or(w)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub trait EntityExtractor: Send + Sync {
    fn extract(&self, text: &str) -> Vec<Entity>;
}

pub struct HeuristicNer;

static MONTHS: &str = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec";

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

struct Patterns {
    fence: Regex,
    inline_code: Regex,
    url: Regex,
    email: Regex,
    iso_date: Regex,
    month_day_year: Regex,
    day_month_year: Regex,
    month_year: Regex,
    year: Regex,
    number: Regex,
    quote: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| Patterns {
        fence: re(r"(?s)```.*?```"),
        inline_code: re(r"`[^`\n]+`"),
        url: re(r"https?://\S+"),
        email: re(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        iso_date: re(r"\b\d{4}-\d{2}-\d{2}\b"),
        month_day_year: re(&format!(
            r"\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
        )),
        day_month_year: re(&format!(
            r"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\.?(?:\s+\d{{4}})?\b"
        )),
        month_year: re(&format!(r"\b(?:{MONTHS})\.?\s+\d{{4}}\b")),
        year: re(r"\b(?:19|20)\d{2}\b"),
        number: re(r"[$€£]?\b\d+(?:[,.]\d+)*%?"),
        quote: re(r#""([^"]{2,120})""#),
    })
}

/// Code carries no named entities; masking it keeps shell flags, JSON
/// fragments, and score-like numbers out of the graph.
fn mask_code(text: &str, p: &Patterns) -> String {
    let no_fences = p.fence.replace_all(text, " ");
    p.inline_code.replace_all(&no_fences, " ").into_owned()
}

/// Quoted shell or JSON fragments that the quote pattern would otherwise
/// promote to entities.
fn is_code_like(s: &str) -> bool {
    s.starts_with('-')
        || s.contains("&&")
        || s.contains("$(")
        || s.contains('`')
        || s.contains('{')
        || s.contains('}')
        || s.contains('\\')
        || s.contains("--")
}

/// Capitalized at sentence start but rarely an entity.
fn sentence_start_noise(w: &str) -> bool {
    let lower = w.to_lowercase();
    is_stopword(&lower)
        || matches!(
            lower.as_str(),
            "hi" | "hey"
                | "hello"
                | "thanks"
                | "thank"
                | "sure"
                | "maybe"
                | "sometimes"
                | "today"
                | "yesterday"
                | "tomorrow"
                | "last"
                | "next"
                | "first"
                | "finally"
                | "anyway"
                | "actually"
                | "honestly"
                | "recently"
                | "everyone"
                | "someone"
                | "nothing"
                | "everything"
                | "please"
                | "sorry"
                | "great"
                | "good"
                | "right"
        )
}

fn is_cap_word(w: &str) -> bool {
    let mut chars = w.chars();
    match chars.next() {
        Some(c) if c.is_uppercase() => {}
        _ => return false,
    }
    w.chars()
        .all(|c| c.is_alphanumeric() || c == '\'' || c == '-' || c == '.')
}

fn is_connector(w: &str) -> bool {
    matches!(
        w,
        "of" | "the" | "and" | "de" | "da" | "di" | "van" | "von" | "la" | "le"
    )
}

impl HeuristicNer {
    fn capitalized_spans(&self, sentence: &str, out: &mut Vec<Entity>) {
        let words: Vec<&str> = sentence
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-' && c != '.')
            })
            .collect();
        let mut i = 0;
        while i < words.len() {
            let w = words[i];
            if w.is_empty() || !is_cap_word(w) {
                i += 1;
                continue;
            }
            let mut span = vec![w];
            let mut j = i + 1;
            while j < words.len() {
                let next = words[j];
                if is_cap_word(next) && !next.is_empty() {
                    span.push(next);
                    j += 1;
                } else if is_connector(next) && j + 1 < words.len() && is_cap_word(words[j + 1]) {
                    span.push(next);
                    span.push(words[j + 1]);
                    j += 2;
                } else {
                    break;
                }
            }
            let at_sentence_start = i == 0;
            let keep = if at_sentence_start && span.len() == 1 {
                !sentence_start_noise(span[0]) && span[0].len() > 2
            } else {
                span.iter().any(|w| !sentence_start_noise(w))
            };
            if keep {
                let display = span.join(" ");
                let trimmed = display.trim_end_matches('.');
                if !trimmed.is_empty() {
                    out.push(Entity::new(trimmed, EntityKind::Named));
                }
            }
            i = j.max(i + 1);
        }
    }
}

impl EntityExtractor for HeuristicNer {
    fn extract(&self, text: &str) -> Vec<Entity> {
        let p = patterns();
        let text = &mask_code(text, p);
        let mut out = Vec::new();
        let mut masked = text.to_string();

        let mut take = |rx: &Regex, kind: EntityKind, masked: &mut String| {
            for m in rx.find_iter(text) {
                out.push(Entity::new(m.as_str(), kind));
            }
            *masked = rx.replace_all(masked, " ").into_owned();
        };

        take(&p.url, EntityKind::Url, &mut masked);
        take(&p.email, EntityKind::Email, &mut masked);
        take(&p.iso_date, EntityKind::Date, &mut masked);

        // Date passes run on the masked text so numbers inside dates are not re-extracted.
        for rx in [&p.month_day_year, &p.day_month_year, &p.month_year] {
            for m in rx.find_iter(&masked.clone()) {
                out.push(Entity::new(m.as_str(), EntityKind::Date));
            }
            masked = rx.replace_all(&masked, " ").into_owned();
        }
        for m in p.year.find_iter(&masked.clone()) {
            out.push(Entity::new(m.as_str(), EntityKind::Date));
        }
        masked = p.year.replace_all(&masked, " ").into_owned();

        for m in p.number.find_iter(&masked) {
            let s = m.as_str();
            if s.len() > 1 || s.parse::<u32>().map_or(false, |n| n > 1) {
                out.push(Entity::new(s, EntityKind::Number));
            }
        }
        for c in p.quote.captures_iter(text) {
            let q = &c[1];
            if q.chars().any(char::is_alphabetic) && !is_code_like(q) {
                out.push(Entity::new(q, EntityKind::Quote));
            }
        }
        for sentence in split_sentences(&masked) {
            self.capitalized_spans(sentence, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canons(text: &str) -> Vec<String> {
        HeuristicNer
            .extract(text)
            .into_iter()
            .map(|e| e.canon)
            .collect()
    }

    #[test]
    fn extracts_multiword_names() {
        let c = canons("Yesterday I met Carrie Vu at the Word Bookstore.");
        assert!(c.contains(&"carrie vu".to_string()), "{c:?}");
        assert!(c.contains(&"word bookstore".to_string()), "{c:?}");
    }

    #[test]
    fn extracts_dates_not_their_digits() {
        let ents = HeuristicNer.extract("We moved on February 14, 2022 and paid $1200.");
        let dates: Vec<_> = ents.iter().filter(|e| e.kind == EntityKind::Date).collect();
        let nums: Vec<_> = ents
            .iter()
            .filter(|e| e.kind == EntityKind::Number)
            .collect();
        assert_eq!(dates.len(), 1, "{ents:?}");
        assert_eq!(nums.len(), 1, "{ents:?}");
        assert_eq!(nums[0].canon, "$1200");
    }

    #[test]
    fn sentence_start_stopword_not_entity() {
        let c = canons("Yesterday was rough.");
        assert!(!c.contains(&"yesterday".to_string()), "{c:?}");
    }

    #[test]
    fn inline_code_is_masked() {
        let c = canons("run `curl -s -w '%{http_code}' -H Auth` and report");
        assert!(c.is_empty(), "{c:?}");
    }

    #[test]
    fn fenced_code_is_masked() {
        let c = canons("look:\n```\nSELECT * FROM Turns WHERE id = 103;\n```\nthanks");
        assert!(c.is_empty(), "{c:?}");
    }

    #[test]
    fn code_like_quotes_dropped() {
        let c = canons(r#"it failed with "&& echo $(date)" and "-d '{" again"#);
        assert!(c.is_empty(), "{c:?}");
    }

    #[test]
    fn prose_quotes_kept() {
        let ents = HeuristicNer.extract(r#"she wrote "back at the shop by noon" on the door"#);
        assert!(ents.iter().any(|e| e.kind == EntityKind::Quote), "{ents:?}");
    }

    #[test]
    fn connector_joins_span() {
        let c = canons("She works at the Museum of Modern Art now.");
        assert!(c.contains(&"museum of modern art".to_string()), "{c:?}");
    }
}
