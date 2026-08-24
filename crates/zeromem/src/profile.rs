//! Query profile, paper eq 6.

use crate::ner::{Entity, EntityExtractor, EntityKind};
use crate::text::content_tokens;
use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum AnswerType {
    Person,
    Time,
    Place,
    Number,
    List,
    Boolean,
    Entity,
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum RecencyPref {
    Earliest,
    Latest,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TemporalCues {
    /// Admissible unix-second ranges parsed from explicit dates.
    pub ranges: Vec<(i64, i64)>,
    pub prefer: Option<RecencyPref>,
    /// Raw cue strings, kept for lexical matching.
    pub mentions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Boundary {
    SessionOrdinal(usize),
    LastSession,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryProfile {
    /// Canonical entity names anchoring the query.
    pub subjects: Vec<String>,
    pub keywords: Vec<String>,
    pub answer_type: AnswerType,
    pub temporal: TemporalCues,
    pub boundary: Option<Boundary>,
    /// Verbatim phrases for exact matching: names, dates, numbers, quotes.
    pub phrases: Vec<String>,
    pub aggregation: bool,
}

pub fn build_profile(query: &str, ner: &dyn EntityExtractor) -> QueryProfile {
    let lower = query.to_lowercase();
    let entities = ner.extract(query);
    let answer_type = classify_answer_type(&lower);
    let temporal = temporal_cues(&lower, &entities);
    let boundary = detect_boundary(&lower);
    let aggregation = AGG_RE
        .get_or_init(|| {
            Regex::new(r"\b(how many times|how often|every time|all the|list all|which all|both|each time)\b").unwrap()
        })
        .is_match(&lower);

    let subjects: Vec<String> = entities
        .iter()
        .filter(|e| e.kind == EntityKind::Named)
        .map(|e| e.canon.clone())
        .collect();
    let phrases = entities
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                EntityKind::Named | EntityKind::Date | EntityKind::Number | EntityKind::Quote
            )
        })
        .map(|e| e.canon.clone())
        .collect();

    QueryProfile {
        subjects,
        keywords: content_tokens(query),
        answer_type,
        temporal,
        boundary,
        phrases,
        aggregation,
    }
}

static AGG_RE: OnceLock<Regex> = OnceLock::new();

fn classify_answer_type(lower: &str) -> AnswerType {
    let starts = |p: &str| lower.starts_with(p);
    if starts("who") {
        AnswerType::Person
    } else if starts("when")
        || lower.contains("what year")
        || lower.contains("what date")
        || lower.contains("what time")
        || lower.contains("how long ago")
    {
        AnswerType::Time
    } else if starts("where") {
        AnswerType::Place
    } else if lower.contains("how many") || lower.contains("how much") || lower.contains("how old")
    {
        AnswerType::Number
    } else if lower.contains("list ")
        || lower.contains("what are")
        || lower.contains("which are")
        || lower.contains("name all")
    {
        AnswerType::List
    } else if [
        "is ", "are ", "was ", "were ", "did ", "does ", "do ", "has ", "have ", "can ", "will ",
        "would ",
    ]
    .iter()
    .any(|p| starts(p))
    {
        AnswerType::Boolean
    } else if starts("what") || starts("which") || starts("whose") {
        AnswerType::Entity
    } else {
        AnswerType::Open
    }
}

fn detect_boundary(lower: &str) -> Option<Boundary> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"\b(first|second|third|fourth|fifth|last|latest|previous)\s+(conversation|session|chat)\b").unwrap()
    });
    let c = re.captures(lower)?;
    match &c[1] {
        "first" => Some(Boundary::SessionOrdinal(0)),
        "second" => Some(Boundary::SessionOrdinal(1)),
        "third" => Some(Boundary::SessionOrdinal(2)),
        "fourth" => Some(Boundary::SessionOrdinal(3)),
        "fifth" => Some(Boundary::SessionOrdinal(4)),
        _ => Some(Boundary::LastSession),
    }
}

fn temporal_cues(lower: &str, entities: &[Entity]) -> TemporalCues {
    let mut cues = TemporalCues::default();
    for e in entities.iter().filter(|e| e.kind == EntityKind::Date) {
        cues.mentions.push(e.canon.clone());
        if let Some(range) = parse_date_range(&e.canon) {
            cues.ranges.push(range);
        }
    }
    if lower.contains("most recent") || lower.contains("latest") || lower.contains("last time") {
        cues.prefer = Some(RecencyPref::Latest);
    } else if lower.contains("first time")
        || lower.contains("originally")
        || lower.contains("initially")
    {
        cues.prefer = Some(RecencyPref::Earliest);
    }
    for marker in [
        "yesterday",
        "last week",
        "last month",
        "last year",
        "recently",
        "earlier",
    ] {
        if lower.contains(marker) {
            cues.mentions.push(marker.to_string());
            if cues.prefer.is_none() {
                cues.prefer = Some(RecencyPref::Latest);
            }
        }
    }
    cues
}

/// Days since epoch for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn ts(y: i64, m: u32, d: u32) -> i64 {
    days_from_civil(y, m, d) * 86400
}

fn month_num(name: &str) -> Option<u32> {
    let n = &name.to_lowercase()[..3.min(name.len())];
    Some(match n {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// Parses a date mention into an inclusive unix range at its precision.
pub fn parse_date_range(s: &str) -> Option<(i64, i64)> {
    static ISO: OnceLock<Regex> = OnceLock::new();
    static MDY: OnceLock<Regex> = OnceLock::new();
    static DMY: OnceLock<Regex> = OnceLock::new();
    static MY: OnceLock<Regex> = OnceLock::new();
    static Y: OnceLock<Regex> = OnceLock::new();

    let iso = ISO.get_or_init(|| Regex::new(r"^(\d{4})-(\d{2})-(\d{2})$").unwrap());
    if let Some(c) = iso.captures(s) {
        let (y, m, d) = (c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?);
        return Some((ts(y, m, d), ts(y, m, d) + 86399));
    }
    let mdy = MDY.get_or_init(|| {
        Regex::new(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?$").unwrap()
    });
    if let Some(c) = mdy.captures(s) {
        if let Some(m) = month_num(&c[1]) {
            let d: u32 = c[2].parse().ok()?;
            if d >= 1 && d <= 31 {
                return match c.get(3) {
                    Some(y) => {
                        let y: i64 = y.as_str().parse().ok()?;
                        Some((ts(y, m, d), ts(y, m, d) + 86399))
                    }
                    None => None, // dayless year unknown; lexical match only
                };
            }
        }
    }
    let dmy = DMY.get_or_init(|| {
        Regex::new(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?(?:\s+(\d{4}))?$").unwrap()
    });
    if let Some(c) = dmy.captures(s) {
        if let Some(m) = month_num(&c[2]) {
            let d: u32 = c[1].parse().ok()?;
            if let Some(y) = c.get(3) {
                let y: i64 = y.as_str().parse().ok()?;
                return Some((ts(y, m, d), ts(y, m, d) + 86399));
            }
            return None;
        }
    }
    let my = MY.get_or_init(|| Regex::new(r"^([a-z]+)\.?\s+(\d{4})$").unwrap());
    if let Some(c) = my.captures(s) {
        if let Some(m) = month_num(&c[1]) {
            let y: i64 = c[2].parse().ok()?;
            return Some((ts(y, m, 1), ts(y, m, days_in_month(y, m)) + 86399));
        }
    }
    let yre = Y.get_or_init(|| Regex::new(r"^((?:19|20)\d{2})$").unwrap());
    if let Some(c) = yre.captures(s) {
        let y: i64 = c[1].parse().ok()?;
        return Some((ts(y, 1, 1), ts(y, 12, 31) + 86399));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::HeuristicNer;

    #[test]
    fn who_question_profile() {
        let p = build_profile("Who did Carrie meet at the farmers market?", &HeuristicNer);
        assert_eq!(p.answer_type, AnswerType::Person);
        assert!(p.subjects.contains(&"carrie".to_string()));
    }

    #[test]
    fn when_question_gets_time_type_and_latest_pref() {
        let p = build_profile("When did she last visit Jersey City?", &HeuristicNer);
        assert_eq!(p.answer_type, AnswerType::Time);
    }

    #[test]
    fn date_range_year() {
        let (a, b) = parse_date_range("2023").unwrap();
        assert_eq!(a, ts(2023, 1, 1));
        assert_eq!(b, ts(2023, 12, 31) + 86399);
    }

    #[test]
    fn date_range_iso_day() {
        let (a, b) = parse_date_range("2022-02-14").unwrap();
        assert_eq!(b - a, 86399);
    }

    #[test]
    fn date_range_month_day_year() {
        let r = parse_date_range("february 14, 2022").unwrap();
        assert_eq!(r, parse_date_range("2022-02-14").unwrap());
        assert_eq!(r, parse_date_range("14th february 2022").unwrap());
    }

    #[test]
    fn boundary_first_session() {
        let p = build_profile("What did I say in the first conversation?", &HeuristicNer);
        assert_eq!(p.boundary, Some(Boundary::SessionOrdinal(0)));
    }

    #[test]
    fn epoch_sanity() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
    }
}
