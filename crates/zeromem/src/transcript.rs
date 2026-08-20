//! Claude Code transcript parsing: session JSONL lines to clean turns.
//!
//! A transcript line is a JSON object with `type` ("user", "assistant",
//! "system", "summary"), a `message` whose content is a string or an array
//! of typed blocks, a `uuid`, and an ISO 8601 `timestamp`. Only human and
//! assistant prose belongs in memory: tool traffic, thinking, meta lines,
//! sidechains (subagent transcripts), and slash-command noise would pollute
//! the entity graph and are dropped here.

/// Longest turn text kept, in characters. Giant pastes drown BM25 and the
/// window centroids without adding recallable facts.
const MAX_TURN_CHARS: usize = 8000;

/// Block or message prefixes that are harness plumbing, not conversation.
const NOISE_PREFIXES: &[&str] = &[
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<system-reminder>",
    "Caveat:",
];

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptTurn {
    pub speaker: String,
    pub text: String,
    /// Unix seconds; 0 when the line carries no parseable timestamp.
    pub ts: i64,
    /// The transcript line's uuid, the dedup key across re-parses.
    pub uuid: String,
}

/// Parses complete transcript lines. Malformed lines are skipped, never
/// fatal: a hook must not fail the session over one odd line.
pub fn parse(input: &str) -> Vec<TranscriptTurn> {
    input.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<TranscriptTurn> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v["isSidechain"].as_bool() == Some(true) || v["isMeta"].as_bool() == Some(true) {
        return None;
    }
    let speaker = match v["type"].as_str()? {
        t @ ("user" | "assistant") => t,
        _ => return None,
    };
    let uuid = v["uuid"].as_str()?.to_string();
    let text = message_text(&v["message"])?;
    Some(TranscriptTurn {
        speaker: speaker.to_string(),
        text,
        ts: v["timestamp"].as_str().map_or(0, |s| parse_iso8601(s).unwrap_or(0)),
        uuid,
    })
}

fn message_text(message: &serde_json::Value) -> Option<String> {
    let content = &message["content"];
    let mut parts: Vec<&str> = Vec::new();
    match content {
        serde_json::Value::String(s) => parts.push(s),
        serde_json::Value::Array(blocks) => {
            for b in blocks {
                if b["type"].as_str() == Some("text") {
                    if let Some(t) = b["text"].as_str() {
                        parts.push(t);
                    }
                }
            }
        }
        _ => return None,
    }
    let kept: Vec<&str> = parts
        .into_iter()
        .map(str::trim)
        .filter(|t| !t.is_empty() && !NOISE_PREFIXES.iter().any(|p| t.starts_with(p)))
        .collect();
    if kept.is_empty() {
        return None;
    }
    Some(truncate(kept.join("\n\n")))
}

fn truncate(text: String) -> String {
    if text.chars().count() <= MAX_TURN_CHARS {
        return text;
    }
    let cut: String = text.chars().take(MAX_TURN_CHARS).collect();
    format!("{cut}\n[truncated]")
}

/// "YYYY-MM-DDTHH:MM:SS(.frac)?Z" to unix seconds. Claude Code stamps in
/// UTC; a non-Z suffix is rejected rather than misread.
pub fn parse_iso8601(s: &str) -> Option<i64> {
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut d = date.split('-');
    let (y, m, day) = (d.next()?.parse().ok()?, d.next()?.parse().ok()?, d.next()?.parse().ok()?);
    if d.next().is_some() || !(1..=12).contains(&m) || !(1..=days_in_month(y, m)).contains(&day) {
        return None;
    }
    let time = time.split_once('.').map_or(time, |(t, _)| t);
    let mut t = time.split(':');
    let (h, min, sec): (i64, i64, i64) =
        (t.next()?.parse().ok()?, t.next()?.parse().ok()?, t.next().unwrap_or("0").parse().ok()?);
    if t.next().is_some() || h > 23 || min > 59 || sec > 59 {
        return None;
    }
    Some(days_from_civil(y, m, day) * 86400 + h * 3600 + min * 60 + sec)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        4 | 6 | 9 | 11 => 30,
        2 => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 31,
    }
}

/// Howard Hinnant's civil-days algorithm; days since 1970-01-01.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(t: &str, uuid: &str, content: serde_json::Value) -> String {
        serde_json::json!({
            "type": t, "uuid": uuid, "timestamp": "2026-08-20T10:00:00.000Z",
            "message": {"role": t, "content": content}
        })
        .to_string()
    }

    #[test]
    fn keeps_user_and_assistant_prose() {
        let input = [
            line("user", "u1", serde_json::json!("I moved to Jersey City.")),
            line(
                "assistant",
                "a1",
                serde_json::json!([
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "Noted."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
                ]),
            ),
        ]
        .join("\n");
        let turns = parse(&input);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].text, "I moved to Jersey City.");
        assert_eq!(turns[1].text, "Noted.");
        assert_eq!(turns[1].speaker, "assistant");
        assert_eq!(turns[0].ts, 1_787_220_000);
    }

    #[test]
    fn drops_noise() {
        let sidechain = format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": "s1", "isSidechain": true,
                "message": {"content": "subagent prompt"}
            })
        );
        let input = [
            sidechain.trim().to_string(),
            line("user", "u1", serde_json::json!([{"type": "tool_result", "content": "445 files"}])),
            line("user", "u2", serde_json::json!("<command-name>/model</command-name>")),
            line("user", "u3", serde_json::json!("<system-reminder>recall</system-reminder>")),
            line("user", "u4", serde_json::json!("Caveat: local command noise")),
            line("system", "s2", serde_json::json!("hook ran")),
            serde_json::json!({"type": "summary", "summary": "compacted"}).to_string(),
            "not json at all".to_string(),
            line("user", "u5", serde_json::json!("real question")),
        ]
        .join("\n");
        let turns = parse(&input);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].uuid, "u5");
    }

    #[test]
    fn meta_lines_dropped() {
        let meta = serde_json::json!({
            "type": "user", "uuid": "m1", "isMeta": true,
            "message": {"content": "injected context"}
        })
        .to_string();
        assert!(parse(&meta).is_empty());
    }

    #[test]
    fn long_text_truncated() {
        let big = "x".repeat(MAX_TURN_CHARS + 50);
        let turns = parse(&line("user", "u1", serde_json::json!(big)));
        assert!(turns[0].text.ends_with("[truncated]"));
        assert!(turns[0].text.chars().count() < MAX_TURN_CHARS + 20);
    }

    #[test]
    fn iso8601_parses_and_rejects() {
        assert_eq!(parse_iso8601("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_iso8601("2022-02-14T00:00:00.500Z"), Some(1_644_796_800));
        assert_eq!(parse_iso8601("2022-02-14T00:00:00+02:00"), None);
        assert_eq!(parse_iso8601("garbage"), None);
        assert_eq!(parse_iso8601("2022-02-31T00:00:00Z"), None);
        assert_eq!(parse_iso8601("2024-02-29T00:00:00Z"), Some(1_709_164_800));
        assert_eq!(parse_iso8601("2100-02-29T00:00:00Z"), None);
        assert_eq!(parse_iso8601("2016-12-31T23:59:60Z"), None);
    }
}
