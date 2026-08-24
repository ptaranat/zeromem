//! `zm hook`: the Claude Code Stop/SessionEnd hook body.
//!
//! Reads the hook event JSON from stdin, parses transcript lines added
//! since the stored byte offset, and spools the clean turns. Never opens
//! the DB (see spool.rs for why). Errors should be logged, not surfaced:
//! a memory hiccup must not block the user's session.

use crate::error::{Error, Result};
use crate::spool::{self, SpoolTurn};
use crate::transcript;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Handles one hook event; returns the number of turns spooled.
pub fn run(home: &Path, input: &str) -> Result<usize> {
    let v: serde_json::Value =
        serde_json::from_str(input).map_err(|e| Error::Invalid(format!("hook input: {e}")))?;
    let session_id = v["session_id"]
        .as_str()
        .ok_or_else(|| Error::Invalid("hook input: missing session_id".into()))?;
    let transcript_path = v["transcript_path"]
        .as_str()
        .ok_or_else(|| Error::Invalid("hook input: missing transcript_path".into()))?;

    let offset_path = offset_path(home, transcript_path);
    let mut offset = std::fs::read_to_string(&offset_path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);

    let mut file = match std::fs::File::open(transcript_path) {
        Ok(f) => f,
        // transcript gone (cleanup, retention): nothing to do
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(err.into()),
    };
    let len = file.metadata()?.len();
    if offset > len {
        // replaced or truncated; re-read from the top, uuid dedup absorbs it
        offset = 0;
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut rest = String::new();
    file.read_to_string(&mut rest)?;

    // consume only complete lines; a line mid-write stays for next time
    let consumed = rest.rfind('\n').map_or(0, |i| i + 1);
    let turns: Vec<SpoolTurn> = transcript::parse(&rest[..consumed])
        .into_iter()
        .map(|t| SpoolTurn {
            session_id: session_id.to_string(),
            speaker: t.speaker,
            text: t.text,
            ts: t.ts,
            uuid: t.uuid,
        })
        .collect();
    if !turns.is_empty() {
        spool::append_event(home, &turns)?;
    }

    if let Some(dir) = offset_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // pid-suffixed tmp so two hook events racing on the same transcript
    // cannot clobber each other's rename
    let tmp = offset_path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, (offset + consumed as u64).to_string())?;
    std::fs::rename(&tmp, &offset_path)?;
    Ok(turns.len())
}

fn offset_path(home: &Path, transcript_path: &str) -> PathBuf {
    home.join("offsets").join(format!("{:016x}", fnv1a(transcript_path)))
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zeromem-hook-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn transcript_line(uuid: &str, text: &str) -> String {
        serde_json::json!({
            "type": "user", "uuid": uuid, "timestamp": "2026-08-20T10:00:00Z",
            "message": {"role": "user", "content": text}
        })
        .to_string()
    }

    fn hook_input(transcript: &Path) -> String {
        serde_json::json!({
            "session_id": "cc-1",
            "transcript_path": transcript.to_str().unwrap(),
            "hook_event_name": "Stop"
        })
        .to_string()
    }

    #[test]
    fn spools_new_lines_and_advances_offset() {
        let home = temp_home("offset");
        let transcript = home.join("transcript.jsonl");
        std::fs::write(&transcript, format!("{}\n", transcript_line("u1", "first"))).unwrap();

        assert_eq!(run(&home, &hook_input(&transcript)).unwrap(), 1);
        // second run, nothing new
        assert_eq!(run(&home, &hook_input(&transcript)).unwrap(), 0);

        // append one line plus one incomplete line
        let mut body = std::fs::read_to_string(&transcript).unwrap();
        body.push_str(&format!("{}\n", transcript_line("u2", "second")));
        body.push_str("{\"type\":\"user\",\"uuid\":\"u3\"");
        std::fs::write(&transcript, &body).unwrap();
        assert_eq!(run(&home, &hook_input(&transcript)).unwrap(), 1);

        let spooled = spool::spool_dir(&home).read_dir().unwrap().count();
        assert_eq!(spooled, 2);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn truncated_transcript_resets_offset() {
        let home = temp_home("reset");
        let transcript = home.join("transcript.jsonl");
        let long = format!("{}\n{}\n", transcript_line("u1", "one"), transcript_line("u2", "two"));
        std::fs::write(&transcript, &long).unwrap();
        assert_eq!(run(&home, &hook_input(&transcript)).unwrap(), 2);

        std::fs::write(&transcript, format!("{}\n", transcript_line("u9", "fresh file"))).unwrap();
        assert_eq!(run(&home, &hook_input(&transcript)).unwrap(), 1);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_transcript_is_quiet() {
        let home = temp_home("missing");
        let input = hook_input(&home.join("nope.jsonl"));
        assert_eq!(run(&home, &input).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&home);
    }
}
