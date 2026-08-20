//! Spool between short-lived hooks and the long-lived MCP server.
//!
//! Hooks must not open the DB: every open replays the whole index and may
//! load the ONNX embedder, and hooks run once per turn. Instead each hook
//! event lands as one JSONL file, written to a tmp name and renamed into
//! place so readers only ever see complete files. The server claims a file
//! by renaming it (atomic, so two servers cannot both ingest it), ingests,
//! and deletes. Claims orphaned by a crashed server are adopted after a
//! grace period; source-uuid dedup in the store makes re-ingesting a
//! half-processed claim harmless.

use crate::error::Result;
use crate::ZeroMem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// How old a claimed file must be before another server adopts it.
const STALE_CLAIM_SECS: u64 = 300;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpoolTurn {
    pub session_id: String,
    pub speaker: String,
    pub text: String,
    pub ts: i64,
    pub uuid: String,
}

/// ZEROMEM_HOME, or ~/.zeromem. One home means one store shared by every
/// project, which is the point: memory that crosses sessions and repos.
pub fn default_home() -> PathBuf {
    if let Ok(dir) = std::env::var("ZEROMEM_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&base).join(".zeromem")
}

pub fn spool_dir(home: &Path) -> PathBuf {
    home.join("spool")
}

/// Writes one hook event as a single spool file. The name sorts by wall
/// clock so drain replays events in the order they happened.
pub fn append_event(home: &Path, turns: &[SpoolTurn]) -> Result<PathBuf> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = spool_dir(home);
    std::fs::create_dir_all(&dir)?;
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let name = format!(
        "{millis:013}-{}-{}.jsonl",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let path = dir.join(&name);
    let tmp = dir.join(format!("{name}.tmp"));
    let mut body = String::new();
    for t in turns {
        body.push_str(&serde_json::to_string(t).expect("spool turn serializes"));
        body.push('\n');
    }
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Ingests every pending spool event, oldest first, and returns the number
/// of turns actually added (dedup may drop some). Files another live
/// server has claimed are skipped; stale claims are adopted.
pub fn drain(home: &Path, zm: &mut ZeroMem) -> Result<usize> {
    let dir = spool_dir(home);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(err.into()),
    };
    let mut pending: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".jsonl") {
            pending.push(path);
        } else if name.contains(".jsonl.claim-") && claim_is_stale(&path) {
            pending.push(path);
        }
    }
    pending.sort();

    let mut added = 0;
    for path in pending {
        let claimed = if path.extension().is_some_and(|e| e == "jsonl") {
            let target = path.with_extension(format!("jsonl.claim-{}", std::process::id()));
            match std::fs::rename(&path, &target) {
                Ok(()) => {
                    // rename keeps the spool file's mtime, so staleness would
                    // measure event age, not claim age: an old backlog file
                    // would look stale the instant it was claimed
                    touch(&target);
                    target
                }
                // another server claimed it between readdir and rename
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err.into()),
            }
        } else {
            path // adopted stale claim, already renamed once
        };
        // an adopted claim can vanish if its slow owner finishes after all
        let body = match std::fs::read_to_string(&claimed) {
            Ok(b) => b,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(t) = serde_json::from_str::<SpoolTurn>(line) else { continue };
            if t.text.trim().is_empty() {
                continue;
            }
            if zm.ingest_turn_dedup(&t.session_id, &t.speaker, &t.text, t.ts, &t.uuid)?.is_some() {
                added += 1;
            }
        }
        match std::fs::remove_file(&claimed) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(added)
}

fn touch(path: &Path) {
    if let Ok(f) = std::fs::File::options().append(true).open(path) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

fn claim_is_stale(path: &Path) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() > STALE_CLAIM_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embed::HashEmbedder;

    fn temp_home(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zeromem-spool-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn turn(uuid: &str, text: &str) -> SpoolTurn {
        SpoolTurn {
            session_id: "s1".into(),
            speaker: "user".into(),
            text: text.into(),
            ts: 1000,
            uuid: uuid.into(),
        }
    }

    #[test]
    fn drain_ingests_and_removes() {
        let home = temp_home("basic");
        let mut zm =
            ZeroMem::open_in_memory(Config::default(), Box::new(HashEmbedder::default())).unwrap();
        append_event(&home, &[turn("u1", "Carrie runs the register."), turn("u2", "Lychee naps.")])
            .unwrap();
        append_event(&home, &[turn("u3", "Slowdive played the Fillmore.")]).unwrap();

        assert_eq!(drain(&home, &mut zm).unwrap(), 3);
        assert_eq!(zm.stats().turns, 3);
        assert!(spool_dir(&home).read_dir().unwrap().next().is_none());
        assert_eq!(drain(&home, &mut zm).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn replayed_event_dedups() {
        let home = temp_home("dedup");
        let mut zm =
            ZeroMem::open_in_memory(Config::default(), Box::new(HashEmbedder::default())).unwrap();
        append_event(&home, &[turn("u1", "Carrie runs the register.")]).unwrap();
        drain(&home, &mut zm).unwrap();
        // same uuid spooled again, e.g. an adopted claim already ingested
        append_event(&home, &[turn("u1", "Carrie runs the register.")]).unwrap();
        assert_eq!(drain(&home, &mut zm).unwrap(), 0);
        assert_eq!(zm.stats().turns, 1);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn fresh_claim_left_alone_stale_claim_adopted() {
        let home = temp_home("claims");
        let mut zm =
            ZeroMem::open_in_memory(Config::default(), Box::new(HashEmbedder::default())).unwrap();
        let path = append_event(&home, &[turn("u1", "Carrie runs the register.")]).unwrap();
        let claim = path.with_extension("jsonl.claim-99999");
        std::fs::rename(&path, &claim).unwrap();

        // fresh claim: some other server is on it
        assert_eq!(drain(&home, &mut zm).unwrap(), 0);
        assert!(claim.exists());

        // stale claim: that server is gone, adopt
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(STALE_CLAIM_SECS * 2);
        let f = std::fs::File::options().append(true).open(&claim).unwrap();
        f.set_modified(old).unwrap();
        drop(f);
        assert_eq!(drain(&home, &mut zm).unwrap(), 1);
        assert!(!claim.exists());
        let _ = std::fs::remove_dir_all(&home);
    }
}
