//! SQLite: turns + cached embeddings. Everything else is derived from turns
//! and rebuilt in memory on open.

use crate::error::Result;
use crate::trace::Turn;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS turns (
                 id INTEGER PRIMARY KEY,
                 session_id TEXT NOT NULL,
                 session_turn INTEGER NOT NULL,
                 speaker TEXT NOT NULL,
                 text TEXT NOT NULL,
                 ts INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, session_turn);
             CREATE TABLE IF NOT EXISTS embeddings (
                 kind TEXT NOT NULL,
                 key TEXT NOT NULL,
                 vec BLOB NOT NULL,
                 PRIMARY KEY (kind, key)
             );
             CREATE TABLE IF NOT EXISTS meta (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );",
        )?;
        Ok(Self { conn })
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }

    pub fn load_turns(&self) -> Result<Vec<Turn>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, session_id, session_turn, speaker, text, ts FROM turns ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(Turn {
                id: r.get(0)?,
                session_id: r.get(1)?,
                session_turn: r.get(2)?,
                speaker: r.get(3)?,
                text: r.get(4)?,
                ts: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    pub fn next_session_turn(&self, session_id: &str) -> Result<i64> {
        let max: Option<i64> = self.conn.query_row(
            "SELECT MAX(session_turn) FROM turns WHERE session_id = ?1",
            [session_id],
            |r| r.get(0),
        )?;
        Ok(max.map_or(0, |m| m + 1))
    }

    pub fn insert_turn(&self, t: &Turn) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO turns (session_id, session_turn, speaker, text, ts) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![t.session_id, t.session_turn, t.speaker, t.text, t.ts],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn embedding(&self, kind: &str, key: &str) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT vec FROM embeddings WHERE kind = ?1 AND key = ?2",
                [kind, key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(blob.map(|b| bytes_to_f32(&b)))
    }

    pub fn put_embedding(&self, kind: &str, key: &str, vec: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO embeddings (kind, key, vec) VALUES (?1, ?2, ?3)
             ON CONFLICT(kind, key) DO UPDATE SET vec = excluded.vec",
            params![kind, key, f32_to_bytes(vec)],
        )?;
        Ok(())
    }

    pub fn clear_embeddings(&self) -> Result<()> {
        self.conn.execute("DELETE FROM embeddings", [])?;
        Ok(())
    }

    pub fn turn_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM turns", [], |r| r.get(0))?)
    }
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_turn_and_embedding() {
        let s = Store::open_in_memory().unwrap();
        let t = Turn {
            id: 0,
            session_id: "s1".into(),
            session_turn: 0,
            speaker: "user".into(),
            text: "hello".into(),
            ts: 1000,
        };
        let id = s.insert_turn(&t).unwrap();
        assert_eq!(id, 1);
        assert_eq!(s.next_session_turn("s1").unwrap(), 1);
        assert_eq!(s.next_session_turn("s2").unwrap(), 0);

        s.put_embedding("turn", "1", &[0.5, -1.0]).unwrap();
        assert_eq!(s.embedding("turn", "1").unwrap().unwrap(), vec![0.5, -1.0]);
        assert!(s.embedding("turn", "2").unwrap().is_none());
    }
}
