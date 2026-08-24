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
             PRAGMA busy_timeout = 5000;
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
        let has_source_uuid = conn
            .prepare("SELECT 1 FROM pragma_table_info('turns') WHERE name = 'source_uuid'")?
            .exists([])?;
        if !has_source_uuid {
            // two processes can race this migration on first open of an old
            // DB; losing the race is fine as long as the column exists
            if let Err(err) = conn.execute("ALTER TABLE turns ADD COLUMN source_uuid TEXT", []) {
                let present = conn
                    .prepare("SELECT 1 FROM pragma_table_info('turns') WHERE name = 'source_uuid'")?
                    .exists([])?;
                if !present {
                    return Err(err.into());
                }
            }
        }
        conn.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS turns_source_uuid
             ON turns(source_uuid) WHERE source_uuid IS NOT NULL;",
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
        self.load_turns_after(0)
    }

    /// Turns with id greater than `id`, in id order. Ids start at 1, so 0
    /// loads everything.
    pub fn load_turns_after(&self, id: i64) -> Result<Vec<Turn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, session_turn, speaker, text, ts FROM turns
             WHERE id > ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map([id], |r| {
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

    /// Inserts a turn, assigning session_turn inside the statement so
    /// concurrent writers cannot race to the same position. A duplicate
    /// source_uuid is ignored and returns None.
    pub fn insert_turn(
        &self,
        session_id: &str,
        speaker: &str,
        text: &str,
        ts: i64,
        source_uuid: Option<&str>,
    ) -> Result<Option<Turn>> {
        let inserted = self.conn.execute(
            "INSERT OR IGNORE INTO turns (session_id, session_turn, speaker, text, ts, source_uuid)
             SELECT ?1,
                    COALESCE((SELECT MAX(session_turn) + 1 FROM turns WHERE session_id = ?1), 0),
                    ?2, ?3, ?4, ?5",
            params![session_id, speaker, text, ts, source_uuid],
        )?;
        if inserted == 0 {
            return Ok(None);
        }
        let id = self.conn.last_insert_rowid();
        let session_turn =
            self.conn
                .query_row("SELECT session_turn FROM turns WHERE id = ?1", [id], |r| r.get(0))?;
        Ok(Some(Turn {
            id,
            session_id: session_id.to_string(),
            session_turn,
            speaker: speaker.to_string(),
            text: text.to_string(),
            ts,
        }))
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

    pub fn delete_session_turns(&self, session_id: &str) -> Result<usize> {
        Ok(self.conn.execute("DELETE FROM turns WHERE session_id = ?1", [session_id])?)
    }

    pub fn embedding_keys(&self, kind: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT key FROM embeddings WHERE kind = ?1")?;
        let rows = stmt.query_map([kind], |r| r.get(0))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Deletes cache rows for turns no longer in the turns table and for
    /// entities absent from `live_entities`, in one transaction. The turn
    /// side keys off an indexed integer column rather than a text CAST.
    pub fn sweep_orphan_embeddings(&mut self, live_entities: &[&str]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "ALTER TABLE embeddings ADD COLUMN turn_id INTEGER;
             CREATE INDEX IF NOT EXISTS embeddings_turn_id ON embeddings(turn_id);
             UPDATE embeddings SET turn_id = CAST(key AS INTEGER) WHERE kind = 'turn' AND turn_id IS NULL;",
        )?;
        tx.execute(
            "DELETE FROM embeddings WHERE kind = 'turn'
             AND turn_id IS NOT NULL
             AND turn_id NOT IN (SELECT id FROM turns)",
            [],
        )?;
        tx.execute("DELETE FROM embeddings WHERE kind = 'turn' AND turn_id IS NULL", [])?;
        tx.execute("CREATE TEMP TABLE live_entity (key TEXT PRIMARY KEY)", [])?;
        {
            let mut ins = tx.prepare("INSERT OR IGNORE INTO live_entity (key) VALUES (?1)")?;
            for key in live_entities {
                ins.execute([key])?;
            }
        }
        tx.execute(
            "DELETE FROM embeddings WHERE kind = 'entity'
             AND key NOT IN (SELECT key FROM live_entity)",
            [],
        )?;
        tx.execute("DROP TABLE live_entity", [])?;
        tx.commit()?;
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
        let t = s.insert_turn("s1", "user", "hello", 1000, None).unwrap().unwrap();
        assert_eq!(t.id, 1);
        assert_eq!(t.session_turn, 0);
        let t2 = s.insert_turn("s1", "user", "again", 1001, None).unwrap().unwrap();
        assert_eq!(t2.session_turn, 1);
        assert_eq!(s.insert_turn("s2", "user", "other", 1002, None).unwrap().unwrap().session_turn, 0);

        s.put_embedding("turn", "1", &[0.5, -1.0]).unwrap();
        assert_eq!(s.embedding("turn", "1").unwrap().unwrap(), vec![0.5, -1.0]);
        assert!(s.embedding("turn", "9").unwrap().is_none());
    }

    #[test]
    fn source_uuid_dedups() {
        let s = Store::open_in_memory().unwrap();
        assert!(s.insert_turn("s1", "user", "hello", 1000, Some("u-1")).unwrap().is_some());
        assert!(s.insert_turn("s1", "user", "hello", 1000, Some("u-1")).unwrap().is_none());
        // untagged inserts never collide
        assert!(s.insert_turn("s1", "user", "hello", 1000, None).unwrap().is_some());
        assert!(s.insert_turn("s1", "user", "hello", 1000, None).unwrap().is_some());
        assert_eq!(s.turn_count().unwrap(), 3);
    }

    #[test]
    fn load_turns_after_skips_indexed() {
        let s = Store::open_in_memory().unwrap();
        s.insert_turn("s1", "user", "one", 1, None).unwrap();
        s.insert_turn("s1", "user", "two", 2, None).unwrap();
        assert_eq!(s.load_turns_after(0).unwrap().len(), 2);
        assert_eq!(s.load_turns_after(1).unwrap().len(), 1);
        assert_eq!(s.load_turns_after(2).unwrap().len(), 0);
    }

    #[test]
    fn migrates_pre_uuid_schema() {
        let dir = std::env::temp_dir().join(format!("zeromem-store-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE turns (
                     id INTEGER PRIMARY KEY,
                     session_id TEXT NOT NULL,
                     session_turn INTEGER NOT NULL,
                     speaker TEXT NOT NULL,
                     text TEXT NOT NULL,
                     ts INTEGER NOT NULL
                 );
                 INSERT INTO turns (session_id, session_turn, speaker, text, ts)
                 VALUES ('s1', 0, 'user', 'legacy', 1000);",
            )
            .unwrap();
        }
        let s = Store::open(&path).unwrap();
        assert_eq!(s.load_turns().unwrap().len(), 1);
        assert!(s.insert_turn("s1", "user", "new", 2000, Some("u-1")).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
