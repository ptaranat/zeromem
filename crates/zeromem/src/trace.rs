//! Trace units are the source of record. Text is never rewritten or summarized;
//! every derived structure points back here (provenance preservation).

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub id: i64,
    pub session_id: String,
    /// Position within the session.
    pub session_turn: i64,
    pub speaker: String,
    pub text: String,
    /// Unix seconds.
    pub ts: i64,
}

impl Turn {
    pub fn provenance(&self) -> String {
        format!("[{} #{} {} @{}]", self.session_id, self.session_turn, self.speaker, self.ts)
    }
}
