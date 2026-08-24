//! `zm mcp`: stdio MCP server for Claude Code.
//!
//! Newline-delimited JSON-RPC 2.0, the subset Claude Code speaks. One
//! server process lives per Claude Code session and is the long-lived home
//! for the index: the store opens lazily on the first tool call (so
//! initialize answers instantly, before any ONNX load or index replay),
//! then every tool call drains the hook spool and refreshes against writes
//! from other sessions.

use crate::config::Config;
use crate::error::Result;
use crate::spool;
use crate::ZeroMem;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;

pub struct Server {
    home: PathBuf,
    use_model: bool,
    zm: Option<ZeroMem>,
}

impl Server {
    pub fn new(home: PathBuf, use_model: bool) -> Self {
        Self { home, use_model, zm: None }
    }

    pub fn serve(&mut self) -> std::io::Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout().lock();
        for line in stdin.lock().lines() {
            if let Some(response) = self.handle_line(&line?) {
                writeln!(stdout, "{response}")?;
                stdout.flush()?;
            }
        }
        Ok(())
    }

    /// One request line in, one response line out; None for notifications.
    /// Unparseable input is dropped rather than answered with the spec's
    /// id:null parse error: Claude Code never sends malformed lines, and
    /// nothing on the far end of the pipe correlates a reply without an id.
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let Ok(req) = serde_json::from_str::<Value>(line) else { return None };
        let id = req.get("id")?.clone();
        let result = match req["method"].as_str().unwrap_or_default() {
            "initialize" => Ok(json!({
                "protocolVersion": req["params"]["protocolVersion"].as_str().unwrap_or("2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "zeromem", "version": env!("CARGO_PKG_VERSION")},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_schemas()})),
            "tools/call" => return Some(self.tool_call(&id, &req["params"]).to_string()),
            other => Err(format!("method not found: {other}")),
        };
        let response = match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(msg) => json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32601, "message": msg},
            }),
        };
        Some(response.to_string())
    }

    fn tool_call(&mut self, id: &Value, params: &Value) -> Value {
        let name = params["name"].as_str().unwrap_or_default().to_string();
        let args = params["arguments"].clone();
        let outcome = self.dispatch(&name, &args);
        let (text, is_error) = match outcome {
            Ok(v) => (serde_json::to_string_pretty(&v).unwrap_or_default(), false),
            Err(e) => (e.to_string(), true),
        };
        json!({
            "jsonrpc": "2.0", "id": id,
            "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
        })
    }

    fn dispatch(&mut self, name: &str, args: &Value) -> Result<Value> {
        let home = self.home.clone();
        let zm = self.open()?;
        spool::drain(&home, zm)?;
        zm.refresh()?;
        match name {
            "zeromem_recall" => {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Invalid("query is required".into()))?;
                let top_k = args["top_k"].as_u64().map(|k| k as usize);
                let exclude = args["exclude_session"].as_str();
                if let Some(sid) = exclude {
                    zm.exclude_session(sid);
                }
                let result = zm.query(query, top_k)?;
                Ok(serde_json::to_value(&result).expect("query result serializes"))            }
            "zeromem_stats" => Ok(serde_json::to_value(zm.stats()).expect("stats serialize")),
            "zeromem_forget_session" => {
                let sid = args["session_id"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Invalid("session_id is required".into()))?;
                let removed = zm.delete_session(sid)?;
                Ok(json!({"session_id": sid, "deleted_turns": removed}))
            }
            other => Err(crate::error::Error::Invalid(format!("unknown tool {other}"))),
        }
    }

    fn open(&mut self) -> Result<&mut ZeroMem> {
        if self.zm.is_none() {
            let db = self.home.join("zeromem.db");
            let embedder = if self.use_model {
                crate::default_embedder(Some(&self.home.join("models")))
            } else {
                Box::new(crate::embed::HashEmbedder::default())
                    as Box<dyn crate::embed::Embedder>
            };
            self.zm = Some(ZeroMem::open(&db, Config::default(), embedder)?);
        }
        Ok(self.zm.as_mut().expect("just opened"))
    }
}

fn tool_schemas() -> Value {
    json!([
        {
            "name": "zeromem_recall",
            "description": "Recall evidence from past conversations across all projects. \
                Returns verbatim turns with provenance (session, turn, speaker, time), \
                never summaries. Results may include turns from the current session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "top_k": {"type": "integer", "description": "Max main evidence items (default 5)"},
                    "exclude_session": {"type": "string", "description": "Session id to drop from results"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "zeromem_stats",
            "description": "Memory store counters: turns, sessions, entities, windows, episodes.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "zeromem_forget_session",
            "description": "Permanently delete one past session from memory: its turns, \
                derived graph state, and cached embeddings. Irreversible; only on explicit \
                user request. A session still open elsewhere will partially return as it \
                keeps talking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to delete"}
                },
                "required": ["session_id"]
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spool::SpoolTurn;

    fn temp_home(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zeromem-mcp-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call(server: &mut Server, id: u64, method: &str, params: Value) -> Value {
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        serde_json::from_str(&server.handle_line(&req.to_string()).unwrap()).unwrap()
    }

    #[test]
    fn handshake_and_tool_listing() {
        let home = temp_home("handshake");
        let mut s = Server::new(home.clone(), false);
        let init = call(&mut s, 1, "initialize", json!({"protocolVersion": "2025-06-18"}));
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["serverInfo"]["name"], "zeromem");

        // notification: no response
        assert!(s.handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());

        let tools = call(&mut s, 2, "tools/list", json!({}));
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["zeromem_recall", "zeromem_stats", "zeromem_forget_session"]);

        let bad = call(&mut s, 3, "resources/list", json!({}));
        assert_eq!(bad["error"]["code"], -32601);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn recall_drains_spool_first() {
        let home = temp_home("recall");
        spool::append_event(
            &home,
            &[SpoolTurn {
                session_id: "cc-1".into(),
                speaker: "user".into(),
                text: "Carrie is handling the Slowdive vinyl order at Dungeon Books.".into(),
                ts: 1000,
                uuid: "u1".into(),
            }],
        )
        .unwrap();
        let mut s = Server::new(home.clone(), false);
        let resp = call(
            &mut s,
            1,
            "tools/call",
            json!({"name": "zeromem_recall", "arguments": {"query": "What is Carrie handling?"}}),
        );
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Slowdive vinyl"), "{text}");

        // exclude_session filters everything from that session out
        let resp = call(
            &mut s,
            2,
            "tools/call",
            json!({"name": "zeromem_recall",
                   "arguments": {"query": "What is Carrie handling?", "exclude_session": "cc-1"}}),
        );
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(!text.contains("Slowdive"), "{text}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn stats_and_forget_roundtrip() {
        let home = temp_home("forget");
        spool::append_event(
            &home,
            &[SpoolTurn {
                session_id: "old".into(),
                speaker: "user".into(),
                text: "Lychee naps on the poetry shelf.".into(),
                ts: 1000,
                uuid: "u1".into(),
            }],
        )
        .unwrap();
        let mut s = Server::new(home.clone(), false);
        let stats = call(&mut s, 1, "tools/call", json!({"name": "zeromem_stats", "arguments": {}}));
        assert!(stats["result"]["content"][0]["text"].as_str().unwrap().contains("\"turns\": 1"));

        let forget = call(
            &mut s,
            2,
            "tools/call",
            json!({"name": "zeromem_forget_session", "arguments": {"session_id": "old"}}),
        );
        assert!(forget["result"]["content"][0]["text"].as_str().unwrap().contains("\"deleted_turns\": 1"));

        let missing = call(&mut s, 3, "tools/call", json!({"name": "zeromem_recall", "arguments": {}}));
        assert_eq!(missing["result"]["isError"], true);
        let _ = std::fs::remove_dir_all(&home);
    }
}
