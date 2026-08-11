"""Zero-token memory provider for Hermes Agent.

Wraps the zeromem Rust core (Zero-Mem, arXiv 2607.29377). Every memory
operation is deterministic and token-free; only the host's own reader call
touches an LLM.

Install: symlink this directory to ~/.hermes/plugins/zeromem/, install
the zeromem wheel into the Hermes environment, set memory.provider: zeromem.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from agent.memory_provider import MemoryProvider
except ImportError:  # standalone tests
    class MemoryProvider:
        pass

_RECALL_SCHEMA = {
    "name": "zeromem_recall",
    "description": (
        "Recall evidence from past conversations. Returns verbatim turns with "
        "provenance (session, turn, speaker, time), never summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for"},
            "top_k": {"type": "integer", "description": "Max main evidence items", "default": 5},
        },
        "required": ["query"],
    },
}

_STATS_SCHEMA = {
    "name": "zeromem_stats",
    "description": "Memory store counters: turns, sessions, entities, windows, episodes.",
    "parameters": {"type": "object", "properties": {}},
}

_FORGET_SCHEMA = {
    "name": "zeromem_forget_session",
    "description": (
        "Permanently delete one past session from memory: its turns, derived "
        "graph state, and cached embeddings. Irreversible; only on explicit "
        "user request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session to delete"},
        },
        "required": ["session_id"],
    },
}


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for e in evidence:
        tag = time.strftime("%Y-%m-%d", time.gmtime(e["ts"])) if e.get("ts") else "?"
        lines.append(f"- [{e['session_id']} #{e['session_turn']} {e['speaker']} {tag}] {e['text']}")
    return "\n".join(lines)


class ZeroMemProvider(MemoryProvider):
    def __init__(self) -> None:
        self._mem = None
        self._session_id = ""
        self._read_only = False
        self._config: Dict[str, Any] = {}
        self._prefetched: Dict[str, str] = {}
        self._forgotten: set = set()
        self._writes: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._writer: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "zeromem"

    def is_available(self) -> bool:
        try:
            import zeromem  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        import zeromem

        hermes_home = Path(kwargs.get("hermes_home") or Path.home() / ".hermes")
        self._session_id = session_id
        # cron/subagent contexts: read-only
        self._read_only = kwargs.get("agent_context", "primary") != "primary"

        cfg_path = hermes_home / "memory" / "zeromem.json"
        if cfg_path.exists():
            try:
                self._config = json.loads(cfg_path.read_text())
            except ValueError:
                logger.warning("zeromem: ignoring malformed %s", cfg_path)
        db = self._config.get("db_path") or str(hermes_home / "memory" / "zeromem.db")
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        self._mem = zeromem.ZeroMem(
            db,
            use_model=self._config.get("use_model", True),
            model_cache_dir=str(hermes_home / "memory" / "zeromem-models"),
        )
        if not self._read_only:
            self._writer = threading.Thread(target=self._drain_writes, daemon=True, name="zeromem-writer")
            self._writer.start()

    def _drain_writes(self) -> None:
        while True:
            item = self._writes.get()
            if item is None:
                return
            session_id, speaker, text = item
            try:
                with self._lock:
                    # a write dequeued before its session was forgotten must
                    # not resurrect it
                    if session_id in self._forgotten:
                        continue
                    self._mem.ingest_turn(session_id, speaker, text, int(time.time()))
            except Exception:
                logger.exception("zeromem: ingest failed")

    def system_prompt_block(self) -> str:
        return (
            "Long-term memory is zeromem: verbatim conversation traces with provenance. "
            "Use zeromem_recall to search past sessions when context is missing."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cached = self._prefetched.pop(session_id or self._session_id, "")
        if cached:
            return cached
        return self._recall_block(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        key = session_id or self._session_id

        def work() -> None:
            self._prefetched[key] = self._recall_block(query)

        threading.Thread(target=work, daemon=True, name="zeromem-prefetch").start()

    def _recall_block(self, query: str) -> str:
        if not self._mem or not query or not query.strip():
            return ""
        try:
            with self._lock:
                result = self._mem.query(query, None)
        except Exception:
            logger.exception("zeromem: query failed")
            return ""
        evidence = result.get("evidence", [])
        # current-session turns are already in context
        evidence = [e for e in evidence if e.get("session_id") != self._session_id]
        if not evidence:
            return ""
        return "Recalled from past sessions (verbatim, with provenance):\n" + _format_evidence(evidence)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None) -> None:
        if self._read_only or not self._mem:
            return
        sid = session_id or self._session_id
        if user_content and user_content.strip():
            self._writes.put((sid, "user", user_content))
        if assistant_content and assistant_content.strip():
            self._writes.put((sid, "assistant", assistant_content))

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [_RECALL_SCHEMA, _STATS_SCHEMA]
        if not self._read_only:
            schemas.append(_FORGET_SCHEMA)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._mem:
            return json.dumps({"error": "zeromem not initialized"})
        try:
            if tool_name == "zeromem_recall":
                with self._lock:
                    result = self._mem.query(args["query"], args.get("top_k"))
                return json.dumps(result)
            if tool_name == "zeromem_stats":
                with self._lock:
                    return json.dumps(self._mem.stats())
            if tool_name == "zeromem_forget_session":
                if self._read_only:
                    return json.dumps({"error": "memory is read-only in this context"})
                sid = args.get("session_id")
                if not sid:
                    return json.dumps({"error": "session_id is required"})
                if sid == self._session_id:
                    return json.dumps({"error": "refusing to delete the active session"})
                with self._lock:
                    removed = self._mem.delete_session(sid)
                    self._forgotten.add(sid)
                return json.dumps({"session_id": sid, "deleted_turns": removed})
        except Exception as exc:
            logger.exception("zeromem: tool %s failed", tool_name)
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown tool {tool_name}"})

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        with self._lock:
            self._session_id = new_session_id
            self._forgotten.discard(new_session_id)
        self._prefetched.clear()

    def shutdown(self) -> None:
        if self._writer:
            self._writes.put(None)
            self._writer.join(timeout=5)
            self._writer = None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "SQLite database path (default: HERMES_HOME/memory/zeromem.db)"},
            {"key": "use_model", "description": "Use the ONNX embedder; false falls back to hashing",
             "type": "boolean", "default": True},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "memory" / "zeromem.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=2))


PROVIDER_CLASS = ZeroMemProvider
