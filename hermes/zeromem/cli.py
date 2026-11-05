"""`hermes zeromem` commands. Hermes loads this file without the provider module."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _provider_module():
    spec = importlib.util.spec_from_file_location("zeromem_provider", Path(__file__).with_name("__init__.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_turns(sessions: List[Dict[str, Any]], existing: Counter) -> List[tuple]:
    remaining = existing.copy()
    turns = []
    for session in sessions:
        for message in session.get("messages") or []:
            speaker, text = message.get("role"), message.get("content")
            if speaker not in ("user", "assistant") or not isinstance(text, str) or not text.strip():
                continue
            key = (session["id"], speaker, text)
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                turns.append((*key, int(message.get("timestamp") or 0)))
    return sorted(turns, key=lambda turn: turn[3])


def _existing_turns(db: str) -> Counter:
    if not Path(db).exists():
        return Counter()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return Counter(conn.execute("SELECT session_id, speaker, text FROM turns"))
    finally:
        conn.close()


def _export_sessions() -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sessions.jsonl"
        subprocess.run(["hermes", "sessions", "export", str(path)], check=True, capture_output=True)
        with path.open(encoding="utf-8") as lines:
            return [json.loads(line) for line in lines if line.strip()]


def _import(args) -> None:
    provider = _provider_module()
    hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    config = provider._load_config(hermes_home)
    turns = new_turns(_export_sessions(), _existing_turns(provider._db_path(hermes_home, config)))
    sessions = len({turn[0] for turn in turns})
    if not turns:
        print("zeromem: nothing new to import")
        return
    if not args.yes:
        print(f"zeromem: would import {len(turns)} turns from {sessions} sessions. Rerun with --yes.")
        return
    store = provider._open_store(hermes_home, config)
    for session_id, speaker, text, ts in turns:
        store.ingest_turn(session_id, speaker, text, ts)
    print(f"zeromem: imported {len(turns)} turns from {sessions} sessions")


def zeromem_command(args) -> None:
    if args.zeromem_command == "import":
        _import(args)


def register_cli(subparser) -> None:
    commands = subparser.add_subparsers(dest="zeromem_command", required=True)
    importer = commands.add_parser("import", help="Import past Hermes conversations into zeromem")
    importer.add_argument("--yes", action="store_true", help="Write the turns instead of previewing")
