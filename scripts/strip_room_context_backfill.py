#!/usr/bin/env python3
"""One-shot backfill: strip the 'Historical room context' block from room
user turns ALREADY ingested into zeromem DBs, and re-embed the cleaned text.

Background: group-chat user turns carry an injected "Historical room context"
block (10-20k chars of stale thread transcripts). The plugin now strips it at
ingest (sync_turn), but rows ingested before that fix keep the block, and its
vectors make stale room turns match almost any casual query. This script
cleans the legacy rows in place.

Properties:
- Idempotent: only touches rows still carrying the marker; re-running after
  heavy group-chat periods is the intended workflow.
- Faithful: reuses the plugin's own _strip_room_context (same regex) and the
  core's own embedder model (fastembed bge-small-en-v1.5, 384 dims, l2-norm).
- Safe: validates the embedding pipeline against stored vectors of UNMODIFIED
  turns (cosine >= 0.99 gate) before writing a single row. Aborts per-DB on
  gate failure or any error; nothing is written before the gate passes.

Usage:
  <venv-with-fastembed>/bin/python scripts/strip_room_context_backfill.py \
      [--db PATH]... [--dry-run] [--gate 0.99]

Defaults: the default-profile DB plus every profiles/*/memory/zeromem.db
under HERMES_HOME.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sqlite3
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = REPO / "hermes" / "zeromem" / "__init__.py"
MARKER = "Historical room context"


def load_strip():
    """Reuse the plugin's exact strip function (single source of truth)."""
    spec = importlib.util.spec_from_file_location("zeromem_plugin", PLUGIN)
    assert spec and spec.loader, f"no se pudo cargar {PLUGIN}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._strip_room_context


def embed_one(model, text: str) -> list[float]:
    for v in model.embed([text]):
        return [float(x) for x in v]
    return []


def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def default_dbs(hermes_home: pathlib.Path) -> list[pathlib.Path]:
    dbs = [hermes_home / "memory" / "zeromem.db"]
    prof = hermes_home / "profiles"
    if prof.is_dir():
        dbs += sorted(prof.glob("*/memory/zeromem.db"))
    return [d for d in dbs if d.exists()]


def migrate_db(db: pathlib.Path, strip, model, gate: float, dry_run: bool) -> int:
    print(f"\n=== {db}")
    con = sqlite3.connect(db, timeout=30)
    try:
        rows = con.execute(
            "SELECT id, LENGTH(text) FROM turns WHERE text LIKE ? ORDER BY ts",
            (f"%{MARKER}%",),
        ).fetchall()
        if not rows:
            print("  limpio: sin turnos con room-context")
            return 0

        # --- validation gate: pipeline must reproduce the core on unmodified
        # turns before we touch anything.
        samples = con.execute(
            "SELECT id, text FROM turns "
            "WHERE text NOT LIKE ? AND LENGTH(text) > 200 LIMIT 3",
            (f"%{MARKER}%",),
        ).fetchall()
        worst = 1.0
        for tid, text in samples:
            row = con.execute(
                "SELECT vec FROM embeddings WHERE kind='turn' AND key=?",
                (str(tid),),
            ).fetchone()
            if row is None:
                print(f"  !! turno {tid} sin embedding guardado; no valido")
                continue
            stored = list(struct.unpack(f"<{len(row[0]) // 4}f", row[0]))
            c = cosine(embed_one(model, text), stored)
            worst = min(worst, c)
            print(f"  validacion turno {tid}: cosine={c:.5f}")
        if worst < gate:
            print(f"  !! ABORT: peor cosine {worst:.5f} < gate {gate}; no se toca nada")
            return 0

        n = moved = 0
        kb_before = kb_after = 0
        for tid, ln in rows:
            text = con.execute("SELECT text FROM turns WHERE id=?", (tid,)).fetchone()[0]
            new = strip(text)
            if new == text:
                print(f"  !! turno {tid}: el regex no corta (estructura distinta?); se salta")
                continue
            vec = embed_one(model, new)
            if len(vec) != 384:
                print(f"  !! turno {tid}: dim inesperada {len(vec)}; se salta")
                continue
            if not dry_run:
                con.execute("UPDATE turns SET text=? WHERE id=?", (new, tid))
                con.execute(
                    "UPDATE embeddings SET vec=? WHERE kind='turn' AND key=?",
                    (to_blob(vec), str(tid)),
                )
            n += 1
            moved += 1
            kb_before += ln // 1024
            kb_after += len(new.encode()) // 1024
        if not dry_run:
            con.commit()
        print(
            f"  {'(dry-run) ' if dry_run else ''}migrados: {moved}/{n} turnos; "
            f"~{kb_before} KB -> ~{kb_after} KB"
        )
        return n
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strip the room-context block from legacy zeromem turns and re-embed"
    )
    ap.add_argument("--db", action="append", help="zeromem.db path (repeatable)")
    ap.add_argument("--hermes-home", default=str(pathlib.Path.home() / ".hermes"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gate", type=float, default=0.99)
    args = ap.parse_args()

    dbs = (
        [pathlib.Path(d) for d in args.db]
        if args.db
        else default_dbs(pathlib.Path(args.hermes_home))
    )

    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    strip = load_strip()

    total = 0
    for db in dbs:
        total += migrate_db(db, strip, model, args.gate, args.dry_run)
    print(f"\nTotal: {total} turnos {'(dry-run)' if args.dry_run else 'migrados'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
