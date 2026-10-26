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
  core's OWN embedder (zm::default_embedder, exposed as zeromem.Embedder).
  The script therefore tracks whatever model the core uses (bge, e5, ...) and
  never hardcodes one, so it stays correct across embedder switches.
- Safe: validates the embed pipeline against stored vectors of UNMODIFIED
  turns (cosine >= gate) before writing a single row. Aborts per-DB on gate
  failure, on an embedder-tag mismatch, or any error; nothing is written
  before the gate passes.

Usage:
  <venv-with-zeromem-wheel>/bin/python scripts/strip_room_context_backfill.py \
      [--db PATH]... [--dry-run] [--gate 0.99] [--model-cache-dir DIR]

Defaults: the default-profile DB plus every profiles/*/memory/zeromem.db
under HERMES_HOME. The model cache dir defaults to <db parent>/zeromem-models
(the layout the plugin uses), so each profile re-embeds against its own cache.
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


def load_embedder(model_cache_dir: pathlib.Path | None):
    """The core's own default embedder — the single source of truth for the
    model + normalization the store actually uses. Loads from (or downloads
    into) the given cache dir, mirroring how the plugin builds its ZeroMem."""
    try:
        import zeromem  # the Rust core binding (wheel)
    except ImportError as e:
        sys.exit(
            f"no se pudo importar 'zeromem' (la wheel del core): {e}.\n"
            "Instala la wheel en este venv:  pip install <ruta>/zeromem_py-*.whl"
        )
    return zeromem.Embedder(str(model_cache_dir) if model_cache_dir else None)


def embed_one(emb, text: str) -> list[float]:
    for v in emb.embed([text]):
        return [float(x) for x in v]
    return []


def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def cosine(a, b) -> float:
    if len(a) != len(b) or not a:
        return 0.0
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


def meta_embedder(con) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
    return row[0] if row else None


def migrate_db(db: pathlib.Path, emb, strip, gate: float, dry_run: bool) -> int:
    print(f"\n=== {db}")
    con = sqlite3.connect(db, timeout=30)
    try:
        # Embedder-tag guard: the store's vectors were produced by meta.embedder.
        # If that differs from the core's current embedder, the core will wipe and
        # re-embed the whole store on next open anyway — re-embedding a subset here
        # with a mismatched model is pointless, so abort cleanly.
        stored_tag = meta_embedder(con)
        live_tag = emb.id()
        if stored_tag is not None and stored_tag != live_tag:
            print(f"  !! ABORT: embedder del store '{stored_tag}' != embedder del core '{live_tag}'.")
            print("     El core borrara y re-embedea todo al abrir; no hace falta backfill.")
            return 0

        dim = emb.dim()
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
                "SELECT vec FROM embeddings WHERE kind='turn' AND key=?", (str(tid),)
            ).fetchone()
            if row is None:
                print(f"  !! turno {tid} sin embedding guardado; no valido")
                continue
            stored = list(struct.unpack(f"<{len(row[0]) // 4}f", row[0]))
            c = cosine(embed_one(emb, text), stored)
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
            vec = embed_one(emb, new)
            if len(vec) != dim:
                print(f"  !! turno {tid}: dim inesperada {len(vec)} != {dim}; se salta")
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
    ap.add_argument(
        "--model-cache-dir",
        default=None,
        help="ONNX model cache for the embedder (default: <db parent>/zeromem-models)",
    )
    args = ap.parse_args()

    dbs = (
        [pathlib.Path(d) for d in args.db]
        if args.db
        else default_dbs(pathlib.Path(args.hermes_home))
    )

    strip = load_strip()
    # One embedder per model cache dir; the default layout keeps each profile's
    # own cache, so every profile re-embeds against the model it already has.
    cache_by_db: dict[pathlib.Path, object] = {}
    total = 0
    for db in dbs:
        cache = (
            pathlib.Path(args.model_cache_dir)
            if args.model_cache_dir
            else db.parent / "zeromem-models"
        )
        if cache not in cache_by_db:
            print(f"  [embedder] cache={cache}")
            cache_by_db[cache] = load_embedder(cache)
        total += migrate_db(db, cache_by_db[cache], strip, args.gate, args.dry_run)
    print(f"\nTotal: {total} turnos {'(dry-run)' if args.dry_run else 'migrados'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
