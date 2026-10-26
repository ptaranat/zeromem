# zeromem

Rust implementation of Zero-Mem (Xiao et al., [arXiv:2607.29377](https://arxiv.org/abs/2607.29377)):
agent memory where every operation outside final question answering makes zero
LLM calls and consumes zero tokens. Raw conversation turns are the source of
record; retrieval is structured search over them, not generated summaries.

Ships as a Rust crate, a `zm` CLI, a Python module (PyO3), and a memory
provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> **Default embedder (this build):** `intfloat/multilingual-e5-small`
> (384-dim, ES/EN plus ~100 languages), in place of the upstream English-only
> `bge-small-en-v1.5` — so non-English conversations embed and recall properly.
> A standalone `zeromem.Embedder` is also exposed for tooling that must produce
> store-compatible vectors.

## How it works

Ingest indexes each turn three ways: an entity-context graph (heuristic NER,
co-occurrence edges weighted per eq 4), a temporal hierarchy (turns, windows,
episodes split on session change, time gap, or topic drift), and BM25 plus
dense embeddings. A query builds a deterministic profile (subject, keywords,
answer type, temporal cues, session boundary), routes to a primary view, runs
both retrievals, fuses with min-max normalization, closes over graph bridges
and local neighbors, and calibrates the evidence set. Optionally the host can
calibrate the reader's answer against typed evidence candidates.

| paper | module |
|---|---|
| eq 3-4 entity-context graph | `graph.rs`, `ner.rs` |
| eq 5 trace hierarchy | `hierarchy.rs` |
| eq 6 query profile | `profile.rs` |
| eq 7 routing | `route.rs` |
| eq 8-10 graph view (PPR) | `graph_view.rs` |
| eq 11 hierarchical view | `hier_view.rs` |
| eq 12-13 fusion | `fuse.rs` |
| eq 14 evidence closure | `closure.rs` |
| eq 15-16 calibration | `calibrate.rs` |

Defaults follow the paper: gamma 0.6, rho 0.6, top-5 evidence.

Persistence is a single SQLite file (turns plus embedding cache). Graph,
hierarchy, and BM25 rebuild from turns on open, so the store cannot drift from
the indexes.

Deviations from the paper: spaCy NER is replaced with a heuristic extractor
behind a trait, and BGE-M3 with a multilingual dense embedder —
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
(384-dim, ES/EN plus ~100 languages) through
[fastembed](https://github.com/Anush008/fastembed-rs) (a deterministic hash
embedder is the offline fallback and what tests use). Both are swappable.

## Build

Rust core:

```
just test        # offline, hash embedder
just demo        # ingest examples/dungeon-books.jsonl, run queries
cargo build --release
```

The `fastembed` feature (default) pulls onnxruntime and downloads the
multilingual-e5-small ONNX model on first use. `--no-default-features` drops both.

CLI:

```
zm --db mem.db ingest turns.jsonl     # {"session_id","speaker","text","ts"}
zm --db mem.db query "What did Carrie handle at the store?"
zm --db mem.db stats
```

Python:

```
just dev         # maturin develop into .venv
```

```python
import zeromem
m = zeromem.ZeroMem("mem.db")
m.ingest_turn("s1", "user", "I moved to Jersey City on February 14, 2022.", 1644796800)
m.query("When did I move?")
m.calibrate_answer("When did I move?", "June 2021", ["I moved to Jersey City on February 14, 2022."])
```

Standalone embedder (no store handle) — the same model and normalization the
store uses, for tooling that must emit compatible vectors (e.g. re-embedding a
batch after a text cleanup):

```python
emb = zeromem.Embedder("path/to/model-cache")  # mirrors the core's default embedder
emb.id()     # e.g. "multilingual-e5-small"
emb.dim()    # 384
emb.embed(["hola", "hello"])  # list of l2-normalized vectors
```

## Hermes Agent

```
ln -s $(pwd)/hermes/zeromem ~/.hermes/plugins/zeromem
uv pip install --python <hermes-python> target/wheels/zeromem_py-*.whl
```

Set in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: zeromem
```

The provider syncs turns in a background thread, prefetches evidence before
each turn, and exposes `zeromem_recall` and `zeromem_stats` tools. Verify with
`just pysmoke`.

## Scripts

`scripts/strip_room_context_backfill.py` is a one-shot, idempotent migration
for the Hermes group-chat case: it strips the injected "Historical room
context" block from turns already ingested before the ingest-time strip, and
re-embeds the cleaned text using the core's own embedder (so it tracks whatever
model the store uses). It validates against stored vectors of untouched turns
before writing anything and aborts per-DB on any mismatch. Run it with a venv
that has the `zeromem` wheel:

```
python scripts/strip_room_context_backfill.py --dry-run   # then drop --dry-run
```

## Claude Code

One shared store across every project, default `~/.zeromem` (override with
`ZEROMEM_HOME`). A Stop/SessionEnd hook (`zm hook`) parses each session's
transcript incrementally and spools clean turns without ever opening the DB;
a stdio MCP server (`zm mcp`) holds the index open for the session, drains
the spool before each call, and serves `zeromem_recall`, `zeromem_stats`,
and `zeromem_forget_session`.

```
cargo install --path crates/zeromem     # zm on PATH
claude plugin marketplace add ptaranat/zeromem
claude plugin install zeromem@zeromem
```

Without the plugin: add the two hooks to `~/.claude/settings.json` (command
`zm hook` on `Stop` and `SessionEnd`) and register the server with
`claude mcp add --scope user zeromem -- zm mcp`. Verify with `just ccsmoke`.

Caveats: Claude Code gives MCP servers no session identity, so recall can
echo turns from the current session (the `exclude_session` argument exists
for hosts that know theirs). Forgetting a session that is still open in
another window partially resurrects it as that session keeps talking. There
is no prompt-time prefetch; recall is tool-driven.

## Credits

Method from the Zero-Mem paper by Yilin Xiao, Zhehan Zhu, Yujing Zhang, Jin
Chen, Zijin Hong, Luyao Zhuang, Qinggang Zhang, Shengyuan Chen, Xiaocao
Ouyang, Lingfei Ren, and Xiao Huang. This is an independent clean-room
implementation from the paper text; the authors' reference code was not
published at the time of writing. Integration shape follows
[mnemosyne](https://github.com/mnemosyne-oss/mnemosyne).

MIT licensed.
