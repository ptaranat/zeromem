# Changelog

## [0.4.0] - 2026-11-04

Retrieval fixes from benchmarking against the authors' released code.
LoCoMo evidence recall at top_k=5 goes from 0.249 to 0.573, and reaches
0.821 at top_k=50 (the official code scores 0.848 at a similar context
size).

- `query()` returns `top_k` main results, each with its supporting turns.
  It used to cut the interleaved main/support list to `top_k`, which kept
  two of five mains.
- The graph view adds BM25 to its turn scores (`graph_lexical_weight`,
  default 1.0), so a turn that shares the question's words scores without
  an entity or dense match.
- Relative time words ("yesterday", "last week", "recently") count as time
  mentions and no longer force latest-only supersession.
- "Last session" means the session before the current one. The Python
  binding takes `query(text, top_k=None, exclude_session=None)`, and the
  Hermes plugin's prefetch and recall tool skip the active session.
- `exclude_session` filters before the final selection, so excluding a
  session no longer shrinks the result set below `top_k`.
- `Stats.embedder_is_fallback`, and `zm mcp` warns on the first recall
  when running the hash fallback embedder.
- `zm hook` writes its offset file under a pid-suffixed temp name, and the
  embedding-cache sweep joins on an indexed integer column.
- The dense embedder is selectable: `bge-small-en-v1.5` (default) or
  `multilingual-e5-small`, via `zm --model`, `ZeroMem(..., model=)` in
  Python, and `model` in the Hermes plugin's `zeromem.json`. An unknown name
  is an error. Switching re-embeds the store on open. Thanks to gas for the
  multilingual push in #8.
- Rust: `default_embedder(dir)` is now `dense_embedder(model, dir)`, and
  `mcp::Server::new` takes `Option<DenseModel>` in place of `use_model`.
||||||| Stash base
- Published to PyPI as `zeromem`, with `abi3` wheels for Linux, macOS and
  Windows. Model downloads use rustls, so the wheels need no system OpenSSL.
- The Hermes plugin registers through `register(ctx)`, declares the wheel in
  `python_dependencies`, and passes `hermes plugins validate`.


## [0.3.0] - 2026-08-20

Claude Code integration.

- `zm mcp`: stdio MCP server for Claude Code. One shared store across every
  project (default `~/.zeromem`, `ZEROMEM_HOME` to override), opened lazily,
  refreshed against concurrent sessions before each tool call. Tools:
  `zeromem_recall`, `zeromem_stats`, `zeromem_forget_session`.
- `zm hook`: Stop/SessionEnd hook that parses the session transcript
  incrementally, filters harness noise, and spools clean turns without
  opening the DB. The server drains the spool; files are claimed by atomic
  rename and crash-orphaned claims are adopted safely.
- Multi-process safe store: busy_timeout, `session_turn` assigned inside the
  INSERT statement, optional `source_uuid` dedup index, and
  `ZeroMem::refresh()` with a generation token so a delete in one process
  forces a rebuild in others.
- Claude Code plugin packaging plus a repo-root marketplace manifest;
  `just ccsmoke` drives the built binary end to end.

Caveats documented in the README: recall can echo the current session,
forgetting a session still open elsewhere partially resurrects it, no
prompt-time prefetch.

## [0.2.0] - 2026-08-11

Session deletion and NER noise filtering.

- `ZeroMem::delete_session(session_id)` removes a session's turns, rebuilds
  derived state from the surviving turns, and sweeps embedding-cache rows
  nothing references anymore. A failed rebuild rolls back cleanly.
- The Python binding exposes `delete_session`.
- The Hermes plugin adds a `zeromem_forget_session` tool: hidden in
  read-only contexts, refuses the active session, and drops queued writes
  for a forgotten session.
- NER masks fenced code blocks and inline backtick spans before extraction
  and drops captures that contain no letters or look like code.

## [0.1.0] - 2026-08-05

Initial release: clean-room Rust implementation of Zero-Mem
(arXiv 2607.29377). Entity-context graph, temporal hierarchy, BM25 plus
dense fusion, evidence closure and calibration. Ships as a crate, the `zm`
CLI, a PyO3 Python module, and a Hermes Agent memory provider. Single
SQLite file; derived state rebuilds from raw turns on open.

[0.3.0]: https://github.com/ptaranat/zeromem/releases/tag/v0.3.0
[0.2.0]: https://github.com/ptaranat/zeromem/releases/tag/v0.2.0
[0.1.0]: https://github.com/ptaranat/zeromem/releases/tag/v0.1.0
