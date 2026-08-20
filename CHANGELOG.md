# Changelog

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
