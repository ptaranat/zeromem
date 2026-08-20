#!/usr/bin/env sh
# End-to-end smoke of the Claude Code integration against the built binary:
# hook spools transcript turns, mcp serves them back over stdio JSON-RPC.
set -eu

zm=${1:?usage: ccsmoke.sh path/to/zm}
home=$(mktemp -d)
trap 'rm -rf "$home"' EXIT

cat > "$home/transcript.jsonl" <<'JSONL'
{"type":"user","uuid":"u1","timestamp":"2026-08-20T10:00:00Z","message":{"role":"user","content":"Slowdive played the Fillmore with MBV."}}
{"type":"assistant","uuid":"a1","timestamp":"2026-08-20T10:00:05Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"skip me"},{"type":"text","text":"Noted: Fillmore show, Slowdive and MBV."}]}}
{"type":"user","uuid":"u2","timestamp":"2026-08-20T10:01:00Z","message":{"role":"user","content":[{"type":"tool_result","content":"tool noise, skip"}]}}
JSONL

printf '{"session_id":"cc-smoke","transcript_path":"%s","hook_event_name":"Stop"}' "$home/transcript.jsonl" \
  | ZEROMEM_HOME="$home" "$zm" hook

[ -n "$(ls "$home/spool")" ] || { echo "ccsmoke: hook spooled nothing" >&2; exit 1; }

out=$(
  {
    printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}'
    printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
    printf '%s\n' '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"zeromem_recall","arguments":{"query":"Who played the Fillmore?"}}}'
  } | "$zm" --no-model mcp --home "$home"
)

echo "$out" | grep -q '"serverInfo"'        || { echo "ccsmoke: no initialize response" >&2; exit 1; }
echo "$out" | grep -q 'zeromem_recall'      || { echo "ccsmoke: recall tool not listed" >&2; exit 1; }
echo "$out" | grep -q 'Fillmore'            || { echo "ccsmoke: recall returned no evidence" >&2; exit 1; }
"$zm" --db "$home/zeromem.db" --no-model stats | grep -q '"turns": 2' \
                                            || { echo "ccsmoke: expected 2 turns in stats" >&2; exit 1; }
[ -z "$(ls "$home/spool")" ]                || { echo "ccsmoke: spool not drained" >&2; exit 1; }

echo "ccsmoke: ok"
