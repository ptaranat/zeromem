#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml)
major=${version%%.*}
rest=${version#*.}
minor=${rest%%.*}
status=0

expect() {
  if [ "$2" != "$3" ]; then
    echo "$1: expected $3, found ${2:-nothing}" >&2
    status=1
  fi
}

expect "hermes/zeromem/plugin.yaml version" \
  "$(sed -n 's/^version: //p' hermes/zeromem/plugin.yaml)" "$version"
expect "claude plugin.json version" \
  "$(sed -n 's/^  "version": "\(.*\)",$/\1/p' claude/zeromem/.claude-plugin/plugin.json)" "$version"
expect "hermes/zeromem/plugin.yaml python_dependencies" \
  "$(sed -n 's/^  - \(zeromem[<>=~!].*\)$/\1/p' hermes/zeromem/plugin.yaml)" "zeromem>=$version,<$major.$((minor + 1))"
expect "CHANGELOG.md latest entry" \
  "$(sed -n 's/^## \[\(.*\)\].*$/\1/p' CHANGELOG.md | head -1)" "$version"
if [ "${GITHUB_REF_TYPE:-}" = tag ]; then
  expect "git tag" "$GITHUB_REF_NAME" "v$version"
fi

[ "$status" -eq 0 ] && echo "versions ok: $version"
exit "$status"
