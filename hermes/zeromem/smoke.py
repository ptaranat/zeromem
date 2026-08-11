"""Provider lifecycle smoke test, no Hermes install required."""

import importlib.util
import json
import tempfile
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("zeromem_provider", Path(__file__).parent / "__init__.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ZeroMemProvider = mod.ZeroMemProvider


def main() -> None:
    provider = ZeroMemProvider()
    assert provider.name == "zeromem"
    assert provider.is_available(), "zeromem wheel not importable"

    home = tempfile.mkdtemp(prefix="zeromem-hermes-")
    provider.save_config({"use_model": False}, home)
    provider.initialize("session-a", hermes_home=home, platform="cli", agent_context="primary")

    provider.sync_turn(
        "We got a dog named Lychee.",
        "Congrats on Lychee!",
        session_id="session-a",
    )
    provider.sync_turn("Lychee is a corgi.", "Corgis are great.", session_id="session-a")
    deadline = time.time() + 5
    while time.time() < deadline:
        stats = json.loads(provider.handle_tool_call("zeromem_stats", {}))
        if stats.get("turns", 0) >= 4:
            break
        time.sleep(0.05)
    assert stats["turns"] >= 4, stats

    out = json.loads(provider.handle_tool_call("zeromem_recall", {"query": "What breed is Lychee?"}))
    texts = [e["text"] for e in out["evidence"]]
    assert any("corgi" in t for t in texts), texts

    provider.on_session_switch("session-b", reset=True)
    block = provider.prefetch("What breed is Lychee?", session_id="session-b")
    assert "corgi" in block, block
    assert "session-a" in block, block

    schemas = provider.get_tool_schemas()
    assert {s["name"] for s in schemas} == {"zeromem_recall", "zeromem_stats", "zeromem_forget_session"}

    refused = json.loads(provider.handle_tool_call("zeromem_forget_session", {"session_id": "session-b"}))
    assert "error" in refused, refused

    gone = json.loads(provider.handle_tool_call("zeromem_forget_session", {"session_id": "session-a"}))
    assert gone["deleted_turns"] == 4, gone
    out = json.loads(provider.handle_tool_call("zeromem_recall", {"query": "What breed is Lychee?"}))
    assert not out["evidence"], out["evidence"]
    stats = json.loads(provider.handle_tool_call("zeromem_stats", {}))
    assert stats["turns"] == 0, stats

    provider.sync_turn("stale write", "stale reply", session_id="session-a")
    deadline = time.time() + 5
    while not provider._writes.empty() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.05)
    stats = json.loads(provider.handle_tool_call("zeromem_stats", {}))
    assert stats["turns"] == 0, f"forgotten session resurrected: {stats}"

    provider.shutdown()
    print("smoke ok:", json.dumps(stats))


if __name__ == "__main__":
    main()
