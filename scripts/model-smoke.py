"""Load the dense embedder from an installed wheel and recall through it."""

import tempfile
import time
from pathlib import Path

import zeromem


def main() -> None:
    home = Path(tempfile.mkdtemp(prefix="zeromem-model-"))
    mem = zeromem.ZeroMem(str(home / "mem.db"), use_model=True, model_cache_dir=str(home / "models"))
    mem.ingest_turn("session-a", "user", "We got a dog named Lychee, she is a corgi.", int(time.time()))

    stats = mem.stats()
    assert not stats["embedder_is_fallback"], stats

    texts = [e["text"] for e in mem.query("What breed is Lychee?", None, None)["evidence"]]
    assert any("corgi" in t for t in texts), texts
    print("model smoke ok:", stats["embedder"])


if __name__ == "__main__":
    main()
