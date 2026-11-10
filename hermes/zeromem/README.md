# zeromem

Memory for Hermes Agent. Every message stays word for word in one SQLite file on your machine. Storing and searching cost zero tokens, and your model only pays to read what comes back.

## Setup

```
hermes plugins install zeromem
```

Then in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: zeromem
```

Switching from another memory provider? Hermes keeps your conversations in its own session history, so you can bring them all in. It costs no tokens, and running it twice won't duplicate anything:

```
hermes zeromem import        # preview
hermes zeromem import --yes
```

Your agent gets `zeromem_recall`, `zeromem_stats` and `zeromem_forget_session`.

## What it does on your machine

- Installs the `zeromem` wheel from PyPI.
- Downloads the bge-small-en-v1.5 embedding model (about 130MB) from Hugging Face on first run, into `~/.hermes/memory/zeromem-models`. After that it makes no network calls. To skip the download, set `{"use_model": false}` in `~/.hermes/memory/zeromem.json`; it then uses a hash embedder with slightly lower recall.
- Stores every user and assistant message as plain text in `~/.hermes/memory/zeromem.db`. Nothing leaves your machine.
- Recalls from every session in the profile, not only the current chat. On a gateway that includes every chat and every sender, so one person's messages can come back in another person's conversation.
- `zeromem_forget_session` permanently deletes a past session's stored turns. The tool tells your agent to use it only when you ask, and it refuses to delete the active session.
- Writes in a background thread and adds recalled messages to your agent's context before each turn.
- `hermes zeromem import` runs `hermes sessions export` to read your session history.

More on how it works and the benchmarks: [github.com/ptaranat/zeromem](https://github.com/ptaranat/zeromem).
