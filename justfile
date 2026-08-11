set quiet

default: test

test:
    cargo test --no-default-features

test-model:
    cargo test

build:
    cargo build --release

demo db="/tmp/zeromem-demo.db":
    rm -f {{db}}
    cargo run -q --no-default-features --bin zm -- --db {{db}} --no-model ingest examples/dungeon-books.jsonl
    cargo run -q --no-default-features --bin zm -- --db {{db}} --no-model query "What is Carrie handling at the store?"
    cargo run -q --no-default-features --bin zm -- --db {{db}} --no-model query "When did I move to Jersey City?"
    cargo run -q --no-default-features --bin zm -- --db {{db}} --no-model stats

venv:
    uv venv --allow-existing .venv
    uv pip install --python .venv maturin

dev: venv
    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop -m crates/zeromem-py/Cargo.toml --release

wheel: venv
    VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH .venv/bin/maturin build -m crates/zeromem-py/Cargo.toml --release

pysmoke: dev
    .venv/bin/python hermes/zeromem/smoke.py
