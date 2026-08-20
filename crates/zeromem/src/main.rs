use std::path::PathBuf;
use zeromem::{config::Config, default_embedder, embed::HashEmbedder, ZeroMem};

const USAGE: &str = "usage: zm [--db PATH] [--no-model] <command>
commands:
  ingest <file.jsonl>   lines: {\"session_id\", \"speaker\", \"text\", \"ts\"?}
  query <text> [-k N]
  stats
  hook                  Claude Code Stop/SessionEnd hook; reads event JSON on stdin
ZEROMEM_HOME overrides the default ~/.zeromem home for mcp and hook.";

fn main() {
    if let Err(e) = run() {
        eprintln!("zm: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut db = PathBuf::from("zeromem.db");
    let mut no_model = false;

    if let Some(i) = args.iter().position(|a| a == "--db") {
        args.remove(i);
        db = PathBuf::from(args.remove(i));
    }
    if let Some(i) = args.iter().position(|a| a == "--no-model") {
        args.remove(i);
        no_model = true;
    }
    let mut home = zeromem::spool::default_home();
    if let Some(i) = args.iter().position(|a| a == "--home") {
        args.remove(i);
        home = PathBuf::from(args.remove(i));
    }

    // mcp and hook manage their own store; no embedder up front
    match args.first().map(String::as_str).unwrap_or("") {
        "hook" => {
            let mut input = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
            // a memory hiccup must not fail the user's session: log and exit 0
            if let Err(e) = zeromem::hook::run(&home, &input) {
                eprintln!("zm hook: {e}");
            }
            return Ok(());
        }
        _ => {}
    }

    let embedder = if no_model {
        Box::new(HashEmbedder::default()) as Box<dyn zeromem::embed::Embedder>
    } else {
        default_embedder(None)
    };

    let cmd = args.first().map(String::as_str).unwrap_or("");
    match cmd {
        "ingest" => {
            let file = args.get(1).ok_or(USAGE)?;
            let mut zm = ZeroMem::open(&db, Config::default(), embedder)?;
            let mut n = 0usize;
            for line in std::fs::read_to_string(file)?.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(line)?;
                let ts = v["ts"].as_i64().unwrap_or(n as i64);
                zm.ingest_turn(
                    v["session_id"].as_str().ok_or("missing session_id")?,
                    v["speaker"].as_str().unwrap_or("user"),
                    v["text"].as_str().ok_or("missing text")?,
                    ts,
                )?;
                n += 1;
            }
            println!("ingested {n} turns into {}", db.display());
        }
        "query" => {
            let text = args.get(1).ok_or(USAGE)?;
            let k = args
                .iter()
                .position(|a| a == "-k")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok());
            let zm = ZeroMem::open(&db, Config::default(), embedder)?;
            let result = zm.query(text, k)?;
            println!("route: {:?}", result.route);
            for e in &result.evidence {
                println!(
                    "{:>6.3} {:?} [{} #{} {}] {}",
                    e.score, e.role, e.session_id, e.session_turn, e.speaker, e.text
                );
            }
        }
        "stats" => {
            let zm = ZeroMem::open(&db, Config::default(), embedder)?;
            println!("{}", serde_json::to_string_pretty(&zm.stats())?);
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}
