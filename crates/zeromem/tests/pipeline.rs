use zeromem::closure::EvidenceRole;
use zeromem::config::Config;
use zeromem::embed::HashEmbedder;
use zeromem::ZeroMem;

fn corpus() -> ZeroMem {
    let mut zm =
        ZeroMem::open_in_memory(Config::default(), Box::new(HashEmbedder::default())).unwrap();
    let data = include_str!("../../../examples/dungeon-books.jsonl");
    for line in data.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        zm.ingest_turn(
            v["session_id"].as_str().unwrap(),
            v["speaker"].as_str().unwrap(),
            v["text"].as_str().unwrap(),
            v["ts"].as_i64().unwrap(),
        )
        .unwrap();
    }
    zm
}

fn top_texts(zm: &ZeroMem, q: &str) -> Vec<String> {
    zm.query(q, None)
        .unwrap()
        .evidence
        .into_iter()
        .map(|e| e.text)
        .collect()
}

#[test]
fn entity_question_finds_cross_session_evidence() {
    let zm = corpus();
    let texts = top_texts(&zm, "What is Carrie handling at the store?");
    assert!(
        texts.iter().any(|t| t.contains("Ingram")),
        "expected the Ingram turn, got {texts:#?}"
    );
}

#[test]
fn temporal_question_surfaces_dated_turn() {
    let zm = corpus();
    let texts = top_texts(&zm, "When did I move to Jersey City?");
    assert!(
        texts.iter().any(|t| t.contains("February 14, 2022")),
        "expected the dated move turn, got {texts:#?}"
    );
}

#[test]
fn updated_state_wins_for_latest_preference() {
    let zm = corpus();
    let texts = top_texts(&zm, "When is the opening at the latest?");
    assert!(
        texts.iter().any(|t| t.contains("August 23")),
        "expected the rescheduled opening, got {texts:#?}"
    );
}

#[test]
fn closure_attaches_supporting_context() {
    let zm = corpus();
    let result = zm.query("What breed is Lychee?", None).unwrap();
    assert!(result.evidence.iter().any(|e| e.role != EvidenceRole::Main));
    assert!(
        result.evidence.iter().any(|e| e.text.contains("corgi")),
        "got {:#?}",
        result.evidence.iter().map(|e| &e.text).collect::<Vec<_>>()
    );
}

#[test]
fn boundary_restricts_to_first_session() {
    let zm = corpus();
    let result = zm
        .query(
            "What did the shelving cost in the first conversation?",
            None,
        )
        .unwrap();
    for e in &result.evidence {
        assert_eq!(e.session_id, "s1", "boundary leak: {e:?}");
    }
    assert!(result.evidence.iter().any(|e| e.text.contains("$1200")));
}

#[test]
fn answer_calibration_replaces_unsupported_date() {
    let zm = corpus();
    let q = "When did I move to Jersey City?";
    let result = zm.query(q, None).unwrap();
    // Unique-candidate replacement: calibrate against the evidence item that
    // carries the date. With multiple dates in scope the answer must be kept.
    let dated: Vec<&str> = result
        .evidence
        .iter()
        .filter(|e| e.text.contains("February 14, 2022"))
        .map(|e| e.text.as_str())
        .collect();
    assert!(!dated.is_empty(), "retrieval missed the dated turn");
    let out = zm.calibrate_answer(q, "June 2021", &dated);
    assert!(out.changed, "{out:?}");
    assert!(out.answer.contains("february 14, 2022"));

    let all: Vec<&str> = result.evidence.iter().map(|e| e.text.as_str()).collect();
    let ambiguous = zm.calibrate_answer(q, "June 2021", &all);
    if ambiguous.candidates.len() > 1 {
        assert!(
            !ambiguous.changed,
            "must not guess among candidates: {ambiguous:?}"
        );
    }
}

#[test]
fn persistence_roundtrip() {
    let dir = std::env::temp_dir().join(format!("zeromem-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = dir.join("t.db");
    {
        let mut zm =
            ZeroMem::open(&db, Config::default(), Box::new(HashEmbedder::default())).unwrap();
        zm.ingest_turn("s1", "user", "Rex the parrot says Copenhagen.", 100)
            .unwrap();
    }
    {
        let zm = ZeroMem::open(&db, Config::default(), Box::new(HashEmbedder::default())).unwrap();
        assert_eq!(zm.stats().turns, 1);
        let texts = top_texts(&zm, "Where does Rex say?");
        assert!(texts.iter().any(|t| t.contains("Copenhagen")), "{texts:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deleted_session_leaves_no_trace_in_retrieval() {
    let mut zm = corpus();
    let sessions_before = zm.stats().sessions;
    let removed = zm.delete_session("s1").unwrap();
    assert!(removed > 0);
    assert_eq!(zm.stats().sessions, sessions_before - 1);

    for q in [
        "What did the shelving cost?",
        "What breed is Lychee?",
        "When did I move to Jersey City?",
    ] {
        let result = zm.query(q, None).unwrap();
        for e in &result.evidence {
            assert_ne!(
                e.session_id, "s1",
                "deleted session leaked for {q:?}: {e:?}"
            );
        }
    }
}
