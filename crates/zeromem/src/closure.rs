//! Evidence closure, paper eq 14: main evidence + graph bridges + local
//! neighbors, deduped by turn.

use crate::config::Config;
use crate::graph::EntityGraph;
use crate::trace::Turn;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum EvidenceRole {
    Main,
    GraphBridge,
    LocalNeighbor,
}

#[derive(Debug, Clone)]
pub struct Selected {
    pub turn: u32,
    pub score: f32,
    pub role: EvidenceRole,
    /// Main turn this supports (itself for mains).
    pub anchor: u32,
}

pub fn close(
    main: &[(u32, f32)],
    graph: &EntityGraph,
    turns: &[Turn],
    cfg: &Config,
) -> Vec<Selected> {
    let mut seen: HashSet<u32> = main.iter().map(|(t, _)| *t).collect();
    let mut out: Vec<Selected> = main
        .iter()
        .map(|&(turn, score)| Selected { turn, score, role: EvidenceRole::Main, anchor: turn })
        .collect();

    for &(m, score) in main {
        if let Some(bridge) = best_graph_neighbor(m, graph, &seen) {
            seen.insert(bridge);
            out.push(Selected {
                turn: bridge,
                score: score * 0.5,
                role: EvidenceRole::GraphBridge,
                anchor: m,
            });
        }
        for nb in local_neighbors(m, turns, cfg) {
            if seen.insert(nb) {
                out.push(Selected {
                    turn: nb,
                    score: score * 0.5,
                    role: EvidenceRole::LocalNeighbor,
                    anchor: m,
                });
            }
        }
    }
    out
}

/// Turn sharing entities with m, scored by min shared weight.
fn best_graph_neighbor(m: u32, graph: &EntityGraph, seen: &HashSet<u32>) -> Option<u32> {
    let m_weights = graph.turn_weights(m);
    let mut best: Option<(u32, f32)> = None;
    for (e, wm) in &m_weights {
        for &(d, _) in &graph.postings[*e as usize] {
            if d == m || seen.contains(&d) {
                continue;
            }
            let wd = graph
                .turn_weights(d)
                .iter()
                .find(|(e2, _)| e2 == e)
                .map(|(_, w)| *w)
                .unwrap_or(0.0);
            let s = wm.min(wd);
            if s > 0.0 {
                best = match best {
                    Some((bd, bs)) if bs >= s => Some((bd, bs)),
                    _ => Some((d, s)),
                };
            }
        }
    }
    best.map(|(d, _)| d)
}

/// Prev/next turn in-session; short or anaphoric turns get the full span.
fn local_neighbors(m: u32, turns: &[Turn], cfg: &Config) -> Vec<u32> {
    let t = &turns[m as usize];
    let needs_span = t.text.len() < 48 || starts_anaphoric(&t.text);
    let radius = if needs_span { cfg.local_span as i64 } else { 1 };
    let mut out = Vec::new();
    for delta in -radius..=radius {
        let idx = m as i64 + delta;
        if delta == 0 || idx < 0 || idx as usize >= turns.len() {
            continue;
        }
        if turns[idx as usize].session_id == t.session_id {
            out.push(idx as u32);
        }
    }
    out
}

fn starts_anaphoric(text: &str) -> bool {
    let first = text.split_whitespace().next().unwrap_or("").to_lowercase();
    matches!(
        first.as_str(),
        "he" | "she" | "they" | "it" | "that" | "this" | "those" | "these" | "yes" | "no" | "yeah" | "same"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::{EntityExtractor, HeuristicNer};

    fn turn(id: i64, session: &str, text: &str) -> Turn {
        Turn {
            id,
            session_id: session.into(),
            session_turn: id,
            speaker: "user".into(),
            text: text.into(),
            ts: 1000 + id,
        }
    }

    #[test]
    fn closure_adds_shared_entity_bridge_and_local_span() {
        let turns = vec![
            turn(0, "s1", "Carrie got a dog named Lychee."),
            turn(1, "s1", "The weather was nice."),
            turn(2, "s1", "Lychee chewed the couch, Carrie was furious about the couch damage."),
        ];
        let mut g = EntityGraph::default();
        for (i, t) in turns.iter().enumerate() {
            g.add_turn(i as u32, &HeuristicNer.extract(&t.text));
        }
        let cfg = Config::default();
        let out = close(&[(0, 1.0)], &g, &turns, &cfg);
        let bridges: Vec<u32> = out
            .iter()
            .filter(|s| s.role == EvidenceRole::GraphBridge)
            .map(|s| s.turn)
            .collect();
        assert_eq!(bridges, vec![2]);
        assert!(out.iter().any(|s| s.role == EvidenceRole::LocalNeighbor && s.turn == 1));
    }
}
