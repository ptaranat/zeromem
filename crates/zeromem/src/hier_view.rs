//! Hierarchical retrieval, paper eq 11: episodes -> windows -> turns.

use crate::config::Config;
use crate::embed::{cosine, l2_normalize};
use crate::graph::EntityGraph;
use crate::hierarchy::Hierarchy;
use crate::lexical::{phrase_matches, Bm25};
use crate::ner::EntityKind;
use crate::profile::{AnswerType, Boundary, QueryProfile};
use crate::trace::Turn;
use std::collections::HashMap;

pub struct HierViewInput<'a> {
    pub hier: &'a Hierarchy,
    pub graph: &'a EntityGraph,
    pub turns: &'a [Turn],
    pub turn_vecs: &'a [Vec<f32>],
    pub query_vec: &'a [f32],
    pub bm25: &'a Bm25,
    pub profile: &'a QueryProfile,
    /// Session ids in first-seen order, for boundary checks.
    pub session_order: &'a [String],
}

pub fn retrieve(input: &HierViewInput, cfg: &Config) -> HashMap<u32, f32> {
    if input.turns.is_empty() {
        return HashMap::new();
    }
    let bm25_scores = input.bm25.scores(&input.profile.keywords);
    let bm25_max = bm25_scores.values().cloned().fold(0.0f32, f32::max).max(1e-6);
    let base = |turn: u32| -> f32 {
        let dense = cosine(input.query_vec, &input.turn_vecs[turn as usize]).max(0.0);
        let lex = bm25_scores.get(&turn).copied().unwrap_or(0.0) / bm25_max;
        0.5 * dense + 0.5 * lex
    };

    // episode beam
    let mut episode_scores: Vec<(u32, f32)> = input
        .hier
        .episodes
        .iter()
        .enumerate()
        .map(|(i, ep)| {
            let mut best = 0.0f32;
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for t in ep.start..=ep.end {
                let s = base(t);
                best = best.max(s);
                sum += s;
                n += 1.0;
            }
            (i as u32, best + 0.1 * sum / n.max(1.0))
        })
        .collect();
    episode_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    episode_scores.truncate(cfg.episode_beam);

    // window beam within surviving episodes
    let mut window_scores: Vec<(u32, f32)> = Vec::new();
    for (ep, _) in &episode_scores {
        for (wid, w) in input.hier.windows_of_episode(*ep) {
            let mut centroid = w.centroid.clone();
            l2_normalize(&mut centroid);
            let dense = cosine(input.query_vec, &centroid).max(0.0);
            let best_turn = (w.start..=w.end).map(base).fold(0.0f32, f32::max);
            window_scores.push((wid, 0.4 * dense + 0.6 * best_turn));
        }
    }
    window_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    window_scores.truncate(cfg.window_beam);

    let mut out: HashMap<u32, f32> = HashMap::new();
    for (wid, _) in &window_scores {
        let w = &input.hier.windows[*wid as usize];
        for t in w.start..=w.end {
            let s = base(t) * (1.0 + compatibility(input, t));
            if s > 0.0 {
                out.insert(t, s);
            }
        }
    }
    out
}

/// Bonus term for the (1 + bonus) multiplier: subject, temporal, boundary,
/// answer-type, and phrase signals. Boundary mismatch goes strongly negative.
fn compatibility(input: &HierViewInput, turn: u32) -> f32 {
    let t = &input.turns[turn as usize];
    let p = input.profile;
    let mut bonus = 0.0f32;

    if !p.subjects.is_empty() {
        let ents = &input.graph.turn_entities[turn as usize];
        let hits = p
            .subjects
            .iter()
            .filter(|s| {
                input
                    .graph
                    .entity_id(s)
                    .map_or(false, |id| ents.iter().any(|(e, _)| *e == id))
            })
            .count();
        bonus += 0.25 * hits as f32 / p.subjects.len() as f32;
    }
    if !p.temporal.ranges.is_empty() {
        let inside = p.temporal.ranges.iter().any(|(a, b)| t.ts >= *a && t.ts <= *b);
        bonus += if inside { 0.3 } else { 0.0 };
    }
    if let Some(boundary) = p.boundary {
        let session_pos = input.session_order.iter().position(|s| *s == t.session_id);
        let matches = match (boundary, session_pos) {
            (Boundary::SessionOrdinal(k), Some(pos)) => pos == k,
            (Boundary::LastSession, Some(pos)) => pos + 1 == input.session_order.len(),
            _ => false,
        };
        bonus += if matches { 0.3 } else { -0.9 };
    }
    let type_hit = match p.answer_type {
        AnswerType::Time => has_kind(input, turn, EntityKind::Date),
        AnswerType::Number => has_kind(input, turn, EntityKind::Number),
        AnswerType::Person | AnswerType::Entity | AnswerType::Place => has_kind(input, turn, EntityKind::Named),
        _ => false,
    };
    if type_hit {
        bonus += 0.15;
    }
    bonus += 0.1 * phrase_matches(&p.phrases, &t.text).min(3) as f32;
    bonus
}

fn has_kind(input: &HierViewInput, turn: u32, kind: EntityKind) -> bool {
    input.graph.turn_entities[turn as usize]
        .iter()
        .any(|(e, _)| input.graph.entities[*e as usize].kind == kind)
}
