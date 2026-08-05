//! Relational retrieval (paper eq 8-10): align query entities to graph
//! entities, propagate activation through co-occurring turns, run Personalized
//! PageRank over the entity-context graph, then refine with exact matches.

use crate::config::Config;
use crate::embed::cosine;
use crate::graph::EntityGraph;
use crate::lexical::phrase_matches;
use crate::profile::QueryProfile;
use crate::trace::Turn;
use std::collections::HashMap;

pub struct GraphViewInput<'a> {
    pub graph: &'a EntityGraph,
    pub turns: &'a [Turn],
    pub turn_vecs: &'a [Vec<f32>],
    pub entity_vecs: &'a [Vec<f32>],
    pub query_vec: &'a [f32],
    pub profile: &'a QueryProfile,
}

/// Sparse turn scores from the graph view.
pub fn retrieve(input: &GraphViewInput, cfg: &Config) -> HashMap<u32, f32> {
    let n_turns = input.turns.len();
    if n_turns == 0 {
        return HashMap::new();
    }

    let activations = seed_and_propagate(input, cfg);
    let pi = personalized_pagerank(input, &activations, cfg);

    let mut scores: HashMap<u32, f32> = HashMap::new();
    for (turn, score) in pi.iter().enumerate() {
        if *score > 0.0 {
            let boost = phrase_matches(&input.profile.phrases, &input.turns[turn].text) as f32;
            scores.insert(turn as u32, score * (1.0 + 0.25 * boost));
        }
    }
    scores
}

/// eq 8-9. Seeds are query entities aligned by cosine; activation spreads to
/// entities co-occurring in turns relevant to the query.
fn seed_and_propagate(input: &GraphViewInput, cfg: &Config) -> HashMap<u32, f32> {
    let g = input.graph;
    let mut act: HashMap<u32, f32> = HashMap::new();

    for subject in input.profile.subjects.iter().chain(&input.profile.phrases) {
        if let Some(id) = g.entity_id(subject) {
            act.insert(id, 1.0);
        }
    }
    // Dense alignment for query entities with no exact match.
    if act.is_empty() && !g.is_empty() {
        for subject in &input.profile.subjects {
            let sub_vec = best_effort_entity_vec(subject, input);
            let Some(sub_vec) = sub_vec else { continue };
            let (mut best, mut best_sim) = (None, cfg.align_threshold);
            for (id, vec) in input.entity_vecs.iter().enumerate() {
                let sim = cosine(&sub_vec, vec);
                if sim > best_sim {
                    best = Some(id as u32);
                    best_sim = sim;
                }
            }
            if let Some(id) = best {
                act.insert(id, best_sim);
            }
        }
    }
    if act.is_empty() {
        return act;
    }

    // Precompute sim(q, z) per turn once; reused across steps.
    let turn_sim: Vec<f32> = input
        .turn_vecs
        .iter()
        .map(|v| cosine(input.query_vec, v).max(0.0))
        .collect();

    for _ in 0..cfg.propagation_steps {
        let mut next = act.clone();
        for (&e, &weight) in &act {
            for &(turn, _) in &g.postings[e as usize] {
                let sim = turn_sim[turn as usize];
                if sim <= 0.0 {
                    continue;
                }
                for &(e2, _) in &g.turn_entities[turn as usize] {
                    if e2 != e {
                        *next.entry(e2).or_default() += weight * sim;
                    }
                }
            }
        }
        let max = next.values().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            next.values_mut().for_each(|v| *v /= max);
        }
        act = next;
    }
    act
}

fn best_effort_entity_vec(subject: &str, input: &GraphViewInput) -> Option<Vec<f32>> {
    // Entities are embedded by name at ingest; query subjects reuse the query
    // embedding as a proxy when the name is unseen. Cheap and adequate: the
    // aligned entity still has to clear align_threshold.
    if input.entity_vecs.is_empty() || subject.is_empty() {
        None
    } else {
        Some(input.query_vec.to_vec())
    }
}

/// eq 10 over the bipartite entity-turn graph plus turn adjacency. Node order:
/// entities then turns. Returns turn-node mass.
fn personalized_pagerank(input: &GraphViewInput, activations: &HashMap<u32, f32>, cfg: &Config) -> Vec<f32> {
    let g = input.graph;
    let ne = g.len();
    let nt = input.turns.len();
    let n = ne + nt;

    // Reset distribution: entity activations plus dense context priors, half mass each.
    let mut reset = vec![0.0f32; n];
    let act_total: f32 = activations.values().sum();
    let priors: Vec<f32> = input
        .turn_vecs
        .iter()
        .map(|v| cosine(input.query_vec, v).max(0.0))
        .collect();
    let prior_total: f32 = priors.iter().sum();
    if act_total == 0.0 && prior_total == 0.0 {
        return vec![0.0; nt];
    }
    let (act_mass, prior_mass) = match (act_total > 0.0, prior_total > 0.0) {
        (true, true) => (0.5, 0.5),
        (true, false) => (1.0, 0.0),
        (false, true) => (0.0, 1.0),
        _ => unreachable!(),
    };
    for (&e, &a) in activations {
        reset[e as usize] = act_mass * a / act_total;
    }
    for (t, p) in priors.iter().enumerate() {
        reset[ne + t] += prior_mass * p / prior_total;
    }

    let mut pi = reset.clone();
    let mut next = vec![0.0f32; n];
    for _ in 0..cfg.ppr_iters {
        next.iter_mut().zip(&reset).for_each(|(x, r)| *x = (1.0 - cfg.gamma) * r);
        for e in 0..ne {
            let mass = pi[e];
            if mass <= 0.0 {
                continue;
            }
            let out = g.entity_weights(e as u32);
            if out.is_empty() {
                next[e] += cfg.gamma * mass;
                continue;
            }
            for (turn, w) in out {
                next[ne + turn as usize] += cfg.gamma * mass * w;
            }
        }
        for t in 0..nt {
            let mass = pi[ne + t];
            if mass <= 0.0 {
                continue;
            }
            let ents = g.turn_weights(t as u32);
            let neighbors = adjacent(t, nt, input.turns);
            let entity_share = if ents.is_empty() { 0.0 } else { 0.8 };
            let adj_share = if neighbors.is_empty() { 0.0 } else { 1.0 - entity_share };
            if ents.is_empty() && neighbors.is_empty() {
                next[ne + t] += cfg.gamma * mass;
                continue;
            }
            for (e, w) in &ents {
                next[*e as usize] += cfg.gamma * mass * entity_share * w;
            }
            let per = adj_share / neighbors.len().max(1) as f32;
            for nb in neighbors {
                next[ne + nb] += cfg.gamma * mass * per;
            }
        }
        let delta: f32 = pi.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
        std::mem::swap(&mut pi, &mut next);
        if delta < 1e-7 {
            break;
        }
    }
    pi[ne..].to_vec()
}

fn adjacent(t: usize, nt: usize, turns: &[Turn]) -> Vec<usize> {
    let mut out = Vec::with_capacity(2);
    if t > 0 && turns[t - 1].session_id == turns[t].session_id {
        out.push(t - 1);
    }
    if t + 1 < nt && turns[t + 1].session_id == turns[t].session_id {
        out.push(t + 1);
    }
    out
}
