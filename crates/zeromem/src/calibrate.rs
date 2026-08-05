//! Deterministic calibration, paper eq 15-16.

use crate::closure::{EvidenceRole, Selected};
use crate::config::Config;
use crate::ner::{EntityExtractor, EntityKind, HeuristicNer};
use crate::profile::{AnswerType, Boundary, QueryProfile, RecencyPref};
use crate::trace::Turn;
use std::collections::HashSet;

pub fn calibrate_evidence(
    selected: Vec<Selected>,
    turns: &[Turn],
    profile: &QueryProfile,
    session_order: &[String],
    cfg: &Config,
) -> Vec<Selected> {
    let admissible: Vec<Selected> = selected
        .into_iter()
        .filter(|s| !violates_boundary(&turns[s.turn as usize], profile, session_order))
        .collect();

    let superseded = superseded_mains(&admissible, turns, profile);

    let mut mains: Vec<&Selected> = admissible
        .iter()
        .filter(|s| s.role == EvidenceRole::Main && !superseded.contains(&s.turn))
        .collect();
    mains.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.turn.cmp(&b.turn)));
    mains.truncate(cfg.top_k);

    // Keep supports attached to surviving mains, one bridge and the nearest
    // neighbor each, total budget 2 * top_k.
    let main_ids: HashSet<u32> = mains.iter().map(|s| s.turn).collect();
    let mut out: Vec<Selected> = Vec::new();
    for m in &mains {
        out.push((*m).clone());
        let mut supports: Vec<&Selected> = admissible
            .iter()
            .filter(|s| s.role != EvidenceRole::Main && s.anchor == m.turn && !main_ids.contains(&s.turn))
            .collect();
        supports.sort_by(|a, b| {
            role_rank(a.role)
                .cmp(&role_rank(b.role))
                .then(distance(a.turn, m.turn).cmp(&distance(b.turn, m.turn)))
        });
        for s in supports.into_iter().take(2) {
            if out.len() < 2 * cfg.top_k && !out.iter().any(|x| x.turn == s.turn) {
                out.push(s.clone());
            }
        }
    }
    out
}

fn role_rank(r: EvidenceRole) -> u8 {
    match r {
        EvidenceRole::Main => 0,
        EvidenceRole::GraphBridge => 1,
        EvidenceRole::LocalNeighbor => 2,
    }
}

fn distance(a: u32, b: u32) -> u32 {
    a.abs_diff(b)
}

fn violates_boundary(turn: &Turn, profile: &QueryProfile, session_order: &[String]) -> bool {
    let Some(boundary) = profile.boundary else { return false };
    let Some(pos) = session_order.iter().position(|s| *s == turn.session_id) else {
        return true;
    };
    match boundary {
        Boundary::SessionOrdinal(k) => pos != k,
        Boundary::LastSession => pos + 1 != session_order.len(),
    }
}

/// Drops mains whose date/number value conflicts with a later-session main
/// when the query prefers the latest state.
fn superseded_mains(selected: &[Selected], turns: &[Turn], profile: &QueryProfile) -> HashSet<u32> {
    let mut out = HashSet::new();
    if !matches!(profile.answer_type, AnswerType::Time | AnswerType::Number)
        || profile.temporal.prefer != Some(RecencyPref::Latest)
    {
        return out;
    }
    let kind = if profile.answer_type == AnswerType::Time { EntityKind::Date } else { EntityKind::Number };
    let mut valued: Vec<(u32, i64, HashSet<String>)> = Vec::new();
    for s in selected.iter().filter(|s| s.role == EvidenceRole::Main) {
        let t = &turns[s.turn as usize];
        let values: HashSet<String> = HeuristicNer
            .extract(&t.text)
            .into_iter()
            .filter(|e| e.kind == kind)
            .map(|e| e.canon)
            .collect();
        if !values.is_empty() {
            valued.push((s.turn, t.ts, values));
        }
    }
    for (turn, ts, values) in &valued {
        let newer_conflict = valued.iter().any(|(t2, ts2, v2)| {
            t2 != turn && ts2 > ts && v2.is_disjoint(values)
        });
        if newer_conflict {
            out.insert(*turn);
        }
    }
    out
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibratedAnswer {
    pub answer: String,
    pub changed: bool,
    pub supported: bool,
    pub candidates: Vec<String>,
}

/// eq 16: replace an unsupported scalar only when evidence has exactly one
/// type-compatible candidate; prune list items absent from evidence.
pub fn calibrate_answer(answer: &str, profile: &QueryProfile, evidence_texts: &[&str]) -> CalibratedAnswer {
    let evidence_lower: Vec<String> = evidence_texts.iter().map(|t| t.to_lowercase()).collect();
    let candidates: Vec<String> = typed_candidates(profile.answer_type, evidence_texts)
        .into_iter()
        .filter(|c| !profile.subjects.contains(c))
        .collect();
    let answer_lower = answer.to_lowercase();

    let supported = candidates.iter().any(|c| answer_lower.contains(c.as_str()))
        || evidence_lower.iter().any(|t| {
            let a = answer_lower.trim().trim_end_matches('.');
            !a.is_empty() && t.contains(a)
        });

    if profile.answer_type == AnswerType::List {
        let items: Vec<&str> = answer
            .split(|c| c == ',' || c == ';' || c == '\n')
            .map(|s| s.trim().trim_start_matches("and ").trim())
            .filter(|s| !s.is_empty())
            .collect();
        let kept: Vec<&str> = items
            .iter()
            .copied()
            .filter(|item| {
                let il = item.to_lowercase();
                evidence_lower.iter().any(|t| t.contains(&il))
            })
            .collect();
        if !kept.is_empty() && kept.len() < items.len() {
            return CalibratedAnswer {
                answer: kept.join(", "),
                changed: true,
                supported: true,
                candidates,
            };
        }
        return CalibratedAnswer { answer: answer.into(), changed: false, supported, candidates };
    }

    if !supported && candidates.len() == 1 && scalar_type(profile.answer_type) {
        return CalibratedAnswer {
            answer: candidates[0].clone(),
            changed: true,
            supported: true,
            candidates,
        };
    }
    CalibratedAnswer { answer: answer.into(), changed: false, supported, candidates }
}

fn scalar_type(t: AnswerType) -> bool {
    matches!(t, AnswerType::Time | AnswerType::Number | AnswerType::Person | AnswerType::Place | AnswerType::Entity)
}

fn typed_candidates(answer_type: AnswerType, evidence_texts: &[&str]) -> Vec<String> {
    let want: &[EntityKind] = match answer_type {
        AnswerType::Time => &[EntityKind::Date],
        AnswerType::Number => &[EntityKind::Number],
        AnswerType::Person | AnswerType::Place | AnswerType::Entity => &[EntityKind::Named],
        AnswerType::List => &[EntityKind::Named, EntityKind::Number, EntityKind::Date],
        _ => return Vec::new(),
    };
    let mut seen = HashSet::new();
    for text in evidence_texts {
        for e in HeuristicNer.extract(text) {
            if want.contains(&e.kind) {
                seen.insert(e.canon);
            }
        }
    }
    let mut out: Vec<String> = seen.into_iter().collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::HeuristicNer as N;
    use crate::profile::build_profile;

    #[test]
    fn unsupported_scalar_replaced_by_unique_candidate() {
        let p = build_profile("When did Carrie move to Jersey City?", &N);
        let out = calibrate_answer("June 2021", &p, &["Carrie moved to Jersey City on February 14, 2022."]);
        assert!(out.changed);
        assert_eq!(out.answer, "february 14, 2022");
    }

    #[test]
    fn supported_answer_kept() {
        let p = build_profile("When did Carrie move to Jersey City?", &N);
        let out = calibrate_answer("February 14, 2022", &p, &["Carrie moved to Jersey City on February 14, 2022."]);
        assert!(!out.changed);
        assert!(out.supported);
    }

    #[test]
    fn list_pruned_to_evidence() {
        let p = build_profile("List all the pets Carrie has.", &N);
        let out = calibrate_answer(
            "Lychee, Mochi, Rex",
            &p,
            &["Carrie has a dog Lychee.", "She adopted a cat named Mochi."],
        );
        assert!(out.changed);
        assert_eq!(out.answer, "Lychee, Mochi");
    }

    #[test]
    fn subject_never_becomes_the_answer() {
        let p = build_profile("Who did Carrie meet at the market?", &N);
        let out = calibrate_answer("someone", &p, &["Carrie met Panat at the market."]);
        assert!(out.changed, "{out:?}");
        assert_eq!(out.answer, "panat");
    }

    #[test]
    fn ambiguous_candidates_leave_answer_alone() {
        let p = build_profile("When did Carrie move?", &N);
        let out = calibrate_answer("sometime in spring", &p, &["She moved February 14, 2022.", "Or was it June 2, 2023?"]);
        assert!(!out.changed);
        assert!(!out.supported);
    }
}
