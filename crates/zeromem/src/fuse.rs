//! Fusion, paper eq 12-13: per-view min-max over own candidates, absent
//! candidates score 0, weighted sum.

use std::collections::HashMap;

pub fn min_max_normalize(scores: &HashMap<u32, f32>) -> HashMap<u32, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let min = scores.values().cloned().fold(f32::INFINITY, f32::min);
    let max = scores.values().cloned().fold(f32::NEG_INFINITY, f32::max);
    scores
        .iter()
        .map(|(&d, &s)| {
            (
                d,
                if max > min {
                    (s - min) / (max - min)
                } else {
                    1.0
                },
            )
        })
        .collect()
}

/// Returns fused candidates sorted descending, truncated to `keep`.
pub fn fuse(
    graph: &HashMap<u32, f32>,
    hier: &HashMap<u32, f32>,
    graph_weight: f32,
    hier_weight: f32,
    keep: usize,
) -> Vec<(u32, f32)> {
    let g = min_max_normalize(graph);
    let h = min_max_normalize(hier);
    let mut fused: HashMap<u32, f32> = HashMap::new();
    for (&d, &s) in &g {
        *fused.entry(d).or_default() += graph_weight * s;
    }
    for (&d, &s) in &h {
        *fused.entry(d).or_default() += hier_weight * s;
    }
    let mut out: Vec<(u32, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(keep);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_candidates_score_zero_in_that_view() {
        let g = HashMap::from([(0, 10.0), (1, 5.0)]);
        let h = HashMap::from([(1, 2.0), (2, 1.0)]);
        let fused = fuse(&g, &h, 0.6, 0.4, 10);
        let score = |d: u32| fused.iter().find(|(x, _)| *x == d).unwrap().1;
        assert!((score(0) - 0.6).abs() < 1e-6);
        assert!((score(1) - 0.4).abs() < 1e-6); // min of g (0.0) + max of h (0.4)
        assert!((score(2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn single_candidate_normalizes_to_one() {
        let g = HashMap::from([(3, 0.2)]);
        let n = min_max_normalize(&g);
        assert_eq!(n[&3], 1.0);
    }
}
