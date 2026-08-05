//! Routing, paper eq 7. Both views always run; the route picks the primary.

use crate::profile::{AnswerType, QueryProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Route {
    Relational,
    Local,
}

pub fn route(profile: &QueryProfile) -> Route {
    let temporal = !profile.temporal.ranges.is_empty()
        || !profile.temporal.mentions.is_empty()
        || profile.answer_type == AnswerType::Time;
    let state_like = profile.boundary.is_some() || temporal;

    if state_like && !profile.aggregation {
        return Route::Local;
    }
    if !profile.subjects.is_empty() {
        return Route::Relational;
    }
    Route::Local
}

/// (graph weight, hierarchy weight) given the shared coefficient rho.
pub fn view_weights(route: Route, rho: f32) -> (f32, f32) {
    match route {
        Route::Relational => (rho, 1.0 - rho),
        Route::Local => (1.0 - rho, rho),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::HeuristicNer;
    use crate::profile::build_profile;

    #[test]
    fn entity_question_routes_relational() {
        let p = build_profile("Who is Panat's roommate?", &HeuristicNer);
        assert_eq!(route(&p), Route::Relational);
    }

    #[test]
    fn temporal_question_routes_local() {
        let p = build_profile("When did Carrie adopt the dog?", &HeuristicNer);
        assert_eq!(route(&p), Route::Local);
    }

    #[test]
    fn weights_flip_with_route() {
        let (g, h) = view_weights(Route::Relational, 0.6);
        assert!((g - 0.6).abs() < 1e-6 && (h - 0.4).abs() < 1e-6);
        let (g, h) = view_weights(Route::Local, 0.6);
        assert!((g - 0.4).abs() < 1e-6 && (h - 0.6).abs() < 1e-6);
    }
}
