//! Entity-context graph (paper eq 3-4). Nodes are entities and turns; edges are
//! observed co-occurrence (entity appears in turn) and turn adjacency. Weights
//! are occurrence counts normalized per turn: w(d,e) = c(e,d) / sum_e' c(e',d).

use crate::ner::{Entity, EntityKind};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EntityNode {
    pub canon: String,
    pub display: String,
    pub kind: EntityKind,
}

#[derive(Default)]
pub struct EntityGraph {
    pub entities: Vec<EntityNode>,
    by_canon: HashMap<String, u32>,
    /// entity -> [(turn, count)]
    pub postings: Vec<Vec<(u32, f32)>>,
    /// turn -> [(entity, count)]
    pub turn_entities: Vec<Vec<(u32, f32)>>,
}

impl EntityGraph {
    pub fn entity_id(&self, canon: &str) -> Option<u32> {
        self.by_canon.get(canon).copied()
    }

    /// Registers a turn's extracted entities. Returns ids of entities new to the graph.
    pub fn add_turn(&mut self, turn: u32, extracted: &[Entity]) -> Vec<u32> {
        debug_assert_eq!(turn as usize, self.turn_entities.len());
        let mut counts: HashMap<u32, f32> = HashMap::new();
        let mut fresh = Vec::new();
        for e in extracted {
            let id = match self.by_canon.get(&e.canon) {
                Some(&id) => id,
                None => {
                    let id = self.entities.len() as u32;
                    self.by_canon.insert(e.canon.clone(), id);
                    self.entities.push(EntityNode {
                        canon: e.canon.clone(),
                        display: e.display.clone(),
                        kind: e.kind,
                    });
                    self.postings.push(Vec::new());
                    fresh.push(id);
                    id
                }
            };
            *counts.entry(id).or_default() += 1.0;
        }
        let mut row: Vec<(u32, f32)> = counts.into_iter().collect();
        row.sort_by_key(|(id, _)| *id);
        for (id, c) in &row {
            self.postings[*id as usize].push((turn, *c));
        }
        self.turn_entities.push(row);
        fresh
    }

    /// w(d,e) rows for one turn.
    pub fn turn_weights(&self, turn: u32) -> Vec<(u32, f32)> {
        let row = &self.turn_entities[turn as usize];
        let total: f32 = row.iter().map(|(_, c)| c).sum();
        if total == 0.0 {
            return Vec::new();
        }
        row.iter().map(|(e, c)| (*e, c / total)).collect()
    }

    /// Turns containing an entity, count-weighted and normalized over the posting.
    pub fn entity_weights(&self, entity: u32) -> Vec<(u32, f32)> {
        let posting = &self.postings[entity as usize];
        let total: f32 = posting.iter().map(|(_, c)| c).sum();
        if total == 0.0 {
            return Vec::new();
        }
        posting.iter().map(|(d, c)| (*d, c / total)).collect()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::{EntityExtractor, HeuristicNer};

    #[test]
    fn weights_normalize_per_turn() {
        let mut g = EntityGraph::default();
        let ents = HeuristicNer.extract("Carrie met Carrie's friend Panat in Jersey City.");
        g.add_turn(0, &ents);
        let w = g.turn_weights(0);
        let total: f32 = w.iter().map(|(_, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-6);
        let carrie = g.entity_id("carrie").unwrap();
        let panat = g.entity_id("panat").unwrap();
        let get = |id: u32| w.iter().find(|(e, _)| *e == id).unwrap().1;
        assert!(get(carrie) > get(panat));
    }
}
