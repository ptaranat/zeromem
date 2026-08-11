//! zeromem: zero-token memory operations for LLM agents.
//!
//! Implementation of Zero-Mem (Xiao et al., arXiv 2607.29377). Interaction
//! traces are the source of record; retrieval runs over an entity-context
//! graph and a temporal hierarchy with no LLM calls anywhere in the pipeline.

pub mod calibrate;
pub mod closure;
pub mod config;
pub mod embed;
pub mod error;
pub mod fuse;
pub mod graph;
pub mod graph_view;
pub mod hier_view;
pub mod hierarchy;
pub mod lexical;
pub mod ner;
pub mod profile;
pub mod route;
pub mod store;
pub mod text;
pub mod trace;

use crate::closure::{EvidenceRole, Selected};
use crate::config::Config;
use crate::embed::Embedder;
use crate::error::Result;
use crate::graph::EntityGraph;
use crate::hierarchy::Hierarchy;
use crate::lexical::Bm25;
use crate::ner::EntityExtractor;
use crate::profile::QueryProfile;
use crate::route::Route;
use crate::store::Store;
use crate::trace::Turn;
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Evidence {
    pub turn_id: i64,
    pub session_id: String,
    pub session_turn: i64,
    pub speaker: String,
    pub text: String,
    pub ts: i64,
    pub score: f32,
    pub role: EvidenceRole,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryResult {
    pub route: Route,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub turns: usize,
    pub sessions: usize,
    pub entities: usize,
    pub windows: usize,
    pub episodes: usize,
    pub embedder: String,
}

pub struct ZeroMem {
    cfg: Config,
    store: Store,
    embedder: Box<dyn Embedder>,
    ner: Box<dyn EntityExtractor>,
    turns: Vec<Turn>,
    turn_vecs: Vec<Vec<f32>>,
    entity_vecs: Vec<Vec<f32>>,
    graph: EntityGraph,
    hier: Hierarchy,
    bm25: Bm25,
    session_order: Vec<String>,
}

impl ZeroMem {
    pub fn open(path: &Path, cfg: Config, embedder: Box<dyn Embedder>) -> Result<Self> {
        Self::build(Store::open(path)?, cfg, embedder)
    }

    pub fn open_in_memory(cfg: Config, embedder: Box<dyn Embedder>) -> Result<Self> {
        Self::build(Store::open_in_memory()?, cfg, embedder)
    }

    fn build(store: Store, cfg: Config, embedder: Box<dyn Embedder>) -> Result<Self> {
        // Cached vectors are only valid for the embedder that produced them.
        let tag = embedder.id();
        match store.meta("embedder")? {
            Some(prev) if prev != tag => {
                store.clear_embeddings()?;
                store.set_meta("embedder", &tag)?;
            }
            None => store.set_meta("embedder", &tag)?,
            _ => {}
        }

        let mut zm = Self {
            cfg,
            store,
            embedder,
            ner: Box::new(ner::HeuristicNer),
            turns: Vec::new(),
            turn_vecs: Vec::new(),
            entity_vecs: Vec::new(),
            graph: EntityGraph::default(),
            hier: Hierarchy::default(),
            bm25: Bm25::default(),
            session_order: Vec::new(),
        };
        for turn in zm.store.load_turns()? {
            zm.index_turn(turn)?;
        }
        Ok(zm)
    }

    pub fn ingest_turn(&mut self, session_id: &str, speaker: &str, text: &str, ts: i64) -> Result<i64> {
        if text.trim().is_empty() {
            return Err(error::Error::Invalid("empty turn text".into()));
        }
        let mut turn = Turn {
            id: 0,
            session_id: session_id.to_string(),
            session_turn: self.store.next_session_turn(session_id)?,
            speaker: speaker.to_string(),
            text: text.to_string(),
            ts,
        };
        turn.id = self.store.insert_turn(&turn)?;
        self.index_turn(turn.clone())?;
        Ok(turn.id)
    }

    /// Deletes a session's turns, rebuilds the derived state from the
    /// surviving turns, and drops embedding-cache rows nothing references
    /// anymore. Returns the number of turns removed.
    pub fn delete_session(&mut self, session_id: &str) -> Result<usize> {
        let removed = self.store.delete_session_turns(session_id)?;
        if removed == 0 {
            return Ok(0);
        }
        self.rebuild()?;
        self.sweep_embedding_cache()?;
        Ok(removed)
    }

    fn rebuild(&mut self) -> Result<()> {
        self.turns.clear();
        self.turn_vecs.clear();
        self.entity_vecs.clear();
        self.graph = EntityGraph::default();
        self.hier = Hierarchy::default();
        self.bm25 = Bm25::default();
        self.session_order.clear();
        for turn in self.store.load_turns()? {
            self.index_turn(turn)?;
        }
        Ok(())
    }

    /// Orphaned cache rows are never loaded (indexing only reads keys for
    /// live turns and entities), so this reclaims space rather than fixing
    /// retrieval; a crash mid-sweep leaves nothing worse than unswept rows.
    fn sweep_embedding_cache(&self) -> Result<()> {
        let live: std::collections::HashSet<String> =
            self.turns.iter().map(|t| t.id.to_string()).collect();
        for key in self.store.embedding_keys("turn")? {
            if !live.contains(&key) {
                self.store.delete_embedding("turn", &key)?;
            }
        }
        let live: std::collections::HashSet<&str> =
            self.graph.entities.iter().map(|e| e.canon.as_str()).collect();
        for key in self.store.embedding_keys("entity")? {
            if !live.contains(key.as_str()) {
                self.store.delete_embedding("entity", &key)?;
            }
        }
        Ok(())
    }

    fn index_turn(&mut self, turn: Turn) -> Result<()> {
        let idx = self.turns.len() as u32;
        let vec = self.embedding("turn", &turn.id.to_string(), &turn.text)?;

        let entities = self.ner.extract(&turn.text);
        let fresh = self.graph.add_turn(idx, &entities);
        for id in fresh {
            let node = self.graph.entities[id as usize].clone();
            let v = self.embedding("entity", &node.canon, &node.display)?;
            debug_assert_eq!(self.entity_vecs.len(), id as usize);
            self.entity_vecs.push(v);
        }

        self.hier.push_turn(idx, &turn.session_id, turn.ts, &vec, &self.cfg);
        self.bm25.add_doc(&text::tokenize(&turn.text));
        if !self.session_order.contains(&turn.session_id) {
            self.session_order.push(turn.session_id.clone());
        }
        self.turn_vecs.push(vec);
        self.turns.push(turn);
        Ok(())
    }

    fn embedding(&self, kind: &str, key: &str, content: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.store.embedding(kind, key)? {
            if v.len() == self.embedder.dim() {
                return Ok(v);
            }
        }
        let v = self.embedder.embed(&[content])?.pop().unwrap_or_default();
        self.store.put_embedding(kind, key, &v)?;
        Ok(v)
    }

    pub fn profile(&self, query: &str) -> QueryProfile {
        profile::build_profile(query, self.ner.as_ref())
    }

    pub fn query(&self, query: &str, top_k: Option<usize>) -> Result<QueryResult> {
        let mut cfg = self.cfg.clone();
        if let Some(k) = top_k {
            cfg.top_k = k;
        }
        let profile = self.profile(query);
        let route = route::route(&profile);
        let (gw, hw) = route::view_weights(route, cfg.rho);

        if self.turns.is_empty() {
            return Ok(QueryResult { route, evidence: Vec::new() });
        }
        let query_vec = self.embedder.embed(&[query])?.pop().unwrap_or_default();

        let graph_scores = graph_view::retrieve(
            &graph_view::GraphViewInput {
                graph: &self.graph,
                turns: &self.turns,
                turn_vecs: &self.turn_vecs,
                entity_vecs: &self.entity_vecs,
                query_vec: &query_vec,
                profile: &profile,
            },
            &cfg,
        );
        let hier_scores = hier_view::retrieve(
            &hier_view::HierViewInput {
                hier: &self.hier,
                graph: &self.graph,
                turns: &self.turns,
                turn_vecs: &self.turn_vecs,
                query_vec: &query_vec,
                bm25: &self.bm25,
                profile: &profile,
                session_order: &self.session_order,
            },
            &cfg,
        );

        let main = fuse::fuse(&graph_scores, &hier_scores, gw, hw, cfg.top_k);
        let closed = closure::close(&main, &self.graph, &self.turns, &cfg);
        let final_set = calibrate::calibrate_evidence(closed, &self.turns, &profile, &self.session_order, &cfg);

        let evidence = final_set
            .into_iter()
            .map(|s: Selected| {
                let t = &self.turns[s.turn as usize];
                Evidence {
                    turn_id: t.id,
                    session_id: t.session_id.clone(),
                    session_turn: t.session_turn,
                    speaker: t.speaker.clone(),
                    text: t.text.clone(),
                    ts: t.ts,
                    score: s.score,
                    role: s.role,
                }
            })
            .collect();
        Ok(QueryResult { route, evidence })
    }

    pub fn calibrate_answer(&self, query: &str, answer: &str, evidence_texts: &[&str]) -> calibrate::CalibratedAnswer {
        let profile = self.profile(query);
        calibrate::calibrate_answer(answer, &profile, evidence_texts)
    }

    pub fn stats(&self) -> Stats {
        Stats {
            turns: self.turns.len(),
            sessions: self.session_order.len(),
            entities: self.graph.len(),
            windows: self.hier.windows.len(),
            episodes: self.hier.episodes.len(),
            embedder: self.embedder.id(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;

    #[test]
    fn delete_session_cascades_to_cache() {
        let mut zm =
            ZeroMem::open_in_memory(Config::default(), Box::new(HashEmbedder::default())).unwrap();
        zm.ingest_turn("keep", "user", "Carrie adopted Lychee in Jersey City.", 1000).unwrap();
        zm.ingest_turn("drop", "user", "Slowdive played the Fillmore with MBV.", 2000).unwrap();

        assert_eq!(zm.delete_session("drop").unwrap(), 1);
        assert_eq!(zm.stats().turns, 1);
        assert_eq!(zm.stats().sessions, 1);

        assert_eq!(zm.store.embedding_keys("turn").unwrap().len(), 1);
        let entity_keys = zm.store.embedding_keys("entity").unwrap();
        assert_eq!(entity_keys.len(), zm.graph.len(), "{entity_keys:?}");
        assert!(!entity_keys.contains(&"slowdive".to_string()), "{entity_keys:?}");

        assert_eq!(zm.delete_session("drop").unwrap(), 0);
    }
}

/// Default embedder: fastembed when compiled in and loadable, hash otherwise.
pub fn default_embedder(cache_dir: Option<&Path>) -> Box<dyn Embedder> {
    #[cfg(feature = "fastembed")]
    {
        let dir = cache_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::temp_dir().join("zeromem-models"));
        match embed::FastEmbedder::new(&dir) {
            Ok(e) => return Box::new(e),
            Err(err) => eprintln!("zeromem: fastembed unavailable ({err}), falling back to hash embedder"),
        }
    }
    let _ = cache_dir;
    Box::new(embed::HashEmbedder::default())
}
