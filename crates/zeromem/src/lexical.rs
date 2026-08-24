//! BM25 over turns, plus exact-phrase matching.

use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Default)]
pub struct Bm25 {
    term_ids: HashMap<String, u32>,
    /// term -> [(doc, tf)]
    postings: Vec<Vec<(u32, f32)>>,
    doc_len: Vec<f32>,
    total_len: f32,
}

impl Bm25 {
    pub fn add_doc(&mut self, tokens: &[String]) {
        let doc = self.doc_len.len() as u32;
        let mut tf: HashMap<u32, f32> = HashMap::new();
        for t in tokens {
            let id = match self.term_ids.get(t) {
                Some(&id) => id,
                None => {
                    let id = self.postings.len() as u32;
                    self.term_ids.insert(t.clone(), id);
                    self.postings.push(Vec::new());
                    id
                }
            };
            *tf.entry(id).or_default() += 1.0;
        }
        for (id, f) in tf {
            self.postings[id as usize].push((doc, f));
        }
        self.doc_len.push(tokens.len() as f32);
        self.total_len += tokens.len() as f32;
    }

    /// Sparse scores over all docs matching at least one query term.
    pub fn scores(&self, query_tokens: &[String]) -> HashMap<u32, f32> {
        let n = self.doc_len.len() as f32;
        if n == 0.0 {
            return HashMap::new();
        }
        let avgdl = self.total_len / n;
        let mut out: HashMap<u32, f32> = HashMap::new();
        for t in query_tokens {
            let Some(&id) = self.term_ids.get(t) else {
                continue;
            };
            let posting = &self.postings[id as usize];
            let df = posting.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in posting {
                let dl = self.doc_len[doc as usize];
                let s = idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl));
                *out.entry(doc).or_default() += s;
            }
        }
        out
    }
}

/// Count of query phrases (names, dates, numbers, quotes) appearing verbatim in text.
pub fn phrase_matches(phrases: &[String], text: &str) -> usize {
    let lower = text.to_lowercase();
    phrases
        .iter()
        .filter(|p| !p.is_empty() && lower.contains(p.as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::tokenize;

    #[test]
    fn rare_term_outranks_common() {
        let mut idx = Bm25::default();
        idx.add_doc(&tokenize("the cat sat on the mat"));
        idx.add_doc(&tokenize("the dog chased the zeppelin"));
        idx.add_doc(&tokenize("the cat and the dog"));
        let s = idx.scores(&tokenize("zeppelin"));
        assert_eq!(s.len(), 1);
        assert!(s.contains_key(&1));
    }

    #[test]
    fn phrase_match_case_insensitive() {
        assert_eq!(
            phrase_matches(&["blue bottle".into()], "at Blue Bottle today"),
            1
        );
    }
}
