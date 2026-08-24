use crate::error::Result;

pub trait Embedder: Send + Sync {
    /// Cache namespace; changing it invalidates stored vectors.
    fn id(&self) -> String;
    fn dim(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// True for lexical-only fallback embedders.
    fn is_fallback(&self) -> bool {
        false
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Signed feature hashing over tokens and char trigrams. Offline fallback and
/// test embedder; real deployments use FastEmbedder.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn bump(&self, v: &mut [f32], feature: &str, weight: f32) {
        let h = fnv1a(feature.as_bytes());
        let idx = (h % self.dim as u64) as usize;
        let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign * weight;
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl Embedder for HashEmbedder {
    fn id(&self) -> String {
        format!("hash-v1-{}", self.dim)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_fallback(&self) -> bool {
        true
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; self.dim];
                for tok in crate::text::tokenize(text) {
                    let w = if crate::text::is_stopword(&tok) {
                        0.2
                    } else {
                        1.0
                    };
                    self.bump(&mut v, &tok, w);
                    let padded: Vec<char> = format!("^{tok}$").chars().collect();
                    for tri in padded.windows(3) {
                        let tri: String = tri.iter().collect();
                        self.bump(&mut v, &tri, 0.4 * w);
                    }
                }
                l2_normalize(&mut v);
                v
            })
            .collect())
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(feature = "fastembed")]
pub use fast::FastEmbedder;

#[cfg(feature = "fastembed")]
mod fast {
    use super::*;
    use crate::error::Error;
    use std::sync::Mutex;

    /// bge-small-en-v1.5 via fastembed/onnxruntime. Downloads the model into
    /// `cache_dir` on first use.
    pub struct FastEmbedder {
        model: Mutex<fastembed::TextEmbedding>,
        dim: usize,
    }

    impl FastEmbedder {
        pub fn new(cache_dir: &std::path::Path) -> Result<Self> {
            let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir.to_path_buf());
            let model =
                fastembed::TextEmbedding::try_new(opts).map_err(|e| Error::Embed(e.to_string()))?;
            Ok(Self {
                model: Mutex::new(model),
                dim: 384,
            })
        }
    }

    impl Embedder for FastEmbedder {
        fn id(&self) -> String {
            "bge-small-en-v1.5".into()
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let model = self.model.lock().unwrap();
            let mut vecs = model
                .embed(texts.to_vec(), None)
                .map_err(|e| Error::Embed(e.to_string()))?;
            vecs.iter_mut().for_each(|v| l2_normalize(v));
            Ok(vecs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_similarity_ordering() {
        let e = HashEmbedder::default();
        let vs = e
            .embed(&[
                "the dog barked at the mailman",
                "a dog was barking loudly",
                "quarterly revenue projections",
            ])
            .unwrap();
        let close = cosine(&vs[0], &vs[1]);
        let far = cosine(&vs[0], &vs[2]);
        assert!(close > far, "close={close} far={far}");
    }

    #[test]
    fn deterministic() {
        let e = HashEmbedder::default();
        let a = e.embed(&["hello world"]).unwrap();
        let b = e.embed(&["hello world"]).unwrap();
        assert_eq!(a, b);
    }
}
