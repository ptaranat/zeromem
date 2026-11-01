use crate::error::Result;

pub trait Embedder: Send + Sync {
    /// Cache namespace; changing it invalidates stored vectors.
    fn id(&self) -> String;
    fn dim(&self) -> usize;
    /// Embeds stored content (turns, windows, entity contexts).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embeds a search query. Models trained with asymmetric query and
    /// passage prefixes override this.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[query])?.pop().unwrap_or_default())
    }
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

/// Dense models a store can embed with. `id` is the tag stored with each
/// vector, so it must never change for an existing model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DenseModel {
    #[default]
    BgeSmallEnV15,
    MultilingualE5Small,
}

impl DenseModel {
    pub const ALL: [DenseModel; 2] = [DenseModel::BgeSmallEnV15, DenseModel::MultilingualE5Small];

    /// The name hosts configure.
    pub fn name(self) -> &'static str {
        match self {
            DenseModel::BgeSmallEnV15 => "bge-small-en-v1.5",
            DenseModel::MultilingualE5Small => "multilingual-e5-small",
        }
    }

    /// The tag stored with each vector. e5 stores written without its
    /// prefixes already exist under the bare name, so the prefixed vectors
    /// get their own tag rather than mixing with them.
    pub fn id(self) -> &'static str {
        match self {
            DenseModel::BgeSmallEnV15 => "bge-small-en-v1.5",
            DenseModel::MultilingualE5Small => "multilingual-e5-small+prefix",
        }
    }

    /// (query, passage) prefixes the model was trained with.
    pub fn prefixes(self) -> (&'static str, &'static str) {
        match self {
            DenseModel::BgeSmallEnV15 => ("", ""),
            DenseModel::MultilingualE5Small => ("query: ", "passage: "),
        }
    }
}

impl std::str::FromStr for DenseModel {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| {
                let known: Vec<&str> = Self::ALL.iter().map(|m| m.name()).collect();
                crate::error::Error::Invalid(format!(
                    "unknown embedding model {s:?}, expected one of: {}",
                    known.join(", ")
                ))
            })
    }
}

#[cfg(feature = "fastembed")]
pub use fast::FastEmbedder;

#[cfg(feature = "fastembed")]
mod fast {
    use super::*;
    use crate::error::Error;
    use std::sync::Mutex;

    /// A [`DenseModel`] via fastembed/onnxruntime. Downloads the model into
    /// `cache_dir` on first use.
    pub struct FastEmbedder {
        model: Mutex<fastembed::TextEmbedding>,
        kind: DenseModel,
        dim: usize,
    }

    impl FastEmbedder {
        pub fn new(cache_dir: &std::path::Path, kind: DenseModel) -> Result<Self> {
            let fastembed_model = match kind {
                DenseModel::BgeSmallEnV15 => fastembed::EmbeddingModel::BGESmallENV15,
                DenseModel::MultilingualE5Small => fastembed::EmbeddingModel::MultilingualE5Small,
            };
            let opts = fastembed::InitOptions::new(fastembed_model)
                .with_cache_dir(cache_dir.to_path_buf());
            let model =
                fastembed::TextEmbedding::try_new(opts).map_err(|e| Error::Embed(e.to_string()))?;
            Ok(Self {
                model: Mutex::new(model),
                kind,
                dim: 384,
            })
        }
    }

    impl Embedder for FastEmbedder {
        fn id(&self) -> String {
            self.kind.id().into()
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.embed_prefixed(self.kind.prefixes().1, texts)
        }

        fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
            Ok(self
                .embed_prefixed(self.kind.prefixes().0, &[query])?
                .pop()
                .unwrap_or_default())
        }
    }

    impl FastEmbedder {
        fn embed_prefixed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
            let model = self.model.lock().unwrap();
            let mut vecs = model
                .embed(inputs, None)
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
    fn dense_model_ids_round_trip() {
        for m in DenseModel::ALL {
            assert_eq!(m.name().parse::<DenseModel>().unwrap(), m);
        }
        assert_eq!(DenseModel::default().id(), "bge-small-en-v1.5");
        let err = "e5".parse::<DenseModel>().unwrap_err().to_string();
        assert!(err.contains("multilingual-e5-small"), "{err}");
    }

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
