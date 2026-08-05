use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::Mutex;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_py(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else {
                n.as_f64().unwrap_or(0.0).into_py(py)
            }
        }
        serde_json::Value::String(s) => s.into_py(py),
        serde_json::Value::Array(a) => {
            let items: Vec<PyObject> = a.iter().map(|v| to_py(py, v)).collect();
            items.into_py(py)
        }
        serde_json::Value::Object(o) => {
            let d = PyDict::new_bound(py);
            for (k, v) in o {
                let _ = d.set_item(k, to_py(py, v));
            }
            d.into_py(py)
        }
    }
}

fn json_out<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<PyObject> {
    let v = serde_json::to_value(value).map_err(err)?;
    Ok(to_py(py, &v))
}

/// Zero-Mem memory store. All operations are token-free; no LLM is called.
#[pyclass]
struct ZeroMem {
    inner: Mutex<zeromem::ZeroMem>,
}

#[pymethods]
impl ZeroMem {
    /// open(path, use_model=True, model_cache_dir=None)
    #[new]
    #[pyo3(signature = (path, use_model = true, model_cache_dir = None))]
    fn new(path: &str, use_model: bool, model_cache_dir: Option<&str>) -> PyResult<Self> {
        let embedder = if use_model {
            zeromem::default_embedder(model_cache_dir.map(std::path::Path::new))
        } else {
            Box::new(zeromem::embed::HashEmbedder::default()) as Box<dyn zeromem::embed::Embedder>
        };
        let inner = zeromem::ZeroMem::open(&PathBuf::from(path), zeromem::config::Config::default(), embedder)
            .map_err(err)?;
        Ok(Self { inner: Mutex::new(inner) })
    }

    #[pyo3(signature = (session_id, speaker, text, ts))]
    fn ingest_turn(&self, session_id: &str, speaker: &str, text: &str, ts: i64) -> PyResult<i64> {
        self.inner
            .lock()
            .unwrap()
            .ingest_turn(session_id, speaker, text, ts)
            .map_err(err)
    }

    /// query(text, top_k=None) -> {"route": ..., "evidence": [...]}
    #[pyo3(signature = (text, top_k = None))]
    fn query(&self, py: Python<'_>, text: &str, top_k: Option<usize>) -> PyResult<PyObject> {
        let result = self.inner.lock().unwrap().query(text, top_k).map_err(err)?;
        json_out(py, &result)
    }

    /// calibrate_answer(query, answer, evidence_texts) -> {"answer", "changed", "supported", "candidates"}
    fn calibrate_answer(
        &self,
        py: Python<'_>,
        query: &str,
        answer: &str,
        evidence_texts: Vec<String>,
    ) -> PyResult<PyObject> {
        let refs: Vec<&str> = evidence_texts.iter().map(String::as_str).collect();
        let out = self.inner.lock().unwrap().calibrate_answer(query, answer, &refs);
        json_out(py, &out)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.lock().unwrap().stats();
        json_out(py, &s)
    }
}

#[pymodule]
fn zeromem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZeroMem>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
