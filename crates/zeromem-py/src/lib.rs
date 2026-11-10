use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::path::PathBuf;
use std::sync::Mutex;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => b.into_bound_py_any(py),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => i.into_bound_py_any(py),
            None => n.as_f64().unwrap_or(0.0).into_bound_py_any(py),
        },
        serde_json::Value::String(s) => s.into_bound_py_any(py),
        serde_json::Value::Array(a) => a
            .iter()
            .map(|v| to_py(py, v))
            .collect::<PyResult<Vec<_>>>()?
            .into_bound_py_any(py),
        serde_json::Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, v) in o {
                d.set_item(k, to_py(py, v)?)?;
            }
            Ok(d.into_any())
        }
    }
}

fn json_out<'py, T: serde::Serialize>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyAny>> {
    let v = serde_json::to_value(value).map_err(err)?;
    to_py(py, &v)
}

/// Zero-Mem store over a single SQLite file.
#[pyclass]
struct ZeroMem {
    inner: Mutex<zm::ZeroMem>,
}

#[pymethods]
impl ZeroMem {
    /// open(path, use_model=True, model_cache_dir=None, model=None)
    ///
    /// model: "bge-small-en-v1.5" (default) or "multilingual-e5-small".
    #[new]
    #[pyo3(signature = (path, use_model = true, model_cache_dir = None, model = None))]
    fn new(
        path: &str,
        use_model: bool,
        model_cache_dir: Option<&str>,
        model: Option<&str>,
    ) -> PyResult<Self> {
        let model: zm::embed::DenseModel = match model {
            Some(name) => name
                .parse()
                .map_err(|e: zm::error::Error| PyValueError::new_err(e.to_string()))?,
            None => Default::default(),
        };
        let embedder = if use_model {
            zm::dense_embedder(model, model_cache_dir.map(std::path::Path::new))
        } else {
            Box::new(zm::embed::HashEmbedder::default()) as Box<dyn zm::embed::Embedder>
        };
        let inner = zm::ZeroMem::open(&PathBuf::from(path), zm::config::Config::default(), embedder)
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

    /// delete_session(session_id) -> number of turns removed. Cascades:
    /// rebuilds derived state and sweeps unreferenced embedding-cache rows.
    fn delete_session(&self, session_id: &str) -> PyResult<usize> {
        self.inner.lock().unwrap().delete_session(session_id).map_err(err)
    }

    /// query(text, top_k=None, exclude_session=None) -> {"route": ..., "evidence": [...]}
    /// exclude_session drops that session from results and from session-boundary
    /// counting, so "last session" means the one before it.
    #[pyo3(signature = (text, top_k = None, exclude_session = None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        top_k: Option<usize>,
        exclude_session: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut inner = self.inner.lock().unwrap();
        inner.exclude_session(exclude_session);
        let result = inner.query(text, top_k);
        inner.exclude_session(None);
        json_out(py, &result.map_err(err)?)
    }

    /// calibrate_answer(query, answer, evidence_texts) -> {"answer", "changed", "supported", "candidates"}
    fn calibrate_answer<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        answer: &str,
        evidence_texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let refs: Vec<&str> = evidence_texts.iter().map(String::as_str).collect();
        let out = self.inner.lock().unwrap().calibrate_answer(query, answer, &refs);
        json_out(py, &out)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self.inner.lock().unwrap().stats();
        json_out(py, &s)
    }
}

#[pymodule(name = "zeromem")]
fn zeromem_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZeroMem>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
