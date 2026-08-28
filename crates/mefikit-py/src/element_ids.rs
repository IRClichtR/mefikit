use numpy as np;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList};

use mefikit::mesh::ElementIds;

use super::element::{etype_to_str, str_to_etype};

pub(crate) fn ids_to_pydict<'py>(py: Python<'py>, eids: &ElementIds) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    for (et, ids) in eids.iter_blocks() {
        let arr = np::PyArray1::from_vec(py, ids.clone());
        dict.set_item(etype_to_str(*et), arr)
            .expect("element type strings are valid dict keys");
    }
    dict
}

#[pyclass]
#[pyo3(name = "ElementIds")]
pub struct PyElementIds {
    inner: ElementIds,
}

impl From<PyElementIds> for Py<PyDict> {
    fn from(eids: PyElementIds) -> Self {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (et, ids) in eids.inner.iter_blocks() {
                let py_ids = np::PyArray1::from_vec(py, ids.clone());
                dict.set_item(etype_to_str(*et), py_ids).unwrap();
            }
            dict.into()
        })
    }
}

impl From<PyElementIds> for ElementIds {
    fn from(pyeids: PyElementIds) -> Self {
        pyeids.inner
    }
}

impl From<ElementIds> for PyElementIds {
    fn from(eids: ElementIds) -> Self {
        PyElementIds { inner: eids }
    }
}

fn extract_ids(value: &Bound<'_, PyAny>) -> Vec<usize> {
    if let Ok(arr) = value.extract::<np::PyReadonlyArray1<usize>>() {
        arr.as_array().to_vec()
    } else if let Ok(list) = value.cast::<PyList>() {
        list.iter()
            .map(|item| item.extract::<usize>().unwrap())
            .collect()
    } else if let Ok(seq) = value.extract::<Vec<usize>>() {
        seq
    } else {
        panic!("Expected a numpy array, a list, or a sequence of integers");
    }
}

impl PyElementIds {
    pub fn from_dict<'py>(dict: &Bound<'py, PyDict>) -> Self {
        let mut eids = ElementIds::new();
        for (key, value) in dict.iter() {
            let et_str: &str = key.extract().unwrap();
            let et = str_to_etype(et_str);
            let ids = extract_ids(&value);
            eids.add_block(et, ids);
        }
        PyElementIds { inner: eids }
    }
}
