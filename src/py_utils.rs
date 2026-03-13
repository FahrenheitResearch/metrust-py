use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ─── Direction / angle conversion ────────────────────────────────────

/// Convert a meteorological angle (degrees clockwise from north) to a
/// 16-point cardinal direction string.
#[pyfunction]
#[pyo3(text_signature = "(degrees)")]
fn angle_to_direction(degrees: f64) -> String {
    metrust::calc::angle_to_direction(degrees).to_string()
}

/// Parse a cardinal direction string into degrees (meteorological convention).
/// Returns None for unrecognised strings.
#[pyfunction]
#[pyo3(text_signature = "(direction)")]
fn parse_angle(direction: &str) -> Option<f64> {
    metrust::calc::parse_angle(direction)
}

// ─── Interpolation helpers ───────────────────────────────────────────

/// Find the two indices in `values` that bracket `target`.
/// Returns None if the target is outside the data range or the array is too short.
#[pyfunction]
#[pyo3(text_signature = "(values, target)")]
fn find_bounding_indices(
    values: PyReadonlyArray1<f64>,
    target: f64,
) -> Option<(usize, usize)> {
    metrust::calc::find_bounding_indices(values.as_slice().unwrap(), target)
}

/// Find the index nearest to where two series cross.
/// Returns None if no crossing is found or inputs are too short.
#[pyfunction]
#[pyo3(text_signature = "(x, y1, y2)")]
fn nearest_intersection_idx(
    x: PyReadonlyArray1<f64>,
    y1: PyReadonlyArray1<f64>,
    y2: PyReadonlyArray1<f64>,
) -> Option<usize> {
    metrust::calc::nearest_intersection_idx(
        x.as_slice().unwrap(),
        y1.as_slice().unwrap(),
        y2.as_slice().unwrap(),
    )
}

// ─── Resampling ──────────────────────────────────────────────────────

/// Nearest-neighbour 1-D resampling.
/// For each value in `x`, finds the closest point in `xp` and returns the
/// corresponding value from `fp`.
#[pyfunction]
#[pyo3(text_signature = "(x, xp, fp)")]
fn resample_nn_1d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    xp: PyReadonlyArray1<f64>,
    fp: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::resample_nn_1d(
        x.as_slice().unwrap(),
        xp.as_slice().unwrap(),
        fp.as_slice().unwrap(),
    );
    result.into_pyarray(py)
}

// ─── Module registration ─────────────────────────────────────────────

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(angle_to_direction, parent)?)?;
    parent.add_function(wrap_pyfunction!(parse_angle, parent)?)?;
    parent.add_function(wrap_pyfunction!(find_bounding_indices, parent)?)?;
    parent.add_function(wrap_pyfunction!(nearest_intersection_idx, parent)?)?;
    parent.add_function(wrap_pyfunction!(resample_nn_1d, parent)?)?;
    Ok(())
}
