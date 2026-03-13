use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ─── Direction / angle conversion ────────────────────────────────────

/// Convert a meteorological angle (degrees clockwise from north) to a
/// cardinal direction string.
///
/// Parameters
/// ----------
/// degrees : float
///     Angle in degrees clockwise from north.
/// level : int, optional
///     Number of compass points: 8, 16 (default), or 32.
/// full : bool, optional
///     If True, return full word names (e.g. "North" instead of "N").
#[pyfunction]
#[pyo3(signature = (degrees, level=16, full=false))]
fn angle_to_direction(degrees: f64, level: u32, full: bool) -> String {
    metrust::calc::angle_to_direction_ext(degrees, level, full).to_string()
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

// ─── Peak detection ─────────────────────────────────────────────────

/// Find peaks (or troughs) in a 1-D array, filtered by IQR.
///
/// Parameters
/// ----------
/// data : array of float64
///     1-D data array to search for peaks.
/// maxima : bool, optional
///     If True (default), find maxima; if False, find minima.
/// iqr_ratio : float, optional
///     Only keep peaks that stand out by at least iqr_ratio * IQR
///     above (or below) the median. Default 0.0 (no filtering).
///
/// Returns
/// -------
/// numpy.ndarray of int
///     Indices of the qualifying peaks.
#[pyfunction]
#[pyo3(signature = (data, maxima=true, iqr_ratio=0.0))]
fn find_peaks<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    maxima: bool,
    iqr_ratio: f64,
) -> Bound<'py, numpy::PyArray1<usize>> {
    let result = metrust::calc::find_peaks(
        data.as_slice().unwrap(),
        maxima,
        iqr_ratio,
    );
    result.into_pyarray(py)
}

/// Topological persistence-based peak detection.
///
/// Parameters
/// ----------
/// data : array of float64
///     1-D data array.
/// maxima : bool, optional
///     If True (default), detect peaks; if False, detect troughs.
///
/// Returns
/// -------
/// list of (int, float)
///     (index, persistence) pairs sorted by descending persistence.
#[pyfunction]
#[pyo3(signature = (data, maxima=true))]
fn peak_persistence(
    data: PyReadonlyArray1<f64>,
    maxima: bool,
) -> Vec<(usize, f64)> {
    metrust::calc::peak_persistence(data.as_slice().unwrap(), maxima)
}

// ─── Radar coordinate conversion ────────────────────────────────────

/// Convert radar azimuth/range to latitude/longitude.
///
/// Parameters
/// ----------
/// azimuths : array of float64
///     Azimuth angles in degrees clockwise from north.
/// ranges : array of float64
///     Range values in meters from the radar.
/// center_lat : float
///     Radar site latitude in degrees.
/// center_lon : float
///     Radar site longitude in degrees.
///
/// Returns
/// -------
/// tuple of (numpy.ndarray, numpy.ndarray)
///     (latitudes, longitudes) arrays of length azimuths * ranges.
#[pyfunction]
#[pyo3(text_signature = "(azimuths, ranges, center_lat, center_lon)")]
fn azimuth_range_to_lat_lon<'py>(
    py: Python<'py>,
    azimuths: PyReadonlyArray1<f64>,
    ranges: PyReadonlyArray1<f64>,
    center_lat: f64,
    center_lon: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (lats, lons) = metrust::calc::azimuth_range_to_lat_lon(
        azimuths.as_slice().unwrap(),
        ranges.as_slice().unwrap(),
        center_lat,
        center_lon,
    );
    (lats.into_pyarray(py), lons.into_pyarray(py))
}

// ─── Module registration ─────────────────────────────────────────────

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(angle_to_direction, parent)?)?;
    parent.add_function(wrap_pyfunction!(parse_angle, parent)?)?;
    parent.add_function(wrap_pyfunction!(find_bounding_indices, parent)?)?;
    parent.add_function(wrap_pyfunction!(nearest_intersection_idx, parent)?)?;
    parent.add_function(wrap_pyfunction!(resample_nn_1d, parent)?)?;
    parent.add_function(wrap_pyfunction!(find_peaks, parent)?)?;
    parent.add_function(wrap_pyfunction!(peak_persistence, parent)?)?;
    parent.add_function(wrap_pyfunction!(azimuth_range_to_lat_lon, parent)?)?;
    Ok(())
}
