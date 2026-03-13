use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

// ─── Helper: parse InterpMethod from a Python string ─────────────────────

fn parse_method(method: &str) -> PyResult<metrust::interpolate::InterpMethod> {
    metrust::interpolate::InterpMethod::from_str_loose(method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown interpolation method '{}'. Use 'nearest', 'bilinear', 'bicubic', or 'budget'.",
            method
        ))
    })
}

// ─── Helper: build a GridSpec from Python kwargs ─────────────────────────

fn build_grid_spec(
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    resolution: f64,
) -> metrust::interpolate::GridSpec {
    metrust::interpolate::GridSpec::regular(lat_min, lat_max, lon_min, lon_max, resolution)
}

// ═════════════════════════════════════════════════════════════════════════
// 1-D interpolation
// ═════════════════════════════════════════════════════════════════════════

/// Piecewise linear interpolation (like numpy.interp).
///
/// Given monotonically increasing breakpoints `xp` with values `fp`,
/// evaluate the piecewise-linear interpolant at each point in `x`.
#[pyfunction]
#[pyo3(text_signature = "(x, xp, fp)")]
fn interpolate_1d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    xp: PyReadonlyArray1<f64>,
    fp: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::interpolate::interpolate_1d(
        x.as_slice()?,
        xp.as_slice()?,
        fp.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

/// Interpolation in log-pressure space (like MetPy's log_interpolate_1d).
///
/// Performs linear interpolation in ln(x) space, appropriate for
/// interpolating meteorological variables with respect to pressure.
#[pyfunction]
#[pyo3(text_signature = "(x, xp, fp)")]
fn log_interpolate_1d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    xp: PyReadonlyArray1<f64>,
    fp: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::interpolate::log_interpolate_1d(
        x.as_slice()?,
        xp.as_slice()?,
        fp.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

/// Fill NaN values by linearly interpolating between surrounding valid points.
///
/// Edge NaNs are filled with the nearest valid value. Returns a new array.
#[pyfunction]
#[pyo3(text_signature = "(values,)")]
fn interpolate_nans_1d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut data = values.as_slice()?.to_vec();
    metrust::interpolate::interpolate_nans_1d(&mut data);
    Ok(data.into_pyarray(py))
}

// ═════════════════════════════════════════════════════════════════════════
// Grid interpolation (scattered -> regular grid)
// ═════════════════════════════════════════════════════════════════════════

/// Interpolate scattered observations onto a regular lat/lon grid.
///
/// Parameters
/// ----------
/// values : array of observation values
/// src_lats : array of observation latitudes
/// src_lons : array of observation longitudes
/// lat_min, lat_max, lon_min, lon_max : grid bounding box (degrees)
/// resolution : grid spacing (degrees)
/// method : interpolation method ('nearest', 'bilinear', 'bicubic', 'budget')
///
/// Returns a 2-D array of shape (ny, nx).
#[pyfunction]
#[pyo3(text_signature = "(values, src_lats, src_lons, lat_min, lat_max, lon_min, lon_max, resolution, method)")]
fn interpolate_to_grid<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    resolution: f64,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let target = build_grid_spec(lat_min, lat_max, lon_min, lon_max, resolution);
    let m = parse_method(method)?;
    let result = metrust::interpolate::interpolate_to_grid(
        values.as_slice()?,
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        &target,
        m,
    );
    let ny = target.ny;
    let nx = target.nx;
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

/// Regrid data from a source grid to a target regular lat/lon grid.
///
/// Parameters
/// ----------
/// src_values : flattened source grid data [ny*nx]
/// src_lats : source latitudes (1D [ny] or 2D [ny*nx])
/// src_lons : source longitudes (1D [nx] or 2D [ny*nx])
/// src_nx, src_ny : source grid dimensions
/// lat_min, lat_max, lon_min, lon_max : target grid bounding box (degrees)
/// resolution : target grid spacing (degrees)
/// method : interpolation method
///
/// Returns a 2-D array of shape (target_ny, target_nx).
#[pyfunction]
#[pyo3(text_signature = "(src_values, src_lats, src_lons, src_nx, src_ny, lat_min, lat_max, lon_min, lon_max, resolution, method)")]
fn regrid<'py>(
    py: Python<'py>,
    src_values: PyReadonlyArray1<f64>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    src_nx: usize,
    src_ny: usize,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    resolution: f64,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let target = build_grid_spec(lat_min, lat_max, lon_min, lon_max, resolution);
    let m = parse_method(method)?;
    let result = metrust::interpolate::regrid(
        src_values.as_slice()?,
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        src_nx,
        src_ny,
        &target,
        m,
    );
    let ny = target.ny;
    let nx = target.nx;
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

// ═════════════════════════════════════════════════════════════════════════
// Point interpolation
// ═════════════════════════════════════════════════════════════════════════

/// Interpolate a gridded field to a single lat/lon point.
///
/// Parameters
/// ----------
/// values : flattened grid data [ny*nx]
/// lats : grid latitudes (1D [ny] or 2D [ny*nx])
/// lons : grid longitudes (1D [nx] or 2D [ny*nx])
/// nx, ny : grid dimensions
/// target_lat, target_lon : point to interpolate to
/// method : interpolation method
#[pyfunction]
#[pyo3(text_signature = "(values, lats, lons, nx, ny, target_lat, target_lon, method)")]
fn interpolate_point(
    values: PyReadonlyArray1<f64>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    target_lat: f64,
    target_lon: f64,
    method: &str,
) -> PyResult<f64> {
    let m = parse_method(method)?;
    Ok(metrust::interpolate::interpolate_point(
        values.as_slice()?,
        lats.as_slice()?,
        lons.as_slice()?,
        nx,
        ny,
        target_lat,
        target_lon,
        m,
    ))
}

/// Interpolate a gridded field to multiple lat/lon points.
///
/// Parameters
/// ----------
/// values : flattened grid data [ny*nx]
/// lats : grid latitudes (1D [ny] or 2D [ny*nx])
/// lons : grid longitudes (1D [nx] or 2D [ny*nx])
/// nx, ny : grid dimensions
/// target_lats, target_lons : arrays of target coordinates
/// method : interpolation method
#[pyfunction]
#[pyo3(text_signature = "(values, lats, lons, nx, ny, target_lats, target_lons, method)")]
fn interpolate_points<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    target_lats: PyReadonlyArray1<f64>,
    target_lons: PyReadonlyArray1<f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = parse_method(method)?;
    let result = metrust::interpolate::interpolate_points(
        values.as_slice()?,
        lats.as_slice()?,
        lons.as_slice()?,
        nx,
        ny,
        target_lats.as_slice()?,
        target_lons.as_slice()?,
        m,
    );
    Ok(result.into_pyarray(py))
}

// ═════════════════════════════════════════════════════════════════════════
// Cross-section / vertical interpolation
// ═════════════════════════════════════════════════════════════════════════

/// Extract a cross-section along a great-circle path between two points.
///
/// Parameters
/// ----------
/// values : flattened grid data [ny*nx]
/// lats : grid latitudes (1D [ny] or 2D [ny*nx])
/// lons : grid longitudes (1D [nx] or 2D [ny*nx])
/// nx, ny : grid dimensions
/// start_lat, start_lon : start point (degrees)
/// end_lat, end_lon : end point (degrees)
/// n_points : number of sample points along the path
/// method : interpolation method
///
/// Returns (interpolated_values, distances_km).
#[pyfunction]
#[pyo3(text_signature = "(values, lats, lons, nx, ny, start_lat, start_lon, end_lat, end_lon, n_points, method)")]
fn cross_section_data<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
    n_points: usize,
    method: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let m = parse_method(method)?;
    let (vals, dists) = metrust::interpolate::cross_section_data(
        values.as_slice()?,
        lats.as_slice()?,
        lons.as_slice()?,
        nx,
        ny,
        (start_lat, start_lon),
        (end_lat, end_lon),
        n_points,
        m,
    );
    Ok((vals.into_pyarray(py), dists.into_pyarray(py)))
}

/// Interpolate a 3-D field to a specific pressure or height level.
///
/// Parameters
/// ----------
/// values_3d : flattened [nz*ny*nx] data (level 0 first)
/// levels : vertical coordinate at each level (length nz)
/// target_level : the level to interpolate to
/// nx, ny, nz : grid dimensions
/// log_interp : if True, use log-linear interpolation (for pressure coords)
///
/// Returns a 2-D array of shape (ny, nx).
#[pyfunction]
#[pyo3(text_signature = "(values_3d, levels, target_level, nx, ny, nz, log_interp)")]
fn interpolate_vertical<'py>(
    py: Python<'py>,
    values_3d: PyReadonlyArray1<f64>,
    levels: PyReadonlyArray1<f64>,
    target_level: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    log_interp: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = metrust::interpolate::interpolate_vertical(
        values_3d.as_slice()?,
        levels.as_slice()?,
        target_level,
        nx,
        ny,
        nz,
        log_interp,
    );
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

/// Interpolate a 3-D field to an isosurface of another 3-D field.
///
/// For each (i, j) column, walks upward through levels and finds where
/// `surface_values` crosses `target`, then linearly interpolates the
/// corresponding value from `values_3d`.
///
/// Parameters
/// ----------
/// values_3d : flattened [nz*ny*nx] data (level-major)
/// surface_values : flattened [nz*ny*nx] surface field
/// target : isosurface target value
/// levels : vertical coordinate at each level (length nz)
/// nx, ny, nz : grid dimensions
///
/// Returns a 2-D array of shape (ny, nx). NaN where no crossing found.
#[pyfunction]
#[pyo3(text_signature = "(values_3d, surface_values, target, levels, nx, ny, nz)")]
fn interpolate_to_isosurface<'py>(
    py: Python<'py>,
    values_3d: PyReadonlyArray1<f64>,
    surface_values: PyReadonlyArray1<f64>,
    target: f64,
    levels: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = metrust::interpolate::interpolate_to_isosurface(
        values_3d.as_slice()?,
        surface_values.as_slice()?,
        target,
        levels.as_slice()?,
        nx,
        ny,
        nz,
    );
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

/// Extract a vertical cross-section from 3-D gridded data along a lat/lon path.
///
/// Parameters
/// ----------
/// values_3d : flattened [nz*ny*nx] data (level-major)
/// levels : vertical coordinate at each level (length nz)
/// lat_slice : latitudes along the slice path
/// lon_slice : longitudes along the slice path
/// src_lats : source grid latitudes (length ny)
/// src_lons : source grid longitudes (length nx)
/// nx, ny, nz : grid dimensions
///
/// Returns a 2-D array of shape (n_points, nz).
#[pyfunction]
#[pyo3(text_signature = "(values_3d, levels, lat_slice, lon_slice, src_lats, src_lons, nx, ny, nz)")]
fn interpolate_to_slice<'py>(
    py: Python<'py>,
    values_3d: PyReadonlyArray1<f64>,
    levels: PyReadonlyArray1<f64>,
    lat_slice: PyReadonlyArray1<f64>,
    lon_slice: PyReadonlyArray1<f64>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let result = metrust::interpolate::interpolate_to_slice(
        values_3d.as_slice()?,
        levels.as_slice()?,
        lat_slice.as_slice()?,
        lon_slice.as_slice()?,
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        nx,
        ny,
        nz,
    );
    // result is Vec<Vec<f64>> of shape [n_points][nz]
    Ok(numpy::PyArray2::from_vec2(py, &result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

// ═════════════════════════════════════════════════════════════════════════
// Inverse distance weighted interpolation
// ═════════════════════════════════════════════════════════════════════════

/// Inverse distance weighted (IDW) interpolation to a regular grid.
///
/// For each target grid point, finds source points within `search_radius`
/// degrees, weights by 1/d^power, and computes the weighted average.
///
/// Parameters
/// ----------
/// lats : source observation latitudes
/// lons : source observation longitudes
/// values : source observation values
/// lat_min, lat_max, lon_min, lon_max : target grid bounding box (degrees)
/// resolution : target grid spacing (degrees)
/// power : distance weighting exponent (typically 2.0)
/// min_neighbors : minimum neighbors required (else NaN)
/// search_radius : search radius in degrees
///
/// Returns a 2-D array of shape (ny, nx).
#[pyfunction]
#[pyo3(text_signature = "(lats, lons, values, lat_min, lat_max, lon_min, lon_max, resolution, power, min_neighbors, search_radius)")]
fn inverse_distance_to_grid<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    resolution: f64,
    power: f64,
    min_neighbors: usize,
    search_radius: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let target = build_grid_spec(lat_min, lat_max, lon_min, lon_max, resolution);
    let result = metrust::interpolate::inverse_distance_to_grid(
        lats.as_slice()?,
        lons.as_slice()?,
        values.as_slice()?,
        &target,
        power,
        min_neighbors,
        search_radius,
    );
    let ny = target.ny;
    let nx = target.nx;
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

/// Inverse distance weighted (IDW) interpolation to arbitrary points.
///
/// Parameters
/// ----------
/// src_lats : source observation latitudes
/// src_lons : source observation longitudes
/// src_values : source observation values
/// target_lats : target point latitudes
/// target_lons : target point longitudes
/// power : distance weighting exponent
/// min_neighbors : minimum neighbors required (else NaN)
/// search_radius : search radius in degrees
#[pyfunction]
#[pyo3(text_signature = "(src_lats, src_lons, src_values, target_lats, target_lons, power, min_neighbors, search_radius)")]
fn inverse_distance_to_points<'py>(
    py: Python<'py>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    src_values: PyReadonlyArray1<f64>,
    target_lats: PyReadonlyArray1<f64>,
    target_lons: PyReadonlyArray1<f64>,
    power: f64,
    min_neighbors: usize,
    search_radius: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::interpolate::inverse_distance_to_points(
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        src_values.as_slice()?,
        target_lats.as_slice()?,
        target_lons.as_slice()?,
        power,
        min_neighbors,
        search_radius,
    );
    Ok(result.into_pyarray(py))
}

// ═════════════════════════════════════════════════════════════════════════
// Natural neighbor interpolation
// ═════════════════════════════════════════════════════════════════════════

/// Approximate natural-neighbor (Sibson) interpolation to a regular grid.
///
/// Parameters
/// ----------
/// lats : source observation latitudes
/// lons : source observation longitudes
/// values : source observation values
/// lat_min, lat_max, lon_min, lon_max : target grid bounding box (degrees)
/// resolution : target grid spacing (degrees)
///
/// Returns a 2-D array of shape (ny, nx).
#[pyfunction]
#[pyo3(text_signature = "(lats, lons, values, lat_min, lat_max, lon_min, lon_max, resolution)")]
fn natural_neighbor_to_grid<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    resolution: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let target = build_grid_spec(lat_min, lat_max, lon_min, lon_max, resolution);
    let result = metrust::interpolate::natural_neighbor_to_grid(
        lats.as_slice()?,
        lons.as_slice()?,
        values.as_slice()?,
        &target,
    );
    let ny = target.ny;
    let nx = target.nx;
    Ok(numpy::PyArray2::from_vec2(py, &vec_to_2d(&result, ny, nx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
}

/// Approximate natural-neighbor (Sibson) interpolation to arbitrary points.
///
/// Parameters
/// ----------
/// src_lats : source observation latitudes
/// src_lons : source observation longitudes
/// src_values : source observation values
/// target_lats : target point latitudes
/// target_lons : target point longitudes
#[pyfunction]
#[pyo3(text_signature = "(src_lats, src_lons, src_values, target_lats, target_lons)")]
fn natural_neighbor_to_points<'py>(
    py: Python<'py>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    src_values: PyReadonlyArray1<f64>,
    target_lats: PyReadonlyArray1<f64>,
    target_lons: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::interpolate::natural_neighbor_to_points(
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        src_values.as_slice()?,
        target_lats.as_slice()?,
        target_lons.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

// ═════════════════════════════════════════════════════════════════════════
// Generic interpolate_to_points dispatcher
// ═════════════════════════════════════════════════════════════════════════

/// Interpolate scattered data to arbitrary points using a specified method.
///
/// Parameters
/// ----------
/// src_lats : source observation latitudes
/// src_lons : source observation longitudes
/// src_values : source observation values
/// target_lats : target point latitudes
/// target_lons : target point longitudes
/// interp_type : interpolation method ('idw', 'linear', 'inverse_distance',
///               'natural_neighbor', 'nn', 'natural')
#[pyfunction]
#[pyo3(signature = (src_lats, src_lons, src_values, target_lats, target_lons, interp_type="linear"))]
fn interpolate_to_points_dispatch<'py>(
    py: Python<'py>,
    src_lats: PyReadonlyArray1<f64>,
    src_lons: PyReadonlyArray1<f64>,
    src_values: PyReadonlyArray1<f64>,
    target_lats: PyReadonlyArray1<f64>,
    target_lons: PyReadonlyArray1<f64>,
    interp_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::interpolate::interpolate_to_points(
        src_lats.as_slice()?,
        src_lons.as_slice()?,
        src_values.as_slice()?,
        target_lats.as_slice()?,
        target_lons.as_slice()?,
        interp_type,
    );
    Ok(result.into_pyarray(py))
}

// ═════════════════════════════════════════════════════════════════════════
// Observation filtering
// ═════════════════════════════════════════════════════════════════════════

/// Remove observations where the value is NaN.
///
/// Returns (lats, lons, values) with NaN entries dropped.
#[pyfunction]
#[pyo3(text_signature = "(lats, lons, values)")]
fn remove_nan_observations<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (out_lats, out_lons, out_vals) = metrust::interpolate::remove_nan_observations(
        lats.as_slice()?,
        lons.as_slice()?,
        values.as_slice()?,
    );
    Ok((
        out_lats.into_pyarray(py),
        out_lons.into_pyarray(py),
        out_vals.into_pyarray(py),
    ))
}

/// Remove observations where the value is below a threshold.
///
/// Returns (lats, lons, values) with only entries where value >= threshold.
#[pyfunction]
#[pyo3(text_signature = "(lats, lons, values, threshold)")]
fn remove_observations_below_value<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (out_lats, out_lons, out_vals) = metrust::interpolate::remove_observations_below_value(
        lats.as_slice()?,
        lons.as_slice()?,
        values.as_slice()?,
        threshold,
    );
    Ok((
        out_lats.into_pyarray(py),
        out_lons.into_pyarray(py),
        out_vals.into_pyarray(py),
    ))
}

/// Remove observations with duplicate (lat, lon) coordinates, keeping the first.
///
/// Returns (lats, lons, values) with duplicates removed.
#[pyfunction]
#[pyo3(text_signature = "(lats, lons, values)")]
fn remove_repeat_coordinates<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (out_lats, out_lons, out_vals) = metrust::interpolate::remove_repeat_coordinates(
        lats.as_slice()?,
        lons.as_slice()?,
        values.as_slice()?,
    );
    Ok((
        out_lats.into_pyarray(py),
        out_lons.into_pyarray(py),
        out_vals.into_pyarray(py),
    ))
}

// ═════════════════════════════════════════════════════════════════════════
// Geodesic
// ═════════════════════════════════════════════════════════════════════════

/// Compute equally-spaced points along the great-circle path between two positions.
///
/// Parameters
/// ----------
/// start_lat, start_lon : start point (degrees)
/// end_lat, end_lon : end point (degrees)
/// n_points : number of sample points (>= 2), including start and end
///
/// Returns (lats, lons) arrays each of length n_points.
#[pyfunction]
#[pyo3(text_signature = "(start_lat, start_lon, end_lat, end_lon, n_points)")]
fn geodesic<'py>(
    py: Python<'py>,
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
    n_points: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (lats, lons) = metrust::interpolate::geodesic(
        (start_lat, start_lon),
        (end_lat, end_lon),
        n_points,
    );
    Ok((lats.into_pyarray(py), lons.into_pyarray(py)))
}

// ═════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═════════════════════════════════════════════════════════════════════════

/// Reshape a flat Vec into Vec<Vec<f64>> for PyArray2::from_vec2.
fn vec_to_2d(flat: &[f64], nrows: usize, ncols: usize) -> Vec<Vec<f64>> {
    (0..nrows)
        .map(|r| flat[r * ncols..(r + 1) * ncols].to_vec())
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════
// Module registration
// ═════════════════════════════════════════════════════════════════════════

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1-D interpolation
    parent.add_function(wrap_pyfunction!(interpolate_1d, parent)?)?;
    parent.add_function(wrap_pyfunction!(log_interpolate_1d, parent)?)?;
    parent.add_function(wrap_pyfunction!(interpolate_nans_1d, parent)?)?;

    // Grid interpolation
    parent.add_function(wrap_pyfunction!(interpolate_to_grid, parent)?)?;
    parent.add_function(wrap_pyfunction!(regrid, parent)?)?;

    // Point interpolation
    parent.add_function(wrap_pyfunction!(interpolate_point, parent)?)?;
    parent.add_function(wrap_pyfunction!(interpolate_points, parent)?)?;

    // Cross-section / vertical
    parent.add_function(wrap_pyfunction!(cross_section_data, parent)?)?;
    parent.add_function(wrap_pyfunction!(interpolate_vertical, parent)?)?;
    parent.add_function(wrap_pyfunction!(interpolate_to_isosurface, parent)?)?;
    parent.add_function(wrap_pyfunction!(interpolate_to_slice, parent)?)?;

    // IDW
    parent.add_function(wrap_pyfunction!(inverse_distance_to_grid, parent)?)?;
    parent.add_function(wrap_pyfunction!(inverse_distance_to_points, parent)?)?;

    // Natural neighbor
    parent.add_function(wrap_pyfunction!(natural_neighbor_to_grid, parent)?)?;
    parent.add_function(wrap_pyfunction!(natural_neighbor_to_points, parent)?)?;

    // Generic dispatcher
    parent.add_function(wrap_pyfunction!(interpolate_to_points_dispatch, parent)?)?;

    // Observation filtering
    parent.add_function(wrap_pyfunction!(remove_nan_observations, parent)?)?;
    parent.add_function(wrap_pyfunction!(remove_observations_below_value, parent)?)?;
    parent.add_function(wrap_pyfunction!(remove_repeat_coordinates, parent)?)?;

    // Geodesic
    parent.add_function(wrap_pyfunction!(geodesic, parent)?)?;

    Ok(())
}
