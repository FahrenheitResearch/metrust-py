use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

// ─── Smoothing filters ──────────────────────────────────────────────

/// 2D Gaussian smoothing with standard deviation `sigma` (in grid-point units).
#[pyfunction]
fn smooth_gaussian<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::smooth_gaussian(&flat, nx, ny, sigma)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Rectangular (box / uniform) smoothing with kernel side length `size`.
///
/// `passes` controls how many times the filter is applied (default 1).
#[pyfunction]
#[pyo3(signature = (data, size, passes=1))]
fn smooth_rectangular<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    size: usize,
    passes: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::smooth_rectangular(&flat, nx, ny, size, passes)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Circular (disk) smoothing with `radius` in grid-point units.
///
/// `passes` controls how many times the filter is applied (default 1).
#[pyfunction]
#[pyo3(signature = (data, radius, passes=1))]
fn smooth_circular<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    radius: f64,
    passes: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::smooth_circular(&flat, nx, ny, radius, passes)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// N-point smoother (n must be 5 or 9).
///
/// `passes` controls how many times the filter is applied (default 1).
#[pyfunction]
#[pyo3(signature = (data, n, passes=1))]
fn smooth_n_point<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    n: usize,
    passes: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::smooth_n_point(&flat, nx, ny, n, passes)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Generic 2D convolution with a user-supplied kernel (window).
///
/// `data` is the 2D input grid and `window` is the 2D kernel. Both are
/// provided as numpy 2D arrays.
///
/// `passes` controls how many times the filter is applied (default 1).
/// `normalize_weights` controls whether weights are normalized before
/// applying (default true).
#[pyfunction]
#[pyo3(signature = (data, window, passes=1, normalize_weights=true))]
fn smooth_window<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    window: PyReadonlyArray2<f64>,
    passes: usize,
    normalize_weights: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d_shape = data.shape();
    let (ny, nx) = (d_shape[0], d_shape[1]);
    let w_shape = window.shape();
    let (window_ny, window_nx) = (w_shape[0], w_shape[1]);
    let flat_data: Vec<f64> = data.as_slice()?.to_vec();
    let flat_window: Vec<f64> = window.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::smooth_window(
            &flat_data, nx, ny, &flat_window, window_nx, window_ny, passes, normalize_weights,
        )
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

// ─── Grid derivative functions ──────────────────────────────────────

/// Partial derivative df/dx using centered finite differences.
///
/// `data` is a 2D grid (ny, nx). `dx` is the uniform grid spacing in meters.
#[pyfunction]
fn gradient_x<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    dx: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::gradient_x(&flat, nx, ny, dx)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Partial derivative df/dy using centered finite differences.
///
/// `data` is a 2D grid (ny, nx). `dy` is the uniform grid spacing in meters.
#[pyfunction]
fn gradient_y<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::gradient_y(&flat, nx, ny, dy)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Laplacian (d2f/dx2 + d2f/dy2) using second-order centered differences.
///
/// `data` is a 2D grid (ny, nx). `dx` and `dy` are grid spacings in meters.
#[pyfunction]
fn laplacian<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::laplacian(&flat, nx, ny, dx, dy)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// First derivative along a chosen axis (0 = x, 1 = y).
///
/// `data` is a 2D grid (ny, nx). `axis_spacing` is the uniform grid spacing
/// along the chosen axis in meters.
#[pyfunction]
fn first_derivative<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    axis_spacing: f64,
    axis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::first_derivative(&flat, axis_spacing, axis, nx, ny)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

/// Second derivative along a chosen axis (0 = x, 1 = y).
///
/// `data` is a 2D grid (ny, nx). `axis_spacing` is the uniform grid spacing
/// along the chosen axis in meters.
#[pyfunction]
fn second_derivative<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    axis_spacing: f64,
    axis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat: Vec<f64> = data.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::second_derivative(&flat, axis_spacing, axis, nx, ny)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

// ─── Geospatial functions ───────────────────────────────────────────

/// Compute physical grid spacings (dx, dy) in meters from lat/lon grids.
///
/// `lats` and `lons` are 2D arrays of shape (ny, nx) in degrees. Returns a
/// tuple of two 2D arrays `(dx, dy)`, each of shape (ny, nx), giving the
/// local east-west and north-south spacing in meters at every grid point.
#[pyfunction]
fn lat_lon_grid_deltas<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray2<f64>,
    lons: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = lats.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat_lats: Vec<f64> = lats.as_slice()?.to_vec();
    let flat_lons: Vec<f64> = lons.as_slice()?.to_vec();
    let (dx_vec, dy_vec) = py.allow_threads(|| {
        metrust::calc::smooth::lat_lon_grid_deltas(&flat_lats, &flat_lons, nx, ny)
    });
    let dx_rows: Vec<Vec<f64>> = dx_vec.chunks(nx).map(|c| c.to_vec()).collect();
    let dy_rows: Vec<Vec<f64>> = dy_vec.chunks(nx).map(|c| c.to_vec()).collect();
    Ok((
        PyArray2::from_vec2(py, &dx_rows)?.into(),
        PyArray2::from_vec2(py, &dy_rows)?.into(),
    ))
}

/// Gradient of a scalar field on a lat/lon grid with geospatially-correct
/// spacings.
///
/// `data`, `lats`, and `lons` are all 2D arrays of shape (ny, nx). Lats and
/// lons are in degrees. Returns `(df_dx, df_dy)` in physical units (per meter).
#[pyfunction]
fn geospatial_gradient<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    lons: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat_data: Vec<f64> = data.as_slice()?.to_vec();
    let flat_lats: Vec<f64> = lats.as_slice()?.to_vec();
    let flat_lons: Vec<f64> = lons.as_slice()?.to_vec();
    let (dfdx, dfdy) = py.allow_threads(|| {
        metrust::calc::smooth::geospatial_gradient(&flat_data, &flat_lats, &flat_lons, nx, ny)
    });
    let dfdx_rows: Vec<Vec<f64>> = dfdx.chunks(nx).map(|c| c.to_vec()).collect();
    let dfdy_rows: Vec<Vec<f64>> = dfdy.chunks(nx).map(|c| c.to_vec()).collect();
    Ok((
        PyArray2::from_vec2(py, &dfdx_rows)?.into(),
        PyArray2::from_vec2(py, &dfdy_rows)?.into(),
    ))
}

/// Laplacian of a scalar field on a lat/lon grid with geospatially-correct
/// spacings.
///
/// `data`, `lats`, and `lons` are all 2D arrays of shape (ny, nx). Lats and
/// lons are in degrees.
#[pyfunction]
fn geospatial_laplacian<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    lons: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let flat_data: Vec<f64> = data.as_slice()?.to_vec();
    let flat_lats: Vec<f64> = lats.as_slice()?.to_vec();
    let flat_lons: Vec<f64> = lons.as_slice()?.to_vec();
    let result = py.allow_threads(|| {
        metrust::calc::smooth::geospatial_laplacian(&flat_data, &flat_lats, &flat_lons, nx, ny)
    });
    let rows: Vec<Vec<f64>> = result.chunks(nx).map(|c| c.to_vec()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?.into())
}

// ─── Module registration ────────────────────────────────────────────

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // Smoothing filters
    parent.add_function(wrap_pyfunction!(smooth_gaussian, parent)?)?;
    parent.add_function(wrap_pyfunction!(smooth_rectangular, parent)?)?;
    parent.add_function(wrap_pyfunction!(smooth_circular, parent)?)?;
    parent.add_function(wrap_pyfunction!(smooth_n_point, parent)?)?;
    parent.add_function(wrap_pyfunction!(smooth_window, parent)?)?;
    // Grid derivatives
    parent.add_function(wrap_pyfunction!(gradient_x, parent)?)?;
    parent.add_function(wrap_pyfunction!(gradient_y, parent)?)?;
    parent.add_function(wrap_pyfunction!(laplacian, parent)?)?;
    parent.add_function(wrap_pyfunction!(first_derivative, parent)?)?;
    parent.add_function(wrap_pyfunction!(second_derivative, parent)?)?;
    // Geospatial functions
    parent.add_function(wrap_pyfunction!(lat_lon_grid_deltas, parent)?)?;
    parent.add_function(wrap_pyfunction!(geospatial_gradient, parent)?)?;
    parent.add_function(wrap_pyfunction!(geospatial_laplacian, parent)?)?;
    Ok(())
}
