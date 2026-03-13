use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

// ─── Helper: reshape a flat Vec<f64> into a 2D Vec<Vec<f64>> (row-major) ───

fn to_2d(flat: &[f64], ny: usize, nx: usize) -> Vec<Vec<f64>> {
    (0..ny)
        .map(|j| flat[j * nx..(j + 1) * nx].to_vec())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
// 2D grid functions (accept numpy 2D arrays, return numpy 2D arrays)
//
// The metrust functions operate on flattened row-major slices with
// explicit (nx, ny) dimensions.  The bindings convert between numpy's
// 2D layout and flat slices transparently.
// ═══════════════════════════════════════════════════════════════════════

/// Horizontal divergence: du/dx + dv/dy.
#[pyfunction]
fn divergence<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::divergence(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Relative vorticity: dv/dx - du/dy.
#[pyfunction]
fn vorticity<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::vorticity(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Absolute vorticity: relative vorticity + Coriolis parameter.
#[pyfunction]
fn absolute_vorticity<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::absolute_vorticity(
        u.as_slice()?, v.as_slice()?, lats.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Advection of a scalar field by a 2D wind: -u(ds/dx) - v(ds/dy).
#[pyfunction]
fn advection<'py>(
    py: Python<'py>,
    scalar: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = scalar.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::advection(
        scalar.as_slice()?, u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// 2D Petterssen frontogenesis function.
#[pyfunction]
fn frontogenesis<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = theta.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::frontogenesis(
        theta.as_slice()?, u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Geostrophic wind from a geopotential height field.
///
/// Returns (u_geo, v_geo) as a tuple of 2D arrays.
#[pyfunction]
fn geostrophic_wind<'py>(
    py: Python<'py>,
    height: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = height.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let (u_geo, v_geo) = metrust::calc::kinematics::geostrophic_wind(
        height.as_slice()?, lats.as_slice()?, nx, ny, dx, dy,
    );
    Ok((
        PyArray2::from_vec2(py, &to_2d(&u_geo, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&v_geo, ny, nx))?,
    ))
}

/// Ageostrophic wind: total wind minus geostrophic wind.
///
/// Returns (u - u_geo, v - v_geo) as a tuple of 1D arrays.
#[pyfunction]
fn ageostrophic_wind<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    u_geo: PyReadonlyArray1<f64>,
    v_geo: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (ua, va) = metrust::calc::kinematics::ageostrophic_wind(
        u.as_slice()?, v.as_slice()?, u_geo.as_slice()?, v_geo.as_slice()?,
    );
    Ok((ua.into_pyarray(py), va.into_pyarray(py)))
}

/// Q-vector components (Q1, Q2) on a constant pressure surface.
///
/// Returns (q1, q2) as a tuple of 2D arrays.
#[pyfunction]
fn q_vector<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<f64>,
    u_geo: PyReadonlyArray2<f64>,
    v_geo: PyReadonlyArray2<f64>,
    p_hpa: f64,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = t.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let (q1, q2) = metrust::calc::kinematics::q_vector(
        t.as_slice()?, u_geo.as_slice()?, v_geo.as_slice()?,
        p_hpa, nx, ny, dx, dy,
    );
    Ok((
        PyArray2::from_vec2(py, &to_2d(&q1, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&q2, ny, nx))?,
    ))
}

/// Q-vector convergence: -2 * div(Q).
#[pyfunction]
fn q_vector_convergence<'py>(
    py: Python<'py>,
    q1: PyReadonlyArray2<f64>,
    q2: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = q1.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::q_vector_convergence(
        q1.as_slice()?, q2.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Stretching deformation: du/dx - dv/dy.
#[pyfunction]
fn stretching_deformation<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::stretching_deformation(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Shearing deformation: dv/dx + du/dy.
#[pyfunction]
fn shearing_deformation<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::shearing_deformation(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Total deformation: sqrt(stretching^2 + shearing^2).
#[pyfunction]
fn total_deformation<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::total_deformation(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Curvature vorticity -- the component of vorticity from streamline curvature.
#[pyfunction]
fn curvature_vorticity<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::curvature_vorticity(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Shear vorticity -- the component of vorticity from cross-stream speed shear.
#[pyfunction]
fn shear_vorticity<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::shear_vorticity(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Temperature advection (convenience wrapper around advection).
#[pyfunction]
fn temperature_advection<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = t.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::temperature_advection(
        t.as_slice()?, u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Moisture advection (convenience wrapper around advection).
#[pyfunction]
fn moisture_advection<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = q.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::moisture_advection(
        q.as_slice()?, u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Inertial-advective wind: advection of the geostrophic wind by the total wind.
///
/// Returns (u_ia, v_ia) as a tuple of 2D arrays.
#[pyfunction]
fn inertial_advective_wind<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    u_geo: PyReadonlyArray2<f64>,
    v_geo: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let (u_ia, v_ia) = metrust::calc::kinematics::inertial_advective_wind(
        u.as_slice()?, v.as_slice()?,
        u_geo.as_slice()?, v_geo.as_slice()?,
        nx, ny, dx, dy,
    );
    Ok((
        PyArray2::from_vec2(py, &to_2d(&u_ia, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&v_ia, ny, nx))?,
    ))
}

/// Compute all four components of the 2D velocity gradient tensor.
///
/// Returns (du_dx, du_dy, dv_dx, dv_dy) as a tuple of 2D arrays.
#[pyfunction]
fn vector_derivative<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let (du_dx, du_dy, dv_dx, dv_dy) = metrust::calc::kinematics::vector_derivative(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    Ok((
        PyArray2::from_vec2(py, &to_2d(&du_dx, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&du_dy, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&dv_dx, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&dv_dy, ny, nx))?,
    ))
}

/// Partial derivative df/dx using centered finite differences.
#[pyfunction]
fn gradient_x<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    dx: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = values.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::gradient_x(
        values.as_slice()?, nx, ny, dx,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Partial derivative df/dy using centered finite differences.
#[pyfunction]
fn gradient_y<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = values.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::gradient_y(
        values.as_slice()?, nx, ny, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Laplacian: d2f/dx2 + d2f/dy2.
#[pyfunction]
fn laplacian<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = values.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::laplacian(
        values.as_slice()?, nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Generalized first derivative along a chosen axis (0 = x, 1 = y).
#[pyfunction]
fn first_derivative<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    axis_spacing: f64,
    axis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = values.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::first_derivative(
        values.as_slice()?, axis_spacing, axis, nx, ny,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Generalized second derivative along a chosen axis (0 = x, 1 = y).
#[pyfunction]
fn second_derivative<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    axis_spacing: f64,
    axis: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = values.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::second_derivative(
        values.as_slice()?, axis_spacing, axis, nx, ny,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Compute physical grid spacings (dx, dy) in meters from lat/lon arrays.
///
/// Returns (dx, dy) as a tuple of 2D arrays.
#[pyfunction]
fn lat_lon_grid_deltas<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray2<f64>,
    lons: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = lats.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let (dx, dy) = metrust::calc::kinematics::lat_lon_grid_deltas(
        lats.as_slice()?, lons.as_slice()?, nx, ny,
    );
    Ok((
        PyArray2::from_vec2(py, &to_2d(&dx, ny, nx))?,
        PyArray2::from_vec2(py, &to_2d(&dy, ny, nx))?,
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar function
// ═══════════════════════════════════════════════════════════════════════

/// Coriolis parameter: f = 2 * Omega * sin(latitude).
#[pyfunction]
fn coriolis_parameter(lat_deg: f64) -> f64 {
    metrust::calc::kinematics::coriolis_parameter(lat_deg)
}

// ═══════════════════════════════════════════════════════════════════════
// 1D element-wise array functions
// ═══════════════════════════════════════════════════════════════════════

/// Scalar wind speed: sqrt(u^2 + v^2).
#[pyfunction]
fn wind_speed<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::wind_speed(
        u.as_slice()?, v.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

/// Meteorological wind direction (degrees, 0 = from north, 90 = from east).
#[pyfunction]
fn wind_direction<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::wind_direction(
        u.as_slice()?, v.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

/// Convert wind speed and meteorological direction to (u, v) components.
#[pyfunction]
fn wind_components<'py>(
    py: Python<'py>,
    speed: PyReadonlyArray1<f64>,
    direction: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (u, v) = metrust::calc::kinematics::wind_components(
        speed.as_slice()?, direction.as_slice()?,
    );
    Ok((u.into_pyarray(py), v.into_pyarray(py)))
}

/// Absolute momentum: M = u - f * y.
#[pyfunction]
fn absolute_momentum<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    lats: PyReadonlyArray1<f64>,
    y_distances: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::absolute_momentum(
        u.as_slice()?, lats.as_slice()?, y_distances.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

/// Kinematic flux: element-wise product of a velocity component and a scalar.
#[pyfunction]
fn kinematic_flux<'py>(
    py: Python<'py>,
    v_component: PyReadonlyArray1<f64>,
    scalar: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::kinematic_flux(
        v_component.as_slice()?, scalar.as_slice()?,
    );
    Ok(result.into_pyarray(py))
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-section functions
// ═══════════════════════════════════════════════════════════════════════

/// Decompose wind into components parallel and perpendicular to a
/// cross-section line.
///
/// Returns (parallel, perpendicular) as a tuple of 1D arrays.
#[pyfunction]
fn cross_section_components<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (par, perp) = metrust::calc::kinematics::cross_section_components(
        u.as_slice()?, v.as_slice()?,
        (start_lat, start_lon),
        (end_lat, end_lon),
    );
    Ok((par.into_pyarray(py), perp.into_pyarray(py)))
}

/// Compute tangent and normal unit vectors for a cross-section line.
///
/// Returns ((tangent_east, tangent_north), (normal_east, normal_north)).
#[pyfunction]
fn unit_vectors_from_cross_section(
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
) -> ((f64, f64), (f64, f64)) {
    metrust::calc::kinematics::unit_vectors_from_cross_section(
        (start_lat, start_lon),
        (end_lat, end_lon),
    )
}

/// Component of wind tangential (parallel) to a cross-section line.
#[pyfunction]
fn tangential_component<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::tangential_component(
        u.as_slice()?, v.as_slice()?,
        (start_lat, start_lon),
        (end_lat, end_lon),
    );
    Ok(result.into_pyarray(py))
}

/// Component of wind normal (perpendicular) to a cross-section line.
#[pyfunction]
fn normal_component<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    start_lat: f64,
    start_lon: f64,
    end_lat: f64,
    end_lon: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = metrust::calc::kinematics::normal_component(
        u.as_slice()?, v.as_slice()?,
        (start_lat, start_lon),
        (end_lat, end_lon),
    );
    Ok(result.into_pyarray(py))
}

// ═══════════════════════════════════════════════════════════════════════
// Potential vorticity
// ═══════════════════════════════════════════════════════════════════════

/// Baroclinic (Ertel) potential vorticity on a 2D isobaric slice.
///
/// `pressure` is a 2-element list [p_below, p_above] in Pa.
#[pyfunction]
fn potential_vorticity_baroclinic<'py>(
    py: Python<'py>,
    potential_temp: PyReadonlyArray2<f64>,
    pressure: [f64; 2],
    theta_below: PyReadonlyArray2<f64>,
    theta_above: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = potential_temp.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::potential_vorticity_baroclinic(
        potential_temp.as_slice()?,
        &pressure,
        theta_below.as_slice()?,
        theta_above.as_slice()?,
        u.as_slice()?,
        v.as_slice()?,
        lats.as_slice()?,
        nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

/// Barotropic potential vorticity: absolute vorticity / layer depth.
#[pyfunction]
fn potential_vorticity_barotropic<'py>(
    py: Python<'py>,
    heights: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    lats: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = heights.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let result = metrust::calc::kinematics::potential_vorticity_barotropic(
        heights.as_slice()?,
        u.as_slice()?,
        v.as_slice()?,
        lats.as_slice()?,
        nx, ny, dx, dy,
    );
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}

// ═══════════════════════════════════════════════════════════════════════
// Module registration
// ═══════════════════════════════════════════════════════════════════════

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // 2D grid functions
    parent.add_function(wrap_pyfunction!(divergence, parent)?)?;
    parent.add_function(wrap_pyfunction!(vorticity, parent)?)?;
    parent.add_function(wrap_pyfunction!(absolute_vorticity, parent)?)?;
    parent.add_function(wrap_pyfunction!(advection, parent)?)?;
    parent.add_function(wrap_pyfunction!(frontogenesis, parent)?)?;
    parent.add_function(wrap_pyfunction!(geostrophic_wind, parent)?)?;
    parent.add_function(wrap_pyfunction!(ageostrophic_wind, parent)?)?;
    parent.add_function(wrap_pyfunction!(q_vector, parent)?)?;
    parent.add_function(wrap_pyfunction!(q_vector_convergence, parent)?)?;
    parent.add_function(wrap_pyfunction!(stretching_deformation, parent)?)?;
    parent.add_function(wrap_pyfunction!(shearing_deformation, parent)?)?;
    parent.add_function(wrap_pyfunction!(total_deformation, parent)?)?;
    parent.add_function(wrap_pyfunction!(curvature_vorticity, parent)?)?;
    parent.add_function(wrap_pyfunction!(shear_vorticity, parent)?)?;
    parent.add_function(wrap_pyfunction!(temperature_advection, parent)?)?;
    parent.add_function(wrap_pyfunction!(moisture_advection, parent)?)?;
    parent.add_function(wrap_pyfunction!(inertial_advective_wind, parent)?)?;
    parent.add_function(wrap_pyfunction!(vector_derivative, parent)?)?;
    parent.add_function(wrap_pyfunction!(gradient_x, parent)?)?;
    parent.add_function(wrap_pyfunction!(gradient_y, parent)?)?;
    parent.add_function(wrap_pyfunction!(laplacian, parent)?)?;
    parent.add_function(wrap_pyfunction!(first_derivative, parent)?)?;
    parent.add_function(wrap_pyfunction!(second_derivative, parent)?)?;
    parent.add_function(wrap_pyfunction!(lat_lon_grid_deltas, parent)?)?;

    // Scalar
    parent.add_function(wrap_pyfunction!(coriolis_parameter, parent)?)?;

    // 1D element-wise
    parent.add_function(wrap_pyfunction!(wind_speed, parent)?)?;
    parent.add_function(wrap_pyfunction!(wind_direction, parent)?)?;
    parent.add_function(wrap_pyfunction!(wind_components, parent)?)?;
    parent.add_function(wrap_pyfunction!(absolute_momentum, parent)?)?;
    parent.add_function(wrap_pyfunction!(kinematic_flux, parent)?)?;

    // Cross-section
    parent.add_function(wrap_pyfunction!(cross_section_components, parent)?)?;
    parent.add_function(wrap_pyfunction!(unit_vectors_from_cross_section, parent)?)?;
    parent.add_function(wrap_pyfunction!(tangential_component, parent)?)?;
    parent.add_function(wrap_pyfunction!(normal_component, parent)?)?;

    // Potential vorticity
    parent.add_function(wrap_pyfunction!(potential_vorticity_baroclinic, parent)?)?;
    parent.add_function(wrap_pyfunction!(potential_vorticity_barotropic, parent)?)?;

    Ok(())
}
