use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ─── Element-wise array functions (re-exported from wx_math::dynamics) ───

#[pyfunction]
fn wind_speed<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::wind_speed(u.as_slice().unwrap(), v.as_slice().unwrap());
    result.into_pyarray(py)
}

#[pyfunction]
fn wind_direction<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::wind_direction(u.as_slice().unwrap(), v.as_slice().unwrap());
    result.into_pyarray(py)
}

#[pyfunction]
fn wind_components<'py>(
    py: Python<'py>,
    speed: PyReadonlyArray1<f64>,
    direction: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (u, v) = metrust::calc::wind_components(
        speed.as_slice().unwrap(),
        direction.as_slice().unwrap(),
    );
    (u.into_pyarray(py), v.into_pyarray(py))
}

// ─── Profile-based functions ─────────────────────────────────────────

#[pyfunction]
fn bulk_shear(
    u_prof: PyReadonlyArray1<f64>,
    v_prof: PyReadonlyArray1<f64>,
    height_prof: PyReadonlyArray1<f64>,
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    metrust::calc::bulk_shear(
        u_prof.as_slice().unwrap(),
        v_prof.as_slice().unwrap(),
        height_prof.as_slice().unwrap(),
        bottom_m,
        top_m,
    )
}

#[pyfunction]
fn storm_relative_helicity(
    u_prof: PyReadonlyArray1<f64>,
    v_prof: PyReadonlyArray1<f64>,
    height_prof: PyReadonlyArray1<f64>,
    depth_m: f64,
    storm_u: f64,
    storm_v: f64,
) -> (f64, f64, f64) {
    metrust::calc::storm_relative_helicity(
        u_prof.as_slice().unwrap(),
        v_prof.as_slice().unwrap(),
        height_prof.as_slice().unwrap(),
        depth_m,
        storm_u,
        storm_v,
    )
}

#[pyfunction]
fn mean_wind(
    u_prof: PyReadonlyArray1<f64>,
    v_prof: PyReadonlyArray1<f64>,
    height_prof: PyReadonlyArray1<f64>,
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    metrust::calc::mean_wind(
        u_prof.as_slice().unwrap(),
        v_prof.as_slice().unwrap(),
        height_prof.as_slice().unwrap(),
        bottom_m,
        top_m,
    )
}

#[pyfunction]
fn bunkers_storm_motion(
    p_prof: PyReadonlyArray1<f64>,
    u_prof: PyReadonlyArray1<f64>,
    v_prof: PyReadonlyArray1<f64>,
    height_prof: PyReadonlyArray1<f64>,
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    metrust::calc::bunkers_storm_motion(
        p_prof.as_slice().unwrap(),
        u_prof.as_slice().unwrap(),
        v_prof.as_slice().unwrap(),
        height_prof.as_slice().unwrap(),
    )
}

#[pyfunction]
fn corfidi_storm_motion(
    u_prof: PyReadonlyArray1<f64>,
    v_prof: PyReadonlyArray1<f64>,
    height_prof: PyReadonlyArray1<f64>,
    u_850: f64,
    v_850: f64,
) -> ((f64, f64), (f64, f64)) {
    metrust::calc::corfidi_storm_motion(
        u_prof.as_slice().unwrap(),
        v_prof.as_slice().unwrap(),
        height_prof.as_slice().unwrap(),
        u_850,
        v_850,
    )
}

// ─── Boundary-layer turbulence functions ──────────────────────────────

/// Friction velocity (m/s) from u and w time series (m/s).
#[pyfunction]
fn friction_velocity(
    u: PyReadonlyArray1<f64>,
    w: PyReadonlyArray1<f64>,
) -> f64 {
    metrust::calc::friction_velocity(
        u.as_slice().unwrap(),
        w.as_slice().unwrap(),
    )
}

/// Turbulent kinetic energy (m^2/s^2) from u, v, w time series (m/s).
#[pyfunction]
fn tke(
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    w: PyReadonlyArray1<f64>,
) -> f64 {
    metrust::calc::tke(
        u.as_slice().unwrap(),
        v.as_slice().unwrap(),
        w.as_slice().unwrap(),
    )
}

/// Gradient Richardson number at each level.
#[pyfunction]
fn gradient_richardson_number<'py>(
    py: Python<'py>,
    height: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray1<f64>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::gradient_richardson_number(
        height.as_slice().unwrap(),
        theta.as_slice().unwrap(),
        u.as_slice().unwrap(),
        v.as_slice().unwrap(),
    );
    result.into_pyarray(py)
}

// ─── Module registration ─────────────────────────────────────────────

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(wind_speed, parent)?)?;
    parent.add_function(wrap_pyfunction!(wind_direction, parent)?)?;
    parent.add_function(wrap_pyfunction!(wind_components, parent)?)?;
    parent.add_function(wrap_pyfunction!(bulk_shear, parent)?)?;
    parent.add_function(wrap_pyfunction!(storm_relative_helicity, parent)?)?;
    parent.add_function(wrap_pyfunction!(mean_wind, parent)?)?;
    parent.add_function(wrap_pyfunction!(bunkers_storm_motion, parent)?)?;
    parent.add_function(wrap_pyfunction!(corfidi_storm_motion, parent)?)?;
    parent.add_function(wrap_pyfunction!(friction_velocity, parent)?)?;
    parent.add_function(wrap_pyfunction!(tke, parent)?)?;
    parent.add_function(wrap_pyfunction!(gradient_richardson_number, parent)?)?;
    Ok(())
}
