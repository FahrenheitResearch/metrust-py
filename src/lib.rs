use pyo3::prelude::*;

mod py_thermo;
mod py_wind;
mod py_kinematics;
mod py_severe;
mod py_atmo;
mod py_smooth;
mod py_utils;
mod py_io;
mod py_interpolate;
mod py_constants;

/// The native Rust module exposed as metrust._metrust
#[pymodule]
fn _metrust(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // calc submodule
    let calc = PyModule::new(py, "calc")?;
    py_thermo::register(py, &calc)?;
    py_wind::register(py, &calc)?;
    py_kinematics::register(py, &calc)?;
    py_severe::register(py, &calc)?;
    py_atmo::register(py, &calc)?;
    py_smooth::register(py, &calc)?;
    py_utils::register(py, &calc)?;
    m.add_submodule(&calc)?;

    // io submodule
    let io_mod = PyModule::new(py, "io")?;
    py_io::register(py, &io_mod)?;
    m.add_submodule(&io_mod)?;

    // interpolate submodule
    let interp = PyModule::new(py, "interpolate")?;
    py_interpolate::register(py, &interp)?;
    m.add_submodule(&interp)?;

    // constants submodule
    let constants = PyModule::new(py, "constants")?;
    py_constants::register(py, &constants)?;
    m.add_submodule(&constants)?;

    Ok(())
}
