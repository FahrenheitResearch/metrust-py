use pyo3::prelude::*;

/// Convert pressure to geometric height using the US Standard Atmosphere 1976.
#[pyfunction]
#[pyo3(text_signature = "(pressure_hpa)")]
fn pressure_to_height_std(pressure_hpa: f64) -> f64 {
    metrust::calc::pressure_to_height_std(pressure_hpa)
}

/// Convert geometric height to pressure using the US Standard Atmosphere 1976.
#[pyfunction]
#[pyo3(text_signature = "(height_m)")]
fn height_to_pressure_std(height_m: f64) -> f64 {
    metrust::calc::height_to_pressure_std(height_m)
}

/// Convert altimeter setting to station pressure.
#[pyfunction]
#[pyo3(text_signature = "(altimeter_hpa, elevation_m)")]
fn altimeter_to_station_pressure(altimeter_hpa: f64, elevation_m: f64) -> f64 {
    metrust::calc::altimeter_to_station_pressure(altimeter_hpa, elevation_m)
}

/// Convert station pressure to altimeter setting.
#[pyfunction]
#[pyo3(text_signature = "(station_hpa, elevation_m)")]
fn station_to_altimeter_pressure(station_hpa: f64, elevation_m: f64) -> f64 {
    metrust::calc::station_to_altimeter_pressure(station_hpa, elevation_m)
}

/// Convert altimeter setting to sea-level pressure accounting for temperature.
#[pyfunction]
#[pyo3(text_signature = "(alt_hpa, elevation_m, t_c)")]
fn altimeter_to_sea_level_pressure(alt_hpa: f64, elevation_m: f64, t_c: f64) -> f64 {
    metrust::calc::altimeter_to_sea_level_pressure(alt_hpa, elevation_m, t_c)
}

/// Convert a sigma coordinate to pressure.
#[pyfunction]
#[pyo3(text_signature = "(sigma, psfc_hpa, ptop_hpa)")]
fn sigma_to_pressure(sigma: f64, psfc_hpa: f64, ptop_hpa: f64) -> f64 {
    metrust::calc::sigma_to_pressure(sigma, psfc_hpa, ptop_hpa)
}

/// Heat index using the Rothfusz regression (NWS formula).
#[pyfunction]
#[pyo3(text_signature = "(temperature_c, relative_humidity_pct)")]
fn heat_index(temperature_c: f64, relative_humidity_pct: f64) -> f64 {
    metrust::calc::heat_index(temperature_c, relative_humidity_pct)
}

/// Wind chill index using the NWS/Environment Canada formula.
#[pyfunction]
#[pyo3(text_signature = "(temperature_c, wind_speed_ms)")]
fn windchill(temperature_c: f64, wind_speed_ms: f64) -> f64 {
    metrust::calc::windchill(temperature_c, wind_speed_ms)
}

/// Apparent temperature combining heat index and wind chill.
#[pyfunction]
#[pyo3(text_signature = "(temperature_c, rh_pct, wind_speed_ms)")]
fn apparent_temperature(temperature_c: f64, rh_pct: f64, wind_speed_ms: f64) -> f64 {
    metrust::calc::apparent_temperature(temperature_c, rh_pct, wind_speed_ms)
}

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(pressure_to_height_std, parent)?)?;
    parent.add_function(wrap_pyfunction!(height_to_pressure_std, parent)?)?;
    parent.add_function(wrap_pyfunction!(altimeter_to_station_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(station_to_altimeter_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(altimeter_to_sea_level_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(sigma_to_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(heat_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(windchill, parent)?)?;
    parent.add_function(wrap_pyfunction!(apparent_temperature, parent)?)?;
    Ok(())
}
