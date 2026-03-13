use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

// =============================================================================
// Scalar thermodynamic functions
// =============================================================================

/// Potential temperature (K) from pressure (hPa) and temperature (Celsius).
#[pyfunction]
fn potential_temperature(py: Python, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::potential_temperature(pressure, temperature)
}

/// Equivalent potential temperature (K) from pressure (hPa), temperature (C), dewpoint (C).
#[pyfunction]
fn equivalent_potential_temperature(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::equivalent_potential_temperature(pressure, temperature, dewpoint)
}

/// Saturation vapor pressure (hPa) from temperature (Celsius).
#[pyfunction]
fn saturation_vapor_pressure(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::saturation_vapor_pressure(temperature)
}

/// Saturation mixing ratio (g/kg) from pressure (hPa) and temperature (Celsius).
#[pyfunction]
fn saturation_mixing_ratio(py: Python, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::saturation_mixing_ratio(pressure, temperature)
}

/// Wet bulb temperature (Celsius) from pressure (hPa), temperature (C), dewpoint (C).
#[pyfunction]
fn wet_bulb_temperature(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::wet_bulb_temperature(pressure, temperature, dewpoint)
}

/// Level of Free Convection. Returns Option<(pressure_hPa, temperature_C)>.
#[pyfunction]
fn lfc(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<Option<(f64, f64)>> {
    let _ = py;
    Ok(metrust::calc::thermo::lfc(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Equilibrium Level. Returns Option<(pressure_hPa, temperature_C)>.
#[pyfunction]
fn el(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<Option<(f64, f64)>> {
    let _ = py;
    Ok(metrust::calc::thermo::el(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Convective Condensation Level. Returns Option<(pressure_hPa, temperature_C)>.
#[pyfunction]
fn ccl(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<Option<(f64, f64)>> {
    let _ = py;
    Ok(metrust::calc::thermo::ccl(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Lifted Index from profiles.
#[pyfunction]
fn lifted_index(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::lifted_index(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Air density (kg/m^3) from pressure (hPa), temperature (C), mixing ratio (g/kg).
#[pyfunction]
fn density(py: Python, pressure: f64, temperature: f64, mixing_ratio: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::density(pressure, temperature, mixing_ratio)
}

/// Dewpoint (C) from temperature (C) and relative humidity (%).
#[pyfunction]
fn dewpoint_from_rh(py: Python, temperature: f64, rh: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::dewpoint_from_rh(temperature, rh)
}

/// Relative humidity (%) from temperature (C) and dewpoint (C).
#[pyfunction]
fn rh_from_dewpoint(py: Python, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::rh_from_dewpoint(temperature, dewpoint)
}

/// Mixing ratio (g/kg) from specific humidity (kg/kg).
#[pyfunction]
fn mixing_ratio_from_specific_humidity(py: Python, specific_humidity: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::mixing_ratio_from_specific_humidity(specific_humidity)
}

/// LCL pressure (hPa) from surface pressure (hPa), temperature (C), dewpoint (C).
#[pyfunction]
fn lcl_pressure(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::lcl_pressure(pressure, temperature, dewpoint)
}

/// Parcel temperature profile (C) from pressure levels, surface T, and surface Td.
#[pyfunction]
fn parcel_profile(py: Python, pressure: PyReadonlyArray1<f64>, t_surface: f64, td_surface: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::parcel_profile(
        pressure.as_slice()?,
        t_surface,
        td_surface,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Dry adiabatic lapse: temperature (C) at each pressure level.
#[pyfunction]
fn dry_lapse(py: Python, pressure: PyReadonlyArray1<f64>, t_surface: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::dry_lapse(
        pressure.as_slice()?,
        t_surface,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Moist adiabatic lapse: temperature (C) at each pressure level.
#[pyfunction]
fn moist_lapse(py: Python, pressure: PyReadonlyArray1<f64>, t_start: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::moist_lapse(
        pressure.as_slice()?,
        t_start,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Convert Celsius to Kelvin.
#[pyfunction]
fn celsius_to_kelvin(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::celsius_to_kelvin(temperature)
}

/// Convert Kelvin to Celsius.
#[pyfunction]
fn kelvin_to_celsius(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::kelvin_to_celsius(temperature)
}

/// Convert Celsius to Fahrenheit.
#[pyfunction]
fn celsius_to_fahrenheit(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::celsius_to_fahrenheit(temperature)
}

/// Convert Fahrenheit to Celsius.
#[pyfunction]
fn fahrenheit_to_celsius(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::fahrenheit_to_celsius(temperature)
}

/// Virtual temperature (C) from temperature (C), pressure (hPa), dewpoint (C).
#[pyfunction]
fn virtual_temp(py: Python, temperature: f64, pressure: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::virtual_temp(temperature, pressure, dewpoint)
}

/// Theta-e (K) from pressure (hPa), temperature (C), dewpoint (C).
#[pyfunction]
fn thetae(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::thetae(pressure, temperature, dewpoint)
}

/// Full CAPE/CIN core computation.
/// Returns (cape, cin, h_lcl, h_lfc) in (J/kg, J/kg, m AGL, m AGL).
#[pyfunction]
#[pyo3(signature = (pressure, temperature, dewpoint, height_agl, psfc, t2m, td2m, parcel_type, ml_depth, mu_depth, top_m=None))]
fn cape_cin_core(
    py: Python,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
    height_agl: PyReadonlyArray1<f64>,
    psfc: f64,
    t2m: f64,
    td2m: f64,
    parcel_type: &str,
    ml_depth: f64,
    mu_depth: f64,
    top_m: Option<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::cape_cin_core(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
        height_agl.as_slice()?,
        psfc,
        t2m,
        td2m,
        parcel_type,
        ml_depth,
        mu_depth,
        top_m,
    ))
}

/// Temperature (K) from potential temperature (K) and pressure (hPa).
#[pyfunction]
fn temperature_from_potential_temperature(py: Python, pressure: f64, theta: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::temperature_from_potential_temperature(pressure, theta)
}

/// Dry static energy (J/kg) from height (m) and temperature (K).
#[pyfunction]
fn dry_static_energy(py: Python, height: f64, temperature_k: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::dry_static_energy(height, temperature_k)
}

/// Moist static energy (J/kg) from height (m), temperature (K), specific humidity (kg/kg).
#[pyfunction]
fn moist_static_energy(py: Python, height: f64, temperature_k: f64, specific_humidity: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::moist_static_energy(height, temperature_k, specific_humidity)
}

/// Static stability parameter at each level.
#[pyfunction]
fn static_stability(py: Python, pressure: PyReadonlyArray1<f64>, temperature_k: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::static_stability(
        pressure.as_slice()?,
        temperature_k.as_slice()?,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Scale height (m) from temperature (K).
#[pyfunction]
fn scale_height(py: Python, temperature_k: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::scale_height(temperature_k)
}

/// Pressure-weighted mean of a quantity.
#[pyfunction]
fn mean_pressure_weighted(py: Python, pressure: PyReadonlyArray1<f64>, values: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::mean_pressure_weighted(
        pressure.as_slice()?,
        values.as_slice()?,
    ))
}

/// Convert geopotential (m^2/s^2) to geopotential height (m).
#[pyfunction]
fn geopotential_to_height(py: Python, geopotential: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::geopotential_to_height(geopotential)
}

/// Convert geopotential height (m) to geopotential (m^2/s^2).
#[pyfunction]
fn height_to_geopotential(py: Python, height: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::height_to_geopotential(height)
}

/// Convert vertical velocity w (m/s) to omega (Pa/s).
#[pyfunction]
fn vertical_velocity_pressure(py: Python, w_ms: f64, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::vertical_velocity_pressure(w_ms, pressure, temperature)
}

/// Convert omega (Pa/s) to vertical velocity w (m/s).
#[pyfunction]
fn vertical_velocity(py: Python, omega: f64, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::vertical_velocity(omega, pressure, temperature)
}

/// Exner function: (p/p0)^(R/Cp).
#[pyfunction]
fn exner_function(py: Python, pressure: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::exner_function(pressure)
}

/// Montgomery streamfunction (J/kg) on an isentropic surface.
#[pyfunction]
fn montgomery_streamfunction(py: Python, theta: f64, pressure: f64, temperature_k: f64, height: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::montgomery_streamfunction(theta, pressure, temperature_k, height)
}

/// Dewpoint (C) from vapor pressure (hPa).
#[pyfunction]
fn dewpoint(py: Python, vapor_pressure: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::dewpoint_from_vapor_pressure(vapor_pressure)
}

/// Mixing ratio (g/kg) from relative humidity (%).
#[pyfunction]
fn mixing_ratio_from_relative_humidity(py: Python, pressure: f64, temperature: f64, rh: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::mixing_ratio_from_relative_humidity(pressure, temperature, rh)
}

/// Relative humidity (%) from mixing ratio (g/kg).
#[pyfunction]
fn relative_humidity_from_mixing_ratio(py: Python, pressure: f64, temperature: f64, mixing_ratio: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio)
}

/// Relative humidity (%) from specific humidity (kg/kg).
#[pyfunction]
fn relative_humidity_from_specific_humidity(py: Python, pressure: f64, temperature: f64, specific_humidity: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity)
}

/// Specific humidity (kg/kg) from dewpoint (C) and pressure (hPa).
#[pyfunction]
fn specific_humidity_from_dewpoint(py: Python, pressure: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::specific_humidity_from_dewpoint(pressure, dewpoint)
}

/// Dewpoint (C) from specific humidity (kg/kg) and pressure (hPa).
#[pyfunction]
fn dewpoint_from_specific_humidity(py: Python, pressure: f64, specific_humidity: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::dewpoint_from_specific_humidity(pressure, specific_humidity)
}

/// Specific humidity (kg/kg) from pressure (hPa) and mixing ratio (g/kg).
#[pyfunction]
fn specific_humidity(py: Python, pressure: f64, mixing_ratio: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::specific_humidity(pressure, mixing_ratio)
}

/// Frost point temperature (C) from temperature (C) and relative humidity (%).
#[pyfunction]
fn frost_point(py: Python, temperature: f64, rh: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::frost_point(temperature, rh)
}

/// Psychrometric vapor pressure (hPa) from dry bulb (C), wet bulb (C), pressure (hPa).
#[pyfunction]
fn psychrometric_vapor_pressure(py: Python, temperature: f64, wet_bulb: f64, pressure: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::psychrometric_vapor_pressure(temperature, wet_bulb, pressure)
}

/// Virtual potential temperature (K) from pressure (hPa), temperature (C), mixing ratio (g/kg).
#[pyfunction]
fn virtual_potential_temperature(py: Python, pressure: f64, temperature: f64, mixing_ratio: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::virtual_potential_temperature(pressure, temperature, mixing_ratio)
}

/// Wet bulb potential temperature (K) from pressure (hPa), temperature (C), dewpoint (C).
#[pyfunction]
fn wet_bulb_potential_temperature(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::wet_bulb_potential_temperature(pressure, temperature, dewpoint)
}

/// Saturation equivalent potential temperature (K) from pressure (hPa), temperature (C).
#[pyfunction]
fn saturation_equivalent_potential_temperature(py: Python, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::saturation_equivalent_potential_temperature(pressure, temperature)
}

/// Hypsometric thickness (m) between two pressure levels.
#[pyfunction]
fn thickness_hypsometric(py: Python, p_bottom: f64, p_top: f64, t_mean_k: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::thickness_hypsometric(p_bottom, p_top, t_mean_k)
}

/// Find intersections between two curves over a shared x-axis.
/// Returns list of (x, y) tuples.
#[pyfunction]
fn find_intersections(py: Python, x: PyReadonlyArray1<f64>, y1: PyReadonlyArray1<f64>, y2: PyReadonlyArray1<f64>) -> PyResult<Vec<(f64, f64)>> {
    let _ = py;
    Ok(metrust::calc::thermo::find_intersections(
        x.as_slice()?,
        y1.as_slice()?,
        y2.as_slice()?,
    ))
}

/// Extract values within a pressure layer. Returns (pressures, values).
#[pyfunction]
fn get_layer(py: Python, pressure: PyReadonlyArray1<f64>, values: PyReadonlyArray1<f64>, p_bottom: f64, p_top: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (p_out, v_out) = metrust::calc::thermo::get_layer(
        pressure.as_slice()?,
        values.as_slice()?,
        p_bottom,
        p_top,
    );
    Ok((
        PyArray1::from_vec(py, p_out).into(),
        PyArray1::from_vec(py, v_out).into(),
    ))
}

/// Extract height values within a pressure layer. Returns (pressures, heights).
#[pyfunction]
fn get_layer_heights(py: Python, pressure: PyReadonlyArray1<f64>, heights: PyReadonlyArray1<f64>, p_bottom: f64, p_top: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (p_out, z_out) = metrust::calc::thermo::get_layer_heights(
        pressure.as_slice()?,
        heights.as_slice()?,
        p_bottom,
        p_top,
    );
    Ok((
        PyArray1::from_vec(py, p_out).into(),
        PyArray1::from_vec(py, z_out).into(),
    ))
}

/// Reduce point density: returns boolean mask (true = keep).
#[pyfunction]
fn reduce_point_density(py: Python, lats: PyReadonlyArray1<f64>, lons: PyReadonlyArray1<f64>, radius_deg: f64) -> PyResult<Vec<bool>> {
    let _ = py;
    Ok(metrust::calc::thermo::reduce_point_density(
        lats.as_slice()?,
        lons.as_slice()?,
        radius_deg,
    ))
}

/// Potential vorticity on isentropic surfaces (PVU).
#[pyfunction]
fn potential_vorticity_baroclinic(
    py: Python,
    theta: PyReadonlyArray1<f64>,
    pressure: PyReadonlyArray1<f64>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    lats: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::potential_vorticity_baroclinic(
        theta.as_slice()?,
        pressure.as_slice()?,
        u.as_slice()?,
        v.as_slice()?,
        lats.as_slice()?,
        nx,
        ny,
        dx,
        dy,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Surface-based CAPE and CIN. Returns (cape, cin).
#[pyfunction]
fn surface_based_cape_cin(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::surface_based_cape_cin(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Mixed-layer CAPE and CIN. Returns (cape, cin).
#[pyfunction]
fn mixed_layer_cape_cin(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>, depth_hpa: f64) -> PyResult<(f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::mixed_layer_cape_cin(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
        depth_hpa,
    ))
}

/// Most-unstable CAPE and CIN. Returns (cape, cin).
#[pyfunction]
fn most_unstable_cape_cin(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::most_unstable_cape_cin(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Mixed-layer average of a quantity in the lowest depth_hpa hPa.
#[pyfunction]
fn mixed_layer(py: Python, pressure: PyReadonlyArray1<f64>, values: PyReadonlyArray1<f64>, depth_hpa: f64) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::mixed_layer(
        pressure.as_slice()?,
        values.as_slice()?,
        depth_hpa,
    ))
}

/// Mixed-layer parcel. Returns (pressure, temperature, dewpoint).
#[pyfunction]
fn get_mixed_layer_parcel(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>, depth: f64) -> PyResult<(f64, f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::get_mixed_layer_parcel(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
        depth,
    ))
}

/// Most-unstable parcel. Returns (pressure, temperature, dewpoint).
#[pyfunction]
fn get_most_unstable_parcel(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>, depth: f64) -> PyResult<(f64, f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::get_most_unstable_parcel(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
        depth,
    ))
}

/// Galvez-Davison Index for tropical convection.
#[pyfunction]
fn galvez_davison_index(py: Python, t950: f64, t850: f64, t700: f64, t500: f64, td950: f64, td850: f64, td700: f64, sst: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst)
}

// =============================================================================
// Wrapper re-exports (MetPy name differs from wx-math name)
// =============================================================================

/// Mixing ratio (g/kg) from pressure (hPa) and temperature (Celsius).
#[pyfunction]
fn mixing_ratio(py: Python, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::mixing_ratio(pressure, temperature)
}

/// Dewpoint (C) from temperature (C) and relative humidity (%).
#[pyfunction]
fn dewpoint_from_relative_humidity(py: Python, temperature: f64, rh: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::dewpoint_from_relative_humidity(temperature, rh)
}

/// Relative humidity (%) from temperature (C) and dewpoint (C).
#[pyfunction]
fn relative_humidity_from_dewpoint(py: Python, temperature: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::relative_humidity_from_dewpoint(temperature, dewpoint)
}

/// Vapor pressure (hPa) from dewpoint temperature (Celsius).
#[pyfunction]
fn vapor_pressure(py: Python, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::vapor_pressure(dewpoint)
}

/// Virtual temperature (C) from temperature (C), pressure (hPa), dewpoint (C).
#[pyfunction]
fn virtual_temperature(py: Python, temperature: f64, pressure: f64, dewpoint: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::virtual_temperature(temperature, pressure, dewpoint)
}

/// Virtual temperature (C) from temperature (C), dewpoint (C), pressure (hPa).
#[pyfunction]
fn virtual_temperature_from_dewpoint(py: Python, temperature: f64, dewpoint: f64, pressure: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::virtual_temperature_from_dewpoint(temperature, dewpoint, pressure)
}

/// LCL via dry-adiabatic ascent. Returns (p_lcl, t_lcl) in (hPa, C).
#[pyfunction]
fn lcl(py: Python, pressure: f64, temperature: f64, dewpoint: f64) -> (f64, f64) {
    let _ = py;
    metrust::calc::thermo::lcl(pressure, temperature, dewpoint)
}

/// CAPE and CIN for a sounding column.
/// Returns (cape, cin, h_lcl, h_lfc) in (J/kg, J/kg, m AGL, m AGL).
#[pyfunction]
#[pyo3(signature = (pressure, temperature, dewpoint, height_agl, psfc, t2m, td2m, parcel_type, ml_depth, mu_depth, top_m=None))]
fn cape_cin(
    py: Python,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
    height_agl: PyReadonlyArray1<f64>,
    psfc: f64,
    t2m: f64,
    td2m: f64,
    parcel_type: &str,
    ml_depth: f64,
    mu_depth: f64,
    top_m: Option<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let _ = py;
    Ok(metrust::calc::thermo::cape_cin(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
        height_agl.as_slice()?,
        psfc,
        t2m,
        td2m,
        parcel_type,
        ml_depth,
        mu_depth,
        top_m,
    ))
}

// =============================================================================
// New implementations in metrust::calc::thermo
// =============================================================================

/// Showalter Index from profiles.
#[pyfunction]
fn showalter_index(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::showalter_index(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// K-Index from standard-level temperatures (all Celsius).
#[pyfunction]
fn k_index(py: Python, t850: f64, td850: f64, t700: f64, td700: f64, t500: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::k_index(t850, td850, t700, td700, t500)
}

/// Vertical Totals: T850 - T500 (Celsius).
#[pyfunction]
fn vertical_totals(py: Python, t850: f64, t500: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::vertical_totals(t850, t500)
}

/// Cross Totals: Td850 - T500 (Celsius).
#[pyfunction]
fn cross_totals(py: Python, td850: f64, t500: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::cross_totals(td850, t500)
}

/// Total Totals Index (Celsius).
#[pyfunction]
fn total_totals(py: Python, t850: f64, td850: f64, t500: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::total_totals(t850, td850, t500)
}

/// SWEAT Index (Severe Weather Threat Index).
#[pyfunction]
fn sweat_index(py: Python, t850: f64, td850: f64, t500: f64, dd850: f64, dd500: f64, ff850: f64, ff500: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
}

/// Downdraft CAPE (J/kg) from profiles.
#[pyfunction]
fn downdraft_cape(py: Python, pressure: PyReadonlyArray1<f64>, temperature: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::downdraft_cape(
        pressure.as_slice()?,
        temperature.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Brunt-Vaisala frequency (s^-1) at each level from height (m) and potential temperature (K).
#[pyfunction]
fn brunt_vaisala_frequency(py: Python, height: PyReadonlyArray1<f64>, potential_temperature: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::brunt_vaisala_frequency(
        height.as_slice()?,
        potential_temperature.as_slice()?,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Brunt-Vaisala period (s) at each level from height (m) and potential temperature (K).
#[pyfunction]
fn brunt_vaisala_period(py: Python, height: PyReadonlyArray1<f64>, potential_temperature: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::brunt_vaisala_period(
        height.as_slice()?,
        potential_temperature.as_slice()?,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Brunt-Vaisala frequency squared (s^-2) at each level.
#[pyfunction]
fn brunt_vaisala_frequency_squared(py: Python, height: PyReadonlyArray1<f64>, potential_temperature: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::brunt_vaisala_frequency_squared(
        height.as_slice()?,
        potential_temperature.as_slice()?,
    );
    Ok(PyArray1::from_vec(py, result).into())
}

/// Precipitable water (mm) from pressure (hPa) and dewpoint (C) profiles.
#[pyfunction]
fn precipitable_water(py: Python, pressure: PyReadonlyArray1<f64>, dewpoint: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::precipitable_water(
        pressure.as_slice()?,
        dewpoint.as_slice()?,
    ))
}

/// Parcel profile with LCL inserted. Returns (pressures, temperatures).
#[pyfunction]
fn parcel_profile_with_lcl(py: Python, pressure: PyReadonlyArray1<f64>, t_surface: f64, td_surface: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (p_out, t_out) = metrust::calc::thermo::parcel_profile_with_lcl(
        pressure.as_slice()?,
        t_surface,
        td_surface,
    );
    Ok((
        PyArray1::from_vec(py, p_out).into(),
        PyArray1::from_vec(py, t_out).into(),
    ))
}

/// Hypsometric thickness (m) using mean layer temperature.
#[pyfunction]
fn thickness_hydrostatic(py: Python, p_bottom: f64, p_top: f64, t_mean_k: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::thickness_hydrostatic(p_bottom, p_top, t_mean_k)
}

/// New pressure (hPa) after ascending/descending by a height increment (m).
#[pyfunction]
fn add_height_to_pressure(py: Python, pressure: f64, delta_h: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::add_height_to_pressure(pressure, delta_h)
}

/// New height (m) after a pressure increment (hPa).
#[pyfunction]
fn add_pressure_to_height(py: Python, height: f64, delta_p: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::add_pressure_to_height(height, delta_p)
}

/// Perturbation (anomaly) from the mean for each element.
#[pyfunction]
fn get_perturbation(py: Python, values: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let result = metrust::calc::thermo::get_perturbation(values.as_slice()?);
    Ok(PyArray1::from_vec(py, result).into())
}

/// Gas constant for moist air (J/(kg K)) from mixing ratio (kg/kg).
#[pyfunction]
fn moist_air_gas_constant(py: Python, mixing_ratio_kgkg: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::moist_air_gas_constant(mixing_ratio_kgkg)
}

/// Specific heat at constant pressure for moist air (J/(kg K)) from mixing ratio (kg/kg).
#[pyfunction]
fn moist_air_specific_heat_pressure(py: Python, mixing_ratio_kgkg: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::moist_air_specific_heat_pressure(mixing_ratio_kgkg)
}

/// Poisson exponent (kappa) for moist air from mixing ratio (kg/kg).
#[pyfunction]
fn moist_air_poisson_exponent(py: Python, mixing_ratio_kgkg: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::moist_air_poisson_exponent(mixing_ratio_kgkg)
}

/// Latent heat of vaporization (J/kg) from temperature (C).
#[pyfunction]
fn water_latent_heat_vaporization(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::water_latent_heat_vaporization(temperature)
}

/// Latent heat of melting (J/kg) from temperature (C).
#[pyfunction]
fn water_latent_heat_melting(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::water_latent_heat_melting(temperature)
}

/// Latent heat of sublimation (J/kg) from temperature (C).
#[pyfunction]
fn water_latent_heat_sublimation(py: Python, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::water_latent_heat_sublimation(temperature)
}

/// Relative humidity (%) from dry bulb (C), wet bulb (C), pressure (hPa).
#[pyfunction]
fn relative_humidity_wet_psychrometric(py: Python, temperature: f64, wet_bulb: f64, pressure: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::relative_humidity_wet_psychrometric(temperature, wet_bulb, pressure)
}

/// Trapezoidal weighted average of values over a coordinate.
#[pyfunction]
fn weighted_continuous_average(py: Python, values: PyReadonlyArray1<f64>, weights: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::weighted_continuous_average(
        values.as_slice()?,
        weights.as_slice()?,
    ))
}

/// Isentropic interpolation of 3D fields to theta surfaces.
/// Returns list of numpy arrays: [pressure_on_theta, t_on_theta, field0_on_theta, ...].
#[pyfunction]
fn isentropic_interpolation(
    py: Python,
    theta_levels: PyReadonlyArray1<f64>,
    p_3d: PyReadonlyArray1<f64>,
    t_3d: PyReadonlyArray1<f64>,
    fields: Vec<PyReadonlyArray1<f64>>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    let field_slices: Vec<&[f64]> = fields
        .iter()
        .map(|f| f.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let result = metrust::calc::thermo::isentropic_interpolation(
        theta_levels.as_slice()?,
        p_3d.as_slice()?,
        t_3d.as_slice()?,
        &field_slices,
        nx,
        ny,
        nz,
    );

    Ok(result
        .into_iter()
        .map(|v| PyArray1::from_vec(py, v).into())
        .collect())
}

/// Specific humidity (kg/kg) from mixing ratio (kg/kg).
#[pyfunction]
fn specific_humidity_from_mixing_ratio(py: Python, mixing_ratio: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::specific_humidity_from_mixing_ratio(mixing_ratio)
}

/// Hypsometric thickness (m) from pressure (hPa), temperature (C), and RH (%) profiles.
#[pyfunction]
fn thickness_hydrostatic_from_relative_humidity(
    py: Python,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    relative_humidity: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let _ = py;
    Ok(metrust::calc::thermo::thickness_hydrostatic_from_relative_humidity(
        pressure.as_slice()?,
        temperature.as_slice()?,
        relative_humidity.as_slice()?,
    ))
}

// =============================================================================
// Array variants of scalar thermodynamic functions
// =============================================================================

/// Potential temperature (K) — array version.
#[pyfunction]
fn potential_temperature_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let result: Vec<f64> = p.iter().zip(t.iter())
        .map(|(&p, &t)| metrust::calc::thermo::potential_temperature(p, t))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Equivalent potential temperature (K) — array version.
#[pyfunction]
fn equivalent_potential_temperature_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = p.iter().zip(t.iter().zip(td.iter()))
        .map(|(&p, (&t, &td))| metrust::calc::thermo::equivalent_potential_temperature(p, t, td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Saturation vapor pressure (hPa) — array version.
#[pyfunction]
fn saturation_vapor_pressure_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let result: Vec<f64> = t.iter()
        .map(|&t| metrust::calc::thermo::saturation_vapor_pressure(t))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Saturation mixing ratio (g/kg) — array version.
#[pyfunction]
fn saturation_mixing_ratio_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let result: Vec<f64> = p.iter().zip(t.iter())
        .map(|(&p, &t)| metrust::calc::thermo::saturation_mixing_ratio(p, t))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Wet bulb temperature (C) — array version.
#[pyfunction]
fn wet_bulb_temperature_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = p.iter().zip(t.iter().zip(td.iter()))
        .map(|(&p, (&t, &td))| metrust::calc::thermo::wet_bulb_temperature(p, t, td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Virtual temperature (C) — array version.
#[pyfunction]
fn virtual_temp_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<f64>,
    pressure: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let p = pressure.as_slice()?;
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = t.iter().zip(p.iter().zip(td.iter()))
        .map(|(&t, (&p, &td))| metrust::calc::thermo::virtual_temp(t, p, td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Dewpoint from RH (C) — array version.
#[pyfunction]
fn dewpoint_from_rh_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<f64>,
    rh: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let r = rh.as_slice()?;
    let result: Vec<f64> = t.iter().zip(r.iter())
        .map(|(&t, &r)| metrust::calc::thermo::dewpoint_from_rh(t, r))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Relative humidity from dewpoint (%) — array version.
#[pyfunction]
fn rh_from_dewpoint_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = t.iter().zip(td.iter())
        .map(|(&t, &td)| metrust::calc::thermo::rh_from_dewpoint(t, td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Mixing ratio (g/kg) from pressure (hPa) and dewpoint (C) — array version.
#[pyfunction]
fn mixing_ratio_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = p.iter().zip(td.iter())
        .map(|(&p, &td)| metrust::calc::thermo::mixing_ratio(p, td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Vapor pressure (hPa) from dewpoint (C) — array version.
#[pyfunction]
fn vapor_pressure_array<'py>(
    py: Python<'py>,
    dewpoint: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let td = dewpoint.as_slice()?;
    let result: Vec<f64> = td.iter()
        .map(|&td| metrust::calc::thermo::vapor_pressure(td))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Density (kg/m^3) — array version.
#[pyfunction]
fn density_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
    mixing_ratio: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let w = mixing_ratio.as_slice()?;
    let result: Vec<f64> = p.iter().zip(t.iter().zip(w.iter()))
        .map(|(&p, (&t, &w))| metrust::calc::thermo::density(p, t, w))
        .collect();
    Ok(result.into_pyarray(py))
}

/// Exner function — array version.
#[pyfunction]
fn exner_function_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let result: Vec<f64> = p.iter()
        .map(|&p| metrust::calc::thermo::exner_function(p))
        .collect();
    Ok(result.into_pyarray(py))
}

// =============================================================================
// Registration
// =============================================================================

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // Direct re-exports (scalar)
    parent.add_function(wrap_pyfunction!(potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(equivalent_potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(saturation_vapor_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(saturation_mixing_ratio, parent)?)?;
    parent.add_function(wrap_pyfunction!(wet_bulb_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(lfc, parent)?)?;
    parent.add_function(wrap_pyfunction!(el, parent)?)?;
    parent.add_function(wrap_pyfunction!(ccl, parent)?)?;
    parent.add_function(wrap_pyfunction!(lifted_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(density, parent)?)?;
    parent.add_function(wrap_pyfunction!(dewpoint_from_rh, parent)?)?;
    parent.add_function(wrap_pyfunction!(rh_from_dewpoint, parent)?)?;
    parent.add_function(wrap_pyfunction!(mixing_ratio_from_specific_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(lcl_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(parcel_profile, parent)?)?;
    parent.add_function(wrap_pyfunction!(dry_lapse, parent)?)?;
    parent.add_function(wrap_pyfunction!(moist_lapse, parent)?)?;
    parent.add_function(wrap_pyfunction!(celsius_to_kelvin, parent)?)?;
    parent.add_function(wrap_pyfunction!(kelvin_to_celsius, parent)?)?;
    parent.add_function(wrap_pyfunction!(celsius_to_fahrenheit, parent)?)?;
    parent.add_function(wrap_pyfunction!(fahrenheit_to_celsius, parent)?)?;
    parent.add_function(wrap_pyfunction!(virtual_temp, parent)?)?;
    parent.add_function(wrap_pyfunction!(thetae, parent)?)?;
    parent.add_function(wrap_pyfunction!(cape_cin_core, parent)?)?;
    parent.add_function(wrap_pyfunction!(temperature_from_potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(dry_static_energy, parent)?)?;
    parent.add_function(wrap_pyfunction!(moist_static_energy, parent)?)?;
    parent.add_function(wrap_pyfunction!(static_stability, parent)?)?;
    parent.add_function(wrap_pyfunction!(scale_height, parent)?)?;
    parent.add_function(wrap_pyfunction!(mean_pressure_weighted, parent)?)?;
    parent.add_function(wrap_pyfunction!(geopotential_to_height, parent)?)?;
    parent.add_function(wrap_pyfunction!(height_to_geopotential, parent)?)?;
    parent.add_function(wrap_pyfunction!(vertical_velocity_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(vertical_velocity, parent)?)?;
    parent.add_function(wrap_pyfunction!(exner_function, parent)?)?;
    parent.add_function(wrap_pyfunction!(montgomery_streamfunction, parent)?)?;
    parent.add_function(wrap_pyfunction!(dewpoint, parent)?)?;
    parent.add_function(wrap_pyfunction!(mixing_ratio_from_relative_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(relative_humidity_from_mixing_ratio, parent)?)?;
    parent.add_function(wrap_pyfunction!(relative_humidity_from_specific_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(specific_humidity_from_dewpoint, parent)?)?;
    parent.add_function(wrap_pyfunction!(dewpoint_from_specific_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(specific_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(frost_point, parent)?)?;
    parent.add_function(wrap_pyfunction!(psychrometric_vapor_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(virtual_potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(wet_bulb_potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(saturation_equivalent_potential_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(thickness_hypsometric, parent)?)?;
    parent.add_function(wrap_pyfunction!(find_intersections, parent)?)?;
    parent.add_function(wrap_pyfunction!(get_layer, parent)?)?;
    parent.add_function(wrap_pyfunction!(get_layer_heights, parent)?)?;
    parent.add_function(wrap_pyfunction!(reduce_point_density, parent)?)?;
    parent.add_function(wrap_pyfunction!(potential_vorticity_baroclinic, parent)?)?;
    parent.add_function(wrap_pyfunction!(isentropic_interpolation, parent)?)?;
    parent.add_function(wrap_pyfunction!(surface_based_cape_cin, parent)?)?;
    parent.add_function(wrap_pyfunction!(mixed_layer_cape_cin, parent)?)?;
    parent.add_function(wrap_pyfunction!(most_unstable_cape_cin, parent)?)?;
    parent.add_function(wrap_pyfunction!(mixed_layer, parent)?)?;
    parent.add_function(wrap_pyfunction!(get_mixed_layer_parcel, parent)?)?;
    parent.add_function(wrap_pyfunction!(get_most_unstable_parcel, parent)?)?;
    parent.add_function(wrap_pyfunction!(galvez_davison_index, parent)?)?;

    // Wrapper re-exports (MetPy-name aliases)
    parent.add_function(wrap_pyfunction!(mixing_ratio, parent)?)?;
    parent.add_function(wrap_pyfunction!(dewpoint_from_relative_humidity, parent)?)?;
    parent.add_function(wrap_pyfunction!(relative_humidity_from_dewpoint, parent)?)?;
    parent.add_function(wrap_pyfunction!(vapor_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(virtual_temperature, parent)?)?;
    parent.add_function(wrap_pyfunction!(virtual_temperature_from_dewpoint, parent)?)?;
    parent.add_function(wrap_pyfunction!(lcl, parent)?)?;
    parent.add_function(wrap_pyfunction!(cape_cin, parent)?)?;

    // New implementations (stability indices)
    parent.add_function(wrap_pyfunction!(showalter_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(k_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(vertical_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(cross_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(total_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(sweat_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(downdraft_cape, parent)?)?;
    parent.add_function(wrap_pyfunction!(brunt_vaisala_frequency, parent)?)?;
    parent.add_function(wrap_pyfunction!(brunt_vaisala_period, parent)?)?;
    parent.add_function(wrap_pyfunction!(brunt_vaisala_frequency_squared, parent)?)?;
    parent.add_function(wrap_pyfunction!(precipitable_water, parent)?)?;
    parent.add_function(wrap_pyfunction!(parcel_profile_with_lcl, parent)?)?;
    parent.add_function(wrap_pyfunction!(thickness_hydrostatic, parent)?)?;
    parent.add_function(wrap_pyfunction!(add_height_to_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(add_pressure_to_height, parent)?)?;
    parent.add_function(wrap_pyfunction!(get_perturbation, parent)?)?;

    // Moist-air properties and latent heats
    parent.add_function(wrap_pyfunction!(moist_air_gas_constant, parent)?)?;
    parent.add_function(wrap_pyfunction!(moist_air_specific_heat_pressure, parent)?)?;
    parent.add_function(wrap_pyfunction!(moist_air_poisson_exponent, parent)?)?;
    parent.add_function(wrap_pyfunction!(water_latent_heat_vaporization, parent)?)?;
    parent.add_function(wrap_pyfunction!(water_latent_heat_melting, parent)?)?;
    parent.add_function(wrap_pyfunction!(water_latent_heat_sublimation, parent)?)?;
    parent.add_function(wrap_pyfunction!(relative_humidity_wet_psychrometric, parent)?)?;
    parent.add_function(wrap_pyfunction!(weighted_continuous_average, parent)?)?;

    // Humidity / thickness
    parent.add_function(wrap_pyfunction!(specific_humidity_from_mixing_ratio, parent)?)?;
    parent.add_function(wrap_pyfunction!(thickness_hydrostatic_from_relative_humidity, parent)?)?;

    // Array variants of scalar thermo functions
    parent.add_function(wrap_pyfunction!(potential_temperature_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(equivalent_potential_temperature_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(saturation_vapor_pressure_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(saturation_mixing_ratio_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(wet_bulb_temperature_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(virtual_temp_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(dewpoint_from_rh_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(rh_from_dewpoint_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(mixing_ratio_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(vapor_pressure_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(density_array, parent)?)?;
    parent.add_function(wrap_pyfunction!(exner_function_array, parent)?)?;

    Ok(())
}
