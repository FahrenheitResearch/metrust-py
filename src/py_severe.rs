use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ===========================================================================
// Point-based severe weather parameters (from metrust::calc::severe)
// ===========================================================================

/// Significant Tornado Parameter (STP).
///
/// Combines mixed-layer CAPE, LCL height, 0-1 km storm-relative helicity,
/// and 0-6 km bulk shear into a single composite. Values above 1.0 are
/// increasingly favorable for significant (EF2+) tornadoes.
#[pyfunction]
#[pyo3(text_signature = "(mlcape, lcl_height_m, srh_0_1km, bulk_shear_0_6km_ms)")]
fn significant_tornado_parameter(
    mlcape: f64,
    lcl_height_m: f64,
    srh_0_1km: f64,
    bulk_shear_0_6km_ms: f64,
) -> f64 {
    metrust::calc::significant_tornado_parameter(mlcape, lcl_height_m, srh_0_1km, bulk_shear_0_6km_ms)
}

/// Supercell Composite Parameter (SCP).
///
/// Combines most-unstable CAPE, effective-layer storm-relative helicity,
/// and effective bulk shear. Values >= 1.0 favor supercells.
#[pyfunction]
#[pyo3(text_signature = "(mucape, srh_eff, bulk_shear_eff_ms)")]
fn supercell_composite_parameter(mucape: f64, srh_eff: f64, bulk_shear_eff_ms: f64) -> f64 {
    metrust::calc::supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff_ms)
}

/// Critical angle between storm-relative inflow and 0-500 m shear vector.
///
/// An angle near 90 degrees most favors low-level mesocyclone development.
/// Returns degrees [0, 180].
#[pyfunction]
#[pyo3(text_signature = "(storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m)")]
fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    metrust::calc::critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m)
}

/// Boyden Index: (Z700 - Z1000) / 10 - T700 - 200.
///
/// - z1000: Geopotential height at 1000 hPa (meters)
/// - z700: Geopotential height at 700 hPa (meters)
/// - t700: Temperature at 700 hPa (Celsius)
#[pyfunction]
#[pyo3(text_signature = "(z1000, z700, t700)")]
fn boyden_index(z1000: f64, z700: f64, t700: f64) -> f64 {
    metrust::calc::boyden_index(z1000, z700, t700)
}

/// Bulk Richardson Number: CAPE / (0.5 * shear^2).
///
/// - cape: CAPE (J/kg)
/// - shear_06_ms: 0-6 km bulk shear magnitude (m/s)
#[pyfunction]
#[pyo3(text_signature = "(cape, shear_06_ms)")]
fn bulk_richardson_number(cape: f64, shear_06_ms: f64) -> f64 {
    metrust::calc::bulk_richardson_number(cape, shear_06_ms)
}

/// Convective Inhibition Depth: depth (hPa) from the surface to the LFC
/// where the parcel is negatively buoyant.
///
/// Profiles are surface-first (decreasing pressure).
/// p in hPa, t and td in Celsius.
#[pyfunction]
#[pyo3(text_signature = "(p, t, td)")]
fn convective_inhibition_depth(
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    td: PyReadonlyArray1<f64>,
) -> f64 {
    metrust::calc::convective_inhibition_depth(
        p.as_slice().unwrap(),
        t.as_slice().unwrap(),
        td.as_slice().unwrap(),
    )
}

/// Dendritic Growth Zone: pressure bounds of the -12C to -18C layer.
///
/// Returns (p_top, p_bottom) in hPa. If the profile never enters the
/// -12 to -18 range, returns (NaN, NaN).
///
/// t_profile: Temperature (Celsius), p_profile: Pressure (hPa).
/// Profiles are surface-first (decreasing pressure).
#[pyfunction]
#[pyo3(text_signature = "(t_profile, p_profile)")]
fn dendritic_growth_zone(
    t_profile: PyReadonlyArray1<f64>,
    p_profile: PyReadonlyArray1<f64>,
) -> (f64, f64) {
    metrust::calc::dendritic_growth_zone(
        t_profile.as_slice().unwrap(),
        p_profile.as_slice().unwrap(),
    )
}

/// Fosberg Fire Weather Index (FFWI).
///
/// - t_f: Temperature (Fahrenheit)
/// - rh: Relative humidity (percent, 0-100)
/// - wspd_mph: Wind speed (mph)
#[pyfunction]
#[pyo3(text_signature = "(t_f, rh, wspd_mph)")]
fn fosberg_fire_weather_index(t_f: f64, rh: f64, wspd_mph: f64) -> f64 {
    metrust::calc::fosberg_fire_weather_index(t_f, rh, wspd_mph)
}

/// Freezing Rain Composite.
///
/// Returns a 0-1 value representing freezing rain likelihood based on
/// warm nose characteristics and precipitation type.
///
/// - t_profile: Temperature (Celsius), p_profile: Pressure (hPa)
/// - precip_type: 0=none, 1=rain, 2=snow, 3=ice_pellets, 4=freezing_rain
#[pyfunction]
#[pyo3(text_signature = "(t_profile, p_profile, precip_type)")]
fn freezing_rain_composite(
    t_profile: PyReadonlyArray1<f64>,
    p_profile: PyReadonlyArray1<f64>,
    precip_type: u8,
) -> f64 {
    metrust::calc::freezing_rain_composite(
        t_profile.as_slice().unwrap(),
        p_profile.as_slice().unwrap(),
        precip_type,
    )
}

/// Haines Index (Low Elevation variant).
///
/// Uses 950 and 850 hPa levels. Returns 2-6. Inputs in Celsius.
#[pyfunction]
#[pyo3(text_signature = "(t_950, t_850, td_850)")]
fn haines_index(t_950: f64, t_850: f64, td_850: f64) -> u8 {
    metrust::calc::haines_index(t_950, t_850, td_850)
}

/// Hot-Dry-Windy Index (HDW): VPD * wind_speed.
///
/// - t_c: Temperature (Celsius)
/// - rh: Relative humidity (0-100)
/// - wspd_ms: Wind speed (m/s)
/// - vpd: Vapor pressure deficit (hPa). If 0, computed from T and RH.
#[pyfunction]
#[pyo3(text_signature = "(t_c, rh, wspd_ms, vpd)")]
fn hot_dry_windy(t_c: f64, rh: f64, wspd_ms: f64, vpd: f64) -> f64 {
    metrust::calc::hot_dry_windy(t_c, rh, wspd_ms, vpd)
}

/// Check for a warm nose: a layer above the surface where T > 0C.
///
/// Returns true if there is a below-freezing layer followed by an
/// above-freezing layer aloft.
///
/// t_profile: Temperature (Celsius), p_profile: Pressure (hPa).
/// Profiles are surface-first (decreasing pressure).
#[pyfunction]
#[pyo3(text_signature = "(t_profile, p_profile)")]
fn warm_nose_check(
    t_profile: PyReadonlyArray1<f64>,
    p_profile: PyReadonlyArray1<f64>,
) -> bool {
    metrust::calc::warm_nose_check(
        t_profile.as_slice().unwrap(),
        p_profile.as_slice().unwrap(),
    )
}

/// Galvez-Davison Index (GDI) for tropical convection potential.
///
/// All temperature and dewpoint inputs in Celsius. sst is sea surface
/// temperature in Celsius.
#[pyfunction]
#[pyo3(text_signature = "(t950, t850, t700, t500, td950, td850, td700, sst)")]
fn galvez_davison_index(
    t950: f64,
    t850: f64,
    t700: f64,
    t500: f64,
    td950: f64,
    td850: f64,
    td700: f64,
    sst: f64,
) -> f64 {
    metrust::calc::galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst)
}

// ===========================================================================
// Grid-based stability indices (from metrust::calc::severe::grid)
// ===========================================================================

/// Showalter Index on a profile: lift 850 hPa parcel to 500 hPa.
///
/// Profiles surface-first (decreasing pressure). p in hPa, t and td in Celsius.
#[pyfunction]
#[pyo3(text_signature = "(p, t, td)")]
fn showalter_index(
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    td: PyReadonlyArray1<f64>,
) -> f64 {
    metrust::calc::severe::grid::showalter_index(
        p.as_slice().unwrap(),
        t.as_slice().unwrap(),
        td.as_slice().unwrap(),
    )
}

/// Lifted Index on a profile: lift surface parcel to 500 hPa.
///
/// Profiles surface-first (decreasing pressure). p in hPa, t and td in Celsius.
#[pyfunction]
#[pyo3(text_signature = "(p, t, td)")]
fn lifted_index(
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    td: PyReadonlyArray1<f64>,
) -> f64 {
    metrust::calc::severe::grid::lifted_index(
        p.as_slice().unwrap(),
        t.as_slice().unwrap(),
        td.as_slice().unwrap(),
    )
}

/// K-Index: (T850 - T500) + Td850 - (T700 - Td700). All Celsius.
#[pyfunction]
#[pyo3(text_signature = "(t850, t700, t500, td850, td700)")]
fn k_index(t850: f64, t700: f64, t500: f64, td850: f64, td700: f64) -> f64 {
    metrust::calc::severe::grid::k_index(t850, t700, t500, td850, td700)
}

/// Total Totals Index: (T850 - T500) + (Td850 - T500). All Celsius.
#[pyfunction]
#[pyo3(text_signature = "(t850, t500, td850)")]
fn total_totals(t850: f64, t500: f64, td850: f64) -> f64 {
    metrust::calc::severe::grid::total_totals(t850, t500, td850)
}

/// Cross Totals: Td850 - T500. All Celsius.
#[pyfunction]
#[pyo3(text_signature = "(td850, t500)")]
fn cross_totals(td850: f64, t500: f64) -> f64 {
    metrust::calc::severe::grid::cross_totals(td850, t500)
}

/// Vertical Totals: T850 - T500. All Celsius.
#[pyfunction]
#[pyo3(text_signature = "(t850, t500)")]
fn vertical_totals(t850: f64, t500: f64) -> f64 {
    metrust::calc::severe::grid::vertical_totals(t850, t500)
}

/// SWEAT (Severe Weather Threat) Index.
///
/// - tt: Total Totals index value
/// - td850: 850 hPa dewpoint (Celsius)
/// - wspd850, wspd500: Wind speeds at 850/500 hPa (knots)
/// - wdir850, wdir500: Wind directions at 850/500 hPa (degrees)
#[pyfunction]
#[pyo3(text_signature = "(tt, td850, wspd850, wdir850, wspd500, wdir500)")]
fn sweat_index(
    tt: f64,
    td850: f64,
    wspd850: f64,
    wdir850: f64,
    wspd500: f64,
    wdir500: f64,
) -> f64 {
    metrust::calc::severe::grid::sweat_index(tt, td850, wspd850, wdir850, wspd500, wdir500)
}

// ===========================================================================
// Grid-based 3-D compute functions (from metrust::calc::severe::grid)
// ===========================================================================

/// Compute CAPE and CIN on a 3-D grid.
///
/// Returns (cape, cin, lcl_p, lfc_p) as four 1-D arrays of length nx*ny.
///
/// - pressure_3d, temperature_c_3d, qvapor_3d, height_agl_3d: flattened [nz][ny][nx]
/// - psfc, t2, q2: flattened [ny][nx] surface fields
/// - parcel_type: "surface", "mixed_layer", or "most_unstable"
#[pyfunction]
#[pyo3(text_signature = "(pressure_3d, temperature_c_3d, qvapor_3d, height_agl_3d, psfc, t2, q2, nx, ny, nz, parcel_type)")]
fn compute_cape_cin<'py>(
    py: Python<'py>,
    pressure_3d: PyReadonlyArray1<f64>,
    temperature_c_3d: PyReadonlyArray1<f64>,
    qvapor_3d: PyReadonlyArray1<f64>,
    height_agl_3d: PyReadonlyArray1<f64>,
    psfc: PyReadonlyArray1<f64>,
    t2: PyReadonlyArray1<f64>,
    q2: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    parcel_type: &str,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let (cape, cin, lcl_p, lfc_p) = metrust::calc::severe::grid::compute_cape_cin(
        pressure_3d.as_slice().unwrap(),
        temperature_c_3d.as_slice().unwrap(),
        qvapor_3d.as_slice().unwrap(),
        height_agl_3d.as_slice().unwrap(),
        psfc.as_slice().unwrap(),
        t2.as_slice().unwrap(),
        q2.as_slice().unwrap(),
        nx,
        ny,
        nz,
        parcel_type,
    );
    (
        cape.into_pyarray(py),
        cin.into_pyarray(py),
        lcl_p.into_pyarray(py),
        lfc_p.into_pyarray(py),
    )
}

/// Compute storm-relative helicity on a 3-D grid.
///
/// Returns a 1-D array of length nx*ny.
///
/// - u_3d, v_3d, height_agl_3d: flattened [nz][ny][nx]
/// - top_m: integration depth (e.g. 1000.0 for 0-1 km, 3000.0 for 0-3 km)
#[pyfunction]
#[pyo3(text_signature = "(u_3d, v_3d, height_agl_3d, nx, ny, nz, top_m)")]
fn compute_srh<'py>(
    py: Python<'py>,
    u_3d: PyReadonlyArray1<f64>,
    v_3d: PyReadonlyArray1<f64>,
    height_agl_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    top_m: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_srh(
        u_3d.as_slice().unwrap(),
        v_3d.as_slice().unwrap(),
        height_agl_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
        top_m,
    );
    result.into_pyarray(py)
}

/// Compute bulk wind shear on a 3-D grid.
///
/// Returns a 1-D array of length nx*ny.
///
/// - u_3d, v_3d, height_agl_3d: flattened [nz][ny][nx]
/// - bottom_m, top_m: shear layer bounds in meters AGL
#[pyfunction]
#[pyo3(text_signature = "(u_3d, v_3d, height_agl_3d, nx, ny, nz, bottom_m, top_m)")]
fn compute_shear<'py>(
    py: Python<'py>,
    u_3d: PyReadonlyArray1<f64>,
    v_3d: PyReadonlyArray1<f64>,
    height_agl_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_m: f64,
    top_m: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_shear(
        u_3d.as_slice().unwrap(),
        v_3d.as_slice().unwrap(),
        height_agl_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
        bottom_m,
        top_m,
    );
    result.into_pyarray(py)
}

/// Compute environmental lapse rate on a 3-D grid.
///
/// Returns a 1-D array of length nx*ny (C/km).
///
/// - temperature_c_3d, qvapor_3d, height_agl_3d: flattened [nz][ny][nx]
/// - bottom_km, top_km: layer bounds in km AGL
#[pyfunction]
#[pyo3(text_signature = "(temperature_c_3d, qvapor_3d, height_agl_3d, nx, ny, nz, bottom_km, top_km)")]
fn compute_lapse_rate<'py>(
    py: Python<'py>,
    temperature_c_3d: PyReadonlyArray1<f64>,
    qvapor_3d: PyReadonlyArray1<f64>,
    height_agl_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_km: f64,
    top_km: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_lapse_rate(
        temperature_c_3d.as_slice().unwrap(),
        qvapor_3d.as_slice().unwrap(),
        height_agl_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
        bottom_km,
        top_km,
    );
    result.into_pyarray(py)
}

/// Compute precipitable water on a 3-D grid.
///
/// Returns a 1-D array of length nx*ny (mm).
///
/// - qvapor_3d: Mixing ratio (kg/kg), flattened [nz][ny][nx]
/// - pressure_3d: Full pressure (Pa), flattened [nz][ny][nx]
#[pyfunction]
#[pyo3(text_signature = "(qvapor_3d, pressure_3d, nx, ny, nz)")]
fn compute_pw<'py>(
    py: Python<'py>,
    qvapor_3d: PyReadonlyArray1<f64>,
    pressure_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_pw(
        qvapor_3d.as_slice().unwrap(),
        pressure_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
    );
    result.into_pyarray(py)
}

// ===========================================================================
// Grid-based 2-D composite parameters (from metrust::calc::severe::grid)
// ===========================================================================

/// Compute STP on pre-computed 2-D fields.
///
/// All inputs are flattened 1-D arrays of length nx*ny.
#[pyfunction]
#[pyo3(text_signature = "(cape, lcl, srh_1km, shear_6km)")]
fn compute_stp<'py>(
    py: Python<'py>,
    cape: PyReadonlyArray1<f64>,
    lcl: PyReadonlyArray1<f64>,
    srh_1km: PyReadonlyArray1<f64>,
    shear_6km: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_stp(
        cape.as_slice().unwrap(),
        lcl.as_slice().unwrap(),
        srh_1km.as_slice().unwrap(),
        shear_6km.as_slice().unwrap(),
    );
    result.into_pyarray(py)
}

/// Compute SCP on pre-computed 2-D fields.
///
/// All inputs are flattened 1-D arrays of length nx*ny.
#[pyfunction]
#[pyo3(text_signature = "(mucape, srh_3km, shear_6km)")]
fn compute_scp<'py>(
    py: Python<'py>,
    mucape: PyReadonlyArray1<f64>,
    srh_3km: PyReadonlyArray1<f64>,
    shear_6km: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_scp(
        mucape.as_slice().unwrap(),
        srh_3km.as_slice().unwrap(),
        shear_6km.as_slice().unwrap(),
    );
    result.into_pyarray(py)
}

/// Energy Helicity Index: EHI = (CAPE * SRH) / 160000.
///
/// Inputs are pre-computed 2-D fields, flattened 1-D arrays.
#[pyfunction]
#[pyo3(text_signature = "(cape, srh)")]
fn compute_ehi<'py>(
    py: Python<'py>,
    cape: PyReadonlyArray1<f64>,
    srh: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::compute_ehi(
        cape.as_slice().unwrap(),
        srh.as_slice().unwrap(),
    );
    result.into_pyarray(py)
}

/// Significant Hail Parameter (SHIP) on 2-D grids.
///
/// All inputs are flattened 1-D arrays of length nx*ny.
/// - cape: MUCAPE (J/kg)
/// - shear06: 0-6 km bulk shear (m/s)
/// - t500: Temperature at 500 hPa (Celsius)
/// - lr_700_500: 700-500 hPa lapse rate (C/km)
/// - mr: Mixing ratio (g/kg)
#[pyfunction]
#[pyo3(text_signature = "(cape, shear06, t500, lr_700_500, mr, nx, ny)")]
fn significant_hail_parameter<'py>(
    py: Python<'py>,
    cape: PyReadonlyArray1<f64>,
    shear06: PyReadonlyArray1<f64>,
    t500: PyReadonlyArray1<f64>,
    lr_700_500: PyReadonlyArray1<f64>,
    mr: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::significant_hail_parameter(
        cape.as_slice().unwrap(),
        shear06.as_slice().unwrap(),
        t500.as_slice().unwrap(),
        lr_700_500.as_slice().unwrap(),
        mr.as_slice().unwrap(),
        nx,
        ny,
    );
    result.into_pyarray(py)
}

/// Derecho Composite Parameter (DCP) on 2-D grids.
///
/// DCP = (DCAPE/980) * (MUCAPE/2000) * (SHEAR_06/20) * (MU_MR/11)
///
/// All inputs are flattened 1-D arrays of length nx*ny.
#[pyfunction]
#[pyo3(text_signature = "(dcape, mu_cape, shear06, mu_mixing_ratio, nx, ny)")]
fn derecho_composite_parameter<'py>(
    py: Python<'py>,
    dcape: PyReadonlyArray1<f64>,
    mu_cape: PyReadonlyArray1<f64>,
    shear06: PyReadonlyArray1<f64>,
    mu_mixing_ratio: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::derecho_composite_parameter(
        dcape.as_slice().unwrap(),
        mu_cape.as_slice().unwrap(),
        shear06.as_slice().unwrap(),
        mu_mixing_ratio.as_slice().unwrap(),
        nx,
        ny,
    );
    result.into_pyarray(py)
}

/// Enhanced Supercell Composite Parameter (SCP) on 2-D grids.
///
/// SCP = (MUCAPE/1000) * (SRH/50) * (SHEAR_06/40) * CIN_term
///
/// All inputs are flattened 1-D arrays of length nx*ny.
#[pyfunction]
#[pyo3(text_signature = "(mu_cape, srh, shear_06, mu_cin, nx, ny)")]
fn grid_supercell_composite_parameter<'py>(
    py: Python<'py>,
    mu_cape: PyReadonlyArray1<f64>,
    srh: PyReadonlyArray1<f64>,
    shear_06: PyReadonlyArray1<f64>,
    mu_cin: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::supercell_composite_parameter(
        mu_cape.as_slice().unwrap(),
        srh.as_slice().unwrap(),
        shear_06.as_slice().unwrap(),
        mu_cin.as_slice().unwrap(),
        nx,
        ny,
    );
    result.into_pyarray(py)
}

/// Critical Angle on 2-D grids.
///
/// Returns angle in degrees (0-180). Values near 90 degrees favor tornadogenesis.
///
/// All inputs are flattened 1-D arrays of length nx*ny.
#[pyfunction]
#[pyo3(text_signature = "(u_storm, v_storm, u_shear, v_shear, nx, ny)")]
fn grid_critical_angle<'py>(
    py: Python<'py>,
    u_storm: PyReadonlyArray1<f64>,
    v_storm: PyReadonlyArray1<f64>,
    u_shear: PyReadonlyArray1<f64>,
    v_shear: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::critical_angle(
        u_storm.as_slice().unwrap(),
        v_storm.as_slice().unwrap(),
        u_shear.as_slice().unwrap(),
        v_shear.as_slice().unwrap(),
        nx,
        ny,
    );
    result.into_pyarray(py)
}

// ===========================================================================
// Grid-based reflectivity composites (from metrust::calc::severe::grid)
// ===========================================================================

/// Composite reflectivity (max in column) in dBZ from a 3-D reflectivity field.
///
/// refl_3d: Reflectivity in dBZ, flattened [nz][ny][nx].
#[pyfunction]
#[pyo3(text_signature = "(refl_3d, nx, ny, nz)")]
fn composite_reflectivity_from_refl<'py>(
    py: Python<'py>,
    refl_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::composite_reflectivity_from_refl(
        refl_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
    );
    result.into_pyarray(py)
}

/// Composite reflectivity (max in column) in dBZ from hydrometeor mixing ratios.
///
/// All 3-D fields flattened [nz][ny][nx]:
/// - pressure_3d: Pa
/// - temperature_c_3d: Celsius
/// - qrain_3d, qsnow_3d, qgraup_3d: kg/kg
#[pyfunction]
#[pyo3(text_signature = "(pressure_3d, temperature_c_3d, qrain_3d, qsnow_3d, qgraup_3d, nx, ny, nz)")]
fn composite_reflectivity_from_hydrometeors<'py>(
    py: Python<'py>,
    pressure_3d: PyReadonlyArray1<f64>,
    temperature_c_3d: PyReadonlyArray1<f64>,
    qrain_3d: PyReadonlyArray1<f64>,
    qsnow_3d: PyReadonlyArray1<f64>,
    qgraup_3d: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = metrust::calc::severe::grid::composite_reflectivity_from_hydrometeors(
        pressure_3d.as_slice().unwrap(),
        temperature_c_3d.as_slice().unwrap(),
        qrain_3d.as_slice().unwrap(),
        qsnow_3d.as_slice().unwrap(),
        qgraup_3d.as_slice().unwrap(),
        nx,
        ny,
        nz,
    );
    result.into_pyarray(py)
}

// ===========================================================================
// Module registration
// ===========================================================================

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // Point-based severe weather parameters
    parent.add_function(wrap_pyfunction!(significant_tornado_parameter, parent)?)?;
    parent.add_function(wrap_pyfunction!(supercell_composite_parameter, parent)?)?;
    parent.add_function(wrap_pyfunction!(critical_angle, parent)?)?;
    parent.add_function(wrap_pyfunction!(boyden_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(bulk_richardson_number, parent)?)?;
    parent.add_function(wrap_pyfunction!(convective_inhibition_depth, parent)?)?;
    parent.add_function(wrap_pyfunction!(dendritic_growth_zone, parent)?)?;
    parent.add_function(wrap_pyfunction!(fosberg_fire_weather_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(freezing_rain_composite, parent)?)?;
    parent.add_function(wrap_pyfunction!(haines_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(hot_dry_windy, parent)?)?;
    parent.add_function(wrap_pyfunction!(warm_nose_check, parent)?)?;
    parent.add_function(wrap_pyfunction!(galvez_davison_index, parent)?)?;

    // Grid-based stability indices
    parent.add_function(wrap_pyfunction!(showalter_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(lifted_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(k_index, parent)?)?;
    parent.add_function(wrap_pyfunction!(total_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(cross_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(vertical_totals, parent)?)?;
    parent.add_function(wrap_pyfunction!(sweat_index, parent)?)?;

    // Grid-based 3-D compute functions
    parent.add_function(wrap_pyfunction!(compute_cape_cin, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_srh, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_shear, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_lapse_rate, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_pw, parent)?)?;

    // Grid-based 2-D composite parameters
    parent.add_function(wrap_pyfunction!(compute_stp, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_scp, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_ehi, parent)?)?;
    parent.add_function(wrap_pyfunction!(significant_hail_parameter, parent)?)?;
    parent.add_function(wrap_pyfunction!(derecho_composite_parameter, parent)?)?;
    parent.add_function(wrap_pyfunction!(grid_supercell_composite_parameter, parent)?)?;
    parent.add_function(wrap_pyfunction!(grid_critical_angle, parent)?)?;

    // Grid-based reflectivity composites
    parent.add_function(wrap_pyfunction!(composite_reflectivity_from_refl, parent)?)?;
    parent.add_function(wrap_pyfunction!(composite_reflectivity_from_hydrometeors, parent)?)?;

    Ok(())
}
