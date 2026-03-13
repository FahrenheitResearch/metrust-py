use pyo3::prelude::*;

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Universal ────────────────────────────────────────────────────
    parent.add("R", metrust::constants::R)?;
    parent.add("stefan_boltzmann", metrust::constants::STEFAN_BOLTZMANN)?;
    parent.add("gravitational_constant", metrust::constants::GRAVITATIONAL_CONSTANT)?;

    // ── Earth ────────────────────────────────────────────────────────
    parent.add("earth_avg_radius", metrust::constants::EARTH_AVG_RADIUS)?;
    parent.add("Re", metrust::constants::Re)?;
    parent.add("noaa_mean_earth_radius", metrust::constants::NOAA_MEAN_EARTH_RADIUS)?;

    parent.add("earth_gravity", metrust::constants::EARTH_GRAVITY)?;
    parent.add("g", metrust::constants::g)?;
    parent.add("earth_gravitational_acceleration", metrust::constants::EARTH_GRAVITATIONAL_ACCELERATION)?;

    parent.add("omega", metrust::constants::OMEGA)?;
    parent.add("earth_avg_density", metrust::constants::EARTH_AVG_DENSITY)?;
    parent.add("earth_max_declination", metrust::constants::EARTH_MAX_DECLINATION)?;

    parent.add("GM", metrust::constants::EARTH_GM)?;
    parent.add("earth_mass", metrust::constants::EARTH_MASS)?;
    parent.add("earth_orbit_eccentricity", metrust::constants::EARTH_ORBIT_ECCENTRICITY)?;
    parent.add("earth_sfc_avg_dist_sun", metrust::constants::EARTH_SFC_AVG_DIST_SUN)?;
    parent.add("earth_solar_irradiance", metrust::constants::EARTH_SOLAR_IRRADIANCE)?;

    // ── Dry air ──────────────────────────────────────────────────────
    parent.add("Rd", metrust::constants::Rd)?;
    parent.add("dry_air_gas_constant", metrust::constants::DRY_AIR_GAS_CONSTANT)?;

    parent.add("Cp_d", metrust::constants::Cp_d)?;
    parent.add("dry_air_spec_heat_press", metrust::constants::DRY_AIR_SPEC_HEAT_PRESS)?;

    parent.add("Cv_d", metrust::constants::Cv_d)?;
    parent.add("dry_air_spec_heat_vol", metrust::constants::DRY_AIR_SPEC_HEAT_VOL)?;

    parent.add("rho_d_stp", metrust::constants::RHO_D_STP)?;
    parent.add("dry_air_density_stp", metrust::constants::DRY_AIR_DENSITY_STP)?;

    parent.add("kappa", metrust::constants::KAPPA)?;
    parent.add("poisson_exponent_dry_air", metrust::constants::POISSON_EXPONENT_DRY_AIR)?;

    parent.add("molecular_weight_dry_air", metrust::constants::MOLECULAR_WEIGHT_DRY_AIR)?;
    parent.add("dry_air_molecular_weight", metrust::constants::DRY_AIR_MOLECULAR_WEIGHT)?;

    parent.add("epsilon", metrust::constants::EPSILON)?;

    parent.add("dry_air_spec_heat_ratio", metrust::constants::DRY_AIR_SPEC_HEAT_RATIO)?;
    parent.add("dry_adiabatic_lapse_rate", metrust::constants::DRY_ADIABATIC_LAPSE_RATE)?;

    // ── Water / moist thermodynamics ─────────────────────────────────
    parent.add("Rv", metrust::constants::Rv)?;
    parent.add("water_gas_constant", metrust::constants::WATER_GAS_CONSTANT)?;

    parent.add("Cp_v", metrust::constants::CP_V)?;
    parent.add("water_specific_heat_vapor", metrust::constants::WATER_SPECIFIC_HEAT_VAPOR)?;

    parent.add("Cv_v", metrust::constants::CV_V)?;

    parent.add("rho_l", metrust::constants::RHO_L)?;
    parent.add("rho_i", metrust::constants::RHO_I)?;
    parent.add("density_ice", metrust::constants::RHO_I)?;

    parent.add("Lv", metrust::constants::Lv)?;
    parent.add("water_heat_vaporization", metrust::constants::WATER_HEAT_VAPORIZATION)?;

    parent.add("Lf", metrust::constants::Lf)?;
    parent.add("water_heat_fusion", metrust::constants::WATER_HEAT_FUSION)?;

    parent.add("Ls", metrust::constants::Ls)?;
    parent.add("water_heat_sublimation", metrust::constants::WATER_HEAT_SUBLIMATION)?;

    parent.add("Cp_l", metrust::constants::CP_L)?;
    parent.add("water_specific_heat_liquid", metrust::constants::WATER_SPECIFIC_HEAT_LIQUID)?;

    parent.add("Cp_i", metrust::constants::CP_I)?;
    parent.add("ice_specific_heat", metrust::constants::CP_I)?;

    parent.add("T_freeze", metrust::constants::T_FREEZE)?;
    parent.add("water_triple_point_temperature", metrust::constants::WATER_TRIPLE_POINT_TEMPERATURE)?;
    parent.add("T0", metrust::constants::WATER_TRIPLE_POINT_TEMPERATURE)?;
    parent.add("sat_pressure_0c", metrust::constants::SAT_PRESSURE_0C)?;

    parent.add("wv_specific_heat_ratio", metrust::constants::WV_SPECIFIC_HEAT_RATIO)?;

    parent.add("molecular_weight_water", metrust::constants::MOLECULAR_WEIGHT_WATER)?;
    parent.add("water_molecular_weight", metrust::constants::WATER_MOLECULAR_WEIGHT)?;

    // ── Standard atmosphere ──────────────────────────────────────────
    parent.add("P0", metrust::constants::POT_TEMP_REF_PRESS)?;
    parent.add("pot_temp_ref_press", metrust::constants::POT_TEMP_REF_PRESS)?;
    parent.add("P_stp", metrust::constants::P_STP)?;
    parent.add("T_stp", metrust::constants::T_STP)?;

    Ok(())
}
