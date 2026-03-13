use pyo3::prelude::*;

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Universal ────────────────────────────────────────────────────
    parent.add("R", metrust::constants::R)?;
    parent.add("stefan_boltzmann", metrust::constants::STEFAN_BOLTZMANN)?;

    // ── Earth ────────────────────────────────────────────────────────
    parent.add("earth_avg_radius", metrust::constants::EARTH_AVG_RADIUS)?;
    parent.add("Re", metrust::constants::Re)?;
    parent.add("noaa_mean_earth_radius", metrust::constants::NOAA_MEAN_EARTH_RADIUS)?;

    parent.add("earth_gravity", metrust::constants::EARTH_GRAVITY)?;
    parent.add("g", metrust::constants::g)?;
    parent.add("G", metrust::constants::g)?;
    parent.add("earth_gravitational_acceleration", metrust::constants::EARTH_GRAVITATIONAL_ACCELERATION)?;

    parent.add("omega", metrust::constants::OMEGA)?;
    parent.add("earth_avg_density", metrust::constants::EARTH_AVG_DENSITY)?;
    parent.add("earth_max_declination", metrust::constants::EARTH_MAX_DECLINATION)?;

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

    // ── Water / moist thermodynamics ─────────────────────────────────
    parent.add("Rv", metrust::constants::Rv)?;
    parent.add("water_gas_constant", metrust::constants::WATER_GAS_CONSTANT)?;

    parent.add("Cp_v", metrust::constants::CP_V)?;
    parent.add("water_specific_heat_vapor", metrust::constants::WATER_SPECIFIC_HEAT_VAPOR)?;

    parent.add("rho_l", metrust::constants::RHO_L)?;

    parent.add("Lv", metrust::constants::Lv)?;
    parent.add("water_heat_vaporization", metrust::constants::WATER_HEAT_VAPORIZATION)?;

    parent.add("Lf", metrust::constants::Lf)?;
    parent.add("water_heat_fusion", metrust::constants::WATER_HEAT_FUSION)?;

    parent.add("Ls", metrust::constants::Ls)?;
    parent.add("water_heat_sublimation", metrust::constants::WATER_HEAT_SUBLIMATION)?;

    parent.add("Cp_l", metrust::constants::CP_L)?;
    parent.add("water_specific_heat_liquid", metrust::constants::WATER_SPECIFIC_HEAT_LIQUID)?;

    parent.add("T_freeze", metrust::constants::T_FREEZE)?;

    parent.add("molecular_weight_water", metrust::constants::MOLECULAR_WEIGHT_WATER)?;
    parent.add("water_molecular_weight", metrust::constants::WATER_MOLECULAR_WEIGHT)?;

    // ── Standard atmosphere ──────────────────────────────────────────
    parent.add("P0", metrust::constants::P_STP)?;
    parent.add("P_stp", metrust::constants::P_STP)?;
    parent.add("T_stp", metrust::constants::T_STP)?;

    Ok(())
}
