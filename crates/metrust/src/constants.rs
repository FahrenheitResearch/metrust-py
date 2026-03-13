// Physical constants mirroring MetPy's metpy.constants module.
//
// All values use SI base units unless otherwise noted. Sources follow
// the same references as MetPy: CODATA 2018, the U.S. Standard
// Atmosphere (1976), and the WMO International Meteorological Tables.

// ---------------------------------------------------------------------------
// Universal
// ---------------------------------------------------------------------------

/// Universal gas constant (J mol^-1 K^-1).
pub const R: f64 = 8.314462618;

/// Stefan-Boltzmann constant (W m^-2 K^-4).
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

// ---------------------------------------------------------------------------
// Earth
// ---------------------------------------------------------------------------

/// Mean radius of the Earth (m).
pub const EARTH_AVG_RADIUS: f64 = 6.371229e6;

/// Standard acceleration of gravity (m s^-2).
pub const EARTH_GRAVITY: f64 = 9.80665;

/// Angular velocity of Earth's rotation (rad s^-1).
pub const OMEGA: f64 = 7.2921159e-5;

/// Mean density of the Earth (kg m^-3).
/// Derived from Earth's mass (5.9722e24 kg) and mean radius.
pub const EARTH_AVG_DENSITY: f64 = 5515.0;

/// Maximum solar declination angle (degrees).
pub const EARTH_MAX_DECLINATION: f64 = 23.45;

// ---------------------------------------------------------------------------
// Dry air
// ---------------------------------------------------------------------------

/// Specific gas constant for dry air (J kg^-1 K^-1).
pub const RD: f64 = 287.058;

/// Specific heat at constant pressure for dry air (J kg^-1 K^-1).
pub const CP_D: f64 = 1005.7;

/// Specific heat at constant volume for dry air (J kg^-1 K^-1).
pub const CV_D: f64 = 718.0;

/// Density of dry air at STP (kg m^-3).
/// Calculated as P_stp / (Rd * T_stp).
pub const RHO_D_STP: f64 = P_STP / (RD * T_STP);

/// Poisson constant for dry air (Rd / Cp_d, dimensionless).
pub const KAPPA: f64 = RD / CP_D;

/// Mean molecular weight of dry air (kg mol^-1).
pub const MOLECULAR_WEIGHT_DRY_AIR: f64 = 28.9645e-3;

/// Ratio of the molecular weight of water to the molecular weight of dry
/// air, also equal to Rd / Rv (dimensionless).
pub const EPSILON: f64 = MOLECULAR_WEIGHT_WATER / MOLECULAR_WEIGHT_DRY_AIR;

// ---------------------------------------------------------------------------
// Water / moist thermodynamics
// ---------------------------------------------------------------------------

/// Specific gas constant for water vapour (J kg^-1 K^-1).
pub const RV: f64 = 461.5;

/// Specific heat at constant pressure for water vapour (J kg^-1 K^-1).
pub const CP_V: f64 = 1875.0;

/// Density of liquid water at 0 degC (kg m^-3).
pub const RHO_L: f64 = 999.97;

/// Latent heat of vaporisation at 0 degC (J kg^-1).
pub const LV: f64 = 2.501e6;

/// Latent heat of fusion at 0 degC (J kg^-1).
pub const LF: f64 = 3.34e5;

/// Latent heat of sublimation at 0 degC (J kg^-1).
pub const LS: f64 = 2.834e6;

/// Specific heat of liquid water at 0 degC (J kg^-1 K^-1).
pub const CP_L: f64 = 4218.0;

/// Freezing point of water (K).
pub const T_FREEZE: f64 = 273.15;

/// Mean molecular weight of water (kg mol^-1).
pub const MOLECULAR_WEIGHT_WATER: f64 = 18.015e-3;

// ---------------------------------------------------------------------------
// Standard atmosphere reference values
// ---------------------------------------------------------------------------

/// Standard atmospheric pressure at sea level (Pa).
pub const P_STP: f64 = 101325.0;

/// Standard temperature at sea level (K).
pub const T_STP: f64 = 288.15;

// ---------------------------------------------------------------------------
// MetPy-compatible named constants
// ---------------------------------------------------------------------------
// These mirror the names MetPy exposes (e.g. `mpconsts.water_heat_vaporization`).
// Where a value duplicates an existing constant the compiler will fold them.

/// Latent heat of vaporisation at 0 degC (J kg^-1). Same as `LV`.
pub const WATER_HEAT_VAPORIZATION: f64 = 2.501e6;

/// Latent heat of fusion at 0 degC (J kg^-1). Same as `LF`.
pub const WATER_HEAT_FUSION: f64 = 3.34e5;

/// Latent heat of sublimation at 0 degC (J kg^-1). Same as `LS`.
pub const WATER_HEAT_SUBLIMATION: f64 = 2.834e6;

/// Specific heat of liquid water at 0 degC (J kg^-1 K^-1). Same as `CP_L`.
pub const WATER_SPECIFIC_HEAT_LIQUID: f64 = 4218.0;

/// Specific heat of water vapour at constant pressure (J kg^-1 K^-1). Same as `CP_V`.
pub const WATER_SPECIFIC_HEAT_VAPOR: f64 = 1875.0;

/// Mean molecular weight of water (kg mol^-1). Same as `MOLECULAR_WEIGHT_WATER`.
pub const WATER_MOLECULAR_WEIGHT: f64 = 18.015e-3;

/// Mean molecular weight of dry air (kg mol^-1). Same as `MOLECULAR_WEIGHT_DRY_AIR`.
pub const DRY_AIR_MOLECULAR_WEIGHT: f64 = 28.9645e-3;

/// Specific heat at constant pressure for dry air (J kg^-1 K^-1). Same as `CP_D`.
pub const DRY_AIR_SPEC_HEAT_PRESS: f64 = 1005.7;

/// Specific heat at constant volume for dry air (J kg^-1 K^-1). Same as `CV_D`.
pub const DRY_AIR_SPEC_HEAT_VOL: f64 = 718.0;

/// Density of dry air at STP (kg m^-3).
pub const DRY_AIR_DENSITY_STP: f64 = 1.275;

/// Specific gas constant for dry air (J kg^-1 K^-1). Same as `RD`.
pub const DRY_AIR_GAS_CONSTANT: f64 = 287.058;

/// Specific gas constant for water vapour (J kg^-1 K^-1). Same as `RV`.
pub const WATER_GAS_CONSTANT: f64 = 461.5;

/// NOAA mean radius of the Earth (m). Same as `EARTH_AVG_RADIUS`.
pub const NOAA_MEAN_EARTH_RADIUS: f64 = 6.371229e6;

/// Standard acceleration of gravity (m s^-2). Same as `EARTH_GRAVITY`.
pub const EARTH_GRAVITATIONAL_ACCELERATION: f64 = 9.80665;

/// Poisson exponent for dry air (Rd / Cp_d, dimensionless). Same as `KAPPA`.
pub const POISSON_EXPONENT_DRY_AIR: f64 = 287.058 / 1005.7;

// ---------------------------------------------------------------------------
// MetPy short aliases (mpconsts.g, mpconsts.Rd, etc.)
// ---------------------------------------------------------------------------

#[allow(non_upper_case_globals)]
/// Standard acceleration of gravity (m s^-2). Alias for `EARTH_GRAVITY`.
pub const g: f64 = EARTH_GRAVITY;

#[allow(non_upper_case_globals)]
/// Specific gas constant for dry air (J kg^-1 K^-1). Alias for `RD`.
pub const Rd: f64 = RD;

#[allow(non_upper_case_globals)]
/// Specific gas constant for water vapour (J kg^-1 K^-1). Alias for `RV`.
pub const Rv: f64 = RV;

#[allow(non_upper_case_globals)]
/// Specific heat at constant pressure for dry air (J kg^-1 K^-1). Alias for `CP_D`.
pub const Cp_d: f64 = CP_D;

#[allow(non_upper_case_globals)]
/// Specific heat at constant volume for dry air (J kg^-1 K^-1). Alias for `CV_D`.
pub const Cv_d: f64 = CV_D;

#[allow(non_upper_case_globals)]
/// Latent heat of vaporisation at 0 degC (J kg^-1). Alias for `LV`.
pub const Lv: f64 = LV;

#[allow(non_upper_case_globals)]
/// Latent heat of fusion at 0 degC (J kg^-1). Alias for `LF`.
pub const Lf: f64 = LF;

#[allow(non_upper_case_globals)]
/// Latent heat of sublimation at 0 degC (J kg^-1). Alias for `LS`.
pub const Ls: f64 = LS;

#[allow(non_upper_case_globals)]
/// Mean radius of the Earth (m). Alias for `EARTH_AVG_RADIUS`.
pub const Re: f64 = EARTH_AVG_RADIUS;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert approximate equality within a relative tolerance.
    fn approx_eq(a: f64, b: f64, rel_tol: f64) {
        let diff = (a - b).abs();
        let magnitude = a.abs().max(b.abs()).max(1e-30);
        assert!(
            diff / magnitude < rel_tol,
            "approx_eq failed: {a} vs {b} (diff {diff}, rel_tol {rel_tol})"
        );
    }

    #[test]
    fn test_earth_constants() {
        assert_eq!(EARTH_AVG_RADIUS, 6.371229e6);
        assert_eq!(EARTH_GRAVITY, 9.80665);
        assert_eq!(OMEGA, 7.2921159e-5);
        assert_eq!(EARTH_AVG_DENSITY, 5515.0);
        assert_eq!(EARTH_MAX_DECLINATION, 23.45);
    }

    #[test]
    fn test_dry_air_constants() {
        assert_eq!(RD, 287.058);
        assert_eq!(CP_D, 1005.7);
        assert_eq!(CV_D, 718.0);
        approx_eq(KAPPA, RD / CP_D, 1e-15);
        assert_eq!(MOLECULAR_WEIGHT_DRY_AIR, 28.9645e-3);
    }

    #[test]
    fn test_epsilon_ratio() {
        // epsilon = Mw / Md, which should also approximate Rd / Rv
        let rd_rv = RD / RV;
        approx_eq(EPSILON, rd_rv, 1e-3);
        approx_eq(EPSILON, 0.62197, 1e-3);
    }

    #[test]
    fn test_water_constants() {
        assert_eq!(RV, 461.5);
        assert_eq!(CP_V, 1875.0);
        assert_eq!(RHO_L, 999.97);
        assert_eq!(LV, 2.501e6);
        assert_eq!(LF, 3.34e5);
        assert_eq!(LS, 2.834e6);
        assert_eq!(CP_L, 4218.0);
        assert_eq!(T_FREEZE, 273.15);
        assert_eq!(MOLECULAR_WEIGHT_WATER, 18.015e-3);
    }

    #[test]
    fn test_standard_atmosphere() {
        assert_eq!(P_STP, 101325.0);
        assert_eq!(T_STP, 288.15);
    }

    #[test]
    fn test_rho_d_stp() {
        // rho = P / (Rd * T)
        let expected = 101325.0 / (287.058 * 288.15);
        approx_eq(RHO_D_STP, expected, 1e-10);
    }

    #[test]
    fn test_universal_constants() {
        approx_eq(R, 8.3145, 1e-3);
        approx_eq(STEFAN_BOLTZMANN, 5.6704e-8, 1e-3);
    }

    #[test]
    fn test_named_water_constants() {
        assert_eq!(WATER_HEAT_VAPORIZATION, LV);
        assert_eq!(WATER_HEAT_FUSION, LF);
        assert_eq!(WATER_HEAT_SUBLIMATION, LS);
        assert_eq!(WATER_SPECIFIC_HEAT_LIQUID, CP_L);
        assert_eq!(WATER_SPECIFIC_HEAT_VAPOR, CP_V);
        assert_eq!(WATER_MOLECULAR_WEIGHT, MOLECULAR_WEIGHT_WATER);
        assert_eq!(WATER_GAS_CONSTANT, RV);
    }

    #[test]
    fn test_named_dry_air_constants() {
        assert_eq!(DRY_AIR_MOLECULAR_WEIGHT, MOLECULAR_WEIGHT_DRY_AIR);
        assert_eq!(DRY_AIR_SPEC_HEAT_PRESS, CP_D);
        assert_eq!(DRY_AIR_SPEC_HEAT_VOL, CV_D);
        assert_eq!(DRY_AIR_GAS_CONSTANT, RD);
        assert_eq!(DRY_AIR_DENSITY_STP, 1.275);
    }

    #[test]
    fn test_named_earth_constants() {
        assert_eq!(NOAA_MEAN_EARTH_RADIUS, EARTH_AVG_RADIUS);
        assert_eq!(EARTH_GRAVITATIONAL_ACCELERATION, EARTH_GRAVITY);
    }

    #[test]
    fn test_poisson_exponent() {
        approx_eq(POISSON_EXPONENT_DRY_AIR, KAPPA, 1e-15);
        approx_eq(POISSON_EXPONENT_DRY_AIR, RD / CP_D, 1e-15);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_metpy_short_aliases() {
        assert_eq!(g, EARTH_GRAVITY);
        assert_eq!(Rd, RD);
        assert_eq!(Rv, RV);
        assert_eq!(Cp_d, CP_D);
        assert_eq!(Cv_d, CV_D);
        assert_eq!(Lv, LV);
        assert_eq!(Lf, LF);
        assert_eq!(Ls, LS);
        assert_eq!(Re, EARTH_AVG_RADIUS);
    }
}
