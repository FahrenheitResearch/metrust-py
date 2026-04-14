//! Severe weather composite parameters.
//!
//! Implements the significant tornado parameter (STP), supercell composite
//! parameter (SCP), and critical angle — standard indices used in operational
//! severe convective weather forecasting.
//!
//! All inputs use SI units: CAPE in J/kg, heights in meters, helicity in
//! m^2/s^2, and bulk shear in m/s.
//!
//! ## Grid-based composites
//!
//! The [`grid`] submodule re-exports grid-oriented functions from
//! `wx_math::composite` that operate on flattened 2-D/3-D arrays and use
//! `rayon` for parallelism. These have different signatures from the
//! point-based functions at the top level of this module.

use std::f64::consts::PI;

// ─────────────────────────────────────────────
// Re-exports from wx_math::composite (point-based helpers)
// ─────────────────────────────────────────────

pub use wx_math::composite::boyden_index;
pub use wx_math::composite::bulk_richardson_number;
pub use wx_math::composite::convective_inhibition_depth;
pub use wx_math::composite::dendritic_growth_zone;
pub use wx_math::composite::fosberg_fire_weather_index;
pub use wx_math::composite::freezing_rain_composite;
pub use wx_math::composite::haines_index;
pub use wx_math::composite::hot_dry_windy;
pub use wx_math::composite::warm_nose_check;

// Re-export galvez_davison_index from wx_math::thermo (it lives there, not composite)
pub use wx_math::thermo::galvez_davison_index;

/// Grid-oriented composite parameters from `wx_math::composite`.
///
/// These functions accept flattened 2-D or 3-D arrays (`&[f64]`) and grid
/// dimensions (`nx`, `ny`, `nz`) and return `Vec<f64>` result grids. They
/// use `rayon` internally for parallel computation.
///
/// Functions whose names overlap with point-based versions in the parent
/// module (e.g. `supercell_composite_parameter`, `critical_angle`) have
/// different signatures here — they take grid slices rather than scalar
/// values.
pub mod grid {
    use ecape_rs::{
        calc_ecape_parcel, CapeType as EcapeCapeType, ParcelOptions,
        StormMotionType as EcapeStormMotionType,
    };
    use rayon::prelude::*;

    // ── Stability indices (profile-based, different arg order from thermo) ──
    pub use wx_math::composite::cross_totals;
    pub use wx_math::composite::k_index;
    pub use wx_math::composite::lifted_index;
    pub use wx_math::composite::showalter_index;
    pub use wx_math::composite::sweat_index;
    pub use wx_math::composite::total_totals;
    pub use wx_math::composite::vertical_totals;

    // ── 3-D grid compute functions ──
    pub use wx_math::composite::compute_cape_cin;
    pub use wx_math::composite::compute_lapse_rate;
    pub use wx_math::composite::compute_pw;
    pub use wx_math::composite::compute_shear;
    pub use wx_math::composite::compute_srh;

    #[derive(Debug, Clone, Copy, Default)]
    struct EcapeSummary {
        ecape: f64,
        ncape: f64,
        cape: f64,
        cin: f64,
        lfc: f64,
        el: f64,
    }

    fn resolve_ecape_cape_type(parcel_type: &str) -> Result<EcapeCapeType, String> {
        match parcel_type.trim().to_ascii_lowercase().as_str() {
            "surface" | "surface_based" | "sb" => Ok(EcapeCapeType::SurfaceBased),
            "mixed_layer" | "ml" => Ok(EcapeCapeType::MixedLayer),
            "most_unstable" | "mu" => Ok(EcapeCapeType::MostUnstable),
            other => Err(format!(
                "unsupported ECAPE parcel_type '{other}'; expected 'surface', 'sb', 'mixed_layer', 'ml', 'most_unstable', or 'mu'"
            )),
        }
    }

    fn resolve_ecape_storm_motion_type(
        storm_motion_type: &str,
    ) -> Result<EcapeStormMotionType, String> {
        match storm_motion_type
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "right_moving" | "bunkers_rm" | "rm" | "right" => {
                Ok(EcapeStormMotionType::RightMoving)
            }
            "left_moving" | "bunkers_lm" | "lm" | "left" => {
                Ok(EcapeStormMotionType::LeftMoving)
            }
            "mean_wind" | "mean" => Ok(EcapeStormMotionType::MeanWind),
            "user_defined" | "custom" => Ok(EcapeStormMotionType::UserDefined),
            other => Err(format!(
                "unsupported ECAPE storm_motion_type '{other}'; expected 'right_moving', 'bunkers_rm', 'left_moving', 'bunkers_lm', or 'mean_wind'"
            )),
        }
    }

    fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
        let q = q_kgkg.max(1.0e-10);
        let p_hpa = p_pa / 100.0;
        let e = (q * p_hpa / (0.622 + q)).max(1.0e-10);
        let ln_e = (e / 6.112).ln();
        let td_c = (243.5 * ln_e) / (17.67 - ln_e);
        (td_c + 273.15).min(temp_k)
    }

    fn push_ecape_level(
        pressure_pa: &mut Vec<f64>,
        height_m: &mut Vec<f64>,
        temp_k: &mut Vec<f64>,
        dewpoint_k: &mut Vec<f64>,
        u_ms: &mut Vec<f64>,
        v_ms: &mut Vec<f64>,
        p: f64,
        z: f64,
        t: f64,
        td: f64,
        u: f64,
        v: f64,
    ) {
        if !p.is_finite()
            || !z.is_finite()
            || !t.is_finite()
            || !td.is_finite()
            || !u.is_finite()
            || !v.is_finite()
        {
            return;
        }

        if let (Some(&last_p), Some(&last_z)) = (pressure_pa.last(), height_m.last()) {
            if p >= last_p || z <= last_z {
                return;
            }
        }

        pressure_pa.push(p);
        height_m.push(z);
        temp_k.push(t);
        dewpoint_k.push(td.min(t));
        u_ms.push(u);
        v_ms.push(v);
    }

    fn build_surface_augmented_ecape_column(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc_pa: f64,
        t2_k: f64,
        q2_kgkg: f64,
        u10_ms: f64,
        v10_ms: f64,
        nz: usize,
        nxy: usize,
        ij: usize,
        model_bottom_up: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pressure_pa = Vec::with_capacity(nz + 1);
        let mut height_m = Vec::with_capacity(nz + 1);
        let mut temp_k = Vec::with_capacity(nz + 1);
        let mut dewpoint_k = Vec::with_capacity(nz + 1);
        let mut u_ms = Vec::with_capacity(nz + 1);
        let mut v_ms = Vec::with_capacity(nz + 1);

        push_ecape_level(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut dewpoint_k,
            &mut u_ms,
            &mut v_ms,
            psfc_pa,
            0.0,
            t2_k,
            dewpoint_k_from_q(q2_kgkg, psfc_pa, t2_k),
            u10_ms,
            v10_ms,
        );

        let push_model_level = |k: usize,
                                pressure_pa: &mut Vec<f64>,
                                height_m: &mut Vec<f64>,
                                temp_k: &mut Vec<f64>,
                                dewpoint_k: &mut Vec<f64>,
                                u_ms: &mut Vec<f64>,
                                v_ms: &mut Vec<f64>| {
            let idx = k * nxy + ij;
            let tk = temperature_c_3d[idx] + 273.15;
            push_ecape_level(
                pressure_pa,
                height_m,
                temp_k,
                dewpoint_k,
                u_ms,
                v_ms,
                pressure_3d[idx],
                height_agl_3d[idx],
                tk,
                dewpoint_k_from_q(qvapor_3d[idx], pressure_3d[idx], tk),
                u_3d[idx],
                v_3d[idx],
            );
        };

        if model_bottom_up {
            for k in 0..nz {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        } else {
            for k in (0..nz).rev() {
                push_model_level(
                    k,
                    &mut pressure_pa,
                    &mut height_m,
                    &mut temp_k,
                    &mut dewpoint_k,
                    &mut u_ms,
                    &mut v_ms,
                );
            }
        }

        (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms)
    }

    /// Compute ECAPE-family diagnostics for every grid point.
    ///
    /// Returns `(ecape, ncape, cape, cin, lfc, el)` as six 1-D arrays of length `nx*ny`.
    pub fn compute_ecape(
        pressure_3d: &[f64],
        temperature_c_3d: &[f64],
        qvapor_3d: &[f64],
        height_agl_3d: &[f64],
        u_3d: &[f64],
        v_3d: &[f64],
        psfc: &[f64],
        t2: &[f64],
        q2: &[f64],
        u10: &[f64],
        v10: &[f64],
        nx: usize,
        ny: usize,
        nz: usize,
        parcel_type: &str,
        storm_motion_type: &str,
        entrainment_rate: Option<f64>,
        pseudoadiabatic: Option<bool>,
        storm_u: Option<f64>,
        storm_v: Option<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let n2d = nx * ny;
        let expected_3d = n2d * nz;
        if pressure_3d.len() != expected_3d
            || temperature_c_3d.len() != expected_3d
            || qvapor_3d.len() != expected_3d
            || height_agl_3d.len() != expected_3d
            || u_3d.len() != expected_3d
            || v_3d.len() != expected_3d
        {
            return Err("ECAPE 3-D inputs must all have length nx*ny*nz".into());
        }
        if psfc.len() != n2d
            || t2.len() != n2d
            || q2.len() != n2d
            || u10.len() != n2d
            || v10.len() != n2d
        {
            return Err("ECAPE surface inputs must all have length nx*ny".into());
        }
        if storm_u.is_some() ^ storm_v.is_some() {
            return Err(
                "storm_u and storm_v must either both be provided or both be omitted".into(),
            );
        }

        let cape_type = resolve_ecape_cape_type(parcel_type)?;
        let mut motion_type = resolve_ecape_storm_motion_type(storm_motion_type)?;
        if storm_u.is_some() && storm_v.is_some() {
            motion_type = EcapeStormMotionType::UserDefined;
        }

        let results: Vec<EcapeSummary> = (0..n2d)
            .into_par_iter()
            .map(|ij| {
                let top_idx = ij;
                let bottom_idx = (nz - 1) * n2d + ij;
                let model_bottom_up = pressure_3d[top_idx] >= pressure_3d[bottom_idx]
                    || height_agl_3d[top_idx] <= height_agl_3d[bottom_idx];

                let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                    build_surface_augmented_ecape_column(
                        pressure_3d,
                        temperature_c_3d,
                        qvapor_3d,
                        height_agl_3d,
                        u_3d,
                        v_3d,
                        psfc[ij],
                        t2[ij],
                        q2[ij],
                        u10[ij],
                        v10[ij],
                        nz,
                        n2d,
                        ij,
                        model_bottom_up,
                    );

                if pressure_pa.len() < 2 {
                    return EcapeSummary::default();
                }

                let mut options = ParcelOptions {
                    cape_type,
                    storm_motion_type: motion_type,
                    entrainment_rate,
                    pseudoadiabatic,
                    ..ParcelOptions::default()
                };

                if let (Some(storm_motion_u_ms), Some(storm_motion_v_ms)) = (storm_u, storm_v) {
                    options.storm_motion_type = EcapeStormMotionType::UserDefined;
                    options.storm_motion_u_ms = Some(storm_motion_u_ms);
                    options.storm_motion_v_ms = Some(storm_motion_v_ms);
                }

                match calc_ecape_parcel(
                    &height_m,
                    &pressure_pa,
                    &temp_k,
                    &dewpoint_k,
                    &u_ms,
                    &v_ms,
                    &options,
                ) {
                    Ok(result) => EcapeSummary {
                        ecape: result.ecape_jkg,
                        ncape: result.ncape_jkg,
                        cape: result.cape_jkg,
                        cin: result.cin_jkg,
                        lfc: result.lfc_m.unwrap_or(0.0),
                        el: result.el_m.unwrap_or(0.0),
                    },
                    Err(_) => EcapeSummary::default(),
                }
            })
            .collect();

        let mut ecape = Vec::with_capacity(n2d);
        let mut ncape = Vec::with_capacity(n2d);
        let mut cape = Vec::with_capacity(n2d);
        let mut cin = Vec::with_capacity(n2d);
        let mut lfc = Vec::with_capacity(n2d);
        let mut el = Vec::with_capacity(n2d);

        for result in results {
            ecape.push(result.ecape);
            ncape.push(result.ncape);
            cape.push(result.cape);
            cin.push(result.cin);
            lfc.push(result.lfc);
            el.push(result.el);
        }

        Ok((ecape, ncape, cape, cin, lfc, el))
    }

    // ── 2-D grid composite parameters ──
    pub use wx_math::composite::compute_ehi;
    pub use wx_math::composite::compute_scp;
    pub use wx_math::composite::compute_stp;
    pub use wx_math::composite::critical_angle;
    pub use wx_math::composite::derecho_composite_parameter;
    pub use wx_math::composite::significant_hail_parameter;
    pub use wx_math::composite::supercell_composite_parameter;

    // ── Reflectivity composites ──
    pub use wx_math::composite::composite_reflectivity_from_hydrometeors;
    pub use wx_math::composite::composite_reflectivity_from_refl;
}

/// Significant Tornado Parameter (STP).
///
/// STP combines mixed-layer CAPE, LCL height, 0-1 km storm-relative helicity,
/// and 0-6 km bulk shear magnitude into a single composite favoring significant
/// (EF2+) tornadoes.
///
/// ```text
/// STP = (mlCAPE / 1500) * ((2000 - mlLCL) / 1000) * (SRH / 150) * (shear / 20)
/// ```
///
/// Each term is clamped so it cannot go below 0 (and the LCL term is also
/// capped at 1.0 when LCL <= 1000 m, per Thompson et al. 2003).
///
/// # Arguments
/// * `mlcape` — Mixed-layer CAPE (J/kg)
/// * `lcl_height_m` — Mixed-layer LCL height AGL (m)
/// * `srh_0_1km` — 0-1 km storm-relative helicity (m^2/s^2)
/// * `bulk_shear_0_6km_ms` — 0-6 km bulk wind shear magnitude (m/s)
///
/// # Returns
/// Dimensionless STP value. Values above 1.0 are increasingly favorable for
/// significant tornadoes.
///
/// # References
/// Thompson, R. L., R. Edwards, J. A. Hart, K. L. Elmore, and P. Markowski,
/// 2003: Close proximity soundings within supercell environments obtained from
/// the Rapid Update Cycle. *Wea. Forecasting*, **18**, 1243-1261.
pub fn significant_tornado_parameter(
    sbcape: f64,
    lcl_height_m: f64,
    srh_0_1km: f64,
    bulk_shear_0_6km_ms: f64,
) -> f64 {
    // CAPE term: SBCAPE / 1500, floored at 0
    // Fixed-layer STP uses surface-based CAPE (not MLCAPE)
    let cape_term = (sbcape / 1500.0).max(0.0);

    // LCL term: (2000 - LCL) / 1000
    //   - Capped at 1.0 when LCL <= 1000 m (very low LCL is always favorable)
    //   - Floored at 0.0 when LCL >= 2000 m (too high, unfavorable)
    let lcl_term = if lcl_height_m <= 1000.0 {
        1.0
    } else {
        ((2000.0 - lcl_height_m) / 1000.0).clamp(0.0, 1.0)
    };

    // SRH term: SRH / 150, floored at 0
    let srh_term = (srh_0_1km / 150.0).max(0.0);

    // Shear term: zero when < 12.5 m/s, capped at 30 m/s, then / 20
    // (per Thompson et al. 2003 / MetPy: shear < 12.5 m/s => 0)
    let shear_term = if bulk_shear_0_6km_ms < 12.5 {
        0.0
    } else {
        (bulk_shear_0_6km_ms.min(30.0) / 20.0).max(0.0)
    };

    cape_term * lcl_term * srh_term * shear_term
}

/// Supercell Composite Parameter (SCP).
///
/// SCP combines most-unstable CAPE, effective-layer storm-relative helicity,
/// and effective bulk shear magnitude.
///
/// ```text
/// SCP = (muCAPE / 1000) * (SRH / 50) * (shear / 20)
/// ```
///
/// Each term is floored at 0.
///
/// # Arguments
/// * `mucape` — Most-unstable CAPE (J/kg)
/// * `srh_eff` — Effective-layer storm-relative helicity (m^2/s^2)
/// * `bulk_shear_eff_ms` — Effective bulk shear magnitude (m/s)
///
/// # Returns
/// Dimensionless SCP value. Values >= 1.0 favor supercells.
///
/// # References
/// Thompson, R. L., R. Edwards, and C. M. Mead, 2004: An update to the
/// supercell composite and significant tornado parameters. Preprints, 22nd
/// Conf. on Severe Local Storms, Hyannis, MA.
pub fn supercell_composite_parameter(mucape: f64, srh_eff: f64, bulk_shear_eff_ms: f64) -> f64 {
    let cape_term = (mucape / 1000.0).max(0.0);
    let srh_term = (srh_eff / 50.0).max(0.0);
    // Shear term: zero when < 10 m/s, capped at 1.0 (i.e. clipped to 20 m/s then / 20)
    // (per Thompson et al. 2004 / MetPy)
    let shear_term = if bulk_shear_eff_ms < 10.0 {
        0.0
    } else {
        (bulk_shear_eff_ms.min(20.0) / 20.0).max(0.0)
    };

    cape_term * srh_term * shear_term
}

/// Critical angle between the storm-relative inflow vector and the 0-500 m
/// shear vector.
///
/// A critical angle near 90 degrees is most favorable for low-level
/// mesocyclone development because it means the storm-relative inflow is
/// perpendicular to the low-level shear, maximizing streamwise vorticity
/// tilting.
///
/// # Arguments
/// * `storm_u`, `storm_v` — Storm motion components (m/s)
/// * `u_sfc`, `v_sfc` — Surface wind components (m/s)
/// * `u_500m`, `v_500m` — Wind at 500 m AGL (m/s)
///
/// # Returns
/// The angle in degrees [0, 180] between the two vectors. Returns 0.0 if
/// either vector has near-zero magnitude.
///
/// # References
/// Esterheld, J. M., and D. J. Giuliano, 2008: Discriminating between
/// tornadic and non-tornadic supercells: A new hodograph technique. *E-Journal
/// of Severe Storms Meteorology*, **3(2)**, 1-13.
pub fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    // Storm-relative inflow vector (surface wind relative to storm)
    let inflow_u = u_sfc - storm_u;
    let inflow_v = v_sfc - storm_v;

    // 0-500 m shear vector
    let shear_u = u_500m - u_sfc;
    let shear_v = v_500m - v_sfc;

    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

    if mag_inflow < 1e-10 || mag_shear < 1e-10 {
        return 0.0;
    }

    let cos_angle = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear);
    // Clamp to [-1, 1] to avoid NaN from floating-point rounding
    cos_angle.clamp(-1.0, 1.0).acos() * (180.0 / PI)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ecape_rs::{calc_ecape_parcel, CapeType, ParcelOptions, StormMotionType};

    // ── significant_tornado_parameter ──

    #[test]
    fn test_stp_nominal() {
        // All terms at their normalizing values => STP = 1.0
        let stp = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp - 1.0).abs() < 1e-10,
            "STP with nominal values = {stp}, expected 1.0"
        );
    }

    #[test]
    fn test_stp_zero_cape() {
        let stp = significant_tornado_parameter(0.0, 800.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with zero CAPE");
    }

    #[test]
    fn test_stp_high_lcl_zero() {
        // LCL at 2000 m makes the LCL term 0 => STP = 0
        let stp = significant_tornado_parameter(2000.0, 2000.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with LCL=2000m");
    }

    #[test]
    fn test_stp_very_low_lcl_capped() {
        // LCL <= 1000m => LCL term capped at 1.0
        let stp_500 = significant_tornado_parameter(1500.0, 500.0, 150.0, 20.0);
        let stp_1000 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp_500 - stp_1000).abs() < 1e-10,
            "LCL 500m and 1000m should give same STP"
        );
    }

    #[test]
    fn test_stp_shear_capped() {
        // Shear term capped at 1.5 when shear = 30 m/s (30/20 = 1.5)
        let stp_30 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 30.0);
        let stp_50 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 50.0);
        assert!(
            (stp_30 - stp_50).abs() < 1e-10,
            "STP should cap shear at 1.5: stp_30={stp_30}, stp_50={stp_50}"
        );
        assert!((stp_30 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_stp_weak_shear_zero() {
        // Shear below 12.5 m/s => shear_term = 0 => STP = 0 (per MetPy)
        let stp = significant_tornado_parameter(2000.0, 800.0, 150.0, 10.0);
        assert!(
            (stp - 0.0).abs() < 1e-10,
            "STP should be 0 with weak shear (<12.5 m/s)"
        );
    }

    #[test]
    fn test_stp_negative_inputs_floored() {
        let stp = significant_tornado_parameter(-500.0, 3000.0, -100.0, -5.0);
        assert!(
            (stp - 0.0).abs() < 1e-10,
            "STP should be 0 with all negative inputs"
        );
    }

    #[test]
    fn test_stp_strong_case() {
        // High CAPE=4000, low LCL=500, strong SRH=400, strong shear=35
        let stp = significant_tornado_parameter(4000.0, 500.0, 400.0, 35.0);
        let expected = (4000.0 / 1500.0) * 1.0 * (400.0 / 150.0) * 1.5;
        assert!(
            (stp - expected).abs() < 1e-10,
            "STP = {stp}, expected {expected}"
        );
    }

    // ── supercell_composite_parameter ──

    #[test]
    fn test_scp_nominal() {
        let scp = supercell_composite_parameter(1000.0, 50.0, 20.0);
        assert!((scp - 1.0).abs() < 1e-10, "SCP with nominal values = {scp}");
    }

    #[test]
    fn test_scp_zero_cape() {
        let scp = supercell_composite_parameter(0.0, 200.0, 30.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scp_strong_case() {
        // Shear 30 m/s is capped at 20 m/s => shear_term = 1.0
        let scp = supercell_composite_parameter(3000.0, 300.0, 30.0);
        let expected = (3000.0 / 1000.0) * (300.0 / 50.0) * 1.0;
        assert!(
            (scp - expected).abs() < 1e-10,
            "SCP = {scp}, expected {expected}"
        );
    }

    #[test]
    fn test_scp_weak_shear_zero() {
        // Shear below 10 m/s => shear_term = 0 => SCP = 0 (per MetPy)
        let scp = supercell_composite_parameter(3000.0, 200.0, 8.0);
        assert!((scp - 0.0).abs() < 1e-10, "SCP should be 0 with weak shear");
    }

    #[test]
    fn test_scp_negative_inputs_floored() {
        let scp = supercell_composite_parameter(-500.0, -100.0, -10.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    // ── critical_angle ──

    #[test]
    fn test_critical_angle_perpendicular() {
        // Inflow from east (storm east of surface obs), shear points north
        // => 90 degrees
        let angle = critical_angle(
            10.0, 0.0, // storm at (10, 0)
            0.0, 0.0, // sfc wind calm
            0.0, 5.0, // 500m wind northward
        );
        assert!(
            (angle - 90.0).abs() < 1e-10,
            "expected 90 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_parallel() {
        // Inflow and shear both pointing north => 0 degrees
        let angle = critical_angle(
            0.0, -10.0, // storm south of sfc
            0.0, 0.0, // sfc calm
            0.0, 5.0, // 500m northward
        );
        assert!(angle.abs() < 1e-10, "expected 0 degrees, got {angle}");
    }

    #[test]
    fn test_critical_angle_antiparallel() {
        // Inflow and shear 180 degrees apart
        let angle = critical_angle(
            0.0, 10.0, // storm north of sfc
            0.0, 0.0, // sfc calm
            0.0, 5.0, // 500m northward
        );
        assert!(
            (angle - 180.0).abs() < 1e-10,
            "expected 180 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_zero_inflow() {
        // Storm motion equals surface wind => zero inflow vector
        let angle = critical_angle(5.0, 3.0, 5.0, 3.0, 10.0, 8.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero inflow should give 0 deg");
    }

    #[test]
    fn test_critical_angle_zero_shear() {
        // 500m wind equals surface wind => zero shear vector
        let angle = critical_angle(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero shear should give 0 deg");
    }

    #[test]
    fn test_critical_angle_45_degrees() {
        // Inflow along x-axis, shear at 45 degrees
        let angle = critical_angle(
            10.0, 0.0, // storm motion
            0.0, 0.0, // sfc
            -5.0, 5.0, // 500m: shear = (-5, 5), inflow = (0,0)-(10,0) = (-10, 0)
        );
        // inflow = (-10, 0), shear = (-5, 5)
        // cos(theta) = ((-10)*(-5) + 0*5) / (10 * sqrt(50)) = 50 / (10*sqrt(50))
        //            = 5 / sqrt(50) = sqrt(50)/10 = 1/sqrt(2)
        // theta = 45 degrees
        assert!(
            (angle - 45.0).abs() < 1e-10,
            "expected 45 degrees, got {angle}"
        );
    }

    fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
        let q = q_kgkg.max(1.0e-10);
        let p_hpa = p_pa / 100.0;
        let e = (q * p_hpa / (0.622 + q)).max(1.0e-10);
        let ln_e = (e / 6.112).ln();
        let td_c = (243.5 * ln_e) / (17.67 - ln_e);
        (td_c + 273.15).min(temp_k)
    }

    #[test]
    fn test_compute_ecape_single_column_matches_direct_solver() {
        let nx = 1;
        let ny = 1;
        let nz = 6;

        let pressure_3d = vec![95000.0, 90000.0, 85000.0, 70000.0, 50000.0, 30000.0];
        let temperature_c_3d = vec![26.0, 22.0, 18.0, 8.0, -10.0, -38.0];
        let qvapor_3d = vec![0.016, 0.013, 0.010, 0.005, 0.0015, 0.0003];
        let height_agl_3d = vec![150.0, 800.0, 1500.0, 3000.0, 5600.0, 9200.0];
        let u_3d = vec![6.0, 9.0, 12.0, 18.0, 26.0, 33.0];
        let v_3d = vec![2.0, 5.0, 8.0, 13.0, 20.0, 28.0];
        let psfc = vec![100000.0];
        let t2 = vec![303.15];
        let q2 = vec![0.018];
        let u10 = vec![5.0];
        let v10 = vec![1.5];

        let (ecape, ncape, cape, cin, lfc, el) = grid::compute_ecape(
            &pressure_3d,
            &temperature_c_3d,
            &qvapor_3d,
            &height_agl_3d,
            &u_3d,
            &v_3d,
            &psfc,
            &t2,
            &q2,
            &u10,
            &v10,
            nx,
            ny,
            nz,
            "ml",
            "bunkers_rm",
            None,
            Some(true),
            None,
            None,
        )
        .unwrap();

        let pressure_pa = vec![
            100000.0, 95000.0, 90000.0, 85000.0, 70000.0, 50000.0, 30000.0,
        ];
        let height_m = vec![0.0, 150.0, 800.0, 1500.0, 3000.0, 5600.0, 9200.0];
        let temp_k = vec![303.15, 299.15, 295.15, 291.15, 281.15, 263.15, 235.15];
        let dewpoint_k = vec![
            dewpoint_k_from_q(0.018, 100000.0, 303.15),
            dewpoint_k_from_q(0.016, 95000.0, 299.15),
            dewpoint_k_from_q(0.013, 90000.0, 295.15),
            dewpoint_k_from_q(0.010, 85000.0, 291.15),
            dewpoint_k_from_q(0.005, 70000.0, 281.15),
            dewpoint_k_from_q(0.0015, 50000.0, 263.15),
            dewpoint_k_from_q(0.0003, 30000.0, 235.15),
        ];
        let u_ms = vec![5.0, 6.0, 9.0, 12.0, 18.0, 26.0, 33.0];
        let v_ms = vec![1.5, 2.0, 5.0, 8.0, 13.0, 20.0, 28.0];
        let options = ParcelOptions {
            cape_type: CapeType::MixedLayer,
            storm_motion_type: StormMotionType::RightMoving,
            pseudoadiabatic: Some(true),
            ..ParcelOptions::default()
        };
        let direct = calc_ecape_parcel(
            &height_m,
            &pressure_pa,
            &temp_k,
            &dewpoint_k,
            &u_ms,
            &v_ms,
            &options,
        )
        .unwrap();

        assert!((ecape[0] - direct.ecape_jkg).abs() < 1.0);
        assert!((ncape[0] - direct.ncape_jkg).abs() < 1.0);
        assert!((cape[0] - direct.cape_jkg).abs() < 1.0);
        assert!((cin[0] - direct.cin_jkg).abs() < 1.0);
        assert!((lfc[0] - direct.lfc_m.unwrap_or(0.0)).abs() < 1.0e-6);
        assert!((el[0] - direct.el_m.unwrap_or(0.0)).abs() < 1.0e-6);
    }

    #[test]
    fn test_compute_ecape_requires_both_storm_motion_components() {
        let err = grid::compute_ecape(
            &[95000.0, 90000.0],
            &[20.0, 10.0],
            &[0.010, 0.004],
            &[500.0, 2000.0],
            &[5.0, 10.0],
            &[0.0, 5.0],
            &[100000.0],
            &[300.0],
            &[0.014],
            &[4.0],
            &[1.0],
            1,
            1,
            2,
            "sb",
            "mean_wind",
            None,
            Some(true),
            Some(8.0),
            None,
        )
        .unwrap_err();
        assert!(err.contains("storm_u and storm_v"));
    }
}
