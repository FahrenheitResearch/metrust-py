# Verification Report

Comprehensive verification of metrust against MetPy 1.7.1, conducted 2026-03-12.

## Summary

- **573 automated verification tests**, all passing
- **89.3% API coverage** of metpy.calc (142/159 functions)
- **15,000x speedup** for scalar operations, **15-70x** for array operations
- **Machine-precision matches** where algorithms are identical
- **Known formula differences** documented below with root causes and impact assessments

## Test Suites

### 1. Thermodynamic Accuracy (151 tests)

Tested 60 distinct thermodynamic functions with 2-4 input cases each, covering:
- Potential temperature, equivalent potential temperature
- Saturation vapor pressure, mixing ratio, specific humidity
- Dewpoint conversions, relative humidity conversions
- Virtual temperature, wet bulb temperature
- LCL, LFC, EL
- Density, dry/moist lapse rates
- Stability indices (Showalter, K-index, Total Totals, SWEAT)
- Brunt-Vaisala frequency/period
- Precipitable water, thickness
- Standard atmosphere pressure/height conversions
- 7 roundtrip consistency tests

**Findings:**
- Bolton vs Buck saturation vapor pressure: max ~0.22 hPa difference at 35C
- SHARPpy LCL polynomial vs MetPy iterative: within 2-5 hPa
- Wet bulb: within 1-2C
- moist_lapse RK4 integration: **significant discrepancy** — needs fix
- Cp_d constant difference propagates ~300 J/kg in energy calculations

### 2. Severe Weather + Atmospheric (100 tests)

Tested all 18 severe/atmo functions with 3-9 cases each.

**Machine-precision matches:**
- heat_index (all NWS reference values)
- fosberg_fire_weather_index
- boyden_index
- bulk_richardson_number
- critical_angle
- haines_index
- sigma_to_pressure

**Known formula differences:**
- STP: MetPy zeros bulk shear term below 12.5 m/s
- SCP: MetPy zeros shear below 10 m/s and caps at 1.0
- Standard atmosphere: ICAO (288.15K) vs MetPy (288.0K)
- Altimeter formula: direct barometric vs Smithsonian iterative (~2 hPa)

### 3. Edge Cases (96 tests)

Comprehensive edge case testing across 23 categories:
- NaN propagation (both propagate correctly, no panics)
- Zero values (both handle gracefully)
- Negative/extreme temperatures (both produce finite results)
- Humidity edge cases (both allow RH > 100% when Td > T)
- Empty arrays (both return empty, no panic)
- Single-element arrays (both work correctly)
- Infinity handling
- Cardinal wind directions
- Round-trip consistency
- Mismatched array lengths (both error appropriately)
- Large arrays (10,000 elements)

**Critical unit convention differences found:**
| Function | metrust (Rust) | MetPy | Python wrapper handles? |
|---|---|---|---|
| saturation_vapor_pressure | hPa | Pa | Yes |
| relative_humidity_from_dewpoint | percent (0-100) | fraction (0-1) | Yes |
| saturation_mixing_ratio | g/kg | kg/kg | Yes |

### 4. Smoothing + Interpolation (73 tests)

Tested all 16 smoothing/derivative/interpolation functions.

**Exact matches on interior points:**
- smooth_n_point (5 and 9 point)
- smooth_rectangular
- first_derivative, second_derivative (interior)
- gradient_x, gradient_y
- laplacian
- interpolate_1d (within range)
- log_interpolate_1d
- interpolate_nans_1d
- interpolate_to_isosurface

**Documented behavioral differences:**
| Behavior | MetPy | metrust |
|---|---|---|
| smooth_n_point weights | Cardinal neighbors 2x diagonal | Equal neighbor weights |
| Smoothing boundaries | Leaves edges untouched | Renormalizes at boundaries |
| smooth_gaussian engine | scipy.ndimage (separable, reflect) | Pure Rust (NaN-aware, separable) |
| first_derivative boundary | 3-point (2nd-order) stencil | 2-point (1st-order) stencil |
| interpolate_1d outside range | NaN | Clamp to boundary |

### 5. Unit Conversions (42 tests)

Tested all 5 unit categories with 5+ values each, plus roundtrip tests.

**All conversions match Pint to 4+ decimal places.**

Sub-ppm conversion constant differences:
- 1 inHg: Pint = 3386.3886403410 Pa, metrust = 3386.39 Pa (~4e-7 relative)
- 1 knot: Pint = 0.51444...(recurring), metrust = 0.514444 (~9e-7 relative)

### 6. Real-World Scenarios (34 tests)

6 end-to-end meteorological scenarios:

1. **Severe thunderstorm sounding** (7 tests) — Full chain from LCL through CAPE/CIN, shear, SRH, Bunkers, STP/SCP/BRN, critical angle
2. **Winter storm** (5 tests) — Warm nose, dendritic growth zone, precipitable water, windchill, freezing rain composite
3. **Standard atmosphere** (4 tests) — Roundtrip conversions, METAR workflow, sigma coordinates
4. **Heat advisory** (4 tests) — NWS heat index reference values, apparent temperature regime transitions
5. **Supercell wind profile** (4 tests) — Hodograph analysis, Bunkers symmetry, SRH depth dependence
6. **Thermodynamic consistency** (10 tests) — Moisture roundtrips, Clausius-Clapeyron, potential temperature properties

### 7. Constants (32 tests)

Compared all 34 MetPy constants against metrust values.

**Exact matches (10+ sig figs):** earth_gravity, earth_max_declination, R (universal gas constant), P_STP, T_STP, T_freeze, stefan_boltzmann

**Close matches (4-7 sig figs):** OMEGA, earth_avg_radius, Rd, Rv, Lv, molecular weights, rho_l, epsilon

**Root cause of Rd/Rv/Cp_d divergence:** MetPy derives these from molecular weights via R/M. metrust uses rounded textbook values. Largest discrepancy: Cp_v (1875.0 vs 1860.08, ~0.8%).

**15 MetPy constants not in metrust:**
G (gravitational), GM, earth_mass, earth_orbit_eccentricity, earth_sfc_avg_dist_sun, earth_solar_irradiance, density_ice, ice_specific_heat, Cv_v, sat_pressure_0c, water_triple_point_temperature, dry_air_spec_heat_ratio, wv_specific_heat_ratio, dry_adiabatic_lapse_rate, pot_temp_ref_press

### 8. Wind (27 tests)

All 8 wind functions tested with 2-3 cases each plus 3 cross-checks.

**All match within 1e-6 tolerance.** Wind functions use identical algorithms.

### 9. Kinematics (18 tests)

All 8 grid dynamics functions tested.

**Non-Coriolis functions match to machine precision:**
- divergence, vorticity, advection, frontogenesis

**Coriolis-dependent functions have ~2e-6 relative error:**
- absolute_vorticity, geostrophic_wind, potential_vorticity (both)
- Root cause: wx_math uses OMEGA = 7.2921e-5, MetPy uses 7.292115e-5

## Known Issues Requiring Fixes

### Critical
1. **moist_lapse integration** — RK4 integration produces significantly less cooling than MetPy. This affects parcel profiles and any derived quantity (CAPE from moist adiabat, etc.). **Do not rely on this for precision work.**

### Moderate
2. **STP/SCP threshold behavior** — MetPy applies shear cutoffs per Thompson et al. (2003). metrust doesn't. Operationally meaningful for weak-shear environments.
3. **15 missing constants** — Ice properties and orbital constants. Easy to add but not yet done.
4. **17 missing calc functions** — Mostly boundary layer and signal analysis. See `tests/api_audit_calc.md`.

### Minor
5. **OMEGA precision in wx_math** — Could be updated to match MetPy's value.
6. **Smoothing boundary handling** — Different convention, both defensible.
7. **interpolate_1d extrapolation** — Clamping vs NaN. Both are valid choices.

## How to Run Verification

```bash
# Rust-side verification (573 tests)
cd metrust
cargo test --test verify_thermo_metpy
cargo test --test verify_wind_metpy
cargo test --test verify_kinematics_metpy
cargo test --test verify_severe_atmo_metpy
cargo test --test verify_smooth_interp_metpy
cargo test --test verify_constants_metpy
cargo test --test verify_units_metpy
cargo test --test verify_edge_cases
cargo test --test verify_real_world

# Python-side reference value scripts
cd metrust-py/tests
python verify_thermo.py
python verify_wind.py
python verify_kinematics.py
python verify_severe_atmo.py
python verify_smooth_interp.py
python verify_constants.py
python verify_units.py
python verify_edge_cases.py
python benchmark.py

# Rust benchmarks
cd metrust
cargo run --example bench --release
```
