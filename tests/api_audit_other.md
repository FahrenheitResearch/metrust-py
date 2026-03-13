# MetPy I/O, Interpolate, Constants, Units API Audit

Audit of metrust coverage against MetPy's `metpy.io`, `metpy.interpolate`,
`metpy.constants`, and `metpy.units` modules.

---

## metpy.io

| MetPy Class/Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| Level2File | `io::Level2File` (re-export from wx-radar) | MATCH | Full NEXRAD Level-II parser |
| Level3File | `io::level3::Level3File` | MATCH | Radial (AF1F) and raster (BA0F/BA07) packets supported |
| GempakGrid | -- | MISSING | GEMPAK grid file reader not implemented |
| GempakSounding | -- | MISSING | GEMPAK sounding file reader not implemented |
| GempakSurface | -- | MISSING | GEMPAK surface file reader not implemented |
| GiniFile | -- | MISSING | GINI satellite image reader not implemented |
| StationLookup | `io::station::StationLookup` | MATCH | Lookup by ID, nearest, within_radius |
| add_station_lat_lon | -- | MISSING | No DataFrame-based station enrichment function |
| is_precip_mode | -- | MISSING | No VCP/scan-mode precipitation detector |
| parse_metar_file | `io::metar::parse_metar_file` | MATCH | Parses multi-line METAR files |
| parse_metar_to_dataframe | -- | MISSING | No DataFrame output variant (Rust has no native DataFrames; `parse_metar_file` returns `Vec<Metar>`) |
| parse_wpc_surface_bulletin | -- | MISSING | WPC surface analysis bulletin parser not implemented |
| set_module | N/A | N/A | Python internal helper, not applicable to Rust |

### Additional metrust I/O not in MetPy

metrust provides several I/O capabilities beyond MetPy's scope:

- **GRIB2**: `Grib2File`, `Grib2Message`, `Grib2Writer`, `StreamingParser`, plus utilities (merge, subset, filter, field_stats, etc.)
- **Download**: `DownloadClient`, idx parsing, model sources, variable groups, streaming fetch
- **Radar products**: `RadarProduct`, `sites` module

**Summary**: 5 of 11 MetPy `io` callables have metrust equivalents (excluding `set_module`).
4 GEMPAK/GINI legacy format readers are missing. `parse_metar_to_dataframe`,
`add_station_lat_lon`, `is_precip_mode`, and `parse_wpc_surface_bulletin` are missing.

---

## metpy.interpolate

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `interpolate_1d(x, xp, *args, axis=0, fill_value=nan, return_list_always=False)` | `interpolate::interpolate_1d(x, xp, fp)` | PARTIAL | Single-array only; MetPy supports multiple `*args` arrays and `fill_value` parameter. metrust clamps instead of filling with NaN. |
| `interpolate_nans_1d(x, y, kind='linear')` | `interpolate::interpolate_nans_1d(values)` | PARTIAL | In-place mutation instead of return. No `kind` parameter (linear only). Operates on values directly without separate x-coordinates. |
| `interpolate_to_grid(x, y, z, interp_type, hres, ...)` | `interpolate::interpolate_to_grid(values, src_lats, src_lons, target, method)` | PARTIAL | Different API shape; no `interp_type` dispatch to Cressman/Barnes/natural_neighbor/rbf. Target is a `GridSpec` instead of auto-generated from `hres`. |
| `interpolate_to_isosurface(level_var, interp_var, level, bottom_up_search)` | `interpolate::interpolate_to_isosurface(values_3d, surface_values, target, levels, nx, ny, nz)` | MATCH | Different parameter names but same algorithm. Always searches bottom-up; no `bottom_up_search` toggle. |
| `interpolate_to_points(points, values, xi, interp_type, ...)` | -- | MISSING | No single generic dispatcher; users must call `inverse_distance_to_points` or `natural_neighbor_to_points` directly |
| `interpolate_to_slice(data, points, interp_type)` | `interpolate::interpolate_to_slice(values_3d, levels, lat_slice, lon_slice, src_lats, src_lons, nx, ny, nz)` | PARTIAL | Different API; requires explicit grid dimensions. MetPy works with xarray DataArrays. Linear only. |
| `inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r, gamma, kappa, min_neighbors, kind)` | `interpolate::inverse_distance_to_grid(lats, lons, values, target, power, min_neighbors, search_radius)` | PARTIAL | Uses `GridSpec` target instead of explicit grid arrays. No `kind` switch for Cressman vs Barnes; single power-law weighting. |
| `inverse_distance_to_points(points, values, xi, r, gamma, kappa, min_neighbors, kind)` | `interpolate::inverse_distance_to_points(src_lats, src_lons, src_values, target_lats, target_lons, power, min_neighbors, search_radius)` | PARTIAL | Same as above: no Cressman/Barnes switch. |
| `log_interpolate_1d(x, xp, *args, axis=0, fill_value=nan)` | `interpolate::log_interpolate_1d(x, xp, fp)` | PARTIAL | Single-array only; MetPy supports multiple `*args` arrays. No `fill_value` or `axis` parameter. |
| `natural_neighbor_to_grid(xp, yp, variable, grid_x, grid_y)` | `interpolate::natural_neighbor_to_grid(lats, lons, values, target)` | PARTIAL | Sibson approximation (K-nearest weighted) rather than true Voronoi-based natural neighbor. Uses `GridSpec` instead of explicit grid arrays. |
| `natural_neighbor_to_points(points, values, xi)` | `interpolate::natural_neighbor_to_points(src_lats, src_lons, src_values, target_lats, target_lons)` | PARTIAL | Same Sibson approximation caveat. |
| `remove_nan_observations(x, y, z)` | `interpolate::remove_nan_observations(lats, lons, values)` | MATCH | Identical semantics |
| `remove_observations_below_value(x, y, z, val=0)` | `interpolate::remove_observations_below_value(lats, lons, values, threshold)` | MATCH | Identical semantics |
| `remove_repeat_coordinates(x, y, z)` | `interpolate::remove_repeat_coordinates(lats, lons, values)` | MATCH | Identical semantics |
| `geodesic(crs, start, end, steps)` | `interpolate::geodesic(start, end, n_points)` | PARTIAL | No CRS parameter; always uses WGS84 spherical. Returns `(Vec<f64>, Vec<f64>)` instead of array of coordinate pairs. |
| `cross_section(data, start, end, steps, interp_type)` | `interpolate::cross_section_data` (re-export from wx_math) | PARTIAL | Re-exported from wx_math::regrid. MetPy version works with xarray DataArrays. |
| `set_module(globls)` | N/A | N/A | Python internal helper |

### Additional metrust interpolation not in MetPy

- `interpolate::interpolate_to_slice` -- vertical cross-section extraction with bilinear horizontal interpolation
- `interpolate::regrid`, `interpolate_point`, `interpolate_points` -- re-exported from wx_math for general regridding
- `interpolate::interpolate_vertical` -- re-exported from wx_math

**Summary**: 14 of 16 MetPy interpolate functions have some metrust equivalent
(excluding `set_module`). 1 is fully missing (`interpolate_to_points` as a
generic dispatcher). Most matches are PARTIAL due to API shape differences
(Rust has no xarray/DataFrames) or missing parameters (Cressman/Barnes
weighting, `fill_value`, multi-array `*args` support).

---

## metpy.constants

### Thermodynamic / Gas Constants

| MetPy Constant | Value | metrust Equivalent | Status | Notes |
|---|---|---|---|---|
| `R` | 8.314462618 J/(mol K) | `R` = 8.314462618 | MATCH | Universal gas constant |
| `Rd` / `dry_air_gas_constant` | 287.047 J/(kg K) | `RD` / `Rd` = 287.058 | MATCH | Minor value difference (MetPy 287.047 vs metrust 287.058) |
| `Rv` / `water_gas_constant` | 461.523 J/(kg K) | `RV` / `Rv` = 461.5 | MATCH | Minor value difference |
| `epsilon` / `molecular_weight_ratio` | 0.62196 | `EPSILON` | MATCH | Computed as Mw/Md |

### Dry Air Properties

| MetPy Constant | Value | metrust Equivalent | Status | Notes |
|---|---|---|---|---|
| `Cp_d` / `dry_air_spec_heat_press` | 1004.67 J/(kg K) | `CP_D` / `Cp_d` = 1005.7 | MATCH | Minor value difference |
| `Cv_d` / `dry_air_spec_heat_vol` | 717.62 J/(kg K) | `CV_D` / `Cv_d` = 718.0 | MATCH | Minor value difference |
| `Md` / `dry_air_molecular_weight` | 0.02897 kg/mol | `MOLECULAR_WEIGHT_DRY_AIR` = 0.028965 | MATCH | |
| `kappa` / `poisson_exponent` | 0.28571 | `KAPPA` / `POISSON_EXPONENT_DRY_AIR` | MATCH | Computed as Rd/Cp_d |
| `rho_d` / `dry_air_density_stp` | 1.2754 kg/m^3 | `RHO_D_STP` / `DRY_AIR_DENSITY_STP` | MATCH | |
| `dry_air_spec_heat_ratio` | 1.4 | -- | MISSING | gamma_d = Cp_d / Cv_d ratio |

### Water / Moisture Properties

| MetPy Constant | Value | metrust Equivalent | Status | Notes |
|---|---|---|---|---|
| `Cp_v` / `wv_specific_heat_press` | 1860.08 J/(kg K) | `CP_V` / `WATER_SPECIFIC_HEAT_VAPOR` = 1875.0 | MATCH | Value difference |
| `Cv_v` / `wv_specific_heat_vol` | 1398.55 J/(kg K) | -- | MISSING | Water vapor specific heat at constant volume |
| `Cp_i` / `ice_specific_heat` | 2090 J/(kg K) | -- | MISSING | Specific heat of ice |
| `Cp_l` / `water_specific_heat` | 4219.4 J/(kg K) | `CP_L` / `WATER_SPECIFIC_HEAT_LIQUID` = 4218.0 | MATCH | Minor value difference |
| `Mw` / `water_molecular_weight` | 18.015268 g/mol | `MOLECULAR_WEIGHT_WATER` = 18.015e-3 kg/mol | MATCH | |
| `Lv` / `water_heat_vaporization` | 2500840 J/kg | `LV` / `Lv` = 2.501e6 | MATCH | |
| `Lf` / `water_heat_fusion` | 333700 J/kg | `LF` / `Lf` = 3.34e5 | MATCH | |
| `Ls` / `water_heat_sublimation` | 2834540 J/kg | `LS` / `Ls` = 2.834e6 | MATCH | |
| `rho_l` / `density_water` | 999.975 kg/m^3 | `RHO_L` = 999.97 | MATCH | |
| `rho_i` / `density_ice` | 917 kg/m^3 | -- | MISSING | Density of ice |
| `T0` / `water_triple_point_temperature` | 273.16 K | -- | MISSING | Triple point temp (metrust has T_FREEZE=273.15 which is freezing point, not triple point) |
| `sat_pressure_0c` | 6.112 mbar | -- | MISSING | Saturation vapor pressure at 0 C |
| `wv_specific_heat_ratio` | 1.33 | -- | MISSING | gamma for water vapor |
| `P0` / `pot_temp_ref_press` | 1000 mbar | -- | MISSING | Reference pressure for potential temperature |

### Earth / Planetary Constants

| MetPy Constant | Value | metrust Equivalent | Status | Notes |
|---|---|---|---|---|
| `g` / `earth_gravity` | 9.80665 m/s^2 | `EARTH_GRAVITY` / `g` | MATCH | |
| `Re` / `earth_avg_radius` | 6371008.77 m | `EARTH_AVG_RADIUS` / `Re` = 6371229 | MATCH | Minor value difference |
| `omega` / `earth_avg_angular_vel` | 7.292115e-5 rad/s | `OMEGA` | MATCH | |
| `delta` / `earth_max_declination` | 23.45 deg | `EARTH_MAX_DECLINATION` | MATCH | |
| `gamma_d` / `dry_adiabatic_lapse_rate` | 9.761 K/km | -- | MISSING | Dry adiabatic lapse rate |
| `me` / `earth_mass` | 5.972e24 kg | -- | MISSING | Mass of the Earth |
| `d` / `earth_sfc_avg_dist_sun` | 1.496e11 m | -- | MISSING | Earth-Sun average distance |
| `e` / `earth_orbit_eccentricity` | 0.0167 | -- | MISSING | Orbital eccentricity |
| `S` / `earth_solar_irradiance` | 1360.8 W/m^2 | -- | MISSING | Total solar irradiance |
| `G` / `gravitational_constant` | 6.6743e-11 m^3/(kg s^2) | -- | MISSING | Newton's gravitational constant |
| `GM` / `geocentric_gravitational_constant` | 3.986e14 m^3/s^2 | -- | MISSING | Geocentric gravitational constant |

### Additional metrust constants not in MetPy

| metrust Constant | Value | Notes |
|---|---|---|
| `STEFAN_BOLTZMANN` | 5.670374419e-8 W/(m^2 K^4) | Stefan-Boltzmann constant |
| `EARTH_AVG_DENSITY` | 5515.0 kg/m^3 | Mean Earth density |
| `P_STP` | 101325.0 Pa | Standard atmosphere pressure |
| `T_STP` | 288.15 K | Standard atmosphere temperature |
| `T_FREEZE` | 273.15 K | Freezing point of water |

**Summary**: 20 of 34 MetPy constants have metrust equivalents (counting
short aliases and long names as the same constant). 14 constants are missing,
mostly planetary/orbital constants and some secondary thermodynamic values
(ice properties, heat capacity ratios, saturation pressure at 0C, etc.).

---

## metpy.units

| MetPy Feature | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `pint.UnitRegistry` (full unit system) | `units::Units` enum | PARTIAL | MetPy uses pint for arbitrary unit expressions and dimensional analysis. metrust has a fixed enum of ~18 unit types. |
| Quantity arithmetic (`3 * units.meter`) | -- | MISSING | No quantity type that carries units through arithmetic. All metrust functions use bare `f64`. |
| `.to()` unit conversion | `units::convert(value, from, to)` | PARTIAL | Function-based conversion instead of method on a quantity object |
| Temperature units (K, degC, degF) | `Units::Kelvin`, `Units::Celsius`, `Units::Fahrenheit` | MATCH | |
| Pressure units (Pa, hPa, mbar, inHg) | `Units::Pascal`, `Units::Hectopascal`, `Units::Millibar`, `Units::InchesOfMercury` | MATCH | |
| Speed units (m/s, knot, mph) | `Units::MetersPerSecond`, `Units::Knots`, `Units::MPH` | MATCH | |
| Length units (m, ft, km, mi) | `Units::Meters`, `Units::Feet`, `Units::Kilometers`, `Units::Miles` | MATCH | |
| Mixing ratio units (kg/kg, g/kg) | `Units::KgPerKg`, `Units::GramsPerKg` | MATCH | |
| Compound units (e.g. `J/(kg K)`, `K/km`) | -- | MISSING | No support for compound/derived units |
| `dimensionless` | -- | MISSING | No explicit dimensionless unit (though `Percent` and `Dbz` exist) |
| Unit parsing from strings | -- | MISSING | Cannot construct units from string names |
| Unit compatibility checking | `units::unit_category()` | PARTIAL | Category-based checking, but not full dimensional analysis |
| Convenience converters | `celsius_to_kelvin`, `knots_to_ms`, `hpa_to_pa`, etc. | MATCH | Inline helpers for common conversions |

### Additional metrust units not in MetPy's core

| metrust Unit | Notes |
|---|---|
| `Units::Dbz` | Radar reflectivity (dBZ) |
| `Units::Degrees` | Angular degrees |
| `Units::Percent` | Percentage |

**Summary**: metrust provides a practical subset of unit conversions covering
the most common meteorological units. However, it lacks the full dimensional
analysis system that pint provides (quantity objects, compound units, unit
arithmetic, string parsing). This is a fundamental architectural difference --
Rust's type system makes a pint-equivalent impractical for the same use
patterns.

---

## Overall Summary

| Module | MetPy Items | metrust Matches | Missing | Coverage |
|---|---|---|---|---|
| `metpy.io` | 11 (excl. set_module) | 5 | 6 | 45% |
| `metpy.interpolate` | 16 (excl. set_module) | 14 (most PARTIAL) | 1 fully missing | 88% |
| `metpy.constants` | ~34 unique constants | 20 | 14 | 59% |
| `metpy.units` | Full unit system | Enum-based subset | Quantity system, compound units | ~40% |

### Key Gaps

**I/O -- 6 missing items:**
1. `GempakGrid` -- GEMPAK grid file reader
2. `GempakSounding` -- GEMPAK sounding file reader
3. `GempakSurface` -- GEMPAK surface file reader
4. `GiniFile` -- GINI satellite image format reader
5. `parse_metar_to_dataframe` -- DataFrame-style METAR output
6. `parse_wpc_surface_bulletin` -- WPC bulletin parser
7. `add_station_lat_lon` -- Station coordinate enrichment
8. `is_precip_mode` -- VCP precipitation mode detection

**Constants -- 14 missing:**
1. `Cv_v` -- Water vapor Cv
2. `Cp_i` / `ice_specific_heat` -- Ice specific heat
3. `rho_i` / `density_ice` -- Ice density
4. `T0` / `water_triple_point_temperature` -- Triple point
5. `sat_pressure_0c` -- Saturation pressure at 0C
6. `P0` / `pot_temp_ref_press` -- Potential temperature reference pressure
7. `dry_air_spec_heat_ratio` -- Dry air gamma
8. `wv_specific_heat_ratio` -- Water vapor gamma
9. `gamma_d` / `dry_adiabatic_lapse_rate` -- Dry adiabatic lapse rate
10. `G` / `gravitational_constant` -- Newton's G
11. `GM` / `geocentric_gravitational_constant`
12. `me` / `earth_mass`
13. `d` / `earth_sfc_avg_dist_sun`
14. `S` / `earth_solar_irradiance`
15. `e` / `earth_orbit_eccentricity`

**Units -- structural gap:**
- No quantity objects (values carrying their units through computation)
- No compound/derived units
- No string-based unit parsing
- No dimensional analysis
