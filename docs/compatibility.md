# metrust / MetPy Compatibility Matrix

This document is the definitive reference for what metrust implements natively
in Rust versus what it delegates to MetPy.

---

## 1. Compatibility Model

metrust is a drop-in replacement for `metpy.calc` backed by a compiled Rust
engine (`_metrust`).  The compatibility model has three tiers:

| Tier | Description |
|------|-------------|
| **Native Rust** | Function is implemented entirely in Rust. The Python wrapper strips Pint units, calls the Rust function, and re-attaches units. No MetPy dependency. |
| **Native + Rust Array Binding** | Same as Native Rust, but the function also exposes a vectorized `_array` variant in the Rust extension so that array inputs are processed in a single FFI call instead of element-wise Python loops. 28 functions have this optimization. |
| **MetPy Shim** | Module forwards attribute lookups to the corresponding MetPy module via lazy import. MetPy must be installed separately. Used for `plots`, `xarray`, and `io.Level2File`. |

When MetPy is not installed, all `metrust.calc` functions still work (they
never import MetPy).  Only the shimmed surfaces (`metrust.plots`,
`metrust.xarray`, `metrust.io.Level2File`) require MetPy at runtime.

---

## 2. metrust.calc Function Matrix

**Status key:**

| Code | Meaning |
|------|---------|
| **Native + Array** | Rust scalar function + Rust `_array` binding for vectorized calls |
| **Native** | Rust scalar function; arrays are handled by `_vec_call` (Python loop over Rust scalar) |
| **Native (profile)** | Rust function that operates on 1-D sounding profiles natively |
| **Native (grid)** | Rust function that operates on 2-D or 3-D gridded arrays natively |
| **Not implemented** | No equivalent in metrust |

### 2.1 Thermodynamic Functions

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `potential_temperature` | Native + Array | `metpy.calc.potential_temperature` | |
| `equivalent_potential_temperature` | Native + Array | `metpy.calc.equivalent_potential_temperature` | |
| `saturation_vapor_pressure` | Native + Array | `metpy.calc.saturation_vapor_pressure` | Liquid phase uses Rust array binding; ice/auto phase computed in Python using Ambaum (2020). See section 5 for numerical differences. |
| `saturation_mixing_ratio` | Native + Array | `metpy.calc.saturation_mixing_ratio` | Liquid phase uses Rust array binding; ice/auto phase falls back to Python. |
| `wet_bulb_temperature` | Native + Array | `metpy.calc.wet_bulb_temperature` | Different iterative approximation from MetPy. See section 5. |
| `virtual_temperature` | Native + Array | `metpy.calc.virtual_temperature` | Two call signatures: (T, mixing_ratio) for MetPy compat, (T, P, Td) for Rust-native path. Array binding via `virtual_temp_array`. |
| `virtual_temperature_from_dewpoint` | Native + Array | `metpy.calc.virtual_temperature_from_dewpoint` | |
| `dewpoint_from_relative_humidity` | Native + Array | `metpy.calc.dewpoint_from_relative_humidity` | Array binding via `dewpoint_from_rh_array`. |
| `relative_humidity_from_dewpoint` | Native + Array | `metpy.calc.relative_humidity_from_dewpoint` | Array binding via `rh_from_dewpoint_array`. Supports phase parameter. |
| `mixing_ratio` | Native + Array | `metpy.calc.mixing_ratio` | Overloaded: (P, T) or (e, p_total). Array binding for (P, T) path. |
| `vapor_pressure` | Native + Array | `metpy.calc.vapor_pressure` | Overloaded: from dewpoint or from (P, w). Array binding for dewpoint path. |
| `density` | Native + Array | `metpy.calc.density` | |
| `exner_function` | Native + Array | `metpy.calc.exner_function` | |
| `dewpoint` | Native + Array | `metpy.calc.dewpoint` | From vapor pressure (hPa). |
| `temperature_from_potential_temperature` | Native + Array | `metpy.calc.temperature_from_potential_temperature` | |
| `virtual_potential_temperature` | Native + Array | `metpy.calc.virtual_potential_temperature` | |
| `saturation_equivalent_potential_temperature` | Native + Array | `metpy.calc.saturation_equivalent_potential_temperature` | |
| `wet_bulb_potential_temperature` | Native + Array | `metpy.calc.wet_bulb_potential_temperature` | |
| `frost_point` | Native + Array | `metpy.calc.frost_point` | |
| `mixing_ratio_from_relative_humidity` | Native + Array | `metpy.calc.mixing_ratio_from_relative_humidity` | |
| `relative_humidity_from_mixing_ratio` | Native + Array | `metpy.calc.relative_humidity_from_mixing_ratio` | |
| `relative_humidity_from_specific_humidity` | Native + Array | `metpy.calc.relative_humidity_from_specific_humidity` | |
| `specific_humidity_from_dewpoint` | Native + Array | `metpy.calc.specific_humidity_from_dewpoint` | |
| `dewpoint_from_specific_humidity` | Native + Array | `metpy.calc.dewpoint_from_specific_humidity` | |
| `mixing_ratio_from_specific_humidity` | Native + Array | `metpy.calc.mixing_ratio_from_specific_humidity` | |
| `specific_humidity_from_mixing_ratio` | Native + Array | `metpy.calc.specific_humidity_from_mixing_ratio` | |
| `lcl` | Native | `metpy.calc.lcl` | Scalar only. Returns (p_lcl, t_lcl). Rust also exposes `lcl_pressure` and `lcl_pressure_array` internally. See section 5 for approximation differences. |
| `lfc` | Native (profile) | `metpy.calc.lfc` | Operates on full sounding profiles. |
| `el` | Native (profile) | `metpy.calc.el` | Operates on full sounding profiles. |
| `cape_cin` | Native (profile) | `metpy.calc.cape_cin` | Extended signature: supports parcel_type, ml_depth, mu_depth, top_m, and returns (CAPE, CIN, LCL height, LFC height). |
| `surface_based_cape_cin` | Native (profile) | `metpy.calc.surface_based_cape_cin` | |
| `mixed_layer_cape_cin` | Native (profile) | `metpy.calc.mixed_layer_cape_cin` | |
| `most_unstable_cape_cin` | Native (profile) | `metpy.calc.most_unstable_cape_cin` | |
| `downdraft_cape` | Native (profile) | `metpy.calc.downdraft_cape` | |
| `parcel_profile` | Native (profile) | `metpy.calc.parcel_profile` | |
| `parcel_profile_with_lcl` | Native (profile) | `metpy.calc.parcel_profile_with_lcl` | |
| `dry_lapse` | Native (profile) | `metpy.calc.dry_lapse` | |
| `moist_lapse` | Native (profile) | `metpy.calc.moist_lapse` | |
| `ccl` | Native (profile) | `metpy.calc.ccl` | Convective Condensation Level. |
| `lifted_index` | Native (profile) | `metpy.calc.lifted_index` | |
| `showalter_index` | Native (profile) | `metpy.calc.showalter_index` | |
| `k_index` | Native | `metpy.calc.k_index` | Takes 5 scalar level values. |
| `total_totals` | Native | `metpy.calc.total_totals` | Takes 3 scalar level values. |
| `cross_totals` | Native | `metpy.calc.cross_totals` | |
| `vertical_totals` | Native | `metpy.calc.vertical_totals` | |
| `sweat_index` | Native | `metpy.calc.sweat_index` | |
| `precipitable_water` | Native (profile) | `metpy.calc.precipitable_water` | |
| `brunt_vaisala_frequency` | Native (profile) | `metpy.calc.brunt_vaisala_frequency` | |
| `brunt_vaisala_period` | Native (profile) | `metpy.calc.brunt_vaisala_period` | |
| `brunt_vaisala_frequency_squared` | Native (profile) | `metpy.calc.brunt_vaisala_frequency_squared` | |
| `static_stability` | Native (profile) | `metpy.calc.static_stability` | |
| `isentropic_interpolation` | Native (grid) | `metpy.calc.isentropic_interpolation` | Operates on 3-D grids. |
| `mean_pressure_weighted` | Native (profile) | `metpy.calc.mean_pressure_weighted` | |
| `mixed_layer` | Native (profile) | `metpy.calc.mixed_layer` | |
| `get_layer` | Native (profile) | `metpy.calc.get_layer` | |
| `get_layer_heights` | Native (profile) | `metpy.calc.get_layer_heights` | |
| `get_mixed_layer_parcel` | Native (profile) | `metpy.calc.mixed_parcel` | |
| `get_most_unstable_parcel` | Native (profile) | `metpy.calc.most_unstable_parcel` | |
| `mixed_parcel` | Native (profile) | `metpy.calc.mixed_parcel` | Alias for `get_mixed_layer_parcel`. |
| `most_unstable_parcel` | Native (profile) | `metpy.calc.most_unstable_parcel` | Alias for `get_most_unstable_parcel`. |
| `find_intersections` | Native (profile) | `metpy.calc.find_intersections` | |
| `moist_air_gas_constant` | Native | `metpy.calc.moist_air_gas_constant` | via `_vec_call` |
| `moist_air_specific_heat_pressure` | Native | `metpy.calc.moist_air_specific_heat_pressure` | via `_vec_call` |
| `moist_air_poisson_exponent` | Native | `metpy.calc.moist_air_poisson_exponent` | via `_vec_call` |
| `water_latent_heat_vaporization` | Native | (no direct MetPy equivalent) | Temperature-dependent L_v. via `_vec_call` |
| `water_latent_heat_melting` | Native | (no direct MetPy equivalent) | Temperature-dependent L_f. via `_vec_call` |
| `water_latent_heat_sublimation` | Native | (no direct MetPy equivalent) | Temperature-dependent L_s. via `_vec_call` |
| `relative_humidity_wet_psychrometric` | Native | `metpy.calc.relative_humidity_wet_psychrometric` | via `_vec_call` |
| `psychrometric_vapor_pressure` | Native | `metpy.calc.psychrometric_vapor_pressure_wet` | via `_vec_call` |
| `psychrometric_vapor_pressure_wet` | Native | `metpy.calc.psychrometric_vapor_pressure_wet` | Alias for `psychrometric_vapor_pressure`. |
| `weighted_continuous_average` | Native (profile) | (no direct MetPy equivalent) | Trapezoidal weighted average. |
| `get_perturbation` | Native (profile) | `metpy.calc.get_perturbation` | |
| `add_height_to_pressure` | Native | `metpy.calc.add_height_to_pressure` | via `_vec_call` |
| `add_pressure_to_height` | Native | `metpy.calc.add_pressure_to_height` | via `_vec_call` |
| `thickness_hydrostatic` | Native | `metpy.calc.thickness_hydrostatic` | Scalar hypsometric equation. via `_vec_call` |
| `thickness_hydrostatic_from_relative_humidity` | Native (profile) | `metpy.calc.thickness_hydrostatic_from_relative_humidity` | Profile-based virtual temperature integration. |
| `dry_static_energy` | Native | `metpy.calc.dry_static_energy` | via `_vec_call` |
| `moist_static_energy` | Native | `metpy.calc.moist_static_energy` | via `_vec_call` |
| `montgomery_streamfunction` | Native | `metpy.calc.montgomery_streamfunction` | via `_vec_call` |
| `vertical_velocity` | Native | `metpy.calc.vertical_velocity` | omega to w conversion. via `_vec_call` |
| `vertical_velocity_pressure` | Native | `metpy.calc.vertical_velocity_pressure` | w to omega conversion. via `_vec_call` |
| `scale_height` | Native | `metpy.calc.scale_height` | via `_vec_call` |
| `geopotential_to_height` | Native | `metpy.calc.geopotential_to_height` | via `_vec_call` |
| `height_to_geopotential` | Native | `metpy.calc.height_to_geopotential` | via `_vec_call` |
| `reduce_point_density` | Native | `metpy.calc.reduce_point_density` | |
| `galvez_davison_index` | Native | `metpy.calc.galvez_davison_index` | |

### 2.2 Moisture Functions

Moisture functions are listed in the thermodynamic table above where they
appear in the source. The 28 functions with Rust array bindings include all
major moisture conversions:

- `dewpoint_from_rh_array` (backing `dewpoint_from_relative_humidity`)
- `rh_from_dewpoint_array` (backing `relative_humidity_from_dewpoint`)
- `mixing_ratio_array` (backing `mixing_ratio`)
- `vapor_pressure_array` (backing `vapor_pressure`)
- `mixing_ratio_from_relative_humidity_array`
- `relative_humidity_from_mixing_ratio_array`
- `relative_humidity_from_specific_humidity_array`
- `specific_humidity_from_dewpoint_array`
- `dewpoint_from_specific_humidity_array`
- `mixing_ratio_from_specific_humidity_array`
- `specific_humidity_from_mixing_ratio_array`
- `specific_humidity_array`
- `frost_point_array`
- `dewpoint_array`
- `density_array`

### 2.3 Wind Functions

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `wind_speed` | Native (profile) | `metpy.calc.wind_speed` | |
| `wind_direction` | Native (profile) | `metpy.calc.wind_direction` | |
| `wind_components` | Native (profile) | `metpy.calc.wind_components` | |
| `bulk_shear` | Native (profile) | `metpy.calc.bulk_shear` | |
| `mean_wind` | Native (profile) | `metpy.calc.mean_wind` | |
| `storm_relative_helicity` | Native (profile) | `metpy.calc.storm_relative_helicity` | Returns (positive, negative, total). |
| `bunkers_storm_motion` | Native (profile) | `metpy.calc.bunkers_storm_motion` | Returns (right, left, mean). |
| `corfidi_storm_motion` | Native (profile) | `metpy.calc.corfidi_storm_motion` | Returns (upwind, downwind). |
| `friction_velocity` | Native (profile) | `metpy.calc.friction_velocity` | |
| `tke` | Native (profile) | `metpy.calc.tke` | Turbulent kinetic energy. |
| `gradient_richardson_number` | Native (profile) | `metpy.calc.gradient_richardson_number` | |

### 2.4 Kinematics (2-D Gridded)

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `divergence` | Native (grid) | `metpy.calc.divergence` | |
| `vorticity` | Native (grid) | `metpy.calc.vorticity` | |
| `absolute_vorticity` | Native (grid) | `metpy.calc.absolute_vorticity` | |
| `advection` | Native (grid) | `metpy.calc.advection` | |
| `frontogenesis` | Native (grid) | `metpy.calc.frontogenesis` | Petterssen frontogenesis. |
| `geostrophic_wind` | Native (grid) | `metpy.calc.geostrophic_wind` | |
| `ageostrophic_wind` | Native (grid) | `metpy.calc.ageostrophic_wind` | |
| `potential_vorticity_baroclinic` | Native (grid) | `metpy.calc.potential_vorticity_baroclinic` | Ertel PV. |
| `potential_vorticity_barotropic` | Native (grid) | `metpy.calc.potential_vorticity_barotropic` | |
| `normal_component` | Native (profile) | `metpy.calc.normal_component` | Cross-section decomposition. |
| `tangential_component` | Native (profile) | `metpy.calc.tangential_component` | Cross-section decomposition. |
| `unit_vectors_from_cross_section` | Native | `metpy.calc.unit_vectors_from_cross_section` | |
| `vector_derivative` | Native (grid) | (no single MetPy equivalent) | Returns all four partials (du/dx, du/dy, dv/dx, dv/dy). |
| `absolute_momentum` | Native (profile) | `metpy.calc.absolute_momentum` | |
| `coriolis_parameter` | Native | `metpy.calc.coriolis_parameter` | |
| `cross_section_components` | Native (profile) | `metpy.calc.cross_section_components` | |
| `curvature_vorticity` | Native (grid) | `metpy.calc.curvature_vorticity` | |
| `inertial_advective_wind` | Native (grid) | `metpy.calc.inertial_advective_wind` | |
| `kinematic_flux` | Native (profile) | `metpy.calc.kinematic_flux` | |
| `q_vector` | Native (grid) | `metpy.calc.q_vector` | |
| `shear_vorticity` | Native (grid) | `metpy.calc.shear_vorticity` | |
| `shearing_deformation` | Native (grid) | `metpy.calc.shearing_deformation` | |
| `stretching_deformation` | Native (grid) | `metpy.calc.stretching_deformation` | |
| `total_deformation` | Native (grid) | `metpy.calc.total_deformation` | |
| `geospatial_gradient` | Native (grid) | (no direct MetPy equivalent) | Gradient on lat/lon grids with spherical corrections. |
| `geospatial_laplacian` | Native (grid) | (no direct MetPy equivalent) | Laplacian on lat/lon grids with spherical corrections. |
| `advection_3d` | Native (grid) | (no direct MetPy equivalent) | 3-D advection including vertical term. |

### 2.5 Severe Weather Parameters

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `significant_tornado_parameter` | Native | `metpy.calc.significant_tornado` | See section 5 for cutoff differences. |
| `supercell_composite_parameter` | Native | `metpy.calc.supercell_composite` | See section 5 for cutoff differences. |
| `critical_angle` | Native | `metpy.calc.critical_angle` | |
| `boyden_index` | Native | (no direct MetPy equivalent) | |
| `bulk_richardson_number` | Native | `metpy.calc.bulk_richardson_number` | |
| `convective_inhibition_depth` | Native (profile) | (no direct MetPy equivalent) | |
| `dendritic_growth_zone` | Native (profile) | (no direct MetPy equivalent) | Returns (p_bottom, p_top). |
| `fosberg_fire_weather_index` | Native | `metpy.calc.fosberg_fire_weather_index` | |
| `freezing_rain_composite` | Native (profile) | (no direct MetPy equivalent) | |
| `haines_index` | Native | (no direct MetPy equivalent) | Fire weather. |
| `hot_dry_windy` | Native | (no direct MetPy equivalent) | |
| `warm_nose_check` | Native (profile) | (no direct MetPy equivalent) | |
| `galvez_davison_index` | Native | `metpy.calc.galvez_davison_index` | Tropical thunderstorm potential. |

### 2.6 Atmospheric / Standard Atmosphere

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `pressure_to_height_std` | Native | `metpy.calc.pressure_to_height_std` | US Standard Atmosphere 1976. |
| `height_to_pressure_std` | Native | `metpy.calc.height_to_pressure_std` | US Standard Atmosphere 1976. |
| `altimeter_to_station_pressure` | Native | `metpy.calc.altimeter_to_station_pressure` | |
| `station_to_altimeter_pressure` | Native | (no direct MetPy equivalent) | |
| `altimeter_to_sea_level_pressure` | Native | `metpy.calc.altimeter_to_sea_level_pressure` | |
| `sigma_to_pressure` | Native | `metpy.calc.sigma_to_pressure` | |
| `heat_index` | Native | `metpy.calc.heat_index` | NWS Rothfusz regression. via `_vec_call` |
| `windchill` | Native | `metpy.calc.windchill` | NWS formula. via `_vec_call` |
| `apparent_temperature` | Native | `metpy.calc.apparent_temperature` | Combines heat index and wind chill. via `_vec_call` |

### 2.7 Smoothing / Spatial Derivatives

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `smooth_gaussian` | Native (grid) | `metpy.calc.smooth_gaussian` | |
| `smooth_rectangular` | Native (grid) | `metpy.calc.smooth_rectangular` | |
| `smooth_circular` | Native (grid) | `metpy.calc.smooth_circular` | |
| `smooth_n_point` | Native (grid) | `metpy.calc.smooth_n_point` | 5-point or 9-point. |
| `smooth_window` | Native (grid) | `metpy.calc.smooth_window` | Generic convolution with user kernel. |
| `gradient` | Native (grid) | `metpy.calc.gradient` | Falls back to numpy.gradient for non-2D. |
| `gradient_x` | Native (grid) | (no direct MetPy equivalent) | Partial df/dx. |
| `gradient_y` | Native (grid) | (no direct MetPy equivalent) | Partial df/dy. |
| `laplacian` | Native (grid) | `metpy.calc.laplacian` | |
| `first_derivative` | Native (grid) | `metpy.calc.first_derivative` | Along a chosen axis. |
| `second_derivative` | Native (grid) | `metpy.calc.second_derivative` | Along a chosen axis. |
| `lat_lon_grid_deltas` | Native (grid) | `metpy.calc.lat_lon_grid_deltas` | |

### 2.8 Utility Functions

| Function | Status | MetPy Equivalent | Notes |
|----------|--------|-------------------|-------|
| `angle_to_direction` | Native | `metpy.calc.angle_to_direction` | 8, 16, or 32 compass points. |
| `parse_angle` | Native | `metpy.calc.parse_angle` | |
| `find_bounding_indices` | Native | (no direct MetPy equivalent) | |
| `nearest_intersection_idx` | Native | (no direct MetPy equivalent) | |
| `resample_nn_1d` | Native | (no direct MetPy equivalent) | Nearest-neighbour 1-D resampling. |
| `find_peaks` | Native | (no direct MetPy equivalent) | IQR-filtered peak detection. |
| `peak_persistence` | Native | (no direct MetPy equivalent) | Topological persistence peak ranking. |
| `azimuth_range_to_lat_lon` | Native | (no direct MetPy equivalent) | Radar azimuth/range to lat/lon. |

### 2.9 Compatibility Aliases

| Alias | Resolves To |
|-------|-------------|
| `significant_tornado` | `significant_tornado_parameter` |
| `supercell_composite` | `supercell_composite_parameter` |
| `total_totals_index` | `total_totals` |
| `mixed_parcel` | `get_mixed_layer_parcel` |
| `most_unstable_parcel` | `get_most_unstable_parcel` |
| `psychrometric_vapor_pressure_wet` | `psychrometric_vapor_pressure` |

### 2.10 Exceptions

| Name | Notes |
|------|-------|
| `InvalidSoundingError` | Raised when sounding data is invalid or insufficient. Exported from `metrust.calc`. |

---

## 3. Grid Composite Functions (metrust-only)

These functions operate on full 3-D (nz, ny, nx) or 2-D (ny, nx) grids and
are parallelized in Rust.  They have no MetPy equivalent -- MetPy does not
provide whole-grid composite kernels.

| Function | Input Shape | Output Shape | Description |
|----------|-------------|--------------|-------------|
| `compute_cape_cin` | 3-D + 2-D surface | (ny, nx) x 4 | CAPE, CIN, LCL height, LFC height for every grid point. Supports surface/ML/MU parcels. |
| `compute_srh` | 3-D | (ny, nx) | Storm-relative helicity (default 0-1 km). |
| `compute_shear` | 3-D | (ny, nx) | Bulk wind shear over a configurable layer. |
| `compute_lapse_rate` | 3-D | (ny, nx) | Environmental lapse rate (C/km) over a configurable layer. |
| `compute_pw` | 3-D | (ny, nx) | Precipitable water (mm). |
| `compute_stp` | 2-D | (ny, nx) | Significant Tornado Parameter from pre-computed fields. |
| `compute_scp` | 2-D | (ny, nx) | Supercell Composite Parameter from pre-computed fields. |
| `compute_ehi` | 2-D | (ny, nx) | Energy-Helicity Index: (CAPE * SRH) / 160000. |
| `compute_ship` | 2-D | (ny, nx) | Significant Hail Parameter. |
| `compute_dcp` | 2-D | (ny, nx) | Derecho Composite Parameter. |
| `compute_grid_scp` | 2-D | (ny, nx) | Enhanced SCP with CIN term. |
| `compute_grid_critical_angle` | 2-D | (ny, nx) | Critical angle on 2-D wind fields. |
| `composite_reflectivity` | 3-D | (ny, nx) | Column-maximum reflectivity from a 3-D dBZ field. |
| `composite_reflectivity_from_hydrometeors` | 3-D | (ny, nx) | Composite reflectivity derived from rain/snow/graupel mixing ratios. |

---

## 4. Shimmed Surfaces

These modules forward attribute lookups to MetPy at runtime.  MetPy must be
installed separately (`pip install metpy`).

| Module | MetPy Module | What It Provides |
|--------|-------------|------------------|
| `metrust.plots` | `metpy.plots` | StationPlot, SkewT, Hodograph, and all other matplotlib-based plotting classes. |
| `metrust.xarray` | `metpy.xarray` | xarray accessor for CRS-aware coordinate handling (`.metpy` accessor). |
| `metrust.io.Level2File` | `metpy.io.Level2File` | NEXRAD Level 2 radar archive file reader. All other I/O classes (Level3File, Metar, GiniFile, GEMPAK formats) are native Rust. |

---

## 5. Known Numerical Differences

metrust and MetPy produce the same results within tight tolerances for most
functions, but several areas use different empirical fits or iteration
strategies.  These are documented here so that users performing exact
regression comparisons know what to expect.

### 5.1 Saturation Vapor Pressure (SVP)

Both metrust and MetPy implement the Ambaum (2020) formulation, but the Rust
engine uses a slightly different constant set for the liquid-phase
calculation.  Typical differences are < 0.01 hPa across the meteorologically
relevant range (-40 to +50 C), but can reach ~0.05 hPa at extreme
temperatures.

The ice-phase SVP in metrust uses Ambaum (2020) Eq. 17 directly, matching
MetPy's implementation.

### 5.2 LCL and Wet-Bulb Temperature

LCL pressure in metrust is computed via direct iterative inversion in Rust,
while MetPy uses a different iterative scheme.  Typical differences are
< 0.5 hPa for LCL pressure and < 0.1 K for LCL temperature.

Wet-bulb temperature uses a Rust-native Newton iteration that can differ from
MetPy's approach by up to ~0.2 C in extreme humidity conditions.

### 5.3 STP and SCP Cutoff Behavior

The Significant Tornado Parameter and Supercell Composite Parameter in
metrust apply hard cutoffs slightly differently from MetPy in the weak-shear
regime:

- **STP**: metrust normalizations use CAPE/1500, LCL/1500, SRH/150,
  shear/20.  Behavior when shear < 12.5 m/s may differ from MetPy's linear
  ramp.
- **SCP**: metrust normalizations use CAPE/1000, SRH/50, shear/40.  Behavior
  when shear < 10 m/s may differ from MetPy's linear ramp.

For typical severe weather environments (moderate to strong shear), the
values agree closely.

### 5.4 CAPE/CIN Integration

metrust uses trapezoidal integration on the full sounding profile with
virtual temperature correction.  MetPy uses a similar approach but may
interpolate differently at the LFC and EL boundaries, leading to differences
of a few J/kg for CAPE and CIN.

---

## 6. Not Yet Implemented

The following MetPy modules and functions do not have native metrust
equivalents.  Some are accessible via the MetPy shim; others are not
available.

### 6.1 Modules Without Native Implementation

| MetPy Module | Status in metrust |
|-------------|-------------------|
| `metpy.plots` | Shimmed (lazy-forwards to MetPy) |
| `metpy.xarray` | Shimmed (lazy-forwards to MetPy) |
| `metpy.io.Level2File` | Shimmed (lazy-forwards to MetPy) |
| `metpy.cbook` | Not available |
| `metpy.testing` | Not available |
| `metpy.deprecation` | Not available |

### 6.2 metpy.calc Functions Not in metrust

The following `metpy.calc` functions are not implemented in metrust.  This
list is based on MetPy 1.6.x and may change as both projects evolve.

| MetPy Function | Category | Notes |
|----------------|----------|-------|
| `cross_section_analysis` | Cross-sections | MetPy uses xarray-based cross-section interpolation |
| `interpolate_to_isosurface` | Interpolation | Use `isentropic_interpolation` for theta surfaces |
| `interpolate_1d` | Interpolation | Use `resample_nn_1d` for nearest-neighbor, or numpy/scipy for linear |
| `log_interpolate_1d` | Interpolation | Log-pressure interpolation |
| `interpolate_nans_1d` | Interpolation | NaN gap filling |
| `thickness_hydrostatic` (profile form) | Thermo | metrust has scalar and RH-based profile forms |
| `smooth_510` | Smoothing | Use `smooth_n_point(data, 5)` followed by `smooth_n_point(data, 9)` |
| `zoom_xarray` | xarray | xarray-specific utility |
| `natural_neighbor_to_grid` | Interpolation | Spatial interpolation (see `metrust.interpolate` for alternatives) |
| `inverse_distance_to_grid` | Interpolation | Spatial interpolation |
| `remove_nan_observations` | Utilities | Data cleaning |
| `remove_repeat_coordinates` | Utilities | Data cleaning |
| `get_wind_dir` | Wind | Use `wind_direction` |
| `get_wind_speed` | Wind | Use `wind_speed` |
| `mean_pressure_weighted` (xarray overload) | Thermo | xarray-aware variant |
| `equivalent_potential_temperature` (profile) | Thermo | metrust has scalar + array, not a full-profile variant |

### 6.3 I/O Formats

metrust natively supports these I/O formats in Rust (no MetPy needed):

| Format | Class/Function |
|--------|---------------|
| NEXRAD Level 3 | `Level3File` |
| METAR | `Metar`, `parse_metar`, `parse_metar_file` |
| GINI satellite | `GiniFile` |
| GEMPAK grids | `GempakGrid`, `GempakGridRecord` |
| GEMPAK soundings | `GempakSounding`, `GempakSoundingStation`, `SoundingData` |
| GEMPAK surface | `GempakSurface`, `GempakSurfaceStation`, `SurfaceObs` |
| WPC surface bulletins | `SurfaceBulletinFeature`, `parse_wpc_surface_bulletin` |
| Station metadata | `StationInfo`, `StationLookup` |
| Radar precip mode detection | `is_precip_mode` |

NEXRAD Level 2 (`Level2File`) requires MetPy via the shim.
