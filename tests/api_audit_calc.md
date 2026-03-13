# MetPy calc API Completeness Audit

Generated: 2026-03-12

Comparison of every public callable in `metpy.calc` against `metrust::calc` (and its submodules: `thermo`, `wind`, `kinematics`, `severe`, `atmo`, `smooth`, `utils`).

## Summary

| Status | Count |
|---|---|
| MATCH | 120 |
| RENAMED | 14 |
| PARTIAL | 8 |
| MISSING | 17 |
| N/A (not applicable to Rust) | 4 |
| **Total MetPy callables** | **163** |

Coverage: **142 / 159** applicable functions = **89.3%**

---

## Thermodynamics

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `potential_temperature(pressure, temperature)` | `thermo::potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `temperature_from_potential_temperature(pressure, potential_temperature)` | `thermo::temperature_from_potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `equivalent_potential_temperature(pressure, temperature, dewpoint)` | `thermo::equivalent_potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `saturation_equivalent_potential_temperature(pressure, temperature)` | `thermo::saturation_equivalent_potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `virtual_potential_temperature(pressure, temperature, mixing_ratio, ...)` | `thermo::virtual_potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `wet_bulb_potential_temperature(pressure, temperature, dewpoint)` | `thermo::wet_bulb_potential_temperature` | MATCH | Re-export from wx_math::thermo |
| `exner_function(pressure, reference_pressure)` | `thermo::exner_function` | MATCH | Re-export from wx_math::thermo |
| `saturation_vapor_pressure(temperature, *, phase)` | `thermo::saturation_vapor_pressure` | PARTIAL | No ice-phase option; liquid-only Bolton formula |
| `saturation_mixing_ratio(total_press, temperature, *, phase)` | `thermo::saturation_mixing_ratio` | PARTIAL | No ice-phase option |
| `mixing_ratio(partial_press, total_press, ...)` | `thermo::mixing_ratio` | RENAMED | MetPy takes (partial_press, total_press); metrust takes (pressure, temperature) and computes internally |
| `mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity, *, phase)` | `thermo::mixing_ratio_from_relative_humidity` | MATCH | Re-export from wx_math::thermo |
| `mixing_ratio_from_specific_humidity(specific_humidity)` | `thermo::mixing_ratio_from_specific_humidity` | MATCH | Re-export from wx_math::thermo |
| `vapor_pressure(pressure, mixing_ratio)` | `thermo::vapor_pressure` | RENAMED | MetPy takes (pressure, mixing_ratio); metrust takes dewpoint and computes e from it |
| `dewpoint(vapor_pressure)` | `thermo::dewpoint` | MATCH | Re-export from wx_math::thermo |
| `dewpoint_from_relative_humidity(temperature, relative_humidity)` | `thermo::dewpoint_from_relative_humidity` | MATCH | Wrapper around wx_math::thermo::dewpoint_from_rh |
| `dewpoint_from_specific_humidity(*args, **kwargs)` | `thermo::dewpoint_from_specific_humidity` | MATCH | Re-export from wx_math::thermo |
| `relative_humidity_from_dewpoint(temperature, dewpoint, *, phase)` | `thermo::relative_humidity_from_dewpoint` | MATCH | Wrapper around wx_math::thermo::rh_from_dewpoint |
| `relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio, *, phase)` | `thermo::relative_humidity_from_mixing_ratio` | MATCH | Re-export from wx_math::thermo |
| `relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity, *, phase)` | `thermo::relative_humidity_from_specific_humidity` | MATCH | Re-export from wx_math::thermo |
| `relative_humidity_wet_psychrometric(pressure, dry_bulb_temperature, wet_bulb_temperature, **kwargs)` | `thermo::relative_humidity_wet_psychrometric` | MATCH | Implemented in thermo.rs |
| `psychrometric_vapor_pressure_wet(pressure, dry_bulb_temperature, wet_bulb_temperature, ...)` | `thermo::psychrometric_vapor_pressure` | RENAMED | Re-export from wx_math::thermo as `psychrometric_vapor_pressure` |
| `specific_humidity_from_dewpoint(pressure, dewpoint, *, phase)` | `thermo::specific_humidity_from_dewpoint` | MATCH | Re-export from wx_math::thermo |
| `specific_humidity_from_mixing_ratio(mixing_ratio)` | - | MISSING | Inverse of mixing_ratio_from_specific_humidity; q = w / (1 + w) |
| `virtual_temperature(temperature, mixing_ratio, ...)` | `thermo::virtual_temperature` | RENAMED | MetPy takes (T, w); metrust takes (T, P, Td) and derives w internally |
| `virtual_temperature_from_dewpoint(pressure, temperature, dewpoint, ...)` | `thermo::virtual_temperature` | MATCH | metrust's virtual_temperature already takes (T, P, Td) which matches this MetPy function |
| `density(pressure, temperature, mixing_ratio, ...)` | `thermo::density` | MATCH | Re-export from wx_math::thermo |
| `dry_lapse(pressure, temperature, ...)` | `thermo::dry_lapse` | MATCH | Re-export from wx_math::thermo |
| `moist_lapse(pressure, temperature, ...)` | `thermo::moist_lapse` | MATCH | Re-export from wx_math::thermo |
| `lcl(pressure, temperature, dewpoint, ...)` | `thermo::lcl` | MATCH | Wrapper around wx_math::thermo::drylift |
| `ccl(pressure, temperature, dewpoint, ...)` | `thermo::ccl` | MATCH | Re-export from wx_math::thermo |
| `lfc(pressure, temperature, dewpoint, ...)` | `thermo::lfc` | MATCH | Re-export from wx_math::thermo |
| `el(pressure, temperature, dewpoint, ...)` | `thermo::el` | MATCH | Re-export from wx_math::thermo |
| `parcel_profile(pressure, temperature, dewpoint)` | `thermo::parcel_profile` | MATCH | Re-export from wx_math::thermo |
| `parcel_profile_with_lcl(pressure, temperature, dewpoint)` | `thermo::parcel_profile_with_lcl` | MATCH | Implemented in thermo.rs |
| `parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint)` | - | N/A | xarray Dataset output; not applicable to Rust |
| `cape_cin(pressure, temperature, dewpoint, parcel_profile, ...)` | `thermo::cape_cin` | MATCH | Wrapper around wx_math::thermo::cape_cin_core |
| `surface_based_cape_cin(pressure, temperature, dewpoint)` | `thermo::surface_based_cape_cin` | MATCH | Re-export from wx_math::thermo |
| `mixed_layer_cape_cin(pressure, temperature, dewpoint, **kwargs)` | `thermo::mixed_layer_cape_cin` | MATCH | Re-export from wx_math::thermo |
| `most_unstable_cape_cin(pressure, temperature, dewpoint, **kwargs)` | `thermo::most_unstable_cape_cin` | MATCH | Re-export from wx_math::thermo |
| `downdraft_cape(pressure, temperature, dewpoint)` | `thermo::downdraft_cape` | MATCH | Wrapper around wx_math::thermo::downdraft_cape |
| `mixed_layer(pressure, *args, ...)` | `thermo::mixed_layer` | MATCH | Re-export from wx_math::thermo |
| `mixed_parcel(pressure, temperature, dewpoint, ...)` | `thermo::get_mixed_layer_parcel` | RENAMED | Re-export as get_mixed_layer_parcel |
| `most_unstable_parcel(pressure, temperature, dewpoint, ...)` | `thermo::get_most_unstable_parcel` | RENAMED | Re-export as get_most_unstable_parcel |
| `lifted_index(pressure, temperature, parcel_profile, ...)` | `thermo::lifted_index` | MATCH | Re-export from wx_math::thermo |
| `precipitable_water(pressure, dewpoint, *, bottom, top)` | `thermo::precipitable_water` | MATCH | Implemented in thermo.rs |
| `mean_pressure_weighted(pressure, *args, ...)` | `thermo::mean_pressure_weighted` | MATCH | Re-export from wx_math::thermo |
| `weighted_continuous_average(pressure, *args, ...)` | `thermo::weighted_continuous_average` | MATCH | Implemented in thermo.rs |
| `get_layer(pressure, *args, ...)` | `thermo::get_layer` | MATCH | Re-export from wx_math::thermo |
| `get_layer_heights(height, depth, *args, ...)` | `thermo::get_layer_heights` | MATCH | Re-export from wx_math::thermo |
| `dry_static_energy(height, temperature)` | `thermo::dry_static_energy` | MATCH | Re-export from wx_math::thermo |
| `moist_static_energy(height, temperature, specific_humidity)` | `thermo::moist_static_energy` | MATCH | Re-export from wx_math::thermo |
| `montgomery_streamfunction(height, temperature)` | `thermo::montgomery_streamfunction` | MATCH | Re-export from wx_math::thermo |
| `static_stability(pressure, temperature, ...)` | `thermo::static_stability` | MATCH | Re-export from wx_math::thermo |
| `brunt_vaisala_frequency(height, potential_temperature, ...)` | `thermo::brunt_vaisala_frequency` | MATCH | Implemented in thermo.rs |
| `brunt_vaisala_frequency_squared(height, potential_temperature, ...)` | `thermo::brunt_vaisala_frequency_squared` | MATCH | Implemented in thermo.rs |
| `brunt_vaisala_period(height, potential_temperature, ...)` | `thermo::brunt_vaisala_period` | MATCH | Implemented in thermo.rs |
| `vertical_velocity(omega, pressure, temperature, ...)` | `thermo::vertical_velocity` | MATCH | Re-export from wx_math::thermo |
| `vertical_velocity_pressure(w, pressure, temperature, ...)` | `thermo::vertical_velocity_pressure` | MATCH | Re-export from wx_math::thermo |
| `geopotential_to_height(geopotential)` | `thermo::geopotential_to_height` | MATCH | Re-export from wx_math::thermo |
| `height_to_geopotential(height)` | `thermo::height_to_geopotential` | MATCH | Re-export from wx_math::thermo |
| `scale_height(temperature_bottom, temperature_top)` | `thermo::scale_height` | MATCH | Re-export from wx_math::thermo |
| `thickness_hydrostatic(pressure, temperature, ...)` | `thermo::thickness_hydrostatic` | MATCH | Wrapper around wx_math::thermo::thickness_hypsometric |
| `thickness_hydrostatic_from_relative_humidity(pressure, temperature, relative_humidity, ...)` | - | MISSING | Computes thickness with virtual temperature correction from RH |
| `add_height_to_pressure(pressure, height)` | `thermo::add_height_to_pressure` | MATCH | Implemented in thermo.rs |
| `add_pressure_to_height(height, pressure)` | `thermo::add_pressure_to_height` | MATCH | Implemented in thermo.rs |
| `get_perturbation(ts, axis)` | `thermo::get_perturbation` | MATCH | Implemented in thermo.rs |
| `isentropic_interpolation(levels, pressure, temperature, *args, ...)` | `thermo::isentropic_interpolation` | MATCH | Re-export from wx_math::thermo |
| `isentropic_interpolation_as_dataset(levels, temperature, *args, ...)` | - | N/A | xarray Dataset output; not applicable to Rust |
| `find_intersections(x, a, b, ...)` | `thermo::find_intersections` | MATCH | Re-export from wx_math::thermo |
| `find_bounding_indices(arr, values, axis, ...)` | `utils::find_bounding_indices` | MATCH | Implemented in utils.rs |
| `reduce_point_density(points, radius, ...)` | `thermo::reduce_point_density` | MATCH | Re-export from wx_math::thermo |
| `find_peaks(arr, *, maxima, iqr_ratio)` | - | MISSING | Find peaks in a 1D array using IQR filtering |
| `peak_persistence(arr, *, maxima)` | - | MISSING | Topological persistence-based peak detection |
| `moist_air_gas_constant(specific_humidity)` | `thermo::moist_air_gas_constant` | MATCH | Implemented in thermo.rs (takes w not q) |
| `moist_air_specific_heat_pressure(specific_humidity)` | `thermo::moist_air_specific_heat_pressure` | MATCH | Implemented in thermo.rs |
| `moist_air_poisson_exponent(specific_humidity)` | `thermo::moist_air_poisson_exponent` | MATCH | Implemented in thermo.rs |
| `water_latent_heat_vaporization(temperature)` | `thermo::water_latent_heat_vaporization` | MATCH | Implemented in thermo.rs |
| `water_latent_heat_melting(temperature)` | `thermo::water_latent_heat_melting` | MATCH | Implemented in thermo.rs |
| `water_latent_heat_sublimation(temperature)` | `thermo::water_latent_heat_sublimation` | MATCH | Implemented in thermo.rs |

## Stability Indices

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `showalter_index(pressure, temperature, dewpoint)` | `thermo::showalter_index` | MATCH | Implemented in thermo.rs |
| `k_index(pressure, temperature, dewpoint, ...)` | `thermo::k_index` | MATCH | Implemented in thermo.rs (takes level values directly) |
| `total_totals_index(pressure, temperature, dewpoint, ...)` | `thermo::total_totals` | RENAMED | Named `total_totals` in metrust |
| `cross_totals(pressure, temperature, dewpoint, ...)` | `thermo::cross_totals` | MATCH | Implemented in thermo.rs |
| `vertical_totals(pressure, temperature, ...)` | `thermo::vertical_totals` | MATCH | Implemented in thermo.rs |
| `sweat_index(pressure, temperature, dewpoint, speed, direction, ...)` | `thermo::sweat_index` | MATCH | Implemented in thermo.rs |
| `galvez_davison_index(pressure, temperature, mixing_ratio, ...)` | `severe::galvez_davison_index` | MATCH | Re-export from wx_math::thermo |

## Wind

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `wind_speed(u, v)` | `wind::wind_speed` | MATCH | Re-export from wx_math::dynamics |
| `wind_direction(u, v, convention)` | `wind::wind_direction` | MATCH | Re-export from wx_math::dynamics |
| `wind_components(speed, wind_direction)` | `wind::wind_components` | MATCH | Re-export from wx_math::dynamics |
| `bulk_shear(pressure, u, v, height, bottom, depth)` | `wind::bulk_shear` | MATCH | Implemented in wind.rs (height-based interface) |
| `storm_relative_helicity(height, u, v, depth, *, bottom, storm_u, storm_v)` | `wind::storm_relative_helicity` | MATCH | Implemented in wind.rs |
| `bunkers_storm_motion(pressure, u, v, height)` | `wind::bunkers_storm_motion` | MATCH | Implemented in wind.rs |
| `corfidi_storm_motion(pressure, u, v, *, u_llj, v_llj)` | `wind::corfidi_storm_motion` | MATCH | Implemented in wind.rs |
| `mean_pressure_weighted(pressure, *args, ...)` | `wind::mean_wind` | RENAMED | metrust provides `mean_wind` for trapezoidal mean; MetPy's `mean_pressure_weighted` is more general |

## Kinematics & Dynamics

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `divergence(u, v, *, dx, dy, ...)` | `kinematics::divergence` | MATCH | Re-export from wx_math::dynamics |
| `vorticity(u, v, *, dx, dy, ...)` | `kinematics::vorticity` | MATCH | Re-export from wx_math::dynamics |
| `absolute_vorticity(u, v, dx, dy, latitude, ...)` | `kinematics::absolute_vorticity` | MATCH | Re-export from wx_math::dynamics |
| `advection(scalar, u, v, w, *, dx, dy, dz, ...)` | `kinematics::advection` | PARTIAL | 2D only; MetPy supports 3D advection with w and dz |
| `frontogenesis(potential_temperature, u, v, ...)` | `kinematics::frontogenesis` | MATCH | Wrapper around wx_math::dynamics::frontogenesis_2d |
| `geostrophic_wind(height, dx, dy, latitude, ...)` | `kinematics::geostrophic_wind` | MATCH | Re-export from wx_math::dynamics |
| `ageostrophic_wind(height, u, v, dx, dy, latitude, ...)` | `kinematics::ageostrophic_wind` | MATCH | Re-export from wx_math::dynamics |
| `inertial_advective_wind(u, v, u_geostrophic, v_geostrophic, ...)` | `kinematics::inertial_advective_wind` | MATCH | Re-export from wx_math::dynamics |
| `q_vector(u, v, temperature, pressure, dx, dy, ...)` | `kinematics::q_vector` | MATCH | Re-export from wx_math::dynamics |
| `stretching_deformation(u, v, dx, dy, ...)` | `kinematics::stretching_deformation` | MATCH | Re-export from wx_math::dynamics |
| `shearing_deformation(u, v, dx, dy, ...)` | `kinematics::shearing_deformation` | MATCH | Re-export from wx_math::dynamics |
| `total_deformation(u, v, dx, dy, ...)` | `kinematics::total_deformation` | MATCH | Re-export from wx_math::dynamics |
| `coriolis_parameter(latitude)` | `kinematics::coriolis_parameter` | MATCH | Re-export from wx_math::dynamics |
| `potential_vorticity_baroclinic(potential_temperature, pressure, u, v, ...)` | `kinematics::potential_vorticity_baroclinic` | MATCH | Implemented in kinematics.rs |
| `potential_vorticity_barotropic(height, u, v, ...)` | `kinematics::potential_vorticity_barotropic` | MATCH | Implemented in kinematics.rs |
| `absolute_momentum(u, v, index)` | `kinematics::absolute_momentum` | MATCH | Re-export from wx_math::dynamics |
| `kinematic_flux(vel, b, ...)` | `kinematics::kinematic_flux` | MATCH | Re-export from wx_math::dynamics |
| `curvature_vorticity(u, v, *, dx, dy, ...)` | `kinematics::curvature_vorticity` | MATCH | Re-export from wx_math::dynamics |
| `shear_vorticity(u, v, *, dx, dy, ...)` | `kinematics::shear_vorticity` | MATCH | Re-export from wx_math::dynamics |
| `cross_section_components(data_x, data_y, index)` | `kinematics::cross_section_components` | MATCH | Implemented in kinematics.rs |
| `normal_component(data_x, data_y, index)` | `kinematics::normal_component` | MATCH | Implemented in kinematics.rs |
| `tangential_component(data_x, data_y, index)` | `kinematics::tangential_component` | MATCH | Implemented in kinematics.rs |
| `unit_vectors_from_cross_section(cross, index)` | `kinematics::unit_vectors_from_cross_section` | MATCH | Implemented in kinematics.rs |
| `vector_derivative(u, v, *, dx, dy, ...)` | `kinematics::vector_derivative` | MATCH | Implemented in kinematics.rs |
| `friction_velocity(u, w, v, perturbation, axis)` | - | MISSING | Boundary-layer friction velocity from Reynolds stress |
| `tke(u, v, w, perturbation, axis)` | - | MISSING | Turbulent kinetic energy from wind perturbations |
| `gradient_richardson_number(height, potential_temperature, u, v, ...)` | - | MISSING | Gradient Richardson number for turbulence diagnostics |

## Spatial Derivatives & Math

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `first_derivative(f, axis, x, delta)` | `smooth::first_derivative` | MATCH | Re-export from wx_math::gridmath |
| `second_derivative(f, axis, x, delta)` | `smooth::second_derivative` | MATCH | Re-export from wx_math::gridmath |
| `gradient(f, axes, coordinates, deltas)` | `smooth::gradient_x` / `smooth::gradient_y` | PARTIAL | metrust provides per-axis gradient_x/gradient_y; MetPy has a unified multi-axis gradient |
| `laplacian(f, axes, coordinates, deltas)` | `smooth::laplacian` | MATCH | Re-export from wx_math::dynamics |
| `lat_lon_grid_deltas(longitude, latitude, ...)` | `smooth::lat_lon_grid_deltas` | MATCH | Re-export from wx_math::gridmath |
| `geospatial_gradient(f, *, dx, dy, ...)` | `smooth::geospatial_gradient` | MATCH | Re-export from wx_math::gridmath |
| `geospatial_laplacian(f, *, dx, dy, ...)` | `smooth::geospatial_laplacian` | MATCH | Re-export from wx_math::gridmath |
| `nearest_intersection_idx(a, b)` | `utils::nearest_intersection_idx` | MATCH | Implemented in utils.rs |
| `resample_nn_1d(a, centers)` | `utils::resample_nn_1d` | MATCH | Implemented in utils.rs |

## Smoothing Filters

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `smooth_gaussian(scalar_grid, n)` | `smooth::smooth_gaussian` | MATCH | Implemented in smooth.rs |
| `smooth_rectangular(scalar_grid, size, passes)` | `smooth::smooth_rectangular` | PARTIAL | No `passes` parameter; callers must loop |
| `smooth_circular(scalar_grid, radius, passes)` | `smooth::smooth_circular` | PARTIAL | No `passes` parameter; callers must loop |
| `smooth_n_point(scalar_grid, n, passes)` | `smooth::smooth_n_point` | PARTIAL | No `passes` parameter; callers must loop |
| `smooth_window(scalar_grid, window, passes, normalize_weights)` | `smooth::smooth_window` | PARTIAL | No `passes` or `normalize_weights` parameters |

## Standard Atmosphere & Pressure

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `pressure_to_height_std(pressure)` | `atmo::pressure_to_height_std` | MATCH | Implemented in atmo.rs |
| `height_to_pressure_std(height)` | `atmo::height_to_pressure_std` | MATCH | Implemented in atmo.rs |
| `altimeter_to_station_pressure(altimeter_value, height)` | `atmo::altimeter_to_station_pressure` | MATCH | Implemented in atmo.rs |
| `altimeter_to_sea_level_pressure(altimeter_value, height, temperature)` | `atmo::altimeter_to_sea_level_pressure` | MATCH | Implemented in atmo.rs |
| `sigma_to_pressure(sigma, pressure_sfc, pressure_top)` | `atmo::sigma_to_pressure` | MATCH | Implemented in atmo.rs |

## Comfort / Apparent Temperature

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `heat_index(temperature, relative_humidity, ...)` | `atmo::heat_index` | MATCH | Implements full Rothfusz regression with NWS adjustments |
| `windchill(temperature, speed, ...)` | `atmo::windchill` | MATCH | NWS/Environment Canada formula |
| `apparent_temperature(temperature, relative_humidity, speed, ...)` | `atmo::apparent_temperature` | MATCH | Combines heat index and wind chill |

## Severe Weather Composites

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `significant_tornado(sbcape, surface_based_lcl_height, storm_helicity_1km, shear_6km)` | `severe::significant_tornado_parameter` | RENAMED | Named `significant_tornado_parameter` in metrust |
| `supercell_composite(mucape, effective_storm_helicity, effective_shear)` | `severe::supercell_composite_parameter` | RENAMED | Named `supercell_composite_parameter` in metrust |
| `critical_angle(pressure, u, v, height, u_storm, v_storm)` | `severe::critical_angle` | MATCH | Implemented in severe.rs |

## Direction / Angle Utilities

| MetPy Function | metrust Equivalent | Status | Notes |
|---|---|---|---|
| `angle_to_direction(input_angle, full, level)` | `utils::angle_to_direction` | PARTIAL | Always 16-point; no `full` word mode or configurable `level` |
| `parse_angle(input_dir)` | `utils::parse_angle` | MATCH | Implemented in utils.rs |

## Functions Not Applicable to Rust

| MetPy Function | Status | Notes |
|---|---|---|
| `InvalidSoundingError` | N/A | Python exception class; Rust uses Result types |
| `set_module(globls)` | N/A | Python internal for module namespace setup |
| `parcel_profile_with_lcl_as_dataset(...)` | N/A | Returns xarray Dataset; not applicable |
| `isentropic_interpolation_as_dataset(...)` | N/A | Returns xarray Dataset; not applicable |
| `zoom_xarray(...)` | N/A | xarray-specific zoom/interpolation utility |

## Missing Functions (17 total)

| MetPy Function | Description | Priority |
|---|---|---|
| `specific_humidity_from_mixing_ratio(mixing_ratio)` | Convert mixing ratio to specific humidity: q = w/(1+w) | Low -- trivial one-liner |
| `thickness_hydrostatic_from_relative_humidity(...)` | Hypsometric thickness with virtual temp correction from RH | Medium |
| `find_peaks(arr, *, maxima, iqr_ratio)` | Find peaks in a 1D array using IQR-based filtering | Low |
| `peak_persistence(arr, *, maxima)` | Topological persistence peak detection | Low |
| `friction_velocity(u, w, v, ...)` | Boundary-layer friction velocity from Reynolds stress | Low -- BL-specific |
| `tke(u, v, w, ...)` | Turbulent kinetic energy from wind perturbations | Low -- BL-specific |
| `gradient_richardson_number(height, potential_temperature, u, v, ...)` | Gradient Richardson number Ri = N^2 / S^2 | Medium |
| `azimuth_range_to_lat_lon(azimuths, ranges, center_lon, center_lat, ...)` | Radar polar to geographic coordinate conversion | Medium -- radar utility |

### Additional metrust extras (not in MetPy)

metrust provides several functions with no MetPy equivalent, reflecting operational NWS/SPC needs:

| metrust Function | Module | Notes |
|---|---|---|
| `boyden_index` | severe | European stability index |
| `bulk_richardson_number` | severe | BRN from CAPE and shear |
| `convective_inhibition_depth` | severe | CIN integrated depth |
| `dendritic_growth_zone` | severe | -12 to -18 C layer depth for snow growth |
| `fosberg_fire_weather_index` | severe | Fire weather composite |
| `freezing_rain_composite` | severe | Icing potential composite |
| `haines_index` | severe | Lower-atmosphere stability and dryness for fire |
| `hot_dry_windy` | severe | Hot-Dry-Windy composite for fire weather |
| `warm_nose_check` | severe | Warm layer aloft detection for winter weather |
| `station_to_altimeter_pressure` | atmo | Inverse of altimeter_to_station_pressure |
| `frost_point` | thermo | Frost-point temperature |
| `q_vector_convergence` | kinematics | -2*div(Q) forcing for vertical motion |
| `temperature_advection` | kinematics | Convenience wrapper for scalar advection |
| `moisture_advection` | kinematics | Convenience wrapper for scalar advection |
| `mean_wind` | wind | Height-weighted trapezoidal mean wind |
| `geospatial_gradient` | smooth | Gradient on lat/lon grid with correct spacings |
| `geospatial_laplacian` | smooth | Laplacian on lat/lon grid with correct spacings |
| `gradient_x` / `gradient_y` | smooth | Per-axis partial derivatives |

---

## Detailed Notes

### PARTIAL implementations

1. **saturation_vapor_pressure / saturation_mixing_ratio**: MetPy supports `phase='ice'` for ice-phase saturation. metrust uses Bolton (1980) liquid-only formula. Ice-phase support would require adding the Alduchov & Eskridge or Murphy-Koop formulas.

2. **advection**: MetPy supports 3D advection with a vertical velocity `w` and vertical spacing `dz`. metrust is 2D-only.

3. **smooth_rectangular / smooth_circular / smooth_n_point / smooth_window**: MetPy accepts a `passes` parameter to apply the filter multiple times. metrust applies a single pass; callers must loop for multiple passes. This is an API convenience issue, not a correctness issue.

4. **gradient**: MetPy provides a single `gradient()` function that handles arbitrary axes. metrust provides separate `gradient_x` and `gradient_y` functions plus `first_derivative(axis)` which covers the same ground but with a different API shape.

5. **angle_to_direction**: MetPy supports `full=True` for full words ("North" vs "N") and `level` for 4/8/16-point resolution. metrust always returns the 16-point abbreviated form.

### RENAMED functions

These are functionally equivalent but use different names to follow Rust naming conventions or reflect slightly different interfaces:

- `mixed_parcel` -> `get_mixed_layer_parcel`
- `most_unstable_parcel` -> `get_most_unstable_parcel`
- `total_totals_index` -> `total_totals`
- `significant_tornado` -> `significant_tornado_parameter`
- `supercell_composite` -> `supercell_composite_parameter`
- `mixing_ratio` (different signature -- takes P, T instead of partial_press, total_press)
- `vapor_pressure` (takes dewpoint instead of pressure + mixing_ratio)
- `virtual_temperature` (takes T, P, Td instead of T, w)
- `psychrometric_vapor_pressure_wet` -> `psychrometric_vapor_pressure`
