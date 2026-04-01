# Migrating from MetPy to metrust

This guide covers everything you need to switch an existing MetPy codebase to
metrust. The short version: change the import, run your tests, ship it.

---

## The one-line migration

```python
# Before
from metpy.calc import potential_temperature, cape_cin
from metpy.units import units

# After
from metrust.calc import potential_temperature, cape_cin
from metrust.units import units
```

That is the entire change for most code. Every function in `metrust.calc`
accepts the same Pint Quantity inputs and returns Pint Quantity outputs, using
the same function names and the same public positional/keyword signatures as MetPy.

A project-wide find-and-replace from `metpy.` to `metrust.` is usually
sufficient. The table below shows the namespace mapping:

| MetPy import               | metrust import              |
|----------------------------|-----------------------------|
| `metpy.calc`               | `metrust.calc`              |
| `metpy.units`              | `metrust.units`             |
| `metpy.constants`          | `metrust.constants`         |
| `metpy.interpolate`        | `metrust.interpolate`       |
| `metpy.io`                 | `metrust.io`                |
| `metpy.plots`              | `metrust.plots` (shim)      |
| `metpy.xarray`             | `metrust.xarray` (shim)     |

---

## What works identically

All `metrust.calc` functions accept Pint Quantity arguments and return Pint
Quantity results, exactly as MetPy does. The function names, parameter names,
and return types are the same. A non-exhaustive list of compatible functions:

**Thermodynamics** --
`potential_temperature`, `equivalent_potential_temperature`,
`saturation_vapor_pressure`, `saturation_mixing_ratio`,
`wet_bulb_temperature`, `wet_bulb_potential_temperature`,
`virtual_temperature`, `virtual_temperature_from_dewpoint`,
`virtual_potential_temperature`,
`temperature_from_potential_temperature`,
`saturation_equivalent_potential_temperature`,
`dewpoint`, `dewpoint_from_relative_humidity`,
`dewpoint_from_specific_humidity`,
`relative_humidity_from_dewpoint`,
`relative_humidity_from_mixing_ratio`,
`relative_humidity_from_specific_humidity`,
`mixing_ratio`, `mixing_ratio_from_relative_humidity`,
`mixing_ratio_from_specific_humidity`,
`specific_humidity_from_dewpoint`,
`specific_humidity_from_mixing_ratio`,
`vapor_pressure`, `lcl`, `lfc`, `el`, `ccl`,
`cape_cin`, `surface_based_cape_cin`,
`mixed_layer_cape_cin`, `most_unstable_cape_cin`,
`downdraft_cape`,
`parcel_profile`, `parcel_profile_with_lcl`,
`dry_lapse`, `moist_lapse`,
`precipitable_water`, `density`,
`dry_static_energy`, `moist_static_energy`,
`exner_function`, `thickness_hydrostatic`,
`thickness_hydrostatic_from_relative_humidity`,
`brunt_vaisala_frequency`, `brunt_vaisala_frequency_squared`,
`brunt_vaisala_period`, `static_stability`,
`montgomery_streamfunction`,
`isentropic_interpolation`,
`add_height_to_pressure`, `add_pressure_to_height`,
`geopotential_to_height`, `height_to_geopotential`,
`scale_height`, `vertical_velocity`, `vertical_velocity_pressure`,
`frost_point`, `psychrometric_vapor_pressure`,
`relative_humidity_wet_psychrometric`

**Wind and kinematics** --
`wind_speed`, `wind_direction`, `wind_components`,
`bulk_shear`, `mean_wind`,
`storm_relative_helicity`,
`bunkers_storm_motion`, `corfidi_storm_motion`,
`divergence`, `vorticity`, `absolute_vorticity`,
`advection`, `frontogenesis`,
`geostrophic_wind`, `ageostrophic_wind`,
`potential_vorticity_baroclinic`, `potential_vorticity_barotropic`,
`shearing_deformation`, `stretching_deformation`, `total_deformation`,
`curvature_vorticity`, `shear_vorticity`,
`q_vector`, `cross_section_components`,
`normal_component`, `tangential_component`,
`coriolis_parameter`, `absolute_momentum`,
`gradient_richardson_number`, `friction_velocity`, `tke`

**Stability indices** --
`showalter_index`, `lifted_index`, `k_index`,
`total_totals`, `cross_totals`, `vertical_totals`,
`sweat_index`

**Severe-weather composites** --
`significant_tornado_parameter`, `supercell_composite_parameter`,
`critical_angle`, `bulk_richardson_number`,
`boyden_index`, `convective_inhibition_depth`,
`dendritic_growth_zone`, `galvez_davison_index`,
`fosberg_fire_weather_index`, `haines_index`,
`hot_dry_windy`, `warm_nose_check`,
`freezing_rain_composite`

**Atmosphere and comfort** --
`pressure_to_height_std`, `height_to_pressure_std`,
`altimeter_to_station_pressure`, `station_to_altimeter_pressure`,
`altimeter_to_sea_level_pressure`, `sigma_to_pressure`,
`heat_index`, `windchill`

**Smoothing** --
`smooth_gaussian`, `smooth_rectangular`, `smooth_circular`,
`smooth_n_point`, `smooth_window`

**Utilities** --
`get_layer`, `get_layer_heights`,
`mixed_layer`, `mean_pressure_weighted`,
`get_mixed_layer_parcel`, `get_most_unstable_parcel`,
`find_intersections`, `reduce_point_density`,
`get_perturbation`, `weighted_continuous_average`

**Interpolation** (`metrust.interpolate`) --
`interpolate_1d`, `log_interpolate_1d`, `interpolate_nans_1d`,
`interpolate_to_isosurface`, `interpolate_to_slice`,
`interpolate_to_grid`, `interpolate_to_points`,
`inverse_distance_to_grid`, `inverse_distance_to_points`,
`natural_neighbor_to_grid`, `natural_neighbor_to_points`,
`remove_nan_observations`, `remove_observations_below_value`,
`remove_repeat_coordinates`, `geodesic`

**Constants** (`metrust.constants`) --
All MetPy constant names are available, including both long-form
(`earth_gravity`, `dry_air_gas_constant`) and short aliases (`g`, `Rd`,
`epsilon`). Values are plain floats in SI base units.

**I/O** (`metrust.io`) --
`Level3File`, `GiniFile`, `Metar`, `parse_metar`, `parse_metar_file`,
`GempakGrid`, `GempakSounding`, `GempakSurface`,
`parse_wpc_surface_bulletin`, `StationLookup`, `is_precip_mode`

---

## Known differences

metrust is a clean-room Rust implementation, not a wrapper around MetPy. The
results are close but not bit-identical in every case.

### Saturation vapor pressure

metrust uses the Ambaum (2020) empirical formulation for saturation vapor
pressure over both liquid water and ice. MetPy uses the same paper but the two
implementations may differ in intermediate precision or constant rounding.
Typical disagreement is on the order of 0.01 hPa.

### LCL

The LCL solver in metrust uses a different iterative approach than MetPy.
Results agree to within approximately 0.1 hPa and 0.05 K for typical
atmospheric soundings.

### Wet-bulb temperature

metrust solves for wet-bulb temperature via its own Newton iteration.
Differences from MetPy are generally under 0.1 K but can be larger near
saturation.

### STP and SCP cutoff behavior

The Significant Tornado Parameter and Supercell Composite Parameter use
clamping terms for weak-shear and high-LCL environments. The exact cutoff
thresholds and ramp shapes in metrust follow the SPC formulation but may
differ from MetPy's implementation in marginal cases (e.g., 0-6 km shear
below 12.5 m/s, LCL heights near 2000 m). The difference is most visible
when composites are near zero.

### Physical constants

metrust derives physical constants from the Rust `metrust::calc` engine.
Most values match MetPy/CODATA to full float64 precision, but a few
(e.g., water vapor gas constant) come from slightly different reference
editions. Differences are negligible for operational use.

### Summary

If your workflow compares output to MetPy at machine-epsilon precision, expect
small discrepancies. If your workflow uses standard meteorological tolerances,
the two libraries are interchangeable.

---

## What is shimmed

Three modules forward to MetPy at runtime when MetPy is installed. They
provide compatibility without requiring you to maintain two separate imports.

### `metrust.plots`

All plotting classes (`SkewT`, `Hodograph`, `StationPlot`, etc.) are lazy-
loaded from `metpy.plots`. If MetPy is not installed, importing any
attribute raises an `ImportError` with installation instructions.

### `metrust.xarray`

The xarray coordinate/CRS accessor (`ds.metpy.parse_cf()`, etc.) is
lazy-loaded from `metpy.xarray`.

### `metrust.io.Level2File`

NEXRAD Level 2 radar file reading is not yet implemented natively in Rust.
Accessing `metrust.io.Level2File` transparently forwards to
`metpy.io.Level2File`.

These shims mean you can do a blanket `metpy` -> `metrust` rename and
everything will continue to work, provided MetPy is installed alongside
metrust for the features that need it.

---

## What is new in metrust

metrust is not just a drop-in replacement. It exposes capabilities that MetPy
does not have, all backed by parallel Rust.

### Grid composite kernels

These functions accept flattened 3-D model grids (e.g., from HRRR, RAP, NAM)
and return 2-D fields. They run in parallel across all grid columns using
Rust threads -- no Python loop required.

**3-D profile kernels** (accept `[nz][ny][nx]` flattened arrays):

| Function                | Returns                          |
|-------------------------|----------------------------------|
| `compute_cape_cin`      | CAPE, CIN, LCL pressure, LFC pressure |
| `compute_srh`           | Storm-relative helicity          |
| `compute_shear`         | Bulk wind shear magnitude        |
| `compute_lapse_rate`    | Environmental lapse rate (C/km)  |
| `compute_pw`            | Precipitable water (mm)          |
| `composite_reflectivity_from_refl` | Column-max reflectivity (dBZ) |
| `composite_reflectivity_from_hydrometeors` | Reflectivity from mixing ratios |

**2-D composite kernels** (accept pre-computed 2-D fields):

| Function                        | Description                      |
|---------------------------------|----------------------------------|
| `compute_stp`                   | Significant Tornado Parameter    |
| `compute_scp`                   | Supercell Composite Parameter    |
| `compute_ehi`                   | Energy-Helicity Index            |
| `significant_hail_parameter`    | SHIP                             |
| `derecho_composite_parameter`   | DCP                              |
| `grid_supercell_composite_parameter` | Enhanced SCP with CIN term  |
| `grid_critical_angle`           | Critical angle field              |

Example -- compute SB CAPE for an entire HRRR grid:

```python
from metrust._metrust import calc

cape, cin, lcl_p, lfc_p = calc.compute_cape_cin(
    pressure_3d,        # flattened [nz][ny][nx], hPa
    temperature_c_3d,   # flattened [nz][ny][nx], Celsius
    qvapor_3d,          # flattened [nz][ny][nx], kg/kg
    height_agl_3d,      # flattened [nz][ny][nx], meters
    psfc,               # flattened [ny][nx], hPa
    t2,                 # flattened [ny][nx], Celsius
    q2,                 # flattened [ny][nx], kg/kg
    nx, ny, nz,
    "surface",          # parcel_type: "surface", "mixed_layer", "most_unstable"
)
```

On a 1059x1799 HRRR grid this completes in seconds, not the minutes it
would take to loop through columns in Python.

### SB3CAPE via `top_m`

Both the point-based `cape_cin` and the grid-based `compute_cape_cin` accept
an optional `top_m` parameter that caps the CAPE integration at a height AGL
in meters. This lets you compute surface-based 0-3 km CAPE (SB3CAPE) directly:

```python
# Point sounding
cape3, cin3, _, _ = metrust.calc.cape_cin(
    p, t, td, height, psfc, t2m, td2m,
    parcel_type="sb", top_m=3000.0,
)

# Grid
cape3, cin3, _, _ = calc.compute_cape_cin(
    ..., parcel_type="surface", top_m=3000.0,
)
```

MetPy does not expose this parameter.

### 28 dedicated Rust array bindings

For the most performance-critical thermodynamic functions, metrust provides
array-native Rust bindings that process entire numpy arrays in a single FFI
call, avoiding per-element Python overhead. These are used automatically when
you pass array Quantities to functions like `potential_temperature`,
`saturation_vapor_pressure`, `wet_bulb_temperature`, `dewpoint`,
`density`, `exner_function`, `virtual_potential_temperature`, and others.

You do not need to change your calling code to benefit from this -- the
Python wrappers detect array inputs and dispatch to the array binding
automatically.

### Native I/O

`metrust.io` provides Rust-native readers for Level 3 NEXRAD, GINI, GEMPAK
(grids, soundings, surface obs), METAR, and WPC surface bulletins. These do
not require MetPy to be installed.

---

## Gradual migration

You can use metrust and MetPy side by side in the same project. This is
useful for incremental migration or for workflows that need MetPy-exclusive
features (plotting, xarray accessors) alongside metrust's compute
performance.

### Pint registry conflict

The one thing to watch for is Pint unit registries. metrust and MetPy each
create their own `pint.UnitRegistry` instance:

```python
from metrust.units import units as mr_units   # metrust's registry
from metpy.units import units as mp_units     # MetPy's registry
```

Pint does not allow arithmetic between Quantities from different registries.
This will raise a `ValueError`:

```python
t_metrust = 25.0 * mr_units.degC
t_metpy   = 30.0 * mp_units.degC
delta = t_metpy - t_metrust   # ValueError: cannot operate with ...
```

The fix is straightforward: pick one registry and use it consistently within
each computation. If you need to pass a metrust result to a MetPy function
(or vice versa), strip the magnitude and re-attach units from the other
registry:

```python
# metrust result -> MetPy input
theta_mr = metrust.calc.potential_temperature(p, t)
theta_mp = theta_mr.magnitude * mp_units.K    # re-wrap for MetPy

# MetPy result -> metrust input
td_mp = metpy.calc.dewpoint_from_relative_humidity(t_mp, rh_mp)
td_mr = td_mp.magnitude * mr_units.degC       # re-wrap for metrust
```

In practice, most codebases use a single import path for `units` and this is
not an issue.

### Recommended migration order

1. **Start with `calc`.**  Replace `from metpy.calc import ...` with
   `from metrust.calc import ...`. Run your existing test suite. The API is
   the same; the only possible failures are numerical precision differences
   (see "Known differences" above).

2. **Switch `units` and `constants`.** Replace `from metpy.units import units`
   and `from metpy.constants import ...`. These are straightforward
   substitutions.

3. **Switch `interpolate`.** The function signatures match MetPy's.

4. **Switch `io`.** Native readers for Level 3, GINI, GEMPAK, and METAR work
   without MetPy. Keep MetPy installed if you need `Level2File`.

5. **Keep MetPy for plotting and xarray.** The `metrust.plots` and
   `metrust.xarray` shims handle this transparently. You can uninstall MetPy
   entirely only when you no longer need these features.

---

## Quick reference

```python
import metrust.calc as mpcalc
from metrust.units import units

# Point sounding -- works exactly like MetPy
p   = [1000, 925, 850, 700, 500] * units.hPa
t   = [25, 20, 15, 5, -15] * units.degC
td  = [20, 15, 10, -5, -30] * units.degC

theta   = mpcalc.potential_temperature(p, t)
theta_e = mpcalc.equivalent_potential_temperature(p, t, td)
t_wb    = mpcalc.wet_bulb_temperature(p, t, td)

# Grid processing -- metrust-exclusive
from metrust._metrust import calc as rcalc

cape, cin, lcl, lfc = rcalc.compute_cape_cin(
    p3d, t3d, qv3d, hagl3d, psfc, t2, q2,
    nx, ny, nz, "surface",
)
stp = rcalc.compute_stp(cape, lcl, srh_1km, shear_6km)
```
