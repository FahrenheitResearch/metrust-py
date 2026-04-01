# metrust

## Meteorological Computation at the Speed of Rust

**metrust** is a Rust-powered, MetPy-compatible calculation layer for [MetPy](https://unidata.github.io/MetPy/)'s calculation workflows.
For many `metpy.calc` workflows, changing one import line is enough. Many thermo and grid workloads run substantially faster.

---

## Install

```bash
pip install metrust
```

That's it. No compiler toolchain, no Rust installation -- prebuilt wheels for Linux, macOS, and Windows.

---

## Migrate in One Line

=== "Before (MetPy)"

    ```python
    from metpy.calc import dewpoint_from_relative_humidity, cape_cin
    from metpy.units import units

    p  = [1000, 925, 850, 700, 500] * units.hPa
    T  = [25, 20, 15, 5, -15] * units.degC
    Td = [20, 15, 10, -5, -25] * units.degC

    dp = dewpoint_from_relative_humidity(T, 0.80)
    ```

=== "After (metrust)"

    ```python
    from metrust.calc import dewpoint_from_relative_humidity, cape_cin
    from metrust.units import units

    p  = [1000, 925, 850, 700, 500] * units.hPa
    T  = [25, 20, 15, 5, -15] * units.degC
    Td = [20, 15, 10, -5, -25] * units.degC

    dp = dewpoint_from_relative_humidity(T, 0.80)  # same API, Rust backend
    ```

The surface is identical. Under the hood, every calculation runs through compiled Rust via [PyO3](https://pyo3.rs/).

---

## Key Features

### 150/150 MetPy Calc Functions (Plus 36 Extras)

Every function in `metpy.calc` has a metrust equivalent -- 100% API coverage. Plus 36 additional functions MetPy doesn't have (grid composites, fire weather indices, and more). The calculation layer is Rust-backed by default, with a small parity-sensitive subset optionally delegating to MetPy when it is installed.
Coverage spans the core of operational meteorology:

- **Thermodynamics** -- potential temperature, equivalent potential temperature, virtual temperature, wet-bulb temperature, LCL, LFC, EL, CAPE/CIN, parcel profiles, precipitable water, thickness hydrostatic, stability indices
- **Wind and Severe** -- wind components, bulk shear, storm-relative helicity, Bunkers storm motion, Corfidi vectors, significant tornado parameter, supercell composite, critical angle
- **Kinematics** -- divergence, vorticity, advection, frontogenesis, geostrophic/ageostrophic wind, potential vorticity, shearing/stretching deformation
- **Smoothing and Interpolation** -- Gaussian, rectangular, circular, n-point, window convolution, log-interpolation, IDW, natural neighbor, isosurface extraction
- **I/O** -- Level III (NEXRAD products), METAR parsing, station lookup, GINI, GEMPAK grid/sounding/surface, WPC surface bulletins
- **Constants** -- the full set of meteorological constants used by the calculation layer

### Scalar, 1-D, and 2-D Grid Support

Every function accepts a single value, a 1-D sounding array, or a full 2-D model grid.
Shape is always preserved -- no reshaping boilerplate required.

```python
import numpy as np
from metrust.calc import potential_temperature
from metrust.units import units

# Scalar
theta = potential_temperature(850 * units.hPa, 20 * units.degC)

# 1-D sounding
p_snd = np.array([1000, 925, 850, 700, 500]) * units.hPa
T_snd = np.array([25, 20, 15, 5, -15]) * units.degC
theta_snd = potential_temperature(p_snd, T_snd)  # shape: (5,)

# 2-D grid (e.g., HRRR 1059x1799)
p_grid = np.full((1059, 1799), 850.0) * units.hPa
T_grid = np.random.uniform(15, 30, (1059, 1799)) * units.degC
theta_grid = potential_temperature(p_grid, T_grid)  # shape: (1059, 1799)
```

28 hot-path functions have dedicated Rust array bindings with zero Python-loop overhead.
Remaining functions use an automatic vectorizer for seamless scalar/array dispatch.

### Full Pint Unit Compatibility

metrust works with the same Pint unit registry that MetPy users already know.
Offset units like `degC` and `degF` are handled correctly throughout.

```python
from metrust.units import units

T = 20 * units.degC
p = 1013.25 * units.hPa
```

### 6--30x Faster on Real Workflows

Validated by running actual MetPy examples and SounderPy with a direct import swap:

| Workflow | Speedup | Source |
|---|---|---|
| MetPy Cookbook sounding analysis | **6.0x** | Full severe weather stack |
| MetPy Cookbook 500 hPa grid | **6.1x** | Vorticity + smoothing + advection |
| MetPy Cookbook Q-vectors | **6.1x** | Q-vector divergence |
| SounderPy compute subset | **29.7x** | Thermo + wind + severe params |
| MetPy isentropic example | **2.3x** | Isentropic interp + Montgomery |

Array operations on 1M elements (32-core Ryzen, rayon parallel):

| Function | Time | Throughput |
|---|---|---|
| `potential_temperature` | 1.8 ms | 550 M/s |
| `wet_bulb_temperature` | 7.3 ms | 137 M/s |
| `wind_speed` | 1.5 ms | 660 M/s |

### Near-Exact Numerical Parity

Verified on MetPy's OUN 2011-05-22 12Z test sounding and NARR isentropic example:

| Metric | Difference from MetPy |
|---|---|
| CAPE | +4.0 J/kg |
| MUCAPE | +7.6 J/kg |
| SRH (0-1 km) | +0.3 m^2/s^2 |
| Critical angle | +0.2 deg |
| Bunkers RM | +0.02 m/s |
| STP | +0.01 |
| Montgomery streamfunction | corr 1.0000 |
| Vorticity (global lat/lon) | corr 1.0000 |

Uses MetPy-exact physical constants, MetPy's CAPE integration formula, pressure-weighted Bunkers algorithm, Newton solver for isentropic interpolation, and spherical metric tensor corrections for lat/lon grids.

---

## What's Different

### Rust Backend via PyO3

All `metrust.calc` functions compile to native machine code through the Rust toolchain and are exposed to Python via [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/). There is no C extension, no Cython, no Numba -- just Rust.

### No MetPy Dependency for Calculations

`pip install metrust` pulls in **only** NumPy and Pint. The calculation layer stays self-contained by default.
A small parity-sensitive subset of `metrust.calc` can delegate to MetPy when MetPy is installed, but the default path remains the Rust backend.

### Optional MetPy for Plots, xarray, and Level 2

A handful of surfaces intentionally delegate to MetPy when it is installed:

| Surface | Behavior |
|---|---|
| `metrust.calc` | Native Rust by default, with limited optional MetPy delegation on a few parity-sensitive paths. |
| `metrust.io.Level2File` | Forwards to MetPy's Level 2 reader when available. |
| `metrust.plots` | Forwards to `metpy.plots`. |
| `metrust.xarray` | Forwards to `metpy.xarray`. |

To use these optional surfaces, install MetPy alongside metrust:

```bash
pip install metrust metpy
```

### Built for Real Workloads

metrust is built and tested against production-scale grids and operational sounding data.
The architecture is designed for the kind of batch processing that meteorological workflows demand -- parallel grid composites, thousands of soundings per cycle, real-time ingest pipelines.

---

## Quick Links

<div class="grid cards" markdown>

-   **Installation**

    ---

    Platform support, from-source builds, and optional dependencies.

    [&rarr;Installation Guide](guides/installation.md)

-   **Migration from MetPy**

    ---

    Detailed guide for moving existing MetPy code to metrust.

    [&rarr;Migration Guide](guides/migration.md)

-   **Array Support**

    ---

    How metrust handles scalars, 1-D arrays, and 2-D grids uniformly.

    [&rarr;Array Guide](guides/arrays.md)

-   **Performance**

    ---

    Three-tier benchmark methodology and full results.

    [&rarr;Benchmarks](performance.md)

-   **API Reference**

    ---

    Full function reference organized by domain.

    [&rarr;Thermodynamics](api/thermodynamics.md) | [&rarr;Wind](api/wind.md) | [&rarr;Kinematics](api/kinematics.md) | [&rarr;Severe](api/severe.md)

-   **Compatibility**

    ---

    What works, what's shimmed, and known numerical differences.

    [&rarr;Compatibility Notes](compatibility.md)

</div>

---

<div class="grid" markdown>

```python title="Five lines to CAPE"
from metrust.calc import cape_cin, parcel_profile
from metrust.units import units

p  = [1000, 925, 850, 700, 500, 300] * units.hPa
T  = [25, 20, 15, 5, -15, -40] * units.degC
Td = [20, 15, 10, -5, -25, -50] * units.degC

prof = parcel_profile(p, T[0], Td[0])
cape, cin = cape_cin(p, T, Td, prof)
```

</div>

---

*metrust v0.3.6 -- MIT License -- [GitHub](https://github.com/FahrenheitResearch/metrust-py)*
