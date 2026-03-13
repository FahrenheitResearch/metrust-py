# metrust

## Meteorological Computation at the Speed of Rust

**metrust** is a Rust-powered, drop-in replacement for [MetPy](https://unidata.github.io/MetPy/)'s calculation layer.
Change one import line. Keep your existing code. Get 10--90x faster results.

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

Every function in `metpy.calc` has a metrust equivalent -- 100% API coverage. Plus 36 additional functions MetPy doesn't have (grid composites, fire weather indices, and more). The entire `metrust.calc` module is backed by Rust -- no MetPy dependency, no Python fallback.
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

### 10--90x Faster Than MetPy

Real benchmarks, real hardware (AMD Ryzen 9), fair comparison (both using Pint wrappers):

| Function | MetPy | metrust | Speedup |
|---|---|---|---|
| `potential_temperature` (scalar) | 129 us | 7.4 us | **17x** |
| `equivalent_potential_temperature` | 300 us | 7.5 us | **40x** |
| `wet_bulb_temperature` (scalar) | 724 us | 8.1 us | **90x** |
| `dewpoint_from_rh` (scalar) | 120 us | 2.7 us | **44x** |
| `parcel_profile` (100 levels) | 2.55 ms | 71 us | **36x** |
| `cape_cin` (100-level sounding) | 1.60 ms | 137 us | **12x** |
| `divergence` (100x100 grid) | 994 us | 12.6 us | **79x** |
| `storm_relative_helicity` (100 levels) | 579 us | 16.5 us | **35x** |

These numbers compare **T3 (MetPy + Pint)** against **T2 (metrust + Pint)** -- the apples-to-apples comparison where both libraries pay Pint wrapper overhead. The raw Rust layer (T1) is faster still.

!!! note "Honest benchmarks"
    A small number of operations are _not_ faster. `wind_speed` on small arrays is dominated by Pint overhead on both sides, and `smooth_gaussian` cannot yet match SciPy's battle-tested C implementation. The benchmark suite does not hide these cases.

---

## What's Different

### Rust Backend via PyO3

All `metrust.calc` functions compile to native machine code through the Rust toolchain and are exposed to Python via [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/). There is no C extension, no Cython, no Numba -- just Rust.

### No MetPy Dependency for Calculations

`pip install metrust` pulls in **only** NumPy and Pint. The entire calculation layer is self-contained.
MetPy is never imported, loaded, or called for any `metrust.calc` function.

### Optional MetPy for Plots, xarray, and Level 2

A handful of surfaces intentionally delegate to MetPy when it is installed:

| Surface | Behavior |
|---|---|
| `metrust.calc` | 100% native Rust. No MetPy fallback. |
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

*metrust v0.3.2 -- MIT License -- [GitHub](https://github.com/FahrenheitResearch/metrust-py)*
