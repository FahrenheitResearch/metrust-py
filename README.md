# metrust

**MetPy-compatible calculation layer, powered by Rust.**

150/150 `metpy.calc` functions implemented natively, plus 36 extras. Often faster on real-world workflows. Verified against MetPy on SounderPy, MetPy Cookbook examples, and synthetic global grids.

```python
# The only change: swap the import
from metrust.calc import cape_cin, potential_temperature, vorticity
from metrust.units import units
```

## Installation

```bash
pip install metrust
```

Optional GPU acceleration for supported calculations:

```bash
pip install "metrust[gpu]"
```

For plotting, xarray accessor, or Level2File (forwarded to MetPy):

```bash
pip install metrust metpy
```

## What It Does

metrust implements every function in `metpy.calc` with a Rust backend compiled via PyO3. The Python API now matches MetPy's public `metpy.calc` signatures and is designed for MetPy-compatible units, return types, and runtime behavior on the shared calculation surface:

```python
import numpy as np
from metrust.calc import (
    cape_cin, parcel_profile, bunkers_storm_motion,
    storm_relative_helicity, significant_tornado_parameter,
    vorticity, divergence, advection,
)
from metrust.units import units

# Sounding analysis (same API as MetPy)
p = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
T = np.array([25, 20, 15, 5, -15, -40]) * units.degC
Td = np.array([20, 15, 10, -5, -25, -50]) * units.degC

prof = parcel_profile(p, T[0], Td[0])
cape, cin = cape_cin(p, T, Td, prof)  # MetPy parcel_profile form works

# Grid kinematics with xarray (dx/dy inferred from lat/lon coords)
vort = vorticity(u_xarray, v_xarray)  # spherical metric corrections included
div = divergence(u_xarray, v_xarray)
```

## Optional GPU Backend

Current users do not need to change anything. `metrust` stays on the Rust CPU backend by default.

When `met-cu` is installed, you can opt in explicitly:

```python
import metrust.calc as mcalc

mcalc.set_backend("gpu")
theta = mcalc.potential_temperature(pressure, temperature)

with mcalc.use_backend("cpu"):
    theta_cpu = mcalc.potential_temperature(pressure, temperature)
```

The GPU backend currently targets the overlap where `met-cu` is already strong and verified:

- `potential_temperature`
- `equivalent_potential_temperature`
- `dewpoint`
- `vorticity`
- `frontogenesis`
- `q_vector`
- `compute_cape_cin`
- `compute_srh`
- `compute_shear`
- `compute_pw`
- `composite_reflectivity_from_hydrometeors`

Eligible dispatch currently focuses on scalar thermo plus uniform 2-D Cartesian grid workloads. Latitude/longitude-derived spacing, map-scale corrections, and other projection-aware cases fall back to the Rust CPU backend.

`metrust` still returns the same Pint/NumPy-facing API surface. Unsupported cases automatically stay on the Rust CPU path.

## Speed

Benchmarked on real-world workflows (v0.3.3, validated by Codex against MetPy):

| Workflow | Speedup | Notes |
|---|---|---|
| MetPy Cookbook sounding analysis | **6.0x** | Full severe weather stack |
| MetPy Cookbook 500 hPa grid | **6.1x** | Vorticity, smoothing, advection |
| MetPy Cookbook Q-vectors | **6.1x** | Q-vector divergence |
| SounderPy compute-heavy subset | **29.7x** | Thermo + wind + severe params |
| MetPy isentropic example | **2.3x** | Isentropic interpolation + Montgomery |
| Vorticity/divergence (global grid) | **2.3x** | Spherical corrections on 721x1440 |

Current replay-harness snapshot from `benches/bench_workflows.py` on Windows 11 / Python `3.13.7`:
sounding `2.16x`, grid diagnostics `8.53x`, xarray workflow `1.46x`.

### Three-Way Benchmark: MetPy vs Rust vs CUDA

Real HRRR model output (40 isobaric levels, 1059 × 1799 grid, ~1.9 M points). RTX 5090, `python tests/benchmark_gpu.py`.

**Scalar Thermodynamics (2D: 1059×1799)**

| Function | MetPy | Rust | CUDA | Rust/MetPy | CUDA/Rust |
|---|---|---|---|---|---|
| potential_temperature ★ | 11.2 ms | 13.0 ms | 7.9 ms | 0.9x | 1.6x |
| equiv_potential_temperature ★ | 303.1 ms | 16.5 ms | 9.1 ms | 18x | 1.8x |
| dewpoint ★ | 33.6 ms | 10.8 ms | 8.8 ms | 3.1x | 1.2x |
| saturation_vapor_pressure | 66.7 ms | 8.2 ms | — | 8.1x | — |
| saturation_mixing_ratio | 78.6 ms | 13.7 ms | — | 5.7x | — |
| dewpoint_from_rh | 110.3 ms | 7.6 ms | — | 14x | — |
| rh_from_dewpoint | 138.7 ms | 10.9 ms | — | 13x | — |
| virtual_temperature | 31.1 ms | 19.7 ms | — | 1.6x | — |
| mixing_ratio | 11.6 ms | 8.5 ms | — | 1.4x | — |
| wet_bulb_temperature | >10 min | 26.9 ms | — | — | — |

**Grid Kinematics (2D: 1059×1799)**

| Function | MetPy | Rust | CUDA | Rust/MetPy | CUDA/Rust |
|---|---|---|---|---|---|
| vorticity ★ | 98.3 ms | 92.8 ms | 9.3 ms | 1.1x | **10x** |
| divergence | 96.1 ms | 91.2 ms | — | 1.1x | — |
| frontogenesis ★ | 733.0 ms | 339.4 ms | 12.2 ms | 2.2x | **28x** |
| q_vector ★ | 390.3 ms | 310.1 ms | 10.8 ms | 1.3x | **29x** |
| advection | 161.8 ms | 87.7 ms | — | 1.8x | — |

**1D Sounding (40 levels, single column)**

| Function | MetPy | Rust | Rust/MetPy |
|---|---|---|---|
| parcel_profile | 5.5 ms | 0.074 ms | **74x** |
| cape_cin | 1.4 ms | 0.254 ms | 5.4x |
| lcl | 0.118 ms | 0.065 ms | 1.8x |
| lfc | 6.7 ms | 0.107 ms | **62x** |
| el | 6.6 ms | 0.112 ms | **59x** |
| precipitable_water | 2.1 ms | 0.057 ms | **36x** |

**Grid Composites (3D: 40×1059×1799 → 2D) — MetPy has no grid equivalents**

| Function | Rust | CUDA | CUDA/Rust |
|---|---|---|---|
| compute_cape_cin ★ | 2.96 s | 674.5 ms | **4.4x** |
| compute_srh ★ | 223.5 ms | 135.8 ms | 1.6x |
| compute_shear ★ | 190.4 ms | 166.5 ms | 1.1x |
| compute_pw ★ | 191.6 ms | 107.9 ms | 1.8x |
| composite_refl_hydrometeors ★ | 154.2 ms | 232.2 ms | 0.7x |

**Summary**

| | MetPy | Rust | CUDA |
|---|---|---|---|
| Scalar thermo (×10) | 785 ms | 136 ms | 26 ms |
| Grid kinematics (×5) | 1.48 s | 921 ms | 32 ms |
| 1D sounding (×6) | 22 ms | 0.67 ms | — |
| Grid composites (×5) | — | 3.72 s | 1.32 s |
| **★ GPU-eligible total** | — | **4.50 s** | **1.37 s (3.3x)** |

★ = dispatches to CUDA when `set_backend("gpu")`. All other functions stay on Rust CPU regardless of backend setting.

### Array Throughput

1M elements, 32-core Ryzen, rayon parallel:

| Function | Time | Throughput |
|---|---|---|
| `potential_temperature` | 1.8 ms | 550 M/s |
| `wet_bulb_temperature` | 7.3 ms | 137 M/s |
| `wind_speed` | 1.5 ms | 660 M/s |

## Numerical Parity

Verified on the MetPy OUN 2011-05-22 12Z test sounding:

| Metric | Difference from MetPy |
|---|---|
| CAPE | +4.0 J/kg |
| MUCAPE | +7.6 J/kg |
| SRH (0-1 km) | +0.3 m^2/s^2 |
| Critical angle | +0.2 deg |
| Bunkers RM | +0.02 m/s |
| STP | +0.01 |
| Montgomery streamfunction | corr 1.0000 |
| Isentropic pressure | 7e-13 hPa diff |
| Vorticity (global lat/lon) | corr 1.0000 |

Uses MetPy-exact physical constants (Rd, Cp, Lv, epsilon), MetPy's CAPE integration formula (`g * dTv/Tv * dz`), pressure-weighted Bunkers algorithm, Newton solver for isentropic interpolation, and spherical metric tensor corrections for lat/lon grid kinematics.

## Coverage

- **150/150** `metpy.calc` functions (100% coverage)
- **36 extras** not in MetPy (grid composites, fire weather indices, etc.)
- **28 Rust array bindings** with rayon parallelism and GIL release
- **Pint application registry** shared with MetPy (no cross-registry errors)
- **xarray support** with coordinate inference and shape preservation

## What's Not Native

These forward to MetPy when installed:

- `metrust.plots` (matplotlib-based plotting)
- `metrust.xarray` (xarray accessor)
- `metrust.io.Level2File` (NEXRAD Level II)

Core `metrust.calc` stays native Rust by default with no required MetPy dependency. The shared calc surface now stays on native metrust implementations even when MetPy is installed. The optional `met-cu` backend is an explicit accelerator, not a requirement.

End-to-end replay benchmarks for sounding, grid, and xarray workflows live in `benches/bench_workflows.py` and the published docs page `workflow-benchmarks.md`.

## Examples

See `examples/` for complete drop-in demos:

- `cookbook_sounding.py` — MetPy Cookbook sounding analysis
- `cookbook_500hpa_grid.py` — MetPy Cookbook 500 hPa vorticity advection
- `sounderpy_dropin.py` — SounderPy-style full sounding pipeline

## Testing

```bash
cargo test --workspace          # 1,186 Rust tests
python -m pytest tests/ -q      # 30 Python tests (including MetPy compatibility regression)
```

## Documentation

Full docs at [fahrenheitresearch.github.io/metrust-py](https://fahrenheitresearch.github.io/metrust-py/), including:

- API reference for all 186 functions
- Beginner tutorials (Weather 101, soundings, grids, recipes)
- Migration guide from MetPy
- Performance benchmarks

## License

MIT
