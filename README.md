# metrust

**Drop-in replacement for MetPy's calculation layer, powered by Rust.**

150/150 `metpy.calc` functions implemented natively, plus 36 extras. 6-30x faster on real-world workflows. Verified against MetPy on SounderPy, MetPy Cookbook examples, and synthetic global grids.

```python
# The only change: swap the import
from metrust.calc import cape_cin, potential_temperature, vorticity
from metrust.units import units
```

## Installation

```bash
pip install metrust
```

For plotting, xarray accessor, or Level2File (forwarded to MetPy):

```bash
pip install metrust metpy
```

## What It Does

metrust implements every function in `metpy.calc` with a Rust backend compiled via PyO3. The Python API matches MetPy's signatures, units, and return types:

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

Array operations on 1M elements (32-core Ryzen, rayon parallel):

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

Core `metrust.calc` is 100% native Rust with no MetPy dependency.

## Examples

See `examples/` for complete drop-in demos:

- `cookbook_sounding.py` — MetPy Cookbook sounding analysis
- `cookbook_500hpa_grid.py` — MetPy Cookbook 500 hPa vorticity advection
- `sounderpy_dropin.py` — SounderPy-style full sounding pipeline

## Testing

```bash
cargo test --workspace          # 1,186 Rust tests
python -m pytest tests/ -q      # 20 Python tests (including MetPy drop-in regression)
```

## Documentation

Full docs at [fahrenheitresearch.github.io/metrust-py](https://fahrenheitresearch.github.io/metrust-py/), including:

- API reference for all 186 functions
- Beginner tutorials (Weather 101, soundings, grids, recipes)
- Migration guide from MetPy
- Performance benchmarks

## License

MIT
