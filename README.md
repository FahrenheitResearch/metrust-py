# metrust

Rust-powered meteorology toolkit with MetPy-compatible Python APIs.

`metrust` uses a Rust backend for core meteorological calculations and exposes a Python surface that aims to feel familiar to MetPy users. The core `metrust.calc` module covers the large majority of `metpy.calc` with native Rust — no MetPy dependency required for typical calc workflows. A handful of MetPy-specific surfaces (`plots`, `xarray` accessor, `Level2File`) optionally forward to MetPy when it is installed.

```python
# Before
from metpy.calc import cape_cin, potential_temperature
from metpy.units import units

# After
from metrust.calc import cape_cin, potential_temperature
from metrust.units import units

p = [1000, 925, 850, 700, 500] * units.hPa
T = [25, 20, 15, 5, -15] * units.degC
Td = [20, 15, 10, -5, -25] * units.degC
```

## Installation

Core install:

```bash
pip install metrust
```

For optional features (plots, xarray accessor, Level2File), install MetPy separately:

```bash
pip install metpy
```

From source:

```bash
git clone https://github.com/FahrenheitResearch/metrust-py
cd metrust-py
python -m pip install -e .
```

## What Works Well Today

Native Rust implementations cover a large portion of the day-to-day meteorology surface:

- Thermodynamics: potential temperature, equivalent potential temperature, CAPE/CIN, parcel profiles, LCL/LFC/EL, virtual temperature, wet bulb, precipitable water, thickness, stability indices
- Wind and severe weather: wind components, bulk shear, storm-relative helicity, Bunkers storm motion, Corfidi vectors, STP, SCP, critical angle
- Kinematics: divergence, vorticity, advection, frontogenesis, geostrophic and ageostrophic wind, potential vorticity, deformation
- Smoothing and interpolation: Gaussian, rectangular, circular, n-point, generic window convolution, 1-D interpolation, log interpolation, NaN fill, isosurface, IDW, natural neighbor
- I/O: Level-III, METAR parsing, station lookup, GINI, GEMPAK grid/sounding/surface, WPC surface bulletin parsing
- Constants: the core meteorological constants used by the calculation layer

On the Python side, `metrust` now also normalizes several wrapper mismatches that previously blocked MetPy-style use:

- Offset temperatures now work with Pint quantities like `20 * units.degC`
- `saturation_vapor_pressure()` returns `Pa`
- `saturation_mixing_ratio()` returns dimensionless `kg/kg`
- `relative_humidity_from_dewpoint()` returns a dimensionless fraction
- Common MetPy signatures such as `cape_cin(p, t, td, parcel_profile=...)` are accepted

## Compatibility Model

`metrust` is best thought of as:

- A fast Rust-backed replacement for much of `metpy.calc`
- A MetPy-compatible Python package for common workflows
- A partial shim over MetPy for surfaces that are still Python-specific

That last point is important. Some compatibility paths intentionally delegate to MetPy when it is installed. This keeps the public API usable while the native Rust/PyO3 surface catches up.

Current shimmed surfaces:

- `metrust.io.Level2File` forwards to MetPy when available
- `metrust.plots` forwards to `metpy.plots`
- `metrust.xarray` forwards to `metpy.xarray`
- Core `metrust.calc` functions are 100% native Rust with no MetPy fallback

## Performance

The benchmark suite (`benches/`) measures performance at three tiers to give an honest picture:

| Tier | What it measures | Import path |
|------|-----------------|-------------|
| **T1: Raw Rust** | Pure FFI, no Pint overhead | `metrust._metrust.calc` |
| **T2: metrust + Pint** | Rust backend with Pint unit wrappers | `metrust.calc` |
| **T3: MetPy + Pint** | Pure Python + Pint baseline | `metpy.calc` |

The **fair comparison** is T3 vs T2 (both use Pint). T1 shows the raw Rust ceiling.

Representative numbers (p50, AMD Ryzen 9):

| Operation | MetPy (T3) | metrust+Pint (T2) | Raw Rust (T1) | Fair speedup (T3/T2) |
|---|---|---|---|---|
| `potential_temperature` (scalar) | 129 us | 7.4 us | 60 ns | **17x** |
| `equivalent_potential_temperature` | 300 us | 7.5 us | 95 ns | **40x** |
| `wet_bulb_temperature` (scalar) | 724 us | 8.1 us | 201 ns | **90x** |
| `dewpoint_from_rh` (scalar) | 120 us | 2.7 us | 76 ns | **44x** |
| `parcel_profile` (100 levels) | 2.55 ms | 71 us | 56 us | **36x** |
| `cape_cin` (100-level sounding) | 1.60 ms | 137 us | 20 us | **12x** |
| `divergence` (100x100 grid) | 994 us | 12.6 us | 12.6 us | **79x** |
| `storm_relative_helicity` (100 levels) | 579 us | 16.5 us | 532 ns | **35x** |

A few operations are **not** faster — this is expected and the benchmarks don't hide it:

- `wind_speed` at small arrays (1k elements): metrust+Pint is ~2x slower than MetPy because Pint wrapper overhead dominates a sub-microsecond Rust call. At 10k+ elements they converge.
- `smooth_gaussian`: MetPy delegates to scipy's heavily-optimized C `gaussian_filter`. The pure Rust implementation is ~10x slower on large grids. The raw Rust API (`T1`) avoids any Pint cost but the algorithm itself is less optimized.

Run the benchmarks yourself:

```bash
# Rust (Criterion, HTML reports in target/criterion/)
cargo bench --package metrust

# Python three-tier (requires metrust; MetPy optional for T3)
python benches/bench_python.py              # T1+T2 only
python benches/bench_python.py --tier 1,2,3 # all three tiers
python benches/bench_python.py --json       # machine-readable output
```

## Known Limits

This is not a full package-level replacement for all of MetPy yet.

- `metrust.plots` and `metrust.xarray` are compatibility shims, not native reimplementations
- Level-II access currently relies on MetPy from the Python surface
- Numerical agreement is close for most shared calculations, but not bit-identical
- `moist_lapse` still needs more scrutiny before being treated as high-confidence parity work

Known numerical differences include:

- Saturation vapor pressure and saturation mixing ratio use different empirical fits than MetPy on the native Rust path
- LCL and wet-bulb calculations use different approximations than MetPy
- STP and SCP cutoff behavior is not identical in every weak-shear environment
- Some constants come from slightly different textbook/reference values

## Verification

The repo currently verifies two different things:

1. The Rust workspace itself via `cargo test --workspace`
2. Python compatibility expectations via `tests/test_python_compat.py`

The Python compatibility tests cover the specific wrapper issues that most affect "drop-in" use:

- Offset temperature handling
- Wrapper return units for key thermodynamic functions
- MetPy-style function signatures such as `cape_cin(..., parcel_profile=...)`
- Public I/O exports
- `plots` and `xarray` shim forwarding

The older `tests/verify_*.py` scripts are still useful for exploratory comparisons, but they are reference scripts, not the authoritative CI gate.

## Running Checks

```bash
cargo test --workspace
python -m pytest tests/test_python_compat.py -q
```

Optional exploratory comparisons:

```bash
python tests/verify_thermo.py
python tests/verify_wind.py
python tests/verify_kinematics.py
python tests/verify_severe_atmo.py
python tests/verify_smooth_interp.py
```

## Should You Use It

Use `metrust` if:

- You want MetPy-style calculations with much lower runtime overhead
- Your workload is dominated by `metpy.calc`-type operations
- You are comfortable with a project that is converging toward broader MetPy compatibility

Keep MetPy in the loop if:

- You depend heavily on the full plotting stack
- You rely on the xarray accessor layer
- You need exact behavioral parity rather than close compatibility

## License

MIT
