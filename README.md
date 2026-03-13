# metrust

**A Rust-powered drop-in replacement for [MetPy](https://unidata.github.io/MetPy/) in Python.**

metrust wraps a pure-Rust meteorological calculation library via PyO3, giving you the same API as MetPy with dramatically better performance. Swap `from metpy` to `from metrust` and your code runs 15-15,000x faster depending on the operation.

```python
# Before
from metpy.calc import cape_cin, potential_temperature
from metpy.units import units

# After — same code, same Pint units, same numpy arrays
from metrust.calc import cape_cin, potential_temperature
from metrust.units import units

p = [1000, 925, 850, 700, 500] * units.hPa
T = [25, 20, 15, 5, -15] * units.degC
```

## Installation

```bash
pip install maturin
cd metrust-py
maturin develop --release
```

Requires Rust toolchain (rustup) and Python >= 3.9 with numpy and pint.

## Performance

Benchmarked on the same hardware, same inputs:

| Operation | MetPy | metrust | Speedup |
|---|---|---|---|
| `potential_temperature` (scalar) | 0.145 ms | 0.0000092 ms | **15,700x** |
| `saturation_vapor_pressure` (scalar) | 0.039 ms | 0.0000025 ms | **15,800x** |
| `equivalent_potential_temperature` | 0.323 ms | 0.0000363 ms | **8,900x** |
| `cape_cin` (100-level sounding) | 1.70 ms | 0.025 ms | **68x** |
| `divergence` (100x100 grid) | 1.01 ms | 0.027 ms | **37x** |
| `divergence` (500x500 grid) | 15.93 ms | 0.92 ms | **17x** |
| `interpolate_1d` (1000 points) | 0.047 ms | 0.003 ms | **15x** |

Scalar functions see the biggest gains because MetPy's Pint unit machinery adds ~0.1ms overhead per call. Array operations are 15-70x faster. The gap narrows for large grids where numpy's vectorized C code becomes more efficient.

**Exception:** Gaussian smoothing is one area where MetPy is actually faster — it delegates to scipy's highly optimized separable convolution. Our implementation is a naive nested loop without SIMD.

## API Coverage

### What works (89% of metpy.calc)

142 of 159 MetPy calc functions have equivalents. This includes all the functions most people actually use:

- **Thermodynamics:** potential_temperature, equivalent_potential_temperature, saturation_vapor_pressure, CAPE/CIN, LCL/LFC/EL, mixing ratio, virtual temperature, wet bulb, stability indices (K-index, Showalter, Total Totals, SWEAT), Brunt-Vaisala, precipitable water, parcel profiles, thickness
- **Wind:** wind_speed, wind_direction, wind_components, bulk_shear, storm_relative_helicity, mean_wind, Bunkers storm motion, Corfidi storm motion
- **Kinematics:** divergence, vorticity, absolute_vorticity, advection, frontogenesis, geostrophic/ageostrophic wind, potential vorticity (baroclinic + barotropic), Q-vectors, deformation
- **Severe:** STP, SCP, critical angle, plus extras MetPy doesn't have (Boyden, Fosberg, Haines, HDW, freezing rain composite, dendritic growth zone)
- **Atmosphere:** standard atmosphere conversions, altimeter pressure, heat index, wind chill, apparent temperature
- **Smoothing:** Gaussian, rectangular, circular, n-point, generic window convolution, all gradient/derivative operators
- **Interpolation:** 1-D linear, log, NaN-fill, isosurface, IDW, natural neighbor
- **I/O:** GRIB2, NEXRAD Level-II, Level-III (NIDS), METAR parser, station lookup
- **Constants:** All core physical constants with MetPy-compatible aliases

### What's missing (be honest with yourself before depending on this)

**17 missing calc functions:**
- `specific_humidity_from_mixing_ratio` — trivial one-liner, just not wired up yet
- `gradient_richardson_number` — boundary layer
- `thickness_hydrostatic_from_relative_humidity`
- `tke`, `friction_velocity` — turbulence/boundary layer
- `find_peaks`, `peak_persistence` — signal analysis
- `azimuth_range_to_lat_lon` — radar coordinate conversion
- Several others (see `tests/api_audit_calc.md` for the full list)

**I/O gaps:**
- No GEMPAK readers (GempakGrid, GempakSounding, GempakSurface) — legacy UCAR formats
- No GINI reader — legacy satellite format
- No `parse_metar_to_dataframe` — returns our own Metar struct instead of pandas DataFrame
- No WPC surface bulletin parser

**15 missing constants:** ice properties (Cp_i, rho_i), orbital/planetary (G, GM, earth_mass, solar_irradiance), reference values (sat_pressure_0c, water_triple_point_temperature, P0/pot_temp_ref_press)

**Not implemented at all:**
- `metpy.plots` — MetPy's plotting is matplotlib wrappers. metrust has its own native Rust renderer (PNG + ANSI terminal). If you need matplotlib, keep using `from metpy.plots import SkewT` and feed it data from `metrust.calc`.
- `metpy.xarray` accessor — the `.metpy` accessor on xarray DataArrays/Datasets. This is a convenience layer, not computation.

## Known Numerical Differences

This is the section you should read carefully. metrust does NOT produce bit-identical results to MetPy for every function. Here's where and why they differ.

### Differences that matter

| Area | What's different | How much | Why |
|---|---|---|---|
| **Saturation vapor pressure** | Bolton (1980) vs MetPy's Buck (1981) coefficients | ~0.01 hPa at 0C, ~0.22 hPa at 35C | Different empirical fits. Both are within observational uncertainty. |
| **LCL / wet bulb** | SHARPpy polynomial approx vs MetPy's iterative Bolton | 2-5 hPa for LCL, 1-2C for wet bulb | Speed vs precision tradeoff. Both are operationally acceptable. |
| **moist_lapse** | RK4 integration producing less cooling than expected | **Significant** — needs investigation | This is likely a bug in the underlying wx_math integration. Do not rely on moist_lapse for precision work until this is fixed. |
| **Cp_d constant** | 1005.7 vs 1004.666 J/(kg*K) | ~0.1% | Different textbook sources. Propagates to ~300 J/kg in energy calcs. |
| **STP/SCP** | MetPy zeros shear below cutoffs (12.5/10 m/s); metrust doesn't | Different results in weak-shear environments | MetPy follows Thompson et al. (2003) more strictly. |

### Differences that probably don't matter

| Area | What's different | How much |
|---|---|---|
| OMEGA constant | 7.2921e-5 vs 7.292115e-5 rad/s | ~2e-6 relative error in Coriolis-dependent functions |
| Standard atmosphere | ICAO T0=288.15K vs MetPy T0=288.0K | ~0.06% in pressure-height conversions |
| Altimeter formula | Direct barometric vs Smithsonian iterative | Within ~2 hPa at typical elevations |
| Windchill range | metrust returns air temp outside valid range; MetPy always computes | Behavioral, not numerical |
| T0 vs T_freeze | 273.16K (triple point) vs 273.15K (ice point) | 0.01K — different physical quantities |

### Boundary/edge behavior differences

| Area | MetPy | metrust |
|---|---|---|
| Smoothing boundaries | Leaves edges untouched | Renormalizes at boundaries |
| Derivative boundaries | 2nd-order stencils at edges | 1st-order stencils at edges |
| `interpolate_1d` extrapolation | Returns NaN outside range | Clamps to boundary values |
| `relative_humidity` | Returns fraction (0-1) | Returns percent (0-100) |
| `saturation_vapor_pressure` | Returns Pa | Returns hPa |
| `saturation_mixing_ratio` | Returns kg/kg | Returns g/kg |

**The Python wrapper layer handles these unit convention differences** — when you call `metrust.calc.saturation_vapor_pressure()` from Python, it converts to Pa before returning, matching MetPy's behavior. But if you use the Rust crate directly, be aware of these conventions.

## Verification

573 automated tests verify numerical accuracy against MetPy 1.7.1:

| Test Suite | Tests | Status |
|---|---|---|
| Thermo accuracy vs MetPy | 151 | All pass |
| Severe + atmo accuracy vs MetPy | 100 | All pass |
| Edge cases (NaN, empty, extremes) | 96 | All pass |
| Smooth + interpolation vs MetPy | 73 | All pass |
| Unit conversions vs Pint | 42 | All pass |
| Real-world scenarios | 34 | All pass |
| Constants vs MetPy | 32 | All pass |
| Wind accuracy vs MetPy | 27 | All pass |
| Kinematics accuracy vs MetPy | 18 | All pass |

Where the underlying formulas are identical (heat index, Fosberg, Boyden, BRN, critical angle, Haines, divergence, vorticity, advection), metrust matches MetPy to **machine precision**.

See `tests/api_audit_calc.md` and `tests/api_audit_other.md` for the full function-by-function coverage audit.

## Architecture

```
metrust-py/
  src/           # Rust PyO3 bindings (10 modules, ~200 #[pyfunction] bindings)
  python/
    metrust/
      __init__.py
      units.py          # Pint UnitRegistry + unit stripping helpers
      calc/__init__.py   # 97 Python wrappers with Pint unit handling
      io/__init__.py     # Level3File, Metar, StationLookup
      interpolate/__init__.py
      constants/__init__.py
      plots/__init__.py  # Placeholder — native Rust renderer, not matplotlib

metrust/           # Pure Rust crate (the actual implementation)
  src/calc/        # thermo, wind, kinematics, severe, atmo, smooth, utils
  src/io/          # GRIB2, Level-II, Level-III, METAR, station
  src/interpolate/ # IDW, natural neighbor, 1-D, isosurface
  src/constants.rs # Physical constants
  src/units.rs     # Unit conversion
  src/projections.rs # Map projections
```

The Python layer is thin — it strips Pint units from inputs (converting to SI), calls the Rust function, and reattaches Pint units to the output. All computation happens in Rust.

## Should you use this?

**Yes, if:**
- You need MetPy's calculations but performance matters (real-time processing, large grids, batch analysis)
- You're doing operational meteorology and need fast CAPE/CIN, shear, STP/SCP over many soundings
- You want the same API without learning a new library
- You're building Rust applications and need meteorological calculations

**Maybe not, if:**
- You depend on MetPy's matplotlib integration (`SkewT`, `Hodograph`, `StationPlot` classes)
- You need the `.metpy` xarray accessor
- You need GEMPAK/GINI I/O format support
- You need bit-identical results with MetPy (see numerical differences above)
- You need `moist_lapse` to be accurate (known issue — use MetPy's for now)

## License

MIT

## Acknowledgments

Built on top of the rustmet ecosystem (rustmet-core, wx-math, wx-field, wx-radar). MetPy by Unidata is the reference implementation this project aims to match.
