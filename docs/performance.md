# Performance

## Why Honest Benchmarking Matters

Performance claims without methodology are meaningless. A Rust FFI call that skips unit handling will always look faster than a Python function that includes it -- but that comparison is not useful to anyone deciding whether to adopt a library. The user's code will have Pint quantities on both sides of the call regardless of which library they choose.

metrust uses a **three-tier benchmark model** to separate the Rust computation cost from the Python/Pint overhead, and to give a fair apples-to-apples comparison against MetPy.

---

## Three-Tier Benchmark Model

Every function in the benchmark suite is measured at three levels:

| Tier | What it measures | Import path | Units |
|------|-----------------|-------------|-------|
| **T1: Raw Rust** | Pure FFI -- Rust function called directly from Python with bare floats/arrays | `metrust._metrust.calc` | None (raw `float` / `ndarray`) |
| **T2: metrust + Pint** | Rust backend with the Pint unit wrapper layer that user code actually calls | `metrust.calc` | Pint quantities in and out |
| **T3: MetPy + Pint** | Pure-Python MetPy with the same Pint unit wrapper layer | `metpy.calc` | Pint quantities in and out |

**The fair comparison is T3 vs T2.** Both tiers pay the same Pint overhead for unit construction, conversion, and validation. The only difference is the computation engine: Python (MetPy) vs Rust (metrust).

**T1 shows the Rust ceiling.** It reveals how fast the computation itself is before Pint gets involved. The gap between T1 and T2 tells you how much time the Python wrapper and Pint unit handling add on top of the raw Rust call.

This distinction matters because:

- At scalar sizes, the T2/T1 ratio is often 100x or more -- almost all the wall-clock time is Pint, not Rust.
- At large array sizes, T2 converges toward T1 because the Rust computation dominates and Pint overhead is amortized.
- Reporting only T1 vs T3 would overstate the benefit users actually see in their code.

---

## Representative Numbers

All measurements are p50 (median) latency, collected on an AMD Ryzen 9 using the three-tier benchmark harness (`benches/bench_python.py`). Each trial targets 0.2 seconds of wall time with 7 independent repeats and 5 warmup calls.

### Core Thermodynamics and Sounding Analysis

| Operation | MetPy (T3) | metrust+Pint (T2) | Raw Rust (T1) | Fair speedup (T3/T2) |
|---|---|---|---|---|
| `potential_temperature` (scalar) | 129 us | 7.4 us | 60 ns | **17x** |
| `equivalent_potential_temperature` (scalar) | 300 us | 7.5 us | 95 ns | **40x** |
| `wet_bulb_temperature` (scalar) | 724 us | 8.1 us | 201 ns | **90x** |
| `dewpoint_from_rh` (scalar) | 120 us | 2.7 us | 76 ns | **44x** |
| `parcel_profile` (100 levels) | 2.55 ms | 71 us | 56 us | **36x** |
| `cape_cin` (100-level sounding) | 1.60 ms | 137 us | 20 us | **12x** |
| `storm_relative_helicity` (100 levels) | 579 us | 16.5 us | 532 ns | **35x** |

### Grid Kinematics

| Operation | MetPy (T3) | metrust+Pint (T2) | Raw Rust (T1) | Fair speedup (T3/T2) |
|---|---|---|---|---|
| `divergence` (100x100 grid) | 994 us | 12.6 us | 12.6 us | **79x** |

For grid kinematics the T2 and T1 numbers are nearly identical. The Rust array binding accepts raw NumPy arrays and performs the full 2-D finite-difference computation in compiled code, so the Pint wrapper cost is negligible relative to the grid traversal.

### Workflow Replay Benchmarks

These are end-to-end import-swap replays, not single-function timings. The
same workflow shapes are also enforced in `tests/test_cookbook_replays.py`.

Local snapshot from `python benches/bench_workflows.py` on Windows 11 with
Python `3.13.7`:

| Workflow | metrust p50 | MetPy p50 | Speedup |
|---|---:|---:|---:|
| Cookbook sounding replay | 10.38 ms | 30.79 ms | **2.97x** |
| Cookbook grid diagnostics replay | 2.59 ms | 22.73 ms | **8.79x** |
| Cookbook xarray replay | 2.51 ms | 3.66 ms | **1.46x** |

The dedicated harness lives in `benches/bench_workflows.py`, with a published
summary on the [Workflow Benchmarks](workflow-benchmarks.md) page. CI also
uploads `workflow_bench_results.json` as a build artifact so benchmark snapshots
are preserved per run.

---

## Where metrust is NOT Faster

The benchmark suite does not hide regressions. Two categories consistently show MetPy matching or beating metrust:

### `wind_speed` at Small Arrays

At 1,000 elements, `metrust.calc.wind_speed` with Pint is roughly 2x slower than MetPy. The reason: `wind_speed` is `sqrt(u^2 + v^2)` -- a single vectorized NumPy operation that MetPy executes in one C call. metrust's Pint wrapper layer adds overhead (unit extraction, conversion, re-wrapping) that exceeds the time saved by the Rust `hypot` loop at this scale.

At 10,000+ elements the two converge, and at grid scale (1M+ points) metrust pulls ahead because its Rust array binding avoids Python-level dispatch entirely.

**Takeaway:** For trivially cheap operations on small arrays, Pint overhead dominates regardless of backend. The speedup appears when the computation itself is non-trivial or the data is large enough to amortize wrapper costs.

### `smooth_gaussian`

MetPy delegates Gaussian smoothing to `scipy.ndimage.gaussian_filter`, which uses an efficient recursive (IIR) approximation. metrust uses a direct FIR convolution with rayon row-level parallelism.

With 32-core parallelism (added in v0.2.0), metrust is now **competitive with or faster than SciPy** on most grid sizes. At sigma=5 on a 500x500 grid, metrust is 2.6x faster. SciPy still wins at very small grids with small sigma where parallelism overhead exceeds the work. Additionally, metrust provides NaN-aware weighted averaging that SciPy does not support at all.

**Takeaway:** Parallelism can overcome an algorithm disadvantage. SciPy's IIR approach is O(n) regardless of sigma, while metrust's FIR is O(n*k), but rayon distributes the work across cores effectively.

---

## Why the Speedup Varies

The fair speedup (T3/T2) ranges from 12x to 90x across different functions. Three factors explain the variation:

### Scalar Functions: Pint is the Bottleneck

For scalar thermodynamic calls -- `potential_temperature`, `wet_bulb_temperature`, `dewpoint_from_rh` -- the actual math takes tens to hundreds of nanoseconds in Rust (T1). The dominant cost in both MetPy and metrust is Pint: constructing unit quantities, checking dimensional compatibility, converting between unit systems, and wrapping the result.

MetPy pays this cost *and* runs the math in Python. metrust pays the Pint cost but completes the math in nanoseconds. The speedup is large (17--90x) because MetPy's Python math is slow relative to Pint overhead, while metrust's Rust math is negligible relative to the same Pint overhead.

Functions with more complex internals (`wet_bulb_temperature` involves iterative solving; `equivalent_potential_temperature` chains multiple thermodynamic calculations) show higher speedups because MetPy spends proportionally more time in Python loops.

### Array Functions: Rust Array Bindings Bypass Python

28 hot-path functions have dedicated Rust array bindings exposed through PyO3. When you pass a NumPy array to `metrust.calc.divergence`, the wrapper extracts the raw buffer pointer once, hands it to Rust, and gets back a new NumPy array. There is no Python loop, no per-element Pint operation, no intermediate array allocation.

MetPy, by contrast, performs element-wise NumPy operations that each create intermediate arrays, apply Pint unit tracking at every step, and incur Python interpreter overhead for each operation in the chain.

The result: grid kinematics like `divergence` on a 100x100 grid show 79x speedup because MetPy's multi-step Python+NumPy+Pint pipeline is replaced by a single Rust traversal.

### Grid Composites: Rayon Parallelism

The `compute_cape_cin`, `compute_srh`, `compute_shear`, and other grid composite functions use [rayon](https://docs.rs/rayon/) to parallelize across grid points. Each column in the 3-D grid is processed independently, and rayon distributes the work across all available CPU cores.

This matters for production workloads where you need CAPE/CIN at every point in a 1059x1799 HRRR grid (1.9 million columns). A Python loop calling MetPy's `cape_cin` at each grid point would take minutes. The rayon-parallelized Rust version processes the entire grid in a single function call, with the GIL released during computation.

---

## Grid Composite Performance

The `compute_cape_cin` function demonstrates production-scale parallelism. It accepts 3-D model fields (pressure, temperature, moisture, height) and 2-D surface fields, extracts a sounding column at every grid point, and computes CAPE, CIN, LCL pressure, and LFC pressure in parallel.

### Interface

```python
from metrust.calc import compute_cape_cin

cape, cin, lcl, lfc = compute_cape_cin(
    pressure_3d,        # (nz, ny, nx), Pa
    temperature_c_3d,   # (nz, ny, nx), Celsius
    qvapor_3d,          # (nz, ny, nx), kg/kg
    height_agl_3d,      # (nz, ny, nx), meters AGL
    psfc,               # (ny, nx), Pa
    t2,                 # (ny, nx), Kelvin
    q2,                 # (ny, nx), kg/kg
    parcel_type="surface",  # "surface", "mixed_layer", or "most_unstable"
    top_m=None,             # optional cap height in meters (e.g., 3000 for SB3CAPE)
)
# Returns four (ny, nx) arrays with Pint units
```

### What Happens Under the Hood

1. Python wrapper flattens the 3-D and 2-D arrays to contiguous 1-D buffers.
2. PyO3 hands the raw `f64` slices to Rust with zero copy.
3. Rust iterates over all `ny * nx` grid points using `rayon::into_par_iter()`.
4. Each parallel task extracts its column, orders it surface-to-top, computes the parcel profile, integrates CAPE/CIN, and identifies LCL/LFC.
5. Results are collected into four flat `Vec<f64>` buffers and returned to Python as NumPy arrays.

The GIL is held only during the brief PyO3 argument marshaling. The entire parallel computation runs without it.

### Full HRRR Grid (1059 x 1799)

A full HRRR domain has 1,905,141 grid points, each requiring a full sounding analysis (parcel profile + buoyancy integration). On a multi-core machine, `compute_cape_cin` processes this grid in a few seconds -- a task that would take far longer in a Python loop even with MetPy's per-column speed.

The same parallel pattern is used by:

- `compute_srh` -- storm-relative helicity at every grid point
- `compute_shear` -- bulk shear at every grid point
- `compute_lapse_rate` -- environmental lapse rate at every grid point
- `compute_stp` -- significant tornado parameter at every grid point
- `compute_scp` -- supercell composite parameter at every grid point

---

## Running Benchmarks Yourself

### Rust Microbenchmarks (Criterion)

```bash
cargo bench --package metrust
```

This runs [Criterion](https://docs.rs/criterion/)-based benchmarks for the Rust calculation layer. Results appear as HTML reports in `target/criterion/` with statistical analysis, regression detection, and comparison against previous runs.

### Python Three-Tier Benchmarks

```bash
# T1 (raw Rust) and T2 (metrust+Pint) -- no MetPy needed
python benches/bench_python.py

# All three tiers -- requires MetPy for T3
python benches/bench_python.py --tier 1,2,3

# Run only a specific category
python benches/bench_python.py --category thermo
python benches/bench_python.py --category wind
python benches/bench_python.py --category grid_kinematics

# Machine-readable JSON output
python benches/bench_python.py --json
python benches/bench_python.py --json --json-file my_results.json
```

The benchmark harness:

- Runs 5 warmup calls to eliminate cold-start effects (JIT, cache priming).
- Auto-tunes the iteration count so each trial takes approximately 0.2 seconds.
- Collects 7 independent trials and reports the p50 (median), p95, and standard deviation.
- Computes the fair speedup ratio (T3/T2) and Pint overhead ratio (T2/T1) automatically.

### HRRR Full-Grid Benchmarks

```bash
# Requires HRRR GRIB2 files in data/
python benches/bench_hrrr.py

# Quick mode (fewer trials)
python benches/bench_hrrr.py --quick

# Specific category
python benches/bench_hrrr.py --category sounding
python benches/bench_hrrr.py --category grid_kinematics

# JSON output
python benches/bench_hrrr.py --json
```

The HRRR benchmark uses real model output (1059x1799 grid, 40 pressure levels) to test every function at production scale. This catches performance characteristics that synthetic data at small sizes cannot reveal.

### MetPy Head-to-Head

```bash
# Requires both metrust and MetPy installed, plus HRRR data
python benches/bench_hrrr_vs_metpy.py
```

This script benchmarks raw Rust (metrust) against MetPy+Pint on identical HRRR data, covering scalar thermodynamics, sounding analysis, and full-grid operations in a single run.

---

## Interpreting the Numbers

A few guidelines for reading benchmark results honestly:

**p50 (median) is the primary metric.** It is more stable than the mean because it is not affected by occasional GC pauses or OS scheduling jitter. The p95 shows the tail, which matters for latency-sensitive applications.

**The fair speedup is T3/T2.** This is what a user actually experiences when switching from `metpy.calc` to `metrust.calc` in code that already uses Pint. Quoting T3/T1 overstates the benefit because no real user code calls the raw Rust API with bare floats.

**Scalar benchmarks measure overhead, not compute.** When the Rust computation takes 60 nanoseconds and the Pint wrapper takes 7 microseconds, the benchmark is primarily measuring Pint performance. This is still the right thing to measure -- it reflects what the user's code actually does -- but it means the speedup comes from avoiding MetPy's Python math, not from Rust being 2000x faster at arithmetic.

**Array benchmarks are more representative of real workloads.** Operational meteorology rarely calls `potential_temperature` on a single value. The array and grid benchmarks reflect the kind of vectorized, batch-oriented processing that weather analysis code actually performs.
