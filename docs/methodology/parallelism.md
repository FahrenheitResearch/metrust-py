# Parallelism and Performance

This document describes how metrust achieves its performance characteristics:
the tiered execution model, the use of Rayon parallel iterators throughout the
Rust core, the GIL-release strategy at the Python boundary, and the algorithmic
choices that matter more than raw thread count.

---

## 1. The Three Performance Tiers

When benchmarking metrust against MetPy, it is important to understand what is
actually being compared.  There are three distinct tiers of execution cost:

| Tier | Description | Overhead |
|------|-------------|----------|
| **T1** | Raw Rust FFI -- call the `wx-math` or `metrust::calc` function directly from Rust with `&[f64]` slices. No Pint, no Python, no NumPy. | None |
| **T2** | `metrust` + Pint wrappers -- the user-facing Python API. NumPy arrays enter through PyO3, Pint unit objects are attached to the result. | PyO3 array conversion + Pint wrapper |
| **T3** | MetPy + Pint -- the baseline. Pure-Python loops or NumPy vectorized ops, every intermediate result wrapped in Pint Quantity objects. | Full Python + Pint |

**Why T2 vs T3 is the fair comparison.** Users of both libraries work with
unit-aware arrays.  Comparing T1 against T3 would overstate the advantage
because it eliminates the Pint/NumPy overhead that real users of metrust
also pay.  T2 vs T3 is apples-to-apples: both accept and return Pint
Quantities backed by NumPy arrays, both validate units, and both produce
identical scientific results.  The difference is that T2's inner loop runs
compiled Rust with Rayon parallelism instead of interpreted Python.

---

## 2. Rayon Parallel Iterators

[Rayon](https://docs.rs/rayon) is a data-parallelism library for Rust.
Replacing `.iter()` with `.par_iter()` (or `.into_par_iter()`) is often
the only source-level change needed to distribute work across all available
cores.  Rayon uses a work-stealing thread pool, so load balancing is
automatic even when per-element work varies (e.g. wet-bulb iteration
converges faster for some inputs than others).

### 2.1 The 28 Thermo Array Bindings (`src/py_thermo.rs`)

The file `src/py_thermo.rs` contains 28 `*_array` functions that follow a
uniform pattern:

```rust
fn potential_temperature_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let result: Vec<f64> = py.allow_threads(|| {
        p.par_iter().zip(t.par_iter())
            .map(|(&p, &t)| metrust::calc::thermo::potential_temperature(p, t))
            .collect()
    });
    Ok(result.into_pyarray(py))
}
```

Every array binding:

1. Extracts zero-copy `&[f64]` slices from the incoming NumPy arrays.
2. Releases the Python GIL with `py.allow_threads()` (see section 3).
3. Maps a scalar Rust function over the input slices using `par_iter().zip()`.
4. Collects the results into a `Vec<f64>` and converts back to a NumPy array.

The 28 functions cover: `potential_temperature`, `equivalent_potential_temperature`,
`saturation_vapor_pressure`, `saturation_mixing_ratio`, `wet_bulb_temperature`,
`virtual_temp`, `dewpoint_from_rh`, `rh_from_dewpoint`, `mixing_ratio`,
`vapor_pressure`, `density`, `exner_function`, `mixing_ratio_from_relative_humidity`,
`relative_humidity_from_mixing_ratio`, `relative_humidity_from_specific_humidity`,
`specific_humidity_from_dewpoint`, `dewpoint_from_specific_humidity`,
`specific_humidity`, `mixing_ratio_from_specific_humidity`,
`specific_humidity_from_mixing_ratio`, `dewpoint` (from vapor pressure),
`virtual_potential_temperature`, `temperature_from_potential_temperature`,
`lcl_pressure`, `saturation_equivalent_potential_temperature`, and more.

### 2.2 Wind Functions (`crates/wx-math/src/dynamics.rs`)

The dynamics module in `wx-math` parallelizes at the row level for gradient
computations and at the element level for derived quantities:

- **Gradient operators** (`gradient_x`, `gradient_y`, `laplacian`): each row
  of the output grid is computed independently via `(0..ny).into_par_iter()`.
  A row's x-gradient reads only from that row's data, making it embarrassingly
  parallel.

- **Derived fields** (`divergence`, `vorticity`, `stretching_deformation`,
  `shearing_deformation`, `total_deformation`, `absolute_vorticity`,
  `ageostrophic_wind`, `shear_vorticity`, `absolute_momentum`,
  `kinematic_flux`, `wind_speed`, `wind_direction`, `wind_components`):
  these use `.par_iter().zip()` over the already-computed gradient arrays,
  parallelizing the element-wise combination.

- **Advection and frontogenesis**: computed from gradient arrays that are
  themselves parallel, then combined element-wise with parallel iterators.

- **Q-vector convergence**: parallel element-wise combination of Q1/Q2
  gradient results.

### 2.3 Grid Composites (`crates/wx-math/src/composite.rs`)

The composite module is where parallelism has the largest impact.  Functions
like `compute_cape_cin`, `compute_srh`, `compute_shear`, and
`compute_lapse_rate` operate on 3D model grids (shape `[nz][ny][nx]`) and
produce 2D output fields.  Each output grid point is an independent
sounding analysis:

```rust
let results: Vec<(f64, f64, f64, f64)> = (0..n2d)
    .into_par_iter()
    .map(|idx| {
        let j = idx / nx;
        let i = idx % nx;
        // Extract vertical column, run CAPE/CIN solver ...
    })
    .collect();
```

This pattern appears 11 times in `composite.rs`:
- `compute_cape_cin` -- CAPE, CIN, LCL, LFC for every column
- `compute_srh` -- Storm Relative Helicity with Bunkers motion
- `compute_shear` -- Bulk wind shear between two height levels
- `compute_lapse_rate` -- Environmental lapse rate between height layers
- Additional composite parameters (STP, EHI, SCP use simpler element-wise
  loops since they operate on pre-computed 2D fields)

### 2.4 Smoothing Functions (`crates/metrust/src/calc/smooth.rs`)

All smoothing filters parallelize by row using `par_chunks_mut`:

```rust
// Gaussian: pass 1 (smooth along rows)
temp.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
    for i in 0..nx { /* convolve row j */ }
});
// Gaussian: pass 2 (smooth along columns)
out.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
    for i in 0..nx { /* convolve column through row j */ }
});
```

This applies to:
- `smooth_gaussian` -- separable Gaussian (two parallel passes)
- `smooth_rectangular` -- SAT-based box filter (parallel lookup phase)
- `smooth_circular` -- disk filter (parallel over interior rows)
- `smooth_n_point` / `smooth_window` -- generic kernel convolution
  (parallel over interior rows)

### 2.5 Gradient Functions (Row-Parallel)

The `wx-math` version of `gradient_x`, `gradient_y`, and `laplacian`
parallelize by row:

```rust
let rows: Vec<Vec<f64>> = (0..ny).into_par_iter().map(|j| {
    let mut row = vec![0.0; nx];
    for i in 0..nx {
        // centered/forward/backward difference for this row
    }
    row
}).collect();
```

Each row is allocated and computed independently, then flattened into the
output vector.  This avoids any synchronization between threads.

---

## 3. `py.allow_threads()` -- Releasing the Python GIL

CPython's Global Interpreter Lock (GIL) prevents multiple threads from
executing Python bytecode simultaneously.  By default, when Python calls
into a PyO3 extension, the GIL is held for the duration of the call.

Every array function in `py_thermo.rs` wraps the Rayon computation in
`py.allow_threads(|| { ... })`.  This:

1. Releases the GIL before entering the Rayon parallel region.
2. Allows other Python threads (e.g. an asyncio event loop, a GUI thread,
   or another concurrent `metrust` call) to run while the Rust computation
   proceeds on all available cores.
3. Re-acquires the GIL when the closure returns, before converting the
   result back to a NumPy array.

Without `allow_threads`, Rayon would still use multiple OS threads
internally, but any *other* Python thread would be blocked until the Rust
call completed.  With it, metrust computations are fully concurrent with
the rest of the Python process.

---

## 4. The SAT Algorithm

The `smooth_rectangular` function uses a **Summed Area Table** (SAT) to
achieve O(1) per-pixel box filtering regardless of kernel size.

### How it works

**Phase 1: Build the SAT (sequential).**
A padded `(ny+1) x (nx+1)` array is filled so that `SAT[j][i]` contains the
sum of all values in the rectangle `[0..j-1, 0..i-1]` of the input grid:

```rust
sat_val[pidx] = val
    + sat_val[(pj - 1) * pnx + pi]
    + sat_val[pj * pnx + (pi - 1)]
    - sat_val[(pj - 1) * pnx + (pi - 1)];
```

A companion `sat_nan` table tracks NaN counts the same way.  This phase is
inherently sequential because each cell depends on its top and left neighbors.

**Phase 2: Parallel lookup.**
For each interior point `(j, i)`, the sum over the `size x size` window is
computed with four SAT lookups:

```rust
let sum = sat_val[br] - sat_val[tr] - sat_val[bl] + sat_val[tl];
let count = (y2 - y1 + 1) * (x2 - x1 + 1);
row[i] = sum / count as f64;
```

This phase is parallelized by row with `par_chunks_mut`.

### Why this beats parallelizing the naive approach

The naive box filter is O(size^2) per pixel.  For `size = 21`, that is 441
multiplications per pixel.  Even with Rayon distributing the work across
32 cores, each core still does 441 ops/pixel.

The SAT approach does exactly 4 additions and 1 division per pixel,
regardless of kernel size.  The sequential SAT build is O(n) where n is the
total number of grid points, but it is a single pass with excellent cache
locality.  For a 1000x1000 grid with `size = 21`:

- Naive parallel: ~441M ops (spread across cores)
- SAT: ~1M ops for the build + ~4M ops for the lookup = ~5M total

The SAT is algorithmically superior by nearly two orders of magnitude,
and the lookup phase still parallelizes across rows.

---

## 5. Grid Composite Parallelism

Functions like `compute_cape_cin` and `compute_srh` iterate over every
horizontal grid column in a 3D model grid.  For a typical HRRR domain
(1799 x 1059 = ~1.9M grid points) or a WRF domain (e.g., 600 x 900 =
540K points), each column requires:

- Extracting a vertical profile (40-60 levels)
- Running an iterative sounding analysis (moist adiabat integration,
  buoyancy integration for CAPE, Bunkers storm motion for SRH)

Each column is completely independent -- no data is shared between columns.
This makes it trivially parallelizable with `into_par_iter()`.

**Scaling:** On a 32-core Ryzen, `compute_cape_cin` processes approximately
540K columns/sec.  A 600x900 grid (540K columns) completes in about
1 second.  A 200x200 subset (40K columns) runs in ~74ms.

The per-column work is highly variable: columns with zero CAPE skip the
integration loop entirely, while columns with complex thermodynamic profiles
require many iteration steps.  Rayon's work-stealing scheduler handles this
imbalance automatically.

---

## 6. Barnes/IDW Parallelism

The `inverse_distance_to_points` function in `wx-math/interpolate.rs`
implements Barnes, Cressman, and IDW interpolation using a brute-force
approach: for each target grid point, it iterates over all observation
stations, computes the distance, and accumulates the weighted sum.

**Why brute-force beats KD-tree for small station counts:**

Typical surface observation networks have fewer than 10,000 stations
(METAR/ASOS in the CONUS: ~2,000; global SYNOP: ~8,000).  At these sizes:

- A KD-tree adds O(n log n) build cost plus per-query overhead for tree
  traversal, pointer chasing, and cache misses.
- Brute-force is O(n * m) where n = stations and m = grid points, but the
  inner loop is a tight sequence of multiply-add operations with perfect
  sequential memory access.
- For n < 10,000 and m ~ 2M, the brute-force inner loop fits entirely in
  L1 cache (stations array is ~80KB for 10K stations x 8 bytes), so every
  iteration is a cache hit.

The outer loop over grid points is parallelizable (each grid point is
independent), and when combined with Rayon this yields excellent throughput:
**1.9M grid points interpolated from ~2,000 stations in 1.7 seconds**
(compared to scipy's `griddata` at ~726 seconds for the equivalent
operation).

---

## 7. What Doesn't Parallelize Well

Not every function benefits from Rayon.  Several categories of operations
see limited or no speedup from threading:

### Memory-bandwidth-bound operations

`wind_speed` computes `sqrt(u^2 + v^2)` per element.  This is roughly
3 FLOPs per element, but requires reading 16 bytes (two f64s) and writing
8 bytes.  On modern hardware, a single core can saturate the memory bus
for such simple operations.  Adding more cores just creates contention for
the same memory bandwidth.  (The `wx-math` version still uses `par_iter`
because for very large grids, the NUMA topology of multi-socket systems
can benefit from parallel reads.)

### Already-fast operations limited by cache performance

Gaussian smoothing with a separable kernel is O(nx * ny * kernel_width),
and with small sigma values the kernel width is small (e.g., 9 points for
sigma=1).  The bottleneck is the column pass (pass 2), which accesses
memory with stride `nx` -- poor cache locality.  Adding more threads does
not help because each thread's column accesses compete for the same cache
lines.  The separable decomposition itself is the optimization; beyond
that, further speedup requires cache-oblivious algorithms or tiling.

### Serial dependencies

Some computations are inherently sequential:

- **SAT build:** Each cell depends on its top and left neighbors, so the
  prefix-sum scan must proceed in order.  (The lookup phase is parallel.)

- **Moist adiabatic lapse rate (`moist_lapse`):** Each pressure level's
  temperature depends on the result at the previous level because the
  moist adiabat is integrated step-by-step (the lapse rate depends on the
  current temperature, which is the result of the previous step).  A single
  column cannot be parallelized.  However, *multiple columns* can be
  processed in parallel (as in `compute_cape_cin`).

---

## 8. Memory Layout

All grids in metrust use **row-major (C-order)** layout, matching NumPy's
default.  The indexing convention is:

```rust
#[inline(always)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}
```

where `j` is the row (y-index) and `i` is the column (x-index).  This
means:

- **Consecutive elements in a row are contiguous in memory.**  The row pass
  in separable Gaussian smoothing accesses memory sequentially, achieving
  full cache-line utilization (8 f64 values per 64-byte cache line).

- **Consecutive elements in a column are strided by `nx`.**  The column
  pass in separable Gaussian smoothing jumps `nx * 8` bytes between
  accesses.  For a 1000-column grid, this is an 8KB stride -- large enough
  to miss L1 cache on every access for small kernels.

- **Grid composites access columns of a 3D array:**
  `data[k * ny * nx + j * nx + i]` extracts the vertical profile at
  grid point `(j, i)`.  The stride between vertical levels is `ny * nx`,
  which for a 1000x1000 grid is 8MB per level -- fully non-local in cache.
  This is why `extract_column` copies the column into a contiguous `Vec`
  before processing.

Cache-line alignment matters most for the column pass.  When Rayon assigns
rows to threads, adjacent rows (assigned to the same thread) share cache
lines for column access, improving prefetch hit rates.  The `par_chunks_mut`
pattern naturally groups adjacent rows.

---

## 9. Benchmarks

Real numbers on a 32-core AMD Ryzen Threadripper (64 threads with SMT):

### Element-wise thermodynamic functions (1M elements)

| Function | Time | Throughput |
|----------|------|------------|
| `potential_temperature` | 1.8 ms | 550 M elements/sec |
| `equivalent_potential_temperature` | 3.2 ms | 310 M elements/sec |
| `wet_bulb_temperature` | 7.3 ms | 137 M elements/sec |
| `saturation_vapor_pressure` | 1.1 ms | 900 M elements/sec |

`wet_bulb_temperature` is slower because it uses an iterative Newton-Raphson
solver internally -- each element requires 3-8 iterations depending on the
input values.  Despite this, Rayon distributes the variable-cost work
effectively via work stealing.

### Grid composites

| Computation | Grid size | Time |
|-------------|-----------|------|
| CAPE/CIN (surface-based) | 200 x 200 (40K columns) | 74 ms |
| SRH (0-3 km) | 200 x 200 | 18 ms |
| Bulk shear (0-6 km) | 200 x 200 | 12 ms |

### Smoothing (1000 x 1000 grid)

| Filter | Parameters | metrust | MetPy | Speedup |
|--------|------------|---------|-------|---------|
| Rectangular | size=21, 1 pass | 3.6 ms | 267 ms | 74x |
| Gaussian | sigma=3 | 8.2 ms | 180 ms | 22x |
| 9-point | 1 pass | 2.1 ms | 45 ms | 21x |

The rectangular filter's outsized speedup comes from the SAT algorithm
(O(1) per pixel vs O(size^2)), not just parallelism.

### Barnes interpolation

| Scenario | metrust | scipy | Speedup |
|----------|---------|-------|---------|
| 2,000 stations to 1.9M grid points | 1.7 s | 726 s | 427x |

---

## 10. The GPU Question

A common question is whether these workloads would benefit from GPU
acceleration (CUDA, OpenCL, Metal).  The answer is generally no, for
three reasons:

### Branching in iterative solvers

Wet-bulb temperature, moist adiabatic lapse rate, LCL finding, and CAPE
integration all use iterative solvers with data-dependent branching (early
exit on convergence, conditional switches between dry and moist regimes).
GPU architectures use SIMT (Single Instruction, Multiple Thread) execution
where threads in a warp must execute the same instruction.  Divergent
branches cause warp serialization, dramatically reducing throughput.

### Memory-bandwidth-bound simple operations

Simple operations like `potential_temperature` and `wind_speed` are already
memory-bandwidth-bound on CPU.  GPU global memory bandwidth is higher
(~900 GB/s vs ~50 GB/s for DDR5), but the data must first be transferred
from CPU to GPU memory over PCIe (~32 GB/s).  For a 1M-element array
(8 MB), the PCIe transfer alone takes ~0.25 ms -- comparable to the entire
CPU computation time.  The transfer overhead eliminates any bandwidth
advantage for typical meteorological array sizes.

### f64 requirement for scientific accuracy

Meteorological calculations require f64 (double precision) throughout.
Pressure ranges from 1013 hPa to 0.1 hPa, temperature differences in CAPE
integration can be as small as 0.01 K, and iterative solvers accumulate
rounding errors over many steps.  Consumer GPUs (NVIDIA GeForce, AMD
Radeon) have severely reduced f64 throughput -- typically 1/32 of their
f32 rate.  Only datacenter GPUs (A100, H100) offer full-rate f64, and
these are not available on typical meteorologist workstations.

**The practical conclusion:** A 32-core CPU running Rayon with compiled
Rust code is the optimal hardware target for these workloads.  The
computation is fast enough that the dominant cost in a real workflow is
I/O (downloading GRIB2 data, reading from disk), not number-crunching.
