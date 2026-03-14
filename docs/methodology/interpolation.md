# Interpolation

This document describes the interpolation algorithms implemented in metrust, covering their mathematical foundations, Rust implementation strategies, and operational guidance for meteorological applications.

Source files:

- **Core engine (wx-math):** `crates/wx-math/src/interpolate.rs`
- **Metrust library layer:** `crates/metrust/src/interpolate/mod.rs`
- **Python bindings:** `src/py_interpolate.rs`
- **Python API:** `python/metrust/interpolate/__init__.py`

---

## Barnes Objective Analysis

Barnes objective analysis is the primary algorithm used for blending scattered surface observations onto a regular grid. It is the workhorse behind mesoanalysis: given a set of irregularly-spaced ASOS/AWOS station reports and a model first-guess field, Barnes analysis produces a smooth, physically plausible gridded field that honors the observations.

### What it does

The method assigns every observation a Gaussian weight that decays with distance from the target grid point. Nearby stations dominate; distant stations contribute negligibly. The weighted average of all station values becomes the gridded value at that point.

This is a form of inverse-distance weighting where the weight function is specifically chosen to be Gaussian, which produces smooth fields without the discontinuities or ringing artifacts that other weight functions can introduce.

### Weight function

For a single grid point, the weight assigned to an observation at distance *d* is:

```
w = exp(-d^2 / kappa)
```

where **kappa** is the smoothing parameter that controls the influence radius. Larger kappa values produce smoother fields (observations influence grid points farther away); smaller kappa values produce fields that track individual station reports more tightly.

The interpolated value at a grid point is the weighted average:

```
V_grid = sum(w_i * V_i) / sum(w_i)
```

where the sum runs over all observations within the search radius.

### Multi-pass refinement with gamma

A single Barnes pass produces a smooth field but may not fit the observations closely enough for mesoscale analysis. The standard approach is a two-pass scheme:

1. **First pass** uses kappa to produce a smooth background field.
2. **Second pass** computes residuals (observed minus first-pass estimate at each station), then performs Barnes analysis on the residuals using an effective smoothing parameter of `kappa * gamma`.

The gamma parameter (typically 0.2--0.4) tightens the Gaussian on the second pass, allowing the analysis to recover smaller-scale features that the first pass smoothed away. The final field is the sum of the first-pass field and the second-pass correction.

In the metrust implementation, the caller is responsible for computing residuals between passes. The `barnes_point` function in `crates/wx-math/src/interpolate.rs` performs a single-pass weighted average. For a two-pass analysis, the caller calls it twice: once with `kappa`, once with `kappa * gamma` on the residual values.

The unified `inverse_distance_to_points` function (kind=1) computes `w = exp(-d^2 / (kappa * gamma))` in a single call, which is the second-pass weighting. For first-pass use, pass `gamma=1.0`.

### Why Barnes analysis is used in mesoanalysis

Surface mesoanalysis merges approximately 1,700 ASOS/AWOS stations across the CONUS with a model first-guess field (typically RAP or HRRR). The stations are irregularly spaced -- dense in urban areas, sparse in the mountain West -- and the first-guess field provides physically consistent values in data-void regions. Barnes analysis handles this naturally: where stations are dense, the analysis tracks observations; where stations are sparse, the smooth Gaussian decay lets the first-guess field dominate.

### Operational kappa values

The smoothing parameter kappa has units of distance-squared (matching the coordinate system of the input). Typical values depend on the application:

| Application | Approximate kappa | Notes |
|---|---|---|
| Synoptic-scale (500 km features) | 500,000--1,000,000 m^2 | Very smooth, suitable for frontal analysis |
| Mesoscale (50--200 km features) | 50,000--200,000 m^2 | Standard SPC mesoanalysis range |
| Storm-scale (10--50 km features) | 5,000--50,000 m^2 | Tight fit, requires dense observations |

When using projected coordinates (metres), kappa is in m^2. When using degree coordinates, kappa is in degrees^2. The default in the Python binding is `kappa=100000.0`.

A typical gamma value is 0.2, meaning the second pass uses a smoothing scale sqrt(0.2) ~ 0.45 times as wide as the first pass.

### Implementation: brute-force beats KD-tree

The Rust implementation uses a brute-force linear scan: for each grid point, it computes the distance to every observation station, filters by the search radius, and accumulates the weighted average.

This is O(n_grid * n_stations) -- seemingly wasteful. A KD-tree would reduce per-grid-point lookups to O(log n_stations). However, for the typical CONUS mesoanalysis problem (~1,700 stations), the KD-tree construction and traversal overhead exceeds the cost of a simple linear scan. The branch-heavy tree traversal also defeats CPU branch prediction and cache prefetching, while the linear scan is perfectly predictable and cache-friendly.

The breakeven point is roughly 10,000--50,000 stations. Below that, brute force wins.

### Rayon parallelism

Each grid point's computation is completely independent: it reads from the shared observation arrays and writes to its own output slot. This is an embarrassingly parallel workload. The `inverse_distance_to_points` function in `crates/wx-math/src/interpolate.rs` uses Rayon's `par_iter` over grid points:

```rust
grid_x
    .par_iter()
    .zip(grid_y.par_iter())
    .map(|(&gx, &gy)| {
        // brute-force scan over all observations
        // compute weighted average
    })
    .collect()
```

Rayon automatically partitions the grid points across available CPU cores. On an 8-core machine, this yields roughly 6--7x speedup over the single-threaded version (the remaining overhead is work-stealing and memory allocation).

### Performance

On a typical CONUS mesoanalysis grid:

- **Grid size:** 1.9 million points (1300 x 1500 at ~3 km spacing)
- **Stations:** ~1,674 ASOS/AWOS reports
- **Wall time:** ~1.7 seconds (8-core Rayon, brute-force)

For comparison, the equivalent scipy-based implementation in Python takes approximately 726 seconds on the same problem -- a **427x speedup**. The gains come from three sources: compiled Rust vs interpreted Python loops, Rayon parallelism, and cache-friendly memory access patterns.

The Python binding releases the GIL during the Rust computation via `py.allow_threads()`, so the Barnes analysis can run concurrently with other Python threads.

---

## Cressman Objective Analysis

Cressman analysis is an older alternative to Barnes that uses a parabolic weight function with a hard cutoff at the search radius.

### Weight function

```
w = (R^2 - d^2) / (R^2 + d^2)    for d <= R
w = 0                              for d > R
```

where *R* is the search radius and *d* is the distance from the grid point to the observation.

Properties of the Cressman weight:

- At d = 0: w = 1 (observation dominates when coincident)
- At d = R: w = 0 (sharp cutoff at the radius boundary)
- Monotonically decreasing from 1 to 0

### Comparison with Barnes

The Cressman weight drops to exactly zero at the search radius, producing a field with a finite influence radius. Barnes weights approach zero asymptotically but never reach it (though the search radius provides a practical cutoff). This means:

- **Cressman** produces sharper gradients near the edge of the influence radius and can leave grid points unanalyzed if no stations fall within *R*.
- **Barnes** produces smoother transitions and is less sensitive to the choice of search radius.

In practice, Barnes is preferred for most meteorological applications because the Gaussian weight function matches the spectral characteristics of atmospheric fields better than the parabolic weight.

### Implementation

The `cressman_point` function in `crates/wx-math/src/interpolate.rs` implements the per-point calculation. It is dispatched via `inverse_distance_to_points` with `kind=2` (or `kind=0` in the standalone `cressman_point` API).

```rust
pub fn cressman_point(distances: &[f64], values: &[f64], radius: f64) -> f64 {
    let r2 = radius * radius;
    // accumulate w and w*v for d^2 <= r2
    // return wv_sum / w_sum, or NaN if no points within radius
}
```

### Typical search radii

| Application | Search radius |
|---|---|
| Synoptic analysis | 300--500 km |
| Mesoscale analysis | 100--200 km |
| Local analysis | 50--100 km |

---

## Inverse Distance Weighting (IDW)

IDW is the general framework that encompasses Barnes and Cressman as special cases. The standard IDW weight function is:

```
w = 1 / d^p
```

where *p* is the power parameter (typically 2).

### Unified API

The `inverse_distance_to_points` function provides a unified entry point for all three weighting schemes via the `kind` parameter:

| kind | Scheme | Weight function |
|---|---|---|
| 0 | Standard IDW | `w = 1 / d^2` |
| 1 | Barnes | `w = exp(-d^2 / (kappa * gamma))` |
| 2 | Cressman | `w = (R^2 - d^2) / (R^2 + d^2)` |

All schemes share the same structure:

1. For each target point, scan all observations within the search radius.
2. Compute the scheme-specific weight for each observation.
3. Return the weighted average, or NaN if fewer than `min_neighbors` observations fall within the radius.

### The min_neighbors parameter

The `min_neighbors` parameter (default 3) prevents the algorithm from producing unreliable estimates in data-sparse regions. If a grid point has fewer observations within its search radius than `min_neighbors`, the output is set to NaN rather than computing a weighted average from too few data points.

This is important for operational use: a grid point influenced by a single station 200 km away should not be treated as a reliable analysis value. The NaN output signals that the grid point is in a data void and should be filled by the first-guess field or flagged for the forecaster.

### Legacy vs unified API

The Python module exposes two IDW interfaces:

- **`inverse_distance_to_points`** (unified): Accepts `kind`, `kappa`, and `gamma` parameters. Uses the wx-math engine with Rayon parallelism. Coordinates are in projected space (metres or arbitrary units).
- **`inverse_distance_to_points_legacy`**: Uses haversine distance in degree-space. Power parameter is configurable. Single-threaded. Preserved for backward compatibility.

The legacy API computes distances using the haversine formula on lat/lon coordinates, which is geodetically correct but slower than the projected-coordinate approach used by the unified API.

---

## Natural Neighbor Interpolation

Natural neighbor (Sibson) interpolation uses the geometry of Voronoi diagrams to assign weights. In theory, inserting the target point into the Voronoi tessellation of the source points, computing the area of overlap between the new Voronoi cell and each original cell, and using those areas as weights produces a smooth, local interpolant.

### Approximate implementation

True natural neighbor interpolation requires incremental Voronoi construction, which is computationally expensive. metrust implements a practical approximation: for each target point, the K nearest source points are found (K = min(12, n)), and weights are computed as inverse-distance-squared:

```
w_i = 1 / d_i^2
```

The K-nearest selection is performed using a partial sort (`select_nth_unstable_by`), which is O(n) rather than O(n log n) for a full sort. This produces results that closely approximate true Sibson weights in smoothly-varying fields.

### When to use natural neighbor vs IDW

| Criterion | Natural neighbor | IDW |
|---|---|---|
| Source density | Works well with irregular spacing | Works well with any spacing |
| Smoothness | Smoother (adaptive neighborhood) | Can show "bull's-eye" artifacts with power >= 2 |
| Parameters | Parameter-free (no radius/kappa to tune) | Requires radius, power, min_neighbors |
| Performance | Slower (K-nearest search per point) | Faster (simple radius filter) |
| Edge behavior | Extrapolates via nearest neighbors | Returns NaN outside radius |

Natural neighbor is the better default choice when you have no prior knowledge of the appropriate search radius or smoothing scale. IDW (and especially Barnes) is preferred when you need explicit control over the smoothing characteristics.

### Python API

```python
from metrust.interpolate import natural_neighbor_to_grid, natural_neighbor_to_points

# Grid output
grid = natural_neighbor_to_grid(lats, lons, values, target_grid)

# Point output
vals = natural_neighbor_to_points(src_lats, src_lons, src_values,
                                   target_lats, target_lons)
```

---

## 1-D Interpolation

### interpolate_1d -- Linear interpolation

Piecewise linear interpolation matching the behavior of `numpy.interp`, with one difference: values outside the range `[xp[0], xp[-1]]` return NaN (no extrapolation), matching MetPy's convention.

The implementation uses binary search to find the enclosing interval, then computes:

```
t = (x - xp[lo]) / (xp[hi] - xp[lo])
result = fp[lo] + t * (fp[hi] - fp[lo])
```

**Complexity:** O(log n) per query point due to binary search.

**Requirements:** `xp` must be monotonically increasing. `xp` and `fp` must have the same length and be non-empty.

### log_interpolate_1d -- Log-pressure interpolation

Performs linear interpolation in ln(x) space. This is the correct approach for interpolating meteorological variables with respect to pressure, because pressure decreases approximately exponentially with height. Interpolating linearly in pressure space would introduce systematic bias; interpolating in log-pressure space preserves the physical relationship.

The implementation transforms both the query points and breakpoints to natural-log space, then delegates to `interpolate_1d`. If the breakpoints are in descending order (typical for pressure coordinates, where the surface is ~1000 hPa and upper levels decrease), the arrays are internally reversed before interpolation.

```python
from metrust.interpolate import log_interpolate_1d

# Interpolate temperature to 925 and 775 hPa
target_p = [925.0, 775.0]
pressure_levels = [1000.0, 850.0, 700.0, 500.0]  # descending
temperature = [20.0, 12.0, 2.0, -15.0]

result = log_interpolate_1d(target_p, pressure_levels, temperature)
```

The function also supports interpolating multiple fields simultaneously via the `*args` pattern in the Python wrapper: pass additional arrays after `xp` and they are all interpolated against the same coordinate.

### interpolate_nans_1d -- Gap filling

Fills NaN gaps in a 1-D array by linearly interpolating between the nearest valid values on either side of each gap. Edge NaNs (leading or trailing) are filled with the nearest valid value (constant extrapolation). If all values are NaN, the array is returned unchanged.

The algorithm:

1. Collect indices of all valid (non-NaN) entries.
2. Fill leading NaNs with the first valid value.
3. Fill trailing NaNs with the last valid value.
4. For each pair of adjacent valid indices, linearly interpolate the interior NaN values.

This is useful for cleaning up sounding data where occasional levels have missing reports, or for preprocessing time series before further analysis.

---

## Isentropic Interpolation

Isentropic interpolation maps 3-D atmospheric fields from pressure levels onto constant potential temperature (theta) surfaces. This is covered in detail in the thermodynamics documentation; the summary here focuses on the interpolation mechanics.

The implementation lives in `crates/wx-math/src/thermo.rs` (`isentropic_interpolation`).

### Algorithm

For each grid column (i, j):

1. **Compute theta** at every pressure level using the Poisson equation: `theta = T * (1000 / p)^(R_d/c_p)`.
2. **Find bounding levels** where theta brackets the target isentropic surface.
3. **Newton iteration** to solve for the pressure where theta equals the target:
   - Model temperature as linear in ln(p) between bounding levels: `T = a * ln(p) + b`.
   - The equation `theta_target = T(ln_p) * (1000/p)^kappa` is solved iteratively.
   - Convergence criterion: `|delta_ln_p| < 1e-10` or 50 iterations.
4. **Interpolate auxiliary fields** (humidity, wind, etc.) using theta-sorted linear interpolation between bounding levels.

### Output

The function returns a vector of flattened 2-D fields at each theta level:

- `output[0]`: pressure on the isentropic surface (hPa)
- `output[1]`: temperature on the isentropic surface (K)
- `output[2..]`: additional fields interpolated to the surface

Grid points where no theta crossing exists (target theta is above or below all model levels) are filled with NaN.

### Newton solver details

The Newton solver models temperature as linear in log-pressure space within the bounding layer. This is physically motivated: in a well-mixed layer, temperature varies approximately linearly with the logarithm of pressure (this is the basis of the skew-T diagram). The solver converges in 3--5 iterations for typical atmospheric profiles.

---

## Data Cleaning

Three preprocessing functions remove problematic observations before interpolation. These are critical for producing reliable analysis fields.

### remove_nan_observations

Drops any observation triplet `(lat, lon, value)` where the value is NaN. This prevents NaN values from poisoning weighted averages -- a single NaN observation would make every grid point within its influence radius return NaN.

```python
from metrust.interpolate import remove_nan_observations

clean_lats, clean_lons, clean_vals = remove_nan_observations(lats, lons, values)
```

### remove_observations_below_value

Drops observations where the value is below a specified threshold. This is used to remove physically implausible reports (e.g., negative dewpoint depressions, sub-zero mixing ratios) or to exclude stations reporting below a quality threshold.

```python
from metrust.interpolate import remove_observations_below_value

# Remove stations reporting dewpoint below -40 C (likely erroneous)
lats, lons, vals = remove_observations_below_value(lats, lons, dewpoints, -40.0)
```

### remove_repeat_coordinates

Drops observations with duplicate (lat, lon) coordinates, keeping the first occurrence. Duplicate coordinates cause numerical problems in interpolation: two stations at the same location with different values create an ambiguous weighted average, and coincident points can produce division-by-zero in IDW schemes.

The implementation uses a HashSet keyed on the bit patterns of the lat/lon floats, which provides exact equality semantics (two floats are considered equal only if they have identical bit representations).

```python
from metrust.interpolate import remove_repeat_coordinates

lats, lons, vals = remove_repeat_coordinates(lats, lons, values)
```

### Recommended preprocessing pipeline

For operational mesoanalysis, apply the cleaning steps in this order before interpolation:

```python
from metrust.interpolate import (
    remove_nan_observations,
    remove_observations_below_value,
    remove_repeat_coordinates,
    inverse_distance_to_points,
)

# 1. Remove NaN values
lats, lons, vals = remove_nan_observations(lats, lons, vals)

# 2. Remove physically implausible values
lats, lons, vals = remove_observations_below_value(lats, lons, vals, threshold)

# 3. Remove duplicate station locations
lats, lons, vals = remove_repeat_coordinates(lats, lons, vals)

# 4. Now interpolate
result = inverse_distance_to_points(obs_x, obs_y, vals, grid_x, grid_y,
                                     radius=R, kind=1, kappa=100000.0, gamma=0.2)
```

The order matters: removing NaNs first ensures that `remove_observations_below_value` does not need to handle NaN comparisons. Removing duplicates last ensures that if two co-located stations have different values, the first (presumably earlier/more reliable) report is kept.
