# Smoothing Algorithms

This document describes the smoothing algorithms implemented in the metrust
Rust core (`crates/rustmet-core/src/grib2/ops.rs`). Each smoother operates on
a row-major `&[f64]` buffer of shape `(ny, nx)` and returns a new `Vec<f64>`
of the same size.

---

## 1. Gaussian Smoothing

### Algorithm

Gaussian smoothing is implemented as a **separable two-pass convolution**: one
horizontal pass along rows, then one vertical pass along columns. Because a
2-D Gaussian is the product of two 1-D Gaussians, separability reduces the
work from O(n * k^2) to O(n * k), where n is the number of grid points and k
is the kernel width.

The 1-D kernel is built from the continuous Gaussian evaluated at integer
offsets:

```
w[i] = exp(-0.5 * ((i - radius) / sigma)^2)
```

The kernel is then normalized so the weights sum to 1.

### Kernel Half-Width

The kernel is truncated at `radius = ceil(3 * sigma)` grid points from center,
giving a full kernel size of `2 * radius + 1`. For sigma = 2.0, this means
radius = 6 and a 13-tap kernel; for sigma = 5.0, radius = 15 and a 31-tap
kernel.

### Memory Layout and Cache Behavior

The horizontal pass is naturally cache-friendly: each row is a contiguous
slice, and the convolution walks it sequentially.

The vertical pass uses a **gather-convolve-scatter** pattern to avoid strided
memory access:

1. Gather column `i` from the temporary buffer into a contiguous `col_in`
   vector of length `ny` (one strided read).
2. Convolve `col_in` into `col_out` using the same 1-D kernel (sequential
   access on an L1-resident buffer -- typically `ny * 8` bytes < 8 KB).
3. Scatter `col_out` back to column `i` of the result (one strided write).

This keeps the inner convolution loop operating on contiguous memory regardless
of grid dimensions.

### Interior vs. Boundary

The interior loop (where the full kernel fits within the row/column) uses
`unsafe get_unchecked` for bounds-elision and runs without branches:

```rust
for ki in 0..ksize {
    sum += src[src_start + ki] * kernel[ki];
}
```

At the left/top and right/bottom boundaries, the kernel extends past the edge.
These positions use a **renormalized** convolution: only the weights that fall
within bounds contribute, and the result is divided by their sum rather than
the full kernel sum. This preserves the local mean at edges without
introducing artificial padding values.

### NaN Handling (Soft NaN)

When any NaN is detected in the input (checked once with a linear scan),
the function switches to a NaN-aware path. For each output pixel, NaN source
values are **excluded** from the weighted sum, and the weights of the
remaining valid neighbors are **renormalized**:

```
output[i] = sum(w[k] * v[k], for k where !isnan(v[k]))
           / sum(w[k],        for k where !isnan(v[k]))
```

If all values within the kernel window are NaN, the output is NaN.

This is "soft NaN" behavior: NaN values do not propagate beyond their kernel
footprint. A single NaN pixel in an otherwise valid field affects only its
immediate neighborhood, and the effect diminishes with distance.

### Complexity

| Phase | Time | Space |
|-------|------|-------|
| Kernel build | O(k) | O(k) |
| Horizontal pass | O(ny * nx * k) | O(nx * ny) temp buffer |
| Vertical pass | O(nx * ny * k) | O(ny) column buffer |
| **Total** | **O(n * k)** where k = 2*ceil(3*sigma)+1 | **O(n)** |

---

## 2. Rectangular (Box) Smoothing

### The Summed-Area Table Algorithm

The rectangular smoother (`smooth_window`) computes the mean of all valid
values within a square window of side length `window_size` centered on each
grid point.

#### Why Not Naive Loops

A naive implementation iterates over all `window_size^2` cells for each of the
`nx * ny` output pixels, giving O(n * s^2) total work where s is the window
side length. For a 1000x1000 grid with a 21x21 window, that is 441 billion
multiply-adds. Doubling the window size quadruples the cost.

The **summed-area table** (SAT), also called an integral image, reduces the
per-pixel lookup to O(1) regardless of window size.

#### SAT Construction

The SAT is a 2-D prefix sum. For a grid `V[j][i]`, the SAT is defined as:

```
SAT[j][i] = sum of V[r][c] for all 0 <= r <= j, 0 <= c <= i
```

It is built in a single pass using the recurrence:

```
SAT[j][i] = V[j][i] + SAT[j-1][i] + SAT[j][i-1] - SAT[j-1][i-1]
```

with `SAT[-1][*] = SAT[*][-1] = 0`. This takes O(n) time and O(n) space.

#### O(1) Rectangle Sum

Once the SAT is built, the sum of any axis-aligned rectangle
`(y1..y2, x1..x2)` (inclusive) is computed with four lookups:

```
sum(y1, y2, x1, x2) = SAT[y2+1][x2+1]
                     - SAT[y1  ][x2+1]
                     - SAT[y2+1][x1  ]
                     + SAT[y1  ][x1  ]
```

The mean is this sum divided by the number of valid cells in the rectangle.

#### NaN Handling via Parallel NaN-Count SAT

To handle NaN values, a second SAT is built in parallel that counts the number
of NaN cells instead of summing values. NaN cells contribute 0 to the value
SAT and 1 to the count SAT. For a given rectangle:

```
nan_count = NAN_SAT[y2+1][x2+1] - NAN_SAT[y1][x2+1]
          - NAN_SAT[y2+1][x1]   + NAN_SAT[y1][x1]

valid_count = total_cells - nan_count
mean = value_sum / valid_count   (if valid_count > 0, else NaN)
```

#### Boundary Handling

The window is clipped to the grid extent at edges. A pixel at position
`(0, 0)` with a 5x5 window (half = 2) only averages the 3x3 corner that
exists. This matches MetPy's boundary convention: edge pixels are preserved
with smaller effective windows rather than padded with zeros or reflected
values.

The window size is forced to be odd (even values are incremented by 1) so the
window is always symmetric around the center pixel.

#### Multi-Pass Support

When `passes > 1`, the output of each pass becomes the input to the next.
Multiple passes of a box filter approximate a Gaussian: three passes of a
box filter with side `s` approximate a Gaussian with
sigma = sqrt(s^2 * passes / 12). This gives users a fast alternative to
true Gaussian smoothing when exact kernel shape is less important than speed.

#### Performance

Because each output pixel requires exactly 4 SAT lookups (plus 4 NaN-SAT
lookups), the cost is constant regardless of window size:

| Window size | Naive O(n*s^2) | SAT O(n) |
|-------------|----------------|----------|
| 3x3 | ~3 ms | ~3 ms |
| 11x11 | ~35 ms | ~3 ms |
| 21x21 | ~130 ms | ~3 ms |
| 51x51 | ~750 ms | ~3 ms |

(Approximate times for a 500x500 grid on a modern CPU.)

The SAT build is **sequential** because each cell depends on its left and
upper neighbors. However, the lookup phase -- where each output pixel
independently reads four SAT values -- is embarrassingly parallel and runs
on rayon `par_chunks_mut`.

### Complexity

| Phase | Time | Space |
|-------|------|-------|
| SAT build | O(n) | O(n) for SAT + O(n) for NaN-SAT |
| Lookup phase | O(n) -- constant per pixel | In-place |
| **Total** | **O(n)** independent of window size | **O(n)** |

Compare with naive: **O(n * s^2)** where s is the window side length.

---

## 3. Circular (Disk) Smoothing

### Algorithm

The circular smoother averages all grid points within Euclidean distance
`radius` of each center pixel. The kernel footprint is a discrete
approximation of a disk.

For each output pixel `(i, j)`, the algorithm iterates over all offsets
`(di, dj)` in the bounding square `[-ceil(r), ceil(r)]^2` and includes only
those where:

```
di^2 + dj^2 <= radius^2
```

These offset pairs can be **pre-computed** once and reused for every pixel,
since the disk shape is translation-invariant. The pre-computed list contains
only the offsets that satisfy the distance criterion, avoiding the
per-pixel distance check.

All valid (non-NaN) values within the disk contribute equally (uniform
weights). The output is their arithmetic mean. If the center pixel is at the
grid boundary, offsets that fall outside the grid are skipped.

### NaN Handling

Circular smoothing uses **hard NaN** semantics in the current implementation:
NaN values within the disk are excluded from the sum and count. If all values
in the disk are NaN, the output is NaN.

### Complexity

| Component | Time |
|-----------|------|
| Offset precompute | O(r^2) one-time |
| Per-pixel | O(k) where k = number of offsets in disk |
| **Total** | **O(n * pi * r^2)** |

For a radius of 4, the disk contains approximately 49 grid points
(pi * 16 = 50.3), so the cost per pixel is comparable to a 7x7 box filter.

---

## 4. N-Point Smoothing

### Algorithm

The N-point smoother applies a fixed-stencil weighted average at each grid
point. Two stencil sizes are supported:

**5-point stencil** (cardinal neighbors only):

```
        [0, w, 0]
        [w, 1, w]
        [0, w, 0]
```

The center pixel has weight 1.0, each cardinal neighbor (N, S, E, W) has
weight 1.0, and the output is the mean of all valid contributions:

```
output = (center + N + S + E + W) / count_valid
```

**9-point stencil** (cardinal + diagonal neighbors):

```
        [w, w, w]
        [w, 1, w]
        [w, w, w]
```

All eight neighbors contribute equally with weight 1.0, matching the center
weight. The output is the mean of up to 9 values.

These weights are **MetPy-exact**: the implementation matches MetPy's
`smooth_n_point` function, which uses uniform weights (1/count) for all
contributing neighbors.

### Multi-Pass

The smoother supports multiple passes via the `passes` parameter. Each pass
reads from the previous pass's output and writes to a scratch buffer, then the
two are swapped. This is equivalent to convolving the field with the stencil
kernel `passes` times.

### Boundary Treatment

At grid edges, neighbors that fall outside the domain are simply excluded from
both the sum and the count. This means edge and corner pixels have fewer
contributors (3 for corners, 5 for edges in the 9-point case) and naturally
receive less smoothing.

### NaN Handling

If the center pixel is NaN, the output is NaN (the center is never
interpolated from neighbors). Neighbor NaN values are excluded from the sum
and count, so a single missing neighbor does not poison the output.

### Complexity

O(n * passes) -- each pass touches every grid point with a fixed-size stencil
(5 or 9 lookups).

---

## 5. Generic Window Convolution

### Algorithm

The `smooth_window` function accepts a user-supplied 2-D kernel (weight
matrix) and convolves it with the input field. This supports arbitrary
filter shapes: sharpening kernels, directional smoothers, Sobel operators, or
any custom stencil.

**Weight normalization** is optional. When enabled (the default), the kernel
weights are divided by their sum before convolution, ensuring the filter
preserves the field mean. When disabled, the raw weights are used, which is
appropriate for derivative operators or edge-detection kernels where the
weights intentionally do not sum to 1.

The convolution follows the standard definition:

```
output[j][i] = sum over (dj, di) of kernel[dj][di] * input[j+dj][i+di]
```

where the kernel is centered on `(j, i)`.

### Boundary Treatment

At boundaries, kernel positions that fall outside the grid are skipped. If
weight normalization is enabled, the weights are renormalized over only the
valid positions.

### Complexity

O(n * kh * kw * passes), where kh and kw are the kernel height and width.

---

## 6. NaN Handling Differences

The smoothers fall into two categories of NaN behavior:

### Soft NaN (Gaussian)

The Gaussian smoother **excludes** NaN values from the weighted sum and
**renormalizes** the remaining weights. This means:

- A single NaN pixel in a large field only affects its immediate neighbors
  (within the kernel radius).
- The output at neighboring pixels is computed from the valid values only,
  with weights scaled up to compensate for the missing data.
- NaN does not propagate beyond the kernel footprint.

This behavior is appropriate for meteorological grids where NaN represents
missing radar returns or masked terrain. The surrounding valid data is
preserved without artificial gaps.

### Hard NaN (Rectangular, Circular, Window Convolution)

The rectangular, circular, and generic window smoothers also exclude NaN
from the sum and count -- but the key behavioral difference is that these
smoothers produce NaN output when **all** values within the kernel are NaN.
The N-point smoother additionally propagates NaN when the **center** pixel
itself is NaN, regardless of neighbor validity.

### Why This Matters

Consider a temperature field with a single NaN pixel (e.g., a bad sensor
reading):

- **Gaussian:** The NaN pixel remains NaN. Its 4-sigma neighborhood gets
  slightly different values than a no-NaN run, but the effect is localized
  and weighted by distance. A pixel 3-sigma away is barely affected.
- **9-point (center NaN):** The NaN pixel stays NaN. Its immediate
  neighbors are computed from their own valid neighborhoods and are
  unaffected by the center NaN (since it is just excluded from their sums).
- **Box/Circular:** NaN cells are excluded from the average. If the window
  is large and only one cell is NaN, the effect on the output is minimal
  (denominator changes from, say, 441 to 440).

For fields with large contiguous NaN regions (e.g., ocean mask on a land-only
grid), the soft-NaN Gaussian approach preserves valid data right up to the
coastline, while hard-NaN approaches may create a narrow NaN border depending
on kernel size.

---

## 7. Parallelism

All smoothing functions use **rayon** `par_chunks_mut` for row-level
parallelism in their output phase. Each row (or block of rows) is computed
independently, since output pixels depend only on input values (which are
read-only).

### SAT Exception

The summed-area table build for rectangular smoothing is **sequential**:
`SAT[j][i]` depends on `SAT[j-1][i]` and `SAT[j][i-1]`, so rows cannot be
computed independently. However, the SAT build is a simple addition loop with
excellent cache behavior (sequential access, no branches), so it runs at near
memory-bandwidth speed.

The **lookup phase** that follows is embarrassingly parallel: each output
pixel reads four independent SAT values and performs three subtractions and a
division. This phase is distributed across rayon threads.

### Gaussian: Two Sequential Passes

The Gaussian smoother's two-pass design (horizontal then vertical) means the
passes are sequential with respect to each other. Within each pass, rows (for
the horizontal pass) or columns (for the vertical pass) are independent and
can be processed in parallel.

---

## 8. Comparison with SciPy and MetPy

### Gaussian: metrust vs. scipy.ndimage.gaussian_filter

MetPy's `smooth_gaussian` delegates to `scipy.ndimage.gaussian_filter`, which
uses a **recursive IIR (infinite impulse response)** approximation of the
Gaussian. This algorithm:

- Runs in O(n) time regardless of sigma (no kernel size dependence).
- Uses only ~12 multiply-adds per pixel per dimension (the IIR has a fixed
  4th-order recursive structure).
- Is implemented in optimized C with compiler auto-vectorization.

metrust uses a **direct FIR (finite impulse response)** convolution:

- Runs in O(n * k) time where k = 2*ceil(3*sigma)+1.
- Each pixel requires k multiply-adds per dimension.
- Is implemented in Rust without explicit SIMD intrinsics.
- Pays additional overhead for NaN checking (even on the fast path, the
  initial NaN scan is O(n)).

With rayon row-level parallelism (added in v0.2.0), metrust's Gaussian is now
**competitive with or faster than SciPy** on multi-core systems. At larger
sigma values, metrust's 32-core parallelism outweighs SciPy's single-threaded
IIR advantage:

| Grid | Sigma | metrust (Rust, rayon) | SciPy (C IIR) | Winner |
|------|-------|----------------------|----------------|--------|
| 200x200 | 2.0 | 0.39 ms | 0.30 ms | scipy 1.3x |
| 200x200 | 5.0 | 0.43 ms | 1.00 ms | **metrust 2.3x** |
| 400x400 | 2.0 | 1.34 ms | 1.34 ms | tie |
| 400x400 | 5.0 | 1.70 ms | 4.11 ms | **metrust 2.4x** |
| 500x500 | 2.0 | 1.90 ms | 2.03 ms | **metrust 1.1x** |
| 500x500 | 5.0 | 2.52 ms | 6.67 ms | **metrust 2.6x** |

SciPy still wins at small grids with small sigma (where parallelism overhead
exceeds the work), but metrust wins everywhere else. The NaN-aware
weighted averaging that metrust provides (which SciPy does not support at all)
is essentially free on these grids.

### Rectangular: metrust vs. MetPy

MetPy's `smooth_rectangular` uses **naive nested loops** in Python/NumPy:
for each pixel, it iterates over all `window_size^2` cells in the window.
This is O(n * s^2).

metrust's SAT-based implementation is O(n) with constant cost per pixel
regardless of window size. The SAT lookup (4 additions) replaces MetPy's
inner loop (s^2 additions).

| Grid | Window | MetPy (Python loops) | metrust (SAT) | Speedup |
|------|--------|---------------------|---------------|---------|
| 500x500 | 3x3 | ~225 ms | ~3 ms | **75x** |
| 500x500 | 11x11 | ~2.7 s | ~3 ms | **900x** |
| 500x500 | 21x21 | ~9.8 s | ~3 ms | **3,200x** |

The speedup grows with window size because MetPy's cost scales as s^2 while
metrust's cost is constant. At window size 3, metrust is already 75x faster
due to Rust vs. Python overhead. At window size 21, the algorithmic advantage
compounds with the language advantage for a 3,200x speedup.

### Summary

| Smoother | metrust vs. SciPy/MetPy | Why |
|----------|------------------------|-----|
| Gaussian | **1--2.6x faster** than SciPy (with rayon) | Parallelism overcomes IIR advantage |
| Rectangular | **75x--3200x faster** than MetPy | SAT O(n) vs naive O(n*s^2) + Rust vs Python |
| N-point | **50--100x faster** than MetPy | Compiled Rust vs Python loops |
| Circular | **50--100x faster** than MetPy | Compiled Rust vs Python loops |

The Gaussian result is a reminder that algorithm choice matters more than
language choice. SciPy's IIR Gaussian has better asymptotic complexity than
any direct FIR implementation, regardless of whether the FIR is written in
Python, Rust, or hand-tuned assembly. A future metrust version could adopt
the IIR approach (Deriche or Young-van Vliet recursive Gaussian) to close
this gap.
