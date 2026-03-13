//! Grid smoothing, spatial derivative utilities, and filtering functions.
//!
//! Re-exports gradient and Laplacian operators from `wx_math::dynamics`
//! and `wx_math::gridmath`, plus geospatial derivatives that work on
//! lat/lon grids.
//!
//! Also implements MetPy's smoothing filters: Gaussian, rectangular (box),
//! circular (disk), and N-point smoothers.
//!
//! All grids are flattened row-major: `index = j * nx + i` where `j` is the
//! row (y-index) and `i` is the column (x-index).
//!
//! NaN values are excluded from weighted averages. At grid edges, only the
//! available neighbors are used.

// ── Basic grid derivatives ───────────────────────────────────────────

/// Partial derivative df/dx using centered finite differences
/// (forward/backward at boundaries).
///
/// Input `values` is flattened row-major with shape `(ny, nx)`.
/// `dx` is the grid spacing in meters.
pub use wx_math::dynamics::gradient_x;

/// Partial derivative df/dy using centered finite differences
/// (forward/backward at boundaries).
///
/// Input `values` is flattened row-major with shape `(ny, nx)`.
/// `dy` is the grid spacing in meters.
pub use wx_math::dynamics::gradient_y;

/// Laplacian: `d2f/dx2 + d2f/dy2` using second-order centered differences.
pub use wx_math::dynamics::laplacian;

// ── Generalized derivatives ──────────────────────────────────────────

/// First derivative along a chosen axis (0 = x, 1 = y).
///
/// Uses centered second-order finite differences in the interior with
/// first-order forward/backward at boundaries.
pub use wx_math::gridmath::first_derivative;

/// Second derivative along a chosen axis (0 = x, 1 = y).
pub use wx_math::gridmath::second_derivative;

// ── Geospatial derivatives ───────────────────────────────────────────

/// Compute physical grid spacings `(dx, dy)` in meters from lat/lon arrays
/// using the haversine formula.
pub use wx_math::gridmath::lat_lon_grid_deltas;

/// Gradient on a lat/lon grid: converts the scalar field to `(df/dx, df/dy)`
/// using geospatially-correct spacings.
pub use wx_math::gridmath::geospatial_gradient;

/// Laplacian on a lat/lon grid with geospatially-correct spacings.
pub use wx_math::gridmath::geospatial_laplacian;

// ── Helpers ──────────────────────────────────────────────────────────

/// Row-major index.
#[inline(always)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}

// ─────────────────────────────────────────────────────────────────────
// Gaussian smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a 2D Gaussian smoothing filter (separable implementation).
///
/// The kernel half-width is `ceil(4 * sigma)` grid points, giving a full
/// kernel size of `2 * half + 1`. The filter is applied separably: first
/// along rows, then along columns, for efficiency.
///
/// NaN values in `data` are excluded from the weighted average. If every
/// neighbor within the kernel is NaN, the output is NaN.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of grid points in the x (column) direction.
/// * `ny` - Number of grid points in the y (row) direction.
/// * `sigma` - Standard deviation of the Gaussian kernel in grid-point units.
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `sigma <= 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_gaussian;
///
/// let nx = 5;
/// let ny = 5;
/// let data = vec![0.0; nx * ny];
/// let smoothed = smooth_gaussian(&data, nx, ny, 1.0);
/// assert_eq!(smoothed.len(), nx * ny);
/// ```
pub fn smooth_gaussian(data: &[f64], nx: usize, ny: usize, sigma: f64) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(sigma > 0.0, "sigma must be positive, got {}", sigma);

    let half = (4.0 * sigma).ceil() as usize;
    let kernel_size = 2 * half + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0; kernel_size];
    let two_sigma2 = 2.0 * sigma * sigma;
    for k in 0..kernel_size {
        let d = k as f64 - half as f64;
        kernel[k] = (-d * d / two_sigma2).exp();
    }

    // Pass 1: smooth along x (rows)
    let mut temp = vec![f64::NAN; n];
    for j in 0..ny {
        for i in 0..nx {
            let mut wsum = 0.0;
            let mut vsum = 0.0;
            for k in 0..kernel_size {
                let ki = k as isize - half as isize;
                let ii = i as isize + ki;
                if ii < 0 || ii >= nx as isize {
                    continue;
                }
                let val = data[idx(j, ii as usize, nx)];
                if val.is_nan() {
                    continue;
                }
                let w = kernel[k];
                wsum += w;
                vsum += w * val;
            }
            temp[idx(j, i, nx)] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    }

    // Pass 2: smooth along y (columns)
    let mut out = vec![f64::NAN; n];
    for j in 0..ny {
        for i in 0..nx {
            let mut wsum = 0.0;
            let mut vsum = 0.0;
            for k in 0..kernel_size {
                let kj = k as isize - half as isize;
                let jj = j as isize + kj;
                if jj < 0 || jj >= ny as isize {
                    continue;
                }
                let val = temp[idx(jj as usize, i, nx)];
                if val.is_nan() {
                    continue;
                }
                let w = kernel[k];
                wsum += w;
                vsum += w * val;
            }
            out[idx(j, i, nx)] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// Rectangular (box) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a rectangular (box / uniform) smoothing filter.
///
/// Each output value is the unweighted mean of the `size x size`
/// neighborhood centered on that grid point. NaN values are excluded.
/// At edges, the window is truncated to the available grid points.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `size` - Side length of the square kernel (should be odd; if even, the
///   effective half-width is `size / 2`).
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `size == 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_rectangular;
///
/// let data = vec![1.0; 9];
/// let out = smooth_rectangular(&data, 3, 3, 3);
/// assert!((out[4] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_rectangular(data: &[f64], nx: usize, ny: usize, size: usize) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(size > 0, "kernel size must be > 0");

    let half = size / 2;
    let mut out = vec![f64::NAN; n];

    for j in 0..ny {
        let j_lo = if j >= half { j - half } else { 0 };
        let j_hi = (j + half).min(ny - 1);

        for i in 0..nx {
            let i_lo = if i >= half { i - half } else { 0 };
            let i_hi = (i + half).min(nx - 1);

            let mut sum = 0.0;
            let mut count = 0u32;

            for jj in j_lo..=j_hi {
                for ii in i_lo..=i_hi {
                    let val = data[idx(jj, ii, nx)];
                    if !val.is_nan() {
                        sum += val;
                        count += 1;
                    }
                }
            }

            out[idx(j, i, nx)] = if count > 0 { sum / count as f64 } else { f64::NAN };
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// Circular (disk) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a circular (disk) smoothing filter.
///
/// Each output value is the unweighted mean of all grid points within
/// `radius` grid-point units of the center. The distance check uses
/// Euclidean distance: `sqrt(di^2 + dj^2) <= radius`.
///
/// NaN values are excluded from the average.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `radius` - Radius of the disk kernel in grid-point units.
///
/// # Panics
///
/// Panics if `data.len() != nx * ny` or `radius <= 0`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_circular;
///
/// let data = vec![1.0; 25];
/// let out = smooth_circular(&data, 5, 5, 2.0);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_circular(data: &[f64], nx: usize, ny: usize, radius: f64) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert!(radius > 0.0, "radius must be positive, got {}", radius);

    // Pre-compute the kernel offsets (dj, di) that fall within the radius
    let half = radius.ceil() as isize;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    for dj in -half..=half {
        for di in -half..=half {
            let dist2 = (di * di + dj * dj) as f64;
            if dist2 <= r2 {
                offsets.push((dj, di));
            }
        }
    }

    let mut out = vec![f64::NAN; n];

    for j in 0..ny {
        for i in 0..nx {
            let mut sum = 0.0;
            let mut count = 0u32;

            for &(dj, di) in &offsets {
                let jj = j as isize + dj;
                let ii = i as isize + di;
                if jj < 0 || jj >= ny as isize || ii < 0 || ii >= nx as isize {
                    continue;
                }
                let val = data[idx(jj as usize, ii as usize, nx)];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }

            out[idx(j, i, nx)] = if count > 0 { sum / count as f64 } else { f64::NAN };
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// N-point smoothing (5-point and 9-point)
// ─────────────────────────────────────────────────────────────────────

/// Apply a 5-point or 9-point smoother.
///
/// This replicates MetPy's `smooth_n_point` filter:
///
/// * **n = 5**: The center gets weight 1.0 and the four cardinal neighbors
///   (N, S, E, W) each get weight 0.5. The weights are normalized.
/// * **n = 9**: The center gets weight 1.0 and all eight surrounding
///   neighbors each get weight 0.5. The weights are normalized.
///
/// At grid edges only the available neighbors are included, and the weights
/// are re-normalized. NaN values are excluded from the average.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns.
/// * `ny` - Number of rows.
/// * `n` - Number of points: must be 5 or 9.
///
/// # Panics
///
/// Panics if `n` is not 5 or 9, or if `data.len() != nx * ny`.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_n_point;
///
/// let data = vec![1.0; 25];
/// let out = smooth_n_point(&data, 5, 5, 5);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_n_point(data: &[f64], nx: usize, ny: usize, n: usize) -> Vec<f64> {
    let len = nx * ny;
    assert_eq!(data.len(), len, "data length must equal nx * ny");
    assert!(n == 5 || n == 9, "n must be 5 or 9, got {}", n);

    // Center always has weight 1.0; neighbors have weight 0.5.
    let neighbor_weight = 0.5;
    let center_weight = 1.0;

    let neighbors: &[(isize, isize)] = if n == 5 {
        // Cardinal: N, S, E, W
        &[(-1, 0), (1, 0), (0, 1), (0, -1)]
    } else {
        // All 8 surrounding
        &[
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
    };

    let mut out = vec![f64::NAN; len];

    for j in 0..ny {
        for i in 0..nx {
            let center_val = data[idx(j, i, nx)];
            let mut wsum = 0.0;
            let mut vsum = 0.0;

            // Center contribution
            if !center_val.is_nan() {
                wsum += center_weight;
                vsum += center_weight * center_val;
            }

            // Neighbor contributions
            for &(dj, di) in neighbors {
                let jj = j as isize + dj;
                let ii = i as isize + di;
                if jj < 0 || jj >= ny as isize || ii < 0 || ii >= nx as isize {
                    continue;
                }
                let val = data[idx(jj as usize, ii as usize, nx)];
                if !val.is_nan() {
                    wsum += neighbor_weight;
                    vsum += neighbor_weight * val;
                }
            }

            out[idx(j, i, nx)] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// Generic window (custom kernel) smoothing
// ─────────────────────────────────────────────────────────────────────

/// Apply a generic 2D convolution with a user-supplied kernel.
///
/// This is the equivalent of MetPy's `smooth_window`, which accepts any
/// custom kernel (e.g., a manually constructed Gaussian, Laplacian, or
/// sharpening filter).
///
/// The kernel is a flattened row-major array of size `window_nx * window_ny`.
/// It is applied as a weighted average: at each grid point the kernel is
/// centered on that point, and the output is `sum(w * val) / sum(w)` over
/// the valid (non-NaN) neighbors that fall within the grid.  At edges the
/// kernel is truncated to the available grid points and the weights are
/// re-normalized.
///
/// # Arguments
///
/// * `data` - Input field, flattened row-major, length `nx * ny`.
/// * `nx` - Number of columns in the data grid.
/// * `ny` - Number of rows in the data grid.
/// * `window` - Flattened row-major kernel weights, length
///   `window_nx * window_ny`.
/// * `window_nx` - Number of columns in the kernel.
/// * `window_ny` - Number of rows in the kernel.
///
/// # Panics
///
/// Panics if `data.len() != nx * ny`, `window.len() != window_nx * window_ny`,
/// or if either kernel dimension is zero.
///
/// # Example
///
/// ```
/// use metrust::calc::smooth::smooth_window;
///
/// // 3x3 uniform kernel (equivalent to smooth_rectangular with size 3)
/// let kernel = vec![1.0; 9];
/// let data = vec![1.0; 25];
/// let out = smooth_window(&data, 5, 5, &kernel, 3, 3);
/// assert!((out[12] - 1.0).abs() < 1e-10);
/// ```
pub fn smooth_window(
    data: &[f64],
    nx: usize,
    ny: usize,
    window: &[f64],
    window_nx: usize,
    window_ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(data.len(), n, "data length must equal nx * ny");
    assert_eq!(
        window.len(),
        window_nx * window_ny,
        "window length must equal window_nx * window_ny"
    );
    assert!(window_nx > 0, "window_nx must be > 0");
    assert!(window_ny > 0, "window_ny must be > 0");

    let half_x = window_nx / 2;
    let half_y = window_ny / 2;

    let mut out = vec![f64::NAN; n];

    for j in 0..ny {
        for i in 0..nx {
            let mut wsum = 0.0;
            let mut vsum = 0.0;

            for wj in 0..window_ny {
                let dj = wj as isize - half_y as isize;
                let jj = j as isize + dj;
                if jj < 0 || jj >= ny as isize {
                    continue;
                }

                for wi in 0..window_nx {
                    let di = wi as isize - half_x as isize;
                    let ii = i as isize + di;
                    if ii < 0 || ii >= nx as isize {
                        continue;
                    }

                    let val = data[idx(jj as usize, ii as usize, nx)];
                    if val.is_nan() {
                        continue;
                    }

                    let w = window[wj * window_nx + wi];
                    wsum += w;
                    vsum += w * val;
                }
            }

            out[idx(j, i, nx)] = if wsum > 0.0 { vsum / wsum } else { f64::NAN };
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two f64 values are approximately equal.
    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "approx failed: {} vs {} (diff {}, tol {})",
            a, b, (a - b).abs(), tol
        );
    }

    // =========================================================
    // Gaussian smoothing
    // =========================================================

    #[test]
    fn test_gaussian_constant_field() {
        // Smoothing a constant field should return the same constant.
        let nx = 7;
        let ny = 7;
        let data = vec![42.0; nx * ny];
        let out = smooth_gaussian(&data, nx, ny, 1.5);
        for val in &out {
            approx(*val, 42.0, 1e-10);
        }
    }

    #[test]
    fn test_gaussian_single_spike() {
        // A single spike should be spread out and reduced in amplitude.
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(5, 5, nx)] = 100.0;

        let out = smooth_gaussian(&data, nx, ny, 1.0);

        // Center should be reduced
        assert!(out[idx(5, 5, nx)] < 100.0);
        assert!(out[idx(5, 5, nx)] > 0.0);

        // Neighbors should pick up some of the value
        assert!(out[idx(5, 6, nx)] > 0.0);
        assert!(out[idx(6, 5, nx)] > 0.0);

        // Far-away points should be near zero
        assert!(out[idx(0, 0, nx)] < 0.01);
    }

    #[test]
    fn test_gaussian_symmetry() {
        // A centered spike should produce a symmetric result.
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(4, 4, nx)] = 100.0;

        let out = smooth_gaussian(&data, nx, ny, 1.5);

        // Check 4-fold symmetry around center
        approx(out[idx(3, 4, nx)], out[idx(5, 4, nx)], 1e-10);
        approx(out[idx(4, 3, nx)], out[idx(4, 5, nx)], 1e-10);
        approx(out[idx(3, 3, nx)], out[idx(5, 5, nx)], 1e-10);
        approx(out[idx(3, 5, nx)], out[idx(5, 3, nx)], 1e-10);
    }

    #[test]
    fn test_gaussian_nan_handling() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![10.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_gaussian(&data, nx, ny, 1.0);

        // Output at the NaN point should not be NaN because neighbors are valid
        assert!(!out[idx(2, 2, nx)].is_nan(), "center should not be NaN");
        // Neighbors of the NaN should still be finite
        assert!(!out[idx(2, 3, nx)].is_nan());
        assert!(!out[idx(1, 2, nx)].is_nan());
    }

    #[test]
    fn test_gaussian_all_nan() {
        let data = vec![f64::NAN; 9];
        let out = smooth_gaussian(&data, 3, 3, 1.0);
        for val in &out {
            assert!(val.is_nan(), "all-NaN input should give all-NaN output");
        }
    }

    #[test]
    fn test_gaussian_larger_sigma_more_smoothing() {
        // Larger sigma should produce a smoother (lower peak) result
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(5, 5, nx)] = 100.0;

        let out_narrow = smooth_gaussian(&data, nx, ny, 0.5);
        let out_wide = smooth_gaussian(&data, nx, ny, 2.0);

        assert!(
            out_wide[idx(5, 5, nx)] < out_narrow[idx(5, 5, nx)],
            "wider sigma should give lower peak: {} vs {}",
            out_wide[idx(5, 5, nx)],
            out_narrow[idx(5, 5, nx)]
        );
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn test_gaussian_nonpositive_sigma_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_gaussian(&data, 2, 2, 0.0);
    }

    // =========================================================
    // Rectangular (box) smoothing
    // =========================================================

    #[test]
    fn test_rectangular_constant_field() {
        let data = vec![7.0; 25];
        let out = smooth_rectangular(&data, 5, 5, 3);
        for val in &out {
            approx(*val, 7.0, 1e-10);
        }
    }

    #[test]
    fn test_rectangular_known_average() {
        // 3x3 grid, all ones except center = 10, box size 3
        // Center average = (8 * 1 + 10) / 9 = 18/9 = 2.0
        let nx = 3;
        let ny = 3;
        let mut data = vec![1.0; 9];
        data[idx(1, 1, nx)] = 10.0;

        let out = smooth_rectangular(&data, nx, ny, 3);
        approx(out[idx(1, 1, nx)], 2.0, 1e-10);
    }

    #[test]
    fn test_rectangular_edge_handling() {
        // 3x3 grid of ones, size 3. Corner (0,0) only sees a 2x2 window.
        let data = vec![1.0; 9];
        let out = smooth_rectangular(&data, 3, 3, 3);
        // All should be 1.0 since all values are the same
        for val in &out {
            approx(*val, 1.0, 1e-10);
        }
    }

    #[test]
    fn test_rectangular_size_1() {
        // Box of size 1 = identity filter
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let out = smooth_rectangular(&data, 4, 3, 1);
        for k in 0..12 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_rectangular_nan_handling() {
        let nx = 3;
        let ny = 3;
        let mut data = vec![4.0; 9];
        data[idx(1, 1, nx)] = f64::NAN;

        let out = smooth_rectangular(&data, nx, ny, 3);

        // Center (1,1) with size 3: 8 valid neighbors of value 4 => avg = 4.0
        approx(out[idx(1, 1, nx)], 4.0, 1e-10);
    }

    #[test]
    fn test_rectangular_all_nan() {
        let data = vec![f64::NAN; 9];
        let out = smooth_rectangular(&data, 3, 3, 3);
        for val in &out {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[should_panic(expected = "kernel size must be > 0")]
    fn test_rectangular_zero_size_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_rectangular(&data, 2, 2, 0);
    }

    #[test]
    fn test_rectangular_large_window() {
        // Window larger than grid => each point sees entire grid => all equal to global mean
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = smooth_rectangular(&data, 3, 3, 99);
        let global_mean = 5.0;
        for val in &out {
            approx(*val, global_mean, 1e-10);
        }
    }

    // =========================================================
    // Circular (disk) smoothing
    // =========================================================

    #[test]
    fn test_circular_constant_field() {
        let data = vec![3.14; 25];
        let out = smooth_circular(&data, 5, 5, 2.0);
        for val in &out {
            approx(*val, 3.14, 1e-10);
        }
    }

    #[test]
    fn test_circular_radius_0_5_is_identity() {
        // Radius 0.5 means only the center point (dist=0) is included.
        // dist to cardinal neighbors = 1.0 > 0.5
        let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
        let out = smooth_circular(&data, 3, 3, 0.5);
        for k in 0..9 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_circular_radius_1_includes_cardinals() {
        // Radius 1.0 includes center (dist 0) and 4 cardinal neighbors (dist 1)
        // = 5-point stencil with equal weights
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(2, 2, nx)] = 5.0;

        let out = smooth_circular(&data, nx, ny, 1.0);

        // Center: only the center has a nonzero value among the 5 points
        // avg = 5.0 / 5 = 1.0
        approx(out[idx(2, 2, nx)], 1.0, 1e-10);

        // Cardinal neighbors: they see center (5.0) + themselves (0.0) + 3 other
        // cardinals of the neighbor (0.0) = 5.0 / 5 = 1.0
        approx(out[idx(1, 2, nx)], 1.0, 1e-10);
        approx(out[idx(3, 2, nx)], 1.0, 1e-10);
        approx(out[idx(2, 1, nx)], 1.0, 1e-10);
        approx(out[idx(2, 3, nx)], 1.0, 1e-10);
    }

    #[test]
    fn test_circular_nan_handling() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![2.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_circular(&data, nx, ny, 1.5);

        // Center should not be NaN (neighbors are valid)
        assert!(!out[idx(2, 2, nx)].is_nan());
    }

    #[test]
    fn test_circular_symmetry() {
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        data[idx(4, 4, nx)] = 100.0;

        let out = smooth_circular(&data, nx, ny, 2.5);

        // 4-fold symmetry
        approx(out[idx(3, 4, nx)], out[idx(5, 4, nx)], 1e-10);
        approx(out[idx(4, 3, nx)], out[idx(4, 5, nx)], 1e-10);
    }

    #[test]
    #[should_panic(expected = "radius must be positive")]
    fn test_circular_nonpositive_radius_panics() {
        let data = vec![1.0; 4];
        let _ = smooth_circular(&data, 2, 2, 0.0);
    }

    // =========================================================
    // N-point smoothing
    // =========================================================

    #[test]
    fn test_5point_constant_field() {
        let data = vec![5.0; 25];
        let out = smooth_n_point(&data, 5, 5, 5);
        for val in &out {
            approx(*val, 5.0, 1e-10);
        }
    }

    #[test]
    fn test_9point_constant_field() {
        let data = vec![5.0; 25];
        let out = smooth_n_point(&data, 5, 5, 9);
        for val in &out {
            approx(*val, 5.0, 1e-10);
        }
    }

    #[test]
    fn test_5point_known_result() {
        // 3x3 grid, center = 10, rest = 0
        // At center: w_center=1.0, 4 cardinal neighbors w=0.5 each
        // sum = 1.0*10 + 0.5*0*4 = 10, wsum = 1.0 + 4*0.5 = 3.0
        // result = 10/3
        let nx = 3;
        let ny = 3;
        let mut data = vec![0.0; 9];
        data[idx(1, 1, nx)] = 10.0;

        let out = smooth_n_point(&data, nx, ny, 5);
        approx(out[idx(1, 1, nx)], 10.0 / 3.0, 1e-10);
    }

    #[test]
    fn test_9point_known_result() {
        // 3x3 grid, center = 9, rest = 0
        // At center: w_center=1.0, 8 neighbors w=0.5 each
        // sum = 1.0*9, wsum = 1.0 + 8*0.5 = 5.0
        // result = 9/5 = 1.8
        let nx = 3;
        let ny = 3;
        let mut data = vec![0.0; 9];
        data[idx(1, 1, nx)] = 9.0;

        let out = smooth_n_point(&data, nx, ny, 9);
        approx(out[idx(1, 1, nx)], 9.0 / 5.0, 1e-10);
    }

    #[test]
    fn test_5point_corner() {
        // At corner (0,0) of a 3x3, only 2 cardinal neighbors are available
        // center weight=1.0, right (0,1) w=0.5, down (1,0) w=0.5
        // If all values = 1.0: sum = 1*1 + 0.5*1 + 0.5*1 = 2, wsum = 2 => result = 1.0
        let data = vec![1.0; 9];
        let out = smooth_n_point(&data, 3, 3, 5);
        approx(out[0], 1.0, 1e-10);
    }

    #[test]
    fn test_9point_corner() {
        // At corner (0,0) of a 3x3, only 3 of 8 neighbors exist: (0,1), (1,0), (1,1)
        // center weight=1.0, 3 neighbors each 0.5 => wsum = 1.0 + 3*0.5 = 2.5
        // If all = 2.0: sum = 1*2 + 3*0.5*2 = 2+3 = 5, result = 5/2.5 = 2.0
        let data = vec![2.0; 9];
        let out = smooth_n_point(&data, 3, 3, 9);
        approx(out[0], 2.0, 1e-10);
    }

    #[test]
    fn test_5point_nan_center() {
        // Center is NaN, but neighbors are valid.
        let nx = 3;
        let ny = 3;
        let mut data = vec![4.0; 9];
        data[idx(1, 1, nx)] = f64::NAN;

        let out = smooth_n_point(&data, nx, ny, 5);
        // 4 cardinal neighbors each 4.0, weight 0.5 => sum = 8, wsum = 2
        approx(out[idx(1, 1, nx)], 4.0, 1e-10);
    }

    #[test]
    fn test_9point_nan_handling() {
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut data = vec![6.0; n];
        data[idx(2, 2, nx)] = f64::NAN;

        let out = smooth_n_point(&data, nx, ny, 9);

        // Center NaN but all 8 neighbors valid: each 0.5*6, sum=24, wsum=4 => 6.0
        approx(out[idx(2, 2, nx)], 6.0, 1e-10);

        // Neighbors of center should still produce finite values
        assert!(!out[idx(1, 1, nx)].is_nan());
    }

    #[test]
    fn test_n_point_all_nan() {
        let data = vec![f64::NAN; 9];
        let out5 = smooth_n_point(&data, 3, 3, 5);
        let out9 = smooth_n_point(&data, 3, 3, 9);
        for val in out5.iter().chain(out9.iter()) {
            assert!(val.is_nan());
        }
    }

    #[test]
    #[should_panic(expected = "n must be 5 or 9")]
    fn test_n_point_invalid_n_panics() {
        let data = vec![1.0; 9];
        let _ = smooth_n_point(&data, 3, 3, 7);
    }

    #[test]
    fn test_5point_edge_row() {
        // At edge (0,2) of a 5x3 grid: center + left (0,1) + right (0,3) + below (1,2)
        // = 3 cardinal neighbors available
        // wsum = 1.0 + 3*0.5 = 2.5
        // All values 10 => result = 10
        let data = vec![10.0; 15];
        let out = smooth_n_point(&data, 5, 3, 5);
        approx(out[idx(0, 2, 5)], 10.0, 1e-10);
    }

    #[test]
    fn test_5point_preserves_linear_field() {
        // A linear field f(i,j) = i + j should be preserved by the 5-point filter
        // at interior points (the 5-point stencil is exact for linear fields).
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                data[j * nx + i] = (i + j) as f64;
            }
        }
        let out = smooth_n_point(&data, nx, ny, 5);
        // Interior points
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                approx(out[k], data[k], 1e-10);
            }
        }
    }

    #[test]
    fn test_9point_preserves_linear_field() {
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut data = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                data[j * nx + i] = 2.0 * i as f64 + 3.0 * j as f64;
            }
        }
        let out = smooth_n_point(&data, nx, ny, 9);
        // Interior: the 9-point stencil with equal neighbor weights also preserves
        // linear fields because symmetric neighbors cancel.
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                approx(out[k], data[k], 1e-10);
            }
        }
    }

    // =========================================================
    // Generic window (custom kernel) smoothing
    // =========================================================

    #[test]
    fn test_window_constant_field() {
        // Any kernel on a constant field should return that constant.
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let data = vec![7.0; 25];
        let out = smooth_window(&data, 5, 5, &kernel, 3, 3);
        for val in &out {
            approx(*val, 7.0, 1e-10);
        }
    }

    #[test]
    fn test_window_uniform_kernel_matches_rectangular() {
        // A uniform kernel should produce the same result as smooth_rectangular.
        let nx = 5;
        let ny = 5;
        let data: Vec<f64> = (0..25).map(|k| (k as f64 * 3.7).sin() * 10.0).collect();
        let kernel = vec![1.0; 9]; // 3x3 uniform
        let from_window = smooth_window(&data, nx, ny, &kernel, 3, 3);
        let from_rect = smooth_rectangular(&data, nx, ny, 3);
        for k in 0..25 {
            approx(from_window[k], from_rect[k], 1e-10);
        }
    }

    #[test]
    fn test_window_single_center_weight() {
        // Kernel with weight only at center acts as identity.
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f64> = (0..9).map(|k| k as f64).collect();
        let out = smooth_window(&data, 3, 3, &kernel, 3, 3);
        for k in 0..9 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_window_nan_exclusion() {
        let nx = 3;
        let ny = 3;
        let mut data = vec![4.0; 9];
        data[idx(1, 1, nx)] = f64::NAN;
        let kernel = vec![1.0; 9]; // 3x3 uniform
        let out = smooth_window(&data, nx, ny, &kernel, 3, 3);
        // Center should average the 8 valid neighbors = 4.0
        approx(out[idx(1, 1, nx)], 4.0, 1e-10);
    }

    #[test]
    fn test_window_all_nan() {
        let data = vec![f64::NAN; 9];
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let out = smooth_window(&data, 3, 3, &kernel, 3, 3);
        for val in &out {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_window_1x1_kernel() {
        // 1x1 kernel = identity.
        let kernel = vec![5.0]; // weight value doesn't matter, just 1 element
        let data: Vec<f64> = (0..12).map(|k| k as f64).collect();
        let out = smooth_window(&data, 4, 3, &kernel, 1, 1);
        for k in 0..12 {
            approx(out[k], data[k], 1e-10);
        }
    }

    #[test]
    fn test_window_asymmetric_kernel() {
        // 1x3 horizontal kernel (row-only smoothing)
        let kernel = vec![1.0, 2.0, 1.0]; // window_nx=3, window_ny=1
        let nx = 5;
        let ny = 1;
        let data = vec![0.0, 0.0, 4.0, 0.0, 0.0];
        let out = smooth_window(&data, nx, ny, &kernel, 3, 1);
        // Center (index 2): w = 1*0 + 2*4 + 1*0 = 8, wsum = 4, out = 2.0
        approx(out[2], 2.0, 1e-10);
        // Index 1: w = 1*0 + 2*0 + 1*4 = 4, wsum = 4, out = 1.0
        approx(out[1], 1.0, 1e-10);
        // Index 3: same as index 1 by symmetry
        approx(out[3], 1.0, 1e-10);
    }

    #[test]
    fn test_window_edge_truncation() {
        // 5x5 kernel on a 3x3 grid: edges must be handled by truncation.
        let kernel = vec![1.0; 25];
        let data = vec![1.0; 9];
        let out = smooth_window(&data, 3, 3, &kernel, 5, 5);
        // Every point should still be 1.0
        for val in &out {
            approx(*val, 1.0, 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "window length must equal")]
    fn test_window_mismatched_kernel_panics() {
        let data = vec![1.0; 9];
        let kernel = vec![1.0; 4]; // 4 != 3*3
        let _ = smooth_window(&data, 3, 3, &kernel, 3, 3);
    }

    #[test]
    fn test_window_reduces_variance() {
        // Any reasonable smoothing kernel should reduce variance on noisy data.
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let data: Vec<f64> = (0..n)
            .map(|k| (k as f64 * 17.3).sin() * 100.0)
            .collect();
        let mean = data.iter().sum::<f64>() / n as f64;
        let var_in: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        // Gaussian-like 3x3 kernel
        let kernel = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
        let out = smooth_window(&data, nx, ny, &kernel, 3, 3);
        let m = out.iter().sum::<f64>() / n as f64;
        let var_out: f64 = out.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
        assert!(
            var_out < var_in,
            "smooth_window should reduce variance: {} vs {}",
            var_out, var_in
        );
    }

    // =========================================================
    // Cross-filter consistency checks
    // =========================================================

    #[test]
    fn test_smoothers_reduce_variance() {
        // Any smoother applied to a noisy field should reduce variance.
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        // Deterministic "noisy" field
        let data: Vec<f64> = (0..n)
            .map(|k| (k as f64 * 17.3).sin() * 100.0)
            .collect();

        let mean = data.iter().sum::<f64>() / n as f64;
        let var_in: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        let gauss = smooth_gaussian(&data, nx, ny, 1.0);
        let rect = smooth_rectangular(&data, nx, ny, 3);
        let circ = smooth_circular(&data, nx, ny, 1.5);
        let s5 = smooth_n_point(&data, nx, ny, 5);
        let s9 = smooth_n_point(&data, nx, ny, 9);

        for (name, out) in [
            ("gaussian", &gauss),
            ("rectangular", &rect),
            ("circular", &circ),
            ("5-point", &s5),
            ("9-point", &s9),
        ] {
            let m = out.iter().sum::<f64>() / n as f64;
            let var: f64 = out.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
            assert!(
                var < var_in,
                "{} did not reduce variance: var_in={}, var_out={}",
                name, var_in, var
            );
        }
    }
}
