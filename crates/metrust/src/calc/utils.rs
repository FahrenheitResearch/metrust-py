//! Meteorological utility functions.
//!
//! General-purpose helpers for direction conversion, interpolation,
//! curve crossing detection, and nearest-neighbor resampling.

// ─────────────────────────────────────────────
// Direction / angle conversion
// ─────────────────────────────────────────────

/// 16-point compass rose in clockwise order starting from North (0 degrees).
const DIRECTIONS: [&str; 16] = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
];

/// Convert a meteorological angle (degrees clockwise from north) to a
/// 16-point cardinal direction string.
///
/// The angle is normalised to [0, 360) before binning into 22.5-degree
/// sectors.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::angle_to_direction;
/// assert_eq!(angle_to_direction(0.0), "N");
/// assert_eq!(angle_to_direction(90.0), "E");
/// assert_eq!(angle_to_direction(225.0), "SW");
/// assert_eq!(angle_to_direction(359.0), "N");
/// ```
pub fn angle_to_direction(angle: f64) -> &'static str {
    let a = ((angle % 360.0) + 360.0) % 360.0; // normalise to [0, 360)
    let idx = ((a + 11.25) / 22.5) as usize % 16;
    DIRECTIONS[idx]
}

/// Parse a cardinal direction string into degrees (meteorological convention).
///
/// Accepts the 16-point compass rose used by [`angle_to_direction`].
/// Case-insensitive.  Returns `None` for unrecognised strings.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::parse_angle;
/// assert_eq!(parse_angle("N"), Some(0.0));
/// assert_eq!(parse_angle("sw"), Some(225.0));
/// assert_eq!(parse_angle("bogus"), None);
/// ```
pub fn parse_angle(dir: &str) -> Option<f64> {
    let upper = dir.to_uppercase();
    DIRECTIONS.iter().position(|&d| d == upper).map(|i| i as f64 * 22.5)
}

// ─────────────────────────────────────────────
// Interpolation helpers
// ─────────────────────────────────────────────

/// Find the two indices in `values` that bracket `target`.
///
/// Searches for the first pair `(i, i+1)` where `target` lies between
/// `values[i]` and `values[i+1]` (inclusive of endpoints). Works for
/// both monotonically increasing and decreasing sequences.
///
/// Returns `None` if `values` has fewer than two elements or `target` is
/// outside the range of the data.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::find_bounding_indices;
/// let v = vec![1.0, 3.0, 5.0, 7.0, 9.0];
/// assert_eq!(find_bounding_indices(&v, 4.0), Some((1, 2)));
/// assert_eq!(find_bounding_indices(&v, 1.0), Some((0, 1)));
/// assert_eq!(find_bounding_indices(&v, 0.0), None);
/// ```
pub fn find_bounding_indices(values: &[f64], target: f64) -> Option<(usize, usize)> {
    if values.len() < 2 {
        return None;
    }
    for i in 0..values.len() - 1 {
        let (lo, hi) = if values[i] <= values[i + 1] {
            (values[i], values[i + 1])
        } else {
            (values[i + 1], values[i])
        };
        if target >= lo && target <= hi {
            return Some((i, i + 1));
        }
    }
    None
}

/// Find the index nearest to the point where two series cross.
///
/// Given a common x-axis and two y-series, locates where
/// `y1[i] - y2[i]` changes sign and returns the index of the crossing
/// point that is closest to zero difference.
///
/// Returns `None` if no crossing is found or inputs are too short.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::nearest_intersection_idx;
/// let x  = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
/// assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(2));
/// ```
pub fn nearest_intersection_idx(x: &[f64], y1: &[f64], y2: &[f64]) -> Option<usize> {
    let n = x.len().min(y1.len()).min(y2.len());
    if n < 2 {
        return None;
    }

    let diff: Vec<f64> = (0..n).map(|i| y1[i] - y2[i]).collect();

    let mut best_idx: Option<usize> = None;
    let mut best_abs = f64::INFINITY;

    for i in 0..n - 1 {
        // Check for a sign change (or zero crossing)
        if diff[i] * diff[i + 1] <= 0.0 {
            // Pick whichever endpoint is closer to zero
            let (idx, abs_val) = if diff[i].abs() <= diff[i + 1].abs() {
                (i, diff[i].abs())
            } else {
                (i + 1, diff[i + 1].abs())
            };
            if abs_val < best_abs {
                best_abs = abs_val;
                best_idx = Some(idx);
            }
        }
    }

    best_idx
}

// ─────────────────────────────────────────────
// Resampling
// ─────────────────────────────────────────────

/// Nearest-neighbour 1-D resampling.
///
/// For each value in `x`, finds the closest point in `xp` and returns the
/// corresponding value from `fp`. This is the 1-D equivalent of
/// `scipy.interpolate.interp1d(kind='nearest')`.
///
/// `xp` and `fp` must have the same length and `xp` should be sorted in
/// ascending order for correct results.
///
/// # Examples
///
/// ```
/// use metrust::calc::utils::resample_nn_1d;
/// let xp = vec![0.0, 1.0, 2.0, 3.0];
/// let fp = vec![10.0, 20.0, 30.0, 40.0];
/// let x  = vec![0.4, 1.6, 2.9];
/// let result = resample_nn_1d(&x, &xp, &fp);
/// assert_eq!(result, vec![10.0, 30.0, 40.0]);
/// ```
pub fn resample_nn_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let np = xp.len().min(fp.len());
    x.iter()
        .map(|&xi| {
            if np == 0 {
                return f64::NAN;
            }
            let mut best_idx = 0;
            let mut best_dist = (xi - xp[0]).abs();
            for j in 1..np {
                let d = (xi - xp[j]).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
            fp[best_idx]
        })
        .collect()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── angle_to_direction ──

    #[test]
    fn test_cardinal_directions() {
        assert_eq!(angle_to_direction(0.0), "N");
        assert_eq!(angle_to_direction(90.0), "E");
        assert_eq!(angle_to_direction(180.0), "S");
        assert_eq!(angle_to_direction(270.0), "W");
    }

    #[test]
    fn test_intercardinal_directions() {
        assert_eq!(angle_to_direction(45.0), "NE");
        assert_eq!(angle_to_direction(135.0), "SE");
        assert_eq!(angle_to_direction(225.0), "SW");
        assert_eq!(angle_to_direction(315.0), "NW");
    }

    #[test]
    fn test_secondary_intercardinals() {
        assert_eq!(angle_to_direction(22.5), "NNE");
        assert_eq!(angle_to_direction(67.5), "ENE");
        assert_eq!(angle_to_direction(202.5), "SSW");
    }

    #[test]
    fn test_angle_wrapping() {
        assert_eq!(angle_to_direction(360.0), "N");
        assert_eq!(angle_to_direction(720.0), "N");
        assert_eq!(angle_to_direction(-90.0), "W");
    }

    #[test]
    fn test_boundary_angles() {
        // 11.25 is the boundary between N and NNE
        assert_eq!(angle_to_direction(11.24), "N");
        assert_eq!(angle_to_direction(11.26), "NNE");
    }

    // ── parse_angle ──

    #[test]
    fn test_parse_angle_all_directions() {
        for (i, &dir) in DIRECTIONS.iter().enumerate() {
            let expected = i as f64 * 22.5;
            assert_eq!(parse_angle(dir), Some(expected), "failed for {}", dir);
        }
    }

    #[test]
    fn test_parse_angle_case_insensitive() {
        assert_eq!(parse_angle("n"), Some(0.0));
        assert_eq!(parse_angle("Sw"), Some(225.0));
        assert_eq!(parse_angle("nnw"), Some(337.5));
    }

    #[test]
    fn test_parse_angle_invalid() {
        assert_eq!(parse_angle("X"), None);
        assert_eq!(parse_angle(""), None);
        assert_eq!(parse_angle("north"), None);
    }

    #[test]
    fn test_roundtrip_angle_direction() {
        for angle in [0.0, 22.5, 45.0, 90.0, 180.0, 270.0, 315.0] {
            let dir = angle_to_direction(angle);
            let parsed = parse_angle(dir).unwrap();
            assert!(
                (parsed - angle).abs() < 1e-10,
                "roundtrip failed for {angle}: got {dir} -> {parsed}"
            );
        }
    }

    // ── find_bounding_indices ──

    #[test]
    fn test_bounding_indices_basic() {
        let v = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(find_bounding_indices(&v, 4.0), Some((1, 2)));
        assert_eq!(find_bounding_indices(&v, 6.0), Some((2, 3)));
    }

    #[test]
    fn test_bounding_indices_on_point() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 3.0), Some((0, 1)));
    }

    #[test]
    fn test_bounding_indices_endpoints() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 1.0), Some((0, 1)));
        assert_eq!(find_bounding_indices(&v, 5.0), Some((1, 2)));
    }

    #[test]
    fn test_bounding_indices_outside() {
        let v = vec![1.0, 3.0, 5.0];
        assert_eq!(find_bounding_indices(&v, 0.0), None);
        assert_eq!(find_bounding_indices(&v, 6.0), None);
    }

    #[test]
    fn test_bounding_indices_decreasing() {
        let v = vec![9.0, 7.0, 5.0, 3.0, 1.0];
        assert_eq!(find_bounding_indices(&v, 4.0), Some((2, 3)));
    }

    #[test]
    fn test_bounding_indices_short() {
        assert_eq!(find_bounding_indices(&[], 1.0), None);
        assert_eq!(find_bounding_indices(&[5.0], 5.0), None);
    }

    // ── nearest_intersection_idx ──

    #[test]
    fn test_intersection_basic() {
        let x  = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(2));
    }

    #[test]
    fn test_intersection_no_crossing() {
        let x  = vec![0.0, 1.0, 2.0];
        let y1 = vec![5.0, 6.0, 7.0];
        let y2 = vec![1.0, 2.0, 3.0];
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), None);
    }

    #[test]
    fn test_intersection_at_endpoints() {
        let x  = vec![0.0, 1.0, 2.0];
        let y1 = vec![0.0, 1.0, 2.0];
        let y2 = vec![0.0, 2.0, 4.0];
        // diff = [0, -1, -2], crossing at index 0 (diff==0)
        assert_eq!(nearest_intersection_idx(&x, &y1, &y2), Some(0));
    }

    #[test]
    fn test_intersection_short_input() {
        assert_eq!(nearest_intersection_idx(&[1.0], &[1.0], &[1.0]), None);
        let empty: &[f64] = &[];
        assert_eq!(nearest_intersection_idx(empty, empty, empty), None);
    }

    // ── resample_nn_1d ──

    #[test]
    fn test_resample_exact_points() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0];
        let x  = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(resample_nn_1d(&x, &xp, &fp), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_resample_midpoints() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0];
        let x  = vec![0.4, 1.6, 2.9];
        let result = resample_nn_1d(&x, &xp, &fp);
        // 0.4 -> nearest 0.0 -> 10.0
        // 1.6 -> nearest 2.0 -> 30.0
        // 2.9 -> nearest 3.0 -> 40.0
        assert_eq!(result, vec![10.0, 30.0, 40.0]);
    }

    #[test]
    fn test_resample_single_point() {
        let xp = vec![5.0];
        let fp = vec![100.0];
        let x  = vec![0.0, 5.0, 10.0];
        let result = resample_nn_1d(&x, &xp, &fp);
        assert_eq!(result, vec![100.0, 100.0, 100.0]);
    }

    #[test]
    fn test_resample_empty_source() {
        let result = resample_nn_1d(&[1.0, 2.0], &[], &[]);
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
