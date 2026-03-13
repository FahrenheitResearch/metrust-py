"""
Verify metrust smoothing and interpolation functions against MetPy reference values.

This script computes reference values using MetPy / NumPy / SciPy so that
the companion Rust integration tests can assert identical (or near-identical)
numerical results.

Usage:
    python verify_smooth_interp.py
"""

import numpy as np
import metpy.calc as mpcalc
import metpy.interpolate as mpinterp
from metpy.units import units
from scipy.ndimage import uniform_filter

# =====================================================================
# Helper: print with high precision for embedding in Rust tests
# =====================================================================
def pv(label, val):
    """Print a scalar value with full precision."""
    print(f"{label} = {val:.15e}")

def pa(label, arr):
    """Print a 1-D array with full precision."""
    vals = ", ".join(f"{v:.15e}" for v in np.asarray(arr).ravel())
    print(f"{label} = [{vals}]")


# =====================================================================
# 1. SMOOTHING TESTS
# =====================================================================
print("=" * 72)
print("SMOOTHING TESTS")
print("=" * 72)

# -- Test grid: 5x5 ramp --
data_5x5 = np.array([[1, 2, 3, 4, 5],
                      [2, 3, 4, 5, 6],
                      [3, 4, 5, 6, 7],
                      [4, 5, 6, 7, 8],
                      [5, 6, 7, 8, 9]], dtype=float)

# -- Test grid: 3x3 with center spike --
data_3x3_spike = np.array([[0, 0, 0],
                            [0, 10, 0],
                            [0, 0, 0]], dtype=float)

# -- Test grid: uniform --
data_uniform = np.ones((5, 5), dtype=float) * 7.0

# -- Test grid: 7x7 noisy (deterministic) --
data_7x7 = np.array([[(j * 7 + i) * 17.3 for i in range(7)] for j in range(7)])

# =====================================================================
# 1a. smooth_n_point (9-point)
# =====================================================================
print("\n--- smooth_n_point (9-point) ---")

# MetPy smooth_n_point with n=9, passes=1
sn9_5x5 = mpcalc.smooth_n_point(data_5x5, 9, passes=1)
print("smooth_n_point(9) on 5x5 ramp:")
pa("  full_grid", sn9_5x5)
pv("  center[2,2]", sn9_5x5[2, 2])

sn9_spike = mpcalc.smooth_n_point(data_3x3_spike, 9, passes=1)
print("smooth_n_point(9) on 3x3 spike:")
pa("  full_grid", sn9_spike)
pv("  center[1,1]", sn9_spike[1, 1])

sn9_uniform = mpcalc.smooth_n_point(data_uniform, 9, passes=1)
pv("smooth_n_point(9) uniform center", sn9_uniform[2, 2])

# =====================================================================
# 1b. smooth_n_point (5-point)
# =====================================================================
print("\n--- smooth_n_point (5-point) ---")
sn5_5x5 = mpcalc.smooth_n_point(data_5x5, 5, passes=1)
print("smooth_n_point(5) on 5x5 ramp:")
pa("  full_grid", sn5_5x5)
pv("  center[2,2]", sn5_5x5[2, 2])

sn5_spike = mpcalc.smooth_n_point(data_3x3_spike, 5, passes=1)
print("smooth_n_point(5) on 3x3 spike:")
pa("  full_grid", sn5_spike)
pv("  center[1,1]", sn5_spike[1, 1])

# =====================================================================
# 1c. smooth_rectangular
# =====================================================================
print("\n--- smooth_rectangular ---")

# MetPy smooth_rectangular takes size as (rows, cols) tuple or int
sr_5x5 = mpcalc.smooth_rectangular(data_5x5, (3, 3), passes=1)
print("smooth_rectangular(3,3) on 5x5 ramp:")
pa("  full_grid", sr_5x5)
pv("  center[2,2]", sr_5x5[2, 2])

sr_spike = mpcalc.smooth_rectangular(data_3x3_spike, (3, 3), passes=1)
print("smooth_rectangular(3,3) on 3x3 spike:")
pa("  full_grid", sr_spike)
pv("  center[1,1]", sr_spike[1, 1])

sr_uniform = mpcalc.smooth_rectangular(data_uniform, (3, 3), passes=1)
pv("smooth_rectangular uniform center", sr_uniform[2, 2])

# Size 5 on 5x5
sr_5x5_big = mpcalc.smooth_rectangular(data_5x5, (5, 5), passes=1)
print("smooth_rectangular(5,5) on 5x5 ramp:")
pa("  full_grid", sr_5x5_big)
pv("  center[2,2]", sr_5x5_big[2, 2])

# =====================================================================
# 1d. smooth_circular
# =====================================================================
print("\n--- smooth_circular ---")

sc_5x5 = mpcalc.smooth_circular(data_5x5, 2, passes=1)
print("smooth_circular(r=2) on 5x5 ramp:")
pa("  full_grid", sc_5x5)
pv("  center[2,2]", sc_5x5[2, 2])

sc_r1 = mpcalc.smooth_circular(data_5x5, 1, passes=1)
print("smooth_circular(r=1) on 5x5 ramp:")
pa("  full_grid", sc_r1)
pv("  center[2,2]", sc_r1[2, 2])

sc_uniform = mpcalc.smooth_circular(data_uniform, 2, passes=1)
pv("smooth_circular uniform center", sc_uniform[2, 2])

# =====================================================================
# 1e. smooth_window (custom kernel)
# =====================================================================
print("\n--- smooth_window ---")

# Uniform 3x3 kernel -- should match smooth_rectangular(3,3)
kernel_uniform = np.ones((3, 3))
sw_uniform = mpcalc.smooth_window(data_5x5, kernel_uniform, passes=1)
print("smooth_window(uniform 3x3) on 5x5 ramp:")
pa("  full_grid", sw_uniform)
pv("  center[2,2]", sw_uniform[2, 2])

# Gaussian-like 3x3 kernel
kernel_gauss = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=float)
sw_gauss = mpcalc.smooth_window(data_5x5, kernel_gauss, passes=1)
print("smooth_window(gaussian-like 3x3) on 5x5 ramp:")
pa("  full_grid", sw_gauss)
pv("  center[2,2]", sw_gauss[2, 2])

# Identity kernel: center-only weight
kernel_identity = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=float)
sw_id = mpcalc.smooth_window(data_5x5, kernel_identity, passes=1)
pv("smooth_window(identity) center", sw_id[2, 2])

# =====================================================================
# 1f. smooth_gaussian
# =====================================================================
print("\n--- smooth_gaussian ---")
# MetPy: sigma = n / (2*pi), truncate = 2*sqrt(2)
# For n=4: sigma = 4/(2*pi) ~ 0.6366
# For n=8: sigma = 8/(2*pi) ~ 1.2732

sg_n4 = mpcalc.smooth_gaussian(data_5x5, 4)
print("smooth_gaussian(n=4) on 5x5 ramp:")
pa("  full_grid", sg_n4)
pv("  center[2,2]", sg_n4[2, 2])

sg_n8 = mpcalc.smooth_gaussian(data_5x5, 8)
print("smooth_gaussian(n=8) on 5x5 ramp:")
pa("  full_grid", sg_n8)
pv("  center[2,2]", sg_n8[2, 2])

sg_uniform = mpcalc.smooth_gaussian(data_uniform, 4)
pv("smooth_gaussian(n=4) uniform center", sg_uniform[2, 2])

# Print the sigma for each n value so Rust test knows what to use
for n in [4, 8]:
    sigma = n / (2 * np.pi)
    truncate = 2 * np.sqrt(2)
    half = int(np.ceil(truncate * sigma))
    print(f"  n={n}: sigma={sigma:.15e}, truncate={truncate:.15e}, half_width={half}")

# =====================================================================
# 2. DERIVATIVE TESTS
# =====================================================================
print("\n" + "=" * 72)
print("DERIVATIVE TESTS")
print("=" * 72)

# =====================================================================
# 2a. first_derivative
# =====================================================================
print("\n--- first_derivative ---")

# Linear field along axis=1 (x): f(i,j) = i+1 for each row
# With delta=100000m (100 km)
dx_m = 100000.0

# Test on the 5x5 ramp (values go 1,2,3,4,5 along columns)
fd_x = mpcalc.first_derivative(data_5x5 * units.K, axis=1, delta=dx_m * units.m)
print("first_derivative(5x5 ramp, axis=1, dx=100km):")
pa("  row_2", fd_x[2, :].magnitude)
pv("  center[2,2]", fd_x[2, 2].magnitude)

fd_y = mpcalc.first_derivative(data_5x5 * units.K, axis=0, delta=dx_m * units.m)
print("first_derivative(5x5 ramp, axis=0, dy=100km):")
pa("  col_2", fd_y[:, 2].magnitude)
pv("  center[2,2]", fd_y[2, 2].magnitude)

# Quadratic: f = i^2, df/dx = 2*i
quad_1d = np.array([0, 1, 4, 9, 16, 25, 36], dtype=float)
quad_2d = np.tile(quad_1d, (3, 1))  # 3x7 grid
fd_quad = mpcalc.first_derivative(quad_2d * units.K, axis=1, delta=1.0 * units.m)
print("first_derivative(quadratic, axis=1, dx=1m):")
pa("  row_0", fd_quad[1, :].magnitude)

# =====================================================================
# 2b. second_derivative
# =====================================================================
print("\n--- second_derivative ---")

sd_x = mpcalc.second_derivative(data_5x5 * units.K, axis=1, delta=dx_m * units.m)
print("second_derivative(5x5 ramp, axis=1, dx=100km):")
pa("  row_2", sd_x[2, :].magnitude)
pv("  center[2,2]", sd_x[2, 2].magnitude)

# Quadratic: f = i^2, d2f/dx2 = 2
sd_quad = mpcalc.second_derivative(quad_2d * units.K, axis=1, delta=1.0 * units.m)
print("second_derivative(quadratic, axis=1, dx=1m):")
pa("  row_1", sd_quad[1, :].magnitude)

# =====================================================================
# 2c. gradient_x / gradient_y (same as first_derivative with axis)
# =====================================================================
print("\n--- gradient (equivalent to first_derivative) ---")

# gradient_x on 5x5 ramp = first_derivative axis=1
# gradient_y on 5x5 ramp = first_derivative axis=0
# These should match what we already computed
pv("gradient_x center (same as fd axis=1)", fd_x[2, 2].magnitude)
pv("gradient_y center (same as fd axis=0)", fd_y[2, 2].magnitude)

# =====================================================================
# 2d. laplacian
# =====================================================================
print("\n--- laplacian ---")

# For the 5x5 ramp (f = i + j), laplacian = 0 since it's linear
lap_ramp = mpcalc.laplacian(data_5x5 * units.K,
                             deltas=[dx_m * units.m, dx_m * units.m])
print("laplacian(5x5 ramp, dx=dy=100km):")
pa("  full_grid", lap_ramp.magnitude)
pv("  center[2,2]", lap_ramp[2, 2].magnitude)

# Quadratic field: f = i^2 + j^2, laplacian = 2 + 2 = 4 / dx^2
quad_5x5 = np.array([[(i**2 + j**2) for i in range(5)] for j in range(5)],
                     dtype=float)
lap_quad = mpcalc.laplacian(quad_5x5 * units.K,
                             deltas=[1.0 * units.m, 1.0 * units.m])
print("laplacian(quadratic i^2+j^2, dx=dy=1m):")
pa("  full_grid", lap_quad.magnitude)
pv("  center[2,2]", lap_quad[2, 2].magnitude)

# =====================================================================
# 2e. lat_lon_grid_deltas
# =====================================================================
print("\n--- lat_lon_grid_deltas ---")

# 3x3 grid at 1-degree spacing near 45N
lons_3x3, lats_3x3 = np.meshgrid(
    np.array([-90.0, -89.0, -88.0]),
    np.array([44.0, 45.0, 46.0])
)
dx_ll, dy_ll = mpcalc.lat_lon_grid_deltas(
    lons_3x3 * units.degrees, lats_3x3 * units.degrees
)
print("lat_lon_grid_deltas (3x3, 1-deg, near 45N):")
print(f"  dx shape: {dx_ll.shape}, dy shape: {dy_ll.shape}")
pa("  dx_row1", dx_ll[1, :].magnitude)
pa("  dy_col1", dy_ll[:, 1].magnitude)
# Note: MetPy returns dx with shape (ny, nx-1) and dy with shape (ny-1, nx)
# while Rust returns same shape as input
# We need to note this difference for the Rust tests

# =====================================================================
# 3. INTERPOLATION TESTS
# =====================================================================
print("\n" + "=" * 72)
print("INTERPOLATION TESTS")
print("=" * 72)

# =====================================================================
# 3a. interpolate_1d
# =====================================================================
print("\n--- interpolate_1d ---")

x_bp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_bp = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
xi = np.array([1.5, 2.5, 3.5])
yi = mpinterp.interpolate_1d(xi, x_bp, y_bp)
print("interpolate_1d (linear, midpoints):")
pa("  result", yi)

# Test clamping / extrapolation
xi_clamp = np.array([0.0, 0.5, 5.5, 6.0])
yi_clamp = mpinterp.interpolate_1d(xi_clamp, x_bp, y_bp)
print("interpolate_1d (extrapolation):")
pa("  result", yi_clamp)

# Non-uniform spacing
x_nu = np.array([0.0, 1.0, 4.0, 5.0])
y_nu = np.array([0.0, 10.0, 40.0, 50.0])
xi_nu = np.array([0.5, 2.0, 4.5])
yi_nu = mpinterp.interpolate_1d(xi_nu, x_nu, y_nu)
print("interpolate_1d (non-uniform):")
pa("  result", yi_nu)

# Exact breakpoint values
xi_exact = np.array([1.0, 3.0, 5.0])
yi_exact = mpinterp.interpolate_1d(xi_exact, x_bp, y_bp)
print("interpolate_1d (at breakpoints):")
pa("  result", yi_exact)

# =====================================================================
# 3b. log_interpolate_1d
# =====================================================================
print("\n--- log_interpolate_1d ---")

# Descending pressure levels (typical met usage)
p_desc = np.array([1000.0, 900.0, 800.0, 700.0, 500.0])
t_desc = np.array([25.0, 20.0, 15.0, 10.0, -5.0])
pi_desc = np.array([950.0, 850.0, 600.0])
ti_desc = mpinterp.log_interpolate_1d(pi_desc, p_desc, t_desc)
print("log_interpolate_1d (descending pressure):")
pa("  result", ti_desc)

# Ascending pressure
p_asc = np.array([500.0, 700.0, 800.0, 900.0, 1000.0])
t_asc = np.array([-5.0, 10.0, 15.0, 20.0, 25.0])
ti_asc = mpinterp.log_interpolate_1d(pi_desc, p_asc, t_asc)
print("log_interpolate_1d (ascending pressure):")
pa("  result", ti_asc)

# Single interpolation point
pi_single = np.array([750.0])
ti_single = mpinterp.log_interpolate_1d(pi_single, p_desc, t_desc)
print("log_interpolate_1d (single point p=750):")
pa("  result", ti_single)

# =====================================================================
# 3c. interpolate_nans_1d
# =====================================================================
print("\n--- interpolate_nans_1d ---")

# Interior NaNs
x_nan1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_nan1 = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
yi_nan1 = mpinterp.interpolate_nans_1d(x_nan1, y_nan1)
print("interpolate_nans_1d (interior gaps):")
pa("  result", yi_nan1)

# Edge NaNs
x_nan2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_nan2 = np.array([np.nan, np.nan, 5.0, 10.0, np.nan])
yi_nan2 = mpinterp.interpolate_nans_1d(x_nan2, y_nan2)
print("interpolate_nans_1d (edge NaNs):")
pa("  result", yi_nan2)

# No NaNs (should return unchanged)
x_nan3 = np.array([0.0, 1.0, 2.0])
y_nan3 = np.array([10.0, 20.0, 30.0])
yi_nan3 = mpinterp.interpolate_nans_1d(x_nan3, y_nan3)
print("interpolate_nans_1d (no NaNs):")
pa("  result", yi_nan3)

# All NaNs -- MetPy raises ValueError, our Rust version leaves unchanged
print("interpolate_nans_1d (all NaN): MetPy raises ValueError (Rust leaves unchanged)")

# =====================================================================
# 3d. interpolate_to_isosurface
# =====================================================================
print("\n--- interpolate_to_isosurface ---")

# NOTE: MetPy's interpolate_to_isosurface has signature:
# interpolate_to_isosurface(data, isosurface, level)
# where:
#   data: 3D array [nz, ny, nx]
#   isosurface: 3D array same shape - the field whose isosurface we seek
#   level: target value

# Simple case: 2x2 grid, 3 levels
# isosurface values: level 0=0, level 1=5, level 2=10 everywhere
# data values: level 0=0, level 1=100, level 2=200
# target = 2.5 => halfway between level 0 and 1 => data = 50
iso_surf = np.array([[[0.0, 0.0], [0.0, 0.0]],
                      [[5.0, 5.0], [5.0, 5.0]],
                      [[10.0, 10.0], [10.0, 10.0]]])
iso_data = np.array([[[0.0, 0.0], [0.0, 0.0]],
                      [[100.0, 100.0], [100.0, 100.0]],
                      [[200.0, 200.0], [200.0, 200.0]]])

try:
    from metpy.interpolate import interpolate_to_isosurface
    iso_result = interpolate_to_isosurface(iso_data, iso_surf, 2.5)
    print("interpolate_to_isosurface (simple, target=2.5):")
    pa("  result", iso_result)
except (ImportError, AttributeError):
    # Try alternative location
    try:
        iso_result = mpcalc.interpolate_to_isosurface(iso_data, iso_surf, 2.5)
        print("interpolate_to_isosurface (simple, target=2.5):")
        pa("  result", iso_result)
    except (ImportError, AttributeError):
        print("  interpolate_to_isosurface not available in this MetPy version")
        # Compute manually: linear interp between levels
        # target=2.5 is between surf[0]=0 and surf[1]=5
        # t = (2.5-0)/(5-0) = 0.5
        # result = data[0] + 0.5*(data[1]-data[0]) = 0 + 0.5*100 = 50
        print("  Manual computation: target=2.5 => result=50.0 everywhere")

# More complex: varying data across grid
iso_surf2 = np.array([[[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]],
                       [[10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0]],
                       [[20.0, 20.0, 20.0],
                        [20.0, 20.0, 20.0]]])
iso_data2 = np.array([[[100.0, 200.0, 300.0],
                        [400.0, 500.0, 600.0]],
                       [[110.0, 210.0, 310.0],
                        [410.0, 510.0, 610.0]],
                       [[120.0, 220.0, 320.0],
                        [420.0, 520.0, 620.0]]])
try:
    iso_result2 = interpolate_to_isosurface(iso_data2, iso_surf2, 5.0)
    print("interpolate_to_isosurface (varying data, target=5.0):")
    pa("  result", iso_result2)
except:
    try:
        iso_result2 = mpcalc.interpolate_to_isosurface(iso_data2, iso_surf2, 5.0)
        print("interpolate_to_isosurface (varying data, target=5.0):")
        pa("  result", iso_result2)
    except:
        # Manual: t = (5-0)/(10-0) = 0.5 for all columns
        # result = data[0] + 0.5*(data[1]-data[0])
        manual = iso_data2[0] + 0.5 * (iso_data2[1] - iso_data2[0])
        print("interpolate_to_isosurface (manual, target=5.0):")
        pa("  result", manual)

# =====================================================================
# 4. ADDITIONAL EDGE CASES
# =====================================================================
print("\n" + "=" * 72)
print("ADDITIONAL EDGE CASES")
print("=" * 72)

# =====================================================================
# 4a. smooth_n_point edge: corner values
# =====================================================================
print("\n--- smooth_n_point corner/edge values ---")

sn9_corners = mpcalc.smooth_n_point(data_5x5, 9, passes=1)
pv("9pt corner[0,0]", sn9_corners[0, 0])
pv("9pt corner[0,4]", sn9_corners[0, 4])
pv("9pt corner[4,0]", sn9_corners[4, 0])
pv("9pt corner[4,4]", sn9_corners[4, 4])
pv("9pt edge[0,2]", sn9_corners[0, 2])
pv("9pt edge[2,0]", sn9_corners[2, 0])

sn5_corners = mpcalc.smooth_n_point(data_5x5, 5, passes=1)
pv("5pt corner[0,0]", sn5_corners[0, 0])
pv("5pt corner[4,4]", sn5_corners[4, 4])
pv("5pt edge[0,2]", sn5_corners[0, 2])

# =====================================================================
# 4b. smooth_rectangular edge: size > grid
# =====================================================================
print("\n--- smooth_rectangular large window ---")
data_small = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=float)
sr_big = mpcalc.smooth_rectangular(data_small, (5, 5), passes=1)
print("smooth_rectangular(5,5) on 3x3:")
pa("  full_grid", sr_big)
pv("  center[1,1]", sr_big[1, 1])

# =====================================================================
# 4c. first_derivative boundary behavior
# =====================================================================
print("\n--- first_derivative boundaries ---")

# 1-D array: [0, 1, 4, 9, 16] = i^2
fd_1d = mpcalc.first_derivative(
    np.array([0, 1, 4, 9, 16], dtype=float) * units.K,
    delta=1.0 * units.m
)
print("first_derivative(i^2, dx=1):")
pa("  result", fd_1d.magnitude)
# Boundary: forward diff at i=0: (1-0)/1 = 1
# Centered at i=1: (4-0)/2 = 2
# Centered at i=2: (9-1)/2 = 4
# Centered at i=3: (16-4)/2 = 6
# Backward at i=4: (16-9)/1 = 7

# =====================================================================
# 4d. second_derivative boundary behavior
# =====================================================================
print("\n--- second_derivative boundaries ---")

sd_1d = mpcalc.second_derivative(
    np.array([0, 1, 4, 9, 16], dtype=float) * units.K,
    delta=1.0 * units.m
)
print("second_derivative(i^2, dx=1):")
pa("  result", sd_1d.magnitude)

# =====================================================================
# 4e. log_interpolate_1d clamping behavior
# =====================================================================
print("\n--- log_interpolate_1d edge cases ---")

# Interpolation at boundary values
pi_boundary = np.array([1000.0, 500.0])
ti_boundary = mpinterp.log_interpolate_1d(pi_boundary, p_desc, t_desc)
print("log_interpolate_1d (at boundaries):")
pa("  result", ti_boundary)

# =====================================================================
# 5. SUMMARY: KEY VALUES FOR RUST TESTS
# =====================================================================
print("\n" + "=" * 72)
print("SUMMARY: KEY REFERENCE VALUES")
print("=" * 72)

print("\n--- smooth_n_point(9) on 5x5 ramp (all 25 values, row-major) ---")
pa("sn9_5x5_flat", sn9_5x5)

print("\n--- smooth_n_point(5) on 5x5 ramp (all 25 values, row-major) ---")
pa("sn5_5x5_flat", sn5_5x5)

print("\n--- smooth_rectangular(3,3) on 5x5 ramp ---")
pa("sr33_5x5_flat", sr_5x5)

print("\n--- smooth_circular(r=2) on 5x5 ramp ---")
pa("sc2_5x5_flat", sc_5x5)

print("\n--- smooth_window(gauss-like 3x3) on 5x5 ramp ---")
pa("sw_gauss_5x5_flat", sw_gauss)

print("\n--- first_derivative(5x5 ramp, axis=x, dx=100km) row 2 ---")
pa("fd_x_row2", fd_x[2, :].magnitude)

print("\n--- first_derivative(5x5 ramp, axis=y, dy=100km) col 2 ---")
pa("fd_y_col2", fd_y[:, 2].magnitude)

print("\n--- interpolate_1d ---")
pa("interp1d_basic", yi)

print("\n--- log_interpolate_1d (descending) ---")
pa("log_interp_desc", ti_desc)

print("\nDone.")
