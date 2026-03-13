"""Performance benchmark: metrust (Rust) vs MetPy (Python)

Compares wall-clock time for key meteorological calculations.
Run after: maturin develop

Usage:
    python tests/benchmark.py              # MetPy only (baseline)
    python tests/benchmark.py --metrust    # MetPy + metrust side-by-side
"""
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bench(name, func, iterations=100):
    """Benchmark a function, return mean time in ms."""
    # Warmup
    func()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    return elapsed


def fmt(ms):
    """Format milliseconds for display."""
    if ms < 0.001:
        return f"{ms * 1000:.2f} us"
    if ms < 1.0:
        return f"{ms:.4f} ms"
    return f"{ms:.2f} ms"


def speedup(t_metpy, t_metrust):
    """Return speedup string."""
    if t_metrust > 0:
        ratio = t_metpy / t_metrust
        return f"{ratio:.1f}x"
    return "inf"


# ---------------------------------------------------------------------------
# Detect mode
# ---------------------------------------------------------------------------

USE_METRUST = "--metrust" in sys.argv

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import interpolate_1d

if USE_METRUST:
    import metrust.calc as mrcalc
    from metrust.interpolate import interpolate_1d as mr_interp_1d

# ---------------------------------------------------------------------------
# Seeds for reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------

print("=" * 72)
print("BENCHMARK: metrust (Rust) vs MetPy (Python)")
print("=" * 72)
if not USE_METRUST:
    print("  Mode: MetPy baseline only  (add --metrust for comparison)")
else:
    print("  Mode: MetPy + metrust side-by-side")
print()

results = []  # (name, t_metpy_ms, t_metrust_ms_or_None)


def record(name, t_metpy, t_metrust=None):
    results.append((name, t_metpy, t_metrust))
    if t_metrust is not None:
        print(f"  {name:40s}  MetPy: {fmt(t_metpy):>12s}  metrust: {fmt(t_metrust):>12s}  -> {speedup(t_metpy, t_metrust):>8s}")
    else:
        print(f"  {name:40s}  MetPy: {fmt(t_metpy):>12s}")


# ===================================================================
# 1. SCALAR THERMODYNAMICS
# ===================================================================

print("-" * 72)
print("SCALAR THERMODYNAMICS")
print("-" * 72)

# potential_temperature
t_mp = bench("MetPy potential_temperature",
    lambda: mpcalc.potential_temperature(850 * units.hPa, 20 * units.degC),
    iterations=1000)

t_mr = None
if USE_METRUST:
    t_mr = bench("metrust potential_temperature",
        lambda: mrcalc.potential_temperature(850.0, 20.0),
        iterations=10000)
record("potential_temperature (scalar)", t_mp, t_mr)

# saturation_vapor_pressure
t_mp = bench("MetPy sat_vp",
    lambda: mpcalc.saturation_vapor_pressure(25 * units.degC),
    iterations=1000)
t_mr = None
if USE_METRUST:
    t_mr = bench("metrust sat_vp",
        lambda: mrcalc.saturation_vapor_pressure(25.0),
        iterations=10000)
record("saturation_vapor_pressure (scalar)", t_mp, t_mr)

# dewpoint_from_relative_humidity
t_mp = bench("MetPy dewpoint_from_rh",
    lambda: mpcalc.dewpoint_from_relative_humidity(25 * units.degC, 60 * units.percent),
    iterations=1000)
t_mr = None
if USE_METRUST:
    t_mr = bench("metrust dewpoint_from_rh",
        lambda: mrcalc.dewpoint_from_relative_humidity(25.0, 60.0),
        iterations=10000)
record("dewpoint_from_rh (scalar)", t_mp, t_mr)

# equivalent_potential_temperature
t_mp = bench("MetPy theta_e",
    lambda: mpcalc.equivalent_potential_temperature(850 * units.hPa, 20 * units.degC, 15 * units.degC),
    iterations=1000)
t_mr = None
if USE_METRUST:
    t_mr = bench("metrust theta_e",
        lambda: mrcalc.equivalent_potential_temperature(850.0, 20.0, 15.0),
        iterations=10000)
record("equivalent_potential_temperature", t_mp, t_mr)


# ===================================================================
# 2. CAPE/CIN (SOUNDING PROFILE)
# ===================================================================

print()
print("-" * 72)
print("CAPE/CIN (SOUNDING PROFILE)")
print("-" * 72)

# Build a realistic 100-level sounding
n = 100
p_snd = np.linspace(1000, 100, n) * units.hPa
t_snd = np.linspace(25, -60, n) * units.degC
td_snd = np.linspace(20, -65, n) * units.degC
pp = mpcalc.parcel_profile(p_snd, t_snd[0], td_snd[0])

t_mp = bench("MetPy CAPE/CIN 100-level",
    lambda: mpcalc.cape_cin(p_snd, t_snd, td_snd, pp),
    iterations=50)

t_mr = None
if USE_METRUST:
    p_raw = np.linspace(1000, 100, n)
    t_raw = np.linspace(25, -60, n)
    td_raw = np.linspace(20, -65, n)
    h_raw = np.linspace(0, 16000, n)
    t_mr = bench("metrust CAPE/CIN 100-level",
        lambda: mrcalc.cape_cin(p_raw, t_raw, td_raw, h_raw,
                                1000.0, 25.0, 20.0, "sb", 100.0, 300.0, None),
        iterations=100)
record("cape_cin (100 levels)", t_mp, t_mr)


# ===================================================================
# 3. GRID KINEMATICS
# ===================================================================

print()
print("-" * 72)
print("GRID KINEMATICS")
print("-" * 72)

for (nx, ny, iters) in [(100, 100, 100), (250, 250, 50), (500, 500, 20)]:
    tag = f"{nx}x{ny}"
    u_grid = np.random.randn(ny, nx) * units('m/s')
    v_grid = np.random.randn(ny, nx) * units('m/s')
    dx_q = 10000 * units.m
    dy_q = 10000 * units.m

    # Divergence
    t_mp = bench(f"MetPy divergence {tag}",
        lambda: mpcalc.divergence(u_grid, v_grid, dx=dx_q, dy=dy_q),
        iterations=iters)
    t_mr = None
    if USE_METRUST:
        u_flat = u_grid.magnitude.ravel()
        v_flat = v_grid.magnitude.ravel()
        t_mr = bench(f"metrust divergence {tag}",
            lambda: mrcalc.divergence(u_flat, v_flat, nx, ny, 10000.0, 10000.0),
            iterations=iters)
    record(f"divergence ({tag})", t_mp, t_mr)

    # Vorticity
    t_mp = bench(f"MetPy vorticity {tag}",
        lambda: mpcalc.vorticity(u_grid, v_grid, dx=dx_q, dy=dy_q),
        iterations=iters)
    t_mr = None
    if USE_METRUST:
        t_mr = bench(f"metrust vorticity {tag}",
            lambda: mrcalc.vorticity(u_flat, v_flat, nx, ny, 10000.0, 10000.0),
            iterations=iters)
    record(f"vorticity ({tag})", t_mp, t_mr)


# ===================================================================
# 4. SMOOTHING
# ===================================================================

print()
print("-" * 72)
print("SMOOTHING")
print("-" * 72)

for (nx, ny, iters) in [(100, 100, 100), (200, 200, 50), (500, 500, 10)]:
    tag = f"{nx}x{ny}"
    data_2d = np.random.randn(ny, nx)

    t_mp = bench(f"MetPy smooth_gaussian {tag}",
        lambda: mpcalc.smooth_gaussian(data_2d, 5),
        iterations=iters)
    t_mr = None
    if USE_METRUST:
        data_flat = data_2d.ravel()
        t_mr = bench(f"metrust smooth_gaussian {tag}",
            lambda: mrcalc.smooth_gaussian(data_flat, nx, ny, 5.0),
            iterations=iters)
    record(f"smooth_gaussian ({tag})", t_mp, t_mr)


# ===================================================================
# 5. INTERPOLATION
# ===================================================================

print()
print("-" * 72)
print("INTERPOLATION")
print("-" * 72)

for n_pts in [100, 1000, 10000]:
    tag = f"{n_pts} pts"
    x_vals = np.sort(np.random.rand(n_pts))
    xp_vals = np.sort(np.random.rand(100))
    fp_vals = np.random.randn(100)

    t_mp = bench(f"MetPy interpolate_1d {tag}",
        lambda: interpolate_1d(x_vals, xp_vals, fp_vals),
        iterations=100)
    t_mr = None
    if USE_METRUST:
        t_mr = bench(f"metrust interpolate_1d {tag}",
            lambda: mr_interp_1d(x_vals, xp_vals, fp_vals),
            iterations=1000)
    record(f"interpolate_1d ({tag})", t_mp, t_mr)


# ===================================================================
# 6. WIND CALCULATIONS
# ===================================================================

print()
print("-" * 72)
print("WIND CALCULATIONS")
print("-" * 72)

# wind_speed - array
for n_wind in [100, 10000]:
    tag = f"{n_wind} pts"
    u_wind = np.random.randn(n_wind) * units('m/s')
    v_wind = np.random.randn(n_wind) * units('m/s')

    t_mp = bench(f"MetPy wind_speed {tag}",
        lambda: mpcalc.wind_speed(u_wind, v_wind),
        iterations=1000)
    t_mr = None
    if USE_METRUST:
        u_raw = u_wind.magnitude
        v_raw = v_wind.magnitude
        t_mr = bench(f"metrust wind_speed {tag}",
            lambda: mrcalc.wind_speed(u_raw, v_raw),
            iterations=1000)
    record(f"wind_speed ({tag})", t_mp, t_mr)


# ===================================================================
# 7. COMPOSITE PARAMETERS
# ===================================================================

print()
print("-" * 72)
print("COMPOSITE PARAMETERS (scalar)")
print("-" * 72)

# STP
t_mp = bench("MetPy STP (manual)",
    lambda: (2000.0 / 1500.0) * ((2000.0 - 800.0) / 1000.0) * (200.0 / 150.0) * min(25.0 / 20.0, 1.5),
    iterations=10000)
t_mr = None
if USE_METRUST:
    t_mr = bench("metrust STP",
        lambda: mrcalc.significant_tornado_parameter(2000.0, 800.0, 200.0, 25.0),
        iterations=10000)
record("significant_tornado_parameter", t_mp, t_mr)


# ===================================================================
# SUMMARY
# ===================================================================

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)

if USE_METRUST:
    print(f"{'Benchmark':40s}  {'MetPy':>12s}  {'metrust':>12s}  {'Speedup':>8s}")
    print("-" * 72)
    for name, t_mp, t_mr in results:
        if t_mr is not None:
            print(f"  {name:40s}  {fmt(t_mp):>12s}  {fmt(t_mr):>12s}  {speedup(t_mp, t_mr):>8s}")
        else:
            print(f"  {name:40s}  {fmt(t_mp):>12s}  {'N/A':>12s}  {'N/A':>8s}")
else:
    print(f"{'Benchmark':40s}  {'MetPy':>12s}")
    print("-" * 72)
    for name, t_mp, _ in results:
        print(f"  {name:40s}  {fmt(t_mp):>12s}")

print()
print("MetPy baseline recorded. Run with --metrust after 'maturin develop' for comparison.")
print("=" * 72)
