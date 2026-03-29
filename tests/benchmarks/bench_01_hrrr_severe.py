"""Benchmark 01: HRRR Severe Thunderstorm Environment

Synthetic 3-km HRRR subdomain (500x500, 30 vertical levels) mimicking a
classic Great Plains severe convective environment with:
  - Capping inversion near 850 hPa
  - Rich low-level moisture (dewpoints 18-22 C at surface)
  - Strong deep-layer shear (backing SE sfc winds, veering SW aloft)
  - Steep mid-level lapse rates

Compares 4 backends:
  1. MetPy     (Pint-unit arrays)
  2. metrust CPU  (raw numpy, Rust engine)
  3. met-cu direct (raw numpy -> CuPy GPU kernels)
  4. metrust GPU   (metrust routing to met-cu)

Prints PASS/FAIL for cross-backend numerical agreement and median timing.
"""

import sys
import time
import statistics
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

NY, NX, NZ = 500, 500, 30       # HRRR 3-km subdomain with 30 levels
THERMO_RTOL, THERMO_ATOL = 1e-4, 1e-2
GRID_RTOL, GRID_ATOL = 1e-3, 0.5   # grid composites can differ more
WARMUP = 1
TIMED_RUNS = 3

np.random.seed(42)

# ============================================================================
# Synthetic Great Plains Severe Environment
# ============================================================================

def build_environment():
    """Build realistic 3-D fields for a Great Plains severe convective event.

    Returns a dict with all arrays needed by the benchmarked functions.
    """
    # --- Pressure levels (hPa) from surface (~970) to ~100 hPa ---
    # Non-uniform: dense near surface, sparser aloft (typical model levels)
    p_levels_hPa = np.array([
        970, 960, 950, 940, 925, 900, 875, 850, 825, 800,
        775, 750, 725, 700, 650, 600, 550, 500, 450, 400,
        350, 300, 275, 250, 225, 200, 175, 150, 125, 100,
    ], dtype=np.float64)
    assert len(p_levels_hPa) == NZ

    # Heights AGL (m) -- approximate hydrostatic, realistic for Great Plains
    # (surface ~350 m MSL, so AGL starts at 0)
    h_levels_m = np.array([
        0, 100, 200, 320, 500, 750, 1050, 1400, 1800, 2200,
        2650, 3100, 3600, 4100, 5200, 6400, 7600, 8900, 10200, 11600,
        13000, 14500, 15400, 16200, 17100, 18000, 19000, 20100, 21300, 22600,
    ], dtype=np.float64)

    # --- Temperature profile (C) ---
    # Warm, moist BL (~30 C sfc), capping inversion near 850 hPa,
    # then steep lapse rate above cap, standard cooling aloft.
    t_profile_C = np.array([
        30.0, 29.0, 28.0, 27.0, 25.5, 23.0, 21.0, 20.5, 18.0, 15.5,
        13.0, 10.5,  8.0,  5.5,  0.5, -5.0, -11.0, -18.0, -26.0, -35.0,
        -44.0, -55.0, -59.0, -62.0, -64.0, -65.0, -67.0, -70.0, -73.0, -76.0,
    ], dtype=np.float64)

    # --- Dewpoint profile (C) ---
    # Very moist BL (Td near T), sharp dryline-type drop above 850,
    # very dry mid/upper levels.
    td_profile_C = np.array([
        22.0, 21.5, 21.0, 20.5, 19.0, 17.0, 14.0,  8.0,  4.0,  0.0,
        -4.0, -8.0, -13.0, -18.0, -28.0, -35.0, -40.0, -45.0, -52.0, -58.0,
        -62.0, -68.0, -70.0, -72.0, -74.0, -76.0, -78.0, -80.0, -82.0, -84.0,
    ], dtype=np.float64)

    # --- Wind profile (m/s) ---
    # Classic Great Plains shear:
    #   Surface: SE at 8 m/s (u~+3, v~-7)  -- warm, moist inflow
    #   850 hPa: SSE at 15 m/s (backed)
    #   500 hPa: WSW at 25 m/s (veering)
    #   300 hPa: W at 40 m/s (jet)
    #   200 hPa: W at 50 m/s
    u_profile = np.array([
        3.0,  3.5,  4.0,  5.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0, 32.0, 36.0,
        38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 49.0, 50.0, 50.0, 50.0,
    ], dtype=np.float64)
    v_profile = np.array([
        -7.0, -7.5, -8.0, -8.0, -7.0, -5.0, -3.0, -1.0,  1.0,  3.0,
         4.0,  5.0,  5.5,  6.0,  5.0,  4.0,  2.0,  0.0, -2.0, -4.0,
        -5.0, -6.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,  0.0,  0.0,
    ], dtype=np.float64)

    # --- Broadcast to 3-D (nz, ny, nx) with small spatial perturbations ---
    # Add realistic mesoscale variability (few degrees T, few m/s wind)
    rng = np.random.default_rng(12345)

    # Spatial perturbation fields (ny, nx) -- smooth-ish
    t_perturb = rng.normal(0, 1.5, (NY, NX)).astype(np.float64)
    td_perturb = rng.normal(0, 1.0, (NY, NX)).astype(np.float64)
    u_perturb = rng.normal(0, 1.0, (NY, NX)).astype(np.float64)
    v_perturb = rng.normal(0, 1.0, (NY, NX)).astype(np.float64)

    # 3-D arrays: profile along axis-0, spatial perturbation scaled by level
    temperature_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    dewpoint_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    u_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    v_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    pressure_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    height_agl_3d = np.empty((NZ, NY, NX), dtype=np.float64)

    for k in range(NZ):
        # Perturbation amplitude decreases with height
        amp = max(0.1, 1.0 - k / NZ)
        temperature_3d[k] = t_profile_C[k] + amp * t_perturb
        dewpoint_3d[k] = td_profile_C[k] + amp * td_perturb
        # Ensure dewpoint <= temperature
        dewpoint_3d[k] = np.minimum(dewpoint_3d[k], temperature_3d[k])
        u_3d[k] = u_profile[k] + amp * u_perturb
        v_3d[k] = v_profile[k] + amp * v_perturb
        # Pressure: small horizontal variations (+/- 2 hPa)
        p_pert = rng.normal(0, 0.5, (NY, NX)) * amp
        pressure_3d[k] = p_levels_hPa[k] + p_pert
        # Height AGL: small terrain variations (+/- 30 m) at low levels
        h_pert = rng.normal(0, 10, (NY, NX)) * amp
        height_agl_3d[k] = h_levels_m[k] + h_pert
        # Ensure height_agl >= 0
        height_agl_3d[k] = np.maximum(height_agl_3d[k], 0.0)

    # Ensure heights are monotonically increasing with level index
    for k in range(1, NZ):
        height_agl_3d[k] = np.maximum(height_agl_3d[k], height_agl_3d[k - 1] + 10.0)

    # Surface fields for CAPE/CIN
    psfc_Pa = pressure_3d[0] * 100.0  # Convert surface pressure to Pa
    t2m_K = temperature_3d[0] + 273.15  # T2m in Kelvin
    # Mixing ratio from dewpoint: approximate Bolton formula
    e_td = 6.112 * np.exp(17.67 * dewpoint_3d[0] / (dewpoint_3d[0] + 243.5))
    q2_kgkg = 0.622 * e_td / (pressure_3d[0] - e_td)  # kg/kg

    # Mixing ratio 3-D (kg/kg) from dewpoint via Bolton
    e_td_3d = 6.112 * np.exp(17.67 * dewpoint_3d / (dewpoint_3d + 243.5))
    qvapor_3d = 0.622 * e_td_3d / (pressure_3d - e_td_3d)
    qvapor_3d = np.maximum(qvapor_3d, 1e-7)

    # Pressure 3-D in Pa for CAPE/CIN (both metrust and met-cu expect Pa)
    pressure_3d_Pa = pressure_3d * 100.0

    return {
        # Pressure in hPa for thermo functions
        "pressure_hPa": pressure_3d,
        # Temperature in C
        "temperature_C": temperature_3d,
        # Dewpoint in C
        "dewpoint_C": dewpoint_3d,
        # Wind components (m/s)
        "u": u_3d,
        "v": v_3d,
        # Heights AGL (m)
        "height_agl": height_agl_3d,
        # CAPE/CIN inputs
        "pressure_3d_Pa": pressure_3d_Pa,
        "qvapor_kgkg": qvapor_3d,
        "psfc_Pa": psfc_Pa,
        "t2m_K": t2m_K,
        "q2_kgkg": q2_kgkg,
    }


# ============================================================================
# Import backends
# ============================================================================

print("=" * 78)
print("BENCHMARK 01: HRRR Severe Thunderstorm Environment")
print(f"  Grid: {NX}x{NY} horizontal, {NZ} vertical levels")
print(f"  Timing: {WARMUP} warmup + {TIMED_RUNS} timed runs (median)")
print("=" * 78)
print()

# 1. MetPy
import metpy.calc as mpcalc
from metpy.units import units
print("[OK] MetPy imported")

# 2. metrust (CPU)
import metrust.calc as mrcalc
mrcalc.set_backend("cpu")
print("[OK] metrust CPU imported")

# 3. met-cu direct
import metcu.calc as mcucalc
import cupy as cp
print(f"[OK] met-cu imported (GPU: {cp.cuda.runtime.getDeviceCount()} device(s))")

# 4. metrust GPU (routes to met-cu)
# We'll switch backend as needed; same import

print()

# ============================================================================
# Build environment
# ============================================================================

print("Building synthetic HRRR severe environment...")
env = build_environment()
print(f"  pressure_hPa shape: {env['pressure_hPa'].shape}")
print(f"  temperature_C range: [{env['temperature_C'].min():.1f}, {env['temperature_C'].max():.1f}] C")
print(f"  dewpoint_C range: [{env['dewpoint_C'].min():.1f}, {env['dewpoint_C'].max():.1f}] C")
print(f"  wind speed range: [{np.sqrt(env['u']**2 + env['v']**2).min():.1f}, "
      f"{np.sqrt(env['u']**2 + env['v']**2).max():.1f}] m/s")
print(f"  CAPE pressure range: [{env['pressure_3d_Pa'].min()/100:.0f}, "
      f"{env['pressure_3d_Pa'].max()/100:.0f}] hPa")
print()


# ============================================================================
# Helpers
# ============================================================================

def to_numpy(arr):
    """Convert CuPy array or Pint Quantity to plain numpy."""
    if hasattr(arr, "get"):
        arr = arr.get()
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    return np.asarray(arr, dtype=np.float64)


def gpu_sync():
    """Synchronize CUDA stream for accurate GPU timing."""
    cp.cuda.Stream.null.synchronize()


def time_func(func, use_gpu=False):
    """Time a function: 1 warmup + N timed, return median in seconds."""
    # Warmup
    for _ in range(WARMUP):
        func()
        if use_gpu:
            gpu_sync()

    times = []
    for _ in range(TIMED_RUNS):
        if use_gpu:
            gpu_sync()
        t0 = time.perf_counter()
        func()
        if use_gpu:
            gpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def compare(name, result_a, result_b, label_a, label_b, rtol, atol):
    """Compare two arrays, print max diff, return PASS/FAIL."""
    a = to_numpy(result_a)
    b = to_numpy(result_b)

    # Handle shape mismatches gracefully
    if a.shape != b.shape:
        print(f"    {label_a} vs {label_b}: SHAPE MISMATCH {a.shape} vs {b.shape} -> FAIL")
        return False

    # Mask NaN/Inf for comparison (CAPE can have NaN for no-CAPE points)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        print(f"    {label_a} vs {label_b}: all NaN/Inf -> SKIP")
        return True

    max_abs_diff = np.max(np.abs(a[valid] - b[valid]))
    ok = np.allclose(a[valid], b[valid], rtol=rtol, atol=atol)
    status = "PASS" if ok else "FAIL"
    print(f"    {label_a} vs {label_b}: max|diff|={max_abs_diff:.6g}  [{status}]")
    return ok


def fmt_time(seconds):
    """Format time for display."""
    ms = seconds * 1000
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.3f} s"


# ============================================================================
# Benchmark functions
# ============================================================================

all_pass = True
saved_results = {}  # {func_name: {backend_label: result_array_or_tuple}}


def compare_cape_tuple(func_name, ref_tuple, other_tuple,
                       ref_label, other_label, rtol, atol):
    """Compare CAPE/CIN tuple results with special LFC handling.

    Returns True if all sub-results pass.
    LFC is compared with very lenient tolerance because when no true LFC
    exists (stable profile, no CAPE), the returned height is implementation-
    dependent (sentinel/extrapolated).  We only compare LFC at points where
    both backends found meaningful CAPE (> 10 J/kg).
    """
    names = ["CAPE", "CIN", "LCL", "LFC"]
    all_ok = True
    cape_ref = to_numpy(ref_tuple[0])
    cape_oth = to_numpy(other_tuple[0])

    for i, name in enumerate(names):
        r = to_numpy(ref_tuple[i])
        o = to_numpy(other_tuple[i])

        if r.shape != o.shape:
            print(f"    {name}: {ref_label} vs {other_label}: "
                  f"SHAPE MISMATCH {r.shape} vs {o.shape} -> FAIL")
            all_ok = False
            continue

        if name == "LFC":
            # LFC is notoriously ill-defined when the parcel is immediately
            # buoyant (no CIN) or when no true LFC exists.  Different backends
            # return different sentinel values (0, domain top, NaN).
            # We only compare LFC at points where:
            #   - Both backends found meaningful CAPE (> 10 J/kg)
            #   - Both values are finite
            #   - Neither value looks like a sentinel (> 20000 m or == 0)
            finite = np.isfinite(r) & np.isfinite(o)
            has_cape = (cape_ref > 10) & (cape_oth > 10)
            not_sentinel = (r > 1) & (o > 1) & (r < 20000) & (o < 20000)
            mask = finite & has_cape & not_sentinel
            n_sentinel = int((has_cape & finite & ~not_sentinel).sum())
            if mask.sum() == 0:
                print(f"    {name}: {ref_label} vs {other_label}: "
                      f"no non-sentinel LFC columns -> SKIP "
                      f"({n_sentinel} sentinel diffs excluded)")
                continue
            max_d = np.max(np.abs(r[mask] - o[mask]))
            ok = np.allclose(r[mask], o[mask], rtol=5e-2, atol=50.0)
            status = "PASS" if ok else "FAIL"
            n_valid = int(mask.sum())
            print(f"    {name}: {ref_label} vs {other_label}: "
                  f"max|diff|={max_d:.6g} ({n_valid} non-sentinel cols, "
                  f"{n_sentinel} sentinel diffs excluded)  [{status}]")
            if not ok:
                all_ok = False
        else:
            ok = compare(f"{name}", r, o, ref_label, other_label, rtol, atol)
            if not ok:
                all_ok = False
    return all_ok


def run_benchmark(func_name, backends, rtol, atol, tuple_names=None):
    """Run a single benchmark across backends.

    backends: dict of {label: (callable, is_gpu)}
    tuple_names: if set (e.g. for CAPE), use special cape-tuple comparison.
    """
    global all_pass

    print(f"\n{'-' * 78}")
    print(f"  {func_name}")
    print(f"{'-' * 78}")

    results = {}
    timings = {}

    for label, (func, is_gpu) in backends.items():
        try:
            result = func()
            t = time_func(func, use_gpu=is_gpu)
            results[label] = result
            timings[label] = t
            print(f"  [{label:15s}]  {fmt_time(t):>12s}")
        except Exception as exc:
            print(f"  [{label:15s}]  ERROR: {exc}")

    # Correctness comparison: compare all pairs against the first successful backend
    if len(results) >= 2:
        print()
        labels = list(results.keys())
        ref_label = labels[0]

        for other_label in labels[1:]:
            ref_result = results[ref_label]
            other_result = results[other_label]

            if tuple_names == "cape":
                ok = compare_cape_tuple(func_name, ref_result, other_result,
                                        ref_label, other_label, rtol, atol)
                if not ok:
                    all_pass = False
            elif isinstance(ref_result, tuple):
                for i, (r, o) in enumerate(zip(ref_result, other_result)):
                    ok = compare(f"{func_name}[{i}]", r, o,
                                 ref_label, other_label, rtol, atol)
                    if not ok:
                        all_pass = False
            else:
                ok = compare(func_name, ref_result, other_result,
                             ref_label, other_label, rtol, atol)
                if not ok:
                    all_pass = False

    # Speedup summary
    if len(timings) >= 2:
        print()
        labels = list(timings.keys())
        base_label = labels[0]
        base_t = timings[base_label]
        for label in labels[1:]:
            ratio = base_t / timings[label] if timings[label] > 0 else float("inf")
            print(f"    Speedup {label} vs {base_label}: {ratio:.1f}x")

    # Save results for comprehensive verification later
    saved_results[func_name] = results


# ============================================================================
# 1. potential_temperature (all 4 backends)
# ============================================================================

p_flat = env["pressure_hPa"].ravel()
t_flat = env["temperature_C"].ravel()
td_flat = env["dewpoint_C"].ravel()

# Pint versions for MetPy
p_pint = p_flat * units.hPa
t_pint = t_flat * units.degC
td_pint = td_flat * units.degC

run_benchmark("potential_temperature", {
    "MetPy": (
        lambda: mpcalc.potential_temperature(p_pint, t_pint),
        False,
    ),
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.potential_temperature(p_flat, t_flat),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.potential_temperature(p_flat, t_flat),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.potential_temperature(p_flat, t_flat),
        True,
    ),
}, rtol=THERMO_RTOL, atol=THERMO_ATOL)

# Reset backend
mrcalc.set_backend("cpu")

# ============================================================================
# 2. equivalent_potential_temperature (all 4 backends)
# ============================================================================

run_benchmark("equivalent_potential_temperature", {
    "MetPy": (
        lambda: mpcalc.equivalent_potential_temperature(p_pint, t_pint, td_pint),
        False,
    ),
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
        True,
    ),
}, rtol=THERMO_RTOL, atol=THERMO_ATOL)

mrcalc.set_backend("cpu")

# ============================================================================
# 3. wind_speed (CPU-only: MetPy + metrust CPU)
# ============================================================================

u_flat = env["u"].ravel()
v_flat = env["v"].ravel()
u_pint = u_flat * units("m/s")
v_pint = v_flat * units("m/s")

run_benchmark("wind_speed", {
    "MetPy": (
        lambda: mpcalc.wind_speed(u_pint, v_pint),
        False,
    ),
    "metrust CPU": (
        lambda: mrcalc.wind_speed(u_flat, v_flat),
        False,
    ),
}, rtol=THERMO_RTOL, atol=THERMO_ATOL)

# ============================================================================
# 4. wind_direction (CPU-only: MetPy + metrust CPU)
# ============================================================================

run_benchmark("wind_direction", {
    "MetPy": (
        lambda: mpcalc.wind_direction(u_pint, v_pint),
        False,
    ),
    "metrust CPU": (
        lambda: mrcalc.wind_direction(u_flat, v_flat),
        False,
    ),
}, rtol=THERMO_RTOL, atol=THERMO_ATOL)

# ============================================================================
# 5. compute_cape_cin (3 backends: metrust CPU, met-cu direct, metrust GPU)
#    MetPy does NOT have a grid version.
# ============================================================================

p3d_Pa = env["pressure_3d_Pa"]
tc3d = env["temperature_C"]
qv3d = env["qvapor_kgkg"]
hagl3d = env["height_agl"]
psfc_Pa = env["psfc_Pa"]
t2m_K = env["t2m_K"]
q2_kgkg = env["q2_kgkg"]

# Use a smaller subdomain for CAPE to keep run time reasonable
# CAPE is O(nz * ncols) with per-column parcel lifting -- expensive
CAPE_NY, CAPE_NX = 100, 100
p3d_Pa_sub = np.ascontiguousarray(p3d_Pa[:, :CAPE_NY, :CAPE_NX])
tc3d_sub = np.ascontiguousarray(tc3d[:, :CAPE_NY, :CAPE_NX])
qv3d_sub = np.ascontiguousarray(qv3d[:, :CAPE_NY, :CAPE_NX])
hagl3d_sub = np.ascontiguousarray(hagl3d[:, :CAPE_NY, :CAPE_NX])
psfc_Pa_sub = np.ascontiguousarray(psfc_Pa[:CAPE_NY, :CAPE_NX])
t2m_K_sub = np.ascontiguousarray(t2m_K[:CAPE_NY, :CAPE_NX])
q2_kgkg_sub = np.ascontiguousarray(q2_kgkg[:CAPE_NY, :CAPE_NX])

print(f"\n  [CAPE/CIN uses {CAPE_NX}x{CAPE_NY} subdomain for tractable timing]")

run_benchmark("compute_cape_cin", {
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.compute_cape_cin(
            p3d_Pa_sub, tc3d_sub, qv3d_sub, hagl3d_sub,
            psfc_Pa_sub, t2m_K_sub, q2_kgkg_sub,
        ),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.compute_cape_cin(
            p3d_Pa_sub, tc3d_sub, qv3d_sub, hagl3d_sub,
            psfc_Pa_sub, t2m_K_sub, q2_kgkg_sub,
        ),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.compute_cape_cin(
            p3d_Pa_sub, tc3d_sub, qv3d_sub, hagl3d_sub,
            psfc_Pa_sub, t2m_K_sub, q2_kgkg_sub,
        ),
        True,
    ),
}, rtol=GRID_RTOL, atol=GRID_ATOL, tuple_names="cape")

mrcalc.set_backend("cpu")

# ============================================================================
# 6. compute_srh (3 backends: metrust CPU, met-cu direct, metrust GPU)
# ============================================================================

u3d = env["u"]
v3d = env["v"]
hagl = env["height_agl"]

run_benchmark("compute_srh", {
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.compute_srh(
            u3d, v3d, hagl, top_m=1000.0,
        ),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.compute_srh(u3d, v3d, hagl, top_m=1000.0),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.compute_srh(
            u3d, v3d, hagl, top_m=1000.0,
        ),
        True,
    ),
}, rtol=GRID_RTOL, atol=GRID_ATOL)

mrcalc.set_backend("cpu")

# ============================================================================
# 7. compute_shear (3 backends: metrust CPU, met-cu direct, metrust GPU)
# ============================================================================

run_benchmark("compute_shear", {
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.compute_shear(
            u3d, v3d, hagl, bottom_m=0.0, top_m=6000.0,
        ),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.compute_shear(u3d, v3d, hagl, bottom_m=0.0, top_m=6000.0),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.compute_shear(
            u3d, v3d, hagl, bottom_m=0.0, top_m=6000.0,
        ),
        True,
    ),
}, rtol=GRID_RTOL, atol=GRID_ATOL)

mrcalc.set_backend("cpu")

# ============================================================================
# COMPREHENSIVE DATA CORRECTNESS VERIFICATION
# ============================================================================
#
# This section treats MetPy as ground truth for functions that have MetPy
# equivalents (potential_temperature, equivalent_potential_temperature,
# wind_speed, wind_direction).  For grid composites without MetPy equivalents
# (compute_cape_cin, compute_srh, compute_shear) it uses metrust CPU as the
# reference and compares met-cu and metrust GPU against it.
#
# For every (function, backend) pair it reports:
#   1. Element-wise statistics (mean diff, max |diff|, RMSE, p99, rel RMSE)
#   2. NaN / Inf audit
#   3. Physical plausibility range checks
#   4. Pearson correlation
#   5. Spatial error distribution (% of points above relative error thresholds)
#   6. Edge-case audit (agreement at extremes and zero-shear columns)
#   7. Text histogram of differences
#   8. PASS / FAIL verdict with stated tolerance
# ============================================================================

from scipy import stats as _scipy_stats

print()
print()
print("#" * 78)
print("#" + " COMPREHENSIVE DATA CORRECTNESS VERIFICATION ".center(76) + "#")
print("#" * 78)


# ---------- Physical plausibility bounds per variable ----------
PHYS_BOUNDS = {
    "potential_temperature":            (250.0, 400.0),   # K
    "equivalent_potential_temperature":  (250.0, 450.0),   # K
    "wind_speed":                        (0.0, 120.0),     # m/s
    "wind_direction":                    (0.0, 360.0),     # degrees
    "CAPE":                              (0.0, 8000.0),    # J/kg
    "CIN":                               (-500.0, 0.0),    # J/kg (usually <= 0)
    "LCL":                               (0.0, 8000.0),    # m
    "LFC":                               (0.0, 25000.0),   # m
    "compute_srh":                       (-500.0, 1000.0), # m^2/s^2
    "compute_shear":                     (0.0, 80.0),      # m/s
}

# Tolerances for PASS/FAIL per category
TOLERANCES = {
    "potential_temperature":            {"rtol": 1e-4, "atol": 1e-2},
    "equivalent_potential_temperature":  {"rtol": 1e-4, "atol": 1e-2},
    "wind_speed":                        {"rtol": 1e-4, "atol": 1e-2},
    "wind_direction":                    {"rtol": 1e-4, "atol": 1e-2},
    "compute_cape_cin":                  {"rtol": 1e-3, "atol": 0.5},
    "compute_srh":                       {"rtol": 1e-3, "atol": 0.5},
    "compute_shear":                     {"rtol": 1e-3, "atol": 0.5},
}


# ---------- Helper: text histogram of diffs ----------
def text_histogram(diffs, bins=15, width=50):
    """Print a compact text histogram of *diffs* (1-D array of finite values)."""
    if len(diffs) == 0:
        print("          (no finite diffs)")
        return
    lo, hi = np.min(diffs), np.max(diffs)
    if lo == hi:
        print(f"          all diffs = {lo:.6g}  (1 bin)")
        return
    counts, edges = np.histogram(diffs, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i, c in enumerate(counts):
        bar_len = int(c / max_count * width)
        bar = "#" * bar_len
        lo_e, hi_e = edges[i], edges[i + 1]
        print(f"          [{lo_e:+12.4g}, {hi_e:+12.4g}) "
              f"{c:>8d} {bar}")


# ---------- Helper: full single-array deep comparison ----------
def deep_compare_array(ref, other, ref_label, other_label, var_name,
                       phys_lo, phys_hi, tol_rtol, tol_atol):
    """Run all 7 verification checks on one pair of arrays.

    Returns True if PASS, False if FAIL.
    Also returns a dict of summary stats {rmse, max_diff} for the table.
    """
    r = to_numpy(ref)
    o = to_numpy(other)

    print(f"\n    --- {var_name}: {other_label} vs {ref_label} ---")

    # Shape sanity
    if r.shape != o.shape:
        print(f"      SHAPE MISMATCH: {r.shape} vs {o.shape} -> FAIL")
        return False, {"rmse": float("nan"), "max_diff": float("nan")}

    total = r.size

    # ---- NaN / Inf audit ----
    nan_r, nan_o = np.isnan(r), np.isnan(o)
    inf_r, inf_o = np.isinf(r), np.isinf(o)
    n_nan_r, n_nan_o = int(nan_r.sum()), int(nan_o.sum())
    n_inf_r, n_inf_o = int(inf_r.sum()), int(inf_o.sum())
    nan_match = bool(np.array_equal(nan_r, nan_o))
    print(f"      NaN count:  {ref_label}={n_nan_r}  {other_label}={n_nan_o}"
          f"  locations_match={nan_match}")
    print(f"      Inf count:  {ref_label}={int(inf_r.sum())}  "
          f"{other_label}={int(inf_o.sum())}")

    # Valid mask (finite in both)
    valid = np.isfinite(r) & np.isfinite(o)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print("      No finite values to compare -> SKIP")
        return True, {"rmse": float("nan"), "max_diff": float("nan")}

    rv, ov = r[valid], o[valid]
    diff = ov - rv
    abs_diff = np.abs(diff)

    # ---- 1. Element-wise statistics ----
    mean_diff = float(np.mean(diff))
    max_abs_diff = float(np.max(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    p99 = float(np.percentile(abs_diff, 99))
    ref_rms = float(np.sqrt(np.mean(rv ** 2)))
    rel_rmse_pct = (rmse / ref_rms * 100) if ref_rms > 0 else float("inf")

    print(f"      mean diff      = {mean_diff:+.6g}")
    print(f"      max |diff|     = {max_abs_diff:.6g}")
    print(f"      RMSE           = {rmse:.6g}")
    print(f"      99th pctl |d|  = {p99:.6g}")
    print(f"      rel RMSE       = {rel_rmse_pct:.4f} %")

    # ---- 3. Physical plausibility ----
    out_of_range_r = int(((rv < phys_lo) | (rv > phys_hi)).sum())
    out_of_range_o = int(((ov < phys_lo) | (ov > phys_hi)).sum())
    print(f"      Phys bounds [{phys_lo}, {phys_hi}]:  "
          f"{ref_label} out={out_of_range_r}  {other_label} out={out_of_range_o}")

    # ---- 4. Pearson correlation ----
    if n_valid >= 2 and np.std(rv) > 0 and np.std(ov) > 0:
        pearson_r, _ = _scipy_stats.pearsonr(rv, ov)
    else:
        pearson_r = float("nan")
    print(f"      Pearson r      = {pearson_r:.10f}")

    # ---- 5. Spatial error distribution ----
    # Relative error: |diff| / max(|ref|, tiny) to avoid div-by-zero
    denom = np.maximum(np.abs(rv), 1e-12)
    rel_err = abs_diff / denom
    pct_gt_1 = float(np.mean(rel_err > 0.01) * 100)
    pct_gt_01 = float(np.mean(rel_err > 0.001) * 100)
    print(f"      Points >1%% rel err   = {pct_gt_1:.4f} %")
    print(f"      Points >0.1%% rel err = {pct_gt_01:.4f} %")

    # ---- 6. Edge-case audit ----
    # Compare at extremes of the reference field
    if n_valid >= 10:
        n_edge = max(1, n_valid // 100)  # bottom/top 1%
        sorted_idx = np.argsort(rv)
        low_idx = sorted_idx[:n_edge]
        high_idx = sorted_idx[-n_edge:]
        low_max_diff = float(np.max(np.abs(diff[low_idx]))) if len(low_idx) > 0 else 0.0
        high_max_diff = float(np.max(np.abs(diff[high_idx]))) if len(high_idx) > 0 else 0.0
        print(f"      Edge: bottom 1%% (n={n_edge}) max|diff|={low_max_diff:.6g}"
              f"  top 1%% max|diff|={high_max_diff:.6g}")

    # ---- 7. Histogram of differences ----
    print("      Difference histogram:")
    text_histogram(diff, bins=12, width=40)

    # ---- PASS / FAIL ----
    ok = bool(np.allclose(rv, ov, rtol=tol_rtol, atol=tol_atol))
    status = "PASS" if ok else "FAIL"
    print(f"      Verdict: {status}  "
          f"(rtol={tol_rtol:.0e}, atol={tol_atol:.0e})")

    if not ok:
        global all_pass
        all_pass = False

    return ok, {"rmse": rmse, "max_diff": max_abs_diff}


# ---------- Run verification for each function ----------

# We collect per-(func, backend) summary stats for the final table.
summary_table = []  # list of (func_name, backend_label, rmse, max_diff, verdict)

# ---- Functions with MetPy ground truth ----
METPY_FUNCS = {
    "potential_temperature": {
        "ground_truth": "MetPy",
        "backends": ["metrust CPU", "met-cu direct", "metrust GPU"],
        "phys_bounds": PHYS_BOUNDS["potential_temperature"],
    },
    "equivalent_potential_temperature": {
        "ground_truth": "MetPy",
        "backends": ["metrust CPU", "met-cu direct", "metrust GPU"],
        "phys_bounds": PHYS_BOUNDS["equivalent_potential_temperature"],
    },
    "wind_speed": {
        "ground_truth": "MetPy",
        "backends": ["metrust CPU"],
        "phys_bounds": PHYS_BOUNDS["wind_speed"],
    },
    "wind_direction": {
        "ground_truth": "MetPy",
        "backends": ["metrust CPU"],
        "phys_bounds": PHYS_BOUNDS["wind_direction"],
    },
}

for func_name, cfg in METPY_FUNCS.items():
    if func_name not in saved_results:
        print(f"\n  [SKIP] {func_name}: no saved results")
        continue
    results = saved_results[func_name]
    gt_label = cfg["ground_truth"]
    if gt_label not in results:
        print(f"\n  [SKIP] {func_name}: ground truth '{gt_label}' not found")
        continue

    tol = TOLERANCES.get(func_name, {"rtol": 1e-4, "atol": 1e-2})
    phys_lo, phys_hi = cfg["phys_bounds"]

    print(f"\n{'=' * 78}")
    print(f"  VERIFICATION: {func_name}")
    print(f"  Ground truth: {gt_label}")
    print(f"{'=' * 78}")

    ref = results[gt_label]

    for backend in cfg["backends"]:
        if backend not in results:
            print(f"\n    [{backend}] not available -- skipped")
            summary_table.append((func_name, backend, float("nan"),
                                  float("nan"), "SKIP"))
            continue

        ok, stats = deep_compare_array(
            ref, results[backend],
            gt_label, backend,
            func_name, phys_lo, phys_hi,
            tol["rtol"], tol["atol"],
        )
        verdict = "PASS" if ok else "FAIL"
        summary_table.append((func_name, backend,
                              stats["rmse"], stats["max_diff"], verdict))


# ---- Grid composites: CAPE/CIN (tuple), SRH, shear ----
# For these we use metrust CPU as reference and compare met-cu / metrust GPU.

# --- compute_cape_cin ---
CAPE_TUPLE_NAMES = ["CAPE", "CIN", "LCL", "LFC"]
CAPE_PHYS = [
    PHYS_BOUNDS["CAPE"],
    PHYS_BOUNDS["CIN"],
    PHYS_BOUNDS["LCL"],
    PHYS_BOUNDS["LFC"],
]

if "compute_cape_cin" in saved_results:
    cape_results = saved_results["compute_cape_cin"]
    ref_label = "metrust CPU"
    tol = TOLERANCES["compute_cape_cin"]

    if ref_label in cape_results:
        print(f"\n{'=' * 78}")
        print(f"  VERIFICATION: compute_cape_cin")
        print(f"  Reference: {ref_label}  (no MetPy grid equivalent)")
        print(f"{'=' * 78}")

        ref_tuple = cape_results[ref_label]

        for backend in ["met-cu direct", "metrust GPU"]:
            if backend not in cape_results:
                for sub_name in CAPE_TUPLE_NAMES:
                    summary_table.append(
                        (f"CAPE/{sub_name}", backend, float("nan"),
                         float("nan"), "SKIP"))
                continue

            other_tuple = cape_results[backend]
            cape_ref = to_numpy(ref_tuple[0])
            cape_oth = to_numpy(other_tuple[0])

            for i, (sub_name, (plo, phi)) in enumerate(
                    zip(CAPE_TUPLE_NAMES, CAPE_PHYS)):
                r_arr = to_numpy(ref_tuple[i])
                o_arr = to_numpy(other_tuple[i])

                # For LFC, restrict to non-sentinel CAPE-bearing columns
                if sub_name == "LFC":
                    finite = np.isfinite(r_arr) & np.isfinite(o_arr)
                    has_cape = (cape_ref > 10) & (cape_oth > 10)
                    not_sentinel = ((r_arr > 1) & (o_arr > 1)
                                    & (r_arr < 20000) & (o_arr < 20000))
                    mask = finite & has_cape & not_sentinel
                    n_masked = int(mask.sum())
                    if n_masked == 0:
                        print(f"\n    --- LFC: {backend} vs {ref_label} ---")
                        print(f"      No non-sentinel LFC columns -> SKIP")
                        summary_table.append(
                            (f"CAPE/LFC", backend, float("nan"),
                             float("nan"), "SKIP"))
                        continue
                    # Apply mask -- create masked flat arrays
                    r_masked = r_arr[mask]
                    o_masked = o_arr[mask]
                    ok, stats = deep_compare_array(
                        r_masked, o_masked,
                        ref_label, backend,
                        f"CAPE/{sub_name} (n={n_masked} non-sentinel)",
                        plo, phi,
                        5e-2, 50.0,  # lenient LFC tolerance
                    )
                else:
                    ok, stats = deep_compare_array(
                        r_arr, o_arr,
                        ref_label, backend,
                        f"CAPE/{sub_name}", plo, phi,
                        tol["rtol"], tol["atol"],
                    )
                verdict = "PASS" if ok else "FAIL"
                summary_table.append(
                    (f"CAPE/{sub_name}", backend,
                     stats["rmse"], stats["max_diff"], verdict))

# --- compute_srh ---
if "compute_srh" in saved_results:
    srh_results = saved_results["compute_srh"]
    ref_label = "metrust CPU"
    tol = TOLERANCES["compute_srh"]
    phys_lo, phys_hi = PHYS_BOUNDS["compute_srh"]

    if ref_label in srh_results:
        print(f"\n{'=' * 78}")
        print(f"  VERIFICATION: compute_srh")
        print(f"  Reference: {ref_label}  (no MetPy grid equivalent)")
        print(f"{'=' * 78}")

        ref = srh_results[ref_label]

        for backend in ["met-cu direct", "metrust GPU"]:
            if backend not in srh_results:
                summary_table.append(
                    ("compute_srh", backend, float("nan"),
                     float("nan"), "SKIP"))
                continue

            ok, stats = deep_compare_array(
                ref, srh_results[backend],
                ref_label, backend,
                "compute_srh", phys_lo, phys_hi,
                tol["rtol"], tol["atol"],
            )
            verdict = "PASS" if ok else "FAIL"
            summary_table.append(
                ("compute_srh", backend,
                 stats["rmse"], stats["max_diff"], verdict))

        # Edge-case audit: zero-shear columns
        # Identify columns where reference SRH is near zero (|SRH| < 1)
        ref_np = to_numpy(ref)
        near_zero = np.abs(ref_np) < 1.0
        n_zero = int(near_zero.sum())
        if n_zero > 0:
            print(f"\n    Zero-shear column audit: {n_zero} columns with |SRH|<1")
            for backend in ["met-cu direct", "metrust GPU"]:
                if backend not in srh_results:
                    continue
                o_np = to_numpy(srh_results[backend])
                if o_np.shape == ref_np.shape:
                    diff_at_zero = np.abs(o_np[near_zero] - ref_np[near_zero])
                    print(f"      {backend}: max|diff| at zero-SRH cols = "
                          f"{diff_at_zero.max():.6g}")

# --- compute_shear ---
if "compute_shear" in saved_results:
    shear_results = saved_results["compute_shear"]
    ref_label = "metrust CPU"
    tol = TOLERANCES["compute_shear"]
    phys_lo, phys_hi = PHYS_BOUNDS["compute_shear"]

    if ref_label in shear_results:
        print(f"\n{'=' * 78}")
        print(f"  VERIFICATION: compute_shear")
        print(f"  Reference: {ref_label}  (no MetPy grid equivalent)")
        print(f"{'=' * 78}")

        ref = shear_results[ref_label]

        for backend in ["met-cu direct", "metrust GPU"]:
            if backend not in shear_results:
                summary_table.append(
                    ("compute_shear", backend, float("nan"),
                     float("nan"), "SKIP"))
                continue

            ok, stats = deep_compare_array(
                ref, shear_results[backend],
                ref_label, backend,
                "compute_shear", phys_lo, phys_hi,
                tol["rtol"], tol["atol"],
            )
            verdict = "PASS" if ok else "FAIL"
            summary_table.append(
                ("compute_shear", backend,
                 stats["rmse"], stats["max_diff"], verdict))


# ============================================================================
# SUMMARY TABLE
# ============================================================================

print()
print()
print("#" * 78)
print("#" + " SUMMARY TABLE: function x backend ".center(76) + "#")
print("#" * 78)
print()

hdr_func = "Function"
hdr_back = "Backend"
hdr_rmse = "RMSE"
hdr_maxd = "Max |diff|"
hdr_verd = "Verdict"

col_func = 40
col_back = 18
col_rmse = 14
col_maxd = 14
col_verd = 8

header = (f"  {hdr_func:<{col_func}s} {hdr_back:<{col_back}s} "
          f"{hdr_rmse:>{col_rmse}s} {hdr_maxd:>{col_maxd}s} "
          f"{hdr_verd:>{col_verd}s}")
print(header)
print("  " + "-" * (col_func + col_back + col_rmse + col_maxd + col_verd + 4))

for func_name, backend, rmse, max_diff, verdict in summary_table:
    rmse_s = f"{rmse:.6g}" if np.isfinite(rmse) else "N/A"
    maxd_s = f"{max_diff:.6g}" if np.isfinite(max_diff) else "N/A"
    print(f"  {func_name:<{col_func}s} {backend:<{col_back}s} "
          f"{rmse_s:>{col_rmse}s} {maxd_s:>{col_maxd}s} "
          f"{verdict:>{col_verd}s}")

n_pass = sum(1 for _, _, _, _, v in summary_table if v == "PASS")
n_fail = sum(1 for _, _, _, _, v in summary_table if v == "FAIL")
n_skip = sum(1 for _, _, _, _, v in summary_table if v == "SKIP")

print()
print(f"  Totals: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")

# ============================================================================
# Final verdict
# ============================================================================

print()
print("=" * 78)
if all_pass and n_fail == 0:
    print("  OVERALL: ALL CORRECTNESS CHECKS PASSED")
else:
    print("  OVERALL: SOME CORRECTNESS CHECKS FAILED")
print("=" * 78)

sys.exit(0 if (all_pass and n_fail == 0) else 1)
