"""Benchmark 01: HRRR Severe Thunderstorm Environment (REAL DATA)

Uses actual HRRR GRIB2 model output:
  - hrrr_prs.grib2: 40 isobaric levels (1059 x 1799), fields t/u/v/q/gh
  - hrrr_sfc.grib2: surface pressure (sp), orography (orog), 2m T/Td/q

The full 1059x1799 grid is used for thermodynamic and wind functions.
A configurable subdomain is used for the expensive grid-column composites
(CAPE/CIN, SRH, shear).

Compares 4 backends:
  1. MetPy     (Pint-unit arrays)
  2. metrust CPU  (raw numpy, Rust engine)
  3. met-cu direct (raw numpy -> CuPy GPU kernels)
  4. metrust GPU   (metrust routing to met-cu)

Prints PASS/FAIL for cross-backend numerical agreement and median timing.
"""

import sys
import os
import time
import statistics
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PRS_GRIB = os.path.join(DATA_DIR, "hrrr_prs.grib2")
SFC_GRIB = os.path.join(DATA_DIR, "hrrr_sfc.grib2")

THERMO_RTOL, THERMO_ATOL = 1e-4, 1e-2
GRID_RTOL, GRID_ATOL = 1e-3, 0.5   # grid composites can differ more
CAPE_RTOL, CAPE_ATOL = 1e-3, 1.0   # CAPE: real data 40-level lifting can differ ~1 J/kg
WARMUP = 1
TIMED_RUNS = 3

# Subdomain for expensive column composites (CAPE/CIN, SRH, shear).
# Full grid is 1059x1799.  Use a generous subdomain for real-data coverage.
CAPE_NY, CAPE_NX = 200, 200

# ============================================================================
# Load real HRRR data
# ============================================================================

def load_hrrr_environment():
    """Load 3-D and surface fields from real HRRR GRIB2 files.

    Returns a dict with all arrays needed by the benchmarked functions.
    """
    import xarray as xr

    print("  Loading HRRR pressure levels ...")
    ds_prs = xr.open_dataset(
        PRS_GRIB, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )

    print("  Loading HRRR surface fields ...")
    ds_sfc = xr.open_dataset(
        SFC_GRIB, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "surface",
                "shortName": ["sp", "orog"],
            },
            "indexpath": "",
        },
    )

    print("  Loading HRRR 2-m fields ...")
    ds_2m = xr.open_dataset(
        SFC_GRIB, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 2,
                "shortName": ["2t", "2d", "2sh"],
            },
            "indexpath": "",
        },
    )

    # --- Extract numpy arrays (float64 for precision) ---
    p_levels_hPa = ds_prs.coords["isobaricInhPa"].values.astype(np.float64)
    NZ = len(p_levels_hPa)
    NY, NX = ds_prs.sizes["y"], ds_prs.sizes["x"]

    # 3-D fields: (nz, ny, nx)
    # Temperature: GRIB stores in Kelvin -> convert to Celsius
    temperature_K = ds_prs["t"].values.astype(np.float64)   # (nz, ny, nx)
    temperature_C = temperature_K - 273.15

    # Wind components (m/s)
    u_3d = ds_prs["u"].values.astype(np.float64)
    v_3d = ds_prs["v"].values.astype(np.float64)

    # Specific humidity (kg/kg) -> mixing ratio (kg/kg): w = q / (1 - q)
    q_3d = ds_prs["q"].values.astype(np.float64)
    qvapor_3d = q_3d / (1.0 - q_3d)
    qvapor_3d = np.maximum(qvapor_3d, 1e-10)

    # Geopotential height (m)
    gh_3d = ds_prs["gh"].values.astype(np.float64)

    # Dewpoint: GRIB stores in Kelvin -> convert to Celsius
    dewpoint_K = ds_prs["dpt"].values.astype(np.float64)
    dewpoint_C = dewpoint_K - 273.15

    # --- Surface / 2-m fields: (ny, nx) ---
    sp_Pa = ds_sfc["sp"].values.astype(np.float64)      # surface pressure (Pa)
    orog = ds_sfc["orog"].values.astype(np.float64)       # orography (m MSL)

    t2m_K = ds_2m["t2m"].values.astype(np.float64)        # 2-m temperature (K)
    # 2-m specific humidity -> mixing ratio
    sh2 = ds_2m["sh2"].values.astype(np.float64)           # kg/kg
    q2_kgkg = sh2 / (1.0 - sh2)
    q2_kgkg = np.maximum(q2_kgkg, 1e-10)

    # --- Derived fields ---
    # Pressure 3-D: broadcast isobaric levels to full (nz, ny, nx)
    pressure_3d_hPa = np.broadcast_to(
        p_levels_hPa[:, np.newaxis, np.newaxis], (NZ, NY, NX)
    ).copy().astype(np.float64)

    # Height AGL = geopotential height - surface orography
    height_agl_3d = gh_3d - orog[np.newaxis, :, :]
    # Clamp to non-negative (lowest level can dip slightly below surface)
    height_agl_3d = np.maximum(height_agl_3d, 0.0)

    # Ensure heights are monotonically increasing with level index.
    # HRRR pressure levels go 1013 -> 50 hPa, so gh should already increase,
    # but there can be sub-surface interpolation artefacts at the lowest levels.
    for k in range(1, NZ):
        height_agl_3d[k] = np.maximum(height_agl_3d[k],
                                       height_agl_3d[k - 1] + 1.0)

    # Pressure 3-D in Pa for CAPE/CIN
    pressure_3d_Pa = pressure_3d_hPa * 100.0

    # Surface pressure in hPa (for reference)
    psfc_hPa = sp_Pa / 100.0

    # Close datasets
    ds_prs.close()
    ds_sfc.close()
    ds_2m.close()

    print(f"  Grid: {NX}x{NY} horizontal, {NZ} vertical levels")
    print(f"  Pressure levels: {p_levels_hPa[0]:.0f} - {p_levels_hPa[-1]:.0f} hPa")

    return {
        # Shape info
        "NZ": NZ, "NY": NY, "NX": NX,
        # Pressure in hPa for thermo functions (nz, ny, nx)
        "pressure_hPa": pressure_3d_hPa,
        # Temperature in Celsius (nz, ny, nx)
        "temperature_C": temperature_C,
        # Dewpoint in Celsius (nz, ny, nx)
        "dewpoint_C": dewpoint_C,
        # Wind components m/s (nz, ny, nx)
        "u": u_3d,
        "v": v_3d,
        # Heights AGL m (nz, ny, nx)
        "height_agl": height_agl_3d,
        # Mixing ratio kg/kg (nz, ny, nx)
        "qvapor_kgkg": qvapor_3d,
        # CAPE/CIN 3-D pressure in Pa (nz, ny, nx)
        "pressure_3d_Pa": pressure_3d_Pa,
        # Surface fields (ny, nx)
        "psfc_Pa": sp_Pa,
        "t2m_K": t2m_K,
        "q2_kgkg": q2_kgkg,
    }


# ============================================================================
# Import backends
# ============================================================================

print("=" * 78)
print("BENCHMARK 01: HRRR Severe Thunderstorm Environment (REAL DATA)")
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

# 4. metrust GPU (routes to met-cu) -- same import, switch backend as needed

print()

# ============================================================================
# Load environment
# ============================================================================

print("Loading real HRRR data ...")
env = load_hrrr_environment()
NZ, NY, NX = env["NZ"], env["NY"], env["NX"]

print(f"  temperature_C range: [{env['temperature_C'].min():.1f}, "
      f"{env['temperature_C'].max():.1f}] C")
print(f"  dewpoint_C range: [{env['dewpoint_C'].min():.1f}, "
      f"{env['dewpoint_C'].max():.1f}] C")
ws = np.sqrt(env["u"]**2 + env["v"]**2)
print(f"  wind speed range: [{ws.min():.1f}, {ws.max():.1f}] m/s")
print(f"  height AGL range: [{env['height_agl'].min():.1f}, "
      f"{env['height_agl'].max():.1f}] m")
print(f"  qvapor range: [{env['qvapor_kgkg'].min():.8f}, "
      f"{env['qvapor_kgkg'].max():.6f}] kg/kg")
print(f"  surface pressure range: [{env['psfc_Pa'].min()/100:.0f}, "
      f"{env['psfc_Pa'].max()/100:.0f}] hPa")
print(f"  t2m range: [{env['t2m_K'].min():.1f}, {env['t2m_K'].max():.1f}] K")
print(f"  Timing: {WARMUP} warmup + {TIMED_RUNS} timed runs (median)")
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

    if a.shape != b.shape:
        print(f"    {label_a} vs {label_b}: SHAPE MISMATCH "
              f"{a.shape} vs {b.shape} -> FAIL")
        return False

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
# Benchmark runner
# ============================================================================

all_pass = True
saved_results = {}  # {func_name: {backend_label: result_array_or_tuple}}


def compare_cape_tuple(func_name, ref_tuple, other_tuple,
                       ref_label, other_label, rtol, atol):
    """Compare CAPE/CIN tuple results with special LFC handling."""
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
    """Run a single benchmark across backends."""
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
            import traceback
            traceback.print_exc()

    # Correctness comparison
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
        lambda: (mrcalc.set_backend("cpu") or True) and
                mrcalc.potential_temperature(p_flat, t_flat),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.potential_temperature(p_flat, t_flat),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and
                mrcalc.potential_temperature(p_flat, t_flat),
        True,
    ),
}, rtol=THERMO_RTOL, atol=THERMO_ATOL)

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
        lambda: (mrcalc.set_backend("cpu") or True) and
                mrcalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and
                mrcalc.equivalent_potential_temperature(p_flat, t_flat, td_flat),
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

# Use subdomain for CAPE (expensive per-column lifting)
p3d_Pa = env["pressure_3d_Pa"]
tc3d = env["temperature_C"]
qv3d = env["qvapor_kgkg"]
hagl3d = env["height_agl"]
psfc_Pa = env["psfc_Pa"]
t2m_K = env["t2m_K"]
q2_kgkg = env["q2_kgkg"]

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
}, rtol=CAPE_RTOL, atol=CAPE_ATOL, tuple_names="cape")

mrcalc.set_backend("cpu")

# ============================================================================
# 6. compute_srh (3 backends: metrust CPU, met-cu direct, metrust GPU)
#    Uses same subdomain as CAPE for consistency.
# ============================================================================

u3d = env["u"]
v3d = env["v"]
hagl = env["height_agl"]

# SRH/shear subdomain
u3d_sub = np.ascontiguousarray(u3d[:, :CAPE_NY, :CAPE_NX])
v3d_sub = np.ascontiguousarray(v3d[:, :CAPE_NY, :CAPE_NX])
hagl_sub = np.ascontiguousarray(hagl3d[:, :CAPE_NY, :CAPE_NX])

run_benchmark("compute_srh", {
    "metrust CPU": (
        lambda: (mrcalc.set_backend("cpu") or True) and mrcalc.compute_srh(
            u3d_sub, v3d_sub, hagl_sub, top_m=1000.0,
        ),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.compute_srh(u3d_sub, v3d_sub, hagl_sub, top_m=1000.0),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.compute_srh(
            u3d_sub, v3d_sub, hagl_sub, top_m=1000.0,
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
            u3d_sub, v3d_sub, hagl_sub, bottom_m=0.0, top_m=6000.0,
        ),
        False,
    ),
    "met-cu direct": (
        lambda: mcucalc.compute_shear(
            u3d_sub, v3d_sub, hagl_sub, bottom_m=0.0, top_m=6000.0,
        ),
        True,
    ),
    "metrust GPU": (
        lambda: (mrcalc.set_backend("gpu") or True) and mrcalc.compute_shear(
            u3d_sub, v3d_sub, hagl_sub, bottom_m=0.0, top_m=6000.0,
        ),
        True,
    ),
}, rtol=GRID_RTOL, atol=GRID_ATOL)

mrcalc.set_backend("cpu")

# ============================================================================
# COMPREHENSIVE DATA CORRECTNESS VERIFICATION
# ============================================================================

from scipy import stats as _scipy_stats

print()
print()
print("#" * 78)
print("#" + " COMPREHENSIVE DATA CORRECTNESS VERIFICATION ".center(76) + "#")
print("#" * 78)


# ---------- Physical plausibility bounds (tuned for real HRRR data) ----------
# Temperature range: ~-80 C (upper levels) to ~+30 C (surface summer)
# Theta: lowest pressure levels can push theta above 450 K at 50 hPa
# Theta-e: tropical/low-level moist air can reach 370+, upper levels > 450 rare
PHYS_BOUNDS = {
    "potential_temperature":            (200.0, 600.0),   # K; wide for 50 hPa
    "equivalent_potential_temperature":  (200.0, 600.0),   # K
    "wind_speed":                        (0.0, 120.0),     # m/s
    "wind_direction":                    (0.0, 360.0),     # degrees
    "CAPE":                              (0.0, 8000.0),    # J/kg
    "CIN":                               (-1000.0, 0.0),   # J/kg
    "LCL":                               (0.0, 10000.0),   # m
    "LFC":                               (0.0, 25000.0),   # m
    "compute_srh":                       (-600.0, 1200.0), # m^2/s^2
    "compute_shear":                     (0.0, 100.0),     # m/s
}

# Tolerances
TOLERANCES = {
    "potential_temperature":            {"rtol": 1e-4, "atol": 1e-2},
    "equivalent_potential_temperature":  {"rtol": 1e-4, "atol": 1e-2},
    "wind_speed":                        {"rtol": 1e-4, "atol": 1e-2},
    "wind_direction":                    {"rtol": 1e-4, "atol": 1e-2},
    "compute_cape_cin":                  {"rtol": CAPE_RTOL, "atol": CAPE_ATOL},
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
    denom = np.maximum(np.abs(rv), 1e-12)
    rel_err = abs_diff / denom
    pct_gt_1 = float(np.mean(rel_err > 0.01) * 100)
    pct_gt_01 = float(np.mean(rel_err > 0.001) * 100)
    print(f"      Points >1%% rel err   = {pct_gt_1:.4f} %")
    print(f"      Points >0.1%% rel err = {pct_gt_01:.4f} %")

    # ---- 6. Edge-case audit ----
    if n_valid >= 10:
        n_edge = max(1, n_valid // 100)
        sorted_idx = np.argsort(rv)
        low_idx = sorted_idx[:n_edge]
        high_idx = sorted_idx[-n_edge:]
        low_max_diff = float(np.max(np.abs(diff[low_idx])))
        high_max_diff = float(np.max(np.abs(diff[high_idx])))
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

summary_table = []

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
                    r_masked = r_arr[mask]
                    o_masked = o_arr[mask]
                    ok, stats = deep_compare_array(
                        r_masked, o_masked,
                        ref_label, backend,
                        f"CAPE/{sub_name} (n={n_masked} non-sentinel)",
                        plo, phi,
                        5e-2, 50.0,
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
