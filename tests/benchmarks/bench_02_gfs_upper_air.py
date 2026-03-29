"""Benchmark 02: GFS 500 hPa Upper-Air Analysis -- REAL DATA

Scenario
--------
Global 0.25-degree grid (721 x 1440), single level.
Real GFS 0.25 analysis data from 2026-03-28 00Z (gfs_0p25.grib2).
500 hPa heights, winds, temperature extracted from isobaric GRIB file.

Functions benchmarked
---------------------
vorticity            (GPU)
divergence           (CPU only)
advection            (CPU only)
frontogenesis        (GPU)  -- uses potential temperature field
wind_speed           (CPU only)
wind_direction       (CPU only)
potential_temperature(GPU)  -- at 500 hPa

Four backends: MetPy, metrust CPU, met-cu (direct GPU), metrust GPU.
"""

import time
import statistics
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Grid configuration -- GFS 0.25 degree, 721 x 1440
# ---------------------------------------------------------------------------

NY, NX = 721, 1440
PRESSURE_HPA = 500.0               # 500 hPa level
DX = 27_800.0                      # ~27.8 km representative mid-latitude
DY = 27_800.0                      # ~27.8 km meridional (meters)

N_WARMUP = 1
N_TIMED  = 3
RTOL     = 1e-4

# ---------------------------------------------------------------------------
# Load REAL GFS data from GRIB2
# ---------------------------------------------------------------------------

print("=" * 80)
print("BENCHMARK 02 -- GFS 500 hPa Upper-Air Analysis  (721 x 1440)  REAL DATA")
print("=" * 80)
print()

GRIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gfs_0p25.grib2")
GRIB_PATH = os.path.normpath(GRIB_PATH)

print(f"Loading GFS GRIB2: {GRIB_PATH}")
if not os.path.isfile(GRIB_PATH):
    print(f"  ERROR: file not found -- {GRIB_PATH}")
    sys.exit(1)

import xarray as xr

ds = xr.open_dataset(
    GRIB_PATH,
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
        "indexpath": "",
    },
)

# Find nearest 500 hPa level index
levels = ds.coords["isobaricInhPa"].values
idx_500 = int(np.argmin(np.abs(levels - PRESSURE_HPA)))
actual_level = levels[idx_500]
print(f"  500 hPa level index: {idx_500}  (actual: {actual_level:.0f} hPa)")

# Extract 500 hPa slices -- GRIB temperature is Kelvin
temp_K = np.ascontiguousarray(ds["t"].values[idx_500], dtype=np.float64)    # K
u_wind = np.ascontiguousarray(ds["u"].values[idx_500], dtype=np.float64)    # m/s
v_wind = np.ascontiguousarray(ds["v"].values[idx_500], dtype=np.float64)    # m/s
heights = np.ascontiguousarray(ds["gh"].values[idx_500], dtype=np.float64)  # gpm

# Convert Kelvin -> Celsius for potential_temperature input
temp = temp_K - 273.15  # degC

# Potential temperature at 500 hPa: theta = T_K * (1000/p)^(R/cp)
theta = temp_K * (1000.0 / PRESSURE_HPA) ** 0.286
theta = np.ascontiguousarray(theta, dtype=np.float64)

# Pressure field for potential_temperature benchmark (uniform 500 hPa)
pressure_field = np.full((NY, NX), PRESSURE_HPA, dtype=np.float64)

ds.close()

speed_check = np.sqrt(u_wind**2 + v_wind**2)
print(f"  Grid shape:    {NY} x {NX}  ({NY * NX:,} points)")
print(f"  Heights range: {heights.min():.0f} - {heights.max():.0f} gpm")
print(f"  Temp range:    {temp.min():.1f} - {temp.max():.1f} C")
print(f"  Wind speed:    {speed_check.min():.1f} - {speed_check.max():.1f} m/s")
print(f"  Theta range:   {theta.min():.1f} - {theta.max():.1f} K")
print(f"  dx = {DX:.0f} m,  dy = {DY:.0f} m")
print()

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed(func, n_warmup=N_WARMUP, n_timed=N_TIMED, sync_gpu=False):
    """Return (median_seconds, last_result)."""
    for _ in range(n_warmup):
        result = func()
        if sync_gpu:
            _gpu_sync()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = func()
        if sync_gpu:
            _gpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times), result


def _gpu_sync():
    """Synchronize CuPy stream if available."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def fmt_ms(seconds):
    ms = seconds * 1000
    if ms < 0.01:
        return f"{ms * 1000:.2f} us"
    if ms < 1.0:
        return f"{ms:.3f} ms"
    return f"{ms:.1f} ms"


def to_numpy(arr):
    """Bring result back to numpy regardless of origin."""
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Import backends
# ---------------------------------------------------------------------------

print("Loading backends ...")

# 1) MetPy
import metpy.calc as mpcalc
from metpy.units import units
print("  [OK] MetPy")

# 2) metrust CPU
import metrust.calc as mrcalc
mrcalc.set_backend("cpu")
print("  [OK] metrust CPU")

# 3) met-cu direct
try:
    import metcu.calc as mcucalc
    HAS_METCU = True
    print("  [OK] met-cu (direct GPU)")
except ImportError as e:
    HAS_METCU = False
    print(f"  [SKIP] met-cu: {e}")

# 4) metrust GPU
try:
    mrcalc.set_backend("gpu")
    _test = mrcalc.potential_temperature(500.0 * units.hPa, -20.0 * units.degC)
    HAS_METRUST_GPU = True
    mrcalc.set_backend("cpu")   # reset for CPU benchmarks first
    print("  [OK] metrust GPU")
except Exception as e:
    HAS_METRUST_GPU = False
    print(f"  [SKIP] metrust GPU: {e}")

print()

# ---------------------------------------------------------------------------
# Prepare MetPy Pint-wrapped inputs (one-time cost, not benchmarked)
# ---------------------------------------------------------------------------

u_q = u_wind * units("m/s")
v_q = v_wind * units("m/s")
dx_q = DX * units.m
dy_q = DY * units.m
theta_q = theta * units.K
temp_q = temp * units.degC            # Celsius for potential_temperature
temp_K_q = temp_K * units.K           # Kelvin for advection (avoids Pint offset pitfall)
pressure_q = pressure_field * units.hPa

# ---------------------------------------------------------------------------
# Prepare metrust Pint-wrapped inputs
# ---------------------------------------------------------------------------

u_mr = u_wind * units("m/s")
v_mr = v_wind * units("m/s")
dx_mr = DX * units.m
dy_mr = DY * units.m
theta_mr = theta * units.K
temp_mr = temp * units.degC           # Celsius for potential_temperature
temp_K_mr = temp_K * units.K          # Kelvin for advection
pressure_mr = pressure_field * units.hPa

# ---------------------------------------------------------------------------
# Results storage
# ---------------------------------------------------------------------------

columns = ["MetPy", "metrust CPU", "met-cu GPU", "metrust GPU"]
results = {}   # func_name -> {backend: (time_s, np_result)}


def record(func_name, backend, time_s, result_arr):
    if func_name not in results:
        results[func_name] = {}
    results[func_name][backend] = (time_s, to_numpy(result_arr))


# ===================================================================
# 1. VORTICITY  (GPU-capable)
# ===================================================================

print("-" * 80)
print("VORTICITY")
print("-" * 80)

t, r = timed(lambda: mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q))
record("vorticity", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.vorticity(u_mr, v_mr, dx=dx_mr, dy=dy_mr))
record("vorticity", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.vorticity(u_wind, v_wind, dx=DX, dy=DY), sync_gpu=True)
    record("vorticity", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.vorticity(u_mr, v_mr, dx=dx_mr, dy=dy_mr), sync_gpu=True)
    record("vorticity", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 2. DIVERGENCE  (CPU only for metrust/MetPy)
# ===================================================================

print()
print("-" * 80)
print("DIVERGENCE")
print("-" * 80)

t, r = timed(lambda: mpcalc.divergence(u_q, v_q, dx=dx_q, dy=dy_q))
record("divergence", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.divergence(u_mr, v_mr, dx=dx_mr, dy=dy_mr))
record("divergence", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.divergence(u_wind, v_wind, dx=DX, dy=DY), sync_gpu=True)
    record("divergence", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.divergence(u_mr, v_mr, dx=dx_mr, dy=dy_mr), sync_gpu=True)
    record("divergence", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 3. ADVECTION  (CPU only for metrust/MetPy)
# ===================================================================

print()
print("-" * 80)
print("ADVECTION  (temperature advection by wind)")
print("-" * 80)

# Use Kelvin for advection scalar -- avoids Pint offset-unit (degC) issues
# where 1 degC/s != 1 K/s in Pint's offset arithmetic.
t, r = timed(lambda: mpcalc.advection(temp_K_q, u_q, v_q, dx=dx_q, dy=dy_q))
record("advection", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.advection(temp_K_mr, u_mr, v_mr, dx=dx_mr, dy=dy_mr))
record("advection", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    # met-cu takes raw numpy; pass Kelvin values for consistency
    t, r = timed(lambda: mcucalc.advection(temp_K, u_wind, v_wind, dx=DX, dy=DY), sync_gpu=True)
    record("advection", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.advection(temp_K_mr, u_mr, v_mr, dx=dx_mr, dy=dy_mr), sync_gpu=True)
    record("advection", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 4. FRONTOGENESIS  (GPU-capable)
# ===================================================================

print()
print("-" * 80)
print("FRONTOGENESIS  (Petterssen, theta field)")
print("-" * 80)

t, r = timed(lambda: mpcalc.frontogenesis(theta_q, u_q, v_q, dx=dx_q, dy=dy_q))
record("frontogenesis", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.frontogenesis(theta_mr, u_mr, v_mr, dx=dx_mr, dy=dy_mr))
record("frontogenesis", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.frontogenesis(theta, u_wind, v_wind, dx=DX, dy=DY), sync_gpu=True)
    record("frontogenesis", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.frontogenesis(theta_mr, u_mr, v_mr, dx=dx_mr, dy=dy_mr), sync_gpu=True)
    record("frontogenesis", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 5. WIND SPEED  (CPU only)
# ===================================================================

print()
print("-" * 80)
print("WIND SPEED")
print("-" * 80)

t, r = timed(lambda: mpcalc.wind_speed(u_q, v_q))
record("wind_speed", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.wind_speed(u_mr, v_mr))
record("wind_speed", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.wind_speed(u_wind, v_wind), sync_gpu=True)
    record("wind_speed", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.wind_speed(u_mr, v_mr), sync_gpu=True)
    record("wind_speed", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 6. WIND DIRECTION  (CPU only)
# ===================================================================

print()
print("-" * 80)
print("WIND DIRECTION")
print("-" * 80)

t, r = timed(lambda: mpcalc.wind_direction(u_q, v_q))
record("wind_direction", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.wind_direction(u_mr, v_mr))
record("wind_direction", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.wind_direction(u_wind, v_wind), sync_gpu=True)
    record("wind_direction", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.wind_direction(u_mr, v_mr), sync_gpu=True)
    record("wind_direction", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 7. POTENTIAL TEMPERATURE  (GPU-capable)
# ===================================================================

print()
print("-" * 80)
print("POTENTIAL TEMPERATURE  (500 hPa)")
print("-" * 80)

t, r = timed(lambda: mpcalc.potential_temperature(pressure_q, temp_q))
record("potential_temperature", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.potential_temperature(pressure_mr, temp_mr))
record("potential_temperature", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    # met-cu expects hPa for pressure, Celsius for temperature
    # Broadcast scalar pressure to array shape for met-cu
    t, r = timed(lambda: mcucalc.potential_temperature(pressure_field, temp), sync_gpu=True)
    record("potential_temperature", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.potential_temperature(pressure_mr, temp_mr), sync_gpu=True)
    record("potential_temperature", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")


# ===================================================================
# COMPREHENSIVE DATA CORRECTNESS VERIFICATION
# MetPy is treated as ground truth for all comparisons.
# ===================================================================

from scipy.stats import pearsonr
import math

# -- Physical plausibility bounds per function --
# Bounds reflect realistic extremes for a 0.25-deg GFS grid with real
# 500 hPa atmospheric data.  Finite-difference gradients on the 27.8 km
# grid can produce values larger than textbook synoptic estimates.
PHYS_BOUNDS = {
    "vorticity":             (-5e-3, 5e-3),      # 1/s  (real vorticity, mesoscale OK)
    "divergence":            (-5e-3, 5e-3),       # 1/s  (real divergence)
    "advection":             (-1.0, 1.0),          # K/s  (temperature advection)
    "frontogenesis":         (-1e-4, 1e-4),        # K/m/s (Petterssen frontogenesis)
    "wind_speed":            (0.0, 100.0),         # m/s  (max ~71 m/s in data)
    "wind_direction":        (0.0, 360.0),         # degrees
    "potential_temperature": (260.0, 350.0),       # K  (real range ~273-333 K)
}

# -- Thresholds for PASS/FAIL --
CORR_THRESHOLD     = 0.9999     # Pearson r must exceed this
REL_ERR_1PCT_MAX   = 0.01       # max fraction of points with >1% relative error
REL_ERR_01PCT_MAX  = 0.05       # max fraction of points with >0.1% relative error

# Pre-compute the wind speed field for edge-case lookups
wspd = np.sqrt(u_wind**2 + v_wind**2)

func_order = [
    "vorticity", "divergence", "advection", "frontogenesis",
    "wind_speed", "wind_direction", "potential_temperature",
]

# Collect per-function per-backend verdicts for summary table
# verdict_table[func_name][bk_name] = {"pass": bool, "rmse": float, "max_diff": float, ...}
verdict_table = {}
all_pass = True

print()
print("=" * 80)
print("COMPREHENSIVE DATA CORRECTNESS VERIFICATION")
print("  Ground truth: MetPy   |   Grid: 721 x 1440  (1,038,240 pts)")
print("  Data source:  GFS 0.25 analysis  2026-03-28 00Z  500 hPa")
print("=" * 80)


def text_histogram(diffs, bins=15, width=50):
    """Print a text histogram of difference values."""
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        print("        (no finite values)")
        return
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        print(f"        all values = {lo:.6e}")
        return
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(finite, bins=edges)
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(bins):
        bar_len = int(round(counts[i] / max_count * width))
        bar = "#" * bar_len
        print(f"        [{edges[i]:+12.4e}, {edges[i+1]:+12.4e})  {counts[i]:>8d}  {bar}")


for func_name in func_order:
    if func_name not in results:
        continue
    backends = results[func_name]
    if "MetPy" not in backends:
        continue
    ref_raw = backends["MetPy"][1]
    ref_2d = ref_raw.reshape(NY, NX) if ref_raw.size == NY * NX else ref_raw

    print()
    print("-" * 80)
    print(f"  {func_name.upper()}")
    print("-" * 80)

    verdict_table[func_name] = {}

    for bk_name in ["metrust CPU", "met-cu GPU", "metrust GPU"]:
        if bk_name not in backends:
            continue

        cand_raw = backends[bk_name][1]

        # Align shapes
        ref_flat = ref_raw.ravel()
        cand_flat = cand_raw.ravel()
        n = min(len(ref_flat), len(cand_flat))
        ref_c = ref_flat[:n]
        cand_c = cand_flat[:n]

        # Valid mask (finite in both)
        valid = np.isfinite(ref_c) & np.isfinite(cand_c)
        n_valid = int(valid.sum())

        if n_valid == 0:
            print(f"\n    {bk_name}: no valid points -- SKIP")
            verdict_table[func_name][bk_name] = {
                "pass": False, "rmse": float("nan"), "max_diff": float("nan")}
            all_pass = False
            continue

        ref_v = ref_c[valid]
        cand_v = cand_c[valid]
        diff = cand_v - ref_v
        abs_diff = np.abs(diff)

        # ==============================================================
        # 1. Element-wise statistics
        # ==============================================================
        mean_diff    = float(np.mean(diff))
        max_abs_diff = float(np.max(abs_diff))
        rmse         = float(np.sqrt(np.mean(diff**2)))
        p99_abs      = float(np.percentile(abs_diff, 99))

        ref_range = float(np.max(np.abs(ref_v))) if np.max(np.abs(ref_v)) > 0 else 1.0
        rel_rmse_pct = rmse / ref_range * 100.0

        print(f"\n    --- MetPy vs {bk_name} ---")
        print(f"    [1] Element-wise stats:")
        print(f"        Mean diff        = {mean_diff:+.6e}")
        print(f"        Max |diff|       = {max_abs_diff:.6e}")
        print(f"        RMSE             = {rmse:.6e}")
        print(f"        99th pctl |diff| = {p99_abs:.6e}")
        print(f"        Relative RMSE    = {rel_rmse_pct:.8f} %")

        # ==============================================================
        # 2. NaN / Inf audit
        # ==============================================================
        ref_nan  = np.isnan(ref_c)
        cand_nan = np.isnan(cand_c)
        ref_inf  = np.isinf(ref_c)
        cand_inf = np.isinf(cand_c)

        ref_nan_ct  = int(ref_nan.sum())
        cand_nan_ct = int(cand_nan.sum())
        ref_inf_ct  = int(ref_inf.sum())
        cand_inf_ct = int(cand_inf.sum())

        nan_loc_agree = int((ref_nan == cand_nan).sum())
        inf_loc_agree = int((ref_inf == cand_inf).sum())

        print(f"    [2] NaN/Inf audit:")
        print(f"        MetPy  NaN={ref_nan_ct:>7d}   Inf={ref_inf_ct:>7d}")
        print(f"        {bk_name:15s} NaN={cand_nan_ct:>7d}   Inf={cand_inf_ct:>7d}")
        print(f"        NaN location agreement: {nan_loc_agree}/{n}  "
              f"({'MATCH' if nan_loc_agree == n else 'MISMATCH'})")
        print(f"        Inf location agreement: {inf_loc_agree}/{n}  "
              f"({'MATCH' if inf_loc_agree == n else 'MISMATCH'})")

        nan_inf_ok = (nan_loc_agree == n) and (inf_loc_agree == n)

        # ==============================================================
        # 3. Physical plausibility (check backend output)
        # ==============================================================
        phys_ok = True
        if func_name in PHYS_BOUNDS:
            lo_b, hi_b = PHYS_BOUNDS[func_name]
            cand_finite = cand_c[np.isfinite(cand_c)]
            cand_min = float(np.min(cand_finite)) if len(cand_finite) > 0 else float("nan")
            cand_max = float(np.max(cand_finite)) if len(cand_finite) > 0 else float("nan")
            in_range = (cand_min >= lo_b) and (cand_max <= hi_b)
            phys_ok = in_range
            status_str = "OK" if in_range else "OUT OF RANGE"
            print(f"    [3] Physical plausibility [{lo_b}, {hi_b}]:")
            print(f"        {bk_name} range: [{cand_min:.6e}, {cand_max:.6e}]  {status_str}")
        else:
            print(f"    [3] Physical plausibility: no bounds defined -- SKIP")

        # ==============================================================
        # 4. Pearson correlation
        # ==============================================================
        # pearsonr needs variance in both; constant arrays get r=NaN
        if np.std(ref_v) > 0 and np.std(cand_v) > 0:
            corr, _ = pearsonr(ref_v, cand_v)
        else:
            corr = 1.0 if np.allclose(ref_v, cand_v) else 0.0
        corr_ok = corr > CORR_THRESHOLD
        print(f"    [4] Pearson correlation:  r = {corr:.10f}  "
              f"(threshold {CORR_THRESHOLD})  {'PASS' if corr_ok else 'FAIL'}")

        # ==============================================================
        # 5. Spatial error distribution
        # ==============================================================
        denom = np.abs(ref_v)
        denom_safe = np.where(denom > 0, denom, 1.0)  # avoid div-by-zero
        rel_err = abs_diff / denom_safe
        # Only count relative error where reference is non-negligible
        # (>1% of max absolute value)
        significant = denom > (0.01 * ref_range)
        n_sig = int(significant.sum())

        if n_sig > 0:
            rel_err_sig = rel_err[significant]
            pct_gt_1pct  = float(np.sum(rel_err_sig > 0.01)) / n_sig * 100.0
            pct_gt_01pct = float(np.sum(rel_err_sig > 0.001)) / n_sig * 100.0
        else:
            pct_gt_1pct = 0.0
            pct_gt_01pct = 0.0

        spatial_ok = (pct_gt_1pct / 100.0 <= REL_ERR_1PCT_MAX) and \
                     (pct_gt_01pct / 100.0 <= REL_ERR_01PCT_MAX)
        print(f"    [5] Spatial error distribution (over {n_sig:,} significant pts):")
        print(f"        Points with >1%   rel error: {pct_gt_1pct:.4f} %"
              f"  (limit {REL_ERR_1PCT_MAX*100:.1f}%)")
        print(f"        Points with >0.1% rel error: {pct_gt_01pct:.4f} %"
              f"  (limit {REL_ERR_01PCT_MAX*100:.1f}%)")

        # ==============================================================
        # 6. Edge case audit (jet core, trough axis, ridge)
        # ==============================================================
        print(f"    [6] Edge case audit:")

        # Use 2D arrays for spatial lookups
        ref_2d_local = ref_c.reshape(NY, NX) if ref_c.size == NY * NX else None
        cand_2d_local = cand_c.reshape(NY, NX) if cand_c.size == NY * NX else None

        if ref_2d_local is not None and cand_2d_local is not None:
            # a) Jet core: location of maximum wind speed
            jet_idx = np.unravel_index(np.argmax(wspd), wspd.shape)
            jr, jc = jet_idx
            ref_jet = ref_2d_local[jr, jc]
            cand_jet = cand_2d_local[jr, jc]
            jet_diff = abs(cand_jet - ref_jet)
            print(f"        Jet core       (row={jr:4d}, col={jc:4d}):  "
                  f"MetPy={ref_jet:+.8e}  {bk_name}={cand_jet:+.8e}  |diff|={jet_diff:.4e}")

            # b) Trough axis: location of max absolute vorticity (from MetPy)
            if "vorticity" in results and "MetPy" in results["vorticity"]:
                vort_ref = results["vorticity"]["MetPy"][1]
                if vort_ref.size == NY * NX:
                    vort_2d = vort_ref.reshape(NY, NX)
                    trough_idx = np.unravel_index(np.argmax(np.abs(vort_2d)), vort_2d.shape)
                    tr, tc = trough_idx
                    ref_trough = ref_2d_local[tr, tc]
                    cand_trough = cand_2d_local[tr, tc]
                    trough_diff = abs(cand_trough - ref_trough)
                    print(f"        Trough axis    (row={tr:4d}, col={tc:4d}):  "
                          f"MetPy={ref_trough:+.8e}  {bk_name}={cand_trough:+.8e}  |diff|={trough_diff:.4e}")
                else:
                    print(f"        Trough axis: vorticity shape mismatch -- SKIP")
            else:
                print(f"        Trough axis: vorticity not available -- SKIP")

            # c) Ridge: location of minimum absolute vorticity
            if "vorticity" in results and "MetPy" in results["vorticity"]:
                vort_ref = results["vorticity"]["MetPy"][1]
                if vort_ref.size == NY * NX:
                    vort_2d = vort_ref.reshape(NY, NX)
                    # Min vorticity = most anticyclonic
                    ridge_idx = np.unravel_index(np.argmin(vort_2d), vort_2d.shape)
                    rr, rc = ridge_idx
                    ref_ridge = ref_2d_local[rr, rc]
                    cand_ridge = cand_2d_local[rr, rc]
                    ridge_diff = abs(cand_ridge - ref_ridge)
                    print(f"        Ridge          (row={rr:4d}, col={rc:4d}):  "
                          f"MetPy={ref_ridge:+.8e}  {bk_name}={cand_ridge:+.8e}  |diff|={ridge_diff:.4e}")
                else:
                    print(f"        Ridge: vorticity shape mismatch -- SKIP")
            else:
                print(f"        Ridge: vorticity not available -- SKIP")
        else:
            print(f"        (cannot reshape to 2D {NY}x{NX} -- size {ref_c.size}) -- SKIP")

        # ==============================================================
        # 7. Histogram of differences
        # ==============================================================
        print(f"    [7] Histogram of (backend - MetPy):")
        text_histogram(diff, bins=15, width=40)

        # ==============================================================
        # Overall PASS / FAIL for this function + backend
        # ==============================================================
        close_ok = np.allclose(ref_v, cand_v, rtol=RTOL, atol=1e-10)
        overall = close_ok and nan_inf_ok and phys_ok and corr_ok and spatial_ok

        if not overall:
            all_pass = False

        fail_reasons = []
        if not close_ok:
            fail_reasons.append("allclose")
        if not nan_inf_ok:
            fail_reasons.append("NaN/Inf mismatch")
        if not phys_ok:
            fail_reasons.append("phys bounds")
        if not corr_ok:
            fail_reasons.append(f"corr {corr:.6f}")
        if not spatial_ok:
            fail_reasons.append("spatial rel err")

        status_str = "PASS" if overall else "FAIL"
        reason_str = f"  ({', '.join(fail_reasons)})" if fail_reasons else ""
        print(f"\n    >>> {func_name} / {bk_name}:  {status_str}{reason_str}")

        verdict_table[func_name][bk_name] = {
            "pass": overall,
            "rmse": rmse,
            "max_diff": max_abs_diff,
            "corr": corr,
            "rel_rmse_pct": rel_rmse_pct,
            "pct_gt_1pct": pct_gt_1pct,
            "pct_gt_01pct": pct_gt_01pct,
        }

# ===================================================================
# VERIFICATION SUMMARY TABLE
# ===================================================================

print()
print("=" * 80)
print("VERIFICATION SUMMARY TABLE")
print("=" * 80)

bk_list = ["metrust CPU", "met-cu GPU", "metrust GPU"]
hdr = f"  {'Function':25s}  {'Backend':15s}  {'Verdict':6s}  {'RMSE':>12s}  {'Max |diff|':>12s}  {'Corr r':>12s}  {'RelRMSE%':>10s}  {'>1%rel':>8s}  {'>0.1%rel':>8s}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for func_name in func_order:
    if func_name not in verdict_table:
        continue
    for bk_name in bk_list:
        if bk_name not in verdict_table[func_name]:
            continue
        v = verdict_table[func_name][bk_name]
        tag = " PASS" if v["pass"] else "*FAIL"
        corr_s = f"{v.get('corr', float('nan')):.8f}" if not math.isnan(v.get("corr", float("nan"))) else "N/A"
        print(f"  {func_name:25s}  {bk_name:15s}  {tag:>6s}  {v['rmse']:12.4e}  {v['max_diff']:12.4e}  {corr_s:>12s}  {v.get('rel_rmse_pct',0):10.6f}  {v.get('pct_gt_1pct',0):7.4f}%  {v.get('pct_gt_01pct',0):7.4f}%")

print()
if all_pass:
    print("  ** ALL VERIFICATION CHECKS PASSED **")
else:
    print("  !! SOME VERIFICATION CHECKS FAILED -- see details above !!")

# ===================================================================
# TIMING SUMMARY TABLE
# ===================================================================

print()
print("=" * 80)
print("TIMING SUMMARY  (median of 3 runs, 721x1440 grid, REAL GFS data)")
print("=" * 80)

header = f"  {'Function':25s}"
for col in columns:
    header += f"  {col:>14s}"
header += f"  {'Speedup':>10s}"
print(header)
print("  " + "-" * (25 + 4 * 16 + 12))

for func_name in func_order:
    if func_name not in results:
        continue
    backends = results[func_name]
    row = f"  {func_name:25s}"
    metpy_t = None
    fastest_t = None
    for col in columns:
        if col in backends:
            t_s = backends[col][0]
            row += f"  {fmt_ms(t_s):>14s}"
            if col == "MetPy":
                metpy_t = t_s
            if fastest_t is None or t_s < fastest_t:
                fastest_t = t_s
        else:
            row += f"  {'--':>14s}"
    # Speedup: MetPy / fastest non-MetPy
    if metpy_t is not None and fastest_t is not None and fastest_t < metpy_t:
        row += f"  {metpy_t / fastest_t:>9.1f}x"
    else:
        row += f"  {'--':>10s}"
    print(row)

print()
print("=" * 80)
print("Benchmark complete.")
print("=" * 80)
