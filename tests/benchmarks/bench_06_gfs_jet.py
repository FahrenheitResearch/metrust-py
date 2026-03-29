#!/usr/bin/env python
"""Benchmark 06: GFS Jet Stream Analysis at 250 hPa -- REAL DATA

Scenario
--------
Real GFS 0.25-degree global grid (721 x 1440) from GRIB2 file.
250 hPa level extracted: actual jet stream with cores reaching ~87 m/s.

Data source
-----------
  C:\\Users\\drew\\metrust-py\\data\\gfs_0p25.grib2
  33 isobaric levels, 721 x 1440 per level.
  Loaded with xarray + cfgrib, filtered for 250 hPa.

  Variables used:
    u   -- zonal wind component (m/s)
    v   -- meridional wind component (m/s)
    t   -- temperature (K in GRIB; converted to C for metrust/met-cu)

  Grid spacing: dx ~ 27800 m, dy ~ 27800 m (constant approximation).

Functions benchmarked
---------------------
  wind_speed          (CPU)       element-wise sqrt(u^2 + v^2)
  wind_direction      (CPU)       element-wise atan2
  wind_components     (CPU)       speed/dir -> u, v
  vorticity           (GPU star)  dv/dx - du/dy on uniform grid
  divergence          (CPU)       du/dx + dv/dy
  potential_temp      (GPU star)  theta at 250 hPa

Backends: MetPy (Pint), metrust CPU, met-cu direct, metrust GPU.

Deep verification against MetPy ground truth:
  - Mean diff, max abs diff, RMSE, 99th percentile, relative RMSE%
  - NaN/Inf audit
  - Physical plausibility bounds
  - Pearson correlation (threshold r > 0.9999)
  - Percentage of points > 1% and > 0.1% relative error
  - Edge case checks (jet core, calm spots, 0/360 wraparound)
  - Histogram of absolute differences

Usage:
    python tests/benchmarks/bench_06_gfs_jet.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Imports
# ============================================================================

import metpy.calc as mpcalc
from metpy.units import units

import metrust.calc as mrcalc

HAS_GPU = False
GPU_NAME = "n/a"
try:
    import cupy as cp
    import metcu.calc as mcucalc
    mrcalc.set_backend("gpu")   # test that GPU path works
    mrcalc.set_backend("cpu")
    HAS_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
except Exception:
    cp = None
    mcucalc = None

# ============================================================================
# Timing harness
# ============================================================================

N_WARMUP = 1
N_ITER = 5


def _sync_gpu():
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()


def bench(func, n=N_ITER, gpu=False):
    """Median wall-clock ms: 1 warmup + n timed runs."""
    if gpu:
        _sync_gpu()
    func()  # warmup
    if gpu:
        _sync_gpu()

    times = []
    for _ in range(n):
        if gpu:
            _sync_gpu()
        t0 = time.perf_counter()
        func()
        if gpu:
            _sync_gpu()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def fmt(ms):
    if ms is None:
        return "---"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd_str(a, b):
    if a is None or b is None or b <= 0:
        return "---"
    r = a / b
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


# ============================================================================
# Load REAL GFS 250 hPa data from GRIB2
# ============================================================================

# Locate the GRIB2 file relative to this script or via absolute path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "data", "gfs_0p25.grib2"))
if not os.path.exists(_DATA_PATH):
    # Fallback to absolute path
    _DATA_PATH = r"C:\Users\drew\metrust-py\data\gfs_0p25.grib2"


def load_gfs_250hpa():
    """Load real GFS 250 hPa data from GRIB2 file.

    Returns dict with u, v (m/s), temp_c (Celsius), temp_k (Kelvin),
    lat/lon arrays, grid spacings, and metadata.
    """
    import xarray as xr

    print(f"  Loading GFS data from: {_DATA_PATH}")
    ds = xr.open_dataset(
        _DATA_PATH,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )

    # Extract 250 hPa level
    u_250 = ds["u"].sel(isobaricInhPa=250.0).values.astype(np.float64)
    v_250 = ds["v"].sel(isobaricInhPa=250.0).values.astype(np.float64)
    t_250_k = ds["t"].sel(isobaricInhPa=250.0).values.astype(np.float64)

    lat = ds.coords["latitude"].values.astype(np.float64)
    lon = ds.coords["longitude"].values.astype(np.float64)

    ds.close()

    ny, nx = u_250.shape
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Temperature: GRIB stores in K, convert to C for metrust/met-cu
    temp_c = t_250_k - 273.15

    # Ensure contiguous float64 arrays
    u_250 = np.ascontiguousarray(u_250, dtype=np.float64)
    v_250 = np.ascontiguousarray(v_250, dtype=np.float64)
    temp_c = np.ascontiguousarray(temp_c, dtype=np.float64)

    # Grid spacings: constant approximation for mid-latitudes
    dx = 27800.0   # ~27.8 km (0.25 deg at ~45N in x)
    dy = 27800.0   # ~27.8 km (0.25 deg in y)

    return dict(
        ny=ny, nx=nx,
        lat=lat, lon=lon,
        lat2d=lat2d, lon2d=lon2d,
        u=u_250, v=v_250,
        temp_c=temp_c,
        temp_k=t_250_k,
        pressure_hpa=250.0,
        dx=dx, dy=dy,
    )


# ============================================================================
# Verification helpers
# ============================================================================

def _to_numpy(arr):
    """Convert Pint Quantity / cupy array / ndarray to plain numpy float64."""
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    if hasattr(arr, "get"):      # cupy -> numpy
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# Physical plausibility bounds per quantity
_PHYS_BOUNDS = {
    "wind_speed":      (0.0,    120.0),   # m/s -- real jets can reach ~90+
    "wind_direction":  (0.0,    360.0),   # degrees (0 inclusive, 360 exclusive ideally)
    "wind_components(u)": (-120.0, 120.0),
    "wind_components(v)": (-120.0, 120.0),
    "vorticity":       (-1e-3,  1e-3),    # 1/s
    "divergence":      (-1e-3,  1e-3),    # 1/s -- real GFS 250 hPa can exceed 5e-4
    "potential_temp":  (280.0,  450.0),   # K -- 250 hPa theta typically 300-400 K
}

# Correlation threshold
_PEARSON_THRESHOLD = 0.9999

# Summary collector (filled during verification, printed at end)
_SUMMARY_ROWS = []


def _diff_histogram_str(diffs_flat, n_bins=10):
    """Return an ASCII histogram of absolute differences."""
    finite = diffs_flat[np.isfinite(diffs_flat)]
    if finite.size == 0:
        return "      (no finite diffs)"
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        return f"      all diffs = {lo:.2e}"
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(finite, bins=edges)
    max_count = max(counts.max(), 1)
    bar_width = 40
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / max_count * bar_width))
        lines.append(f"      [{edges[i]:+.2e}, {edges[i+1]:+.2e}) {c:>8d}  {bar}")
    return "\n".join(lines)


def _circular_diff(a_deg, b_deg):
    """Signed angular difference in degrees, accounting for 0/360 wrap."""
    d = a_deg - b_deg
    return (d + 180.0) % 360.0 - 180.0


def deep_verify(name, ref, test, label, u_arr=None, v_arr=None):
    """Exhaustive correctness verification of *test* vs MetPy *ref*.

    Checks performed:
      1. Shape match
      2. NaN / Inf audit
      3. Mean diff, max abs diff, RMSE, 99th percentile abs diff
      4. Relative RMSE (%)
      5. Pearson r (must exceed _PEARSON_THRESHOLD)
      6. Percentage of points with > 1% and > 0.1% relative error
      7. Physical plausibility (bounds from _PHYS_BOUNDS)
      8. Edge cases:
         - Jet core (top 1% wind speed): agreement of derived qty
         - Wind direction at calm spots (speed < 0.5 m/s) -- relaxed
         - Wind direction 0/360 wraparound zone
      9. Histogram of absolute differences

    Returns True if all sub-checks pass.
    """
    ref_np = _to_numpy(ref)
    test_np = _to_numpy(test)
    all_ok = True

    print(f"\n    --- {name} vs {label} ---")

    # ---- 1. Shape ----
    if ref_np.shape != test_np.shape:
        print(f"      FAIL  shape mismatch: ref {ref_np.shape} vs test {test_np.shape}")
        _SUMMARY_ROWS.append((name, label, "FAIL", "shape mismatch", "", "", "", "", ""))
        return False
    n_pts = ref_np.size
    print(f"      shape: {ref_np.shape}  ({n_pts:,} points)")

    # ---- 2. NaN / Inf audit ----
    ref_nan = int(np.count_nonzero(np.isnan(ref_np)))
    ref_inf = int(np.count_nonzero(np.isinf(ref_np)))
    test_nan = int(np.count_nonzero(np.isnan(test_np)))
    test_inf = int(np.count_nonzero(np.isinf(test_np)))
    nan_ok = (ref_nan == test_nan) and (ref_inf == test_inf)
    if not nan_ok:
        # Allow test to have fewer NaN (it may fill differently), but flag new NaN/Inf
        extra_nan = test_nan - ref_nan
        extra_inf = test_inf - ref_inf
        if extra_nan > 0 or extra_inf > 0:
            print(f"      FAIL  NaN/Inf audit: ref NaN={ref_nan} Inf={ref_inf}, "
                  f"test NaN={test_nan} Inf={test_inf} (+{extra_nan} NaN, +{extra_inf} Inf)")
            all_ok = False
        else:
            print(f"      WARN  NaN/Inf audit: ref NaN={ref_nan} Inf={ref_inf}, "
                  f"test NaN={test_nan} Inf={test_inf} (test has fewer -- OK)")
    else:
        print(f"      PASS  NaN/Inf audit: NaN={ref_nan}, Inf={ref_inf} (match)")

    # Build finite mask for remaining comparisons
    finite_mask = np.isfinite(ref_np) & np.isfinite(test_np)
    n_finite = int(np.count_nonzero(finite_mask))
    if n_finite == 0:
        print(f"      SKIP  no finite points to compare")
        _SUMMARY_ROWS.append((name, label, "SKIP", "no finite points", "", "", "", "", ""))
        return all_ok

    ref_f = ref_np[finite_mask]
    test_f = test_np[finite_mask]

    # For wind direction, use circular difference
    is_direction = ("direction" in name.lower())
    if is_direction:
        diffs = _circular_diff(ref_f, test_f)
    else:
        diffs = ref_f - test_f
    abs_diffs = np.abs(diffs)

    # ---- 3. Mean diff, max abs diff, RMSE, 99th percentile ----
    mean_diff = float(np.mean(diffs))
    max_abs_diff = float(np.max(abs_diffs))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    pct99 = float(np.percentile(abs_diffs, 99))

    print(f"      mean_diff     = {mean_diff:+.6e}")
    print(f"      max_abs_diff  = {max_abs_diff:.6e}")
    print(f"      RMSE          = {rmse:.6e}")
    print(f"      99th pct      = {pct99:.6e}")

    # ---- 4. Relative RMSE (%) ----
    ref_range = float(np.max(ref_f) - np.min(ref_f))
    ref_abs_mean = float(np.mean(np.abs(ref_f)))
    denom = max(ref_abs_mean, 1e-30)
    rel_rmse_pct = rmse / denom * 100.0
    print(f"      rel RMSE      = {rel_rmse_pct:.6f}%  (vs mean |ref|={denom:.4e})")

    # ---- 5. Pearson r ----
    if n_finite >= 2 and np.std(ref_f) > 0 and np.std(test_f) > 0:
        r_val, p_val = scipy_stats.pearsonr(ref_f, test_f)
        pearson_ok = r_val >= _PEARSON_THRESHOLD
        status_r = "PASS" if pearson_ok else "FAIL"
        print(f"      {status_r}  Pearson r   = {r_val:.10f}  (threshold {_PEARSON_THRESHOLD})")
        if not pearson_ok:
            all_ok = False
    else:
        r_val = float("nan")
        print(f"      SKIP  Pearson r (constant or too few points)")

    # ---- 6. Percentage > 1% and > 0.1% relative error ----
    safe_denom = np.maximum(np.abs(ref_f), 1e-30)
    rel_err = abs_diffs / safe_denom
    # For direction, relative error is wrt 360 scale
    if is_direction:
        rel_err = abs_diffs / 360.0

    pct_gt_1 = float(np.count_nonzero(rel_err > 0.01) / n_finite * 100.0)
    pct_gt_01 = float(np.count_nonzero(rel_err > 0.001) / n_finite * 100.0)
    print(f"      pts > 1%  rel err  = {pct_gt_1:.4f}%")
    print(f"      pts > 0.1% rel err = {pct_gt_01:.4f}%")

    # ---- 7. Physical plausibility ----
    bounds = _PHYS_BOUNDS.get(name, None)
    if bounds is not None:
        lo, hi = bounds
        test_min = float(np.min(test_f))
        test_max = float(np.max(test_f))
        in_bounds = (test_min >= lo - 1e-10) and (test_max <= hi + 1e-10)
        status_b = "PASS" if in_bounds else "FAIL"
        print(f"      {status_b}  physical bounds [{lo}, {hi}]: "
              f"test range [{test_min:.6e}, {test_max:.6e}]")
        if not in_bounds:
            all_ok = False
    else:
        print(f"      SKIP  physical bounds (not defined for '{name}')")

    # ---- 8. Edge-case checks ----
    # 8a. Jet core region: top 1% by wind speed (if u, v are provided)
    if u_arr is not None and v_arr is not None:
        wspd_full = np.sqrt(u_arr ** 2 + v_arr ** 2)
        jet_threshold = np.percentile(wspd_full, 99)
        jet_mask = (wspd_full >= jet_threshold) & finite_mask
        n_jet = int(np.count_nonzero(jet_mask))
        if n_jet > 0:
            if is_direction:
                jet_diffs = np.abs(_circular_diff(ref_np[jet_mask], test_np[jet_mask]))
            else:
                jet_diffs = np.abs(ref_np[jet_mask] - test_np[jet_mask])
            jet_max = float(np.max(jet_diffs))
            jet_rmse = float(np.sqrt(np.mean(jet_diffs ** 2)))
            print(f"      jet core  (top 1%, n={n_jet}, wspd>={jet_threshold:.1f} m/s): "
                  f"max_abs={jet_max:.6e}, RMSE={jet_rmse:.6e}")

    # 8b. Wind direction at calm spots (speed < 0.5 m/s)
    if is_direction and u_arr is not None and v_arr is not None:
        wspd_full = np.sqrt(u_arr ** 2 + v_arr ** 2)
        calm_mask = (wspd_full < 0.5) & finite_mask
        n_calm = int(np.count_nonzero(calm_mask))
        if n_calm > 0:
            calm_diffs = np.abs(_circular_diff(ref_np[calm_mask], test_np[calm_mask]))
            calm_max = float(np.max(calm_diffs))
            calm_mean = float(np.mean(calm_diffs))
            # At calm spots, direction is essentially undefined; large diffs are expected
            print(f"      calm spots (wspd<0.5, n={n_calm}): mean_circ_diff={calm_mean:.2f} deg, "
                  f"max_circ_diff={calm_max:.2f} deg  (relaxed -- direction undefined at calm)")
        else:
            print(f"      calm spots (wspd<0.5): none found")

    # 8c. Wind direction near 0/360 wraparound
    if is_direction:
        near_0_360 = ((ref_np > 350) | (ref_np < 10)) & finite_mask
        n_wrap = int(np.count_nonzero(near_0_360))
        if n_wrap > 0:
            wrap_diffs = np.abs(_circular_diff(ref_np[near_0_360], test_np[near_0_360]))
            wrap_max = float(np.max(wrap_diffs))
            wrap_mean = float(np.mean(wrap_diffs))
            wrap_ok = wrap_max < 1.0  # within 1 degree
            status_w = "PASS" if wrap_ok else "FAIL"
            print(f"      {status_w}  0/360 wraparound zone (n={n_wrap}): "
                  f"mean_circ_diff={wrap_mean:.6f} deg, max_circ_diff={wrap_max:.6f} deg")
            if not wrap_ok:
                all_ok = False
        else:
            print(f"      0/360 wraparound zone: no points near 0/360")

    # ---- 9. Histogram of absolute differences ----
    print(f"      abs-diff histogram ({n_finite} finite points):")
    print(_diff_histogram_str(abs_diffs))

    # ---- Final verdict for this quantity ----
    overall = "PASS" if all_ok else "FAIL"
    print(f"      >>> {name} vs {label}: {overall}")

    # Collect for summary table
    _SUMMARY_ROWS.append((
        name, label, overall,
        f"{mean_diff:+.2e}", f"{max_abs_diff:.2e}", f"{rmse:.2e}",
        f"{pct99:.2e}", f"{rel_rmse_pct:.4f}%",
        f"{r_val:.8f}" if np.isfinite(r_val) else "N/A",
    ))
    return all_ok


def print_summary_table():
    """Print a compact summary table of all deep verification results."""
    print("\n" + "=" * 130)
    print("  DEEP VERIFICATION SUMMARY TABLE")
    print("=" * 130)
    hdr = (f"  {'Quantity':<25s} {'Backend':<15s} {'Result':<7s} "
           f"{'Mean Diff':>12s} {'Max Abs':>12s} {'RMSE':>12s} "
           f"{'99th Pct':>12s} {'Rel RMSE%':>12s} {'Pearson r':>14s}")
    print(hdr)
    print("  " + "-" * 126)
    for row in _SUMMARY_ROWS:
        (qty, backend, result, mean_d, max_a, rmse_s, p99, rel_r, pearson) = row
        print(f"  {qty:<25s} {backend:<15s} {result:<7s} "
              f"{mean_d:>12s} {max_a:>12s} {rmse_s:>12s} "
              f"{p99:>12s} {rel_r:>12s} {pearson:>14s}")
    print("=" * 130)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("  BENCHMARK 06: GFS Jet Stream Analysis at 250 hPa -- REAL DATA")
    print(f"  Backends: MetPy | metrust CPU | met-cu direct | metrust GPU")
    print(f"  GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print("=" * 100)

    d = load_gfs_250hpa()
    ny, nx = d["ny"], d["nx"]
    u, v = d["u"], d["v"]
    temp_c = d["temp_c"]
    temp_k = d["temp_k"]
    dx, dy = d["dx"], d["dy"]
    pressure_hpa = d["pressure_hpa"]

    wspd_max = np.max(np.sqrt(u**2 + v**2))
    wspd_min = np.min(np.sqrt(u**2 + v**2))

    print(f"\n  Grid: {ny} x {nx} = {ny * nx:,} points  (REAL GFS 0.25-degree)")
    print(f"  Jet stream wind range: {wspd_min:.2f} - {wspd_max:.2f} m/s")
    print(f"  Temperature range: {temp_k.min():.2f} - {temp_k.max():.2f} K "
          f"({temp_c.min():.2f} - {temp_c.max():.2f} C)")
    print(f"  u range: {u.min():.2f} to {u.max():.2f} m/s")
    print(f"  v range: {v.min():.2f} to {v.max():.2f} m/s")
    print(f"  dx = {dx:.0f} m, dy = {dy:.0f} m")
    print(f"  Pressure: {pressure_hpa} hPa")

    # ---- MetPy Pint quantities (creation not timed) ----
    u_q = u * units("m/s")
    v_q = v * units("m/s")
    dx_q = dx * units.m
    dy_q = dy * units.m
    temp_q = temp_c * units.degC
    pres_q = pressure_hpa * units.hPa

    # ==================================================================
    # DEEP VERIFICATION (MetPy = ground truth)
    # ==================================================================
    print("\n" + "=" * 130)
    print("  DEEP DATA CORRECTNESS VERIFICATION  (Real GFS 250 hPa)")
    print("  Ground truth: MetPy (Pint).  Every backend compared element-wise.")
    print("=" * 130)

    all_pass = True
    _SUMMARY_ROWS.clear()

    # --- Reference: MetPy ---
    print("\n  Computing MetPy reference values ...")
    ref_wspd = mpcalc.wind_speed(u_q, v_q)
    ref_wdir = mpcalc.wind_direction(u_q, v_q)
    ref_uc, ref_vc = mpcalc.wind_components(ref_wspd, ref_wdir)
    ref_vort = mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q)
    ref_div = mpcalc.divergence(u_q, v_q, dx=dx_q, dy=dy_q)
    ref_theta = mpcalc.potential_temperature(pres_q, temp_q)

    # Precompute speed/dir for wind_components tests
    spd_np = np.sqrt(u ** 2 + v ** 2)
    dir_np = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    # ------------------------------------------------------------------
    # metrust CPU
    # ------------------------------------------------------------------
    print("\n" + "-" * 130)
    print("  [metrust CPU vs MetPy]")
    print("-" * 130)
    mrcalc.set_backend("cpu")
    mr_wspd = mrcalc.wind_speed(u, v)
    mr_wdir = mrcalc.wind_direction(u, v)
    mr_uc, mr_vc = mrcalc.wind_components(spd_np, dir_np)
    mr_vort = mrcalc.vorticity(u, v, dx=dx, dy=dy)
    mr_div = mrcalc.divergence(u, v, dx=dx, dy=dy)
    mr_theta = mrcalc.potential_temperature(pressure_hpa, temp_c)

    all_pass &= deep_verify("wind_speed", ref_wspd, mr_wspd, "metrust CPU", u, v)
    all_pass &= deep_verify("wind_direction", ref_wdir, mr_wdir, "metrust CPU", u, v)
    all_pass &= deep_verify("wind_components(u)", ref_uc, mr_uc, "metrust CPU", u, v)
    all_pass &= deep_verify("wind_components(v)", ref_vc, mr_vc, "metrust CPU", u, v)
    all_pass &= deep_verify("vorticity", ref_vort, mr_vort, "metrust CPU", u, v)
    all_pass &= deep_verify("divergence", ref_div, mr_div, "metrust CPU", u, v)
    all_pass &= deep_verify("potential_temp", ref_theta, mr_theta, "metrust CPU", u, v)

    # ------------------------------------------------------------------
    # met-cu direct (GPU)
    # ------------------------------------------------------------------
    if HAS_GPU:
        print("\n" + "-" * 130)
        print("  [met-cu direct vs MetPy]")
        print("-" * 130)
        cu_wspd = mcucalc.wind_speed(u, v)
        cu_wdir = mcucalc.wind_direction(u, v)
        cu_uc, cu_vc = mcucalc.wind_components(spd_np, dir_np)
        cu_vort = mcucalc.vorticity(u, v, dx=dx, dy=dy)
        cu_div = mcucalc.divergence(u, v, dx=dx, dy=dy)
        cu_theta = mcucalc.potential_temperature(pressure_hpa, temp_c)
        cp.cuda.Stream.null.synchronize()

        all_pass &= deep_verify("wind_speed", ref_wspd, cu_wspd, "met-cu", u, v)
        all_pass &= deep_verify("wind_direction", ref_wdir, cu_wdir, "met-cu", u, v)
        all_pass &= deep_verify("wind_components(u)", ref_uc, cu_uc, "met-cu", u, v)
        all_pass &= deep_verify("wind_components(v)", ref_vc, cu_vc, "met-cu", u, v)
        all_pass &= deep_verify("vorticity", ref_vort, cu_vort, "met-cu", u, v)
        all_pass &= deep_verify("divergence", ref_div, cu_div, "met-cu", u, v)
        all_pass &= deep_verify("potential_temp", ref_theta, cu_theta, "met-cu", u, v)

        # --- metrust GPU (only GPU-eligible functions) ---
        print("\n" + "-" * 130)
        print("  [metrust GPU vs MetPy]")
        print("-" * 130)
        mrcalc.set_backend("gpu")
        mg_vort = mrcalc.vorticity(u, v, dx=dx, dy=dy)
        mg_theta = mrcalc.potential_temperature(pressure_hpa, temp_c)
        mrcalc.set_backend("cpu")

        all_pass &= deep_verify("vorticity", ref_vort, mg_vort, "metrust GPU", u, v)
        all_pass &= deep_verify("potential_temp", ref_theta, mg_theta, "metrust GPU", u, v)
    else:
        print("\n  [GPU backends skipped -- no CUDA]")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print_summary_table()

    overall_str = "ALL PASS" if all_pass else "*** SOME FAILURES ***"
    print(f"\n  Overall verification: {overall_str}")

    # ==================================================================
    # TIMING
    # ==================================================================
    print("\n" + "=" * 100)
    print("  TIMING (1 warmup + 5, median)")
    print("=" * 100)

    # Column header
    hdr = (f"  {'Function':30s}"
           f" {'MetPy':>11s} {'Rust CPU':>11s}"
           f" {'met-cu':>11s} {'mrust GPU':>11s}"
           f" {'Rust/MP':>8s} {'cu/Rust':>8s}")
    sep = "-" * 100

    # ------------------------------------------------------------------
    # WIND SPEED
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"  WIND  ({ny}x{nx})")
    print(f"{sep}")
    print(hdr)

    # wind_speed
    t_mp = bench(lambda: mpcalc.wind_speed(u_q, v_q))
    t_cpu = bench(lambda: mrcalc.wind_speed(u, v))
    t_cu = bench(lambda: mcucalc.wind_speed(u, v), gpu=True) if HAS_GPU else None
    # wind_speed has no GPU path in metrust; it always uses Rust CPU
    t_gpu = None

    print(f"  {'wind_speed':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # wind_direction
    t_mp = bench(lambda: mpcalc.wind_direction(u_q, v_q))
    t_cpu = bench(lambda: mrcalc.wind_direction(u, v))
    t_cu = bench(lambda: mcucalc.wind_direction(u, v), gpu=True) if HAS_GPU else None
    t_gpu = None

    print(f"  {'wind_direction':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # wind_components
    spd_q = spd_np * units("m/s")
    dir_q = dir_np * units.degree

    t_mp = bench(lambda: mpcalc.wind_components(spd_q, dir_q))
    t_cpu = bench(lambda: mrcalc.wind_components(spd_np, dir_np))
    t_cu = bench(lambda: mcucalc.wind_components(spd_np, dir_np), gpu=True) if HAS_GPU else None
    t_gpu = None

    print(f"  {'wind_components':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # ------------------------------------------------------------------
    # KINEMATICS
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"  KINEMATICS  ({ny}x{nx})")
    print(f"{sep}")
    print(hdr)

    # vorticity (GPU-eligible)
    mrcalc.set_backend("cpu")
    t_mp = bench(lambda: mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q))
    t_cpu = bench(lambda: mrcalc.vorticity(u, v, dx=dx, dy=dy))
    t_cu = bench(lambda: mcucalc.vorticity(u, v, dx=dx, dy=dy), gpu=True) if HAS_GPU else None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.vorticity(u, v, dx=dx, dy=dy), gpu=True)
        mrcalc.set_backend("cpu")
    else:
        t_gpu = None

    print(f"  {'vorticity (*)':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # divergence
    mrcalc.set_backend("cpu")
    t_mp = bench(lambda: mpcalc.divergence(u_q, v_q, dx=dx_q, dy=dy_q))
    t_cpu = bench(lambda: mrcalc.divergence(u, v, dx=dx, dy=dy))
    t_cu = bench(lambda: mcucalc.divergence(u, v, dx=dx, dy=dy), gpu=True) if HAS_GPU else None
    t_gpu = None  # divergence has no metrust GPU dispatch

    print(f"  {'divergence':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # ------------------------------------------------------------------
    # THERMODYNAMICS
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"  THERMODYNAMICS  ({ny}x{nx})")
    print(f"{sep}")
    print(hdr)

    # potential_temperature (GPU-eligible)
    mrcalc.set_backend("cpu")
    t_mp = bench(lambda: mpcalc.potential_temperature(pres_q, temp_q))
    t_cpu = bench(lambda: mrcalc.potential_temperature(pressure_hpa, temp_c))
    t_cu = bench(lambda: mcucalc.potential_temperature(pressure_hpa, temp_c), gpu=True) if HAS_GPU else None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.potential_temperature(pressure_hpa, temp_c), gpu=True)
        mrcalc.set_backend("cpu")
    else:
        t_gpu = None

    print(f"  {'potential_temperature (*)':30s}"
          f" {fmt(t_mp):>11s} {fmt(t_cpu):>11s}"
          f" {fmt(t_cu):>11s} {fmt(t_gpu):>11s}"
          f" {spd_str(t_mp, t_cpu):>8s} {spd_str(t_cpu, t_cu):>8s}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 130}")
    print("  (*) = GPU-eligible function")
    print(f"  Grid: {ny}x{nx} = {ny * nx:,} points  |  250 hPa GFS jet stream -- REAL DATA")
    overall_str = "ALL PASS" if all_pass else "*** SOME FAILURES ***"
    print(f"  Deep verification: {overall_str}")
    n_pass = sum(1 for r in _SUMMARY_ROWS if r[2] == "PASS")
    n_fail = sum(1 for r in _SUMMARY_ROWS if r[2] == "FAIL")
    n_skip = sum(1 for r in _SUMMARY_ROWS if r[2] == "SKIP")
    print(f"  Checks: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP  "
          f"(total {len(_SUMMARY_ROWS)} quantity-backend pairs)")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
