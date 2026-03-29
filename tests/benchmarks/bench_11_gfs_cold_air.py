#!/usr/bin/env python
"""Bench 11 -- GFS Cold Air Outbreak verification benchmark (REAL DATA).

Scenario
--------
Real GFS 0.25-deg analysis at 850 hPa from ``data/gfs_0p25.grib2``.
Mid-latitude baroclinic zone slice: 30-70 N (161 x 1440).
  - Temperature, winds, specific humidity read directly from GRIB2
  - Dewpoint derived from specific humidity via vapor pressure
  - dx = dy = 27 800 m  (GFS 0.25-deg mid-latitude spacing)

Functions
---------
  potential_temperature           (GPU-eligible)
  equivalent_potential_temperature (GPU-eligible)
  dewpoint (from vapor pressure)  (GPU-eligible)
  vorticity                       (GPU-eligible)
  frontogenesis                   (GPU-eligible)
  saturation_mixing_ratio         (CPU + met-cu)
  virtual_temperature             (CPU + met-cu)

Backends: MetPy (Pint), metrust CPU, met-cu direct, metrust GPU.

Deep Verification
-----------------
For every function, every backend vs MetPy reference:
  - Mean diff, max abs diff, RMSE, 99th percentile abs diff
  - Relative RMSE%, NaN/Inf audit
  - Physical plausibility bounds
  - Pearson correlation coefficient
  - Percentage of points > 1% and > 0.1% relative error
  - Histogram of diffs (text)
  - Edge cases: coldest point, warmest point, strongest gradient

Timing: perf_counter, cupy sync, 1 warmup + 5 timed, median.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Load REAL GFS 850 hPa data
# ---------------------------------------------------------------------------
import xarray as xr

GRIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gfs_0p25.grib2")
GRIB_PATH = os.path.normpath(GRIB_PATH)
if not os.path.isfile(GRIB_PATH):
    # fallback: try relative to cwd
    GRIB_PATH = os.path.normpath("data/gfs_0p25.grib2")

print(f"Loading GRIB2: {GRIB_PATH}")
ds = xr.open_dataset(
    GRIB_PATH,
    engine="cfgrib",
    backend_kwargs={
        "indexpath": "",
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 850},
    },
    errors="ignore",
)

# Latitude slice: 30-70 N (baroclinic zone)
# GFS lat: 90 -> -90 in 0.25 steps.  Index for 70N = (90-70)/0.25 = 80
# Index for 30N = (90-30)/0.25 = 240 (inclusive) so slice is [80:241]
LAT_70N = int((90 - 70) / 0.25)       # 80
LAT_30N = int((90 - 30) / 0.25) + 1   # 241
lat_vals = ds.latitude.values[LAT_70N:LAT_30N]

NY = LAT_30N - LAT_70N   # 161
NX = 1440
DX = 27_800.0   # metres -- approximate GFS 0.25-deg at mid-latitudes
DY = 27_800.0
P_HPA = 850.0

# Extract 2-D slices as contiguous float64
t_K_arr  = np.ascontiguousarray(ds["t"].values[LAT_70N:LAT_30N, :], dtype=np.float64)   # Kelvin
t_arr    = t_K_arr - 273.15                                                               # Celsius
u_arr    = np.ascontiguousarray(ds["u"].values[LAT_70N:LAT_30N, :], dtype=np.float64)   # m/s
v_arr    = np.ascontiguousarray(ds["v"].values[LAT_70N:LAT_30N, :], dtype=np.float64)   # m/s
q_arr    = np.ascontiguousarray(ds["q"].values[LAT_70N:LAT_30N, :], dtype=np.float64)   # kg/kg specific humidity

ds.close()

print(f"  Grid: {NY} x {NX}   lat {lat_vals[0]:.2f} N to {lat_vals[-1]:.2f} N")
print(f"  T(C) range : [{t_arr.min():.2f}, {t_arr.max():.2f}]")
print(f"  u range    : [{u_arr.min():.2f}, {u_arr.max():.2f}] m/s")
print(f"  v range    : [{v_arr.min():.2f}, {v_arr.max():.2f}] m/s")
print(f"  q range    : [{q_arr.min():.6f}, {q_arr.max():.6f}] kg/kg")

# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------
# Dewpoint: compute vapor pressure from specific humidity, then invert Bolton
#   e = q * p / (0.622 + 0.378 * q)   [hPa]
vp_arr = np.ascontiguousarray(q_arr * P_HPA / (0.62197 + 0.37803 * q_arr), dtype=np.float64)
# Bolton inversion: Td(C) = 243.5 * ln(e/6.112) / (17.67 - ln(e/6.112))
_log_ratio = np.log(vp_arr / 6.112)
td_arr = np.ascontiguousarray(243.5 * _log_ratio / (17.67 - _log_ratio), dtype=np.float64)
# Ensure Td <= T - 0.1 (physical constraint)
td_arr = np.minimum(td_arr, t_arr - 0.1)

# Mixing ratio from specific humidity:  w = q / (1 - q)
mr_arr = np.ascontiguousarray(q_arr / (1.0 - q_arr), dtype=np.float64)
mr_arr = np.clip(mr_arr, 0.0, None)

# Potential temperature field for frontogenesis (compute from T and p)
# theta = T * (1000/p)^(R/cp),  R/cp = 0.2854
theta_arr = np.ascontiguousarray(t_K_arr * (1000.0 / P_HPA) ** 0.2854, dtype=np.float64)

# Full pressure array (GPU kernels need broadcasted arrays, not scalars)
p_arr = np.full((NY, NX), P_HPA, dtype=np.float64)

print(f"  Td(C) range: [{td_arr.min():.2f}, {td_arr.max():.2f}]")
print(f"  VP range   : [{vp_arr.min():.4f}, {vp_arr.max():.4f}] hPa")
print(f"  w range    : [{mr_arr.min():.6f}, {mr_arr.max():.6f}] kg/kg")
print(f"  theta range: [{theta_arr.min():.2f}, {theta_arr.max():.2f}] K")

# ---------------------------------------------------------------------------
# Imports -- MetPy, metrust, met-cu
# ---------------------------------------------------------------------------
import metpy.calc as mpcalc
from metpy.units import units

import metrust.calc as mrcalc

HAS_GPU = False
GPU_NAME = "n/a"
try:
    import cupy as cp
    import metcu.calc as mcucalc
    mrcalc.set_backend("gpu")
    mrcalc.set_backend("cpu")
    HAS_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
except Exception:
    pass

# ---------------------------------------------------------------------------
# Pint-wrapped fields for MetPy
# ---------------------------------------------------------------------------
p_q     = P_HPA * units.hPa
t_q     = t_K_arr * units.K          # MetPy expects Kelvin
td_q    = (td_arr + 273.15) * units.K
u_q     = u_arr * units("m/s")
v_q     = v_arr * units("m/s")
theta_q = theta_arr * units.K
dx_q    = DX * units.m
dy_q    = DY * units.m
vp_q    = vp_arr * units.hPa
mr_q    = mr_arr * units("kg/kg")

# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
def _sync():
    cp.cuda.Stream.null.synchronize()


def bench(func, n=5, gpu=False):
    """1 warmup + n timed, return median ms."""
    if gpu:
        _sync()
    func()                    # warmup
    if gpu:
        _sync()
    times = []
    for _ in range(n):
        if gpu:
            _sync()
        t0 = time.perf_counter()
        func()
        if gpu:
            _sync()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def fmt(ms):
    if ms is None:
        return "-"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1.0:
        return f"{ms:.3f} ms"
    if ms < 1000.0:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd(ref, val):
    if ref is None or val is None or val <= 0:
        return "-"
    r = ref / val
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


# ---------------------------------------------------------------------------
# Edge-case indices: coldest, warmest, strongest gradient
# ---------------------------------------------------------------------------
# Coldest point: find actual minimum T in array
_flat_cold = np.argmin(t_arr)
IDX_COLD = np.unravel_index(_flat_cold, t_arr.shape)

# Warmest point: find actual maximum T in array
_flat_warm = np.argmax(t_arr)
IDX_WARM = np.unravel_index(_flat_warm, t_arr.shape)

# Strongest gradient: highest dT/dy magnitude in interior
t_grad_y = np.abs(np.gradient(t_arr, axis=0))
_grad_flat = t_grad_y[2:-2, 2:-2].ravel()
_grad_idx = np.argmax(_grad_flat)
_gy, _gx = np.unravel_index(_grad_idx, t_grad_y[2:-2, 2:-2].shape)
IDX_GRAD = (_gy + 2, _gx + 2)

EDGE_LABELS = {
    "coldest":      IDX_COLD,
    "warmest":      IDX_WARM,
    "max gradient": IDX_GRAD,
}


# ===========================================================================
# DEEP VERIFICATION ENGINE
# ===========================================================================

def _to_numpy(arr):
    """Extract a numpy float64 array from any backend result."""
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    if hasattr(arr, "get"):        # cupy
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


def _ascii_histogram(diffs, bins=10, width=40):
    """Return a list of strings forming an ASCII histogram of diffs."""
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return ["      (no finite values)"]
    counts, edges = np.histogram(finite, bins=bins)
    max_c = max(counts) if max(counts) > 0 else 1
    lines = []
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(round(counts[i] / max_c * width))
        bar = "#" * bar_len
        lines.append(f"      [{lo:+12.5e}, {hi:+12.5e}) {counts[i]:>8d} {bar}")
    return lines


def deep_verify(func_name, ref, test, backend_label,
                phys_lo, phys_hi, phys_unit,
                mask_zero_ref=1e-20):
    """Run full statistical + physical verification of test vs ref.

    Returns a dict with all metrics and a bool 'pass' flag.
    """
    ref_a = _to_numpy(ref)
    test_a = _to_numpy(test)

    result = {
        "func": func_name,
        "backend": backend_label,
        "pass": True,
        "notes": [],
    }

    # --- Shape check ---
    if ref_a.shape != test_a.shape:
        result["pass"] = False
        result["notes"].append(f"SHAPE MISMATCH: ref {ref_a.shape} vs test {test_a.shape}")
        return result

    N = ref_a.size

    # --- NaN / Inf audit ---
    nan_ref = int(np.isnan(ref_a).sum())
    nan_test = int(np.isnan(test_a).sum())
    inf_ref = int(np.isinf(ref_a).sum())
    inf_test = int(np.isinf(test_a).sum())
    result["nan_ref"] = nan_ref
    result["nan_test"] = nan_test
    result["inf_ref"] = inf_ref
    result["inf_test"] = inf_test

    if nan_test > nan_ref:
        result["notes"].append(f"extra NaN: test has {nan_test} vs ref {nan_ref}")
    if inf_test > 0 and inf_ref == 0:
        result["pass"] = False
        result["notes"].append(f"Inf introduced: test has {inf_test} Inf values")

    # --- Finite mask ---
    finite_mask = np.isfinite(ref_a) & np.isfinite(test_a)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        result["pass"] = False
        result["notes"].append("no finite points to compare")
        return result

    r = ref_a[finite_mask]
    t = test_a[finite_mask]
    diff = t - r
    abs_diff = np.abs(diff)

    # --- Core statistics ---
    result["mean_diff"] = float(np.mean(diff))
    result["max_abs_diff"] = float(np.max(abs_diff))
    result["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    result["pct99_abs_diff"] = float(np.percentile(abs_diff, 99))

    # --- Relative RMSE% ---
    ref_range = float(np.max(r) - np.min(r))
    if ref_range > 0:
        result["rel_rmse_pct"] = result["rmse"] / ref_range * 100.0
    else:
        result["rel_rmse_pct"] = 0.0

    # --- Pearson correlation ---
    if np.std(r) > 0 and np.std(t) > 0:
        result["pearson_r"] = float(np.corrcoef(r, t)[0, 1])
    else:
        result["pearson_r"] = 1.0  # constant arrays are trivially correlated

    # --- Relative error percentages (where ref is non-negligible) ---
    nontrivial = np.abs(r) > mask_zero_ref
    n_nontrivial = int(nontrivial.sum())
    if n_nontrivial > 0:
        rel_err = np.abs(diff[nontrivial]) / np.abs(r[nontrivial])
        result["pct_gt_1pct_relerr"] = float(np.mean(rel_err > 0.01) * 100.0)
        result["pct_gt_01pct_relerr"] = float(np.mean(rel_err > 0.001) * 100.0)
    else:
        result["pct_gt_1pct_relerr"] = 0.0
        result["pct_gt_01pct_relerr"] = 0.0

    # --- Physical plausibility on test output ---
    test_finite = test_a[np.isfinite(test_a)]
    if len(test_finite) > 0:
        t_min = float(np.min(test_finite))
        t_max = float(np.max(test_finite))
        result["phys_min"] = t_min
        result["phys_max"] = t_max
        if t_min < phys_lo or t_max > phys_hi:
            result["pass"] = False
            result["notes"].append(
                f"PHYS BOUNDS VIOLATED: [{t_min:.4f}, {t_max:.4f}] "
                f"outside [{phys_lo}, {phys_hi}] {phys_unit}")

    # --- Histogram of diffs ---
    result["hist_lines"] = _ascii_histogram(diff)

    # --- Edge case checks ---
    result["edge_cases"] = {}
    for label, idx in EDGE_LABELS.items():
        r_val = float(ref_a[idx])
        t_val = float(test_a[idx])
        d_val = t_val - r_val
        rel = abs(d_val / r_val) if abs(r_val) > mask_zero_ref else 0.0
        result["edge_cases"][label] = {
            "ref": r_val, "test": t_val, "diff": d_val, "rel": rel
        }

    # --- PASS/FAIL criteria ---
    # Strict: RMSE must be tiny relative to range, Pearson > 0.9999
    if result["rel_rmse_pct"] > 0.1:  # >0.1% of range
        result["pass"] = False
        result["notes"].append(f"rel RMSE% too high: {result['rel_rmse_pct']:.4f}%")
    if result["pearson_r"] < 0.9999:
        result["pass"] = False
        result["notes"].append(f"Pearson r too low: {result['pearson_r']:.8f}")
    if result["pct_gt_1pct_relerr"] > 1.0:
        result["pass"] = False
        result["notes"].append(
            f">1% rel error at {result['pct_gt_1pct_relerr']:.2f}% of points")

    return result


def print_deep_result(res):
    """Pretty-print a deep_verify result dict."""
    tag = "PASS" if res["pass"] else "FAIL"
    print(f"\n    --- {res['func']} [{res['backend']}] : {tag} ---")

    if "mean_diff" not in res:
        for n in res.get("notes", []):
            print(f"      !! {n}")
        return

    print(f"      Mean diff        : {res['mean_diff']:+.6e}")
    print(f"      Max |diff|       : {res['max_abs_diff']:.6e}")
    print(f"      RMSE             : {res['rmse']:.6e}")
    print(f"      99th pct |diff|  : {res['pct99_abs_diff']:.6e}")
    print(f"      Relative RMSE%   : {res['rel_rmse_pct']:.6f}%")
    print(f"      Pearson r        : {res['pearson_r']:.10f}")
    print(f"      NaN (ref/test)   : {res['nan_ref']} / {res['nan_test']}")
    print(f"      Inf (ref/test)   : {res['inf_ref']} / {res['inf_test']}")
    print(f"      % pts >1% rel err: {res['pct_gt_1pct_relerr']:.4f}%")
    print(f"      % pts >0.1% rel  : {res['pct_gt_01pct_relerr']:.4f}%")

    if "phys_min" in res:
        print(f"      Output range     : [{res['phys_min']:.4f}, {res['phys_max']:.4f}]")

    # Edge cases
    for label, ec in res.get("edge_cases", {}).items():
        print(f"      Edge [{label:16s}]: ref={ec['ref']:+.6f}  "
              f"test={ec['test']:+.6f}  diff={ec['diff']:+.4e}  "
              f"rel={ec['rel']:.4e}")

    # Histogram
    print("      Diff histogram:")
    for line in res.get("hist_lines", []):
        print(line)

    # Notes / warnings
    for n in res.get("notes", []):
        print(f"      !! {n}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
results = []  # (name, t_mp, t_cpu, t_gpu_direct, t_gpu_metrust, verified)
all_deep = []  # all deep_verify result dicts


def record(name, t_mp, t_cpu, t_gpu_direct, t_gpu_metrust, verified):
    results.append((name, t_mp, t_cpu, t_gpu_direct, t_gpu_metrust, verified))
    v_tag = "PASS" if verified else "FAIL"
    parts = [f"  {name:42s}"]
    parts.append(f"MetPy {fmt(t_mp):>10s}")
    parts.append(f"CPU {fmt(t_cpu):>10s} ({spd(t_mp, t_cpu):>5s})")
    if t_gpu_direct is not None:
        parts.append(f"GPU-d {fmt(t_gpu_direct):>10s} ({spd(t_mp, t_gpu_direct):>5s})")
    else:
        parts.append(f"{'GPU-d':>5s} {'  -':>10s} {'':>7s}")
    if t_gpu_metrust is not None:
        parts.append(f"GPU-mr {fmt(t_gpu_metrust):>10s} ({spd(t_mp, t_gpu_metrust):>5s})")
    else:
        parts.append(f"{'GPU-mr':>6s} {'  -':>10s} {'':>7s}")
    parts.append(f"[{v_tag}]")
    print(" | ".join(parts))


# ===========================================================================
# Header
# ===========================================================================
print()
print("=" * 120)
print("BENCH 11 -- GFS 850 hPa Real Data  |  grid %d x %d  |  30-70 N baroclinic zone" % (NY, NX))
print("  DEEP DATA CORRECTNESS VERIFICATION")
print("=" * 120)
print(f"  GPU: {GPU_NAME}" if HAS_GPU else "  GPU: not available")
print(f"  Edge-case indices:")
for label, idx in EDGE_LABELS.items():
    print(f"    {label:20s} : row={idx[0]:3d}  col={idx[1]:4d}  "
          f"T={t_arr[idx]:.1f} C  Td={td_arr[idx]:.1f} C")
print()


# ===========================================================================
# Physical plausibility bounds for each function
# ===========================================================================
# Based on real 850 hPa mid-latitude data ranges (30-70N).
# T ranges roughly -36 C to +30 C at 850 hPa.
PHYS_BOUNDS = {
    "potential_temperature":            (245.0, 325.0, "K"),
    "equivalent_potential_temperature": (245.0, 380.0, "K"),
    "dewpoint":                         (-70.0, 30.0,  "degC"),
    "vorticity":                        (-5e-3, 5e-3,  "1/s"),
    "frontogenesis":                    (-1e-6, 1e-6,  "K/m/s"),
    "saturation_mixing_ratio":          (0.0,   0.040, "kg/kg"),
    "virtual_temperature":              (-40.0, 35.0,  "degC"),
}


# ===========================================================================
# Helper: run deep verification for all backends of a function
# ===========================================================================
def verify_all_backends(func_name, ref, cpu_result, gpu_direct_result, gpu_mr_result):
    """Run deep_verify for each backend, print results, return combined pass."""
    lo, hi, unit_label = PHYS_BOUNDS[func_name]
    combined_pass = True

    # CPU vs MetPy
    res_cpu = deep_verify(func_name, ref, cpu_result, "metrust-CPU",
                          lo, hi, unit_label)
    all_deep.append(res_cpu)
    print_deep_result(res_cpu)
    if not res_cpu["pass"]:
        combined_pass = False

    # GPU-direct vs MetPy
    if gpu_direct_result is not None:
        res_gpud = deep_verify(func_name, ref, gpu_direct_result, "met-cu-direct",
                               lo, hi, unit_label)
        all_deep.append(res_gpud)
        print_deep_result(res_gpud)
        if not res_gpud["pass"]:
            combined_pass = False

    # GPU-metrust vs MetPy
    if gpu_mr_result is not None:
        res_gpumr = deep_verify(func_name, ref, gpu_mr_result, "metrust-GPU",
                                lo, hi, unit_label)
        all_deep.append(res_gpumr)
        print_deep_result(res_gpumr)
        if not res_gpumr["pass"]:
            combined_pass = False

    return combined_pass


# ===========================================================================
# 1. potential_temperature  (GPU-eligible)
# ===========================================================================
print("-" * 120)
print("THERMODYNAMIC FUNCTIONS")
print("-" * 120)

# -- MetPy reference --
ref_pt = mpcalc.potential_temperature(p_q, t_q).to("K").magnitude

# -- metrust CPU (takes hPa scalar, Celsius array) --
mrcalc.set_backend("cpu")
mr_pt = mrcalc.potential_temperature(P_HPA, t_arr)
if hasattr(mr_pt, "magnitude"):
    mr_pt = mr_pt.magnitude
mr_pt = np.asarray(mr_pt, dtype=np.float64)

# -- met-cu direct (needs 2-D pressure array) --
mcu_pt = None
if HAS_GPU:
    mcu_pt_raw = mcucalc.potential_temperature(p_arr, t_arr)
    mcu_pt = mcu_pt_raw.get() if hasattr(mcu_pt_raw, "get") else np.asarray(mcu_pt_raw)

# -- metrust GPU --
mrgpu_pt = None
if HAS_GPU:
    mrcalc.set_backend("gpu")
    mrgpu_pt_raw = mrcalc.potential_temperature(p_arr, t_arr)
    if hasattr(mrgpu_pt_raw, "magnitude"):
        mrgpu_pt_raw = mrgpu_pt_raw.magnitude
    mrgpu_pt = mrgpu_pt_raw.get() if hasattr(mrgpu_pt_raw, "get") else np.asarray(mrgpu_pt_raw)
    mrcalc.set_backend("cpu")

# -- Deep verify --
verified = verify_all_backends("potential_temperature", ref_pt, mr_pt, mcu_pt, mrgpu_pt)

# -- Timing --
t_mp = bench(lambda: mpcalc.potential_temperature(p_q, t_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.potential_temperature(P_HPA, t_arr))
t_gpu_d = bench(lambda: mcucalc.potential_temperature(p_arr, t_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("gpu")
t_gpu_mr = bench(lambda: mrcalc.potential_temperature(p_arr, t_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("cpu")

record("potential_temperature", t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified)

# ===========================================================================
# 2. equivalent_potential_temperature  (GPU-eligible)
# ===========================================================================

# -- MetPy reference --
ref_ept = mpcalc.equivalent_potential_temperature(p_q, t_q, td_q).to("K").magnitude

# -- metrust CPU (scalar pressure, Celsius arrays) --
mrcalc.set_backend("cpu")
mr_ept = mrcalc.equivalent_potential_temperature(P_HPA, t_arr, td_arr)
if hasattr(mr_ept, "magnitude"):
    mr_ept = mr_ept.magnitude
mr_ept = np.asarray(mr_ept, dtype=np.float64)

# -- met-cu direct (broadcast scalar pressure to 2-D array for GPU) --
mcu_ept = None
if HAS_GPU:
    mcu_ept_raw = mcucalc.equivalent_potential_temperature(p_arr, t_arr, td_arr)
    mcu_ept = mcu_ept_raw.get() if hasattr(mcu_ept_raw, "get") else np.asarray(mcu_ept_raw)

# -- metrust GPU --
mrgpu_ept = None
if HAS_GPU:
    mrcalc.set_backend("gpu")
    mrgpu_ept_raw = mrcalc.equivalent_potential_temperature(p_arr, t_arr, td_arr)
    if hasattr(mrgpu_ept_raw, "magnitude"):
        mrgpu_ept_raw = mrgpu_ept_raw.magnitude
    mrgpu_ept = mrgpu_ept_raw.get() if hasattr(mrgpu_ept_raw, "get") else np.asarray(mrgpu_ept_raw)
    mrcalc.set_backend("cpu")

# -- Deep verify --
verified = verify_all_backends("equivalent_potential_temperature",
                               ref_ept, mr_ept, mcu_ept, mrgpu_ept)

# -- Timing --
t_mp = bench(lambda: mpcalc.equivalent_potential_temperature(p_q, t_q, td_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.equivalent_potential_temperature(P_HPA, t_arr, td_arr))
t_gpu_d = bench(lambda: mcucalc.equivalent_potential_temperature(p_arr, t_arr, td_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("gpu")
t_gpu_mr = bench(lambda: mrcalc.equivalent_potential_temperature(p_arr, t_arr, td_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("cpu")

record("equivalent_potential_temperature", t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified)

# ===========================================================================
# 3. dewpoint from vapor pressure  (GPU-eligible)
# ===========================================================================

# -- MetPy reference --
ref_dp = mpcalc.dewpoint(vp_q).to("degC").magnitude

# -- metrust CPU --
mrcalc.set_backend("cpu")
mr_dp = mrcalc.dewpoint(vp_arr)
if hasattr(mr_dp, "magnitude"):
    mr_dp = mr_dp.magnitude
mr_dp = np.asarray(mr_dp, dtype=np.float64)

# -- met-cu direct --
mcu_dp = None
if HAS_GPU:
    mcu_dp_raw = mcucalc.dewpoint(vp_arr)
    mcu_dp = mcu_dp_raw.get() if hasattr(mcu_dp_raw, "get") else np.asarray(mcu_dp_raw)

# -- metrust GPU --
mrgpu_dp = None
if HAS_GPU:
    mrcalc.set_backend("gpu")
    mrgpu_dp_raw = mrcalc.dewpoint(vp_arr)
    if hasattr(mrgpu_dp_raw, "magnitude"):
        mrgpu_dp_raw = mrgpu_dp_raw.magnitude
    mrgpu_dp = mrgpu_dp_raw.get() if hasattr(mrgpu_dp_raw, "get") else np.asarray(mrgpu_dp_raw)
    mrcalc.set_backend("cpu")

# -- Deep verify --
verified = verify_all_backends("dewpoint", ref_dp, mr_dp, mcu_dp, mrgpu_dp)

# -- Timing --
t_mp = bench(lambda: mpcalc.dewpoint(vp_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.dewpoint(vp_arr))
t_gpu_d = bench(lambda: mcucalc.dewpoint(vp_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("gpu")
t_gpu_mr = bench(lambda: mrcalc.dewpoint(vp_arr), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("cpu")

record("dewpoint (from VP)", t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified)

# ===========================================================================
# 4. vorticity  (GPU-eligible)
# ===========================================================================
print()
print("-" * 120)
print("KINEMATIC FUNCTIONS")
print("-" * 120)

# -- MetPy reference --
ref_vort = mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q).to("1/s").magnitude

# -- metrust CPU --
mrcalc.set_backend("cpu")
mr_vort = mrcalc.vorticity(u_arr, v_arr, dx=DX, dy=DY)
if hasattr(mr_vort, "magnitude"):
    mr_vort = mr_vort.magnitude
mr_vort = np.asarray(mr_vort, dtype=np.float64)

# -- met-cu direct --
mcu_vort = None
if HAS_GPU:
    mcu_vort_raw = mcucalc.vorticity(u_arr, v_arr, dx=DX, dy=DY)
    mcu_vort = mcu_vort_raw.get() if hasattr(mcu_vort_raw, "get") else np.asarray(mcu_vort_raw)

# -- metrust GPU --
mrgpu_vort = None
if HAS_GPU:
    mrcalc.set_backend("gpu")
    mrgpu_vort_raw = mrcalc.vorticity(u_arr, v_arr, dx=DX, dy=DY)
    if hasattr(mrgpu_vort_raw, "magnitude"):
        mrgpu_vort_raw = mrgpu_vort_raw.magnitude
    mrgpu_vort = mrgpu_vort_raw.get() if hasattr(mrgpu_vort_raw, "get") else np.asarray(mrgpu_vort_raw)
    mrcalc.set_backend("cpu")

# -- Deep verify --
verified = verify_all_backends("vorticity", ref_vort, mr_vort, mcu_vort, mrgpu_vort)

# -- Timing --
t_mp = bench(lambda: mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.vorticity(u_arr, v_arr, dx=DX, dy=DY))
t_gpu_d = bench(lambda: mcucalc.vorticity(u_arr, v_arr, dx=DX, dy=DY), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("gpu")
t_gpu_mr = bench(lambda: mrcalc.vorticity(u_arr, v_arr, dx=DX, dy=DY), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("cpu")

record("vorticity", t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified)

# ===========================================================================
# 5. frontogenesis  (GPU-eligible)
# ===========================================================================

# -- MetPy reference --
ref_fronto = mpcalc.frontogenesis(theta_q, u_q, v_q, dx=dx_q, dy=dy_q).to("K/m/s").magnitude

# -- metrust CPU (raw arrays, scalar dx/dy) --
mrcalc.set_backend("cpu")
mr_fronto = mrcalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY)
if hasattr(mr_fronto, "magnitude"):
    mr_fronto = mr_fronto.magnitude
mr_fronto = np.asarray(mr_fronto, dtype=np.float64)

# -- met-cu direct --
mcu_fronto = None
if HAS_GPU:
    mcu_fronto_raw = mcucalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY)
    mcu_fronto = mcu_fronto_raw.get() if hasattr(mcu_fronto_raw, "get") else np.asarray(mcu_fronto_raw)

# -- metrust GPU --
mrgpu_fronto = None
if HAS_GPU:
    mrcalc.set_backend("gpu")
    mrgpu_fronto_raw = mrcalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY)
    if hasattr(mrgpu_fronto_raw, "magnitude"):
        mrgpu_fronto_raw = mrgpu_fronto_raw.magnitude
    mrgpu_fronto = mrgpu_fronto_raw.get() if hasattr(mrgpu_fronto_raw, "get") else np.asarray(mrgpu_fronto_raw)
    mrcalc.set_backend("cpu")

# -- Deep verify (frontogenesis: mask near-zero ref for relative error) --
verified = verify_all_backends("frontogenesis", ref_fronto, mr_fronto,
                               mcu_fronto, mrgpu_fronto)

# -- Timing --
t_mp = bench(lambda: mpcalc.frontogenesis(theta_q, u_q, v_q, dx=dx_q, dy=dy_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY))
t_gpu_d = bench(lambda: mcucalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("gpu")
t_gpu_mr = bench(lambda: mrcalc.frontogenesis(theta_arr, u_arr, v_arr, dx=DX, dy=DY), gpu=True) if HAS_GPU else None
if HAS_GPU:
    mrcalc.set_backend("cpu")

record("frontogenesis", t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified)

# ===========================================================================
# 6. saturation_mixing_ratio  (CPU + met-cu)
# ===========================================================================
print()
print("-" * 120)
print("MOISTURE / VIRTUAL TEMP FUNCTIONS")
print("-" * 120)

# -- MetPy reference (takes pressure in hPa, temperature in K) --
ref_smr = mpcalc.saturation_mixing_ratio(p_q, t_q).to("kg/kg").magnitude

# -- metrust CPU (scalar pressure hPa, Celsius array) --
mrcalc.set_backend("cpu")
mr_smr = mrcalc.saturation_mixing_ratio(P_HPA, t_arr)
if hasattr(mr_smr, "magnitude"):
    mr_smr = mr_smr.magnitude
mr_smr = np.asarray(mr_smr, dtype=np.float64)

# -- met-cu direct (needs 2-D pressure array for GPU kernel) --
mcu_smr = None
if HAS_GPU:
    mcu_smr_raw = mcucalc.saturation_mixing_ratio(p_arr, t_arr)
    mcu_smr = mcu_smr_raw.get() if hasattr(mcu_smr_raw, "get") else np.asarray(mcu_smr_raw)

# -- Deep verify --
verified = verify_all_backends("saturation_mixing_ratio",
                               ref_smr, mr_smr, mcu_smr, None)

# -- Timing --
t_mp = bench(lambda: mpcalc.saturation_mixing_ratio(p_q, t_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.saturation_mixing_ratio(P_HPA, t_arr))
t_gpu_d = bench(lambda: mcucalc.saturation_mixing_ratio(p_arr, t_arr), gpu=True) if HAS_GPU else None

record("saturation_mixing_ratio", t_mp, t_cpu, t_gpu_d, None, verified)

# ===========================================================================
# 7. virtual_temperature  (CPU + met-cu)
# ===========================================================================

# MetPy: virtual_temperature(temperature_K, mixing_ratio_kg/kg) -> K
ref_vt = mpcalc.virtual_temperature(t_q, mr_q).to("degC").magnitude

# -- metrust CPU: virtual_temperature(T_degC, mixing_ratio_kg/kg) -> degC --
mrcalc.set_backend("cpu")
mr_vt = mrcalc.virtual_temperature(t_arr, mr_arr)
if hasattr(mr_vt, "magnitude"):
    mr_vt = mr_vt.magnitude
mr_vt = np.asarray(mr_vt, dtype=np.float64)

# -- met-cu direct: virtual_temperature(T_degC, mixing_ratio_kg/kg) -> degC --
mcu_vt = None
if HAS_GPU:
    mcu_vt_raw = mcucalc.virtual_temperature(t_arr, mr_arr)
    mcu_vt = mcu_vt_raw.get() if hasattr(mcu_vt_raw, "get") else np.asarray(mcu_vt_raw)

# -- Deep verify --
verified = verify_all_backends("virtual_temperature",
                               ref_vt, mr_vt, mcu_vt, None)

# -- Timing --
t_mp = bench(lambda: mpcalc.virtual_temperature(t_q, mr_q))
mrcalc.set_backend("cpu")
t_cpu = bench(lambda: mrcalc.virtual_temperature(t_arr, mr_arr))
t_gpu_d = bench(lambda: mcucalc.virtual_temperature(t_arr, mr_arr), gpu=True) if HAS_GPU else None

record("virtual_temperature", t_mp, t_cpu, t_gpu_d, None, verified)


# ===========================================================================
# SUMMARY TABLE (timing)
# ===========================================================================
print()
print("=" * 120)
print("TIMING SUMMARY")
print("=" * 120)

hdr = f"  {'Function':42s} | {'MetPy':>10s} | {'CPU':>10s} {'(vs MP)':>7s} | {'GPU-d':>10s} {'(vs MP)':>7s} | {'GPU-mr':>10s} {'(vs MP)':>7s} | {'OK':>4s}"
print(hdr)
print("-" * 120)

n_pass = 0
n_fail = 0
for name, t_mp, t_cpu, t_gpu_d, t_gpu_mr, verified in results:
    tag = "PASS" if verified else "FAIL"
    if verified:
        n_pass += 1
    else:
        n_fail += 1
    parts = [f"  {name:42s}"]
    parts.append(f"{fmt(t_mp):>10s}")
    parts.append(f"{fmt(t_cpu):>10s} {spd(t_mp, t_cpu):>7s}")
    parts.append(f"{fmt(t_gpu_d):>10s} {spd(t_mp, t_gpu_d):>7s}" if t_gpu_d is not None else f"{'  -':>10s} {'':>7s}")
    parts.append(f"{fmt(t_gpu_mr):>10s} {spd(t_mp, t_gpu_mr):>7s}" if t_gpu_mr is not None else f"{'  -':>10s} {'':>7s}")
    parts.append(f"{tag:>4s}")
    print(" | ".join(parts))

print("-" * 120)
print(f"  Timing verification: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")


# ===========================================================================
# DEEP VERIFICATION SUMMARY TABLE
# ===========================================================================
print()
print("=" * 120)
print("DEEP VERIFICATION SUMMARY")
print("=" * 120)

# Table header
hdr2 = (f"  {'Function':36s} {'Backend':14s} | {'MeanDiff':>12s} {'MaxAbsDiff':>12s} "
        f"{'RMSE':>12s} {'99pct':>12s} {'RelRMSE%':>10s} "
        f"{'Pearson':>12s} {'>1%err':>8s} {'>0.1%err':>9s} "
        f"{'NaN':>5s} {'Inf':>5s} | {'OK':>4s}")
print(hdr2)
print("-" * 160)

deep_pass = 0
deep_fail = 0
for res in all_deep:
    tag = "PASS" if res["pass"] else "FAIL"
    if res["pass"]:
        deep_pass += 1
    else:
        deep_fail += 1

    if "mean_diff" not in res:
        print(f"  {res['func']:36s} {res['backend']:14s} | {'(no data)':>12s} "
              f"{'':>12s} {'':>12s} {'':>12s} {'':>10s} "
              f"{'':>12s} {'':>8s} {'':>9s} "
              f"{'':>5s} {'':>5s} | {tag:>4s}")
        continue

    print(f"  {res['func']:36s} {res['backend']:14s} | "
          f"{res['mean_diff']:>+12.4e} {res['max_abs_diff']:>12.4e} "
          f"{res['rmse']:>12.4e} {res['pct99_abs_diff']:>12.4e} {res['rel_rmse_pct']:>9.5f}% "
          f"{res['pearson_r']:>12.10f} {res['pct_gt_1pct_relerr']:>7.3f}% "
          f"{res['pct_gt_01pct_relerr']:>8.3f}% "
          f"{res['nan_test']:>5d} {res['inf_test']:>5d} | {tag:>4s}")

print("-" * 160)
print(f"  Deep verification: {deep_pass} passed, {deep_fail} failed out of {deep_pass + deep_fail}")

if deep_fail > 0:
    print()
    print("  ** FAILURES detected -- details above **")
    print("  Failed checks:")
    for res in all_deep:
        if not res["pass"]:
            notes = "; ".join(res.get("notes", []))
            print(f"    - {res['func']} [{res['backend']}]: {notes}")
else:
    print()
    print("  ALL DEEP VERIFICATION CHECKS PASSED")

print("=" * 120)

sys.exit(0 if deep_fail == 0 else 1)
