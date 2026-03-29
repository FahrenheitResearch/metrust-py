#!/usr/bin/env python
"""Bench-05: HRRR Supercell Environment -- REAL HRRR DATA, deep verification.

Scenario
--------
Full HRRR 3-km grid (40 levels, 1059x1799) loaded from actual GRIB2 files.
Real severe weather pipeline: CAPE/CIN, SRH, 0-6 km shear, composite
reflectivity from hydrometeors, theta-e, dewpoint -- all from genuine 3-D
model output.

Data sources
------------
- data/hrrr_prs.grib2 : 40 isobaric levels, 1059x1799
    Vars: t, u, v, q, gh, r, dpt, rwmr, snmr, grle
- data/hrrr_sfc.grib2 : surface + 2m fields
    Surface: sp, orog   (typeOfLevel='surface')
    2m:      t2m, d2m, sh2  (typeOfLevel='heightAboveGround', level=2)

Data prep
---------
T(K) -> T(C), q -> mixing ratio, height AGL = gh - orog, pressure 3D in Pa.
Surface: psfc(Pa), t2m(K), q2 = sh2/(1-sh2).

Functions benchmarked
---------------------
- compute_cape_cin                          (GPU-eligible)
- compute_srh                               (GPU-eligible)
- compute_shear                             (GPU-eligible)
- composite_reflectivity_from_hydrometeors  (GPU-eligible)
- equivalent_potential_temperature           (GPU-eligible)
- dewpoint                                  (GPU-eligible)

Backends: MetPy (thermo only, no grid composites), metrust CPU, met-cu
direct (CUDA), metrust GPU.

Timing: perf_counter, cupy sync, 1 warmup + 3 timed, median.

Verification (per-function)
---------------------------
For EVERY function:
  - mean diff, max abs diff, RMSE, 99th percentile abs diff
  - relative RMSE %
  - NaN/Inf audit
  - physical plausibility bounds
  - Pearson correlation coefficient
  - % points > 1% relative error, % points > 0.1% relative error
  - edge case checks (max CAPE column, zero-CAPE column, max refl)
  - histogram of diffs (10-bin text histogram)
  - CAPE/CIN special: distribution comparison (mean, median, std, min, max),
    flag columns where one backend finds CAPE>0 and another finds CAPE=0
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Imports
# ============================================================================
import metrust.calc as mrcalc
from metrust.units import units

HAS_METPY = True
try:
    import metpy.calc as mpcalc
    from metpy.units import units as mp_units
except ImportError:
    HAS_METPY = False

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

# ============================================================================
# Data paths
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


# ============================================================================
# Load REAL HRRR data from GRIB2
# ============================================================================

def load_hrrr_data():
    """Load real HRRR GRIB2 data and derive all fields needed for the
    full severe weather benchmark suite.

    Returns dict with all arrays as contiguous float64.
    """
    import xarray as xr

    prs_path = os.path.join(DATA_DIR, "hrrr_prs.grib2")
    sfc_path = os.path.join(DATA_DIR, "hrrr_sfc.grib2")

    for p, label in [(prs_path, "hrrr_prs.grib2"), (sfc_path, "hrrr_sfc.grib2")]:
        if not os.path.exists(p):
            sys.exit(f"HRRR data not found: {p}\n"
                     f"Place {label} in the data/ directory.")

    print("  Loading HRRR pressure levels ... ", end="", flush=True)
    t0 = time.perf_counter()

    ds3 = xr.open_dataset(prs_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                        "indexpath": ""})

    # -- Pressure coordinate (ensure surface-first = descending hPa) ----------
    plev = np.asarray(ds3.isobaricInhPa.values, dtype=np.float64)
    flip = plev[0] < plev[-1]          # need surface (high hPa) first
    if flip:
        plev = plev[::-1]

    def g3(name):
        a = np.asarray(ds3[name].values, dtype=np.float64)
        return a[::-1] if flip else a

    nz, ny, nx = g3("t").shape
    print(f"{time.perf_counter() - t0:.1f} s  ({nz} levels, {ny}x{nx})")

    # -- 3-D fields -----------------------------------------------------------
    print("  Extracting 3-D fields ... ", end="", flush=True)
    t1 = time.perf_counter()

    t_k   = g3("t")                       # K
    t_c   = t_k - 273.15                   # Celsius
    dpt_k = g3("dpt")                      # K
    dpt_c = dpt_k - 273.15                 # Celsius
    u     = g3("u")                        # m/s
    v     = g3("v")                        # m/s
    q     = g3("q")                        # specific humidity kg/kg
    gh    = g3("gh")                       # geopotential height m
    rh    = g3("r")                        # relative humidity %
    rwmr  = g3("rwmr")                     # rain water kg/kg
    snmr  = g3("snmr")                     # snow       kg/kg
    grle  = g3("grle")                     # graupel    kg/kg

    # Mixing ratio from specific humidity: w = q / (1 - q)
    w_mr = q / (1.0 - q)

    # 3-D pressure field in Pa
    p3_hPa = np.broadcast_to(plev[:, None, None], (nz, ny, nx)).copy()
    p3_Pa  = p3_hPa * 100.0

    print(f"{time.perf_counter() - t1:.1f} s")

    # -- Surface + 2m fields --------------------------------------------------
    print("  Loading surface/2m fields ... ", end="", flush=True)
    t2 = time.perf_counter()

    ds_sfc = xr.open_dataset(sfc_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface",
                                           "shortName": ["sp", "orog"]},
                        "indexpath": ""})
    ds_2m = xr.open_dataset(sfc_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround",
                                           "level": 2,
                                           "shortName": ["2t", "2d", "2sh"]},
                        "indexpath": ""})

    psfc = np.asarray(ds_sfc["sp"].values, dtype=np.float64)     # Pa
    orog = np.asarray(ds_sfc["orog"].values, dtype=np.float64)   # m
    t2m  = np.asarray(ds_2m["t2m"].values, dtype=np.float64)     # K
    d2m  = np.asarray(ds_2m["d2m"].values, dtype=np.float64)     # K
    sh2  = np.asarray(ds_2m["sh2"].values, dtype=np.float64)     # kg/kg

    # Surface mixing ratio from specific humidity
    q2 = sh2 / (1.0 - sh2)

    # Height AGL = geopotential height minus terrain
    h_agl = gh - orog[None, :, :]

    print(f"{time.perf_counter() - t2:.1f} s")

    # -- 850 hPa slice for thermo benchmarks ----------------------------------
    i850 = int(np.argmin(np.abs(plev - 850.0)))
    tc850  = t_c[i850].copy()
    dc850  = dpt_c[i850].copy()

    # Vapor pressure from dewpoint (Tetens formula) for dewpoint() benchmark
    vp850 = 6.1078 * np.exp(17.27 * dc850 / (dc850 + 237.3))

    # Close datasets
    ds3.close()
    ds_sfc.close()
    ds_2m.close()

    # Package everything as contiguous float64
    def c(a):
        return np.ascontiguousarray(a, dtype=np.float64)

    return dict(
        # Grid dimensions
        nz=nz, ny=ny, nx=nx,
        plev=plev, i850=i850,
        # 3-D fields
        p_3d_hPa=c(p3_hPa), p_3d_Pa=c(p3_Pa),
        tc_3d=c(t_c), td_3d=c(dpt_c),
        w_mr=c(w_mr), h_agl=c(h_agl),
        u_3d=c(u), v_3d=c(v),
        qrain=c(rwmr), qsnow=c(snmr), qgraup=c(grle),
        # Surface / 2m
        psfc=c(psfc), t2m=c(t2m), q2=c(q2), orog=c(orog),
        # 850 hPa slices
        tc850=c(tc850), dc850=c(dc850), vp850=c(vp850),
        # Extras
        rh=c(rh), d2m=c(d2m), sh2=c(sh2),
    )


# ============================================================================
# Timing harness
# ============================================================================

def _sync():
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()


def bench(func, n=3, gpu=False):
    """1 warmup + n timed, return median ms."""
    if gpu:
        _sync()
    func()  # warmup
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
        return "--"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd(slow, fast):
    if slow is None or fast is None or fast <= 0:
        return "--"
    r = slow / fast
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


def _to_numpy(val):
    """Convert cupy array to numpy if needed."""
    if hasattr(val, 'get'):
        return val.get()
    return val


def _strip_units(val):
    """Extract raw numpy array from a pint Quantity or cupy array."""
    if hasattr(val, "magnitude"):
        v = val.magnitude
        return np.asarray(_to_numpy(v))
    return np.asarray(_to_numpy(val))


# ============================================================================
# Deep verification engine
# ============================================================================

# Physical plausibility bounds:  (lo, hi)
PHYS_BOUNDS = {
    "cape":         (0.0, 8000.0),       # J/kg
    "cin":          (-1000.0, 0.0),      # J/kg  (real HRRR: tall mountains -> extreme CIN)
    "lcl":          (0.0, 6000.0),       # m AGL
    "lfc":          (0.0, 15000.0),      # m AGL
    "srh":          (-500.0, 1000.0),    # m^2/s^2
    "shear":        (0.0, 80.0),         # m/s
    "refl":         (-30.0, 80.0),       # dBZ  (-30 is floor sentinel from hydrometeor calc)
    "theta_e":      (260.0, 400.0),      # K  (widened for real data extremes)
    "dewpoint":     (-90.0, 40.0),       # degC  (widened for real data)
}

# Global verification ledger: list of (name, pair_label, passed_bool)
_VERIFY_LEDGER = []


def _text_histogram(diffs, name, nbins=10):
    """Print a compact 10-bin text histogram of difference values."""
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        print(f"      [{name}] histogram: no finite diffs")
        return
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo < 1e-30:
        print(f"      [{name}] histogram: all diffs ~ {lo:.4e} (constant)")
        return
    counts, edges = np.histogram(finite, bins=nbins)
    max_c = max(counts) if max(counts) > 0 else 1
    bar_width = 30
    print(f"      [{name}] diff histogram ({finite.size:,} pts):")
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / max_c * bar_width))
        pct = c / finite.size * 100
        print(f"        [{edges[i]:+10.3e}, {edges[i+1]:+10.3e}) "
              f"{bar:<{bar_width}s} {c:>8,} ({pct:5.1f}%)")


def deep_verify(name, ref, test, phys_key, label_ref="ref", label_test="test",
                rtol_pass=1e-3):
    """Full statistical comparison of two arrays.

    Returns True if the comparison passes all critical checks.
    """
    tag = f"{name} [{label_ref} vs {label_test}]"
    ref_a = np.asarray(_strip_units(ref), dtype=np.float64).ravel()
    test_a = np.asarray(_strip_units(test), dtype=np.float64).ravel()

    if ref_a.shape != test_a.shape:
        msg = f"    {tag}: FAIL  SHAPE MISMATCH {ref_a.shape} vs {test_a.shape}"
        print(msg)
        _VERIFY_LEDGER.append((name, f"{label_ref} vs {label_test}", False))
        return False

    n_total = ref_a.size
    passed = True

    # --- NaN / Inf audit -----------------------------------------------------
    nan_ref = int(np.isnan(ref_a).sum())
    nan_test = int(np.isnan(test_a).sum())
    inf_ref = int(np.isinf(ref_a).sum())
    inf_test = int(np.isinf(test_a).sum())
    nan_ok = (nan_ref == nan_test) and (inf_ref == 0) and (inf_test == 0)
    if not nan_ok:
        passed = False
    print(f"      NaN/Inf audit: ref NaN={nan_ref:,} Inf={inf_ref:,}  "
          f"test NaN={nan_test:,} Inf={inf_test:,}  "
          f"{'OK' if nan_ok else 'MISMATCH'}")

    # Work on finite-both mask
    mask = np.isfinite(ref_a) & np.isfinite(test_a)
    n_valid = int(mask.sum())
    if n_valid == 0:
        print(f"    {tag}: SKIP  no valid points")
        _VERIFY_LEDGER.append((name, f"{label_ref} vs {label_test}", True))
        return True

    r = ref_a[mask]
    t = test_a[mask]
    diffs = t - r

    # --- Basic statistics ----------------------------------------------------
    mean_diff = float(np.mean(diffs))
    max_abs_diff = float(np.max(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    pct99 = float(np.percentile(np.abs(diffs), 99))

    # Relative RMSE%: RMSE / mean(|ref|), guarded against zero
    mean_abs_ref = float(np.mean(np.abs(r)))
    rel_rmse_pct = (rmse / mean_abs_ref * 100) if mean_abs_ref > 1e-12 else 0.0

    print(f"      mean_diff={mean_diff:+.4e}  max_abs_diff={max_abs_diff:.4e}  "
          f"RMSE={rmse:.4e}  99th_pct={pct99:.4e}  "
          f"rel_RMSE={rel_rmse_pct:.4f}%")

    # --- Pearson correlation -------------------------------------------------
    if n_valid > 1 and np.std(r) > 1e-12 and np.std(t) > 1e-12:
        pearson_r, pearson_p = sp_stats.pearsonr(r, t)
    else:
        pearson_r, pearson_p = 1.0, 0.0
    print(f"      Pearson r={pearson_r:.8f}  (p={pearson_p:.2e})")
    if pearson_r < 0.999:
        passed = False

    # --- Relative error percentages ------------------------------------------
    denom = np.abs(r)
    denom_safe = np.where(denom > 1e-10, denom, 1e-10)
    rel_err = np.abs(diffs) / denom_safe
    pct_gt_1pct = float(np.mean(rel_err > 0.01) * 100)
    pct_gt_01pct = float(np.mean(rel_err > 0.001) * 100)
    print(f"      %pts >1% rel err: {pct_gt_1pct:.2f}%   "
          f"%pts >0.1% rel err: {pct_gt_01pct:.2f}%")

    # Check median relative error against rtol
    median_rel = float(np.median(rel_err))
    if median_rel > rtol_pass:
        passed = False

    # --- Physical plausibility -----------------------------------------------
    lo, hi = PHYS_BOUNDS.get(phys_key, (-1e30, 1e30))
    oob_ref = int(np.sum((r < lo) | (r > hi)))
    oob_test = int(np.sum((t < lo) | (t > hi)))
    phys_ok = (oob_test == 0)
    # Be lenient: only fail if test has OOB and ref does not
    if oob_test > 0 and oob_ref == 0:
        passed = False
    print(f"      Physical [{phys_key}] bounds [{lo}, {hi}]: "
          f"ref OOB={oob_ref:,}  test OOB={oob_test:,}  "
          f"{'OK' if phys_ok else 'WARN'}")

    # --- Histogram of diffs --------------------------------------------------
    _text_histogram(diffs, tag)

    # --- Verdict -------------------------------------------------------------
    verdict = "PASS" if passed else "FAIL"
    print(f"    {tag}: {verdict}  "
          f"(median_rel={median_rel:.2e}, rtol={rtol_pass:.0e})")
    _VERIFY_LEDGER.append((name, f"{label_ref} vs {label_test}", passed))
    return passed


def deep_verify_tuple(name, refs, tests, phys_keys, field_names,
                      label_ref="ref", label_test="test", rtol_pass=1e-3):
    """Deep-verify each element of a tuple result."""
    all_ok = True
    for i, fn in enumerate(field_names):
        r = refs[i] if i < len(refs) else None
        t = tests[i] if i < len(tests) else None
        if r is None or t is None:
            continue
        pk = phys_keys[i] if i < len(phys_keys) else "cape"
        ok = deep_verify(f"{name}.{fn}", r, t, pk, label_ref, label_test,
                         rtol_pass)
        all_ok = all_ok and ok
    return all_ok


# ============================================================================
# CAPE/CIN special analysis
# ============================================================================

def cape_distribution_analysis(cape_dict):
    """Compare distributions of CAPE/CIN across backends.

    cape_dict: {backend_name: (cape, cin, lcl, lfc)} with raw numpy arrays.
    """
    print("\n      --- CAPE/CIN Distribution Comparison ---")
    backends = list(cape_dict.keys())
    if len(backends) < 2:
        print("      (need >=2 backends for comparison)")
        return

    # Distribution stats table
    header_line = (f"      {'Backend':<12s} {'field':<6s} {'mean':>10s} "
                   f"{'median':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    print(header_line)
    print("      " + "-" * len(header_line.strip()))

    for bk in backends:
        for fi, fn in enumerate(["cape", "cin"]):
            arr = np.asarray(_strip_units(cape_dict[bk][fi]),
                             dtype=np.float64).ravel()
            fin = arr[np.isfinite(arr)]
            if fin.size == 0:
                print(f"      {bk:<12s} {fn:<6s} "
                      f"{'--':>10s} {'--':>10s} {'--':>10s} "
                      f"{'--':>10s} {'--':>10s}")
                continue
            print(f"      {bk:<12s} {fn:<6s} "
                  f"{np.mean(fin):10.2f} {np.median(fin):10.2f} "
                  f"{np.std(fin):10.2f} {np.min(fin):10.2f} "
                  f"{np.max(fin):10.2f}")

    # --- Flag columns where one backend finds CAPE>0 and another finds CAPE=0
    print("\n      --- CAPE zero/nonzero disagreement ---")
    ref_name = backends[0]
    cape_ref = np.asarray(_strip_units(cape_dict[ref_name][0]),
                          dtype=np.float64).ravel()
    total_cols = cape_ref.size
    for bk in backends[1:]:
        cape_other = np.asarray(_strip_units(cape_dict[bk][0]),
                                dtype=np.float64).ravel()
        if cape_ref.size != cape_other.size:
            print(f"      {ref_name} vs {bk}: SIZE MISMATCH, cannot compare")
            continue

        ref_pos = cape_ref > 0
        oth_pos = cape_other > 0

        # Ref has CAPE>0 but other has CAPE==0
        disagree_a = ref_pos & (~oth_pos)
        count_a = int(np.sum(disagree_a))
        # Other has CAPE>0 but ref has CAPE==0
        disagree_b = (~ref_pos) & oth_pos
        count_b = int(np.sum(disagree_b))

        total_disagree = count_a + count_b
        pct = total_disagree / total_cols * 100 if total_cols > 0 else 0
        tag = "OK" if total_disagree == 0 else "FLAG"
        print(f"      {ref_name} vs {bk}: "
              f"{count_a:,} cols {ref_name}>0 & {bk}==0, "
              f"{count_b:,} cols {bk}>0 & {ref_name}==0  "
              f"({pct:.3f}% of {total_cols:,})  [{tag}]")

        # Show worst disagreement columns (up to 5)
        if total_disagree > 0:
            disagree_mask = disagree_a | disagree_b
            disagree_idx = np.where(disagree_mask)[0]
            show = min(5, len(disagree_idx))
            diff_at_disagree = np.abs(cape_ref[disagree_idx] -
                                      cape_other[disagree_idx])
            order = np.argsort(-diff_at_disagree)[:show]
            for rank, oi in enumerate(order):
                idx = disagree_idx[oi]
                print(f"        #{rank+1} col {idx}: "
                      f"{ref_name}={cape_ref[idx]:.1f}  "
                      f"{bk}={cape_other[idx]:.1f}")


# ============================================================================
# Edge case analysis
# ============================================================================

def edge_case_analysis(name, results_dict, phys_key, find_max=True):
    """Examine specific columns: max-value column, zero-value column, etc."""
    print(f"\n      --- Edge cases for {name} ---")
    backends = list(results_dict.keys())
    if len(backends) == 0:
        return

    ref_name = backends[0]
    ref_arr = np.asarray(_strip_units(results_dict[ref_name]),
                         dtype=np.float64).ravel()

    finite = ref_arr[np.isfinite(ref_arr)]
    if finite.size == 0:
        print("      No finite values in reference.")
        return

    # Max-value column
    max_idx = int(np.nanargmax(ref_arr))
    print(f"      Max-value column (idx={max_idx}):")
    for bk in backends:
        arr = np.asarray(_strip_units(results_dict[bk]),
                         dtype=np.float64).ravel()
        val = arr[max_idx] if max_idx < arr.size else float('nan')
        print(f"        {bk}: {val:.4f}")

    # Min-value or zero column
    abs_ref = np.abs(ref_arr)
    abs_ref[~np.isfinite(abs_ref)] = 1e30
    min_idx = int(np.argmin(abs_ref))
    print(f"      Min-abs-value column (idx={min_idx}):")
    for bk in backends:
        arr = np.asarray(_strip_units(results_dict[bk]),
                         dtype=np.float64).ravel()
        val = arr[min_idx] if min_idx < arr.size else float('nan')
        print(f"        {bk}: {val:.6e}")

    # Column with max inter-backend disagreement
    if len(backends) >= 2:
        other_name = backends[1]
        other_arr = np.asarray(_strip_units(results_dict[other_name]),
                               dtype=np.float64).ravel()
        if ref_arr.size == other_arr.size:
            abs_diff = np.abs(ref_arr - other_arr)
            abs_diff[~np.isfinite(abs_diff)] = 0
            worst_idx = int(np.argmax(abs_diff))
            print(f"      Worst-disagreement column (idx={worst_idx}, "
                  f"{ref_name} vs {other_name}):")
            for bk in backends:
                arr = np.asarray(_strip_units(results_dict[bk]),
                                 dtype=np.float64).ravel()
                val = arr[worst_idx] if worst_idx < arr.size else float('nan')
                print(f"        {bk}: {val:.4f}")


# ============================================================================
# Display
# ============================================================================

COL_NAME = 40
COL_T = 12
COL_S = 10

rows = []


def header():
    print(f"\n  {'':2s} {'Function':{COL_NAME}s}"
          f" {'MetPy':>{COL_T}s}"
          f" {'Rust/CPU':>{COL_T}s}"
          f" {'met-cu':>{COL_T}s}"
          f" {'Rust/GPU':>{COL_T}s}"
          f" {'CPU/MetPy':>{COL_S}s}"
          f" {'GPU/CPU':>{COL_S}s}")
    print("  " + "-" * (COL_NAME + 4 * COL_T + 2 * COL_S + 8))


def record(name, t_mp, t_cpu, t_mcu, t_gpu):
    rows.append((name, t_mp, t_cpu, t_mcu, t_gpu))
    print(f"  * {name:{COL_NAME}s}"
          f" {fmt(t_mp):>{COL_T}s}"
          f" {fmt(t_cpu):>{COL_T}s}"
          f" {fmt(t_mcu):>{COL_T}s}"
          f" {fmt(t_gpu):>{COL_T}s}"
          f" {spd(t_mp, t_cpu):>{COL_S}s}"
          f" {spd(t_cpu, t_gpu):>{COL_S}s}")


# ============================================================================
# Backend wrappers
# ============================================================================

def _run_metpy(func, n=3):
    if not HAS_METPY:
        return None
    try:
        return bench(func, n=n)
    except Exception as e:
        print(f"    [MetPy error: {e}]")
        return None


def _run_cpu(func, n=3):
    mrcalc.set_backend("cpu")
    return bench(func, n=n)


def _run_mcu(func, n=3):
    """Run met-cu (direct CUDA) benchmark."""
    if not HAS_GPU:
        return None
    try:
        return bench(func, n=n, gpu=True)
    except Exception as e:
        print(f"    [met-cu error: {e}]")
        return None


def _run_gpu(func, n=3):
    """Run metrust GPU backend benchmark."""
    if not HAS_GPU:
        return None
    mrcalc.set_backend("gpu")
    t = bench(func, n=n, gpu=True)
    mrcalc.set_backend("cpu")
    return t


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 110)
    print("  BENCH-05: HRRR Supercell -- REAL HRRR DATA  (DEEP VERIFICATION)")
    print(f"  GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print(f"  MetPy: {'available' if HAS_METPY else 'not available'}")
    print("=" * 110)

    # ==================================================================
    # Load real HRRR data
    # ==================================================================
    print("\n  Loading real HRRR GRIB2 data ...")
    t_load = time.perf_counter()
    d = load_hrrr_data()
    load_s = time.perf_counter() - t_load
    nz, ny, nx = d['nz'], d['ny'], d['nx']

    print(f"\n  Data loaded in {load_s:.1f} s")
    print(f"  Grid: {nz} levels x {ny}x{nx} = {nz*ny*nx:,} 3-D points, "
          f"{ny*nx:,} columns")

    # Sanity-check the real data
    print(f"\n  --- Real HRRR Data Summary ---")
    print(f"  Pressure levels (hPa): {d['plev'][0]:.0f} .. {d['plev'][-1]:.0f}")
    print(f"  T surface range: {d['tc_3d'][0].min():.1f} to "
          f"{d['tc_3d'][0].max():.1f} C")
    print(f"  T 500hPa range: "
          f"{d['tc_3d'][int(np.argmin(np.abs(d['plev']-500)))].min():.1f} to "
          f"{d['tc_3d'][int(np.argmin(np.abs(d['plev']-500)))].max():.1f} C")
    print(f"  Td surface range: {d['td_3d'][0].min():.1f} to "
          f"{d['td_3d'][0].max():.1f} C")
    print(f"  T2m range: {d['t2m'].min():.1f} to {d['t2m'].max():.1f} K")
    print(f"  Psfc range: {d['psfc'].min()/100:.0f} to "
          f"{d['psfc'].max()/100:.0f} hPa")
    print(f"  Terrain range: {d['orog'].min():.0f} to "
          f"{d['orog'].max():.0f} m")
    print(f"  h_agl bot range: {d['h_agl'][0].min():.0f} to "
          f"{d['h_agl'][0].max():.0f} m")
    print(f"  h_agl top range: {d['h_agl'][-1].min():.0f} to "
          f"{d['h_agl'][-1].max():.0f} m")
    print(f"  Mixing ratio sfc max: {d['w_mr'][0].max()*1000:.2f} g/kg")
    print(f"  qrain max: {d['qrain'].max()*1000:.4f} g/kg")
    print(f"  qsnow max: {d['qsnow'].max()*1000:.4f} g/kg")
    print(f"  qgraup max: {d['qgraup'].max()*1000:.4f} g/kg")

    # Approx bulk shear
    u_sfc_mean = float(np.mean(d['u_3d'][0]))
    v_sfc_mean = float(np.mean(d['v_3d'][0]))
    u_top_mean = float(np.mean(d['u_3d'][-1]))
    v_top_mean = float(np.mean(d['v_3d'][-1]))
    shear_approx = np.sqrt((u_top_mean - u_sfc_mean)**2 +
                           (v_top_mean - v_sfc_mean)**2)
    print(f"  Approx 0-top shear: {shear_approx:.1f} m/s "
          f"({shear_approx * 1.944:.0f} kt)")

    N = 3  # timed iterations

    # ==================================================================
    # Prepare Pint quantities for MetPy (not timed)
    # ==================================================================
    if HAS_METPY:
        mp_p850 = 850.0 * mp_units.hPa
        mp_tc850 = d['tc850'] * mp_units.degC
        mp_td850 = d['dc850'] * mp_units.degC
        mp_vp850 = d['vp850'] * mp_units.hPa

    # ==================================================================
    # SECTION 1: THERMODYNAMICS (2D slice at 850 hPa)
    # ==================================================================
    print("\n" + "=" * 110)
    print(f"  SECTION 1: THERMODYNAMICS (2D slice at 850 hPa, {ny}x{nx} "
          f"= {ny*nx:,} points)")
    header()

    # --- equivalent_potential_temperature ---
    tc_slice = d['tc850']
    td_slice = d['dc850']
    p850_2d = np.full_like(tc_slice, 850.0)

    t_mp = _run_metpy(
        lambda: mpcalc.equivalent_potential_temperature(
            mp_p850, mp_tc850, mp_td850), N) if HAS_METPY else None

    t_cpu = _run_cpu(
        lambda: mrcalc.equivalent_potential_temperature(
            850.0, tc_slice, td_slice), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.equivalent_potential_temperature(
            p850_2d, tc_slice, td_slice), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.equivalent_potential_temperature(
            850.0, tc_slice, td_slice), N)

    record("equivalent_potential_temperature", t_mp, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: theta-e --
    print("\n    [Deep Verification: theta_e]")
    mrcalc.set_backend("cpu")
    ref_cpu = mrcalc.equivalent_potential_temperature(850.0, tc_slice, td_slice)
    theta_e_results = {"CPU": ref_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        ref_gpu = mrcalc.equivalent_potential_temperature(
            850.0, tc_slice, td_slice)
        mrcalc.set_backend("cpu")
        theta_e_results["GPU"] = ref_gpu
        deep_verify("theta_e", ref_cpu, ref_gpu, "theta_e", "CPU", "GPU",
                    rtol_pass=1e-4)

        ref_mcu = mcucalc.equivalent_potential_temperature(
            p850_2d, tc_slice, td_slice)
        theta_e_results["met-cu"] = ref_mcu
        deep_verify("theta_e", ref_cpu, ref_mcu, "theta_e", "CPU", "met-cu",
                    rtol_pass=1e-4)

    if HAS_METPY:
        ref_mp = mpcalc.equivalent_potential_temperature(
            mp_p850, mp_tc850, mp_td850)
        theta_e_results["MetPy"] = ref_mp
        deep_verify("theta_e", ref_cpu, ref_mp, "theta_e", "CPU", "MetPy",
                    rtol_pass=1e-4)

    edge_case_analysis("theta_e", theta_e_results, "theta_e")

    # --- dewpoint ---
    vp_slice = d['vp850']

    t_mp = _run_metpy(
        lambda: mpcalc.dewpoint(mp_vp850), N) if HAS_METPY else None

    t_cpu = _run_cpu(
        lambda: mrcalc.dewpoint(vp_slice), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.dewpoint(vp_slice), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.dewpoint(vp_slice), N)

    record("dewpoint", t_mp, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: dewpoint --
    print("\n    [Deep Verification: dewpoint]")
    mrcalc.set_backend("cpu")
    ref_cpu = mrcalc.dewpoint(vp_slice)
    dp_results = {"CPU": ref_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        ref_gpu = mrcalc.dewpoint(vp_slice)
        mrcalc.set_backend("cpu")
        dp_results["GPU"] = ref_gpu
        deep_verify("dewpoint", ref_cpu, ref_gpu, "dewpoint", "CPU", "GPU",
                    rtol_pass=1e-4)

        ref_mcu = mcucalc.dewpoint(vp_slice)
        dp_results["met-cu"] = ref_mcu
        deep_verify("dewpoint", ref_cpu, ref_mcu, "dewpoint", "CPU", "met-cu",
                    rtol_pass=1e-4)

    if HAS_METPY:
        ref_mp = mpcalc.dewpoint(mp_vp850)
        dp_results["MetPy"] = ref_mp
        deep_verify("dewpoint", ref_cpu, ref_mp, "dewpoint", "CPU", "MetPy",
                    rtol_pass=1e-4)

    edge_case_analysis("dewpoint", dp_results, "dewpoint")

    # ==================================================================
    # SECTION 2: GRID COMPOSITES (full 3D -> 2D, 3 backends)
    # ==================================================================
    print("\n" + "=" * 110)
    print(f"  SECTION 2: GRID COMPOSITES ({nz}x{ny}x{nx} -> {ny}x{nx})")
    header()

    # --- compute_cape_cin ---
    t_cpu = _run_cpu(
        lambda: mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2m'], d['q2']), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2m'], d['q2']), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2m'], d['q2']), N)

    record("compute_cape_cin", None, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: CAPE/CIN --
    print("\n    [Deep Verification: CAPE/CIN]")
    mrcalc.set_backend("cpu")
    cape_cpu = mrcalc.compute_cape_cin(
        d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
        d['psfc'], d['t2m'], d['q2'])
    cape_all = {"CPU": cape_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        cape_gpu = mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2m'], d['q2'])
        mrcalc.set_backend("cpu")
        cape_all["GPU"] = cape_gpu
        deep_verify_tuple("cape_cin", cape_cpu, cape_gpu,
                          ["cape", "cin", "lcl", "lfc"],
                          ["cape", "cin", "lcl", "lfc"],
                          "CPU", "GPU", rtol_pass=1e-3)

        cape_mcu = mcucalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2m'], d['q2'])
        cape_all["met-cu"] = cape_mcu
        deep_verify_tuple("cape_cin", cape_cpu, cape_mcu,
                          ["cape", "cin", "lcl", "lfc"],
                          ["cape", "cin", "lcl", "lfc"],
                          "CPU", "met-cu", rtol_pass=1e-3)

    # CAPE/CIN distribution analysis
    cape_distribution_analysis(cape_all)

    # Edge case: max CAPE column, zero-CAPE column
    cape_results_flat = {"CPU": cape_cpu[0]}
    if HAS_GPU:
        cape_results_flat["GPU"] = cape_gpu[0]
        cape_results_flat["met-cu"] = cape_mcu[0]
    edge_case_analysis("cape", cape_results_flat, "cape")

    cin_results_flat = {"CPU": cape_cpu[1]}
    if HAS_GPU:
        cin_results_flat["GPU"] = cape_gpu[1]
        cin_results_flat["met-cu"] = cape_mcu[1]
    edge_case_analysis("cin", cin_results_flat, "cin")

    # --- compute_srh ---
    t_cpu = _run_cpu(
        lambda: mrcalc.compute_srh(
            d['u_3d'], d['v_3d'], d['h_agl']), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.compute_srh(
            d['u_3d'], d['v_3d'], d['h_agl']), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.compute_srh(
            d['u_3d'], d['v_3d'], d['h_agl']), N)

    record("compute_srh", None, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: SRH --
    print("\n    [Deep Verification: SRH]")
    mrcalc.set_backend("cpu")
    srh_cpu = mrcalc.compute_srh(d['u_3d'], d['v_3d'], d['h_agl'])
    srh_results = {"CPU": srh_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        srh_gpu = mrcalc.compute_srh(d['u_3d'], d['v_3d'], d['h_agl'])
        mrcalc.set_backend("cpu")
        srh_results["GPU"] = srh_gpu
        deep_verify("srh", srh_cpu, srh_gpu, "srh", "CPU", "GPU",
                    rtol_pass=1e-3)

        srh_mcu = mcucalc.compute_srh(d['u_3d'], d['v_3d'], d['h_agl'])
        srh_results["met-cu"] = srh_mcu
        deep_verify("srh", srh_cpu, srh_mcu, "srh", "CPU", "met-cu",
                    rtol_pass=1e-3)

    edge_case_analysis("srh", srh_results, "srh")

    # --- compute_shear ---
    t_cpu = _run_cpu(
        lambda: mrcalc.compute_shear(
            d['u_3d'], d['v_3d'], d['h_agl']), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.compute_shear(
            d['u_3d'], d['v_3d'], d['h_agl']), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.compute_shear(
            d['u_3d'], d['v_3d'], d['h_agl']), N)

    record("compute_shear", None, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: shear --
    print("\n    [Deep Verification: shear]")
    mrcalc.set_backend("cpu")
    shear_cpu = mrcalc.compute_shear(d['u_3d'], d['v_3d'], d['h_agl'])
    shear_results = {"CPU": shear_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        shear_gpu = mrcalc.compute_shear(d['u_3d'], d['v_3d'], d['h_agl'])
        mrcalc.set_backend("cpu")
        shear_results["GPU"] = shear_gpu
        deep_verify("shear", shear_cpu, shear_gpu, "shear", "CPU", "GPU",
                    rtol_pass=1e-3)

        shear_mcu = mcucalc.compute_shear(d['u_3d'], d['v_3d'], d['h_agl'])
        shear_results["met-cu"] = shear_mcu
        deep_verify("shear", shear_cpu, shear_mcu, "shear", "CPU", "met-cu",
                    rtol_pass=1e-3)

    edge_case_analysis("shear", shear_results, "shear")

    # --- composite_reflectivity_from_hydrometeors ---
    t_cpu = _run_cpu(
        lambda: mrcalc.composite_reflectivity_from_hydrometeors(
            d['p_3d_Pa'], d['tc_3d'],
            d['qrain'], d['qsnow'], d['qgraup']), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.composite_reflectivity_from_hydrometeors(
            d['p_3d_Pa'], d['tc_3d'],
            d['qrain'], d['qsnow'], d['qgraup']), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.composite_reflectivity_from_hydrometeors(
            d['p_3d_Pa'], d['tc_3d'],
            d['qrain'], d['qsnow'], d['qgraup']), N)

    record("composite_refl_hydrometeors", None, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: composite reflectivity --
    print("\n    [Deep Verification: composite reflectivity]")
    mrcalc.set_backend("cpu")
    refl_cpu = mrcalc.composite_reflectivity_from_hydrometeors(
        d['p_3d_Pa'], d['tc_3d'],
        d['qrain'], d['qsnow'], d['qgraup'])
    refl_results = {"CPU": refl_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        refl_gpu = mrcalc.composite_reflectivity_from_hydrometeors(
            d['p_3d_Pa'], d['tc_3d'],
            d['qrain'], d['qsnow'], d['qgraup'])
        mrcalc.set_backend("cpu")
        refl_results["GPU"] = refl_gpu
        deep_verify("composite_refl", refl_cpu, refl_gpu, "refl",
                    "CPU", "GPU", rtol_pass=1e-3)

        refl_mcu = mcucalc.composite_reflectivity_from_hydrometeors(
            d['p_3d_Pa'], d['tc_3d'],
            d['qrain'], d['qsnow'], d['qgraup'])
        refl_results["met-cu"] = refl_mcu
        deep_verify("composite_refl", refl_cpu, refl_mcu, "refl",
                    "CPU", "met-cu", rtol_pass=1e-3)

    edge_case_analysis("composite_refl", refl_results, "refl")

    # ==================================================================
    # TIMING SUMMARY
    # ==================================================================
    print("\n" + "=" * 110)
    print("  TIMING SUMMARY")
    print("=" * 110)

    total_cpu = 0.0
    total_gpu = 0.0
    total_mcu = 0.0

    for name, t_mp, t_cpu_r, t_mcu_r, t_gpu_r in rows:
        if t_cpu_r is not None:
            total_cpu += t_cpu_r
        if t_gpu_r is not None:
            total_gpu += t_gpu_r
        if t_mcu_r is not None:
            total_mcu += t_mcu_r

    print(f"  Total Rust/CPU:  {fmt(total_cpu)}")
    if total_mcu > 0:
        print(f"  Total met-cu:    {fmt(total_mcu)}")
    if total_gpu > 0:
        print(f"  Total Rust/GPU:  {fmt(total_gpu)}")
        print(f"  Overall CPU->GPU speedup: {spd(total_cpu, total_gpu)}")
        if total_mcu > 0:
            print(f"  Overall CPU->met-cu speedup: {spd(total_cpu, total_mcu)}")

    # ==================================================================
    # VERIFICATION SUMMARY TABLE
    # ==================================================================
    print("\n" + "=" * 110)
    print("  VERIFICATION SUMMARY")
    print("=" * 110)

    n_pass = sum(1 for _, _, p in _VERIFY_LEDGER if p)
    n_fail = sum(1 for _, _, p in _VERIFY_LEDGER if not p)
    n_total_checks = len(_VERIFY_LEDGER)

    print(f"\n  {'Check':<40s} {'Pair':<25s} {'Result':<8s}")
    print("  " + "-" * 73)
    for check_name, pair_label, passed in _VERIFY_LEDGER:
        tag = "PASS" if passed else "FAIL"
        print(f"  {check_name:<40s} {pair_label:<25s} {tag:<8s}")

    print("\n  " + "-" * 73)
    print(f"  TOTAL: {n_pass} PASS, {n_fail} FAIL "
          f"out of {n_total_checks} checks")

    if n_fail > 0:
        print("\n  *** FAILURES DETECTED ***")
        for check_name, pair_label, passed in _VERIFY_LEDGER:
            if not passed:
                print(f"    FAIL: {check_name} ({pair_label})")
    else:
        print("\n  ALL CHECKS PASSED")

    # ==================================================================
    # PHYSICAL PLAUSIBILITY FINAL REPORT
    # ==================================================================
    print("\n" + "=" * 110)
    print("  PHYSICAL PLAUSIBILITY FINAL REPORT")
    print("=" * 110)

    mrcalc.set_backend("cpu")
    cape_vals = mrcalc.compute_cape_cin(
        d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
        d['psfc'], d['t2m'], d['q2'])
    cape_arr = _strip_units(cape_vals[0])
    cin_arr = _strip_units(cape_vals[1])
    srh_arr = _strip_units(mrcalc.compute_srh(
        d['u_3d'], d['v_3d'], d['h_agl']))
    shear_arr = _strip_units(mrcalc.compute_shear(
        d['u_3d'], d['v_3d'], d['h_agl']))
    refl_arr = _strip_units(mrcalc.composite_reflectivity_from_hydrometeors(
        d['p_3d_Pa'], d['tc_3d'],
        d['qrain'], d['qsnow'], d['qgraup']))
    theta_e_arr = _strip_units(mrcalc.equivalent_potential_temperature(
        850.0, tc_slice, td_slice))
    dp_arr = _strip_units(mrcalc.dewpoint(vp_slice))

    fields = [
        ("CAPE",        cape_arr,    "cape",     "J/kg"),
        ("CIN",         cin_arr,     "cin",      "J/kg"),
        ("SRH",         srh_arr,     "srh",      "m^2/s^2"),
        ("0-6km shear", shear_arr,   "shear",    "m/s"),
        ("Comp refl",   refl_arr,    "refl",     "dBZ"),
        ("theta-e",     theta_e_arr, "theta_e",  "K"),
        ("dewpoint",    dp_arr,      "dewpoint", "degC"),
    ]

    any_phys_fail = False
    print(f"\n  {'Field':<16s} {'min':>12s} {'max':>12s} {'median':>12s} "
          f"{'Bounds':>20s} {'OOB':>8s} {'Status':<8s}")
    print("  " + "-" * 90)

    for fname, arr, pk, unit_str in fields:
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            print(f"  {fname:<16s} {'--':>12s} {'--':>12s} {'--':>12s} "
                  f"{'--':>20s} {'--':>8s} {'SKIP':<8s}")
            continue
        lo, hi = PHYS_BOUNDS.get(pk, (-1e30, 1e30))
        oob = int(np.sum((fin < lo) | (fin > hi)))
        status = "PASS" if oob == 0 else "FAIL"
        if oob > 0:
            any_phys_fail = True
        print(f"  {fname:<16s} {np.min(fin):12.2f} {np.max(fin):12.2f} "
              f"{np.median(fin):12.2f} "
              f"{'[' + str(lo) + ', ' + str(hi) + ']':>20s} "
              f"{oob:>8,} {status:<8s}")

    if any_phys_fail:
        print("\n  *** PHYSICAL PLAUSIBILITY FAILURES ***")
    else:
        print("\n  All fields within physical bounds.")

    print("\n  Done.")
    print("=" * 110)

    # Return exit code
    return 1 if n_fail > 0 or any_phys_fail else 0


if __name__ == "__main__":
    sys.exit(main())
