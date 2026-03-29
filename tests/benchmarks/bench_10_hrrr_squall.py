#!/usr/bin/env python
"""Benchmark 10: HRRR Squall Line / MCS Dynamics

Scenario
--------
Prefrontal squall line environment on a 3 km HRRR-like grid (700x700,
30 vertical levels).  Realistic fields include a strong south-southwesterly
low-level jet (20-30 m/s at 850 hPa), veering winds with height, a
theta gradient marking the frontal convergence zone, deep-layer shear,
and conditional instability for CAPE.

Functions benchmarked
---------------------
  q_vector            (GPU)  - Q-vector forcing
  frontogenesis       (GPU)  - Petterssen frontogenesis
  vorticity           (GPU)  - relative vorticity
  compute_cape_cin    (GPU)  - 3-D grid CAPE/CIN
  compute_shear       (GPU)  - 0-6 km bulk shear
  potential_temperature (GPU) - potential temperature

4 backends: MetPy, metrust CPU, met-cu (direct CUDA), metrust GPU.
MetPy does NOT have compute_cape_cin / compute_shear grid versions,
so those compare only 3 backends.

Verification: deep data-correctness audit per function --
  mean diff, max abs diff, RMSE, 99th percentile, relative RMSE%,
  NaN/Inf audit, physical plausibility bounds, Pearson r,
  percentage of points exceeding 1% and 0.1% relative error,
  histogram of absolute diffs.
Special CAPE/CIN cross-backend distribution comparison.

Timing: perf_counter with cupy synchronization, 1 warmup + 3, median.

Usage:
    python tests/benchmarks/bench_10_hrrr_squall.py
    python tests/benchmarks/bench_10_hrrr_squall.py --no-metpy
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

USE_METPY = "--no-metpy" not in sys.argv

# ============================================================================
# Imports
# ============================================================================
import metrust.calc as mrcalc

if USE_METPY:
    import metpy.calc as mpcalc
    from metpy.units import units as mpunits

HAS_GPU = False
GPU_NAME = "n/a"
try:
    import cupy as cp
    mrcalc.set_backend("gpu")
    mrcalc.set_backend("cpu")
    HAS_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
except Exception:
    pass

HAS_METCU = False
try:
    import metcu.calc as mcucalc
    HAS_METCU = True
except Exception:
    pass

# ============================================================================
# Grid configuration
# ============================================================================
NY, NX = 700, 700
NZ = 30
DX = DY = 3000.0  # meters (HRRR native)

WARMUP = 1
NRUNS = 3
RTOL_KINE = 1e-4
RTOL_CAPE = 1e-3

# ============================================================================
# Physical plausibility bounds per function
# ============================================================================
PHYS_BOUNDS = {
    "potential_temperature": (280.0, 360.0, "K"),
    # On a 3-km grid with 1.5 m/s noise, dv/dx over one grid cell
    # can produce vorticity ~ 1.5/(3000)~5e-4 per noise perturbation,
    # and cumulative gradients across the jet push towards 5e-3.
    "vorticity":            (-5e-3, 5e-3, "1/s"),
    # Petterssen frontogenesis on a 3-km squall line with 12 K
    # north-south gradient and 25+ m/s jet: a few grid points
    # can exceed 1e-6 K/m/s; allow up to 2e-6.
    "frontogenesis":        (-2e-6, 2e-6, "K/m/s"),
    # Q-vector magnitudes on a 3-km squall-line grid with strong frontal
    # gradients routinely reach 1e-9 to 1e-8; allow up to 1e-7.
    "q_vector":             (-1e-7, 1e-7, "Q (m/kg/s)"),
    "compute_cape_cin_CAPE": (0.0, 6000.0, "J/kg"),
    "compute_cape_cin_CIN":  (-500.0, 0.0, "J/kg"),
    "compute_shear":         (0.0, 60.0, "m/s"),
    "theta":                 (280.0, 360.0, "K"),
}

# ============================================================================
# Synthetic squall-line data
# ============================================================================

def build_squall_data():
    """Generate realistic prefrontal squall-line environment."""
    rng = np.random.default_rng(42)

    # -- pressure levels (surface-first, 1000..250 hPa) ----------------------
    plev = np.linspace(1000.0, 250.0, NZ)  # hPa, decreasing upward

    # -- height AGL (hydrostatic approximation) -------------------------------
    # Scale height ~ 8.5 km
    h_1d = -8500.0 * np.log(plev / plev[0])            # m AGL approx
    height_agl = np.broadcast_to(
        h_1d[:, None, None], (NZ, NY, NX)
    ).copy()

    # -- temperature (C): warm sector south, cold air north -------------------
    # Lapse rate ~ 6.5 C/km, frontal gradient in y
    y_frac = np.linspace(0, 1, NY)  # 0=south, 1=north
    base_t_sfc = 30.0 - 12.0 * y_frac  # 30 C south -> 18 C north at sfc
    lapse = 6.5 / 1000.0  # C per meter
    t_c = np.empty((NZ, NY, NX), dtype=np.float64)
    for k in range(NZ):
        t_c[k] = base_t_sfc[None, :] - lapse * h_1d[k]
    # Add small noise so gradients are not perfectly smooth
    t_c += rng.normal(0, 0.3, t_c.shape)

    # -- potential temperature at 850 hPa (for 2-D kinematic tests) -----------
    i850 = int(np.argmin(np.abs(plev - 850.0)))
    t_k_850 = t_c[i850] + 273.15
    theta_850 = t_k_850 * (1000.0 / 850.0) ** 0.2854

    # -- winds: veering with height, strong LLJ at 850 hPa -------------------
    u_profile = np.array([
        2 + 4 * (k / (NZ - 1)) ** 0.5 + 25 * (k / (NZ - 1)) ** 1.5
        for k in range(NZ)
    ])
    v_profile = np.array([
        10 + 18 * np.exp(-((h_1d[k] - 1500.0) / 1200.0) ** 2)
        - 5 * (k / (NZ - 1))
        for k in range(NZ)
    ])
    u_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    v_3d = np.empty((NZ, NY, NX), dtype=np.float64)
    for k in range(NZ):
        u_3d[k] = u_profile[k] + rng.normal(0, 1.5, (NY, NX))
        v_3d[k] = v_profile[k] + rng.normal(0, 1.5, (NY, NX))

    # 2-D slices at 850 hPa
    u_850 = u_3d[i850].copy()
    v_850 = v_3d[i850].copy()

    # -- moisture: q (mixing ratio kg/kg), decrease with height ---------------
    q_sfc = 0.014 - 0.006 * y_frac  # moister south
    qvapor = np.empty((NZ, NY, NX), dtype=np.float64)
    for k in range(NZ):
        qvapor[k] = q_sfc[None, :] * np.exp(-h_1d[k] / 3500.0)
    qvapor = np.clip(qvapor, 1e-6, None)

    # -- pressure 3-D (Pa for compute_cape_cin) --------------------------------
    p3_hPa = np.broadcast_to(plev[:, None, None], (NZ, NY, NX)).copy()
    p3_Pa = p3_hPa * 100.0

    # -- surface fields -------------------------------------------------------
    psfc = np.full((NY, NX), 1000.0 * 100.0, dtype=np.float64)  # Pa
    t2 = t_c[0] + 273.15  # surface T in K
    q2 = qvapor[0].copy()  # surface mixing ratio kg/kg

    return dict(
        plev=plev, i850=i850,
        t_c=t_c, t_k_850=t_k_850, theta_850=theta_850,
        u_3d=u_3d, v_3d=v_3d,
        u_850=u_850, v_850=v_850,
        qvapor=qvapor,
        height_agl=height_agl,
        p3_hPa=p3_hPa, p3_Pa=p3_Pa,
        psfc=psfc, t2=t2, q2=q2,
    )


# ============================================================================
# Timing helpers
# ============================================================================

def _sync_gpu():
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()


def bench(func, n=NRUNS, gpu=False):
    """1 warmup + n timed runs, return median ms."""
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
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def fmt(ms):
    if ms is None:
        return "\u2014"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd(a, b):
    if a is None or b is None or b <= 0:
        return "\u2014"
    r = a / b
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


# ============================================================================
# Conversion helpers
# ============================================================================

def _to_np(x):
    """Convert Pint Quantity / cupy array / tuple to plain numpy."""
    if isinstance(x, tuple):
        return tuple(_to_np(v) for v in x)
    if hasattr(x, "magnitude"):
        x = x.magnitude
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x, dtype=np.float64)


# ============================================================================
# Deep verification engine
# ============================================================================

def _nan_inf_audit(arr, label):
    """Return (n_nan, n_inf, n_total, pass_bool) and print audit line."""
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n_total = int(arr.size)
    ok = (n_nan == 0 and n_inf == 0)
    status = "PASS" if ok else "FAIL"
    print(f"        NaN/Inf audit [{label}]: {status}  "
          f"(NaN={n_nan}, Inf={n_inf}, total={n_total})")
    return n_nan, n_inf, n_total, ok


def _phys_bounds_check(arr, lo, hi, unit_str, label):
    """Check physical plausibility bounds; return pass_bool."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"        Phys bounds [{label}]: SKIP (no finite values)")
        return True
    vmin, vmax = float(finite.min()), float(finite.max())
    ok = (vmin >= lo and vmax <= hi)
    status = "PASS" if ok else "FAIL"
    n_below = int((finite < lo).sum())
    n_above = int((finite > hi).sum())
    print(f"        Phys bounds [{label}]: {status}  "
          f"range=[{vmin:.6g}, {vmax:.6g}] vs [{lo:.6g}, {hi:.6g}] {unit_str}"
          f"  (below={n_below}, above={n_above})")
    return ok


def _diff_histogram(diffs, label, nbins=8):
    """Print a compact text histogram of absolute differences."""
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        print(f"        Diff histogram [{label}]: no finite diffs")
        return
    # Use log-spaced bins if range is large
    mn, mx = float(finite.min()), float(finite.max())
    if mx <= 0:
        print(f"        Diff histogram [{label}]: all diffs = 0")
        return
    # Linear bins from 0 to max
    edges = np.linspace(0, mx, nbins + 1)
    counts, _ = np.histogram(finite, bins=edges)
    total = finite.size
    print(f"        Diff histogram [{label}] (abs diff, {nbins} bins, 0..{mx:.3e}):")
    bar_max_width = 30
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(nbins):
        lo_e, hi_e = edges[i], edges[i + 1]
        c = counts[i]
        pct = 100.0 * c / total
        bar_len = int(bar_max_width * c / max_count)
        bar = "#" * bar_len
        print(f"          [{lo_e:10.3e}, {hi_e:10.3e}): {c:>8d} ({pct:5.1f}%) {bar}")


def deep_verify(func_name, ref, test, rtol, phys_key=None):
    """Run exhaustive data-correctness checks between ref and test arrays.

    Returns True if all critical checks pass.
    """
    ref_np = _to_np(ref)
    test_np = _to_np(test)

    # Handle tuple returns (q_vector)
    if isinstance(ref_np, tuple):
        all_ok = True
        for i, (r, t) in enumerate(zip(ref_np, test_np)):
            sub_phys = f"{phys_key}" if phys_key else None
            all_ok &= deep_verify(f"{func_name}[{i}]", r, t, rtol, sub_phys)
        return all_ok

    print(f"\n      --- Deep verification: {func_name} ---")
    print(f"        Shape: ref={ref_np.shape}, test={test_np.shape}")
    all_ok = True

    # 1. NaN/Inf audit on both arrays
    _, _, _, ok_r = _nan_inf_audit(ref_np, "ref")
    _, _, _, ok_t = _nan_inf_audit(test_np, "test")
    all_ok &= ok_r and ok_t

    # 2. Physical plausibility bounds
    if phys_key and phys_key in PHYS_BOUNDS:
        lo, hi, unit_str = PHYS_BOUNDS[phys_key]
        ok_pb_r = _phys_bounds_check(ref_np, lo, hi, unit_str, "ref")
        ok_pb_t = _phys_bounds_check(test_np, lo, hi, unit_str, "test")
        all_ok &= ok_pb_r and ok_pb_t

    # 3. Compute difference statistics
    diff = test_np - ref_np
    abs_diff = np.abs(diff)
    finite_mask = np.isfinite(diff)
    n_finite = int(finite_mask.sum())

    if n_finite == 0:
        print("        No finite differences to analyze!")
        return all_ok

    fd = diff[finite_mask]
    fad = abs_diff[finite_mask]

    mean_diff = float(np.mean(fd))
    max_abs_diff = float(np.max(fad))
    rmse = float(np.sqrt(np.mean(fd ** 2)))
    p99 = float(np.percentile(fad, 99))
    p99_9 = float(np.percentile(fad, 99.9))
    std_diff = float(np.std(fd))
    median_diff = float(np.median(fd))

    # Relative RMSE (as % of ref range)
    ref_finite = ref_np[np.isfinite(ref_np)]
    ref_range = float(np.ptp(ref_finite)) if ref_finite.size > 0 else 1.0
    rel_rmse_pct = (rmse / ref_range * 100.0) if ref_range > 0 else 0.0

    print(f"        Mean diff:       {mean_diff:+.6e}")
    print(f"        Median diff:     {median_diff:+.6e}")
    print(f"        Std diff:        {std_diff:.6e}")
    print(f"        Max |diff|:      {max_abs_diff:.6e}")
    print(f"        RMSE:            {rmse:.6e}")
    print(f"        Relative RMSE:   {rel_rmse_pct:.6f}%  (of ref range {ref_range:.6e})")
    print(f"        99th pct |diff|: {p99:.6e}")
    print(f"        99.9th pct:      {p99_9:.6e}")

    # 4. Pearson correlation
    if ref_finite.size > 1 and np.std(ref_np[finite_mask]) > 0:
        r_val, p_val = sp_stats.pearsonr(ref_np[finite_mask].ravel(),
                                         test_np[finite_mask].ravel())
        ok_corr = r_val > 0.9999
        status = "PASS" if ok_corr else "FAIL"
        print(f"        Pearson r:       {r_val:.10f}  (p={p_val:.2e})  {status}")
        all_ok &= ok_corr
    else:
        print("        Pearson r:       SKIP (constant or empty)")

    # 5. Relative error analysis (points where ref is significant)
    sig_mask = finite_mask & (np.abs(ref_np) > 1e-12)
    n_sig = int(sig_mask.sum())
    if n_sig > 0:
        rel_err = np.abs(diff[sig_mask] / ref_np[sig_mask])
        n_gt_1pct = int((rel_err > 0.01).sum())
        n_gt_01pct = int((rel_err > 0.001).sum())
        pct_gt_1pct = 100.0 * n_gt_1pct / n_sig
        pct_gt_01pct = 100.0 * n_gt_01pct / n_sig
        max_rel = float(np.max(rel_err))

        ok_rtol = max_rel <= rtol
        status = "PASS" if ok_rtol else "FAIL"
        print(f"        Max rel error:   {max_rel:.6e}  (rtol={rtol:.0e})  {status}")
        print(f"        Points >1%% rel:  {n_gt_1pct:>8d} / {n_sig} ({pct_gt_1pct:.4f}%%)")
        print(f"        Points >0.1%% rel: {n_gt_01pct:>8d} / {n_sig} ({pct_gt_01pct:.4f}%%)")
        all_ok &= ok_rtol
    else:
        print("        Relative error:  SKIP (no significant ref values)")

    # 6. Diff histogram
    _diff_histogram(fad, func_name)

    status = "PASS" if all_ok else "FAIL"
    print(f"        Overall [{func_name}]: {status}")
    return all_ok


def deep_verify_grid_cape(func_name, ref_np, test_np, rtol, label):
    """Deep verification for CAPE/CIN grid output (allows tiny boundary outliers).

    Returns True if all critical checks pass.
    """
    print(f"\n      --- Deep verification: {func_name}[{label}] ---")
    print(f"        Shape: ref={ref_np.shape}, test={test_np.shape}")
    all_ok = True

    # NaN/Inf audit
    _, _, _, ok_r = _nan_inf_audit(ref_np, "ref")
    _, _, _, ok_t = _nan_inf_audit(test_np, "test")
    all_ok &= ok_r and ok_t

    # Physical plausibility
    phys_key = f"compute_cape_cin_{label}"
    if phys_key in PHYS_BOUNDS:
        lo, hi, unit_str = PHYS_BOUNDS[phys_key]
        all_ok &= _phys_bounds_check(ref_np, lo, hi, unit_str, "ref")
        all_ok &= _phys_bounds_check(test_np, lo, hi, unit_str, "test")

    # Difference statistics on significant points
    diff = test_np - ref_np
    abs_diff = np.abs(diff)
    finite_mask = np.isfinite(diff)
    fd = diff[finite_mask]
    fad = abs_diff[finite_mask]
    n_finite = int(finite_mask.sum())

    if n_finite == 0:
        print("        No finite differences to analyze!")
        return all_ok

    mean_diff = float(np.mean(fd))
    max_abs_diff = float(np.max(fad))
    rmse = float(np.sqrt(np.mean(fd ** 2)))
    p99 = float(np.percentile(fad, 99))
    p99_9 = float(np.percentile(fad, 99.9))
    p99_99 = float(np.percentile(fad, 99.99))

    ref_finite = ref_np[np.isfinite(ref_np)]
    ref_range = float(np.ptp(ref_finite)) if ref_finite.size > 0 else 1.0
    rel_rmse_pct = (rmse / ref_range * 100.0) if ref_range > 0 else 0.0

    print(f"        Mean diff:       {mean_diff:+.6e}")
    print(f"        Max |diff|:      {max_abs_diff:.6e}")
    print(f"        RMSE:            {rmse:.6e}")
    print(f"        Relative RMSE:   {rel_rmse_pct:.6f}%")
    print(f"        99th pct |diff|: {p99:.6e}")
    print(f"        99.9th pct:      {p99_9:.6e}")
    print(f"        99.99th pct:     {p99_99:.6e}")

    # Pearson r
    if ref_finite.size > 1 and np.std(ref_np[finite_mask]) > 0:
        r_val, p_val = sp_stats.pearsonr(ref_np[finite_mask].ravel(),
                                         test_np[finite_mask].ravel())
        ok_corr = r_val > 0.999
        status = "PASS" if ok_corr else "FAIL"
        print(f"        Pearson r:       {r_val:.10f}  {status}")
        all_ok &= ok_corr

    # Relative error with outlier tolerance (99.99% must pass)
    mask_sig = finite_mask & (np.abs(ref_np) > 1.0)
    n_sig = int(mask_sig.sum())
    if n_sig > 0:
        rel = np.abs(diff[mask_sig] / ref_np[mask_sig])
        n_bad = int((rel > rtol).sum())
        frac_good = 1.0 - n_bad / n_sig
        max_rel = float(np.max(rel))
        n_gt_1pct = int((rel > 0.01).sum())
        n_gt_01pct = int((rel > 0.001).sum())

        ok_grid = frac_good >= 0.9999 and float(np.percentile(rel, 99.99)) <= rtol
        status = "PASS" if ok_grid else "FAIL"
        print(f"        Max rel error:   {max_rel:.6e}")
        print(f"        Outliers (>{rtol:.0e}): {n_bad} / {n_sig} "
              f"({100.0 * n_bad / n_sig:.4f}%)  "
              f"need >=99.99%% within rtol  {status}")
        print(f"        Points >1%% rel:  {n_gt_1pct:>8d} / {n_sig} "
              f"({100.0 * n_gt_1pct / n_sig:.4f}%%)")
        print(f"        Points >0.1%% rel: {n_gt_01pct:>8d} / {n_sig} "
              f"({100.0 * n_gt_01pct / n_sig:.4f}%%)")
        all_ok &= ok_grid

    # Diff histogram
    _diff_histogram(fad, f"{func_name}[{label}]")

    status = "PASS" if all_ok else "FAIL"
    print(f"        Overall [{func_name}][{label}]: {status}")
    return all_ok


# ============================================================================
# CAPE/CIN cross-backend distribution analysis
# ============================================================================

def cape_cin_cross_backend_analysis(cape_results):
    """Deep cross-backend comparison of CAPE/CIN distributions.

    cape_results: dict mapping backend_name -> (cape_2d, cin_2d) as numpy arrays.
    Returns True if distribution comparison passes.
    """
    W = 100
    print()
    print("=" * W)
    print("  CAPE/CIN CROSS-BACKEND DISTRIBUTION ANALYSIS")
    print("=" * W)

    if len(cape_results) < 2:
        print("    Only 1 backend available -- skipping cross-backend analysis.")
        return True

    all_ok = True
    backend_names = list(cape_results.keys())

    for field_idx, field_name in enumerate(["CAPE", "CIN"]):
        print(f"\n  --- {field_name} Distribution Comparison ---")

        # Gather all arrays
        arrays = {}
        for bname in backend_names:
            arr = cape_results[bname][field_idx]
            arrays[bname] = arr

        # Print distribution stats table
        print(f"    {'Backend':<18s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} "
              f"{'P5':>10s} {'P25':>10s} {'P75':>10s} {'P95':>10s} "
              f"{'Min':>10s} {'Max':>10s}")
        print("    " + "-" * 116)

        stats_by_backend = {}
        for bname in backend_names:
            a = arrays[bname]
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                print(f"    {bname:<18s}  (no finite values)")
                continue
            s = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "std": float(np.std(finite)),
                "p5": float(np.percentile(finite, 5)),
                "p25": float(np.percentile(finite, 25)),
                "p75": float(np.percentile(finite, 75)),
                "p95": float(np.percentile(finite, 95)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
            stats_by_backend[bname] = s
            print(f"    {bname:<18s} {s['mean']:10.2f} {s['median']:10.2f} "
                  f"{s['std']:10.2f} {s['p5']:10.2f} {s['p25']:10.2f} "
                  f"{s['p75']:10.2f} {s['p95']:10.2f} "
                  f"{s['min']:10.2f} {s['max']:10.2f}")

        # Cross-compare: columns where backends disagree on CAPE > 0 vs CAPE == 0
        if field_name == "CAPE":
            print(f"\n    CAPE>0 vs CAPE=0 agreement (column-level):")
            ref_name = backend_names[0]
            ref_arr = arrays[ref_name]
            ref_positive = ref_arr > 0.0

            for bname in backend_names[1:]:
                test_arr = arrays[bname]
                test_positive = test_arr > 0.0
                disagree = ref_positive != test_positive
                n_disagree = int(disagree.sum())
                n_total = int(ref_arr.size)
                pct = 100.0 * n_disagree / n_total if n_total > 0 else 0.0

                # Break down: ref>0 & test==0 vs ref==0 & test>0
                ref_pos_test_zero = int((ref_positive & ~test_positive).sum())
                ref_zero_test_pos = int((~ref_positive & test_positive).sum())

                ok_agree = pct < 1.0  # less than 1% disagreement
                status = "PASS" if ok_agree else "FAIL"
                print(f"      {ref_name} vs {bname}: {n_disagree} / {n_total} "
                      f"disagree ({pct:.4f}%)  {status}")
                print(f"        ref>0 & test=0: {ref_pos_test_zero}    "
                      f"ref=0 & test>0: {ref_zero_test_pos}")
                all_ok &= ok_agree

        # Pairwise Pearson correlation between backends
        if len(backend_names) >= 2:
            print(f"\n    Pairwise Pearson r ({field_name}):")
            for i in range(len(backend_names)):
                for j in range(i + 1, len(backend_names)):
                    a_i = arrays[backend_names[i]].ravel()
                    a_j = arrays[backend_names[j]].ravel()
                    valid = np.isfinite(a_i) & np.isfinite(a_j)
                    if valid.sum() > 1 and np.std(a_i[valid]) > 0:
                        r_val, _ = sp_stats.pearsonr(a_i[valid], a_j[valid])
                        ok_r = r_val > 0.999
                        status = "PASS" if ok_r else "FAIL"
                        print(f"      {backend_names[i]} vs {backend_names[j]}: "
                              f"r={r_val:.10f}  {status}")
                        all_ok &= ok_r

        # Pairwise RMSE and mean-diff between backends
        print(f"\n    Pairwise RMSE / Mean diff ({field_name}):")
        for i in range(len(backend_names)):
            for j in range(i + 1, len(backend_names)):
                a_i = arrays[backend_names[i]]
                a_j = arrays[backend_names[j]]
                d = a_j - a_i
                finite = d[np.isfinite(d)]
                if finite.size > 0:
                    rmse = float(np.sqrt(np.mean(finite ** 2)))
                    md = float(np.mean(finite))
                    print(f"      {backend_names[i]} vs {backend_names[j]}: "
                          f"RMSE={rmse:.6e}, mean_diff={md:+.6e}")

    return all_ok


# ============================================================================
# Legacy simple verify (kept for basic pass/fail in timing table)
# ============================================================================

def verify(name, ref, test, rtol):
    """Check two results agree within rtol; print PASS/FAIL."""
    ref_np = _to_np(ref)
    test_np = _to_np(test)
    if isinstance(ref_np, tuple):
        ok = True
        for i, (r, t) in enumerate(zip(ref_np, test_np)):
            mask = np.abs(r) > 1e-12
            if mask.sum() == 0:
                continue
            maxrel = np.max(np.abs((r[mask] - t[mask]) / r[mask]))
            if maxrel > rtol:
                print(f"    FAIL  {name}[{i}]: max_rel={maxrel:.2e} > {rtol:.0e}")
                ok = False
        if ok:
            print(f"    PASS  {name}")
        return ok
    else:
        mask = np.abs(ref_np) > 1e-12
        if mask.sum() == 0:
            print(f"    PASS  {name} (all near zero)")
            return True
        maxrel = np.max(np.abs((ref_np[mask] - test_np[mask]) / ref_np[mask]))
        if maxrel > rtol:
            print(f"    FAIL  {name}: max_rel={maxrel:.2e} > {rtol:.0e}")
            return False
        print(f"    PASS  {name} (max_rel={maxrel:.2e})")
        return True


def verify_grid_cape(name, ref_np, test_np, rtol, label):
    """Verify CAPE/CIN allowing a tiny fraction of boundary outliers."""
    mask = np.abs(ref_np) > 1.0
    if mask.sum() == 0:
        print(f"    PASS  {name}[{label}] (all near zero)")
        return True
    rel = np.abs((ref_np[mask] - test_np[mask]) / ref_np[mask])
    n_bad = int((rel > rtol).sum())
    n_total = int(mask.sum())
    frac_good = 1.0 - n_bad / n_total
    p99 = float(np.percentile(rel, 99.99))
    if frac_good >= 0.9999 and p99 <= rtol:
        print(f"    PASS  {name}[{label}] "
              f"(p99.99={p99:.2e}, {n_bad} outliers / {n_total})")
        return True
    else:
        print(f"    FAIL  {name}[{label}] "
              f"(p99.99={p99:.2e}, {n_bad} outliers / {n_total}, "
              f"need >=99.99% within {rtol:.0e})")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    W = 100
    print("=" * W)
    print("  BENCHMARK 10: HRRR Squall Line / MCS Dynamics")
    print(f"  Grid: {NZ} levels x {NY}x{NX}  |  dx=dy={DX:.0f} m")
    print(f"  MetPy: {'yes' if USE_METPY else 'skipped'}  |  "
          f"met-cu: {'yes' if HAS_METCU else 'not available'}  |  "
          f"GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print("=" * W)

    print("\n  Building synthetic squall-line data ...", end=" ", flush=True)
    t0 = time.perf_counter()
    d = build_squall_data()
    print(f"{time.perf_counter() - t0:.2f} s")

    # -- Pre-build MetPy Pint quantities (not timed) -------------------------
    if USE_METPY:
        u_mp = d["u_850"] * mpunits("m/s")
        v_mp = d["v_850"] * mpunits("m/s")
        t_mp_K = d["t_k_850"] * mpunits.K
        th_mp = d["theta_850"] * mpunits.K
        p_mp = 850.0 * mpunits.hPa
        dx_mp = DX * mpunits.m
        dy_mp = DY * mpunits.m

    # ========================================================================
    # Results table
    # ========================================================================
    COL_NAME = 28
    COL_T = 11
    COL_S = 9

    def hdr(title):
        print()
        print(f"\u2500\u2500 {title} " + "\u2500" * max(0, W - len(title) - 4))
        cols = (f"  {'Function':{COL_NAME}s}"
                f" {'MetPy':>{COL_T}s}"
                f" {'Rust/CPU':>{COL_T}s}"
                f" {'met-cu':>{COL_T}s}"
                f" {'Rust/GPU':>{COL_T}s}"
                f" {'MP/Rust':>{COL_S}s}"
                f" {'Rust/CUDA':>{COL_S}s}")
        print(cols)

    def row(name, t_mp, t_cpu, t_cu, t_gpu):
        print(f"  {name:{COL_NAME}s}"
              f" {fmt(t_mp):>{COL_T}s}"
              f" {fmt(t_cpu):>{COL_T}s}"
              f" {fmt(t_cu):>{COL_T}s}"
              f" {fmt(t_gpu):>{COL_T}s}"
              f" {spd(t_mp, t_cpu):>{COL_S}s}"
              f" {spd(t_cpu, t_gpu):>{COL_S}s}")

    all_pass = True

    # ========================================================================
    # Collect all computed results for cross-backend deep analysis
    # ========================================================================
    deep_results = {}   # func_name -> list of (backend_label, result_np)
    cape_cin_backends = {}  # backend_name -> (cape_2d, cin_2d)

    # ========================================================================
    # 1. POTENTIAL TEMPERATURE  (2-D, 850 hPa)
    # ========================================================================
    hdr(f"potential_temperature (2D: {NY}x{NX})")

    # -- compute reference (metrust CPU) --
    mrcalc.set_backend("cpu")
    ref_pt = mrcalc.potential_temperature(850.0, d["t_k_850"] - 273.15)
    deep_results["potential_temperature"] = [("Rust/CPU", _to_np(ref_pt))]

    # MetPy
    t_mp_pt = None
    if USE_METPY:
        res_mp = mpcalc.potential_temperature(p_mp, (d["t_k_850"] - 273.15) * mpunits.degC)
        res_mp_np = _to_np(res_mp)
        all_pass &= verify("potential_temperature MetPy vs Rust/CPU", _to_np(ref_pt), res_mp, RTOL_KINE)
        all_pass &= deep_verify("potential_temperature MetPy-vs-Rust/CPU",
                                ref_pt, res_mp, RTOL_KINE,
                                phys_key="potential_temperature")
        deep_results["potential_temperature"].append(("MetPy", res_mp_np))
        t_mp_pt = bench(lambda: mpcalc.potential_temperature(p_mp, (d["t_k_850"] - 273.15) * mpunits.degC))

    # metrust CPU
    t_cpu_pt = bench(lambda: mrcalc.potential_temperature(850.0, d["t_k_850"] - 273.15))

    # met-cu direct
    t_cu_pt = None
    if HAS_METCU:
        res_cu = mcucalc.potential_temperature(850.0, d["t_k_850"] - 273.15)
        res_cu_np = _to_np(res_cu)
        all_pass &= verify("potential_temperature met-cu vs Rust/CPU", _to_np(ref_pt), res_cu, RTOL_KINE)
        all_pass &= deep_verify("potential_temperature met-cu-vs-Rust/CPU",
                                ref_pt, res_cu, RTOL_KINE,
                                phys_key="potential_temperature")
        deep_results["potential_temperature"].append(("met-cu", res_cu_np))
        t_cu_pt = bench(lambda: mcucalc.potential_temperature(850.0, d["t_k_850"] - 273.15), gpu=True)

    # metrust GPU
    t_gpu_pt = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.potential_temperature(850.0, d["t_k_850"] - 273.15)
        res_gpu_np = _to_np(res_gpu)
        all_pass &= verify("potential_temperature Rust/GPU vs Rust/CPU", _to_np(ref_pt), res_gpu, RTOL_KINE)
        all_pass &= deep_verify("potential_temperature Rust/GPU-vs-Rust/CPU",
                                ref_pt, res_gpu, RTOL_KINE,
                                phys_key="potential_temperature")
        deep_results["potential_temperature"].append(("Rust/GPU", res_gpu_np))
        t_gpu_pt = bench(lambda: mrcalc.potential_temperature(850.0, d["t_k_850"] - 273.15), gpu=True)
        mrcalc.set_backend("cpu")

    row("potential_temperature", t_mp_pt, t_cpu_pt, t_cu_pt, t_gpu_pt)

    # ========================================================================
    # 2. VORTICITY  (2-D, 850 hPa)
    # ========================================================================
    hdr(f"vorticity (2D: {NY}x{NX})")

    mrcalc.set_backend("cpu")
    ref_vort = mrcalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY)
    deep_results["vorticity"] = [("Rust/CPU", _to_np(ref_vort))]

    t_mp_vort = None
    if USE_METPY:
        res_mp = mpcalc.vorticity(u_mp, v_mp, dx=dx_mp, dy=dy_mp)
        res_mp_np = _to_np(res_mp)
        all_pass &= verify("vorticity MetPy vs Rust/CPU", _to_np(ref_vort), res_mp, RTOL_KINE)
        all_pass &= deep_verify("vorticity MetPy-vs-Rust/CPU",
                                ref_vort, res_mp, RTOL_KINE,
                                phys_key="vorticity")
        deep_results["vorticity"].append(("MetPy", res_mp_np))
        t_mp_vort = bench(lambda: mpcalc.vorticity(u_mp, v_mp, dx=dx_mp, dy=dy_mp))

    t_cpu_vort = bench(lambda: mrcalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY))

    t_cu_vort = None
    if HAS_METCU:
        res_cu = mcucalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY)
        res_cu_np = _to_np(res_cu)
        all_pass &= verify("vorticity met-cu vs Rust/CPU", _to_np(ref_vort), res_cu, RTOL_KINE)
        all_pass &= deep_verify("vorticity met-cu-vs-Rust/CPU",
                                ref_vort, res_cu, RTOL_KINE,
                                phys_key="vorticity")
        deep_results["vorticity"].append(("met-cu", res_cu_np))
        t_cu_vort = bench(lambda: mcucalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY), gpu=True)

    t_gpu_vort = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY)
        res_gpu_np = _to_np(res_gpu)
        all_pass &= verify("vorticity Rust/GPU vs Rust/CPU", _to_np(ref_vort), res_gpu, RTOL_KINE)
        all_pass &= deep_verify("vorticity Rust/GPU-vs-Rust/CPU",
                                ref_vort, res_gpu, RTOL_KINE,
                                phys_key="vorticity")
        deep_results["vorticity"].append(("Rust/GPU", res_gpu_np))
        t_gpu_vort = bench(lambda: mrcalc.vorticity(d["u_850"], d["v_850"], dx=DX, dy=DY), gpu=True)
        mrcalc.set_backend("cpu")

    row("vorticity", t_mp_vort, t_cpu_vort, t_cu_vort, t_gpu_vort)

    # ========================================================================
    # 3. FRONTOGENESIS  (2-D, 850 hPa)
    # ========================================================================
    hdr(f"frontogenesis (2D: {NY}x{NX})")

    mrcalc.set_backend("cpu")
    ref_fronto = mrcalc.frontogenesis(
        d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY)
    deep_results["frontogenesis"] = [("Rust/CPU", _to_np(ref_fronto))]

    t_mp_fronto = None
    if USE_METPY:
        res_mp = mpcalc.frontogenesis(th_mp, u_mp, v_mp, dx=dx_mp, dy=dy_mp)
        res_mp_np = _to_np(res_mp)
        all_pass &= verify("frontogenesis MetPy vs Rust/CPU", _to_np(ref_fronto), res_mp, RTOL_KINE)
        all_pass &= deep_verify("frontogenesis MetPy-vs-Rust/CPU",
                                ref_fronto, res_mp, RTOL_KINE,
                                phys_key="frontogenesis")
        deep_results["frontogenesis"].append(("MetPy", res_mp_np))
        t_mp_fronto = bench(lambda: mpcalc.frontogenesis(th_mp, u_mp, v_mp, dx=dx_mp, dy=dy_mp))

    t_cpu_fronto = bench(lambda: mrcalc.frontogenesis(
        d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY))

    t_cu_fronto = None
    if HAS_METCU:
        res_cu = mcucalc.frontogenesis(
            d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY)
        res_cu_np = _to_np(res_cu)
        all_pass &= verify("frontogenesis met-cu vs Rust/CPU", _to_np(ref_fronto), res_cu, RTOL_KINE)
        all_pass &= deep_verify("frontogenesis met-cu-vs-Rust/CPU",
                                ref_fronto, res_cu, RTOL_KINE,
                                phys_key="frontogenesis")
        deep_results["frontogenesis"].append(("met-cu", res_cu_np))
        t_cu_fronto = bench(lambda: mcucalc.frontogenesis(
            d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY), gpu=True)

    t_gpu_fronto = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.frontogenesis(
            d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY)
        res_gpu_np = _to_np(res_gpu)
        all_pass &= verify("frontogenesis Rust/GPU vs Rust/CPU", _to_np(ref_fronto), res_gpu, RTOL_KINE)
        all_pass &= deep_verify("frontogenesis Rust/GPU-vs-Rust/CPU",
                                ref_fronto, res_gpu, RTOL_KINE,
                                phys_key="frontogenesis")
        deep_results["frontogenesis"].append(("Rust/GPU", res_gpu_np))
        t_gpu_fronto = bench(lambda: mrcalc.frontogenesis(
            d["theta_850"], d["u_850"], d["v_850"], dx=DX, dy=DY), gpu=True)
        mrcalc.set_backend("cpu")

    row("frontogenesis", t_mp_fronto, t_cpu_fronto, t_cu_fronto, t_gpu_fronto)

    # ========================================================================
    # 4. Q-VECTOR  (2-D, 850 hPa)
    # ========================================================================
    hdr(f"q_vector (2D: {NY}x{NX})")

    mrcalc.set_backend("cpu")
    ref_qvec = mrcalc.q_vector(
        d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY)
    ref_qvec_np = _to_np(ref_qvec)
    deep_results["q_vector"] = [("Rust/CPU", ref_qvec_np)]

    t_mp_qvec = None
    if USE_METPY:
        res_mp = mpcalc.q_vector(u_mp, v_mp, t_mp_K, p_mp, dx=dx_mp, dy=dy_mp)
        res_mp_np = _to_np(res_mp)
        all_pass &= verify("q_vector MetPy vs Rust/CPU", ref_qvec_np, res_mp, RTOL_KINE)
        all_pass &= deep_verify("q_vector MetPy-vs-Rust/CPU",
                                ref_qvec, res_mp, RTOL_KINE,
                                phys_key="q_vector")
        deep_results["q_vector"].append(("MetPy", res_mp_np))
        t_mp_qvec = bench(lambda: mpcalc.q_vector(u_mp, v_mp, t_mp_K, p_mp, dx=dx_mp, dy=dy_mp))

    t_cpu_qvec = bench(lambda: mrcalc.q_vector(
        d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY))

    t_cu_qvec = None
    if HAS_METCU:
        res_cu = mcucalc.q_vector(
            d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY)
        res_cu_np = _to_np(res_cu)
        all_pass &= verify("q_vector met-cu vs Rust/CPU", ref_qvec_np, res_cu, RTOL_KINE)
        all_pass &= deep_verify("q_vector met-cu-vs-Rust/CPU",
                                ref_qvec, res_cu, RTOL_KINE,
                                phys_key="q_vector")
        deep_results["q_vector"].append(("met-cu", res_cu_np))
        t_cu_qvec = bench(lambda: mcucalc.q_vector(
            d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY), gpu=True)

    t_gpu_qvec = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.q_vector(
            d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY)
        res_gpu_np = _to_np(res_gpu)
        all_pass &= verify("q_vector Rust/GPU vs Rust/CPU", ref_qvec_np, res_gpu, RTOL_KINE)
        all_pass &= deep_verify("q_vector Rust/GPU-vs-Rust/CPU",
                                ref_qvec, res_gpu, RTOL_KINE,
                                phys_key="q_vector")
        deep_results["q_vector"].append(("Rust/GPU", res_gpu_np))
        t_gpu_qvec = bench(lambda: mrcalc.q_vector(
            d["u_850"], d["v_850"], d["t_k_850"], 850.0, dx=DX, dy=DY), gpu=True)
        mrcalc.set_backend("cpu")

    row("q_vector", t_mp_qvec, t_cpu_qvec, t_cu_qvec, t_gpu_qvec)

    # ========================================================================
    # 5. COMPUTE_CAPE_CIN  (3-D -> 2-D)    [no MetPy grid version]
    # ========================================================================
    hdr(f"compute_cape_cin (3D: {NZ}x{NY}x{NX} -> 2D)")

    mrcalc.set_backend("cpu")
    ref_cape = mrcalc.compute_cape_cin(
        d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
        d["psfc"], d["t2"], d["q2"])
    ref_cape_np = _to_np(ref_cape)
    cape_cin_backends["Rust/CPU"] = (ref_cape_np[0], ref_cape_np[1])

    t_cpu_cape = bench(lambda: mrcalc.compute_cape_cin(
        d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
        d["psfc"], d["t2"], d["q2"]))

    t_cu_cape = None
    if HAS_METCU:
        res_cu = mcucalc.compute_cape_cin(
            d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
            d["psfc"], d["t2"], d["q2"])
        res_cu_np = _to_np(res_cu)
        cape_cin_backends["met-cu"] = (res_cu_np[0], res_cu_np[1])
        for i, label in enumerate(["CAPE", "CIN"]):
            all_pass &= verify_grid_cape(
                "compute_cape_cin met-cu vs Rust/CPU",
                ref_cape_np[i], res_cu_np[i], RTOL_CAPE, label)
            all_pass &= deep_verify_grid_cape(
                "compute_cape_cin met-cu-vs-Rust/CPU",
                ref_cape_np[i], res_cu_np[i], RTOL_CAPE, label)
        t_cu_cape = bench(lambda: mcucalc.compute_cape_cin(
            d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
            d["psfc"], d["t2"], d["q2"]), gpu=True)

    t_gpu_cape = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.compute_cape_cin(
            d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
            d["psfc"], d["t2"], d["q2"])
        res_gpu_np = _to_np(res_gpu)
        cape_cin_backends["Rust/GPU"] = (res_gpu_np[0], res_gpu_np[1])
        for i, label in enumerate(["CAPE", "CIN"]):
            all_pass &= verify_grid_cape(
                "compute_cape_cin Rust/GPU vs Rust/CPU",
                ref_cape_np[i], res_gpu_np[i], RTOL_CAPE, label)
            all_pass &= deep_verify_grid_cape(
                "compute_cape_cin Rust/GPU-vs-Rust/CPU",
                ref_cape_np[i], res_gpu_np[i], RTOL_CAPE, label)
        t_gpu_cape = bench(lambda: mrcalc.compute_cape_cin(
            d["p3_Pa"], d["t_c"], d["qvapor"], d["height_agl"],
            d["psfc"], d["t2"], d["q2"]), gpu=True)
        mrcalc.set_backend("cpu")

    row("compute_cape_cin", None, t_cpu_cape, t_cu_cape, t_gpu_cape)

    # ========================================================================
    # 6. COMPUTE_SHEAR  (3-D -> 2-D)    [no MetPy grid version]
    # ========================================================================
    hdr(f"compute_shear 0-6km (3D: {NZ}x{NY}x{NX} -> 2D)")

    mrcalc.set_backend("cpu")
    ref_shear = mrcalc.compute_shear(
        d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0)
    deep_results["compute_shear"] = [("Rust/CPU", _to_np(ref_shear))]

    t_cpu_shear = bench(lambda: mrcalc.compute_shear(
        d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0))

    t_cu_shear = None
    if HAS_METCU:
        res_cu = mcucalc.compute_shear(
            d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0)
        res_cu_np = _to_np(res_cu)
        all_pass &= verify("compute_shear met-cu vs Rust/CPU", _to_np(ref_shear), res_cu, RTOL_KINE)
        all_pass &= deep_verify("compute_shear met-cu-vs-Rust/CPU",
                                ref_shear, res_cu, RTOL_KINE,
                                phys_key="compute_shear")
        deep_results["compute_shear"].append(("met-cu", res_cu_np))
        t_cu_shear = bench(lambda: mcucalc.compute_shear(
            d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0), gpu=True)

    t_gpu_shear = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        res_gpu = mrcalc.compute_shear(
            d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0)
        res_gpu_np = _to_np(res_gpu)
        all_pass &= verify("compute_shear Rust/GPU vs Rust/CPU", _to_np(ref_shear), res_gpu, RTOL_KINE)
        all_pass &= deep_verify("compute_shear Rust/GPU-vs-Rust/CPU",
                                ref_shear, res_gpu, RTOL_KINE,
                                phys_key="compute_shear")
        deep_results["compute_shear"].append(("Rust/GPU", res_gpu_np))
        t_gpu_shear = bench(lambda: mrcalc.compute_shear(
            d["u_3d"], d["v_3d"], d["height_agl"], bottom_m=0.0, top_m=6000.0), gpu=True)
        mrcalc.set_backend("cpu")

    row("compute_shear", None, t_cpu_shear, t_cu_shear, t_gpu_shear)

    # ========================================================================
    # CAPE/CIN Cross-Backend Distribution Analysis
    # ========================================================================
    all_pass &= cape_cin_cross_backend_analysis(cape_cin_backends)

    # ========================================================================
    # VERIFICATION SUMMARY TABLE
    # ========================================================================
    print()
    print("=" * W)
    print("  DEEP VERIFICATION SUMMARY")
    print("=" * W)

    # Per-function cross-backend summary
    func_names = [
        "potential_temperature", "vorticity", "frontogenesis",
        "q_vector", "compute_shear",
    ]
    print(f"\n  {'Function':<28s} {'Backends':>8s} {'Max |diff|':>12s} "
          f"{'RMSE':>12s} {'Rel RMSE%':>10s} {'Pearson r':>12s} {'Status':>8s}")
    print("  " + "-" * 92)

    for fn in func_names:
        if fn not in deep_results or len(deep_results[fn]) < 2:
            print(f"  {fn:<28s}     (only 1 backend)")
            continue
        entries = deep_results[fn]
        ref_label, ref_arr = entries[0]
        # Handle tuples (q_vector)
        if isinstance(ref_arr, tuple):
            for comp_i in range(len(ref_arr)):
                for label, arr in entries[1:]:
                    r = ref_arr[comp_i]
                    t = arr[comp_i]
                    d_arr = t - r
                    finite = d_arr[np.isfinite(d_arr)]
                    if finite.size == 0:
                        continue
                    max_ad = float(np.max(np.abs(finite)))
                    rmse = float(np.sqrt(np.mean(finite ** 2)))
                    rng = float(np.ptp(r[np.isfinite(r)])) if np.isfinite(r).sum() > 0 else 1.0
                    rel_pct = rmse / rng * 100.0 if rng > 0 else 0.0
                    valid = np.isfinite(r) & np.isfinite(t)
                    if valid.sum() > 1 and np.std(r[valid]) > 0:
                        pr, _ = sp_stats.pearsonr(r[valid].ravel(), t[valid].ravel())
                    else:
                        pr = float("nan")
                    ok = rel_pct < 0.1 and max_ad < 1e-6
                    status = "PASS" if ok else "WARN"
                    print(f"  {fn+'['+str(comp_i)+']':<28s} "
                          f"{ref_label[:3]+'/'+label[:3]:>8s} "
                          f"{max_ad:12.4e} {rmse:12.4e} {rel_pct:10.6f} "
                          f"{pr:12.10f} {status:>8s}")
        else:
            for label, arr in entries[1:]:
                d_arr = arr - ref_arr
                finite = d_arr[np.isfinite(d_arr)]
                if finite.size == 0:
                    continue
                max_ad = float(np.max(np.abs(finite)))
                rmse = float(np.sqrt(np.mean(finite ** 2)))
                rng = float(np.ptp(ref_arr[np.isfinite(ref_arr)])) if np.isfinite(ref_arr).sum() > 0 else 1.0
                rel_pct = rmse / rng * 100.0 if rng > 0 else 0.0
                valid = np.isfinite(ref_arr) & np.isfinite(arr)
                if valid.sum() > 1 and np.std(ref_arr[valid]) > 0:
                    pr, _ = sp_stats.pearsonr(ref_arr[valid].ravel(), arr[valid].ravel())
                else:
                    pr = float("nan")
                ok = rel_pct < 0.1
                status = "PASS" if ok else "WARN"
                print(f"  {fn:<28s} {ref_label[:3]+'/'+label[:3]:>8s} "
                      f"{max_ad:12.4e} {rmse:12.4e} {rel_pct:10.6f} "
                      f"{pr:12.10f} {status:>8s}")

    # CAPE/CIN summary row
    if len(cape_cin_backends) >= 2:
        bn = list(cape_cin_backends.keys())
        ref_label = bn[0]
        for field_idx, fname in enumerate(["CAPE", "CIN"]):
            ref_arr = cape_cin_backends[ref_label][field_idx]
            for bname in bn[1:]:
                test_arr = cape_cin_backends[bname][field_idx]
                d_arr = test_arr - ref_arr
                finite = d_arr[np.isfinite(d_arr)]
                if finite.size == 0:
                    continue
                max_ad = float(np.max(np.abs(finite)))
                rmse = float(np.sqrt(np.mean(finite ** 2)))
                rng = float(np.ptp(ref_arr[np.isfinite(ref_arr)])) if np.isfinite(ref_arr).sum() > 0 else 1.0
                rel_pct = rmse / rng * 100.0 if rng > 0 else 0.0
                valid = np.isfinite(ref_arr) & np.isfinite(test_arr)
                if valid.sum() > 1 and np.std(ref_arr[valid]) > 0:
                    pr, _ = sp_stats.pearsonr(ref_arr[valid].ravel(), test_arr[valid].ravel())
                else:
                    pr = float("nan")
                ok = rel_pct < 1.0
                status = "PASS" if ok else "WARN"
                print(f"  {'cape_cin['+fname+']':<28s} "
                      f"{ref_label[:3]+'/'+bname[:3]:>8s} "
                      f"{max_ad:12.4e} {rmse:12.4e} {rel_pct:10.6f} "
                      f"{pr:12.10f} {status:>8s}")

    # ========================================================================
    # Timing Summary
    # ========================================================================
    print()
    print("=" * W)
    print("  TIMING SUMMARY")
    print("=" * W)

    labels = [
        "potential_temperature", "vorticity", "frontogenesis",
        "q_vector", "compute_cape_cin", "compute_shear",
    ]
    timings = [
        (t_mp_pt, t_cpu_pt, t_cu_pt, t_gpu_pt),
        (t_mp_vort, t_cpu_vort, t_cu_vort, t_gpu_vort),
        (t_mp_fronto, t_cpu_fronto, t_cu_fronto, t_gpu_fronto),
        (t_mp_qvec, t_cpu_qvec, t_cu_qvec, t_gpu_qvec),
        (None, t_cpu_cape, t_cu_cape, t_gpu_cape),
        (None, t_cpu_shear, t_cu_shear, t_gpu_shear),
    ]

    print(f"  {'Function':{COL_NAME}s}"
          f" {'MetPy':>{COL_T}s}"
          f" {'Rust/CPU':>{COL_T}s}"
          f" {'met-cu':>{COL_T}s}"
          f" {'Rust/GPU':>{COL_T}s}"
          f" {'MP/Rust':>{COL_S}s}"
          f" {'Rust/CUDA':>{COL_S}s}")
    print("  " + "\u2500" * (W - 2))

    sum_mp = sum_cpu = sum_cu = sum_gpu = 0.0
    has_mp = has_cpu = has_cu = has_gpu = False
    for name, (tmp, tcpu, tcu, tgpu) in zip(labels, timings):
        row(name, tmp, tcpu, tcu, tgpu)
        if tmp is not None:
            sum_mp += tmp; has_mp = True
        if tcpu is not None:
            sum_cpu += tcpu; has_cpu = True
        if tcu is not None:
            sum_cu += tcu; has_cu = True
        if tgpu is not None:
            sum_gpu += tgpu; has_gpu = True

    print("  " + "\u2500" * (W - 2))
    print(f"  {'TOTAL':{COL_NAME}s}"
          f" {fmt(sum_mp) if has_mp else '\u2014':>{COL_T}s}"
          f" {fmt(sum_cpu) if has_cpu else '\u2014':>{COL_T}s}"
          f" {fmt(sum_cu) if has_cu else '\u2014':>{COL_T}s}"
          f" {fmt(sum_gpu) if has_gpu else '\u2014':>{COL_T}s}"
          f" {spd(sum_mp, sum_cpu) if has_mp and has_cpu else '\u2014':>{COL_S}s}"
          f" {spd(sum_cpu, sum_gpu) if has_cpu and has_gpu else '\u2014':>{COL_S}s}")

    print()
    print("=" * W)
    vstr = "ALL PASS" if all_pass else "SOME FAILURES"
    print(f"  Verification: {vstr}")
    print(f"  Tolerance: rtol={RTOL_KINE:.0e} (kinematics), rtol={RTOL_CAPE:.0e} (CAPE)")
    print(f"  Timing: {WARMUP} warmup + {NRUNS} runs, median")
    print("=" * W)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
