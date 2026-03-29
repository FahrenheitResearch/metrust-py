#!/usr/bin/env python
"""Benchmark 03 -- RAP Warm Front Analysis (850 hPa)

Scenario: RAP 13-km grid (337x451), single level at 850 hPa.
Synthetic warm front: warm moist air overrunning cold surface air.
    - Temperature 0-15 C (warm tongue from south)
    - Dewpoint -5 to 12 C (correlated with temperature)
    - Southerly winds veering to westerly across the front
    - Strong theta gradient in the frontal zone

Functions benchmarked:
    frontogenesis          (GPU)   -- key function for this scenario
    q_vector               (GPU)
    potential_temperature  (GPU)
    equiv_potential_temp   (GPU)
    dewpoint               (GPU)   -- from vapor pressure
    vorticity              (GPU)
    saturation_mixing_ratio (CPU only)

Backends: MetPy (Pint), metrust CPU, met-cu direct, metrust GPU.

Data correctness verification (per function, per backend):
    - Mean diff, max abs diff, RMSE, 99th percentile abs diff
    - Relative RMSE %, Pearson r
    - NaN / Inf audit
    - Physical plausibility range checks
    - Percentage of points with >1% and >0.1% relative error
    - Edge-case frontal-zone analysis
    - Histogram of absolute differences (log-scale buckets)

Usage:
    python tests/benchmarks/bench_03_rap_warm_front.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NY, NX = 337, 451           # RAP 13-km grid
DX = DY = 13_000.0          # metres
PRESSURE_HPA = 850.0
WARMUP = 1
TIMED = 3

# ---------------------------------------------------------------------------
# Imports -- all four backends
# ---------------------------------------------------------------------------
import metpy.calc as mpcalc
from metpy.units import units

import metrust.calc as mrcalc

HAS_GPU = False
GPU_NAME = "n/a"
mcucalc = None
try:
    import cupy as cp
    # Ensure met-cu is importable
    sys.path.insert(0, os.path.join("C:\\Users\\drew\\met-cu", "python"))
    import metcu.calc as mcucalc
    HAS_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Physical plausibility ranges per function
# ---------------------------------------------------------------------------
PHYS_RANGES = {
    "potential_temperature":  (200.0, 400.0),     # K -- theta at 850 hPa
    "equiv_potential_temp":   (200.0, 450.0),     # K -- theta_e can be higher
    "dewpoint":               (-80.0, 50.0),      # degC
    "saturation_mixing_ratio":(0.0, 0.2),         # kg/kg dimensionless
    "vorticity":              (-1e-3, 1e-3),       # 1/s
    "frontogenesis":          (-1e-6, 1e-6),       # K/m/s
    "q_vector[0]":            (-1e-10, 1e-10),     # m/s/Pa (Q1)
    "q_vector[1]":            (-1e-10, 1e-10),     # m/s/Pa (Q2)
}

# Pearson-r threshold (extremely tight -- backends should nearly perfectly
# correlate with MetPy ground truth)
PEARSON_R_THRESHOLD = 0.9999


# ---------------------------------------------------------------------------
# Synthetic warm-front data
# ---------------------------------------------------------------------------
def make_warm_front_data():
    """Build a realistic 850 hPa warm-front environment on a RAP-like grid.

    Returns a dict of plain numpy float64 arrays (no Pint units).
    """
    y = np.linspace(0, 1, NY)
    x = np.linspace(0, 1, NX)
    X, Y = np.meshgrid(x, y)   # (NY, NX)

    # -- warm tongue: Gaussian ridge from SSW to NNE --
    # Centre line tilts from (0.3, 0.0) to (0.7, 1.0)
    cx = 0.3 + 0.4 * Y
    sigma = 0.18
    warm_ridge = np.exp(-((X - cx) ** 2) / (2 * sigma ** 2))

    # Temperature (Celsius): 0 C in cold air, up to 15 C in warm tongue
    temperature = 0.0 + 15.0 * warm_ridge
    # Small N-S background gradient: warmer to south
    temperature += 3.0 * (1.0 - Y)

    # Dewpoint (Celsius): -5 in cold/dry, up to 12 in warm/moist tongue
    dewpoint_c = -5.0 + 17.0 * warm_ridge
    dewpoint_c = np.minimum(dewpoint_c, temperature - 0.5)  # keep Td < T

    # Potential temperature (K) at 850 hPa
    theta = (temperature + 273.15) * (1000.0 / PRESSURE_HPA) ** 0.2854

    # Winds: southerly in warm sector, veering to westerly in cold air
    # Blend factor: 1 in warm tongue, 0 in cold air
    blend = warm_ridge
    u_warm, v_warm = -2.0, 12.0    # SSW flow  (m/s)
    u_cold, v_cold = 8.0, 2.0      # W flow    (m/s)
    u = u_cold * (1 - blend) + u_warm * blend
    v = v_cold * (1 - blend) + v_warm * blend
    # Add smooth perturbation for realism
    u += 1.5 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y)
    v += 1.0 * np.cos(np.pi * X) * np.sin(2 * np.pi * Y)

    # Vapor pressure (hPa) from dewpoint via Tetens
    vp = 6.1078 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))

    # Pressure broadcast to 2D (needed by met-cu kernels that ravel all inputs)
    pressure_2d = np.full_like(temperature, PRESSURE_HPA)

    return dict(
        temperature=np.ascontiguousarray(temperature, dtype=np.float64),
        dewpoint_c=np.ascontiguousarray(dewpoint_c, dtype=np.float64),
        theta=np.ascontiguousarray(theta, dtype=np.float64),
        pressure_2d=np.ascontiguousarray(pressure_2d, dtype=np.float64),
        u=np.ascontiguousarray(u, dtype=np.float64),
        v=np.ascontiguousarray(v, dtype=np.float64),
        vp=np.ascontiguousarray(vp, dtype=np.float64),
        warm_ridge=warm_ridge,   # for frontal-zone mask
        X=X, Y=Y,
    )


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
def _sync_gpu():
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()


def bench(func, gpu=False):
    """1 warmup + 3 timed, return median ms."""
    if gpu:
        _sync_gpu()
    func()  # warmup
    if gpu:
        _sync_gpu()

    times = []
    for _ in range(TIMED):
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
        return "--"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd(ref, val):
    if ref is None or val is None or val <= 0:
        return "--"
    r = ref / val
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


# ---------------------------------------------------------------------------
# Magnitude extraction helpers
# ---------------------------------------------------------------------------
def _mag(x):
    """Extract magnitude from Pint or cupy, return numpy array."""
    if hasattr(x, "magnitude"):
        return np.asarray(x.magnitude)
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _mag_tuple(t):
    """Extract magnitudes from a tuple of results."""
    return tuple(_mag(x) for x in t)


# ---------------------------------------------------------------------------
# Deep verification engine
# ---------------------------------------------------------------------------
def _histogram_buckets(abs_diffs):
    """Return a compact string showing the distribution of absolute diffs
    across log-scale buckets."""
    n = abs_diffs.size
    if n == 0:
        return "  (no data)"
    edges = [0, 1e-30, 1e-20, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, np.inf]
    labels = ["==0", "<1e-30", "<1e-20", "<1e-15", "<1e-12",
              "<1e-10", "<1e-8", "<1e-6", "<1e-4", ">=1e-4"]
    # exact-zero bucket
    exact_zero = int(np.sum(abs_diffs == 0))
    counts = [exact_zero]
    for lo, hi in zip(edges[1:-1], edges[2:]):
        counts.append(int(np.sum((abs_diffs > lo) & (abs_diffs <= hi))))
    # keep only nonempty
    parts = []
    for lbl, cnt in zip(labels, counts):
        if cnt > 0:
            pct = 100.0 * cnt / n
            parts.append(f"{lbl}:{pct:.1f}%")
    return "  ".join(parts)


class DeepVerifyResult:
    """Stores all metrics for a single (function, backend) comparison."""
    __slots__ = (
        "name", "backend", "passed",
        "mean_diff", "max_abs_diff", "rmse", "pct99_abs_diff",
        "rel_rmse_pct", "pearson_r",
        "nan_count_ref", "nan_count_test", "inf_count_ref", "inf_count_test",
        "phys_lo", "phys_hi", "oob_ref", "oob_test",
        "pct_gt_1pct_relerr", "pct_gt_01pct_relerr",
        "frontal_zone_max_abs", "frontal_zone_rmse",
        "histogram_str",
    )

    def __init__(self, name, backend):
        self.name = name
        self.backend = backend
        self.passed = True


def deep_verify(func_name, backend_label, ref, test, warm_ridge,
                rtol=1e-4, atol=1e-20):
    """Exhaustive numerical comparison of *test* against *ref* (MetPy ground truth).

    Returns a DeepVerifyResult with all metrics populated.
    """
    res = DeepVerifyResult(func_name, backend_label)

    if ref is None or test is None:
        return res

    a = np.asarray(ref, dtype=np.float64).ravel()
    b = np.asarray(test, dtype=np.float64).ravel()

    if a.size != b.size:
        print(f"      DEEP VERIFY {func_name} [{backend_label}]: "
              f"FAIL  shape mismatch ref={a.size} test={b.size}")
        res.passed = False
        return res

    # --- NaN / Inf audit ---
    res.nan_count_ref = int(np.sum(np.isnan(a)))
    res.nan_count_test = int(np.sum(np.isnan(b)))
    res.inf_count_ref = int(np.sum(np.isinf(a)))
    res.inf_count_test = int(np.sum(np.isinf(b)))

    nan_inf_ok = True
    if res.nan_count_test > res.nan_count_ref:
        nan_inf_ok = False
    if res.inf_count_test > 0 and res.inf_count_ref == 0:
        nan_inf_ok = False

    # Finite mask for remaining analysis
    mask = np.isfinite(a) & np.isfinite(b)
    n_valid = int(mask.sum())
    if n_valid == 0:
        print(f"      DEEP VERIFY {func_name} [{backend_label}]: all NaN/Inf, skip")
        return res

    a_m, b_m = a[mask], b[mask]
    diffs = a_m - b_m
    abs_diffs = np.abs(diffs)

    # --- Core metrics ---
    res.mean_diff = float(np.mean(diffs))
    res.max_abs_diff = float(np.max(abs_diffs))
    res.rmse = float(np.sqrt(np.mean(diffs ** 2)))
    res.pct99_abs_diff = float(np.percentile(abs_diffs, 99))

    # Relative RMSE% (relative to ref magnitude range)
    ref_range = float(np.max(np.abs(a_m)))
    if ref_range > 0:
        res.rel_rmse_pct = 100.0 * res.rmse / ref_range
    else:
        res.rel_rmse_pct = 0.0

    # --- Pearson r ---
    if n_valid >= 2 and np.std(a_m) > 0 and np.std(b_m) > 0:
        r_val, _ = pearsonr(a_m, b_m)
        res.pearson_r = float(r_val)
    else:
        # Perfect match by construction (constant field)
        res.pearson_r = 1.0

    # --- Physical plausibility ---
    phys_key = func_name
    if phys_key in PHYS_RANGES:
        lo, hi = PHYS_RANGES[phys_key]
        res.phys_lo, res.phys_hi = lo, hi
        res.oob_ref = int(np.sum((a_m < lo) | (a_m > hi)))
        res.oob_test = int(np.sum((b_m < lo) | (b_m > hi)))
    else:
        res.phys_lo = res.phys_hi = None
        res.oob_ref = res.oob_test = 0

    # --- % points with relative error > 1% and > 0.1% ---
    denom = np.maximum(np.abs(a_m), 1e-30)
    rel_err = abs_diffs / denom
    res.pct_gt_1pct_relerr = 100.0 * float(np.sum(rel_err > 0.01)) / n_valid
    res.pct_gt_01pct_relerr = 100.0 * float(np.sum(rel_err > 0.001)) / n_valid

    # --- Frontal-zone edge-case analysis ---
    # The frontal zone is where the warm_ridge gradient is steepest --
    # use 0.2 < warm_ridge < 0.8 as the transition band
    if warm_ridge is not None:
        fz = warm_ridge.ravel()
        if fz.size == a.size:
            fz_mask = mask & (fz > 0.2) & (fz < 0.8)
            n_fz = int(fz_mask.sum())
            if n_fz > 0:
                fz_diffs = np.abs(a[fz_mask] - b[fz_mask])
                res.frontal_zone_max_abs = float(np.max(fz_diffs))
                res.frontal_zone_rmse = float(np.sqrt(np.mean(fz_diffs ** 2)))
            else:
                res.frontal_zone_max_abs = 0.0
                res.frontal_zone_rmse = 0.0
        else:
            res.frontal_zone_max_abs = None
            res.frontal_zone_rmse = None
    else:
        res.frontal_zone_max_abs = None
        res.frontal_zone_rmse = None

    # --- Histogram of diffs ---
    res.histogram_str = _histogram_buckets(abs_diffs)

    # --- Overall PASS/FAIL ---
    allclose_ok = np.allclose(a_m, b_m, rtol=rtol, atol=atol)
    pearson_ok = res.pearson_r >= PEARSON_R_THRESHOLD
    phys_ok = (res.oob_test == 0) if res.phys_lo is not None else True

    res.passed = allclose_ok and pearson_ok and nan_inf_ok and phys_ok
    return res


def print_deep_result(r):
    """Pretty-print a single DeepVerifyResult."""
    tag = "PASS" if r.passed else "FAIL"
    print(f"      [{tag}] {r.name} [{r.backend}]")
    print(f"        Mean diff        : {r.mean_diff:+.6e}")
    print(f"        Max |diff|       : {r.max_abs_diff:.6e}")
    print(f"        RMSE             : {r.rmse:.6e}")
    print(f"        99th pct |diff|  : {r.pct99_abs_diff:.6e}")
    print(f"        Relative RMSE %  : {r.rel_rmse_pct:.10f}%")
    print(f"        Pearson r        : {r.pearson_r:.10f}  "
          f"(threshold {PEARSON_R_THRESHOLD})")
    print(f"        NaN  ref/test    : {r.nan_count_ref} / {r.nan_count_test}")
    print(f"        Inf  ref/test    : {r.inf_count_ref} / {r.inf_count_test}")
    if r.phys_lo is not None:
        print(f"        Phys range       : [{r.phys_lo:.2e}, {r.phys_hi:.2e}]"
              f"  OOB ref={r.oob_ref} test={r.oob_test}")
    print(f"        Pts >1%  rel err : {r.pct_gt_1pct_relerr:.4f}%")
    print(f"        Pts >0.1% rel err: {r.pct_gt_01pct_relerr:.4f}%")
    if r.frontal_zone_max_abs is not None:
        print(f"        Frontal zone max : {r.frontal_zone_max_abs:.6e}")
        print(f"        Frontal zone RMSE: {r.frontal_zone_rmse:.6e}")
    print(f"        Diff histogram   : {r.histogram_str}")


def deep_verify_all(func_name, ref, results_dict, warm_ridge,
                    rtol=1e-4, atol=1e-20):
    """Run deep_verify for every backend in results_dict against *ref*.

    results_dict: {"cpu": array, "mcu": array_or_None, "gpu": array_or_None}
    Returns list of DeepVerifyResult, all_ok bool.
    """
    all_results = []
    all_ok = True
    for backend_label, test_arr in results_dict.items():
        if test_arr is None:
            continue
        r = deep_verify(func_name, backend_label, ref, test_arr,
                        warm_ridge, rtol=rtol, atol=atol)
        all_results.append(r)
        print_deep_result(r)
        if not r.passed:
            all_ok = False
    return all_results, all_ok


def deep_verify_tuple(func_name, ref_tuple, results_dict, warm_ridge,
                      rtol=1e-4, atol=1e-20):
    """Deep-verify tuple-valued results (e.g., q_vector returning (Q1, Q2))."""
    all_results = []
    all_ok = True
    for i, ref_component in enumerate(ref_tuple):
        comp_name = f"{func_name}[{i}]"
        comp_dict = {}
        for backend_label, test_tuple in results_dict.items():
            if test_tuple is None:
                continue
            comp_dict[backend_label] = test_tuple[i]
        results, ok = deep_verify_all(comp_name, ref_component, comp_dict,
                                      warm_ridge, rtol=rtol, atol=atol)
        all_results.extend(results)
        all_ok = all_ok and ok
    return all_results, all_ok


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
COL_NAME = 32
COL_T = 12
COL_S = 9

rows = []   # (name, t_mp, t_cpu, t_mcu, t_gpu, gpu_flag)
all_deep_results = []   # list of DeepVerifyResult for summary table


def header():
    print()
    hdr = (f"  {'Function':<{COL_NAME}s}"
           f" {'MetPy':>{COL_T}s}"
           f" {'Rust/CPU':>{COL_T}s}"
           f" {'met-cu':>{COL_T}s}"
           f" {'Rust/GPU':>{COL_T}s}"
           f" {'Rust/MP':>{COL_S}s}"
           f" {'GPU/CPU':>{COL_S}s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))


def record(name, t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=False):
    rows.append((name, t_mp, t_cpu, t_mcu, t_gpu, gpu_flag))
    star = "*" if gpu_flag else " "
    print(f" {star}{name:<{COL_NAME}s}"
          f" {fmt(t_mp):>{COL_T}s}"
          f" {fmt(t_cpu):>{COL_T}s}"
          f" {fmt(t_mcu):>{COL_T}s}"
          f" {fmt(t_gpu):>{COL_T}s}"
          f" {spd(t_mp, t_cpu):>{COL_S}s}"
          f" {spd(t_cpu, t_gpu):>{COL_S}s}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    global all_deep_results

    print("=" * 100)
    print("  BENCHMARK 03: RAP Warm Front Analysis (850 hPa)")
    print(f"  Grid: {NY}x{NX} ({NY*NX:,} pts)  dx=dy={DX/1000:.0f} km")
    print(f"  GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print(f"  Timing: {WARMUP} warmup + {TIMED} timed, median")
    print("=" * 100)

    d = make_warm_front_data()
    warm_ridge = d["warm_ridge"]
    print(f"\n  Synthetic data: T=[{d['temperature'].min():.1f}, {d['temperature'].max():.1f}] C"
          f"  Td=[{d['dewpoint_c'].min():.1f}, {d['dewpoint_c'].max():.1f}] C"
          f"  theta=[{d['theta'].min():.1f}, {d['theta'].max():.1f}] K"
          f"  |V|=[{np.hypot(d['u'], d['v']).min():.1f}, {np.hypot(d['u'], d['v']).max():.1f}] m/s")

    # -- MetPy Pint quantities (not timed) --
    p_q = PRESSURE_HPA * units.hPa
    t_q = d["temperature"] * units.degC
    td_q = d["dewpoint_c"] * units.degC
    theta_q = d["theta"] * units.K
    u_q = d["u"] * units("m/s")
    v_q = d["v"] * units("m/s")
    dx_q = DX * units.m
    dy_q = DY * units.m
    vp_q = d["vp"] * units.hPa

    all_ok = True

    # ==================================================================
    # 1. potential_temperature (GPU)
    # ==================================================================
    header()
    print()
    print("  --- Thermodynamics ---")

    # MetPy
    t_mp = bench(lambda: mpcalc.potential_temperature(p_q, t_q))
    ref_pt = _mag(mpcalc.potential_temperature(p_q, t_q))

    # metrust CPU
    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.potential_temperature(PRESSURE_HPA, d["temperature"]))
    res_cpu = _mag(mrcalc.potential_temperature(PRESSURE_HPA, d["temperature"]))

    # met-cu direct
    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.potential_temperature(PRESSURE_HPA, d["temperature"]), gpu=True)
        res_mcu = _mag(mcucalc.potential_temperature(PRESSURE_HPA, d["temperature"]))

    # metrust GPU
    t_gpu = None
    res_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.potential_temperature(PRESSURE_HPA, d["temperature"]), gpu=True)
        res_gpu = _mag(mrcalc.potential_temperature(PRESSURE_HPA, d["temperature"]))
        mrcalc.set_backend("cpu")

    record("potential_temperature", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: potential_temperature --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu, "gpu": res_gpu}
    dv_results, dv_ok = deep_verify_all(
        "potential_temperature", ref_pt, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 2. equivalent_potential_temperature (GPU)
    # ==================================================================
    t_mp = bench(lambda: mpcalc.equivalent_potential_temperature(p_q, t_q, td_q))
    ref_ept = _mag(mpcalc.equivalent_potential_temperature(p_q, t_q, td_q))

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.equivalent_potential_temperature(
        PRESSURE_HPA, d["temperature"], d["dewpoint_c"]))
    res_cpu = _mag(mrcalc.equivalent_potential_temperature(
        PRESSURE_HPA, d["temperature"], d["dewpoint_c"]))

    # met-cu direct (needs 2D pressure to avoid scalar-broadcast kernel bug)
    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.equivalent_potential_temperature(
            d["pressure_2d"], d["temperature"], d["dewpoint_c"]), gpu=True)
        res_mcu = _mag(mcucalc.equivalent_potential_temperature(
            d["pressure_2d"], d["temperature"], d["dewpoint_c"]))

    # metrust GPU (use 2D pressure to work around met-cu scalar-broadcast bug)
    t_gpu = None
    res_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.equivalent_potential_temperature(
            d["pressure_2d"], d["temperature"], d["dewpoint_c"]), gpu=True)
        res_gpu = _mag(mrcalc.equivalent_potential_temperature(
            d["pressure_2d"], d["temperature"], d["dewpoint_c"]))
        mrcalc.set_backend("cpu")

    record("equiv_potential_temp", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: equiv_potential_temp --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu, "gpu": res_gpu}
    dv_results, dv_ok = deep_verify_all(
        "equiv_potential_temp", ref_ept, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 3. dewpoint from vapor pressure (GPU)
    # ==================================================================
    t_mp = bench(lambda: mpcalc.dewpoint(vp_q))
    ref_dp = _mag(mpcalc.dewpoint(vp_q))

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.dewpoint(d["vp"]))
    res_cpu = _mag(mrcalc.dewpoint(d["vp"]))

    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.dewpoint(d["vp"]), gpu=True)
        res_mcu = _mag(mcucalc.dewpoint(d["vp"]))

    t_gpu = None
    res_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.dewpoint(d["vp"]), gpu=True)
        res_gpu = _mag(mrcalc.dewpoint(d["vp"]))
        mrcalc.set_backend("cpu")

    record("dewpoint", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: dewpoint --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu, "gpu": res_gpu}
    dv_results, dv_ok = deep_verify_all(
        "dewpoint", ref_dp, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 4. saturation_mixing_ratio (CPU only)
    # ==================================================================
    t_mp = bench(lambda: mpcalc.saturation_mixing_ratio(p_q, t_q))
    ref_smr = _mag(mpcalc.saturation_mixing_ratio(p_q, t_q))

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.saturation_mixing_ratio(PRESSURE_HPA, d["temperature"]))
    res_cpu = _mag(mrcalc.saturation_mixing_ratio(PRESSURE_HPA, d["temperature"]))

    # met-cu (needs 2D pressure to avoid scalar-broadcast kernel bug)
    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.saturation_mixing_ratio(d["pressure_2d"], d["temperature"]), gpu=True)
        res_mcu = _mag(mcucalc.saturation_mixing_ratio(d["pressure_2d"], d["temperature"]))

    record("saturation_mixing_ratio", t_mp, t_cpu, t_mcu, None, gpu_flag=False)

    print()
    print("    -- Deep verification: saturation_mixing_ratio --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu}
    dv_results, dv_ok = deep_verify_all(
        "saturation_mixing_ratio", ref_smr, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 5. vorticity (GPU)
    # ==================================================================
    print()
    print("  --- Kinematics ---")

    t_mp = bench(lambda: mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q))
    ref_vort = _mag(mpcalc.vorticity(u_q, v_q, dx=dx_q, dy=dy_q))

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.vorticity(d["u"], d["v"], dx=DX, dy=DY))
    res_cpu = _mag(mrcalc.vorticity(d["u"], d["v"], dx=DX, dy=DY))

    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.vorticity(d["u"], d["v"], dx=DX, dy=DY), gpu=True)
        res_mcu = _mag(mcucalc.vorticity(d["u"], d["v"], dx=DX, dy=DY))

    t_gpu = None
    res_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.vorticity(d["u"], d["v"], dx=DX, dy=DY), gpu=True)
        res_gpu = _mag(mrcalc.vorticity(d["u"], d["v"], dx=DX, dy=DY))
        mrcalc.set_backend("cpu")

    record("vorticity", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: vorticity --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu, "gpu": res_gpu}
    dv_results, dv_ok = deep_verify_all(
        "vorticity", ref_vort, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 6. frontogenesis (GPU) -- KEY function
    # ==================================================================
    t_mp = bench(lambda: mpcalc.frontogenesis(theta_q, u_q, v_q, dx=dx_q, dy=dy_q))
    ref_fronto = _mag(mpcalc.frontogenesis(theta_q, u_q, v_q, dx=dx_q, dy=dy_q))

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY))
    res_cpu = _mag(mrcalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY))

    t_mcu = None
    res_mcu = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY), gpu=True)
        res_mcu = _mag(mcucalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY))

    t_gpu = None
    res_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY), gpu=True)
        res_gpu = _mag(mrcalc.frontogenesis(d["theta"], d["u"], d["v"], dx=DX, dy=DY))
        mrcalc.set_backend("cpu")

    record("frontogenesis", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: frontogenesis --")
    results_dict = {"cpu": res_cpu, "mcu": res_mcu, "gpu": res_gpu}
    dv_results, dv_ok = deep_verify_all(
        "frontogenesis", ref_fronto, results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # 7. q_vector (GPU)
    # ==================================================================
    # MetPy: q_vector(u, v, temperature, pressure, dx, dy) -- temperature in K
    t_k_q = (d["temperature"] + 273.15) * units.K
    t_mp = bench(lambda: mpcalc.q_vector(u_q, v_q, t_k_q, p_q, dx=dx_q, dy=dy_q))
    ref_qv = _mag_tuple(mpcalc.q_vector(u_q, v_q, t_k_q, p_q, dx=dx_q, dy=dy_q))

    # metrust CPU: temperature in degC, pressure scalar hPa
    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.q_vector(d["u"], d["v"], d["temperature"], PRESSURE_HPA, dx=DX, dy=DY))
    res_cpu = _mag_tuple(mrcalc.q_vector(d["u"], d["v"], d["temperature"], PRESSURE_HPA, dx=DX, dy=DY))

    # met-cu direct: temperature in K, pressure scalar hPa
    t_k_raw = d["temperature"] + 273.15
    t_mcu = None
    res_mcu_qv = None
    if mcucalc:
        t_mcu = bench(lambda: mcucalc.q_vector(d["u"], d["v"], t_k_raw, PRESSURE_HPA, dx=DX, dy=DY), gpu=True)
        res_mcu_qv = _mag_tuple(mcucalc.q_vector(d["u"], d["v"], t_k_raw, PRESSURE_HPA, dx=DX, dy=DY))

    # metrust GPU: temperature in degC, pressure scalar hPa
    t_gpu = None
    res_gpu_qv = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.q_vector(d["u"], d["v"], d["temperature"], PRESSURE_HPA, dx=DX, dy=DY), gpu=True)
        res_gpu_qv = _mag_tuple(mrcalc.q_vector(d["u"], d["v"], d["temperature"], PRESSURE_HPA, dx=DX, dy=DY))
        mrcalc.set_backend("cpu")

    record("q_vector", t_mp, t_cpu, t_mcu, t_gpu, gpu_flag=True)

    print()
    print("    -- Deep verification: q_vector --")
    qv_results_dict = {"cpu": res_cpu, "mcu": res_mcu_qv, "gpu": res_gpu_qv}
    dv_results, dv_ok = deep_verify_tuple(
        "q_vector", ref_qv, qv_results_dict, warm_ridge)
    all_deep_results.extend(dv_results)
    all_ok = all_ok and dv_ok

    # ==================================================================
    # Summary Tables
    # ==================================================================
    print()
    print("=" * 100)
    print("  TIMING SUMMARY")
    print("=" * 100)
    print()
    print(f"  {'Function':<{COL_NAME}s}"
          f" {'MetPy':>{COL_T}s}"
          f" {'Rust/CPU':>{COL_T}s}"
          f" {'met-cu':>{COL_T}s}"
          f" {'Rust/GPU':>{COL_T}s}"
          f" {'Rust/MP':>{COL_S}s}"
          f" {'GPU/CPU':>{COL_S}s}")
    print("  " + "-" * 96)
    for name, t_mp, t_cpu, t_mcu, t_gpu, gf in rows:
        star = "*" if gf else " "
        print(f" {star}{name:<{COL_NAME}s}"
              f" {fmt(t_mp):>{COL_T}s}"
              f" {fmt(t_cpu):>{COL_T}s}"
              f" {fmt(t_mcu):>{COL_T}s}"
              f" {fmt(t_gpu):>{COL_T}s}"
              f" {spd(t_mp, t_cpu):>{COL_S}s}"
              f" {spd(t_cpu, t_gpu):>{COL_S}s}")

    # Geometric-mean speedups
    cpu_ratios = [t_mp / t_cpu for _, t_mp, t_cpu, _, _, _ in rows
                  if t_mp and t_cpu and t_cpu > 0]
    if cpu_ratios:
        geo_cpu = np.exp(np.mean(np.log(cpu_ratios)))
        print(f"\n  Geometric mean Rust/CPU vs MetPy: {geo_cpu:.1f}x")

    gpu_ratios = [t_cpu / t_gpu for _, _, t_cpu, _, t_gpu, _ in rows
                  if t_cpu and t_gpu and t_gpu > 0]
    if gpu_ratios:
        geo_gpu = np.exp(np.mean(np.log(gpu_ratios)))
        print(f"  Geometric mean GPU vs Rust/CPU:   {geo_gpu:.1f}x")

    # ==================================================================
    # Data Correctness Summary Table
    # ==================================================================
    print()
    print("=" * 100)
    print("  DATA CORRECTNESS SUMMARY  (vs MetPy ground truth)")
    print("=" * 100)
    print()

    hdr_fmt = ("  {:<30s} {:<8s} {:>12s} {:>12s} {:>12s} {:>12s}"
               " {:>8s} {:>6s}")
    print(hdr_fmt.format(
        "Function", "Backend", "RMSE", "Max|diff|", "99th pct",
        "RelRMSE%", "Pearson", "P/F"))
    print("  " + "-" * 108)
    row_fmt = ("  {:<30s} {:<8s} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.8f}"
               " {:>8.6f} {:>6s}")
    for r in all_deep_results:
        pf = "PASS" if r.passed else "FAIL"
        print(row_fmt.format(
            r.name, r.backend,
            r.rmse, r.max_abs_diff, r.pct99_abs_diff,
            r.rel_rmse_pct, r.pearson_r, pf))

    # ==================================================================
    # Per-function PASS/FAIL roll-up
    # ==================================================================
    print()
    print("  Per-function roll-up:")
    print("  " + "-" * 60)

    func_names_seen = []
    for r in all_deep_results:
        if r.name not in func_names_seen:
            func_names_seen.append(r.name)

    for fname in func_names_seen:
        func_results = [r for r in all_deep_results if r.name == fname]
        func_ok = all(r.passed for r in func_results)
        backends_str = ", ".join(
            f"{r.backend}={'PASS' if r.passed else 'FAIL'}"
            for r in func_results)
        overall = "PASS" if func_ok else "FAIL"
        print(f"    [{overall}] {fname:<30s} -- {backends_str}")

    # ==================================================================
    # Final verdict
    # ==================================================================
    print()
    if all_ok:
        print("  ALL VERIFICATION CHECKS PASSED")
    else:
        print("  *** SOME VERIFICATION CHECKS FAILED ***")
    print()
    print("  * = GPU-accelerated kernel available")


if __name__ == "__main__":
    main()
