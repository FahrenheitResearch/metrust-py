#!/usr/bin/env python
"""Benchmark 04 -- NAM Winter Storm Precipitable Water & Moisture

Scenario: NAM 12 km grid (428x614), 30 vertical levels.
Classic Nor'easter moisture plume with realistic 3-D temperature, moisture,
pressure, and height profiles.

Functions benchmarked:
  - compute_pw          (GPU)  -- precipitable water, 3D -> 2D
  - dewpoint            (GPU)  -- dewpoint from vapor pressure
  - saturation_vapor_pressure  (CPU)
  - relative_humidity_from_dewpoint  (CPU)
  - mixing_ratio        (CPU)
  - potential_temperature (GPU)

Four backends: MetPy (Pint), metrust CPU, met-cu direct (CUDA), metrust GPU.

MetPy has only 1-D precipitable_water(pressure, dewpoint) -- no grid compute_pw.
For compute_pw we compare metrust CPU / met-cu / metrust GPU only.

Usage:
    python tests/benchmarks/bench_04_nam_winter.py
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Imports
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
    mrcalc.set_backend("cpu")  # reset to cpu
    HAS_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
except Exception as exc:
    print(f"  GPU init note: {exc}")


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def _sync():
    cp.cuda.Stream.null.synchronize()


def bench(func, n=3, gpu=False):
    """1 warmup + n timed runs, return median ms."""
    # warmup
    if gpu:
        _sync()
    func()
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


def spd(ref, fast):
    if ref is None or fast is None or fast <= 0:
        return "--"
    r = ref / fast
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


def _mag(x):
    """Strip Pint units and cupy, return numpy array."""
    if hasattr(x, "magnitude"):
        x = x.magnitude
    # Handle cupy arrays
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Synthetic NAM winter-storm data
# ---------------------------------------------------------------------------

def build_nam_data():
    """Build synthetic NAM-like 3-D fields mimicking a Nor'easter moisture plume."""
    np.random.seed(2024)

    ny, nx, nz = 428, 614, 30

    # --- Pressure levels: 1000 -> 100 hPa (surface first, decreasing upward) ---
    plev_hPa = np.linspace(1000, 100, nz)  # hPa, shape (nz,)
    plev_Pa = plev_hPa * 100.0             # Pa

    # 3-D pressure field (nz, ny, nx) - broadcast
    p3_hPa = np.broadcast_to(plev_hPa[:, None, None], (nz, ny, nx)).copy()
    p3_Pa = p3_hPa * 100.0

    # --- Temperature: warm moist surface, decreasing with height ---
    # Surface temp: 2-15 C (warmer in south/east = moisture plume)
    lat_frac = np.linspace(0, 1, ny)[:, None]   # 0=south, 1=north
    lon_frac = np.linspace(0, 1, nx)[None, :]   # 0=west, 1=east

    # Surface temperature: warmer in SE (moisture plume), colder in NW
    t_sfc_C = 15.0 - 10.0 * lat_frac - 3.0 * (1 - lon_frac)
    t_sfc_C += np.random.randn(ny, nx) * 0.5  # slight noise

    # Lapse rate: ~6.5 C/km in troposphere, moist adiabatic lower levels
    z_km = np.linspace(0, 16, nz)  # approximate heights
    lapse = 6.5  # C/km

    # 3-D temperature (Celsius)
    t3_C = np.empty((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        t3_C[k] = t_sfc_C - lapse * z_km[k]
    # Clamp minimum to prevent unrealistic values
    t3_C = np.clip(t3_C, -80.0, 45.0)

    # --- Dewpoint: close to temperature in moisture plume, drier aloft ---
    # Dewpoint depression increases with height and distance from plume
    plume_center_y = int(ny * 0.35)  # plume centered in southern portion
    plume_center_x = int(nx * 0.6)   # eastern portion
    dy_plume = (np.arange(ny)[:, None] - plume_center_y) / ny
    dx_plume = (np.arange(nx)[None, :] - plume_center_x) / nx
    dist_plume = np.sqrt(dy_plume**2 + dx_plume**2)

    # Dewpoint depression: small in plume core, larger away & aloft
    td3_C = np.empty((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        height_factor = (z_km[k] / 16.0) ** 0.5  # dries faster aloft
        spatial_depression = 2.0 + 15.0 * dist_plume + 20.0 * height_factor
        td3_C[k] = t3_C[k] - spatial_depression
    td3_C = np.clip(td3_C, -85.0, t3_C - 0.1)  # Td <= T

    # --- Mixing ratio (kg/kg) from temperature and dewpoint ---
    # Use Tetens formula: e_s = 6.112 * exp(17.67 * Td / (Td + 243.5))
    e_td = 6.112 * np.exp(17.67 * td3_C / (td3_C + 243.5))  # hPa
    qvapor_kgkg = 0.622 * e_td / (p3_hPa - e_td)
    qvapor_kgkg = np.clip(qvapor_kgkg, 1e-7, 0.04)

    # --- Vapor pressure (hPa) for dewpoint benchmark ---
    # Tetens from dewpoint: gives actual vapor pressure
    # We'll use the 850 hPa slice for 2-D benchmarks
    i850 = int(np.argmin(np.abs(plev_hPa - 850.0)))

    # 2-D slices at 850 hPa
    t850_C = t3_C[i850].copy()
    td850_C = td3_C[i850].copy()
    vp850_hPa = 6.112 * np.exp(17.67 * td850_C / (td850_C + 243.5))

    # --- Geopotential heights (approximate, for context only) ---
    H = 7.4  # scale height km
    z3_m = np.empty((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        z3_m[k] = H * 1000.0 * np.log(1013.25 / plev_hPa[k])
    z3_m += np.random.randn(nz, ny, nx) * 10.0  # small perturbation

    print(f"  Grid: {ny}x{nx}, {nz} levels ({plev_hPa[0]:.0f}-{plev_hPa[-1]:.0f} hPa)")
    print(f"  T range:  [{t3_C.min():.1f}, {t3_C.max():.1f}] C")
    print(f"  Td range: [{td3_C.min():.1f}, {td3_C.max():.1f}] C")
    print(f"  qv range: [{qvapor_kgkg.min():.2e}, {qvapor_kgkg.max():.2e}] kg/kg")
    print(f"  PW estimate: {np.mean(qvapor_kgkg.sum(axis=0)) * 100:.1f} mm (crude sum)")

    return dict(
        ny=ny, nx=nx, nz=nz,
        plev_hPa=plev_hPa, plev_Pa=plev_Pa,
        p3_hPa=p3_hPa, p3_Pa=p3_Pa,
        t3_C=t3_C, td3_C=td3_C,
        qvapor_kgkg=qvapor_kgkg,
        z3_m=z3_m,
        i850=i850,
        t850_C=t850_C, td850_C=td850_C,
        vp850_hPa=vp850_hPa,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    W = 106
    print("=" * W)
    print("  BENCHMARK 04: NAM Winter Storm -- Precipitable Water & Moisture")
    print(f"  GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print("=" * W)
    print()

    d = build_nam_data()
    ny, nx, nz = d["ny"], d["nx"], d["nz"]
    N = 3   # timed iterations (median of 3)

    # Pre-build MetPy Pint arrays (not timed)
    t850q = d["t850_C"] * units.degC
    td850q = d["td850_C"] * units.degC
    vp850q = d["vp850_hPa"] * units.hPa
    p850q = 850.0 * units.hPa

    results = []   # (name, t_metpy, t_mr_cpu, t_mcu, t_mr_gpu, note)

    def hdr():
        print(f"  {'Function':32s}"
              f" {'MetPy':>10s} {'MR-CPU':>10s} {'met-cu':>10s} {'MR-GPU':>10s}"
              f" {'MP/CPU':>8s} {'CPU/GPU':>8s}")
        print("  " + "-" * (W - 4))

    def row(name, t_mp, t_cpu, t_mcu, t_gpu, note=""):
        results.append((name, t_mp, t_cpu, t_mcu, t_gpu, note))
        print(f"  {name:32s}"
              f" {fmt(t_mp):>10s} {fmt(t_cpu):>10s}"
              f" {fmt(t_mcu):>10s} {fmt(t_gpu):>10s}"
              f" {spd(t_mp, t_cpu):>8s} {spd(t_cpu, t_gpu):>8s}"
              f"  {note}")

    # ==================================================================
    # SECTION 1: 2-D thermodynamics at 850 hPa (428x614 = 262,792 pts)
    # ==================================================================
    print()
    print(f"-- 2D THERMODYNAMICS (850 hPa, {ny}x{nx}) " + "-" * 50)
    hdr()

    # --- potential_temperature (GPU) ---
    t_mp = bench(lambda: mpcalc.potential_temperature(p850q, t850q), N)

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.potential_temperature(850.0, d["t850_C"]), N)

    t_mcu = None
    t_gpu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.potential_temperature(850.0, d["t850_C"]), N, gpu=True)
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.potential_temperature(850.0, d["t850_C"]), N, gpu=True)
        mrcalc.set_backend("cpu")

    row("potential_temperature", t_mp, t_cpu, t_mcu, t_gpu, "GPU")

    # --- dewpoint (GPU) ---
    t_mp = bench(lambda: mpcalc.dewpoint(vp850q), N)

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.dewpoint(d["vp850_hPa"]), N)

    t_mcu = None
    t_gpu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.dewpoint(d["vp850_hPa"]), N, gpu=True)
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.dewpoint(d["vp850_hPa"]), N, gpu=True)
        mrcalc.set_backend("cpu")

    row("dewpoint", t_mp, t_cpu, t_mcu, t_gpu, "GPU")

    # --- saturation_vapor_pressure (CPU) ---
    t_mp = bench(lambda: mpcalc.saturation_vapor_pressure(t850q), N)

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.saturation_vapor_pressure(d["t850_C"]), N)

    t_mcu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.saturation_vapor_pressure(d["t850_C"]), N, gpu=True)

    row("saturation_vapor_pressure", t_mp, t_cpu, t_mcu, None, "CPU")

    # --- relative_humidity_from_dewpoint (CPU) ---
    t_mp = bench(lambda: mpcalc.relative_humidity_from_dewpoint(t850q, td850q), N)

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.relative_humidity_from_dewpoint(d["t850_C"], d["td850_C"]), N)

    t_mcu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.relative_humidity_from_dewpoint(d["t850_C"], d["td850_C"]), N, gpu=True)

    row("rh_from_dewpoint", t_mp, t_cpu, t_mcu, None, "CPU")

    # --- mixing_ratio (CPU) ---
    # MetPy: mixing_ratio(partial_pressure, total_pressure)
    # metrust: mixing_ratio(partial_pressure, total_pressure) -- same path
    # met-cu: mixing_ratio(partial_pressure, total_pressure)
    t_mp = bench(lambda: mpcalc.mixing_ratio(vp850q, p850q), N)

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.mixing_ratio(d["vp850_hPa"], 850.0), N)

    t_mcu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.mixing_ratio(d["vp850_hPa"], 850.0), N, gpu=True)

    row("mixing_ratio", t_mp, t_cpu, t_mcu, None, "CPU")

    # ==================================================================
    # SECTION 2: 3-D grid compute_pw (30 x 428 x 614 -> 428 x 614)
    # ==================================================================
    print()
    print(f"-- 3D GRID: compute_pw ({nz}x{ny}x{nx} -> {ny}x{nx}) " + "-" * 40)
    hdr()

    # MetPy has no grid compute_pw -- 1D only
    t_mp = None

    mrcalc.set_backend("cpu")
    t_cpu = bench(lambda: mrcalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"]), N)

    t_mcu = None
    t_gpu = None
    if HAS_GPU:
        t_mcu = bench(lambda: mcucalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"]), N, gpu=True)
        mrcalc.set_backend("gpu")
        t_gpu = bench(lambda: mrcalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"]), N, gpu=True)
        mrcalc.set_backend("cpu")

    row("compute_pw (3D->2D)", t_mp, t_cpu, t_mcu, t_gpu, "GPU, no MetPy equiv")

    # ==================================================================
    # DEEP DATA CORRECTNESS VERIFICATION
    # ==================================================================
    print()
    print("=" * W)
    print("  DEEP DATA CORRECTNESS VERIFICATION")
    print("=" * W)

    from scipy.stats import pearsonr

    all_pass = True
    verify_rows = []  # (func, pair, pass/fail, mean_diff, max_abs, rmse, pct99, rel_rmse, r, pct_gt_1pct, pct_gt_01pct)

    def _to_np(x):
        """Strip Pint units and cupy, return float64 numpy array."""
        v = _mag(x)
        if hasattr(v, "get"):
            v = v.get()
        return np.asarray(v, dtype=np.float64)

    def _nan_inf_audit(arr, label):
        """Check for NaN/Inf, return (n_nan, n_inf, total)."""
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        total = arr.size
        return n_nan, n_inf, total

    def _physical_bounds_check(arr, lo, hi, name, unit_str):
        """Check that all finite values fall within [lo, hi]. Return (n_out, fraction)."""
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0, 0.0
        n_out = int(np.sum((finite < lo) | (finite > hi)))
        frac = n_out / finite.size
        return n_out, frac

    def _histogram_line(diffs, n_bins=10):
        """Return a compact one-line ASCII histogram of diffs."""
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            return "(no finite diffs)"
        counts, edges = np.histogram(finite, bins=n_bins)
        mx = max(counts) if max(counts) > 0 else 1
        bar = ""
        for c in counts:
            filled = int(round(8 * c / mx))
            bar += ["_", ".", ":", "-", "=", "+", "#", "@", "@"][min(filled, 8)]
        lo_str = f"{edges[0]:.2e}"
        hi_str = f"{edges[-1]:.2e}"
        return f"[{lo_str}]{bar}[{hi_str}]"

    def deep_check(func_name, a_raw, b_raw, label_a, label_b,
                   phys_lo=None, phys_hi=None, phys_unit="",
                   rtol=1e-4, atol=0.01):
        """Comprehensive correctness check between two result arrays.

        Computes: mean diff, max abs diff, RMSE, 99th percentile abs diff,
        relative RMSE%, Pearson r, % points >1% and >0.1% relative error,
        NaN/Inf audit, physical plausibility, histogram of diffs.
        """
        nonlocal all_pass

        a = _to_np(a_raw).ravel()
        b = _to_np(b_raw).ravel()

        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

        pair = f"{label_a} vs {label_b}"

        # --- NaN/Inf audit ---
        nan_a, inf_a, tot_a = _nan_inf_audit(a, label_a)
        nan_b, inf_b, tot_b = _nan_inf_audit(b, label_b)

        print(f"\n  --- {func_name} [{pair}] ---")
        print(f"      NaN/Inf audit: {label_a}: {nan_a} NaN, {inf_a} Inf / {tot_a} pts"
              f"  |  {label_b}: {nan_b} NaN, {inf_b} Inf / {tot_b} pts")

        nan_inf_ok = (nan_a == 0 and inf_a == 0 and nan_b == 0 and inf_b == 0)
        if not nan_inf_ok:
            print(f"      FAIL  NaN/Inf detected")
            all_pass = False

        # --- Physical plausibility ---
        phys_ok = True
        if phys_lo is not None and phys_hi is not None:
            for arr, lbl in [(a, label_a), (b, label_b)]:
                n_out, frac = _physical_bounds_check(arr, phys_lo, phys_hi, func_name, phys_unit)
                if n_out > 0:
                    print(f"      FAIL  {lbl}: {n_out} pts ({frac*100:.4f}%) outside [{phys_lo}, {phys_hi}] {phys_unit}")
                    phys_ok = False
                    all_pass = False
            if phys_ok:
                print(f"      PASS  Physical bounds [{phys_lo}, {phys_hi}] {phys_unit}")

        # --- Mask to finite-in-both for numerical comparison ---
        valid = np.isfinite(a) & np.isfinite(b)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            print(f"      SKIP  No valid (finite-in-both) points for comparison")
            verify_rows.append((func_name, pair, "SKIP", *([np.nan]*8)))
            return

        av = a[valid]
        bv = b[valid]
        diff = av - bv
        abs_diff = np.abs(diff)

        # --- Core statistics ---
        mean_diff = float(np.mean(diff))
        max_abs = float(np.max(abs_diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        pct99 = float(np.percentile(abs_diff, 99))
        med_abs = float(np.median(abs_diff))

        # Relative RMSE (% of reference mean absolute value)
        ref_scale = float(np.mean(np.abs(bv)))
        rel_rmse_pct = (rmse / ref_scale * 100.0) if ref_scale > 1e-30 else 0.0

        # Pearson r
        if np.std(av) > 1e-30 and np.std(bv) > 1e-30:
            r_val, _ = pearsonr(av, bv)
        else:
            r_val = 1.0 if np.allclose(av, bv) else 0.0

        # Relative error per-point (against |b| + small epsilon to avoid div/0)
        denom = np.abs(bv) + 1e-30
        rel_err = abs_diff / denom
        pct_gt_1pct = float(np.sum(rel_err > 0.01) / n_valid * 100.0)
        pct_gt_01pct = float(np.sum(rel_err > 0.001) / n_valid * 100.0)

        # --- allclose verdict ---
        ac_ok = bool(np.allclose(av, bv, rtol=rtol, atol=atol))
        status = "PASS" if ac_ok else "FAIL"
        if not ac_ok:
            all_pass = False

        # --- Histogram ---
        hist_str = _histogram_line(diff)

        # --- Print all metrics ---
        print(f"      {status}  allclose(rtol={rtol:.0e}, atol={atol:.0e})  n_valid={n_valid}")
        print(f"      mean_diff   = {mean_diff:+.6e}")
        print(f"      max_abs_diff= {max_abs:.6e}")
        print(f"      median_abs  = {med_abs:.6e}")
        print(f"      RMSE        = {rmse:.6e}")
        print(f"      99th pct    = {pct99:.6e}")
        print(f"      rel RMSE    = {rel_rmse_pct:.6f}%")
        print(f"      Pearson r   = {r_val:.12f}")
        print(f"      >1% rel err = {pct_gt_1pct:.4f}%  |  >0.1% rel err = {pct_gt_01pct:.4f}%")
        print(f"      diff hist:    {hist_str}")

        verify_rows.append((func_name, pair, status,
                            mean_diff, max_abs, rmse, pct99,
                            rel_rmse_pct, r_val, pct_gt_1pct, pct_gt_01pct))

    # ------------------------------------------------------------------
    # Compute all backend results
    # ------------------------------------------------------------------

    # --- potential_temperature ---
    mrcalc.set_backend("cpu")
    pt_mp = mpcalc.potential_temperature(p850q, t850q)
    pt_cpu = mrcalc.potential_temperature(850.0, d["t850_C"])
    pt_mcu = None
    pt_gpu = None
    if HAS_GPU:
        pt_mcu = mcucalc.potential_temperature(850.0, d["t850_C"])
        mrcalc.set_backend("gpu")
        pt_gpu = mrcalc.potential_temperature(850.0, d["t850_C"])
        mrcalc.set_backend("cpu")

    # --- dewpoint ---
    dp_mp = mpcalc.dewpoint(vp850q)
    mrcalc.set_backend("cpu")
    dp_cpu = mrcalc.dewpoint(d["vp850_hPa"])
    dp_mcu = None
    dp_gpu = None
    if HAS_GPU:
        dp_mcu = mcucalc.dewpoint(d["vp850_hPa"])
        mrcalc.set_backend("gpu")
        dp_gpu = mrcalc.dewpoint(d["vp850_hPa"])
        mrcalc.set_backend("cpu")

    # --- saturation_vapor_pressure ---
    svp_mp = mpcalc.saturation_vapor_pressure(t850q)
    mrcalc.set_backend("cpu")
    svp_cpu = mrcalc.saturation_vapor_pressure(d["t850_C"])
    svp_mcu = None
    if HAS_GPU:
        svp_mcu = mcucalc.saturation_vapor_pressure(d["t850_C"])

    # --- relative_humidity_from_dewpoint ---
    rh_mp = mpcalc.relative_humidity_from_dewpoint(t850q, td850q)
    mrcalc.set_backend("cpu")
    rh_cpu = mrcalc.relative_humidity_from_dewpoint(d["t850_C"], d["td850_C"])
    rh_mcu = None
    if HAS_GPU:
        rh_mcu = mcucalc.relative_humidity_from_dewpoint(d["t850_C"], d["td850_C"])

    # --- mixing_ratio ---
    mr_mp = mpcalc.mixing_ratio(vp850q, p850q)
    mrcalc.set_backend("cpu")
    mr_cpu = mrcalc.mixing_ratio(d["vp850_hPa"], 850.0)
    mr_mcu = None
    if HAS_GPU:
        mr_mcu = mcucalc.mixing_ratio(d["vp850_hPa"], 850.0)

    # --- compute_pw (no MetPy grid equivalent) ---
    mrcalc.set_backend("cpu")
    pw_cpu = mrcalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"])
    pw_mcu = None
    pw_gpu = None
    if HAS_GPU:
        pw_mcu = mcucalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"])
        mrcalc.set_backend("gpu")
        pw_gpu = mrcalc.compute_pw(d["qvapor_kgkg"], d["p3_Pa"])
        mrcalc.set_backend("cpu")

    # ------------------------------------------------------------------
    # Deep checks: potential_temperature  (physical: 250-400 K)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [1/6] POTENTIAL TEMPERATURE")
    print("=" * W)

    deep_check("potential_temperature", pt_mp, pt_cpu, "MetPy", "MR-CPU",
               phys_lo=250, phys_hi=400, phys_unit="K",
               rtol=1e-4, atol=0.01)
    if HAS_GPU:
        deep_check("potential_temperature", pt_cpu, pt_mcu, "MR-CPU", "met-cu",
                   phys_lo=250, phys_hi=400, phys_unit="K",
                   rtol=1e-4, atol=0.01)
        deep_check("potential_temperature", pt_cpu, pt_gpu, "MR-CPU", "MR-GPU",
                   phys_lo=250, phys_hi=400, phys_unit="K",
                   rtol=1e-4, atol=0.01)

    # ------------------------------------------------------------------
    # Deep checks: dewpoint  (physical: -80 to 40 degC)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [2/6] DEWPOINT")
    print("=" * W)

    deep_check("dewpoint", dp_mp, dp_cpu, "MetPy", "MR-CPU",
               phys_lo=-80, phys_hi=40, phys_unit="degC",
               rtol=1e-4, atol=0.01)
    if HAS_GPU:
        deep_check("dewpoint", dp_cpu, dp_mcu, "MR-CPU", "met-cu",
                   phys_lo=-80, phys_hi=40, phys_unit="degC",
                   rtol=1e-4, atol=0.01)
        deep_check("dewpoint", dp_cpu, dp_gpu, "MR-CPU", "MR-GPU",
                   phys_lo=-80, phys_hi=40, phys_unit="degC",
                   rtol=1e-4, atol=0.01)

    # ------------------------------------------------------------------
    # Deep checks: saturation_vapor_pressure  (physical: 0-60 hPa = 0-6000 Pa)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [3/6] SATURATION VAPOR PRESSURE")
    print("=" * W)

    # Both MetPy and metrust return Pa
    deep_check("saturation_vapor_pressure", svp_mp, svp_cpu, "MetPy", "MR-CPU",
               phys_lo=0, phys_hi=6000, phys_unit="Pa (0-60 hPa)",
               rtol=1e-4, atol=1.0)
    if HAS_GPU:
        deep_check("saturation_vapor_pressure", svp_cpu, svp_mcu, "MR-CPU", "met-cu",
                   phys_lo=0, phys_hi=6000, phys_unit="Pa (0-60 hPa)",
                   rtol=1e-4, atol=1.0)

    # ------------------------------------------------------------------
    # Deep checks: relative_humidity_from_dewpoint  (physical: 0-1 fraction)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [4/6] RELATIVE HUMIDITY FROM DEWPOINT")
    print("=" * W)

    deep_check("rh_from_dewpoint", rh_mp, rh_cpu, "MetPy", "MR-CPU",
               phys_lo=0.0, phys_hi=1.0, phys_unit="fraction",
               rtol=1e-3, atol=0.01)
    if HAS_GPU:
        deep_check("rh_from_dewpoint", rh_cpu, rh_mcu, "MR-CPU", "met-cu",
                   phys_lo=0.0, phys_hi=1.0, phys_unit="fraction",
                   rtol=1e-3, atol=0.01)

    # ------------------------------------------------------------------
    # Deep checks: mixing_ratio  (physical: 0-0.04 kg/kg)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [5/6] MIXING RATIO")
    print("=" * W)

    deep_check("mixing_ratio", mr_mp, mr_cpu, "MetPy", "MR-CPU",
               phys_lo=0.0, phys_hi=0.04, phys_unit="kg/kg",
               rtol=1e-3, atol=1e-4)
    if HAS_GPU:
        deep_check("mixing_ratio", mr_cpu, mr_mcu, "MR-CPU", "met-cu",
                   phys_lo=0.0, phys_hi=0.04, phys_unit="kg/kg",
                   rtol=1e-3, atol=1e-4)

    # ------------------------------------------------------------------
    # Deep checks: compute_pw  (physical: 0-80 mm; no MetPy grid version)
    # ------------------------------------------------------------------
    print()
    print("=" * W)
    print("  [6/6] COMPUTE_PW (3D grid)")
    print("=" * W)

    # Always check CPU result physically
    pw_cpu_np = _to_np(pw_cpu)
    n_out_pw, frac_out_pw = _physical_bounds_check(pw_cpu_np.ravel(), 0, 80, "compute_pw", "mm")
    nan_pw, inf_pw, tot_pw = _nan_inf_audit(pw_cpu_np.ravel(), "MR-CPU")
    print(f"\n  --- compute_pw MR-CPU physical audit ---")
    print(f"      NaN/Inf: {nan_pw} NaN, {inf_pw} Inf / {tot_pw} pts")
    print(f"      Range: [{float(np.nanmin(pw_cpu_np)):.4f}, {float(np.nanmax(pw_cpu_np)):.4f}] mm")
    print(f"      Mean: {float(np.nanmean(pw_cpu_np)):.4f} mm")
    if n_out_pw > 0:
        print(f"      FAIL  {n_out_pw} pts ({frac_out_pw*100:.4f}%) outside [0, 80] mm")
        all_pass = False
    else:
        print(f"      PASS  All values within [0, 80] mm")
    if nan_pw > 0 or inf_pw > 0:
        print(f"      FAIL  NaN/Inf detected in PW")
        all_pass = False

    if HAS_GPU:
        deep_check("compute_pw", pw_cpu, pw_mcu, "MR-CPU", "met-cu",
                   phys_lo=0, phys_hi=80, phys_unit="mm",
                   rtol=1e-4, atol=0.1)
        deep_check("compute_pw", pw_cpu, pw_gpu, "MR-CPU", "MR-GPU",
                   phys_lo=0, phys_hi=80, phys_unit="mm",
                   rtol=1e-4, atol=0.1)

    # ------------------------------------------------------------------
    # Edge case: driest / wettest columns for PW
    # ------------------------------------------------------------------
    print()
    print("  --- compute_pw edge-case: driest & wettest columns ---")

    pw_2d = _to_np(pw_cpu).reshape(d["ny"], d["nx"])
    col_mean = np.nanmean(pw_2d, axis=0)  # mean PW per longitude column
    driest_col = int(np.argmin(col_mean))
    wettest_col = int(np.argmax(col_mean))
    print(f"      Driest column  ix={driest_col}: mean PW={col_mean[driest_col]:.4f} mm"
          f"  range=[{pw_2d[:, driest_col].min():.4f}, {pw_2d[:, driest_col].max():.4f}]")
    print(f"      Wettest column ix={wettest_col}: mean PW={col_mean[wettest_col]:.4f} mm"
          f"  range=[{pw_2d[:, wettest_col].min():.4f}, {pw_2d[:, wettest_col].max():.4f}]")

    if HAS_GPU:
        pw_mcu_2d = _to_np(pw_mcu).reshape(d["ny"], d["nx"])
        pw_gpu_2d = _to_np(pw_gpu).reshape(d["ny"], d["nx"])

        # Driest column cross-check
        dry_cpu = pw_2d[:, driest_col]
        dry_mcu = pw_mcu_2d[:, driest_col]
        dry_gpu = pw_gpu_2d[:, driest_col]
        dry_maxdiff_mcu = float(np.max(np.abs(dry_cpu - dry_mcu)))
        dry_maxdiff_gpu = float(np.max(np.abs(dry_cpu - dry_gpu)))
        print(f"      Driest col  max|CPU-mcu|={dry_maxdiff_mcu:.6e}  max|CPU-GPU|={dry_maxdiff_gpu:.6e}")

        # Wettest column cross-check
        wet_cpu = pw_2d[:, wettest_col]
        wet_mcu = pw_mcu_2d[:, wettest_col]
        wet_gpu = pw_gpu_2d[:, wettest_col]
        wet_maxdiff_mcu = float(np.max(np.abs(wet_cpu - wet_mcu)))
        wet_maxdiff_gpu = float(np.max(np.abs(wet_cpu - wet_gpu)))
        print(f"      Wettest col max|CPU-mcu|={wet_maxdiff_mcu:.6e}  max|CPU-GPU|={wet_maxdiff_gpu:.6e}")

        edge_tol = 0.1  # mm
        edge_ok = (dry_maxdiff_mcu < edge_tol and dry_maxdiff_gpu < edge_tol
                   and wet_maxdiff_mcu < edge_tol and wet_maxdiff_gpu < edge_tol)
        if edge_ok:
            print(f"      PASS  Edge columns agree within {edge_tol} mm")
        else:
            print(f"      FAIL  Edge column disagreement exceeds {edge_tol} mm")
            all_pass = False
    else:
        print("      (GPU not available -- skipping cross-backend edge check)")

    # ==================================================================
    # VERIFICATION SUMMARY TABLE
    # ==================================================================
    print()
    print("=" * W)
    print("  VERIFICATION SUMMARY TABLE")
    print("=" * W)
    hdr_fmt = ("  {:<28s} {:<18s} {:>6s} {:>12s} {:>12s} {:>12s} {:>12s}"
               " {:>10s} {:>12s} {:>8s} {:>9s}")
    print(hdr_fmt.format(
        "Function", "Comparison", "P/F", "mean_diff", "max_abs", "RMSE",
        "99th pct", "relRMSE%", "Pearson r", ">1%rel", ">0.1%rel"))
    print("  " + "-" * (W - 4))
    for (fn, pair, status, md, ma, rmse_v, p99, rr, pr, g1, g01) in verify_rows:
        if status == "SKIP":
            print(f"  {fn:<28s} {pair:<18s} {'SKIP':>6s}")
            continue
        print(f"  {fn:<28s} {pair:<18s} {status:>6s}"
              f" {md:>+12.4e} {ma:>12.4e} {rmse_v:>12.4e} {p99:>12.4e}"
              f" {rr:>10.6f} {pr:>12.10f} {g1:>8.4f} {g01:>9.4f}")

    n_pass = sum(1 for r in verify_rows if r[2] == "PASS")
    n_fail = sum(1 for r in verify_rows if r[2] == "FAIL")
    n_skip = sum(1 for r in verify_rows if r[2] == "SKIP")
    print()
    print(f"  Checks: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP"
          f"  (total {len(verify_rows)})")

    # ==================================================================
    # TIMING SUMMARY TABLE
    # ==================================================================
    print()
    print("=" * W)
    print("  TIMING SUMMARY")
    print("=" * W)
    print(f"  {'Function':32s}"
          f" {'MetPy':>10s} {'MR-CPU':>10s} {'met-cu':>10s} {'MR-GPU':>10s}"
          f" {'MP/CPU':>8s} {'CPU/GPU':>8s}")
    print("  " + "-" * (W - 4))
    for name, t_mp, t_cpu, t_mcu, t_gpu, note in results:
        star = "*" if "GPU" in note else " "
        print(f" {star}{name:32s}"
              f" {fmt(t_mp):>10s} {fmt(t_cpu):>10s}"
              f" {fmt(t_mcu):>10s} {fmt(t_gpu):>10s}"
              f" {spd(t_mp, t_cpu):>8s} {spd(t_cpu, t_gpu):>8s}")

    # Totals
    total_mp = sum(t for _, t, _, _, _, _ in results if t is not None)
    total_cpu = sum(t for _, _, t, _, _, _ in results if t is not None)
    total_mcu = sum(t for _, _, _, t, _, _ in results if t is not None)
    total_gpu = sum(t for _, _, _, _, t, _ in results if t is not None)
    print("  " + "-" * (W - 4))
    print(f"  {'TOTAL':32s}"
          f" {fmt(total_mp) if total_mp else '--':>10s}"
          f" {fmt(total_cpu):>10s}"
          f" {fmt(total_mcu) if total_mcu else '--':>10s}"
          f" {fmt(total_gpu) if total_gpu else '--':>10s}"
          f" {spd(total_mp, total_cpu) if total_mp else '--':>8s}"
          f" {spd(total_cpu, total_gpu) if total_gpu else '--':>8s}")

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================
    print()
    print("=" * W)
    if all_pass:
        print("  FINAL VERDICT: ALL CHECKS PASSED")
    else:
        print("  FINAL VERDICT: SOME CHECKS FAILED")
    print("=" * W)


if __name__ == "__main__":
    main()
