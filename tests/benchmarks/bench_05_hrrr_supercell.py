#!/usr/bin/env python
"""Bench-05: HRRR Supercell Environment — with deep data-correctness verification.

Scenario
--------
HRRR-like 3 km grid, 600x600 subdomain, 30 vertical levels.
Classic supercell environment: rich BL moisture (Td 18-22 C), strong 0-6 km
shear (>40 kt), backed surface winds, strong capping with EML, realistic
hydrometeor profiles (rain in warm cloud, snow above freezing level, graupel
in updraft).

Functions benchmarked
---------------------
- compute_cape_cin        (GPU-eligible)
- compute_srh             (GPU-eligible)
- compute_shear           (GPU-eligible)
- composite_reflectivity_from_hydrometeors  (GPU-eligible)
- equivalent_potential_temperature          (GPU-eligible)
- dewpoint                                  (GPU-eligible)

Backends: MetPy, metrust CPU, met-cu direct (CUDA), metrust GPU.
MetPy lacks the grid composites, so only thermo functions compare all four.

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
# Grid parameters
# ============================================================================
NX, NY, NZ = 600, 600, 30
SEED = 42
np.random.seed(SEED)

# ============================================================================
# Build realistic supercell environment
# ============================================================================

def build_supercell_environment():
    """Construct a physically consistent HRRR-like supercell subdomain."""

    rng = np.random.default_rng(SEED)

    # -- Pressure levels (surface ~1013 hPa to ~100 hPa, 30 levels) ----------
    p_levels_hPa = np.linspace(1013.0, 100.0, NZ)   # hPa, surface first
    p_3d_hPa = np.broadcast_to(
        p_levels_hPa[:, None, None], (NZ, NY, NX)
    ).copy()
    p_3d_Pa = p_3d_hPa * 100.0

    # -- Temperature profile (supercell: warm BL, strong cap, cold aloft) -----
    z_approx_km = 7.0 * np.log(1013.0 / p_levels_hPa)

    t_profile_C = np.zeros(NZ)
    for k in range(NZ):
        z = z_approx_km[k]
        if z < 1.0:       # BL: warm, moist (32 C sfc, ~8 C/km lapse)
            t_profile_C[k] = 32.0 - z * 8.0
        elif z < 2.0:     # Cap: thin warm nose (only 3 C/km lapse)
            t_profile_C[k] = t_profile_C[k-1] + (z_approx_km[k-1] - z) * 3.0
        elif z < 8.0:     # Free tropo: steep EML ~8.5 C/km
            t_profile_C[k] = t_profile_C[k-1] + (z_approx_km[k-1] - z) * 8.5
        elif z < 12.0:    # Upper tropo: ~6.5 C/km
            t_profile_C[k] = t_profile_C[k-1] + (z_approx_km[k-1] - z) * 6.5
        else:             # Stratosphere: isothermal to slight warming
            t_profile_C[k] = t_profile_C[k-1] + (z_approx_km[k-1] - z) * (-1.0)

    tc_3d = np.broadcast_to(
        t_profile_C[:, None, None], (NZ, NY, NX)
    ).copy()
    tc_3d += rng.normal(0, 0.3, (NZ, NY, NX))

    # -- Dewpoint (rich BL moisture, dry EML, moderate mid-level) -------------
    td_profile_C = np.zeros(NZ)
    for k in range(NZ):
        z = z_approx_km[k]
        if z < 1.0:       # BL: rich moisture, Td 21-22 C
            td_profile_C[k] = 22.0 - z * 2.0
        elif z < 2.0:     # Cap: dry (Td depression 20 C)
            td_profile_C[k] = t_profile_C[k] - 20.0
        elif z < 8.0:     # Mid-tropo: moderate moisture
            td_profile_C[k] = t_profile_C[k] - 14.0
        elif z < 12.0:    # Upper tropo: dry
            td_profile_C[k] = t_profile_C[k] - 25.0
        else:
            td_profile_C[k] = t_profile_C[k] - 35.0

    td_3d = np.broadcast_to(
        td_profile_C[:, None, None], (NZ, NY, NX)
    ).copy()
    td_3d += rng.normal(0, 0.3, (NZ, NY, NX))
    # Ensure Td <= T everywhere
    td_3d = np.minimum(td_3d, tc_3d - 0.1)

    # -- Vapor pressure for dewpoint() benchmark (Tetens from dewpoint) -------
    vp_hPa = 6.1078 * np.exp(17.27 * td_3d / (td_3d + 237.3))

    # -- Mixing ratio from dewpoint (Bolton) ----------------------------------
    es_td = 6.112 * np.exp(17.67 * td_3d / (td_3d + 243.5))
    w_mr = 0.622 * es_td / (p_3d_hPa - es_td)         # kg/kg
    w_mr = np.clip(w_mr, 0.0, 0.040)

    # -- Height AGL (hypsometric approximation) -------------------------------
    h_agl = np.zeros((NZ, NY, NX))
    for k in range(1, NZ):
        t_mean_K = 0.5 * (tc_3d[k-1] + tc_3d[k]) + 273.15
        h_agl[k] = h_agl[k-1] + (287.05 * t_mean_K / 9.81) * np.log(
            p_3d_Pa[k-1] / p_3d_Pa[k]
        )

    # -- Wind profile (backed sfc, strong shear >40 kt 0-6 km) ---------------
    u_profile = np.zeros(NZ)
    v_profile = np.zeros(NZ)
    for k in range(NZ):
        h = float(np.mean(h_agl[k]))
        if h < 1000:
            frac = h / 1000.0
            u_profile[k] = -5.0 + frac * 10.0
            v_profile[k] = 10.0 + frac * 5.0
        elif h < 6000:
            frac = (h - 1000.0) / 5000.0
            u_profile[k] = 5.0 + frac * 25.0
            v_profile[k] = 15.0 - frac * 15.0
        else:
            u_profile[k] = 30.0 + (h - 6000.0) * 0.001
            v_profile[k] = 0.0

    u_3d = np.broadcast_to(
        u_profile[:, None, None], (NZ, NY, NX)
    ).copy()
    v_3d = np.broadcast_to(
        v_profile[:, None, None], (NZ, NY, NX)
    ).copy()
    u_3d += rng.normal(0, 1.0, (NZ, NY, NX))
    v_3d += rng.normal(0, 1.0, (NZ, NY, NX))

    # -- Surface fields -------------------------------------------------------
    psfc = p_3d_Pa[0, :, :].copy()
    t2 = tc_3d[0, :, :] + 273.15
    q2 = w_mr[0, :, :].copy()

    # -- Hydrometeor profiles -------------------------------------------------
    qrain = np.zeros((NZ, NY, NX))
    qsnow = np.zeros((NZ, NY, NX))
    qgraup = np.zeros((NZ, NY, NX))

    yy, xx = np.ogrid[0:NY, 0:NX]
    dist_sq = ((yy - NY / 2) ** 2 + (xx - NX / 2) ** 2).astype(np.float64)
    updraft_envelope = np.exp(-dist_sq / (2 * 50.0 ** 2))

    for k in range(NZ):
        t_level = tc_3d[k]

        warm_mask = t_level > 0
        rain_intensity = np.clip(t_level / 30.0 * 3.0, 0, 3.0)
        qrain[k] = np.where(warm_mask, rain_intensity * 1e-3 * updraft_envelope, 0)

        snow_mask = (t_level < 0) & (t_level > -40)
        snow_intensity = np.clip(-t_level / 20.0 * 2.0, 0, 2.0)
        qsnow[k] = np.where(snow_mask, snow_intensity * 1e-3 * updraft_envelope, 0)

        graup_mask = (t_level < -5) & (t_level > -30)
        graup_intensity = np.clip(
            (1.0 - np.abs(t_level + 15) / 15.0) * 5.0, 0, 5.0)
        qgraup[k] = np.where(
            graup_mask, graup_intensity * 1e-3 * updraft_envelope * 1.5, 0)

    for q_arr in [qrain, qsnow, qgraup]:
        noise = rng.uniform(0, 1e-5, q_arr.shape)
        q_arr[:] = np.clip(q_arr + noise * (q_arr > 0), 0, 0.005)

    return dict(
        p_3d_hPa=np.ascontiguousarray(p_3d_hPa, dtype=np.float64),
        p_3d_Pa=np.ascontiguousarray(p_3d_Pa, dtype=np.float64),
        tc_3d=np.ascontiguousarray(tc_3d, dtype=np.float64),
        td_3d=np.ascontiguousarray(td_3d, dtype=np.float64),
        vp_hPa=np.ascontiguousarray(vp_hPa, dtype=np.float64),
        w_mr=np.ascontiguousarray(w_mr, dtype=np.float64),
        h_agl=np.ascontiguousarray(h_agl, dtype=np.float64),
        u_3d=np.ascontiguousarray(u_3d, dtype=np.float64),
        v_3d=np.ascontiguousarray(v_3d, dtype=np.float64),
        psfc=np.ascontiguousarray(psfc, dtype=np.float64),
        t2=np.ascontiguousarray(t2, dtype=np.float64),
        q2=np.ascontiguousarray(q2, dtype=np.float64),
        qrain=np.ascontiguousarray(qrain, dtype=np.float64),
        qsnow=np.ascontiguousarray(qsnow, dtype=np.float64),
        qgraup=np.ascontiguousarray(qgraup, dtype=np.float64),
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
    "cin":          (-600.0, 0.0),        # J/kg  (strong caps can push below -500)
    "lcl":          (0.0, 6000.0),        # m AGL
    "lfc":          (0.0, 15000.0),       # m AGL
    "srh":          (-500.0, 1000.0),     # m^2/s^2
    "shear":        (0.0, 80.0),          # m/s
    "refl":         (-20.0, 80.0),        # dBZ
    "theta_e":      (290.0, 380.0),       # K
    "dewpoint":     (-90.0, 35.0),        # degC
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
    header_line = f"      {'Backend':<12s} {'field':<6s} {'mean':>10s} {'median':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    print(header_line)
    print("      " + "-" * len(header_line.strip()))

    for bk in backends:
        for fi, fn in enumerate(["cape", "cin"]):
            arr = np.asarray(_strip_units(cape_dict[bk][fi]), dtype=np.float64).ravel()
            fin = arr[np.isfinite(arr)]
            if fin.size == 0:
                print(f"      {bk:<12s} {fn:<6s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s}")
                continue
            print(f"      {bk:<12s} {fn:<6s} "
                  f"{np.mean(fin):10.2f} {np.median(fin):10.2f} "
                  f"{np.std(fin):10.2f} {np.min(fin):10.2f} {np.max(fin):10.2f}")

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
            # Sort by magnitude of difference
            diff_at_disagree = np.abs(cape_ref[disagree_idx] - cape_other[disagree_idx])
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
    print("  BENCH-05: HRRR Supercell Environment  (DEEP VERIFICATION)")
    print(f"  Grid: {NZ} levels x {NY}x{NX} = {NZ*NY*NX:,} points")
    print(f"  GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print(f"  MetPy: {'available' if HAS_METPY else 'not available'}")
    print("=" * 110)

    print("\n  Building synthetic supercell environment ... ", end="", flush=True)
    t0 = time.perf_counter()
    d = build_supercell_environment()
    print(f"{time.perf_counter() - t0:.1f} s")

    # Sanity-check the data
    print(f"  T surface range: {d['tc_3d'][0].min():.1f} to "
          f"{d['tc_3d'][0].max():.1f} C")
    print(f"  Td surface range: {d['td_3d'][0].min():.1f} to "
          f"{d['td_3d'][0].max():.1f} C")
    shear_approx = np.sqrt(
        (float(np.mean(d['u_3d'][-1])) - float(np.mean(d['u_3d'][0]))) ** 2 +
        (float(np.mean(d['v_3d'][-1])) - float(np.mean(d['v_3d'][0]))) ** 2
    )
    print(f"  Approx 0-top shear: {shear_approx:.1f} m/s "
          f"({shear_approx * 1.944:.0f} kt)")
    print(f"  qrain max: {d['qrain'].max()*1000:.2f} g/kg")
    print(f"  qsnow max: {d['qsnow'].max()*1000:.2f} g/kg")
    print(f"  qgraup max: {d['qgraup'].max()*1000:.2f} g/kg")

    N = 3  # timed iterations

    # ==================================================================
    # Prepare Pint quantities for MetPy (not timed)
    # ==================================================================
    if HAS_METPY:
        p_levels = d['p_3d_hPa'][:, 0, 0]
        i850 = int(np.argmin(np.abs(p_levels - 850.0)))
        tc_850 = d['tc_3d'][i850]
        td_850 = d['td_3d'][i850]
        vp_850 = d['vp_hPa'][i850]

        mp_p850 = 850.0 * mp_units.hPa
        mp_tc850 = tc_850 * mp_units.degC
        mp_td850 = td_850 * mp_units.degC
        mp_vp850 = vp_850 * mp_units.hPa

    # ==================================================================
    # SECTION 1: THERMODYNAMICS (2D slice at 850 hPa)
    # ==================================================================
    print("\n" + "=" * 110)
    print("  SECTION 1: THERMODYNAMICS (2D slice at 850 hPa)")
    header()

    # --- equivalent_potential_temperature ---
    i850_idx = int(np.argmin(np.abs(d['p_3d_hPa'][:, 0, 0] - 850.0)))
    tc_slice = d['tc_3d'][i850_idx]
    td_slice = d['td_3d'][i850_idx]
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
    vp_slice = d['vp_hPa'][i850_idx]

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
    # SECTION 2: GRID COMPOSITES (3 backends: CPU, met-cu, GPU)
    # ==================================================================
    print("\n" + "=" * 110)
    print(f"  SECTION 2: GRID COMPOSITES ({NZ}x{NY}x{NX} -> {NY}x{NX})")
    header()

    # --- compute_cape_cin ---
    t_cpu = _run_cpu(
        lambda: mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2'], d['q2']), N)

    t_mcu = _run_mcu(
        lambda: mcucalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2'], d['q2']), N) if HAS_GPU else None

    t_gpu = _run_gpu(
        lambda: mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2'], d['q2']), N)

    record("compute_cape_cin", None, t_cpu, t_mcu, t_gpu)

    # -- Deep verification: CAPE/CIN --
    print("\n    [Deep Verification: CAPE/CIN]")
    mrcalc.set_backend("cpu")
    cape_cpu = mrcalc.compute_cape_cin(
        d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
        d['psfc'], d['t2'], d['q2'])
    cape_all = {"CPU": cape_cpu}

    if HAS_GPU:
        mrcalc.set_backend("gpu")
        cape_gpu = mrcalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2'], d['q2'])
        mrcalc.set_backend("cpu")
        cape_all["GPU"] = cape_gpu
        deep_verify_tuple("cape_cin", cape_cpu, cape_gpu,
                          ["cape", "cin", "lcl", "lfc"],
                          ["cape", "cin", "lcl", "lfc"],
                          "CPU", "GPU", rtol_pass=1e-3)

        cape_mcu = mcucalc.compute_cape_cin(
            d['p_3d_Pa'], d['tc_3d'], d['w_mr'], d['h_agl'],
            d['psfc'], d['t2'], d['q2'])
        cape_all["met-cu"] = cape_mcu
        deep_verify_tuple("cape_cin", cape_cpu, cape_mcu,
                          ["cape", "cin", "lcl", "lfc"],
                          ["cape", "cin", "lcl", "lfc"],
                          "CPU", "met-cu", rtol_pass=1e-3)

    # CAPE/CIN distribution analysis
    cape_distribution_analysis(cape_all)

    # Edge case: max CAPE column, zero-CAPE column
    cape_flat_cpu = np.asarray(_strip_units(cape_cpu[0]), dtype=np.float64).ravel()
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
    n_total = len(_VERIFY_LEDGER)

    print(f"\n  {'Check':<40s} {'Pair':<25s} {'Result':<8s}")
    print("  " + "-" * 73)
    for check_name, pair_label, passed in _VERIFY_LEDGER:
        tag = "PASS" if passed else "FAIL"
        print(f"  {check_name:<40s} {pair_label:<25s} {tag:<8s}")

    print("\n  " + "-" * 73)
    print(f"  TOTAL: {n_pass} PASS, {n_fail} FAIL out of {n_total} checks")

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
        d['psfc'], d['t2'], d['q2'])
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
        ("CAPE",      cape_arr,    "cape",     "J/kg"),
        ("CIN",       cin_arr,     "cin",      "J/kg"),
        ("SRH",       srh_arr,     "srh",      "m^2/s^2"),
        ("0-6km shear", shear_arr, "shear",    "m/s"),
        ("Comp refl", refl_arr,    "refl",     "dBZ"),
        ("theta-e",   theta_e_arr, "theta_e",  "K"),
        ("dewpoint",  dp_arr,      "dewpoint", "degC"),
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
