"""Benchmark 07 -- HRRR Fire Weather Assessment (3 km, 800x800, surface level)

Scenario
--------
Red-flag-warning conditions over the western US.  Hot, dry, windy: temperature
30-45 degC, relative humidity 5-25 %, sustained winds 5-20 m/s with gusts.
Smooth spatial gradients, not random noise.

Functions benchmarked
---------------------
  CPU-only : heat_index, dewpoint_from_relative_humidity,
             relative_humidity_from_dewpoint, saturation_vapor_pressure,
             wind_speed
  GPU-capable : potential_temperature, dewpoint (from vapor pressure)

Backends
--------
  1. MetPy  (Pint Quantity)
  2. metrust CPU  (raw numpy, Rust engine)
  3. met-cu direct  (raw numpy -> CuPy CUDA kernels)
  4. metrust GPU  (metrust with backend="gpu", wraps met-cu)

Verification
------------
  Deep data-correctness audit for every function x backend:
    - Mean diff, max abs diff, RMSE, 99th percentile diff
    - Relative RMSE (%)
    - NaN / Inf audit
    - Physical plausibility bounds check
    - Pearson correlation coefficient
    - Percentage of points exceeding 1% and 0.1% relative error
    - Histogram of absolute differences (10 bins)
  Special heat_index extreme-condition stress test (T>40 C, RH<10%).
  Summary table with PASS/FAIL per backend per function.

Timing: time.perf_counter, 1 warmup + 5 timed runs, report median.
"""

import sys
import time
import textwrap

import numpy as np


# ---- Synthetic HRRR fire-weather grid ----------------------------------------

NY, NX = 800, 800
N = NY * NX

# Smooth temperature gradient: 30 C in NW corner to 45 C in SE corner
_row = np.linspace(0.0, 1.0, NY).reshape(NY, 1)
_col = np.linspace(0.0, 1.0, NX).reshape(1, NX)
temperature_c = np.ascontiguousarray(
    30.0 + 15.0 * (0.5 * _row + 0.5 * _col), dtype=np.float64)  # 30-45 C

# Smooth RH gradient: 25 % in the hot SE to 5 % in dry NW (inverted somewhat)
# Use a slightly different gradient axis for realism
relative_humidity_pct = np.ascontiguousarray(
    25.0 - 20.0 * (0.4 * _row + 0.6 * _col), dtype=np.float64)  # 5-25 %

# Surface pressure: slight gradient from 1013 hPa (low elevation) to 850 hPa
# (high terrain in the west)
pressure_hpa = np.broadcast_to(
    1013.0 - 163.0 * _col, (NY, NX)).copy()  # west side is higher terrain

# Wind components (m/s): southwest flow, stronger in the south
# Broadcast to full grid so every backend sees (800, 800)
u_ms = np.broadcast_to(5.0 + 10.0 * _row, (NY, NX)).copy()   # 5-15 m/s
v_ms = np.broadcast_to(3.0 + 7.0 * _col, (NY, NX)).copy()    # 3-10 m/s

# Vapor pressure for dewpoint benchmark: compute from SVP and RH
# e = es(T) * RH/100;  es from Ambaum (2020) approximation in hPa
_es_hpa = 6.1078 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))
vapor_pressure_hpa = _es_hpa * relative_humidity_pct / 100.0

# Precompute dewpoint array (needed for rh_from_dewpoint benchmark)
# Use Magnus formula: Td = 237.3 * ln(e/6.1078) / (17.27 - ln(e/6.1078))
_x = np.log(vapor_pressure_hpa / 6.1078)
dewpoint_c = 237.3 * _x / (17.27 - _x)


# ---- Imports -----------------------------------------------------------------

import metpy.calc as mpcalc
from metpy.units import units as mpunits

import metrust.calc as mcalc
from metrust.calc import use_backend
from metrust.units import units as mrunits
from metrust._metrust import calc as _rcalc   # raw Rust bindings

import cupy as cp
from metcu import calc as gpcalc              # met-cu direct GPU


# ---- Helpers -----------------------------------------------------------------

WARMUP = 1
REPEATS = 5


def _sync_gpu():
    """Synchronize CUDA stream so timing is accurate."""
    cp.cuda.Stream.null.synchronize()


def _to_numpy(x):
    """Convert anything to a plain numpy array."""
    if hasattr(x, "magnitude"):
        x = x.magnitude
    if hasattr(x, "get"):          # cupy
        x = cp.asnumpy(x)
    return np.asarray(x, dtype=np.float64)


def bench(label, fn, *, gpu=False):
    """Time *fn* (1 warmup + REPEATS), return (median_ms, result_as_numpy)."""
    # warmup
    for _ in range(WARMUP):
        r = fn()
        if gpu:
            _sync_gpu()

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = fn()
        if gpu:
            _sync_gpu()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    return np.median(times), _to_numpy(r)


def check(name, ref, got, rtol=1e-4):
    """Verify result matches reference within tolerance."""
    ref_f = ref.ravel()
    got_f = got.ravel()
    if ref_f.shape != got_f.shape:
        return False, f"shape mismatch {ref_f.shape} vs {got_f.shape}"
    ok = np.allclose(ref_f, got_f, rtol=rtol, atol=1e-8, equal_nan=True)
    if not ok:
        diff = np.abs(ref_f - got_f)
        maxdiff = np.nanmax(diff)
        idx = np.nanargmax(diff)
        return False, f"max diff {maxdiff:.6e} at [{idx}] ref={ref_f[idx]:.6f} got={got_f[idx]:.6f}"
    return True, "OK"


# ==============================================================================
# Deep verification engine
# ==============================================================================

# Physical plausibility bounds: (low, high) in the *output* unit of each func.
# These are generous fire-weather limits (not climatological, but physically
# sane).  A value outside these bounds is a clear computation error.
PHYS_BOUNDS = {
    "heat_index":                       (25.0, 75.0),     # degC
    "dewpoint_from_relative_humidity":  (-30.0, 30.0),    # degC
    "relative_humidity_from_dewpoint":  (0.0, 1.05),      # fractional 0-1
    "saturation_vapor_pressure":        (0.0, 15000.0),   # Pa (0-150 hPa)
    "wind_speed":                       (0.0, 50.0),      # m/s
    "potential_temperature":            (280.0, 360.0),    # K
    "dewpoint":                         (-30.0, 35.0),     # degC
}


def deep_verify(func_name, ref, got, backend_label):
    """Run the full verification suite comparing *got* to *ref* (MetPy).

    Returns a dict of metrics and a boolean pass/fail.
    """
    ref_f = ref.ravel().astype(np.float64)
    got_f = got.ravel().astype(np.float64)

    report = {"backend": backend_label, "function": func_name}

    # --- Shape check ---
    if ref_f.shape != got_f.shape:
        report["error"] = f"shape mismatch {ref_f.shape} vs {got_f.shape}"
        return report, False

    n = ref_f.size

    # --- NaN / Inf audit ---
    ref_nan = int(np.isnan(ref_f).sum())
    got_nan = int(np.isnan(got_f).sum())
    ref_inf = int(np.isinf(ref_f).sum())
    got_inf = int(np.isinf(got_f).sum())
    report["ref_nan"] = ref_nan
    report["got_nan"] = got_nan
    report["ref_inf"] = ref_inf
    report["got_inf"] = got_inf

    # Work on finite mask for remaining metrics
    finite_mask = np.isfinite(ref_f) & np.isfinite(got_f)
    n_finite = int(finite_mask.sum())
    report["n_finite"] = n_finite

    if n_finite == 0:
        report["error"] = "no finite points to compare"
        return report, False

    r = ref_f[finite_mask]
    g = got_f[finite_mask]
    diff = g - r
    absdiff = np.abs(diff)

    # --- Core accuracy metrics ---
    report["mean_diff"] = float(np.mean(diff))
    report["max_abs_diff"] = float(np.max(absdiff))
    report["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    report["pct99_abs_diff"] = float(np.percentile(absdiff, 99))

    # --- Relative RMSE (%) ---
    ref_range = float(np.max(r) - np.min(r))
    ref_mean_abs = float(np.mean(np.abs(r)))
    denom = max(ref_mean_abs, 1e-15)
    report["rel_rmse_pct"] = float(report["rmse"] / denom * 100.0)

    # --- Pearson correlation ---
    if np.std(r) > 0 and np.std(g) > 0:
        report["pearson_r"] = float(np.corrcoef(r, g)[0, 1])
    else:
        report["pearson_r"] = float("nan")

    # --- Relative error exceedances ---
    # Avoid division by zero: only consider points where |ref| > threshold
    rel_thresh = 1e-10
    rel_mask = np.abs(r) > rel_thresh
    if rel_mask.sum() > 0:
        rel_err = absdiff[rel_mask] / np.abs(r[rel_mask])
        report["pct_gt_1pct_rel"] = float(
            (rel_err > 0.01).sum() / rel_mask.sum() * 100.0)
        report["pct_gt_0p1pct_rel"] = float(
            (rel_err > 0.001).sum() / rel_mask.sum() * 100.0)
    else:
        report["pct_gt_1pct_rel"] = 0.0
        report["pct_gt_0p1pct_rel"] = 0.0

    # --- Histogram of absolute differences (10 bins, log-spaced if needed) ---
    if report["max_abs_diff"] > 0:
        lo = max(absdiff[absdiff > 0].min(), 1e-18) if (absdiff > 0).any() else 1e-18
        hi = report["max_abs_diff"]
        if hi / max(lo, 1e-30) > 100:
            edges = np.geomspace(lo, hi, 11)
        else:
            edges = np.linspace(0, hi, 11)
        counts, _ = np.histogram(absdiff, bins=edges)
        report["hist_counts"] = counts.tolist()
        report["hist_edges"] = [float(e) for e in edges]
    else:
        report["hist_counts"] = [n_finite]
        report["hist_edges"] = [0.0, 0.0]

    # --- Physical plausibility check on got ---
    bounds = PHYS_BOUNDS.get(func_name)
    if bounds is not None:
        lo_b, hi_b = bounds
        got_all = got_f[np.isfinite(got_f)]
        below = int((got_all < lo_b).sum())
        above = int((got_all > hi_b).sum())
        report["phys_below"] = below
        report["phys_above"] = above
        report["phys_bounds"] = bounds
    else:
        report["phys_below"] = 0
        report["phys_above"] = 0

    # --- PASS / FAIL decision ---
    passed = True
    failures = []

    # Accuracy gate: RMSE < 0.01% of mean absolute value
    if report["rel_rmse_pct"] > 0.01:
        passed = False
        failures.append(f"rel_rmse={report['rel_rmse_pct']:.4f}%>0.01%")

    # No spurious NaN/Inf
    extra_nan = got_nan - ref_nan
    if extra_nan > 0:
        passed = False
        failures.append(f"+{extra_nan} NaNs vs ref")
    if got_inf > ref_inf:
        passed = False
        failures.append(f"+{got_inf - ref_inf} Infs vs ref")

    # Pearson r must be essentially 1
    if report["pearson_r"] < 0.999999:
        passed = False
        failures.append(f"pearson_r={report['pearson_r']:.8f}")

    # Physical bounds
    if report["phys_below"] > 0 or report["phys_above"] > 0:
        passed = False
        failures.append(
            f"phys_oob: {report['phys_below']} below, {report['phys_above']} above")

    # No points > 1% relative error
    if report["pct_gt_1pct_rel"] > 0:
        passed = False
        failures.append(f"{report['pct_gt_1pct_rel']:.4f}% pts >1% rel err")

    report["passed"] = passed
    report["failures"] = failures

    return report, passed


def print_report(report):
    """Pretty-print a single deep-verification report."""
    fn = report["function"]
    bk = report["backend"]
    tag = "PASS" if report.get("passed", False) else "FAIL"
    print(f"  [{tag}] {fn} / {bk}")

    if "error" in report:
        print(f"         Error: {report['error']}")
        return

    print(f"         Mean diff:     {report['mean_diff']:>+14.8e}")
    print(f"         Max |diff|:    {report['max_abs_diff']:>14.8e}")
    print(f"         RMSE:          {report['rmse']:>14.8e}")
    print(f"         99th pct:      {report['pct99_abs_diff']:>14.8e}")
    print(f"         Rel RMSE:      {report['rel_rmse_pct']:>14.8f} %")
    print(f"         Pearson r:     {report['pearson_r']:>14.10f}")
    print(f"         NaN (ref/got): {report['ref_nan']}/{report['got_nan']}   "
          f"Inf (ref/got): {report['ref_inf']}/{report['got_inf']}")
    print(f"         Pts >1%% rel:  {report['pct_gt_1pct_rel']:.4f}%   "
          f">0.1%% rel: {report['pct_gt_0p1pct_rel']:.4f}%")

    if report.get("phys_bounds"):
        lo_b, hi_b = report["phys_bounds"]
        print(f"         Phys bounds [{lo_b}, {hi_b}]: "
              f"{report['phys_below']} below, {report['phys_above']} above")

    # Compact histogram
    if report.get("hist_counts") and len(report["hist_edges"]) > 2:
        edges = report["hist_edges"]
        counts = report["hist_counts"]
        max_c = max(counts) if counts else 1
        bar_w = 20
        print("         Diff histogram:")
        for i, c in enumerate(counts):
            bar_len = int(c / max(max_c, 1) * bar_w)
            bar = "#" * bar_len
            print(f"           [{edges[i]:.2e}, {edges[i+1]:.2e}): "
                  f"{c:>8d}  {bar}")

    if not report.get("passed", False) and report.get("failures"):
        print(f"         ** Reasons: {'; '.join(report['failures'])}")


# ==============================================================================
# Heat index extreme-condition stress test
# ==============================================================================

def heat_index_stress_test():
    """Compare all backends at T>40C, low RH, checking for physical sanity."""
    print()
    print("=" * 85)
    print("HEAT INDEX EXTREME-CONDITION STRESS TEST")
    print("  Testing T = 40-55 degC, RH = 2-15 %  (well beyond standard chart)")
    print("=" * 85)

    # Build small stress-test arrays
    n_t = 50
    n_rh = 50
    t_vals = np.linspace(40.0, 55.0, n_t)
    rh_vals = np.linspace(2.0, 15.0, n_rh)
    tt, rr = np.meshgrid(t_vals, rh_vals)
    t_stress = tt.ravel().astype(np.float64)
    rh_stress = rr.ravel().astype(np.float64)

    # MetPy reference
    ref = _to_numpy(mpcalc.heat_index(
        t_stress * mpunits.degC, rh_stress * mpunits.percent,
        mask_undefined=False))

    # metrust CPU
    mrust = _to_numpy(mcalc.heat_index(
        t_stress * mrunits.degC, rh_stress * mrunits.percent))

    # met-cu GPU
    t_2d_s = np.ascontiguousarray(t_stress, dtype=np.float64)
    rh_2d_s = np.ascontiguousarray(rh_stress, dtype=np.float64)
    metcu = _to_numpy(gpcalc.heat_index(t_2d_s, rh_2d_s))

    all_ok = True
    for label, arr in [("MetPy", ref), ("metrust_cpu", mrust), ("met-cu", metcu)]:
        finite = arr[np.isfinite(arr)]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        oob_lo = int((finite < 25.0).sum()) if finite.size > 0 else 0
        oob_hi = int((finite > 75.0).sum()) if finite.size > 0 else 0
        ok = (n_nan == 0 and n_inf == 0 and oob_lo == 0 and oob_hi == 0)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{tag}] {label:15s}  range [{finite.min():.2f}, {finite.max():.2f}] degC   "
              f"NaN={n_nan}  Inf={n_inf}  <25C={oob_lo}  >75C={oob_hi}")

    # Cross-backend comparison
    for label, arr in [("metrust_cpu", mrust), ("met-cu", metcu)]:
        diff = arr - ref
        absdiff = np.abs(diff)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        maxd = float(np.max(absdiff))
        r_val = float(np.corrcoef(ref, arr)[0, 1]) if np.std(ref) > 0 else float("nan")
        print(f"  {label:15s} vs MetPy:  RMSE={rmse:.6e}  max|diff|={maxd:.6e}  r={r_val:.10f}")

    # Focus on the most extreme corner: T>=50C, RH<=5%
    extreme_mask = (t_stress >= 50.0) & (rh_stress <= 5.0)
    if extreme_mask.sum() > 0:
        print(f"\n  Extreme corner (T>=50C, RH<=5%): {extreme_mask.sum()} points")
        for label, arr in [("MetPy", ref), ("metrust_cpu", mrust), ("met-cu", metcu)]:
            sub = arr[extreme_mask]
            print(f"    {label:15s}  min={sub.min():.4f}  max={sub.max():.4f}  "
                  f"mean={sub.mean():.4f}  std={sub.std():.4f}")
        # Check if any backend diverges from MetPy by >0.5C in this extreme region
        for label, arr in [("metrust_cpu", mrust), ("met-cu", metcu)]:
            maxd_ext = float(np.max(np.abs(arr[extreme_mask] - ref[extreme_mask])))
            tag_ext = "PASS" if maxd_ext < 0.5 else "WARN"
            if maxd_ext >= 0.5:
                all_ok = False
            print(f"    [{tag_ext}] {label} extreme max|diff|={maxd_ext:.6e}")

    print()
    return all_ok


# ---- Pint-wrapped inputs for MetPy / metrust-with-units ---------------------

t_q = temperature_c * mpunits.degC
rh_q = relative_humidity_pct * mpunits.percent
p_q = pressure_hpa * mpunits.hPa
u_q = u_ms * mpunits("m/s")
v_q = v_ms * mpunits("m/s")
e_q = vapor_pressure_hpa * mpunits.hPa
td_q = dewpoint_c * mpunits.degC

# Same with metrust units (identical pint registry)
t_mr = temperature_c * mrunits.degC
rh_mr = relative_humidity_pct * mrunits.percent
p_mr = pressure_hpa * mrunits.hPa
u_mr = u_ms * mrunits("m/s")
v_mr = v_ms * mrunits("m/s")
e_mr = vapor_pressure_hpa * mrunits.hPa
td_mr = dewpoint_c * mrunits.degC


# ---- Benchmark definitions ---------------------------------------------------

# Flat contiguous float64 for raw Rust / met-cu
t_flat = np.ascontiguousarray(temperature_c.ravel(), dtype=np.float64)
rh_flat = np.ascontiguousarray(relative_humidity_pct.ravel(), dtype=np.float64)
p_flat = np.ascontiguousarray(pressure_hpa.ravel(), dtype=np.float64)
u_flat = np.ascontiguousarray(u_ms.ravel(), dtype=np.float64)
v_flat = np.ascontiguousarray(v_ms.ravel(), dtype=np.float64)
e_flat = np.ascontiguousarray(vapor_pressure_hpa.ravel(), dtype=np.float64)
td_flat = np.ascontiguousarray(dewpoint_c.ravel(), dtype=np.float64)

# 2-D contiguous for met-cu / metrust-gpu (they accept ndarray directly)
t_2d = np.ascontiguousarray(temperature_c, dtype=np.float64)
rh_2d = np.ascontiguousarray(relative_humidity_pct, dtype=np.float64)
p_2d = np.ascontiguousarray(pressure_hpa, dtype=np.float64)
u_2d = np.ascontiguousarray(u_ms, dtype=np.float64)
v_2d = np.ascontiguousarray(v_ms, dtype=np.float64)
e_2d = np.ascontiguousarray(vapor_pressure_hpa, dtype=np.float64)
td_2d = np.ascontiguousarray(dewpoint_c, dtype=np.float64)


FUNCTIONS = [
    # ---- heat_index (CPU only -- no array kernel in Rust; met-cu has it) -----
    {
        "name": "heat_index",
        "metpy":       lambda: mpcalc.heat_index(t_q, rh_q, mask_undefined=False),
        "metrust_cpu": lambda: mcalc.heat_index(t_mr, rh_mr),
        "metcu":       lambda: gpcalc.heat_index(t_2d, rh_2d),
        "metrust_gpu": None,       # heat_index not routed to GPU in metrust
        # MetPy returns degC, metrust returns degC, met-cu returns degC
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1},
    },
    # ---- dewpoint_from_relative_humidity ------------------------------------
    {
        "name": "dewpoint_from_relative_humidity",
        "metpy":       lambda: mpcalc.dewpoint_from_relative_humidity(t_q, rh_q),
        "metrust_cpu": lambda: mcalc.dewpoint_from_relative_humidity(t_mr, rh_mr),
        "metcu":       lambda: gpcalc.dewpoint_from_relative_humidity(t_2d, rh_2d),
        "metrust_gpu": None,
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1},
    },
    # ---- relative_humidity_from_dewpoint ------------------------------------
    {
        "name": "relative_humidity_from_dewpoint",
        "metpy":       lambda: mpcalc.relative_humidity_from_dewpoint(t_q, td_q),
        "metrust_cpu": lambda: mcalc.relative_humidity_from_dewpoint(t_mr, td_mr),
        # met-cu returns 0-1 fractional; MetPy returns 0-1 dimensionless
        "metcu":       lambda: gpcalc.relative_humidity_from_dewpoint(t_2d, td_2d),
        "metrust_gpu": None,
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1},
    },
    # ---- saturation_vapor_pressure ------------------------------------------
    {
        "name": "saturation_vapor_pressure",
        "metpy":       lambda: mpcalc.saturation_vapor_pressure(t_q),
        "metrust_cpu": lambda: mcalc.saturation_vapor_pressure(t_mr),
        # met-cu returns Pa (same as MetPy/metrust which also return Pa)
        "metcu":       lambda: gpcalc.saturation_vapor_pressure(t_2d),
        "metrust_gpu": None,
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1},
    },
    # ---- wind_speed ---------------------------------------------------------
    {
        "name": "wind_speed",
        "metpy":       lambda: mpcalc.wind_speed(u_q, v_q),
        "metrust_cpu": lambda: mcalc.wind_speed(u_mr, v_mr),
        "metcu":       lambda: gpcalc.wind_speed(u_2d, v_2d),
        "metrust_gpu": None,
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1},
    },
    # ---- potential_temperature (GPU) ----------------------------------------
    {
        "name": "potential_temperature",
        "metpy":       lambda: mpcalc.potential_temperature(p_q, t_q),
        "metrust_cpu": lambda: mcalc.potential_temperature(p_mr, t_mr),
        "metcu":       lambda: gpcalc.potential_temperature(p_2d, t_2d),
        "metrust_gpu": "gpu",      # marker: run metrust with gpu backend
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1, "metrust_gpu": 1},
    },
    # ---- dewpoint (from vapor pressure, GPU) --------------------------------
    {
        "name": "dewpoint",
        "metpy":       lambda: mpcalc.dewpoint(e_q),
        "metrust_cpu": lambda: mcalc.dewpoint(e_mr),
        "metcu":       lambda: gpcalc.dewpoint(e_2d),
        "metrust_gpu": "gpu",
        "unit_scale":  {"metpy": 1, "metrust_cpu": 1, "metcu": 1, "metrust_gpu": 1},
    },
]


def _make_metrust_gpu_fn(spec):
    """Build a lambda that calls metrust with the GPU backend for the GPU-capable functions."""
    name = spec["name"]
    if name == "potential_temperature":
        def fn():
            with use_backend("gpu"):
                return mcalc.potential_temperature(p_mr, t_mr)
        return fn
    elif name == "dewpoint":
        def fn():
            with use_backend("gpu"):
                return mcalc.dewpoint(e_mr)
        return fn
    return None


# ---- Main --------------------------------------------------------------------

def main():
    width = 85
    hdr = f"{'Function':<40} {'MetPy':>10} {'metrust':>10} {'met-cu':>10} {'mrust-gpu':>10}  {'Speedup':>8}"
    sep = "-" * width

    print()
    print("=" * width)
    print("Benchmark 07 -- HRRR Fire Weather (800x800 = 640k points)")
    print("=" * width)
    print(f"  Grid: {NY}x{NX} ({N:,} points)")
    print(f"  Temperature: {temperature_c.min():.1f} - {temperature_c.max():.1f} degC")
    print(f"  RH: {relative_humidity_pct.min():.1f} - {relative_humidity_pct.max():.1f} %")
    print(f"  Wind: {np.hypot(u_ms, v_ms).min():.1f} - {np.hypot(u_ms, v_ms).max():.1f} m/s")
    print(f"  Pressure: {pressure_hpa.min():.0f} - {pressure_hpa.max():.0f} hPa")
    print(f"  Timing: {WARMUP} warmup + {REPEATS} runs, median")
    print()
    print(hdr)
    print(sep)

    all_pass = True
    all_reports = []   # (func_name, backend, report_dict)
    summary_rows = []  # for final summary table

    for spec in FUNCTIONS:
        name = spec["name"]
        results = {}
        timings = {}
        backends_run = []

        # ---- MetPy (reference) ----
        ms, ref = bench("MetPy", spec["metpy"])
        timings["metpy"] = ms
        results["metpy"] = ref
        backends_run.append("metpy")

        # ---- metrust CPU ----
        ms, val = bench("metrust", spec["metrust_cpu"])
        timings["metrust_cpu"] = ms
        results["metrust_cpu"] = val
        backends_run.append("metrust_cpu")

        # ---- met-cu direct ----
        if spec["metcu"] is not None:
            ms, val = bench("met-cu", spec["metcu"], gpu=True)
            timings["metcu"] = ms
            results["metcu"] = val
            backends_run.append("metcu")

        # ---- metrust GPU ----
        if spec["metrust_gpu"] == "gpu":
            gpu_fn = _make_metrust_gpu_fn(spec)
            if gpu_fn is not None:
                ms, val = bench("mrust-gpu", gpu_fn, gpu=True)
                timings["metrust_gpu"] = ms
                results["metrust_gpu"] = val
                backends_run.append("metrust_gpu")

        # ---- Deep verification against MetPy ----
        ref_np = results["metpy"]
        for bk in backends_run:
            if bk == "metpy":
                continue
            report, passed = deep_verify(name, ref_np, results[bk], bk)
            all_reports.append(report)
            if not passed:
                all_pass = False
            summary_rows.append((name, bk, report))

        # ---- Format output row (timing) ----
        t_metpy = timings.get("metpy", float("nan"))
        t_mrust = timings.get("metrust_cpu", float("nan"))
        t_metcu = timings.get("metcu", float("nan"))
        t_mrgpu = timings.get("metrust_gpu", float("nan"))

        # Best non-MetPy time for speedup
        candidates = [t for t in [t_mrust, t_metcu, t_mrgpu] if not np.isnan(t)]
        best = min(candidates) if candidates else float("nan")
        speedup = t_metpy / best if best > 0 else float("nan")

        row = f"{name:<40} {t_metpy:>9.2f}ms {t_mrust:>9.2f}ms"
        if not np.isnan(t_metcu):
            row += f" {t_metcu:>9.2f}ms"
        else:
            row += f" {'--':>10}"
        if not np.isnan(t_mrgpu):
            row += f" {t_mrgpu:>9.2f}ms"
        else:
            row += f" {'--':>10}"
        row += f"  {speedup:>7.1f}x"

        print(row)

    print(sep)
    print()

    # ==================================================================
    # Detailed per-backend verification reports
    # ==================================================================
    print("=" * width)
    print("DEEP DATA CORRECTNESS VERIFICATION")
    print("=" * width)
    print()

    for report in all_reports:
        print_report(report)
        print()

    # ==================================================================
    # Heat index extreme-condition stress test
    # ==================================================================
    stress_ok = heat_index_stress_test()
    if not stress_ok:
        all_pass = False

    # ==================================================================
    # Summary table
    # ==================================================================
    print("=" * width)
    print("SUMMARY TABLE")
    print("=" * width)
    print()

    col_w = {
        "Function": 36,
        "Backend": 14,
        "RMSE": 12,
        "RelRMSE%": 11,
        "Max|d|": 12,
        "99pct": 12,
        "r": 13,
        ">1%": 7,
        ">0.1%": 7,
        "NaN+": 5,
        "OOB": 5,
        "": 6,
    }

    header = (f"  {'Function':<36s} {'Backend':<14s} {'RMSE':>12s} {'RelRMSE%':>11s} "
              f"{'Max|d|':>12s} {'99pct':>12s} {'r':>13s} "
              f"{'>1%':>7s} {'>0.1%':>7s} {'NaN+':>5s} {'OOB':>5s} {'':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for func_name, bk, report in summary_rows:
        if "error" in report:
            print(f"  {func_name:<36s} {bk:<14s}  *** ERROR: {report['error']}")
            continue

        tag = "PASS" if report["passed"] else "FAIL"
        extra_nan = report["got_nan"] - report["ref_nan"]
        oob = report["phys_below"] + report["phys_above"]

        print(
            f"  {func_name:<36s} {bk:<14s} "
            f"{report['rmse']:>12.4e} {report['rel_rmse_pct']:>10.6f}% "
            f"{report['max_abs_diff']:>12.4e} {report['pct99_abs_diff']:>12.4e} "
            f"{report['pearson_r']:>13.10f} "
            f"{report['pct_gt_1pct_rel']:>6.2f}% {report['pct_gt_0p1pct_rel']:>6.2f}% "
            f"{extra_nan:>5d} {oob:>5d} "
            f"[{tag}]"
        )

    print()

    # ==================================================================
    # Final verdict
    # ==================================================================
    n_pass = sum(1 for r in all_reports if r.get("passed", False))
    n_fail = len(all_reports) - n_pass

    print("=" * width)
    if all_pass:
        print(f"VERDICT: ALL PASSED  ({n_pass}/{len(all_reports)} checks, "
              f"+ heat_index stress test)")
    else:
        print(f"VERDICT: ** FAILURES DETECTED **  "
              f"({n_pass} passed, {n_fail} failed out of {len(all_reports)} checks)")
    print("=" * width)
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
