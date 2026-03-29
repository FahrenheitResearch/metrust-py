#!/usr/bin/env python
"""Benchmark 08: ERA5-like Tropical Cyclone Analysis

Scenario: 0.25 degree grid, 200x200 subdomain centred on a synthetic TC,
20 vertical levels (1000-50 hPa).  Rankine-vortex wind field, warm-core
temperature anomaly, high moisture in the eyewall.

Functions benchmarked
---------------------
  vorticity                           (GPU)  -- the canonical TC field
  equivalent_potential_temperature    (GPU)  -- warm-core identification
  potential_temperature               (GPU)
  compute_pw                          (GPU)  -- 3-D precipitable water
  wind_speed                          (CPU)
  divergence                          (CPU)

Backends: MetPy (Pint), metrust CPU, met-cu direct, metrust GPU.
MetPy does NOT have a grid compute_pw -- that row uses 3 backends only.

Verification : deep correctness audit per function --
  mean diff, max abs diff, RMSE, 99th pct, relative RMSE%, NaN/Inf audit,
  physical plausibility bounds, Pearson r, % points > 1%/0.1% relative error,
  histogram of diffs, and special TC structural checks.

Timing       : perf_counter, cupy sync, 1 warmup + 3 timed, median
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
import metpy.calc as mpcalc
from metpy.units import units

import metrust.calc as mrcalc

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
# Timing helpers
# ============================================================================
WARMUP = 1
TRIALS = 3


def _sync_gpu():
    cp.cuda.Stream.null.synchronize()


def _time_func(func, gpu=False):
    """1 warmup + TRIALS timed runs, return median ms."""
    if gpu:
        _sync_gpu()
    func()  # warmup
    if gpu:
        _sync_gpu()

    times = []
    for _ in range(TRIALS):
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
        return "--"
    if ms < 0.01:
        return f"{ms * 1000:.1f} us"
    if ms < 1:
        return f"{ms:.3f} ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def spd(ref, tgt):
    if ref is None or tgt is None or tgt <= 0:
        return "--"
    r = ref / tgt
    return f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"


# ============================================================================
# Synthetic tropical cyclone field generator
# ============================================================================
def make_tc_fields():
    """Build synthetic TC data on a 200x200 x 20-level grid.

    Returns a dict with all arrays needed by the six benchmarked functions.
    """
    NX = NY = 200
    NZ = 20
    DX = DY = 27800.0  # metres (0.25 deg at tropics)

    # Pressure levels (hPa) -- surface to top
    p_levels_hPa = np.array([
        1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
        600, 550, 500, 400, 300, 250, 200, 150, 100, 50,
    ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Horizontal grid (distance from storm centre)
    # ------------------------------------------------------------------
    cx, cy = NX // 2, NY // 2
    x = (np.arange(NX) - cx) * DX  # metres
    y = (np.arange(NY) - cy) * DY
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)        # radius from centre
    THETA = np.arctan2(Y, X)         # azimuthal angle

    # ------------------------------------------------------------------
    # Rankine vortex (tangential wind)
    # ------------------------------------------------------------------
    Rmax = 50_000.0      # eye radius ~ 50 km
    Vmax = 50.0           # max wind ~ 50 m/s
    V_t = np.where(R <= Rmax,
                   Vmax * R / Rmax,
                   Vmax * (Rmax / np.maximum(R, 1.0)))

    # Cyclonic (counter-clockwise in NH): tangential -> u,v
    u_2d = -V_t * np.sin(THETA)
    v_2d =  V_t * np.cos(THETA)

    # ------------------------------------------------------------------
    # Vertical decay of vortex wind (strongest at 850 hPa, weakens aloft)
    # ------------------------------------------------------------------
    weight_wind = np.exp(-0.5 * ((p_levels_hPa - 850.0) / 300.0)**2)
    u_3d = u_2d[None, :, :] * weight_wind[:, None, None]
    v_3d = v_2d[None, :, :] * weight_wind[:, None, None]

    # ------------------------------------------------------------------
    # Base temperature profile (tropical sounding, Celsius)
    # ------------------------------------------------------------------
    T_sfc = 28.0  # SST-ish
    T_top = -75.0
    T_base = np.linspace(T_sfc, T_top, NZ)  # 1-D profile

    # Warm-core anomaly: Gaussian in r, peaked at 300 hPa
    warm_core_r = np.exp(-(R / 100_000.0)**2)
    warm_core_z = np.exp(-0.5 * ((p_levels_hPa - 300.0) / 150.0)**2)
    dT_warm = 10.0 * warm_core_z[:, None, None] * warm_core_r[None, :, :]
    T_3d = T_base[:, None, None] + dT_warm  # (nz, ny, nx)  Celsius

    # ------------------------------------------------------------------
    # Dewpoint (moisture: high near eyewall, drier in eye and outskirts)
    # ------------------------------------------------------------------
    rh_base = np.linspace(0.85, 0.20, NZ)  # tropical RH profile
    eyewall_mask = np.exp(-((R - Rmax) / 80_000.0)**2)  # enhance near eyewall
    eye_drying = np.where(R < Rmax * 0.5, 0.6, 1.0)     # drier inside eye
    rh_3d = np.clip(
        rh_base[:, None, None] * (1.0 + 0.15 * eyewall_mask[None, :, :]) * eye_drying[None, :, :],
        0.05, 0.99,
    )

    # Dewpoint from RH and T (Magnus formula, approximate)
    a, b = 17.27, 237.3
    gamma = a * T_3d / (b + T_3d) + np.log(rh_3d)
    Td_3d = b * gamma / (a - gamma)  # Celsius

    # ------------------------------------------------------------------
    # Mixing ratio (for compute_pw)
    # ------------------------------------------------------------------
    e_s = 6.112 * np.exp(a * T_3d / (b + T_3d))      # sat vp (hPa)
    e   = rh_3d * e_s                                   # actual vp (hPa)
    w_3d = 0.622 * e / (p_levels_hPa[:, None, None] - e)  # mixing ratio kg/kg

    # ------------------------------------------------------------------
    # Pressure 3-D (Pa) for compute_pw
    # ------------------------------------------------------------------
    p3_Pa = np.broadcast_to(
        p_levels_hPa[:, None, None] * 100.0, (NZ, NY, NX)
    ).copy()

    return dict(
        NX=NX, NY=NY, NZ=NZ, DX=DX, DY=DY,
        cx=cx, cy=cy,
        Rmax=Rmax,
        R=R,
        p_levels_hPa=p_levels_hPa,
        u_2d=u_2d, v_2d=v_2d,
        u_3d=u_3d, v_3d=v_3d,
        T_3d=T_3d,        # Celsius (nz, ny, nx)
        Td_3d=Td_3d,      # Celsius (nz, ny, nx)
        w_3d=w_3d,         # mixing ratio kg/kg (nz, ny, nx)
        p3_Pa=p3_Pa,       # Pa (nz, ny, nx)
    )


# ============================================================================
# Results table
# ============================================================================
COL_W = {"name": 38, "t": 11, "r": 9}
rows = []


def header():
    print(f"  {'Function':{COL_W['name']}s}"
          f"  {'MetPy':>{COL_W['t']}s}"
          f"  {'Rust CPU':>{COL_W['t']}s}"
          f"  {'met-cu':>{COL_W['t']}s}"
          f"  {'Rust GPU':>{COL_W['t']}s}"
          f"  {'Py/Rust':>{COL_W['r']}s}"
          f"  {'Rust/GPU':>{COL_W['r']}s}")
    print("  " + "-" * (COL_W["name"] + 4 * COL_W["t"] + 2 * COL_W["r"] + 10))


def record(name, t_mp, t_cpu, t_mcu, t_gpu, vfy_detail):
    """vfy_detail: dict of pair_name -> bool, e.g. {'MP-CPU': True, ...}"""
    all_ok = all(vfy_detail.values()) if vfy_detail else True
    rows.append((name, t_mp, t_cpu, t_mcu, t_gpu, all_ok, vfy_detail))
    mark = "PASS" if all_ok else "FAIL"
    fails = [k for k, v in vfy_detail.items() if not v]
    detail = ""
    if fails:
        detail = f"  ({', '.join(fails)})"
    print(f"  {name:{COL_W['name']}s}"
          f"  {fmt(t_mp):>{COL_W['t']}s}"
          f"  {fmt(t_cpu):>{COL_W['t']}s}"
          f"  {fmt(t_mcu):>{COL_W['t']}s}"
          f"  {fmt(t_gpu):>{COL_W['t']}s}"
          f"  {spd(t_mp, t_cpu):>{COL_W['r']}s}"
          f"  {spd(t_cpu, t_gpu):>{COL_W['r']}s}"
          f"  [{mark}]{detail}")


# ============================================================================
# Helpers: strip Pint units to plain ndarray
# ============================================================================
def _strip(val):
    """Return plain float64 ndarray, stripping Pint if present."""
    if hasattr(val, "magnitude"):
        return np.asarray(val.magnitude, dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


def _asnp(val):
    """CuPy -> NumPy, or just ensure numpy float64."""
    if hasattr(val, "get"):
        return np.asarray(val.get(), dtype=np.float64)
    return np.asarray(val, dtype=np.float64)


def _check(ref, test, rtol=1e-4, atol=1e-12):
    """allclose that returns False (not error) when NaN present."""
    if np.any(np.isnan(test)):
        return False
    return bool(np.allclose(ref, test, rtol=rtol, atol=atol))


# ============================================================================
# Deep verification framework
# ============================================================================
class DeepVerifier:
    """Accumulates per-function verification results across all backends."""

    def __init__(self):
        self.reports = []  # list of dicts, one per (function, pair) comparison
        self.tc_checks = []  # list of dicts for TC-specific structural checks
        self.all_pass = True

    # ------------------------------------------------------------------
    # Core statistical comparison
    # ------------------------------------------------------------------
    def compare(self, func_name, pair_name, ref, test, *,
                phys_lo=None, phys_hi=None, phys_label="",
                rtol_pass=1e-4, atol_pass=1e-12):
        """Run full statistical battery on (ref, test) pair.

        Returns True if PASS, False if FAIL.
        """
        ref = np.asarray(ref, dtype=np.float64).ravel()
        test = np.asarray(test, dtype=np.float64).ravel()
        n = ref.size

        report = {
            "func": func_name, "pair": pair_name, "n_points": n,
            "pass": True, "fail_reasons": [],
        }

        # --- NaN / Inf audit ---
        nan_ref = int(np.count_nonzero(np.isnan(ref)))
        nan_test = int(np.count_nonzero(np.isnan(test)))
        inf_ref = int(np.count_nonzero(np.isinf(ref)))
        inf_test = int(np.count_nonzero(np.isinf(test)))
        report["nan_ref"] = nan_ref
        report["nan_test"] = nan_test
        report["inf_ref"] = inf_ref
        report["inf_test"] = inf_test

        if nan_test > nan_ref:
            report["fail_reasons"].append(
                f"NaN count grew: ref={nan_ref}, test={nan_test}")
            report["pass"] = False
        if inf_test > inf_ref:
            report["fail_reasons"].append(
                f"Inf count grew: ref={inf_ref}, test={inf_test}")
            report["pass"] = False

        # Mask out NaN/Inf for stats
        valid = np.isfinite(ref) & np.isfinite(test)
        n_valid = int(np.count_nonzero(valid))
        report["n_valid"] = n_valid
        if n_valid == 0:
            report["fail_reasons"].append("No finite points to compare")
            report["pass"] = False
            self.reports.append(report)
            self.all_pass = self.all_pass and report["pass"]
            return report["pass"]

        r, t = ref[valid], test[valid]
        diff = t - r
        abs_diff = np.abs(diff)

        # --- Basic error statistics ---
        report["mean_diff"] = float(np.mean(diff))
        report["max_abs_diff"] = float(np.max(abs_diff))
        report["rmse"] = float(np.sqrt(np.mean(diff**2)))
        report["pct99"] = float(np.percentile(abs_diff, 99))
        report["median_abs_diff"] = float(np.median(abs_diff))

        # --- Relative RMSE % ---
        ref_range = float(np.max(np.abs(r)))
        if ref_range > 0:
            report["rel_rmse_pct"] = report["rmse"] / ref_range * 100.0
        else:
            report["rel_rmse_pct"] = 0.0

        # --- Pearson r ---
        if n_valid >= 2 and np.std(r) > 0 and np.std(t) > 0:
            pr = sp_stats.pearsonr(r, t)
            report["pearson_r"] = float(pr.statistic)
            report["pearson_p"] = float(pr.pvalue)
        else:
            report["pearson_r"] = float("nan")
            report["pearson_p"] = float("nan")

        # --- % points exceeding relative thresholds ---
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(np.abs(r) > 1e-30, abs_diff / np.abs(r), 0.0)
        report["pct_gt_1pct"] = float(np.count_nonzero(rel_err > 0.01) / n_valid * 100)
        report["pct_gt_0_1pct"] = float(np.count_nonzero(rel_err > 0.001) / n_valid * 100)

        # --- Histogram of absolute diffs (log10 bins) ---
        bins = [0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1, 10, 100, np.inf]
        hist_counts, _ = np.histogram(abs_diff, bins=bins)
        report["diff_histogram"] = {
            "bins": [str(b) for b in bins],
            "counts": [int(c) for c in hist_counts],
        }

        # --- Physical plausibility ---
        if phys_lo is not None or phys_hi is not None:
            test_full = np.asarray(test, dtype=np.float64)  # already raveled
            oob = np.zeros(test_full.size, dtype=bool)
            if phys_lo is not None:
                oob |= test_full < phys_lo
            if phys_hi is not None:
                oob |= test_full > phys_hi
            n_oob = int(np.count_nonzero(oob & np.isfinite(test_full)))
            report["phys_label"] = phys_label
            report["phys_lo"] = phys_lo
            report["phys_hi"] = phys_hi
            report["phys_oob"] = n_oob
            report["phys_oob_pct"] = n_oob / max(n_valid, 1) * 100.0
            if n_oob > 0:
                report["fail_reasons"].append(
                    f"Phys bounds [{phys_lo}, {phys_hi}] {phys_label}: "
                    f"{n_oob} OOB ({report['phys_oob_pct']:.3f}%)")
                report["pass"] = False

        # --- allclose gate ---
        if not np.allclose(r, t, rtol=rtol_pass, atol=atol_pass):
            report["fail_reasons"].append(
                f"allclose(rtol={rtol_pass}, atol={atol_pass}) FAILED")
            report["pass"] = False

        # --- Pearson correlation gate ---
        if report.get("pearson_r") is not None and np.isfinite(report["pearson_r"]):
            if report["pearson_r"] < 0.99999:
                report["fail_reasons"].append(
                    f"Pearson r={report['pearson_r']:.8f} < 0.99999")
                report["pass"] = False

        self.reports.append(report)
        self.all_pass = self.all_pass and report["pass"]
        return report["pass"]

    # ------------------------------------------------------------------
    # TC-specific structural checks
    # ------------------------------------------------------------------
    def tc_vorticity_check(self, vort_2d, cx, cy, Rmax, DX):
        """Verify vorticity max is near storm center and has correct sign."""
        v = np.asarray(vort_2d, dtype=np.float64)
        check = {"name": "TC vorticity structure", "pass": True, "notes": []}

        # Max vorticity location
        max_idx = np.unravel_index(np.argmax(v), v.shape)
        max_val = float(v[max_idx])
        dist_from_center_km = np.sqrt(
            (max_idx[1] - cx)**2 + (max_idx[0] - cy)**2) * DX / 1000.0

        check["vort_max"] = max_val
        check["vort_max_ij"] = max_idx
        check["vort_max_dist_km"] = dist_from_center_km
        check["notes"].append(
            f"Vort max = {max_val:.6e} at ({max_idx[0]},{max_idx[1]}), "
            f"{dist_from_center_km:.1f} km from center")

        # Should be near center (within ~2 * Rmax)
        if dist_from_center_km > 2.0 * Rmax / 1000.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: vort max {dist_from_center_km:.1f} km from center, "
                f"expected within {2.0 * Rmax / 1000.0:.0f} km")

        # Physical bounds: vorticity should be within +/-5e-3 /s near TC
        if abs(max_val) > 5e-3:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: vort max {max_val:.4e} exceeds 5e-3 /s")

        # Positive vorticity at center for NH cyclone
        center_vort = float(v[cy, cx])
        check["vort_center"] = center_vort
        if center_vort <= 0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: center vort = {center_vort:.4e}, expected > 0 for NH cyclone")
        else:
            check["notes"].append(
                f"Center vort = {center_vort:.6e} (positive, NH cyclone: OK)")

        self.tc_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def tc_wind_speed_check(self, ws_2d, u_2d, v_2d, cx, cy, Rmax, DX):
        """Verify cyclonic pattern: min at center, max near Rmax."""
        ws = np.asarray(ws_2d, dtype=np.float64)
        check = {"name": "TC wind speed structure", "pass": True, "notes": []}

        # Center should be calm (eye)
        center_ws = float(ws[cy, cx])
        check["ws_center"] = center_ws
        if center_ws > 5.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: eye wind speed = {center_ws:.2f} m/s, expected < 5 m/s")
        else:
            check["notes"].append(f"Eye wind speed = {center_ws:.2f} m/s (calm: OK)")

        # Max should be near Rmax
        max_idx = np.unravel_index(np.argmax(ws), ws.shape)
        max_val = float(ws[max_idx])
        dist_km = np.sqrt(
            (max_idx[1] - cx)**2 + (max_idx[0] - cy)**2) * DX / 1000.0
        rmax_km = Rmax / 1000.0

        check["ws_max"] = max_val
        check["ws_max_dist_km"] = dist_km
        check["notes"].append(
            f"WS max = {max_val:.2f} m/s at {dist_km:.1f} km "
            f"(Rmax = {rmax_km:.0f} km)")

        if abs(dist_km - rmax_km) > rmax_km * 1.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: WS max at {dist_km:.1f} km, expected near {rmax_km:.0f} km")

        # Physical: wind speed 0-60 m/s
        if max_val > 60.0 or float(np.min(ws)) < 0.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: WS range [{np.min(ws):.2f}, {max_val:.2f}] outside [0, 60]")

        # Cyclonic check: verify tangential component is counter-clockwise (NH)
        # Sample points east of center at Rmax: v should be positive (northward)
        j_east = int(round(cx + Rmax / DX))
        if 0 <= j_east < ws.shape[1]:
            v_east = float(v_2d[cy, j_east])
            check["v_east_of_center"] = v_east
            if v_east > 0:
                check["notes"].append(
                    f"East-of-center v = {v_east:.2f} m/s (northward, CCW: OK)")
            else:
                check["pass"] = False
                check["notes"].append(
                    f"FAIL: East-of-center v = {v_east:.2f} m/s, expected > 0 (CCW)")

        self.tc_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def tc_theta_e_check(self, ept_2d, cx, cy, Rmax, DX):
        """Verify theta-e is highest in eyewall, reasonable tropical range."""
        e = np.asarray(ept_2d, dtype=np.float64)
        check = {"name": "TC theta-e structure", "pass": True, "notes": []}

        # Build radial distance array
        ny, nx = e.shape
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        r_grid = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2) * DX

        # Eyewall ring: Rmax +/- 50% Rmax
        eyewall = (r_grid >= 0.5 * Rmax) & (r_grid <= 1.5 * Rmax)
        outer = r_grid > 3.0 * Rmax
        eye = r_grid < 0.3 * Rmax

        ept_eyewall_mean = float(np.mean(e[eyewall])) if np.any(eyewall) else float("nan")
        ept_outer_mean = float(np.mean(e[outer])) if np.any(outer) else float("nan")
        ept_eye_mean = float(np.mean(e[eye])) if np.any(eye) else float("nan")

        check["ept_eyewall_mean"] = ept_eyewall_mean
        check["ept_outer_mean"] = ept_outer_mean
        check["ept_eye_mean"] = ept_eye_mean
        check["notes"].append(
            f"Theta-e: eyewall={ept_eyewall_mean:.1f} K, "
            f"outer={ept_outer_mean:.1f} K, eye={ept_eye_mean:.1f} K")

        # Eyewall should be warmest (highest theta-e)
        if np.isfinite(ept_eyewall_mean) and np.isfinite(ept_outer_mean):
            if ept_eyewall_mean <= ept_outer_mean:
                check["pass"] = False
                check["notes"].append(
                    f"FAIL: eyewall theta-e ({ept_eyewall_mean:.1f}) "
                    f"<= outer ({ept_outer_mean:.1f})")
            else:
                check["notes"].append(
                    f"Eyewall > outer by {ept_eyewall_mean - ept_outer_mean:.1f} K: OK")

        # Physical range: theta-e must be positive and reasonable
        # Note: this synthetic sounding has a cold 850 hPa level (~1 C base)
        # so eyewall theta-e is ~293-298 K, not real-world tropical ~340-365 K.
        # We check a broad sanity band and rely on structural checks above.
        if np.isfinite(ept_eyewall_mean):
            if ept_eyewall_mean < 200 or ept_eyewall_mean > 450:
                check["pass"] = False
                check["notes"].append(
                    f"FAIL: eyewall theta-e {ept_eyewall_mean:.1f} K "
                    f"outside sanity range [200, 450]")
            else:
                check["notes"].append(
                    f"Eyewall theta-e {ept_eyewall_mean:.1f} K in [200, 450]: OK")

        self.tc_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def tc_pw_check(self, pw_2d, cx, cy, Rmax, DX):
        """Verify PW is in tropical range and enhanced near eyewall."""
        pw = np.asarray(pw_2d, dtype=np.float64)
        check = {"name": "TC precipitable water structure", "pass": True, "notes": []}

        ny, nx = pw.shape
        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        r_grid = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2) * DX

        eyewall = (r_grid >= 0.5 * Rmax) & (r_grid <= 1.5 * Rmax)
        outer = r_grid > 3.0 * Rmax

        pw_max = float(np.max(pw))
        pw_min = float(np.min(pw[np.isfinite(pw)]))
        pw_eyewall = float(np.mean(pw[eyewall])) if np.any(eyewall) else float("nan")
        pw_outer = float(np.mean(pw[outer])) if np.any(outer) else float("nan")

        check["pw_range"] = (pw_min, pw_max)
        check["pw_eyewall_mean"] = pw_eyewall
        check["pw_outer_mean"] = pw_outer
        check["notes"].append(
            f"PW range: [{pw_min:.1f}, {pw_max:.1f}] mm")
        check["notes"].append(
            f"PW: eyewall={pw_eyewall:.1f} mm, outer={pw_outer:.1f} mm")

        # Physical sanity: PW must be non-negative, < 100 mm
        # Note: this synthetic sounding yields ~12-21 mm (limited vertical
        # extent & cold mid-levels), not real-world tropical 40-70 mm.
        if pw_min < 0.0:
            check["pass"] = False
            check["notes"].append(f"FAIL: PW min {pw_min:.1f} mm is negative")
        if pw_max > 100.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: PW max {pw_max:.1f} mm exceeds 100 mm sanity cap")
        if pw_min >= 0.0 and pw_max <= 100.0:
            check["notes"].append(f"PW range [{pw_min:.1f}, {pw_max:.1f}] mm: OK")

        # Eyewall should have higher PW than outer
        if np.isfinite(pw_eyewall) and np.isfinite(pw_outer):
            if pw_eyewall > pw_outer:
                check["notes"].append(
                    f"PW eyewall > outer by {pw_eyewall - pw_outer:.1f} mm: OK")
            else:
                check["notes"].append(
                    f"PW eyewall ({pw_eyewall:.1f}) <= outer ({pw_outer:.1f}): "
                    f"unexpected but not fatal for synthetic data")

        self.tc_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    # ------------------------------------------------------------------
    # Pretty-print all results
    # ------------------------------------------------------------------
    def print_detailed_report(self):
        """Print the full verification report."""
        W = 110
        print()
        print("=" * W)
        print("  DEEP VERIFICATION REPORT")
        print("=" * W)

        # Group reports by function
        funcs_seen = []
        for r in self.reports:
            if r["func"] not in funcs_seen:
                funcs_seen.append(r["func"])

        for func in funcs_seen:
            func_reports = [r for r in self.reports if r["func"] == func]
            print()
            print(f"  --- {func} ---")
            for r in func_reports:
                status = "PASS" if r["pass"] else "FAIL"
                print(f"    [{status}] {r['pair']}  "
                      f"({r['n_points']} pts, {r.get('n_valid', 0)} valid)")

                if r.get("n_valid", 0) == 0:
                    for reason in r["fail_reasons"]:
                        print(f"           {reason}")
                    continue

                # Stats table
                print(f"           Mean diff   = {r['mean_diff']:+.6e}")
                print(f"           Max |diff|  = {r['max_abs_diff']:.6e}")
                print(f"           Median |d|  = {r['median_abs_diff']:.6e}")
                print(f"           RMSE        = {r['rmse']:.6e}")
                print(f"           99th pct    = {r['pct99']:.6e}")
                print(f"           Rel RMSE    = {r['rel_rmse_pct']:.6f} %")
                if np.isfinite(r.get("pearson_r", float("nan"))):
                    print(f"           Pearson r   = {r['pearson_r']:.10f}")
                print(f"           NaN  ref/test = {r['nan_ref']}/{r['nan_test']}")
                print(f"           Inf  ref/test = {r['inf_ref']}/{r['inf_test']}")
                print(f"           >1% rel err = {r['pct_gt_1pct']:.4f} %")
                print(f"           >0.1% rel   = {r['pct_gt_0_1pct']:.4f} %")

                # Physical bounds
                if "phys_label" in r:
                    oob_str = "OK" if r["phys_oob"] == 0 else f"{r['phys_oob']} OOB"
                    print(f"           Phys [{r['phys_lo']}, {r['phys_hi']}] "
                          f"{r['phys_label']}: {oob_str}")

                # Histogram
                h = r["diff_histogram"]
                bin_labels = []
                for i in range(len(h["bins"]) - 1):
                    bin_labels.append(f"[{h['bins'][i]},{h['bins'][i+1]})")
                print(f"           Diff histogram (|diff| bins):")
                for bl, c in zip(bin_labels, h["counts"]):
                    bar = "#" * min(c * 40 // max(max(h["counts"]), 1), 40)
                    print(f"             {bl:>22s}: {c:>7d} {bar}")

                # Failures
                for reason in r["fail_reasons"]:
                    print(f"           ** {reason}")

        # TC structural checks
        if self.tc_checks:
            print()
            print(f"  --- TC structural checks ---")
            for c in self.tc_checks:
                status = "PASS" if c["pass"] else "FAIL"
                print(f"    [{status}] {c['name']}")
                for note in c["notes"]:
                    print(f"           {note}")

        # Summary table
        print()
        print("-" * W)
        print(f"  {'Function':<30s} {'Pair':<12s} {'RMSE':>12s} "
              f"{'MaxDiff':>12s} {'RelRMSE%':>10s} {'Pearson r':>12s} "
              f"{'Status':>8s}")
        print("  " + "-" * (W - 4))
        for r in self.reports:
            pr_str = (f"{r['pearson_r']:.8f}"
                      if np.isfinite(r.get("pearson_r", float("nan")))
                      else "n/a")
            status = "PASS" if r["pass"] else "** FAIL"
            print(f"  {r['func']:<30s} {r['pair']:<12s} "
                  f"{r['rmse']:>12.6e} {r['max_abs_diff']:>12.6e} "
                  f"{r['rel_rmse_pct']:>9.4f}% {pr_str:>12s} {status:>8s}")

        n_pass = sum(1 for r in self.reports if r["pass"])
        n_fail = sum(1 for r in self.reports if not r["pass"])
        tc_pass = sum(1 for c in self.tc_checks if c["pass"])
        tc_fail = sum(1 for c in self.tc_checks if not c["pass"])

        print()
        print(f"  Statistical comparisons: {n_pass} PASS, {n_fail} FAIL "
              f"(of {n_pass + n_fail})")
        print(f"  TC structural checks:    {tc_pass} PASS, {tc_fail} FAIL "
              f"(of {tc_pass + tc_fail})")
        overall = "ALL PASS" if self.all_pass else "SOME FAILURES"
        print(f"  Overall verification:    {overall}")
        print("=" * W)


# ============================================================================
# Main benchmark
# ============================================================================
def main():
    print("=" * 110)
    print("  BENCHMARK 08 : ERA5-like Tropical Cyclone Analysis")
    print(f"  Grid: 200x200 x 20 levels  |  dx=dy=27.8 km (0.25 deg at tropics)")
    print(f"  GPU : {GPU_NAME if HAS_GPU else 'not available'}  |  met-cu: {'yes' if HAS_METCU else 'no'}")
    print("=" * 110)

    # -- build fields --
    print("  Generating synthetic tropical cyclone fields ... ", end="", flush=True)
    t0 = time.perf_counter()
    d = make_tc_fields()
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    NX, NY, NZ = d["NX"], d["NY"], d["NZ"]
    DX, DY = d["DX"], d["DY"]
    cx, cy = d["cx"], d["cy"]
    Rmax = d["Rmax"]

    verifier = DeepVerifier()

    # -- MetPy Pint wrappers (not timed) --
    u2_q   = d["u_2d"] * units("m/s")
    v2_q   = d["v_2d"] * units("m/s")
    dx_q   = DX * units.m
    dy_q   = DY * units.m

    # Use 850 hPa slice (index 5) for 2-D thermo tests
    i850 = 5  # p_levels_hPa[5] = 850
    T850   = d["T_3d"][i850]
    Td850  = d["Td_3d"][i850]
    T850_q = T850 * units.degC
    Td850_q = Td850 * units.degC
    p850_q = 850.0 * units.hPa
    # met-cu needs pressure broadcast to match array shape
    p850_arr = np.full_like(T850, 850.0)

    # ================================================================
    print()
    header()

    all_pass = True

    # ----------------------------------------------------------------
    # 1. VORTICITY (GPU)
    # ----------------------------------------------------------------
    # MetPy
    t_mp = _time_func(lambda: mpcalc.vorticity(u2_q, v2_q, dx=dx_q, dy=dy_q))
    ref_vort = _strip(mpcalc.vorticity(u2_q, v2_q, dx=dx_q, dy=dy_q))

    # metrust CPU
    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY))
    mr_vort = _strip(mrcalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY))

    # met-cu
    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY),
            gpu=True)
        mcu_vort = _asnp(mcucalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY))

    # metrust GPU
    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY),
            gpu=True)
        mr_gpu_vort = _strip(mrcalc.vorticity(d["u_2d"], d["v_2d"], dx=DX, dy=DY))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_vort, mr_vort)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_vort, mr_gpu_vort)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_vort, mcu_vort)
    all_pass = all_pass and all(vd.values())
    record("vorticity", t_mp, t_cpu, t_mcu, t_gpu, vd)

    # Deep verification: vorticity
    verifier.compare("vorticity", "MP-CPU", ref_vort, mr_vort,
                     phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    if HAS_GPU:
        verifier.compare("vorticity", "MP-GPU", ref_vort, mr_gpu_vort,
                         phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    if HAS_METCU:
        verifier.compare("vorticity", "MP-MCU", ref_vort, mcu_vort,
                         phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    # TC structural: vorticity max near center
    verifier.tc_vorticity_check(ref_vort, cx, cy, Rmax, DX)
    verifier.tc_vorticity_check(mr_vort, cx, cy, Rmax, DX)

    # ----------------------------------------------------------------
    # 2. EQUIVALENT POTENTIAL TEMPERATURE (GPU)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.equivalent_potential_temperature(
        p850_q, T850_q, Td850_q))
    ref_ept = _strip(mpcalc.equivalent_potential_temperature(
        p850_q, T850_q, Td850_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.equivalent_potential_temperature(
        850.0, T850, Td850))
    mr_ept = _strip(mrcalc.equivalent_potential_temperature(850.0, T850, Td850))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.equivalent_potential_temperature(p850_arr, T850, Td850),
            gpu=True)
        mcu_ept = _asnp(mcucalc.equivalent_potential_temperature(p850_arr, T850, Td850))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.equivalent_potential_temperature(850.0, T850, Td850),
            gpu=True)
        mr_gpu_ept = _strip(mrcalc.equivalent_potential_temperature(850.0, T850, Td850))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_ept, mr_ept)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_ept, mr_gpu_ept)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_ept, mcu_ept)
    all_pass = all_pass and all(vd.values())
    record("equiv_potential_temp", t_mp, t_cpu, t_mcu, t_gpu, vd)

    # Deep verification: equiv_potential_temperature
    # Synthetic sounding theta-e at 850 hPa is ~293-298 K (cold level)
    verifier.compare("equiv_potential_temp", "MP-CPU", ref_ept, mr_ept,
                     phys_lo=200, phys_hi=450, phys_label="K")
    if HAS_GPU:
        verifier.compare("equiv_potential_temp", "MP-GPU", ref_ept, mr_gpu_ept,
                         phys_lo=200, phys_hi=450, phys_label="K")
    if HAS_METCU:
        verifier.compare("equiv_potential_temp", "MP-MCU", ref_ept, mcu_ept,
                         phys_lo=200, phys_hi=450, phys_label="K")
    # TC structural: theta-e highest in eyewall
    verifier.tc_theta_e_check(ref_ept, cx, cy, Rmax, DX)

    # ----------------------------------------------------------------
    # 3. POTENTIAL TEMPERATURE (GPU)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.potential_temperature(p850_q, T850_q))
    ref_pt = _strip(mpcalc.potential_temperature(p850_q, T850_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.potential_temperature(850.0, T850))
    mr_pt = _strip(mrcalc.potential_temperature(850.0, T850))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.potential_temperature(p850_arr, T850),
            gpu=True)
        mcu_pt = _asnp(mcucalc.potential_temperature(p850_arr, T850))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.potential_temperature(850.0, T850),
            gpu=True)
        mr_gpu_pt = _strip(mrcalc.potential_temperature(850.0, T850))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_pt, mr_pt)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_pt, mr_gpu_pt)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_pt, mcu_pt)
    all_pass = all_pass and all(vd.values())
    record("potential_temperature", t_mp, t_cpu, t_mcu, t_gpu, vd)

    # Deep verification: potential_temperature
    verifier.compare("potential_temperature", "MP-CPU", ref_pt, mr_pt,
                     phys_lo=250, phys_hi=400, phys_label="K")
    if HAS_GPU:
        verifier.compare("potential_temperature", "MP-GPU", ref_pt, mr_gpu_pt,
                         phys_lo=250, phys_hi=400, phys_label="K")
    if HAS_METCU:
        verifier.compare("potential_temperature", "MP-MCU", ref_pt, mcu_pt,
                         phys_lo=250, phys_hi=400, phys_label="K")

    # ----------------------------------------------------------------
    # 4. COMPUTE_PW (GPU) -- no MetPy grid equivalent
    # ----------------------------------------------------------------
    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.compute_pw(d["w_3d"], d["p3_Pa"]))
    mr_pw = _strip(mrcalc.compute_pw(d["w_3d"], d["p3_Pa"]))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.compute_pw(d["w_3d"], d["p3_Pa"]),
            gpu=True)
        mcu_pw = _asnp(mcucalc.compute_pw(d["w_3d"], d["p3_Pa"]))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.compute_pw(d["w_3d"], d["p3_Pa"]),
            gpu=True)
        mr_gpu_pw = _strip(mrcalc.compute_pw(d["w_3d"], d["p3_Pa"]))
        mrcalc.set_backend("cpu")

    vd = {}
    if HAS_METCU:
        vd["CPU-MCU"] = _check(mr_pw, mcu_pw)
    if HAS_GPU:
        vd["CPU-GPU"] = _check(mr_pw, mr_gpu_pw)
    all_pass = all_pass and all(vd.values())
    record("compute_pw (3D)", None, t_cpu, t_mcu, t_gpu, vd)

    # Deep verification: compute_pw (cross-compare, no MetPy reference)
    if HAS_METCU:
        verifier.compare("compute_pw", "CPU-MCU", mr_pw, mcu_pw,
                         phys_lo=0, phys_hi=100, phys_label="mm")
    if HAS_GPU:
        verifier.compare("compute_pw", "CPU-GPU", mr_pw, mr_gpu_pw,
                         phys_lo=0, phys_hi=100, phys_label="mm")
    # TC structural: PW in tropical range
    verifier.tc_pw_check(mr_pw, cx, cy, Rmax, DX)

    # ----------------------------------------------------------------
    # 5. WIND SPEED (CPU only -- no GPU kernel)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.wind_speed(u2_q, v2_q))
    ref_ws = _strip(mpcalc.wind_speed(u2_q, v2_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.wind_speed(d["u_2d"], d["v_2d"]))
    mr_ws = _strip(mrcalc.wind_speed(d["u_2d"], d["v_2d"]))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.wind_speed(d["u_2d"], d["v_2d"]),
            gpu=True)
        mcu_ws = _asnp(mcucalc.wind_speed(d["u_2d"], d["v_2d"]))

    vd = {"MP-CPU": _check(ref_ws, mr_ws)}
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_ws, mcu_ws)
    all_pass = all_pass and all(vd.values())
    record("wind_speed", t_mp, t_cpu, t_mcu, None, vd)

    # Deep verification: wind_speed
    verifier.compare("wind_speed", "MP-CPU", ref_ws, mr_ws,
                     phys_lo=0.0, phys_hi=60.0, phys_label="m/s")
    if HAS_METCU:
        verifier.compare("wind_speed", "MP-MCU", ref_ws, mcu_ws,
                         phys_lo=0.0, phys_hi=60.0, phys_label="m/s")
    # TC structural: cyclonic pattern
    verifier.tc_wind_speed_check(ref_ws, d["u_2d"], d["v_2d"], cx, cy, Rmax, DX)

    # ----------------------------------------------------------------
    # 6. DIVERGENCE (CPU only -- no GPU kernel)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.divergence(u2_q, v2_q, dx=dx_q, dy=dy_q))
    ref_div = _strip(mpcalc.divergence(u2_q, v2_q, dx=dx_q, dy=dy_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.divergence(d["u_2d"], d["v_2d"], dx=DX, dy=DY))
    mr_div = _strip(mrcalc.divergence(d["u_2d"], d["v_2d"], dx=DX, dy=DY))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.divergence(d["u_2d"], d["v_2d"], dx=DX, dy=DY),
            gpu=True)
        mcu_div = _asnp(mcucalc.divergence(d["u_2d"], d["v_2d"], dx=DX, dy=DY))

    vd = {"MP-CPU": _check(ref_div, mr_div)}
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_div, mcu_div)
    all_pass = all_pass and all(vd.values())
    record("divergence", t_mp, t_cpu, t_mcu, None, vd)

    # Deep verification: divergence
    verifier.compare("divergence", "MP-CPU", ref_div, mr_div,
                     phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    if HAS_METCU:
        verifier.compare("divergence", "MP-MCU", ref_div, mcu_div,
                         phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")

    # ================================================================
    # TIMING SUMMARY
    # ================================================================
    print()
    print("=" * 110)
    print("  TIMING SUMMARY")
    print("=" * 110)
    print(f"  {'Function':{COL_W['name']}s}"
          f"  {'MetPy':>{COL_W['t']}s}"
          f"  {'Rust CPU':>{COL_W['t']}s}"
          f"  {'met-cu':>{COL_W['t']}s}"
          f"  {'Rust GPU':>{COL_W['t']}s}"
          f"  {'Py/Rust':>{COL_W['r']}s}"
          f"  {'Rust/GPU':>{COL_W['r']}s}"
          f"  Verify")
    print("  " + "-" * (COL_W["name"] + 4 * COL_W["t"] + 2 * COL_W["r"] + 18))
    for name, t_mp, t_cpu, t_mcu, t_gpu, ok, vd in rows:
        mark = "PASS" if ok else "FAIL"
        fails = [k for k, v in vd.items() if not v]
        detail = f"  ({', '.join(fails)})" if fails else ""
        print(f"  {name:{COL_W['name']}s}"
              f"  {fmt(t_mp):>{COL_W['t']}s}"
              f"  {fmt(t_cpu):>{COL_W['t']}s}"
              f"  {fmt(t_mcu):>{COL_W['t']}s}"
              f"  {fmt(t_gpu):>{COL_W['t']}s}"
              f"  {spd(t_mp, t_cpu):>{COL_W['r']}s}"
              f"  {spd(t_cpu, t_gpu):>{COL_W['r']}s}"
              f"  [{mark}]{detail}")

    # Totals
    tot_mp = sum(t for _, t, _, _, _, _, _ in rows if t is not None)
    tot_cpu = sum(t for _, _, t, _, _, _, _ in rows if t is not None)
    tot_mcu = sum(t for _, _, _, t, _, _, _ in rows if t is not None)
    tot_gpu = sum(t for _, _, _, _, t, _, _ in rows if t is not None)
    print("  " + "-" * (COL_W["name"] + 4 * COL_W["t"] + 2 * COL_W["r"] + 18))
    print(f"  {'TOTAL':{COL_W['name']}s}"
          f"  {fmt(tot_mp):>{COL_W['t']}s}"
          f"  {fmt(tot_cpu):>{COL_W['t']}s}"
          f"  {fmt(tot_mcu):>{COL_W['t']}s}"
          f"  {fmt(tot_gpu):>{COL_W['t']}s}"
          f"  {spd(tot_mp, tot_cpu):>{COL_W['r']}s}"
          f"  {spd(tot_cpu, tot_gpu):>{COL_W['r']}s}")

    # ================================================================
    # DEEP VERIFICATION REPORT
    # ================================================================
    verifier.print_detailed_report()

    # Final verdict combines both timing-pass and deep-pass
    deep_ok = verifier.all_pass
    combined = all_pass and deep_ok
    print()
    status = "ALL PASS" if combined else "SOME FAILURES"
    print(f"  Quick verify (allclose): {'PASS' if all_pass else 'FAIL'}")
    print(f"  Deep verification:       {'PASS' if deep_ok else 'FAIL'}")
    print(f"  Combined:                {status}")
    print("=" * 110)

    return 0 if combined else 1


if __name__ == "__main__":
    sys.exit(main())
