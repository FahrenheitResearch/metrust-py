#!/usr/bin/env python
"""Benchmark 08: Real GFS Tropical Analysis (Western Pacific)

Scenario: Real GFS 0.25-degree data, tropical subdomain (0-30N, 100-150E).
33 isobaric levels (1000-1 hPa), ~121 x 201 horizontal grid.

Data source: C:\\Users\\drew\\metrust-py\\data\\gfs_0p25.grib2

Functions benchmarked
---------------------
  vorticity                           (GPU)
  equivalent_potential_temperature    (GPU)
  potential_temperature               (GPU)
  compute_pw                          (GPU)  -- 3-D precipitable water
  wind_speed                          (CPU)
  divergence                          (CPU)

Backends: MetPy (Pint), metrust CPU, met-cu direct, metrust GPU.
MetPy does NOT have a grid compute_pw -- that row uses 3 backends only.

Verification : deep correctness audit per function --
  mean diff, max abs diff, RMSE, 99th pct, relative RMSE%, NaN/Inf audit,
  physical plausibility bounds, Pearson r, % points > 1%/0.1% relative error,
  histogram of diffs, and tropical atmosphere structural checks.

Timing       : perf_counter, cupy sync, 1 warmup + 3 timed, median
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
import xarray as xr
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
# GFS data path
# ============================================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gfs_0p25.grib2")
DATA_FILE = os.path.normpath(DATA_FILE)

# Tropical subdomain: Western Pacific 0-30N, 100-150E
LAT_SLICE = slice(30, 0)       # GFS lats run 90 to -90
LON_SLICE = slice(100, 150)

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
# Load real GFS data
# ============================================================================
def load_gfs_tropical():
    """Load GFS isobaric data and extract tropical Western Pacific subdomain.

    Returns a dict with all arrays needed by the six benchmarked functions.
    """
    print(f"  Loading GFS data from: {DATA_FILE}")

    ds = xr.open_dataset(
        DATA_FILE, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )

    # Extract tropical subdomain
    sub = ds.sel(latitude=LAT_SLICE, longitude=LON_SLICE)

    p_levels_hPa = sub.isobaricInhPa.values.astype(np.float64)
    NZ = len(p_levels_hPa)
    lats = sub.latitude.values
    lons = sub.longitude.values
    NY = len(lats)
    NX = len(lons)

    # Grid spacing (metres) -- 0.25 deg at ~15N
    DX = 0.25 * 111_320.0 * np.cos(np.radians(15.0))  # ~26.9 km
    DY = 0.25 * 110_540.0                              # ~27.6 km

    # 3-D fields: (nz, ny, nx), float64
    T_K = sub["t"].values.astype(np.float64)        # temperature in K
    q   = sub["q"].values.astype(np.float64)        # specific humidity kg/kg
    u   = sub["u"].values.astype(np.float64)        # u-wind m/s
    v   = sub["v"].values.astype(np.float64)        # v-wind m/s

    # Derived fields
    T_C = T_K - 273.15                              # Celsius for metrust/met-cu

    # Mixing ratio from specific humidity: w = q / (1 - q)
    w_3d = q / (1.0 - q)

    # Dewpoint from specific humidity and pressure
    # e = q * p / (0.622 + q * 0.378)
    p3_Pa = np.broadcast_to(
        p_levels_hPa[:, None, None] * 100.0, (NZ, NY, NX)
    ).copy().astype(np.float64)

    e = q * p3_Pa / (0.622 + q * 0.378)   # vapour pressure (Pa)
    e_hPa = e / 100.0
    # Clamp e_hPa > 0 to avoid log domain errors
    e_hPa = np.clip(e_hPa, 1e-6, None)
    # Magnus formula: Td (Celsius)
    Td_C = 243.04 * np.log(e_hPa / 6.1078) / (17.27 - np.log(e_hPa / 6.1078))

    ds.close()

    return dict(
        NX=NX, NY=NY, NZ=NZ, DX=DX, DY=DY,
        lats=lats, lons=lons,
        p_levels_hPa=p_levels_hPa,
        u_3d=u, v_3d=v,          # (nz, ny, nx) m/s
        T_K=T_K,                  # (nz, ny, nx) Kelvin
        T_C=T_C,                  # (nz, ny, nx) Celsius
        Td_C=Td_C,               # (nz, ny, nx) Celsius
        q=q,                      # (nz, ny, nx) specific humidity
        w_3d=w_3d,                # (nz, ny, nx) mixing ratio kg/kg
        p3_Pa=p3_Pa,              # (nz, ny, nx) pressure Pa
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
        self.reports = []
        self.trop_checks = []
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
            test_full = np.asarray(test, dtype=np.float64)
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
    # Tropical atmosphere structural checks
    # ------------------------------------------------------------------
    def trop_theta_e_check(self, ept_2d, label="MetPy"):
        """Verify theta-e is in realistic tropical range at 850 hPa."""
        e = np.asarray(ept_2d, dtype=np.float64)
        check = {"name": f"Tropical theta-e structure ({label})", "pass": True, "notes": []}

        ept_mean = float(np.nanmean(e))
        ept_min = float(np.nanmin(e))
        ept_max = float(np.nanmax(e))
        check["ept_mean"] = ept_mean
        check["ept_range"] = (ept_min, ept_max)
        check["notes"].append(
            f"Theta-e 850 hPa: mean={ept_mean:.1f} K, "
            f"range=[{ept_min:.1f}, {ept_max:.1f}] K")

        # Tropical 850 hPa theta-e should be 320-380 K
        if ept_mean < 310 or ept_mean > 390:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: mean theta-e {ept_mean:.1f} K outside tropical range [310, 390]")
        else:
            check["notes"].append(
                f"Mean theta-e {ept_mean:.1f} K in tropical range [310, 390]: OK")

        # Should have high values (>340 K) somewhere in the domain
        pct_warm = float(np.count_nonzero(e > 340) / e.size * 100)
        check["pct_gt_340K"] = pct_warm
        check["notes"].append(f"{pct_warm:.1f}% of points have theta-e > 340 K")

        self.trop_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def trop_pw_check(self, pw_2d, label="metrust"):
        """Verify PW is in realistic tropical range."""
        pw = np.asarray(pw_2d, dtype=np.float64)
        check = {"name": f"Tropical PW structure ({label})", "pass": True, "notes": []}

        pw_mean = float(np.nanmean(pw))
        pw_min = float(np.nanmin(pw))
        pw_max = float(np.nanmax(pw))
        check["pw_mean"] = pw_mean
        check["pw_range"] = (pw_min, pw_max)
        check["notes"].append(
            f"PW: mean={pw_mean:.1f} mm, range=[{pw_min:.1f}, {pw_max:.1f}] mm")

        # Tropical PW should be 20-80 mm typically
        if pw_mean < 10 or pw_mean > 90:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: mean PW {pw_mean:.1f} mm outside tropical range [10, 90]")
        else:
            check["notes"].append(
                f"Mean PW {pw_mean:.1f} mm in tropical range [10, 90]: OK")

        # Must be non-negative
        if pw_min < -0.1:
            check["pass"] = False
            check["notes"].append(f"FAIL: PW min {pw_min:.2f} mm is negative")

        # Sanity cap
        if pw_max > 120:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: PW max {pw_max:.1f} mm exceeds 120 mm sanity cap")

        self.trop_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def trop_vorticity_check(self, vort_2d, label="MetPy"):
        """Verify vorticity is in reasonable range for tropical low-level flow."""
        v = np.asarray(vort_2d, dtype=np.float64)
        check = {"name": f"Tropical vorticity structure ({label})", "pass": True, "notes": []}

        vort_mean = float(np.nanmean(v))
        vort_min = float(np.nanmin(v))
        vort_max = float(np.nanmax(v))
        vort_absmax = max(abs(vort_min), abs(vort_max))
        check["vort_range"] = (vort_min, vort_max)
        check["notes"].append(
            f"Vorticity: mean={vort_mean:.4e}, range=[{vort_min:.4e}, {vort_max:.4e}] /s")

        # Typical tropical 850 hPa vorticity: +/- 1e-4 to 1e-3 /s
        if vort_absmax > 5e-3:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: max |vorticity| {vort_absmax:.4e} exceeds 5e-3 /s")
        else:
            check["notes"].append(
                f"Max |vorticity| {vort_absmax:.4e} within reasonable bounds: OK")

        # Should have both positive and negative values in the tropics
        has_pos = bool(np.any(v > 0))
        has_neg = bool(np.any(v < 0))
        if has_pos and has_neg:
            check["notes"].append("Both positive and negative vorticity present: OK")
        else:
            check["notes"].append("WARNING: vorticity is all one sign")

        self.trop_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def trop_wind_speed_check(self, ws_2d, label="MetPy"):
        """Verify wind speed is in realistic tropical range at 850 hPa."""
        ws = np.asarray(ws_2d, dtype=np.float64)
        check = {"name": f"Tropical wind speed structure ({label})", "pass": True, "notes": []}

        ws_mean = float(np.nanmean(ws))
        ws_min = float(np.nanmin(ws))
        ws_max = float(np.nanmax(ws))
        check["ws_range"] = (ws_min, ws_max)
        check["notes"].append(
            f"Wind speed 850 hPa: mean={ws_mean:.1f}, "
            f"range=[{ws_min:.1f}, {ws_max:.1f}] m/s")

        # Tropical 850 hPa winds: typically 0-30 m/s, can reach 40 in TCs
        if ws_max > 50.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: max wind speed {ws_max:.1f} m/s exceeds 50 m/s")
        if ws_min < 0.0:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: min wind speed {ws_min:.2f} m/s is negative")

        if ws_min >= 0.0 and ws_max <= 50.0:
            check["notes"].append(
                f"Wind speed range [{ws_min:.1f}, {ws_max:.1f}] m/s: OK")

        self.trop_checks.append(check)
        self.all_pass = self.all_pass and check["pass"]
        return check["pass"]

    def trop_potential_temp_check(self, pt_2d, label="MetPy"):
        """Verify potential temperature is in realistic range at 850 hPa."""
        pt = np.asarray(pt_2d, dtype=np.float64)
        check = {"name": f"Tropical potential temp ({label})", "pass": True, "notes": []}

        pt_mean = float(np.nanmean(pt))
        pt_min = float(np.nanmin(pt))
        pt_max = float(np.nanmax(pt))
        check["pt_range"] = (pt_min, pt_max)
        check["notes"].append(
            f"Theta 850 hPa: mean={pt_mean:.1f} K, "
            f"range=[{pt_min:.1f}, {pt_max:.1f}] K")

        # Tropical 850 hPa theta: ~295-315 K
        if pt_mean < 285 or pt_mean > 325:
            check["pass"] = False
            check["notes"].append(
                f"FAIL: mean theta {pt_mean:.1f} K outside range [285, 325]")
        else:
            check["notes"].append(
                f"Mean theta {pt_mean:.1f} K in expected range [285, 325]: OK")

        self.trop_checks.append(check)
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

        # Tropical structural checks
        if self.trop_checks:
            print()
            print(f"  --- Tropical atmosphere structural checks ---")
            for c in self.trop_checks:
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
        tc_pass = sum(1 for c in self.trop_checks if c["pass"])
        tc_fail = sum(1 for c in self.trop_checks if not c["pass"])

        print()
        print(f"  Statistical comparisons:     {n_pass} PASS, {n_fail} FAIL "
              f"(of {n_pass + n_fail})")
        print(f"  Tropical structural checks:  {tc_pass} PASS, {tc_fail} FAIL "
              f"(of {tc_pass + tc_fail})")
        overall = "ALL PASS" if self.all_pass else "SOME FAILURES"
        print(f"  Overall verification:        {overall}")
        print("=" * W)


# ============================================================================
# Main benchmark
# ============================================================================
def main():
    print("=" * 110)
    print("  BENCHMARK 08 : Real GFS Tropical Analysis (Western Pacific 0-30N, 100-150E)")
    print(f"  Data: GFS 0.25 deg, 33 levels, tropical subdomain ~121 x 201")
    print(f"  GPU : {GPU_NAME if HAS_GPU else 'not available'}  |  met-cu: {'yes' if HAS_METCU else 'no'}")
    print("=" * 110)

    # -- load data --
    print()
    t0 = time.perf_counter()
    d = load_gfs_tropical()
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Loaded in {load_ms:.0f} ms  |  "
          f"shape: {d['NZ']} levels x {d['NY']} lat x {d['NX']} lon")
    print(f"  Pressure levels: {d['p_levels_hPa'][0]:.0f} - {d['p_levels_hPa'][-1]:.0f} hPa "
          f"({d['NZ']} levels)")
    print(f"  dx={d['DX']:.0f} m  dy={d['DY']:.0f} m")
    print()

    NX, NY, NZ = d["NX"], d["NY"], d["NZ"]
    DX, DY = d["DX"], d["DY"]

    verifier = DeepVerifier()

    # -- 850 hPa slice for 2-D tests --
    # Find 850 hPa index
    i850 = int(np.argmin(np.abs(d["p_levels_hPa"] - 850.0)))
    p850_hPa = d["p_levels_hPa"][i850]
    print(f"  Using level index {i850} = {p850_hPa:.0f} hPa for 2-D tests")

    u850 = d["u_3d"][i850].copy()       # (ny, nx)
    v850 = d["v_3d"][i850].copy()
    T850_C = d["T_C"][i850].copy()      # Celsius
    Td850_C = d["Td_C"][i850].copy()    # Celsius

    print(f"  T850: {T850_C.min():.1f} to {T850_C.max():.1f} C")
    print(f"  Td850: {Td850_C.min():.1f} to {Td850_C.max():.1f} C")
    print(f"  U850: {u850.min():.1f} to {u850.max():.1f} m/s")
    print(f"  V850: {v850.min():.1f} to {v850.max():.1f} m/s")

    # MetPy Pint wrappers
    u850_q = u850 * units("m/s")
    v850_q = v850 * units("m/s")
    dx_q = DX * units.m
    dy_q = DY * units.m
    T850_q = T850_C * units.degC
    Td850_q = Td850_C * units.degC
    p850_q = p850_hPa * units.hPa

    # met-cu needs pressure broadcast to array shape
    p850_arr = np.full_like(T850_C, p850_hPa)

    # ================================================================
    print()
    header()

    all_pass = True

    # ----------------------------------------------------------------
    # 1. VORTICITY (GPU)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.vorticity(u850_q, v850_q, dx=dx_q, dy=dy_q))
    ref_vort = _strip(mpcalc.vorticity(u850_q, v850_q, dx=dx_q, dy=dy_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.vorticity(u850, v850, dx=DX, dy=DY))
    mr_vort = _strip(mrcalc.vorticity(u850, v850, dx=DX, dy=DY))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.vorticity(u850, v850, dx=DX, dy=DY),
            gpu=True)
        mcu_vort = _asnp(mcucalc.vorticity(u850, v850, dx=DX, dy=DY))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.vorticity(u850, v850, dx=DX, dy=DY),
            gpu=True)
        mr_gpu_vort = _strip(mrcalc.vorticity(u850, v850, dx=DX, dy=DY))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_vort, mr_vort)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_vort, mr_gpu_vort)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_vort, mcu_vort)
    all_pass = all_pass and all(vd.values())
    record("vorticity", t_mp, t_cpu, t_mcu, t_gpu, vd)

    verifier.compare("vorticity", "MP-CPU", ref_vort, mr_vort,
                     phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    if HAS_GPU:
        verifier.compare("vorticity", "MP-GPU", ref_vort, mr_gpu_vort,
                         phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    if HAS_METCU:
        verifier.compare("vorticity", "MP-MCU", ref_vort, mcu_vort,
                         phys_lo=-5e-3, phys_hi=5e-3, phys_label="/s")
    verifier.trop_vorticity_check(ref_vort, label="MetPy")

    # ----------------------------------------------------------------
    # 2. EQUIVALENT POTENTIAL TEMPERATURE (GPU)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.equivalent_potential_temperature(
        p850_q, T850_q, Td850_q))
    ref_ept = _strip(mpcalc.equivalent_potential_temperature(
        p850_q, T850_q, Td850_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.equivalent_potential_temperature(
        p850_hPa, T850_C, Td850_C))
    mr_ept = _strip(mrcalc.equivalent_potential_temperature(
        p850_hPa, T850_C, Td850_C))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.equivalent_potential_temperature(p850_arr, T850_C, Td850_C),
            gpu=True)
        mcu_ept = _asnp(mcucalc.equivalent_potential_temperature(p850_arr, T850_C, Td850_C))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.equivalent_potential_temperature(p850_hPa, T850_C, Td850_C),
            gpu=True)
        mr_gpu_ept = _strip(mrcalc.equivalent_potential_temperature(
            p850_hPa, T850_C, Td850_C))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_ept, mr_ept)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_ept, mr_gpu_ept)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_ept, mcu_ept)
    all_pass = all_pass and all(vd.values())
    record("equiv_potential_temp", t_mp, t_cpu, t_mcu, t_gpu, vd)

    verifier.compare("equiv_potential_temp", "MP-CPU", ref_ept, mr_ept,
                     phys_lo=280, phys_hi=400, phys_label="K")
    if HAS_GPU:
        verifier.compare("equiv_potential_temp", "MP-GPU", ref_ept, mr_gpu_ept,
                         phys_lo=280, phys_hi=400, phys_label="K")
    if HAS_METCU:
        verifier.compare("equiv_potential_temp", "MP-MCU", ref_ept, mcu_ept,
                         phys_lo=280, phys_hi=400, phys_label="K")
    verifier.trop_theta_e_check(ref_ept, label="MetPy")

    # ----------------------------------------------------------------
    # 3. POTENTIAL TEMPERATURE (GPU)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.potential_temperature(p850_q, T850_q))
    ref_pt = _strip(mpcalc.potential_temperature(p850_q, T850_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.potential_temperature(p850_hPa, T850_C))
    mr_pt = _strip(mrcalc.potential_temperature(p850_hPa, T850_C))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.potential_temperature(p850_arr, T850_C),
            gpu=True)
        mcu_pt = _asnp(mcucalc.potential_temperature(p850_arr, T850_C))

    t_gpu = None
    if HAS_GPU:
        mrcalc.set_backend("gpu")
        t_gpu = _time_func(
            lambda: mrcalc.potential_temperature(p850_hPa, T850_C),
            gpu=True)
        mr_gpu_pt = _strip(mrcalc.potential_temperature(p850_hPa, T850_C))
        mrcalc.set_backend("cpu")

    vd = {"MP-CPU": _check(ref_pt, mr_pt)}
    if HAS_GPU:
        vd["MP-GPU"] = _check(ref_pt, mr_gpu_pt)
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_pt, mcu_pt)
    all_pass = all_pass and all(vd.values())
    record("potential_temperature", t_mp, t_cpu, t_mcu, t_gpu, vd)

    verifier.compare("potential_temperature", "MP-CPU", ref_pt, mr_pt,
                     phys_lo=280, phys_hi=330, phys_label="K")
    if HAS_GPU:
        verifier.compare("potential_temperature", "MP-GPU", ref_pt, mr_gpu_pt,
                         phys_lo=280, phys_hi=330, phys_label="K")
    if HAS_METCU:
        verifier.compare("potential_temperature", "MP-MCU", ref_pt, mcu_pt,
                         phys_lo=280, phys_hi=330, phys_label="K")
    verifier.trop_potential_temp_check(ref_pt, label="MetPy")

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

    if HAS_METCU:
        verifier.compare("compute_pw", "CPU-MCU", mr_pw, mcu_pw,
                         phys_lo=0, phys_hi=120, phys_label="mm")
    if HAS_GPU:
        verifier.compare("compute_pw", "CPU-GPU", mr_pw, mr_gpu_pw,
                         phys_lo=0, phys_hi=120, phys_label="mm")
    verifier.trop_pw_check(mr_pw, label="metrust CPU")

    # ----------------------------------------------------------------
    # 5. WIND SPEED (CPU only -- no GPU kernel)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.wind_speed(u850_q, v850_q))
    ref_ws = _strip(mpcalc.wind_speed(u850_q, v850_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.wind_speed(u850, v850))
    mr_ws = _strip(mrcalc.wind_speed(u850, v850))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.wind_speed(u850, v850),
            gpu=True)
        mcu_ws = _asnp(mcucalc.wind_speed(u850, v850))

    vd = {"MP-CPU": _check(ref_ws, mr_ws)}
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_ws, mcu_ws)
    all_pass = all_pass and all(vd.values())
    record("wind_speed", t_mp, t_cpu, t_mcu, None, vd)

    verifier.compare("wind_speed", "MP-CPU", ref_ws, mr_ws,
                     phys_lo=0.0, phys_hi=50.0, phys_label="m/s")
    if HAS_METCU:
        verifier.compare("wind_speed", "MP-MCU", ref_ws, mcu_ws,
                         phys_lo=0.0, phys_hi=50.0, phys_label="m/s")
    verifier.trop_wind_speed_check(ref_ws, label="MetPy")

    # ----------------------------------------------------------------
    # 6. DIVERGENCE (CPU only -- no GPU kernel)
    # ----------------------------------------------------------------
    t_mp = _time_func(lambda: mpcalc.divergence(u850_q, v850_q, dx=dx_q, dy=dy_q))
    ref_div = _strip(mpcalc.divergence(u850_q, v850_q, dx=dx_q, dy=dy_q))

    mrcalc.set_backend("cpu")
    t_cpu = _time_func(lambda: mrcalc.divergence(u850, v850, dx=DX, dy=DY))
    mr_div = _strip(mrcalc.divergence(u850, v850, dx=DX, dy=DY))

    t_mcu = None
    if HAS_METCU:
        t_mcu = _time_func(
            lambda: mcucalc.divergence(u850, v850, dx=DX, dy=DY),
            gpu=True)
        mcu_div = _asnp(mcucalc.divergence(u850, v850, dx=DX, dy=DY))

    vd = {"MP-CPU": _check(ref_div, mr_div)}
    if HAS_METCU:
        vd["MP-MCU"] = _check(ref_div, mcu_div)
    all_pass = all_pass and all(vd.values())
    record("divergence", t_mp, t_cpu, t_mcu, None, vd)

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
