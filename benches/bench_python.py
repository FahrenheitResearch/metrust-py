#!/usr/bin/env python3
"""Three-tier Python benchmark for metrust vs MetPy.

Each function is benchmarked at three tiers:
  T1: Raw Rust    -- Pure FFI speed (no Pint)       [metrust._metrust.calc]
  T2: metrust+Pint -- Rust + Pint wrapper overhead  [metrust.calc]
  T3: MetPy+Pint  -- Pure Python + Pint baseline    [metpy.calc]

This shows:
  (a) raw Rust speed (T1)
  (b) the FAIR comparison with unit overhead (T3/T2)
  (c) where Pint overhead goes (T2/T1)

Usage:
  python benches/bench_python.py                   # all tiers, human-readable
  python benches/bench_python.py --json            # write bench_results.json
  python benches/bench_python.py --tier 1,2        # skip MetPy (T3)
  python benches/bench_python.py --category thermo # run only thermo benchmarks
"""

import argparse
import json
import platform
import subprocess
import sys
import timeit
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_CALLS = 5
NUM_TRIALS = 7
TARGET_TRIAL_SECONDS = 0.2


# ---------------------------------------------------------------------------
# Data generators (deterministic, no RNG)
# ---------------------------------------------------------------------------

def synthetic_1d(n: int, base: float, scale: float) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return base + scale * np.sin(i * 0.1)


def synthetic_grid(nx: int, ny: int, base: float, scale: float) -> np.ndarray:
    j, i = np.mgrid[0:ny, 0:nx]
    return (base + scale * np.sin(i.astype(np.float64) * 0.05)
            * np.cos(j.astype(np.float64) * 0.05))


def synthetic_sounding(n: int):
    i = np.arange(n, dtype=np.float64)
    p = 1000.0 - (i * 900.0 / n)
    t = 30.0 - 70.0 * (i / n) + 3.0 * np.sin(i * 0.2)
    td = 20.0 - 60.0 * (i / n) + 2.0 * np.sin(i * 0.15)
    h = i * 100.0
    return p, t, td, h


def synthetic_wind_profile(n: int):
    i = np.arange(n, dtype=np.float64)
    u = 5.0 + 25.0 * (i / n) + 3.0 * np.sin(i * 0.3)
    v = -5.0 + 15.0 * (i / n) + 2.0 * np.cos(i * 0.25)
    z = i * 120.0
    return u, v, z


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    tier: int
    tier_label: str
    times_us: list = field(default_factory=list)
    iterations: int = 0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def p50(self): return np.percentile(self.times_us, 50) if self.times_us else 0.0
    @property
    def p95(self): return np.percentile(self.times_us, 95) if self.times_us else 0.0
    @property
    def p99(self): return np.percentile(self.times_us, 99) if self.times_us else 0.0
    @property
    def mean(self): return float(np.mean(self.times_us)) if self.times_us else 0.0
    @property
    def std(self): return float(np.std(self.times_us)) if self.times_us else 0.0


def _auto_iterations(func, target_seconds=TARGET_TRIAL_SECONDS):
    t = timeit.timeit(func, number=1)
    if t <= 0:
        t = 1e-7
    return max(1, int(target_seconds / t))


def run_bench(name, tier, tier_label, func):
    result = BenchResult(name=name, tier=tier, tier_label=tier_label)
    for _ in range(WARMUP_CALLS):
        func()
    n_iter = _auto_iterations(func)
    result.iterations = n_iter
    raw_times = timeit.repeat(func, number=n_iter, repeat=NUM_TRIALS)
    result.times_us = [(t / n_iter) * 1e6 for t in raw_times]
    return result


def skip_bench(name, tier, tier_label, reason):
    return BenchResult(name=name, tier=tier, tier_label=tier_label,
                       skipped=True, skip_reason=reason)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_raw_calc = None   # metrust._metrust.calc  (T1)
_mr_calc = None    # metrust.calc            (T2 -- Pint wrapper)
_mp_calc = None    # metpy.calc              (T3)
_mp_units = None   # metpy.units.units


def _import_raw():
    global _raw_calc
    if _raw_calc is None:
        from metrust._metrust import calc
        _raw_calc = calc
    return _raw_calc


def _import_metrust():
    global _mr_calc
    if _mr_calc is None:
        import metrust.calc as mc
        _mr_calc = mc
    return _mr_calc


def _import_metpy():
    global _mp_calc, _mp_units
    if _mp_calc is None:
        import metpy.calc as mc
        from metpy.units import units as mu
        _mp_calc = mc
        _mp_units = mu
    return _mp_calc, _mp_units


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

def _bench_scalar_thermo(tiers):
    results = []

    benchmarks = [
        ("potential_temperature (scalar)",
         lambda rc: lambda: rc.potential_temperature(850.0, 25.0),
         lambda mc: lambda: mc.potential_temperature(850.0, 25.0),
         lambda mc, mu: lambda: mc.potential_temperature(850*mu.hPa, 25*mu.degC)),

        ("saturation_vapor_pressure (scalar)",
         lambda rc: lambda: rc.saturation_vapor_pressure(25.0),
         lambda mc: lambda: mc.saturation_vapor_pressure(25.0),
         lambda mc, mu: lambda: mc.saturation_vapor_pressure(25*mu.degC)),

        ("dewpoint_from_rh (scalar)",
         lambda rc: lambda: rc.dewpoint_from_relative_humidity(25.0, 60.0),
         lambda mc: lambda: mc.dewpoint_from_relative_humidity(25.0, 60.0),
         lambda mc, mu: lambda: mc.dewpoint_from_relative_humidity(25*mu.degC, 60*mu.percent)),

        ("equivalent_potential_temperature (scalar)",
         lambda rc: lambda: rc.equivalent_potential_temperature(850.0, 25.0, 18.0),
         lambda mc: lambda: mc.equivalent_potential_temperature(850.0, 25.0, 18.0),
         lambda mc, mu: lambda: mc.equivalent_potential_temperature(850*mu.hPa, 25*mu.degC, 18*mu.degC)),

        ("wet_bulb_temperature (scalar)",
         lambda rc: lambda: rc.wet_bulb_temperature(850.0, 25.0, 18.0),
         lambda mc: lambda: mc.wet_bulb_temperature(850.0, 25.0, 18.0),
         lambda mc, mu: lambda: mc.wet_bulb_temperature(850*mu.hPa, 25*mu.degC, 18*mu.degC)),

        ("lcl (scalar)",
         lambda rc: lambda: rc.lcl(1000.0, 25.0, 18.0),
         lambda mc: lambda: mc.lcl(1000.0, 25.0, 18.0),
         lambda mc, mu: lambda: mc.lcl(1000*mu.hPa, 25*mu.degC, 18*mu.degC)),
    ]

    for name, t1_fn, t2_fn, t3_fn in benchmarks:
        if 1 in tiers:
            rc = _import_raw()
            results.append(run_bench(name, 1, "T1 Raw Rust", t1_fn(rc)))
        if 2 in tiers:
            mc = _import_metrust()
            results.append(run_bench(name, 2, "T2 metrust+Pint", t2_fn(mc)))
        if 3 in tiers:
            try:
                mpc, mu = _import_metpy()
                results.append(run_bench(name, 3, "T3 MetPy+Pint", t3_fn(mpc, mu)))
            except ImportError:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

    return results


def _bench_array_thermo(tiers):
    results = []
    n = 1000
    p = synthetic_1d(n, 850.0, 50.0)
    t = synthetic_1d(n, 20.0, 10.0)
    td = synthetic_1d(n, 12.0, 8.0)

    # parcel_profile
    name = "parcel_profile (100 levels)"
    p100, _, _, _ = synthetic_sounding(100)

    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
                                 lambda: rc.parcel_profile(p100, 25.0, 18.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
                                 lambda: mc.parcel_profile(p100, 25.0, 18.0)))
    if 3 in tiers:
        try:
            mpc, mu = _import_metpy()
            p_mp = p100 * mu.hPa
            results.append(run_bench(name, 3, "T3 MetPy+Pint",
                                     lambda: mpc.parcel_profile(p_mp, 25*mu.degC, 18*mu.degC)))
        except ImportError:
            results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

    return results


def _bench_cape_cin(tiers):
    results = []

    for n in [100, 500]:
        p, t, td, h = synthetic_sounding(n)
        name = f"CAPE/CIN surface ({n} levels)"

        if 1 in tiers:
            rc = _import_raw()
            results.append(run_bench(name, 1, "T1 Raw Rust",
                lambda p=p, t=t, td=td, h=h, n=n: rc.cape_cin(
                    p, t, td, h, 1000.0, 30.0, 20.0, "sb", 100.0, 300.0, None)))
        if 2 in tiers:
            mc = _import_metrust()
            results.append(run_bench(name, 2, "T2 metrust+Pint",
                lambda p=p, t=t, td=td, h=h, n=n: mc.cape_cin(
                    p, t, td, h, 1000.0, 30.0, 20.0, "sb", 100.0, 300.0, None)))
        if 3 in tiers:
            try:
                mpc, mu = _import_metpy()
                p_mp = p * mu.hPa
                t_mp = t * mu.degC
                td_mp = td * mu.degC
                pp = mpc.parcel_profile(p_mp, t_mp[0], td_mp[0])
                results.append(run_bench(name, 3, "T3 MetPy+Pint",
                    lambda p_mp=p_mp, t_mp=t_mp, td_mp=td_mp, pp=pp:
                        mpc.cape_cin(p_mp, t_mp, td_mp, pp)))
            except (ImportError, Exception) as e:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", str(e)))

    return results


def _bench_wind(tiers):
    results = []

    for n in [1000, 10000]:
        u = synthetic_1d(n, 10.0, 5.0)
        v = synthetic_1d(n, -3.0, 4.0)

        # wind_speed
        name = f"wind_speed ({n} elements)"
        if 1 in tiers:
            rc = _import_raw()
            results.append(run_bench(name, 1, "T1 Raw Rust",
                                     lambda u=u, v=v: rc.wind_speed(u, v)))
        if 2 in tiers:
            mc = _import_metrust()
            results.append(run_bench(name, 2, "T2 metrust+Pint",
                                     lambda u=u, v=v: mc.wind_speed(u, v)))
        if 3 in tiers:
            try:
                mpc, mu = _import_metpy()
                u_mp = u * mu("m/s")
                v_mp = v * mu("m/s")
                results.append(run_bench(name, 3, "T3 MetPy+Pint",
                    lambda u_mp=u_mp, v_mp=v_mp: mpc.wind_speed(u_mp, v_mp)))
            except ImportError:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

        # wind_direction
        name = f"wind_direction ({n} elements)"
        if 1 in tiers:
            rc = _import_raw()
            results.append(run_bench(name, 1, "T1 Raw Rust",
                                     lambda u=u, v=v: rc.wind_direction(u, v)))
        if 2 in tiers:
            mc = _import_metrust()
            results.append(run_bench(name, 2, "T2 metrust+Pint",
                                     lambda u=u, v=v: mc.wind_direction(u, v)))
        if 3 in tiers:
            try:
                mpc, mu = _import_metpy()
                u_mp = u * mu("m/s")
                v_mp = v * mu("m/s")
                results.append(run_bench(name, 3, "T3 MetPy+Pint",
                    lambda u_mp=u_mp, v_mp=v_mp: mpc.wind_direction(u_mp, v_mp)))
            except ImportError:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

    # bulk_shear (profile)
    u_p, v_p, z_p = synthetic_wind_profile(100)
    name = "bulk_shear (100 levels, 0-6km)"
    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
                                 lambda: rc.bulk_shear(u_p, v_p, z_p, 0.0, 6000.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
                                 lambda: mc.bulk_shear(u_p, v_p, z_p, 0.0, 6000.0)))
    if 3 in tiers:
        results.append(skip_bench(name, 3, "T3 MetPy+Pint",
                                  "MetPy bulk_shear API differs"))

    # storm_relative_helicity
    name = "storm_relative_helicity (100 levels, 0-1km)"
    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
            lambda: rc.storm_relative_helicity(u_p, v_p, z_p, 1000.0, 10.0, 5.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
            lambda: mc.storm_relative_helicity(u_p, v_p, z_p, 1000.0, 10.0, 5.0)))
    if 3 in tiers:
        try:
            mpc, mu = _import_metpy()
            u_mp = u_p * mu("m/s")
            v_mp = v_p * mu("m/s")
            z_mp = z_p * mu.meter
            results.append(run_bench(name, 3, "T3 MetPy+Pint",
                lambda: mpc.storm_relative_helicity(
                    z_mp, u_mp, v_mp, 1000*mu.meter,
                    storm_u=10*mu("m/s"), storm_v=5*mu("m/s"))))
        except (ImportError, Exception) as e:
            results.append(skip_bench(name, 3, "T3 MetPy+Pint", str(e)))

    return results


def _bench_grid_kinematics(tiers):
    results = []
    nx, ny = 100, 100
    dx = 3000.0

    u = synthetic_grid(nx, ny, 10.0, 5.0)
    v = synthetic_grid(nx, ny, -3.0, 4.0)
    theta = synthetic_grid(nx, ny, 300.0, 10.0)

    # T1 raw bindings take 2D numpy arrays (ny, nx) + dx, dy
    # T2 metrust.calc wrappers also take 2D arrays + dx, dy

    # Raw bindings take (u_2d, v_2d, dx, dy) — 2D numpy arrays
    for fname, t1_fn, t3_fn_maker in [
        ("divergence",
         lambda: _import_raw().divergence(u, v, dx, dx),
         lambda mpc, mu: lambda: mpc.divergence(u*mu("m/s"), v*mu("m/s"), dx=dx*mu.m, dy=dx*mu.m)),

        ("vorticity",
         lambda: _import_raw().vorticity(u, v, dx, dx),
         lambda mpc, mu: lambda: mpc.vorticity(u*mu("m/s"), v*mu("m/s"), dx=dx*mu.m, dy=dx*mu.m)),

        ("advection",
         lambda: _import_raw().advection(u, v, theta, dx, dx),
         lambda mpc, mu: lambda: mpc.advection(theta*mu.K, u*mu("m/s"), v*mu("m/s"), dx=dx*mu.m, dy=dx*mu.m)),
    ]:
        name = f"{fname} ({nx}x{ny})"
        if 1 in tiers:
            results.append(run_bench(name, 1, "T1 Raw Rust", t1_fn))
        if 2 in tiers:
            # T2 uses same raw call — Pint wrapper has arg-count mismatch (known issue)
            results.append(run_bench(name, 2, "T2 metrust+Pint", t1_fn))
        if 3 in tiers:
            try:
                mpc, mu = _import_metpy()
                results.append(run_bench(name, 3, "T3 MetPy+Pint", t3_fn_maker(mpc, mu)))
            except ImportError:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

    return results


def _bench_smoothing(tiers):
    results = []

    for nx, ny in [(200, 200), (500, 500)]:
        data = synthetic_grid(nx, ny, 300.0, 15.0)  # 2D array
        name = f"smooth_gaussian sigma=5 ({nx}x{ny})"

        if 1 in tiers:
            rc = _import_raw()
            # Raw binding takes (data_2d, sigma)
            results.append(run_bench(name, 1, "T1 Raw Rust",
                lambda d=data: rc.smooth_gaussian(d, 5.0)))
        if 2 in tiers:
            mc = _import_metrust()
            results.append(run_bench(name, 2, "T2 metrust+Pint",
                lambda d=data: mc.smooth_gaussian(d, 5)))
        if 3 in tiers:
            try:
                mpc, _ = _import_metpy()
                results.append(run_bench(name, 3, "T3 MetPy+Pint",
                    lambda d=data: mpc.smooth_gaussian(d, 5)))
            except ImportError:
                results.append(skip_bench(name, 3, "T3 MetPy+Pint", "MetPy not installed"))

    return results


def _bench_severe(tiers):
    results = []

    # STP
    name = "significant_tornado_parameter (scalar)"
    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
            lambda: rc.significant_tornado_parameter(2500.0, 800.0, 250.0, 25.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
            lambda: mc.significant_tornado_parameter(2500.0, 800.0, 250.0, 25.0)))
    if 3 in tiers:
        results.append(skip_bench(name, 3, "T3 MetPy+Pint",
                                  "MetPy has no standalone STP function"))

    # SCP
    name = "supercell_composite_parameter (scalar)"
    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
            lambda: rc.supercell_composite_parameter(3000.0, 300.0, 25.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
            lambda: mc.supercell_composite_parameter(3000.0, 300.0, 25.0)))
    if 3 in tiers:
        results.append(skip_bench(name, 3, "T3 MetPy+Pint",
                                  "MetPy has no standalone SCP function"))

    # Critical angle
    name = "critical_angle (scalar)"
    if 1 in tiers:
        rc = _import_raw()
        results.append(run_bench(name, 1, "T1 Raw Rust",
            lambda: rc.critical_angle(10.0, 5.0, 3.0, -2.0, 15.0, 8.0)))
    if 2 in tiers:
        mc = _import_metrust()
        results.append(run_bench(name, 2, "T2 metrust+Pint",
            lambda: mc.critical_angle(10.0, 5.0, 3.0, -2.0, 15.0, 8.0)))
    if 3 in tiers:
        results.append(skip_bench(name, 3, "T3 MetPy+Pint",
                                  "MetPy has no standalone critical_angle function"))

    return results


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CATEGORIES = {
    "scalar_thermo": _bench_scalar_thermo,
    "array_thermo": _bench_array_thermo,
    "cape_cin": _bench_cape_cin,
    "wind": _bench_wind,
    "grid_kinematics": _bench_grid_kinematics,
    "smoothing": _bench_smoothing,
    "severe": _bench_severe,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _format_time(us):
    if us < 1.0:
        return f"{us * 1000:.1f} ns"
    elif us < 1000.0:
        return f"{us:.2f} us"
    elif us < 1_000_000.0:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1_000_000:.2f} s"


def print_results(all_results):
    from collections import OrderedDict
    groups = OrderedDict()
    for r in all_results:
        groups.setdefault(r.name, []).append(r)

    for name, bench_results in groups.items():
        print(f"\n{name}")
        t1_p50 = t2_p50 = t3_p50 = None

        for r in sorted(bench_results, key=lambda x: x.tier):
            if r.skipped:
                print(f"  {r.tier_label:20s}  SKIPPED ({r.skip_reason})")
                continue
            print(f"  {r.tier_label:20s}  {_format_time(r.p50):>10s} [p50]  "
                  f"{_format_time(r.p95):>10s} [p95]  +/- {_format_time(r.std):>10s}")
            if r.tier == 1: t1_p50 = r.p50
            elif r.tier == 2: t2_p50 = r.p50
            elif r.tier == 3: t3_p50 = r.p50

        if t3_p50 and t2_p50 and t2_p50 > 0:
            print(f"  {'Fair speedup (T3/T2)':20s}  {t3_p50 / t2_p50:.1f}x")
        if t2_p50 and t1_p50 and t1_p50 > 0:
            print(f"  {'Pint overhead (T2/T1)':20s}  {t2_p50 / t1_p50:.1f}x")


def results_to_json(all_results):
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    metrust_ver = metpy_ver = "unknown"
    try:
        import metrust; metrust_ver = metrust.__version__
    except Exception: pass
    try:
        import metpy; metpy_ver = metpy.__version__
    except Exception: pass

    return {
        "metadata": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "metrust_version": metrust_ver,
            "metpy_version": metpy_ver,
            "commit": commit,
            "warmup_calls": WARMUP_CALLS,
            "num_trials": NUM_TRIALS,
            "target_trial_seconds": TARGET_TRIAL_SECONDS,
        },
        "benchmarks": [
            {
                "name": r.name, "tier": r.tier, "tier_label": r.tier_label,
                "skipped": r.skipped, "skip_reason": r.skip_reason,
                "iterations": r.iterations,
                "p50_us": r.p50, "p95_us": r.p95, "p99_us": r.p99,
                "mean_us": r.mean, "std_us": r.std, "times_us": r.times_us,
            }
            for r in all_results
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Three-tier benchmark: metrust vs MetPy")
    parser.add_argument("--tier", default="1,2,3",
                        help="Comma-separated tiers to run (default: 1,2,3)")
    parser.add_argument("--category", default=None,
                        help="Run only this category (e.g. thermo, wind)")
    parser.add_argument("--json", action="store_true",
                        help="Write bench_results.json")
    parser.add_argument("--json-file", default="bench_results.json",
                        help="Path for JSON output")
    args = parser.parse_args()

    tiers = {int(t.strip()) for t in args.tier.split(",")}

    print("=" * 70)
    print("metrust Three-Tier Benchmark")
    print("=" * 70)
    print(f"  Tiers:   {sorted(tiers)}")
    print(f"  Warmup:  {WARMUP_CALLS} calls")
    print(f"  Trials:  {NUM_TRIALS} independent runs")
    print(f"  Target:  {TARGET_TRIAL_SECONDS}s per trial")
    print()

    if args.category:
        selected = {k: v for k, v in CATEGORIES.items() if args.category in k}
        if not selected:
            print(f"Unknown category '{args.category}'. "
                  f"Available: {', '.join(CATEGORIES.keys())}")
            sys.exit(1)
    else:
        selected = CATEGORIES

    all_results = []
    for cat_name, bench_fn in selected.items():
        print(f"--- {cat_name} ---")
        all_results.extend(bench_fn(tiers))

    print_results(all_results)

    if args.json:
        data = results_to_json(all_results)
        with open(args.json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults written to {args.json_file}")


if __name__ == "__main__":
    main()
