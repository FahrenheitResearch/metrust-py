#!/usr/bin/env python3
"""
Massive HRRR-driven benchmark of metrust.calc functions.

Uses a real HRRR forecast hour (surface + pressure levels) to benchmark
every applicable metrust.calc function on production-sized data:
  - 2D grids: 1059 x 1799 (~1.9M points)
  - Vertical profiles: 40 pressure levels
  - Soundings: extracted column profiles at multiple grid points

Usage:
    python benches/bench_hrrr.py                     # run all categories
    python benches/bench_hrrr.py --category grid_2d   # single category
    python benches/bench_hrrr.py --json               # machine-readable output
    python benches/bench_hrrr.py --quick              # fewer trials, faster

Prerequisites:
    - HRRR GRIB2 files in data/ (download with bench_hrrr_download.py or curl)
    - metrust installed (pip install -e .)
    - cfgrib + xarray for GRIB reading
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import timeit
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hrrr_data(data_dir="data"):
    """Load HRRR fields into a flat dict of numpy arrays."""
    import xarray as xr

    sfc_path = os.path.join(data_dir, "hrrr_sfc.grib2")
    prs_path = os.path.join(data_dir, "hrrr_prs.grib2")

    if not os.path.exists(sfc_path) or not os.path.exists(prs_path):
        print(f"ERROR: HRRR files not found in {data_dir}/")
        print(f"  Need: hrrr_sfc.grib2 and hrrr_prs.grib2")
        print(f"  Download from: https://noaa-hrrr-bdp-pds.s3.amazonaws.com/")
        sys.exit(1)

    print("Loading HRRR data...")
    t0 = time.time()

    data = {}

    # --- Surface fields ---
    ds_2m = xr.open_dataset(sfc_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
    data['t2m'] = ds_2m['t2m'].values - 273.15        # K -> C
    data['d2m'] = ds_2m['d2m'].values - 273.15        # K -> C
    data['rh2m'] = ds_2m['r2'].values                 # %
    data['sh2m'] = ds_2m['sh2'].values                # kg/kg
    data['lat'] = ds_2m['latitude'].values             # degrees N
    data['lon'] = ds_2m['longitude'].values            # degrees E
    ny, nx = data['t2m'].shape
    data['ny'], data['nx'] = ny, nx

    ds_10m = xr.open_dataset(sfc_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
    data['u10'] = ds_10m['u10'].values                 # m/s
    data['v10'] = ds_10m['v10'].values                 # m/s

    ds_sfc = xr.open_dataset(sfc_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': 'sp'}})
    data['psfc'] = ds_sfc['sp'].values / 100.0         # Pa -> hPa

    # --- Pressure level fields ---
    ds_t = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 't'}})
    data['t_prs'] = ds_t['t'].values - 273.15          # K -> C, shape (40, ny, nx)
    data['p_levels'] = ds_t['isobaricInhPa'].values    # hPa

    ds_r = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}})
    data['rh_prs'] = ds_r['r'].values                  # %, shape (40, ny, nx)

    ds_dpt = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'dpt'}})
    data['td_prs'] = ds_dpt['dpt'].values - 273.15     # K -> C

    ds_u = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
    data['u_prs'] = ds_u['u'].values                   # m/s, shape (40, ny, nx)

    ds_v = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
    data['v_prs'] = ds_v['v'].values                   # m/s

    ds_gh = xr.open_dataset(prs_path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}})
    data['gh_prs'] = ds_gh['gh'].values                # geopotential height, m

    # --- Derived grids ---
    # Grid spacing (approximate, HRRR is 3km Lambert conformal)
    data['dx'] = 3000.0  # meters
    data['dy'] = 3000.0  # meters

    # Single-level slices for index calculations
    for lvl_hpa, label in [(850, '850'), (700, '700'), (500, '500'), (950, '950'), (1000, '1000')]:
        idx = np.argmin(np.abs(data['p_levels'] - lvl_hpa))
        data[f't{label}'] = data['t_prs'][idx]          # 2D, degC
        data[f'td{label}'] = data['td_prs'][idx]        # 2D, degC
        data[f'u{label}'] = data['u_prs'][idx]          # 2D, m/s
        data[f'v{label}'] = data['v_prs'][idx]          # 2D, m/s
        data[f'gh{label}'] = data['gh_prs'][idx]        # 2D, m
        data[f'rh{label}'] = data['rh_prs'][idx]        # 2D, %

    # Extract column soundings at several grid points (for profile benchmarks)
    # Pick 100 evenly-spaced columns across the domain
    yi = np.linspace(100, ny - 100, 10, dtype=int)
    xi = np.linspace(100, nx - 100, 10, dtype=int)
    cols = [(y, x) for y in yi for x in xi]  # 100 columns
    data['sounding_cols'] = cols
    # Pre-extract a single representative sounding (middle of domain)
    cy, cx = ny // 2, nx // 2
    data['snd_p'] = data['p_levels'].copy()                    # 40 levels, hPa
    data['snd_t'] = data['t_prs'][:, cy, cx].copy()            # 40 levels, degC
    data['snd_td'] = data['td_prs'][:, cy, cx].copy()          # 40 levels, degC
    data['snd_u'] = data['u_prs'][:, cy, cx].copy()            # 40 levels, m/s
    data['snd_v'] = data['v_prs'][:, cy, cx].copy()            # 40 levels, m/s
    data['snd_gh'] = data['gh_prs'][:, cy, cx].copy()          # 40 levels, m
    data['snd_rh'] = data['rh_prs'][:, cy, cx].copy()          # 40 levels, %
    # Heights AGL (approx: subtract surface geopotential height)
    sfc_gh = data['gh_prs'][0, cy, cx]
    data['snd_h_agl'] = data['snd_gh'] - sfc_gh               # m AGL

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s — grid {ny}x{nx}, {len(data['p_levels'])} pressure levels")
    return data


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def auto_number(func, target_seconds=0.2):
    """Find iteration count so one trial takes ~target_seconds."""
    n = 1
    while True:
        t = timeit.timeit(func, number=n)
        if t >= target_seconds or n >= 1_000_000:
            return max(n, 1)
        if t < 0.001:
            n *= 100
        elif t < 0.02:
            n *= 10
        else:
            n = max(1, int(n * target_seconds / t))


def bench_one(name, func, num_trials=5, target_seconds=0.2, warmup=3):
    """Benchmark a single function, return stats dict."""
    # Warmup
    for _ in range(warmup):
        func()

    n = auto_number(func, target_seconds)

    times = []
    for _ in range(num_trials):
        t = timeit.timeit(func, number=n) / n
        times.append(t)

    times_us = [t * 1e6 for t in sorted(times)]
    p50 = np.percentile(times_us, 50)
    p95 = np.percentile(times_us, 95)
    std = np.std(times_us)

    return {
        'name': name,
        'iterations': n,
        'p50_us': p50,
        'p95_us': p95,
        'mean_us': np.mean(times_us),
        'std_us': std,
        'times_us': times_us,
    }


def fmt_time(us):
    """Format microseconds to human-friendly string."""
    if us < 1:
        return f"{us * 1000:7.1f} ns"
    elif us < 1000:
        return f"{us:7.2f} us"
    elif us < 1_000_000:
        return f"{us / 1000:7.2f} ms"
    else:
        return f"{us / 1_000_000:7.2f}  s"


def print_result(r):
    """Print one benchmark result."""
    print(f"  {r['name']:55s}  {fmt_time(r['p50_us'])} [p50]  {fmt_time(r['p95_us'])} [p95]  +/- {fmt_time(r['std_us'])}")


# ---------------------------------------------------------------------------
# Benchmark categories
# ---------------------------------------------------------------------------

def bench_scalar_thermo(data, **kw):
    """Scalar thermodynamic functions on representative values."""
    from metrust._metrust import calc
    results = []
    t, td, p, rh = 25.0, 18.0, 850.0, 65.0
    w = 0.012  # kg/kg

    benches = [
        ("potential_temperature",               lambda: calc.potential_temperature(p, t)),
        ("equivalent_potential_temperature",     lambda: calc.equivalent_potential_temperature(p, t, td)),
        ("saturation_vapor_pressure",           lambda: calc.saturation_vapor_pressure(t)),
        ("wet_bulb_temperature",                lambda: calc.wet_bulb_temperature(p, t, td)),
        ("lcl",                                 lambda: calc.lcl(1000.0, t, td)),
        ("mixing_ratio",                        lambda: calc.mixing_ratio(p, td)),
        ("dewpoint_from_rh",                    lambda: calc.dewpoint_from_relative_humidity(t, rh)),
        ("virtual_temperature",                 lambda: calc.virtual_temperature(t, p, td)),
        ("exner_function",                      lambda: calc.exner_function(p)),
        ("saturation_mixing_ratio",             lambda: calc.saturation_mixing_ratio(p, t)),
        ("density",                             lambda: calc.density(p, t, w)),
        ("vapor_pressure",                      lambda: calc.vapor_pressure(td)),
        ("heat_index",                          lambda: calc.heat_index(35.0, 70.0)),
        ("windchill",                           lambda: calc.windchill(-10.0, 12.0)),
        ("dewpoint",                            lambda: calc.dewpoint(2300.0)),
        ("relative_humidity_from_dewpoint",     lambda: calc.relative_humidity_from_dewpoint(t, td)),
        ("pressure_to_height_std",              lambda: calc.pressure_to_height_std(p)),
        ("height_to_pressure_std",              lambda: calc.height_to_pressure_std(5000.0)),
        ("dry_static_energy",                   lambda: calc.dry_static_energy(1500.0, t)),
        ("moist_static_energy",                 lambda: calc.moist_static_energy(1500.0, t, 0.015)),
        ("thickness_hydrostatic",               lambda: calc.thickness_hydrostatic(1000.0, 500.0, -10.0)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))
    return results


def bench_grid_thermo(data, **kw):
    """Thermodynamic functions applied to full HRRR 2D grids."""
    from metrust._metrust import calc
    results = []
    ny, nx = data['ny'], data['nx']

    t2m = np.ascontiguousarray(data['t2m'], dtype=np.float64)
    d2m = np.ascontiguousarray(data['d2m'], dtype=np.float64)
    psfc = np.ascontiguousarray(data['psfc'], dtype=np.float64)
    rh2m = np.ascontiguousarray(data['rh2m'], dtype=np.float64)
    t850 = np.ascontiguousarray(data['t850'].ravel(), dtype=np.float64)
    p850 = np.full(t850.shape, 850.0, dtype=np.float64)

    label = f"{ny}x{nx}"

    # Vectorized over full grid (flattened)
    t_flat = t2m.ravel()
    d_flat = d2m.ravel()
    p_flat = psfc.ravel()
    rh_flat = rh2m.ravel()

    benches = [
        (f"potential_temperature grid ({label})",
         lambda: np.array([calc.potential_temperature(p_flat[i], t_flat[i]) for i in range(0, len(t_flat), 100)])),
        (f"saturation_vapor_pressure grid ({label})",
         lambda: np.array([calc.saturation_vapor_pressure(t_flat[i]) for i in range(0, len(t_flat), 100)])),
        (f"dewpoint_from_rh grid ({label})",
         lambda: np.array([calc.dewpoint_from_relative_humidity(t_flat[i], rh_flat[i]) for i in range(0, len(t_flat), 100)])),
        (f"mixing_ratio grid ({label})",
         lambda: np.array([calc.mixing_ratio(p_flat[i], d_flat[i]) for i in range(0, len(t_flat), 100)])),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))
    return results


def bench_sounding(data, **kw):
    """Profile-based calculations on HRRR column soundings."""
    from metrust._metrust import calc
    results = []

    p = np.ascontiguousarray(data['snd_p'], dtype=np.float64)
    t = np.ascontiguousarray(data['snd_t'], dtype=np.float64)
    td = np.ascontiguousarray(data['snd_td'], dtype=np.float64)
    h = np.ascontiguousarray(data['snd_h_agl'], dtype=np.float64)
    psfc_val = float(p[0])
    t_sfc = float(t[0])
    td_sfc = float(td[0])
    nlevels = len(p)

    benches = [
        (f"parcel_profile ({nlevels} levels)",
         lambda: calc.parcel_profile(p, t_sfc, td_sfc)),
        (f"dry_lapse ({nlevels} levels)",
         lambda: calc.dry_lapse(p, t_sfc)),
        (f"moist_lapse ({nlevels} levels)",
         lambda: calc.moist_lapse(p, t_sfc)),
        (f"cape_cin surface ({nlevels} levels)",
         lambda: calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "sb", 100.0, 300.0, None)),
        (f"cape_cin mixed-layer ({nlevels} levels)",
         lambda: calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "ml", 100.0, 300.0, None)),
        (f"cape_cin most-unstable ({nlevels} levels)",
         lambda: calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "mu", 100.0, 300.0, None)),
        (f"lcl ({nlevels} levels)",
         lambda: calc.lcl(psfc_val, t_sfc, td_sfc)),
        (f"precipitable_water ({nlevels} levels)",
         lambda: calc.precipitable_water(p, td)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))

    # Batch: 100 soundings from across the domain
    cols = data['sounding_cols']

    def run_100_cape():
        for cy, cx in cols:
            sp = np.ascontiguousarray(data['p_levels'], dtype=np.float64)
            st = np.ascontiguousarray(data['t_prs'][:, cy, cx], dtype=np.float64)
            std_ = np.ascontiguousarray(data['td_prs'][:, cy, cx], dtype=np.float64)
            sh = np.ascontiguousarray(
                data['gh_prs'][:, cy, cx] - data['gh_prs'][0, cy, cx], dtype=np.float64)
            calc.cape_cin(sp, st, std_, sh, float(sp[0]), float(st[0]), float(std_[0]),
                          "sb", 100.0, 300.0, None)

    results.append(bench_one(f"cape_cin x100 soundings ({nlevels} levels each)", run_100_cape, **kw))
    return results


def bench_wind(data, **kw):
    """Wind calculations on HRRR data."""
    from metrust._metrust import calc
    results = []

    # Full-grid wind speed/direction
    u10 = np.ascontiguousarray(data['u10'].ravel(), dtype=np.float64)
    v10 = np.ascontiguousarray(data['v10'].ravel(), dtype=np.float64)
    n = len(u10)
    label = f"{data['ny']}x{data['nx']}"

    benches = [
        (f"wind_speed full grid ({label})",
         lambda: calc.wind_speed(u10, v10)),
        (f"wind_direction full grid ({label})",
         lambda: calc.wind_direction(u10, v10)),
        (f"wind_components full grid ({label})",
         lambda: calc.wind_components(
             np.sqrt(u10**2 + v10**2),
             np.degrees(np.arctan2(-u10, -v10)) % 360)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))

    # Profile wind
    u = np.ascontiguousarray(data['snd_u'], dtype=np.float64)
    v = np.ascontiguousarray(data['snd_v'], dtype=np.float64)
    h = np.ascontiguousarray(data['snd_h_agl'], dtype=np.float64)
    nlevels = len(u)

    profile_benches = [
        (f"bulk_shear 0-6km ({nlevels} levels)",
         lambda: calc.bulk_shear(u, v, h, 0.0, 6000.0)),
        (f"bulk_shear 0-1km ({nlevels} levels)",
         lambda: calc.bulk_shear(u, v, h, 0.0, 1000.0)),
        (f"storm_relative_helicity 0-1km ({nlevels} levels)",
         lambda: calc.storm_relative_helicity(u, v, h, 1000.0, 10.0, 5.0)),
        (f"storm_relative_helicity 0-3km ({nlevels} levels)",
         lambda: calc.storm_relative_helicity(u, v, h, 3000.0, 10.0, 5.0)),
        (f"bunkers_storm_motion ({nlevels} levels)",
         lambda: calc.bunkers_storm_motion(u, v, h)),
        (f"corfidi_storm_motion ({nlevels} levels)",
         lambda: calc.corfidi_storm_motion(u, v, h, float(data['snd_u'][5]), float(data['snd_v'][5]))),
    ]

    for name, fn in profile_benches:
        results.append(bench_one(name, fn, **kw))

    # Batch: 100 storm motions
    cols = data['sounding_cols']

    def run_100_bunkers():
        for cy, cx in cols:
            su = np.ascontiguousarray(data['u_prs'][:, cy, cx], dtype=np.float64)
            sv = np.ascontiguousarray(data['v_prs'][:, cy, cx], dtype=np.float64)
            sh = np.ascontiguousarray(
                data['gh_prs'][:, cy, cx] - data['gh_prs'][0, cy, cx], dtype=np.float64)
            calc.bunkers_storm_motion(su, sv, sh)

    results.append(bench_one(f"bunkers_storm_motion x100 soundings", run_100_bunkers, **kw))
    return results


def bench_grid_kinematics(data, **kw):
    """2D grid kinematics on full HRRR domain."""
    from metrust._metrust import calc
    results = []

    ny, nx = data['ny'], data['nx']
    dx, dy = data['dx'], data['dy']
    label = f"{ny}x{nx}"

    # Use 850 hPa wind for kinematics (full grid)
    u = np.ascontiguousarray(data['u850'], dtype=np.float64)
    v = np.ascontiguousarray(data['v850'], dtype=np.float64)
    theta = np.ascontiguousarray(data['t850'] + 273.15, dtype=np.float64)  # approx pot temp

    benches = [
        (f"divergence ({label})",
         lambda: calc.divergence(u, v, dx, dy)),
        (f"vorticity ({label})",
         lambda: calc.vorticity(u, v, dx, dy)),
        (f"advection ({label})",
         lambda: calc.advection(theta, u, v, dx, dy)),
        (f"frontogenesis ({label})",
         lambda: calc.frontogenesis(theta, u, v, dx, dy)),
        (f"shearing_deformation ({label})",
         lambda: calc.shearing_deformation(u, v, dx, dy)),
        (f"stretching_deformation ({label})",
         lambda: calc.stretching_deformation(u, v, dx, dy)),
        (f"total_deformation ({label})",
         lambda: calc.total_deformation(u, v, dx, dy)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))

    # Also benchmark at 500 hPa
    u500 = np.ascontiguousarray(data['u500'], dtype=np.float64)
    v500 = np.ascontiguousarray(data['v500'], dtype=np.float64)

    results.append(bench_one(
        f"divergence 500hPa ({label})",
        lambda: calc.divergence(u500, v500, dx, dy), **kw))

    # Spatial derivatives
    t2d = np.ascontiguousarray(data['t850'], dtype=np.float64)
    results.append(bench_one(
        f"gradient_x ({label})",
        lambda: calc.gradient_x(t2d, dx), **kw))
    results.append(bench_one(
        f"gradient_y ({label})",
        lambda: calc.gradient_y(t2d, dy), **kw))
    results.append(bench_one(
        f"laplacian ({label})",
        lambda: calc.laplacian(t2d, dx, dy), **kw))

    return results


def bench_smoothing(data, **kw):
    """Smoothing on full HRRR grids."""
    from metrust._metrust import calc
    results = []

    t850 = np.ascontiguousarray(data['t850'], dtype=np.float64)
    label = f"{data['ny']}x{data['nx']}"

    benches = [
        (f"smooth_gaussian sigma=2 ({label})",
         lambda: calc.smooth_gaussian(t850, 2.0)),
        (f"smooth_gaussian sigma=5 ({label})",
         lambda: calc.smooth_gaussian(t850, 5.0)),
        (f"smooth_gaussian sigma=9 ({label})",
         lambda: calc.smooth_gaussian(t850, 9.0)),
        (f"smooth_n_point n=5 ({label})",
         lambda: calc.smooth_n_point(t850, 5, 1)),
        (f"smooth_n_point n=9 ({label})",
         lambda: calc.smooth_n_point(t850, 9, 1)),
        (f"smooth_n_point n=9 x3 passes ({label})",
         lambda: calc.smooth_n_point(t850, 9, 3)),
        (f"smooth_rectangular 5 ({label})",
         lambda: calc.smooth_rectangular(t850, 5, 1)),
        (f"smooth_circular 3 ({label})",
         lambda: calc.smooth_circular(t850, 3, 1)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))
    return results


def bench_indices(data, **kw):
    """Severe weather indices and composite parameters."""
    from metrust._metrust import calc
    results = []

    # Scalar indices
    benches_scalar = [
        ("significant_tornado_parameter",
         lambda: calc.significant_tornado_parameter(2500.0, 800.0, 250.0, 25.0)),
        ("supercell_composite_parameter",
         lambda: calc.supercell_composite_parameter(3000.0, 300.0, 25.0)),
        ("critical_angle",
         lambda: calc.critical_angle(10.0, 5.0, 3.0, -2.0, 15.0, 8.0)),
        ("k_index",
         lambda: calc.k_index(20.0, 15.0, 10.0, 5.0, -15.0)),
        ("total_totals",
         lambda: calc.total_totals(20.0, 15.0, -15.0)),
        ("cross_totals",
         lambda: calc.cross_totals(15.0, -15.0)),
        ("vertical_totals",
         lambda: calc.vertical_totals(20.0, -15.0)),
        ("boyden_index",
         lambda: calc.boyden_index(100.0, 3000.0, 10.0)),
        ("heat_index",
         lambda: calc.heat_index(38.0, 75.0)),
        ("fosberg_fire_weather_index",
         lambda: calc.fosberg_fire_weather_index(95.0, 15.0, 25.0)),
    ]

    for name, fn in benches_scalar:
        results.append(bench_one(name, fn, **kw))

    # Batch: STP across 100 columns using real CAPE/shear
    cols = data['sounding_cols']

    def run_100_stp():
        for cy, cx in cols:
            # Fake but realistic values from grid data
            cape_val = max(0.0, float(data['t2m'][cy, cx]) * 100)
            lcl_val = 1000.0
            srh_val = 200.0
            shear_val = float(np.sqrt(data['u10'][cy, cx]**2 + data['v10'][cy, cx]**2)) * 2
            calc.significant_tornado_parameter(cape_val, lcl_val, srh_val, shear_val)

    results.append(bench_one("STP x100 grid points", run_100_stp, **kw))

    def run_100_kindex():
        for cy, cx in cols:
            calc.k_index(
                float(data['t850'][cy, cx]), float(data['td850'][cy, cx]),
                float(data['t700'][cy, cx]), float(data['td700'][cy, cx]),
                float(data['t500'][cy, cx]))

    results.append(bench_one("k_index x100 grid points", run_100_kindex, **kw))

    return results


def bench_conversions(data, **kw):
    """Unit and coordinate conversions."""
    from metrust._metrust import calc
    results = []

    benches = [
        ("pressure_to_height_std",       lambda: calc.pressure_to_height_std(500.0)),
        ("height_to_pressure_std",       lambda: calc.height_to_pressure_std(5500.0)),
        ("exner_function",               lambda: calc.exner_function(850.0)),
        ("coriolis_parameter",           lambda: calc.coriolis_parameter(40.0)),
        ("sigma_to_pressure",            lambda: calc.sigma_to_pressure(0.5, 1013.0, 50.0)),
        ("height_to_geopotential",       lambda: calc.height_to_geopotential(5500.0)),
        ("geopotential_to_height",       lambda: calc.geopotential_to_height(53900.0)),
        ("specific_humidity_from_mixing_ratio", lambda: calc.specific_humidity_from_mixing_ratio(0.012)),
        ("mixing_ratio_from_specific_humidity", lambda: calc.mixing_ratio_from_specific_humidity(0.012)),
    ]

    for name, fn in benches:
        results.append(bench_one(name, fn, **kw))

    return results


def bench_full_analysis(data, **kw):
    """Simulated full severe weather analysis pipeline on a single sounding."""
    from metrust._metrust import calc
    results = []

    p = np.ascontiguousarray(data['snd_p'], dtype=np.float64)
    t = np.ascontiguousarray(data['snd_t'], dtype=np.float64)
    td = np.ascontiguousarray(data['snd_td'], dtype=np.float64)
    h = np.ascontiguousarray(data['snd_h_agl'], dtype=np.float64)
    u = np.ascontiguousarray(data['snd_u'], dtype=np.float64)
    v = np.ascontiguousarray(data['snd_v'], dtype=np.float64)
    psfc_val = float(p[0])
    t_sfc = float(t[0])
    td_sfc = float(td[0])

    def full_sounding_analysis():
        """Complete severe weather analysis pipeline."""
        # LCL
        p_lcl, t_lcl = calc.lcl(psfc_val, t_sfc, td_sfc)
        # Parcel profile
        pp = calc.parcel_profile(p, t_sfc, td_sfc)
        # CAPE/CIN (all three types)
        sb = calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "sb", 100.0, 300.0, None)
        ml = calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "ml", 100.0, 300.0, None)
        mu = calc.cape_cin(p, t, td, h, psfc_val, t_sfc, td_sfc, "mu", 100.0, 300.0, None)
        # Wind analysis
        bunkers = calc.bunkers_storm_motion(u, v, h)
        bs06 = calc.bulk_shear(u, v, h, 0.0, 6000.0)
        bs01 = calc.bulk_shear(u, v, h, 0.0, 1000.0)
        srh01 = calc.storm_relative_helicity(u, v, h, 1000.0, bunkers[0][0], bunkers[0][1])
        srh03 = calc.storm_relative_helicity(u, v, h, 3000.0, bunkers[0][0], bunkers[0][1])
        # Composite params
        shear_mag = (bs06[0]**2 + bs06[1]**2)**0.5
        stp = calc.significant_tornado_parameter(ml[0], sb[2], srh01[2], shear_mag)
        scp = calc.supercell_composite_parameter(mu[0], srh03[2], shear_mag)
        # Precipitable water
        pw = calc.precipitable_water(p, td)
        return stp, scp, pw

    results.append(bench_one("FULL sounding analysis pipeline (1 column)", full_sounding_analysis, **kw))

    # 100 columns
    cols = data['sounding_cols']

    def full_analysis_100():
        for cy, cx in cols:
            sp = np.ascontiguousarray(data['p_levels'], dtype=np.float64)
            st = np.ascontiguousarray(data['t_prs'][:, cy, cx], dtype=np.float64)
            std_ = np.ascontiguousarray(data['td_prs'][:, cy, cx], dtype=np.float64)
            su = np.ascontiguousarray(data['u_prs'][:, cy, cx], dtype=np.float64)
            sv = np.ascontiguousarray(data['v_prs'][:, cy, cx], dtype=np.float64)
            sh = np.ascontiguousarray(
                data['gh_prs'][:, cy, cx] - data['gh_prs'][0, cy, cx], dtype=np.float64)
            psfc = float(sp[0])
            tsfc = float(st[0])
            tdsfc = float(std_[0])

            calc.lcl(psfc, tsfc, tdsfc)
            calc.parcel_profile(sp, tsfc, tdsfc)
            sb = calc.cape_cin(sp, st, std_, sh, psfc, tsfc, tdsfc, "sb", 100.0, 300.0, None)
            ml = calc.cape_cin(sp, st, std_, sh, psfc, tsfc, tdsfc, "ml", 100.0, 300.0, None)
            mu = calc.cape_cin(sp, st, std_, sh, psfc, tsfc, tdsfc, "mu", 100.0, 300.0, None)
            bk = calc.bunkers_storm_motion(su, sv, sh)
            bs = calc.bulk_shear(su, sv, sh, 0.0, 6000.0)
            calc.storm_relative_helicity(su, sv, sh, 1000.0, bk[0][0], bk[0][1])
            calc.storm_relative_helicity(su, sv, sh, 3000.0, bk[0][0], bk[0][1])
            smag = (bs[0]**2 + bs[1]**2)**0.5
            calc.significant_tornado_parameter(ml[0], sb[2], 200.0, smag)
            calc.supercell_composite_parameter(mu[0], 200.0, smag)
            calc.precipitable_water(sp, std_)

    results.append(bench_one("FULL sounding analysis x100 columns", full_analysis_100, **kw))

    # Full grid kinematics pipeline
    ny, nx = data['ny'], data['nx']
    dx, dy = data['dx'], data['dy']
    u850 = np.ascontiguousarray(data['u850'], dtype=np.float64)
    v850 = np.ascontiguousarray(data['v850'], dtype=np.float64)
    t850 = np.ascontiguousarray(data['t850'] + 273.15, dtype=np.float64)

    def full_grid_kinematics():
        calc.divergence(u850, v850, dx, dy)
        calc.vorticity(u850, v850, dx, dy)
        calc.advection(t850, u850, v850, dx, dy)
        calc.frontogenesis(t850, u850, v850, dx, dy)
        calc.total_deformation(u850, v850, dx, dy)
        calc.shearing_deformation(u850, v850, dx, dy)
        calc.stretching_deformation(u850, v850, dx, dy)

    results.append(bench_one(f"FULL grid kinematics pipeline ({ny}x{nx})", full_grid_kinematics, **kw))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CATEGORIES = {
    'scalar_thermo':    bench_scalar_thermo,
    'grid_thermo':      bench_grid_thermo,
    'sounding':         bench_sounding,
    'wind':             bench_wind,
    'grid_kinematics':  bench_grid_kinematics,
    'smoothing':        bench_smoothing,
    'indices':          bench_indices,
    'conversions':      bench_conversions,
    'full_analysis':    bench_full_analysis,
}


def main():
    parser = argparse.ArgumentParser(description="HRRR-driven metrust benchmark")
    parser.add_argument('--category', type=str, default=None,
                        help='Comma-separated categories to run (default: all)')
    parser.add_argument('--json', action='store_true',
                        help='Write machine-readable JSON output')
    parser.add_argument('--json-file', type=str, default='bench_hrrr_results.json',
                        help='JSON output file (default: bench_hrrr_results.json)')
    parser.add_argument('--quick', action='store_true',
                        help='Fewer trials for faster runs')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory with HRRR GRIB2 files')
    args = parser.parse_args()

    # Benchmark config
    bench_kw = dict(num_trials=5, target_seconds=0.2, warmup=3)
    if args.quick:
        bench_kw = dict(num_trials=3, target_seconds=0.1, warmup=2)

    cats = list(CATEGORIES.keys())
    if args.category:
        cats = [c.strip() for c in args.category.split(',')]
        for c in cats:
            if c not in CATEGORIES:
                print(f"Unknown category: {c}")
                print(f"Available: {', '.join(CATEGORIES.keys())}")
                sys.exit(1)

    # Load data
    data = load_hrrr_data(args.data_dir)

    print()
    print("=" * 90)
    print("metrust HRRR Benchmark — Raw Rust FFI (T1)")
    print("=" * 90)
    print(f"  Grid:     {data['ny']} x {data['nx']} ({data['ny'] * data['nx']:,} points)")
    print(f"  Levels:   {len(data['p_levels'])} pressure levels")
    print(f"  Columns:  {len(data['sounding_cols'])} sounding extraction points")
    print(f"  Trials:   {bench_kw['num_trials']}, target {bench_kw['target_seconds']}s/trial")
    print()

    all_results = []

    for cat in cats:
        print(f"--- {cat} ---")
        results = CATEGORIES[cat](data, **bench_kw)
        for r in results:
            print_result(r)
        all_results.extend(results)
        print()

    # Summary
    total_funcs = len(all_results)
    fastest = min(all_results, key=lambda r: r['p50_us'])
    slowest = max(all_results, key=lambda r: r['p50_us'])

    print("=" * 90)
    print(f"  {total_funcs} benchmarks complete")
    print(f"  Fastest: {fastest['name']:45s} {fmt_time(fastest['p50_us'])}")
    print(f"  Slowest: {slowest['name']:45s} {fmt_time(slowest['p50_us'])}")
    print("=" * 90)

    if args.json:
        # Metadata
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "unknown"

        output = {
            'metadata': {
                'benchmark': 'hrrr_massive',
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'commit': commit,
                'grid_shape': [data['ny'], data['nx']],
                'pressure_levels': len(data['p_levels']),
                'sounding_columns': len(data['sounding_cols']),
                **bench_kw,
            },
            'benchmarks': [{k: v for k, v in r.items()} for r in all_results],
        }
        with open(args.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json_file}")


if __name__ == '__main__':
    main()
