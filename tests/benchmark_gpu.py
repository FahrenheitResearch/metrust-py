#!/usr/bin/env python
"""Benchmark: MetPy vs metrust (Rust/CPU) vs metrust[gpu] (CUDA)

Real HRRR model output · 40 isobaric levels · 1059 × 1799 grid (~1.9 M pts)

Memory: ~5 GB RAM for data, ~3 GB VRAM for 3D GPU composites.

Usage:
    python tests/benchmark_gpu.py              # all three backends
    python tests/benchmark_gpu.py --no-metpy   # skip MetPy column (faster)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows so box-drawing / star characters render correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USE_METPY = "--no-metpy" not in sys.argv

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
import metrust.calc as mrcalc

if USE_METPY:
    import metpy.calc as mpcalc
    from metpy.units import units

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

# ══════════════════════════════════════════════════════════════════════════════
# Timing harness
# ══════════════════════════════════════════════════════════════════════════════

def _sync():
    cp.cuda.Stream.null.synchronize()


def bench(func, n=5, gpu=False, timeout_s=180):
    """Median wall-clock ms over *n* calls.  One warmup, auto-adapts *n*."""
    if gpu:
        _sync()
    t0 = time.perf_counter()
    func()
    if gpu:
        _sync()
    warmup_ms = (time.perf_counter() - t0) * 1000

    # adapt iteration count to fit within timeout
    if warmup_ms > timeout_s * 1000:
        return warmup_ms
    n = min(n, max(1, int(timeout_s * 1000 / max(warmup_ms, 0.001))))

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
        return "\u2014"
    if ms < 0.01:
        return f"{ms * 1000:.1f} \u00b5s"
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


# ══════════════════════════════════════════════════════════════════════════════
# Results collector
# ══════════════════════════════════════════════════════════════════════════════
COL = {"name": 40, "t": 11, "r": 10}

rows = []          # (name, t_mp, t_cpu, t_gpu, gpu_eligible)
category_rows = {} # category -> [indices into rows]


def section(title):
    category_rows[title] = []
    print()
    print(f"\u2500\u2500 {title} " + "\u2500" * max(0, 96 - len(title) - 4))
    print(f"  {'':2s} {'Function':{COL['name']}s}"
          f" {'MetPy':>{COL['t']}s} {'Rust':>{COL['t']}s}"
          f" {'CUDA':>{COL['t']}s}"
          f" {'Rust/MetPy':>{COL['r']}s} {'CUDA/Rust':>{COL['r']}s}")


def record(cat, name, t_mp, t_cpu, t_gpu, gpu_eligible=False):
    idx = len(rows)
    rows.append((name, t_mp, t_cpu, t_gpu, gpu_eligible))
    category_rows[cat].append(idx)
    star = "\u2605" if gpu_eligible else " "
    print(f"  {star} {name:{COL['name']}s}"
          f" {fmt(t_mp):>{COL['t']}s} {fmt(t_cpu):>{COL['t']}s}"
          f" {fmt(t_gpu):>{COL['t']}s}"
          f" {spd(t_mp, t_cpu):>{COL['r']}s} {spd(t_cpu, t_gpu):>{COL['r']}s}")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers for backend switching
# ══════════════════════════════════════════════════════════════════════════════

def _mp(func, n, timeout_s=120):
    """Benchmark a MetPy call.  Adapts iterations so total stays under timeout."""
    if not USE_METPY:
        return None
    try:
        t0 = time.perf_counter()
        func()  # warmup
        warmup_s = time.perf_counter() - t0
        if warmup_s > timeout_s:
            return warmup_s * 1000  # just report the one call
        actual_n = min(n, max(1, int(timeout_s / max(warmup_s, 0.001))))
        times = []
        for _ in range(actual_n):
            t0 = time.perf_counter()
            func()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))
    except Exception:
        return None


def _cpu(func, n):
    mrcalc.set_backend("cpu")
    return bench(func, n=n)


def _gpu(func, n, eligible=True):
    if not HAS_GPU or not eligible:
        return None
    mrcalc.set_backend("gpu")
    t = bench(func, n=n, gpu=True)
    mrcalc.set_backend("cpu")
    return t


# ══════════════════════════════════════════════════════════════════════════════
# HRRR data loader
# ══════════════════════════════════════════════════════════════════════════════

def load_hrrr():
    import xarray as xr

    prs_path = os.path.join(DATA_DIR, "hrrr_prs.grib2")
    if not os.path.exists(prs_path):
        sys.exit(f"HRRR data not found at {prs_path}\n"
                 "Place hrrr_prs.grib2 in the data/ directory.")

    print("  Loading HRRR GRIB ... ", end="", flush=True)
    t0 = time.perf_counter()

    # Filtered loads (~6 s total vs 100+ s for cfgrib.open_datasets)
    ds3 = xr.open_dataset(prs_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                        "indexpath": ""})
    ds_sfc = xr.open_dataset(prs_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface",
                                           "shortName": ["sp", "orog"]},
                        "indexpath": ""})
    ds_2m = xr.open_dataset(prs_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround",
                                           "level": 2,
                                           "shortName": ["2t", "2d", "2sh"]},
                        "indexpath": ""})

    # ── pressure coordinate (ensure surface-first) ───────────────────────────
    plev = np.asarray(ds3.isobaricInhPa.values, dtype=np.float64)
    flip = plev[0] < plev[-1]
    if flip:
        plev = plev[::-1]

    def g3(name):
        a = np.asarray(ds3[name].values, dtype=np.float64)
        return a[::-1] if flip else a

    nz, ny, nx = g3("t").shape

    # ── 3-D fields ───────────────────────────────────────────────────────────
    t_k  = g3("t")
    t_c  = t_k - 273.15
    dpt_c = g3("dpt") - 273.15
    u, v = g3("u"), g3("v")
    q    = g3("q")               # specific humidity  kg/kg
    gh   = g3("gh")              # geopotential height  ~m
    rh   = g3("r")               # relative humidity  %
    rwmr = g3("rwmr")            # rain water  kg/kg
    snmr = g3("snmr")            # snow         kg/kg
    grle = g3("grle")            # graupel      kg/kg

    w_mr = q / (1.0 - q)         # mixing ratio
    p3   = np.broadcast_to(plev[:, None, None], (nz, ny, nx)).copy()
    p3_pa = p3 * 100.0

    # ── surface ──────────────────────────────────────────────────────────────
    psfc = np.asarray(ds_sfc["sp"].values,   dtype=np.float64)   # Pa
    orog = np.asarray(ds_sfc["orog"].values, dtype=np.float64)   # m
    t2m  = np.asarray(ds_2m["t2m"].values,   dtype=np.float64)   # K
    d2m  = np.asarray(ds_2m["d2m"].values,   dtype=np.float64)   # K
    sh2  = np.asarray(ds_2m["sh2"].values,   dtype=np.float64)   # kg/kg
    q2_w = sh2 / (1.0 - sh2)

    h_agl = gh - orog[None, :, :]

    # ── 850 hPa slice ────────────────────────────────────────────────────────
    i850 = int(np.argmin(np.abs(plev - 850.0)))
    tc850  = t_c[i850].copy()
    dc850  = dpt_c[i850].copy()
    u850   = u[i850].copy()
    v850   = v[i850].copy()
    rh850  = rh[i850].copy()
    w850   = w_mr[i850].copy()
    # vapor pressure (Tetens from dewpoint)
    vp850  = 6.1078 * np.exp(17.27 * dc850 / (dc850 + 237.3))
    # potential temperature for frontogenesis
    th850  = (tc850 + 273.15) * (1000.0 / 850.0) ** 0.2854

    # ── single sounding column (grid centre) ─────────────────────────────────
    cy, cx = ny // 2, nx // 2

    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.1f} s  ({nz} levels, {ny}\u00d7{nx})")

    return dict(
        nz=nz, ny=ny, nx=nx, plev=plev, i850=i850,
        t_c=t_c, dpt_c=dpt_c, u=u, v=v, w_mr=w_mr, q=q, gh=gh, rh=rh,
        p3=p3, p3_pa=p3_pa, h_agl=h_agl,
        rwmr=rwmr, snmr=snmr, grle=grle,
        psfc=psfc, orog=orog, t2m=t2m, d2m=d2m, q2_w=q2_w,
        tc850=tc850, dc850=dc850, u850=u850, v850=v850,
        rh850=rh850, w850=w850, vp850=vp850, th850=th850,
        p_snd=plev.copy(),
        t_snd=t_c[:, cy, cx].copy(),
        d_snd=dpt_c[:, cy, cx].copy(),
        h_snd=h_agl[:, cy, cx].copy(),
        u_snd=u[:, cy, cx].copy(),
        v_snd=v[:, cy, cx].copy(),
        dx=3000.0, dy=3000.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 98)
    print("  BENCHMARK: MetPy  vs  metrust (Rust/CPU)  vs  metrust[gpu] (CUDA)")
    print(f"  MetPy: {'yes' if USE_METPY else 'skipped'}  |  "
          f"GPU: {GPU_NAME if HAS_GPU else 'not available'}")
    print("=" * 98)

    d = load_hrrr()
    ny, nx = d["ny"], d["nx"]
    dx, dy = d["dx"], d["dy"]

    N = 5     # 2-D operations
    N1 = 20   # 1-D sounding (fast)
    N3 = 3    # 3-D composites (slow)

    # ── MetPy Pint quantities (creation not timed) ───────────────────────────
    if USE_METPY:
        p850q   = 850.0 * units.hPa
        tc850q  = d["tc850"] * units.degC
        dc850q  = d["dc850"] * units.degC
        u850q   = d["u850"]  * units("m/s")
        v850q   = d["v850"]  * units("m/s")
        dxq     = dx * units.m
        dyq     = dy * units.m
        vp850q  = d["vp850"] * units.hPa
        w850q   = d["w850"]  * units("dimensionless")
        rh850q  = d["rh850"] * units.percent
        th850q  = d["th850"] * units.K

        psndq   = d["p_snd"] * units.hPa
        tsndq   = d["t_snd"] * units.degC
        dsndq   = d["d_snd"] * units.degC

        # parcel profile for MetPy cape_cin
        pp_mp = mpcalc.parcel_profile(psndq, tsndq[0], dsndq[0])

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: SCALAR THERMODYNAMICS  (2-D: ny × nx)
    # ══════════════════════════════════════════════════════════════════════════
    CAT1 = f"SCALAR THERMODYNAMICS (2D: {ny}\u00d7{nx})"
    section(CAT1)

    record(CAT1, "potential_temperature",
           _mp(lambda: mpcalc.potential_temperature(p850q, tc850q), N),
           _cpu(lambda: mrcalc.potential_temperature(850.0, d["tc850"]), N),
           _gpu(lambda: mrcalc.potential_temperature(850.0, d["tc850"]), N),
           gpu_eligible=True)

    record(CAT1, "equiv_potential_temperature",
           _mp(lambda: mpcalc.equivalent_potential_temperature(
               p850q, tc850q, dc850q), N),
           _cpu(lambda: mrcalc.equivalent_potential_temperature(
               850.0, d["tc850"], d["dc850"]), N),
           _gpu(lambda: mrcalc.equivalent_potential_temperature(
               850.0, d["tc850"], d["dc850"]), N),
           gpu_eligible=True)

    record(CAT1, "dewpoint",
           _mp(lambda: mpcalc.dewpoint(vp850q), N),
           _cpu(lambda: mrcalc.dewpoint(d["vp850"]), N),
           _gpu(lambda: mrcalc.dewpoint(d["vp850"]), N),
           gpu_eligible=True)

    record(CAT1, "saturation_vapor_pressure",
           _mp(lambda: mpcalc.saturation_vapor_pressure(tc850q), N),
           _cpu(lambda: mrcalc.saturation_vapor_pressure(d["tc850"]), N),
           _gpu(lambda: mrcalc.saturation_vapor_pressure(d["tc850"]), N,
                eligible=False))

    record(CAT1, "saturation_mixing_ratio",
           _mp(lambda: mpcalc.saturation_mixing_ratio(p850q, tc850q), N),
           _cpu(lambda: mrcalc.saturation_mixing_ratio(850.0, d["tc850"]), N),
           _gpu(lambda: mrcalc.saturation_mixing_ratio(850.0, d["tc850"]), N,
                eligible=False))

    record(CAT1, "dewpoint_from_rh",
           _mp(lambda: mpcalc.dewpoint_from_relative_humidity(
               tc850q, rh850q), N),
           _cpu(lambda: mrcalc.dewpoint_from_relative_humidity(
               d["tc850"], d["rh850"]), N),
           _gpu(lambda: mrcalc.dewpoint_from_relative_humidity(
               d["tc850"], d["rh850"]), N, eligible=False))

    record(CAT1, "rh_from_dewpoint",
           _mp(lambda: mpcalc.relative_humidity_from_dewpoint(
               tc850q, dc850q), N),
           _cpu(lambda: mrcalc.relative_humidity_from_dewpoint(
               d["tc850"], d["dc850"]), N),
           _gpu(lambda: mrcalc.relative_humidity_from_dewpoint(
               d["tc850"], d["dc850"]), N, eligible=False))

    record(CAT1, "virtual_temperature",
           _mp(lambda: mpcalc.virtual_temperature(tc850q, w850q), N),
           _cpu(lambda: mrcalc.virtual_temperature(d["tc850"], d["w850"]), N),
           _gpu(lambda: mrcalc.virtual_temperature(d["tc850"], d["w850"]), N,
                eligible=False))

    record(CAT1, "mixing_ratio",
           _mp(lambda: mpcalc.mixing_ratio(vp850q, p850q), N),
           _cpu(lambda: mrcalc.mixing_ratio(d["vp850"], 850.0), N),
           _gpu(lambda: mrcalc.mixing_ratio(d["vp850"], 850.0), N,
                eligible=False))

    # MetPy wet_bulb does per-element root-finding; skip on full grid
    record(CAT1, "wet_bulb_temperature",
           None,  # MetPy: skipped (iterative solver, >10 min on 1.9M pts)
           _cpu(lambda: mrcalc.wet_bulb_temperature(
               850.0, d["tc850"], d["dc850"]), N),
           _gpu(lambda: mrcalc.wet_bulb_temperature(
               850.0, d["tc850"], d["dc850"]), N, eligible=False))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: GRID KINEMATICS  (2-D: ny × nx)
    # ══════════════════════════════════════════════════════════════════════════
    CAT2 = f"GRID KINEMATICS (2D: {ny}\u00d7{nx})"
    section(CAT2)

    record(CAT2, "vorticity",
           _mp(lambda: mpcalc.vorticity(u850q, v850q, dx=dxq, dy=dyq), N),
           _cpu(lambda: mrcalc.vorticity(
               d["u850"], d["v850"], dx=dx, dy=dy), N),
           _gpu(lambda: mrcalc.vorticity(
               d["u850"], d["v850"], dx=dx, dy=dy), N),
           gpu_eligible=True)

    record(CAT2, "divergence",
           _mp(lambda: mpcalc.divergence(u850q, v850q, dx=dxq, dy=dyq), N),
           _cpu(lambda: mrcalc.divergence(
               d["u850"], d["v850"], dx=dx, dy=dy), N),
           _gpu(lambda: mrcalc.divergence(
               d["u850"], d["v850"], dx=dx, dy=dy), N, eligible=False))

    record(CAT2, "frontogenesis",
           _mp(lambda: mpcalc.frontogenesis(
               th850q, u850q, v850q, dx=dxq, dy=dyq), N),
           _cpu(lambda: mrcalc.frontogenesis(
               d["th850"], d["u850"], d["v850"], dx=dx, dy=dy), N),
           _gpu(lambda: mrcalc.frontogenesis(
               d["th850"], d["u850"], d["v850"], dx=dx, dy=dy), N),
           gpu_eligible=True)

    record(CAT2, "q_vector",
           _mp(lambda: mpcalc.q_vector(
               u850q, v850q, tc850q, p850q, dx=dxq, dy=dyq), N),
           _cpu(lambda: mrcalc.q_vector(
               d["u850"], d["v850"], d["tc850"], 850.0, dx=dx, dy=dy), N),
           _gpu(lambda: mrcalc.q_vector(
               d["u850"], d["v850"], d["tc850"], 850.0, dx=dx, dy=dy), N),
           gpu_eligible=True)

    record(CAT2, "advection",
           _mp(lambda: mpcalc.advection(
               tc850q, u850q, v850q, dx=dxq, dy=dyq), N),
           _cpu(lambda: mrcalc.advection(
               d["tc850"], d["u850"], d["v850"], dx=dx, dy=dy), N),
           _gpu(lambda: mrcalc.advection(
               d["tc850"], d["u850"], d["v850"], dx=dx, dy=dy), N,
                eligible=False))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: 1-D SOUNDING (single column, nz levels)
    # ══════════════════════════════════════════════════════════════════════════
    CAT3 = f"1D SOUNDING ({d['nz']} levels, grid centre)"
    section(CAT3)

    record(CAT3, "parcel_profile",
           _mp(lambda: mpcalc.parcel_profile(psndq, tsndq[0], dsndq[0]), N1),
           _cpu(lambda: mrcalc.parcel_profile(
               d["p_snd"], d["t_snd"][0], d["d_snd"][0]), N1),
           None)

    record(CAT3, "cape_cin",
           _mp(lambda: mpcalc.cape_cin(psndq, tsndq, dsndq, pp_mp), N1),
           _cpu(lambda: mrcalc.cape_cin(
               d["p_snd"], d["t_snd"], d["d_snd"]), N1),
           None)

    record(CAT3, "lcl",
           _mp(lambda: mpcalc.lcl(psndq[0], tsndq[0], dsndq[0]), N1),
           _cpu(lambda: mrcalc.lcl(
               d["p_snd"][0], d["t_snd"][0], d["d_snd"][0]), N1),
           None)

    record(CAT3, "lfc",
           _mp(lambda: mpcalc.lfc(psndq, tsndq, dsndq), N1),
           _cpu(lambda: mrcalc.lfc(
               d["p_snd"], d["t_snd"], d["d_snd"]), N1),
           None)

    record(CAT3, "el",
           _mp(lambda: mpcalc.el(psndq, tsndq, dsndq), N1),
           _cpu(lambda: mrcalc.el(
               d["p_snd"], d["t_snd"], d["d_snd"]), N1),
           None)

    record(CAT3, "precipitable_water",
           _mp(lambda: mpcalc.precipitable_water(psndq, dsndq), N1),
           _cpu(lambda: mrcalc.precipitable_water(
               d["p_snd"], d["d_snd"]), N1),
           None)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: GRID COMPOSITES  (3-D → 2-D)
    # ══════════════════════════════════════════════════════════════════════════
    CAT4 = f"GRID COMPOSITES (3D: {d['nz']}\u00d7{ny}\u00d7{nx} \u2192 2D)"
    section(CAT4)

    record(CAT4, "compute_cape_cin",
           None,
           _cpu(lambda: mrcalc.compute_cape_cin(
               d["p3_pa"], d["t_c"], d["w_mr"], d["h_agl"],
               d["psfc"], d["t2m"], d["q2_w"]), N3),
           _gpu(lambda: mrcalc.compute_cape_cin(
               d["p3_pa"], d["t_c"], d["w_mr"], d["h_agl"],
               d["psfc"], d["t2m"], d["q2_w"]), N3),
           gpu_eligible=True)

    record(CAT4, "compute_srh",
           None,
           _cpu(lambda: mrcalc.compute_srh(
               d["u"], d["v"], d["h_agl"]), N3),
           _gpu(lambda: mrcalc.compute_srh(
               d["u"], d["v"], d["h_agl"]), N3),
           gpu_eligible=True)

    record(CAT4, "compute_shear",
           None,
           _cpu(lambda: mrcalc.compute_shear(
               d["u"], d["v"], d["h_agl"]), N3),
           _gpu(lambda: mrcalc.compute_shear(
               d["u"], d["v"], d["h_agl"]), N3),
           gpu_eligible=True)

    record(CAT4, "compute_pw",
           None,
           _cpu(lambda: mrcalc.compute_pw(d["w_mr"], d["p3_pa"]), N3),
           _gpu(lambda: mrcalc.compute_pw(d["w_mr"], d["p3_pa"]), N3),
           gpu_eligible=True)

    record(CAT4, "composite_refl_hydrometeors",
           None,
           _cpu(lambda: mrcalc.composite_reflectivity_from_hydrometeors(
               d["p3_pa"], d["t_c"],
               d["rwmr"], d["snmr"], d["grle"]), N3),
           _gpu(lambda: mrcalc.composite_reflectivity_from_hydrometeors(
               d["p3_pa"], d["t_c"],
               d["rwmr"], d["snmr"], d["grle"]), N3),
           gpu_eligible=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: WIND & INDICES  (2-D: ny × nx)
    # ══════════════════════════════════════════════════════════════════════════
    CAT5 = f"WIND & INDICES (2D: {ny}\u00d7{nx})"
    section(CAT5)

    record(CAT5, "wind_speed",
           _mp(lambda: mpcalc.wind_speed(u850q, v850q), N),
           _cpu(lambda: mrcalc.wind_speed(d["u850"], d["v850"]), N),
           None)

    record(CAT5, "wind_direction",
           _mp(lambda: mpcalc.wind_direction(u850q, v850q), N),
           _cpu(lambda: mrcalc.wind_direction(d["u850"], d["v850"]), N),
           None)

    # precompute speed/direction for wind_components
    spd_850 = np.sqrt(d["u850"] ** 2 + d["v850"] ** 2)
    dir_850 = (270.0 - np.degrees(np.arctan2(d["v850"], d["u850"]))) % 360.0
    if USE_METPY:
        spd850q = spd_850 * units("m/s")
        dir850q = dir_850 * units.degree

    record(CAT5, "wind_components",
           _mp(lambda: mpcalc.wind_components(spd850q, dir850q), N),
           _cpu(lambda: mrcalc.wind_components(spd_850, dir_850), N),
           None)

    record(CAT5, "heat_index",
           _mp(lambda: mpcalc.heat_index(
               tc850q, rh850q, mask_undefined=False), N),
           _cpu(lambda: mrcalc.heat_index(d["tc850"], d["rh850"]), N),
           None)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 98)
    print("  SUMMARY BY CATEGORY")
    print("=" * 98)
    print(f"  {'Category':40s}"
          f" {'MetPy':>{COL['t']}s} {'Rust':>{COL['t']}s}"
          f" {'CUDA':>{COL['t']}s}"
          f" {'Rust/MetPy':>{COL['r']}s} {'CUDA/Rust':>{COL['r']}s}")
    print("  " + "\u2500" * 94)

    total_mp = total_cpu = total_gpu = 0.0
    total_mp_ok = total_cpu_ok = total_gpu_ok = False

    for cat, idxs in category_rows.items():
        s_mp = s_cpu = s_gpu = 0.0
        mp_ok = cpu_ok = gpu_ok = False
        for i in idxs:
            _, t_mp, t_cpu, t_gpu, _ = rows[i]
            if t_mp is not None:
                s_mp += t_mp; mp_ok = True
            if t_cpu is not None:
                s_cpu += t_cpu; cpu_ok = True
            if t_gpu is not None:
                s_gpu += t_gpu; gpu_ok = True

        print(f"  {cat:40s}"
              f" {fmt(s_mp) if mp_ok else '\u2014':>{COL['t']}s}"
              f" {fmt(s_cpu) if cpu_ok else '\u2014':>{COL['t']}s}"
              f" {fmt(s_gpu) if gpu_ok else '\u2014':>{COL['t']}s}"
              f" {spd(s_mp, s_cpu) if mp_ok and cpu_ok else '\u2014':>{COL['r']}s}"
              f" {spd(s_cpu, s_gpu) if cpu_ok and gpu_ok else '\u2014':>{COL['r']}s}")

        if mp_ok:
            total_mp += s_mp; total_mp_ok = True
        if cpu_ok:
            total_cpu += s_cpu; total_cpu_ok = True
        if gpu_ok:
            total_gpu += s_gpu; total_gpu_ok = True

    print("  " + "\u2500" * 94)
    print(f"  {'TOTAL':40s}"
          f" {fmt(total_mp) if total_mp_ok else '\u2014':>{COL['t']}s}"
          f" {fmt(total_cpu) if total_cpu_ok else '\u2014':>{COL['t']}s}"
          f" {fmt(total_gpu) if total_gpu_ok else '\u2014':>{COL['t']}s}"
          f" {spd(total_mp, total_cpu) if total_mp_ok and total_cpu_ok else '\u2014':>{COL['r']}s}"
          f" {spd(total_cpu, total_gpu) if total_cpu_ok and total_gpu_ok else '\u2014':>{COL['r']}s}")

    # GPU-eligible subtotal
    ge_cpu = ge_gpu = 0.0
    ge_cpu_ok = ge_gpu_ok = False
    for _, t_mp, t_cpu, t_gpu, eligible in rows:
        if eligible:
            if t_cpu is not None:
                ge_cpu += t_cpu; ge_cpu_ok = True
            if t_gpu is not None:
                ge_gpu += t_gpu; ge_gpu_ok = True
    if ge_cpu_ok and ge_gpu_ok:
        print(f"  {'\u2605 GPU-eligible only':40s}"
              f" {'\u2014':>{COL['t']}s}"
              f" {fmt(ge_cpu):>{COL['t']}s}"
              f" {fmt(ge_gpu):>{COL['t']}s}"
              f" {'\u2014':>{COL['r']}s}"
              f" {spd(ge_cpu, ge_gpu):>{COL['r']}s}")

    print()
    print("  \u2605 = GPU-eligible (dispatches to CUDA when backend=\"gpu\")")
    print("  All other functions run on CPU regardless of backend setting.")
    print("=" * 98)


if __name__ == "__main__":
    main()
