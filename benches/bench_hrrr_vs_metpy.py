#!/usr/bin/env python3
"""Head-to-head: metrust (raw Rust) vs MetPy on real HRRR 1059x1799 grid."""
import warnings, time, timeit, numpy as np, xarray as xr
warnings.filterwarnings("ignore")

# --- Load data ---
print("Loading HRRR data...")
t0 = time.time()

ds_2m = xr.open_dataset('data/hrrr_sfc.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
t2m_C = ds_2m['t2m'].values - 273.15
d2m_C = ds_2m['d2m'].values - 273.15

ds_10m = xr.open_dataset('data/hrrr_sfc.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}})
u10 = ds_10m['u10'].values
v10 = ds_10m['v10'].values

ds_t = xr.open_dataset('data/hrrr_prs.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 't'}})
t_prs = ds_t['t'].values - 273.15
p_levels = ds_t['isobaricInhPa'].values

ds_dpt = xr.open_dataset('data/hrrr_prs.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'dpt'}})
td_prs = ds_dpt['dpt'].values - 273.15

ds_u = xr.open_dataset('data/hrrr_prs.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
u_prs = ds_u['u'].values

ds_v = xr.open_dataset('data/hrrr_prs.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
v_prs = ds_v['v'].values

ds_gh = xr.open_dataset('data/hrrr_prs.grib2', engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}})
gh_prs = ds_gh['gh'].values

ny, nx = t2m_C.shape
print(f"  Loaded in {time.time()-t0:.1f}s -- grid {ny}x{nx} ({ny*nx:,} points)")

# Level slices
i850 = np.argmin(np.abs(p_levels - 850))
i700 = np.argmin(np.abs(p_levels - 700))
i500 = np.argmin(np.abs(p_levels - 500))
t850 = t_prs[i850]
u850 = np.ascontiguousarray(u_prs[i850], dtype=np.float64)
v850 = np.ascontiguousarray(v_prs[i850], dtype=np.float64)

# Sounding at domain center
cy, cx = ny // 2, nx // 2
snd_p = np.ascontiguousarray(p_levels, dtype=np.float64)
snd_t = np.ascontiguousarray(t_prs[:, cy, cx], dtype=np.float64)
snd_td = np.ascontiguousarray(td_prs[:, cy, cx], dtype=np.float64)
snd_u = np.ascontiguousarray(u_prs[:, cy, cx], dtype=np.float64)
snd_v = np.ascontiguousarray(v_prs[:, cy, cx], dtype=np.float64)
snd_h = np.ascontiguousarray(gh_prs[:, cy, cx] - gh_prs[0, cy, cx], dtype=np.float64)

# --- Imports ---
from metrust._metrust import calc as mr
import metpy.calc as mpc
from metpy.units import units as mu

dx = dy = 3000.0

# --- Pint arrays for MetPy ---
u10_pint = u10 * mu('m/s')
v10_pint = v10 * mu('m/s')
u850_pint = u850 * mu('m/s')
v850_pint = v850 * mu('m/s')
theta850_pint = (t850 + 273.15) * mu.K
dx_pint = np.ones((ny, nx - 1)) * 3000.0 * mu.meter
dy_pint = np.ones((ny - 1, nx)) * 3000.0 * mu.meter

snd_p_pint = snd_p * mu.hPa
snd_t_pint = snd_t * mu.degC
snd_td_pint = snd_td * mu.degC
snd_u_pint = snd_u * mu('m/s')
snd_v_pint = snd_v * mu('m/s')
snd_h_pint = snd_h * mu.meter


# --- Benchmark helper ---
def bench(fn, trials=3, target=0.15):
    for _ in range(3):
        fn()
    n = 1
    while True:
        t = timeit.timeit(fn, number=n)
        if t >= target or n >= 500000:
            break
        n = max(1, int(n * target / max(t, 1e-9)))
    times = sorted([timeit.timeit(fn, number=n) / n for _ in range(trials)])
    return times[len(times) // 2] * 1e6  # p50 in us


def fmt(us):
    if us < 1:
        return f"{us * 1000:.0f} ns"
    if us < 1000:
        return f"{us:.1f} us"
    if us < 1e6:
        return f"{us / 1000:.2f} ms"
    return f"{us / 1e6:.2f} s"


# --- Head to head ---
results = []


def row(name, mr_fn, mp_fn):
    mr_us = bench(mr_fn)
    mp_us = bench(mp_fn)
    ratio = mp_us / mr_us if mr_us > 0 else 0
    print(f"  {name:<45s} {fmt(mr_us):>12s} {fmt(mp_us):>12s} {ratio:>8.1f}x")
    results.append((name, mr_us, mp_us, ratio))


print()
print(f"{'Operation':<47s} {'metrust':>12s} {'MetPy':>12s} {'Speedup':>9s}")
print("=" * 82)

# --- Scalar thermo ---
print("\n  SCALAR THERMODYNAMICS (single point)")
print("  " + "-" * 78)

row("potential_temperature",
    lambda: mr.potential_temperature(850.0, 25.0),
    lambda: mpc.potential_temperature(850.0 * mu.hPa, 25.0 * mu.degC))

row("saturation_vapor_pressure",
    lambda: mr.saturation_vapor_pressure(25.0),
    lambda: mpc.saturation_vapor_pressure(25.0 * mu.degC))

row("dewpoint_from_rh",
    lambda: mr.dewpoint_from_relative_humidity(25.0, 65.0),
    lambda: mpc.dewpoint_from_relative_humidity(25.0 * mu.degC, 65.0 * mu.percent))

row("equivalent_potential_temperature",
    lambda: mr.equivalent_potential_temperature(850.0, 25.0, 18.0),
    lambda: mpc.equivalent_potential_temperature(850.0 * mu.hPa, 25.0 * mu.degC, 18.0 * mu.degC))

row("wet_bulb_temperature",
    lambda: mr.wet_bulb_temperature(850.0, 25.0, 18.0),
    lambda: mpc.wet_bulb_temperature(850.0 * mu.hPa, 25.0 * mu.degC, 18.0 * mu.degC))

row("lcl",
    lambda: mr.lcl(1000.0, 25.0, 18.0),
    lambda: mpc.lcl(1000.0 * mu.hPa, 25.0 * mu.degC, 18.0 * mu.degC))

# --- Sounding ---
print(f"\n  SOUNDING ANALYSIS (40-level real HRRR column)")
print("  " + "-" * 78)

row("parcel_profile",
    lambda: mr.parcel_profile(snd_p, float(snd_t[0]), float(snd_td[0])),
    lambda: mpc.parcel_profile(snd_p_pint, snd_t_pint[0], snd_td_pint[0]))

row("cape_cin (surface-based)",
    lambda: mr.cape_cin(snd_p, snd_t, snd_td, snd_h,
                        float(snd_p[0]), float(snd_t[0]), float(snd_td[0]),
                        "sb", 100.0, 300.0, None),
    lambda: mpc.cape_cin(snd_p_pint, snd_t_pint, snd_td_pint,
                         mpc.parcel_profile(snd_p_pint, snd_t_pint[0], snd_td_pint[0])))

row("precipitable_water",
    lambda: mr.precipitable_water(snd_p, snd_td),
    lambda: mpc.precipitable_water(snd_p_pint, snd_td_pint))

row("bulk_shear 0-6km",
    lambda: mr.bulk_shear(snd_u, snd_v, snd_h, 0.0, 6000.0),
    lambda: mpc.bulk_shear(snd_p_pint, snd_u_pint, snd_v_pint,
                           height=snd_h_pint, depth=6000.0 * mu.meter))

row("storm_relative_helicity 0-1km",
    lambda: mr.storm_relative_helicity(snd_u, snd_v, snd_h, 1000.0, 10.0, 5.0),
    lambda: mpc.storm_relative_helicity(snd_h_pint, snd_u_pint, snd_v_pint,
                                        depth=1000.0 * mu.meter))

row("bunkers_storm_motion",
    lambda: mr.bunkers_storm_motion(snd_u, snd_v, snd_h),
    lambda: mpc.bunkers_storm_motion(snd_p_pint, snd_u_pint, snd_v_pint, snd_h_pint))

# --- Full grid ---
print(f"\n  FULL GRID ({ny}x{nx} = {ny * nx:,} points)")
print("  " + "-" * 78)

u10_flat = np.ascontiguousarray(u10.ravel(), dtype=np.float64)
v10_flat = np.ascontiguousarray(v10.ravel(), dtype=np.float64)

row("wind_speed",
    lambda: mr.wind_speed(u10_flat, v10_flat),
    lambda: mpc.wind_speed(u10_pint, v10_pint))

row("wind_direction",
    lambda: mr.wind_direction(u10_flat, v10_flat),
    lambda: mpc.wind_direction(u10_pint, v10_pint))

row("divergence",
    lambda: mr.divergence(u850, v850, dx, dy),
    lambda: mpc.divergence(u850_pint, v850_pint, dx=dx_pint, dy=dy_pint))

row("vorticity",
    lambda: mr.vorticity(u850, v850, dx, dy),
    lambda: mpc.vorticity(u850_pint, v850_pint, dx=dx_pint, dy=dy_pint))

row("advection",
    lambda: mr.advection(np.ascontiguousarray(t850, dtype=np.float64), u850, v850, dx, dy),
    lambda: mpc.advection(theta850_pint, u850_pint, v850_pint, dx=dx_pint, dy=dy_pint))

row("smooth_gaussian sigma=5",
    lambda: mr.smooth_gaussian(np.ascontiguousarray(t850, dtype=np.float64), 5.0),
    lambda: mpc.smooth_gaussian(t850, 5))

row("smooth_n_point n=9",
    lambda: mr.smooth_n_point(np.ascontiguousarray(t850, dtype=np.float64), 9, 1),
    lambda: mpc.smooth_n_point(t850, 9, 1))

# --- Summary ---
print()
print("=" * 82)
wins = sum(1 for _, _, _, r in results if r > 1)
losses = sum(1 for _, _, _, r in results if r < 1)
geomean = np.exp(np.mean(np.log([r for _, _, _, r in results if r > 0])))
print(f"  {len(results)} benchmarks | {wins} faster | {losses} slower | "
      f"geometric mean speedup: {geomean:.1f}x")
best = max(results, key=lambda x: x[3])
worst = min(results, key=lambda x: x[3])
print(f"  Best:  {best[0]} ({best[3]:.0f}x)")
print(f"  Worst: {worst[0]} ({worst[3]:.1f}x)")
print("=" * 82)
