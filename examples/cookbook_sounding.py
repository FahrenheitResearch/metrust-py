"""MetPy Cookbook: Sounding Calculations — metrust drop-in demo.

This is the MetPy "Sounding Calculations" cookbook example, running entirely
on metrust.  The only change from the original MetPy version is the import
line: ``import metrust.calc as mpcalc`` instead of ``import metpy.calc as mpcalc``.

Benchmark (v0.2.3, 32-core Ryzen):
    MetPy:   96 ms
    metrust: 11 ms  (8.9x faster)

Data: uses MetPy's bundled OUN 2011-05-22 12Z sounding (no network needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from metpy.cbook import get_test_data
from metpy.units import units

# === THE ONLY CHANGE: swap metpy.calc -> metrust.calc ===
import metrust.calc as mpcalc

# ---------- Load sounding data ----------
col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]
sounding_data = pd.read_fwf(
    get_test_data("20110522_OUN_12Z.txt", as_file_obj=False),
    skiprows=7,
    usecols=[0, 1, 2, 3, 6, 7],
    names=col_names,
)
sounding_data = sounding_data.dropna(
    subset=("temperature", "dewpoint", "direction", "speed"),
    how="all",
).reset_index(drop=True)

pres = sounding_data["pressure"].values * units.hPa
temp = sounding_data["temperature"].values * units.degC
dewpoint = sounding_data["dewpoint"].values * units.degC
height = sounding_data["height"].values * units.meter
wind_speed = sounding_data["speed"].values * units.knots
wind_dir = sounding_data["direction"].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)

# ---------- Stability indices ----------
ctotals = mpcalc.cross_totals(pres, temp, dewpoint)
kindex = mpcalc.k_index(pres, temp, dewpoint)
showalter = mpcalc.showalter_index(pres, temp, dewpoint)
total_totals = mpcalc.total_totals_index(pres, temp, dewpoint)
vert_totals = mpcalc.vertical_totals(pres, temp)

print("=== Stability Indices ===")
print(f"  K-Index:       {kindex:.1f}")
print(f"  Showalter:     {showalter:.1f}")
print(f"  Total Totals:  {total_totals:.1f}")
print(f"  Cross Totals:  {ctotals:.1f}")
print(f"  Vert Totals:   {vert_totals:.1f}")

# ---------- Parcel analysis ----------
prof = mpcalc.parcel_profile(pres, temp[0], dewpoint[0])
cape, cin = mpcalc.cape_cin(pres, temp, dewpoint, prof)
lclp, lclt = mpcalc.lcl(pres[0], temp[0], dewpoint[0])
lfcp, _ = mpcalc.lfc(pres, temp, dewpoint)
el_pressure, _ = mpcalc.el(pres, temp, dewpoint, prof)
lift_index = mpcalc.lifted_index(pres, temp, prof)
sbcape, sbcin = mpcalc.surface_based_cape_cin(pres, temp, dewpoint)

print("\n=== Parcel Analysis ===")
print(f"  SBCAPE:        {sbcape:.1f}")
print(f"  SBCIN:         {sbcin:.1f}")
print(f"  CAPE:          {cape:.1f}")
print(f"  CIN:           {cin:.1f}")
print(f"  LCL:           {lclp:.1f}")
print(f"  LFC:           {lfcp:.1f}")
print(f"  EL:            {el_pressure:.1f}")
print(f"  Lifted Index:  {lift_index:.1f}")

# ---------- Mixed-layer / Most-unstable ----------
ml_t, ml_td = mpcalc.mixed_layer(pres, temp, dewpoint, depth=50 * units.hPa)
ml_p, _, _ = mpcalc.mixed_parcel(pres, temp, dewpoint, depth=50 * units.hPa)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pres, temp, dewpoint, depth=50 * units.hPa)
mu_p, mu_t, mu_td, mu_idx = mpcalc.most_unstable_parcel(pres, temp, dewpoint, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(pres, temp, dewpoint)

print("\n=== Mixed-Layer / Most-Unstable ===")
print(f"  MLCAPE:        {mlcape:.1f}")
print(f"  MLCIN:         {mlcin:.1f}")
print(f"  MUCAPE:        {mucape:.1f}")
print(f"  MUCIN:         {mucin:.1f}")
print(f"  MU parcel idx: {mu_idx}")

# ---------- Wind analysis ----------
(u_storm, v_storm), (u_left, v_left), (u_mean, v_mean) = mpcalc.bunkers_storm_motion(pres, u, v, height)
critical_angle = mpcalc.critical_angle(pres, u, v, height, u_storm, v_storm)
_, _, total_helicity = mpcalc.storm_relative_helicity(
    height, u, v, depth=1 * units.km, storm_u=u_storm, storm_v=v_storm,
)
ubshr, vbshr = mpcalc.bulk_shear(pres, u, v, height=height, depth=6 * units.km)
bshear = mpcalc.wind_speed(ubshr, vbshr)

print("\n=== Wind Analysis ===")
print(f"  Bunkers RM:    ({u_storm:.1f}, {v_storm:.1f})")
print(f"  Critical Angle:{critical_angle:.1f}")
print(f"  0-1km SRH:     {total_helicity:.1f}")
print(f"  0-6km Shear:   {bshear:.1f}")

# ---------- Severe parameters ----------
new_p = np.append(pres[pres > lclp], lclp)
new_t = np.append(temp[pres > lclp], lclt)
lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)
sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity, bshear).to_base_units()
super_comp = mpcalc.supercell_composite(mucape, total_helicity, bshear)

print("\n=== Severe Parameters ===")
print(f"  STP:           {sig_tor:.2f}")
print(f"  SCP:           {super_comp:.2f}")

print("\n--- Demo complete. All calculations used metrust as a drop-in for MetPy. ---")
