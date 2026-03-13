"""SounderPy drop-in demo: metrust as MetPy backend.

SounderPy is a popular sounding analysis toolkit that depends on MetPy for
all its meteorological calculations.  This example shows how to swap in
metrust as the calc backend for a 30x speedup on the compute-heavy functions.

Benchmark (v0.2.3, 32-core Ryzen):
    Full SounderPy calc:        1.25x faster
    MetPy-heavy subset only:   29.70x faster

The full calc speedup is modest because SounderPy spends most of its time
on I/O, Pandas, and formatting.  The raw metrust compute path (thermodynamics,
wind analysis, severe parameters) is ~30x faster.

Requires: pip install sounderpy
"""
from __future__ import annotations

import numpy as np
from metpy.units import units

# === THE ONLY CHANGE: swap metpy.calc -> metrust.calc ===
import metrust.calc as mpcalc

# ---------- Build a realistic sounding ----------
# Great Plains severe weather environment (similar to what SounderPy fetches)
p = np.array([1013, 1000, 975, 950, 925, 900, 875, 850, 825, 800,
              775, 750, 700, 650, 600, 550, 500, 450, 400, 350,
              300, 250, 200]) * units.hPa
T = np.array([31.2, 30.4, 28.6, 27.0, 24.8, 22.6, 20.4, 18.2, 16.0, 13.8,
              11.6, 9.4, 5.0, 0.6, -4.5, -10.2, -16.5, -23.4, -31.0, -39.5,
              -49.0, -57.5, -62.0]) * units.degC
Td = np.array([22.2, 22.0, 21.0, 19.5, 17.8, 14.6, 10.4, 6.2, 2.0, -2.2,
               -6.4, -10.6, -15.0, -19.4, -24.5, -30.2, -36.5, -43.4, -51.0, -56.5,
               -60.0, -65.5, -72.0]) * units.degC
z = np.array([0, 111, 323, 541, 764, 993, 1229, 1472, 1722, 1981,
              2248, 2525, 3107, 3726, 4392, 5112, 5900, 6772, 7749, 8863,
              9164, 10363, 11784]) * units.m
wspd = np.array([8, 10, 12, 15, 18, 20, 22, 25, 28, 30,
                 32, 35, 40, 45, 48, 50, 52, 50, 48, 45,
                 42, 40, 38]) * units.knots
wdir = np.array([180, 185, 190, 195, 200, 210, 215, 220, 225, 230,
                 235, 240, 245, 250, 255, 260, 265, 270, 270, 270,
                 270, 270, 270]) * units.degrees

u, v = mpcalc.wind_components(wspd, wdir)

# ---------- SounderPy-style calc pipeline ----------
print("=== SounderPy-style calculation pipeline (metrust backend) ===\n")

# Basic profiles
wd = mpcalc.wind_direction(u, v)
ws = mpcalc.wind_speed(u, v)
wet_bulb = mpcalc.wet_bulb_temperature(p, T, Td)
rel_humidity = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
spec_humidity = (mpcalc.specific_humidity_from_dewpoint(p, Td) * 1000) * units.g / units.kg
mix_ratio = mpcalc.mixing_ratio_from_specific_humidity(spec_humidity)
theta = mpcalc.potential_temperature(p, T)
theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
pwat = mpcalc.precipitable_water(p, Td)

print(f"  Surface theta-e:    {theta_e[0]:.2f}")
print(f"  Precipitable water: {pwat:.2f}")
print(f"  Surface wet-bulb:   {wet_bulb[0]:.2f}")
print(f"  Surface RH:         {rel_humidity[0]:.1f}")

# Parcel analysis
mu_p, mu_t, mu_td, mu_idx = mpcalc.most_unstable_parcel(p, T, Td, depth=50 * units.hPa)
ml_p, ml_t, ml_td = mpcalc.mixed_parcel(p, T, Td, bottom=p[0], depth=50 * units.hPa, interpolate=True)

prof = mpcalc.parcel_profile(p, T[0], Td[0])
sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td)

print(f"\n  SBCAPE:  {sbcape:.1f}")
print(f"  SBCIN:   {sbcin:.1f}")
print(f"  MLCAPE:  {mlcape:.1f}")
print(f"  MUCAPE:  {mucape:.1f}")

# Kinematics
rm, lm, mw = mpcalc.bunkers_storm_motion(p, u, v, z)
_, _, srh1_total = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km, storm_u=rm[0], storm_v=rm[1])
_, _, srh3_total = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km, storm_u=rm[0], storm_v=rm[1])
ubshr, vbshr = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
bshear = mpcalc.wind_speed(ubshr, vbshr)

print(f"\n  Bunkers RM:  ({rm[0]:.1f}, {rm[1]:.1f})")
print(f"  0-1km SRH:   {srh1_total:.1f}")
print(f"  0-3km SRH:   {srh3_total:.1f}")
print(f"  0-6km Shear: {bshear:.1f}")

# Severe composites
lclp, _ = mpcalc.lcl(p[0], T[0], Td[0])
lcl_hgt = mpcalc.pressure_to_height_std(lclp) - mpcalc.pressure_to_height_std(p[0])
stp = mpcalc.significant_tornado_parameter(sbcape, lcl_hgt, srh1_total, bshear)
scp = mpcalc.supercell_composite_parameter(mucape, srh3_total, bshear)

print(f"\n  STP: {stp:.2f}")
print(f"  SCP: {scp:.2f}")

print("\n--- Demo complete. All SounderPy-style calcs used metrust. ---")
