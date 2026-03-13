"""
Verification script: compute reference values from MetPy for severe weather
composites and atmospheric functions.

These values are used by the Rust integration tests in metrust to verify
numerical agreement.

Run: python tests/verify_severe_atmo.py
"""

import metpy.calc as mpcalc
from metpy.units import units
import numpy as np


def scalar(q):
    """Extract a scalar float from a pint Quantity (MetPy often returns 1-d arrays)."""
    m = q.magnitude
    if hasattr(m, 'flat'):
        return float(np.asarray(m).flat[0])
    return float(m)


# ============================================================================
# 1. Significant Tornado Parameter (STP)
# ============================================================================
# MetPy function: mpcalc.significant_tornado
# Formula: (SBCAPE/1500) * ((2000-LCL)/1000) * (SRH/150) * (shear/20)
# MetPy differences from metrust:
#   - shear term is set to 0 when shear < 12.5 m/s (metrust just floors at 0)
#   - Both cap shear at 30 m/s (term = 1.5)
#   - Both clamp LCL: capped at 1.0 when LCL <= 1000m, 0 when LCL >= 2000m

print("=" * 60)
print("SIGNIFICANT TORNADO PARAMETER (STP)")
print("=" * 60)

# Case 1: Nominal values (all terms = 1.0)
stp = mpcalc.significant_tornado(
    1500 * units('J/kg'), 1000 * units.m,
    150 * units('m^2/s^2'), 20 * units('m/s'),
)
print(f"STP case1 (nominal) = {scalar(stp):.10f}")

# Case 2: Strong case
stp = mpcalc.significant_tornado(
    4000 * units('J/kg'), 500 * units.m,
    400 * units('m^2/s^2'), 35 * units('m/s'),
)
print(f"STP case2 (strong) = {scalar(stp):.10f}")

# Case 3: Weak shear (below 12.5 m/s => MetPy sets to 0, metrust does not)
stp = mpcalc.significant_tornado(
    2000 * units('J/kg'), 800 * units.m,
    200 * units('m^2/s^2'), 10 * units('m/s'),
)
print(f"STP case3 (shear=10 m/s, MetPy zero) = {scalar(stp):.10f}")

# Case 4: LCL at 1500m
stp = mpcalc.significant_tornado(
    3000 * units('J/kg'), 1500 * units.m,
    250 * units('m^2/s^2'), 25 * units('m/s'),
)
print(f"STP case4 (LCL=1500m) = {scalar(stp):.10f}")

# Case 5: Zero CAPE
stp = mpcalc.significant_tornado(
    0 * units('J/kg'), 800 * units.m,
    200 * units('m^2/s^2'), 25 * units('m/s'),
)
print(f"STP case5 (zero CAPE) = {scalar(stp):.10f}")

# Case 6: LCL at 2000m (term = 0)
stp = mpcalc.significant_tornado(
    2000 * units('J/kg'), 2000 * units.m,
    200 * units('m^2/s^2'), 25 * units('m/s'),
)
print(f"STP case6 (LCL=2000m) = {scalar(stp):.10f}")

# Case 7: Shear exactly 30 m/s (capped at 1.5)
stp = mpcalc.significant_tornado(
    1500 * units('J/kg'), 1000 * units.m,
    150 * units('m^2/s^2'), 30 * units('m/s'),
)
print(f"STP case7 (shear=30) = {scalar(stp):.10f}")

# Case 8: Shear exactly 12.5 m/s (MetPy boundary -- NOT zeroed)
stp = mpcalc.significant_tornado(
    1500 * units('J/kg'), 1000 * units.m,
    150 * units('m^2/s^2'), 12.5 * units('m/s'),
)
print(f"STP case8 (shear=12.5) = {scalar(stp):.10f}")

# ============================================================================
# 2. Supercell Composite Parameter (SCP)
# ============================================================================
# MetPy: mpcalc.supercell_composite
# MetPy: shear set to 0 when < 10 m/s, capped at 20 m/s (term = 1.0)
# metrust: shear_term = max(shear/20, 0) (no lower cutoff, no upper cap)

print("\n" + "=" * 60)
print("SUPERCELL COMPOSITE PARAMETER (SCP)")
print("=" * 60)

scp = mpcalc.supercell_composite(
    1000 * units('J/kg'), 50 * units('m^2/s^2'), 20 * units('m/s'),
)
print(f"SCP case1 (nominal) = {scalar(scp):.10f}")

scp = mpcalc.supercell_composite(
    3000 * units('J/kg'), 300 * units('m^2/s^2'), 30 * units('m/s'),
)
print(f"SCP case2 (shear capped at 20) = {scalar(scp):.10f}")

scp = mpcalc.supercell_composite(
    2000 * units('J/kg'), 200 * units('m^2/s^2'), 8 * units('m/s'),
)
print(f"SCP case3 (shear=8, MetPy zero) = {scalar(scp):.10f}")

scp = mpcalc.supercell_composite(
    0 * units('J/kg'), 200 * units('m^2/s^2'), 25 * units('m/s'),
)
print(f"SCP case4 (zero CAPE) = {scalar(scp):.10f}")

scp = mpcalc.supercell_composite(
    5000 * units('J/kg'), 500 * units('m^2/s^2'), 25 * units('m/s'),
)
print(f"SCP case5 (very strong) = {scalar(scp):.10f}")

scp = mpcalc.supercell_composite(
    2000 * units('J/kg'), 100 * units('m^2/s^2'), 15 * units('m/s'),
)
print(f"SCP case6 (shear=15) = {scalar(scp):.10f}")

# ============================================================================
# 3. Pressure to Height (Standard Atmosphere)
# ============================================================================

print("\n" + "=" * 60)
print("PRESSURE TO HEIGHT STD")
print("=" * 60)

for p_val in [1013.25, 850.0, 700.0, 500.0, 300.0, 200.0]:
    h = mpcalc.pressure_to_height_std(p_val * units.hPa)
    print(f"pressure_to_height_std({p_val}) = {scalar(h.to('m')):.6f} m")

# ============================================================================
# 4. Height to Pressure (Standard Atmosphere)
# ============================================================================

print("\n" + "=" * 60)
print("HEIGHT TO PRESSURE STD")
print("=" * 60)

for h_val in [0.0, 1000.0, 1500.0, 3000.0, 5500.0, 8000.0, 10000.0]:
    p = mpcalc.height_to_pressure_std(h_val * units.m)
    print(f"height_to_pressure_std({h_val}) = {scalar(p.to('hPa')):.6f} hPa")

# ============================================================================
# 5. Altimeter to Station Pressure
# ============================================================================

print("\n" + "=" * 60)
print("ALTIMETER TO STATION PRESSURE")
print("=" * 60)

for alt_hpa, elev_m in [(1013.25, 0.0), (1013.25, 300.0), (1013.25, 1609.0),
                         (1013.25, 100.0), (1020.0, 500.0)]:
    asp = mpcalc.altimeter_to_station_pressure(alt_hpa * units.hPa, elev_m * units.m)
    print(f"altimeter_to_station_pressure({alt_hpa}, {elev_m}) = {scalar(asp.to('hPa')):.6f} hPa")

# ============================================================================
# 6. Altimeter to Sea-Level Pressure
# ============================================================================

print("\n" + "=" * 60)
print("ALTIMETER TO SEA LEVEL PRESSURE")
print("=" * 60)

for alt_hpa, elev_m, t_c in [(1013.25, 0.0, 15.0), (1013.25, 300.0, 10.0),
                               (1013.25, 1609.0, 20.0), (1020.0, 500.0, 25.0)]:
    slp = mpcalc.altimeter_to_sea_level_pressure(
        alt_hpa * units.hPa, elev_m * units.m, t_c * units.degC
    )
    print(f"altimeter_to_sea_level_pressure({alt_hpa}, {elev_m}, {t_c}) = {scalar(slp.to('hPa')):.6f} hPa")

# ============================================================================
# 7. Sigma to Pressure
# ============================================================================

print("\n" + "=" * 60)
print("SIGMA TO PRESSURE")
print("=" * 60)

for sigma, psfc, ptop in [(1.0, 1000.0, 100.0), (0.0, 1000.0, 100.0),
                           (0.5, 1000.0, 100.0), (0.5, 1013.25, 10.0),
                           (0.2, 1013.25, 50.0), (0.8, 1013.25, 50.0)]:
    p = mpcalc.sigma_to_pressure(
        np.array([sigma]), psfc * units.hPa, ptop * units.hPa,
    )
    print(f"sigma_to_pressure({sigma}, {psfc}, {ptop}) = {scalar(p.to('hPa')):.6f} hPa")

# ============================================================================
# 8. Heat Index
# ============================================================================

print("\n" + "=" * 60)
print("HEAT INDEX")
print("=" * 60)

for t_c, rh in [(35.0, 60.0), (35.0, 80.0), (40.0, 10.0), (40.0, 80.0),
                 (32.2, 65.0), (30.0, 90.0), (38.0, 40.0)]:
    hi = mpcalc.heat_index(t_c * units.degC, rh * units.percent, mask_undefined=False)
    print(f"heat_index({t_c}, {rh}) = {scalar(hi.to('degC')):.6f} degC")

# Below threshold
hi = mpcalc.heat_index(20.0 * units.degC, 50.0 * units.percent, mask_undefined=False)
print(f"heat_index(20.0, 50.0) = {scalar(hi.to('degC')):.6f} degC")

hi = mpcalc.heat_index(25.0 * units.degC, 70.0 * units.percent, mask_undefined=False)
print(f"heat_index(25.0, 70.0) = {scalar(hi.to('degC')):.6f} degC")

# ============================================================================
# 9. Wind Chill
# ============================================================================

print("\n" + "=" * 60)
print("WIND CHILL")
print("=" * 60)

for t_c, ws_ms in [(-10.0, 8.3333), (-10.0, 10.0), (-5.0, 5.0), (-5.0, 15.0),
                    (-17.8, 6.7), (-20.0, 12.0), (0.0, 8.0)]:
    wc = mpcalc.windchill(t_c * units.degC, ws_ms * units('m/s'), mask_undefined=False)
    print(f"windchill({t_c}, {ws_ms}) = {scalar(wc.to('degC')):.6f} degC")

# Edge: warm passthrough
wc = mpcalc.windchill(15.0 * units.degC, 10.0 * units('m/s'), mask_undefined=False)
print(f"windchill(15.0, 10.0) = {scalar(wc.to('degC')):.6f} degC")

# Edge: calm passthrough
wc = mpcalc.windchill(-10.0 * units.degC, 1.0 * units('m/s'), mask_undefined=False)
print(f"windchill(-10.0, 1.0) = {scalar(wc.to('degC')):.6f} degC")

# ============================================================================
# 10. Apparent Temperature
# ============================================================================

print("\n" + "=" * 60)
print("APPARENT TEMPERATURE")
print("=" * 60)

# Hot case (heat index)
at = mpcalc.apparent_temperature(
    30.0 * units.degC, 50.0 * units.percent, 5.0 * units('m/s'), mask_undefined=False
)
print(f"apparent_temperature(30.0, 50.0, 5.0) = {scalar(at.to('degC')):.6f} degC")

# Hot + humid
at = mpcalc.apparent_temperature(
    35.0 * units.degC, 70.0 * units.percent, 2.0 * units('m/s'), mask_undefined=False
)
print(f"apparent_temperature(35.0, 70.0, 2.0) = {scalar(at.to('degC')):.6f} degC")

# Cold + windy (windchill)
at = mpcalc.apparent_temperature(
    -10.0 * units.degC, 50.0 * units.percent, 10.0 * units('m/s'), mask_undefined=False
)
print(f"apparent_temperature(-10.0, 50.0, 10.0) = {scalar(at.to('degC')):.6f} degC")

# Mild (returns air temperature)
at = mpcalc.apparent_temperature(
    18.0 * units.degC, 50.0 * units.percent, 3.0 * units('m/s'), mask_undefined=False
)
print(f"apparent_temperature(18.0, 50.0, 3.0) = {scalar(at.to('degC')):.6f} degC")

# Cold but calm
at = mpcalc.apparent_temperature(
    -5.0 * units.degC, 50.0 * units.percent, 1.0 * units('m/s'), mask_undefined=False
)
print(f"apparent_temperature(-5.0, 50.0, 1.0) = {scalar(at.to('degC')):.6f} degC")

# ============================================================================
# 11. Bulk Richardson Number (manual -- same formula everywhere)
# ============================================================================

print("\n" + "=" * 60)
print("BULK RICHARDSON NUMBER")
print("=" * 60)

for cape, shear in [(2000.0, 20.0), (3000.0, 25.0), (1000.0, 10.0),
                     (500.0, 30.0), (4000.0, 15.0)]:
    brn = cape / (0.5 * shear**2)
    print(f"bulk_richardson_number({cape}, {shear}) = {brn:.10f}")

# ============================================================================
# 12. Boyden Index (manual -- trivial formula)
# ============================================================================

print("\n" + "=" * 60)
print("BOYDEN INDEX")
print("=" * 60)

for z1000, z700, t700 in [(100.0, 3100.0, -5.0), (50.0, 3050.0, 0.0),
                            (200.0, 3200.0, -10.0), (120.0, 3000.0, -3.0)]:
    bi = (z700 - z1000) / 10.0 - t700 - 200.0
    print(f"boyden_index({z1000}, {z700}, {t700}) = {bi:.10f}")

# ============================================================================
# 13. Haines Index (manual -- categorical)
# ============================================================================

print("\n" + "=" * 60)
print("HAINES INDEX")
print("=" * 60)

for t950, t850, td850 in [(20.0, 15.0, 10.0), (25.0, 18.0, 5.0),
                            (30.0, 20.0, 18.0), (22.0, 20.0, 11.0),
                            (28.0, 19.0, 6.0)]:
    dt = t950 - t850
    dd = t850 - td850
    a = 1 if dt <= 3 else (2 if dt <= 7 else 3)
    b = 1 if dd <= 5 else (2 if dd <= 9 else 3)
    hi = a + b
    print(f"haines_index({t950}, {t850}, {td850}) = {hi}")

# ============================================================================
# 14. Fosberg Fire Weather Index (manual -- same formula)
# ============================================================================

print("\n" + "=" * 60)
print("FOSBERG FIRE WEATHER INDEX")
print("=" * 60)


def fosberg_manual(t_f, rh, wspd_mph):
    rh = max(0.0, min(100.0, rh))
    if rh <= 10.0:
        emc = 0.03229 + 0.281073 * rh - 0.000578 * rh * t_f
    elif rh <= 50.0:
        emc = 2.22749 + 0.160107 * rh - 0.01478 * t_f
    else:
        emc = 21.0606 + 0.005565 * rh * rh - 0.00035 * rh * t_f - 0.483199 * rh
    emc = emc / 30.0
    m = max(emc, 0.0)
    eta = 1.0 - 2.0 * m + 1.5 * m * m - 0.5 * m * m * m
    fw = eta * (1.0 + wspd_mph * wspd_mph) ** 0.5
    return max(0.0, min(100.0, fw * 10.0 / 3.0))


for t_f, rh, wspd in [(90.0, 20.0, 10.0), (100.0, 5.0, 25.0),
                        (80.0, 60.0, 5.0), (95.0, 15.0, 15.0),
                        (105.0, 8.0, 30.0)]:
    ffwi = fosberg_manual(t_f, rh, wspd)
    print(f"fosberg_fire_weather_index({t_f}, {rh}, {wspd}) = {ffwi:.10f}")

# ============================================================================
# 15. Hot-Dry-Windy Index (manual)
# ============================================================================

print("\n" + "=" * 60)
print("HOT DRY WINDY INDEX")
print("=" * 60)

# HDW = VPD * wind_speed
# When vpd=0 (auto), metrust computes VPD from wx_math::thermo::vappres()
# which is different from MetPy's saturation_vapor_pressure.
# Test with explicit VPD to avoid the vapor pressure formula difference.
for t_c, rh, ws, vpd_input in [(35.0, 20.0, 10.0, 0.0), (40.0, 10.0, 15.0, 0.0),
                                 (30.0, 50.0, 5.0, 0.0), (25.0, 30.0, 8.0, 20.0)]:
    if vpd_input > 0:
        hdw = vpd_input * ws
        print(f"hot_dry_windy({t_c}, {rh}, {ws}, vpd={vpd_input}) = {hdw:.10f}")
    else:
        es = mpcalc.saturation_vapor_pressure(t_c * units.degC)
        es_hpa = scalar(es.to('hPa'))
        ea = es_hpa * (rh / 100.0)
        vpd = max(es_hpa - ea, 0.0)
        hdw = vpd * ws
        print(f"hot_dry_windy({t_c}, {rh}, {ws}, vpd=auto) = {hdw:.10f}  [metpy_es={es_hpa:.6f} vpd={vpd:.6f}]")

# ============================================================================
# 16. Critical Angle (manual -- vector geometry)
# ============================================================================

print("\n" + "=" * 60)
print("CRITICAL ANGLE")
print("=" * 60)


def critical_angle_py(storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m):
    inflow_u = u_sfc - storm_u
    inflow_v = v_sfc - storm_v
    shear_u = u_500m - u_sfc
    shear_v = v_500m - v_sfc
    mag_inflow = (inflow_u**2 + inflow_v**2)**0.5
    mag_shear = (shear_u**2 + shear_v**2)**0.5
    if mag_inflow < 1e-10 or mag_shear < 1e-10:
        return 0.0
    cos_a = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


for su, sv, usfc, vsfc, u500, v500 in [
    (10.0, 0.0, 0.0, 0.0, 0.0, 5.0),    # 90
    (0.0, -10.0, 0.0, 0.0, 0.0, 5.0),   # 0
    (0.0, 10.0, 0.0, 0.0, 0.0, 5.0),    # 180
    (10.0, 0.0, 0.0, 0.0, -5.0, 5.0),   # 45
    (5.0, 3.0, 5.0, 3.0, 10.0, 8.0),    # 0 (zero inflow)
    (10.0, 0.0, 0.0, 0.0, 0.0, 0.0),    # 0 (zero shear)
    (8.0, 5.0, 3.0, 2.0, 7.0, 9.0),     # general
]:
    angle = critical_angle_py(su, sv, usfc, vsfc, u500, v500)
    print(f"critical_angle({su},{sv},{usfc},{vsfc},{u500},{v500}) = {angle:.10f}")

# ============================================================================
# 17. Galvez-Davison Index
# ============================================================================

print("\n" + "=" * 60)
print("GALVEZ-DAVISON INDEX")
print("=" * 60)
print("NOTE: MetPy GDI uses a different profile-based interface.")
print("Computing using MetPy's equivalent_potential_temperature + metrust's formula.")

from metpy.calc import equivalent_potential_temperature as ept

for t950, t850, t700, t500, td950, td850, td700, sst in [
    (25.0, 18.0, 5.0, -15.0, 22.0, 14.0, -2.0, 28.0),
    (28.0, 20.0, 8.0, -20.0, 25.0, 16.0, 0.0, 30.0),
    (20.0, 12.0, 0.0, -25.0, 15.0, 8.0, -8.0, 22.0),
]:
    te950 = scalar(ept(950 * units.hPa, t950 * units.degC, td950 * units.degC).to('K'))
    te850 = scalar(ept(850 * units.hPa, t850 * units.degC, td850 * units.degC).to('K'))
    te700 = scalar(ept(700 * units.hPa, t700 * units.degC, td700 * units.degC).to('K'))
    te_low = (te950 + te850) / 2.0
    cbi = te_low - te700
    t500_k = t500 + 273.15
    mwi = (t500_k - 243.15) * 1.5
    ii = max(sst - 25.0, 0.0) * 5.0
    gdi = cbi + ii - mwi
    print(f"GDI(t950={t950},t850={t850},t700={t700},t500={t500},"
          f"td950={td950},td850={td850},td700={td700},sst={sst}) = {gdi:.6f}")
    print(f"  theta_e: 950={te950:.6f} 850={te850:.6f} 700={te700:.6f}")
    print(f"  cbi={cbi:.6f} mwi={mwi:.6f} ii={ii:.6f}")

# ============================================================================
# Summary of known differences
# ============================================================================

print("\n" + "=" * 60)
print("KNOWN FORMULA DIFFERENCES")
print("=" * 60)
print("""
STP: MetPy zeros shear below 12.5 m/s; metrust does not.
SCP: MetPy zeros shear below 10 m/s and caps at 1.0; metrust does neither.
altimeter_to_station_pressure: MetPy uses Smithsonian (1951) iterative; metrust uses barometric.
altimeter_to_sea_level_pressure: Different reduction methods.
HDW auto-VPD: Different saturation vapor pressure polynomials.
GDI: Different theta-e formulas (MetPy Bolton vs wx_math Bolton variant).
""")
