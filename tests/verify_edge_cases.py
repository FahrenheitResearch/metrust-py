"""
Edge case verification for metrust vs MetPy compatibility.

Tests NaN handling, boundary conditions, empty/single-element arrays,
extreme values, and special floating-point behavior to document
MetPy's exact behavior for each case.
"""

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import warnings

# Suppress pint/metpy warnings so we can see clean output
warnings.filterwarnings("ignore")

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

# ============================================================
# 1. NaN Handling
# ============================================================
section("NaN HANDLING - Scalar NaN inputs")

# potential_temperature with NaN pressure
try:
    result = mpcalc.potential_temperature(np.nan * units.hPa, 25 * units.degC)
    print(f"pot_temp(NaN hPa, 25C) = {result.magnitude:.6f} K  [is_nan={np.isnan(result.magnitude)}]")
except Exception as e:
    print(f"pot_temp(NaN hPa, 25C) -> ERROR: {type(e).__name__}: {e}")

# potential_temperature with NaN temperature
try:
    result = mpcalc.potential_temperature(1000 * units.hPa, np.nan * units.degC)
    print(f"pot_temp(1000hPa, NaN C) = {result.magnitude:.6f} K  [is_nan={np.isnan(result.magnitude)}]")
except Exception as e:
    print(f"pot_temp(1000hPa, NaN C) -> ERROR: {type(e).__name__}: {e}")

# potential_temperature with both NaN
try:
    result = mpcalc.potential_temperature(np.nan * units.hPa, np.nan * units.degC)
    print(f"pot_temp(NaN, NaN) = {result.magnitude:.6f} K  [is_nan={np.isnan(result.magnitude)}]")
except Exception as e:
    print(f"pot_temp(NaN, NaN) -> ERROR: {type(e).__name__}: {e}")

# saturation_vapor_pressure with NaN
try:
    result = mpcalc.saturation_vapor_pressure(np.nan * units.degC)
    print(f"sat_vp(NaN C) = {result.magnitude:.6f}  [is_nan={np.isnan(result.magnitude)}]")
except Exception as e:
    print(f"sat_vp(NaN C) -> ERROR: {type(e).__name__}: {e}")

# wind_speed with NaN
try:
    ws = mpcalc.wind_speed(np.nan * units('m/s'), 5 * units('m/s'))
    print(f"wind_speed(NaN, 5) = {ws.magnitude:.6f}  [is_nan={np.isnan(ws.magnitude)}]")
except Exception as e:
    print(f"wind_speed(NaN, 5) -> ERROR: {type(e).__name__}: {e}")

try:
    ws = mpcalc.wind_speed(5 * units('m/s'), np.nan * units('m/s'))
    print(f"wind_speed(5, NaN) = {ws.magnitude:.6f}  [is_nan={np.isnan(ws.magnitude)}]")
except Exception as e:
    print(f"wind_speed(5, NaN) -> ERROR: {type(e).__name__}: {e}")

# wind_direction with NaN
try:
    wd = mpcalc.wind_direction(np.nan * units('m/s'), 5 * units('m/s'))
    print(f"wind_direction(NaN, 5) = {wd.magnitude:.6f}  [is_nan={np.isnan(wd.magnitude)}]")
except Exception as e:
    print(f"wind_direction(NaN, 5) -> ERROR: {type(e).__name__}: {e}")

# relative_humidity_from_dewpoint with NaN
try:
    rh = mpcalc.relative_humidity_from_dewpoint(np.nan * units.degC, 15 * units.degC)
    print(f"rh_from_dp(NaN, 15C) = {rh.magnitude:.6f}  [is_nan={np.isnan(rh.magnitude)}]")
except Exception as e:
    print(f"rh_from_dp(NaN, 15C) -> ERROR: {type(e).__name__}: {e}")

try:
    rh = mpcalc.relative_humidity_from_dewpoint(25 * units.degC, np.nan * units.degC)
    print(f"rh_from_dp(25C, NaN) = {rh.magnitude:.6f}  [is_nan={np.isnan(rh.magnitude)}]")
except Exception as e:
    print(f"rh_from_dp(25C, NaN) -> ERROR: {type(e).__name__}: {e}")

# mixing_ratio_from_relative_humidity with NaN
try:
    mr = mpcalc.mixing_ratio_from_relative_humidity(
        1000 * units.hPa, np.nan * units.degC, 50 * units.percent
    )
    print(f"mixing_ratio_from_rh(1000, NaN, 50%) = {mr.magnitude:.6f}  [is_nan={np.isnan(mr.magnitude)}]")
except Exception as e:
    print(f"mixing_ratio_from_rh(1000, NaN, 50%) -> ERROR: {type(e).__name__}: {e}")

# equivalent_potential_temperature with NaN
try:
    result = mpcalc.equivalent_potential_temperature(
        np.nan * units.hPa, 25 * units.degC, 15 * units.degC
    )
    print(f"equiv_pot_temp(NaN, 25C, 15C) = {result.magnitude:.6f}  [is_nan={np.isnan(result.magnitude)}]")
except Exception as e:
    print(f"equiv_pot_temp(NaN, 25C, 15C) -> ERROR: {type(e).__name__}: {e}")

# wet_bulb_temperature with NaN
try:
    wb = mpcalc.wet_bulb_temperature(
        np.nan * units.hPa, 25 * units.degC, 15 * units.degC
    )
    print(f"wet_bulb(NaN, 25C, 15C) = {wb.magnitude:.6f}  [is_nan={np.isnan(wb.magnitude)}]")
except Exception as e:
    print(f"wet_bulb(NaN, 25C, 15C) -> ERROR: {type(e).__name__}: {e}")

section("NaN HANDLING - Array with NaN elements")

# potential_temperature array with NaN
try:
    p = np.array([1000, np.nan, 800, 700]) * units.hPa
    t = np.array([25, 20, np.nan, 10]) * units.degC
    result = mpcalc.potential_temperature(p, t)
    print(f"pot_temp with NaN in arrays:")
    for i, v in enumerate(result.magnitude):
        print(f"  [{i}] = {v:.6f}  [is_nan={np.isnan(v)}]")
except Exception as e:
    print(f"pot_temp with NaN arrays -> ERROR: {type(e).__name__}: {e}")

# wind_speed array with NaN
try:
    u = np.array([3, np.nan, 0, 5]) * units('m/s')
    v = np.array([4, 5, np.nan, 12]) * units('m/s')
    ws = mpcalc.wind_speed(u, v)
    print(f"wind_speed with NaN in arrays:")
    for i, val in enumerate(ws.magnitude):
        print(f"  [{i}] = {val:.6f}  [is_nan={np.isnan(val)}]")
except Exception as e:
    print(f"wind_speed with NaN arrays -> ERROR: {type(e).__name__}: {e}")

# wind_direction array with NaN
try:
    u = np.array([3, np.nan, 0, 5]) * units('m/s')
    v = np.array([4, 5, np.nan, 12]) * units('m/s')
    wd = mpcalc.wind_direction(u, v)
    print(f"wind_direction with NaN in arrays:")
    for i, val in enumerate(wd.magnitude):
        print(f"  [{i}] = {val:.6f}  [is_nan={np.isnan(val)}]")
except Exception as e:
    print(f"wind_direction with NaN arrays -> ERROR: {type(e).__name__}: {e}")

# saturation_vapor_pressure array with NaN
try:
    t = np.array([25, np.nan, -40, 0]) * units.degC
    result = mpcalc.saturation_vapor_pressure(t)
    print(f"sat_vp with NaN in array:")
    for i, v in enumerate(result.magnitude):
        print(f"  [{i}] = {v:.6f}  [is_nan={np.isnan(v)}]")
except Exception as e:
    print(f"sat_vp with NaN array -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 2. Boundary Conditions - Zero Values
# ============================================================
section("BOUNDARY CONDITIONS - Zero values")

# Zero pressure
try:
    h = mpcalc.pressure_to_height_std(0 * units.hPa)
    print(f"pressure_to_height(0 hPa) = {h.magnitude:.6f} m  [is_inf={np.isinf(h.magnitude)}, is_nan={np.isnan(h.magnitude)}]")
except Exception as e:
    print(f"pressure_to_height(0 hPa) -> ERROR: {type(e).__name__}: {e}")

# Zero temperature (Celsius) for saturation vapor pressure
try:
    es = mpcalc.saturation_vapor_pressure(0 * units.degC)
    print(f"sat_vp(0C) = {es.magnitude:.6f} hPa")
except Exception as e:
    print(f"sat_vp(0C) -> ERROR: {type(e).__name__}: {e}")

# Zero wind speed
try:
    ws = mpcalc.wind_speed(0 * units('m/s'), 0 * units('m/s'))
    print(f"wind_speed(0, 0) = {ws.magnitude:.6f}")
except Exception as e:
    print(f"wind_speed(0, 0) -> ERROR: {type(e).__name__}: {e}")

# wind_direction with zero wind
try:
    wd = mpcalc.wind_direction(0 * units('m/s'), 0 * units('m/s'))
    print(f"wind_direction(0, 0) = {wd.magnitude:.6f} deg")
except Exception as e:
    print(f"wind_direction(0, 0) -> ERROR: {type(e).__name__}: {e}")

# wind_components with zero speed
try:
    u, v = mpcalc.wind_components(0 * units('m/s'), 270 * units.degree)
    print(f"wind_components(0, 270) = u={u.magnitude:.6f}, v={v.magnitude:.6f}")
except Exception as e:
    print(f"wind_components(0, 270) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 3. Boundary Conditions - Negative/Below-Zero Values
# ============================================================
section("BOUNDARY CONDITIONS - Negative and extreme temperature values")

# Negative pressure (unphysical)
try:
    h = mpcalc.pressure_to_height_std(-100 * units.hPa)
    print(f"pressure_to_height(-100 hPa) = {h.magnitude:.6f}  [is_nan={np.isnan(h.magnitude)}]")
except Exception as e:
    print(f"pressure_to_height(-100 hPa) -> ERROR: {type(e).__name__}: {e}")

# Below absolute zero
try:
    es = mpcalc.saturation_vapor_pressure(-274 * units.degC)
    print(f"sat_vp(-274C) = {es.magnitude:.10e}  [is_nan={np.isnan(es.magnitude)}]")
except Exception as e:
    print(f"sat_vp(-274C) -> ERROR: {type(e).__name__}: {e}")

# Very cold temperature
try:
    es = mpcalc.saturation_vapor_pressure(-100 * units.degC)
    print(f"sat_vp(-100C) = {es.magnitude:.10e}")
except Exception as e:
    print(f"sat_vp(-100C) -> ERROR: {type(e).__name__}: {e}")

# potential_temperature with very low pressure
try:
    pt = mpcalc.potential_temperature(1 * units.hPa, -50 * units.degC)
    print(f"pot_temp(1 hPa, -50C) = {pt.magnitude:.6f} K")
except Exception as e:
    print(f"pot_temp(1 hPa, -50C) -> ERROR: {type(e).__name__}: {e}")

# potential_temperature with very high pressure
try:
    pt = mpcalc.potential_temperature(2000 * units.hPa, 50 * units.degC)
    print(f"pot_temp(2000 hPa, 50C) = {pt.magnitude:.6f} K")
except Exception as e:
    print(f"pot_temp(2000 hPa, 50C) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 4. Boundary Conditions - Humidity Edge Cases
# ============================================================
section("BOUNDARY CONDITIONS - Humidity edge cases")

# RH when T = Td (saturated, 100%)
try:
    rh = mpcalc.relative_humidity_from_dewpoint(25 * units.degC, 25 * units.degC)
    print(f"RH when T=Td=25C: {rh.magnitude:.10f}  (expected ~1.0)")
except Exception as e:
    print(f"RH when T=Td -> ERROR: {type(e).__name__}: {e}")

# RH when Td very low (very dry)
try:
    rh = mpcalc.relative_humidity_from_dewpoint(25 * units.degC, -40 * units.degC)
    print(f"RH when T=25C, Td=-40C: {rh.magnitude:.10f}")
except Exception as e:
    print(f"RH when Td very low -> ERROR: {type(e).__name__}: {e}")

# RH when Td > T (supersaturation, unphysical)
try:
    rh = mpcalc.relative_humidity_from_dewpoint(20 * units.degC, 30 * units.degC)
    print(f"RH when Td>T (20C, 30C): {rh.magnitude:.10f}  (>1 = supersaturated)")
except Exception as e:
    print(f"RH when Td>T -> ERROR: {type(e).__name__}: {e}")

# Dewpoint from 100% RH
try:
    td = mpcalc.dewpoint_from_relative_humidity(25 * units.degC, 100 * units.percent)
    print(f"dewpoint(25C, 100%RH) = {td.magnitude:.6f} C  (expected ~25)")
except Exception as e:
    print(f"dewpoint(25C, 100%RH) -> ERROR: {type(e).__name__}: {e}")

# Dewpoint from 0% RH
try:
    td = mpcalc.dewpoint_from_relative_humidity(25 * units.degC, 0 * units.percent)
    print(f"dewpoint(25C, 0%RH) = {td.magnitude:.6f} C  [is_inf={np.isinf(td.magnitude)}, is_nan={np.isnan(td.magnitude)}]")
except Exception as e:
    print(f"dewpoint(25C, 0%RH) -> ERROR: {type(e).__name__}: {e}")

# Dewpoint from very small RH
try:
    td = mpcalc.dewpoint_from_relative_humidity(25 * units.degC, 0.01 * units.percent)
    print(f"dewpoint(25C, 0.01%RH) = {td.magnitude:.6f} C")
except Exception as e:
    print(f"dewpoint(25C, 0.01%RH) -> ERROR: {type(e).__name__}: {e}")

# Mixing ratio at very low pressure
try:
    mr = mpcalc.saturation_mixing_ratio(10 * units.hPa, -50 * units.degC)
    print(f"sat_mix_ratio(10 hPa, -50C) = {mr.magnitude:.6f}")
except Exception as e:
    print(f"sat_mix_ratio(10 hPa, -50C) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 5. Empty and Single-Element Arrays
# ============================================================
section("EMPTY AND SINGLE-ELEMENT ARRAYS")

# Empty arrays - wind_speed
try:
    ws = mpcalc.wind_speed(np.array([]) * units('m/s'), np.array([]) * units('m/s'))
    print(f"wind_speed([], []) = {ws.magnitude}  [len={len(ws.magnitude)}]")
except Exception as e:
    print(f"wind_speed([], []) -> ERROR: {type(e).__name__}: {e}")

# Empty arrays - wind_direction
try:
    wd = mpcalc.wind_direction(np.array([]) * units('m/s'), np.array([]) * units('m/s'))
    print(f"wind_direction([], []) = {wd.magnitude}  [len={len(wd.magnitude)}]")
except Exception as e:
    print(f"wind_direction([], []) -> ERROR: {type(e).__name__}: {e}")

# Empty arrays - potential_temperature
try:
    pt = mpcalc.potential_temperature(np.array([]) * units.hPa, np.array([]) * units.degC)
    print(f"pot_temp([], []) = {pt.magnitude}  [len={len(pt.magnitude)}]")
except Exception as e:
    print(f"pot_temp([], []) -> ERROR: {type(e).__name__}: {e}")

# Empty arrays - saturation_vapor_pressure
try:
    es = mpcalc.saturation_vapor_pressure(np.array([]) * units.degC)
    print(f"sat_vp([]) = {es.magnitude}  [len={len(es.magnitude)}]")
except Exception as e:
    print(f"sat_vp([]) -> ERROR: {type(e).__name__}: {e}")

# Single element arrays
try:
    ws = mpcalc.wind_speed(np.array([5]) * units('m/s'), np.array([3]) * units('m/s'))
    print(f"wind_speed([5], [3]) = {ws.magnitude}  [expected ~5.831]")
except Exception as e:
    print(f"wind_speed([5], [3]) -> ERROR: {type(e).__name__}: {e}")

try:
    wd = mpcalc.wind_direction(np.array([5]) * units('m/s'), np.array([3]) * units('m/s'))
    print(f"wind_direction([5], [3]) = {wd.magnitude}")
except Exception as e:
    print(f"wind_direction([5], [3]) -> ERROR: {type(e).__name__}: {e}")

try:
    pt = mpcalc.potential_temperature(np.array([1000]) * units.hPa, np.array([25]) * units.degC)
    print(f"pot_temp([1000], [25]) = {pt.magnitude}")
except Exception as e:
    print(f"pot_temp([1000], [25]) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 6. Special Floating-Point Values (Infinity)
# ============================================================
section("INFINITY HANDLING")

# Infinity pressure
try:
    pt = mpcalc.potential_temperature(np.inf * units.hPa, 25 * units.degC)
    print(f"pot_temp(inf hPa, 25C) = {pt.magnitude:.6f}  [is_nan={np.isnan(pt.magnitude)}, is_zero={pt.magnitude==0.0}]")
except Exception as e:
    print(f"pot_temp(inf, 25C) -> ERROR: {type(e).__name__}: {e}")

# Negative infinity temperature
try:
    es = mpcalc.saturation_vapor_pressure(-np.inf * units.degC)
    print(f"sat_vp(-inf C) = {es.magnitude:.6e}  [is_zero={es.magnitude==0.0}, is_nan={np.isnan(es.magnitude)}]")
except Exception as e:
    print(f"sat_vp(-inf C) -> ERROR: {type(e).__name__}: {e}")

# Positive infinity temperature
try:
    es = mpcalc.saturation_vapor_pressure(np.inf * units.degC)
    print(f"sat_vp(inf C) = {es.magnitude:.6e}  [is_inf={np.isinf(es.magnitude)}]")
except Exception as e:
    print(f"sat_vp(inf C) -> ERROR: {type(e).__name__}: {e}")

# wind_speed with infinity
try:
    ws = mpcalc.wind_speed(np.inf * units('m/s'), 5 * units('m/s'))
    print(f"wind_speed(inf, 5) = {ws.magnitude:.6e}  [is_inf={np.isinf(ws.magnitude)}]")
except Exception as e:
    print(f"wind_speed(inf, 5) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 7. Extreme Values
# ============================================================
section("EXTREME VALUES")

# Very high pressure (bottom of deep atmosphere)
try:
    pt = mpcalc.potential_temperature(1100 * units.hPa, 45 * units.degC)
    print(f"pot_temp(1100 hPa, 45C) = {pt.magnitude:.6f} K")
except Exception as e:
    print(f"pot_temp(1100 hPa, 45C) -> ERROR: {type(e).__name__}: {e}")

# Very low pressure (stratosphere)
try:
    pt = mpcalc.potential_temperature(10 * units.hPa, -60 * units.degC)
    print(f"pot_temp(10 hPa, -60C) = {pt.magnitude:.6f} K")
except Exception as e:
    print(f"pot_temp(10 hPa, -60C) -> ERROR: {type(e).__name__}: {e}")

# Very strong winds
try:
    ws = mpcalc.wind_speed(100 * units('m/s'), 100 * units('m/s'))
    print(f"wind_speed(100, 100) = {ws.magnitude:.6f} m/s  [expected ~141.421]")
except Exception as e:
    print(f"wind_speed(100, 100) -> ERROR: {type(e).__name__}: {e}")

# Very weak winds (near zero but not zero)
try:
    ws = mpcalc.wind_speed(1e-10 * units('m/s'), 1e-10 * units('m/s'))
    print(f"wind_speed(1e-10, 1e-10) = {ws.magnitude:.6e}")
except Exception as e:
    print(f"wind_speed(1e-10, 1e-10) -> ERROR: {type(e).__name__}: {e}")

try:
    wd = mpcalc.wind_direction(1e-10 * units('m/s'), 1e-10 * units('m/s'))
    print(f"wind_direction(1e-10, 1e-10) = {wd.magnitude:.6f}")
except Exception as e:
    print(f"wind_direction(1e-10, 1e-10) -> ERROR: {type(e).__name__}: {e}")

# Negative wind components
try:
    ws = mpcalc.wind_speed(-10 * units('m/s'), -10 * units('m/s'))
    print(f"wind_speed(-10, -10) = {ws.magnitude:.6f}  [expected ~14.142]")
except Exception as e:
    print(f"wind_speed(-10, -10) -> ERROR: {type(e).__name__}: {e}")

try:
    wd = mpcalc.wind_direction(-10 * units('m/s'), -10 * units('m/s'))
    print(f"wind_direction(-10, -10) = {wd.magnitude:.6f} deg")
except Exception as e:
    print(f"wind_direction(-10, -10) -> ERROR: {type(e).__name__}: {e}")

# Very high temperature for saturation vapor pressure
try:
    es = mpcalc.saturation_vapor_pressure(60 * units.degC)
    print(f"sat_vp(60C) = {es.magnitude:.6f} hPa")
except Exception as e:
    print(f"sat_vp(60C) -> ERROR: {type(e).__name__}: {e}")

# Very low temperature for saturation vapor pressure
try:
    es = mpcalc.saturation_vapor_pressure(-80 * units.degC)
    print(f"sat_vp(-80C) = {es.magnitude:.10e} hPa")
except Exception as e:
    print(f"sat_vp(-80C) -> ERROR: {type(e).__name__}: {e}")

# Heat index at extreme values
try:
    hi = mpcalc.heat_index(50 * units.degC, 100 * units.percent)
    mag = hi.to('degC').magnitude
    val = float(mag) if np.ndim(mag) == 0 else float(mag.flat[0])
    print(f"heat_index(50C, 100%RH) = {val:.6f} degC")
except Exception as e:
    print(f"heat_index(50C, 100%RH) -> ERROR: {type(e).__name__}: {e}")

# Windchill at extreme values
try:
    wc = mpcalc.windchill(-50 * units.degC, 30 * units('m/s'))
    print(f"windchill(-50C, 30m/s) = {wc.magnitude:.6f} degC")
except Exception as e:
    print(f"windchill(-50C, 30m/s) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 8. Wind Direction Edge Cases
# ============================================================
section("WIND DIRECTION EDGE CASES")

# Cardinal directions
for u_val, v_val, expected_name in [
    (0, -10, "North wind (from N)"),
    (0, 10, "South wind (from S)"),
    (-10, 0, "East wind (from E)"),
    (10, 0, "West wind (from W)"),
]:
    try:
        wd = mpcalc.wind_direction(u_val * units('m/s'), v_val * units('m/s'))
        print(f"wind_dir(u={u_val:3}, v={v_val:3}) = {wd.magnitude:.6f} deg  [{expected_name}]")
    except Exception as e:
        print(f"wind_dir(u={u_val}, v={v_val}) -> ERROR: {e}")

# Wind direction angles > 360 or < 0 input to wind_components
try:
    u, v = mpcalc.wind_components(10 * units('m/s'), 450 * units.degree)
    print(f"wind_components(10, 450deg) = u={u.magnitude:.6f}, v={v.magnitude:.6f}  [same as 90deg]")
except Exception as e:
    print(f"wind_components(10, 450deg) -> ERROR: {type(e).__name__}: {e}")

try:
    u, v = mpcalc.wind_components(10 * units('m/s'), -90 * units.degree)
    print(f"wind_components(10, -90deg) = u={u.magnitude:.6f}, v={v.magnitude:.6f}")
except Exception as e:
    print(f"wind_components(10, -90deg) -> ERROR: {type(e).__name__}: {e}")

try:
    u, v = mpcalc.wind_components(10 * units('m/s'), 0 * units.degree)
    print(f"wind_components(10, 0deg) = u={u.magnitude:.6f}, v={v.magnitude:.6f}  [from north]")
except Exception as e:
    print(f"wind_components(10, 0deg) -> ERROR: {type(e).__name__}: {e}")

try:
    u, v = mpcalc.wind_components(10 * units('m/s'), 360 * units.degree)
    print(f"wind_components(10, 360deg) = u={u.magnitude:.6f}, v={v.magnitude:.6f}  [same as 0deg]")
except Exception as e:
    print(f"wind_components(10, 360deg) -> ERROR: {type(e).__name__}: {e}")

# Negative wind speed in wind_components
try:
    u, v = mpcalc.wind_components(-10 * units('m/s'), 180 * units.degree)
    print(f"wind_components(-10, 180deg) = u={u.magnitude:.6f}, v={v.magnitude:.6f}")
except Exception as e:
    print(f"wind_components(-10, 180deg) -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 9. Pressure-Height Conversion Edge Cases
# ============================================================
section("PRESSURE-HEIGHT CONVERSION EDGE CASES")

# Standard sea level
try:
    h = mpcalc.pressure_to_height_std(1013.25 * units.hPa)
    print(f"p_to_h(1013.25) = {h.to('m').magnitude:.6f} m  [expected ~0]")
except Exception as e:
    print(f"p_to_h(1013.25) -> ERROR: {e}")

# Very small pressure
try:
    h = mpcalc.pressure_to_height_std(1 * units.hPa)
    print(f"p_to_h(1 hPa) = {h.to('m').magnitude:.6f} m")
except Exception as e:
    print(f"p_to_h(1 hPa) -> ERROR: {e}")

# Very large pressure
try:
    h = mpcalc.pressure_to_height_std(2000 * units.hPa)
    print(f"p_to_h(2000 hPa) = {h.to('m').magnitude:.6f} m  [negative expected]")
except Exception as e:
    print(f"p_to_h(2000 hPa) -> ERROR: {e}")

# ============================================================
# 10. Large Arrays (Performance + Correctness)
# ============================================================
section("LARGE ARRAYS")

# Large array wind_speed
n = 10000
u = np.random.randn(n) * 10
v = np.random.randn(n) * 10
try:
    ws = mpcalc.wind_speed(u * units('m/s'), v * units('m/s'))
    print(f"wind_speed(n={n}): min={ws.magnitude.min():.6f}, max={ws.magnitude.max():.6f}, all_finite={np.all(np.isfinite(ws.magnitude))}")
except Exception as e:
    print(f"wind_speed(n={n}) -> ERROR: {type(e).__name__}: {e}")

# Large array with some NaN
u_nan = u.copy()
v_nan = v.copy()
u_nan[::100] = np.nan  # every 100th element
try:
    ws = mpcalc.wind_speed(u_nan * units('m/s'), v_nan * units('m/s'))
    nan_count = np.sum(np.isnan(ws.magnitude))
    print(f"wind_speed(n={n}, NaN every 100th): nan_count={nan_count}  [expected {n//100}]")
except Exception as e:
    print(f"wind_speed with sparse NaN -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 11. Mismatched Array Lengths
# ============================================================
section("MISMATCHED ARRAY LENGTHS")

try:
    ws = mpcalc.wind_speed(np.array([1, 2, 3]) * units('m/s'), np.array([1, 2]) * units('m/s'))
    print(f"wind_speed([1,2,3], [1,2]) = {ws.magnitude}")
except Exception as e:
    print(f"wind_speed mismatched lengths -> ERROR: {type(e).__name__}: {e}")

try:
    pt = mpcalc.potential_temperature(np.array([1000, 900]) * units.hPa, np.array([25]) * units.degC)
    print(f"pot_temp([1000,900], [25]) = {pt.magnitude}")
except Exception as e:
    print(f"pot_temp mismatched lengths -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 12. Dewpoint and Vapor Pressure Round-Trip
# ============================================================
section("ROUND-TRIP CONSISTENCY")

# dewpoint -> vapor_pressure -> dewpoint
for t_test in [25.0, 0.0, -20.0, -50.0, 40.0]:
    try:
        td = t_test * units.degC
        e = mpcalc.saturation_vapor_pressure(td)
        td_back = mpcalc.dewpoint(e)
        diff = abs(td_back.magnitude - t_test)
        print(f"  T={t_test:6.1f}C -> e={e.magnitude:.6f} hPa -> Td_back={td_back.magnitude:.6f}C  [diff={diff:.2e}]")
    except Exception as e:
        print(f"  T={t_test}C round-trip -> ERROR: {type(e).__name__}: {e}")

# RH round-trip: T,Td -> RH -> T,Td
for t_test, td_test in [(25, 15), (25, 25), (25, -10), (0, -5)]:
    try:
        rh = mpcalc.relative_humidity_from_dewpoint(t_test * units.degC, td_test * units.degC)
        td_back = mpcalc.dewpoint_from_relative_humidity(t_test * units.degC, rh)
        diff = abs(td_back.magnitude - td_test)
        print(f"  T={t_test}C, Td={td_test}C -> RH={rh.magnitude:.6f} -> Td_back={td_back.magnitude:.6f}C  [diff={diff:.2e}]")
    except Exception as e:
        print(f"  T={t_test}C, Td={td_test}C round-trip -> ERROR: {type(e).__name__}: {e}")

# ============================================================
# 13. Exact Numeric Reference Values
# ============================================================
section("EXACT NUMERIC REFERENCE VALUES (for Rust test assertions)")

# These are exact MetPy outputs to use as ground truth in Rust tests
print("\n--- potential_temperature ---")
for p, t in [(1000, 25), (850, 20), (500, -10), (300, -40), (1013.25, 15)]:
    pt = mpcalc.potential_temperature(p * units.hPa, t * units.degC)
    print(f"  pot_temp({p}, {t}C) = {pt.magnitude:.10f} K")

print("\n--- saturation_vapor_pressure ---")
for t in [0, 10, 20, 25, 30, -10, -20, -40]:
    es = mpcalc.saturation_vapor_pressure(t * units.degC)
    print(f"  sat_vp({t}C) = {es.magnitude:.10f} hPa")

print("\n--- relative_humidity_from_dewpoint ---")
for t, td in [(25, 15), (25, 25), (30, 10), (0, -5), (25, -40)]:
    rh = mpcalc.relative_humidity_from_dewpoint(t * units.degC, td * units.degC)
    print(f"  rh({t}C, {td}C) = {rh.magnitude:.10f}")

print("\n--- wind_speed ---")
for u, v in [(3, 4), (0, 0), (10, 0), (0, 10), (-5, -12), (1e-15, 1e-15)]:
    ws = mpcalc.wind_speed(u * units('m/s'), v * units('m/s'))
    print(f"  wind_speed({u}, {v}) = {ws.magnitude:.10f}")

print("\n--- wind_direction ---")
for u, v in [(0, -10), (10, 0), (0, 10), (-10, 0), (5, -5), (-5, 5)]:
    wd = mpcalc.wind_direction(u * units('m/s'), v * units('m/s'))
    print(f"  wind_dir({u}, {v}) = {wd.magnitude:.10f} deg")

print("\n--- pressure_to_height_std ---")
for p in [1013.25, 500, 300, 100, 850, 700, 10]:
    h = mpcalc.pressure_to_height_std(p * units.hPa)
    print(f"  p_to_h({p}) = {h.to('m').magnitude:.6f} m")

print("\n--- wind_components ---")
for spd, d in [(10, 0), (10, 90), (10, 180), (10, 270), (10, 45), (10, 360)]:
    u, v = mpcalc.wind_components(spd * units('m/s'), d * units.degree)
    print(f"  wind_comp({spd}, {d}deg) = u={u.magnitude:.10f}, v={v.magnitude:.10f}")

print("\n--- heat_index ---")
for t, rh in [(35, 80), (40, 10), (25, 50), (32.2, 65), (20, 50)]:
    hi = mpcalc.heat_index(t * units.degC, rh * units.percent)
    mag = hi.to('degC').magnitude
    val = float(mag) if np.ndim(mag) == 0 else float(mag.flat[0])
    print(f"  heat_index({t}C, {rh}%) = {val:.6f} degC")

print("\n--- windchill ---")
for t, ws in [(-10, 10), (-20, 15), (-5, 5), (15, 10), (-10, 1)]:
    wc = mpcalc.windchill(t * units.degC, ws * units('m/s'))
    mag = wc.to('degC').magnitude
    val = float(mag) if np.ndim(mag) == 0 else float(mag.flat[0])
    print(f"  windchill({t}C, {ws}m/s) = {val:.6f} degC")

print("\n\nDone. All edge case behaviors documented.")
