"""
Compute MetPy reference values for all thermodynamic functions that exist in metrust.

metrust conventions:
  - Pressures: hPa (millibars)
  - Temperatures: Celsius (functions return K where noted, e.g. potential_temperature)
  - Mixing ratio: g/kg
  - Relative humidity: percent (0-100)
  - Specific humidity: kg/kg

MetPy conventions:
  - Uses Pint quantities with explicit units

This script computes reference values using MetPy, then prints them in a format
suitable for hardcoding into Rust integration tests.
"""

import metpy.calc as mpcalc
from metpy.units import units
import numpy as np

print("=" * 70)
print("MetPy Reference Values for metrust verification")
print(f"MetPy version: {__import__('metpy').__version__}")
print("=" * 70)

# ============================================================================
# 1. Potential Temperature (returns K)
#    metrust: potential_temperature(p_hpa, t_c) -> K
# ============================================================================
print("\n--- potential_temperature(p_hpa, t_c) -> K ---")
for p_hpa, t_c in [(1000.0, 25.0), (850.0, 10.0), (500.0, -20.0), (700.0, 5.0)]:
    t_k_input = (t_c + 273.15) * units.K
    result = mpcalc.potential_temperature(p_hpa * units.hPa, t_k_input)
    print(f"  potential_temperature({p_hpa}, {t_c}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 2. Equivalent Potential Temperature (returns K)
#    metrust: equivalent_potential_temperature(p_hpa, t_c, td_c) -> K
#    Uses Bolton (1980) formula
# ============================================================================
print("\n--- equivalent_potential_temperature(p_hpa, t_c, td_c) -> K ---")
for p_hpa, t_c, td_c in [(1000.0, 25.0, 20.0), (850.0, 10.0, 5.0), (1000.0, 30.0, 25.0)]:
    t_k = (t_c + 273.15) * units.K
    td_k = (td_c + 273.15) * units.K
    result = mpcalc.equivalent_potential_temperature(p_hpa * units.hPa, t_k, td_k)
    print(f"  equiv_potential_temp({p_hpa}, {t_c}, {td_c}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 3. Saturation Vapor Pressure (returns hPa)
#    metrust: saturation_vapor_pressure(t_c) -> hPa
#    Bolton (1980) formula
# ============================================================================
print("\n--- saturation_vapor_pressure(t_c) -> hPa ---")
for t_c in [0.0, 10.0, 20.0, 25.0, 30.0, 35.0, -10.0, -20.0]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.saturation_vapor_pressure(t_k)
    print(f"  saturation_vapor_pressure({t_c}) = {result.to('hPa').magnitude:.6f} hPa")

# ============================================================================
# 4. Saturation Mixing Ratio (returns g/kg in metrust, dimensionless in MetPy)
#    metrust: saturation_mixing_ratio(p_hpa, t_c) -> g/kg
# ============================================================================
print("\n--- saturation_mixing_ratio(p_hpa, t_c) -> dimensionless (MetPy), g/kg (metrust) ---")
for p_hpa, t_c in [(1013.25, 20.0), (1000.0, 25.0), (850.0, 10.0), (500.0, -20.0)]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.saturation_mixing_ratio(p_hpa * units.hPa, t_k)
    val = result.to('g/kg').magnitude
    print(f"  saturation_mixing_ratio({p_hpa}, {t_c}) = {val:.6f} g/kg")

# ============================================================================
# 5. Dewpoint from Relative Humidity (returns Celsius)
#    metrust: dewpoint_from_rh(t_c, rh%) -> Celsius
# ============================================================================
print("\n--- dewpoint_from_rh(t_c, rh) -> C ---")
for t_c, rh in [(25.0, 50.0), (25.0, 80.0), (30.0, 40.0), (10.0, 90.0), (20.0, 100.0)]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.dewpoint_from_relative_humidity(t_k, rh / 100.0)
    print(f"  dewpoint_from_rh({t_c}, {rh}) = {result.to('degC').magnitude:.6f} C")

# ============================================================================
# 6. Relative Humidity from Dewpoint (returns %)
#    metrust: rh_from_dewpoint(t_c, td_c) -> %
# ============================================================================
print("\n--- rh_from_dewpoint(t_c, td_c) -> % ---")
for t_c, td_c in [(25.0, 15.0), (30.0, 20.0), (20.0, 20.0), (10.0, -5.0)]:
    t_k = (t_c + 273.15) * units.K
    td_k = (td_c + 273.15) * units.K
    result = mpcalc.relative_humidity_from_dewpoint(t_k, td_k)
    print(f"  rh_from_dewpoint({t_c}, {td_c}) = {result.to('percent').magnitude:.6f} %")

# ============================================================================
# 7. Vapor Pressure (from dewpoint) -> hPa
#    metrust: vapor_pressure_from_dewpoint(td_c) -> hPa
#    In metrust this is just saturation_vapor_pressure(td_c), since e(Td) = es(Td)
# ============================================================================
print("\n--- vapor_pressure(td_c) -> hPa ---")
for td_c in [15.0, 20.0, 25.0, -10.0]:
    td_k = (td_c + 273.15) * units.K
    # vapor_pressure_from_dewpoint = saturation_vapor_pressure(td)
    result = mpcalc.saturation_vapor_pressure(td_k)
    print(f"  vapor_pressure({td_c}) = {result.to('hPa').magnitude:.6f} hPa")

# ============================================================================
# 8. Virtual Temperature (returns Celsius in metrust)
#    metrust: virtual_temp(t_c, p_hpa, td_c) -> Celsius
# ============================================================================
print("\n--- virtual_temperature(t_c, p_hpa, td_c) -> C ---")
for t_c, p_hpa, td_c in [(25.0, 1000.0, 20.0), (20.0, 850.0, 15.0), (30.0, 1013.25, 25.0)]:
    t_k = (t_c + 273.15) * units.K
    # MetPy virtual_temperature takes T and mixing ratio
    # We compute mixing ratio from dewpoint first
    td_k = (td_c + 273.15) * units.K
    e = mpcalc.saturation_vapor_pressure(td_k)
    w = mpcalc.mixing_ratio(e, p_hpa * units.hPa)
    result = mpcalc.virtual_temperature(t_k, w)
    print(f"  virtual_temperature({t_c}, {p_hpa}, {td_c}) = {result.to('degC').magnitude:.6f} C")

# ============================================================================
# 9. Wet Bulb Temperature (returns Celsius)
#    metrust: wet_bulb_temperature(p_hpa, t_c, td_c) -> Celsius
# ============================================================================
print("\n--- wet_bulb_temperature(p_hpa, t_c, td_c) -> C ---")
for p_hpa, t_c, td_c in [(1000.0, 25.0, 15.0), (850.0, 10.0, 5.0), (1013.25, 30.0, 20.0)]:
    t_k = (t_c + 273.15) * units.K
    td_k = (td_c + 273.15) * units.K
    result = mpcalc.wet_bulb_temperature(p_hpa * units.hPa, t_k, td_k)
    print(f"  wet_bulb_temperature({p_hpa}, {t_c}, {td_c}) = {result.to('degC').magnitude:.6f} C")

# ============================================================================
# 10. LCL (returns pressure hPa and temperature Celsius)
#     metrust: lcl(p, t, td) -> (p_lcl, t_lcl) in (hPa, Celsius)
#     (internally uses drylift)
# ============================================================================
print("\n--- lcl(p_hpa, t_c, td_c) -> (p_lcl hPa, t_lcl C) ---")
for p_hpa, t_c, td_c in [(1000.0, 25.0, 15.0), (1013.25, 30.0, 20.0), (850.0, 15.0, 10.0)]:
    t_k = (t_c + 273.15) * units.K
    td_k = (td_c + 273.15) * units.K
    p_lcl, t_lcl = mpcalc.lcl(p_hpa * units.hPa, t_k, td_k)
    print(f"  lcl({p_hpa}, {t_c}, {td_c}) = ({p_lcl.to('hPa').magnitude:.6f} hPa, {t_lcl.to('degC').magnitude:.6f} C)")

# ============================================================================
# 11. Density (returns kg/m^3)
#     metrust: density(p_hpa, t_c, w_gkg) -> kg/m^3
# ============================================================================
print("\n--- density(p_hpa, t_c, w_gkg) -> kg/m^3 ---")
for p_hpa, t_c, w_gkg in [(1013.25, 15.0, 0.0), (1000.0, 25.0, 10.0), (850.0, 5.0, 3.0)]:
    t_k = (t_c + 273.15) * units.K
    w_kgkg = w_gkg / 1000.0
    # MetPy density: rho = p / (Rd * Tv)
    tv = mpcalc.virtual_temperature(t_k, w_kgkg * units('kg/kg'))
    rho = (p_hpa * units.hPa).to('Pa') / (287.058 * units('J/(kg*K)') * tv.to('K'))
    print(f"  density({p_hpa}, {t_c}, {w_gkg}) = {rho.to('kg/m^3').magnitude:.6f} kg/m^3")

# ============================================================================
# 12. Dry Lapse (returns Celsius at each level)
#     metrust: dry_lapse(p_levels, t_surface_c) -> Vec<Celsius>
# ============================================================================
print("\n--- dry_lapse(p_levels, t_surface_c) -> C at each level ---")
p_levels = [1000.0, 900.0, 800.0, 700.0, 500.0]
t_surface_c = 25.0
t_surface_k = (t_surface_c + 273.15) * units.K
result = mpcalc.dry_lapse(
    np.array(p_levels) * units.hPa,
    t_surface_k
)
vals = result.to('degC').magnitude
print(f"  dry_lapse({p_levels}, {t_surface_c}) = {[f'{v:.6f}' for v in vals]}")

# ============================================================================
# 13. Moist Lapse (returns Celsius at each level)
#     metrust: moist_lapse(p_levels, t_start_c) -> Vec<Celsius>
# ============================================================================
print("\n--- moist_lapse(p_levels, t_start_c) -> C at each level ---")
p_levels_arr = np.array([1000.0, 900.0, 800.0, 700.0, 500.0]) * units.hPa
t_start_c = 20.0
t_start_k = (t_start_c + 273.15) * units.K
result = mpcalc.moist_lapse(p_levels_arr, t_start_k)
vals = result.to('degC').magnitude
print(f"  moist_lapse({[1000.0, 900.0, 800.0, 700.0, 500.0]}, {t_start_c}) = {[f'{v:.6f}' for v in vals]}")

# ============================================================================
# 14. Mixing Ratio from RH (returns g/kg in metrust, dimensionless in MetPy)
#     metrust: mixing_ratio_from_relative_humidity(p_hpa, t_c, rh) -> g/kg
# ============================================================================
print("\n--- mixing_ratio_from_relative_humidity(p_hpa, t_c, rh) -> g/kg ---")
for p_hpa, t_c, rh in [(1013.25, 25.0, 50.0), (850.0, 10.0, 80.0), (1000.0, 30.0, 60.0)]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.mixing_ratio_from_relative_humidity(p_hpa * units.hPa, t_k, rh / 100.0)
    val = result.to('g/kg').magnitude
    print(f"  mixing_ratio_from_rh({p_hpa}, {t_c}, {rh}) = {val:.6f} g/kg")

# ============================================================================
# 15. Relative Humidity from Mixing Ratio (returns %)
#     metrust: relative_humidity_from_mixing_ratio(p_hpa, t_c, w_gkg) -> %
# ============================================================================
print("\n--- relative_humidity_from_mixing_ratio(p_hpa, t_c, w_gkg) -> % ---")
for p_hpa, t_c, w_gkg in [(1013.25, 25.0, 10.0), (850.0, 10.0, 5.0), (1000.0, 20.0, 7.0)]:
    t_k = (t_c + 273.15) * units.K
    w_kgkg = w_gkg / 1000.0
    result = mpcalc.relative_humidity_from_mixing_ratio(p_hpa * units.hPa, t_k, w_kgkg * units('kg/kg'))
    print(f"  rh_from_mixing_ratio({p_hpa}, {t_c}, {w_gkg}) = {result.to('percent').magnitude:.6f} %")

# ============================================================================
# 16. Specific Humidity from Mixing Ratio and vice-versa
#     metrust: specific_humidity(p_hpa, w_gkg) -> kg/kg
#     metrust: mixing_ratio_from_specific_humidity(q_kgkg) -> g/kg
# ============================================================================
print("\n--- specific_humidity / mixing_ratio_from_specific_humidity ---")
for w_gkg in [10.0, 5.0, 20.0]:
    w_kgkg = w_gkg / 1000.0
    q = w_kgkg / (1.0 + w_kgkg)  # exact formula
    print(f"  specific_humidity_from_w({w_gkg} g/kg) = {q:.10f} kg/kg")
    w_back = (q / (1.0 - q)) * 1000.0
    print(f"  mixing_ratio_from_q({q:.10f} kg/kg) = {w_back:.6f} g/kg")

# ============================================================================
# 17. Dewpoint from Vapor Pressure (returns Celsius)
#     metrust: dewpoint(e_hpa) -> Celsius
# ============================================================================
print("\n--- dewpoint(vapor_pressure_hpa) -> C ---")
for e_hpa in [10.0, 23.37, 31.67, 6.112]:
    result = mpcalc.dewpoint(e_hpa * units.hPa)
    print(f"  dewpoint({e_hpa}) = {result.to('degC').magnitude:.6f} C")

# ============================================================================
# 18. Temperature from Potential Temperature (inverse Poisson)
#     metrust: temperature_from_potential_temperature(p_hpa, theta_K) -> K
# ============================================================================
print("\n--- temperature_from_potential_temperature(p_hpa, theta_K) -> K ---")
for p_hpa, theta_k in [(1000.0, 300.0), (850.0, 310.0), (500.0, 320.0)]:
    result = mpcalc.temperature_from_potential_temperature(p_hpa * units.hPa, theta_k * units.K)
    print(f"  temp_from_theta({p_hpa}, {theta_k}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 19. Dry Static Energy: DSE = Cp*T + g*z (J/kg)
#     metrust: dry_static_energy(height_m, t_k) -> J/kg
# ============================================================================
print("\n--- dry_static_energy(height_m, t_k) -> J/kg ---")
for h_m, t_k in [(0.0, 288.15), (1000.0, 280.0), (5000.0, 260.0)]:
    result = mpcalc.dry_static_energy(h_m * units.m, t_k * units.K)
    print(f"  dry_static_energy({h_m}, {t_k}) = {result.to('J/kg').magnitude:.6f} J/kg")

# ============================================================================
# 20. Moist Static Energy: MSE = Cp*T + g*z + Lv*q (J/kg)
#     metrust: moist_static_energy(height_m, t_k, q_kgkg) -> J/kg
# ============================================================================
print("\n--- moist_static_energy(height_m, t_k, q_kgkg) -> J/kg ---")
for h_m, t_k, q_kgkg in [(0.0, 288.15, 0.010), (1000.0, 280.0, 0.005), (5000.0, 260.0, 0.001)]:
    result = mpcalc.moist_static_energy(h_m * units.m, t_k * units.K, q_kgkg * units('kg/kg'))
    print(f"  moist_static_energy({h_m}, {t_k}, {q_kgkg}) = {result.to('J/kg').magnitude:.6f} J/kg")

# ============================================================================
# 21. Thickness Hypsometric (returns meters)
#     metrust: thickness_hypsometric(p_bottom, p_top, t_mean_k) -> meters
# ============================================================================
print("\n--- thickness_hypsometric(p_bottom, p_top, t_mean_k) -> m ---")
for p_bot, p_top, t_mean_k in [(1000.0, 500.0, 260.0), (850.0, 500.0, 255.0), (1000.0, 700.0, 275.0)]:
    result = mpcalc.thickness_hydrostatic(
        np.array([p_bot, p_top]) * units.hPa,
        np.array([t_mean_k, t_mean_k]) * units.K
    )
    # MetPy thickness_hydrostatic integrates; for two-level constant T, it equals hypsometric
    # Direct formula: dz = (Rd * T_mean / g) * ln(p_bot / p_top)
    Rd = 287.058
    g = 9.80665
    dz = (Rd * t_mean_k / g) * np.log(p_bot / p_top)
    print(f"  thickness_hypsometric({p_bot}, {p_top}, {t_mean_k}) = {dz:.6f} m")

# ============================================================================
# 22. Pressure to Height (Standard Atmosphere) -> meters
#     metrust: pressure_to_height_std(p_hpa) -> meters
# ============================================================================
print("\n--- pressure_to_height_std(p_hpa) -> m ---")
for p_hpa in [1013.25, 850.0, 700.0, 500.0, 300.0]:
    result = mpcalc.pressure_to_height_std(p_hpa * units.hPa)
    print(f"  pressure_to_height_std({p_hpa}) = {result.to('m').magnitude:.6f} m")

# ============================================================================
# 23. Height to Pressure (Standard Atmosphere) -> hPa
#     metrust: height_to_pressure_std(h_m) -> hPa
# ============================================================================
print("\n--- height_to_pressure_std(h_m) -> hPa ---")
for h_m in [0.0, 1000.0, 3000.0, 5500.0, 9000.0]:
    result = mpcalc.height_to_pressure_std(h_m * units.m)
    print(f"  height_to_pressure_std({h_m}) = {result.to('hPa').magnitude:.6f} hPa")

# ============================================================================
# 24. Virtual Potential Temperature (returns K)
#     metrust: virtual_potential_temperature(p_hpa, t_c, w_gkg) -> K
# ============================================================================
print("\n--- virtual_potential_temperature(p_hpa, t_c, w_gkg) -> K ---")
for p_hpa, t_c, w_gkg in [(1000.0, 25.0, 10.0), (850.0, 10.0, 5.0), (700.0, 0.0, 3.0)]:
    t_k = (t_c + 273.15) * units.K
    w_kgkg = w_gkg / 1000.0
    result = mpcalc.virtual_potential_temperature(p_hpa * units.hPa, t_k, w_kgkg * units('kg/kg'))
    print(f"  virtual_potential_temp({p_hpa}, {t_c}, {w_gkg}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 25. Exner Function: Pi = Cp * (p/p0)^(R/Cp)
#     metrust: exner_function(p_hpa) -> dimensionless (just (p/p0)^(R/Cp))
# ============================================================================
print("\n--- exner_function(p_hpa) -> dimensionless ---")
for p_hpa in [1000.0, 850.0, 500.0, 300.0]:
    result = mpcalc.exner_function(p_hpa * units.hPa)
    # MetPy returns Cp*(p/p0)^kappa in J/(kg*K); metrust returns just (p/p0)^kappa
    val_dimensionless = (p_hpa / 1000.0) ** 0.28571426
    print(f"  exner_function({p_hpa}) = {val_dimensionless:.10f} (dimensionless)")

# ============================================================================
# 26. Geopotential <-> Height
#     metrust: geopotential_to_height(geopot) -> m (Phi/g)
#     metrust: height_to_geopotential(h_m) -> m^2/s^2 (g*h)
# ============================================================================
print("\n--- geopotential_to_height / height_to_geopotential ---")
g = 9.80665
for h_m in [0.0, 1000.0, 5000.0, 10000.0]:
    geopot = g * h_m
    print(f"  height_to_geopotential({h_m}) = {geopot:.6f} m^2/s^2")
    print(f"  geopotential_to_height({geopot}) = {geopot / g:.6f} m")

# ============================================================================
# 27. Scale Height: H = Rd*T/g (meters)
#     metrust: scale_height(t_k) -> meters
# ============================================================================
print("\n--- scale_height(t_k) -> m ---")
Rd = 287.058
for t_k in [250.0, 273.15, 288.15, 300.0]:
    h = Rd * t_k / g
    print(f"  scale_height({t_k}) = {h:.6f} m")

# ============================================================================
# 28. Sigma to Pressure
#     metrust: sigma_to_pressure(sigma, p_sfc, p_top) -> hPa
# ============================================================================
print("\n--- sigma_to_pressure(sigma, p_sfc, p_top) -> hPa ---")
for sigma, p_sfc, p_top in [(0.0, 1013.25, 50.0), (0.5, 1013.25, 50.0), (1.0, 1013.25, 50.0)]:
    p = sigma * (p_sfc - p_top) + p_top
    print(f"  sigma_to_pressure({sigma}, {p_sfc}, {p_top}) = {p:.6f} hPa")

# ============================================================================
# 29. Temperature Conversions
# ============================================================================
print("\n--- temperature conversions ---")
for t_c in [-40.0, 0.0, 20.0, 37.0, 100.0]:
    t_k = t_c + 273.15
    t_f = t_c * 9.0 / 5.0 + 32.0
    print(f"  C_to_K({t_c}) = {t_k:.6f}, C_to_F({t_c}) = {t_f:.6f}")
for t_f in [-40.0, 32.0, 68.0, 98.6, 212.0]:
    t_c = (t_f - 32.0) * 5.0 / 9.0
    print(f"  F_to_C({t_f}) = {t_c:.6f}")

# ============================================================================
# 30. Vertical Velocity conversions
#     metrust: vertical_velocity_pressure(w_ms, p_hpa, t_c) -> Pa/s
#     metrust: vertical_velocity(omega_pas, p_hpa, t_c) -> m/s
# ============================================================================
print("\n--- vertical_velocity_pressure / vertical_velocity ---")
for w_ms, p_hpa, t_c in [(1.0, 1000.0, 15.0), (0.5, 850.0, 5.0), (-1.0, 500.0, -20.0)]:
    t_k = t_c + 273.15
    p_pa = p_hpa * 100.0
    rho = p_pa / (287.058 * t_k)
    omega = -rho * 9.80665 * w_ms
    print(f"  vertical_velocity_pressure({w_ms}, {p_hpa}, {t_c}) = {omega:.6f} Pa/s")
    w_back = -omega / (rho * 9.80665)
    print(f"  vertical_velocity({omega:.6f}, {p_hpa}, {t_c}) = {w_back:.6f} m/s")

# ============================================================================
# 31. Specific Humidity from Dewpoint
#     metrust: specific_humidity_from_dewpoint(p_hpa, td_c) -> kg/kg
# ============================================================================
print("\n--- specific_humidity_from_dewpoint(p_hpa, td_c) -> kg/kg ---")
for p_hpa, td_c in [(1013.25, 20.0), (1000.0, 15.0), (850.0, 5.0)]:
    td_k = (td_c + 273.15) * units.K
    result = mpcalc.specific_humidity_from_dewpoint(p_hpa * units.hPa, td_k)
    print(f"  specific_humidity_from_dewpoint({p_hpa}, {td_c}) = {result.to('kg/kg').magnitude:.10f} kg/kg")

# ============================================================================
# 32. Dewpoint from Specific Humidity
#     metrust: dewpoint_from_specific_humidity(p_hpa, q_kgkg) -> Celsius
# ============================================================================
print("\n--- dewpoint_from_specific_humidity(p_hpa, q) -> C ---")
for p_hpa, q in [(1013.25, 0.010), (1000.0, 0.005), (850.0, 0.003)]:
    result = mpcalc.dewpoint_from_specific_humidity(p_hpa * units.hPa, q * units('kg/kg'))
    print(f"  dewpoint_from_specific_humidity({p_hpa}, {q}) = {result.to('degC').magnitude:.6f} C")

# ============================================================================
# 33. Wet Bulb Potential Temperature (returns K)
#     metrust: wet_bulb_potential_temperature(p_hpa, t_c, td_c) -> K
# ============================================================================
print("\n--- wet_bulb_potential_temperature(p_hpa, t_c, td_c) -> K ---")
for p_hpa, t_c, td_c in [(1000.0, 25.0, 15.0), (850.0, 10.0, 5.0), (1013.25, 30.0, 25.0)]:
    t_k = (t_c + 273.15) * units.K
    td_k = (td_c + 273.15) * units.K
    result = mpcalc.wet_bulb_potential_temperature(p_hpa * units.hPa, t_k, td_k)
    print(f"  wet_bulb_potential_temp({p_hpa}, {t_c}, {td_c}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 34. Saturation Equivalent Potential Temperature (returns K)
#     metrust: saturation_equivalent_potential_temperature(p_hpa, t_c) -> K
#     = equivalent_potential_temperature(p, t, td=t) i.e., assume saturated
# ============================================================================
print("\n--- saturation_equivalent_potential_temperature(p_hpa, t_c) -> K ---")
for p_hpa, t_c in [(1000.0, 25.0), (850.0, 10.0), (500.0, -20.0)]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.saturation_equivalent_potential_temperature(p_hpa * units.hPa, t_k)
    print(f"  sat_equiv_pot_temp({p_hpa}, {t_c}) = {result.to('K').magnitude:.6f} K")

# ============================================================================
# 35. Relative Humidity from Specific Humidity
#     metrust: relative_humidity_from_specific_humidity(p_hpa, t_c, q_kgkg) -> %
# ============================================================================
print("\n--- relative_humidity_from_specific_humidity(p_hpa, t_c, q) -> % ---")
for p_hpa, t_c, q in [(1013.25, 25.0, 0.010), (850.0, 10.0, 0.005), (1000.0, 20.0, 0.008)]:
    t_k = (t_c + 273.15) * units.K
    result = mpcalc.relative_humidity_from_specific_humidity(p_hpa * units.hPa, t_k, q * units('kg/kg'))
    print(f"  rh_from_specific_humidity({p_hpa}, {t_c}, {q}) = {result.to('percent').magnitude:.6f} %")

print("\n" + "=" * 70)
print("Done. All values above are MetPy reference values.")
print("=" * 70)
