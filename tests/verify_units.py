"""
Verify MetPy/Pint unit conversions to generate reference values for metrust.
"""
from metpy.units import units
import numpy as np

print("=== Temperature conversions ===")
# Celsius to Kelvin
vals = [-40, -20, 0, 15, 20, 25, 30, 37, 100]
for v in vals:
    k = (v * units.degC).to('K').magnitude
    print(f"  {v}°C = {k:.4f} K")

# Celsius to Fahrenheit
for v in vals:
    f = (v * units.degC).to('degF').magnitude
    print(f"  {v}°C = {f:.4f} °F")

# Fahrenheit to Celsius
f_vals = [-40, 0, 32, 68, 98.6, 212]
for v in f_vals:
    c = (v * units.degF).to('degC').magnitude
    print(f"  {v}°F = {c:.4f} °C")

# Kelvin to Celsius
k_vals = [0, 233.15, 273.15, 293.15, 373.15]
for v in k_vals:
    c = (v * units.K).to('degC').magnitude
    print(f"  {v} K = {c:.4f} °C")

# Kelvin to Fahrenheit
for v in k_vals:
    f = (v * units.K).to('degF').magnitude
    print(f"  {v} K = {f:.4f} °F")

# Fahrenheit to Kelvin
for v in f_vals:
    k = (v * units.degF).to('K').magnitude
    print(f"  {v}°F = {k:.4f} K")

print("\n=== Pressure conversions ===")
# hPa to Pa
pressures = [1013.25, 850, 700, 500, 300, 200, 100]
for p in pressures:
    pa = (p * units.hPa).to('Pa').magnitude
    print(f"  {p} hPa = {pa:.2f} Pa")

# inHg to hPa
inhg_vals = [29.92, 30.00, 28.00, 31.00]
for v in inhg_vals:
    hpa = (v * units.inHg).to('hPa').magnitude
    print(f"  {v} inHg = {hpa:.4f} hPa")

# hPa to inHg
for p in [1013.25, 850, 1000]:
    inhg = (p * units.hPa).to('inHg').magnitude
    print(f"  {p} hPa = {inhg:.6f} inHg")

# Pa to hPa
for pa in [101325.0, 85000.0, 50000.0]:
    hpa = (pa * units.Pa).to('hPa').magnitude
    print(f"  {pa} Pa = {hpa:.4f} hPa")

print("\n=== Speed conversions ===")
speeds_ms = [1, 5, 10, 25, 50, 100]
for s in speeds_ms:
    kt = (s * units('m/s')).to('knots').magnitude
    mph = (s * units('m/s')).to('mph').magnitude
    kmh = (s * units('m/s')).to('km/hr').magnitude
    print(f"  {s} m/s = {kt:.6f} kt = {mph:.6f} mph = {kmh:.6f} km/h")

# knots to m/s
for kt in [1, 10, 50, 100, 150]:
    ms = (kt * units.knots).to('m/s').magnitude
    print(f"  {kt} kt = {ms:.6f} m/s")

# mph to m/s
for mph in [1, 60, 100, 150]:
    ms = (mph * units.mph).to('m/s').magnitude
    print(f"  {mph} mph = {ms:.6f} m/s")

print("\n=== Length conversions ===")
lengths_m = [1, 100, 1000, 5280, 10000]
for l in lengths_m:
    ft = (l * units.m).to('feet').magnitude
    mi = (l * units.m).to('miles').magnitude
    km = (l * units.m).to('km').magnitude
    print(f"  {l} m = {ft:.6f} ft = {mi:.6f} mi = {km:.6f} km")

# feet to meters
for ft in [1, 100, 5280, 10000]:
    m = (ft * units.feet).to('m').magnitude
    print(f"  {ft} ft = {m:.6f} m")

# km to miles
for km in [1, 1.609344, 10, 100]:
    mi = (km * units.km).to('miles').magnitude
    print(f"  {km} km = {mi:.6f} mi")

# miles to km
for mi in [1, 10, 26.2, 100]:
    km = (mi * units.miles).to('km').magnitude
    print(f"  {mi} mi = {km:.6f} km")

print("\n=== Mixing ratio conversions ===")
# g/kg to kg/kg and back
for mr in [1, 5, 10, 15, 20]:
    kgkg = (mr * units('g/kg')).to('kg/kg').magnitude
    print(f"  {mr} g/kg = {kgkg:.6f} kg/kg")

# kg/kg to g/kg
for mr in [0.001, 0.005, 0.010, 0.015, 0.020]:
    gkg = (mr * units('kg/kg')).to('g/kg').magnitude
    print(f"  {mr} kg/kg = {gkg:.6f} g/kg")

print("\n=== Roundtrip tests ===")
# Temperature roundtrip C -> K -> C
test_temp = 25.0
k = (test_temp * units.degC).to('K').magnitude
back = (k * units.K).to('degC').magnitude
print(f"  25°C -> {k} K -> {back}°C (diff: {abs(test_temp - back):.10e})")

# Temperature roundtrip C -> F -> C
f = (test_temp * units.degC).to('degF').magnitude
back = (f * units.degF).to('degC').magnitude
print(f"  25°C -> {f}°F -> {back}°C (diff: {abs(test_temp - back):.10e})")

# Pressure roundtrip inHg -> hPa -> inHg
test_p = 29.92
hpa = (test_p * units.inHg).to('hPa').magnitude
back = (hpa * units.hPa).to('inHg').magnitude
print(f"  29.92 inHg -> {hpa} hPa -> {back} inHg (diff: {abs(test_p - back):.10e})")

# Speed roundtrip m/s -> kt -> mph -> m/s
test_s = 50.0
kt = (test_s * units('m/s')).to('knots').magnitude
mph = (kt * units.knots).to('mph').magnitude
ms = (mph * units.mph).to('m/s').magnitude
print(f"  50 m/s -> {kt} kt -> {mph} mph -> {ms} m/s (diff: {abs(test_s - ms):.10e})")

# Length roundtrip m -> ft -> mi -> km -> m
test_l = 5280.0
ft = (test_l * units.m).to('feet').magnitude
mi = (ft * units.feet).to('miles').magnitude
km = (mi * units.miles).to('km').magnitude
m = (km * units.km).to('m').magnitude
print(f"  5280 m -> {ft} ft -> {mi} mi -> {km} km -> {m} m (diff: {abs(test_l - m):.10e})")

# Mixing ratio roundtrip g/kg -> kg/kg -> g/kg
test_mr = 12.5
kgkg = (test_mr * units('g/kg')).to('kg/kg').magnitude
back = (kgkg * units('kg/kg')).to('g/kg').magnitude
print(f"  12.5 g/kg -> {kgkg} kg/kg -> {back} g/kg (diff: {abs(test_mr - back):.10e})")

print("\n=== Edge cases ===")
# Absolute zero
abs_zero_k = 0.0
c = (abs_zero_k * units.K).to('degC').magnitude
f = (abs_zero_k * units.K).to('degF').magnitude
print(f"  0 K = {c:.4f}°C = {f:.4f}°F")

# -40 is the same in C and F
neg40c = (-40.0 * units.degC).to('degF').magnitude
neg40f = (-40.0 * units.degF).to('degC').magnitude
print(f"  -40°C = {neg40c:.4f}°F (should be -40)")
print(f"  -40°F = {neg40f:.4f}°C (should be -40)")

# Boiling point
boil_f = (100.0 * units.degC).to('degF').magnitude
boil_k = (100.0 * units.degC).to('K').magnitude
print(f"  100°C = {boil_f:.4f}°F = {boil_k:.4f} K")

# Zero speed
zero_kt = (0.0 * units('m/s')).to('knots').magnitude
zero_mph = (0.0 * units('m/s')).to('mph').magnitude
print(f"  0 m/s = {zero_kt:.6f} kt = {zero_mph:.6f} mph")

# Zero pressure
zero_pa = (0.0 * units.hPa).to('Pa').magnitude
print(f"  0 hPa = {zero_pa:.2f} Pa")

# Standard atmosphere
std_atm_pa = (1.0 * units.atm).to('Pa').magnitude
std_atm_hpa = (1.0 * units.atm).to('hPa').magnitude
std_atm_inhg = (1.0 * units.atm).to('inHg').magnitude
print(f"  1 atm = {std_atm_pa:.2f} Pa = {std_atm_hpa:.4f} hPa = {std_atm_inhg:.6f} inHg")

print("\nDone! All reference values generated.")
