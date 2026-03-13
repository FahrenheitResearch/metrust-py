# Common Workflows and Recipes

A collection of copy-paste-ready code snippets for everyday meteorological
analysis with metrust. Each recipe is self-contained -- just change the input
data to match your use case.

All recipes assume `metrust` is installed (`pip install metrust`).

---

## Complete Sounding Analysis

Given pressure, temperature, and dewpoint arrays, compute every standard
thermodynamic quantity a forecaster would want in a single pass.

```python
import numpy as np
from metrust.calc import (
    potential_temperature, equivalent_potential_temperature,
    relative_humidity_from_dewpoint, mixing_ratio,
    lcl, lfc, el, parcel_profile, cape_cin,
    precipitable_water, wet_bulb_temperature,
)
from metrust.units import units

# --- Input sounding ---
p  = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T  = np.array([  30,  24,  18,   6, -14, -42, -58]) * units.degC
Td = np.array([  22,  18,  14,  -2, -24, -48, -64]) * units.degC
hgt = np.array([  0, 750,1500,3000,5500,9000,12000]) * units.m

# Thermodynamic profiles
theta   = potential_temperature(p, T)
theta_e = equivalent_potential_temperature(p, T, Td)
rh      = relative_humidity_from_dewpoint(T, Td)
w       = mixing_ratio(p, T)

# Key levels
p_lcl, T_lcl = lcl(p[0], T[0], Td[0])
p_lfc        = lfc(p, T, Td)
p_el         = el(p, T, Td)

# Parcel profile and CAPE/CIN
prof = parcel_profile(p, T[0], Td[0])
cape, cin, h_lcl, h_lfc = cape_cin(
    p, T, Td, hgt, p[0], T[0], Td[0], parcel_type="sb",
)

# Column moisture
pw = precipitable_water(p, Td)
Tw = wet_bulb_temperature(p[0], T[0], Td[0])

print(f"CAPE: {cape:.0f}   CIN: {cin:.0f}")
print(f"LCL: {p_lcl:.1f}   LFC: {p_lfc:.1f}   EL: {p_el:.1f}")
print(f"PW: {pw:.1f}   Wet bulb: {Tw:.1f}")
print(f"Surface theta: {theta[0]:.1f}   theta-e: {theta_e[0]:.1f}")
print(f"Surface RH: {rh[0]:.2f}   w: {w[0].to('g/kg'):.1f}")
```

!!! tip
    The `parcel_type` parameter in `cape_cin` accepts `"sb"` (surface-based),
    `"ml"` (mixed-layer), and `"mu"` (most-unstable). Operational forecasters
    typically compute all three.

---

## Severe Weather Quick Assessment

Given a sounding with winds, compute every parameter a severe weather
forecaster checks and print a text summary of the threat level.

```python
import numpy as np
from metrust.calc import (
    cape_cin, bulk_shear, storm_relative_helicity,
    bunkers_storm_motion,
    significant_tornado_parameter, supercell_composite_parameter,
)
from metrust.units import units

# --- Input sounding with winds ---
p   = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T   = np.array([  30,  24,  18,   6, -14, -42, -58]) * units.degC
Td  = np.array([  22,  18,  14,  -2, -24, -48, -64]) * units.degC
hgt = np.array([   0, 750,1500,3000,5500,9000,12000]) * units.m
u   = np.array([   2,   8,  14,  20,  28,  34,  36]) * units("m/s")
v   = np.array([  10,  12,   8,   2,  -4,  -6,  -8]) * units("m/s")

# Thermodynamics
cape, cin, h_lcl, h_lfc = cape_cin(
    p, T, Td, hgt, p[0], T[0], Td[0], parcel_type="sb",
)

# Shear
su6, sv6 = bulk_shear(u, v, hgt, bottom=0*units.m, top=6000*units.m)
shear_06 = np.sqrt(su6**2 + sv6**2)

# Storm motion and SRH
(rm_u, rm_v), _, _ = bunkers_storm_motion(u, v, hgt)
_, _, srh_01 = storm_relative_helicity(u, v, hgt, 1000*units.m, rm_u, rm_v)
_, _, srh_03 = storm_relative_helicity(u, v, hgt, 3000*units.m, rm_u, rm_v)

# Composite indices
stp = significant_tornado_parameter(cape, h_lcl, srh_01, shear_06)
scp = supercell_composite_parameter(cape, srh_03, shear_06)

# Threat assessment
print("=== Severe Weather Assessment ===")
print(f"CAPE:          {cape:>8.0f}")
print(f"CIN:           {cin:>8.0f}")
print(f"LCL height:    {h_lcl:>8.0f}")
print(f"0-6 km shear:  {shear_06:>8.1f}")
print(f"0-1 km SRH:    {srh_01:>8.0f}")
print(f"0-3 km SRH:    {srh_03:>8.0f}")
print(f"STP:           {stp:>8.1f}")
print(f"SCP:           {scp:>8.1f}")
print()

if stp.magnitude >= 3:
    print("THREAT: Significant tornadoes likely.")
elif stp.magnitude >= 1:
    print("THREAT: Supercells with tornado potential.")
elif scp.magnitude >= 1:
    print("THREAT: Supercells possible, limited tornado risk.")
elif cape.magnitude >= 1000:
    print("THREAT: Thunderstorms possible, weak shear.")
else:
    print("THREAT: Severe weather unlikely.")
```

---

## Surface Observation Processing

Given surface temperature, dewpoint, pressure, and wind speed, compute
comfort indices and derived quantities useful for station-level analysis.

```python
import numpy as np
from metrust.calc import (
    heat_index, windchill, apparent_temperature,
    relative_humidity_from_dewpoint, wet_bulb_temperature,
    density, mixing_ratio, altimeter_to_station_pressure,
)
from metrust.units import units

# --- Surface observation ---
T  = 35 * units.degC
Td = 24 * units.degC
p  = 1013 * units.hPa
ws = 3 * units("m/s")

# Relative humidity
rh = relative_humidity_from_dewpoint(T, Td)
rh_pct = rh.magnitude * 100

# Comfort indices
hi = heat_index(T, rh_pct)
at = apparent_temperature(T, rh_pct, ws)
Tw = wet_bulb_temperature(p, T, Td)

# Air density (affects aviation density altitude concept)
w  = mixing_ratio(p, Td)
rho = density(p, T, w)

print(f"Temperature:    {T}")
print(f"Dewpoint:       {Td}")
print(f"RH:             {rh_pct:.0f}%")
print(f"Heat index:     {hi:.1f}")
print(f"Apparent temp:  {at:.1f}")
print(f"Wet bulb:       {Tw:.1f}")
print(f"Air density:    {rho:.3f}")
print()

# Cold weather scenario
T_cold  = -10 * units.degC
ws_cold = 8 * units("m/s")
wc = windchill(T_cold, ws_cold)
print(f"Wind chill at {T_cold}, {ws_cold} wind: {wc:.1f}")
```

!!! tip
    `heat_index` expects RH as a percent (0--100). `relative_humidity_from_dewpoint`
    returns a fraction (0--1), so multiply by 100 before passing it to `heat_index`.

---

## Precipitation Type Check

Given a temperature sounding, check the wet-bulb temperature profile to
determine whether precipitation will reach the surface as rain, snow,
sleet, or freezing rain.

```python
import numpy as np
from metrust.calc import wet_bulb_temperature, warm_nose_check
from metrust.units import units

# --- Winter sounding with warm nose aloft ---
p  = np.array([1000, 950, 900, 850, 800, 700, 600, 500]) * units.hPa
T  = np.array([  -1,   0,   2,   3,   1,  -5, -15, -25]) * units.degC
Td = np.array([  -2,  -1,   1,   1,  -1,  -8, -20, -35]) * units.degC

# Wet-bulb temperature profile
Tw = wet_bulb_temperature(p, T, Td)

# Check for a warm nose (above-freezing layer aloft)
has_warm_nose = warm_nose_check(T, p)
sfc_T  = T[0].magnitude
sfc_Tw = Tw[0].magnitude
warm_layer = np.any(Tw.magnitude > 0)

print(f"Surface T: {sfc_T:.1f} C  |  Surface Tw: {sfc_Tw:.1f} C  |  Warm nose: {has_warm_nose}")

if sfc_Tw <= 0 and not warm_layer:
    print("Diagnosis: SNOW -- entire column below freezing wet-bulb")
elif sfc_T <= 0 and has_warm_nose:
    warm_depth = np.sum(Tw.magnitude > 0)
    if warm_depth >= 3:
        print("Diagnosis: FREEZING RAIN -- deep warm nose, surface refreeze")
    else:
        print("Diagnosis: SLEET -- shallow warm nose, partial melt then refreeze")
elif sfc_T > 0:
    print("Diagnosis: RAIN -- above-freezing surface")
else:
    print("Diagnosis: RAIN -- wet-bulb profile above freezing")
```

---

## Batch Process Multiple Soundings

Loop over a list of soundings from different stations or time steps,
compute CAPE for each, and collect results. This pattern is common when
processing observation networks or model time series.

```python
import numpy as np
import time
from metrust.calc import cape_cin
from metrust.units import units

# Shared vertical structure
p   = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
hgt = np.array([   0, 750,1500,3000,5500,9000]) * units.m

# Simulated 12Z soundings from 5 stations
stations = ["OUN", "DDC", "TOP", "SGF", "LBF"]
sfc_temps = [30, 28, 26, 24, 22]
sfc_tds   = [22, 20, 18, 16, 14]
T_data  = [np.array([st, st-6, st-12, st-24, st-44, st-72]) * units.degC for st in sfc_temps]
Td_data = [np.array([sd, sd-2, sd-4,  sd-12, sd-34, sd-58]) * units.degC for sd in sfc_tds]

# Batch processing
start = time.perf_counter()
results = []
for i, stn in enumerate(stations):
    cape, cin, h_lcl, h_lfc = cape_cin(
        p, T_data[i], Td_data[i], hgt,
        p[0], T_data[i][0], Td_data[i][0],
        parcel_type="sb",
    )
    results.append((stn, cape.magnitude, cin.magnitude))

elapsed = time.perf_counter() - start

print(f"{'Station':>8}  {'CAPE (J/kg)':>12}  {'CIN (J/kg)':>12}")
print("-" * 36)
for stn, cape_val, cin_val in results:
    print(f"{stn:>8}  {cape_val:>12.0f}  {cin_val:>12.0f}")
print(f"\nProcessed {len(stations)} soundings in {elapsed*1000:.1f} ms")
```

!!! tip
    For production workflows with hundreds of soundings, metrust's Rust
    backend processes each sounding in microseconds. The bottleneck is usually
    I/O, not computation. Compare this to MetPy, where each sounding can take
    tens of milliseconds in pure Python.

---

## Grid CAPE/CIN Map

Given 3-D arrays of temperature, moisture, pressure, and height from model
output, compute a 2-D CAPE field for every grid column in a single call.
This is one of metrust's flagship features -- it parallelizes the column
calculations across all CPU cores.

```python
import numpy as np
from metrust.calc import compute_cape_cin

# --- Synthetic 3-D model data (replace with real HRRR/RAP/WRF output) ---
nz, ny, nx = 30, 100, 100

pressure_3d   = np.linspace(100000, 20000, nz).reshape(nz,1,1) * np.ones((1,ny,nx))
temperature_c = np.linspace(28, -55, nz).reshape(nz,1,1) * np.ones((1,ny,nx))
qvapor_3d     = np.linspace(0.015, 0.0001, nz).reshape(nz,1,1) * np.ones((1,ny,nx))
height_agl_3d = np.linspace(0, 14000, nz).reshape(nz,1,1) * np.ones((1,ny,nx))

# Add horizontal variation to make it interesting
temperature_c += np.random.normal(0, 2, (nz, ny, nx))

# Surface fields
psfc = np.full((ny, nx), 100000.0)
t2   = np.full((ny, nx), 301.15)    # K
q2   = np.full((ny, nx), 0.015)     # kg/kg

# Compute CAPE/CIN for every column (Rust-parallel)
cape, cin, lcl_hgt, lfc_hgt = compute_cape_cin(
    pressure_3d, temperature_c, qvapor_3d, height_agl_3d,
    psfc, t2, q2,
    parcel_type="mixed_layer",
)

print(f"CAPE grid shape: {cape.shape}")
print(f"CAPE range: {cape.magnitude.min():.0f} to {cape.magnitude.max():.0f} J/kg")
print(f"CIN  range: {cin.magnitude.min():.0f} to {cin.magnitude.max():.0f} J/kg")

# Plot with matplotlib (if desired)
# import matplotlib.pyplot as plt
# plt.pcolormesh(cape.magnitude, cmap="YlOrRd")
# plt.colorbar(label="MLCAPE (J/kg)")
# plt.title("Mixed-Layer CAPE")
# plt.savefig("cape_map.png", dpi=150)
```

!!! tip
    `compute_cape_cin` expects pressure in Pa, temperature in Celsius,
    moisture as mixing ratio in kg/kg, and height in meters AGL. The surface
    temperature `t2` is in Kelvin. These conventions match standard NWP model
    output variables.

---

## Wind Shear Profile

Compute bulk shear at multiple layer depths and storm-relative helicity
at multiple integration depths. This gives a complete picture of the
vertical wind shear structure.

```python
import numpy as np
from metrust.calc import (
    bulk_shear, storm_relative_helicity, bunkers_storm_motion,
)
from metrust.units import units

# --- Wind profile ---
hgt = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 9000]) * units.m
u   = np.array([2,   4,   7,   12,   16,   19,   24,   28,   34,   38]) * units("m/s")
v   = np.array([8,   9,  10,    9,    7,    5,    1,   -2,   -5,   -7]) * units("m/s")

# Storm motion
(rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v) = bunkers_storm_motion(u, v, hgt)

# Bulk shear at multiple depths
shear_layers = [1000, 3000, 6000]
print("=== Bulk Shear Profile ===")
for top in shear_layers:
    su, sv = bulk_shear(u, v, hgt, bottom=0*units.m, top=top*units.m)
    mag = np.sqrt(su**2 + sv**2)
    print(f"  0-{top/1000:.0f} km:  {mag:.1f}")

# SRH at multiple depths
srh_layers = [500, 1000, 3000]
print("\n=== Storm-Relative Helicity ===")
for depth in srh_layers:
    pos, neg, total = storm_relative_helicity(
        u, v, hgt, depth*units.m, rm_u, rm_v,
    )
    print(f"  0-{depth/1000:.1f} km:  {total:.0f}  (pos={pos:.0f}, neg={neg:.0f})")

# Storm motion summary
print(f"\nBunkers right-mover: u={rm_u:.1f}, v={rm_v:.1f}")
print(f"Bunkers left-mover:  u={lm_u:.1f}, v={lm_v:.1f}")
```

---

## Theta-E Analysis for Boundaries

Compute equivalent potential temperature to identify airmass boundaries.
Theta-e gradients mark outflow boundaries, fronts, and drylines -- the
features that trigger thunderstorm initiation.

```python
import numpy as np
from metrust.calc import equivalent_potential_temperature
from metrust.units import units

# --- Surface observations across a boundary ---
# Imagine a west-to-east transect across a cold front
stations   = ["West1", "West2", "Front", "East1", "East2"]
p_sfc      = np.array([1010, 1012, 1014, 1016, 1018]) * units.hPa
T_sfc      = np.array([  32,   30,   24,   18,   16]) * units.degC
Td_sfc     = np.array([  22,   20,   16,   14,   13]) * units.degC

# Theta-e at each station
theta_e = equivalent_potential_temperature(p_sfc, T_sfc, Td_sfc)

# Find the gradient
theta_e_vals = theta_e.magnitude
gradient = np.diff(theta_e_vals)

for i in range(len(stations)):
    print(f"{stations[i]:>8}  T={T_sfc[i].m:.0f}C  Td={Td_sfc[i].m:.0f}C"
          f"  theta-e={theta_e_vals[i]:.1f} K")

# Flag the strongest gradient as the likely boundary location
max_grad_idx = np.argmax(np.abs(gradient))
print(f"\nStrongest theta-e gradient: {gradient[max_grad_idx]:+.1f} K "
      f"between {stations[max_grad_idx]} and {stations[max_grad_idx+1]}")
```

!!! tip
    A theta-e drop of 8--10 K or more across a short distance is a strong
    signal of a boundary (front, outflow, dryline). Storms preferentially
    initiate along these features.

---

## Mixing Height Estimation

Estimate the planetary boundary layer depth by finding where a surface
parcel's potential temperature matches the environmental theta (the
"parcel method" PBL height).

```python
import numpy as np
from metrust.calc import potential_temperature
from metrust.units import units

# --- Afternoon sounding ---
p   = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700]) * units.hPa
T   = np.array([  30,  28,  26,  24,  22,  18,  14,  10,   6]) * units.degC
hgt = np.array([   0, 250, 500, 750,1000,1500,2000,2500,3000]) * units.m

# Environmental theta profile
theta = potential_temperature(p, T)
theta_sfc = theta[0].magnitude
theta_vals = theta.magnitude
hgt_vals = hgt.magnitude

# Find where theta_env first exceeds theta_sfc (PBL top)
above_idx = np.where(theta_vals > theta_sfc + 0.5)[0]

if len(above_idx) > 0:
    idx = above_idx[0]
    if idx > 0:
        frac = ((theta_sfc + 0.5) - theta_vals[idx-1]) / (theta_vals[idx] - theta_vals[idx-1])
        mix_height = hgt_vals[idx-1] + frac * (hgt_vals[idx] - hgt_vals[idx-1])
    else:
        mix_height = hgt_vals[0]
    print(f"Surface theta: {theta_sfc:.1f} K")
    print(f"Mixing height: {mix_height:.0f} m AGL")
else:
    print("Mixing height above sounding top")
```

---

## Unit Conversion Roundtrip

Show how to work with non-standard input units (Fahrenheit, knots, inches
of mercury) and convert them to standard meteorological units for use with
metrust.

```python
import numpy as np
from metrust.units import units
from metrust.calc import (
    heat_index, windchill,
    altimeter_to_station_pressure, relative_humidity_from_dewpoint,
)

# --- Input in non-metric units ---
T_f    = 95 * units.degF
Td_f   = 72 * units.degF
wind_kt = 15 * units.knot
altimeter_inHg = 30.12 * units("inHg")
elevation_ft   = 1200 * units("ft")

# Convert to standard meteorological units
T_c    = T_f.to(units.degC)
Td_c   = Td_f.to(units.degC)
wind_ms = wind_kt.to(units("m/s"))
altimeter_hPa = altimeter_inHg.to(units.hPa)
elevation_m   = elevation_ft.to(units.m)

print("=== Unit Conversions ===")
print(f"Temperature:  {T_f:.1f}  ->  {T_c:.1f}")
print(f"Dewpoint:     {Td_f:.1f}  ->  {Td_c:.1f}")
print(f"Wind:         {wind_kt:.1f}  ->  {wind_ms:.1f}")
print(f"Altimeter:    {altimeter_inHg:.2f}  ->  {altimeter_hPa:.1f}")
print(f"Elevation:    {elevation_ft:.0f}  ->  {elevation_m:.1f}")

# Use the converted values with metrust functions
rh = relative_humidity_from_dewpoint(T_c, Td_c)
stn_p = altimeter_to_station_pressure(altimeter_hPa, elevation_m)
hi = heat_index(T_c, rh.magnitude * 100)

print(f"\nRH:              {rh.magnitude*100:.0f}%")
print(f"Station pressure: {stn_p:.1f}")
print(f"Heat index:       {hi:.1f}  ({hi.to('degF'):.1f})")

# You can also pass non-metric units directly -- metrust auto-converts
hi_direct = heat_index(T_f, rh.magnitude * 100)
print(f"Heat index (direct from F): {hi_direct:.1f}")
```

!!! tip
    Most metrust functions accept Pint Quantities in any compatible unit and
    convert internally. You can pass `degF` directly to `heat_index` without
    manually converting to Celsius first.

---

## Performance: Scalar vs Array

The same calculation done one element at a time in a Python loop versus
vectorized over an array. Always use arrays when possible -- the Rust
backend processes entire arrays in a single FFI call.

```python
import numpy as np
import time
from metrust.calc import potential_temperature
from metrust.units import units

# Generate a large array
n = 100_000
p_arr = np.random.uniform(300, 1000, n) * units.hPa
T_arr = np.random.uniform(-60, 35, n) * units.degC

# --- Vectorized (fast): pass the entire array ---
start = time.perf_counter()
theta_vec = potential_temperature(p_arr, T_arr)
t_vec = time.perf_counter() - start

# --- Scalar loop (slow): process one element at a time ---
start = time.perf_counter()
theta_loop = np.empty(n)
for i in range(n):
    result = potential_temperature(p_arr[i], T_arr[i])
    theta_loop[i] = result.magnitude
t_loop = time.perf_counter() - start

print(f"Array ({n:,} elements): {t_vec*1000:.1f} ms")
print(f"Scalar loop:           {t_loop*1000:.1f} ms")
print(f"Speedup:               {t_loop/t_vec:.0f}x")
print(f"\nResults match: {np.allclose(theta_vec.magnitude, theta_loop)}")
```

!!! tip
    Functions marked "Rust array binding" in the API docs have dedicated
    compiled entry points that process an entire NumPy array in a single
    Rust call with zero per-element Python overhead. Functions using
    `_vec_call` still benefit from Rust speed per element, but pay Python
    dispatch cost per iteration. For the best performance, prefer the
    array-binding functions and always pass arrays rather than looping over
    scalars.

---

## Integrating with Real Data

Brief patterns for loading real-world data and processing it through metrust.

### Loading a CSV of observations

```python
import numpy as np
import csv
from metrust.calc import heat_index, relative_humidity_from_dewpoint
from metrust.units import units

# Suppose you have a CSV with columns: station, temp_c, dewpoint_c
# station,temp_c,dewpoint_c
# KATL,32,24
# KJFK,28,20
# KORD,30,22

stations, temps, dewpoints = [], [], []
with open("observations.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stations.append(row["station"])
        temps.append(float(row["temp_c"]))
        dewpoints.append(float(row["dewpoint_c"]))

T  = np.array(temps) * units.degC
Td = np.array(dewpoints) * units.degC

rh = relative_humidity_from_dewpoint(T, Td)
hi = heat_index(T, rh.magnitude * 100)

for i, stn in enumerate(stations):
    print(f"{stn}: T={T[i].m:.0f}C  Td={Td[i].m:.0f}C  "
          f"RH={rh[i].magnitude*100:.0f}%  HI={hi[i].m:.1f}C")
```

### Working with xarray datasets

```python
import xarray as xr
from metrust.calc import compute_cape_cin

# metrust.xarray provides a passthrough to MetPy's xarray accessor
# for coordinate-aware operations. For raw grid computations, extract
# the .values arrays and pass them directly to the grid-composite
# functions.
ds = xr.open_dataset("hrrr.grib2", engine="cfgrib")

cape, cin, lcl_hgt, lfc_hgt = compute_cape_cin(
    pressure_3d=ds["P"].values + ds["PB"].values,   # Pa
    temperature_c_3d=ds["T"].values - 273.15,        # K -> C
    qvapor_3d=ds["QVAPOR"].values,                   # kg/kg
    height_agl_3d=ds["height_agl"].values,            # m AGL
    psfc=ds["PSFC"].values,                           # Pa
    t2=ds["T2"].values,                               # K
    q2=ds["Q2"].values,                               # kg/kg
    parcel_type="mixed_layer",
)

# Wrap result back into xarray for plotting
ds["CAPE"] = (("south_north", "west_east"), cape.magnitude)
```

### Using metrust.io for NEXRAD Level III files

```python
import numpy as np
from metrust.io import Level3File

# Read a NEXRAD Level III product (native Rust parser, no MetPy needed)
f = Level3File.from_file("KFWS_N0Q_20230501_1200.nids")

print(f"Product code: {f.product_code}")
print(f"Radar:        lat={f.latitude:.3f}, lon={f.longitude:.3f}")
print(f"Time:         {f.volume_time}")
print(f"Grid:         {f.num_radials} radials x {f.num_bins} bins")

# Reshape into a 2-D polar grid
data_2d = f.data.reshape(f.num_radials, f.num_bins)
print(f"Max reflectivity: {np.nanmax(data_2d):.1f} dBZ")
```
