# Your First Sounding Analysis

This tutorial walks you through a complete upper-air sounding analysis. By
the end you will know how to compute every parameter a severe weather
forecaster looks at and understand what the numbers mean.

No meteorology background required. Every function call is explained.

---

## What We're Building

We will take a single atmospheric sounding -- pressure, temperature,
dewpoint, height, wind speed, and wind direction from the surface to the
tropopause -- and extract every useful parameter from it. Our sounding
represents a warm, humid summer afternoon over the southern Great Plains: the
kind of environment that spawns violent supercells.

---

## The Data

```python
import numpy as np
from metrust.calc import (
    potential_temperature, virtual_temperature,
    equivalent_potential_temperature,
    relative_humidity_from_dewpoint,
    mixing_ratio_from_relative_humidity, vapor_pressure,
    lcl, lfc, el, parcel_profile, cape_cin,
    wind_components, bulk_shear, storm_relative_helicity,
    significant_tornado_parameter, supercell_composite_parameter,
)
from metrust.units import units
```

### Pressure, temperature, dewpoint

```python
pressure = np.array([
    1013, 1000, 975, 950, 925, 900, 850, 800,
    750,  700,  650, 600, 550, 500, 400, 300, 250, 200,
]) * units.hPa

temperature = np.array([
    33.0, 32.0, 29.5, 27.0, 25.0, 23.0, 19.0, 14.5,
    10.0,  5.5,  1.0, -4.5, -10.0, -16.0, -30.0, -44.0, -52.0, -58.0,
]) * units.degC

dewpoint = np.array([
    23.0, 22.5, 21.0, 20.0, 18.5, 16.5, 12.0,  5.0,
     0.0, -5.0, -10.0, -17.0, -24.0, -32.0, -44.0, -55.0, -62.0, -70.0,
]) * units.degC
```

Pressure decreases with altitude (1013 hPa at the surface, 200 hPa near
12 km). Temperature drops at roughly 6-7 C/km. The surface dewpoint of 23 C
is oppressively humid, but dewpoints drop quickly above 850 hPa -- moist
boundary layer capped by dry mid-levels, the classic severe weather setup.

### Height and wind

```python
height = np.array([
       0,  100,  350,  600,  850, 1100, 1500, 2000,
    2550, 3100, 3700, 4350, 5050, 5800, 7500, 9500, 10700, 12100,
]) * units.m

wind_spd = np.array([
     8,  10,  12,  14,  16,  18,  22,  26,
    30,  34,  36,  37,  38,  40,  44,  50, 52, 54,
]) * units('m/s')

wind_dir = np.array([
    170, 175, 185, 195, 210, 220, 230, 240,
    245, 250, 255, 258, 260, 262, 265, 268, 270, 272,
]) * units.degree
```

Wind increases from 8 m/s at the surface to 54 m/s (120 mph) aloft, veering
from southerly to westerly. This increasing-speed-and-veering-direction
pattern is **wind shear** -- the key ingredient alongside instability for
supercells and tornadoes.

---

## Step 1: Basic Thermodynamics

### Potential temperature

```python
theta = potential_temperature(pressure, temperature)
print(f"Surface:  {theta[0]:.1f}")   # ~306 K
print(f"500 hPa:  {theta[13]:.1f}")  # ~310 K
```

Potential temperature is what air would measure if brought adiabatically to
1000 hPa. It removes the effect of pressure so you can compare air at
different altitudes. Theta increasing with height means the atmosphere is
stable to dry processes -- but as we will see, moisture changes everything.

### Virtual temperature

```python
tv = virtual_temperature(temperature, pressure, dewpoint=dewpoint)
print(f"Surface T:   {temperature[0]:.1f}")  # 33.0 degC
print(f"Surface T_v: {tv[0]:.1f}")           # ~36.1 degC
```

Moist air is lighter than dry air (water vapor, MW 18, replaces heavier N2
and O2). Virtual temperature adjusts for this. The ~3 C difference at the
surface confirms heavy moisture loading in the boundary layer.

### Equivalent potential temperature

```python
theta_e = equivalent_potential_temperature(pressure, temperature, dewpoint)
print(f"Surface theta-e:  {theta_e[0]:.1f}")   # ~354 K
print(f"500 hPa theta-e:  {theta_e[13]:.1f}")  # ~315 K
```

Theta-e accounts for latent heat released during condensation. The steep
drop from 354 K at the surface to 315 K at 500 hPa tells us the low levels
are loaded with energy from both warmth and moisture, while mid-levels are
relatively cold and dry. When the cap breaks, all that latent heat fuels an
intense updraft.

!!! note "Why theta-e matters"
    A large surface-to-mid-level theta-e difference is one of the clearest
    signals of severe weather potential. It means the atmosphere has a deep
    reservoir of energy waiting to be released.

---

## Step 2: Moisture Profile

### Relative humidity

```python
rh = relative_humidity_from_dewpoint(temperature, dewpoint)
for i in [0, 6, 13]:
    print(f"{pressure[i].magnitude:.0f} hPa:  {rh[i].magnitude * 100:.1f}%")
```

```title="Output"
1013 hPa:  59.4%
850 hPa:   62.6%
500 hPa:   22.3%
```

RH is 60-67% in the boundary layer (moist but not saturated) and drops to
about 22% by 500 hPa. This moist-low-dry-aloft profile is textbook severe
weather.

!!! warning "RH can mislead"
    Cold air with low absolute moisture can show high RH. Use mixing ratio
    for a true measure of water vapor content.

### Mixing ratio

```python
w = mixing_ratio_from_relative_humidity(pressure, temperature, rh)
for i in [0, 6, 13]:
    print(f"{pressure[i].magnitude:.0f} hPa:  {w[i].to('g/kg').magnitude:.1f} g/kg")
```

```title="Output"
1013 hPa:  17.6 g/kg
850 hPa:   8.9 g/kg
500 hPa:   0.7 g/kg
```

The surface mixing ratio of ~18 g/kg is very high -- typical of Gulf of
Mexico air masses that fuel Great Plains severe weather. By 500 hPa it has
dropped to less than 1 g/kg. Almost all the moisture is concentrated in the
lowest 3 km.

### Vapor pressure

```python
e = vapor_pressure(dewpoint)
for i in [0, 6, 13]:
    print(f"{pressure[i].magnitude:.0f} hPa:  {e[i].to('hPa').magnitude:.1f} hPa")
```

```title="Output"
1013 hPa:  28.1 hPa
850 hPa:   14.0 hPa
500 hPa:   0.4 hPa
```

Vapor pressure drops exponentially with height. The surface value of 28 hPa
means nearly 3% of total atmospheric pressure is from water vapor alone.

!!! tip "All three are related"
    RH, mixing ratio, and vapor pressure are different ways to express the
    same thing: how much water vapor is present. Use RH for cloud formation,
    mixing ratio for air-mass tracking, vapor pressure for energy budgets.

---

## Step 3: Find Key Levels

Three critical altitudes determine whether storms form and how strong they
can become.

### LCL -- Lifting Condensation Level (cloud base)

```python
p_lcl, T_lcl = lcl(pressure[0], temperature[0], dewpoint[0])
print(f"LCL: {p_lcl:.0f}  ({T_lcl:.1f})")  # ~881 hPa, ~19.6 C
```

The LCL is where a lifted surface parcel cools to saturation and clouds
form. Ours is about 881 hPa (~1100 m AGL). Low LCLs (below ~1500 m) favor
tornadoes because they keep the rotating updraft close to the ground.

### LFC -- Level of Free Convection (storms become self-sustaining)

```python
p_lfc = lfc(pressure, temperature, dewpoint)
print(f"LFC: {p_lfc:.0f}")  # ~756 hPa
```

Below the LFC, a rising parcel is cooler than its surroundings and needs a
push (cold front, dryline, outflow). Above the LFC, the parcel is buoyant
and accelerates upward on its own. Our LFC near 756 hPa means a moderate
trigger is needed.

### EL -- Equilibrium Level (storm top)

```python
p_el = el(pressure, temperature, dewpoint)
print(f"EL: {p_el:.0f}")  # ~195 hPa
```

The EL is where the parcel cools back to the environment temperature and
loses buoyancy. It marks the anvil level. The LFC-to-EL gap (~756 to
~195 hPa) spans roughly 10 km of buoyant acceleration -- an enormous
updraft. Storm tops would exceed 40,000 feet.

---

## Step 4: Parcel Profile

```python
prof = parcel_profile(pressure, temperature[0], dewpoint[0])
for i in [0, 6, 9, 13, 17]:
    diff = prof[i].magnitude - temperature[i].magnitude
    label = "WARMER" if diff > 0 else "cooler"
    print(f"  {pressure[i].magnitude:6.0f} hPa: env {temperature[i].magnitude:+6.1f} C  "
          f"parcel {prof[i].magnitude:+6.1f} C  ({label})")
```

```title="Output"
  1013 hPa: env  +33.0 C  parcel  +33.0 C  (cooler)
   850 hPa: env  +19.0 C  parcel  +17.3 C  (cooler)
   700 hPa: env   +5.5 C  parcel   +7.2 C  (WARMER)
   500 hPa: env  -16.0 C  parcel  -12.8 C  (WARMER)
   200 hPa: env  -58.0 C  parcel  -57.5 C  (WARMER)
```

The parcel profile traces the temperature a surface parcel would have if
lifted through the sounding. Below the LCL it cools at the dry adiabatic
rate (~10 C/km). Above the LCL, condensation releases latent heat and it
cools more slowly (~6 C/km). The parcel is cooler than the environment in the
boundary layer (CIN zone) but warmer from ~750 hPa all the way to ~195 hPa
(CAPE zone).

---

## Step 5: CAPE and CIN

### Surface-based

```python
sb_cape, sb_cin, sb_lcl_h, sb_lfc_h = cape_cin(
    pressure, temperature, dewpoint, height,
    pressure[0], temperature[0], dewpoint[0],
    parcel_type="sb",
)
print(f"SBCAPE: {sb_cape:.0f}")    # ~2687 J/kg
print(f"CIN:    {sb_cin:.0f}")     # ~-58 J/kg
print(f"LCL:    {sb_lcl_h:.0f}")   # ~1148 m AGL
print(f"LFC:    {sb_lfc_h:.0f}")   # ~2490 m AGL
```

| CAPE (J/kg) | Instability | Typical outcome                         |
|-------------|-------------|------------------------------------------|
| < 300       | Weak        | Showers, no severe weather                |
| 300--1000   | Moderate    | Thunderstorms, small hail possible        |
| 1000--2500  | Strong      | Severe storms, large hail, damaging winds |
| 2500+       | Extreme     | Violent storms, giant hail, tornadoes     |

| CIN (J/kg)   | Cap strength | Meaning                                |
|--------------|-------------|------------------------------------------|
| 0 to -25     | Weak        | Storms fire with minimal trigger          |
| -25 to -100  | Moderate    | Needs a front, dryline, or boundary       |
| < -200       | Very strong | Almost nothing can break through          |

Our CAPE of ~2700 J/kg is extreme. CIN of ~-58 J/kg is a moderate cap --
storms need a trigger, but once it breaks, the atmosphere is loaded.

!!! note "Some CIN is good"
    A moderate cap prevents early, disorganized storms. CAPE builds through
    the day. When the cap finally breaks in late afternoon, the result is
    fewer but much more intense storms.

### Mixed-layer and most-unstable

```python
ml_cape, ml_cin, _, _ = cape_cin(
    pressure, temperature, dewpoint, height,
    pressure[0], temperature[0], dewpoint[0],
    parcel_type="ml",
)
mu_cape, mu_cin, _, _ = cape_cin(
    pressure, temperature, dewpoint, height,
    pressure[0], temperature[0], dewpoint[0],
    parcel_type="mu",
)
print(f"MLCAPE: {ml_cape:.0f}")  # ~2305 J/kg
print(f"MUCAPE: {mu_cape:.0f}")  # ~2780 J/kg
```

!!! tip "Three parcel types"
    - **Surface-based (SB):** Uses the surface observation directly.
    - **Mixed-layer (ML):** Averages the lowest 100 hPa. More representative
      of what a storm actually ingests. Standard for STP.
    - **Most-unstable (MU):** Finds the single most buoyant parcel in the
      lowest 300 hPa. Important for elevated convection.

---

## Step 6: Wind Analysis

### Convert to u/v components

```python
u, v = wind_components(wind_spd, wind_dir)
print(f"Surface:  u={u[0]:.1f}, v={v[0]:.1f}")
print(f"500 hPa:  u={u[13]:.1f}, v={v[13]:.1f}")
```

```title="Output"
Surface:  u=1.4 m/s, v=-7.9 m/s
500 hPa:  u=35.3 m/s, v=-5.6 m/s
```

At the surface, wind is almost entirely from the south (strong -v). Aloft,
u dominates -- nearly westerly. This south-to-west veering is the classic
severe weather hodograph.

### 0-6 km bulk shear

```python
shear_u, shear_v = bulk_shear(u, v, height, bottom=0 * units.m, top=6000 * units.m)
shear_mag = np.sqrt(shear_u.magnitude**2 + shear_v.magnitude**2)
print(f"0-6 km bulk shear: {shear_mag:.1f} m/s")  # ~35 m/s
```

| 0-6 km shear (m/s) | Storm mode                            |
|---------------------|---------------------------------------|
| < 10                | Weak; short-lived pulse storms         |
| 10--20              | Moderate; multicell clusters           |
| 20--30              | Strong; supercells likely              |
| 30+                 | Extreme; long-lived violent supercells |

Our ~35 m/s is extreme. Any storm that forms will almost certainly rotate.

### Storm-relative helicity (SRH)

```python
from metrust.calc import bunkers_storm_motion

(rm_u, rm_v), _, _ = bunkers_storm_motion(u, v, height)
rm_speed = np.sqrt(rm_u.magnitude**2 + rm_v.magnitude**2)
print(f"Bunkers right-mover: {rm_speed:.1f} m/s")

_, _, srh_01 = storm_relative_helicity(u, v, height, 1000 * units.m, rm_u, rm_v)
_, _, srh_03 = storm_relative_helicity(u, v, height, 3000 * units.m, rm_u, rm_v)
print(f"0-1 km SRH: {srh_01:.0f}")  # ~203 m^2/s^2
print(f"0-3 km SRH: {srh_03:.0f}")  # ~347 m^2/s^2
```

SRH measures the corkscrew rotation in the wind profile relative to the
storm. Higher SRH means the updraft ingests more rotation, spinning up a
mesocyclone.

| 0-1 km SRH (m^2/s^2) | Tornado potential                 |
|-----------------------|-----------------------------------|
| < 100                 | Low; tornadoes unlikely            |
| 100--200              | Moderate; weak tornadoes possible  |
| 200--300              | High; significant tornadoes likely |
| 300+                  | Very high; violent tornadoes       |

!!! note "Why the 0-1 km layer matters most for tornadoes"
    Tornadoes are a near-surface phenomenon. Strong low-level SRH means
    the storm can produce rotation right down to the ground. The 0-3 km SRH
    captures the broader mesocyclone, but it is the 0-1 km layer that
    discriminates tornadic from non-tornadic supercells.

---

## Step 7: Severe Weather Composite Parameters

### Significant Tornado Parameter (STP)

```python
stp = significant_tornado_parameter(
    ml_cape,                       # mixed-layer CAPE
    sb_lcl_h,                      # LCL height AGL
    srh_01,                        # 0-1 km SRH
    shear_mag * units('m/s'),      # 0-6 km shear magnitude
)
print(f"STP: {stp:.1f}")  # ~4.6
```

STP combines four ingredients, each normalized by its "significant tornado"
baseline:

1. **MLCAPE / 1500** -- instability
2. **(2000 - LCL) / 1000** -- low cloud base (0 when LCL > 2000 m)
3. **SRH / 150** -- low-level rotation
4. **Shear / 20** -- deep-layer organization

| STP     | Interpretation                                    |
|---------|---------------------------------------------------|
| < 1     | Significant tornadoes unlikely                     |
| 1--3    | Environment supports significant tornadoes         |
| 3--6    | Very favorable; multiple significant tornadoes     |
| 6+      | Extremely favorable; violent (EF4+) possible       |

!!! warning "STP is not a guarantee"
    An STP of 4.6 means the ingredients are strongly favorable. Storms still
    need a trigger, and mesoscale factors determine which cells actually
    produce tornadoes.

### Supercell Composite Parameter (SCP)

```python
scp = supercell_composite_parameter(
    mu_cape,                       # most-unstable CAPE
    srh_03,                        # effective SRH
    shear_mag * units('m/s'),      # effective bulk shear
)
print(f"SCP: {scp:.1f}")  # ~17.0
```

SCP estimates whether storms will be supercells:

1. **MUCAPE / 1000** -- buoyancy
2. **SRH / 50** -- rotation supply
3. **Shear / 20** -- organizational shear

| SCP     | Interpretation                        |
|---------|---------------------------------------|
| < 1     | Supercells unlikely                    |
| 1--4    | Supercells possible                    |
| 4--10   | Supercells very likely                 |
| 10+     | Discrete, intense supercells favored   |

Our SCP of ~17 means any storms that form will be long-lived rotating
supercells.

---

## Putting It All Together

```python
print("=" * 50)
print("  SOUNDING ANALYSIS SUMMARY")
print("=" * 50)
print(f"  Surface T / Td:    {temperature[0].magnitude:.0f} C / {dewpoint[0].magnitude:.0f} C")
print(f"  LCL:  {p_lcl:.0f}  (~{sb_lcl_h:.0f} AGL)")
print(f"  LFC:  {p_lfc:.0f}    EL: {p_el:.0f}")
print(f"  SBCAPE / CIN:      {sb_cape.magnitude:.0f} / {sb_cin.magnitude:.0f} J/kg")
print(f"  MLCAPE:            {ml_cape.magnitude:.0f} J/kg")
print(f"  MUCAPE:            {mu_cape.magnitude:.0f} J/kg")
print(f"  0-6 km shear:      {shear_mag:.1f} m/s")
print(f"  0-1 km SRH:        {srh_01.magnitude:.0f} m^2/s^2")
print(f"  0-3 km SRH:        {srh_03.magnitude:.0f} m^2/s^2")
print(f"  STP: {stp.magnitude:.1f}    SCP: {scp.magnitude:.1f}")
print("=" * 50)
```

**The forecast:** This sounding shows extreme instability (CAPE ~2700 J/kg)
combined with extreme deep-layer shear (35 m/s) and strong low-level
rotation (0-1 km SRH ~200). A moderate cap will focus storm initiation on a
mesoscale trigger. Once storms develop, discrete long-lived supercells are
essentially guaranteed. The STP of 4.6 places this environment well above
the significant tornado threshold -- multiple significant tornadoes are
possible with any supercell that can sustain a low-level mesocyclone.

---

## What to Try Next

- **Kill the moisture.** Drop surface dewpoint from 23 C to 12 C. Watch CAPE
  collapse, LCL height jump, and STP fall below 1.

- **Remove the shear.** Set wind to a constant 10 m/s from 250 degrees at
  all levels. CAPE stays the same, but SRH drops to zero and composites fall
  below 1. This is an "airmass thunderstorm" environment.

- **Strengthen the cap.** Add 5 C to temperatures at 850 and 800 hPa. CIN
  becomes much more negative and storms may not fire at all.

- **Try all parcel types.** Switch between `"sb"`, `"ml"`, and `"mu"` in
  `cape_cin`. Operational forecasters look at all three.

- **Use real data.** Load a radiosonde from the University of Wyoming archive
  or a model GRIB file and run this same pipeline.

!!! tip "Further reading"
    - [Weather 101 for Developers](weather-101.md) -- background concepts
    - [Thermodynamics API](../api/thermodynamics.md) -- full thermo reference
    - [Wind API](../api/wind.md) -- shear, helicity, storm motion
    - [Severe Weather API](../api/severe.md) -- all composite indices
