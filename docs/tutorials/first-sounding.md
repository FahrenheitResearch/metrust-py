# Your First Sounding Analysis

This tutorial walks you through a complete upper-air sounding analysis from
scratch. By the end, you will understand how meteorologists diagnose severe
weather potential from a single atmospheric profile, and you will know how to
use metrust to do it yourself.

No meteorology background is assumed. Every line of code is explained.

---

## What is a sounding?

A **sounding** is a vertical snapshot of the atmosphere. Twice a day, weather
offices around the world launch helium-filled balloons carrying instrument
packages called **radiosondes**. As the balloon rises from the surface to the
upper atmosphere (~20 km), it measures temperature, humidity, wind speed, and
wind direction at many altitudes. The result is a column of data that tells us
everything about the air above a single location at a single moment in time.

Meteorologists use soundings to answer questions like:

- Will thunderstorms form today?
- If storms form, could they produce tornadoes or large hail?
- How tall will the storm clouds be?
- How fast and in what direction will storms move?

This tutorial shows you how to answer all of those questions with metrust.

---

## Setup

```python
import numpy as np
from metrust.calc import (
    potential_temperature, dewpoint_from_relative_humidity,
    lcl, lfc, el, cape_cin, parcel_profile,
    mixing_ratio, precipitable_water,
    wet_bulb_temperature, virtual_temperature,
    bulk_shear, storm_relative_helicity, bunkers_storm_motion,
    significant_tornado_parameter, supercell_composite_parameter,
    wind_speed, wind_direction,
    pressure_to_height_std,
)
from metrust.units import units
```

**What each import does:**

- `numpy` provides the array type we use for profile data.
- Functions from `metrust.calc` are the meteorological calculations, each
  backed by a compiled Rust engine so they run fast.
- `units` is a Pint unit registry. It lets us attach physical units to numbers
  (e.g., `1000 * units.hPa`), which prevents unit-mismatch bugs and makes the
  code self-documenting. When you pass a Quantity to a metrust function, it
  automatically converts to whatever the Rust backend expects.

---

## Step 1: Create the sounding data

We will build a sounding by hand that represents a warm, humid summer afternoon
in the Great Plains -- the kind of day that produces violent thunderstorms. In
practice you would load this from a radiosonde file or a model forecast, but
typed-out arrays let us see exactly what every number means.

### Pressure levels

```python
p = np.array([1000, 950, 900, 850, 800, 750, 700, 650,
              600, 550, 500, 450, 400, 350, 300, 250, 200]) * units.hPa
```

Pressure decreases with altitude. The surface is around 1000 hPa (also called
millibars). Each successive value is a higher altitude. By the time we reach
200 hPa, we are near the tropopause -- roughly 12 km (39,000 feet) above the
ground, around the cruising altitude of a commercial jet.

The unit `hPa` (hectopascals) is standard in meteorology. 1013.25 hPa is
average sea-level pressure.

### Temperature

```python
T = np.array([32, 28, 24, 20, 17, 14, 10, 6,
              2, -3, -8, -15, -22, -30, -38, -48, -58]) * units.degC
```

Temperature generally drops as you go up. Our surface temperature is 32 C
(about 90 F) -- a hot day. By 200 hPa it has fallen to -58 C. The rate at
which temperature drops with altitude matters enormously: a faster drop means
the atmosphere is more **unstable**, which favors stronger thunderstorms.

### Dewpoint temperature

```python
Td = np.array([24, 22, 19, 15, 10, 5, 1, -4,
               -10, -16, -22, -30, -38, -46, -54, -62, -72]) * units.degC
```

The **dewpoint** tells you how much moisture the air contains. It is the
temperature to which air must be cooled (at constant pressure) for water vapor
to begin condensing. A higher dewpoint means more moisture.

Our surface dewpoint is 24 C (75 F), which is oppressively humid. Notice that
the dewpoint is always less than or equal to the temperature -- the gap between
them (called the **dewpoint depression**) tells you how far the air is from
saturation. A small gap means the air is nearly saturated; a large gap means it
is dry. Near the surface our gap is only 8 C, confirming a very moist boundary
layer.

### Wind components

```python
u = np.array([5, 8, 12, 15, 18, 20, 22, 25,
              27, 28, 30, 32, 33, 35, 38, 40, 42]) * units('m/s')
v = np.array([-2, 0, 3, 5, 7, 8, 8, 7,
              5, 3, 0, -3, -5, -7, -8, -8, -7]) * units('m/s')
```

Wind is split into two components:

- **u**: the east-west component. Positive means the wind is blowing *toward
  the east*.
- **v**: the north-south component. Positive means the wind is blowing *toward
  the north*.

At the surface, winds are light (about 5 m/s from the south-southeast). Aloft,
they increase dramatically and veer (shift clockwise with height). By 200 hPa,
the u-component is 42 m/s -- nearly 95 mph. This increase in wind speed with
height is called **wind shear**, and it is one of the most important
ingredients for supercell thunderstorms and tornadoes.

### Height profile

Many metrust functions need a height array (meters above ground level). We will
estimate heights from pressure using the US Standard Atmosphere:

```python
height = np.array([pressure_to_height_std(pi).magnitude for pi in p]) * units.m
height = height - height[0]  # convert to Above Ground Level (AGL)
```

This gives us approximate AGL heights at each pressure level. The surface is
0 m, and 200 hPa is roughly 12 km AGL.

---

## Step 2: Basic thermodynamics

Before diving into severe weather analysis, let's compute some fundamental
properties of this atmosphere.

### Potential temperature

```python
theta = potential_temperature(p, T)
print(f"Potential temperature at surface: {theta[0]:.1f}")
print(f"Potential temperature at 500 hPa: {theta[10]:.1f}")
```

**What is potential temperature?** Imagine taking a parcel of air from some
level in the atmosphere and compressing it (or expanding it) until it reaches
1000 hPa, without adding or removing any heat. The temperature it would have
at 1000 hPa is its potential temperature, measured in Kelvin (K).

Why bother? Because potential temperature is *conserved* during dry adiabatic
processes. If you see potential temperature increasing with height, the
atmosphere is stable (air resists being displaced vertically). If it decreases
with height, the atmosphere is absolutely unstable. Our sounding should show
potential temperature increasing with height -- the atmosphere is not
spontaneously overturning, but as we will see, lifting the very moist surface
air will unlock enormous energy.

### Mixing ratio

```python
w = mixing_ratio(p, T)
print(f"Surface mixing ratio: {w[0].to('g/kg'):.1f}")
print(f"500 hPa mixing ratio: {w[10].to('g/kg'):.1f}")
```

The **mixing ratio** is the mass of water vapor per mass of dry air. It is
usually expressed in grams per kilogram (g/kg). Our surface mixing ratio should
be quite high -- around 20 g/kg -- reflecting the humid boundary layer. By 500
hPa, the air is much drier.

### Precipitable water

```python
pw = precipitable_water(p, Td)
print(f"Precipitable water: {pw:.1f}")
```

**Precipitable water** answers the question: "If you squeezed every last drop
of moisture out of this entire atmospheric column and collected it in a
bucket, how deep would the water be?" It is measured in millimeters.

Typical values range from 5 mm in cold, dry arctic air to 70+ mm in tropical
air masses. A precipitable water value above about 40 mm in the Great Plains
signals an extremely moist atmosphere capable of producing heavy rainfall.

### Wet-bulb temperature

```python
Tw = wet_bulb_temperature(p[0], T[0], Td[0])
print(f"Surface wet-bulb temperature: {Tw:.1f}")
```

The **wet-bulb temperature** is the lowest temperature air can reach through
evaporative cooling alone. Think of it as the temperature a wet thermometer
reads when you wave it around. It is always between the dewpoint and the air
temperature.

Wet-bulb temperature matters for human health (dangerous heat stress occurs
above ~35 C wet bulb) and for severe weather (it influences hail melting and
cold pool strength in thunderstorms).

---

## Step 3: Find the key levels

When meteorologists look at a sounding, they immediately search for three
critical altitudes. These levels determine whether storms will form and how
strong they will be.

### LCL -- Lifting Condensation Level

```python
p_lcl, T_lcl = lcl(p[0], T[0], Td[0])
h_lcl = pressure_to_height_std(p_lcl) - pressure_to_height_std(p[0])
print(f"LCL pressure:    {p_lcl:.1f}")
print(f"LCL temperature: {T_lcl:.1f}")
print(f"LCL height AGL:  ~{h_lcl.to('m'):.0f}")
```

The **Lifting Condensation Level** is where clouds form. If you take a parcel
of surface air and push it upward, it cools as it expands (about 10 C per
kilometer). At some point it cools enough that the water vapor in it begins to
condense into tiny droplets -- that is the LCL. It marks the base of cumulus
clouds.

Our sounding has a surface temperature-dewpoint spread of 8 C (32 minus 24),
so the LCL should be roughly 800-900 meters above the ground. Low LCLs (below
~1500 m) are important for tornadoes because they keep the storm's rotating
updraft close to the surface.

### LFC -- Level of Free Convection

```python
p_lfc = lfc(p, T, Td)
print(f"LFC pressure: {p_lfc:.1f}")
```

The **Level of Free Convection** is where a storm becomes self-sustaining.
Below the LFC, a rising parcel is cooler than the surrounding air, so it needs
an external push to keep rising (from a cold front, a dryline, or terrain, for
example). Above the LFC, the parcel is warmer than its surroundings and
accelerates upward on its own -- it is **buoyant**.

A lower LFC (higher pressure) means less energy is needed to initiate storms.
If the LFC is very high (low pressure), storms may not form even if the
atmosphere is otherwise favorable, because nothing can push air parcels high
enough.

### EL -- Equilibrium Level

```python
p_el = el(p, T, Td)
print(f"EL pressure: {p_el:.1f}")
```

The **Equilibrium Level** is where the rising parcel finally becomes cooler
than the environment again, losing its buoyancy. This marks approximately the
top of the thunderstorm cloud.

The height difference between the LFC and EL determines the depth over which
the storm's updraft is accelerating. A large LFC-to-EL gap means a deep,
powerful updraft. On our sounding, the EL should be near the tropopause
(around 200-250 hPa), meaning storms would punch up to 40,000 feet or higher.

---

## Step 4: Compute CAPE and CIN

These are the two most important numbers on a sounding for severe weather
forecasting.

### Generate a parcel profile

```python
prof = parcel_profile(p, T[0], Td[0])
```

The **parcel profile** traces the temperature a surface air parcel would have
if lifted through the entire atmosphere. Below the LCL it cools at the dry
adiabatic rate (~10 C/km). Above the LCL, condensation releases latent heat,
so it cools more slowly along the moist adiabat (~6 C/km, varying with
temperature).

### Compute CAPE and CIN

```python
cape_val, cin_val, h_lcl_agl, h_lfc_agl = cape_cin(
    p, T, Td, height,
    p[0], T[0], Td[0],
    parcel_type="sb",
)
print(f"CAPE: {cape_val:.0f}")
print(f"CIN:  {cin_val:.0f}")
```

**CAPE** (Convective Available Potential Energy, in J/kg) measures the total
energy available to a rising parcel between the LFC and the EL. It is the area
on a thermodynamic diagram where the parcel is warmer than the environment.
More CAPE means a stronger updraft.

Here is a rough guide to CAPE values:

| CAPE (J/kg)  | Instability  | What to expect                          |
|--------------|--------------|-----------------------------------------|
| < 300        | Weak         | Showers possible, no severe weather     |
| 300 -- 1000  | Moderate     | Thunderstorms, small hail possible      |
| 1000 -- 2500 | Strong       | Severe storms, large hail, damaging wind|
| 2500+        | Extreme      | Violent storms, tornadoes, giant hail   |

**CIN** (Convective Inhibition, in J/kg) is the energy barrier a parcel must
overcome to reach the LFC. Think of it as a "cap" or lid on the atmosphere.
CIN is reported as a negative number. A CIN of -50 J/kg means storms need a
moderate trigger (a front, outflow boundary, or terrain-forced lifting) to
break through the cap. A CIN of -200 J/kg means the cap is very strong and
storms are unlikely unless a powerful forcing mechanism is present.

Interestingly, some CIN is actually *good* for severe weather. A moderate cap
prevents storms from firing too early in the day. The atmosphere continues to
heat and moisten, building up even more CAPE. When the cap finally breaks --
often in late afternoon -- the result is fewer but much more intense storms.

---

## Step 5: Wind analysis

Wind shear is what separates ordinary thunderstorms from supercells and
tornadoes. An ordinary storm has its updraft and downdraft stacked on top of
each other, so the downdraft chokes off the updraft within 30-60 minutes. A
supercell lives in a strongly sheared environment, which tilts the updraft away
from the downdraft, allowing the storm to persist for hours.

### Wind speed and direction

```python
sfc_speed = wind_speed(u[0:1], v[0:1])
sfc_dir = wind_direction(u[0:1], v[0:1])
print(f"Surface wind: {sfc_speed[0]:.1f} from {sfc_dir[0]:.0f}")

idx_500 = 10  # 500 hPa is the 11th level (index 10)
spd_500 = wind_speed(u[idx_500:idx_500+1], v[idx_500:idx_500+1])
dir_500 = wind_direction(u[idx_500:idx_500+1], v[idx_500:idx_500+1])
print(f"500 hPa wind: {spd_500[0]:.1f} from {dir_500[0]:.0f}")
```

`wind_speed` computes the magnitude from the u and v components:
`speed = sqrt(u^2 + v^2)`. `wind_direction` returns the meteorological
convention -- the direction the wind is blowing *from*, where 0/360 is north
and 90 is east.

### 0-6 km bulk shear

```python
shear_u, shear_v = bulk_shear(u, v, height, 0 * units.m, top=6000 * units.m)
shear_mag = np.sqrt(shear_u.magnitude**2 + shear_v.magnitude**2)
print(f"0-6 km bulk shear: {shear_mag:.1f} m/s")
```

**Bulk shear** is the vector difference between the wind at the top and bottom
of a layer. The 0-6 km bulk shear is the single most important shear parameter
for severe weather forecasting.

| 0-6 km shear (m/s) | Significance                              |
|---------------------|-------------------------------------------|
| < 10                | Weak shear; ordinary pulse storms         |
| 10 -- 20            | Moderate; organized multicell storms      |
| 20 -- 30            | Strong; supercells likely                 |
| 30+                 | Extreme; long-lived violent supercells    |

### Bunkers storm motion

```python
(rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v) = bunkers_storm_motion(u, v, height)
rm_speed = np.sqrt(rm_u.magnitude**2 + rm_v.magnitude**2)
print(f"Right-mover storm motion: u={rm_u:.1f}, v={rm_v:.1f} ({rm_speed:.1f} m/s)")
print(f"Mean wind:                u={mw_u:.1f}, v={mw_v:.1f}")
```

Supercells do not move with the mean wind. They deviate to the right (in the
Northern Hemisphere) because of the interaction between the updraft and the
environmental wind shear. The **Bunkers method** estimates where a right-moving
supercell and a left-moving supercell would travel, based on the 0-6 km mean
wind and shear vector.

The right-mover estimate is used as the storm motion when computing
storm-relative helicity (next).

### Storm-relative helicity (SRH)

```python
pos_1, neg_1, srh_0_1 = storm_relative_helicity(
    u, v, height, 1000 * units.m, rm_u, rm_v
)
pos_3, neg_3, srh_0_3 = storm_relative_helicity(
    u, v, height, 3000 * units.m, rm_u, rm_v
)
print(f"0-1 km SRH: {srh_0_1:.0f}")
print(f"0-3 km SRH: {srh_0_3:.0f}")
```

**Storm-relative helicity** measures the amount of streamwise (corkscrew)
rotation in the low-level wind relative to the storm's motion. It is the
parameter most directly linked to tornadoes.

Think of it this way: SRH measures how much the wind "spirals" into the storm.
Higher SRH means the storm's updraft ingests more horizontal rotation, which it
tilts into the vertical to create a mesocyclone (the rotating updraft of a
supercell).

| 0-1 km SRH (m^2/s^2) | Tornado potential                  |
|-----------------------|------------------------------------|
| < 100                 | Low; tornadoes unlikely            |
| 100 -- 200            | Moderate; weak tornadoes possible  |
| 200 -- 300            | High; significant tornadoes        |
| 300+                  | Very high; violent tornadoes       |

The 0-1 km layer is especially critical because tornadoes are a near-surface
phenomenon. Strong low-level SRH signals that the storm can produce rotation
right down to the ground.

---

## Step 6: Severe weather indices

Severe weather composite indices combine multiple parameters into a single
number that estimates the overall threat. They are widely used in operational
forecasting.

### Significant Tornado Parameter (STP)

```python
stp = significant_tornado_parameter(
    cape_val,                  # CAPE (J/kg)
    h_lcl_agl,                # LCL height (m AGL)
    srh_0_1,                  # 0-1 km SRH (m^2/s^2)
    shear_mag * units('m/s'), # 0-6 km bulk shear (m/s)
)
print(f"STP: {stp:.1f}")
```

The **Significant Tornado Parameter** (STP) combines four ingredients:

1. **CAPE** -- fuel for the storm (normalized by 1500 J/kg)
2. **LCL height** -- low cloud bases favor tornadoes (0 when LCL > 2000 m)
3. **0-1 km SRH** -- low-level rotation (normalized by 150 m^2/s^2)
4. **0-6 km shear** -- storm organization (normalized by 20 m/s)

An STP of 1.0 means all four ingredients are at their "significant tornado"
threshold values simultaneously. Higher values indicate more favorable
conditions.

| STP     | Interpretation                              |
|---------|---------------------------------------------|
| < 1     | Significant tornadoes unlikely               |
| 1 -- 3  | Conditions favorable for significant tornadoes|
| 3 -- 6  | Conditions very favorable                    |
| 6+      | Extremely favorable; violent tornadoes possible|

### Supercell Composite Parameter (SCP)

```python
scp = supercell_composite_parameter(
    cape_val,                  # CAPE (J/kg)
    srh_0_3,                  # 0-3 km SRH (effective SRH; m^2/s^2)
    shear_mag * units('m/s'), # effective bulk shear (m/s)
)
print(f"SCP: {scp:.1f}")
```

The **Supercell Composite Parameter** (SCP) estimates the likelihood that any
storm that does form will be a supercell (rotating thunderstorm). It combines:

1. **CAPE** (normalized by 1000 J/kg)
2. **Effective SRH** (normalized by 50 m^2/s^2)
3. **Effective bulk shear** (normalized by 20 m/s)

| SCP     | Interpretation                       |
|---------|--------------------------------------|
| < 1     | Supercells unlikely                  |
| 1 -- 4  | Supercells possible                  |
| 4 -- 10 | Supercells very likely               |
| 10+     | Discrete, intense supercells favored |

### Putting it all together

Now we have all the pieces to write a plain-English summary of this sounding:

```python
# Interpret CAPE
if cape_val.magnitude < 300:
    cape_label = "weak"
elif cape_val.magnitude < 1000:
    cape_label = "moderate"
elif cape_val.magnitude < 2500:
    cape_label = "strong"
else:
    cape_label = "extreme"

# Determine verdict
if stp.magnitude >= 3 and scp.magnitude >= 4:
    verdict = "Conditions are favorable for significant severe weather, including violent tornadoes."
elif stp.magnitude >= 1 and scp.magnitude >= 1:
    verdict = "Conditions are favorable for supercells and significant tornadoes."
elif scp.magnitude >= 1:
    verdict = "Conditions favor supercell thunderstorms with large hail and damaging winds."
elif cape_val.magnitude >= 1000:
    verdict = "Conditions favor strong thunderstorms, but lack of shear limits tornado potential."
else:
    verdict = "Severe weather is unlikely."

print()
print(f"This sounding shows a warm, moist boundary layer with {cape_label} "
      f"instability ({cape_val.magnitude:.0f} J/kg of CAPE), strong deep-layer "
      f"shear of {shear_mag:.1f} m/s, and 0-1 km SRH of "
      f"{srh_0_1.magnitude:.0f} m^2/s^2.")
print(f"The STP of {stp.magnitude:.1f} and SCP of {scp.magnitude:.1f} suggest:")
print(f"  {verdict}")
```

---

## Step 7: Summary table

Finally, let's print a clean summary of everything we computed. This is the
kind of output a forecaster would glance at to quickly assess the day's severe
weather potential.

```python
print()
print("=" * 40)
print("  SOUNDING ANALYSIS SUMMARY")
print("=" * 40)
print(f"  Surface T/Td:   {T[0].magnitude:.1f} C / {Td[0].magnitude:.1f} C")
print(f"  LCL:            {p_lcl:.0f} (~{h_lcl.to('m'):.0f} AGL)")
print(f"  LFC:            {p_lfc:.0f}")
print(f"  EL:             {p_el:.0f}")
print(f"  CAPE:           {cape_val.magnitude:.0f} J/kg ({cape_label})")
print(f"  CIN:            {cin_val.magnitude:.0f} J/kg")
print(f"  PW:             {pw:.1f}")
print(f"  0-6 km Shear:   {shear_mag:.1f} m/s")
print(f"  0-1 km SRH:     {srh_0_1.magnitude:.0f} m^2/s^2")
print(f"  0-3 km SRH:     {srh_0_3.magnitude:.0f} m^2/s^2")
print(f"  Bunkers RM:     {rm_speed:.1f} m/s")
print(f"  STP:            {stp.magnitude:.1f}")
print(f"  SCP:            {scp.magnitude:.1f}")
print(f"  Verdict:        {verdict}")
print("=" * 40)
```

---

## Complete script

Here is the entire analysis as a single copy-paste-ready script:

```python
import numpy as np
from metrust.calc import (
    potential_temperature, dewpoint_from_relative_humidity,
    lcl, lfc, el, cape_cin, parcel_profile,
    mixing_ratio, precipitable_water,
    wet_bulb_temperature, virtual_temperature,
    bulk_shear, storm_relative_helicity, bunkers_storm_motion,
    significant_tornado_parameter, supercell_composite_parameter,
    wind_speed, wind_direction,
    pressure_to_height_std,
)
from metrust.units import units

# ── Step 1: Sounding data ──────────────────────────────────────────────

p = np.array([1000, 950, 900, 850, 800, 750, 700, 650,
              600, 550, 500, 450, 400, 350, 300, 250, 200]) * units.hPa

T = np.array([32, 28, 24, 20, 17, 14, 10, 6,
              2, -3, -8, -15, -22, -30, -38, -48, -58]) * units.degC

Td = np.array([24, 22, 19, 15, 10, 5, 1, -4,
               -10, -16, -22, -30, -38, -46, -54, -62, -72]) * units.degC

u = np.array([5, 8, 12, 15, 18, 20, 22, 25,
              27, 28, 30, 32, 33, 35, 38, 40, 42]) * units('m/s')

v = np.array([-2, 0, 3, 5, 7, 8, 8, 7,
              5, 3, 0, -3, -5, -7, -8, -8, -7]) * units('m/s')

# Estimate height AGL from pressure using the standard atmosphere
height = np.array([pressure_to_height_std(pi).magnitude for pi in p]) * units.m
height = height - height[0]


# ── Step 2: Basic thermodynamics ───────────────────────────────────────

theta = potential_temperature(p, T)
print(f"Potential temperature at surface: {theta[0]:.1f}")
print(f"Potential temperature at 500 hPa: {theta[10]:.1f}")

w = mixing_ratio(p, T)
print(f"Surface mixing ratio: {w[0].to('g/kg'):.1f}")
print(f"500 hPa mixing ratio: {w[10].to('g/kg'):.1f}")

pw = precipitable_water(p, Td)
print(f"Precipitable water: {pw:.1f}")

Tw = wet_bulb_temperature(p[0], T[0], Td[0])
print(f"Surface wet-bulb temperature: {Tw:.1f}")


# ── Step 3: Key levels ────────────────────────────────────────────────

p_lcl, T_lcl = lcl(p[0], T[0], Td[0])
h_lcl = pressure_to_height_std(p_lcl) - pressure_to_height_std(p[0])
print(f"\nLCL pressure:    {p_lcl:.1f}")
print(f"LCL temperature: {T_lcl:.1f}")
print(f"LCL height AGL:  ~{h_lcl.to('m'):.0f}")

p_lfc = lfc(p, T, Td)
print(f"LFC pressure: {p_lfc:.1f}")

p_el = el(p, T, Td)
print(f"EL pressure:  {p_el:.1f}")


# ── Step 4: CAPE and CIN ──────────────────────────────────────────────

prof = parcel_profile(p, T[0], Td[0])

cape_val, cin_val, h_lcl_agl, h_lfc_agl = cape_cin(
    p, T, Td, height,
    p[0], T[0], Td[0],
    parcel_type="sb",
)
print(f"\nCAPE: {cape_val:.0f}")
print(f"CIN:  {cin_val:.0f}")

if cape_val.magnitude < 300:
    cape_label = "weak"
elif cape_val.magnitude < 1000:
    cape_label = "moderate"
elif cape_val.magnitude < 2500:
    cape_label = "strong"
else:
    cape_label = "extreme"


# ── Step 5: Wind analysis ─────────────────────────────────────────────

sfc_speed = wind_speed(u[0:1], v[0:1])
sfc_dir = wind_direction(u[0:1], v[0:1])
print(f"\nSurface wind: {sfc_speed[0]:.1f} from {sfc_dir[0]:.0f}")

idx_500 = 10
spd_500 = wind_speed(u[idx_500:idx_500+1], v[idx_500:idx_500+1])
dir_500 = wind_direction(u[idx_500:idx_500+1], v[idx_500:idx_500+1])
print(f"500 hPa wind: {spd_500[0]:.1f} from {dir_500[0]:.0f}")

shear_u, shear_v = bulk_shear(u, v, height, 0 * units.m, top=6000 * units.m)
shear_mag = np.sqrt(shear_u.magnitude**2 + shear_v.magnitude**2)
print(f"0-6 km bulk shear: {shear_mag:.1f} m/s")

(rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v) = bunkers_storm_motion(u, v, height)
rm_speed = np.sqrt(rm_u.magnitude**2 + rm_v.magnitude**2)
print(f"Right-mover storm motion: u={rm_u:.1f}, v={rm_v:.1f} ({rm_speed:.1f} m/s)")
print(f"Mean wind:                u={mw_u:.1f}, v={mw_v:.1f}")

pos_1, neg_1, srh_0_1 = storm_relative_helicity(
    u, v, height, 1000 * units.m, rm_u, rm_v
)
pos_3, neg_3, srh_0_3 = storm_relative_helicity(
    u, v, height, 3000 * units.m, rm_u, rm_v
)
print(f"0-1 km SRH: {srh_0_1:.0f}")
print(f"0-3 km SRH: {srh_0_3:.0f}")


# ── Step 6: Severe weather indices ────────────────────────────────────

stp = significant_tornado_parameter(
    cape_val, h_lcl_agl, srh_0_1, shear_mag * units('m/s'),
)
scp = supercell_composite_parameter(
    cape_val, srh_0_3, shear_mag * units('m/s'),
)
print(f"\nSTP: {stp:.1f}")
print(f"SCP: {scp:.1f}")

if stp.magnitude >= 3 and scp.magnitude >= 4:
    verdict = "Conditions are favorable for significant severe weather, including violent tornadoes."
elif stp.magnitude >= 1 and scp.magnitude >= 1:
    verdict = "Conditions are favorable for supercells and significant tornadoes."
elif scp.magnitude >= 1:
    verdict = "Conditions favor supercell thunderstorms with large hail and damaging winds."
elif cape_val.magnitude >= 1000:
    verdict = "Conditions favor strong thunderstorms, but lack of shear limits tornado potential."
else:
    verdict = "Severe weather is unlikely."


# ── Step 7: Summary ───────────────────────────────────────────────────

print()
print("=" * 40)
print("  SOUNDING ANALYSIS SUMMARY")
print("=" * 40)
print(f"  Surface T/Td:   {T[0].magnitude:.1f} C / {Td[0].magnitude:.1f} C")
print(f"  LCL:            {p_lcl:.0f} (~{h_lcl.to('m'):.0f} AGL)")
print(f"  LFC:            {p_lfc:.0f}")
print(f"  EL:             {p_el:.0f}")
print(f"  CAPE:           {cape_val.magnitude:.0f} J/kg ({cape_label})")
print(f"  CIN:            {cin_val.magnitude:.0f} J/kg")
print(f"  PW:             {pw:.1f}")
print(f"  0-6 km Shear:   {shear_mag:.1f} m/s")
print(f"  0-1 km SRH:     {srh_0_1.magnitude:.0f} m^2/s^2")
print(f"  0-3 km SRH:     {srh_0_3.magnitude:.0f} m^2/s^2")
print(f"  Bunkers RM:     {rm_speed:.1f} m/s")
print(f"  STP:            {stp.magnitude:.1f}")
print(f"  SCP:            {scp.magnitude:.1f}")
print(f"  Verdict:        {verdict}")
print("=" * 40)
```

---

## What to try next

Now that you have the fundamentals, here are some ideas for further
exploration:

- **Change the moisture.** Drop the surface dewpoint from 24 C to 15 C and
  re-run the analysis. Watch how CAPE, LCL height, and STP all respond. Dry
  boundary layers produce higher LCLs and weaker tornado parameters.

- **Remove the shear.** Set `u` and `v` to constant values at all levels. CAPE
  stays the same, but SRH drops to zero and the composite indices collapse.
  This is what an ordinary airmass thunderstorm environment looks like.

- **Try different parcel types.** Change `parcel_type="sb"` to `"ml"`
  (mixed-layer, averaging the lowest 100 hPa) or `"mu"` (most-unstable,
  searching for the parcel with the highest CAPE). Operational forecasters
  typically look at all three.

- **Use real data.** Load an actual radiosonde observation from the University
  of Wyoming sounding archive or from an NWP model output file, and run the
  same analysis on it.
