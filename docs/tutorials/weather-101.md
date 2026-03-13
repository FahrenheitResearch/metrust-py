# Weather Data 101 -- A Crash Course

You can write code. You can debug distributed systems at 2 AM. But you have
no idea what "850 hPa dewpoint" means, and the last time you thought about
the atmosphere was when your flight hit turbulence.

This guide will fix that. By the end, you will understand what metrust is
actually computing, why meteorologists care about these numbers, and how to
use the library to calculate them yourself. We will not turn you into a
forecaster, but you will be dangerous enough to build weather tools that
make sense.

---

## 1. The Atmosphere as a Stack of Layers

Think of the atmosphere as a tall column of air sitting on top of you.
The air at the bottom is compressed by everything above it, so it is dense
and has high pressure. As you go up, there is less air overhead, so
pressure drops.

```
            Altitude        Pressure
            --------        --------
  Space     ~100 km         ~0 hPa

  ....................................

            ~9 km           ~300 hPa     <-- jet stream, cloud tops
            ~5.5 km         ~500 hPa     <-- "middle" of the atmosphere by mass
            ~3 km           ~700 hPa
            ~1.5 km         ~850 hPa     <-- low-level moisture, fronts
  Surface   0 km            ~1013 hPa    <-- you are here
```

**Why pressure instead of altitude?** Weather instruments on balloons
measure pressure directly -- it is the fundamental vertical coordinate.
Two locations at the "same altitude" can have different pressures depending
on temperature and weather patterns. Pressure surfaces are where the
physics actually happens: fronts, jet streams, and storm dynamics are all
defined on pressure levels. So when a meteorologist says "the temperature
at 500 hPa," they mean "the temperature at the level where pressure equals
500 hectopascals" -- roughly 5.5 km up, give or take.

!!! note "hPa and mb are the same thing"
    The unit **hectopascal (hPa)** is standard in meteorology. 1 hPa = 1
    millibar (mb). Standard sea-level pressure is 1013.25 hPa. You will see
    both hPa and mb in the wild; they are interchangeable.

**Temperature decreases with height** at roughly 6.5 degC per kilometer
in the lower atmosphere. This rate of decrease is called the *lapse rate*.
It matters because it controls whether air parcels rise or sink -- the
foundation of all storm forecasting.

**Key levels you will encounter constantly:**

| Pressure | Approx. Altitude | What happens here |
|----------|-------------------|-------------------|
| Surface (~1013 hPa) | 0 km | Where we live. Surface observations. |
| 850 hPa | ~1.5 km | Low-level moisture transport, fronts |
| 700 hPa | ~3 km | Mid-level moisture, capping inversions |
| 500 hPa | ~5.5 km | "Middle" of the atmosphere by mass |
| 300 hPa | ~9 km | Jet stream level, storm cloud tops |
| 200 hPa | ~12 km | Tropopause. Boundary with the stratosphere. |

---

## 2. Temperature Flavors

A single "temperature" is not enough to describe the atmosphere. Meteorologists
use several related quantities, each revealing something different about the air.

### Temperature (T)

What the thermometer reads. Simple. In metrust, temperatures are in Celsius
(or attached with Pint units, which handle conversions automatically).

### Dewpoint Temperature (Td)

The temperature at which the air becomes saturated -- meaning it holds all
the water vapor it can, and any further cooling would produce condensation
(fog, clouds, dew).

- **Td close to T** &rarr; humid air. Small cooling produces condensation.
- **Td far below T** &rarr; dry air. A lot of cooling needed before moisture condenses.

The difference `T - Td` is called the **dewpoint depression** -- a quick
humidity gauge. A depression of 2--3 degC means fog or low clouds. A depression
of 20+ degC means bone-dry air.

```python
import numpy as np
from metrust.calc import dewpoint_from_relative_humidity
from metrust.units import units

# Surface observation: 30 degC temperature, 22 degC dewpoint
T  = 30 * units.degC
Td = 22 * units.degC

# Dewpoint depression -- how dry is it?
depression = T - Td  # 8 degC -- moderately humid
```

### Potential Temperature (theta)

Imagine grabbing a parcel of air at some altitude and bringing it down to
the surface (1000 hPa) without adding or removing heat -- just letting it
compress and warm as pressure increases. The temperature it would have at
1000 hPa is its *potential temperature*.

Why bother? Because potential temperature is **conserved** when air rises
or sinks without exchanging heat (dry adiabatic processes). If two air
parcels at different altitudes have the same theta, they came from the same
air mass. It lets you compare temperatures at different altitudes on equal
footing.

```python
from metrust.calc import potential_temperature
from metrust.units import units

# Air at 700 hPa and 5 degC -- what is its potential temperature?
theta = potential_temperature(700 * units.hPa, 5 * units.degC)
print(theta)  # ~299.5 K -- warmer than 5 degC because compression heats it
```

### Virtual Temperature (Tv)

Moist air is lighter than dry air at the same temperature and pressure.
(Water vapor, H2O, molecular weight 18, replaces nitrogen, N2, molecular
weight 28, and oxygen, O2, molecular weight 32.) Virtual temperature
adjusts for this: it is the temperature that *dry* air would need to have
the same density as the actual moist air.

Tv is always >= T. The difference is small (a few tenths to a couple of
degrees) but matters for precise buoyancy and pressure calculations.

```python
from metrust.calc import virtual_temperature
from metrust.units import units

Tv = virtual_temperature(30 * units.degC, 1000 * units.hPa, dewpoint=22 * units.degC)
print(Tv)  # ~32 degC -- slightly warmer to account for moisture
```

### Equivalent Potential Temperature (theta-e)

Like potential temperature, but also accounts for the latent heat released
when water vapor condenses. Theta-e is conserved in *moist* adiabatic processes
(when clouds are forming and releasing heat). This makes it the best tracer
for tracking air masses through storms and fronts.

```python
from metrust.calc import equivalent_potential_temperature
from metrust.units import units

theta_e = equivalent_potential_temperature(
    1000 * units.hPa, 30 * units.degC, 22 * units.degC
)
print(theta_e)  # ~354 K -- much higher than theta because of latent heat
```

!!! tip "When to use which temperature"
    - **T**: what you feel outside, what instruments measure.
    - **Td**: how much moisture is in the air.
    - **theta**: comparing air masses at different altitudes (dry processes).
    - **theta-e**: tracking air masses through clouds and storms (moist processes).
    - **Tv**: precise density and buoyancy calculations.

### Summary Table

```
  Name                  Symbol    What it tells you
  --------------------  ------    ----------------------------------------
  Temperature           T         What the thermometer reads
  Dewpoint              Td        When would this air saturate?
  Potential temp        theta     T normalized to 1000 hPa (conserved dry)
  Equiv. potential temp theta_e   theta + latent heat (conserved moist)
  Virtual temp          Tv        T adjusted for moisture (density)
```

---

## 3. Moisture Variables

Water vapor is invisible but it drives almost everything interesting in
weather -- clouds, rain, thunderstorms, hurricanes. Meteorologists have
several ways to measure "how much moisture is in the air," each useful in
different contexts.

### Relative Humidity (RH)

The ratio of the actual water vapor in the air to the maximum it could
hold at that temperature, expressed as a percentage.

- **RH = 100%** &rarr; saturated. Fog, clouds, or precipitation.
- **RH = 50%** &rarr; holding half of what it could.
- **RH is temperature-dependent.** Cool the same air and RH goes up
  (because cold air holds less vapor). That is why dew forms at night.

```python
from metrust.calc import (
    relative_humidity_from_dewpoint,
    dewpoint_from_relative_humidity,
)
from metrust.units import units

# Compute RH from temperature and dewpoint
rh = relative_humidity_from_dewpoint(30 * units.degC, 22 * units.degC)
print(rh)  # ~0.62 (62%)

# Go the other way: find dewpoint from T and RH
Td = dewpoint_from_relative_humidity(30 * units.degC, 0.62)
print(Td)  # ~22 degC
```

### Mixing Ratio (w)

The mass of water vapor per mass of *dry* air. Typically expressed in
grams per kilogram (g/kg). Unlike RH, mixing ratio does not change when
air warms or cools (as long as no water condenses or evaporates). This
makes it useful for tracking moisture through the atmosphere.

```python
from metrust.calc import (
    saturation_mixing_ratio,
    mixing_ratio_from_relative_humidity,
)
from metrust.units import units

# Saturation mixing ratio -- the maximum the air could hold
w_sat = saturation_mixing_ratio(1000 * units.hPa, 30 * units.degC)
print(w_sat)  # ~0.027 kg/kg -- warm air holds a LOT of moisture

# Actual mixing ratio from pressure, temperature, and RH
w = mixing_ratio_from_relative_humidity(
    1000 * units.hPa, 30 * units.degC, 62 * units.percent
)
print(w.to("g/kg"))  # ~17 g/kg
```

### Specific Humidity (q)

Almost the same as mixing ratio, but per kilogram of *moist* air
(dry air + vapor) instead of dry air alone. Numerically very close to
mixing ratio for typical atmospheric moisture levels. Used in many NWP
model outputs because it is bounded between 0 and 1.

```python
from metrust.calc import (
    specific_humidity_from_mixing_ratio,
    mixing_ratio_from_specific_humidity,
)
from metrust.units import units

# Convert between mixing ratio and specific humidity
q = specific_humidity_from_mixing_ratio(0.017 * units("kg/kg"))
print(q)  # slightly less than 0.017

w_back = mixing_ratio_from_specific_humidity(q)
print(w_back)  # round-trips to the original
```

### Vapor Pressure (e)

The partial pressure exerted by water vapor in the air. At saturation,
this is the *saturation vapor pressure* (es). The Clausius-Clapeyron
equation governs how es increases exponentially with temperature -- which
is why warm air can hold so much more moisture than cold air.

```python
from metrust.calc import saturation_vapor_pressure, vapor_pressure
from metrust.units import units

# Saturation vapor pressure at 30 degC
es = saturation_vapor_pressure(30 * units.degC)
print(es.to("hPa"))  # ~42.4 hPa

# Actual vapor pressure from the dewpoint
e = vapor_pressure(22 * units.degC)
print(e.to("hPa"))  # ~26.4 hPa

# RH is just the ratio: e / es
print(f"RH check: {(e / es).magnitude:.1%}")  # ~62%
```

### Dewpoint -- Tying It All Together

You can compute dewpoint from any other moisture variable. It is
the "universal adapter" of moisture measurements:

```python
from metrust.calc import (
    dewpoint_from_relative_humidity,
    dewpoint_from_specific_humidity,
)
from metrust.units import units

# From RH
Td = dewpoint_from_relative_humidity(30 * units.degC, 0.62)

# From specific humidity and pressure
Td = dewpoint_from_specific_humidity(1013 * units.hPa, 0.014 * units("kg/kg"))
```

!!! note "How they all relate"
    All of these are just different ways to express the same underlying
    quantity: how much water vapor is in the air. metrust provides conversion
    functions between all of them. If you have any one moisture variable plus
    pressure and temperature, you can derive all the others.

---

## 4. What is a Sounding?

A **sounding** is the single most important data structure in meteorology.
It is a vertical profile of the atmosphere at a specific time and place --
think of it as a "core sample" of the air column above one location.

### How soundings are collected

Twice a day (0000 and 1200 UTC), weather stations around the world launch
*radiosondes* -- instrument packages attached to helium balloons. As the
balloon ascends, the radiosonde measures temperature, humidity, pressure,
and wind at each level, transmitting data back to the ground. The balloon
eventually pops at around 30 km altitude.

### The data format

A sounding is a set of parallel arrays, one value per pressure level:

```python
import numpy as np
from metrust.calc import lcl, lfc, el, cape_cin, parcel_profile
from metrust.units import units

# A realistic 7-level sounding (surface to ~12 km)
p  = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T  = np.array([  30,  25,  20,  10, -10, -35, -55]) * units.degC
Td = np.array([  22,  18,  14,   2, -20, -45, -65]) * units.degC
h  = np.array([   0, 750,1500,3000,5500,9000,12000]) * units.m
```

!!! warning "Pressure must decrease going up"
    The first element (index 0) is the surface, where pressure is highest.
    The last element is the top of the sounding. This is the standard
    convention in metrust (and MetPy). Getting this backwards will produce
    wrong results.

### Why meteorologists care about soundings

From this one data structure you can determine:

- **Where clouds will form** &rarr; the Lifting Condensation Level (LCL)
- **Whether thunderstorms are possible** &rarr; CAPE and CIN
- **How strong those storms might be** &rarr; severe weather parameters
- **What type of storms to expect** &rarr; wind shear and hodograph shape
- **Precipitation type** &rarr; temperature profile through the column

### Basic sounding analysis

Let's walk through the key calculations.

**Step 1: Find the cloud base (LCL).**

```python
p_lcl, T_lcl = lcl(1000 * units.hPa, 30 * units.degC, 22 * units.degC)
print(f"Cloud base: {p_lcl:.0f}")   # ~887 hPa (~1100 m up)
print(f"LCL temp:   {T_lcl:.1f}")
```

**Step 2: Trace the parcel upward.**

The parcel profile computes what temperature a surface air parcel would have
if lifted through the entire sounding. Below the LCL it cools at ~10 degC/km
(dry adiabat). Above the LCL, condensation releases heat, so it cools more
slowly (~6 degC/km, the moist adiabat).

```python
prof = parcel_profile(p, 30 * units.degC, 22 * units.degC)
# prof is an array of temperatures -- one per pressure level
```

**Step 3: Find CAPE and CIN.**

```python
cape, cin, lcl_h, lfc_h = cape_cin(
    p, T, Td, h,
    psfc=1000 * units.hPa,
    t2m=30 * units.degC,
    td2m=22 * units.degC,
)
print(f"CAPE: {cape:.0f}")         # Energy for storms (J/kg)
print(f"CIN:  {cin:.0f}")          # Cap strength (J/kg)
print(f"LCL height: {lcl_h:.0f}") # Cloud base (m AGL)
print(f"LFC height: {lfc_h:.0f}") # Free convection level (m AGL)
```

We will explain what CAPE and CIN mean in the next section.

---

## 5. Stability and Buoyancy

Stability is the central question in weather forecasting: **if I push
a parcel of air upward, does it keep going or sink back down?**

### The Parcel Model

Imagine taking a small "parcel" of air from the surface and lifting it.
As it rises, pressure drops and the parcel expands and cools. The key
question: how does the parcel's temperature compare to its surroundings?

- **Parcel warmer than environment** &rarr; buoyant &rarr; keeps rising &rarr; **UNSTABLE**
- **Parcel cooler than environment** &rarr; sinks back down &rarr; **STABLE**

### Critical Levels

As the parcel rises through the sounding, it crosses several important
boundaries:

**LCL (Lifting Condensation Level):** Where the parcel cools to its dewpoint.
Cloud forms. This is the cloud base.

**LFC (Level of Free Convection):** Where the parcel first becomes warmer than
the environment. Above this point it rises freely -- a thunderstorm is born.

**EL (Equilibrium Level):** Where the parcel finally matches the environment
temperature again. This is approximately the cloud top.

```
  Pressure (hPa)
    200  +---------+
         |     EL  x............  <-- parcel meets environment
    300  |        / :
         |       /  : <-- CAPE
    400  |      /   :    (parcel warmer than environment)
         |     /    :    (total energy available for storms)
    500  |    /     :
         |   /  LFC x............  <-- parcel becomes buoyant
    600  |  :   /
         |  :  /    <-- CIN
    700  |  : /         (parcel cooler than environment)
         |  :/          (energy barrier to overcome)
    800  |  x LCL
         | /
    900  |/
         x Surface
   1000  +---------+
```

### CAPE and CIN

These two numbers are the most important stability parameters in severe
weather forecasting.

**CAPE (Convective Available Potential Energy):** The total energy
available to accelerate the parcel upward between the LFC and the EL.
Measured in J/kg. Think of it as the fuel for thunderstorms.

| CAPE (J/kg) | Meaning |
|-------------|---------|
| 0 | No instability. No thunderstorms. |
| 100--1000 | Marginal. Weak storms possible. |
| 1000--2500 | Moderate. Strong thunderstorms likely. |
| 2500--4000 | High. Severe storms possible. |
| >4000 | Extreme. Violent storms possible. |

**CIN (Convective Inhibition):** The energy barrier the parcel must
overcome to reach the LFC. Think of it as a "cap" or lid on convection.
CIN is reported as a negative value.

| CIN (J/kg) | Meaning |
|------------|---------|
| 0 to -50 | Weak cap. Storms can fire easily. |
| -50 to -200 | Moderate cap. Needs a trigger (front, dryline). |
| < -200 | Strong cap. Very hard to break. |

!!! tip "Some CIN is actually good"
    A moderate cap prevents storms from firing too early. The atmosphere
    keeps heating and moistening, building more CAPE. When the cap finally
    breaks -- often in late afternoon -- storms explode. This is the classic
    "loaded gun" sounding pattern.

```python
import numpy as np
from metrust.calc import cape_cin
from metrust.units import units

# Full sounding data
p  = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T  = np.array([  30,  25,  20,  10, -10, -35, -55]) * units.degC
Td = np.array([  22,  18,  14,   2, -20, -45, -65]) * units.degC
h  = np.array([   0, 750,1500,3000,5500,9000,12000]) * units.m

cape, cin, lcl_h, lfc_h = cape_cin(
    p, T, Td, h,
    psfc=1000 * units.hPa,
    t2m=30 * units.degC,
    td2m=22 * units.degC,
)
print(f"CAPE: {cape:.0f}")
print(f"CIN:  {cin:.0f}")

# Interpret
if cape.magnitude > 2500:
    print("High instability -- severe storms possible")
elif cape.magnitude > 1000:
    print("Moderate instability -- thunderstorms likely")
else:
    print("Marginal or no instability")
```

---

## 6. Wind

Wind might seem straightforward -- it blows, things move. But in
meteorology, *how wind changes with height* is as important as the
wind itself, especially for severe weather.

### Components: u and v

Wind in atmospheric data is almost always stored as two components:

- **u**: east-west component. Positive u = wind blowing toward the east.
- **v**: north-south component. Positive v = wind blowing toward the north.

Why components instead of speed and direction? Because components are what
the physics equations use. You can add, subtract, and integrate them
directly. Speed and direction require trigonometry for every operation.

```python
from metrust.calc import wind_speed, wind_direction, wind_components
from metrust.units import units
import numpy as np

# Compute speed and direction from components
u = np.array([10, -5, 0, 8]) * units("m/s")
v = np.array([ 5, -5, 10, 0]) * units("m/s")

speed = wind_speed(u, v)
direction = wind_direction(u, v)
print(speed)      # [11.2, 7.1, 10.0, 8.0] m/s
print(direction)  # meteorological direction (FROM)
```

!!! warning "Wind direction convention"
    Wind direction is where the wind blows **FROM**, not where it blows TO.
    A 270-degree wind blows from the west (toward the east). A 180-degree
    wind blows from the south. This trips up many developers.

```python
# Go the other way: speed + direction -> u, v
u_calc, v_calc = wind_components(
    np.array([15, 10]) * units("m/s"),
    np.array([225, 180]) * units.degree
)
```

### Bulk Shear -- Change in Wind with Height

Wind shear is the change in wind speed or direction (or both) with height.
It is the single most important factor in determining storm type.

**Bulk shear** is the vector difference between wind at two levels.
The 0--6 km bulk shear is the standard measure:

```python
from metrust.calc import bulk_shear, wind_speed
from metrust.units import units
import numpy as np

height = np.array([0, 1000, 3000, 6000]) * units.m
u_prof = np.array([2, 8, 20, 35]) * units("m/s")
v_prof = np.array([5, 8, 10, 5]) * units("m/s")

shear_u, shear_v = bulk_shear(
    u_prof, v_prof, height,
    bottom=0 * units.m,
    top=6000 * units.m,
)
shear_mag = wind_speed(
    np.array([shear_u.magnitude]) * units("m/s"),
    np.array([shear_v.magnitude]) * units("m/s"),
)
print(f"0-6 km shear: {shear_mag[0]:.1f}")
```

| 0--6 km Shear | Storm Type |
|----------------|------------|
| < 10 m/s | Single-cell storms. Weak, short-lived. |
| 10--20 m/s | Multicell storms. Clusters, squall lines. |
| 20--30 m/s | Supercells possible. Long-lived, rotating. |
| > 30 m/s | Strongly supercellular. Tornado risk. |

### Storm-Relative Helicity (SRH)

SRH measures how much the wind profile would cause rotation in a storm's
updraft. It is computed relative to the storm's motion -- hence
"storm-relative." Think of it as measuring how much the low-level wind
spirals into the storm like a corkscrew.

Computing SRH requires knowing the storm motion. The **Bunkers method**
estimates how a supercell would move:

```python
from metrust.calc import bunkers_storm_motion, storm_relative_helicity
from metrust.units import units
import numpy as np

height = np.array([0, 250, 500, 750, 1000, 2000, 3000, 6000]) * units.m
u_prof = np.array([0, 3, 5, 8, 12, 20, 25, 35]) * units("m/s")
v_prof = np.array([5, 7, 10, 12, 10, 8, 5, 0]) * units("m/s")

# Estimate storm motion using Bunkers
(ru, rv), _, _ = bunkers_storm_motion(u_prof, v_prof, height)

# Compute 0-1 km SRH (most important for tornadoes)
pos_srh, neg_srh, total_srh = storm_relative_helicity(
    u_prof, v_prof, height,
    1000 * units.m,  # depth of integration
    ru, rv            # storm motion components
)
print(f"0-1 km SRH: {total_srh:.0f}")
```

| 0--1 km SRH (m^2/s^2) | Interpretation |
|------------------------|----------------|
| 50--150 | Weak rotation potential |
| 150--300 | Significant tornado risk |
| 300--500 | Very large tornado risk |
| > 500 | Extreme (rare, violent tornadoes) |

---

## 7. Severe Weather Parameters

Severe weather composite indices combine multiple ingredients into a single
number. They are the bread and butter of operational forecasting.

!!! warning "Parameters are guides, not guarantees"
    A high value means the *ingredients* are present, not that severe weather
    *will* happen. A trigger mechanism (front, outflow boundary, etc.) is
    also needed. Conversely, severe weather occasionally occurs with low
    parameter values. Treat these as probabilistic guides.

### Significant Tornado Parameter (STP)

STP combines four ingredients that research has found are present in
environments that produce significant (EF2+) tornadoes:

- **MLCAPE**: instability (mixed-layer CAPE, in J/kg)
- **LCL height**: cloud base height (lower = more tornado-friendly)
- **0--1 km SRH**: low-level rotation (m^2/s^2)
- **0--6 km bulk shear**: deep-layer shear (m/s)

| STP | Interpretation |
|-----|----------------|
| < 1 | Below threshold. Tornadoes unlikely. |
| 1--3 | Favorable for significant tornadoes. |
| 3--6 | Very favorable for significant tornadoes. |
| > 6 | Extreme. Violent tornado environment. |

```python
from metrust.calc import significant_tornado_parameter
from metrust.units import units

stp = significant_tornado_parameter(
    2500 * units("J/kg"),       # MLCAPE
    800 * units.m,              # LCL height AGL
    250 * units("m**2/s**2"),   # 0-1 km SRH
    25 * units("m/s"),          # 0-6 km bulk shear
)
print(f"STP: {stp:.1f}")  # > 1 means favorable
```

### Supercell Composite Parameter (SCP)

SCP assesses whether the environment supports supercell thunderstorms --
the long-lived rotating storms that produce the majority of significant
tornadoes, large hail, and damaging winds.

| SCP | Interpretation |
|-----|----------------|
| < 1 | Supercells unlikely. |
| 1--4 | Favorable for supercells. |
| > 4 | Strongly favorable for supercells. |

```python
from metrust.calc import supercell_composite_parameter
from metrust.units import units

scp = supercell_composite_parameter(
    3000 * units("J/kg"),       # MUCAPE
    200 * units("m**2/s**2"),   # Effective SRH
    25 * units("m/s"),          # Effective bulk shear
)
print(f"SCP: {scp:.1f}")
```

### Significant Hail Parameter (SHIP)

SHIP identifies environments favorable for significant (2-inch+) hail by
combining CAPE, shear, 500 hPa temperature, mid-level lapse rates, and
low-level moisture.

| SHIP | Interpretation |
|------|----------------|
| < 1 | Significant hail unlikely. |
| 1--2 | Significant hail possible. |
| > 4 | Extreme hail environment. |

```python
from metrust.calc import compute_ship
import numpy as np

# compute_ship works on 2-D grids, but scalars wrapped in arrays work too
ship = compute_ship(
    np.array([[3000.0]]),    # MUCAPE (J/kg)
    np.array([[25.0]]),      # 0-6 km shear (m/s)
    np.array([[-15.0]]),     # 500 hPa temperature (degC)
    np.array([[7.5]]),       # 700-500 hPa lapse rate (degC/km)
    np.array([[14.0]]),      # low-level mixing ratio (g/kg)
)
print(f"SHIP: {ship[0, 0]:.1f}")
```

### Putting It All Together

In practice, forecasters never look at one parameter in isolation. Here is
a complete severe weather assessment from a sounding:

```python
import numpy as np
from metrust.calc import (
    cape_cin, bunkers_storm_motion, storm_relative_helicity,
    bulk_shear, wind_speed,
    significant_tornado_parameter, supercell_composite_parameter,
)
from metrust.units import units

# --- Sounding data ---
p  = np.array([1013,1000,975,950,925,900,850,800,750,700,
               650,600,550,500,400,300,250,200,150]) * units.hPa
T  = np.array([  32, 31, 28, 26, 24, 22, 18, 14, 10, 6,
                  1, -4,-10,-16,-28,-42,-50,-58,-65]) * units.degC
Td = np.array([  23, 22, 21, 20, 18, 16, 12,  5,  0,-8,
               -15,-22,-28,-35,-45,-55,-62,-70,-77]) * units.degC
h  = np.array([   0,100,350,600,850,1100,1500,2000,2500,3000,
              3600,4200,4900,5600,7200,9200,10400,11800,13500]) * units.m
u  = np.array([   2,  3,  5,  7, 10, 12, 15, 18, 22, 25,
                 28, 30, 32, 35, 38, 40, 42, 43, 44]) * units("m/s")
v  = np.array([   8,  9, 10, 11, 12, 12, 10,  8,  5,  2,
                  0, -2, -3, -4, -5, -5, -4, -3, -2]) * units("m/s")

# --- Stability ---
cape, cin, lcl_h, lfc_h = cape_cin(
    p, T, Td, h,
    psfc=1013 * units.hPa,
    t2m=32 * units.degC,
    td2m=23 * units.degC,
)

# --- Storm motion ---
(ru, rv), _, _ = bunkers_storm_motion(u, v, h)

# --- 0-1 km SRH ---
_, _, srh_01 = storm_relative_helicity(
    u, v, h, 1000 * units.m, ru, rv
)

# --- 0-6 km bulk shear ---
su, sv = bulk_shear(u, v, h, 0 * units.m, top=6000 * units.m)
shear_06 = wind_speed(
    np.array([su.magnitude]) * units("m/s"),
    np.array([sv.magnitude]) * units("m/s"),
)

# --- Composite parameters ---
stp = significant_tornado_parameter(cape, lcl_h, srh_01, shear_06[0])
scp = supercell_composite_parameter(cape, srh_01, shear_06[0])

# --- Report ---
print(f"CAPE:         {cape:.0f}")
print(f"CIN:          {cin:.0f}")
print(f"0-1 km SRH:   {srh_01:.0f}")
print(f"0-6 km shear: {shear_06[0]:.1f}")
print(f"STP:          {stp:.1f}")
print(f"SCP:          {scp:.1f}")
```

---

## 8. Grids vs. Soundings

So far, everything has been about a single sounding -- one vertical
column. But weather models produce 3-D grids covering entire regions or
the globe. This is where metrust really shines.

### What a Weather Model Grid Looks Like

Numerical Weather Prediction (NWP) models divide the atmosphere into a
3-D grid of cells:

| Model | Resolution | Coverage | Update Frequency |
|-------|-----------|----------|------------------|
| HRRR | 3 km | CONUS | Hourly |
| NAM | 12 km | N. America | Every 6 hours |
| GFS | 0.25 deg | Global | Every 6 hours |
| ERA5 | 0.25 deg | Global | Hourly (reanalysis) |

Each grid point has values for temperature, moisture, wind, and pressure at
every vertical level. The data shape is typically `(nz, ny, nx)` -- levels
by latitude by longitude. **Each grid column is like a sounding.**

```
               nx (longitude points)
             <----------------------->
       ^    +--+--+--+--+--+--+--+--+    Level nz (top)
  ny   |    |  |  |  |  |  |  |  |  |
(lat   |    +--+--+--+--+--+--+--+--+    Level nz-1
points)|    |  |  |  |  |  |  |  |  |
       |    +--+--+--+--+--+--+--+--+    ...
       v    |  |  |  |  |  |  |  |  |
            +--+--+--+--+--+--+--+--+    Level 1 (surface)
```

### The Problem: Millions of Soundings

Computing CAPE for a single sounding is fast. Computing it for every one of
the 1.9 million grid columns in an HRRR domain is where performance matters.
Looping in Python would take 10--30 minutes. metrust's `compute_*` functions
process the entire grid in parallel using Rust, finishing in seconds.

```python
import numpy as np
from metrust.calc import compute_cape_cin, compute_srh, compute_shear, compute_stp
from metrust.units import units

# Suppose you have loaded 3-D model data (e.g., from GRIB files via xarray)
# Shape: (nz, ny, nx) = (50, 1059, 1799) for a typical HRRR grid
# These would normally come from xarray/cfgrib:

# pressure_3d    shape (nz, ny, nx) in Pa
# temperature_3d shape (nz, ny, nx) in degrees Celsius
# qvapor_3d      shape (nz, ny, nx) in kg/kg
# height_agl_3d  shape (nz, ny, nx) in meters AGL
# u_3d, v_3d     shape (nz, ny, nx) in m/s
# psfc           shape (ny, nx) in Pa
# t2m            shape (ny, nx) in K
# q2m            shape (ny, nx) in kg/kg

# --- CAPE and CIN for every grid point (seconds, not minutes) ---
cape, cin, lcl_h, lfc_h = compute_cape_cin(
    pressure_3d, temperature_3d, qvapor_3d, height_agl_3d,
    psfc, t2m, q2m,
    parcel_type="surface",
)
# cape.shape = (ny, nx)

# --- Storm-relative helicity (0-1 km) ---
srh_1km = compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0)

# --- 0-6 km bulk shear ---
shear_06 = compute_shear(u_3d, v_3d, height_agl_3d, top_m=6000.0)

# --- Significant Tornado Parameter from the 2-D fields ---
stp = compute_stp(cape, lcl_h, srh_1km, shear_06)
# stp.shape = (ny, nx) -- a full STP map in one call
```

!!! tip "Performance comparison"
    On a 1059 x 1799 HRRR grid (50 levels, ~95 million data points):

    | Function | Wall time |
    |----------|-----------|
    | `compute_cape_cin` | 2--5 s |
    | `compute_srh` | < 1 s |
    | `compute_shear` | < 1 s |
    | `compute_stp` | < 0.1 s |
    | **Total** | **3--7 s** |

    The equivalent pure-Python loop takes 10--30 minutes.

---

## Quick Reference: "What Function Do I Call?"

| I want to compute... | metrust function |
|---|---|
| Potential temperature | `calc.potential_temperature(p, T)` |
| Equivalent potential temperature | `calc.equivalent_potential_temperature(p, T, Td)` |
| Wet-bulb temperature | `calc.wet_bulb_temperature(p, T, Td)` |
| Virtual temperature | `calc.virtual_temperature(T, p, dewpoint=Td)` |
| Dewpoint from RH | `calc.dewpoint_from_relative_humidity(T, rh)` |
| RH from dewpoint | `calc.relative_humidity_from_dewpoint(T, Td)` |
| Saturation vapor pressure | `calc.saturation_vapor_pressure(T)` |
| Vapor pressure | `calc.vapor_pressure(Td)` |
| Saturation mixing ratio | `calc.saturation_mixing_ratio(p, T)` |
| Mixing ratio from RH | `calc.mixing_ratio_from_relative_humidity(p, T, rh)` |
| Specific humidity &harr; mixing ratio | `calc.specific_humidity_from_mixing_ratio(w)` |
| LCL | `calc.lcl(p_sfc, T_sfc, Td_sfc)` |
| LFC | `calc.lfc(p, T, Td)` |
| EL | `calc.el(p, T, Td)` |
| CAPE and CIN | `calc.cape_cin(p, T, Td, h, psfc=, t2m=, td2m=)` |
| Parcel profile | `calc.parcel_profile(p, T_sfc, Td_sfc)` |
| Wind speed | `calc.wind_speed(u, v)` |
| Wind direction | `calc.wind_direction(u, v)` |
| u, v from speed/direction | `calc.wind_components(speed, direction)` |
| Bulk shear | `calc.bulk_shear(u, v, h, bottom, top=)` |
| Storm motion | `calc.bunkers_storm_motion(u, v, h)` |
| Storm-relative helicity | `calc.storm_relative_helicity(u, v, h, depth, su, sv)` |
| Significant Tornado Parameter | `calc.significant_tornado_parameter(cape, lcl_h, srh, shear)` |
| Supercell Composite Parameter | `calc.supercell_composite_parameter(cape, srh, shear)` |
| Significant Hail Parameter | `calc.compute_ship(cape, shear, t500, lr, mr)` |
| Grid CAPE/CIN (parallel) | `calc.compute_cape_cin(...)` |
| Grid SRH (parallel) | `calc.compute_srh(...)` |
| Grid shear (parallel) | `calc.compute_shear(...)` |
| Grid STP (parallel) | `calc.compute_stp(...)` |

---

## What Next?

You now understand the building blocks of meteorological computation. From here:

- **[Your First Sounding Analysis](first-sounding.md)** -- a full hands-on tutorial
- **[API Reference: Thermodynamics](/api/thermodynamics/)** -- detailed docs for every function
- **[API Reference: Wind & Kinematics](/api/wind/)** -- full wind analysis API
- **[API Reference: Severe Weather](/api/severe/)** -- all composite parameters
- **[Grid Composites](/api/grid-composites/)** -- parallel grid computation
- **[Migration from MetPy](/guides/migration/)** -- if you are coming from MetPy
