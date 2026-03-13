# Weather 101 for Developers

You can write code. You can debug distributed systems at 2 AM. But you have
no idea what "850 hPa dewpoint" means, and the last time you thought about
the atmosphere was when your flight hit turbulence.

This guide will fix that. By the end, you will understand what metrust is
actually computing, why meteorologists care about these numbers, and how to
use the library to calculate them yourself. We will not turn you into a
forecaster, but you will be dangerous enough to build weather tools that
make sense.

---

## The Atmosphere as a Stack of Layers

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

**The unit: hectopascals (hPa).** 1 hPa = 1 millibar (mb). Standard sea-level
pressure is 1013.25 hPa. You will see both hPa and mb in the wild; they are
the same thing.

**Temperature decreases with height** at roughly 6.5 degC per kilometer
in the lower atmosphere. This rate of decrease is called the *lapse rate*.
It matters because it controls whether air parcels rise or sink -- the
foundation of all storm forecasting.

```
  Altitude    Typical Temperature
  ---------   -------------------
  9 km        -45 degC
  5.5 km      -20 degC
  3 km         5 degC
  1.5 km      15 degC
  Surface     25 degC
```

---

## Temperature Flavors

A single "temperature" is not enough to describe the atmosphere. Meteorologists
use several related quantities, each revealing something different about the air.
Here is what they are and how to compute them.

### Temperature (T)

What the thermometer reads. Simple. In metrust, temperatures are in Celsius
(or attached with pint units, which handle conversions automatically).

### Dewpoint Temperature (Td)

The temperature at which the air becomes saturated -- meaning it holds all
the water vapor it can, and any further cooling would produce condensation
(fog, clouds, dew).

- **Td close to T** = humid air. The gap is small, so it would not take
  much cooling to reach saturation.
- **Td far below T** = dry air. The air needs to cool a lot before moisture
  condenses.

The difference `T - Td` is called the **dewpoint depression** and is a
quick-and-dirty humidity gauge. A depression of 2-3 degC means fog or low
clouds are likely. A depression of 20+ degC means the air is bone dry.

```python
import numpy as np
from metrust.units import units
from metrust import calc

# Surface observation: 30 degC temperature, 22 degC dewpoint
T = 30 * units.degC
Td = 22 * units.degC

# Dewpoint depression -- how dry is it?
depression = T - Td  # 8 degC -- moderately humid
```

### Potential Temperature (theta)

Imagine grabbing a parcel of air at some altitude and bringing it down to
the surface (1000 hPa) without adding or removing heat -- just letting it
compress and warm as pressure increases. The temperature it would have at
the surface is its *potential temperature*.

Why bother? Because potential temperature is **conserved** when air rises
or sinks without exchanging heat (dry adiabatic processes). If two air
parcels have the same theta, they came from the same air mass. It lets you
compare temperatures at different altitudes on equal footing.

```python
# Air at 700 hPa and 5 degC -- what is its potential temperature?
theta = calc.potential_temperature(700 * units.hPa, 5 * units.degC)
print(theta)  # ~299.5 K -- warmer than 5 degC because compression heats it
```

### Wet-Bulb Temperature (Tw)

Wrap a thermometer in a wet cloth and swing it through the air. Evaporation
cools the thermometer until the air around it reaches saturation. The
temperature it settles at is the wet-bulb temperature.

It always sits between T and Td:  `Td <= Tw <= T`

Wet-bulb temperature matters for heat stress (the human body cools by
sweating, which is evaporation), precipitation type (rain vs. snow vs.
sleet), and agricultural applications.

```python
Tw = calc.wet_bulb_temperature(1000 * units.hPa, 30 * units.degC, 22 * units.degC)
print(Tw)  # ~25 degC
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
Tv = calc.virtual_temperature(30 * units.degC, 1000 * units.hPa, 22 * units.degC)
print(Tv)  # ~32 degC -- slightly warmer to account for moisture
```

### Equivalent Potential Temperature (theta-e)

Like potential temperature, but also accounts for the latent heat released
when water vapor condenses. It is conserved in *moist* adiabatic processes
(when clouds are forming and releasing heat). Useful for tracking air masses
through storms.

```python
theta_e = calc.equivalent_potential_temperature(
    1000 * units.hPa, 30 * units.degC, 22 * units.degC
)
print(theta_e)  # ~354 K -- much higher than theta because of latent heat
```

### Summary: Temperature Flavors at a Glance

```
  Name                  Symbol    What it tells you
  --------------------  ------    ----------------------------------------
  Temperature           T         What the thermometer reads
  Dewpoint              Td        When would this air saturate?
  Potential temp        theta     T normalized to 1000 hPa (conserved dry)
  Equiv. potential temp theta_e   theta + latent heat (conserved moist)
  Wet-bulb temp         Tw        Lowest T reachable by evaporation
  Virtual temp          Tv        T adjusted for moisture (density)
```

---

## Moisture Variables

Water vapor is invisible but it drives almost everything interesting in
weather -- clouds, rain, thunderstorms, hurricanes. Meteorologists have
several ways to measure "how much moisture is in the air," each useful in
different contexts.

### Relative Humidity (RH)

The ratio of the actual water vapor in the air to the maximum it could
hold at that temperature, expressed as a percentage.

- **RH = 100%**: the air is saturated. Fog, clouds, or precipitation is
  occurring (or about to).
- **RH = 50%**: the air is holding half of what it could.
- **RH is temperature-dependent.** Cool the same air and RH goes up
  (because cold air holds less vapor). That is why dew forms at night.

```python
# Compute RH from temperature and dewpoint
rh = calc.relative_humidity_from_dewpoint(30 * units.degC, 22 * units.degC)
print(rh)  # ~0.62 (62%)

# Go the other way: find dewpoint from T and RH
Td = calc.dewpoint_from_relative_humidity(30 * units.degC, 62 * units.percent)
print(Td)  # ~22 degC
```

### Mixing Ratio (w)

The mass of water vapor per mass of *dry* air. Typically expressed in
grams per kilogram (g/kg). Unlike RH, mixing ratio does not change when
air warms or cools (as long as no water condenses or evaporates). This
makes it useful for tracking moisture through the atmosphere.

```python
# Mixing ratio from pressure and temperature (saturation mixing ratio)
w_sat = calc.saturation_mixing_ratio(1000 * units.hPa, 30 * units.degC)
print(w_sat.to("g/kg"))  # ~27 g/kg -- warm air holds a LOT of moisture

# Mixing ratio from pressure, temperature, and RH
w = calc.mixing_ratio_from_relative_humidity(
    1000 * units.hPa, 30 * units.degC, 62 * units.percent
)
print(w.to("g/kg"))  # ~17 g/kg
```

### Specific Humidity (q)

Almost the same as mixing ratio, but per kilogram of *moist* air
(dry air + vapor) instead of dry air alone. Numerically very close to
mixing ratio for typical atmospheric moisture levels. Used in many NWP
model outputs.

```python
# Convert between mixing ratio and specific humidity
q = calc.specific_humidity_from_mixing_ratio(w)
print(q)

w_back = calc.mixing_ratio_from_specific_humidity(q)
print(w_back)  # round-trips to the original
```

### Vapor Pressure (e)

The partial pressure exerted by water vapor in the air. At saturation,
this is the *saturation vapor pressure* (es). The Clausius-Clapeyron
equation governs how es increases exponentially with temperature -- which
is why warm air can hold so much more moisture than cold air.

```python
# Saturation vapor pressure at 30 degC
es = calc.saturation_vapor_pressure(30 * units.degC)
print(es.to("hPa"))  # ~42.4 hPa

# Actual vapor pressure from the dewpoint
e = calc.vapor_pressure(22 * units.degC)
print(e.to("hPa"))  # ~26.4 hPa

# RH is just the ratio: e / es
print(f"RH check: {(e / es).magnitude:.1%}")  # ~62%
```

### Dewpoint Depression (T - Td)

Not a separate variable -- just `T - Td`. But it is so commonly used as a
quick moisture proxy that it deserves a mention. You will see it on weather
maps, sounding plots, and in severe weather discussions.

```
  T - Td       Meaning
  --------     ----------------------------------------
  0-3 degC     Near saturation. Fog, low clouds likely.
  5-10 degC    Moderate moisture. Typical summer day.
  15+ degC     Very dry air. Desert, or dry intrusion aloft.
```

### How They All Relate

```
                    Vapor Pressure (e)
                   /        |
                  /         |  e = es(Td)
                 /          |
   Mixing Ratio (w)    Saturation VP (es)
        |                   |
        |  q = w/(1+w)      |  RH = e / es
        |                   |
   Specific Humidity (q)    Relative Humidity
```

All of these are just different ways to express the same underlying
quantity: how much water vapor is in the air. metrust provides conversion
functions between all of them.

---

## Soundings -- A Vertical Slice of the Atmosphere

A **sounding** is the single most important data structure in meteorology.
It is a vertical profile of the atmosphere at a specific time and place.

### How soundings are collected

Twice a day (0000 and 1200 UTC), weather stations around the world launch
*radiosondes* -- instrument packages attached to helium balloons. As the
balloon ascends, the radiosonde measures temperature, humidity, pressure,
and wind at each level, transmitting data back to the ground. The balloon
eventually pops at around 30 km altitude.

```
     .  .  .  .  .  .  .  .  .  .   <- ~30 km, balloon pops
     |
     |   radiosonde measures T, Td, wind
     |   at each pressure level
     |
     O  <- balloon
     |
     |
  ///|///  <- surface
```

### The data format

A sounding is a set of parallel arrays, one value per pressure level:

```python
import numpy as np
from metrust.units import units

# A simplified 7-level sounding
p  = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T  = np.array([  30,  25,  20,  10, -10, -35, -55]) * units.degC
Td = np.array([  22,  18,  14,   2, -20, -45, -65]) * units.degC

# Wind as u and v components (m/s)
u  = np.array([  2,   5,   8,  15,  25,  35,  40]) * units("m/s")
v  = np.array([  5,   8,  10,  12,  10,   5,   0]) * units("m/s")
```

**Key convention: pressure DECREASES going up.** The first element
(index 0) is the surface, where pressure is highest. The last element is
the top of the sounding. This is the standard in metrust (and MetPy).

```
  Index    Pressure    Altitude (approx)
  -----    --------    -----------------
  [0]      1000 hPa    Surface
  [1]       925 hPa    ~750 m
  [2]       850 hPa    ~1500 m
  [3]       700 hPa    ~3000 m
  [4]       500 hPa    ~5500 m
  [5]       300 hPa    ~9000 m
  [6]       200 hPa    ~12000 m
```

### What you can compute from a sounding

A sounding is the input for almost every calculation in metrust. From this
one data structure you can determine:

- Where clouds will form (LCL)
- Whether thunderstorms are possible (CAPE, CIN)
- How strong those storms might be (severe weather parameters)
- What type of storms to expect (wind shear, hodograph shape)
- Precipitation type (temperature profile through the column)

We will cover all of these next.

---

## Stability -- Will Air Rise or Sink?

Stability is the central question in weather forecasting: **if I push
a parcel of air upward, does it keep going or does it sink back down?**

### The Parcel Model

Imagine taking a small "parcel" of air from the surface and lifting it.
As it rises, pressure drops and the parcel expands and cools. The key
question is: how does the parcel's temperature compare to its surroundings?

```
        UNSTABLE                          STABLE
  parcel is WARMER                  parcel is COOLER
  than environment                  than environment
  -> keeps rising                   -> sinks back down

  Height                            Height
    |         x  <- parcel            |    x  <- parcel
    |        /   (warmer, rises)      |     \   (cooler, sinks)
    |       /                         |      \
    |      /                          |       \
    |     /                           |        \
    |    /  <- environment            |     <- environment
    |   /                             |       /
    +---------> Temperature           +---------> Temperature
```

### Critical Levels

Meteorologists have names for the key levels a rising parcel encounters.
These are computed from the sounding data.

**LCL (Lifting Condensation Level):** The altitude where the rising parcel
cools to its dewpoint. Water vapor condenses and a cloud forms. This is
literally the cloud base.

```python
# Find the LCL from surface conditions
p_lcl, T_lcl = calc.lcl(1000 * units.hPa, 30 * units.degC, 22 * units.degC)
print(f"Cloud base: {p_lcl:.0f}")   # ~887 hPa
print(f"LCL temp:   {T_lcl:.1f}")   # ~22.4 degC
```

**LFC (Level of Free Convection):** Above the LCL, the parcel follows a
moist adiabat (cools more slowly because condensation releases latent
heat). If it eventually becomes warmer than the environment, that crossover
point is the LFC. Above the LFC, the parcel is buoyant and rises freely --
a thunderstorm is developing.

```python
p_lfc = calc.lfc(p, T, Td)
print(f"LFC: {p_lfc:.0f}")
```

**EL (Equilibrium Level):** As the parcel continues to rise above the LFC,
it is warmer than the environment and accelerates upward. Eventually, it
reaches a level where its temperature matches the environment again. This
is the EL -- the theoretical cloud top. Overshooting tops on severe
thunderstorms punch above the EL.

```python
p_el = calc.el(p, T, Td)
print(f"EL: {p_el:.0f}")
```

### CAPE and CIN

These two numbers are the most important stability parameters in severe
weather forecasting.

```
  Pressure (hPa)
    200  +---------+
         |     EL  x............  <- parcel meets environment
    300  |        / :
         |       /  : <- CAPE
    400  |      /   :    (parcel warmer than environment)
         |     /    :    (total energy available for storms)
    500  |    /     :
         |   /  LFC x............  <- parcel becomes buoyant
    600  |  :   /
         |  :  /    <- CIN
    700  |  : /         (parcel cooler than environment)
         |  :/          (energy barrier to overcome)
    800  |  x LCL
         | /
    900  |/
         x Surface
   1000  +---------+
```

**CAPE (Convective Available Potential Energy):** The total energy
available to accelerate the parcel upward between the LFC and the EL.
Measured in J/kg. Think of it as the fuel for thunderstorms.

```
  CAPE Value      What It Means
  ----------      ----------------------------------------
  0 J/kg          No instability. No thunderstorms.
  100-1000        Marginal. Weak storms possible.
  1000-2500       Moderate. Strong thunderstorms likely.
  2500-4000       High. Severe storms possible.
  >4000           Extreme. Violent storms possible.
```

**CIN (Convective Inhibition):** The energy barrier the parcel must
overcome to reach the LFC. Think of it as a "cap" or lid on convection.
Some CIN is often *desired* in severe weather setups -- it lets CAPE build
all day, then when the cap breaks (from heating, a front, etc.), storms
explode.

```
  CIN Value        What It Means
  ---------        ----------------------------------------
  0 to -50 J/kg    Weak cap. Storms can fire easily.
  -50 to -200      Moderate cap. Needs a trigger (front, dryline).
  < -200           Strong cap. Very hard to break.
```

```python
# Full CAPE/CIN calculation from a sounding
# Requires pressure, temperature, dewpoint, height, surface obs
p   = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
T   = np.array([  30,  25,  20,  10, -10, -35, -55]) * units.degC
Td  = np.array([  22,  18,  14,   2, -20, -45, -65]) * units.degC
h   = np.array([   0, 750,1500,3000,5500,9000,12000]) * units.m

cape, cin, lcl_h, lfc_h = calc.cape_cin(
    p, T, Td, h,
    psfc=1000 * units.hPa,
    t2m=30 * units.degC,
    td2m=22 * units.degC,
)
print(f"CAPE: {cape:.0f}")         # Energy for storms
print(f"CIN:  {cin:.0f}")          # Cap strength
print(f"LCL height: {lcl_h:.0f}") # Cloud base height AGL
print(f"LFC height: {lfc_h:.0f}") # Free convection height AGL
```

---

## Wind

Wind might seem straightforward -- it blows, things move. But in
meteorology, *how wind changes with height* is as important as the
wind itself, especially for severe weather.

### Components: u and v

Wind in atmospheric data is almost always stored as two components:

- **u**: east-west component. Positive u = wind blowing toward the east.
- **v**: north-south component. Positive v = wind blowing toward the north.

```
              North (+v)
                ^
                |
                |
  West (-u) <--+---> East (+u)
                |
                |
                v
              South (-v)
```

### Speed and Direction

Wind speed is just the magnitude: `speed = sqrt(u^2 + v^2)`

Wind direction follows a critical convention that trips up many developers:
**wind direction is where the wind blows FROM, not where it blows TO.**

- A **270-degree wind** blows FROM the west (toward the east).
- A **180-degree wind** blows FROM the south (toward the north).
- A **360-degree (or 0-degree) wind** blows FROM the north.

```python
# Compute speed and direction from components
u = np.array([10, -5, 0, 8]) * units("m/s")
v = np.array([ 5, -5, 10, 0]) * units("m/s")

speed = calc.wind_speed(u, v)
direction = calc.wind_direction(u, v)

print(speed)      # [11.2, 7.1, 10.0, 8.0] m/s
print(direction)  # meteorological direction (FROM)

# Go the other way: speed + direction -> u, v
u_calc, v_calc = calc.wind_components(
    np.array([15, 10]) * units("m/s"),
    np.array([225, 180]) * units.degree
)
```

### Vertical Wind Shear

Wind shear is the change in wind speed or direction (or both) with height.
It is the single most important factor in determining storm type.

**Bulk shear** is the simple vector difference between wind at two levels.
The 0-6 km bulk shear is the standard measure:

```python
# Wind shear between surface and 6 km
height = np.array([0, 1000, 3000, 6000]) * units.m
u_prof = np.array([2, 8, 20, 35]) * units("m/s")
v_prof = np.array([5, 8, 10, 5]) * units("m/s")

shear_u, shear_v = calc.bulk_shear(
    u_prof, v_prof, height,
    0 * units.m,       # bottom
    top=6000 * units.m  # top
)
shear_mag = calc.wind_speed(
    np.array([shear_u.magnitude]) * units("m/s"),
    np.array([shear_v.magnitude]) * units("m/s")
)
print(f"0-6 km shear: {shear_mag[0]:.1f}")  # magnitude in m/s
```

```
  Bulk Shear (0-6 km)    Storm Type
  --------------------    ----------------------------------------
  < 20 kt (~10 m/s)      Single-cell storms. Weak, short-lived.
  20-40 kt (10-20 m/s)   Multicell storms. Clusters, squall lines.
  > 40 kt (20+ m/s)      Supercells possible. Long-lived, rotating.
  > 60 kt (30+ m/s)      Strongly supercellular. Tornado risk.
```

### Wind Veering and Backing

When wind direction turns clockwise with height (e.g., south at the
surface, southwest at 1 km, west at 3 km), it is called **veering**.
Veering indicates warm air advection -- warm air is being transported
into the area. This is the classic severe weather wind profile.

Counterclockwise turning is **backing**, indicating cold advection.

### Hodograph

A hodograph plots the wind vector at each level as a curve. The shape
of the hodograph tells you about storm behavior:

```
  v (m/s)
  20 +
     |          6 km
  15 +         /
     |        /
  10 +    3 km
     |      /
   5 +  1 km
     |  /
   0 + sfc
     +--+--+--+--+--+
     0  5  10 15 20 25  u (m/s)
```

A straight-line hodograph favors splitting storms. A curved (veering)
hodograph favors right-moving supercells -- the type that produce the
most significant tornadoes.

### Storm Motion

Storms do not just move with the wind. Supercells deviate from the mean
wind due to their rotating updraft. The **Bunkers method** estimates
right-moving and left-moving supercell motion:

```python
right_mover, left_mover, mean = calc.bunkers_storm_motion(
    u_prof, v_prof, height
)
print(f"Right-mover: u={right_mover[0]:.1f}, v={right_mover[1]:.1f}")
print(f"Left-mover:  u={left_mover[0]:.1f}, v={left_mover[1]:.1f}")
print(f"Mean wind:   u={mean[0]:.1f}, v={mean[1]:.1f}")
```

---

## Severe Weather Parameters

These are composite indices that combine multiple ingredients to assess
the likelihood and potential severity of different storm types. They
are the bread and butter of operational severe weather forecasting.

**Important caveat:** These parameters are *ingredients-based*. A high
value means the ingredients are present, not that severe weather *will*
happen. A significant tornado parameter (STP) of 5 means the environment
strongly favors significant tornadoes -- but a tornado also needs a
trigger (front, outflow boundary, etc.) and the right storm-scale
dynamics. Conversely, tornadoes occasionally occur with low parameter
values. Treat these as probabilistic guides, not deterministic forecasts.

### Storm-Relative Helicity (SRH)

SRH measures how much the wind profile would cause rotation in a storm's
updraft. It is computed relative to the storm's motion -- hence
"storm-relative."

The 0-1 km SRH focuses on low-level rotation (tornado potential).
The 0-3 km SRH captures broader mesocyclone rotation.

```
  SRH (0-1 km)           Interpretation
  ---------------         ----------------------------------------
  50-150 m^2/s^2          Weak rotation potential
  150-300 m^2/s^2         Significant tornado risk
  300-500 m^2/s^2         Very large tornado risk
  > 500 m^2/s^2           Extreme (rare, violent tornadoes)
```

```python
height = np.array([0, 250, 500, 750, 1000, 2000, 3000, 6000]) * units.m
u_prof = np.array([0, 3, 5, 8, 12, 20, 25, 35]) * units("m/s")
v_prof = np.array([5, 7, 10, 12, 10, 8, 5, 0]) * units("m/s")

# Need storm motion -- use Bunkers
(ru, rv), _, _ = calc.bunkers_storm_motion(u_prof, v_prof, height)

# Compute 0-1 km SRH
pos_srh, neg_srh, total_srh = calc.storm_relative_helicity(
    u_prof, v_prof, height,
    1000 * units.m,  # depth
    ru, rv            # storm motion
)
print(f"0-1 km SRH: {total_srh:.0f}")
```

### Significant Tornado Parameter (STP)

STP combines four ingredients that research has found are present in
environments that produce significant (EF2+) tornadoes:

```
  STP = (MLCAPE / 1500) * ((2000 - LCL) / 1000) * (SRH / 150) * (shear / 20)
```

Each term captures a different ingredient:
- **MLCAPE**: instability (mixed-layer CAPE, in J/kg)
- **LCL height**: cloud base height (lower = more tornado-friendly)
- **SRH**: low-level rotation (0-1 km, in m^2/s^2)
- **Bulk shear**: deep-layer shear (0-6 km, in m/s)

```
  STP Value     Interpretation
  ---------     ----------------------------------------
  < 1           Below threshold. Tornadoes unlikely.
  1-3           Favorable for significant tornadoes.
  3-6           Very favorable for significant tornadoes.
  > 6           Extreme. Violent tornado environment.
```

```python
stp = calc.significant_tornado_parameter(
    2500 * units("J/kg"),       # MLCAPE
    800 * units.m,              # LCL height AGL
    250 * units("m**2/s**2"),   # 0-1 km SRH
    25 * units("m/s"),          # 0-6 km bulk shear
)
print(f"STP: {stp:.1f}")  # > 1 means favorable
```

### Supercell Composite Parameter (SCP)

SCP assesses whether the environment supports supercell thunderstorms
(long-lived rotating storms). Supercells produce the majority of
significant tornadoes, large hail, and damaging winds.

```
  SCP = (MUCAPE / 1000) * (SRH / 50) * (shear / 40)
```

```
  SCP Value     Interpretation
  ---------     ----------------------------------------
  < 1           Supercells unlikely from this ingredient alone.
  1-4           Favorable for supercells.
  > 4           Strongly favorable for supercells.
```

```python
scp = calc.supercell_composite_parameter(
    3000 * units("J/kg"),       # MUCAPE
    200 * units("m**2/s**2"),   # Effective SRH
    25 * units("m/s"),          # Effective bulk shear
)
print(f"SCP: {scp:.1f}")
```

### Putting It Together: A Severe Weather Assessment

In practice, forecasters look at multiple parameters together. Here is
a pattern you might use to assess a sounding:

```python
import numpy as np
from metrust.units import units
from metrust import calc

# --- Define sounding data ---
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
cape, cin, lcl_h, lfc_h = calc.cape_cin(
    p, T, Td, h,
    psfc=1013 * units.hPa,
    t2m=32 * units.degC,
    td2m=23 * units.degC,
)
print(f"CAPE:  {cape:.0f}")
print(f"CIN:   {cin:.0f}")

# --- Storm motion ---
(ru, rv), _, _ = calc.bunkers_storm_motion(u, v, h)

# --- 0-1 km SRH ---
_, _, srh_01 = calc.storm_relative_helicity(
    u, v, h, 1000 * units.m, ru, rv
)
print(f"0-1 km SRH: {srh_01:.0f}")

# --- 0-6 km bulk shear ---
su, sv = calc.bulk_shear(u, v, h, 0 * units.m, top=6000 * units.m)
shear_06 = calc.wind_speed(
    np.array([su.magnitude]) * units("m/s"),
    np.array([sv.magnitude]) * units("m/s")
)
print(f"0-6 km shear: {shear_06[0]:.1f}")

# --- Composite parameters ---
stp = calc.significant_tornado_parameter(cape, lcl_h, srh_01, shear_06[0])
print(f"STP: {stp:.1f}")
```

---

## Grids -- The Full Picture

So far, everything has been about a single sounding -- one vertical
column. But weather models produce 3-D grids covering entire regions
or the globe. This is where metrust's Rust backend really shines.

### What a Weather Model Grid Looks Like

Numerical Weather Prediction (NWP) models divide the atmosphere into a
3-D grid of cells. Common models:

```
  Model    Resolution    Coverage    Update Frequency
  -----    ----------    --------    ----------------
  HRRR     3 km          CONUS       Hourly
  NAM      12 km         N. America  Every 6 hours
  GFS      0.25 deg      Global      Every 6 hours
  ERA5     0.25 deg      Global      Hourly (reanalysis)
```

Each grid point has values for temperature, moisture, wind, pressure, and
other variables at every vertical level. The data shape is typically
`(nz, ny, nx)` -- levels by latitude by longitude.

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

### Computing Derived Fields Over the Grid

Computing CAPE for a single sounding is straightforward. Computing it for
every one of the 500,000+ grid points in an HRRR domain is where
performance matters. metrust's `compute_*` functions process the entire
grid in parallel using Rust, which is dramatically faster than looping in
Python.

```python
import numpy as np
from metrust.units import units
from metrust import calc

# Suppose you have loaded 3-D model data (e.g., from GRIB files)
# Shape: (nz, ny, nx) = (50, 1059, 1799) for a typical HRRR grid
# These would normally come from xarray/cfgrib, shown here as concept:

# pressure_3d    shape (nz, ny, nx) in Pa
# temperature_3d shape (nz, ny, nx) in Celsius
# qvapor_3d      shape (nz, ny, nx) in kg/kg
# height_agl_3d  shape (nz, ny, nx) in meters AGL
# psfc           shape (ny, nx) in Pa
# t2m            shape (ny, nx) in K
# q2m            shape (ny, nx) in kg/kg

# --- CAPE and CIN for every grid point ---
cape, cin, lcl_h, lfc_h = calc.compute_cape_cin(
    pressure_3d, temperature_3d, qvapor_3d, height_agl_3d,
    psfc, t2m, q2m,
    parcel_type="surface",
)
# cape.shape = (ny, nx), units J/kg

# --- Storm-relative helicity ---
srh_1km = calc.compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0)
# srh_1km.shape = (ny, nx), units m^2/s^2

# --- 0-6 km bulk shear ---
shear_06 = calc.compute_shear(u_3d, v_3d, height_agl_3d,
                               bottom_m=0.0, top_m=6000.0)
# shear_06.shape = (ny, nx), units m/s

# --- STP and SCP from the 2-D fields ---
stp = calc.compute_stp(cape, lcl_h, srh_1km, shear_06)
scp = calc.compute_scp(cape, srh_1km, shear_06)
# Both shape (ny, nx), dimensionless
```

### Other Grid-Level Computations

metrust provides parallel grid computations for many derived fields:

| Function | What it computes |
|---|---|
| `compute_cape_cin()` | CAPE, CIN, LCL height, LFC height |
| `compute_srh()` | Storm-relative helicity |
| `compute_shear()` | Bulk wind shear |
| `compute_stp()` | Significant Tornado Parameter |
| `compute_scp()` | Supercell Composite Parameter |
| `compute_ehi()` | Energy-Helicity Index |
| `compute_ship()` | Significant Hail Parameter |
| `compute_dcp()` | Derecho Composite Parameter |
| `compute_lapse_rate()` | Environmental lapse rate |
| `compute_pw()` | Precipitable water |

All of these take (nz, ny, nx) or (ny, nx) shaped arrays and return
(ny, nx) results, processing every grid column in parallel.

---

## Quick Reference: "What Function Do I Call?"

| I want to compute... | metrust function |
|---|---|
| Potential temperature | `calc.potential_temperature(p, T)` |
| Equivalent potential temperature | `calc.equivalent_potential_temperature(p, T, Td)` |
| Wet-bulb temperature | `calc.wet_bulb_temperature(p, T, Td)` |
| Virtual temperature | `calc.virtual_temperature(T, p, Td)` |
| Dewpoint from RH | `calc.dewpoint_from_relative_humidity(T, rh)` |
| RH from dewpoint | `calc.relative_humidity_from_dewpoint(T, Td)` |
| Saturation vapor pressure | `calc.saturation_vapor_pressure(T)` |
| Vapor pressure | `calc.vapor_pressure(Td)` |
| Mixing ratio | `calc.mixing_ratio(p, T)` |
| Saturation mixing ratio | `calc.saturation_mixing_ratio(p, T)` |
| Specific humidity from mixing ratio | `calc.specific_humidity_from_mixing_ratio(w)` |
| Mixing ratio from specific humidity | `calc.mixing_ratio_from_specific_humidity(q)` |
| LCL | `calc.lcl(p_sfc, T_sfc, Td_sfc)` |
| LFC | `calc.lfc(p, T, Td)` |
| EL | `calc.el(p, T, Td)` |
| CAPE and CIN | `calc.cape_cin(p, T, Td, h, psfc=, t2m=, td2m=)` |
| Wind speed | `calc.wind_speed(u, v)` |
| Wind direction | `calc.wind_direction(u, v)` |
| u, v from speed/direction | `calc.wind_components(speed, direction)` |
| Bulk shear | `calc.bulk_shear(u, v, h, bottom, top=)` |
| Storm motion | `calc.bunkers_storm_motion(u, v, h)` |
| Storm-relative helicity | `calc.storm_relative_helicity(u, v, h, depth, su, sv)` |
| Significant Tornado Parameter | `calc.significant_tornado_parameter(cape, lcl_h, srh, shear)` |
| Supercell Composite Parameter | `calc.supercell_composite_parameter(cape, srh, shear)` |

---

## What Next?

You now understand the building blocks. From here:

- **[API Reference](../api/thermodynamics.md)** -- detailed docs for every function
- **[Wind & Kinematics](../api/wind.md)** -- full wind analysis API
- **[Severe Weather](../api/severe.md)** -- all composite parameters
- **[Grid Composites](../api/grid-composites.md)** -- parallel grid computation
- **[Migration from MetPy](../guides/migration.md)** -- if you are coming from MetPy
