# Reading the Numbers -- A Meteorological Cheat Sheet

So you ran `cape_cin()` and got back 3500 J/kg. Is that a lot?

This page is your field reference. Every parameter that metrust can compute is
listed here with its units, operational thresholds, and plain-English meaning.
Bookmark it, print it, tape it to your monitor.

---

## Instability

### CAPE -- Convective Available Potential Energy

**Units:** J/kg

CAPE is the total energy available to a thunderstorm's updraft. On a
thermodynamic diagram, it is the area between the lifted parcel curve and the
environmental temperature where the parcel is warmer. Bigger CAPE means a
more violent updraft, stronger hail, and heavier rain.

| CAPE (J/kg)  | Category              | What to expect                                    |
|--------------|-----------------------|---------------------------------------------------|
| 0--500       | Marginal instability  | Weak showers, maybe isolated thunder.             |
| 500--1000    | Weak instability      | Thunderstorms likely if triggered.                |
| 1000--2500   | Moderate instability  | Strong thunderstorms, large hail possible.        |
| 2500--4000   | Strong instability    | Severe thunderstorms, very large hail.            |
| 4000+        | Extreme instability   | Violent storms if anything triggers them.         |

!!! warning "CAPE alone does not equal severe weather"
    A sounding can have 5000 J/kg of CAPE and produce nothing. Without a
    trigger mechanism (front, dryline, outflow boundary) and sufficient wind
    shear, all that energy just sits there. Conversely, strong tornadoes have
    occurred with CAPE as low as 500 J/kg when shear and low-level moisture
    were favorable. Always evaluate CAPE alongside shear and CIN.

**Variants you will encounter:**

- **SBCAPE** -- Surface-based. Uses the observed surface parcel.
- **MLCAPE** -- Mixed-layer. Averages the lowest ~100 hPa. This is the
  standard for SPC operations and the input to STP.
- **MUCAPE** -- Most-unstable. Uses the parcel with the highest theta-e in
  the lowest 300 hPa. Best for elevated convection.

```python
from metrust.calc import cape_cin

# Surface-based CAPE
cape, cin, h_lcl, h_lfc = cape_cin(
    p, t, td, hgt, psfc, t2m, td2m, parcel_type="sb"
)

# Mixed-layer CAPE
cape, cin, h_lcl, h_lfc = cape_cin(
    p, t, td, hgt, psfc, t2m, td2m, parcel_type="ml"
)

# Most-unstable CAPE
cape, cin, h_lcl, h_lfc = cape_cin(
    p, t, td, hgt, psfc, t2m, td2m, parcel_type="mu"
)
```

---

### CIN -- Convective Inhibition

**Units:** J/kg (negative values)

CIN is the energy barrier a parcel must overcome before it can rise freely --
the "cap." It is the area on a sounding where the lifted parcel is cooler than
the environment, below the LFC. CIN is why not every hot, humid day produces
storms.

| CIN (J/kg)   | Category        | What to expect                                         |
|--------------|-----------------|--------------------------------------------------------|
| 0 to -25     | Weak cap        | Storms initiate easily; widespread convection likely.  |
| -25 to -50   | Moderate cap    | Need a trigger (front, dryline, terrain).              |
| -50 to -100  | Strong cap      | Hard to break. If broken: explosive development.       |
| < -100       | Very strong cap | Unlikely to break without extreme forcing.             |

!!! tip "The cap is not always your enemy"
    A moderate cap (-25 to -75 J/kg) can actually help severe weather
    forecasts. It prevents storms from firing early and scattering the
    instability across many weak cells. Instead, the boundary layer continues
    to heat and moisten throughout the day, building even more CAPE. When the
    cap finally breaks -- often along a front or dryline in late afternoon --
    the result is fewer but much more intense storms. Forecasters call this
    "uncapping."

**metrust:** Returned as the second value from `cape_cin()`. See CAPE section above.

---

### Lapse Rates

**Units:** deg C/km

The rate at which temperature decreases with height. The **700--500 hPa lapse
rate** is the standard mid-level measure. Steeper lapse rates mean a parcel
that starts rising stays warmer than its surroundings longer, producing
stronger buoyancy.

| Lapse Rate (deg C/km) | Category                | What to expect                               |
|------------------------|-------------------------|----------------------------------------------|
| < 6                    | Stable                  | Weak convection, limited hail growth.        |
| 6--7                   | Conditionally unstable  | Moderate storms with adequate moisture.       |
| 7--8                   | Steep                   | Strong updrafts, favorable for large hail.   |
| > 8                    | Very steep              | Extreme instability. Approaching dry adiabatic (~9.8). |

!!! tip "Mid-level lapse rates and hail"
    The 700--500 hPa lapse rate is one of the best discriminators for
    significant hail. Steep mid-level lapse rates (> 7 deg C/km) keep
    hailstones suspended in the updraft longer, allowing them to grow larger
    before falling out. When you see steep lapse rates combined with high
    CAPE, think "big hail day."

**metrust:** Compute from temperature profiles at 700 and 500 hPa, or derive
from `dry_lapse()` / `moist_lapse()`.

---

## Sounding Levels

### LCL -- Lifting Condensation Level

**Units:** hPa (pressure) or m AGL (height)

The height where clouds form when surface air is lifted. Air cools
dry-adiabatically until it saturates -- that point is the LCL. The LCL marks
the base of cumulus clouds and is one of the critical factors for tornado
potential.

| LCL Height (m AGL) | What it means                              |
|---------------------|--------------------------------------------|
| < 1000              | Very low cloud bases. Favorable for tornadoes. |
| 1000--1500          | Moderate. Tornadoes possible.              |
| > 1500              | High cloud bases. Tornadoes less likely.   |

!!! warning "Low LCL = tornado risk factor"
    A low LCL means less distance between the cloud base and the ground.
    When a mesocyclone's rotation stretches downward from cloud base, a
    shorter distance means it is more likely to reach the surface as a
    tornado. Most significant tornadoes occur with LCL heights below 1500 m
    AGL. The STP formula explicitly penalizes LCL heights above 2000 m.

```python
from metrust.calc import lcl
from metrust.units import units

p_lcl, t_lcl = lcl(1000 * units.hPa, 30 * units.degC, 20 * units.degC)
```

LCL height AGL is also returned as the third value from `cape_cin()`.

---

### LFC -- Level of Free Convection

**Units:** hPa

The height where a lifted parcel becomes warmer than its surroundings and
starts rising on its own. Below the LFC, external forcing (a front, outflow
boundary, terrain) must push the parcel upward. Above the LFC, buoyancy takes
over and the parcel accelerates freely.

A lower LFC (higher pressure) means less forcing is needed to initiate
convection. The gap between the LCL and LFC corresponds roughly to CIN -- a
large gap means a strong cap.

```python
from metrust.calc import lfc

p_lfc = lfc(p, t, td)
```

LFC height AGL is also returned as the fourth value from `cape_cin()`.

---

### EL -- Equilibrium Level

**Units:** hPa

The height where the rising parcel stops being warmer than the environment.
This is approximately the storm cloud top. A higher EL (lower pressure)
means taller storms with more total energy. The CAPE integral runs from the
LFC to the EL, so a high EL directly implies large CAPE.

Storm tops near or above the tropopause (roughly 100--200 hPa, depending on
season and latitude) indicate deep, powerful updrafts. Overshooting tops --
where the updraft punches above the EL -- are a radar signature of the
most intense storms.

```python
from metrust.calc import el

p_el = el(p, t, td)
```

---

## Wind Shear and Rotation

### Bulk Wind Shear (0--6 km)

**Units:** m/s or knots

The vector difference in wind between two heights. The 0--6 km layer is the
standard measure for deep-layer shear, which controls storm mode: whether a
thunderstorm is a brief pulse, a multicell cluster, or a long-lived supercell.

| 0--6 km Shear          | Category | Storm mode                                  |
|-------------------------|----------|---------------------------------------------|
| < 10 m/s (< 20 kt)     | Weak     | Single-cell storms, brief and disorganized. |
| 10--18 m/s (20--35 kt)  | Moderate | Multicell storms, some organization.        |
| 18--25 m/s (35--50 kt)  | Strong   | Supercells likely.                          |
| > 25 m/s (> 50 kt)     | Extreme  | Long-lived, potentially violent supercells. |

!!! tip "Low-level shear matters too"
    The 0--1 km shear is a better discriminator for tornadoes than the 0--6 km
    shear. Strong low-level shear (> 10 m/s in the lowest kilometer) increases
    the likelihood that a supercell will produce a tornado, even when deep-layer
    shear is only moderate.

```python
from metrust.calc import bulk_shear
from metrust.units import units

su, sv = bulk_shear(u, v, height, bottom=0 * units.m, top=6000 * units.m)
```

---

### SRH -- Storm-Relative Helicity

**Units:** m^2/s^2

SRH measures the potential for rotating updrafts (mesocyclones). It answers:
"How much does the wind vector curl with height, relative to the storm
motion?" The 0--1 km layer targets low-level rotation (tornado potential);
the 0--3 km layer targets overall supercell and mesocyclone potential.

**0--1 km SRH (tornado potential):**

| SRH (m^2/s^2) | Category | What to expect                        |
|----------------|----------|---------------------------------------|
| < 50           | Weak     | Minimal low-level rotation.           |
| 50--150        | Moderate | Weak tornadoes possible.              |
| 150--300       | Strong   | Significant tornadoes possible.       |
| 300+           | Extreme  | Violent tornadoes possible.           |

**0--3 km SRH (supercell/mesocyclone potential):**

| SRH (m^2/s^2) | Category | What to expect                        |
|----------------|----------|---------------------------------------|
| < 100          | Weak     | Minimal mesocyclone potential.        |
| 100--250       | Moderate | Mesocyclones likely with storms.      |
| 250--450       | Strong   | Long-lived supercells.                |
| 450+           | Extreme  | Violent long-track supercells.        |

```python
from metrust.calc import storm_relative_helicity, bunkers_storm_motion
from metrust.units import units

# Get storm motion first
(rm_u, rm_v), _, _ = bunkers_storm_motion(u, v, height)

# 0-1 km SRH
pos, neg, total_1km = storm_relative_helicity(
    u, v, height, 1000 * units.m, rm_u, rm_v
)

# 0-3 km SRH
pos, neg, total_3km = storm_relative_helicity(
    u, v, height, 3000 * units.m, rm_u, rm_v
)
```

---

## Composite Parameters

These indices combine multiple ingredients into a single number. They are
the workhorses of operational severe weather forecasting.

### STP -- Significant Tornado Parameter

**Units:** dimensionless

The primary composite used by the Storm Prediction Center to assess tornado
threat. STP combines four ingredients, each normalized so that a value of 1
represents a climatologically favorable threshold:

    STP = (MLCAPE / 1500) * ((2000 - LCL) / 1000) * (SRH_1km / 150) * (shear_6km / 20)

| STP   | Category   | What to expect                                  |
|-------|------------|-------------------------------------------------|
| < 0.5 | Low        | Significant tornadoes unlikely.                 |
| 0.5--1| Conditional| Need a strong trigger. Watch for mesocyclones.  |
| 1--3  | Moderate   | Significant tornadoes possible.                 |
| 3--6  | High       | Multiple significant tornadoes likely.           |
| > 6   | Extreme    | Violent tornado outbreak. Particularly dangerous situation. |

!!! tip "What the components tell you"
    When STP is high, look at which terms are contributing. An STP of 4 driven
    primarily by extreme CAPE (4000+ J/kg) with modest SRH is a different
    threat than an STP of 4 driven by extreme SRH (400+ m^2/s^2) with
    moderate CAPE. The latter pattern -- high SRH, moderate CAPE -- is often
    associated with strong, long-track tornadoes in the Southeast US, sometimes
    embedded in squall lines where they are hard to see on radar.

```python
from metrust.calc import significant_tornado_parameter

stp = significant_tornado_parameter(
    mlcape,           # J/kg
    lcl_height,       # m AGL
    srh_0_1km,        # m^2/s^2
    bulk_shear_0_6km, # m/s
)
```

---

### SCP -- Supercell Composite Parameter

**Units:** dimensionless

Discriminates supercell environments from ordinary thunderstorm environments.
SCP combines most-unstable CAPE, effective-layer storm-relative helicity, and
effective bulk shear.

| SCP   | Category | What to expect                                |
|-------|----------|-----------------------------------------------|
| < 1   | Low      | Supercells unlikely.                          |
| 1--4  | Moderate | Supercells possible if storms initiate.       |
| 4--10 | High     | Supercells likely with any convective trigger.|
| > 10  | Extreme  | Long-lived violent supercells favored.        |

!!! warning "SCP does not predict tornadoes"
    SCP tells you whether supercells will form, not whether those supercells
    will produce tornadoes. A high SCP with low 0--1 km SRH and high LCL
    heights favors supercells that produce large hail and damaging winds but
    not tornadoes. Use STP for tornado discrimination.

```python
from metrust.calc import supercell_composite_parameter

scp = supercell_composite_parameter(
    mucape,          # J/kg
    srh_eff,         # m^2/s^2
    bulk_shear_eff,  # m/s
)
```

---

### SHIP -- Significant Hail Parameter

**Units:** dimensionless

SHIP estimates the likelihood of significant hail (diameter >= 2 inches / 5 cm).
It combines MUCAPE, mixing ratio, 700--500 hPa lapse rate, 500 hPa
temperature, and 0--6 km bulk shear. The freezing level and mid-level lapse
rates are the key discriminators: steep lapse rates aloft keep hailstones in
the growth zone longer.

| SHIP  | Category | What to expect                                |
|-------|----------|-----------------------------------------------|
| < 0.5 | Low      | Significant hail unlikely.                    |
| 0.5--1| Moderate | Large hail possible with strong updrafts.     |
| 1--2  | High     | Significant hail likely.                      |
| > 2   | Very high| Very large hail (> 2 in) strongly favored.   |

!!! tip "SHIP and the freezing level"
    SHIP works best in environments where the freezing level is between about
    3000 and 4500 m AGL. If the freezing level is very low (< 2500 m),
    hailstones melt before reaching the surface. If it is very high (> 5000 m),
    the warm layer below is deep enough to melt even large stones. The sweet
    spot for giant hail is a high freezing level combined with steep mid-level
    lapse rates and extreme CAPE.

**metrust:** Compute SHIP from its component parts:

```python
import numpy as np
from metrust.calc import cape_cin, bulk_shear, mixing_ratio
from metrust.units import units

# Gather components, then:
# SHIP = (MUCAPE * W * LR * (-T500) * SHEAR) / 44000000
# where W = mixing ratio (g/kg), LR = 700-500 hPa lapse rate,
# T500 = 500 hPa temperature (degC, negative), SHEAR = 0-6 km (m/s)
```

---

## Moisture

### Precipitable Water (PWAT)

**Units:** mm (or inches)

Total water vapor in the atmospheric column if all of it condensed out. PWAT
does not tell you how much rain will fall, but it sets an upper bound and
indicates how much moisture is available for storms to tap.

!!! warning "Thresholds are regional and seasonal"
    A PWAT of 40 mm in July over Oklahoma is unremarkable. The same value in
    January over Montana is extraordinary. Always compare PWAT to local
    climatological normals, not to a fixed table. The SPC sounding
    climatology and the NWS precipitable water climatology pages provide
    percentile ranks by station and month.

The table below gives rough warm-season mid-latitude guidelines:

| PWAT (mm) | PWAT (in) | Category       | What to expect                          |
|-----------|-----------|----------------|-----------------------------------------|
| < 25      | < 1.0     | Dry            | Limited moisture for storms.            |
| 25--40    | 1.0--1.6  | Moderate       | Adequate moisture for convection.       |
| 40--60    | 1.6--2.4  | Moist          | Heavy rain potential with any lift.     |
| 60--75    | 2.4--3.0  | Very moist     | Flash flood risk. Near-tropical moisture.|
| > 75      | > 3.0     | Extreme        | Tropical air mass. Extreme flood risk.  |

```python
from metrust.calc import precipitable_water

pw = precipitable_water(p, td)  # result in mm
```

---

## Kinematics

### Divergence and Vorticity

**Units:** 1/s (per second)

Divergence measures whether air is spreading apart (positive) or converging
(negative) at a given level. Vorticity measures the spin of the air -- positive
values indicate cyclonic (counterclockwise in the Northern Hemisphere) rotation.

The classic pattern that supports deep convection:

- **Upper-level divergence** (air spreading out aloft) removes mass from the
  column, lowering surface pressure.
- **Low-level convergence** (air piling in near the surface) forces upward
  motion to replace the evacuated air.
- **Positive vorticity advection (PVA)** at 500 hPa enhances upward motion
  downstream of a trough axis.

| Feature                      | Typical magnitude | Interpretation                          |
|------------------------------|-------------------|-----------------------------------------|
| Upper-level divergence       | 1e-5 to 5e-5 1/s | Supports ascent; stronger = more lift.  |
| Low-level convergence        | -1e-5 to -5e-5 1/s| Forces upward motion into convection.  |
| Positive vorticity (500 hPa) | 1e-5 to 1e-4 1/s | Cyclonic spin; PVA = storm support.     |

!!! tip "Divergence-vorticity coupling"
    In an idealized setting, upper-level divergence exceeding low-level
    convergence produces net mass evacuation from the column -- surface
    pressure falls and lift intensifies. When you see a diffluent jet-stream
    pattern over a surface low with converging winds, you have the textbook
    setup for explosive cyclogenesis or a severe weather outbreak.

```python
from metrust.calc import divergence, vorticity
from metrust.units import units

div  = divergence(u_grid, v_grid, dx, dy)   # 1/s, 2-D field
vort = vorticity(u_grid, v_grid, dx, dy)    # 1/s, 2-D field
```

---

## Precipitation Type

### Wet-Bulb Temperature

**Units:** deg C

The wet-bulb temperature is the lowest temperature air can reach through
evaporative cooling alone. It sits between the dewpoint and the air
temperature, and it is the single best predictor of surface precipitation type.

As precipitation falls through a layer, it cools the air toward the wet-bulb
temperature via evaporation and melting. If the wet-bulb temperature is below
freezing through the entire column, snow reaches the surface. If it is above
freezing, rain reaches the surface.

| Wet-Bulb Temp (deg C) | Precipitation type                               |
|------------------------|----------------------------------------------------|
| < 0                    | Snow likely (all ice to the surface).              |
| 0--1.3                 | Transition zone: snow, sleet, or freezing rain.    |
| > 1.3                  | Rain (enough warmth to fully melt frozen precip).  |

!!! warning "The 1 deg C rule of thumb"
    Many forecasters use a surface wet-bulb of 1 deg C as the rain/snow
    line, but this is only a starting point. Precipitation intensity matters:
    heavy precipitation cools the column toward the wet-bulb faster than
    light precipitation. A surface wet-bulb of 2 deg C can still produce
    accumulating snow at the onset of heavy precipitation before the column
    has fully cooled. The full wet-bulb profile through the column matters
    more than the surface value alone.

```python
from metrust.calc import wet_bulb_temperature
from metrust.units import units

tw = wet_bulb_temperature(
    1000 * units.hPa, 2 * units.degC, -1 * units.degC
)
```

---

## Stability Indices

Quick-look numbers computed from fixed pressure levels. Less sophisticated
than CAPE/shear composites, but fast and widely used.

| Index              | Thunderstorm Threshold | Severe Threshold          | Direction          |
|--------------------|------------------------|---------------------------|--------------------|
| **K-Index**        | > 30                   | > 40                      | Higher = worse     |
| **Total Totals**   | > 44                   | > 50--55                  | Higher = worse     |
| **Showalter Index**| < 0                    | < -3                      | More negative = worse |
| **Lifted Index**   | < 0                    | < -3 (very), < -6 (extreme)| More negative = worse |
| **SWEAT**          | > 300 (severe)         | > 400 (tornadoes)         | Higher = worse     |

```python
from metrust.calc import (
    k_index, total_totals, showalter_index, lifted_index, sweat_index,
)
from metrust.units import units

ki = k_index(t850, td850, t700, td700, t500)
tt = total_totals(t850, td850, t500)
si = showalter_index(p, t, td)
li = lifted_index(p, t, td)
sw = sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
```

---

## Other Composites

### EHI -- Energy-Helicity Index

**Units:** dimensionless

A simple composite that multiplies buoyancy and rotation potential:

    EHI = (CAPE * SRH) / 160000

| EHI  | What to expect                             |
|------|--------------------------------------------|
| < 1  | Minimal tornado potential.                 |
| 1--2 | Significant tornado potential.             |
| > 2  | Strong tornado potential.                  |

```python
ehi = (cape.magnitude * srh.magnitude) / 160000.0
```

### BRN -- Bulk Richardson Number

**Units:** dimensionless

The ratio of CAPE to the kinetic energy of the 0--6 km bulk wind shear.
Distinguishes supercellular from multicellular storm modes.

| BRN    | Storm mode                                             |
|--------|--------------------------------------------------------|
| < 10   | Shear-dominated. Storms may split or fail to sustain.  |
| 10--45 | Supercells favored.                                    |
| > 45   | Buoyancy-dominated. Multicell storms.                  |

```python
from metrust.calc import bulk_richardson_number
from metrust.units import units

brn = bulk_richardson_number(2000 * units("J/kg"), 20 * units("m/s"))
```

---

## Comfort and Safety

### Heat Index

**Units:** deg C or deg F

Apparent temperature accounting for humidity.

| Heat Index                          | Category        | Risk                                             |
|-------------------------------------|-----------------|--------------------------------------------------|
| 27--32 deg C (80--90 deg F)         | Caution         | Fatigue possible with prolonged exposure.        |
| 32--41 deg C (90--105 deg F)        | Extreme caution | Heat cramps and exhaustion possible.             |
| 41--54 deg C (105--130 deg F)       | Danger          | Heat cramps/exhaustion likely. Heatstroke possible.|
| > 54 deg C (> 130 deg F)           | Extreme danger  | Heatstroke highly likely.                        |

```python
from metrust.calc import heat_index

hi = heat_index(35 * units.degC, 80)  # 80% RH
```

### Wind Chill

**Units:** deg C or deg F

Apparent temperature accounting for wind.

| Wind Chill                                 | Risk                                       |
|--------------------------------------------|--------------------------------------------|
| -18 to -28 deg C (0 to -18 deg F)         | Frostbite possible in 30 minutes.          |
| -28 to -40 deg C (-18 to -40 deg F)       | Frostbite possible in 10--15 minutes.      |
| -40 to -48 deg C (-40 to -55 deg F)       | Frostbite possible in 5--10 minutes.       |
| < -48 deg C (< -55 deg F)                 | Frostbite possible in under 5 minutes.     |

```python
from metrust.calc import windchill

wc = windchill(-10 * units.degC, 8 * units("m/s"))
```

---

## Quick Reference Card

One table to rule them all. Clip this for the field.

| Parameter              | Units           | "Watch out" threshold              | "This is serious" threshold          |
|------------------------|-----------------|------------------------------------|--------------------------------------|
| **CAPE**               | J/kg            | > 1000                             | > 2500                               |
| **CIN**                | J/kg            | -50 (strong cap)                   | < -100 (very strong cap)             |
| **700--500 hPa LR**   | deg C/km        | > 7 (steep)                        | > 8 (very steep)                     |
| **LCL height**         | m AGL           | < 1500 (tornado risk)              | < 1000 (high tornado risk)           |
| **LFC**                | hPa             | Low LFC = easy initiation          | Very low LFC + low CIN = explosive   |
| **EL**                 | hPa             | < 250 hPa (tall storms)           | < 200 hPa (extreme depth)           |
| **0--6 km shear**      | m/s             | > 18 (supercells possible)         | > 25 (violent supercells)            |
| **0--1 km SRH**        | m^2/s^2         | > 150 (sig tornadoes possible)     | > 300 (violent tornadoes)            |
| **0--3 km SRH**        | m^2/s^2         | > 250 (long-lived supercells)      | > 450 (extreme rotation)            |
| **STP**                | dimensionless   | > 1 (sig tornado environment)      | > 4 (PDS-level tornado environment)  |
| **SCP**                | dimensionless   | > 1 (supercells possible)          | > 4 (supercells very likely)         |
| **SHIP**               | dimensionless   | > 1 (sig hail possible)            | > 2 (very large hail favored)        |
| **PWAT**               | mm              | > 40 (heavy rain potential)        | > 60 (flash flood risk)             |
| **Wet-bulb temp**      | deg C           | 0--1.3 (wintry mix zone)          | < 0 (snow likely)                    |
| **K-Index**            | dimensionless   | > 30 (thunderstorms)               | > 40 (numerous storms)              |
| **Total Totals**       | dimensionless   | > 44 (storms possible)             | > 55 (severe storms)                |
| **Lifted Index**       | deg C           | < -3 (very unstable)              | < -6 (extreme instability)           |
| **Divergence (upper)** | 1/s             | > 1e-5 (ascent support)           | > 3e-5 (strong forcing)             |
| **Vorticity (500 hPa)**| 1/s             | > 1e-5 (PVA likely)               | > 5e-5 (strong PVA)                 |
| **Heat Index**         | deg C           | > 32 (extreme caution)            | > 41 (danger)                        |
| **Wind Chill**         | deg C           | < -18 (frostbite in 30 min)       | < -40 (frostbite in 5 min)          |
| **EHI**                | dimensionless   | > 1 (sig tornado potential)        | > 2 (strong tornado potential)       |
| **BRN**                | dimensionless   | 10--45 (supercell sweet spot)      | < 10 or > 45 (less favorable)        |

!!! tip "How to use this table"
    Do not treat these thresholds as on/off switches. Weather is a
    continuous spectrum, and these numbers represent climatological
    tendencies, not physical laws. A parameter at 90% of its "watch out"
    threshold in combination with three other parameters all near their
    thresholds can be more dangerous than one parameter that is off the
    charts while everything else is zero. Always evaluate the full
    parameter space, not a single number.

---

## See Also

- [Thermodynamics API](../api/thermodynamics.md) -- CAPE/CIN, parcel profiles, LCL/LFC/EL, precipitable water.
- [Wind API](../api/wind.md) -- Bulk shear, storm-relative helicity, Bunkers storm motion.
- [Severe Weather API](../api/severe.md) -- STP, SCP, BRN, stability indices, SWEAT.
- [Kinematics API](../api/kinematics.md) -- Divergence, vorticity, frontogenesis, advection.
- [Atmospheric API](../api/atmospheric.md) -- Heat index, wind chill, layer functions.
- [Your First Sounding Analysis](first-sounding.md) -- Step-by-step tutorial that computes all of these parameters from scratch.
- [Common Workflows and Recipes](recipes.md) -- Copy-paste-ready code snippets.
