# Severe Weather

Functions for computing convective indices, composite parameters, and fire weather diagnostics.
All functions accept Pint Quantities and return Pint Quantities (or plain values where noted).

---

## Composite Parameters

### `significant_tornado_parameter`

Significant Tornado Parameter (STP). The primary composite used by the Storm Prediction Center
to discriminate between significant tornadic supercells (EF2+) and non-tornadic supercells.
STP combines mixed-layer CAPE, LCL height, 0--1 km storm-relative helicity, and 0--6 km
bulk wind shear into a single dimensionless number. Values above 1 indicate environments
increasingly favorable for significant tornadoes; most significant tornadoes occur with
STP in the 1--8 range. An STP of 0 means one or more ingredients are absent or below threshold.

```python
significant_tornado_parameter(mlcape, lcl_height, srh_0_1km, bulk_shear_0_6km)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `mlcape` | `Quantity (J/kg)` | Mixed-layer CAPE (typically 100 hPa mixed-layer parcel). |
| `lcl_height` | `Quantity (m)` | Lifting condensation level height above ground. |
| `srh_0_1km` | `Quantity (m^2/s^2)` | 0--1 km storm-relative helicity. |
| `bulk_shear_0_6km` | `Quantity (m/s)` | 0--6 km bulk wind shear magnitude. |

**Returns**

`Quantity (dimensionless)` -- the STP value.

**Example**

```python
from metrust.calc import significant_tornado_parameter
from metrust.units import units

stp = significant_tornado_parameter(
    2500 * units("J/kg"),
    800 * units.m,
    250 * units("m**2/s**2"),
    25 * units("m/s"),
)
print(stp)  # > 1 indicates significant tornado environment
```

---

### `supercell_composite_parameter`

Supercell Composite Parameter (SCP). Discriminates supercell environments from ordinary
thunderstorms. SCP multiplies most-unstable CAPE, effective-layer storm-relative helicity,
and effective bulk shear, each normalized by climatological thresholds. Values above 1
favor supercell development; values above 4--5 are associated with long-lived, strongly
rotating supercells. SPC mesoanalysts use SCP alongside STP to gauge the supercell and
tornado threat.

```python
supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `mucape` | `Quantity (J/kg)` | Most-unstable CAPE. |
| `srh_eff` | `Quantity (m^2/s^2)` | Effective-layer storm-relative helicity. |
| `bulk_shear_eff` | `Quantity (m/s)` | Effective bulk wind shear magnitude. |

**Returns**

`Quantity (dimensionless)` -- the SCP value.

**Example**

```python
from metrust.calc import supercell_composite_parameter
from metrust.units import units

scp = supercell_composite_parameter(
    3000 * units("J/kg"),
    300 * units("m**2/s**2"),
    20 * units("m/s"),
)
```

---

### `critical_angle`

Critical angle between the storm-relative inflow vector and the 0--500 m shear vector.
Research by Esterheld and Giuliano (2008) found that a critical angle near 90 degrees
is strongly associated with tornadic supercells. Angles well below 90 degrees suggest
the low-level shear and inflow are poorly aligned for mesocyclone intensification.

```python
critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `storm_u` | `Quantity (m/s)` | Storm motion u-component. |
| `storm_v` | `Quantity (m/s)` | Storm motion v-component. |
| `u_sfc` | `Quantity (m/s)` | Surface u-wind component. |
| `v_sfc` | `Quantity (m/s)` | Surface v-wind component. |
| `u_500m` | `Quantity (m/s)` | 500 m AGL u-wind component. |
| `v_500m` | `Quantity (m/s)` | 500 m AGL v-wind component. |

**Returns**

`Quantity (degree)` -- the critical angle in degrees.

**Example**

```python
from metrust.calc import critical_angle
from metrust.units import units

ca = critical_angle(
    10 * units("m/s"), 5 * units("m/s"),   # storm motion
    2 * units("m/s"), 8 * units("m/s"),    # surface wind
    12 * units("m/s"), 15 * units("m/s"),  # 500 m wind
)
# Values near 90 degrees favor tornadic supercells
```

---

### `bulk_richardson_number`

Bulk Richardson Number (BRN). The ratio of CAPE to the kinetic energy of the 0--6 km
bulk wind shear. BRN helps distinguish between supercellular and multicellular convective
modes. Values of 10--45 favor supercells; values below 10 suggest the shear may be
too strong relative to buoyancy (splitting or shear-dominated failure), and values
above 45 favor multicell storms.

```python
bulk_richardson_number(cape, shear_0_6km)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `cape` | `Quantity (J/kg)` | Convective available potential energy. |
| `shear_0_6km` | `Quantity (m/s)` | 0--6 km bulk wind shear magnitude. |

**Returns**

`Quantity (dimensionless)` -- the BRN value.

**Example**

```python
from metrust.calc import bulk_richardson_number
from metrust.units import units

brn = bulk_richardson_number(2000 * units("J/kg"), 20 * units("m/s"))
# BRN 10-45: supercell likely; > 45: multicell; < 10: too much shear
```

---

## Stability Indices

### `showalter_index`

Showalter Index (SI). A parcel-based stability index that lifts air from 850 hPa to
500 hPa and compares the parcel temperature to the environment. Negative values indicate
instability; values below -3 suggest moderate-to-strong instability. The Showalter Index
is less sensitive to boundary-layer moisture than the Lifted Index, making it useful when
an elevated mixed layer or cap is present.

```python
showalter_index(pressure, temperature, dewpoint)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `dewpoint` | `array Quantity (temperature)` | Dewpoint profile. |

**Returns**

`Quantity (delta_degC)` -- the Showalter Index value.

**Example**

```python
from metrust.calc import showalter_index
from metrust.units import units

p  = [1000, 925, 850, 700, 500] * units.hPa
T  = [25, 20, 15, 5, -15] * units.degC
Td = [20, 15, 10, -5, -25] * units.degC

si = showalter_index(p, T, Td)
# Negative values = unstable; < -3 = significant instability
```

---

### `lifted_index`

Lifted Index (LI). Computes the difference between the 500 hPa environmental temperature
and the temperature of a surface-based parcel lifted to 500 hPa. Negative values mean
the parcel is warmer than the environment (unstable). Operational thresholds: 0 to -2
is marginally unstable, -2 to -6 is moderately unstable, and below -6 is extremely
unstable with the potential for intense convection.

```python
lifted_index(pressure, temperature, dewpoint)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `dewpoint` | `array Quantity (temperature)` | Dewpoint profile. |

**Returns**

`Quantity (delta_degK)` -- the Lifted Index value.

**Example**

```python
from metrust.calc import lifted_index
from metrust.units import units

p  = [1000, 925, 850, 700, 500] * units.hPa
T  = [30, 24, 18, 4, -18] * units.degC
Td = [22, 18, 14, -4, -28] * units.degC

li = lifted_index(p, T, Td)
# LI < 0: unstable; LI < -6: extreme instability
```

---

### `k_index`

K-Index (KI). A multi-level stability index that combines low-level moisture depth
and mid-level lapse rate into a single number. Higher values indicate greater potential
for air-mass thunderstorms. Thresholds: below 20 means thunderstorms unlikely; 20--25
means isolated storms possible; 26--30 means widely scattered storms; 31--35 means
scattered storms likely; above 35 means numerous storms. The K-Index does not account
for wind shear and is best suited for air-mass (non-severe) convection forecasting.

```python
k_index(t850, td850, t700, td700, t500)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t850` | `Quantity (temperature)` | 850 hPa temperature (degC). |
| `td850` | `Quantity (temperature)` | 850 hPa dewpoint (degC). |
| `t700` | `Quantity (temperature)` | 700 hPa temperature (degC). |
| `td700` | `Quantity (temperature)` | 700 hPa dewpoint (degC). |
| `t500` | `Quantity (temperature)` | 500 hPa temperature (degC). |

**Returns**

`Quantity (delta_degC)` -- the K-Index value.

**Example**

```python
from metrust.calc import k_index
from metrust.units import units

ki = k_index(
    20 * units.degC,   # T 850
    16 * units.degC,   # Td 850
    8 * units.degC,    # T 700
    4 * units.degC,    # Td 700
    -12 * units.degC,  # T 500
)
# KI > 30: scattered thunderstorms likely
```

---

### `total_totals`

Total Totals Index (TT). The sum of the Vertical Totals (T850 - T500) and
Cross Totals (Td850 - T500). Values above 44 indicate a possibility of
thunderstorms; above 50--55 suggests a risk of severe thunderstorms with large
hail. TT is widely used in operational forecasting because it requires only
standard-level data and is straightforward to compute.

```python
total_totals(t850, td850, t500)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t850` | `Quantity (temperature)` | 850 hPa temperature (degC). |
| `td850` | `Quantity (temperature)` | 850 hPa dewpoint (degC). |
| `t500` | `Quantity (temperature)` | 500 hPa temperature (degC). |

**Returns**

`Quantity (delta_degC)` -- the Total Totals index value.

**Example**

```python
from metrust.calc import total_totals
from metrust.units import units

tt = total_totals(20 * units.degC, 16 * units.degC, -12 * units.degC)
# TT > 44: thunderstorms possible; > 55: severe storms possible
```

---

### `cross_totals`

Cross Totals (CT). Defined as Td850 - T500. The Cross Totals component captures
the combined effect of low-level moisture and mid-level cold air. Values above 18
indicate moderate instability; above 22 suggests strong instability favorable for
convection.

```python
cross_totals(td850, t500)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `td850` | `Quantity (temperature)` | 850 hPa dewpoint (degC). |
| `t500` | `Quantity (temperature)` | 500 hPa temperature (degC). |

**Returns**

`Quantity (delta_degC)` -- the Cross Totals value.

**Example**

```python
from metrust.calc import cross_totals
from metrust.units import units

ct = cross_totals(16 * units.degC, -12 * units.degC)
```

---

### `vertical_totals`

Vertical Totals (VT). Defined as T850 - T500. Measures the temperature lapse rate
between 850 hPa and 500 hPa. Values above 26 indicate steep lapse rates favorable
for deep convection, though Vertical Totals alone do not account for moisture.

```python
vertical_totals(t850, t500)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t850` | `Quantity (temperature)` | 850 hPa temperature (degC). |
| `t500` | `Quantity (temperature)` | 500 hPa temperature (degC). |

**Returns**

`Quantity (delta_degC)` -- the Vertical Totals value.

**Example**

```python
from metrust.calc import vertical_totals
from metrust.units import units

vt = vertical_totals(20 * units.degC, -12 * units.degC)
```

---

### `sweat_index`

Severe Weather Threat Index (SWEAT). An older composite that blends low-level
moisture, instability (via the Total Totals), wind speeds at 850 and 500 hPa,
and directional wind shear (veering) between those two levels. Values above 300
indicate a significant severe weather threat; above 400 suggests tornado potential.
Because SWEAT relies on fixed-level wind data and lacks direct shear-profile
information, it has been largely superseded by STP and SCP in U.S. operations
but remains in use internationally.

```python
sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t850` | `Quantity (temperature)` | 850 hPa temperature (degC). |
| `td850` | `Quantity (temperature)` | 850 hPa dewpoint (degC). |
| `t500` | `Quantity (temperature)` | 500 hPa temperature (degC). |
| `dd850` | `Quantity (degrees)` | 850 hPa wind direction. |
| `dd500` | `Quantity (degrees)` | 500 hPa wind direction. |
| `ff850` | `Quantity (speed)` | 850 hPa wind speed (knots). |
| `ff500` | `Quantity (speed)` | 500 hPa wind speed (knots). |

**Returns**

`Quantity (dimensionless)` -- the SWEAT index value.

**Example**

```python
from metrust.calc import sweat_index
from metrust.units import units

sweat = sweat_index(
    20 * units.degC,      # T 850
    16 * units.degC,      # Td 850
    -12 * units.degC,     # T 500
    210 * units.degree,   # wind dir 850
    250 * units.degree,   # wind dir 500
    30 * units.knot,      # wind speed 850
    50 * units.knot,      # wind speed 500
)
# SWEAT > 300: severe threat; > 400: tornado potential
```

---

### `boyden_index`

Boyden Index. A European stability index originally developed for thunderstorm
forecasting in the UK. It combines the 1000--700 hPa thickness with the 700 hPa
temperature. Values above 94--95 have been associated with thunderstorm development
in mid-latitude maritime environments.

```python
boyden_index(z1000, z700, t700)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `z1000` | `Quantity (m)` | 1000 hPa geopotential height. |
| `z700` | `Quantity (m)` | 700 hPa geopotential height. |
| `t700` | `Quantity (temperature)` | 700 hPa temperature. |

**Returns**

`Quantity (dimensionless)` -- the Boyden Index value.

**Example**

```python
from metrust.calc import boyden_index
from metrust.units import units

bi = boyden_index(120 * units.m, 3100 * units.m, 4 * units.degC)
# Values > 94 suggest thunderstorm potential (European climates)
```

---

## CAPE and CIN Variants

### `downdraft_cape`

Downdraft CAPE (DCAPE). Estimates the maximum kinetic energy available to a
descending parcel driven by evaporative cooling. Larger DCAPE values correlate
with stronger thunderstorm outflow and damaging surface winds. Values above
1000 J/kg indicate potential for significant wind damage; above 1500 J/kg
suggests extreme outflow potential.

```python
downdraft_cape(pressure, temperature, dewpoint)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `dewpoint` | `array Quantity (temperature)` | Dewpoint profile. |

**Returns**

`Quantity (J/kg)` -- the DCAPE value.

**Example**

```python
from metrust.calc import downdraft_cape
from metrust.units import units

p  = [1000, 925, 850, 700, 500, 300] * units.hPa
T  = [28, 22, 16, 4, -14, -38] * units.degC
Td = [22, 16, 10, -8, -30, -50] * units.degC

dcape = downdraft_cape(p, T, Td)
# DCAPE > 1000 J/kg: strong outflow potential
```

---

### `convective_inhibition_depth`

Convective inhibition depth. Computes the pressure depth (in hPa) of the layer
through which convective inhibition acts on a lifted parcel. This supplements
the CIN magnitude by indicating how deep the capping inversion extends. A
shallow but strong cap may break more easily than a deep cap with the same
total CIN.

```python
convective_inhibition_depth(pressure, temperature, dewpoint)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `dewpoint` | `array Quantity (temperature)` | Dewpoint profile. |

**Returns**

`Quantity (hPa)` -- the pressure depth of the CIN layer.

**Example**

```python
from metrust.calc import convective_inhibition_depth
from metrust.units import units

p  = [1000, 950, 925, 900, 850, 700, 500] * units.hPa
T  = [30, 26, 24, 22, 18, 4, -14] * units.degC
Td = [22, 20, 18, 14, 10, -6, -28] * units.degC

cin_depth = convective_inhibition_depth(p, T, Td)
```

---

## Sounding Diagnostics

### `dendritic_growth_zone`

Dendritic growth zone bounds. Identifies the pressure layer where temperatures
fall between -12 degC and -18 degC, the range in which dendritic ice crystals
grow most efficiently. This zone is operationally significant for snowfall
forecasting: a deep dendritic growth zone with adequate moisture supports
higher snow-to-liquid ratios and greater snowfall accumulation.

```python
dendritic_growth_zone(temperature, pressure)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |

**Returns**

`tuple of (Quantity (hPa), Quantity (hPa))` -- bottom and top pressure of the dendritic growth zone.

**Example**

```python
from metrust.calc import dendritic_growth_zone
from metrust.units import units

T = [-5, -8, -12, -15, -18, -22] * units.degC
p = [850, 800, 750, 700, 650, 600] * units.hPa

bot, top = dendritic_growth_zone(T, p)
# A deeper zone (larger pressure difference) favors heavier snow
```

---

### `warm_nose_check`

Check for a warm nose -- a layer above the surface where the temperature
exceeds 0 degC in an otherwise sub-freezing profile. Warm noses are critical
for winter precipitation type forecasting: their presence and depth govern
whether falling snow melts completely (producing rain or freezing rain) or
only partially (producing sleet/ice pellets).

```python
warm_nose_check(temperature, pressure)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |

**Returns**

`bool` -- `True` if a warm nose is detected in the profile.

**Example**

```python
from metrust.calc import warm_nose_check
from metrust.units import units

T = [-2, 1, 3, 1, -4, -12] * units.degC
p = [1000, 950, 900, 850, 700, 500] * units.hPa

has_warm_nose = warm_nose_check(T, p)
# True indicates a melting layer aloft -- freezing rain or sleet risk
```

---

### `freezing_rain_composite`

Freezing rain composite index. A multi-factor diagnostic that combines thermal
profile information and precipitation type into a single index for assessing
the likelihood of freezing rain. Higher values indicate environments more
conducive to freezing rain accumulation.

```python
freezing_rain_composite(temperature, pressure, precip_type)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `temperature` | `array Quantity (temperature)` | Temperature profile. |
| `pressure` | `array Quantity (pressure)` | Pressure profile (hPa). |
| `precip_type` | `int` | Precipitation type flag. |

**Returns**

`Quantity (dimensionless)` -- the freezing rain composite index value.

**Example**

```python
from metrust.calc import freezing_rain_composite
from metrust.units import units

T = [-2, 1, 3, 1, -5, -15] * units.degC
p = [1000, 950, 900, 850, 700, 500] * units.hPa

fzra = freezing_rain_composite(T, p, precip_type=1)
```

---

### `galvez_davison_index`

Galvez-Davison Index (GDI). A stability index developed specifically for tropical
environments where traditional mid-latitude indices (LI, CAPE) often perform poorly.
The GDI uses temperatures and dewpoints at 950, 850, 700, and 500 hPa along with
sea surface temperature to assess the potential for tropical thunderstorm development.
Positive values indicate instability; values above 25--30 suggest conditions favorable
for deep tropical convection. Widely used by NWS offices in Puerto Rico and the
tropical Pacific.

```python
galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t950` | `Quantity (temperature)` | 950 hPa temperature. |
| `t850` | `Quantity (temperature)` | 850 hPa temperature. |
| `t700` | `Quantity (temperature)` | 700 hPa temperature. |
| `t500` | `Quantity (temperature)` | 500 hPa temperature. |
| `td950` | `Quantity (temperature)` | 950 hPa dewpoint. |
| `td850` | `Quantity (temperature)` | 850 hPa dewpoint. |
| `td700` | `Quantity (temperature)` | 700 hPa dewpoint. |
| `sst` | `Quantity (temperature)` | Sea surface temperature. |

**Returns**

`Quantity (dimensionless)` -- the GDI value.

**Example**

```python
from metrust.calc import galvez_davison_index
from metrust.units import units

gdi = galvez_davison_index(
    24 * units.degC,   # T 950
    20 * units.degC,   # T 850
    10 * units.degC,   # T 700
    -8 * units.degC,   # T 500
    22 * units.degC,   # Td 950
    18 * units.degC,   # Td 850
    4 * units.degC,    # Td 700
    28 * units.degC,   # SST
)
# GDI > 25-30: favorable for tropical deep convection
```

---

## Fire Weather

### `fosberg_fire_weather_index`

Fosberg Fire Weather Index (FFWI). Combines temperature, relative humidity, and
wind speed into a single index that measures the potential for wildfire spread.
Values are scaled 0--100; higher values indicate greater fire danger. The FFWI
is standard in U.S. fire weather forecasting and is included in NWS fire weather
spot forecasts. Values above 50 are considered high; above 75 is extreme.

```python
fosberg_fire_weather_index(temperature, relative_humidity, wind_speed)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `temperature` | `Quantity (temperature)` | Temperature in Fahrenheit. |
| `relative_humidity` | `Quantity (percent)` or `float` | Relative humidity (0--100 scale). |
| `wind_speed` | `Quantity (speed)` | Wind speed in mph. |

**Returns**

`Quantity (dimensionless)` -- the FFWI value.

**Example**

```python
from metrust.calc import fosberg_fire_weather_index
from metrust.units import units

ffwi = fosberg_fire_weather_index(
    95 * units.degF,
    15 * units.percent,
    25 * units.mph,
)
# FFWI > 50: high fire danger; > 75: extreme
```

---

### `haines_index`

Haines Index (Lower Atmosphere Severity Index). A fire weather index that
combines the stability and moisture content of the lower atmosphere to assess
the potential for large, erratic fire behavior. Uses 950 and 850 hPa
temperatures along with the 850 hPa dewpoint depression. The index ranges
from 2 to 6: values of 2--3 are low potential, 4 is moderate, 5 is high,
and 6 is very high potential for large fire growth.

```python
haines_index(t_950, t_850, td_850)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t_950` | `Quantity (temperature)` | 950 hPa temperature. |
| `t_850` | `Quantity (temperature)` | 850 hPa temperature. |
| `td_850` | `Quantity (temperature)` | 850 hPa dewpoint. |

**Returns**

`int` -- the Haines Index value (2--6).

**Example**

```python
from metrust.calc import haines_index
from metrust.units import units

hi = haines_index(
    22 * units.degC,  # T 950
    18 * units.degC,  # T 850
    6 * units.degC,   # Td 850
)
# 5-6: high potential for large, erratic fire behavior
```

---

### `hot_dry_windy`

Hot-Dry-Windy Index (HDW). A composite fire weather diagnostic that captures
the combined effect of temperature, moisture deficit, and wind on fire behavior.
Unlike the Fosberg index, HDW explicitly includes vapor pressure deficit (VPD),
which better represents the drying power of the atmosphere. When `vpd` is set
to 0, it is computed internally from the temperature and humidity inputs.
Higher values indicate more dangerous fire weather conditions.

```python
hot_dry_windy(temperature, relative_humidity, wind_speed, vpd=0.0)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `temperature` | `Quantity (temperature)` | Temperature. |
| `relative_humidity` | `float` or `Quantity (percent)` | Relative humidity. |
| `wind_speed` | `Quantity (m/s)` | Wind speed. |
| `vpd` | `float` | Vapor pressure deficit in hPa. If 0, computed internally from temperature and humidity. |

**Returns**

`Quantity (dimensionless)` -- the HDW index value.

**Example**

```python
from metrust.calc import hot_dry_windy
from metrust.units import units

hdw = hot_dry_windy(
    35 * units.degC,
    12 * units.percent,
    15 * units("m/s"),
)
```

---

## See Also

- [Thermodynamics](thermodynamics.md) -- CAPE/CIN, parcel profiles, LCL/LFC/EL, and moisture calculations.
- [Wind](wind.md) -- Bulk shear, storm-relative helicity, Bunkers storm motion, and Corfidi vectors.
- [Kinematics](kinematics.md) -- Divergence, vorticity, advection, and frontogenesis.
