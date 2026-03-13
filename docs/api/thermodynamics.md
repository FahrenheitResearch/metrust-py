# Thermodynamics API Reference

All functions live in `metrust.calc` and accept
[Pint](https://pint.readthedocs.io/) `Quantity` objects. Units are
automatically converted to the Rust-native convention before calling the
compiled backend, and appropriate units are attached to the result.

Plain `float` / `ndarray` values (without Pint units) are passed through
as-is; callers who already work in the Rust-native unit system can skip the
Pint overhead entirely.

**Rust-native unit conventions:**

| Quantity | Unit |
|----------|------|
| Pressure | hPa (millibars) |
| Temperature | degC (potential temperature in K) |
| Mixing ratio | g/kg |
| Height | m |

**Dispatch model:** Functions that note "Rust array binding" call a dedicated
`_array` entry point compiled from Rust, which processes the entire array in a
single FFI call. Functions that note "`_vec_call` fallback" iterate over the
array element-wise in Python, calling the scalar Rust function once per
element. Both paths produce identical results; the array binding is faster for
large inputs.

---

## Potential Temperature

```python
potential_temperature(pressure, temperature)
```

Calculate potential temperature using Poisson's equation.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |

**Returns** -- `Quantity` in **K** (Kelvin).

**Dispatch** -- Rust array binding (`potential_temperature_array`).

```python
from metrust.calc import potential_temperature
from metrust.units import units

theta = potential_temperature(850 * units.hPa, 25 * units.degC)
# ~298.9 K
```

---

## Equivalent Potential Temperature

```python
equivalent_potential_temperature(pressure, temperature, dewpoint)
```

Equivalent potential temperature (theta-e), accounting for latent heat release
during condensation.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `dewpoint` | `Quantity` | degC | Dewpoint temperature |

**Returns** -- `Quantity` in **K**.

**Dispatch** -- Rust array binding (`equivalent_potential_temperature_array`).

```python
from metrust.calc import equivalent_potential_temperature
from metrust.units import units

theta_e = equivalent_potential_temperature(
    850 * units.hPa, 25 * units.degC, 20 * units.degC
)
# ~348 K
```

---

## Saturation Vapor Pressure

```python
saturation_vapor_pressure(temperature, phase="liquid")
```

Saturation vapor pressure using the Ambaum (2020) formulation, matching MetPy
exactly.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | `Quantity` | degC | Air temperature |
| `phase` | `str` | -- | `"liquid"` (default), `"ice"`, or `"auto"` (ice below 0 degC) |

**Returns** -- `Quantity` in **Pa**.

**Dispatch** -- Rust array binding (`saturation_vapor_pressure_array`) for
`phase="liquid"`. The `"ice"` and `"auto"` paths use a Python implementation
of the Ambaum (2020) ice-phase formula.

```python
from metrust.calc import saturation_vapor_pressure
from metrust.units import units

es = saturation_vapor_pressure(20 * units.degC)
# ~2338 Pa (23.38 hPa)

es_ice = saturation_vapor_pressure(-10 * units.degC, phase="ice")
```

---

## Saturation Mixing Ratio

```python
saturation_mixing_ratio(pressure, temperature, phase="liquid")
```

Saturation mixing ratio at a given pressure and temperature.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `phase` | `str` | -- | `"liquid"` (default), `"ice"`, or `"auto"` |

**Returns** -- `Quantity` in **kg/kg** (dimensionless).

**Dispatch** -- Rust array binding (`saturation_mixing_ratio_array`) for
`phase="liquid"`. The Rust engine returns g/kg internally; the Python wrapper
divides by 1000 to yield kg/kg. Ice/auto paths use Python SVP + epsilon
formula.

```python
from metrust.calc import saturation_mixing_ratio
from metrust.units import units

ws = saturation_mixing_ratio(1000 * units.hPa, 25 * units.degC)
# ~0.020 kg/kg
```

---

## Wet-Bulb Temperature

```python
wet_bulb_temperature(pressure, temperature, dewpoint)
```

Wet-bulb temperature via iterative solution along the moist adiabat.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `dewpoint` | `Quantity` | degC | Dewpoint temperature |

**Returns** -- `Quantity` in **degC**.

**Dispatch** -- Rust array binding (`wet_bulb_temperature_array`).

```python
from metrust.calc import wet_bulb_temperature
from metrust.units import units

tw = wet_bulb_temperature(
    1000 * units.hPa, 30 * units.degC, 20 * units.degC
)
# ~23.5 degC
```

---

## Virtual Temperature

```python
virtual_temperature(temperature, pressure_or_mixing_ratio, dewpoint=None)
```

Virtual temperature. Supports two calling conventions:

1. **MetPy-compatible:** `virtual_temperature(T, mixing_ratio)` -- mixing ratio
   is dimensionless (kg/kg). Computes `T_v = T * (1 + w/eps) / (1 + w)` in
   pure Python/NumPy.
2. **Rust-native:** `virtual_temperature(T, pressure, dewpoint)` -- pressure
   in hPa, dewpoint in degC.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | `Quantity` | degC | Air temperature |
| `pressure_or_mixing_ratio` | `Quantity` | hPa or kg/kg | Pressure (if `dewpoint` given) or mixing ratio |
| `dewpoint` | `Quantity`, optional | degC | Dewpoint temperature (triggers Rust-native path) |

**Returns** -- `Quantity` in **degC**.

**Dispatch** -- Rust array binding (`virtual_temp_array`) when `dewpoint` is
provided. Pure NumPy when called with mixing ratio only.

```python
from metrust.calc import virtual_temperature
from metrust.units import units

# Rust-native path (pressure + dewpoint)
tv = virtual_temperature(
    25 * units.degC, 1000 * units.hPa, dewpoint=20 * units.degC
)

# MetPy-compatible path (mixing ratio)
tv = virtual_temperature(25 * units.degC, 0.015 * units("kg/kg"))
```

---

## Virtual Temperature from Dewpoint

```python
virtual_temperature_from_dewpoint(pressure, temperature, dewpoint)
```

Virtual temperature computed from pressure, temperature, and dewpoint.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `dewpoint` | `Quantity` | degC | Dewpoint temperature |

**Returns** -- `Quantity` in **degC**.

**Dispatch** -- Rust array binding (`virtual_temperature_from_dewpoint_array`).

!!! note
    The keyword arguments `molecular_weight_ratio` and `phase` are accepted
    for MetPy API compatibility but are ignored.

```python
from metrust.calc import virtual_temperature_from_dewpoint
from metrust.units import units

tv = virtual_temperature_from_dewpoint(
    1000 * units.hPa, 25 * units.degC, 20 * units.degC
)
```

---

## Virtual Potential Temperature

```python
virtual_potential_temperature(pressure, temperature, mixing_ratio)
```

Virtual potential temperature.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `mixing_ratio` | `Quantity` | g/kg or kg/kg | Water vapor mixing ratio (auto-converted to g/kg for Rust) |

**Returns** -- `Quantity` in **K**.

**Dispatch** -- Rust array binding (`virtual_potential_temperature_array`).

```python
from metrust.calc import virtual_potential_temperature
from metrust.units import units

theta_v = virtual_potential_temperature(
    850 * units.hPa, 25 * units.degC, 15 * units("g/kg")
)
```

---

## Temperature from Potential Temperature

```python
temperature_from_potential_temperature(pressure, theta)
```

Recover temperature from potential temperature by inverting Poisson's equation.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `theta` | `Quantity` | K | Potential temperature |

**Returns** -- `Quantity` in **K**.

**Dispatch** -- Rust array binding
(`temperature_from_potential_temperature_array`).

```python
from metrust.calc import temperature_from_potential_temperature
from metrust.units import units

t = temperature_from_potential_temperature(850 * units.hPa, 300 * units.K)
# ~275 K
```

---

## Saturation Equivalent Potential Temperature

```python
saturation_equivalent_potential_temperature(pressure, temperature)
```

Equivalent potential temperature assuming the air is saturated (dewpoint equals
temperature).

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |

**Returns** -- `Quantity` in **K**.

**Dispatch** -- Rust array binding
(`saturation_equivalent_potential_temperature_array`).

```python
from metrust.calc import saturation_equivalent_potential_temperature
from metrust.units import units

theta_es = saturation_equivalent_potential_temperature(
    850 * units.hPa, 25 * units.degC
)
```

---

## Wet-Bulb Potential Temperature

```python
wet_bulb_potential_temperature(pressure, temperature, dewpoint)
```

Wet-bulb potential temperature -- the temperature a parcel would reach if
lifted to saturation and then brought moist-adiabatically to 1000 hPa.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `dewpoint` | `Quantity` | degC | Dewpoint temperature |

**Returns** -- `Quantity` in **K**.

**Dispatch** -- Rust array binding
(`wet_bulb_potential_temperature_array`).

```python
from metrust.calc import wet_bulb_potential_temperature
from metrust.units import units

theta_w = wet_bulb_potential_temperature(
    850 * units.hPa, 25 * units.degC, 20 * units.degC
)
```

---

## Lifting Condensation Level (LCL)

```python
lcl(pressure, temperature, dewpoint)
```

Find the Lifting Condensation Level -- the pressure and temperature at which a
parcel lifted dry-adiabatically from the surface first reaches saturation.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` (scalar) | hPa | Starting pressure |
| `temperature` | `Quantity` (scalar) | degC | Starting temperature |
| `dewpoint` | `Quantity` (scalar) | degC | Starting dewpoint |

**Returns** -- `tuple` of (`Quantity` **hPa**, `Quantity` **degC**) -- LCL
pressure and temperature.

**Dispatch** -- Direct Rust scalar call (`_calc.lcl`).

```python
from metrust.calc import lcl
from metrust.units import units

p_lcl, t_lcl = lcl(1000 * units.hPa, 30 * units.degC, 20 * units.degC)
# p_lcl ~875 hPa, t_lcl ~20 degC
```

---

## Level of Free Convection (LFC)

```python
lfc(pressure, temperature, dewpoint)
```

Find the Level of Free Convection -- the pressure where a lifted parcel first
becomes warmer than the environment.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile (surface first, descending) |
| `temperature` | array `Quantity` | degC | Environmental temperature profile |
| `dewpoint` | array `Quantity` | degC | Environmental dewpoint profile |

**Returns** -- `Quantity` in **hPa**.

**Dispatch** -- Direct Rust sounding call (`_calc.lfc`).

```python
from metrust.calc import lfc
from metrust.units import units
import numpy as np

p = np.array([1000, 925, 850, 700, 500]) * units.hPa
t = np.array([30, 25, 20, 10, -10]) * units.degC
td = np.array([20, 18, 15, 5, -20]) * units.degC

p_lfc = lfc(p, t, td)
```

---

## Equilibrium Level (EL)

```python
el(pressure, temperature, dewpoint)
```

Find the Equilibrium Level -- the pressure above the LFC where the lifted
parcel temperature crosses back below the environment temperature.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile (surface first) |
| `temperature` | array `Quantity` | degC | Environmental temperature profile |
| `dewpoint` | array `Quantity` | degC | Environmental dewpoint profile |

**Returns** -- `Quantity` in **hPa**.

**Dispatch** -- Direct Rust sounding call (`_calc.el`).

```python
from metrust.calc import el
from metrust.units import units
import numpy as np

p = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
t = np.array([30, 25, 20, 10, -10, -40, -55]) * units.degC
td = np.array([20, 18, 15, 5, -20, -45, -60]) * units.degC

p_el = el(p, t, td)
```

---

## Convective Condensation Level (CCL)

```python
ccl(pressure, temperature, dewpoint)
```

Find the Convective Condensation Level -- the level where the saturation
mixing ratio line through the surface dewpoint intersects the environment
temperature profile.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile (surface first) |
| `temperature` | array `Quantity` | degC | Environmental temperature profile |
| `dewpoint` | array `Quantity` | degC | Environmental dewpoint profile |

**Returns** -- `tuple` of (`Quantity` **hPa**, `Quantity` **degC**), or `None`
if no CCL is found.

**Dispatch** -- Direct Rust sounding call (`_calc.ccl`).

```python
from metrust.calc import ccl
from metrust.units import units
import numpy as np

p = np.array([1000, 925, 850, 700, 500]) * units.hPa
t = np.array([30, 25, 20, 10, -10]) * units.degC
td = np.array([20, 18, 15, 5, -20]) * units.degC

result = ccl(p, t, td)
if result is not None:
    p_ccl, t_ccl = result
```

---

## CAPE and CIN

```python
cape_cin(pressure, temperature, dewpoint, height,
         psfc, t2m, td2m,
         parcel_type="sb", ml_depth=100.0, mu_depth=300.0, top_m=None)
```

Convective Available Potential Energy and Convective Inhibition for a full
sounding. Supports surface-based, mixed-layer, and most-unstable parcel types.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile (surface first) |
| `temperature` | array `Quantity` | degC | Temperature profile |
| `dewpoint` | array `Quantity` | degC | Dewpoint profile |
| `height` | array `Quantity` | m | Height AGL profile |
| `psfc` | `Quantity` | hPa | Surface pressure |
| `t2m` | `Quantity` | degC | 2-m temperature |
| `td2m` | `Quantity` | degC | 2-m dewpoint |
| `parcel_type` | `str` | -- | `"sb"` (surface-based), `"ml"` (mixed-layer), or `"mu"` (most-unstable) |
| `ml_depth` | `float` | hPa | Mixed-layer depth (default 100) |
| `mu_depth` | `float` | hPa | Most-unstable search depth (default 300) |
| `top_m` | `float`, optional | m | Height AGL cap for integration |

**Returns** -- `tuple` of four `Quantity` values:

| Index | Units | Description |
|-------|-------|-------------|
| 0 | J/kg | CAPE |
| 1 | J/kg | CIN |
| 2 | m | LCL height AGL |
| 3 | m | LFC height AGL |

**Dispatch** -- Direct Rust sounding call (`_calc.cape_cin`).

```python
from metrust.calc import cape_cin
from metrust.units import units
import numpy as np

p   = np.array([1000, 925, 850, 700, 500, 300, 200]) * units.hPa
t   = np.array([30, 25, 20, 10, -10, -40, -55]) * units.degC
td  = np.array([20, 18, 15, 5, -20, -45, -60]) * units.degC
hgt = np.array([0, 750, 1500, 3000, 5500, 9000, 12000]) * units.m

cape, cin, h_lcl, h_lfc = cape_cin(
    p, t, td, hgt,
    1000 * units.hPa, 30 * units.degC, 20 * units.degC,
    parcel_type="sb",
)
```

---

## Parcel Profile

```python
parcel_profile(pressure, temperature, dewpoint)
```

Compute the temperature a lifted parcel would have at each pressure level.
Follows a dry adiabat below the LCL and a moist adiabat above.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure levels (surface first) |
| `temperature` | `Quantity` (scalar) | degC | Starting (surface) temperature |
| `dewpoint` | `Quantity` (scalar) | degC | Starting (surface) dewpoint |

**Returns** -- array `Quantity` in **degC**, same length as `pressure`.

**Dispatch** -- Direct Rust sounding call (`_calc.parcel_profile`).

```python
from metrust.calc import parcel_profile
from metrust.units import units
import numpy as np

p = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
t_parcel = parcel_profile(p, 30 * units.degC, 20 * units.degC)
```

---

## Dry Adiabatic Lapse Rate

```python
dry_lapse(pressure, temperature)
```

Compute the temperature of a parcel at each pressure level assuming a dry
adiabatic process (Poisson's equation applied along the profile).

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure levels |
| `temperature` | `Quantity` (scalar) | degC | Starting temperature |

**Returns** -- array `Quantity` in **degC**, same length as `pressure`.

**Dispatch** -- Direct Rust sounding call (`_calc.dry_lapse`).

```python
from metrust.calc import dry_lapse
from metrust.units import units
import numpy as np

p = np.array([1000, 900, 800, 700]) * units.hPa
t_dry = dry_lapse(p, 30 * units.degC)
```

---

## Moist Adiabatic Lapse Rate

```python
moist_lapse(pressure, temperature)
```

Compute the temperature of a parcel at each pressure level following the
pseudo-adiabatic moist lapse rate.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure levels |
| `temperature` | `Quantity` (scalar) | degC | Starting temperature |

**Returns** -- array `Quantity` in **degC**, same length as `pressure`.

**Dispatch** -- Direct Rust sounding call (`_calc.moist_lapse`).

```python
from metrust.calc import moist_lapse
from metrust.units import units
import numpy as np

p = np.array([850, 700, 500, 300]) * units.hPa
t_moist = moist_lapse(p, 20 * units.degC)
```

---

## Density

```python
density(pressure, temperature, mixing_ratio)
```

Air density from the equation of state for moist air.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |
| `temperature` | `Quantity` | degC | Air temperature |
| `mixing_ratio` | `Quantity` | g/kg or kg/kg | Water vapor mixing ratio (auto-converted to g/kg for Rust) |

**Returns** -- `Quantity` in **kg/m^3**.

**Dispatch** -- Rust array binding (`density_array`).

```python
from metrust.calc import density
from metrust.units import units

rho = density(1013.25 * units.hPa, 20 * units.degC, 10 * units("g/kg"))
# ~1.19 kg/m^3
```

---

## Exner Function

```python
exner_function(pressure)
```

The Exner function, a non-dimensional pressure used in potential temperature
calculations: `(p / p0) ^ (Rd / cp)`.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | `Quantity` | hPa | Atmospheric pressure |

**Returns** -- `Quantity` (dimensionless).

**Dispatch** -- Rust array binding (`exner_function_array`).

```python
from metrust.calc import exner_function
from metrust.units import units

pi = exner_function(850 * units.hPa)
# ~0.956
```

---

## Brunt-Vaisala Frequency

```python
brunt_vaisala_frequency(height, potential_temperature)
```

Brunt-Vaisala buoyancy frequency `N` at each level of a sounding.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | array `Quantity` | m | Height profile |
| `potential_temperature` | array `Quantity` | K | Potential temperature profile |

**Returns** -- array `Quantity` in **1/s**.

**Dispatch** -- Direct Rust sounding call (`_calc.brunt_vaisala_frequency`).

```python
from metrust.calc import brunt_vaisala_frequency
from metrust.units import units
import numpy as np

z = np.array([0, 1000, 2000, 3000]) * units.m
theta = np.array([300, 302, 305, 309]) * units.K

N = brunt_vaisala_frequency(z, theta)
```

---

## Brunt-Vaisala Period

```python
brunt_vaisala_period(height, potential_temperature)
```

Brunt-Vaisala oscillation period `2*pi/N` at each level.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | array `Quantity` | m | Height profile |
| `potential_temperature` | array `Quantity` | K | Potential temperature profile |

**Returns** -- array `Quantity` in **s**.

**Dispatch** -- Direct Rust sounding call (`_calc.brunt_vaisala_period`).

```python
from metrust.calc import brunt_vaisala_period
from metrust.units import units
import numpy as np

z = np.array([0, 1000, 2000, 3000]) * units.m
theta = np.array([300, 302, 305, 309]) * units.K

period = brunt_vaisala_period(z, theta)
```

---

## Brunt-Vaisala Frequency Squared

```python
brunt_vaisala_frequency_squared(height, potential_temperature)
```

Brunt-Vaisala frequency squared (`N^2`) at each level, useful for stability
analysis without the square-root cost.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | array `Quantity` | m | Height profile |
| `potential_temperature` | array `Quantity` | K | Potential temperature profile |

**Returns** -- array `Quantity` in **1/s^2**.

**Dispatch** -- Direct Rust sounding call
(`_calc.brunt_vaisala_frequency_squared`).

```python
from metrust.calc import brunt_vaisala_frequency_squared
from metrust.units import units
import numpy as np

z = np.array([0, 1000, 2000, 3000]) * units.m
theta = np.array([300, 302, 305, 309]) * units.K

N2 = brunt_vaisala_frequency_squared(z, theta)
```

---

## Static Stability

```python
static_stability(pressure, temperature)
```

Static stability parameter `-(T / theta) * (d_theta / d_p)` from pressure and
temperature profiles.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile |
| `temperature` | array `Quantity` | K | Temperature profile (in Kelvin) |

**Returns** -- array `Quantity` in **K/Pa**.

**Dispatch** -- Direct Rust sounding call (`_calc.static_stability`).

```python
from metrust.calc import static_stability
from metrust.units import units
import numpy as np

p = np.array([1000, 850, 700, 500]) * units.hPa
t = np.array([288, 278, 265, 245]) * units.K

sigma = static_stability(p, t)
```

---

## Precipitable Water

```python
precipitable_water(pressure, dewpoint)
```

Total precipitable water in the column, integrated from pressure and dewpoint
profiles.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array `Quantity` | hPa | Pressure profile |
| `dewpoint` | array `Quantity` | degC | Dewpoint profile |

**Returns** -- `Quantity` in **mm**.

**Dispatch** -- Direct Rust sounding call (`_calc.precipitable_water`).

```python
from metrust.calc import precipitable_water
from metrust.units import units
import numpy as np

p = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
td = np.array([20, 18, 15, 5, -20, -45]) * units.degC

pw = precipitable_water(p, td)
# result in mm
```

---

## Dry Static Energy

```python
dry_static_energy(height, temperature)
```

Dry static energy: `DSE = cp*T + g*z`.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | `Quantity` | m | Geopotential height |
| `temperature` | `Quantity` | K | Temperature (Kelvin) |

**Returns** -- `Quantity` in **J/kg**.

**Dispatch** -- `_vec_call` fallback (scalar Rust function called per element).

```python
from metrust.calc import dry_static_energy
from metrust.units import units

dse = dry_static_energy(1500 * units.m, 280 * units.K)
```

---

## Moist Static Energy

```python
moist_static_energy(height, temperature, specific_humidity)
```

Moist static energy: `MSE = cp*T + g*z + Lv*q`.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | `Quantity` | m | Geopotential height |
| `temperature` | `Quantity` | K | Temperature (Kelvin) |
| `specific_humidity` | `Quantity` | kg/kg | Specific humidity |

**Returns** -- `Quantity` in **J/kg**.

**Dispatch** -- `_vec_call` fallback (scalar Rust function called per element).

```python
from metrust.calc import moist_static_energy
from metrust.units import units

mse = moist_static_energy(
    1500 * units.m, 280 * units.K, 0.008 * units("kg/kg")
)
```
