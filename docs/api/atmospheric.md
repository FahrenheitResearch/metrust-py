# Atmospheric Functions

Pressure/height conversions, comfort indices, and layer analysis functions.

All functions accept and return [Pint](https://pint.readthedocs.io/) Quantity
objects, matching the MetPy API. Plain floats and NumPy arrays (without units)
are accepted as-is and treated as values in the Rust-native unit system
(hPa for pressure, Celsius for temperature, m/s for wind, meters for height).

```python
from metrust.calc import (
    pressure_to_height_std, height_to_pressure_std,
    altimeter_to_station_pressure, heat_index, windchill,
    thickness_hydrostatic, get_layer, scale_height,
)
from metrust.units import units
```

---

## Pressure / Height Conversions

### `pressure_to_height_std`

Convert pressure to height using the U.S. Standard Atmosphere 1976.

```python
pressure_to_height_std(pressure)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | Quantity or float | hPa | Atmospheric pressure |

**Returns** -- Quantity in **m** (meters above sea level in the standard atmosphere).

```python
>>> from metrust.calc import pressure_to_height_std
>>> from metrust.units import units
>>> pressure_to_height_std(500 * units.hPa)
<Quantity(5574.05, 'meter')>
```

---

### `height_to_pressure_std`

Convert height to pressure using the U.S. Standard Atmosphere 1976.

```python
height_to_pressure_std(height)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | Quantity or float | m | Height above sea level |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import height_to_pressure_std
>>> from metrust.units import units
>>> height_to_pressure_std(5500 * units.m)
<Quantity(505.53, 'hectopascal')>
```

---

### `altimeter_to_station_pressure`

Convert altimeter setting to station pressure.

```python
altimeter_to_station_pressure(altimeter, elevation)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `altimeter` | Quantity or float | hPa | Altimeter setting |
| `elevation` | Quantity or float | m | Station elevation above sea level |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import altimeter_to_station_pressure
>>> from metrust.units import units
>>> altimeter_to_station_pressure(1013.25 * units.hPa, 300 * units.m)
<Quantity(977.48, 'hectopascal')>
```

---

### `station_to_altimeter_pressure`

Convert station pressure to altimeter setting.

```python
station_to_altimeter_pressure(station_pressure, elevation)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `station_pressure` | Quantity or float | hPa | Measured station pressure |
| `elevation` | Quantity or float | m | Station elevation above sea level |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import station_to_altimeter_pressure
>>> from metrust.units import units
>>> station_to_altimeter_pressure(977 * units.hPa, 300 * units.m)
<Quantity(1012.75, 'hectopascal')>
```

---

### `altimeter_to_sea_level_pressure`

Convert altimeter setting to sea-level pressure, accounting for station
elevation and temperature.

```python
altimeter_to_sea_level_pressure(altimeter, elevation, temperature)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `altimeter` | Quantity or float | hPa | Altimeter setting |
| `elevation` | Quantity or float | m | Station elevation above sea level |
| `temperature` | Quantity or float | degC | Station temperature |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import altimeter_to_sea_level_pressure
>>> from metrust.units import units
>>> altimeter_to_sea_level_pressure(1013.25 * units.hPa, 300 * units.m, 20 * units.degC)
<Quantity(1013.52, 'hectopascal')>
```

---

### `sigma_to_pressure`

Convert a sigma (terrain-following) coordinate to pressure.

Formula: `p = ptop + sigma * (psfc - ptop)`

```python
sigma_to_pressure(sigma, psfc, ptop)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `sigma` | float | dimensionless | Sigma coordinate (0 at model top, 1 at surface) |
| `psfc` | Quantity or float | hPa | Surface pressure |
| `ptop` | Quantity or float | hPa | Pressure at model top |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import sigma_to_pressure
>>> from metrust.units import units
>>> sigma_to_pressure(0.5, 1013.25 * units.hPa, 50 * units.hPa)
<Quantity(531.625, 'hectopascal')>
```

---

### `add_height_to_pressure`

Compute the new pressure after ascending or descending by a height increment,
using the hypsometric equation with standard atmosphere assumptions.

Supports array inputs via `_vec_call`.

```python
add_height_to_pressure(pressure, delta_height)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | Quantity or float | hPa | Starting pressure |
| `delta_height` | Quantity or float | m | Height change (positive = ascend) |

**Returns** -- Quantity in **hPa**.

```python
>>> from metrust.calc import add_height_to_pressure
>>> from metrust.units import units
>>> add_height_to_pressure(1000 * units.hPa, 1000 * units.m)
<Quantity(886.39, 'hectopascal')>
```

---

### `add_pressure_to_height`

Compute the new height after a pressure increment, using the hypsometric
equation with standard atmosphere assumptions.

Supports array inputs via `_vec_call`.

```python
add_pressure_to_height(height, delta_pressure)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | Quantity or float | m | Starting height |
| `delta_pressure` | Quantity or float | hPa | Pressure change |

**Returns** -- Quantity in **m**.

```python
>>> from metrust.calc import add_pressure_to_height
>>> from metrust.units import units
>>> add_pressure_to_height(0 * units.m, -100 * units.hPa)
<Quantity(879.47, 'meter')>
```

---

## Comfort Indices

All comfort index functions support array inputs via `_vec_call`. Pass scalar
Quantity values for single-point calculations or NumPy arrays for vectorized
batch computation across grids or station lists.

### `heat_index`

Heat index using the NWS Rothfusz regression.

```python
heat_index(temperature, relative_humidity)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | Quantity or float | degC | Air temperature |
| `relative_humidity` | Quantity or float | percent (0--100) | Relative humidity |

**Returns** -- Quantity in **degC**.

```python
>>> from metrust.calc import heat_index
>>> from metrust.units import units
>>> heat_index(35 * units.degC, 80)
<Quantity(49.54, 'degree_Celsius')>
```

Array example:

```python
>>> import numpy as np
>>> T = np.array([30, 33, 36]) * units.degC
>>> rh = np.array([60, 70, 80])
>>> heat_index(T, rh)
<Quantity([32.73 40.28 55.15], 'degree_Celsius')>
```

---

### `windchill`

Wind chill index using the NWS formula.

```python
windchill(temperature, wind_speed)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | Quantity or float | degC | Air temperature |
| `wind_speed` | Quantity or float | m/s | Wind speed |

**Returns** -- Quantity in **degC**.

```python
>>> from metrust.calc import windchill
>>> from metrust.units import units
>>> windchill(-10 * units.degC, 8 * units.("m/s"))
<Quantity(-19.03, 'degree_Celsius')>
```

Array example:

```python
>>> import numpy as np
>>> T = np.array([-5, -10, -15]) * units.degC
>>> ws = np.array([5, 10, 15]) * units("m/s")
>>> windchill(T, ws)
<Quantity([-11.38 -21.16 -30.41], 'degree_Celsius')>
```

---

### `apparent_temperature`

Apparent temperature that combines heat index and wind chill into a single
comfort metric. Uses the appropriate regime based on the temperature value.

```python
apparent_temperature(temperature, relative_humidity, wind_speed)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | Quantity or float | degC | Air temperature |
| `relative_humidity` | Quantity or float | percent (0--100) | Relative humidity |
| `wind_speed` | Quantity or float | m/s | Wind speed |

**Returns** -- Quantity in **degC**.

```python
>>> from metrust.calc import apparent_temperature
>>> from metrust.units import units
>>> apparent_temperature(35 * units.degC, 80, 2 * units("m/s"))
<Quantity(47.91, 'degree_Celsius')>
```

Array example:

```python
>>> import numpy as np
>>> T = np.array([-10, 20, 38]) * units.degC
>>> rh = np.array([50, 50, 75])
>>> ws = np.array([10, 3, 1]) * units("m/s")
>>> apparent_temperature(T, rh, ws)
<Quantity([-20.85  20.0  50.12], 'degree_Celsius')>
```

---

## Layer Functions

### `thickness_hydrostatic`

Hypsometric thickness between two pressure levels given a mean layer
temperature. Uses the hypsometric equation: `dz = (Rd * T_mean / g) * ln(p_bottom / p_top)`.

Supports array inputs via `_vec_call`.

```python
thickness_hydrostatic(p_bottom, p_top, t_mean)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `p_bottom` | Quantity or float | hPa | Pressure at the bottom of the layer |
| `p_top` | Quantity or float | hPa | Pressure at the top of the layer |
| `t_mean` | Quantity or float | K | Mean layer temperature (Kelvin) |

**Returns** -- Quantity in **m**.

```python
>>> from metrust.calc import thickness_hydrostatic
>>> from metrust.units import units
>>> thickness_hydrostatic(1000 * units.hPa, 500 * units.hPa, 260 * units.K)
<Quantity(5303.57, 'meter')>
```

---

### `thickness_hydrostatic_from_relative_humidity`

Hypsometric thickness computed from full pressure, temperature, and relative
humidity profiles. Virtual temperature is derived from the RH profile to
account for moisture effects on layer thickness.

```python
thickness_hydrostatic_from_relative_humidity(pressure, temperature, relative_humidity)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array Quantity | hPa | Pressure profile (1-D, top-down or bottom-up) |
| `temperature` | array Quantity | degC | Temperature profile |
| `relative_humidity` | array Quantity | dimensionless (0--1) or percent (0--100) | Relative humidity profile |

**Returns** -- Quantity in **m**.

The function auto-detects whether RH is provided as a ratio (0--1) or as
percent (0--100) and normalizes internally.

```python
>>> import numpy as np
>>> from metrust.calc import thickness_hydrostatic_from_relative_humidity
>>> from metrust.units import units
>>> p  = np.array([1000, 925, 850, 700, 500]) * units.hPa
>>> T  = np.array([25, 20, 15, 5, -15]) * units.degC
>>> rh = np.array([0.80, 0.75, 0.70, 0.50, 0.30])
>>> thickness_hydrostatic_from_relative_humidity(p, T, rh)
<Quantity(5598.42, 'meter')>
```

---

### `get_layer`

Extract a layer from a sounding between two pressure levels. Interpolates
values at the boundary pressures if they do not fall exactly on observed
levels.

```python
get_layer(pressure, values, p_bottom, p_top)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array Quantity | hPa | Full sounding pressure array |
| `values` | array Quantity | any | Corresponding values (temperature, dewpoint, etc.) |
| `p_bottom` | Quantity or float | hPa | Bottom pressure of the layer |
| `p_top` | Quantity or float | hPa | Top pressure of the layer |

**Returns** -- Tuple of `(pressure_layer, values_layer)`, both as array
Quantities. Pressure is in **hPa**; values retain their original units.

```python
>>> import numpy as np
>>> from metrust.calc import get_layer
>>> from metrust.units import units
>>> p = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
>>> T = np.array([25, 20, 15, 5, -15, -40]) * units.degC
>>> p_layer, T_layer = get_layer(p, T, 1000 * units.hPa, 700 * units.hPa)
```

---

### `get_layer_heights`

Extract layer heights between two pressure levels. Interpolates heights at
the boundary pressures.

```python
get_layer_heights(pressure, heights, p_bottom, p_top)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array Quantity | hPa | Full sounding pressure array |
| `heights` | array Quantity | m | Corresponding height array |
| `p_bottom` | Quantity or float | hPa | Bottom pressure of the layer |
| `p_top` | Quantity or float | hPa | Top pressure of the layer |

**Returns** -- Tuple of `(pressure_layer, heights_layer)` as array Quantities
in **hPa** and **m** respectively.

```python
>>> import numpy as np
>>> from metrust.calc import get_layer_heights
>>> from metrust.units import units
>>> p = np.array([1000, 925, 850, 700, 500]) * units.hPa
>>> z = np.array([0, 750, 1500, 3000, 5500]) * units.m
>>> p_layer, z_layer = get_layer_heights(p, z, 1000 * units.hPa, 700 * units.hPa)
```

---

### `mixed_layer`

Pressure-weighted mean of a quantity over the lowest N hPa of a sounding
(the mixed layer).

```python
mixed_layer(pressure, values, depth=100.0)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `pressure` | array Quantity | hPa | Sounding pressure profile |
| `values` | array Quantity or array | any | Values to average |
| `depth` | float or Quantity | hPa | Depth of the mixed layer (default: 100 hPa) |

**Returns** -- float (pressure-weighted mean value, in the same unit system as
the input values).

```python
>>> import numpy as np
>>> from metrust.calc import mixed_layer
>>> from metrust.units import units
>>> p = np.array([1000, 975, 950, 925, 900, 850]) * units.hPa
>>> T = np.array([25, 23, 21, 19, 17, 13]) * units.degC
>>> mixed_layer(p, T, depth=100.0)
21.03
```

---

### `scale_height`

Atmospheric scale height: `H = R_d * T / g`, the e-folding height for
pressure in an isothermal atmosphere.

Supports array inputs via `_vec_call`.

```python
scale_height(temperature)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `temperature` | Quantity or float | K | Temperature (Kelvin) |

**Returns** -- Quantity in **m**.

```python
>>> from metrust.calc import scale_height
>>> from metrust.units import units
>>> scale_height(270 * units.K)
<Quantity(7902.14, 'meter')>
```

---

### `montgomery_streamfunction`

Montgomery streamfunction for isentropic analysis. Defined as
`M = c_p * T + g * z`, used to diagnose flow on isentropic surfaces.

Supports array inputs via `_vec_call`.

```python
montgomery_streamfunction(theta, pressure, temperature, height)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `theta` | Quantity or float | K | Potential temperature |
| `pressure` | Quantity or float | hPa | Pressure |
| `temperature` | Quantity or float | K | Temperature (Kelvin) |
| `height` | Quantity or float | m | Geopotential height |

**Returns** -- Quantity in **J/kg**.

```python
>>> from metrust.calc import montgomery_streamfunction
>>> from metrust.units import units
>>> montgomery_streamfunction(
...     300 * units.K, 700 * units.hPa, 268 * units.K, 3100 * units.m
... )
<Quantity(299629.2, 'joule / kilogram')>
```

---

### `geopotential_to_height`

Convert geopotential to geopotential height using the relationship
`z = phi / g_0` (with gravity adjusted for the mean Earth radius).

Supports array inputs via `_vec_call`.

```python
geopotential_to_height(geopotential)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `geopotential` | Quantity or float | m^2/s^2 | Geopotential value |

**Returns** -- Quantity in **m**.

```python
>>> from metrust.calc import geopotential_to_height
>>> from metrust.units import units
>>> geopotential_to_height(50000 * units("m**2/s**2"))
<Quantity(5098.55, 'meter')>
```

---

### `height_to_geopotential`

Convert geopotential height to geopotential.

Supports array inputs via `_vec_call`.

```python
height_to_geopotential(height)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | Quantity or float | m | Geopotential height |

**Returns** -- Quantity in **m^2/s^2**.

```python
>>> from metrust.calc import height_to_geopotential
>>> from metrust.units import units
>>> height_to_geopotential(5000 * units.m)
<Quantity(49038.55, 'm ** 2 / s ** 2')>
```

---

### `vertical_velocity`

Convert pressure vertical velocity (omega, dp/dt) to geometric vertical
velocity (w, dz/dt).

Supports array inputs via `_vec_call`.

```python
vertical_velocity(omega, pressure, temperature)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `omega` | Quantity or float | Pa/s | Pressure vertical velocity |
| `pressure` | Quantity or float | hPa | Pressure at the level |
| `temperature` | Quantity or float | degC | Temperature at the level |

**Returns** -- Quantity in **m/s** (positive = upward).

```python
>>> from metrust.calc import vertical_velocity
>>> from metrust.units import units
>>> vertical_velocity(-2 * units("Pa/s"), 500 * units.hPa, -20 * units.degC)
<Quantity(1.47, 'm / s')>
```

---

### `vertical_velocity_pressure`

Convert geometric vertical velocity (w, dz/dt) to pressure vertical velocity
(omega, dp/dt). The inverse of `vertical_velocity`.

Supports array inputs via `_vec_call`.

```python
vertical_velocity_pressure(w, pressure, temperature)
```

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `w` | Quantity or float | m/s | Geometric vertical velocity (positive = upward) |
| `pressure` | Quantity or float | hPa | Pressure at the level |
| `temperature` | Quantity or float | degC | Temperature at the level |

**Returns** -- Quantity in **Pa/s**.

```python
>>> from metrust.calc import vertical_velocity_pressure
>>> from metrust.units import units
>>> vertical_velocity_pressure(1.0 * units("m/s"), 500 * units.hPa, -20 * units.degC)
<Quantity(-1.36, 'pascal / second')>
```

---

## Notes

### Array support

All pressure/height conversion functions and comfort indices support
element-wise array dispatch through the internal `_vec_call` helper. When you
pass array Quantities (or NumPy arrays), the Rust scalar function is called for
each element and the result is returned with the original array shape preserved.

```python
import numpy as np
from metrust.calc import pressure_to_height_std, heat_index
from metrust.units import units

# Vectorized pressure-to-height
p = np.array([1000, 850, 700, 500, 300]) * units.hPa
z = pressure_to_height_std(p)  # shape: (5,)

# Vectorized heat index over a grid
T_grid  = np.random.uniform(28, 40, (100, 100)) * units.degC
rh_grid = np.random.uniform(40, 90, (100, 100))
hi_grid = heat_index(T_grid, rh_grid)  # shape: (100, 100)
```

Layer functions (`get_layer`, `get_layer_heights`, `mixed_layer`,
`thickness_hydrostatic_from_relative_humidity`) operate on 1-D sounding
profiles and return appropriately sized results.

### Unit handling

All functions accept any compatible Pint unit and convert internally:

```python
# Feet work -- converted to meters before calling Rust
z = pressure_to_height_std(500 * units.hPa).to("ft")

# Fahrenheit works -- converted to Celsius before calling Rust
hi = heat_index(95 * units.degF, 80)
```

The Rust engine uses a fixed internal convention (hPa, Celsius, m/s, meters,
Kelvin for potential temperature). The Python wrapper handles all conversions
transparently.
