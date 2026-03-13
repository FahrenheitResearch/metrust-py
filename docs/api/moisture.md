# Moisture & Humidity

Functions for computing dewpoint, relative humidity, mixing ratio, specific
humidity, vapor pressure, frost point, psychrometric quantities, and moist-air
thermodynamic properties.

All functions accept and return **Pint Quantity** objects.  Internally, units
are stripped to the Rust-native convention (hPa for pressure, Celsius for
temperature, g/kg for mixing ratio, percent 0--100 for RH), the Rust function
is called, and appropriate units are attached to the result.

!!! info "Array dispatch"
    Functions marked **Rust array binding** have a dedicated Rust entry point
    that processes the entire array in compiled code with zero Python-loop
    overhead.  Functions marked **`_vec_call`** use an automatic vectorizer
    that calls the scalar Rust function element-wise -- still fast, but with
    per-element Python dispatch.

---

## Dewpoint

### `dewpoint`

Dewpoint temperature from vapor pressure, using the inverse of the
Ambaum (2020) saturation vapor pressure formula.

**Rust array binding** -- `_calc.dewpoint` / `_calc.dewpoint_array`

```python
dewpoint(vapor_pressure_val)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `vapor_pressure_val` | `Quantity` | hPa (pressure) | Vapor pressure |

**Returns:** `Quantity` in **degC** -- dewpoint temperature.

```python
from metrust.calc import dewpoint
from metrust.units import units

td = dewpoint(12.27 * units.hPa)
print(td)  # ~10.0 degC
```

---

### `dewpoint_from_relative_humidity`

Dewpoint temperature computed from air temperature and relative humidity.

**Rust array binding** -- `_calc.dewpoint_from_relative_humidity` / `_calc.dewpoint_from_rh_array`

```python
dewpoint_from_relative_humidity(temperature, relative_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `relative_humidity` | `Quantity` | dimensionless (0--1) or percent (0--100) | Relative humidity |

**Returns:** `Quantity` in **degC** -- dewpoint temperature.

```python
from metrust.calc import dewpoint_from_relative_humidity
from metrust.units import units

td = dewpoint_from_relative_humidity(25 * units.degC, 0.65)
print(td)  # ~17.8 degC
```

---

### `dewpoint_from_specific_humidity`

Dewpoint temperature from pressure and specific humidity.  Internally converts
specific humidity to vapor pressure, then inverts the SVP equation.

**Rust array binding** -- `_calc.dewpoint_from_specific_humidity` / `_calc.dewpoint_from_specific_humidity_array`

```python
dewpoint_from_specific_humidity(pressure, specific_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure` | `Quantity` | hPa (pressure) | Total air pressure |
| `specific_humidity` | `Quantity` | kg/kg (dimensionless) | Specific humidity |

**Returns:** `Quantity` in **degC** -- dewpoint temperature.

```python
from metrust.calc import dewpoint_from_specific_humidity
from metrust.units import units

td = dewpoint_from_specific_humidity(1013.25 * units.hPa, 0.008 * units("kg/kg"))
print(td)  # ~11.0 degC
```

---

## Relative Humidity

### `relative_humidity_from_dewpoint`

Relative humidity from temperature and dewpoint.  Supports explicit phase
selection for ice-phase saturation below freezing.

**Rust array binding** (liquid phase) -- `_calc.relative_humidity_from_dewpoint` / `_calc.rh_from_dewpoint_array`.
Ice and auto phases fall back to a Python SVP calculation.

```python
relative_humidity_from_dewpoint(temperature, dewpoint, phase="liquid")
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `dewpoint` | `Quantity` | degC (temperature) | Dewpoint temperature |
| `phase` | `str` | -- | `"liquid"` (default), `"ice"`, or `"auto"` |

**Returns:** `Quantity` -- **dimensionless (0--1)** relative humidity.

```python
from metrust.calc import relative_humidity_from_dewpoint
from metrust.units import units

rh = relative_humidity_from_dewpoint(25 * units.degC, 18 * units.degC)
print(rh)  # ~0.65

# Ice-phase RH at sub-freezing temperatures
rh_ice = relative_humidity_from_dewpoint(
    -10 * units.degC, -12 * units.degC, phase="ice"
)
```

---

### `relative_humidity_from_mixing_ratio`

Relative humidity from pressure, temperature, and mixing ratio.

**Rust array binding** -- `_calc.relative_humidity_from_mixing_ratio` / `_calc.relative_humidity_from_mixing_ratio_array`

```python
relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure` | `Quantity` | hPa (pressure) | Total air pressure |
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `mixing_ratio` | `Quantity` | g/kg or kg/kg | Mixing ratio (auto-converted) |

**Returns:** `Quantity` -- **dimensionless (0--1)** relative humidity.

!!! note
    The wrapper auto-detects whether the input is in g/kg or kg/kg and
    converts to the g/kg convention expected by the Rust layer.

```python
from metrust.calc import relative_humidity_from_mixing_ratio
from metrust.units import units

rh = relative_humidity_from_mixing_ratio(
    1013 * units.hPa, 25 * units.degC, 10 * units("g/kg")
)
print(rh)  # ~0.50
```

---

### `relative_humidity_from_specific_humidity`

Relative humidity from pressure, temperature, and specific humidity.

**Rust array binding** -- `_calc.relative_humidity_from_specific_humidity` / `_calc.relative_humidity_from_specific_humidity_array`

```python
relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure` | `Quantity` | hPa (pressure) | Total air pressure |
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `specific_humidity` | `Quantity` | kg/kg (dimensionless) | Specific humidity |

**Returns:** `Quantity` -- **dimensionless (0--1)** relative humidity.

```python
from metrust.calc import relative_humidity_from_specific_humidity
from metrust.units import units

rh = relative_humidity_from_specific_humidity(
    1013 * units.hPa, 25 * units.degC, 0.010 * units("kg/kg")
)
print(rh)  # ~0.50
```

---

### `relative_humidity_wet_psychrometric`

Relative humidity derived from dry-bulb temperature, wet-bulb temperature, and
station pressure using the psychrometric equation.

**`_vec_call`** -- calls `_calc.relative_humidity_wet_psychrometric` per element.

```python
relative_humidity_wet_psychrometric(temperature, wet_bulb, pressure)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Dry-bulb temperature |
| `wet_bulb` | `Quantity` | degC (temperature) | Wet-bulb temperature |
| `pressure` | `Quantity` | hPa (pressure) | Station pressure |

**Returns:** `Quantity` in **percent** -- relative humidity.

```python
from metrust.calc import relative_humidity_wet_psychrometric
from metrust.units import units

rh = relative_humidity_wet_psychrometric(
    30 * units.degC, 22 * units.degC, 1013 * units.hPa
)
print(rh)  # percent
```

---

## Mixing Ratio

### `mixing_ratio`

Mixing ratio, supporting two calling conventions:

- `mixing_ratio(pressure, temperature)` -- saturation mixing ratio from total
  pressure and temperature (Rust path).
- `mixing_ratio(partial_pressure, total_pressure)` -- from the ratio of vapor
  pressure to total pressure using `w = eps * e / (p - e)` (Python path).

The function detects the second argument's unit dimensionality to choose the
code path automatically.

**Rust array binding** (pressure + temperature path) -- `_calc.mixing_ratio` / `_calc.mixing_ratio_array`.
The partial-pressure path is computed in Python.

```python
mixing_ratio(
    partial_press_or_pressure,
    total_press_or_temperature,
    molecular_weight_ratio=0.6219569100577033,
)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `partial_press_or_pressure` | `Quantity` | hPa (pressure) | Total pressure *or* partial (vapor) pressure |
| `total_press_or_temperature` | `Quantity` | degC (temperature) *or* hPa (pressure) | Temperature *or* total pressure |
| `molecular_weight_ratio` | `float` | -- | Mv/Md ratio (default: 0.622, used only in partial-pressure path) |

**Returns:** `Quantity` in **kg/kg** (dimensionless) -- mixing ratio.

```python
from metrust.calc import mixing_ratio
from metrust.units import units

# From pressure and temperature
w = mixing_ratio(1013 * units.hPa, 25 * units.degC)
print(w.to("g/kg"))  # saturation mixing ratio

# From partial pressure and total pressure
w2 = mixing_ratio(12.27 * units.hPa, 1013 * units.hPa)
print(w2.to("g/kg"))
```

---

### `mixing_ratio_from_relative_humidity`

Mixing ratio from pressure, temperature, and relative humidity.

**Rust array binding** -- `_calc.mixing_ratio_from_relative_humidity` / `_calc.mixing_ratio_from_relative_humidity_array`

```python
mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure` | `Quantity` | hPa (pressure) | Total air pressure |
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `relative_humidity` | `Quantity` | percent or dimensionless | Relative humidity |

**Returns:** `Quantity` in **kg/kg** (dimensionless) -- mixing ratio.

```python
from metrust.calc import mixing_ratio_from_relative_humidity
from metrust.units import units

w = mixing_ratio_from_relative_humidity(
    1013 * units.hPa, 25 * units.degC, 50 * units.percent
)
print(w.to("g/kg"))  # ~10 g/kg
```

---

### `mixing_ratio_from_specific_humidity`

Mixing ratio from specific humidity, using the identity `w = q / (1 - q)`.

**Rust array binding** -- `_calc.mixing_ratio_from_specific_humidity` / `_calc.mixing_ratio_from_specific_humidity_array`

```python
mixing_ratio_from_specific_humidity(specific_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `specific_humidity` | `Quantity` | kg/kg (dimensionless) | Specific humidity |

**Returns:** `Quantity` in **kg/kg** (dimensionless) -- mixing ratio.

```python
from metrust.calc import mixing_ratio_from_specific_humidity
from metrust.units import units

w = mixing_ratio_from_specific_humidity(0.010 * units("kg/kg"))
print(w.to("g/kg"))  # ~10.1 g/kg
```

---

## Specific Humidity

### `specific_humidity_from_dewpoint`

Specific humidity from pressure and dewpoint temperature.

**Rust array binding** -- `_calc.specific_humidity_from_dewpoint` / `_calc.specific_humidity_from_dewpoint_array`

```python
specific_humidity_from_dewpoint(pressure, dewpoint)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure` | `Quantity` | hPa (pressure) | Total air pressure |
| `dewpoint` | `Quantity` | degC (temperature) | Dewpoint temperature |

**Returns:** `Quantity` in **kg/kg** (dimensionless) -- specific humidity.

```python
from metrust.calc import specific_humidity_from_dewpoint
from metrust.units import units

q = specific_humidity_from_dewpoint(1013 * units.hPa, 15 * units.degC)
print(q.to("g/kg"))  # ~10.6 g/kg
```

---

### `specific_humidity_from_mixing_ratio`

Specific humidity from mixing ratio, using the identity `q = w / (1 + w)`.

**Rust array binding** -- `_calc.specific_humidity_from_mixing_ratio` / `_calc.specific_humidity_from_mixing_ratio_array`

```python
specific_humidity_from_mixing_ratio(mixing_ratio)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `mixing_ratio` | `Quantity` | kg/kg (dimensionless) | Mixing ratio |

**Returns:** `Quantity` in **kg/kg** (dimensionless) -- specific humidity.

```python
from metrust.calc import specific_humidity_from_mixing_ratio
from metrust.units import units

q = specific_humidity_from_mixing_ratio(0.010 * units("kg/kg"))
print(q.to("g/kg"))  # ~9.9 g/kg
```

---

## Vapor Pressure

### `vapor_pressure`

Vapor pressure, supporting two calling conventions:

- `vapor_pressure(dewpoint)` -- from dewpoint temperature (equivalent to
  saturation vapor pressure at the dewpoint).
- `vapor_pressure(pressure, mixing_ratio=w)` -- from total pressure and mixing
  ratio using `e = p * w / (eps + w)`.

**Rust array binding** (dewpoint path) -- `_calc.vapor_pressure` / `_calc.vapor_pressure_array`.
The pressure + mixing ratio path is computed in Python/NumPy.

```python
vapor_pressure(
    pressure_or_dewpoint,
    mixing_ratio=None,
    molecular_weight_ratio=0.6219569100577033,
)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `pressure_or_dewpoint` | `Quantity` | degC (temperature) *or* Pa (pressure) | Dewpoint *or* total pressure |
| `mixing_ratio` | `Quantity`, optional | kg/kg (dimensionless) | If provided, selects the pressure path |
| `molecular_weight_ratio` | `float` | -- | Mv/Md ratio (default: 0.622) |

**Returns:** `Quantity` in **Pa** -- vapor pressure.

```python
from metrust.calc import vapor_pressure
from metrust.units import units

# From dewpoint
e = vapor_pressure(15 * units.degC)
print(e.to("hPa"))  # ~17.04 hPa

# From pressure and mixing ratio
e2 = vapor_pressure(1013 * units.hPa, mixing_ratio=0.010 * units("kg/kg"))
print(e2.to("hPa"))
```

---

## Frost Point

### `frost_point`

Frost point temperature -- the temperature at which the air becomes saturated
with respect to ice.

**Rust array binding** -- `_calc.frost_point` / `_calc.frost_point_array`

```python
frost_point(temperature, relative_humidity)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |
| `relative_humidity` | `Quantity` | percent or dimensionless | Relative humidity |

**Returns:** `Quantity` in **degC** -- frost point temperature.

```python
from metrust.calc import frost_point
from metrust.units import units

fp = frost_point(-5 * units.degC, 80 * units.percent)
print(fp)  # degC, below the dewpoint
```

---

## Psychrometric Functions

### `psychrometric_vapor_pressure`

Vapor pressure computed from dry-bulb temperature, wet-bulb temperature, and
station pressure via the psychrometric equation.

**`_vec_call`** -- calls `_calc.psychrometric_vapor_pressure` per element.

```python
psychrometric_vapor_pressure(temperature, wet_bulb, pressure)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Dry-bulb temperature |
| `wet_bulb` | `Quantity` | degC (temperature) | Wet-bulb temperature |
| `pressure` | `Quantity` | hPa (pressure) | Station pressure |

**Returns:** `Quantity` in **hPa** -- psychrometric vapor pressure.

```python
from metrust.calc import psychrometric_vapor_pressure
from metrust.units import units

e = psychrometric_vapor_pressure(
    30 * units.degC, 22 * units.degC, 1013 * units.hPa
)
print(e)  # hPa
```

!!! note "Alias"
    `psychrometric_vapor_pressure_wet` is an alias for
    `psychrometric_vapor_pressure` with identical signature and behavior.

---

## Moist-Air Thermodynamic Properties

These functions compute thermodynamic constants that vary with moisture content.
All use the `_vec_call` dispatcher (scalar Rust function called per element).

### `moist_air_gas_constant`

Gas constant for moist air, accounting for the contribution of water vapor.

**`_vec_call`** -- calls `_calc.moist_air_gas_constant` per element.

```python
moist_air_gas_constant(mixing_ratio_kgkg)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `mixing_ratio_kgkg` | `Quantity` or `float` | kg/kg | Mixing ratio |

**Returns:** `Quantity` in **J/(kg*K)** -- gas constant for moist air.

```python
from metrust.calc import moist_air_gas_constant
from metrust.units import units

Rm = moist_air_gas_constant(0.012 * units("kg/kg"))
print(Rm)  # ~289 J/(kg*K)
```

---

### `moist_air_specific_heat_pressure`

Specific heat at constant pressure (cp) for moist air.

**`_vec_call`** -- calls `_calc.moist_air_specific_heat_pressure` per element.

```python
moist_air_specific_heat_pressure(mixing_ratio_kgkg)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `mixing_ratio_kgkg` | `Quantity` or `float` | kg/kg | Mixing ratio |

**Returns:** `Quantity` in **J/(kg*K)** -- specific heat at constant pressure.

```python
from metrust.calc import moist_air_specific_heat_pressure
from metrust.units import units

cp = moist_air_specific_heat_pressure(0.012 * units("kg/kg"))
print(cp)  # ~1012 J/(kg*K)
```

---

### `moist_air_poisson_exponent`

Poisson exponent (kappa) for moist air.  Used in computing potential
temperature with moisture corrections.

**`_vec_call`** -- calls `_calc.moist_air_poisson_exponent` per element.

```python
moist_air_poisson_exponent(mixing_ratio_kgkg)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `mixing_ratio_kgkg` | `Quantity` or `float` | kg/kg | Mixing ratio |

**Returns:** `Quantity` -- **dimensionless** Poisson exponent.

```python
from metrust.calc import moist_air_poisson_exponent
from metrust.units import units

kappa = moist_air_poisson_exponent(0.012 * units("kg/kg"))
print(kappa)  # ~0.286
```

---

## Latent Heat Functions

Temperature-dependent latent heat values following the Ambaum (2020)
formulation.  All use the `_vec_call` dispatcher.

### `water_latent_heat_vaporization`

Latent heat of vaporization (liquid to vapor).

**`_vec_call`** -- calls `_calc.water_latent_heat_vaporization` per element.

```python
water_latent_heat_vaporization(temperature)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |

**Returns:** `Quantity` in **J/kg** -- latent heat of vaporization.

```python
from metrust.calc import water_latent_heat_vaporization
from metrust.units import units

Lv = water_latent_heat_vaporization(20 * units.degC)
print(Lv)  # ~2.45e6 J/kg
```

---

### `water_latent_heat_melting`

Latent heat of melting (ice to liquid).

**`_vec_call`** -- calls `_calc.water_latent_heat_melting` per element.

```python
water_latent_heat_melting(temperature)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |

**Returns:** `Quantity` in **J/kg** -- latent heat of melting.

```python
from metrust.calc import water_latent_heat_melting
from metrust.units import units

Lm = water_latent_heat_melting(0 * units.degC)
print(Lm)  # ~3.34e5 J/kg
```

---

### `water_latent_heat_sublimation`

Latent heat of sublimation (ice directly to vapor).

**`_vec_call`** -- calls `_calc.water_latent_heat_sublimation` per element.

```python
water_latent_heat_sublimation(temperature)
```

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `temperature` | `Quantity` | degC (temperature) | Air temperature |

**Returns:** `Quantity` in **J/kg** -- latent heat of sublimation.

```python
from metrust.calc import water_latent_heat_sublimation
from metrust.units import units

Ls = water_latent_heat_sublimation(-10 * units.degC)
print(Ls)  # ~2.83e6 J/kg
```

---

## Array Dispatch Summary

The table below shows which dispatch mechanism each function uses.

| Function | Dispatch | Rust scalar | Rust array |
|----------|----------|-------------|------------|
| `dewpoint` | Rust array binding | `_calc.dewpoint` | `_calc.dewpoint_array` |
| `dewpoint_from_relative_humidity` | Rust array binding | `_calc.dewpoint_from_relative_humidity` | `_calc.dewpoint_from_rh_array` |
| `dewpoint_from_specific_humidity` | Rust array binding | `_calc.dewpoint_from_specific_humidity` | `_calc.dewpoint_from_specific_humidity_array` |
| `relative_humidity_from_dewpoint` | Rust array binding (liquid) | `_calc.relative_humidity_from_dewpoint` | `_calc.rh_from_dewpoint_array` |
| `relative_humidity_from_mixing_ratio` | Rust array binding | `_calc.relative_humidity_from_mixing_ratio` | `_calc.relative_humidity_from_mixing_ratio_array` |
| `relative_humidity_from_specific_humidity` | Rust array binding | `_calc.relative_humidity_from_specific_humidity` | `_calc.relative_humidity_from_specific_humidity_array` |
| `mixing_ratio` | Rust array binding (T path) | `_calc.mixing_ratio` | `_calc.mixing_ratio_array` |
| `mixing_ratio_from_relative_humidity` | Rust array binding | `_calc.mixing_ratio_from_relative_humidity` | `_calc.mixing_ratio_from_relative_humidity_array` |
| `mixing_ratio_from_specific_humidity` | Rust array binding | `_calc.mixing_ratio_from_specific_humidity` | `_calc.mixing_ratio_from_specific_humidity_array` |
| `specific_humidity_from_dewpoint` | Rust array binding | `_calc.specific_humidity_from_dewpoint` | `_calc.specific_humidity_from_dewpoint_array` |
| `specific_humidity_from_mixing_ratio` | Rust array binding | `_calc.specific_humidity_from_mixing_ratio` | `_calc.specific_humidity_from_mixing_ratio_array` |
| `vapor_pressure` | Rust array binding (Td path) | `_calc.vapor_pressure` | `_calc.vapor_pressure_array` |
| `frost_point` | Rust array binding | `_calc.frost_point` | `_calc.frost_point_array` |
| `relative_humidity_wet_psychrometric` | `_vec_call` | `_calc.relative_humidity_wet_psychrometric` | -- |
| `psychrometric_vapor_pressure` | `_vec_call` | `_calc.psychrometric_vapor_pressure` | -- |
| `moist_air_gas_constant` | `_vec_call` | `_calc.moist_air_gas_constant` | -- |
| `moist_air_specific_heat_pressure` | `_vec_call` | `_calc.moist_air_specific_heat_pressure` | -- |
| `moist_air_poisson_exponent` | `_vec_call` | `_calc.moist_air_poisson_exponent` | -- |
| `water_latent_heat_vaporization` | `_vec_call` | `_calc.water_latent_heat_vaporization` | -- |
| `water_latent_heat_melting` | `_vec_call` | `_calc.water_latent_heat_melting` | -- |
| `water_latent_heat_sublimation` | `_vec_call` | `_calc.water_latent_heat_sublimation` | -- |

Functions with **Rust array binding** process entire NumPy arrays in a single
Rust call with no per-element Python overhead.  Functions using **`_vec_call`**
call the scalar Rust function in a Python loop -- still backed by compiled Rust,
but with per-element dispatch cost that matters on large grids.
