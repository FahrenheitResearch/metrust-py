# Wind

Functions for wind speed, direction, shear, helicity, storm motion, and
boundary-layer turbulence diagnostics. All functions live in `metrust.calc`.

```python
from metrust.calc import wind_speed, wind_direction, wind_components
from metrust.units import units
```

---

## wind_speed

```python
wind_speed(u, v)
```

Compute scalar wind speed from u and v components.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component |
| `v` | array Quantity | m/s | North-south wind component |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| array Quantity | m/s | Scalar wind speed |

**Example**

```python
from metrust.calc import wind_speed
from metrust.units import units

u = [5.0, -3.0] * units("m/s")
v = [8.0, 4.0] * units("m/s")
speed = wind_speed(u, v)
print(speed)  # [9.43, 5.0] m/s
```

---

## wind_direction

```python
wind_direction(u, v)
```

Meteorological wind direction from u and v components. Returns the direction
the wind is blowing *from*, measured clockwise from north (0/360 = north,
90 = east, 180 = south, 270 = west).

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component |
| `v` | array Quantity | m/s | North-south wind component |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| array Quantity | degree | Meteorological wind direction |

**Example**

```python
from metrust.calc import wind_direction
from metrust.units import units

u = [0.0, -10.0] * units("m/s")
v = [-10.0, 0.0] * units("m/s")
wdir = wind_direction(u, v)
print(wdir)  # [180.0, 270.0] degree
```

---

## wind_components

```python
wind_components(speed, direction)
```

Convert wind speed and meteorological direction to u and v components.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `speed` | array Quantity | m/s | Wind speed |
| `direction` | array Quantity | degree | Meteorological wind direction (from) |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of (array Quantity, array Quantity) | (m/s, m/s) | `(u, v)` wind components |

**Example**

```python
from metrust.calc import wind_components
from metrust.units import units

u, v = wind_components(10 * units("m/s"), 270 * units.degree)
print(u)  # 10.0 m/s  (westerly wind -> positive u)
print(v)  # ~0.0 m/s
```

---

## bulk_shear

```python
bulk_shear(pressure_or_u, u_or_v, v_or_height, height=None,
           bottom=None, depth=None, top=None)
```

Bulk wind shear (vector difference) over a height layer. The layer is
defined by `bottom` and `top` heights; `depth` may be used as an
alternative to `top` when combined with `bottom`.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |
| `height` | array Quantity | m | Height profile (AGL or MSL, consistent usage) |
| `bottom` | Quantity | m | Bottom of the shear layer |
| `top` | Quantity | m | Top of the shear layer |
| `depth` | Quantity | m | Layer depth (alternative to `top`) |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of (Quantity, Quantity) | (m/s, m/s) | `(shear_u, shear_v)` shear components |

**Example**

```python
import numpy as np
from metrust.calc import bulk_shear
from metrust.units import units

height = np.array([0, 1000, 2000, 3000, 6000]) * units.m
u = np.array([2, 8, 14, 18, 30]) * units("m/s")
v = np.array([5, 6, 4, 2, -2]) * units("m/s")

su, sv = bulk_shear(u, v, height, bottom=0 * units.m, top=6000 * units.m)
print(su, sv)  # shear components over 0-6 km
```

---

## mean_wind

```python
mean_wind(u, v, height, bottom, top)
```

Pressure-weighted mean wind over a height layer.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |
| `height` | array Quantity | m | Height profile |
| `bottom` | Quantity | m | Bottom of the layer |
| `top` | Quantity | m | Top of the layer |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of (Quantity, Quantity) | (m/s, m/s) | `(mean_u, mean_v)` components of the mean wind |

**Example**

```python
import numpy as np
from metrust.calc import mean_wind
from metrust.units import units

height = np.array([0, 500, 1000, 2000, 3000, 6000]) * units.m
u = np.array([2, 5, 10, 15, 20, 30]) * units("m/s")
v = np.array([5, 5, 4, 3, 1, -3]) * units("m/s")

mu, mv = mean_wind(u, v, height, 0 * units.m, 6000 * units.m)
print(mu, mv)  # 0-6 km mean wind components
```

---

## storm_relative_helicity

```python
storm_relative_helicity(u, v, height, depth, storm_u, storm_v)
```

Storm-relative helicity (SRH). All six arguments are positional.

Returns positive, negative, and total SRH. Positive SRH indicates
cyclonic (counterclockwise) rotation potential; negative SRH indicates
anticyclonic rotation.

!!! note "Sign convention"
    The sign convention matches MetPy: positive SRH corresponds to
    cyclonic (right-moving supercell) rotation in the Northern Hemisphere.
    Total SRH is the algebraic sum of the positive and negative
    contributions.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |
| `height` | array Quantity | m | Height profile (AGL) |
| `depth` | Quantity | m | Depth of the SRH integration layer |
| `storm_u` | Quantity | m/s | Storm motion u component |
| `storm_v` | Quantity | m/s | Storm motion v component |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of (Quantity, Quantity, Quantity) | (m^2/s^2, m^2/s^2, m^2/s^2) | `(positive, negative, total)` SRH |

**Example**

```python
import numpy as np
from metrust.calc import storm_relative_helicity, bunkers_storm_motion
from metrust.units import units

height = np.array([0, 500, 1000, 1500, 2000, 3000]) * units.m
u = np.array([0, 5, 10, 15, 20, 25]) * units("m/s")
v = np.array([0, 5, 8, 6, 3, 0]) * units("m/s")

# Use Bunkers right-mover as storm motion
(rm_u, rm_v), _, _ = bunkers_storm_motion(u, v, height)

pos, neg, total = storm_relative_helicity(
    u, v, height, 3000 * units.m, rm_u, rm_v
)
print(f"0-3 km SRH: {total}")
```

---

## bunkers_storm_motion

```python
bunkers_storm_motion(pressure_or_u, u_or_v, v_or_height, height=None)
```

Bunkers internal dynamics method for estimating supercell storm motion.
Returns right-mover, left-mover, and mean wind vectors.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |
| `height` | array Quantity | m | Height profile |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of 3 tuples | (m/s, m/s) each | `(right_u, right_v), (left_u, left_v), (mean_u, mean_v)` |

The first tuple is the right-moving supercell motion, the second is the
left-moving supercell motion, and the third is the 0--6 km mean wind.

**Example**

```python
import numpy as np
from metrust.calc import bunkers_storm_motion
from metrust.units import units

height = np.linspace(0, 10000, 50) * units.m
u = np.linspace(0, 30, 50) * units("m/s")
v = np.linspace(10, -5, 50) * units("m/s")

right, left, mean = bunkers_storm_motion(u, v, height)
print(f"Right-mover: u={right[0]:.1f}, v={right[1]:.1f}")
print(f"Left-mover:  u={left[0]:.1f}, v={left[1]:.1f}")
print(f"Mean wind:   u={mean[0]:.1f}, v={mean[1]:.1f}")
```

---

## corfidi_storm_motion

```python
corfidi_storm_motion(u, v, height, u_850, v_850)
```

Corfidi upwind and downwind propagation vectors for estimating the motion
of mesoscale convective systems (MCSs). Requires the 850 hPa wind
components as separate arguments.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |
| `height` | array Quantity | m | Height profile |
| `u_850` | Quantity | m/s | 850 hPa u wind component (or low-level jet u) |
| `v_850` | Quantity | m/s | 850 hPa v wind component (or low-level jet v) |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| tuple of 2 tuples | (m/s, m/s) each | `(upwind_u, upwind_v), (downwind_u, downwind_v)` |

**Example**

```python
import numpy as np
from metrust.calc import corfidi_storm_motion
from metrust.units import units

height = np.linspace(0, 10000, 50) * units.m
u = np.linspace(5, 25, 50) * units("m/s")
v = np.linspace(15, -5, 50) * units("m/s")

upwind, downwind = corfidi_storm_motion(
    u, v, height, 8 * units("m/s"), 12 * units("m/s")
)
print(f"Corfidi upwind:   u={upwind[0]:.1f}, v={upwind[1]:.1f}")
print(f"Corfidi downwind: u={downwind[0]:.1f}, v={downwind[1]:.1f}")
```

---

## friction_velocity

```python
friction_velocity(u, w)
```

Friction velocity (u\*) computed from time series of the along-wind (u)
and vertical (w) wind components. Uses the eddy covariance definition:

    u* = (-<u'w'>)^(1/2)

where `u'` and `w'` are the fluctuations from the respective means.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | Along-wind component time series |
| `w` | array Quantity | m/s | Vertical wind component time series |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| Quantity | m/s | Friction velocity |

**Example**

```python
import numpy as np
from metrust.calc import friction_velocity
from metrust.units import units

rng = np.random.default_rng(42)
u = (10 + rng.normal(0, 1.5, 1000)) * units("m/s")
w = (0 + rng.normal(0, 0.5, 1000)) * units("m/s")

u_star = friction_velocity(u, w)
print(f"u* = {u_star:.3f}")
```

---

## tke

```python
tke(u, v, w)
```

Turbulent kinetic energy from time series of the three wind components.
Computed as:

    TKE = 0.5 * (<u'^2> + <v'^2> + <w'^2>)

where primes denote departures from the time-series mean.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `u` | array Quantity | m/s | East-west wind component time series |
| `v` | array Quantity | m/s | North-south wind component time series |
| `w` | array Quantity | m/s | Vertical wind component time series |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| Quantity | m^2/s^2 | Turbulent kinetic energy |

**Example**

```python
import numpy as np
from metrust.calc import tke
from metrust.units import units

rng = np.random.default_rng(0)
u = (8 + rng.normal(0, 2.0, 5000)) * units("m/s")
v = (3 + rng.normal(0, 1.5, 5000)) * units("m/s")
w = (0 + rng.normal(0, 0.8, 5000)) * units("m/s")

energy = tke(u, v, w)
print(f"TKE = {energy:.2f}")
```

---

## gradient_richardson_number

```python
gradient_richardson_number(height, potential_temperature, u, v)
```

Gradient Richardson number at each level. Quantifies the ratio of
buoyancy suppression of turbulence to mechanical generation by wind shear:

    Ri = (g / theta) * (d_theta/dz) / ((du/dz)^2 + (dv/dz)^2)

Values below 0.25 indicate the onset of dynamic (shear-driven) instability.
Negative values indicate statically unstable conditions.

**Parameters**

| Name | Type | Units | Description |
|------|------|-------|-------------|
| `height` | array Quantity | m | Height profile |
| `potential_temperature` | array Quantity | K | Potential temperature profile |
| `u` | array Quantity | m/s | East-west wind component profile |
| `v` | array Quantity | m/s | North-south wind component profile |

**Returns**

| Type | Units | Description |
|------|-------|-------------|
| array Quantity | dimensionless | Gradient Richardson number at each level |

**Example**

```python
import numpy as np
from metrust.calc import gradient_richardson_number
from metrust.units import units

height = np.array([0, 100, 200, 500, 1000, 2000]) * units.m
theta = np.array([300, 300.5, 301, 303, 306, 312]) * units.K
u = np.array([2, 5, 8, 12, 15, 20]) * units("m/s")
v = np.array([1, 2, 3, 3, 2, 1]) * units("m/s")

ri = gradient_richardson_number(height, theta, u, v)
print(ri)  # Ri < 0.25 indicates shear instability
```
