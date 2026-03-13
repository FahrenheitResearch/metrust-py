# Kinematics

Grid-scale kinematic analysis functions. All functions in this module use Rust
finite-difference kernels internally for performance. They accept Pint Quantity
arrays and return Pint Quantity results, matching the MetPy API.

Grid spacing parameters `dx` and `dy` are scalar values in meters representing
uniform spacing in the x (east-west) and y (north-south) directions. All 2-D
input arrays must share the same shape.

```python
from metrust.calc import divergence, vorticity, advection
from metrust.units import units
```

---

## Divergence and Vorticity

### `divergence`

Horizontal divergence of a 2-D wind field.

```python
divergence(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import divergence
from metrust.units import units

u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
dx = 25000 * units.m
dy = 25000 * units.m

div = divergence(u, v, dx, dy)  # shape: (50, 50), units: 1/s
```

---

### `vorticity`

Relative (vertical) vorticity of a 2-D wind field.

```python
vorticity(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import vorticity
from metrust.units import units

u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
dx = 25000 * units.m
dy = 25000 * units.m

vort = vorticity(u, v, dx, dy)  # shape: (50, 50), units: 1/s
```

---

### `absolute_vorticity`

Absolute vorticity: relative vorticity plus the Coriolis parameter at each
grid point.

```python
absolute_vorticity(u, v, lats, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import absolute_vorticity
from metrust.units import units

u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
lats = np.linspace(30, 45, 50).reshape(1, -1) * np.ones((50, 1))
dx = 25000 * units.m
dy = 25000 * units.m

abs_vort = absolute_vorticity(u, v, lats, dx, dy)  # 1/s
```

---

### `curvature_vorticity`

Curvature component of vorticity on a 2-D grid. Separates the part of
vorticity due to flow curvature from the shear contribution.

```python
curvature_vorticity(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import curvature_vorticity
from metrust.units import units

u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")

curv_vort = curvature_vorticity(u, v, 25000 * units.m, 25000 * units.m)
```

---

### `shear_vorticity`

Shear component of vorticity on a 2-D grid. Separates the part of vorticity
due to speed shear across the flow from the curvature contribution.

```python
shear_vorticity(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import shear_vorticity
from metrust.units import units

u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")

shear_v = shear_vorticity(u, v, 25000 * units.m, 25000 * units.m)
```

---

### `coriolis_parameter`

Coriolis parameter (f) for a given latitude. Computed as `2 * omega * sin(lat)`.

```python
coriolis_parameter(latitude)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `latitude` | `Quantity` (degrees) or `float` | Latitude in degrees. Plain floats are interpreted as degrees. |

**Returns** -- `Quantity` in 1/s.

```python
from metrust.calc import coriolis_parameter
from metrust.units import units

f = coriolis_parameter(45.0)                  # float, degrees
f = coriolis_parameter(45.0 * units.degree)   # Pint Quantity
# f ~ 1.03e-4  1/s
```

---

## Advection

### `advection`

Advection of a scalar field by a 2-D horizontal wind. Computes
`-u * (ds/dx) - v * (ds/dy)`.

```python
advection(scalar, u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `scalar` | 2-D array `Quantity` | Scalar field to be advected |
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in scalar_units/s. The output units are the
units of the scalar field divided by seconds.

```python
import numpy as np
from metrust.calc import advection
from metrust.units import units

temperature = np.random.uniform(270, 300, (50, 50)) * units.K
u = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
v = np.random.uniform(-10, 10, (50, 50)) * units("m/s")
dx = 25000 * units.m
dy = 25000 * units.m

temp_adv = advection(temperature, u, v, dx, dy)  # units: K/s
```

---

### `advection_3d`

Advection of a scalar field by a 3-D wind. Extends 2-D advection with the
vertical term: `-u * (ds/dx) - v * (ds/dy) - w * (ds/dz)`.

```python
advection_3d(scalar, u, v, w, dx, dy, dz)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `scalar` | 3-D array `Quantity` | Scalar field, shape `(nz, ny, nx)` |
| `u` | 3-D array `Quantity` | U-component of wind (m/s), same shape |
| `v` | 3-D array `Quantity` | V-component of wind (m/s), same shape |
| `w` | 3-D array `Quantity` | Vertical velocity (m/s), same shape |
| `dx` | scalar `Quantity` | Horizontal grid spacing in x (m) |
| `dy` | scalar `Quantity` | Horizontal grid spacing in y (m) |
| `dz` | scalar `Quantity` | Vertical grid spacing (m) |

**Returns** -- 3-D array `Quantity` in scalar_units/s, shape `(nz, ny, nx)`.

The input `scalar` must be a 3-D array. The function infers `nz`, `ny`, `nx`
from its shape.

```python
import numpy as np
from metrust.calc import advection_3d
from metrust.units import units

nz, ny, nx = 10, 50, 50
theta = np.random.uniform(290, 310, (nz, ny, nx)) * units.K
u = np.random.uniform(-10, 10, (nz, ny, nx)) * units("m/s")
v = np.random.uniform(-10, 10, (nz, ny, nx)) * units("m/s")
w = np.random.uniform(-1, 1, (nz, ny, nx)) * units("m/s")

adv = advection_3d(theta, u, v, w, 25000 * units.m, 25000 * units.m, 500 * units.m)
# shape: (10, 50, 50), units: K/s
```

---

## Frontogenesis

### `frontogenesis`

2-D Petterssen frontogenesis function. Quantifies the rate of change of the
horizontal potential temperature gradient magnitude due to the deformation
and convergence of the wind field.

```python
frontogenesis(theta, u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `theta` | 2-D array `Quantity` | Potential temperature field (K) |
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in K/(m*s). Positive values indicate
frontogenesis (tightening gradients); negative values indicate frontolysis.

```python
import numpy as np
from metrust.calc import frontogenesis
from metrust.units import units

theta = np.random.uniform(290, 310, (50, 50)) * units.K
u = np.random.uniform(-15, 15, (50, 50)) * units("m/s")
v = np.random.uniform(-15, 15, (50, 50)) * units("m/s")
dx = 25000 * units.m
dy = 25000 * units.m

fronto = frontogenesis(theta, u, v, dx, dy)  # K/(m*s)
```

---

## Geostrophic and Ageostrophic Wind

### `geostrophic_wind`

Geostrophic wind components from a geopotential height field. Uses the
geostrophic wind equations with the Coriolis parameter derived from the
latitude grid.

```python
geostrophic_wind(heights, lats, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `heights` | 2-D array `Quantity` | Geopotential height field (m) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- Tuple of `(u_geo, v_geo)`, each a 2-D array `Quantity` in m/s.

```python
import numpy as np
from metrust.calc import geostrophic_wind
from metrust.units import units

heights = np.random.uniform(5400, 5800, (50, 50)) * units.m
lats = np.linspace(30, 50, 50).reshape(1, -1) * np.ones((50, 1))
dx = 25000 * units.m
dy = 25000 * units.m

u_geo, v_geo = geostrophic_wind(heights, lats, dx, dy)  # m/s each
```

---

### `ageostrophic_wind`

Ageostrophic wind: the difference between the observed wind and the
geostrophic wind. Computed as `(u - u_geo, v - v_geo)`.

```python
ageostrophic_wind(u, v, heights, lats, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | Observed U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | Observed V-component of wind (m/s) |
| `heights` | 2-D array `Quantity` | Geopotential height field (m) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- Tuple of `(u_ageo, v_ageo)`, each a 2-D array `Quantity` in m/s.

```python
import numpy as np
from metrust.calc import ageostrophic_wind
from metrust.units import units

u = np.random.uniform(-20, 20, (50, 50)) * units("m/s")
v = np.random.uniform(-20, 20, (50, 50)) * units("m/s")
heights = np.random.uniform(5400, 5800, (50, 50)) * units.m
lats = np.linspace(30, 50, 50).reshape(1, -1) * np.ones((50, 1))
dx = 25000 * units.m
dy = 25000 * units.m

u_ageo, v_ageo = ageostrophic_wind(u, v, heights, lats, dx, dy)
```

---

### `inertial_advective_wind`

Inertial-advective wind. Represents the wind that would result from the
advection of the geostrophic wind by the total wind.

```python
inertial_advective_wind(u, v, u_geo, v_geo, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | Total U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | Total V-component of wind (m/s) |
| `u_geo` | 2-D array `Quantity` | Geostrophic U-component (m/s) |
| `v_geo` | 2-D array `Quantity` | Geostrophic V-component (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- Tuple of `(u_ia, v_ia)`, each a 2-D array `Quantity` in m/s.

```python
import numpy as np
from metrust.calc import geostrophic_wind, inertial_advective_wind
from metrust.units import units

heights = np.random.uniform(5400, 5800, (50, 50)) * units.m
lats = np.linspace(30, 50, 50).reshape(1, -1) * np.ones((50, 1))
dx = 25000 * units.m
dy = 25000 * units.m

u_geo, v_geo = geostrophic_wind(heights, lats, dx, dy)
u = np.random.uniform(-20, 20, (50, 50)) * units("m/s")
v = np.random.uniform(-20, 20, (50, 50)) * units("m/s")

u_ia, v_ia = inertial_advective_wind(u, v, u_geo, v_geo, dx, dy)
```

---

## Potential Vorticity

### `potential_vorticity_baroclinic`

Baroclinic (Ertel) potential vorticity on a 2-D isobaric slice. Requires
potential temperature on the target level and on the levels immediately above
and below, along with the bounding pressure values.

```python
potential_vorticity_baroclinic(potential_temp, pressure, theta_below,
                               theta_above, u, v, lats, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `potential_temp` | 2-D array `Quantity` | Potential temperature on the target level (K) |
| `pressure` | length-2 sequence `Quantity` | Pressure pair `[p_below, p_above]` (Pa) |
| `theta_below` | 2-D array `Quantity` | Potential temperature on the level below (K) |
| `theta_above` | 2-D array `Quantity` | Potential temperature on the level above (K) |
| `u` | 2-D array `Quantity` | U-component of wind on the target level (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind on the target level (m/s) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in K * m^2 / (kg * s).

```python
import numpy as np
from metrust.calc import potential_vorticity_baroclinic
from metrust.units import units

shape = (50, 50)
theta_500 = np.full(shape, 320.0) * units.K
theta_700 = np.full(shape, 310.0) * units.K
theta_300 = np.full(shape, 330.0) * units.K
pressure = [70000, 30000] * units.Pa   # 700 hPa below, 300 hPa above
u = np.random.uniform(-20, 20, shape) * units("m/s")
v = np.random.uniform(-20, 20, shape) * units("m/s")
lats = np.linspace(30, 50, 50).reshape(1, -1) * np.ones((50, 1))

pv = potential_vorticity_baroclinic(
    theta_500, pressure, theta_700, theta_300,
    u, v, lats, 25000 * units.m, 25000 * units.m,
)
```

---

### `potential_vorticity_barotropic`

Barotropic potential vorticity, computed from geopotential height and the
horizontal wind field.

```python
potential_vorticity_barotropic(heights, u, v, lats, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `heights` | 2-D array `Quantity` | Geopotential height field (m) |
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/(m*s).

```python
import numpy as np
from metrust.calc import potential_vorticity_barotropic
from metrust.units import units

shape = (50, 50)
heights = np.random.uniform(5400, 5800, shape) * units.m
u = np.random.uniform(-20, 20, shape) * units("m/s")
v = np.random.uniform(-20, 20, shape) * units("m/s")
lats = np.linspace(30, 50, 50).reshape(1, -1) * np.ones((50, 1))

pv_bt = potential_vorticity_barotropic(
    heights, u, v, lats, 25000 * units.m, 25000 * units.m,
)
```

---

## Deformation

### `shearing_deformation`

Shearing deformation of a 2-D wind field. Measures the rate at which fluid
elements are being deformed by differential motion along the axis of
dilatation.

```python
shearing_deformation(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import shearing_deformation
from metrust.units import units

u = np.random.uniform(-15, 15, (50, 50)) * units("m/s")
v = np.random.uniform(-15, 15, (50, 50)) * units("m/s")

shear_def = shearing_deformation(u, v, 25000 * units.m, 25000 * units.m)
```

---

### `stretching_deformation`

Stretching deformation of a 2-D wind field. Measures the rate at which fluid
elements are elongated along one axis and compressed along the perpendicular
axis.

```python
stretching_deformation(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import stretching_deformation
from metrust.units import units

u = np.random.uniform(-15, 15, (50, 50)) * units("m/s")
v = np.random.uniform(-15, 15, (50, 50)) * units("m/s")

stretch_def = stretching_deformation(u, v, 25000 * units.m, 25000 * units.m)
```

---

### `total_deformation`

Total deformation of a 2-D wind field. The magnitude of the combined shearing
and stretching deformation: `sqrt(shearing^2 + stretching^2)`.

```python
total_deformation(u, v, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

**Returns** -- 2-D array `Quantity` in 1/s.

```python
import numpy as np
from metrust.calc import total_deformation
from metrust.units import units

u = np.random.uniform(-15, 15, (50, 50)) * units("m/s")
v = np.random.uniform(-15, 15, (50, 50)) * units("m/s")

total_def = total_deformation(u, v, 25000 * units.m, 25000 * units.m)
```

---

## Q-Vectors

### `q_vector`

Q-vector on a 2-D grid. Q-vectors are used to diagnose quasigeostrophic
vertical motion. The Q-vector convergence is proportional to forcing for
ascent.

```python
q_vector(u, v, temperature, pressure, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | 2-D array `Quantity` | U-component of wind (m/s) |
| `v` | 2-D array `Quantity` | V-component of wind (m/s) |
| `temperature` | 2-D array `Quantity` | Temperature field (K or degC) |
| `pressure` | scalar `Quantity` | Pressure level of the analysis (hPa) |
| `dx` | scalar `Quantity` | Grid spacing in the x-direction (m) |
| `dy` | scalar `Quantity` | Grid spacing in the y-direction (m) |

Additional keyword arguments (`x_dim`, `y_dim`) are accepted for MetPy
compatibility but are ignored.

**Returns** -- Tuple of `(q1, q2)`, each a 2-D array. The Q-vector x and y
components.

```python
import numpy as np
from metrust.calc import q_vector
from metrust.units import units

u = np.random.uniform(-20, 20, (50, 50)) * units("m/s")
v = np.random.uniform(-20, 20, (50, 50)) * units("m/s")
temp = np.random.uniform(250, 280, (50, 50)) * units.K
dx = 25000 * units.m
dy = 25000 * units.m

q1, q2 = q_vector(u, v, temp, 500 * units.hPa, dx, dy)
```

---

## Geospatial Operators

### `geospatial_gradient`

Gradient of a scalar field on an irregular latitude/longitude grid. Returns
physical-space derivatives in meters, accounting for the convergence of
meridians.

```python
geospatial_gradient(data, lats, lons)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array | Scalar field (optionally with Pint units) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `lons` | 2-D array | Longitude at each grid point (degrees) |

**Returns** -- Tuple of `(df/dx, df/dy)`, each a 2-D array. If the input `data`
carries Pint units, the output units are `data_units / m`. Otherwise the arrays
are dimensionless.

```python
import numpy as np
from metrust.calc import geospatial_gradient
from metrust.units import units

lon, lat = np.meshgrid(np.linspace(-100, -90, 50), np.linspace(30, 40, 50))
height = np.random.uniform(5400, 5800, (50, 50)) * units.m

dhdx, dhdy = geospatial_gradient(height, lat, lon)  # units: m/m (dimensionless)
```

---

### `geospatial_laplacian`

Laplacian of a scalar field on an irregular latitude/longitude grid.

```python
geospatial_laplacian(data, lats, lons)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array | Scalar field (optionally with Pint units) |
| `lats` | 2-D array | Latitude at each grid point (degrees) |
| `lons` | 2-D array | Longitude at each grid point (degrees) |

**Returns** -- 2-D array. If the input `data` carries Pint units, the output
units are `data_units / m^2`. Otherwise the array is dimensionless.

```python
import numpy as np
from metrust.calc import geospatial_laplacian
from metrust.units import units

lon, lat = np.meshgrid(np.linspace(-100, -90, 50), np.linspace(30, 40, 50))
height = np.random.uniform(5400, 5800, (50, 50)) * units.m

lap = geospatial_laplacian(height, lat, lon)  # units: m / m^2 = 1/m
```

---

## Cross-Section Components

### `normal_component`

Normal (perpendicular) component of wind relative to a cross-section line
defined by start and end lat/lon points.

```python
normal_component(u, v, start, end)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | array `Quantity` | U-component of wind (m/s) |
| `v` | array `Quantity` | V-component of wind (m/s) |
| `start` | tuple of `(lat, lon)` | Start point of the cross-section (degrees) |
| `end` | tuple of `(lat, lon)` | End point of the cross-section (degrees) |

**Returns** -- Array `Quantity` in m/s.

```python
import numpy as np
from metrust.calc import normal_component
from metrust.units import units

u = np.array([10, 15, 20]) * units("m/s")
v = np.array([5, 10, 15]) * units("m/s")

norm = normal_component(u, v, (30.0, -95.0), (45.0, -85.0))
```

---

### `tangential_component`

Tangential (parallel) component of wind relative to a cross-section line
defined by start and end lat/lon points.

```python
tangential_component(u, v, start, end)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | array `Quantity` | U-component of wind (m/s) |
| `v` | array `Quantity` | V-component of wind (m/s) |
| `start` | tuple of `(lat, lon)` | Start point of the cross-section (degrees) |
| `end` | tuple of `(lat, lon)` | End point of the cross-section (degrees) |

**Returns** -- Array `Quantity` in m/s.

```python
import numpy as np
from metrust.calc import tangential_component
from metrust.units import units

u = np.array([10, 15, 20]) * units("m/s")
v = np.array([5, 10, 15]) * units("m/s")

tang = tangential_component(u, v, (30.0, -95.0), (45.0, -85.0))
```

---

### `cross_section_components`

Decompose wind into components parallel and perpendicular to a cross-section
line. Combines `tangential_component` and `normal_component` in a single call.

```python
cross_section_components(u, v, start_lat, start_lon, end_lat, end_lon)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `u` | array `Quantity` | U-component of wind (m/s) |
| `v` | array `Quantity` | V-component of wind (m/s) |
| `start_lat` | `float` | Start latitude (degrees) |
| `start_lon` | `float` | Start longitude (degrees) |
| `end_lat` | `float` | End latitude (degrees) |
| `end_lon` | `float` | End longitude (degrees) |

**Returns** -- Tuple of `(parallel, perpendicular)`, each an array `Quantity` in
m/s.

```python
import numpy as np
from metrust.calc import cross_section_components
from metrust.units import units

u = np.array([10, 15, 20, 25]) * units("m/s")
v = np.array([5, 10, 15, 20]) * units("m/s")

parallel, perpendicular = cross_section_components(
    u, v, 30.0, -95.0, 45.0, -85.0,
)
```

---

## Notes

### Grid spacing convention

All `dx` and `dy` parameters are scalar grid spacings in meters. They
represent uniform spacing and are not per-grid-point arrays. Pass them as Pint
Quantities:

```python
dx = 25000 * units.m   # 25 km grid spacing
dy = 25000 * units.m
```

### Rust kernels

Every function in this module delegates to a compiled Rust kernel via PyO3.
The Python layer handles unit conversion (stripping Pint units to the
Rust-native convention, calling the Rust function, and re-attaching
appropriate units to the result). There is no Python-level finite-difference
loop -- all differentiation happens in Rust.

### Input array requirements

2-D grid functions expect arrays where the first axis is y (north-south, rows)
and the second axis is x (east-west, columns). This matches the standard
NumPy/MetPy convention for gridded meteorological data. Arrays are internally
converted to contiguous `float64` before being passed to Rust.
