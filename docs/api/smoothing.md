# Smoothing and Interpolation

Grid smoothing filters and finite-difference calculus operators.
All smoothing functions delegate to compiled Rust kernels for performance.
Calculus functions compute spatial derivatives on 2-D grids using
second-order centered finite differences (also in Rust).

Every function transparently handles `pint.Quantity` arrays: pass data
with units in, get results with correct units out.

---

## Smoothing

### `smooth_gaussian`

2-D Gaussian smoothing of a scalar grid.

```python
metrust.calc.smooth_gaussian(data, sigma)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `sigma` | float | Standard deviation of the Gaussian kernel, in grid-point units. |

**Returns**

2-D `ndarray` (or Quantity if the input carried units).

**Example**

```python
import numpy as np
from metrust.calc import smooth_gaussian

temperature = np.random.rand(100, 100)
smoothed = smooth_gaussian(temperature, sigma=3.0)
```

---

### `smooth_rectangular`

Rectangular (box-average) smoothing of a scalar grid.

```python
metrust.calc.smooth_rectangular(data, size, passes=1)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `size` | int | Side length of the square averaging kernel. |
| `passes` | int, optional | Number of times to apply the filter. Default `1`. |

**Returns**

2-D `ndarray` (or Quantity if the input carried units).

**Example**

```python
from metrust.calc import smooth_rectangular

smoothed = smooth_rectangular(temperature, size=5, passes=2)
```

---

### `smooth_circular`

Circular (disk) smoothing of a scalar grid.

```python
metrust.calc.smooth_circular(data, radius, passes=1)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `radius` | float | Radius of the circular kernel, in grid-point units. |
| `passes` | int, optional | Number of times to apply the filter. Default `1`. |

**Returns**

2-D `ndarray` (or Quantity if the input carried units).

**Example**

```python
from metrust.calc import smooth_circular

smoothed = smooth_circular(temperature, radius=4.0, passes=1)
```

---

### `smooth_n_point`

Classic N-point smoother (5-point or 9-point stencils).

```python
metrust.calc.smooth_n_point(data, n, passes=1)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `n` | int | Stencil size. Must be `5` or `9`. |
| `passes` | int, optional | Number of times to apply the filter. Default `1`. |

**Returns**

2-D `ndarray` (or Quantity if the input carried units).

**Example**

```python
from metrust.calc import smooth_n_point

# Nine-point smoother, applied three times
smoothed = smooth_n_point(temperature, n=9, passes=3)
```

---

### `smooth_window`

Generic 2-D convolution with a user-supplied kernel.

```python
metrust.calc.smooth_window(data, window, passes=1, normalize_weights=True)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `window` | 2-D array | Convolution kernel (weights). |
| `passes` | int, optional | Number of times to apply the filter. Default `1`. |
| `normalize_weights` | bool, optional | Whether to normalize the kernel so its weights sum to 1 before applying. Default `True`. |

**Returns**

2-D `ndarray` (or Quantity if the input carried units).

**Example**

```python
import numpy as np
from metrust.calc import smooth_window

# Custom 3x3 sharpening-aware kernel
kernel = np.array([[0.5, 1.0, 0.5],
                   [1.0, 2.0, 1.0],
                   [0.5, 1.0, 0.5]])

smoothed = smooth_window(temperature, kernel, passes=1)
```

---

## Calculus

### `gradient`

Calculate the gradient of a scalar field. This is a convenience wrapper
that matches the MetPy / NumPy calling convention.

When called with a 2-D field and `deltas`, it uses the native Rust
`gradient_x` / `gradient_y` implementations. Otherwise it falls back
to `numpy.gradient`.

```python
metrust.calc.gradient(f, **kwargs)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `f` | array-like or Quantity | Scalar field (any dimensionality). |
| `deltas` | list of float or Quantity, keyword | Grid spacings, one per dimension. For a 2-D field, `deltas[0]` is dy and `deltas[1]` is dx. |
| `axes` | int or tuple, keyword | Axis or axes along which to compute the gradient (passed to `numpy.gradient` in the fallback path). |

**Returns**

`list` of arrays, one gradient array per dimension. For a 2-D field with
`deltas` the order is `[df/dy, df/dx]`.

**Example**

```python
from metrust.calc import gradient
from metrust.units import units

temperature = np.random.rand(50, 50) * units.K
dx = 1000.0 * units.m
dy = 1000.0 * units.m

grad_y, grad_x = gradient(temperature, deltas=[dy, dx])
# grad_x and grad_y are Quantity arrays in K/m
```

---

### `gradient_x`

Partial derivative df/dx along the x-axis (columns) of a 2-D field.

```python
metrust.calc.gradient_x(data, dx)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `dx` | Quantity (m) or float | Grid spacing in the x-direction (meters). |

**Returns**

2-D array or Quantity with units `data_units / m`.

**Example**

```python
from metrust.calc import gradient_x
from metrust.units import units

dT_dx = gradient_x(temperature, dx=1000.0 * units.m)
```

---

### `gradient_y`

Partial derivative df/dy along the y-axis (rows) of a 2-D field.

```python
metrust.calc.gradient_y(data, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `dy` | Quantity (m) or float | Grid spacing in the y-direction (meters). |

**Returns**

2-D array or Quantity with units `data_units / m`.

**Example**

```python
from metrust.calc import gradient_y
from metrust.units import units

dT_dy = gradient_y(temperature, dy=1000.0 * units.m)
```

---

### `first_derivative`

First derivative of a 2-D field along a chosen axis.

```python
metrust.calc.first_derivative(data, axis_spacing, axis=0)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `axis_spacing` | Quantity (m) or float | Uniform grid spacing along the selected axis. |
| `axis` | int, optional | Axis along which to differentiate: `0` for x (columns), `1` for y (rows). Default `0`. |

**Returns**

2-D array or Quantity with units `data_units / m`.

**Example**

```python
from metrust.calc import first_derivative
from metrust.units import units

dT_dx = first_derivative(temperature, 1000.0 * units.m, axis=0)
dT_dy = first_derivative(temperature, 1000.0 * units.m, axis=1)
```

---

### `second_derivative`

Second derivative of a 2-D field along a chosen axis.

```python
metrust.calc.second_derivative(data, axis_spacing, axis=0)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `axis_spacing` | Quantity (m) or float | Uniform grid spacing along the selected axis. |
| `axis` | int, optional | Axis along which to differentiate: `0` for x, `1` for y. Default `0`. |

**Returns**

2-D array or Quantity with units `data_units / m^2`.

**Example**

```python
from metrust.calc import second_derivative
from metrust.units import units

d2T_dx2 = second_derivative(temperature, 1000.0 * units.m, axis=0)
```

---

### `laplacian`

Laplacian of a 2-D scalar field (d2f/dx2 + d2f/dy2).

```python
metrust.calc.laplacian(data, dx, dy)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | 2-D array or Quantity | Input scalar field. |
| `dx` | Quantity (m) or float | Grid spacing in the x-direction. |
| `dy` | Quantity (m) or float | Grid spacing in the y-direction. |

**Returns**

2-D array or Quantity with units `data_units / m^2`.

**Example**

```python
from metrust.calc import laplacian
from metrust.units import units

lap = laplacian(temperature, dx=1000.0 * units.m, dy=1000.0 * units.m)
```

---

### `lat_lon_grid_deltas`

Compute physical grid spacings (dx, dy) in meters from 2-D latitude and
longitude arrays. Uses the Haversine formula on the Rust side for accuracy.

```python
metrust.calc.lat_lon_grid_deltas(lats, lons)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `lats` | 2-D array (degrees) | Latitude grid. |
| `lons` | 2-D array (degrees) | Longitude grid. |

**Returns**

`tuple` of `(dx, dy)` where each element is a 2-D Quantity array in meters.

**Example**

```python
import numpy as np
from metrust.calc import lat_lon_grid_deltas

lons, lats = np.meshgrid(np.linspace(-100, -90, 50),
                         np.linspace(30, 40, 50))

dx, dy = lat_lon_grid_deltas(lats, lons)
# dx and dy are 2-D arrays in units of meters
```

---

## Notes

- **Rust kernels.** All smoothing and derivative functions call into compiled
  Rust via PyO3. There is no pure-Python fallback path for these operations.
  The one exception is `gradient`, which falls back to `numpy.gradient` when
  called on arrays that are not 2-D or when `deltas` is not provided.

- **Unit handling.** If the input array is a `pint.Quantity`, the result
  carries the correct derived units automatically (e.g., `K` in, `K/m` out
  for a first derivative, `K/m^2` out for a second derivative or Laplacian).
  Plain `ndarray` inputs produce plain `ndarray` outputs.

- **Edge treatment.** Centered finite differences are used in the interior.
  Forward and backward differences are used at the boundaries, matching the
  behavior of `numpy.gradient`.

- **Multi-pass smoothing.** For `smooth_rectangular`, `smooth_circular`,
  `smooth_n_point`, and `smooth_window`, passing `passes > 1` applies the
  filter repeatedly. Multiple passes of a box filter approximate a Gaussian.
