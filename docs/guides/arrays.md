# Array Support

metrust accepts scalars, 1-D arrays, and full 2-D grids through the same
function signatures. This is a key difference from MetPy, which often requires
separate code paths or manual iteration for grid-scale data. In metrust, you
call the same function regardless of whether the input is a single observation
or an HRRR-sized grid, and the library dispatches to the fastest available
implementation automatically.

---

## Three dispatch tiers

Every function in `metrust.calc` handles three tiers of input:

### Scalar

Pass a single float or a Pint scalar quantity. The result is a scalar.

```python
from metrust.calc import potential_temperature
from metrust.units import units

theta = potential_temperature(850 * units.hPa, 25 * units.degC)
# Returns a single Pint Quantity: ~298.9 K
```

### 1-D array

Pass a numpy array (with or without Pint units). The result is an array of
the same length.

```python
import numpy as np
from metrust.calc import dewpoint_from_relative_humidity
from metrust.units import units

t = np.array([20.0, 25.0, 30.0]) * units.degC
rh = np.array([50.0, 65.0, 80.0]) * units.percent
td = dewpoint_from_relative_humidity(t, rh)
# Returns a 1-D array of length 3
```

### 2-D grid

Pass a 2-D `(ny, nx)` array. The original shape is preserved through the Rust
computation and back into the returned quantity.

```python
import numpy as np
from metrust.calc import potential_temperature, dewpoint_from_relative_humidity
from metrust.units import units

p_grid = np.full((1059, 1799), 850.0) * units.hPa
t_grid = np.random.uniform(15, 30, (1059, 1799)) * units.degC
theta = potential_temperature(p_grid, t_grid)  # shape (1059, 1799) preserved
```

No special grid mode, wrapper, or flag is needed. The function detects the
input shape and restores it on the way out.

---

## How it works under the hood

The dispatch system has three layers, chosen at call time depending on what
the Rust extension exposes for that function.

### Dedicated Rust `_array` bindings (fastest path)

26 high-traffic thermodynamic functions have hand-written Rust bindings that
accept a flat `float64` array and return a flat `float64` array. The entire
loop runs in compiled Rust with no per-element Python overhead. These
functions include:

- `potential_temperature`
- `equivalent_potential_temperature`
- `saturation_vapor_pressure`
- `saturation_mixing_ratio`
- `wet_bulb_temperature`
- `dewpoint_from_relative_humidity`
- `relative_humidity_from_dewpoint`
- `virtual_temperature`
- `mixing_ratio`
- `density`
- `dewpoint`
- `exner_function`
- `vapor_pressure`
- `frost_point`
- `temperature_from_potential_temperature`
- `virtual_potential_temperature`
- `wet_bulb_potential_temperature`
- `saturation_equivalent_potential_temperature`
- `specific_humidity_from_mixing_ratio`
- `specific_humidity_from_dewpoint`
- `dewpoint_from_specific_humidity`
- `mixing_ratio_from_relative_humidity`
- `mixing_ratio_from_specific_humidity`
- `relative_humidity_from_mixing_ratio`
- `relative_humidity_from_specific_humidity`
- `virtual_temperature_from_dewpoint`

When you call one of these with array input, the Python wrapper calls the
corresponding `_calc.<name>_array(flat_array, ...)` binding directly. No
Python-level loop is involved.

### `_vec_call` Python-side vectorizer (fallback path)

Functions that do not yet have a dedicated `_array` Rust binding use the
`_vec_call` helper. This flattens the inputs, loops in Python calling the
scalar Rust function once per element, and reshapes the result. It is slower
than a native array binding but still faster than a pure-Python implementation
because each element-level computation runs in Rust.

Functions that currently use `_vec_call` include `heat_index`, `windchill`,
`apparent_temperature`, `dry_static_energy`, `moist_static_energy`,
`scale_height`, `vertical_velocity`, `geopotential_to_height`, and others.

### The `_prep()` helper

Both paths share the `_prep()` helper, which is the core of the dispatch
logic. Given one or more stripped (unit-free) values, it determines how to
proceed:

1. **All inputs are 0-d (scalars):** returns plain floats and marks the call
   as scalar dispatch.
2. **Any input is an array:** broadcasts all inputs to a common shape (using
   `np.broadcast_shapes`, the same rules as numpy), flattens them to
   contiguous 1-D `float64` arrays, and records the original shape for later
   reshape.

The calling function then either sends the flat arrays to the Rust `_array`
binding or iterates with `_vec_call`, and reshapes the result back to the
original shape before attaching units.

```
caller passes (ny, nx) arrays with Pint units
       |
   _strip() removes units, converts to target unit system
       |
   _prep() detects shapes, broadcasts, flattens to 1-D
       |
       +---> _calc.<fn>_array(flat)   [Rust, zero Python loop]
       |           or
       +---> _vec_call(scalar_fn, flat)  [Python loop, Rust per element]
       |
   result.reshape(original_shape)
       |
   _attach() or * units.<unit> adds Pint units back
       |
   returns Quantity with original (ny, nx) shape
```

---

## Broadcasting

When inputs have different shapes, they are broadcast together following
standard numpy broadcasting rules (via `np.broadcast_shapes`). This means you
can mix scalars and arrays freely:

```python
import numpy as np
from metrust.calc import dewpoint_from_relative_humidity
from metrust.units import units

# Single pressure, grid of temperatures
t_grid = np.random.uniform(15, 35, (100, 200)) * units.degC
rh_scalar = 60.0 * units.percent

td = dewpoint_from_relative_humidity(t_grid, rh_scalar)
# Result shape: (100, 200)
```

A 1-D column of pressures can be broadcast against a 2-D temperature field
as long as the shapes are compatible under numpy rules:

```python
import numpy as np
from metrust.calc import potential_temperature
from metrust.units import units

pressure_col = np.linspace(1000, 200, 50).reshape(50, 1) * units.hPa
t_field = np.random.uniform(-40, 30, (50, 100)) * units.degC

theta = potential_temperature(pressure_col, t_field)
# Result shape: (50, 100)
```

---

## Grid composite functions

For full 3-D model fields (e.g., WRF, HRRR, RAP output on native levels),
metrust provides a set of `compute_*` functions that accept `(nz, ny, nx)`
arrays directly and return `(ny, nx)` results. These are parallelized in Rust
and handle the vertical integration or search internally, so there is no need
to write column-by-column loops in Python.

### Available grid composites

| Function | 3-D inputs | 2-D inputs | Returns |
|----------|-----------|-----------|---------|
| `compute_cape_cin` | pressure, T, qvapor, height AGL | psfc, T2m, Q2m | CAPE, CIN, LCL height, LFC height |
| `compute_ecape` | pressure, T, qvapor, height AGL, u, v | psfc, T2m, Q2m, U10m, V10m | ECAPE, NCAPE, CAPE, CIN, LFC height, EL height |
| `compute_srh` | u, v, height AGL | -- | storm-relative helicity |
| `compute_shear` | u, v, height AGL | -- | bulk wind shear magnitude |
| `compute_lapse_rate` | T, qvapor, height AGL | -- | environmental lapse rate |
| `compute_pw` | qvapor, pressure | -- | precipitable water |

These accept the raw model arrays. Units are expected in the Rust-native
convention (Pa or hPa for pressure, Celsius for temperature, kg/kg for mixing
ratio, m for height, m/s for wind). Pint quantities are also accepted and
will be stripped automatically.

### Example: CAPE/CIN on a full HRRR grid

```python
import numpy as np
from metrust.calc import compute_cape_cin

# Shapes: (nz, ny, nx) for 3-D fields, (ny, nx) for surface fields
nz, ny, nx = 50, 1059, 1799

pressure_3d = ...   # (nz, ny, nx) in Pa
temperature_3d = ... # (nz, ny, nx) in Celsius
qvapor_3d = ...     # (nz, ny, nx) in kg/kg
height_agl_3d = ... # (nz, ny, nx) in meters
psfc = ...          # (ny, nx) in Pa
t2m = ...           # (ny, nx) in K
q2m = ...           # (ny, nx) in kg/kg

cape, cin, lcl_hgt, lfc_hgt = compute_cape_cin(
    pressure_3d, temperature_3d, qvapor_3d, height_agl_3d,
    psfc, t2m, q2m,
    parcel_type="surface",
)
# Each output is shape (1059, 1799) with Pint units
```

### 2-D composite parameters

Once you have CAPE, shear, and helicity grids, the `compute_stp`,
`compute_scp`, `compute_ehi`, `compute_ship`, `compute_dcp`, and
`compute_grid_scp` functions combine them element-wise on 2-D `(ny, nx)`
fields:

```python
from metrust.calc import compute_stp

stp = compute_stp(cape, lcl_hgt, srh_1km, shear_6km)
# Input and output are all (ny, nx)
```

---

## Comparison with MetPy

MetPy's thermodynamic functions generally operate on scalars or 1-D arrays
that represent a single sounding profile. Computing a field like potential
temperature over a 2-D grid requires either manual vectorization or relying
on numpy broadcasting within MetPy's Python-level math. For vertical
integration functions like CAPE, MetPy operates on one column at a time,
so covering a full grid means writing an explicit loop over grid points.

metrust eliminates both of these issues:

- **Element-wise functions** (potential temperature, dewpoint, etc.) dispatch
  to Rust array bindings that process the entire flattened grid in a single
  call with no Python loop.
- **Column-integration functions** (CAPE/CIN, SRH, shear, lapse rate, PW)
  accept the full 3-D field and parallelize across grid columns in Rust.

The result is that the same code works for a single observation, a sounding
profile, or a million-point grid, with no API changes and no manual
iteration.
