# Architecture

How the Rust and Python layers of metrust connect, from low-level crate
structure through the PyO3 bridge to the Python compatibility wrappers.

---

## 1. Crate Structure

The workspace lives in two repositories. The **rustmet** repo contains the
foundational crates; the **metrust-py** repo adds the MetPy-compatible library
crate and the Python cdylib.

### rustmet crates (upstream)

| Crate | Role |
|---|---|
| **wx-field** | Shared type foundation. `Field2D`, `Projection`, `RadialSweep`, `SoundingProfile`, `FieldMeta`, `ValidTime` -- the common data structures every other crate depends on. Minimal dependencies (just `chrono`). |
| **wx-math** | Pure-math meteorological computations. Submodules: `thermo`, `dynamics`, `gridmath`, `composite`, `regrid`, `interpolate`. Depends only on `wx-field` and `rayon`. No I/O, no rendering. |
| **wx-radar** | NEXRAD Level-II parser, PPI renderer, color tables, storm cell detection. Depends on `wx-field`, `byteorder`, `bzip2`, `flate2`, `rayon`. |
| **wx-core** | GRIB2 parser/writer, model download client (HRRR/GFS/NAM/RAP/...), rendering pipeline (Skew-T, hodograph, contour, raster), unit conversion, grid operations. Pulls in `wx-field` and `wx-math`. |

### metrust-py crates

| Crate | Role |
|---|---|
| **metrust** (library) | The aggregation crate. Re-exports and organises everything into a MetPy-shaped namespace: `calc::{thermo, wind, kinematics, severe, atmo, smooth, utils}`, `constants`, `interpolate`, `io`, `plots`, `projections`, `units`. Depends on `wx-core`, `wx-math`, `wx-field`, `wx-radar`. |
| **metrust-py** (root, cdylib) | The PyO3 extension module. Compiles to `_metrust.pyd` / `_metrust.so`. Depends on `metrust`, `pyo3`, `numpy`, `rayon`. |

### Dependency graph (simplified)

```
wx-field
  ├── wx-math
  ├── wx-radar
  └── wx-core  (also depends on wx-math)
        └── metrust  (also depends on wx-math, wx-field, wx-radar)
              └── metrust-py  (cdylib, links pyo3 + numpy)
                    └── python/metrust/  (pure-Python wrapper layer)
```

---

## 2. The PyO3 Bridge

The bridge is defined in `src/lib.rs` of the metrust-py root crate. It creates
a single native module called `_metrust` with nested submodules:

```rust
#[pymodule]
fn _metrust(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let calc = PyModule::new(py, "calc")?;
    py_thermo::register(py, &calc)?;
    py_wind::register(py, &calc)?;
    py_kinematics::register(py, &calc)?;
    py_severe::register(py, &calc)?;
    py_atmo::register(py, &calc)?;
    py_smooth::register(py, &calc)?;
    py_utils::register(py, &calc)?;
    m.add_submodule(&calc)?;

    let io_mod = PyModule::new(py, "io")?;
    py_io::register(py, &io_mod)?;
    m.add_submodule(&io_mod)?;

    let interp = PyModule::new(py, "interpolate")?;
    py_interpolate::register(py, &interp)?;
    m.add_submodule(&interp)?;

    let constants = PyModule::new(py, "constants")?;
    py_constants::register(py, &constants)?;
    m.add_submodule(&constants)?;

    Ok(())
}
```

Each `py_*.rs` file exposes one domain's functions and uses a `register()`
function to wire them into the parent module with `wrap_pyfunction!`.

### How numpy arrays cross the FFI boundary

**Python to Rust (input):**

```rust
fn divergence<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,    // zero-copy borrow of a numpy array
    v: PyReadonlyArray2<f64>,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = u.shape();
    let (ny, nx) = (shape[0], shape[1]);
    // .as_slice() gives &[f64] -- no copy, no allocation
    let result = metrust::calc::kinematics::divergence(
        u.as_slice()?, v.as_slice()?, nx, ny, dx, dy,
    );
    // Allocate a new numpy array from the Rust Vec
    Ok(PyArray2::from_vec2(py, &to_2d(&result, ny, nx))?)
}
```

The pattern is always the same:

1. Accept `PyReadonlyArray1<f64>` or `PyReadonlyArray2<f64>` (zero-copy
   borrow of the numpy buffer).
2. Call `.as_slice()?` to get `&[f64]` -- a direct pointer into numpy's
   memory, no copies.
3. Pass the slice to the pure-Rust function.
4. Return results via `PyArray1::from_vec(py, result)` or
   `result.into_pyarray(py)` -- allocates a new numpy array backed by the
   Rust `Vec<f64>`.

### The allow_threads pattern for GIL release

Array-variant bindings release the GIL so Rayon's thread pool can run
without blocking other Python threads:

```rust
fn potential_temperature_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let result: Vec<f64> = py.allow_threads(|| {
        p.par_iter().zip(t.par_iter())
            .map(|(&p, &t)| metrust::calc::thermo::potential_temperature(p, t))
            .collect()
    });
    Ok(result.into_pyarray(py))
}
```

`py.allow_threads(|| ...)` releases the GIL for the duration of the closure.
Inside, `par_iter()` fans the work across Rayon's thread pool. The result
`Vec<f64>` is collected before re-acquiring the GIL, then converted to a
numpy array.

---

## 3. The Python Wrapper Layer

The pure-Python layer lives in `python/metrust/calc/__init__.py` (~5700
lines). It provides MetPy-compatible function signatures, handles Pint unit
conversion, manages xarray coordinate propagation, and dispatches to the
Rust bindings.

### Key internal helpers

#### `_strip(value, target_unit)`

Converts a Pint Quantity to the unit system the Rust functions expect and
extracts the bare magnitude. Plain floats pass through unchanged.

```python
def _strip(quantity, target_unit):
    if hasattr(quantity, "magnitude"):
        return quantity.to(target_unit).magnitude
    return quantity
```

#### `_prep(*values)`

Broadcasts stripped values to a common shape and flattens them for dispatch.
Returns `(flat_arrays, orig_shape, is_array)`. Used to decide scalar vs
array path.

```python
def _prep(*values):
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    if all(a.ndim == 0 for a in arrays):
        return [float(a) for a in arrays], (), False
    # ... broadcasting, shape detection, flattening ...
    return flat, orig_shape, True
```

#### `_vec_call(fn, *stripped_args)`

Element-wise dispatch over a scalar Rust function when no dedicated array
binding exists. Loops in Python, calling the Rust scalar function once per
element.

```python
def _vec_call(fn, *stripped_args):
    vals, shape, is_arr = _prep(*stripped_args)
    if not is_arr:
        return fn(*vals)
    n = vals[0].size
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = fn(*[v[i] for v in vals])
    return result.reshape(shape)
```

#### `_flat(data, unit=None)`

Strips units and returns a contiguous `float64` array, preserving the
original shape (unlike `_prep` which flattens).

#### `_wrap_result_like(template, values, unit_str=None)`

Re-wraps a raw numpy result to match the input type. If the template is an
xarray `DataArray`, the result gets the same coordinates and dimensions. If
the template is a Pint Quantity, units are reattached.

```python
def _wrap_result_like(template, values, unit_str=None):
    arr = np.asarray(values, dtype=np.float64)
    if hasattr(template, "coords") and hasattr(template, "dims"):
        result = xr.DataArray(arr, coords=template.coords, dims=template.dims)
        if unit_str is not None:
            result.attrs["units"] = unit_str
        return result
    return arr * units(unit_str) if unit_str is not None else arr
```

#### `_resolve_dx_dy(data, dx, dy, latitude, longitude)`

Infers grid spacing when not explicitly provided. Tries `data.metpy.latitude`
/ `data.metpy.longitude`, then falls back to xarray coordinate names
(`latitude`, `lat`, `longitude`, `lon`). Computes deltas via
`lat_lon_grid_deltas()`.

#### `_get_scale_factors(data)`

Extracts map-projection scale factors from xarray CRS metadata
(`data.metpy.cartopy_crs`). Returns `(parallel_scale, meridional_scale)` as
2D arrays matching the data shape, used for CRS-corrected metric gradients.

---

## 4. The Three Dispatch Paths

Every Python-facing function in `metrust.calc` uses one of three strategies
to reach the Rust computation:

### Path A: Rust array bindings (`_array` variants)

For the most performance-critical functions, a dedicated `_array` binding
exists in the PyO3 layer. These use `par_iter` + `allow_threads` for full
multithreaded parallelism with the GIL released.

```python
def potential_temperature(pressure, temperature):
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"),
                                _strip(temperature, "degC"))
    if is_arr:
        result = np.asarray(
            _calc.potential_temperature_array(vals[0], vals[1])
        ).reshape(shape)
    else:
        result = _calc.potential_temperature(vals[0], vals[1])
    return result * units.K
```

The `_calc.potential_temperature_array` call crosses into Rust where
`par_iter` distributes the work. This is the fastest path.

### Path B: Python vectorizer (`_vec_call`)

For functions that have only a scalar Rust binding, `_vec_call` loops in
Python and calls the Rust scalar function once per element:

```python
def some_function(pressure, temperature, dewpoint):
    return _vec_call(
        _calc.some_function,
        _strip(pressure, "hPa"),
        _strip(temperature, "degC"),
        _strip(dewpoint, "degC"),
    ) * units("J/kg")
```

This avoids reimplementing the math in Python but has per-call FFI overhead.
Adding an `_array` variant to the Rust side is the standard upgrade path.

### Path C: Python-side computation

Some functions are too complex for a simple scalar-to-array lift, or depend
on Python-ecosystem features. These are implemented entirely in Python,
typically calling multiple Rust scalar or array functions internally:

- **Variable-spacing gradients**: compute first/second derivatives on
  non-uniform grids where dx/dy vary per grid point.
- **CAPE integration with user parcel profiles**: `cape_cin()` accepts
  arbitrary parcel profiles, performs pressure-level interpolation in Python,
  then delegates the core integration to `_calc.cape_cin_core()`.
- **cross_section**: delegates to MetPy when available.
- **Natural neighbor interpolation**: delegates to SciPy.

---

## 5. Unit Flow

The unit lifecycle for a typical function call:

```
Python caller
  │  potential_temperature(1000 * units.hPa, 20 * units.degC)
  │
  ▼
_strip(pressure, "hPa")        →  1000.0  (bare float64)
_strip(temperature, "degC")    →  20.0    (bare float64)
  │
  ▼
_prep(1000.0, 20.0)            →  ([1000.0], [20.0], (), False)
  │
  ▼
_calc.potential_temperature(1000.0, 20.0)   ← Rust FFI call
  │                                            (pure f64, no units)
  ▼
293.15                          ← bare float64 from Rust
  │
  ▼
result * units.K               →  293.15 K  (Pint Quantity)
  │
  ▼
returned to caller
```

Conventions inside Rust:

| Domain | Unit |
|---|---|
| Pressure | hPa (millibars) |
| Temperature | Celsius (potential temperature in Kelvin) |
| Mixing ratio | g/kg |
| Relative humidity | percent (0--100) |
| Wind speed | m/s |
| Height | meters |
| Grid spacing (dx, dy) | meters |
| Angles | degrees |
| Energy | J/kg |

The Python wrapper is responsible for converting to these conventions on
input (`_strip`) and attaching the correct output units (`* units.K`,
`* units("J/kg")`, etc.).

---

## 6. The Compatibility Layer

### Function signature detection

Some functions accept positional arguments whose meaning depends on type.
For example, `cape_cin` detects whether the 4th positional argument is a
parcel profile (temperature-like) or height (length-like) by calling
`_can_convert(value, "hPa")` and `_can_convert(value, "degC")`.

### MetPy alias mapping

A `_COMPAT_ALIASES` dict maps MetPy's alternative names to the canonical
metrust names:

```python
_COMPAT_ALIASES = {
    "significant_tornado": "significant_tornado_parameter",
    "supercell_composite": "supercell_composite_parameter",
    "total_totals_index": "total_totals",
}
```

A module-level `__getattr__` intercepts attribute lookups for these aliases
and redirects to the canonical function:

```python
def __getattr__(name):
    if name in _COMPAT_ALIASES:
        return globals()[_COMPAT_ALIASES[name]]
    raise AttributeError(...)
```

### xarray coordinate inference

Functions that need grid coordinates (kinematics, smoothing) use
`_infer_lat_lon()` to pull latitude/longitude from:

1. `data.metpy.latitude` / `data.metpy.longitude` (MetPy accessor)
2. xarray coordinate names: `latitude`, `lat`, `longitude`, `lon`

Results are re-wrapped via `_wrap_result_like()` to preserve `DataArray`
coordinates and dimensions.

### Fallback paths

When a feature requires a library that metrust does not reimplement:

- **Natural neighbor interpolation**: delegates to `scipy.interpolate`.
- **cross_section**: delegates to MetPy's `cross_section()` if available.

---

## 7. Adding a New Function End-to-End

### Step 1: Rust implementation

Write the core function in the appropriate `metrust` submodule. All inputs
and outputs are plain `f64` or `&[f64]`/`Vec<f64>`. No units, no Python
types.

```
crates/metrust/src/calc/thermo.rs   (or wind.rs, kinematics.rs, etc.)
```

Re-export it from `crates/metrust/src/calc/mod.rs`:

```rust
pub use thermo::my_new_function;
```

### Step 2: PyO3 binding

Add the `#[pyfunction]` in the corresponding `src/py_*.rs` file. For scalar
functions:

```rust
#[pyfunction]
fn my_new_function(py: Python, pressure: f64, temperature: f64) -> f64 {
    let _ = py;
    metrust::calc::thermo::my_new_function(pressure, temperature)
}
```

For an array variant (optional but recommended for performance):

```rust
#[pyfunction]
fn my_new_function_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<f64>,
    temperature: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let t = temperature.as_slice()?;
    let result: Vec<f64> = py.allow_threads(|| {
        p.par_iter().zip(t.par_iter())
            .map(|(&p, &t)| metrust::calc::thermo::my_new_function(p, t))
            .collect()
    });
    Ok(result.into_pyarray(py))
}
```

Register both in the `register()` function:

```rust
parent.add_function(wrap_pyfunction!(my_new_function, parent)?)?;
parent.add_function(wrap_pyfunction!(my_new_function_array, parent)?)?;
```

### Step 3: Python wrapper

Add a user-facing function in `python/metrust/calc/__init__.py`:

```python
def my_new_function(pressure, temperature):
    """One-line description.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (K)
    """
    vals, shape, is_arr = _prep(
        _strip(pressure, "hPa"),
        _strip(temperature, "degC"),
    )
    if is_arr:
        result = np.asarray(
            _calc.my_new_function_array(vals[0], vals[1])
        ).reshape(shape)
    else:
        result = _calc.my_new_function(vals[0], vals[1])
    return result * units.K
```

### Step 4: Export

Add `"my_new_function"` to the `__all__` list at the bottom of
`python/metrust/calc/__init__.py`.

### Step 5: Tests

Add verification tests in `tests/verify_thermo.py` (or the appropriate
`verify_*.py` file), comparing results against MetPy's output.
