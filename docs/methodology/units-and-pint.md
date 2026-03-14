# Units and Pint Integration

metrust wraps a compiled Rust backend with a Python layer that speaks
[Pint](https://pint.readthedocs.io/) Quantities. This document explains
how the unit layer works, the cross-registry problem with MetPy, and the
patterns used to safely shuttle data between Pint and Rust FFI. The
techniques are applicable to any scientific Python project that wraps a
compiled backend -- not just meteorology.

Source files discussed:

- `python/metrust/units.py` -- registry setup, strip/attach helpers
- `python/metrust/calc/__init__.py` -- public API, prep/dispatch, xarray
  preservation, grid spacing handling

---

## 1. The Application Registry

This is the single most important design decision in the unit layer.

### The problem

Pint tracks which `UnitRegistry` created every `Unit` and `Quantity`. Two
registries cannot interoperate:

```python
import pint

reg_a = pint.UnitRegistry()
reg_b = pint.UnitRegistry()

x = 5 * reg_a.meter
y = 10 * reg_b.meter

x + y  # pint.errors.DimensionalityError:
       # "Cannot operate with Unit and Unit of different registries"
```

MetPy creates its own registry at import time. If metrust also calls
`pint.UnitRegistry()`, any program that imports both libraries will hit
this error the moment a Quantity from one library touches a Quantity from
the other:

```python
from metpy.units import units as metpy_units
from metrust.units import units as metrust_units

T = 25 * metpy_units.degC
metrust.calc.saturation_vapor_pressure(T)  # BOOM
```

### The fix

One line in `units.py`:

```python
units = pint.get_application_registry()
```

`pint.get_application_registry()` returns a process-wide singleton. MetPy
uses the same call. When both libraries use the application registry, they
share a single registry instance, and all Quantities are compatible.

If you are building a Pint-aware wrapper around any compiled backend, this
is the rule: **never call `pint.UnitRegistry()` -- always use
`pint.get_application_registry()`**.

### Offset autoconversion

Immediately after obtaining the registry, metrust enables automatic
conversion of offset units:

```python
units.autoconvert_offset_to_baseunit = True
```

This is discussed further in Section 4.

### Defensive alias registration

Older Pint versions may lack convenience aliases like `degC`, `degF`,
`hPa`, and `knot`. The module guards each with a try/except and defines
them if missing, so metrust works across Pint versions without requiring
users to upgrade.

---

## 2. Unit Stripping and Reattaching

Rust functions accept plain `float` or `numpy.ndarray` values -- they
know nothing about Pint. The unit layer's job is:

1. **Strip**: convert the user's Quantity to the unit convention the Rust
   engine expects, then extract the bare numeric value.
2. **Call Rust** with the raw number.
3. **Attach**: wrap the Rust result back in a Pint Quantity with the
   correct output unit.

### `_strip(quantity, target_unit)`

```python
def _strip(quantity, target_unit):
    if hasattr(quantity, "magnitude"):
        return quantity.to(target_unit).magnitude
    return quantity
```

The `hasattr` check makes the helper tolerant of plain floats/arrays that
already use the Rust-native convention. This lets power users skip Pint
overhead entirely.

There is also `_strip_or_none()`, which passes `None` through unchanged
for optional parameters.

### `_attach(value, unit_str)`

```python
def _attach(value, unit_str):
    return units.Quantity(value, unit_str)
```

Uses the shared application registry to create the output Quantity.

### Rust-native unit conventions

The Rust engine uses a fixed set of internal conventions:

| Quantity         | Rust unit     |
|------------------|---------------|
| Pressure         | hPa (mbar)    |
| Temperature      | Celsius       |
| Potential temp.  | Kelvin        |
| Mixing ratio     | g/kg          |
| Relative humidity| percent 0-100 |
| Wind speed       | m/s           |
| Height           | meters        |
| Grid spacing     | meters        |
| Angles           | degrees       |

The Python layer converts to these conventions on the way in, then
converts the result to the MetPy-compatible output convention on the way
out (see Section 5 for where those differ).

---

## 3. The `_prep()` Helper

After stripping units, values may be scalars, 0-d arrays, 1-D arrays, or
N-D arrays. `_prep()` normalizes them for dispatch:

```python
def _prep(*values):
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    if all(a.ndim == 0 for a in arrays):
        return [float(a) for a in arrays], (), False
    # ... broadcasting logic ...
    return flat, orig_shape, True
```

It returns a triple `(processed_values, orig_shape, is_array)`:

- **All scalars**: returns plain floats, empty shape, `is_array=False`.
  The caller invokes the scalar Rust function directly.
- **Any arrays**: broadcasts all inputs to a common shape, flattens them
  to contiguous 1-D float64 arrays, and returns `is_array=True`. The
  caller invokes the array Rust function, then reshapes the result back
  to `orig_shape`.

For mixed-dimensionality inputs (e.g., a 1-D pressure profile broadcast
against a 2-D temperature field), `_prep` promotes lower-dimensional
arrays by inserting trailing dimensions before broadcasting.

This pattern appears throughout the codebase:

```python
vals, shape, is_arr = _prep(t_raw)
if is_arr:
    result = np.asarray(_calc.some_fn_array(vals[0])).reshape(shape)
else:
    result = _calc.some_fn(vals[0])
```

---

## 4. Temperature Offset Handling

Celsius and Fahrenheit are *offset* units in Pint. By default, Pint
refuses to do arithmetic on them because operations like `25 degC * 2`
are physically ambiguous (does the user mean 50 degC or 25 delta_degC
times 2?).

metrust sets:

```python
units.autoconvert_offset_to_baseunit = True
```

This tells Pint to silently convert offset units to their base unit
(Kelvin for temperature) before arithmetic. Without this flag, any
calculation that internally multiplies or divides a temperature Quantity
would raise a Pint `OffsetUnitCalculusError`.

The stripping layer sidesteps most of this by converting to Celsius (or
Kelvin, depending on the function) before calling Rust, but the flag is
still needed for user-side arithmetic with metrust Quantities (e.g.,
computing a temperature difference and passing it to another function).

---

## 5. Common Unit Gotchas

The Rust engine's internal conventions differ from MetPy's API conventions
in several places. The Python wrapper performs silent conversions to match
MetPy's return types. If you are comparing metrust output to raw Rust
output, these are the transforms to be aware of:

### `saturation_vapor_pressure` returns Pa, not hPa

Rust returns SVP in hPa (its internal convention). The Python wrapper
multiplies by 100 and attaches `"Pa"` to match MetPy:

```python
result = _calc.saturation_vapor_pressure(vals[0]) * 100.0
return _attach(result, "Pa")
```

### `mixing_ratio` and `saturation_mixing_ratio` return kg/kg

Rust returns mixing ratio in g/kg. The wrapper divides by 1000 and
attaches `"kg/kg"`:

```python
result = _calc.mixing_ratio(vals[0], vals[1]) / 1000.0
return _attach(result, "kg/kg")
```

### `relative_humidity_from_dewpoint` returns a fraction [0, 1]

Rust returns RH as a percentage (0-100). The wrapper divides by 100 and
attaches dimensionless units:

```python
result = _calc.relative_humidity_from_dewpoint(vals[0], vals[1]) / 100.0
return _attach(result, "")
```

The inverse helper `_rh_to_percent()` handles the opposite direction:
converting user input (which may be a fraction, a percentage, or a Pint
Quantity with `%` units) back to the 0-100 scale Rust expects.

### Montgomery streamfunction returns kJ/kg

Following MetPy convention, the 2-argument form divides by 1000 and
returns `kJ/kg`:

```python
result = (cp * t_arr + g * h_arr) / 1000.0
return result * units("kJ/kg")
```

### STP and SCP are dimensionless

`significant_tornado_parameter` and `supercell_composite_parameter` are
composite indices with no physical unit:

```python
return _calc.significant_tornado_parameter(cape, lcl, srh, shear) * units.dimensionless
```

---

## 6. Cross-Registry Safe Patterns

Even with the shared application registry, there are edge cases where
unit objects from different sources (xarray `attrs["units"]` strings,
MetPy Quantities, metrust Quantities) can collide. The advection
functions demonstrate the defensive pattern.

### The problem

When computing advection, the output unit is `scalar_units / second`. A
naive implementation might write:

```python
return result * (scalar.units / units.s)   # DANGER
```

If `scalar` came from MetPy or from an xarray DataArray with a string
`units` attribute, `scalar.units` may be a `pint.Unit` from a different
registry, or even a plain string. Dividing it by `units.s` can raise the
cross-registry error.

### The fix: `_safe_unit_str()` and string-based construction

```python
def _safe_unit_str(unit_obj):
    """Return a unit string usable with *our* registry, avoiding cross-registry ops."""
    return str(unit_obj)
```

The advection functions serialize the input unit to a string, then build
the output unit entirely within metrust's registry:

```python
s_unit_str = _safe_unit_str(scalar.units)
return result.reshape(nz, ny, nx) * units(f"({s_unit_str}) / s")
```

By going through a string intermediary, the code never performs arithmetic
between `Unit` objects from different registries. The `units(...)` call
parses the string using the application registry, guaranteeing a
single-registry result.

The 2-D advection function uses a similar pattern:

```python
s_u = units.Unit(str(scalar.units))
out = _wrap_result_like(scalar, result, str(s_u / units.s))
```

Here `str(scalar.units)` serializes whatever the input unit is, and
`units.Unit(...)` re-parses it within the shared registry before dividing
by `units.s`.

**Rule of thumb**: whenever you need to combine a unit from user input
with a unit from your own registry, convert the user's unit to a string
first, re-parse it with your registry, and then combine.

---

## 7. xarray Preservation

Many meteorological workflows use `xarray.DataArray` objects with named
coordinates, dimensions, and attributes. metrust preserves these through
the `_wrap_result_like()` helper:

```python
def _wrap_result_like(template, values, unit_str=None):
    arr = np.asarray(values, dtype=np.float64)
    template_shape = np.asarray(template).shape
    if arr.shape != template_shape:
        arr = arr.reshape(template_shape)
    if hasattr(template, "coords") and hasattr(template, "dims"):
        result = xr.DataArray(arr, coords=template.coords, dims=template.dims)
        if unit_str is not None:
            result.attrs["units"] = unit_str
        return result
    return arr * units(unit_str) if unit_str is not None else arr
```

The logic:

1. Reshape the flat Rust output to match the input template's shape.
2. If the template is an xarray DataArray, return a new DataArray with
   the same `coords`, `dims`, and a `units` attribute in `attrs`.
3. Otherwise, return a Pint Quantity.

This means users can pass xarray DataArrays directly into metrust
functions and get xarray DataArrays back, with coordinates and dimension
names preserved. The unit is stored as a string in `attrs["units"]`
rather than as a Pint Quantity wrapping the DataArray, which matches
the CF convention and avoids the performance overhead of Pint-wrapped
xarray objects.

---

## 8. Grid Spacing Handling

Grid-based functions (advection, divergence, vorticity, frontogenesis)
need horizontal grid spacings `dx` and `dy`. These can arrive in several
forms:

- A scalar Quantity (`100 * units.km`)
- A 1-D array from `lat_lon_grid_deltas`
- A 2-D array of per-gridpoint spacings (latitude-dependent grids)

### `_mean_spacing(val, target_unit="m")`

Extracts a single scalar value for Rust functions that expect uniform
spacing:

```python
def _mean_spacing(val, target_unit="m"):
    if hasattr(val, "magnitude"):
        arr = np.asarray(val.to(target_unit).magnitude, dtype=np.float64)
    else:
        arr = np.asarray(val, dtype=np.float64)
    return float(arr.mean()) if arr.ndim > 0 and arr.size > 1 else float(arr)
```

For a 2-D array of spacings from a lat/lon grid, this returns the mean
value in meters.

### `_is_variable_spacing(val)`

Detects whether grid spacing varies enough to require the variable-spacing
code path:

```python
def _is_variable_spacing(val):
    # ... extract array ...
    if arr.ndim < 2:
        return False
    rng = finite.max() - finite.min()
    return rng > 0.05 * abs(finite.mean())
```

A relative range threshold of 5% distinguishes "effectively uniform"
spacing (e.g., a fine-resolution Lambert conformal grid) from genuinely
variable spacing (e.g., a coarse lat/lon grid where dx shrinks toward the
poles).

### Three-tier dispatch

Grid functions use a three-level strategy:

1. **Uniform spacing** (`dx` and `dy` are scalars, or 2-D with <5%
   variation): call the Rust kernel with a single `dx`/`dy` float via
   `_mean_spacing()`.

2. **Variable 2-D spacing** (`_is_variable_spacing()` returns `True`):
   fall back to `_gradient_2d_variable()`, a pure-NumPy implementation
   using centered finite differences with per-gridpoint spacing. This
   handles MetPy's `lat_lon_grid_deltas` output, which returns arrays of
   shape `(ny, nx-1)` and `(ny-1, nx)` that are padded to full grid size.

3. **Spherical grid** (latitude and longitude coordinates are available):
   `_resolve_dx_dy()` infers lat/lon from xarray coordinates or MetPy's
   `.metpy.latitude`/`.metpy.longitude` accessors, calls
   `lat_lon_grid_deltas()` to compute per-gridpoint spacings, and then
   dispatches to tier 1 or 2 above.

This layering means users can pass any of:

```python
advection(T, u, v, dx=100*units.km, dy=100*units.km)  # uniform
advection(T, u, v, dx=dx_2d, dy=dy_2d)                # variable
advection(T, u, v)                                     # inferred from coords
```

and the function selects the appropriate code path automatically.

---

## Summary for Library Authors

If you are wrapping a compiled backend (Rust, C, Fortran) with a
Pint-aware Python layer, the key takeaways are:

1. **Use `pint.get_application_registry()`**, never `pint.UnitRegistry()`.
2. **Set `autoconvert_offset_to_baseunit = True`** if you handle
   temperature.
3. **Strip units on the way in, attach on the way out.** Your compiled
   code should never see a Pint object.
4. **Normalize shapes** before calling into compiled code -- handle
   scalar vs array dispatch in Python.
5. **Build output units from strings**, not by combining `Unit` objects
   that may come from different registries.
6. **Preserve xarray metadata** by copying `coords`, `dims`, and `attrs`
   from the input to the output.
7. **Document unit conversions** between your internal convention and your
   API convention -- silent scaling factors (like dividing g/kg by 1000
   to get kg/kg) are the most common source of bugs.
