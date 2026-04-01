"""metrust.calc -- MetPy-compatible calculation layer

Every public function accepts and returns Pint Quantity objects with a
MetPy-compatible API surface. Internally, units are stripped to the convention
expected by the Rust engine (hPa for pressure, Celsius for temperature,
m/s for wind, m for height, etc.), the Rust function is called, and
appropriate units are attached to the result.

Plain floats / ndarrays (without Pint units) are passed through as-is,
so callers who already work in the Rust-native unit system can skip the
Pint overhead.

Rust-native conventions
-----------------------
- Pressure:  hPa (millibars)
- Temperature:  Celsius  (potential temperature in Kelvin)
- Mixing ratio:  g/kg
- Relative humidity:  percent [0-100]
- Wind speed:  m/s
- Height:  meters
- Grid spacings (dx, dy):  meters
- Angles:  degrees
"""

import importlib
import inspect
import sys
from contextlib import contextmanager
import numpy as np
try:
    import xarray as xr
except ImportError:
    xr = None
from metrust._metrust import calc as _calc
from metrust.units import units, _strip, _strip_or_none, _attach, _as_float, _as_1d


class InvalidSoundingError(Exception):
    """Raised when sounding data is invalid or insufficient for the requested calculation."""
    pass


_BACKEND = "cpu"
_GPU_CALC = None
_UNIT_HPA = units.hPa
_UNIT_PA = units.Pa
_UNIT_DEGC = units.degC
_UNIT_KELVIN = units.kelvin
_UNIT_JPKG = units.joule / units.kilogram

METPY_COMPATIBILITY_TARGET = {
    "metpy": "1.7.1",
    "python": ("3.10", "3.11", "3.12", "3.13"),
}

METPY_OPTIONAL_CALC_DELEGATIONS = ()


def _normalize_backend_name(backend):
    name = str(backend).strip().lower()
    if name not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return name


def _load_gpu_calc():
    global _GPU_CALC
    if _GPU_CALC is None:
        try:
            _GPU_CALC = importlib.import_module("metcu.calc")
        except Exception as exc:
            raise ImportError(
                "GPU backend requires met-cu and a working CuPy/CUDA environment. "
                "Install with `pip install \"metrust[gpu]\"` or `pip install met-cu`."
            ) from exc
    return _GPU_CALC


def get_backend():
    """Return the active backend name."""
    return _BACKEND


def set_backend(backend):
    """Set the active backend for eligible calculations."""
    global _BACKEND
    name = _normalize_backend_name(backend)
    if name == "gpu":
        _load_gpu_calc()
    _BACKEND = name


@contextmanager
def use_backend(backend):
    """Temporarily switch to a specific backend."""
    previous = get_backend()
    set_backend(backend)
    try:
        yield
    finally:
        set_backend(previous)


def _can_convert(value, unit_str):
    if not hasattr(value, "to"):
        return False
    try:
        value.to(unit_str)
        return True
    except Exception:
        return False


def _rh_to_percent(relative_humidity):
    if hasattr(relative_humidity, "to"):
        try:
            return relative_humidity.to("percent").magnitude
        except Exception:
            return np.asarray(relative_humidity.to("").magnitude, dtype=np.float64) * 100.0
    val = np.asarray(relative_humidity, dtype=np.float64)
    if val.ndim == 0:
        v = float(val)
        return v * 100.0 if abs(v) <= 1.5 else v
    finite = val[np.isfinite(val)]
    if finite.size and np.nanmax(np.abs(finite)) <= 1.5:
        return val * 100.0
    return val


def _rh_to_fraction(relative_humidity):
    return np.asarray(_rh_to_percent(relative_humidity), dtype=np.float64) / 100.0


def _as_2d(data, unit=None):
    """Convert to a contiguous float64 2-D numpy array, optionally stripping units."""
    if unit and hasattr(data, "magnitude"):
        arr = np.asarray(data.to(unit).magnitude, dtype=np.float64)
    elif hasattr(data, "magnitude"):
        arr = np.asarray(data.magnitude, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _as_2d_raw(data):
    """Convert to contiguous float64 2-D array, stripping units if present."""
    if hasattr(data, "magnitude"):
        arr = np.asarray(data.magnitude, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _prep(*values):
    """Convert stripped values to arrays; determine scalar vs array dispatch.

    Returns (processed_values, orig_shape, is_array).
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    if all(a.ndim == 0 for a in arrays):
        return [float(a) for a in arrays], (), False
    target_ndim = max(a.ndim for a in arrays)
    if target_ndim > 1:
        leading_dim = next((a.shape[0] for a in arrays if a.ndim == target_ndim), None)
        if leading_dim is not None:
            promoted = []
            for a in arrays:
                if a.ndim == 1 and a.shape[0] == leading_dim:
                    a = a.reshape((a.shape[0],) + (1,) * (target_ndim - 1))
                promoted.append(a)
            arrays = promoted
    shapes = [a.shape for a in arrays if a.ndim > 0]
    orig_shape = np.broadcast_shapes(*shapes)
    flat = [np.ascontiguousarray(np.broadcast_to(a, orig_shape).ravel()) for a in arrays]
    return flat, orig_shape, True


def _gpu_to_numpy(value):
    if isinstance(value, tuple):
        return tuple(_gpu_to_numpy(v) for v in value)
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _interp_profile_level(pressure, values, target_pressure_hpa, value_unit=None):
    """Interpolate a profile to a pressure level in hPa."""
    p_arr = np.asarray(_strip(pressure, "hPa"), dtype=np.float64).ravel()
    if value_unit is not None and hasattr(values, "magnitude"):
        v_arr = np.asarray(values.to(value_unit).magnitude, dtype=np.float64).ravel()
    elif hasattr(values, "magnitude"):
        v_arr = np.asarray(values.magnitude, dtype=np.float64).ravel()
    else:
        v_arr = np.asarray(values, dtype=np.float64).ravel()

    if p_arr[0] > p_arr[-1]:
        p_arr = p_arr[::-1]
        v_arr = v_arr[::-1]

    return float(np.interp(float(target_pressure_hpa), p_arr, v_arr))


def _coord_values(coord):
    """Extract plain float coordinate values from xarray-like or Pint inputs."""
    if coord is None:
        return None
    if hasattr(coord, "values"):
        coord = coord.values
    elif hasattr(coord, "magnitude"):
        coord = coord.magnitude
    return np.asarray(coord, dtype=np.float64)


def _as_array_with_unit(value, unit=None):
    """Return float64 magnitudes plus the associated Pint unit when available."""
    if hasattr(value, "metpy"):
        try:
            value = value.metpy.unit_array
        except Exception:
            pass

    if hasattr(value, "to"):
        quantity = value.to(unit) if unit is not None else value
        return np.asarray(quantity.magnitude, dtype=np.float64), quantity.units

    if hasattr(value, "values"):
        raw = np.asarray(value.values, dtype=np.float64)
    else:
        raw = np.asarray(value, dtype=np.float64)

    unit_attr = getattr(getattr(value, "attrs", None), "get", lambda *_: None)("units")
    if unit_attr:
        quantity = raw * units(unit_attr)
        quantity = quantity.to(unit) if unit is not None else quantity
        return np.asarray(quantity.magnitude, dtype=np.float64), quantity.units

    return raw, units(unit) if unit is not None else None


def _infer_lat_lon(data, latitude=None, longitude=None):
    """Infer latitude/longitude grids from a field for MetPy-style grid calls."""
    lat = latitude
    lon = longitude

    if lat is None or lon is None:
        try:
            lat = data.metpy.latitude
            lon = data.metpy.longitude
        except Exception:
            pass

    coords = getattr(data, "coords", None)
    if coords is not None:
        if lat is None:
            for name in ("latitude", "lat"):
                if name in coords:
                    lat = coords[name]
                    break
        if lon is None:
            for name in ("longitude", "lon"):
                if name in coords:
                    lon = coords[name]
                    break

    if lat is None or lon is None:
        return None, None

    lat_arr = _coord_values(lat)
    lon_arr = _coord_values(lon)
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr)
    return lat_arr, lon_arr


def _resolve_dx_dy(data, dx=None, dy=None, latitude=None, longitude=None):
    """Resolve grid spacing from explicit args or inferred lat/lon coordinates."""
    if dx is not None and dy is not None:
        return dx, dy

    lat_arr, lon_arr = _infer_lat_lon(data, latitude=latitude, longitude=longitude)
    if lat_arr is None or lon_arr is None:
        return dx, dy

    return lat_lon_grid_deltas(lon_arr, lat_arr)


def _wrap_result_like(template, values, unit_str=None):
    arr = np.asarray(values, dtype=np.float64)
    template_shape = getattr(template, "shape", None)
    if template_shape is None:
        template_shape = np.asarray(template).shape
    if arr.shape != template_shape:
        arr = arr.reshape(template_shape)
    if hasattr(template, "coords") and hasattr(template, "dims"):
        result = xr.DataArray(arr, coords=template.coords, dims=template.dims)
        if unit_str is not None:
            result.attrs["units"] = unit_str
        return result
    return arr * units(unit_str) if unit_str is not None else arr


def _raw_array_ndim(data):
    if hasattr(data, "ndim"):
        return data.ndim
    if hasattr(data, "values"):
        data = data.values
    elif hasattr(data, "magnitude"):
        data = data.magnitude
    return np.asarray(data).ndim


def _broadcast_for_gpu(*args):
    """Broadcast scalar args to match the largest array shape for GPU kernels.

    met-cu CUDA kernels ``.ravel()`` every argument and index by thread-id,
    so all inputs must have the same number of elements.  This helper expands
    any 0-d (scalar) argument to a full array matching the largest shape.
    """
    shapes = [np.shape(a) for a in args]
    ref = max(shapes, key=lambda s: len(s) if s else 0)
    if not ref:
        return args
    out = []
    for a, s in zip(args, shapes):
        if not s and ref:
            out.append(np.broadcast_to(float(a), ref).copy())
        else:
            out.append(a)
    return tuple(out)


def _gpu_uniform_grid_supported(*arrays, dx=None, dy=None, parallel_scale=None,
                                meridional_scale=None, latitude=None,
                                longitude=None, crs=None):
    if _BACKEND != "gpu":
        return False
    if dx is None or dy is None:
        return False
    if any(value is not None for value in (
        parallel_scale, meridional_scale, latitude, longitude, crs,
    )):
        return False
    if _is_variable_spacing(dx) or _is_variable_spacing(dy):
        return False
    return all(_raw_array_ndim(arr) == 2 for arr in arrays)


def _vec_call(fn, *stripped_args):
    """Call scalar Rust fn element-wise over array args; return scalar or reshaped array."""
    vals, shape, is_arr = _prep(*stripped_args)
    if not is_arr:
        return fn(*vals)
    n = vals[0].size
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = fn(*[v[i] for v in vals])
    return result.reshape(shape)


def _is_pressure_like(value):
    return _can_convert(value, "hPa")


def _is_temperature_like(value):
    return _can_convert(value, "degC") or _can_convert(value, "K")


# ---------- SVP ice-phase (Ambaum 2020, matching MetPy / Rust) ----------
_T0 = 273.16
_SAT_P0 = 611.2        # Pa at 0 degC
_CP_L = 4219.4
_CP_V = 1860.078011865639
_CP_I = 2090.0
_RV = 461.52311572606084
_LV0 = 2_500_840.0
_LS0 = 2_834_540.0
_ZEROCNK = 273.15


def _svp_ice_pa(t_k):
    """SVP over ice in Pa (Ambaum 2020 Eq. 17)."""
    lat = _LS0 - (_CP_I - _CP_V) * (t_k - _T0)
    pw = (_CP_I - _CP_V) / _RV
    ex = (_LS0 / _T0 - lat / t_k) / _RV
    return _SAT_P0 * (_T0 / t_k) ** pw * np.exp(ex)


def _normalize_phase(phase):
    phase = str(phase).strip().lower()
    if phase == "solid":
        return "ice"
    return phase


def _svp_with_phase(t_c, phase):
    """SVP in hPa with explicit phase selection."""
    phase = _normalize_phase(phase)
    t_k = np.asarray(t_c, dtype=np.float64) + _ZEROCNK
    if phase == "ice":
        return _svp_ice_pa(t_k) / 100.0
    # auto: ice below T0
    liquid = np.asarray(_calc.saturation_vapor_pressure_array(np.asarray(t_c, dtype=np.float64).ravel()), dtype=np.float64).reshape(t_k.shape)
    ice = _svp_ice_pa(t_k) / 100.0
    if phase == "auto":
        result = np.where(t_k > _T0, liquid, ice)
        return float(result) if np.ndim(result) == 0 else result
    return float(ice) if np.ndim(ice) == 0 else ice

# ============================================================================
# Thermo
# ============================================================================

def potential_temperature(pressure, temperature):
    """Potential temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature, Celsius)

    Returns
    -------
    Quantity (K)
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(
            _load_gpu_calc().potential_temperature(
                _strip(pressure, "hPa"),
                _strip(temperature, "degC"),
            )
        )
        return result * units.K
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"))
    if is_arr:
        result = np.asarray(_calc.potential_temperature_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.potential_temperature(vals[0], vals[1])
    return result * units.K


def equivalent_potential_temperature(pressure, temperature, dewpoint):
    """Equivalent potential temperature (theta-e).

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (K)
    """
    if _BACKEND == "gpu":
        p, t, td = _broadcast_for_gpu(
            _strip(pressure, "hPa"),
            _strip(temperature, "degC"),
            _strip(dewpoint, "degC"),
        )
        result = _gpu_to_numpy(
            _load_gpu_calc().equivalent_potential_temperature(p, t, td)
        )
        return result * units.K
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"), _strip(dewpoint, "degC"))
    if is_arr:
        result = np.asarray(_calc.equivalent_potential_temperature_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.equivalent_potential_temperature(vals[0], vals[1], vals[2])
    return result * units.K


def saturation_vapor_pressure(temperature, phase="liquid"):
    """Saturation vapor pressure.

    Uses Ambaum (2020) matching MetPy exactly.
    ``phase="auto"`` selects ice below 0 degC.

    Parameters
    ----------
    temperature : Quantity (temperature)
    phase : str
        ``"liquid"`` (default), ``"ice"``, or ``"auto"``.

    Returns
    -------
    Quantity (hPa)
    """
    phase = _normalize_phase(phase)
    t_raw = _strip(temperature, "degC")
    if phase == "liquid":
        vals, shape, is_arr = _prep(t_raw)
        if is_arr:
            result = np.asarray(_calc.saturation_vapor_pressure_array(vals[0])).reshape(shape) * 100.0
        else:
            result = _calc.saturation_vapor_pressure(vals[0]) * 100.0
        return _attach(result, "Pa")
    # ice / auto -- Ambaum (2020) ice-phase SVP (same as Rust impl)
    t = _as_float(t_raw)
    return _attach(_svp_with_phase(t, phase) * 100.0, "Pa")


def saturation_mixing_ratio(pressure, temperature, phase="liquid"):
    """Saturation mixing ratio.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    phase : str
        ``"liquid"`` (default), ``"ice"``, or ``"auto"``.

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    phase = _normalize_phase(phase)
    p_raw = _strip(pressure, "hPa")
    t_raw = _strip(temperature, "degC")
    if phase == "liquid":
        vals, shape, is_arr = _prep(p_raw, t_raw)
        if is_arr:
            result = np.asarray(_calc.saturation_mixing_ratio_array(vals[0], vals[1])).reshape(shape) / 1000.0
        else:
            result = _calc.saturation_mixing_ratio(vals[0], vals[1]) / 1000.0
        return _attach(result, "kg/kg")
    # ice / auto
    p = np.asarray(p_raw, dtype=np.float64)
    t = np.asarray(t_raw, dtype=np.float64)
    _EPS = 0.6219569100577033
    es = _svp_with_phase(t, phase)
    result = np.maximum(_EPS * es / (p - es), 0.0)
    return _attach(float(result) if np.ndim(result) == 0 else result, "kg/kg")


def wet_bulb_temperature(pressure, temperature, dewpoint):
    """Wet-bulb temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (degC)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"), _strip(dewpoint, "degC"))
    if is_arr:
        result = np.asarray(_calc.wet_bulb_temperature_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.wet_bulb_temperature(vals[0], vals[1], vals[2])
    return result * units.degC


def _drop_nan_profiles(*arrays):
    mask = np.ones(len(np.asarray(arrays[0], dtype=np.float64)), dtype=bool)
    cleaned = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=np.float64).ravel()
        mask &= np.isfinite(arr_np)
        cleaned.append(arr_np)
    return tuple(arr[mask] for arr in cleaned)


def _interp_profile_pressure(pressure_hpa, values, target_pressure_hpa):
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    v_arr = np.asarray(values, dtype=np.float64).ravel()
    if p_arr.size != v_arr.size:
        raise ValueError("pressure and values must have matching sizes")
    if p_arr[0] > p_arr[-1]:
        p_arr = p_arr[::-1]
        v_arr = v_arr[::-1]
    return float(np.interp(float(target_pressure_hpa), p_arr, v_arr))


def _log_pressure_intersections(pressure_hpa, y1, y2, *, direction):
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    y1_arr = np.asarray(y1, dtype=np.float64).ravel()
    y2_arr = np.asarray(y2, dtype=np.float64).ravel()
    diff = y1_arr - y2_arr
    log_p = np.log(p_arr)
    tol = 1e-9
    x_out = []
    y_out = []

    for i in range(len(p_arr) - 1):
        d0 = diff[i]
        d1 = diff[i + 1]
        if not np.isfinite(d0) or not np.isfinite(d1):
            continue

        if direction == "increasing":
            crossing = (d0 <= 0 and d1 > 0) or (d0 < 0 and d1 >= 0)
        elif direction == "decreasing":
            crossing = (d0 >= 0 and d1 < 0) or (d0 > 0 and d1 <= 0)
        else:
            raise ValueError("direction must be 'increasing' or 'decreasing'")

        if not crossing and abs(d0) <= tol:
            if direction == "increasing" and d1 > 0:
                crossing = True
            elif direction == "decreasing" and d1 < 0:
                crossing = True

        if not crossing:
            continue

        if abs(d1 - d0) <= tol:
            frac = 0.0
        else:
            frac = -d0 / (d1 - d0)
        frac = float(np.clip(frac, 0.0, 1.0))

        log_px = log_p[i] + frac * (log_p[i + 1] - log_p[i])
        x_val = float(np.exp(log_px))
        y_val = float(y1_arr[i] + frac * (y1_arr[i + 1] - y1_arr[i]))

        if x_out and abs(x_val - x_out[-1]) <= 1e-6:
            continue

        x_out.append(x_val)
        y_out.append(y_val)

    return np.asarray(x_out, dtype=np.float64), np.asarray(y_out, dtype=np.float64)


def _find_log_pressure_intersections_native(pressure_hpa, profile_a, profile_b):
    """Find all intersections between two profiles in log-pressure space."""
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    a_arr = np.asarray(profile_a, dtype=np.float64).ravel()
    b_arr = np.asarray(profile_b, dtype=np.float64).ravel()
    log_p = np.log(p_arr)
    tol = 1e-12

    x_out = []
    y_out = []
    for idx in range(len(p_arr) - 1):
        p0 = p_arr[idx]
        p1 = p_arr[idx + 1]
        d0 = a_arr[idx] - b_arr[idx]
        d1 = a_arr[idx + 1] - b_arr[idx + 1]
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(d0) and np.isfinite(d1)):
            continue
        if abs(d0) <= tol:
            x_out.append(float(p0))
            y_out.append(float(b_arr[idx]))
            continue
        if d0 * d1 > 0:
            continue

        log_px = log_p[idx] - d0 * (log_p[idx + 1] - log_p[idx]) / (d1 - d0)
        frac = (log_px - log_p[idx]) / (log_p[idx + 1] - log_p[idx])
        x_out.append(float(np.exp(log_px)))
        y_out.append(float(b_arr[idx] + frac * (b_arr[idx + 1] - b_arr[idx])))

    return np.asarray(x_out, dtype=np.float64), np.asarray(y_out, dtype=np.float64)


def _select_profile_intersection(pressures_hpa, temperatures_c, which):
    p_arr = np.asarray(pressures_hpa, dtype=np.float64).ravel()
    t_arr = np.asarray(temperatures_c, dtype=np.float64).ravel()

    if which == "all":
        return p_arr, t_arr
    if p_arr.size == 0:
        return np.nan, np.nan
    if which == "bottom":
        idx = 0
    elif which == "top":
        idx = -1
    else:
        raise ValueError('Invalid option for "which". Valid options are "top", "bottom", '
                         '"wide", "most_cape", and "all".')
    return float(p_arr[idx]), float(t_arr[idx])


def _multiple_el_lfc_options_native(intersect_pressures_hpa, intersect_temperatures_c, valid_mask,
                                    which, pressure_hpa, parcel_temperature_c, temperature_c,
                                    dewpoint_c, intersect_type):
    """Choose which EL/LFC to return without delegating to MetPy."""
    p_list = np.asarray(intersect_pressures_hpa, dtype=np.float64).ravel()[valid_mask]
    t_list = np.asarray(intersect_temperatures_c, dtype=np.float64).ravel()[valid_mask]

    if which in {"top", "bottom", "all"}:
        return _select_profile_intersection(p_list, t_list, which)

    if p_list.size == 0:
        return np.nan, np.nan

    if which == "wide":
        if intersect_type == "LFC":
            other_p_list, _ = _log_pressure_intersections(
                pressure_hpa[1:],
                parcel_temperature_c[1:],
                temperature_c[1:],
                direction="decreasing",
            )
            diffs = [lfc_p - el_p for lfc_p, el_p in zip(p_list, other_p_list, strict=False)]
        else:
            other_p_list, _ = _log_pressure_intersections(
                pressure_hpa,
                parcel_temperature_c,
                temperature_c,
                direction="increasing",
            )
            diffs = [lfc_p - el_p for lfc_p, el_p in zip(other_p_list, p_list, strict=False)]
        if not diffs:
            return _select_profile_intersection(p_list, t_list, "top")
        idx = int(np.nanargmax(np.asarray(diffs, dtype=np.float64)))
        return float(p_list[idx]), float(t_list[idx])

    if which == "most_cape":
        cape_pairs = []
        for which_lfc in ("top", "bottom"):
            for which_el in ("top", "bottom"):
                cape, _ = _cape_cin_profile_native(
                    pressure_hpa,
                    temperature_c,
                    dewpoint_c,
                    parcel_temperature_c,
                    which_lfc,
                    which_el,
                )
                cape_pairs.append((float(cape), which_lfc, which_el))
        _, lfc_choice, el_choice = max(cape_pairs, key=lambda item: item[0])
        choice = lfc_choice if intersect_type == "LFC" else el_choice
        return _select_profile_intersection(p_list, t_list, choice)

    raise ValueError('Invalid option for "which". Valid options are "top", "bottom", '
                     '"wide", "most_cape", and "all".')


def _insert_lcl_level(pressure, values, lcl_pressure):
    """Insert the LCL pressure into a profile using linear pressure interpolation."""
    pressure_vals = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    lcl_val = float(_strip(lcl_pressure, "hPa"))
    if hasattr(values, "units"):
        value_unit = values.units
        value_vals = np.asarray(values.magnitude, dtype=np.float64)
    else:
        value_unit = None
        value_vals = np.asarray(values, dtype=np.float64)

    interp_val = np.interp(lcl_val, pressure_vals[::-1], value_vals[::-1])
    loc = pressure_vals.size - np.searchsorted(pressure_vals[::-1], lcl_val)
    inserted = np.insert(value_vals, loc, interp_val)
    if value_unit is not None:
        return inserted * value_unit
    return inserted


def _parcel_profile_helper_native(pressure, temperature, dewpoint):
    """Native equivalent of MetPy's parcel-profile helper."""
    pressure_q = pressure if hasattr(pressure, "units") else np.asarray(pressure, dtype=np.float64) * units.hPa
    temperature_q = temperature if hasattr(temperature, "units") else np.asarray(temperature, dtype=np.float64) * units.degC
    dewpoint_q = dewpoint if hasattr(dewpoint, "units") else np.asarray(dewpoint, dtype=np.float64) * units.degC

    press_lcl, temp_lcl = lcl(pressure_q[0], temperature_q, dewpoint_q)
    press_lcl = press_lcl.to(pressure_q.units)

    press_lower_vals = np.concatenate((
        pressure_q[pressure_q >= press_lcl].to(pressure_q.units).magnitude,
        [press_lcl.to(pressure_q.units).magnitude],
    ))
    press_lower = press_lower_vals * pressure_q.units
    temp_lower = dry_lapse(press_lower, temperature_q).to(temperature_q.units)

    if np.nanmin(pressure_q.to(pressure_q.units).magnitude) >= press_lcl.to(pressure_q.units).magnitude:
        return (
            press_lower[:-1],
            press_lcl,
            units.Quantity(np.array([]), press_lower.units),
            temp_lower[:-1],
            temp_lcl.to(temperature_q.units),
            units.Quantity(np.array([]), temp_lower.units),
        )

    press_upper_vals = np.concatenate((
        [press_lcl.to(pressure_q.units).magnitude],
        pressure_q[pressure_q < press_lcl].to(pressure_q.units).magnitude,
    ))
    press_upper = press_upper_vals * pressure_q.units
    unique_vals, indices = np.unique(press_upper.magnitude, return_inverse=True)
    unique_press = unique_vals * press_upper.units
    temp_upper = moist_lapse(unique_press[::-1], temp_lower[-1]).to(temp_lower.units)
    temp_upper = temp_upper[::-1][indices]

    return (
        press_lower[:-1],
        press_lcl,
        press_upper[1:],
        temp_lower[:-1],
        temp_lcl.to(temperature_q.units),
        temp_upper[1:],
    )


def _find_append_zero_crossings_native(pressure, profile_delta):
    """Append log-pressure zero crossings and return sorted unique arrays."""
    if hasattr(profile_delta, "units"):
        delta_unit = profile_delta.units
        profile_arr = np.asarray(profile_delta.to(delta_unit).magnitude, dtype=np.float64)
    else:
        delta_unit = None
        profile_arr = np.asarray(profile_delta, dtype=np.float64)
    pressure_arr = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    zero_profile = np.zeros_like(profile_arr)

    crossings_p, crossings_y = _find_log_pressure_intersections_native(
        pressure_arr[1:],
        profile_arr[1:],
        zero_profile[1:],
    )

    pressure_vals = np.concatenate((
        pressure_arr,
        crossings_p,
    ))
    profile_vals = np.concatenate((profile_arr, crossings_y))

    sort_idx = np.argsort(pressure_vals)
    pressure_vals = pressure_vals[sort_idx]
    profile_vals = profile_vals[sort_idx]

    keep_idx = np.ediff1d(pressure_vals, to_end=[1.0]) > 1e-6
    pressure_vals = pressure_vals[keep_idx]
    profile_vals = profile_vals[keep_idx]

    if delta_unit is not None:
        return pressure_vals * _UNIT_HPA, profile_vals * delta_unit
    return pressure_vals, profile_vals


def _lcl_native_hpa_degc(pressure_hpa, temperature_c, dewpoint_c):
    return _calc.lcl(float(pressure_hpa), float(temperature_c), float(dewpoint_c))


def _saturation_mixing_ratio_native(pressure_hpa, temperature_c):
    p_arr, t_arr = np.broadcast_arrays(
        np.asarray(pressure_hpa, dtype=np.float64),
        np.asarray(temperature_c, dtype=np.float64),
    )
    if p_arr.ndim == 0:
        return _calc.saturation_mixing_ratio(float(p_arr), float(t_arr)) / 1000.0
    result = _calc.saturation_mixing_ratio_array(
        np.ascontiguousarray(p_arr.ravel()),
        np.ascontiguousarray(t_arr.ravel()),
    )
    return np.asarray(result, dtype=np.float64).reshape(p_arr.shape) / 1000.0


def _virtual_temperature_from_mixing_ratio_native(temperature_c, mixing_ratio_kgkg, molecular_weight_ratio=0.6219569100577033):
    t_arr = np.asarray(temperature_c, dtype=np.float64)
    w_arr = np.asarray(mixing_ratio_kgkg, dtype=np.float64)
    t_k = t_arr + 273.15
    tv_k = t_k * (1.0 + w_arr / molecular_weight_ratio) / (1.0 + w_arr)
    return tv_k - 273.15


def _pressure_quantity_from_hpa(values_hpa, target_unit):
    values = np.asarray(values_hpa, dtype=np.float64)
    target_name = str(target_unit)
    if target_name == "hectopascal":
        return values * _UNIT_HPA
    if target_name == "pascal":
        return (values * 100.0) * _UNIT_PA
    return (values * _UNIT_HPA).to(target_unit)


def _temperature_quantity_from_degc(values_c, target_unit):
    values = np.asarray(values_c, dtype=np.float64)
    target_name = str(target_unit)
    if target_name == "degree_Celsius":
        return values * _UNIT_DEGC
    if target_name == "kelvin":
        return (values + 273.15) * _UNIT_KELVIN
    return (values * _UNIT_DEGC).to(target_unit)


def _lfc_native_arrays(pressure_hpa, temperature_c, dewpoint_c, parcel_temperature_c, dewpoint_start_c, which):
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    t_arr = np.asarray(temperature_c, dtype=np.float64).ravel()
    td_arr = np.asarray(dewpoint_c, dtype=np.float64).ravel()
    parcel_arr = np.asarray(parcel_temperature_c, dtype=np.float64).ravel()

    if np.isclose(parcel_arr[0], t_arr[0], atol=1e-9, rtol=1e-9):
        x, y = _log_pressure_intersections(p_arr[1:], parcel_arr[1:], t_arr[1:], direction="increasing")
    else:
        x, y = _log_pressure_intersections(p_arr, parcel_arr, t_arr, direction="increasing")

    lcl_p, lcl_t = _lcl_native_hpa_degc(p_arr[0], parcel_arr[0], dewpoint_start_c)

    if x.size == 0:
        mask = p_arr < lcl_p
        if np.all(parcel_arr[mask] <= t_arr[mask] + 1e-9):
            return np.nan, np.nan
        return lcl_p, lcl_t

    valid = x < lcl_p
    if not np.any(valid):
        el_x, _ = _log_pressure_intersections(
            p_arr[1:],
            parcel_arr[1:],
            t_arr[1:],
            direction="decreasing",
        )
        if el_x.size and np.min(el_x) > lcl_p:
            return np.nan, np.nan
        return lcl_p, lcl_t

    return _multiple_el_lfc_options_native(
        x,
        y,
        valid,
        which,
        p_arr,
        parcel_arr,
        t_arr,
        td_arr,
        "LFC",
    )


def _el_native_arrays(pressure_hpa, temperature_c, dewpoint_c, parcel_temperature_c, which):
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    t_arr = np.asarray(temperature_c, dtype=np.float64).ravel()
    td_arr = np.asarray(dewpoint_c, dtype=np.float64).ravel()
    parcel_arr = np.asarray(parcel_temperature_c, dtype=np.float64).ravel()

    if parcel_arr[-1] > t_arr[-1]:
        return np.nan, np.nan

    x, y = _log_pressure_intersections(p_arr[1:], parcel_arr[1:], t_arr[1:], direction="decreasing")
    lcl_p, _ = _lcl_native_hpa_degc(p_arr[0], t_arr[0], td_arr[0])
    valid = x < lcl_p
    if np.any(valid):
        return _multiple_el_lfc_options_native(
            x,
            y,
            valid,
            which,
            p_arr,
            parcel_arr,
            t_arr,
            td_arr,
            "EL",
        )
    return np.nan, np.nan


def _cape_cin_profile_native(pressure_hpa, temperature_c, dewpoint_c, parcel_temperature_c, which_lfc, which_el):
    p_arr = np.asarray(pressure_hpa, dtype=np.float64).ravel()
    t_arr = np.asarray(temperature_c, dtype=np.float64).ravel()
    td_arr = np.asarray(dewpoint_c, dtype=np.float64).ravel()
    parcel_arr = np.asarray(parcel_temperature_c, dtype=np.float64).ravel()

    lcl_p, _ = _lcl_native_hpa_degc(p_arr[0], t_arr[0], td_arr[0])
    below_lcl = p_arr > lcl_p
    parcel_mixing_ratio = np.where(
        below_lcl,
        _saturation_mixing_ratio_native(p_arr[0], td_arr[0]),
        _saturation_mixing_ratio_native(p_arr, parcel_arr),
    )
    env_mixing_ratio = _saturation_mixing_ratio_native(p_arr, td_arr)
    env_virtual_temperature = _virtual_temperature_from_mixing_ratio_native(t_arr, env_mixing_ratio)
    parcel_virtual_temperature = _virtual_temperature_from_mixing_ratio_native(parcel_arr, parcel_mixing_ratio)

    lfc_pressure_hpa, _ = _lfc_native_arrays(
        p_arr,
        env_virtual_temperature,
        td_arr,
        parcel_virtual_temperature,
        td_arr[0],
        which_lfc,
    )
    if np.isnan(lfc_pressure_hpa):
        return 0.0, 0.0

    el_pressure_hpa, _ = _el_native_arrays(
        p_arr,
        env_virtual_temperature,
        td_arr,
        parcel_virtual_temperature,
        which_el,
    )
    if np.isnan(el_pressure_hpa):
        el_pressure_hpa = p_arr[-1]

    y = parcel_virtual_temperature - env_virtual_temperature
    x_vals, y_vals = _find_append_zero_crossings_native(p_arr, y)
    trapz = getattr(np, "trapezoid", None) or np.trapz

    cape_mask = (x_vals <= lfc_pressure_hpa + 1e-9) & (x_vals >= el_pressure_hpa - 1e-9)
    cin_mask = x_vals >= lfc_pressure_hpa - 1e-9

    cape = 287.04749097718457 * trapz(y_vals[cape_mask], np.log(x_vals[cape_mask]))
    cin = 287.04749097718457 * trapz(y_vals[cin_mask], np.log(x_vals[cin_mask]))
    if cin > 0.0:
        cin = 0.0
    return float(cape), float(cin)


def lfc(pressure, temperature, dewpoint, parcel_temperature_profile=None,
        dewpoint_start=None, which="top"):
    """Level of Free Convection.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC))
        LFC pressure and parcel temperature at the LFC.
    """
    p_arr = np.asarray(_strip(pressure, "hPa"), dtype=np.float64).ravel()
    t_arr = np.asarray(_strip(temperature, "degC"), dtype=np.float64).ravel()
    td_arr = np.asarray(_strip(dewpoint, "degC"), dtype=np.float64).ravel()

    output_temp_unit = getattr(parcel_temperature_profile, "units", None)

    if parcel_temperature_profile is None:
        p_arr, t_arr, td_arr = _drop_nan_profiles(p_arr, t_arr, td_arr)
        p_q, t_q, td_q, parcel_q = parcel_profile_with_lcl(
            p_arr * _UNIT_HPA,
            t_arr * _UNIT_DEGC,
            td_arr * _UNIT_DEGC,
        )
        p_arr = np.asarray(p_q.to("hPa").magnitude, dtype=np.float64)
        t_arr = np.asarray(t_q.to("degC").magnitude, dtype=np.float64)
        td_arr = np.asarray(td_q.to("degC").magnitude, dtype=np.float64)
        parcel_arr = np.asarray(parcel_q.to("degC").magnitude, dtype=np.float64)
        output_temp_unit = parcel_q.units
    else:
        parcel_arr = np.asarray(_strip(parcel_temperature_profile, "degC"), dtype=np.float64).ravel()
        p_arr, t_arr, td_arr, parcel_arr = _drop_nan_profiles(p_arr, t_arr, td_arr, parcel_arr)

    if output_temp_unit is None:
        output_temp_unit = _UNIT_DEGC

    if dewpoint_start is None:
        dewpoint_start_c = float(td_arr[0])
    else:
        dewpoint_start_c = _as_float(_strip(dewpoint_start, "degC"))

    p_result, t_result = _lfc_native_arrays(
        p_arr,
        t_arr,
        td_arr,
        parcel_arr,
        dewpoint_start_c,
        which,
    )
    return (
        _pressure_quantity_from_hpa(p_result, getattr(pressure, "units", _UNIT_HPA)),
        _temperature_quantity_from_degc(t_result, output_temp_unit),
    )


def el(pressure, temperature, dewpoint, parcel_temperature_profile=None, which="top"):
    """Equilibrium Level.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC))
        EL pressure and parcel temperature at the EL.
    """
    p_arr = np.asarray(_strip(pressure, "hPa"), dtype=np.float64).ravel()
    t_arr = np.asarray(_strip(temperature, "degC"), dtype=np.float64).ravel()
    td_arr = np.asarray(_strip(dewpoint, "degC"), dtype=np.float64).ravel()

    output_temp_unit = getattr(parcel_temperature_profile, "units", None)

    if parcel_temperature_profile is None:
        p_arr, t_arr, td_arr = _drop_nan_profiles(p_arr, t_arr, td_arr)
        p_q, t_q, td_q, parcel_q = parcel_profile_with_lcl(
            p_arr * _UNIT_HPA,
            t_arr * _UNIT_DEGC,
            td_arr * _UNIT_DEGC,
        )
        p_arr = np.asarray(p_q.to("hPa").magnitude, dtype=np.float64)
        t_arr = np.asarray(t_q.to("degC").magnitude, dtype=np.float64)
        td_arr = np.asarray(td_q.to("degC").magnitude, dtype=np.float64)
        parcel_arr = np.asarray(parcel_q.to("degC").magnitude, dtype=np.float64)
        output_temp_unit = parcel_q.units
    else:
        parcel_arr = np.asarray(_strip(parcel_temperature_profile, "degC"), dtype=np.float64).ravel()
        p_arr, t_arr, td_arr, parcel_arr = _drop_nan_profiles(p_arr, t_arr, td_arr, parcel_arr)

    if output_temp_unit is None:
        output_temp_unit = _UNIT_DEGC

    p_result, t_result = _el_native_arrays(
        p_arr,
        t_arr,
        td_arr,
        parcel_arr,
        which,
    )
    return (
        _pressure_quantity_from_hpa(p_result, getattr(pressure, "units", _UNIT_HPA)),
        _temperature_quantity_from_degc(t_result, output_temp_unit),
    )


def lcl(pressure, temperature, dewpoint):
    """Lifting Condensation Level.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC))
        LCL pressure and temperature.
    """
    p_lcl, t_lcl = _lcl_native_hpa_degc(
        _as_float(_strip(pressure, "hPa")),
        _as_float(_strip(temperature, "degC")),
        _as_float(_strip(dewpoint, "degC")),
    )
    pressure_unit = getattr(pressure, "units", _UNIT_HPA)
    temperature_unit = getattr(temperature, "units", _UNIT_DEGC)
    return (
        _pressure_quantity_from_hpa(p_lcl, pressure_unit),
        _temperature_quantity_from_degc(t_lcl, temperature_unit),
    )


def dewpoint_from_relative_humidity(temperature, relative_humidity):
    """Dewpoint from temperature and relative humidity.

    Parameters
    ----------
    temperature : Quantity (temperature)
    relative_humidity : Quantity (dimensionless or %)

    Returns
    -------
    Quantity (degC)
    """
    t_raw = _strip(temperature, "degC")
    rh = _rh_to_percent(relative_humidity)
    vals, shape, is_arr = _prep(t_raw, rh)
    if is_arr:
        result = np.asarray(_calc.dewpoint_from_rh_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.dewpoint_from_relative_humidity(vals[0], vals[1])
    return _attach(result, "degC")


def relative_humidity_from_dewpoint(temperature, dewpoint, phase="liquid"):
    """Relative humidity from temperature and dewpoint.

    Parameters
    ----------
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)
    phase : str
        ``"liquid"`` (default), ``"ice"``, or ``"auto"``.

    Returns
    -------
    Quantity (dimensionless, 0-1)
    """
    phase = _normalize_phase(phase)
    t_raw = _strip(temperature, "degC")
    td_raw = _strip(dewpoint, "degC")
    if phase == "liquid":
        vals, shape, is_arr = _prep(t_raw, td_raw)
        if is_arr:
            result = np.asarray(_calc.rh_from_dewpoint_array(vals[0], vals[1])).reshape(shape) / 100.0
        else:
            result = _calc.relative_humidity_from_dewpoint(vals[0], vals[1]) / 100.0
        return _attach(result, "")
    # e(Td) / es(T) with requested phase
    t = _as_float(t_raw)
    td = _as_float(td_raw)
    e_td = _svp_with_phase(td, "liquid")  # actual vapor pressure always liquid
    es_t = _svp_with_phase(t, phase)
    return _attach(e_td / es_t, "")


def virtual_temperature(temperature, pressure_or_mixing_ratio, dewpoint=None,
                        molecular_weight_ratio=0.6219569100577033):
    """Virtual temperature.

    Can be called as:
    - ``virtual_temperature(T, mixing_ratio)`` (MetPy-compatible, dimensionless)
    - ``virtual_temperature(T, pressure, dewpoint)`` (Rust-native path)

    Parameters
    ----------
    temperature : Quantity (temperature)
    pressure_or_mixing_ratio : Quantity (pressure) or Quantity (dimensionless)
    dewpoint : Quantity (temperature), optional

    Returns
    -------
    Quantity (degC)
    """
    if dewpoint is None:
        # MetPy-compatible path: T * (1 + w/eps) / (1 + w)
        t = np.asarray(_strip(temperature, "degC"), dtype=np.float64)
        w = np.asarray(_strip(pressure_or_mixing_ratio, "kg/kg") if hasattr(pressure_or_mixing_ratio, "magnitude") else pressure_or_mixing_ratio, dtype=np.float64)
        eps = molecular_weight_ratio
        t_k = t + 273.15
        tv_k = t_k * (1.0 + w / eps) / (1.0 + w)
        result = tv_k - 273.15
        return _attach(float(result) if np.ndim(result) == 0 else result, "degC")
    vals, shape, is_arr = _prep(
        _strip(temperature, "degC"),
        _strip(pressure_or_mixing_ratio, "hPa"),
        _strip(dewpoint, "degC"),
    )
    if is_arr:
        result = np.asarray(_calc.virtual_temp_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.virtual_temperature(vals[0], vals[1], vals[2])
    return _attach(result, "degC")


def virtual_temperature_from_dewpoint(pressure, temperature, dewpoint,
                                      molecular_weight_ratio=None, phase=None):
    """Virtual temperature from pressure, temperature, and dewpoint.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)
    molecular_weight_ratio : float, optional
        Accepted for MetPy compatibility; ignored (uses standard Rd/Rv).
    phase : str, optional
        Accepted for MetPy compatibility; ignored.

    Returns
    -------
    Quantity (degC)
    """
    vals, shape, is_arr = _prep(
        _strip(temperature, "degC"),
        _strip(dewpoint, "degC"),
        _strip(pressure, "hPa"),
    )
    if is_arr:
        result = np.asarray(_calc.virtual_temperature_from_dewpoint_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.virtual_temperature_from_dewpoint(vals[0], vals[1], vals[2])
    return result * units.degC


def cape_cin(pressure, temperature, dewpoint, parcel_profile_or_height=None,
             *args, parcel_profile=None, which_lfc="bottom", which_el="top",
             parcel_type="sb", ml_depth=100.0, mu_depth=300.0, top_m=None, **kwargs):
    """CAPE and CIN for a sounding.

    Parameters
    ----------
    pressure : array Quantity (pressure)
        Pressure profile (surface first).
    temperature : array Quantity (temperature)
        Temperature profile.
    dewpoint : array Quantity (temperature)
        Dewpoint profile.
    height : array Quantity (length)
        Height AGL profile.
    psfc : Quantity (pressure)
        Surface pressure.
    t2m : Quantity (temperature)
        2-m temperature.
    td2m : Quantity (temperature)
        2-m dewpoint.
    parcel_type : str
        "sb", "ml", or "mu".
    ml_depth : float
        Mixed-layer depth (hPa).
    mu_depth : float
        Most-unstable search depth (hPa).
    top_m : float, optional
        Height cap for integration (m AGL).

    Returns
    -------
    tuple of (Quantity (J/kg), Quantity (J/kg), Quantity (m), Quantity (m))
        CAPE, CIN, LCL height AGL, LFC height AGL.
    """
    psfc = kwargs.pop("psfc", None)
    t2m = kwargs.pop("t2m", None)
    td2m = kwargs.pop("td2m", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    if parcel_profile is not None:
        parcel_profile_or_height = parcel_profile

    if len(args) >= 3:
        psfc, t2m, td2m = args[:3]

    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))

    # Detect whether the 4th arg is height (length units) or parcel profile (temperature units).
    # MetPy's cape_cin signature is cape_cin(p, T, Td, parcel_profile).
    fourth = parcel_profile_or_height
    metpy_profile_form = fourth is not None and _is_temperature_like(fourth)
    if metpy_profile_form:
        p_profile = np.asarray(_strip(pressure, "hPa"), dtype=np.float64).ravel()
        t_profile = np.asarray(_strip(temperature, "degC"), dtype=np.float64).ravel()
        td_profile = np.asarray(_strip(dewpoint, "degC"), dtype=np.float64).ravel()
        parcel_profile_c = np.asarray(_strip(fourth, "degC"), dtype=np.float64).ravel()
        p_profile, t_profile, td_profile, parcel_profile_c = _drop_nan_profiles(
            p_profile,
            t_profile,
            td_profile,
            parcel_profile_c,
        )
        cape_val, cin_val = _cape_cin_profile_native(
            p_profile,
            t_profile,
            td_profile,
            parcel_profile_c,
            which_lfc,
            which_el,
        )
        return (cape_val * _UNIT_JPKG, cin_val * _UNIT_JPKG)

    elif fourth is not None:
        h = _as_1d(_strip(fourth, "m"))
    else:
        # No 4th arg — derive height from standard atmosphere
        h_vals = np.array([_calc.pressure_to_height_std(float(pi)) for pi in p])
        h = _as_1d(h_vals - h_vals[0])
        if psfc is None:
            psfc = pressure[0] if hasattr(pressure, '__getitem__') else pressure
        if t2m is None:
            t2m = temperature[0] if hasattr(temperature, '__getitem__') else temperature
        if td2m is None:
            td2m = dewpoint[0] if hasattr(dewpoint, '__getitem__') else dewpoint

    ps = _as_float(_strip(psfc, "hPa"))
    t2 = _as_float(_strip(t2m, "degC"))
    td2 = _as_float(_strip(td2m, "degC"))
    cape_val, cin_val, h_lcl, h_lfc = _calc.cape_cin(
        p, t, td, h, ps, t2, td2, parcel_type,
        float(ml_depth), float(mu_depth), top_m,
    )
    return (
        cape_val * _UNIT_JPKG,
        cin_val * _UNIT_JPKG,
        h_lcl * units.m,
        h_lfc * units.m,
    )


def mixing_ratio(partial_press_or_pressure, total_press_or_temperature,
                 molecular_weight_ratio=0.6219569100577033):
    """Mixing ratio.

    Can be called as:
    - ``mixing_ratio(pressure, temperature)`` -- from pressure & temperature
    - ``mixing_ratio(partial_pressure, total_pressure)`` -- from vapor/total pressure

    Parameters
    ----------
    partial_press_or_pressure : Quantity (pressure)
    total_press_or_temperature : Quantity (temperature or pressure)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    if _is_temperature_like(total_press_or_temperature):
        vals, shape, is_arr = _prep(
            _strip(partial_press_or_pressure, "hPa"),
            _strip(total_press_or_temperature, "degC"),
        )
        if is_arr:
            result = np.asarray(_calc.mixing_ratio_array(vals[0], vals[1])).reshape(shape) / 1000.0
        else:
            result = _calc.mixing_ratio(vals[0], vals[1]) / 1000.0
        return _attach(result, "kg/kg")
    # partial_pressure / total_pressure path: w = eps * e / (p - e)
    e = np.asarray(_strip(partial_press_or_pressure, "Pa"), dtype=np.float64)
    p = np.asarray(_strip(total_press_or_temperature, "Pa"), dtype=np.float64)
    eps = molecular_weight_ratio
    return _attach(eps * e / (p - e), "kg/kg")


def showalter_index(pressure, temperature, dewpoint):
    """Showalter Index.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    Quantity (delta_degC)
    """
    t850 = _interp_profile_level(pressure, temperature, 850.0, "degC") * units.degC
    td850 = _interp_profile_level(pressure, dewpoint, 850.0, "degC") * units.degC
    t500 = np.asarray([_interp_profile_level(pressure, temperature, 500.0, "degC")]) * units.degC
    prof = parcel_profile(np.asarray([850.0, 500.0]) * units.hPa, t850, td850)
    return t500 - prof[-1]


def k_index(*args, vertical_dim=0):
    """K-Index.

    All temperatures in Celsius (or Quantity).

    Returns
    -------
    Quantity (delta_degC)
    """
    if len(args) == 3:
        pressure, temperature, dewpoint = args
        t850 = _interp_profile_level(pressure, temperature, 850.0, "degC")
        td850 = _interp_profile_level(pressure, dewpoint, 850.0, "degC")
        t700 = _interp_profile_level(pressure, temperature, 700.0, "degC")
        td700 = _interp_profile_level(pressure, dewpoint, 700.0, "degC")
        t500 = _interp_profile_level(pressure, temperature, 500.0, "degC")
    elif len(args) == 5:
        t850, td850, t700, td700, t500 = args
    else:
        raise TypeError("k_index expects either (pressure, temperature, dewpoint) or 5 scalar level values")
    return _attach(
        _calc.k_index(
            _as_float(_strip(t850, "degC")),
            _as_float(_strip(t700, "degC")),
            _as_float(_strip(t500, "degC")),
            _as_float(_strip(td850, "degC")),
            _as_float(_strip(td700, "degC")),
        ),
        "delta_degC",
    )


def total_totals(*args, vertical_dim=0):
    """Total Totals Index.

    Returns
    -------
    Quantity (delta_degC)
    """
    if len(args) == 3 and np.asarray(_strip(args[0], "hPa")).ndim > 0:
        pressure, temperature, dewpoint = args
        t850 = _interp_profile_level(pressure, temperature, 850.0, "degC")
        td850 = _interp_profile_level(pressure, dewpoint, 850.0, "degC")
        t500 = _interp_profile_level(pressure, temperature, 500.0, "degC")
    elif len(args) == 3:
        t850, td850, t500 = args
    else:
        raise TypeError("total_totals expects either (pressure, temperature, dewpoint) or 3 scalar level values")
    return _attach(
        _calc.total_totals(
            _as_float(_strip(t850, "degC")),
            _as_float(_strip(t500, "degC")),
            _as_float(_strip(td850, "degC")),
        ),
        "delta_degC",
    )


def downdraft_cape(pressure, temperature, dewpoint):
    """Downdraft CAPE (DCAPE).

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    pressure_q = pressure if hasattr(pressure, "to") else np.asarray(pressure, dtype=np.float64) * units.hPa
    temperature_q = temperature if hasattr(temperature, "to") else np.asarray(temperature, dtype=np.float64) * units.degC
    dewpoint_q = dewpoint if hasattr(dewpoint, "to") else np.asarray(dewpoint, dtype=np.float64) * units.degC

    p_arr = np.asarray(pressure_q.to("hPa").magnitude, dtype=np.float64).ravel()
    t_arr = np.asarray(temperature_q.to("degC").magnitude, dtype=np.float64).ravel()
    td_arr = np.asarray(dewpoint_q.to("degC").magnitude, dtype=np.float64).ravel()
    p_arr, t_arr, td_arr = _drop_nan_profiles(p_arr, t_arr, td_arr)
    pressure_q = p_arr * units.hPa
    temperature_q = t_arr * units.degC
    dewpoint_q = td_arr * units.degC

    p_layer, t_layer, td_layer = get_layer(
        pressure_q,
        temperature_q,
        dewpoint_q,
        bottom=700 * units.hPa,
        depth=200 * units.hPa,
        interpolate=True,
    )
    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    min_idx = int(np.nanargmin(theta_e.magnitude))
    parcel_start_p = p_layer[min_idx]
    parcel_lcl_p, parcel_lcl_t = lcl(parcel_start_p, t_layer[min_idx], td_layer[min_idx])
    parcel_start_wb = moist_lapse(
        parcel_start_p,
        parcel_lcl_t,
        reference_pressure=parcel_lcl_p,
    )
    if np.ndim(getattr(parcel_start_wb, "magnitude", parcel_start_wb)) > 0:
        parcel_start_wb = parcel_start_wb[0]

    mask = pressure_q >= parcel_start_p
    down_pressure = pressure_q[mask].to("hPa")
    down_parcel_trace = moist_lapse(
        down_pressure,
        parcel_start_wb,
        reference_pressure=parcel_start_p,
    ).to("degC")

    parcel_virt_temp = virtual_temperature_from_dewpoint(
        down_pressure,
        down_parcel_trace,
        down_parcel_trace,
    ).to("K")
    env_virt_temp = virtual_temperature_from_dewpoint(
        down_pressure,
        temperature_q[mask],
        dewpoint_q[mask],
    ).to("K")

    diff = (env_virt_temp - parcel_virt_temp).to("K").magnitude
    lnp = np.log(down_pressure.to("hPa").magnitude)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    dcape = -(287.04749097718457 * trapz(diff, lnp)) * units("J/kg")
    return dcape, down_pressure, down_parcel_trace


def cross_totals(*args, vertical_dim=0):
    """Cross Totals: Td850 - T500.

    Returns
    -------
    Quantity (delta_degC)
    """
    if len(args) == 3:
        pressure, temperature, dewpoint = args
        td850 = _interp_profile_level(pressure, dewpoint, 850.0, "degC")
        t500 = _interp_profile_level(pressure, temperature, 500.0, "degC")
    elif len(args) == 2:
        td850, t500 = args
    else:
        raise TypeError("cross_totals expects either (pressure, temperature, dewpoint) or (td850, t500)")
    return _attach(
        _calc.cross_totals(
            _as_float(_strip(td850, "degC")),
            _as_float(_strip(t500, "degC")),
        ),
        "delta_degC",
    )


def vertical_totals(*args, vertical_dim=0):
    """Vertical Totals: T850 - T500.

    Returns
    -------
    Quantity (delta_degC)
    """
    if len(args) == 2 and np.asarray(_strip(args[0], "hPa")).ndim > 0:
        pressure, temperature = args
        t850 = _interp_profile_level(pressure, temperature, 850.0, "degC")
        t500 = _interp_profile_level(pressure, temperature, 500.0, "degC")
    elif len(args) == 2:
        t850, t500 = args
    else:
        raise TypeError("vertical_totals expects either (pressure, temperature) or (t850, t500)")
    return _attach(
        _calc.vertical_totals(
            _as_float(_strip(t850, "degC")),
            _as_float(_strip(t500, "degC")),
        ),
        "delta_degC",
    )


def sweat_index(*args, vertical_dim=0):
    """SWEAT Index.

    Parameters
    ----------
    pressure, temperature, dewpoint : profile Quantities
    speed, direction : profile Quantities
    vertical_dim : int, optional
        Axis corresponding to the vertical dimension for profile-form calls.

    Or legacy scalar-level form:
    t850, td850, t500, dd850, dd500, ff850, ff500

    Returns
    -------
    Quantity (dimensionless)
    """
    if len(args) == 5:
        pressure, temperature, dewpoint, speed, direction = args
        td850 = np.asarray([_interp_profile_level(pressure, dewpoint, 850.0, "degC")], dtype=np.float64)
        tt_mag = np.atleast_1d(np.asarray(total_totals(pressure, temperature, dewpoint).to("delta_degC").magnitude, dtype=np.float64))
        f850 = np.asarray([_interp_profile_level(pressure, speed, 850.0, "knot")], dtype=np.float64)
        f500 = np.asarray([_interp_profile_level(pressure, speed, 500.0, "knot")], dtype=np.float64)
        dd850 = np.asarray([_interp_profile_level(pressure, direction, 850.0, "degree")], dtype=np.float64)
        dd500 = np.asarray([_interp_profile_level(pressure, direction, 500.0, "degree")], dtype=np.float64)
    elif len(args) == 7:
        t850, td850, t500, dd850, dd500, ff850, ff500 = args
        td850 = np.atleast_1d(np.asarray(_strip(td850, "degC"), dtype=np.float64))
        tt_mag = np.atleast_1d(np.asarray(total_totals(t850, td850, t500).to("delta_degC").magnitude, dtype=np.float64))
        f850 = np.atleast_1d(np.asarray(_strip(ff850, "knot"), dtype=np.float64))
        f500 = np.atleast_1d(np.asarray(_strip(ff500, "knot"), dtype=np.float64))
        dd850 = np.atleast_1d(np.asarray(_strip(dd850, "degree"), dtype=np.float64))
        dd500 = np.atleast_1d(np.asarray(_strip(dd500, "degree"), dtype=np.float64))
    else:
        raise TypeError(
            "sweat_index expects either (pressure, temperature, dewpoint, speed, direction) "
            "or (t850, td850, t500, dd850, dd500, ff850, ff500)"
        )

    first_term = 12.0 * np.clip(np.asarray(td850, dtype=np.float64), 0.0, None)
    second_term = 20.0 * np.clip(tt_mag - 49.0, 0.0, None)
    required = (
        (np.asarray(dd850, dtype=np.float64) >= 130.0)
        & (np.asarray(dd850, dtype=np.float64) <= 250.0)
        & (np.asarray(dd500, dtype=np.float64) >= 210.0)
        & (np.asarray(dd500, dtype=np.float64) <= 310.0)
        & ((np.asarray(dd500, dtype=np.float64) - np.asarray(dd850, dtype=np.float64)) > 0.0)
        & (np.asarray(f850, dtype=np.float64) >= 15.0)
        & (np.asarray(f500, dtype=np.float64) >= 15.0)
    )
    shear_term = 125.0 * (np.sin(np.deg2rad(np.asarray(dd500, dtype=np.float64) - np.asarray(dd850, dtype=np.float64))) + 0.2)
    shear_term = np.asarray(shear_term, dtype=np.float64)
    shear_term = np.where(required, shear_term, 0.0)

    result = np.atleast_1d(
        first_term
        + second_term
        + (2.0 * np.asarray(f850, dtype=np.float64))
        + np.asarray(f500, dtype=np.float64)
        + shear_term
    )
    return _attach(result, "")


def brunt_vaisala_frequency(height, potential_temp, vertical_dim=0):
    """Brunt-Vaisala frequency at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (1/s)
    """
    n_squared = brunt_vaisala_frequency_squared(height, potential_temp, vertical_dim=vertical_dim)
    arr = np.asarray(n_squared.to("1/s**2").magnitude, dtype=np.float64).copy()
    arr[arr < 0.0] = np.nan
    return np.sqrt(arr) * units("1/s")


def brunt_vaisala_period(height, potential_temp, vertical_dim=0):
    """Brunt-Vaisala period at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (s)
    """
    n_squared = brunt_vaisala_frequency_squared(height, potential_temp, vertical_dim=vertical_dim)
    arr = np.asarray(n_squared.to("1/s**2").magnitude, dtype=np.float64).copy()
    arr[arr <= 0.0] = np.nan
    return (2.0 * np.pi / np.sqrt(arr)) * units.s


def brunt_vaisala_frequency_squared(height, potential_temp, vertical_dim=0):
    """Brunt-Vaisala frequency squared (N^2) at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (1/s^2)
    """
    theta = potential_temp.to("K") if hasattr(potential_temp, "to") else np.asarray(potential_temp, dtype=np.float64) * units.K
    dtheta_dz = first_derivative(theta, x=height, axis=vertical_dim)
    return (9.80665 * units("m/s**2") / theta) * dtheta_dz


def precipitable_water(pressure, dewpoint, *, bottom=None, top=None):
    """Precipitable water from pressure and dewpoint profiles.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    dewpoint : array Quantity (temperature)
    bottom: Quantity (pressure), optional
        Bottom of the layer. Defaults to the highest pressure in the profile.
    top: Quantity (pressure), optional
        Top of the layer. Defaults to the lowest pressure in the profile.

    Returns
    -------
    Quantity (mm)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    td = _as_1d(_strip(dewpoint, "degC"))
    mask = ~(np.isnan(p) | np.isnan(td))
    p = p[mask]
    td = td[mask]
    if p.size == 0:
        raise ValueError("precipitable_water requires at least one finite pressure/dewpoint pair")

    sort_inds = np.argsort(p)[::-1]
    p = p[sort_inds]
    td = td[sort_inds]

    min_pressure = float(np.nanmin(p))
    max_pressure = float(np.nanmax(p))

    if top is None:
        top_hpa = min_pressure
    else:
        top_hpa = _as_float(_strip(top, "hPa"))
        if not min_pressure <= top_hpa <= max_pressure:
            raise ValueError(
                f"The pressure and dewpoint profile ranges from {max_pressure} to "
                f"{min_pressure} hPa after removing missing values. {top_hpa} hPa is "
                "outside this range."
            )

    if bottom is None:
        bottom_hpa = max_pressure
    else:
        bottom_hpa = _as_float(_strip(bottom, "hPa"))
        if not min_pressure <= bottom_hpa <= max_pressure:
            raise ValueError(
                f"The pressure and dewpoint profile ranges from {max_pressure} to "
                f"{min_pressure} hPa after removing missing values. {bottom_hpa} hPa is "
                "outside this range."
            )

    pres_layer, dewpoint_layer = get_layer(
        p * units.hPa,
        td * units.degC,
        bottom=bottom_hpa * units.hPa,
        depth=(bottom_hpa - top_hpa) * units.hPa,
        interpolate=True,
    )
    w = mixing_ratio(saturation_vapor_pressure(dewpoint_layer), pres_layer.to("Pa"))
    # Pressure decreases with height, so the integral is negated to yield positive PW.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    pw = -trapz(w.to("kg/kg").magnitude, pres_layer.to("Pa").magnitude) / (9.80665 * 1000.0)
    return (pw * units.m).to("mm")


def parcel_profile_with_lcl(pressure, temperature, dewpoint):
    """Parcel temperature profile with the LCL level inserted.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : Quantity or array Quantity (temperature)
    dewpoint : Quantity or array Quantity (temperature)

    Returns
    -------
    tuple
        Pressure levels, ambient temperature, ambient dewpoint, and parcel
        temperatures on a profile that includes the LCL.
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t_raw = np.asarray(_strip(temperature, "degC"), dtype=np.float64)
    td_raw = np.asarray(_strip(dewpoint, "degC"), dtype=np.float64)

    # Legacy scalar mode retained for downstream callers that only need the parcel trace.
    if t_raw.ndim == 0 and td_raw.ndim == 0:
        t = float(t_raw)
        td = float(td_raw)
        p_out, t_out = _calc.parcel_profile_with_lcl(p, t, td)
        return np.asarray(p_out, dtype=np.float64) * units.hPa, np.asarray(t_out, dtype=np.float64) * units.degC

    pressure_q = pressure if hasattr(pressure, "units") else np.asarray(p, dtype=np.float64) * units.hPa
    temperature_q = temperature if hasattr(temperature, "units") else np.asarray(t_raw, dtype=np.float64) * units.degC
    dewpoint_q = dewpoint if hasattr(dewpoint, "units") else np.asarray(td_raw, dtype=np.float64) * units.degC
    pressure_q, temperature_q, dewpoint_q = _drop_nan_profiles(pressure_q, temperature_q, dewpoint_q)
    if len(pressure_q) == 0:
        raise ValueError("parcel_profile_with_lcl requires at least one finite profile level")

    press_lower, press_lcl, press_upper, temp_lower, temp_lcl, temp_upper = _parcel_profile_helper_native(
        pressure_q,
        temperature_q[0],
        dewpoint_q[0],
    )
    pressure_out = np.concatenate((
        press_lower.to("hPa").magnitude,
        [press_lcl.to("hPa").magnitude],
        press_upper.to("hPa").magnitude,
    )) * units.hPa
    parcel_temperature = np.concatenate((
        temp_lower.to(temperature_q.units).magnitude,
        [temp_lcl.to(temperature_q.units).magnitude],
        temp_upper.to(temperature_q.units).magnitude,
    )) * temperature_q.units
    ambient_temperature = _insert_lcl_level(pressure_q, temperature_q, press_lcl)
    ambient_dewpoint = _insert_lcl_level(pressure_q, dewpoint_q, press_lcl)
    return pressure_out, ambient_temperature, ambient_dewpoint, parcel_temperature


def moist_air_gas_constant(specific_humidity):
    """Gas constant for moist air."""
    q = np.asarray(
        _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity,
        dtype=np.float64,
    )
    result = 287.04749097718457 + q * (461.52311572606084 - 287.04749097718457)
    return _attach(float(result) if np.ndim(result) == 0 else result, "J/(kg*K)")


def moist_air_specific_heat_pressure(specific_humidity):
    """Specific heat at constant pressure for moist air."""
    q = np.asarray(
        _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity,
        dtype=np.float64,
    )
    result = 1004.6662184201462 + q * (1860.078011865639 - 1004.6662184201462)
    return _attach(float(result) if np.ndim(result) == 0 else result, "J/(kg*K)")


def moist_air_poisson_exponent(specific_humidity):
    """Poisson exponent (kappa) for moist air."""
    result = moist_air_gas_constant(specific_humidity) / moist_air_specific_heat_pressure(specific_humidity)
    return result.to("")


def water_latent_heat_vaporization(temperature):
    """Latent heat of vaporization (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    temp_k = np.asarray(
        _strip(temperature, "K") if hasattr(temperature, "to") else np.asarray(temperature, dtype=np.float64) + 273.15,
        dtype=np.float64,
    )
    result = _LV0 - (_CP_L - _CP_V) * (temp_k - _T0)
    return _attach(float(result) if np.ndim(result) == 0 else result, "J/kg")


def water_latent_heat_melting(temperature):
    """Latent heat of melting (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    temp_k = np.asarray(
        _strip(temperature, "K") if hasattr(temperature, "to") else np.asarray(temperature, dtype=np.float64) + 273.15,
        dtype=np.float64,
    )
    result = (_LS0 - _LV0) - (_CP_L - _CP_I) * (temp_k - _T0)
    return _attach(float(result) if np.ndim(result) == 0 else result, "J/kg")


def water_latent_heat_sublimation(temperature):
    """Latent heat of sublimation (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    temp_k = np.asarray(
        _strip(temperature, "K") if hasattr(temperature, "to") else np.asarray(temperature, dtype=np.float64) + 273.15,
        dtype=np.float64,
    )
    result = _LS0 - (_CP_I - _CP_V) * (temp_k - _T0)
    return _attach(float(result) if np.ndim(result) == 0 else result, "J/kg")


def relative_humidity_wet_psychrometric(pressure, dry_bulb_temperature, wet_bulb_temperature,
                                        **kwargs):
    """Relative humidity from dry-bulb, wet-bulb, and pressure.
    """
    if not _is_pressure_like(pressure) and _is_pressure_like(wet_bulb_temperature):
        pressure, dry_bulb_temperature, wet_bulb_temperature = (
            wet_bulb_temperature,
            pressure,
            dry_bulb_temperature,
        )

    psychrometer_coefficient = kwargs.get("psychrometer_coefficient")
    if psychrometer_coefficient is None:
        psychrometer_coefficient = units.Quantity(6.21e-4, "1/K")

    vapor_pressure_wet = (
        saturation_vapor_pressure(wet_bulb_temperature)
        - psychrometer_coefficient
        * pressure.to("Pa")
        * (dry_bulb_temperature - wet_bulb_temperature).to("kelvin")
    )
    return (vapor_pressure_wet / saturation_vapor_pressure(dry_bulb_temperature)).to("")


def _drop_nan_profiles(*profiles):
    """Remove rows where any profile contains NaN values."""
    raw = []
    for profile in profiles:
        values = profile.magnitude if hasattr(profile, "magnitude") else profile
        raw.append(np.asarray(values, dtype=np.float64))
    mask = np.ones(raw[0].shape, dtype=bool)
    for arr in raw:
        mask &= np.isfinite(arr)

    cleaned = []
    for profile, arr in zip(profiles, raw):
        subset = arr[mask]
        cleaned.append(subset * profile.units if hasattr(profile, "units") else subset)
    return tuple(cleaned)


def _interp_profile_to_pressures(target_pressure, pressure, values):
    """Interpolate profile values to target pressures in log-pressure space."""
    target = np.asarray(_strip(target_pressure, "hPa"), dtype=np.float64)
    source_pressure = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    if hasattr(values, "units"):
        unit = values.units
        source_values = np.asarray(values.magnitude, dtype=np.float64)
    else:
        unit = None
        source_values = np.asarray(values, dtype=np.float64)

    order = np.argsort(source_pressure)
    interp = np.interp(np.log(target), np.log(source_pressure[order]), source_values[order])
    if unit is not None:
        return interp * unit
    return interp


def _interp_profile_to_heights(target_height, height, values):
    """Interpolate profile values to target heights."""
    target = np.asarray(_strip(target_height, "m"), dtype=np.float64)
    source_height = np.asarray(_strip(height, "m"), dtype=np.float64)
    if hasattr(values, "units"):
        unit = values.units
        source_values = np.asarray(values.magnitude, dtype=np.float64)
    else:
        unit = None
        source_values = np.asarray(values, dtype=np.float64)

    order = np.argsort(source_height)
    interp = np.interp(target, source_height[order], source_values[order])
    if unit is not None:
        return interp * unit
    return interp


def _find_log_pressure_intersections(pressure, profile_a, profile_b):
    """Find all intersections between two temperature profiles in log-pressure space."""
    pressure_vals = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    b_unit = profile_b.units if hasattr(profile_b, "units") else units.degC
    a_vals = np.asarray(
        profile_a.to(b_unit).magnitude if hasattr(profile_a, "to") else profile_a,
        dtype=np.float64,
    )
    b_vals = np.asarray(
        profile_b.to(b_unit).magnitude if hasattr(profile_b, "to") else profile_b,
        dtype=np.float64,
    )

    x_cross = []
    y_cross = []
    for idx in range(len(pressure_vals) - 1):
        p0 = pressure_vals[idx]
        p1 = pressure_vals[idx + 1]
        d0 = a_vals[idx] - b_vals[idx]
        d1 = a_vals[idx + 1] - b_vals[idx + 1]
        if not np.all(np.isfinite([p0, p1, d0, d1])):
            continue
        if np.isclose(d0, 0.0):
            x_cross.append(p0)
            y_cross.append(b_vals[idx])
            continue
        if d0 * d1 > 0:
            continue

        log_p0 = np.log(p0)
        log_p1 = np.log(p1)
        log_px = log_p0 - d0 * (log_p1 - log_p0) / (d1 - d0)
        frac = (log_px - log_p0) / (log_p1 - log_p0)
        x_cross.append(float(np.exp(log_px)))
        y_cross.append(float(b_vals[idx] + frac * (b_vals[idx + 1] - b_vals[idx])))

    return np.asarray(x_cross) * units.hPa, np.asarray(y_cross) * b_unit


def _parcel_profile_with_environment(pressure, temperature, dewpoint):
    """Build the parcel profile plus environmental profiles with the LCL inserted."""
    lcl_pressure, _ = lcl(pressure[0], temperature[0], dewpoint[0])
    pressure_values = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    parcel_pressure = np.sort(np.append(pressure_values, lcl_pressure.to("hPa").magnitude))[::-1] * units.hPa
    parcel_temperature = parcel_profile(parcel_pressure, temperature[0], dewpoint[0])
    ambient_temperature = _interp_profile_to_pressures(parcel_pressure, pressure, temperature)
    ambient_dewpoint = _interp_profile_to_pressures(parcel_pressure, pressure, dewpoint)
    return parcel_pressure, ambient_temperature, ambient_dewpoint, parcel_temperature


def weighted_continuous_average(pressure, *args, height=None, bottom=None, depth=None):
    """Trapezoidal weighted average over a coordinate.
    """
    if not args:
        raise TypeError("weighted_continuous_average requires at least one value profile")

    pressure_layer, *layers = get_layer(
        pressure,
        *args,
        height=height,
        bottom=bottom,
        depth=depth,
    )
    trapz = getattr(np, "trapezoid", None) or np.trapz
    denominator = pressure_layer[-1].to("Pa").magnitude - pressure_layer[0].to("Pa").magnitude
    results = []
    for layer in layers:
        if hasattr(layer, "units"):
            value = trapz(layer.magnitude, x=pressure_layer.to("Pa").magnitude) / denominator
            results.append(value * layer.units)
        else:
            results.append(trapz(np.asarray(layer, dtype=np.float64),
                                 x=pressure_layer.to("Pa").magnitude) / denominator)
    return results


def get_perturbation(values):
    """Anomaly (perturbation) from the mean.

    Parameters
    ----------
    values : array Quantity or array-like

    Returns
    -------
    array (same units as input, or dimensionless)
    """
    has_units = hasattr(values, "magnitude")
    if has_units:
        u = values.units
        v = _as_1d(values.magnitude)
    else:
        u = None
        v = _as_1d(values)
    result = np.array(_calc.get_perturbation(v))
    if u is not None:
        return result * u
    return result


def add_height_to_pressure(pressure, height):
    """New pressure after ascending/descending by a height increment.
    """
    return height_to_pressure_std(pressure_to_height_std(pressure) + height)


def add_pressure_to_height(height, pressure):
    """New height after a pressure increment.
    """
    return pressure_to_height_std(height_to_pressure_std(height) - pressure)


def thickness_hydrostatic(pressure_or_bottom, temperature_or_top, t_mean=None,
                          mixing_ratio=None,
                          molecular_weight_ratio=0.6219569100577033,
                          bottom=None, depth=None):
    """Hypsometric thickness.

    Supports both the original scalar form ``thickness_hydrostatic(p_bottom, p_top, t_mean)``
    and the MetPy-style profile form ``thickness_hydrostatic(pressure, temperature, ...)``.
    """
    if t_mean is not None:
        result = _vec_call(
            _calc.thickness_hydrostatic,
            _strip(pressure_or_bottom, "hPa"),
            _strip(temperature_or_top, "hPa"),
            _strip(t_mean, "K"),
        )
        return result * units.m

    pressure = pressure_or_bottom
    temperature = temperature_or_top
    if bottom is not None or depth is not None:
        extracted = get_layer(pressure, temperature, *( [mixing_ratio] if mixing_ratio is not None else [] ),
                              bottom=bottom if bottom is not None else pressure[0],
                              depth=depth if depth is not None else (pressure[0] - pressure[-1]),
                              interpolate=True)
        if mixing_ratio is not None:
            pressure, temperature, mixing_ratio = extracted
        else:
            pressure, temperature = extracted

    p = _as_1d(_strip(pressure, "Pa"))
    t = _as_1d(_strip(temperature, "K"))
    if mixing_ratio is not None:
        w = _as_1d(_strip(mixing_ratio, "kg/kg"))
        eps = molecular_weight_ratio
        t = t * (1.0 + w / eps) / (1.0 + w)

    if p[0] < p[-1]:
        p = p[::-1]
        t = t[::-1]

    thickness = -(287.05 / 9.80665) * np.trapezoid(t, np.log(p))
    return thickness * units.m


def vapor_pressure(pressure_or_dewpoint, mixing_ratio=None,
                   molecular_weight_ratio=0.6219569100577033):
    """Vapor pressure from dewpoint temperature.

    Parameters
    ----------
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
    """
    if mixing_ratio is not None:
        p_raw = np.asarray(_strip(pressure_or_dewpoint, "Pa"), dtype=np.float64)
        w_raw = np.asarray(_strip(mixing_ratio, "kg/kg"), dtype=np.float64)
        result = p_raw * w_raw / (molecular_weight_ratio + w_raw)
        return _attach(result, "Pa")
    td_raw = _strip(pressure_or_dewpoint, "degC")
    vals, shape, is_arr = _prep(td_raw)
    if is_arr:
        result = np.asarray(_calc.vapor_pressure_array(vals[0])).reshape(shape) * 100.0
    else:
        result = _calc.vapor_pressure(vals[0]) * 100.0
    return _attach(result, "Pa")


def specific_humidity_from_mixing_ratio(mixing_ratio):
    """Specific humidity from mixing ratio.

    Parameters
    ----------
    mixing_ratio : Quantity (dimensionless, kg/kg)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    vals, shape, is_arr = _prep(_strip(mixing_ratio, "kg/kg"))
    if is_arr:
        result = np.asarray(_calc.specific_humidity_from_mixing_ratio_array(vals[0])).reshape(shape)
    else:
        result = _calc.specific_humidity_from_mixing_ratio(vals[0])
    return result * units("kg/kg")


def thickness_hydrostatic_from_relative_humidity(pressure, temperature,
                                                 relative_humidity, bottom=None,
                                                 depth=None):
    """Hypsometric thickness from pressure, temperature, and relative humidity profiles.
    """
    mixing = mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity)
    return thickness_hydrostatic(
        pressure,
        temperature,
        mixing_ratio=mixing,
        bottom=bottom,
        depth=depth,
    )


def ccl(pressure, temperature, dewpoint, height=None, mixed_layer_depth=None, which="top"):
    """Convective Condensation Level.
    """
    pressure, temperature, dewpoint_profile = _drop_nan_profiles(pressure, temperature, dewpoint)

    if mixed_layer_depth is None:
        vapor_pressure_start = saturation_vapor_pressure(dewpoint_profile[0])
        r_start = mixing_ratio(vapor_pressure_start, pressure[0])
    else:
        vapor_pressure_profile = saturation_vapor_pressure(dewpoint_profile)
        r_profile = mixing_ratio(vapor_pressure_profile, pressure)
        r_start = mixed_layer(
            pressure,
            r_profile,
            height=height,
            depth=mixed_layer_depth,
        )[0]

    rt_profile = globals()["dewpoint"](vapor_pressure(pressure, r_start))
    ccl_pressure, ccl_temperature = _find_log_pressure_intersections(
        pressure,
        rt_profile,
        temperature,
    )
    if ccl_pressure.size == 0:
        raise ValueError("No CCL intersection found in the provided profile.")

    if which == "top":
        ccl_pressure = ccl_pressure[-1]
        ccl_temperature = ccl_temperature[-1]
    elif which == "bottom":
        ccl_pressure = ccl_pressure[0]
        ccl_temperature = ccl_temperature[0]
    elif which != "all":
        raise ValueError('Invalid option for "which". Valid options are "top", "bottom", and "all".')

    convective_temperature = temperature_from_potential_temperature(
        pressure[0],
        potential_temperature(ccl_pressure, ccl_temperature),
    ).to(temperature.units)
    return ccl_pressure.to(pressure.units), ccl_temperature.to(temperature.units), convective_temperature


def lifted_index(pressure, temperature, parcel_profile, vertical_dim=0):
    """Lifted Index.
    """
    target = np.array([500.0], dtype=np.float64)
    pressure_vals = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
    temp_vals = np.asarray(_strip(temperature, "degC"), dtype=np.float64)
    parcel_vals = np.asarray(_strip(parcel_profile, "degC"), dtype=np.float64)

    axis = vertical_dim % pressure_vals.ndim
    pressure_moved = np.moveaxis(pressure_vals, axis, 0)
    temp_moved = np.moveaxis(temp_vals, axis, 0)
    parcel_moved = np.moveaxis(parcel_vals, axis, 0)
    out_shape = (1,) + pressure_moved.shape[1:]
    t500 = np.empty(out_shape, dtype=np.float64)
    tp500 = np.empty(out_shape, dtype=np.float64)

    for idx in np.ndindex(pressure_moved.shape[1:]):
        p_column = pressure_moved[(slice(None),) + idx]
        order = np.argsort(p_column)
        p_sorted = p_column[order]
        t500[(slice(None),) + idx] = np.interp(target, p_sorted, temp_moved[(slice(None),) + idx][order])
        tp500[(slice(None),) + idx] = np.interp(target, p_sorted, parcel_moved[(slice(None),) + idx][order])

    lifted = np.moveaxis(t500 - tp500, 0, axis)
    return _attach(np.asarray(lifted, dtype=np.float64), "delta_degC")


def density(pressure, temperature, mixing_ratio,
            molecular_weight_ratio=0.6219569100577033):
    """Air density from pressure, temperature, and mixing ratio.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    mixing_ratio : Quantity (g/kg or kg/kg)

    Returns
    -------
    Quantity (kg/m^3)
    """
    virtual_temp = virtual_temperature(
        temperature,
        mixing_ratio,
        molecular_weight_ratio=molecular_weight_ratio,
    ).to("K")
    result = pressure.to("Pa") / (287.04749097718457 * units("J/(kg*K)") * virtual_temp)
    return result.to("kg/m**3")


def dewpoint(vapor_pressure_val):
    """Dewpoint from vapor pressure.

    Parameters
    ----------
    vapor_pressure_val : Quantity (pressure, hPa)

    Returns
    -------
    Quantity (degC)
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(_load_gpu_calc().dewpoint(_strip(vapor_pressure_val, "hPa")))
        return result * units.degC
    vals, shape, is_arr = _prep(_strip(vapor_pressure_val, "hPa"))
    if is_arr:
        result = np.asarray(_calc.dewpoint_array(vals[0])).reshape(shape)
    else:
        result = _calc.dewpoint(vals[0])
    return result * units.degC


def dewpoint_from_specific_humidity(pressure, specific_humidity):
    """Dewpoint from pressure and specific humidity.

    Parameters
    ----------
    pressure : Quantity (pressure)
    specific_humidity : Quantity (kg/kg)

    Returns
    -------
    Quantity (degC)
    """
    vals, shape, is_arr = _prep(
        _strip(pressure, "hPa"),
        _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity,
    )
    if is_arr:
        result = np.asarray(_calc.dewpoint_from_specific_humidity_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.dewpoint_from_specific_humidity(vals[0], vals[1])
    return result * units.degC


def dry_lapse(pressure, t_surface):
    """Dry adiabatic lapse rate temperature profile.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    t_surface : Quantity (temperature)

    Returns
    -------
    array Quantity (degC)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(t_surface, "degC"))
    result = np.asarray(_calc.dry_lapse(p, t))
    return result * units.degC


def dry_static_energy(height, temperature):
    """Dry static energy.

    Parameters
    ----------
    height : Quantity (m)
    temperature : Quantity (K)

    Returns
    -------
    Quantity (J/kg)
    """
    result = (
        9.80665 * units("m/s^2") * height.to("m")
        + 1004.6662184201462 * units("J/(kg*K)") * temperature.to("K")
    )
    return result.to("kJ/kg")


def exner_function(pressure):
    """Exner function.

    Parameters
    ----------
    pressure : Quantity (pressure)

    Returns
    -------
    Quantity (dimensionless)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"))
    if is_arr:
        result = np.asarray(_calc.exner_function_array(vals[0])).reshape(shape)
    else:
        result = _calc.exner_function(vals[0])
    return result * units.dimensionless


def find_intersections(x, y1, y2):
    """Find intersections of two curves.

    Parameters
    ----------
    x : array Quantity or array-like
    y1, y2 : array Quantity or array-like

    Returns
    -------
    list of (x, y) tuples
    """
    x_arr = _as_1d(_strip(x, "")) if hasattr(x, "magnitude") else _as_1d(x)
    y1_arr = _as_1d(_strip(y1, "")) if hasattr(y1, "magnitude") else _as_1d(y1)
    y2_arr = _as_1d(_strip(y2, "")) if hasattr(y2, "magnitude") else _as_1d(y2)
    return _calc.find_intersections(x_arr, y1_arr, y2_arr)


def geopotential_to_height(geopotential):
    """Convert geopotential to height.

    Parameters
    ----------
    geopotential : Quantity (m^2/s^2)

    Returns
    -------
    Quantity (m)
    """
    geopotential = geopotential.to("m**2/s**2")
    earth_radius = 6371008.7714 * units.m
    gravity = 9.80665 * units("m/s^2")
    return ((geopotential * earth_radius) / (gravity * earth_radius - geopotential)).to("m")


def get_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True,
              p_bottom=None, p_top=None):
    """Extract one or more fields from a sounding layer.

    Supports both the original form ``get_layer(pressure, values, p_bottom, p_top)``
    and the MetPy-style pressure-coordinate form
    ``get_layer(pressure, values..., height=None, bottom=..., depth=...)``.
    """
    if not args:
        raise TypeError("get_layer requires at least one value profile")

    value_args = args
    if p_bottom is None and p_top is None and len(args) >= 3:
        tail_bottom, tail_top = args[-2], args[-1]
        if _is_pressure_like(tail_bottom) and _is_pressure_like(tail_top):
            value_args = args[:-2]
            p_bottom = tail_bottom
            p_top = tail_top

    if p_bottom is None:
        if bottom is not None:
            p_bottom = bottom
        else:
            p_vals = np.asarray(_strip(pressure, "hPa"), dtype=np.float64)
            p_bottom = np.nanmax(p_vals) * units.hPa
    if p_top is None:
        if depth is None:
            depth = 100 * units.hPa
        p_top = p_bottom - depth

    p = _as_1d(_strip(pressure, "hPa"))
    pb = _as_float(_strip(p_bottom, "hPa"))
    pt = _as_float(_strip(p_top, "hPa"))
    p_result = None
    value_results = []

    for values in value_args:
        has_units = hasattr(values, "units")
        v_unit = values.units if has_units else None
        if has_units:
            v = _as_1d(np.asarray(values.magnitude, dtype=np.float64))
        elif hasattr(values, "values"):
            v = _as_1d(np.asarray(values.values, dtype=np.float64))
        else:
            v = _as_1d(values)
        p_out, v_out = _calc.get_layer(p, v, pb, pt)
        p_out = np.asarray(p_out, dtype=np.float64)
        v_out = np.asarray(v_out)
        keep = np.ediff1d(p_out, to_end=[1.0]) != 0
        p_out = p_out[keep]
        v_out = v_out[keep]
        if p_result is None:
            p_result = p_out * units.hPa
        v_result = v_out
        if v_unit is not None:
            v_result = v_result * v_unit
        value_results.append(v_result)

    return [p_result, *value_results]


def get_layer_heights(height, depth, *args, bottom=None, interpolate=True, with_agl=False):
    """Extract layer heights between two pressures.
    """
    if _is_pressure_like(height) and _can_convert(depth, "m") and len(args) == 2 and all(
        _is_pressure_like(arg) for arg in args
    ):
        pressure = _as_1d(_strip(height, "hPa"))
        heights = _as_1d(_strip(depth, "m"))
        p_bottom = _as_float(_strip(args[0], "hPa"))
        p_top = _as_float(_strip(args[1], "hPa"))
        p_out, h_out = _calc.get_layer_heights(pressure, heights, p_bottom, p_top)
        return np.asarray(p_out) * units.hPa, np.asarray(h_out) * units.m

    for datavar in args:
        if len(height) != len(datavar):
            raise ValueError("Height and data variables must have the same length.")

    working_height = height
    if with_agl:
        working_height = height - np.min(height)
    if bottom is None:
        bottom = working_height[0]

    working_height = working_height.to_base_units()
    bottom = bottom.to_base_units()
    top = bottom + depth.to_base_units()

    sort_idx = np.argsort(working_height.magnitude)
    height_sorted = working_height[sort_idx]
    mask = (
        height_sorted.magnitude >= bottom.magnitude - 1e-9
    ) & (
        height_sorted.magnitude <= top.magnitude + 1e-9
    )
    layer_heights = height_sorted[mask]

    if interpolate:
        points = layer_heights.magnitude
        if not np.any(np.isclose(points, top.magnitude)):
            points = np.append(points, top.magnitude)
        if not np.any(np.isclose(points, bottom.magnitude)):
            points = np.append(points, bottom.magnitude)
        layer_heights = np.sort(points) * working_height.units

    results = [layer_heights]
    for datavar in args:
        sorted_var = datavar[sort_idx]
        if interpolate:
            results.append(_interp_profile_to_heights(layer_heights, height_sorted, sorted_var))
        else:
            results.append(sorted_var[mask])
    return results


def height_to_geopotential(height):
    """Convert height to geopotential.

    Parameters
    ----------
    height : Quantity (m)

    Returns
    -------
    Quantity (m^2/s^2)
    """
    height = height.to("m")
    earth_radius = 6371008.7714 * units.m
    gravity = 9.80665 * units("m/s^2")
    return ((gravity * earth_radius * height) / (earth_radius + height)).to("m**2/s**2")


def isentropic_interpolation(levels, pressure, temperature, *args, vertical_dim=0,
                             temperature_out=False, max_iters=50, eps=1e-6,
                             bottom_up_search=True, **kwargs):
    """Interpolate fields to isentropic surfaces.

    Parameters
    ----------
    levels : array (K)
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    *args : array-like, optional
        Additional fields to interpolate to each isentropic surface.
    vertical_dim : int, optional
        Axis corresponding to the vertical dimension. Defaults to 0.
    temperature_out : bool, optional
        Include interpolated temperature in the returned list when ``True``.

    Returns
    -------
    list
        Pressure at each requested theta level, followed by interpolated
        temperature when requested, then each extra field.
    """
    del max_iters, eps, bottom_up_search, kwargs

    theta = _as_1d(_strip(levels, "K")) if hasattr(levels, "magnitude") else _as_1d(levels)
    temperature_arr, _ = _as_array_with_unit(temperature, "kelvin")
    temperature_arr = np.asarray(temperature_arr, dtype=np.float64)
    original_shape = temperature_arr.shape
    vertical_axis = vertical_dim % temperature_arr.ndim
    temperature_arr = np.moveaxis(temperature_arr, vertical_axis, 0)

    pressure_arr, _ = _as_array_with_unit(pressure, "hPa")
    pressure_arr = np.asarray(pressure_arr, dtype=np.float64)
    if pressure_arr.ndim == 1:
        if pressure_arr.shape[0] != temperature_arr.shape[0]:
            raise ValueError("Pressure levels must match the temperature vertical dimension.")
        pressure_arr = np.broadcast_to(
            pressure_arr.reshape((pressure_arr.shape[0],) + (1,) * (temperature_arr.ndim - 1)),
            temperature_arr.shape,
        ).copy()
    else:
        pressure_arr = np.moveaxis(pressure_arr, vertical_axis, 0)
        if pressure_arr.shape != temperature_arr.shape:
            raise ValueError("Pressure and temperature must have matching shapes.")

    field_arrays = []
    field_units = []
    for field in args:
        field_arr, field_unit = _as_array_with_unit(field)
        field_arr = np.asarray(field_arr, dtype=np.float64)
        field_arr = np.moveaxis(field_arr, vertical_axis, 0)
        if field_arr.shape != temperature_arr.shape:
            raise ValueError("Additional isentropic interpolation fields must match temperature.")
        field_arrays.append(np.ascontiguousarray(field_arr.reshape(-1)))
        field_units.append(field_unit)

    nz = int(temperature_arr.shape[0])
    spatial_shape = tuple(int(v) for v in temperature_arr.shape[1:])
    nx = int(np.prod(spatial_shape, dtype=np.int64)) if spatial_shape else 1
    ny = 1

    result = _calc.isentropic_interpolation(
        theta,
        np.ascontiguousarray(pressure_arr.reshape(-1)),
        np.ascontiguousarray(temperature_arr.reshape(-1)),
        field_arrays,
        nx,
        ny,
        nz,
    )

    target_shape = (len(theta),) + spatial_shape

    def _restore_output(values):
        arr = np.asarray(values, dtype=np.float64).reshape(target_shape)
        if len(original_shape) > 1 and vertical_axis != 0:
            arr = np.moveaxis(arr, 0, vertical_axis)
        return arr

    pressure_out = _restore_output(result[0]) * units.hPa
    temperature_interp = _restore_output(result[1]) * units.kelvin
    outputs = [pressure_out]
    if temperature_out:
        outputs.append(temperature_interp)
    for field_unit, values in zip(field_units, result[2:]):
        field_out = _restore_output(values)
        if field_unit is not None:
            field_out = field_out * field_unit
        outputs.append(field_out)
    return outputs


def mean_pressure_weighted(pressure, *args, height=None, bottom=None, depth=None):
    """Pressure-weighted mean of a quantity.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    values : array Quantity or array-like

    Returns
    -------
    float
    """
    if not args:
        raise TypeError("mean_pressure_weighted requires at least one value profile")

    layer = get_layer(pressure, *args, height=height, bottom=bottom, depth=depth)
    p_layer = layer[0]
    value_layers = layer[1:]

    trapz = getattr(np, "trapezoid", None) or np.trapz
    pres_int = 0.5 * (p_layer[-1] ** 2 - p_layer[0] ** 2)

    results = []
    for values in value_layers:
        if hasattr(values, "to") and _can_convert(values, "kelvin"):
            values = values.to("kelvin")
        results.append(trapz(values * p_layer, x=p_layer) / pres_int)
    return results


def mixed_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True):
    """Mixed-layer mean of one or more profiles.

    Supports both the original form ``mixed_layer(pressure, values, depth=...)``
    and the MetPy-style form ``mixed_layer(pressure, values..., bottom=..., depth=...)``.
    """
    if not args:
        raise TypeError("mixed_layer requires at least one value profile")
    if depth is None:
        depth = 100 * units.hPa

    layer = get_layer(
        pressure,
        *args,
        height=height,
        bottom=bottom,
        depth=depth,
        interpolate=interpolate,
    )
    p_layer = layer[0]
    value_layers = layer[1:]

    trapz = getattr(np, "trapezoid", None) or np.trapz
    actual_depth = abs(p_layer[0] - p_layer[-1])
    results = []
    for values in value_layers:
        if hasattr(values, "units"):
            mixed = trapz(values.magnitude, p_layer.magnitude) / -actual_depth.magnitude
            results.append(units.Quantity(mixed, values.units))
        else:
            mixed = trapz(np.asarray(values, dtype=np.float64), p_layer.magnitude) / -actual_depth.magnitude
            results.append(mixed)
    return results


def mixed_layer_cape_cin(pressure, temperature, dewpoint, **kwargs):
    """Mixed-layer CAPE and CIN.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    kwargs
        Additional keyword arguments to pass to :func:`mixed_parcel`.

    Returns
    -------
    tuple of (Quantity (J/kg), Quantity (J/kg))
    """
    depth = kwargs.get("depth", units.Quantity(100, "hPa"))
    start_p = kwargs.get("parcel_start_pressure", pressure[0])
    parcel_pressure, parcel_temp, parcel_dewpoint = mixed_parcel(
        pressure,
        temperature,
        dewpoint,
        **kwargs,
    )

    pressure_prof = pressure[pressure < (start_p - depth)]
    temp_prof = temperature[pressure < (start_p - depth)]
    dew_prof = dewpoint[pressure < (start_p - depth)]
    pressure_prof = units.Quantity(
        np.concatenate((
            [parcel_pressure.to(parcel_pressure.units).magnitude],
            pressure_prof.to(parcel_pressure.units).magnitude,
        )),
        parcel_pressure.units,
    )
    temperature_unit = getattr(temperature, "units", units.degC)
    dewpoint_unit = getattr(dewpoint, "units", units.degC)
    temp_prof = units.Quantity(
        np.concatenate((
            [parcel_temp.to(temperature_unit).magnitude],
            temp_prof.to(temperature_unit).magnitude,
        )),
        temperature_unit,
    )
    dew_prof = units.Quantity(
        np.concatenate((
            [parcel_dewpoint.to(dewpoint_unit).magnitude],
            dew_prof.to(dewpoint_unit).magnitude,
        )),
        dewpoint_unit,
    )

    p_prof, t_prof, td_prof, ml_profile = parcel_profile_with_lcl(
        pressure_prof,
        temp_prof,
        dew_prof,
    )
    return cape_cin(p_prof, t_prof, td_prof, ml_profile)


def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity, *, phase="liquid"):
    """Mixing ratio from pressure, temperature, and relative humidity.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    relative_humidity : Quantity (percent or dimensionless)
    phase : str, optional
        ``"liquid"`` (default), ``"ice"``, or ``"auto"``.

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    phase = _normalize_phase(phase)
    if phase != "liquid":
        rh = _rh_to_fraction(relative_humidity)
        return _attach(
            rh * saturation_mixing_ratio(pressure, temperature, phase=phase).to("kg/kg").magnitude,
            "kg/kg",
        )
    rh = _rh_to_percent(relative_humidity)
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"), rh)
    if is_arr:
        result = np.asarray(_calc.mixing_ratio_from_relative_humidity_array(vals[0], vals[1], vals[2])).reshape(shape) / 1000.0
    else:
        result = _calc.mixing_ratio_from_relative_humidity(vals[0], vals[1], vals[2]) / 1000.0
    return _attach(result, "kg/kg")


def mixing_ratio_from_specific_humidity(specific_humidity):
    """Mixing ratio from specific humidity.

    Parameters
    ----------
    specific_humidity : Quantity (kg/kg)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    vals, shape, is_arr = _prep(_strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity)
    if is_arr:
        result = np.asarray(_calc.mixing_ratio_from_specific_humidity_array(vals[0])).reshape(shape) / 1000.0
    else:
        result = _calc.mixing_ratio_from_specific_humidity(vals[0]) / 1000.0
    return _attach(result, "kg/kg")


def moist_lapse(pressure, temperature, reference_pressure=None):
    """Moist adiabatic lapse rate temperature profile.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    array Quantity (degC)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    if reference_pressure is None:
        result = np.asarray(_calc.moist_lapse(p, t))
    else:
        ref_p = _as_float(_strip(reference_pressure, "hPa"))
        result = np.empty_like(p, dtype=np.float64)
        equal_mask = np.isclose(p, ref_p)
        lower_mask = p < ref_p
        higher_mask = p > ref_p

        if np.any(lower_mask):
            lower_levels = np.sort(p[lower_mask])[::-1]
            up_path = np.ascontiguousarray(np.concatenate(([ref_p], lower_levels)))
            up_result = np.asarray(_calc.moist_lapse(up_path, t))
            for level, value in zip(up_path[1:], up_result[1:]):
                result[np.isclose(p, level)] = value

        if np.any(higher_mask):
            higher_levels = np.sort(p[higher_mask])
            down_path = np.ascontiguousarray(np.concatenate(([ref_p], higher_levels)))
            down_result = np.asarray(_calc.moist_lapse(down_path, t))
            for level, value in zip(down_path[1:], down_result[1:]):
                result[np.isclose(p, level)] = value

        result[equal_mask] = t

        if not np.any(equal_mask):
            sort_idx = np.argsort(p)
            p_sorted = p[sort_idx]
            result_sorted = result[sort_idx]
            result = np.interp(p, p_sorted, result_sorted)
    return result * units.degC


def moist_static_energy(height, temperature, specific_humidity):
    """Moist static energy.

    Parameters
    ----------
    height : Quantity (m)
    temperature : Quantity (K)
    specific_humidity : Quantity (kg/kg)

    Returns
    -------
    Quantity (J/kg)
    """
    specific_humidity = specific_humidity.to("dimensionless")
    result = dry_static_energy(height, temperature) + (
        2500840.0 * units("J/kg") * specific_humidity
    ).to("kJ/kg")
    return result.to("kJ/kg")


def montgomery_streamfunction(height_or_theta, temperature_or_pressure=None,
                              temperature=None, height=None):
    """Montgomery streamfunction on isentropic surfaces.

    MetPy form: ``montgomery_streamfunction(height, temperature)``
    Legacy form: ``montgomery_streamfunction(theta, pressure, temperature, height)``

    Parameters
    ----------
    height : Quantity (m)
        Geopotential height on isentropic surface.
    temperature : Quantity (K)
        Temperature on isentropic surface.

    Returns
    -------
    Quantity (J/kg)
        Montgomery streamfunction: M = c_p * T + g * z
    """
    if temperature is not None and height is not None:
        # Explicit keyword form
        h = temperature_or_pressure  # actually height passed as 1st positional
        t = temperature
        # Hmm, ambiguous. Let's handle all cases below.
        pass

    if temperature_or_pressure is not None and temperature is None and height is None:
        return dry_static_energy(height_or_theta, temperature_or_pressure)
    elif temperature is not None and height is not None:
        # 4-arg legacy form: (theta, pressure, temperature, height)
        result = _vec_call(_calc.montgomery_streamfunction,
                           _strip(height_or_theta, "K"),
                           _strip(temperature_or_pressure, "hPa"),
                           _strip(temperature, "K"),
                           _strip(height, "m"))
        return result * units("J/kg")
    else:
        raise TypeError(
            "montgomery_streamfunction expects (height, temperature) or "
            "(theta, pressure, temperature, height)"
        )


def most_unstable_cape_cin(pressure, temperature, dewpoint, depth=300, **kwargs):
    """Most-unstable CAPE and CIN.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    depth : Quantity or float, optional
        Search depth in hPa (default 300).

    Returns
    -------
    tuple of (Quantity (J/kg), Quantity (J/kg))
    """
    if "depth" not in kwargs:
        kwargs["depth"] = depth
    pressure, temperature, dewpoint = _drop_nan_profiles(pressure, temperature, dewpoint)
    _, _, _, parcel_idx = most_unstable_parcel(pressure, temperature, dewpoint, **kwargs)
    pressure = pressure[parcel_idx:]
    temperature = temperature[parcel_idx:]
    dewpoint = dewpoint[parcel_idx:]
    p_prof, t_prof, td_prof, parcel_prof = _parcel_profile_with_environment(
        pressure,
        temperature,
        dewpoint,
    )
    return cape_cin(p_prof, t_prof, td_prof, parcel_prof)


def parcel_profile(pressure, temperature, dewpoint):
    """Parcel temperature profile.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    array Quantity (K)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    result = np.asarray(_calc.parcel_profile(p, t, td))
    return (result * units.degC).to("kelvin")


def reduce_point_density(points, radius, priority=None):
    """Reduce point density by removing points too close together.

    Parameters
    ----------
    points : (N, M) array-like
    radius : Quantity or float
    priority : (N,) array-like, optional

    Returns
    -------
    ndarray[bool]
    """
    from scipy.spatial import KDTree

    if hasattr(radius, "units"):
        radius = radius.to("m").magnitude

    if hasattr(points, "units"):
        points = points.to("m").magnitude

    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 2:
        points = points.reshape(-1, 1)

    good_vals = np.isfinite(points)
    points = np.where(good_vals, points, 0.0)
    keep = np.logical_and.reduce(good_vals, axis=-1)
    tree = KDTree(points)

    if priority is None:
        sorted_indices = range(len(points))
    else:
        sorted_indices = np.argsort(np.asarray(priority))[::-1]

    for ind in sorted_indices:
        if keep[ind]:
            neighbors = tree.query_ball_point(points[ind], radius)
            keep[neighbors] = False
            keep[ind] = True

    return keep


def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio_val, *, phase="liquid"):
    """Relative humidity from mixing ratio.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    mixing_ratio_val : Quantity (g/kg or kg/kg)

    Returns
    -------
    Quantity (dimensionless, 0-1)
    """
    phase = _normalize_phase(phase)
    if phase != "liquid":
        w_kgkg = _strip(mixing_ratio_val, "kg/kg") if _can_convert(mixing_ratio_val, "kg/kg") else mixing_ratio_val
        ws_kgkg = saturation_mixing_ratio(pressure, temperature, phase=phase).to("kg/kg").magnitude
        return _attach(np.asarray(w_kgkg, dtype=np.float64) / ws_kgkg, "")
    p_raw = _strip(pressure, "hPa")
    t_raw = _strip(temperature, "degC")
    w_raw = _strip(mixing_ratio_val, "g/kg") if _can_convert(mixing_ratio_val, "g/kg") else np.asarray(_strip(mixing_ratio_val, "kg/kg"), dtype=np.float64) * 1000.0
    vals, shape, is_arr = _prep(p_raw, t_raw, w_raw)
    if is_arr:
        result = np.asarray(_calc.relative_humidity_from_mixing_ratio_array(vals[0], vals[1], vals[2])).reshape(shape) / 100.0
    else:
        result = _calc.relative_humidity_from_mixing_ratio(vals[0], vals[1], vals[2]) / 100.0
    return _attach(result, "")


def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity, *, phase="liquid"):
    """Relative humidity from specific humidity.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    specific_humidity : Quantity (kg/kg)

    Returns
    -------
    Quantity (dimensionless, 0-1)
    """
    phase = _normalize_phase(phase)
    if phase != "liquid":
        q = np.asarray(
            _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity,
            dtype=np.float64,
        )
        mixing_ratio_val = q / np.clip(1.0 - q, 1e-12, None)
        return relative_humidity_from_mixing_ratio(
            pressure,
            temperature,
            mixing_ratio_val * units("kg/kg"),
            phase=phase,
        )
    vals, shape, is_arr = _prep(
        _strip(pressure, "hPa"),
        _strip(temperature, "degC"),
        _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity,
    )
    if is_arr:
        result = np.asarray(_calc.relative_humidity_from_specific_humidity_array(vals[0], vals[1], vals[2])).reshape(shape) / 100.0
    else:
        result = _calc.relative_humidity_from_specific_humidity(vals[0], vals[1], vals[2]) / 100.0
    return _attach(result, "")


def saturation_equivalent_potential_temperature(pressure, temperature):
    """Saturation equivalent potential temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (K)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"))
    if is_arr:
        result = np.asarray(_calc.saturation_equivalent_potential_temperature_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.saturation_equivalent_potential_temperature(vals[0], vals[1])
    return result * units.K


def scale_height(temperature_bottom, temperature_top=None):
    """Atmospheric scale height.
    """
    if temperature_top is None:
        result = _vec_call(_calc.scale_height, _strip(temperature_bottom, "K"))
        return result * units.m
    mean_temperature = 0.5 * (temperature_bottom.to("K") + temperature_top.to("K"))
    return ((287.04749097718457 * units("J/(kg*K)") * mean_temperature)
            / (9.80665 * units("m/s^2"))).to("m")


def specific_humidity_from_dewpoint(pressure, dewpoint, *, phase="liquid"):
    """Specific humidity from pressure and dewpoint.
    """
    phase = _normalize_phase(phase)
    mixing = saturation_mixing_ratio(pressure, dewpoint, phase=phase)
    return specific_humidity_from_mixing_ratio(mixing)


def static_stability(pressure, temperature, vertical_dim=0):
    """Static stability.

    Parameters
    ----------
    pressure : array Quantity (pressure, hPa)
    temperature : array Quantity (K)

    Returns
    -------
    array Quantity (K/Pa)
    """
    p_hpa = pressure.to("hPa") if hasattr(pressure, "to") else np.asarray(pressure, dtype=np.float64) * units.hPa
    t_k = temperature.to("K") if hasattr(temperature, "to") else np.asarray(temperature, dtype=np.float64) * units.K
    theta = potential_temperature(pressure, temperature).to("K")
    p_vals = np.asarray(p_hpa.magnitude, dtype=np.float64)
    log_theta = np.asarray(np.log(theta.magnitude), dtype=np.float64)
    ax = vertical_dim % log_theta.ndim
    dlogtheta_dp = np.gradient(log_theta, p_vals, axis=ax, edge_order=2) / units.hPa
    rd = 287.05 * units("J/(kg*K)")
    return -rd * t_k / p_hpa * dlogtheta_dp


def surface_based_cape_cin(pressure, temperature, dewpoint):
    """Surface-based CAPE and CIN.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    tuple of (Quantity (J/kg), Quantity (J/kg))
    """
    pressure, temperature, dewpoint = _drop_nan_profiles(pressure, temperature, dewpoint)
    p_prof, t_prof, td_prof, parcel_prof = _parcel_profile_with_environment(
        pressure,
        temperature,
        dewpoint,
    )
    return cape_cin(p_prof, t_prof, td_prof, parcel_prof)


def temperature_from_potential_temperature(pressure, theta):
    """Temperature from potential temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    theta : Quantity (K)

    Returns
    -------
    Quantity (K)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(theta, "K"))
    if is_arr:
        result = np.asarray(_calc.temperature_from_potential_temperature_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.temperature_from_potential_temperature(vals[0], vals[1])
    return result * units.K


def vertical_velocity(omega, pressure, temperature):
    """Convert pressure vertical velocity (omega) to w (m/s).

    Parameters
    ----------
    omega : Quantity (Pa/s)
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (m/s)
    """
    result = _vec_call(_calc.vertical_velocity, _strip(omega, "Pa/s"), _strip(pressure, "hPa"), _strip(temperature, "degC"))
    return result * units("m/s")


def vertical_velocity_pressure(w, pressure, temperature):
    """Convert w (m/s) to pressure vertical velocity (omega, Pa/s).

    Parameters
    ----------
    w : Quantity (m/s)
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (Pa/s)
    """
    result = _vec_call(_calc.vertical_velocity_pressure, _strip(w, "m/s"), _strip(pressure, "hPa"), _strip(temperature, "degC"))
    return result * units("Pa/s")


def virtual_potential_temperature(pressure, temperature, mixing_ratio_val):
    """Virtual potential temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    mixing_ratio_val : Quantity (g/kg or kg/kg)

    Returns
    -------
    Quantity (K)
    """
    p_raw = _strip(pressure, "hPa")
    t_raw = _strip(temperature, "degC")
    w_raw = _strip(mixing_ratio_val, "g/kg") if _can_convert(mixing_ratio_val, "g/kg") else np.asarray(_strip(mixing_ratio_val, "kg/kg"), dtype=np.float64) * 1000.0
    vals, shape, is_arr = _prep(p_raw, t_raw, w_raw)
    if is_arr:
        result = np.asarray(_calc.virtual_potential_temperature_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.virtual_potential_temperature(vals[0], vals[1], vals[2])
    return result * units.K


def wet_bulb_potential_temperature(pressure, temperature, dewpoint):
    """Wet-bulb potential temperature."""
    theta_e = equivalent_potential_temperature(pressure, temperature, dewpoint).to("kelvin")
    theta_e_vals = np.asarray(theta_e.magnitude, dtype=np.float64)
    x = theta_e_vals / 273.15
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    a = 7.101574 - 20.68208 * x + 16.11182 * x2 + 2.574631 * x3 - 5.205688 * x4
    b = 1 - 3.552497 * x + 3.781782 * x2 - 0.6899655 * x3 - 0.5929340 * x4
    theta_w_vals = theta_e_vals - np.exp(a / b)
    if np.isscalar(theta_e_vals):
        return theta_e if theta_e <= units.Quantity(173.15, "kelvin") else theta_w_vals * units.kelvin
    result = np.where(theta_e_vals <= 173.15, theta_e_vals, theta_w_vals)
    return result * units.kelvin


def _mixed_parcel_impl(pressure, temperature, dewpoint, parcel_start_pressure=None,
                       height=None, bottom=None, depth=None, interpolate=True):
    if parcel_start_pressure is None:
        parcel_start_pressure = pressure[0]
    if depth is None:
        depth = units.Quantity(100, "hPa")

    theta = potential_temperature(pressure, temperature)
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    mean_theta, mean_mixing_ratio = mixed_layer(
        pressure,
        theta,
        mixing_ratio,
        bottom=bottom,
        height=height,
        depth=depth,
        interpolate=interpolate,
    )
    mean_temperature = mean_theta * exner_function(parcel_start_pressure)
    mean_vapor_pressure = vapor_pressure(parcel_start_pressure, mean_mixing_ratio)
    mean_dewpoint = globals()["dewpoint"](mean_vapor_pressure)

    temperature_unit = getattr(temperature, "units", units.degC)
    dewpoint_unit = getattr(dewpoint, "units", units.degC)
    return (
        parcel_start_pressure,
        mean_temperature.to(temperature_unit),
        mean_dewpoint.to(dewpoint_unit),
    )


def get_mixed_layer_parcel(pressure, temperature, dewpoint, depth=100.0):
    """Get mixed-layer parcel properties."""
    if not hasattr(depth, "magnitude"):
        depth = depth * units.hPa
    return _mixed_parcel_impl(pressure, temperature, dewpoint, depth=depth)


def _most_unstable_parcel_impl(pressure, temperature, dewpoint, height=None, bottom=None,
                               depth=None):
    if depth is None:
        depth = units.Quantity(300, "hPa")
    p_layer, t_layer, td_layer = get_layer(
        pressure,
        temperature,
        dewpoint,
        bottom=bottom,
        depth=depth,
        height=height,
        interpolate=False,
    )
    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    theta_e_vals = np.asarray(theta_e.to("kelvin").magnitude, dtype=np.float64)
    max_idx = int(np.argmax(theta_e_vals))
    return p_layer[max_idx], t_layer[max_idx], td_layer[max_idx], max_idx


def get_most_unstable_parcel(pressure, temperature, dewpoint,
                             height=None, bottom=None, depth=300.0):
    """Get most-unstable parcel properties."""
    if not hasattr(depth, "magnitude"):
        depth = depth * units.hPa
    return _most_unstable_parcel_impl(
        pressure,
        temperature,
        dewpoint,
        height=height,
        bottom=bottom,
        depth=depth,
    )


def psychrometric_vapor_pressure(temperature, wet_bulb, pressure):
    """Psychrometric vapor pressure from dry-bulb, wet-bulb, and pressure.

    Parameters
    ----------
    temperature : Quantity (temperature)
    wet_bulb : Quantity (temperature)
    pressure : Quantity (pressure)

    Returns
    -------
    Quantity (hPa)
    """
    result = _vec_call(_calc.psychrometric_vapor_pressure, _strip(temperature, "degC"), _strip(wet_bulb, "degC"), _strip(pressure, "hPa"))
    return result * units.hPa


def frost_point(temperature, relative_humidity):
    """Frost point temperature.

    Parameters
    ----------
    temperature : Quantity (temperature)
    relative_humidity : Quantity (percent or dimensionless)

    Returns
    -------
    Quantity (degC)
    """
    rh = _rh_to_percent(relative_humidity)
    vals, shape, is_arr = _prep(_strip(temperature, "degC"), rh)
    if is_arr:
        result = np.asarray(_calc.frost_point_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.frost_point(vals[0], vals[1])
    return result * units.degC


# ============================================================================
# Aliases
# ============================================================================

def mixed_parcel(pressure, temperature, dewpoint, parcel_start_pressure=None,
                 height=None, bottom=None, depth=100, interpolate=True):
    """Mixed-layer parcel (MetPy-compatible)."""
    if depth is None:
        depth = units.Quantity(100, "hPa")
    elif not hasattr(depth, "magnitude"):
        depth = depth * units.hPa
    return _mixed_parcel_impl(
        pressure,
        temperature,
        dewpoint,
        parcel_start_pressure=parcel_start_pressure,
        height=height,
        bottom=bottom,
        depth=depth,
        interpolate=interpolate,
    )


def most_unstable_parcel(pressure, temperature, dewpoint, height=None,
                         bottom=None, depth=300):
    """Most-unstable parcel (MetPy-compatible)."""
    if depth is None:
        depth = units.Quantity(300, "hPa")
    elif not hasattr(depth, "magnitude"):
        depth = depth * units.hPa
    return _most_unstable_parcel_impl(
        pressure,
        temperature,
        dewpoint,
        height=height,
        bottom=bottom,
        depth=depth,
    )


def psychrometric_vapor_pressure_wet(pressure, dry_bulb_temperature, wet_bulb_temperature,
                                     psychrometer_coefficient=None):
    """Calculate vapor pressure from wet- and dry-bulb temperatures."""
    if psychrometer_coefficient is None:
        psychrometer_coefficient = units.Quantity(6.21e-4, "1/K")
    return (
        saturation_vapor_pressure(wet_bulb_temperature)
        - psychrometer_coefficient
        * pressure
        * (dry_bulb_temperature - wet_bulb_temperature).to("kelvin")
    )


def _get_layer_heights_runtime(height, depth, *args, bottom=None, interpolate=True,
                               with_agl=False):
    if with_agl:
        height = height - np.min(height)
    if bottom is None:
        bottom = height[0]

    height = height.to_base_units()
    bottom = bottom.to(height.units)
    depth = depth.to(height.units)
    top = bottom + depth

    sort_inds = np.argsort(np.asarray(height.magnitude, dtype=np.float64))
    height_sorted = height[sort_inds]
    in_layer = (height_sorted >= bottom) & (height_sorted <= top)
    layer_height = height_sorted[in_layer]

    if interpolate:
        layer_vals = np.asarray(layer_height.magnitude, dtype=np.float64)
        if layer_vals.size == 0 or not np.any(np.isclose(layer_vals, bottom.magnitude)):
            layer_vals = np.append(layer_vals, bottom.magnitude)
        if layer_vals.size == 0 or not np.any(np.isclose(layer_vals, top.magnitude)):
            layer_vals = np.append(layer_vals, top.magnitude)
        layer_height = units.Quantity(np.sort(layer_vals), height.units)

    ret = [layer_height]
    for datavar in args:
        datavar_sorted = datavar[sort_inds]
        if interpolate:
            x_new = np.asarray(layer_height.to(height.units).magnitude, dtype=np.float64)
            x_old = np.asarray(height_sorted.magnitude, dtype=np.float64)
            if hasattr(datavar_sorted, "units"):
                datavar_vals = np.interp(
                    x_new,
                    x_old,
                    np.asarray(datavar_sorted.magnitude, dtype=np.float64),
                )
                datavar_sorted = datavar_vals * datavar_sorted.units
            else:
                datavar_sorted = np.interp(
                    x_new,
                    x_old,
                    np.asarray(datavar_sorted, dtype=np.float64),
                )
        else:
            datavar_sorted = datavar_sorted[in_layer]
        ret.append(datavar_sorted)
    return ret


def _kinematic_flux_runtime(vel, scalar, perturbation=False, axis=-1):
    flux = np.mean(vel * scalar, axis=axis)
    if not perturbation:
        flux = flux - np.mean(vel, axis=axis) * np.mean(scalar, axis=axis)
    if hasattr(flux, "units"):
        return units.Quantity(np.atleast_1d(np.asarray(flux.magnitude, dtype=np.float64)), flux.units)
    return np.atleast_1d(np.asarray(flux, dtype=np.float64))


# ============================================================================
# Wind
# ============================================================================

def wind_speed(u, v):
    """Wind speed from (u, v) components.

    Parameters
    ----------
    u, v : array Quantity (m/s)

    Returns
    -------
    array Quantity (m/s)
    """
    orig_shape = np.asarray(_strip(u, "m/s")).shape
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.wind_speed(u_arr, v_arr))
    return result.reshape(orig_shape) * units("m/s")


def wind_direction(u, v, convention="from"):
    """Meteorological wind direction from (u, v).

    Parameters
    ----------
    u, v : array Quantity (m/s)

    Returns
    -------
    array Quantity (degree)
    """
    if convention not in {"from", "to"}:
        raise ValueError('Invalid kwarg for "convention". Valid options are "from" or "to".')
    orig_shape = np.asarray(_strip(u, "m/s")).shape
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.wind_direction(u_arr, v_arr))
    if convention == "to":
        result = np.where(result <= 180.0, result + 180.0, result - 180.0)
        calm_mask = (u_arr == 0.0) & (v_arr == 0.0)
        result = np.where(calm_mask, 0.0, result)
    return result.reshape(orig_shape) * units.degree


def wind_components(speed, direction):
    """Convert (speed, direction) to (u, v) components.

    Parameters
    ----------
    speed : array Quantity (m/s)
    direction : array Quantity (degree)

    Returns
    -------
    tuple of (array Quantity (m/s), array Quantity (m/s))
    """
    orig_shape = np.asarray(_strip(speed, "m/s")).shape
    spd = _as_1d(_strip(speed, "m/s"))
    dirn = _as_1d(_strip(direction, "degree"))
    u, v = _calc.wind_components(spd, dirn)
    ms = units("m/s")
    return np.asarray(u).reshape(orig_shape) * ms, np.asarray(v).reshape(orig_shape) * ms


def bulk_shear(pressure_or_u, u_or_v, v_or_height=None, height=None,
               bottom=None, depth=None, top=None):
    """Bulk wind shear over a height layer.

    Supports both MetPy form ``bulk_shear(p, u, v, height=z, depth=6*km)``
    and direct form ``bulk_shear(u, v, height, bottom, top=top)``.

    Parameters
    ----------
    pressure_or_u : array Quantity
    u_or_v : array Quantity
    v_or_height : array Quantity, optional
    height : array Quantity (m), optional
    bottom : Quantity (m), optional
    depth : Quantity (m), optional
    top : Quantity (m), optional

    Returns
    -------
    tuple of (Quantity (m/s), Quantity (m/s))
        Shear u and v components.
    """
    # Detect MetPy form: bulk_shear(pressure, u, v, height=z, depth=6km)
    # vs direct form: bulk_shear(u, v, height, bottom, top=top)
    if height is not None:
        # MetPy form: 1st arg is pressure (ignored), 2nd=u, 3rd=v, height=keyword
        u_arr = _as_1d(_strip(u_or_v, "m/s"))
        v_arr = _as_1d(_strip(v_or_height, "m/s"))
        h_arr = _as_1d(_strip(height, "m"))
    else:
        # Direct form: 1st=u, 2nd=v, 3rd=height
        u_arr = _as_1d(_strip(pressure_or_u, "m/s"))
        v_arr = _as_1d(_strip(u_or_v, "m/s"))
        h_arr = _as_1d(_strip(v_or_height, "m"))

    # Resolve bottom/top from depth (default bottom = first height, matching MetPy)
    bot_val = _scalar_strip(bottom, "m") if bottom is not None else float(h_arr[0])
    if top is not None:
        top_val = _scalar_strip(top, "m")
    elif depth is not None:
        top_val = bot_val + _scalar_strip(depth, "m")
    else:
        top_val = float(h_arr[-1])  # default: full profile

    su, sv = _calc.bulk_shear(u_arr, v_arr, h_arr, bot_val, top_val)
    return _attach(su, "m/s"), _attach(sv, "m/s")


def mean_wind(u, v, height, bottom, top):
    """Pressure-weighted mean wind over a height layer.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    height : array Quantity (m)
    bottom, top : Quantity (m)

    Returns
    -------
    tuple of (Quantity (m/s), Quantity (m/s))
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    bot = _as_float(_strip(bottom, "m"))
    top_val = _as_float(_strip(top, "m"))
    mu, mv = _calc.mean_wind(u_arr, v_arr, h_arr, bot, top_val)
    return mu * units("m/s"), mv * units("m/s")


def storm_relative_helicity(*args, bottom=None, depth=None, storm_u=None, storm_v=None):
    """Storm-relative helicity.

    MetPy form: ``storm_relative_helicity(height, u, v, depth, *, bottom, storm_u, storm_v)``
    Legacy form: ``storm_relative_helicity(u, v, height, depth, storm_u, storm_v)``

    Parameters
    ----------
    height : array Quantity (m)
    u, v : array Quantity (m/s)
    depth : Quantity (m)
    storm_u, storm_v : Quantity (m/s)

    Returns
    -------
    tuple of (Quantity (m^2/s^2), Quantity (m^2/s^2), Quantity (m^2/s^2))
        Positive, negative, and total SRH.
    """
    if len(args) == 6:
        # Legacy positional: (u, v, height, depth, storm_u, storm_v)
        u, v, height_a, depth_a, storm_u, storm_v = args
    elif len(args) == 4:
        # MetPy form: (height, u, v, depth, *, storm_u=, storm_v=)
        height_a, u, v, depth_a = args
        if depth is not None:
            depth_a = depth
    elif len(args) == 3:
        # Keyword form: (height, u, v, *, depth=, storm_u=, storm_v=)
        height_a, u, v = args
        depth_a = depth
    else:
        raise TypeError(
            "storm_relative_helicity expects (height, u, v, depth, *, storm_u, storm_v) "
            f"— got {len(args)} positional args"
        )

    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height_a, "m"))
    d = _scalar_strip(depth_a, "m")
    if storm_u is None or storm_v is None:
        # Auto-compute Bunkers right-mover if storm motion not provided
        right_mover, _, _ = bunkers_storm_motion(u, v, height_a)
        if storm_u is None:
            storm_u = right_mover[0]
        if storm_v is None:
            storm_v = right_mover[1]
    su = _scalar_strip(storm_u, "m/s")
    sv = _scalar_strip(storm_v, "m/s")
    pos, neg, total = _calc.storm_relative_helicity(u_arr, v_arr, h_arr, d, su, sv)
    return _attach(pos, "m**2/s**2"), _attach(neg, "m**2/s**2"), _attach(total, "m**2/s**2")


def bunkers_storm_motion(pressure_or_u, u_or_v, v_or_height, height=None):
    """Bunkers storm motion (right-mover, left-mover, mean wind).

    MetPy form: ``bunkers_storm_motion(pressure, u, v, height)``
    Direct form: ``bunkers_storm_motion(u, v, height)``

    Parameters
    ----------
    pressure_or_u : array Quantity
    u_or_v : array Quantity
    v_or_height : array Quantity
    height : array Quantity (m), optional

    Returns
    -------
    tuple of 3 tuples, each (Quantity (m/s), Quantity (m/s))
        (right_u, right_v), (left_u, left_v), (mean_u, mean_v)
    """
    if height is not None:
        # MetPy form: (pressure, u, v, height)
        p_arr = _as_1d(_strip(pressure_or_u, "hPa"))
        u_arr = _as_1d(_strip(u_or_v, "m/s"))
        v_arr = _as_1d(_strip(v_or_height, "m/s"))
        h_arr = _as_1d(_strip(height, "m"))
    else:
        # Direct form: (u, v, height) — derive pressure from std atmosphere
        u_arr = _as_1d(_strip(pressure_or_u, "m/s"))
        v_arr = _as_1d(_strip(u_or_v, "m/s"))
        h_arr = _as_1d(_strip(v_or_height, "m"))
        p_arr = np.array([_calc.height_to_pressure_std(float(hi)) for hi in h_arr])
    (ru, rv), (lu, lv), (mu, mv) = _calc.bunkers_storm_motion(p_arr, u_arr, v_arr, h_arr)
    out_unit = getattr(pressure_or_u, "units", None) if height is None else getattr(u_or_v, "units", None)
    right = units.Quantity(np.asarray([ru, rv]), "m/s")
    left = units.Quantity(np.asarray([lu, lv]), "m/s")
    mean = units.Quantity(np.asarray([mu, mv]), "m/s")
    if out_unit is not None:
        right = right.to(out_unit)
        left = left.to(out_unit)
        mean = mean.to(out_unit)
    return right, left, mean


def corfidi_storm_motion(pressure_or_u, u_or_v, v_or_height, *args, u_llj=None, v_llj=None):
    """Corfidi upwind and downwind vectors for MCS motion.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    height : array Quantity (m)
    u_850, v_850 : Quantity (m/s)

    Returns
    -------
    tuple of 2 tuples, each (Quantity (m/s), Quantity (m/s))
        (upwind_u, upwind_v), (downwind_u, downwind_v)
    """
    if len(args) != 2:
        raise TypeError(
            "corfidi_storm_motion expects either (pressure, u, v, *, u_llj, v_llj) "
            "or legacy positional (u, v, height, u_850, v_850)"
        )
    u_850, v_850 = args
    u_arr = _as_1d(_strip(pressure_or_u, "m/s"))
    v_arr = _as_1d(_strip(u_or_v, "m/s"))
    h_arr = _as_1d(_strip(v_or_height, "m"))
    u8 = _as_float(_strip(u_850, "m/s"))
    v8 = _as_float(_strip(v_850, "m/s"))
    (uu, uv), (du, dv) = _calc.corfidi_storm_motion(u_arr, v_arr, h_arr, u8, v8)
    return (_attach(uu, "m/s"), _attach(uv, "m/s")), (_attach(du, "m/s"), _attach(dv, "m/s"))


def friction_velocity(u, w):
    """Friction velocity from time series of u and w components.

    Parameters
    ----------
    u : array Quantity (m/s)
        Along-wind component time series.
    w : array Quantity (m/s)
        Vertical wind component time series.

    Returns
    -------
    Quantity (m/s)
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    w_arr = _as_1d(_strip(w, "m/s"))
    return _calc.friction_velocity(u_arr, w_arr) * units("m/s")


def tke(u, v, w):
    """Turbulent kinetic energy from time series of wind components.

    Parameters
    ----------
    u, v, w : array Quantity (m/s)
        Wind component time series.

    Returns
    -------
    Quantity (m**2/s**2)
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    w_arr = _as_1d(_strip(w, "m/s"))
    return _calc.tke(u_arr, v_arr, w_arr) * units("m**2/s**2")


def gradient_richardson_number(height, potential_temperature, u, v):
    """Gradient Richardson number at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temperature : array Quantity (K)
    u, v : array Quantity (m/s)

    Returns
    -------
    array Quantity (dimensionless)
    """
    z = _as_1d(_strip(height, "m"))
    theta = _as_1d(_strip(potential_temperature, "K"))
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.gradient_richardson_number(z, theta, u_arr, v_arr))
    return result * units.dimensionless


# MetPy-aligned redefinitions for the wind/profile runtime parity campaign.
def storm_relative_helicity(*args, bottom=None, depth=None, storm_u=None, storm_v=None):
    """Storm-relative helicity."""
    if len(args) == 6:
        u, v, height_a, depth_a, storm_u, storm_v = args
    elif len(args) == 4:
        height_a, u, v, depth_a = args
        if depth is not None:
            depth_a = depth
    elif len(args) == 3:
        height_a, u, v = args
        depth_a = depth
    else:
        raise TypeError(
            "storm_relative_helicity expects (height, u, v, depth, *, storm_u, storm_v) "
            f"but got {len(args)} positional args"
        )

    if depth_a is None:
        raise TypeError("storm_relative_helicity missing required argument: 'depth'")
    if bottom is None:
        bottom = units.Quantity(0, "m")
    if storm_u is None:
        storm_u = units.Quantity(0, "m/s")
    if storm_v is None:
        storm_v = units.Quantity(0, "m/s")

    _, u_layer, v_layer = _get_layer_heights_runtime(
        height_a,
        depth_a,
        u,
        v,
        bottom=bottom,
        interpolate=True,
        with_agl=True,
    )
    storm_relative_u = u_layer - storm_u
    storm_relative_v = v_layer - storm_v
    int_layers = (
        storm_relative_u[1:] * storm_relative_v[:-1]
        - storm_relative_u[:-1] * storm_relative_v[1:]
    )
    positive_mask = np.asarray(int_layers.magnitude, dtype=np.float64) > 0.0
    negative_mask = np.asarray(int_layers.magnitude, dtype=np.float64) < 0.0
    srh_unit = units("meter**2 / second**2")
    positive_srh = (
        int_layers[positive_mask].sum().to(srh_unit)
        if np.any(positive_mask)
        else units.Quantity(0.0, srh_unit)
    )
    negative_srh = (
        int_layers[negative_mask].sum().to(srh_unit)
        if np.any(negative_mask)
        else units.Quantity(0.0, srh_unit)
    )
    total_srh = (positive_srh + negative_srh).to(srh_unit)
    return positive_srh, negative_srh, total_srh


def corfidi_storm_motion(pressure_or_u, u_or_v, v_or_height, *args, u_llj=None, v_llj=None):
    """Corfidi upwind- and downwind-propagating MCS motion."""
    if (u_llj is None) ^ (v_llj is None):
        raise ValueError("Must specify both u_llj and v_llj or neither")

    if len(args) == 0:
        pressure = pressure_or_u
        u = u_or_v
        v = v_or_height
    elif len(args) == 2:
        pressure = units.Quantity(
            np.asarray([_calc.height_to_pressure_std(float(hi)) for hi in _as_1d(_strip(v_or_height, "m"))]),
            "hPa",
        )
        u = pressure_or_u
        v = u_or_v
        u_llj, v_llj = args
    else:
        raise TypeError(
            "corfidi_storm_motion expects (pressure, u, v, *, u_llj, v_llj) "
            "or legacy positional (u, v, height, u_850, v_850)"
        )

    p_mag = np.asarray(pressure.to("hPa").magnitude, dtype=np.float64)
    u_mag = np.asarray(u.magnitude, dtype=np.float64)
    v_mag = np.asarray(v.magnitude, dtype=np.float64)
    finite_mask = np.isfinite(p_mag) & np.isfinite(u_mag) & np.isfinite(v_mag)
    pressure = pressure[finite_mask]
    u = u[finite_mask]
    v = v[finite_mask]

    if u_llj is not None and v_llj is not None:
        llj_inverse = units.Quantity.from_list((-1 * u_llj, -1 * v_llj))
    elif np.max(pressure) >= units.Quantity(850, "hPa"):
        wind_magnitude = wind_speed(u, v)
        lowlevel_index = int(np.argmin(pressure >= units.Quantity(850, "hPa")))
        llj_index = int(np.argmax(wind_magnitude[:lowlevel_index]))
        llj_inverse = units.Quantity.from_list((-u[llj_index], -v[llj_index]))
    else:
        raise ValueError("Must specify low-level jet or specify pressure values below 850 hPa")

    bottom = pressure[0] if pressure[0] < units.Quantity(850, "hPa") else units.Quantity(850, "hPa")
    depth = (
        bottom - pressure[-1]
        if pressure[-1] > units.Quantity(300, "hPa")
        else units.Quantity(550, "hPa")
    )
    cloud_layer_winds = units.Quantity.from_list(
        mean_pressure_weighted(pressure, u, v, bottom=bottom, depth=depth)
    )
    upwind = cloud_layer_winds + llj_inverse
    downwind = 2 * cloud_layer_winds + llj_inverse
    return upwind, downwind


def friction_velocity(u, w, v=None, perturbation=False, axis=-1):
    """Compute friction velocity from wind component time series."""
    uw = _kinematic_flux_runtime(u, w, perturbation=perturbation, axis=axis)
    flux_sum = uw ** 2
    if v is not None:
        vw = _kinematic_flux_runtime(v, w, perturbation=perturbation, axis=axis)
        flux_sum = flux_sum + vw ** 2
    return np.sqrt(np.sqrt(flux_sum))


def tke(u, v, w, perturbation=False, axis=-1):
    """Compute turbulence kinetic energy."""
    if not perturbation:
        u = u - np.mean(u, axis=axis, keepdims=True)
        v = v - np.mean(v, axis=axis, keepdims=True)
        w = w - np.mean(w, axis=axis, keepdims=True)
    u_cont = np.mean(u ** 2, axis=axis)
    v_cont = np.mean(v ** 2, axis=axis)
    w_cont = np.mean(w ** 2, axis=axis)
    return 0.5 * (u_cont + v_cont + w_cont)


def gradient_richardson_number(height, potential_temperature, u, v, vertical_dim=0):
    """Calculate the gradient Richardson number."""
    gravity = units.Quantity(9.80665, "m/s^2")
    dthetadz = first_derivative(potential_temperature, x=height, axis=vertical_dim)
    dudz = first_derivative(u, x=height, axis=vertical_dim)
    dvdz = first_derivative(v, x=height, axis=vertical_dim)
    return (gravity / potential_temperature) * (dthetadz / (dudz ** 2 + dvdz ** 2))


# ============================================================================
# Kinematics (2-D gridded fields)
# ============================================================================

def _grid_shape(data):
    """Extract (ny, nx) from a 2-D array."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr.shape[0], arr.shape[1]
    raise ValueError(f"Expected 2-D array, got {arr.ndim}-D")


def _flat(data, unit=None):
    """Strip units and return a contiguous float64 array preserving shape."""
    if unit and hasattr(data, "magnitude"):
        arr = np.asarray(data.to(unit).magnitude, dtype=np.float64)
    elif hasattr(data, "magnitude"):
        arr = np.asarray(data.magnitude, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _mean_spacing(val, target_unit="m"):
    """Extract a scalar grid spacing from a scalar or array Quantity.

    If *val* is a 2-D array (e.g. from ``lat_lon_grid_deltas``), the mean
    value is returned so that the Rust function receives a single float.
    """
    if hasattr(val, "magnitude"):
        arr = np.asarray(val.to(target_unit).magnitude, dtype=np.float64)
    else:
        arr = np.asarray(val, dtype=np.float64)
    return float(arr.mean()) if arr.ndim > 0 and arr.size > 1 else float(arr)


def _is_variable_spacing(val):
    """Check if dx/dy is a 2D array (variable spacing, e.g., lat/lon grid)."""
    if hasattr(val, "magnitude"):
        arr = np.asarray(val.magnitude)
    else:
        arr = np.asarray(val)
    if arr.ndim < 2:
        return False
    # Check if values vary significantly (>5% relative range)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return False
    rng = finite.max() - finite.min()
    return rng > 0.05 * abs(finite.mean()) if abs(finite.mean()) > 1e-10 else rng > 1e-10


def _gradient_2d_variable(field, dx_2d, dy_2d):
    """Compute ∂f/∂x and ∂f/∂y with variable 2D grid spacing.

    Uses centered differences in the interior, one-sided at boundaries,
    matching MetPy's first_derivative behavior on variable-spaced grids.
    """
    arr = np.asarray(field, dtype=np.float64)
    ny, nx = arr.shape
    dx = np.asarray(dx_2d, dtype=np.float64)
    dy = np.asarray(dy_2d, dtype=np.float64)

    # Pad dx/dy to match field shape if needed (MetPy grid_deltas returns (ny, nx-1) and (ny-1, nx))
    if dx.shape != (ny, nx):
        if dx.shape == (ny, nx - 1):
            # Average adjacent to get (ny, nx)
            dx_full = np.empty((ny, nx))
            dx_full[:, 0] = dx[:, 0]
            dx_full[:, -1] = dx[:, -1]
            dx_full[:, 1:-1] = (dx[:, :-1] + dx[:, 1:]) / 2.0
            dx = dx_full
        else:
            dx = np.broadcast_to(dx, (ny, nx)).copy()
    if dy.shape != (ny, nx):
        if dy.shape == (ny - 1, nx):
            dy_full = np.empty((ny, nx))
            dy_full[0, :] = dy[0, :]
            dy_full[-1, :] = dy[-1, :]
            dy_full[1:-1, :] = (dy[:-1, :] + dy[1:, :]) / 2.0
            dy = dy_full
        else:
            dy = np.broadcast_to(dy, (ny, nx)).copy()

    # Replace zeros/near-zeros with NaN to avoid division by zero (poles)
    dx[np.abs(dx) < 1.0] = np.nan
    dy[np.abs(dy) < 1.0] = np.nan

    # ∂f/∂x — centered differences along axis=1
    dfdx = np.full_like(arr, np.nan)
    dfdx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0 * dx[:, 1:-1])
    dfdx[:, 0] = (arr[:, 1] - arr[:, 0]) / dx[:, 0]
    dfdx[:, -1] = (arr[:, -1] - arr[:, -2]) / dx[:, -1]

    # ∂f/∂y — centered differences along axis=0
    dfdy = np.full_like(arr, np.nan)
    dfdy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2.0 * dy[1:-1, :])
    dfdy[0, :] = (arr[1, :] - arr[0, :]) / dy[0, :]
    dfdy[-1, :] = (arr[-1, :] - arr[-2, :]) / dy[-1, :]

    return dfdx, dfdy


def _safe_unit_str(unit_obj):
    """Return a unit string usable with *our* registry, avoiding cross-registry ops."""
    return str(unit_obj)


def _first_derivative_variable(field, delta, axis):
    """Compute first derivative with variable spacing along an axis.

    Matches MetPy's first_derivative: centered differences in interior,
    one-sided at boundaries. *delta* is a 1-D or 2-D array of grid spacings
    (one fewer element than field along *axis*).
    """
    arr = np.asarray(field, dtype=np.float64)
    axis = axis % arr.ndim
    d = np.asarray(delta, dtype=np.float64).copy()
    # Replace near-zero spacings with NaN to avoid division by zero (e.g., at poles)
    d[np.abs(d) < 1.0] = np.nan
    n = arr.shape[axis]
    if n < 3:
        return np.gradient(arr, axis=axis)

    if d.size == 1:
        return np.gradient(arr, float(d.ravel()[0]), axis=axis)

    target_shape = list(arr.shape)
    target_shape[axis] = n - 1
    if d.ndim == 1:
        if d.size == n:
            d = np.diff(d)
        elif d.size != n - 1:
            return np.gradient(arr, float(np.mean(d)), axis=axis)
        reshape = [1] * arr.ndim
        reshape[axis] = d.shape[0]
        d = d.reshape(reshape)
    else:
        if d.shape[axis] == n:
            d = np.diff(d, axis=axis)
        elif d.shape[axis] != n - 1:
            return np.gradient(arr, float(np.mean(d)), axis=axis)
        try:
            d = np.broadcast_to(d, tuple(target_shape)).copy()
        except ValueError:
            return np.gradient(arr, float(np.mean(d)), axis=axis)

    def _axis_slice(spec):
        slices = [slice(None)] * arr.ndim
        slices[axis] = spec
        return tuple(slices)

    delta_slice0 = _axis_slice(slice(None, -1))
    delta_slice1 = _axis_slice(slice(1, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        combined_delta = d[delta_slice0] + d[delta_slice1]
        delta_diff = d[delta_slice1] - d[delta_slice0]

        center = (
            -d[delta_slice1] / (combined_delta * d[delta_slice0]) * arr[_axis_slice(slice(None, -2))]
            + delta_diff / (d[delta_slice0] * d[delta_slice1]) * arr[_axis_slice(slice(1, -1))]
            + d[delta_slice0] / (combined_delta * d[delta_slice1]) * arr[_axis_slice(slice(2, None))]
        )

        combined_delta = d[_axis_slice(slice(None, 1))] + d[_axis_slice(slice(1, 2))]
        big_delta = combined_delta + d[_axis_slice(slice(None, 1))]
        left = (
            -big_delta / (combined_delta * d[_axis_slice(slice(None, 1))]) * arr[_axis_slice(slice(None, 1))]
            + combined_delta / (d[_axis_slice(slice(None, 1))] * d[_axis_slice(slice(1, 2))]) * arr[_axis_slice(slice(1, 2))]
            - d[_axis_slice(slice(None, 1))] / (combined_delta * d[_axis_slice(slice(1, 2))]) * arr[_axis_slice(slice(2, 3))]
        )

        combined_delta = d[_axis_slice(slice(-2, -1))] + d[_axis_slice(slice(-1, None))]
        big_delta = combined_delta + d[_axis_slice(slice(-1, None))]
        right = (
            d[_axis_slice(slice(-1, None))] / (combined_delta * d[_axis_slice(slice(-2, -1))]) * arr[_axis_slice(slice(-3, -2))]
            - combined_delta / (d[_axis_slice(slice(-2, -1))] * d[_axis_slice(slice(-1, None))]) * arr[_axis_slice(slice(-2, -1))]
            + big_delta / (combined_delta * d[_axis_slice(slice(-1, None))]) * arr[_axis_slice(slice(-1, None))]
        )

    return np.concatenate((left, center, right), axis=axis)


def _get_scale_factors(data):
    """Extract map projection scale factors from xarray CRS metadata.

    Returns (parallel_scale, meridional_scale) as 2D arrays matching data shape,
    or (None, None) if no CRS is available.
    """
    if not hasattr(data, "metpy"):
        return None, None
    try:
        crs = data.metpy.cartopy_crs
    except Exception:
        return None, None

    lat_arr, lon_arr = _infer_lat_lon(data)
    if lat_arr is None:
        return None, None

    lat_2d = np.asarray(lat_arr.magnitude if hasattr(lat_arr, "magnitude") else lat_arr, dtype=np.float64)
    lon_2d = np.asarray(lon_arr.magnitude if hasattr(lon_arr, "magnitude") else lon_arr, dtype=np.float64)

    ny, nx = data.shape[-2:]
    if lat_2d.ndim == 1:
        lat_2d = np.broadcast_to(lat_2d[:, None], (ny, nx))
        lon_2d = np.broadcast_to(lon_2d[None, :], (ny, nx))

    try:
        from pyproj import Proj
        proj = Proj(crs)
        factors = proj.get_factors(lon_2d.ravel(), lat_2d.ravel())
        ps = np.asarray(factors.parallel_scale).reshape(ny, nx)
        ms = np.asarray(factors.meridional_scale).reshape(ny, nx)
        return ps, ms
    except Exception:
        # Fallback for lat/lon: parallel_scale = 1/cos(lat), meridional_scale = 1
        ps = 1.0 / np.cos(np.deg2rad(lat_2d))
        ms = np.ones_like(ps)
        return ps, ms


def _vector_derivative_corrected(u_arr, v_arr, dx, dy, parallel_scale, meridional_scale):
    """Compute map-projection-corrected vector derivative components.

    Returns (du_dy_corrected, dv_dx_corrected) for vorticity,
    or (du_dx_corrected, dv_dy_corrected) for divergence.
    """
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    # Cartesian derivatives with variable spacing
    du_dx = _first_derivative_variable(u_arr, dx_m, axis=-1)
    du_dy = _first_derivative_variable(u_arr, dy_m, axis=-2)
    dv_dx = _first_derivative_variable(v_arr, dx_m, axis=-1)
    dv_dy = _first_derivative_variable(v_arr, dy_m, axis=-2)

    ps = np.asarray(parallel_scale, dtype=np.float64)
    ms = np.asarray(meridional_scale, dtype=np.float64)

    # Scale factor derivatives
    dp_dy = _first_derivative_variable(ps, dy_m, axis=-2)
    dm_dx = _first_derivative_variable(ms, dx_m, axis=-1)

    # Map factor corrections (MetPy vector_derivative formula)
    dx_correction = ms / ps * dp_dy
    dy_correction = ps / ms * dm_dx

    du_dx_corr = ps * du_dx - v_arr * dx_correction
    du_dy_corr = ms * du_dy + v_arr * dy_correction
    dv_dx_corr = ps * dv_dx + u_arr * dx_correction
    dv_dy_corr = ms * dv_dy - u_arr * dy_correction

    return du_dx_corr, du_dy_corr, dv_dx_corr, dv_dy_corr


def divergence(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
               meridional_scale=None, latitude=None, longitude=None, crs=None):
    """Horizontal divergence on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if (dx is None or dy is None) and v is not None:
        dx, dy = _resolve_dx_dy(v, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("divergence requires dx/dy or inferable latitude/longitude coordinates")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)

    # Check for map projection scale factors (spherical corrections)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None

    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    # If we have scale factors, use full vector derivative with metric corrections
    if ps is not None and ms is not None:
        du_dx_corr, _, _, dv_dy_corr = _vector_derivative_corrected(
            u_arr, v_arr, dx, dy, ps, ms)
        result = du_dx_corr + dv_dy_corr
        return _wrap_result_like(u, result, "1/s")

    # Variable spacing without scale factors
    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        dudx = _first_derivative_variable(u_arr, dx_m, axis=-1)
        dvdy = _first_derivative_variable(v_arr, dy_m, axis=-2)
        result = dudx + dvdy
        return _wrap_result_like(u, result, "1/s")

    # Uniform grid: fast Rust path
    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    if u_arr.ndim == 2:
        result = np.asarray(_calc.divergence(np.ascontiguousarray(u_arr), np.ascontiguousarray(v_arr), dx_val, dy_val))
        return _wrap_result_like(u, result, "1/s")
    lead_shape = u_arr.shape[:-2]
    ny, nx = u_arr.shape[-2:]
    result = np.empty(u_arr.shape, dtype=np.float64)
    u_flat = u_arr.reshape((-1, ny, nx))
    v_flat = v_arr.reshape((-1, ny, nx))
    for idx in range(u_flat.shape[0]):
        result.reshape((-1, ny, nx))[idx] = np.asarray(
            _calc.divergence(np.ascontiguousarray(u_flat[idx]), np.ascontiguousarray(v_flat[idx]), dx_val, dy_val)
        ).reshape((ny, nx))
    return _wrap_result_like(u, result, "1/s")


def vorticity(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
              parallel_scale=None, meridional_scale=None,
              latitude=None, longitude=None, crs=None):
    """Relative vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("vorticity requires dx/dy or inferable latitude/longitude coordinates")
    if _gpu_uniform_grid_supported(
        u, v, dx=dx, dy=dy,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        latitude=latitude, longitude=longitude, crs=crs,
    ):
        result = _gpu_to_numpy(
            _load_gpu_calc().vorticity(
                _as_2d(u, "m/s"),
                _as_2d(v, "m/s"),
                dx=_mean_spacing(dx, "m"),
                dy=_mean_spacing(dy, "m"),
                x_dim=x_dim,
                y_dim=y_dim,
            )
        )
        return _wrap_result_like(u, result, "1/s")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)

    # Check for map projection scale factors (spherical corrections)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None

    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    # If we have scale factors, use full vector derivative with metric corrections
    if ps is not None and ms is not None:
        _, du_dy_corr, dv_dx_corr, _ = _vector_derivative_corrected(
            u_arr, v_arr, dx, dy, ps, ms)
        result = dv_dx_corr - du_dy_corr
        return _wrap_result_like(u, result, "1/s")

    # Variable spacing without scale factors (flat-Earth, variable grid)
    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        dvdx = _first_derivative_variable(v_arr, dx_m, axis=-1)
        dudy = _first_derivative_variable(u_arr, dy_m, axis=-2)
        result = dvdx - dudy
        return _wrap_result_like(u, result, "1/s")

    # Uniform grid: fast Rust path
    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    result = np.asarray(_calc.vorticity(np.ascontiguousarray(u_arr), np.ascontiguousarray(v_arr), dx_val, dy_val))
    return _wrap_result_like(u, result, "1/s")


def absolute_vorticity(u, v, lats=None, dx=None, dy=None, latitude=None,
                       longitude=None, x_dim=-1, y_dim=-2, crs=None,
                       parallel_scale=None, meridional_scale=None):
    """Absolute vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    lats : 2-D array (degrees)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    lat_source = lats if lats is not None else latitude
    if lat_source is None:
        lat_source, _ = _infer_lat_lon(u, latitude=latitude, longitude=longitude)
    if lat_source is None:
        raise TypeError("absolute_vorticity requires latitude or inferable coordinates")
    relative_vorticity = vorticity(
        u,
        v,
        dx=dx,
        dy=dy,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        latitude=lat_source,
        longitude=longitude,
        crs=crs,
    )
    return relative_vorticity + coriolis_parameter(lat_source)


def advection(scalar, *args, dx=None, dy=None, dz=None, x_dim=-1, y_dim=-2,
              vertical_dim=-3, parallel_scale=None, meridional_scale=None,
              latitude=None, longitude=None, crs=None):
    """Advection of a scalar field by a 2-D wind.

    Parameters
    ----------
    scalar : 2-D array Quantity
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (scalar_units / s)
    """
    if len(args) == 2:
        u, v = args
        w = None
    elif len(args) == 3:
        u, v, w = args
    elif len(args) == 4:
        u, v, dx_pos, dy_pos = args
        w = None
        dx = dx_pos if dx is None else dx
        dy = dy_pos if dy is None else dy
    elif len(args) == 5:
        u, v, w, dx_pos, dy_pos = args
        dx = dx_pos if dx is None else dx
        dy = dy_pos if dy is None else dy
    else:
        raise TypeError("advection expects (scalar, u, v, ...) with optional dx/dy or 3-D placeholders")

    if w is not None or dz is not None:
        raise NotImplementedError("3-D advection with w/dz is not yet supported in metrust.advection; use advection_3d")

    dx, dy = _resolve_dx_dy(scalar, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if (dx is None or dy is None) and u is not None:
        dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("advection requires dx/dy or inferable latitude/longitude coordinates")
    has_units = hasattr(scalar, "units")

    # Build output unit string safely
    if has_units:
        try:
            s_u = units.Unit(str(scalar.units))
        except Exception:
            s_u = units.dimensionless
        out_unit = str(s_u / units.s)
    else:
        out_unit = "1/s"

    # Check for variable spacing (lat/lon grids)
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        # Variable-spacing: compute gradients with full 2D dx/dy
        s_arr = _flat(scalar)
        u_arr = _flat(u, "m/s")
        v_arr = _flat(v, "m/s")
        dsdx = _first_derivative_variable(s_arr, dx_m, axis=-1)
        dsdy = _first_derivative_variable(s_arr, dy_m, axis=-2)
        result = -(u_arr * dsdx + v_arr * dsdy)
        return _wrap_result_like(scalar, result, out_unit)

    # Uniform spacing: fast Rust path
    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    result = np.asarray(_calc.advection(_flat(scalar), _flat(u, "m/s"), _flat(v, "m/s"), dx_val, dy_val))
    return _wrap_result_like(scalar, result, out_unit)


def frontogenesis(potential_temperature, u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                  *, parallel_scale=None, meridional_scale=None,
                  latitude=None, longitude=None, crs=None):
    """2-D Petterssen frontogenesis function.

    Parameters
    ----------
    potential_temperature : 2-D array Quantity (K)
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (K/m/s)
    """
    theta = potential_temperature
    dx, dy = _resolve_dx_dy(theta, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if (dx is None or dy is None) and u is not None:
        dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("frontogenesis requires dx/dy or inferable latitude/longitude coordinates")
    if _gpu_uniform_grid_supported(
        theta, u, v, dx=dx, dy=dy,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        latitude=latitude, longitude=longitude, crs=crs,
    ):
        result = _gpu_to_numpy(
            _load_gpu_calc().frontogenesis(
                _as_2d(theta, "K"),
                _as_2d(u, "m/s"),
                _as_2d(v, "m/s"),
                dx=_mean_spacing(dx, "m"),
                dy=_mean_spacing(dy, "m"),
                x_dim=x_dim,
                y_dim=y_dim,
            )
        )
        return _wrap_result_like(theta, result, "K/m/s")
    t_arr = np.asarray(_strip(theta, "K"), dtype=np.float64)
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        dtdx = _first_derivative_variable(t_arr, dx_m, axis=-1)
        dtdy = _first_derivative_variable(t_arr, dy_m, axis=-2)
        dudx = _first_derivative_variable(u_arr, dx_m, axis=-1)
        dvdy = _first_derivative_variable(v_arr, dy_m, axis=-2)
        dudy = _first_derivative_variable(u_arr, dy_m, axis=-2)
        dvdx = _first_derivative_variable(v_arr, dx_m, axis=-1)
        mag_t = np.sqrt(dtdx**2 + dtdy**2)
        shrd = dvdx + dudy
        strd = dudx - dvdy
        tdef = np.sqrt(strd**2 + shrd**2)
        div = dudx + dvdy
        psi = 0.5 * np.arctan2(shrd, strd)
        sin_beta = np.divide(
            dtdx * np.cos(psi) + dtdy * np.sin(psi),
            mag_t,
            out=np.zeros_like(mag_t),
            where=mag_t != 0,
        )
        result = 0.5 * mag_t * (tdef * (1.0 - 2.0 * sin_beta**2) - div)
        return _wrap_result_like(theta, result, "K/m/s")

    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    result = np.asarray(_calc.frontogenesis(_flat(theta, "K"), _flat(u, "m/s"), _flat(v, "m/s"), dx_val, dy_val))
    return _wrap_result_like(theta, result, "K/m/s")


def geostrophic_wind(heights, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2,
                     parallel_scale=None, meridional_scale=None,
                     longitude=None, crs=None):
    """Geostrophic wind from geopotential height.

    Parameters
    ----------
    heights : 2-D array Quantity (m)
    lats : 2-D array (degrees)
    dx, dy : Quantity (m)

    Returns
    -------
    tuple of (2-D array Quantity (m/s), 2-D array Quantity (m/s))
    """
    if latitude is not None and dx is not None and dy is not None:
        old_style = False
        try:
            old_style = (_can_convert(latitude, "m") and np.ndim(_coord_values(dx)) > 0)
        except Exception:
            old_style = False
        if old_style:
            latitude, dx, dy = dx, dy, latitude
    if latitude is None:
        latitude, _ = _infer_lat_lon(heights, latitude=latitude, longitude=longitude)
    dx, dy = _resolve_dx_dy(heights, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None or latitude is None:
        raise TypeError("geostrophic_wind requires latitude plus dx/dy or inferable coordinates")
    h_arr = np.asarray(_strip(heights, "m"), dtype=np.float64)
    lat_arr = np.asarray(latitude.magnitude if hasattr(latitude, "magnitude") else latitude, dtype=np.float64)
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        # Variable-spacing: compute geostrophic wind with full 2D dx/dy
        # u_g = -(g/f) * dZ/dy,  v_g = (g/f) * dZ/dx
        g = 9.80665
        omega = 7.2921159e-5
        if lat_arr.ndim == 2:
            lat_2d = lat_arr
        elif lat_arr.ndim == 1:
            lat_2d = np.broadcast_to(lat_arr[:, None], h_arr.shape)
        else:
            lat_2d = np.broadcast_to(lat_arr, h_arr.shape)
        f = 2.0 * omega * np.sin(np.deg2rad(lat_2d))
        f = np.where(np.abs(f) < 1e-10, np.nan, f)
        dzdx = _first_derivative_variable(h_arr, dx_m, axis=-1)
        dzdy = _first_derivative_variable(h_arr, dy_m, axis=-2)
        u_g = -(g / f) * dzdy
        v_g = (g / f) * dzdx
        return _wrap_result_like(heights, u_g, "m/s"), _wrap_result_like(heights, v_g, "m/s")

    h_f = _as_2d(heights, "m")
    if lat_arr.ndim == 0:
        lat_input = np.broadcast_to(lat_arr, h_arr.shape)
    else:
        lat_input = latitude
    lats_f = _as_2d(lat_input, "degree") if hasattr(lat_input, "magnitude") else _as_2d(lat_input)
    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    u_g, v_g = _calc.geostrophic_wind(h_f, lats_f, dx_val, dy_val)
    ms = units("m/s")
    return np.asarray(u_g) * ms, np.asarray(v_g) * ms


def _is_dataarray_like(value):
    return xr is not None and isinstance(value, xr.DataArray)


def _dataarray_unit_array(data, unit=None):
    if hasattr(data, "metpy"):
        try:
            arr = data.metpy.unit_array
            return arr.to(unit) if unit is not None and hasattr(arr, "to") else arr
        except Exception:
            pass
    raw = data.data if hasattr(data, "data") else data
    if hasattr(raw, "to"):
        return raw.to(unit) if unit is not None else raw
    values = np.asarray(data.values if hasattr(data, "values") else data, dtype=np.float64)
    unit_name = None
    if hasattr(data, "attrs"):
        unit_name = data.attrs.get("units")
    if unit_name:
        quantity = values * units(unit_name)
        return quantity.to(unit) if unit is not None else quantity
    return values * units(unit) if unit is not None else values


def _cross_section_index_name(cross, index):
    if isinstance(index, int):
        return cross.dims[index]
    return index


def _cross_section_distance_coords(cross):
    coords = getattr(cross, "coords", {})
    if "longitude" in coords and "latitude" in coords:
        lon_coord = coords["longitude"]
        lat_coord = coords["latitude"]
        lon = _dataarray_unit_array(lon_coord, "degree").to("degree").magnitude
        lat = _dataarray_unit_array(lat_coord, "degree").to("degree").magnitude
        try:
            crs = cross.metpy.pyproj_crs
        except Exception:
            from pyproj import CRS
            crs = CRS.from_epsg(4326)
        geod = crs.get_geod()
        forward_az, _, distance = geod.inv(
            lon[0] * np.ones_like(lon),
            lat[0] * np.ones_like(lat),
            lon,
            lat,
        )
        x = xr.DataArray(
            units.Quantity(distance * np.sin(np.deg2rad(forward_az)), "meter"),
            coords=lon_coord.coords,
            dims=lon_coord.dims,
        )
        y = xr.DataArray(
            units.Quantity(distance * np.cos(np.deg2rad(forward_az)), "meter"),
            coords=lat_coord.coords,
            dims=lat_coord.dims,
        )
        return x, y
    if "x" in coords and "y" in coords:
        x_coord = coords["x"]
        y_coord = coords["y"]
        x = xr.DataArray(_dataarray_unit_array(x_coord, "meter"), coords=x_coord.coords, dims=x_coord.dims)
        y = xr.DataArray(_dataarray_unit_array(y_coord, "meter"), coords=y_coord.coords, dims=y_coord.dims)
        return x, y
    raise AttributeError("Sufficient horizontal coordinates not defined.")


def _cross_section_latitude_coord(cross):
    coords = getattr(cross, "coords", {})
    if "latitude" in coords:
        latitude = coords["latitude"]
        return xr.DataArray(
            _dataarray_unit_array(latitude, "degree"),
            coords=latitude.coords,
            dims=latitude.dims,
        )
    if "lat" in coords:
        latitude = coords["lat"]
        return xr.DataArray(
            _dataarray_unit_array(latitude, "degree"),
            coords=latitude.coords,
            dims=latitude.dims,
        )
    if hasattr(cross, "metpy"):
        try:
            from pyproj import Proj
            y = cross.metpy.y
            latitude = Proj(cross.metpy.pyproj_crs)(
                cross.metpy.x.values,
                y.values,
                inverse=True,
                radians=False,
            )[1]
            return xr.DataArray(units.Quantity(latitude, "degrees_north"), coords=y.coords,
                                dims=y.dims)
        except Exception:
            pass
    raise AttributeError("Latitude coordinates are required for absolute_momentum.")


def _cross_section_unit_vectors(cross, index="index"):
    x, y = _cross_section_distance_coords(cross)
    index_name = _cross_section_index_name(cross, index)
    index_coord = cross.coords[index_name]
    index_values = _coord_values(index_coord)
    if index_values.ndim != 1 or index_values.size != x.sizes[index_name]:
        index_values = np.arange(x.sizes[index_name], dtype=np.float64)
    dx_di = first_derivative(_dataarray_unit_array(x, "meter"), x=index_values, axis=0).to("")
    dy_di = first_derivative(_dataarray_unit_array(y, "meter"), x=index_values, axis=0).to("")
    tangent_mag = np.hypot(dx_di.magnitude, dy_di.magnitude)
    unit_tangent = np.vstack([dx_di.magnitude / tangent_mag, dy_di.magnitude / tangent_mag]) * units.dimensionless
    unit_normal = np.vstack([-dy_di.magnitude / tangent_mag, dx_di.magnitude / tangent_mag]) * units.dimensionless
    return unit_tangent, unit_normal


def ageostrophic_wind(height, u=None, v=None, dx=None, dy=None, latitude=None,
                      x_dim=-1, y_dim=-2, *, parallel_scale=None,
                      meridional_scale=None, longitude=None, crs=None):
    """Ageostrophic wind: total wind minus geostrophic wind.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    heights : 2-D array Quantity (m)
    lats : 2-D array (degrees)
    dx, dy : Quantity (m)

    Returns
    -------
    tuple of (2-D array Quantity (m/s), 2-D array Quantity (m/s))
    """
    old_style = False
    try:
        old_style = (
            u is not None
            and v is not None
            and dx is not None
            and dy is not None
            and latitude is not None
            and _can_convert(v, "m")
            and _can_convert(dx, "degree")
            and _can_convert(dy, "m")
            and _can_convert(latitude, "m")
        )
    except Exception:
        old_style = False
    if old_style:
        total_u, total_v = height, u
        height, u, v = v, total_u, total_v
        latitude, dx, dy = dx, dy, latitude

    u_geostrophic, v_geostrophic = geostrophic_wind(
        height,
        dx=dx,
        dy=dy,
        latitude=latitude,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        longitude=longitude,
        crs=crs,
    )
    return u - u_geostrophic, v - v_geostrophic


def potential_vorticity_baroclinic(potential_temp, pressure, *args, dx=None, dy=None,
                                   latitude=None, x_dim=-1, y_dim=-2, vertical_dim=-3,
                                   longitude=None, crs=None, parallel_scale=None,
                                   meridional_scale=None):
    """Baroclinic (Ertel) potential vorticity on a 2-D isobaric slice.

    Parameters
    ----------
    potential_temp : 2-D array Quantity (K)
    pressure : length-2 sequence Quantity (Pa)
        [p_below, p_above]
    theta_below : 2-D array Quantity (K)
    theta_above : 2-D array Quantity (K)
    u, v : 2-D array Quantity (m/s)
    lats : 2-D array (degrees)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (K*m^2/(kg*s))
    """
    if len(args) == 2:
        u, v = args
        dx, dy = _resolve_dx_dy(potential_temp, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
        if (dx is None or dy is None) and u is not None:
            dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
        if latitude is None:
            latitude, _ = _infer_lat_lon(potential_temp, latitude=latitude, longitude=longitude)
            if latitude is None:
                latitude, _ = _infer_lat_lon(u, latitude=latitude, longitude=longitude)
        if dx is None or dy is None or latitude is None:
            raise TypeError("potential_vorticity_baroclinic requires latitude plus dx/dy or inferable coordinates")
        try:
            theta_arr, theta_unit = _as_array_with_unit(potential_temp, "kelvin")
        except Exception:
            theta_arr = np.asarray(getattr(potential_temp, "values", potential_temp), dtype=np.float64)
            theta_unit = units.kelvin
        try:
            u_arr, _ = _as_array_with_unit(u, "m/s")
        except Exception:
            u_arr = np.asarray(getattr(u, "values", u), dtype=np.float64)
        try:
            v_arr, _ = _as_array_with_unit(v, "m/s")
        except Exception:
            v_arr = np.asarray(getattr(v, "values", v), dtype=np.float64)
        pressure_arr, _ = _as_array_with_unit(pressure, "Pa")
        theta_arr = np.asarray(theta_arr, dtype=np.float64)
        u_arr = np.asarray(u_arr, dtype=np.float64)
        v_arr = np.asarray(v_arr, dtype=np.float64)
        pressure_arr = np.asarray(pressure_arr, dtype=np.float64)
        vertical_axis = vertical_dim % theta_arr.ndim
        vertical_size = theta_arr.shape[vertical_axis]
        if vertical_size < 3:
            raise ValueError(
                f"Length of potential temperature along the vertical axis {vertical_dim} must be at least 3."
            )
        if pressure_arr.ndim == 1:
            pressure_levels = pressure_arr
        else:
            pressure_levels = np.moveaxis(pressure_arr, vertical_axis, 0).reshape(vertical_size, -1)[:, 0]

        dudy, dvdx = vector_derivative(
            u_arr * units("m/s"),
            v_arr * units("m/s"),
            dx=dx,
            dy=dy,
            x_dim=x_dim,
            y_dim=y_dim,
            parallel_scale=parallel_scale,
            meridional_scale=meridional_scale,
            latitude=latitude,
            longitude=longitude,
            crs=crs,
            return_only=("du/dy", "dv/dx"),
        )
        avor = dvdx - dudy + coriolis_parameter(latitude)
        if (
            theta_arr.shape[y_dim % theta_arr.ndim] == 1
            and theta_arr.shape[x_dim % theta_arr.ndim] == 1
        ):
            dthetadx = np.zeros_like(theta_arr) * ((theta_unit or units.kelvin) / units.m)
            dthetady = np.zeros_like(theta_arr) * ((theta_unit or units.kelvin) / units.m)
        else:
            dthetadx, dthetady = geospatial_gradient(
                theta_arr * (theta_unit or units.kelvin),
                dx=dx,
                dy=dy,
                x_dim=x_dim,
                y_dim=y_dim,
                parallel_scale=parallel_scale,
                meridional_scale=meridional_scale,
                latitude=latitude,
                longitude=longitude,
                crs=crs,
            )
        dthetadp = np.gradient(theta_arr, pressure_levels, axis=vertical_axis, edge_order=2) * (
            (theta_unit or units.kelvin) / units.Pa
        )
        dudp = np.gradient(u_arr, pressure_levels, axis=vertical_axis, edge_order=2) * units("m/s/Pa")
        dvdp = np.gradient(v_arr, pressure_levels, axis=vertical_axis, edge_order=2) * units("m/s/Pa")
        result = (
            -units.Quantity(9.80665, "m/s**2")
            * (dudp * dthetady - dvdp * dthetadx + avor * dthetadp)
        ).to("K * m**2 / (s * kg)")
        if _is_dataarray_like(potential_temp):
            return _wrap_result_like(potential_temp, result.magnitude, "K*m**2/(kg*s)")
        return result
    if len(args) != 7:
        raise TypeError("potential_vorticity_baroclinic expects either (theta, pressure, u, v, ...) or the legacy 2-D form")
    theta_below, theta_above, u, v, lats, dx_pos, dy_pos = args
    dx = dx_pos if dx is None else dx
    dy = dy_pos if dy is None else dy
    pt_f = _as_2d(potential_temp, "K")
    tb_f = _as_2d(theta_below, "K")
    ta_f = _as_2d(theta_above, "K")
    u_f = _as_2d(u, "m/s")
    v_f = _as_2d(v, "m/s")
    lats_f = _as_2d(lats, "degree") if hasattr(lats, "magnitude") else _as_2d(lats)
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    p_arr = np.asarray(_strip(pressure, "Pa"), dtype=np.float64) if hasattr(pressure, "magnitude") else np.asarray(pressure, dtype=np.float64)
    result = np.asarray(_calc.potential_vorticity_baroclinic(pt_f, p_arr, tb_f, ta_f, u_f, v_f, lats_f, dx_val, dy_val))
    return result * units("K*m**2/(kg*s)")


def potential_vorticity_barotropic(height, u, v, dx=None, dy=None, latitude=None, x_dim=-1,
                                   y_dim=-2, *, parallel_scale=None, meridional_scale=None,
                                   longitude=None, crs=None):
    """Barotropic potential vorticity.

    Parameters
    ----------
    heights : 2-D array Quantity (m)
    u, v : 2-D array Quantity (m/s)
    lats : 2-D array (degrees)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/(m*s))
    """
    old_style = False
    try:
        old_style = (
            latitude is not None
            and _can_convert(dx, "degree")
            and _can_convert(dy, "m")
            and _can_convert(latitude, "m")
        )
    except Exception:
        old_style = False
    if old_style:
        latitude, dx, dy = dx, dy, latitude
    avor = absolute_vorticity(
        u,
        v,
        dx=dx,
        dy=dy,
        latitude=latitude,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        longitude=longitude,
        crs=crs,
    )
    return (avor / height).to("1/(m*s)")


def normal_component(data_x, data_y, index="index", *legacy_args):
    """Normal (perpendicular) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    array Quantity (m/s)
    """
    if _is_dataarray_like(data_x) and _is_dataarray_like(data_y) and not legacy_args:
        data_x = data_x.metpy.quantify() if hasattr(data_x, "metpy") else data_x
        data_y = data_y.metpy.quantify() if hasattr(data_y, "metpy") else data_y
        _, unit_norm = _cross_section_unit_vectors(data_x, index=index)
        component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]
        if "grid_mapping" in data_x.attrs:
            component_norm.attrs["grid_mapping"] = data_x.attrs["grid_mapping"]
        return component_norm
    start = index
    end = legacy_args[0] if legacy_args else None
    if end is None:
        raise TypeError("normal_component requires either (data_x, data_y, index='index') or legacy (u, v, start, end)")
    u_arr = _as_1d(_strip(data_x, "m/s"))
    v_arr = _as_1d(_strip(data_y, "m/s"))
    result = np.array(_calc.normal_component(u_arr, v_arr, start, end))
    return result * units("m/s")


def tangential_component(data_x, data_y, index="index", *legacy_args):
    """Tangential (parallel) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    array Quantity (m/s)
    """
    if _is_dataarray_like(data_x) and _is_dataarray_like(data_y) and not legacy_args:
        data_x = data_x.metpy.quantify() if hasattr(data_x, "metpy") else data_x
        data_y = data_y.metpy.quantify() if hasattr(data_y, "metpy") else data_y
        unit_tang, _ = _cross_section_unit_vectors(data_x, index=index)
        component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]
        if "grid_mapping" in data_x.attrs:
            component_tang.attrs["grid_mapping"] = data_x.attrs["grid_mapping"]
        return component_tang
    start = index
    end = legacy_args[0] if legacy_args else None
    if end is None:
        raise TypeError("tangential_component requires either (data_x, data_y, index='index') or legacy (u, v, start, end)")
    u_arr = _as_1d(_strip(data_x, "m/s"))
    v_arr = _as_1d(_strip(data_y, "m/s"))
    result = np.array(_calc.tangential_component(u_arr, v_arr, start, end))
    return result * units("m/s")


def unit_vectors_from_cross_section(cross, end=None, index="index"):
    """Tangent and normal unit vectors for a cross-section line.

    Parameters
    ----------
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    tuple of ((east, north), (east, north))
        Tangent and normal unit vector components.
    """
    if end is None and _is_dataarray_like(cross):
        return _cross_section_unit_vectors(cross, index=index)
    if end is None:
        raise TypeError("unit_vectors_from_cross_section requires either (cross, index='index') or legacy (start, end)")
    return _calc.unit_vectors_from_cross_section(cross, end)


def vector_derivative(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
                      meridional_scale=None, return_only=None, latitude=None,
                      longitude=None, crs=None):
    """All four partial derivatives of a 2-D vector field.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    tuple of four 2-D arrays Quantity (1/s)
        (du/dx, du/dy, dv/dx, dv/dy)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("vector_derivative requires dx/dy or inferable latitude/longitude coordinates")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)
    if ps is not None and ms is not None:
        dudx, dudy, dvdx, dvdy = _vector_derivative_corrected(u_arr, v_arr, dx, dy, ps, ms)
    elif _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        dudx = _first_derivative_variable(u_arr, dx_m, axis=x_dim)
        dudy = _first_derivative_variable(u_arr, dy_m, axis=y_dim)
        dvdx = _first_derivative_variable(v_arr, dx_m, axis=x_dim)
        dvdy = _first_derivative_variable(v_arr, dy_m, axis=y_dim)
    elif u_arr.ndim > 2 or v_arr.ndim > 2:
        x_axis = x_dim % u_arr.ndim
        y_axis = y_dim % u_arr.ndim
        dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
        dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
        dudx = np.gradient(u_arr, dx_val, axis=x_axis, edge_order=2)
        dudy = np.gradient(u_arr, dy_val, axis=y_axis, edge_order=2)
        dvdx = np.gradient(v_arr, dx_val, axis=x_axis, edge_order=2)
        dvdy = np.gradient(v_arr, dy_val, axis=y_axis, edge_order=2)
    else:
        dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
        dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
        dudx, dudy, dvdx, dvdy = _calc.vector_derivative(
            np.ascontiguousarray(u_arr),
            np.ascontiguousarray(v_arr),
            dx_val,
            dy_val,
        )
    derivatives = {
        "du/dx": _wrap_result_like(u, dudx, "1/s"),
        "du/dy": _wrap_result_like(u, dudy, "1/s"),
        "dv/dx": _wrap_result_like(v, dvdx, "1/s"),
        "dv/dy": _wrap_result_like(v, dvdy, "1/s"),
    }
    if return_only is None:
        return (
            derivatives["du/dx"],
            derivatives["du/dy"],
            derivatives["dv/dx"],
            derivatives["dv/dy"],
        )
    if isinstance(return_only, str):
        return derivatives[return_only]
    return tuple(derivatives[component] for component in return_only)


def absolute_momentum(u, v=None, index="index"):
    """Absolute momentum.

    Parameters
    ----------
    u : array Quantity (m/s)
    lats : array (degrees)
    y_distances : array Quantity (m)

    Returns
    -------
    array Quantity (m/s)
    """
    if _is_dataarray_like(u) and _is_dataarray_like(v):
        u = u.metpy.quantify() if hasattr(u, "metpy") else u
        v = v.metpy.quantify() if hasattr(v, "metpy") else v
        norm_wind = normal_component(u, v, index=index)
        latitude = _cross_section_latitude_coord(norm_wind)
        _, latitude = xr.broadcast(norm_wind, latitude)
        f = coriolis_parameter(latitude)
        x, y = _cross_section_distance_coords(norm_wind)
        distance_q = np.hypot(_dataarray_unit_array(x, "meter"), _dataarray_unit_array(y, "meter"))
        distance = xr.DataArray(distance_q, coords=x.coords, dims=x.dims)
        _, distance = xr.broadcast(norm_wind, distance)
        result = norm_wind + f * distance
        return result.metpy.convert_units("m/s") if hasattr(result, "metpy") else result
    lats = v
    y_distances = index
    u_arr = _as_1d(_strip(u, "m/s"))
    lat_arr = _as_1d(_strip(lats, "degree")) if hasattr(lats, "magnitude") else _as_1d(lats)
    yd = _as_1d(_strip(y_distances, "m"))
    result = np.asarray(_calc.absolute_momentum(u_arr, lat_arr, yd))
    return result * units("m/s")


def coriolis_parameter(latitude):
    """Coriolis parameter.

    Parameters
    ----------
    latitude : Quantity (degree) or float

    Returns
    -------
    Quantity (1/s)
    """
    lat_stripped = _strip(latitude, "degree") if hasattr(latitude, "magnitude") else latitude
    lat_arr = np.asarray(lat_stripped, dtype=np.float64)
    result = 2.0 * 7.292115e-5 * np.sin(np.deg2rad(lat_arr))
    if _is_dataarray_like(latitude):
        return xr.DataArray(
            np.asarray(result, dtype=np.float64) * units("1/s"),
            coords=latitude.coords,
            dims=latitude.dims,
            attrs=dict(latitude.attrs),
        )
    return _wrap_result_like(latitude, result, "1/s")


def cross_section_components(data_x, data_y, index="index", *legacy_args):
    """Decompose wind into parallel and perpendicular cross-section components.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    start_lat, start_lon : float (degrees)
    end_lat, end_lon : float (degrees)

    Returns
    -------
    tuple of (array Quantity (m/s), array Quantity (m/s))
        (parallel, perpendicular) components.
    """
    if _is_dataarray_like(data_x) and _is_dataarray_like(data_y) and not legacy_args:
        data_x = data_x.metpy.quantify() if hasattr(data_x, "metpy") else data_x
        data_y = data_y.metpy.quantify() if hasattr(data_y, "metpy") else data_y
        unit_tang, unit_norm = _cross_section_unit_vectors(data_x, index=index)
        component_tang = data_x * unit_tang[0] + data_y * unit_tang[1]
        component_norm = data_x * unit_norm[0] + data_y * unit_norm[1]
        return component_tang, component_norm
    if len(legacy_args) != 3:
        raise TypeError(
            "cross_section_components requires either (data_x, data_y, index='index') "
            "or legacy (u, v, start_lat, start_lon, end_lat, end_lon)"
        )
    start_lat = index
    start_lon, end_lat, end_lon = legacy_args
    u_arr = _as_1d(_strip(data_x, "m/s"))
    v_arr = _as_1d(_strip(data_y, "m/s"))
    slat = _as_float(_strip(start_lat, "degree")) if hasattr(start_lat, "magnitude") else float(start_lat)
    slon = _as_float(_strip(start_lon, "degree")) if hasattr(start_lon, "magnitude") else float(start_lon)
    elat = _as_float(_strip(end_lat, "degree")) if hasattr(end_lat, "magnitude") else float(end_lat)
    elon = _as_float(_strip(end_lon, "degree")) if hasattr(end_lon, "magnitude") else float(end_lon)
    par, perp = _calc.cross_section_components(u_arr, v_arr, slat, slon, elat, elon)
    ms = units("m/s")
    return np.asarray(par) * ms, np.asarray(perp) * ms


def curvature_vorticity(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
                        meridional_scale=None, latitude=None, longitude=None, crs=None):
    """Curvature vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u,
        v,
        dx=dx,
        dy=dy,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        latitude=latitude,
        longitude=longitude,
        crs=crs,
    )
    return (u * u * dvdx - v * v * dudy - v * u * dudx + u * v * dvdy) / (u ** 2 + v ** 2)


def inertial_advective_wind(u, v, u_geostrophic, v_geostrophic, dx=None, dy=None,
                            latitude=None, x_dim=-1, y_dim=-2, *,
                            parallel_scale=None, meridional_scale=None,
                            longitude=None, crs=None):
    """Inertial-advective wind.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    u_geo, v_geo : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    tuple of (2-D array Quantity (m/s), 2-D array Quantity (m/s))
    """
    if latitude is None:
        latitude, _ = _infer_lat_lon(u, latitude=latitude, longitude=longitude)
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None or latitude is None:
        raise TypeError("inertial_advective_wind requires latitude plus dx/dy or inferable coordinates")
    f = coriolis_parameter(latitude)
    dugdx, dugdy = geospatial_gradient(
        u_geostrophic,
        dx=dx,
        dy=dy,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        latitude=latitude,
        longitude=longitude,
        crs=crs,
    )
    dvgdx, dvgdy = geospatial_gradient(
        v_geostrophic,
        dx=dx,
        dy=dy,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        latitude=latitude,
        longitude=longitude,
        crs=crs,
    )
    return -(u * dvgdx + v * dvgdy) / f, (u * dugdx + v * dugdy) / f


def kinematic_flux(vel, b, perturbation=False, axis=-1):
    """Kinematic flux (element-wise product).

    Parameters
    ----------
    v_component : array Quantity (m/s)
    scalar : array Quantity or array-like

    Returns
    -------
    array (product units)
    """
    flux = np.mean(vel * b, axis=axis)
    if not perturbation:
        flux = flux - np.mean(vel, axis=axis) * np.mean(b, axis=axis)
    if hasattr(flux, "units"):
        return units.Quantity(np.atleast_1d(np.asarray(flux.magnitude, dtype=np.float64)), flux.units)
    return np.atleast_1d(np.asarray(flux, dtype=np.float64))


def q_vector(u, v, temperature, pressure, dx=None, dy=None, **kwargs):
    """Q-vector on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    temperature : 2-D array Quantity (K or degC)
    pressure : Quantity (pressure, hPa)
    dx, dy : Quantity (m)
    **kwargs : dict
        Accepts x_dim, y_dim for MetPy compatibility (ignored).

    Returns
    -------
    tuple of (2-D array, 2-D array)
    """
    latitude = kwargs.get("latitude")
    longitude = kwargs.get("longitude")
    dx, dy = _resolve_dx_dy(temperature, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if (dx is None or dy is None) and u is not None:
        dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("q_vector requires dx/dy or inferable latitude/longitude coordinates")
    if _gpu_uniform_grid_supported(
        u, v, temperature, dx=dx, dy=dy, latitude=latitude, longitude=longitude,
    ) and all(key in {"x_dim", "y_dim", "latitude", "longitude"} for key in kwargs):
        q1, q2 = _gpu_to_numpy(
            _load_gpu_calc().q_vector(
                _as_2d(u, "m/s"),
                _as_2d(v, "m/s"),
                _as_2d(temperature, "K") if _can_convert(temperature, "K") else _as_2d(temperature, "degC"),
                _as_float(_strip(pressure, "hPa")),
                dx=_mean_spacing(dx, "m"),
                dy=_mean_spacing(dy, "m"),
                **kwargs,
            )
        )
        return _wrap_result_like(u, q1), _wrap_result_like(v, q2)
    t_2d = _as_2d(temperature, "K") if _can_convert(temperature, "K") else _as_2d(temperature, "degC")
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    p_val = _as_float(_strip(pressure, "hPa"))
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    q1, q2 = _calc.q_vector(t_2d, u_2d, v_2d, p_val, dx_val, dy_val)
    return _wrap_result_like(u, q1, "m**2/(kg*s)"), _wrap_result_like(v, q2, "m**2/(kg*s)")


def shear_vorticity(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
                    meridional_scale=None, latitude=None, longitude=None, crs=None):
    """Shear vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u,
        v,
        dx=dx,
        dy=dy,
        x_dim=x_dim,
        y_dim=y_dim,
        parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
        latitude=latitude,
        longitude=longitude,
        crs=crs,
    )
    return (v * u * dudx + v * v * dvdx - u * u * dudy - u * v * dvdy) / (u ** 2 + v ** 2)


def shearing_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                         parallel_scale=None, meridional_scale=None,
                         latitude=None, longitude=None, crs=None):
    """Shearing deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("shearing_deformation requires dx/dy or inferable latitude/longitude coordinates")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None

    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if ps is not None and ms is not None:
        _, du_dy_corr, dv_dx_corr, _ = _vector_derivative_corrected(u_arr, v_arr, dx, dy, ps, ms)
        return _wrap_result_like(u, dv_dx_corr + du_dy_corr, "1/s")
    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        result = _first_derivative_variable(v_arr, dx_m, axis=-1) + _first_derivative_variable(u_arr, dy_m, axis=-2)
        return _wrap_result_like(u, result, "1/s")

    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.shearing_deformation(_as_2d(u, "m/s"), _as_2d(v, "m/s"), dx_val, dy_val))
    return _wrap_result_like(u, result, "1/s")


def stretching_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                           parallel_scale=None, meridional_scale=None,
                           latitude=None, longitude=None, crs=None):
    """Stretching deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("stretching_deformation requires dx/dy or inferable latitude/longitude coordinates")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None

    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if ps is not None and ms is not None:
        du_dx_corr, _, _, dv_dy_corr = _vector_derivative_corrected(u_arr, v_arr, dx, dy, ps, ms)
        return _wrap_result_like(u, du_dx_corr - dv_dy_corr, "1/s")
    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        result = _first_derivative_variable(u_arr, dx_m, axis=-1) - _first_derivative_variable(v_arr, dy_m, axis=-2)
        return _wrap_result_like(u, result, "1/s")

    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.stretching_deformation(_as_2d(u, "m/s"), _as_2d(v, "m/s"), dx_val, dy_val))
    return _wrap_result_like(u, result, "1/s")


def total_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                      parallel_scale=None, meridional_scale=None,
                      latitude=None, longitude=None, crs=None):
    """Total deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("total_deformation requires dx/dy or inferable latitude/longitude coordinates")
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(u)
    else:
        ps = np.asarray(parallel_scale, dtype=np.float64) if parallel_scale is not None else None
        ms = np.asarray(meridional_scale, dtype=np.float64) if meridional_scale is not None else None

    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if ps is not None and ms is not None:
        du_dx_corr, du_dy_corr, dv_dx_corr, dv_dy_corr = _vector_derivative_corrected(
            u_arr, v_arr, dx, dy, ps, ms)
        strd = du_dx_corr - dv_dy_corr
        shrd = dv_dx_corr + du_dy_corr
        return _wrap_result_like(u, np.sqrt(strd**2 + shrd**2), "1/s")
    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        dudx = _first_derivative_variable(u_arr, dx_m, axis=-1)
        dvdy = _first_derivative_variable(v_arr, dy_m, axis=-2)
        dudy = _first_derivative_variable(u_arr, dy_m, axis=-2)
        dvdx = _first_derivative_variable(v_arr, dx_m, axis=-1)
        strd = dudx - dvdy
        shrd = dvdx + dudy
        return _wrap_result_like(u, np.sqrt(strd**2 + shrd**2), "1/s")

    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.total_deformation(_as_2d(u, "m/s"), _as_2d(v, "m/s"), dx_val, dy_val))
    return _wrap_result_like(u, result, "1/s")


def geospatial_gradient(f, *args, dx=None, dy=None, x_dim=-1, y_dim=-2,
                        parallel_scale=None, meridional_scale=None,
                        return_only=None, latitude=None, longitude=None, crs=None):
    """Gradient of a scalar field on a lat/lon grid.

    Parameters
    ----------
    data : 2-D array
    lats : 2-D array (degrees)
    lons : 2-D array (degrees)

    Returns
    -------
    tuple of (2-D array, 2-D array)
        (df/dx, df/dy) in physical units (per meter).
    """
    if len(args) == 2 and dx is None and dy is None and latitude is None and longitude is None:
        latitude, longitude = args
    if dx is None or dy is None:
        dx, dy = _resolve_dx_dy(f, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("geospatial_gradient requires dx/dy or inferable latitude/longitude coordinates")
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(f)
        parallel_scale = ps if ps is not None else parallel_scale
        meridional_scale = ms if ms is not None else meridional_scale
    derivatives = {
        component: None
        for component in ("df/dx", "df/dy")
        if return_only is None or component in return_only
    }
    scales = {"df/dx": parallel_scale, "df/dy": meridional_scale}
    map_factor_correction = parallel_scale is not None and meridional_scale is not None
    f_arr, f_unit = _as_array_with_unit(f)
    f_arr = np.asarray(f_arr, dtype=np.float64)
    unit_str = str((f_unit or units.dimensionless) / units.m)
    for component in derivatives:
        delta, dim = (dx, x_dim) if component.endswith("dx") else (dy, y_dim)
        delta_arr = np.asarray(delta.to("m").magnitude if hasattr(delta, "to") else delta, dtype=np.float64)
        axis = dim % f_arr.ndim
        if delta_arr.size == 1:
            spacing = float(delta_arr.mean())
            values = np.gradient(f_arr, spacing, axis=axis, edge_order=2)
        else:
            values = _first_derivative_variable(f_arr, delta_arr, axis=axis)
        derivatives[component] = _wrap_result_like(f, values, unit_str)
        if map_factor_correction:
            derivatives[component] = derivatives[component] * scales[component]
    if return_only is None:
        return derivatives["df/dx"], derivatives["df/dy"]
    if isinstance(return_only, str):
        return derivatives[return_only]
    return tuple(derivatives[component] for component in return_only)


def geospatial_laplacian(f, *args, dx=None, dy=None, x_dim=-1, y_dim=-2,
                         parallel_scale=None, meridional_scale=None,
                         latitude=None, longitude=None, crs=None):
    """Laplacian of a scalar field on a lat/lon grid.

    Parameters
    ----------
    data : 2-D array
    lats : 2-D array (degrees)
    lons : 2-D array (degrees)

    Returns
    -------
    2-D array
    """
    if len(args) == 2 and dx is None and dy is None and latitude is None and longitude is None:
        latitude, longitude = args
    if dx is None or dy is None:
        dx, dy = _resolve_dx_dy(f, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("geospatial_laplacian requires dx/dy or inferable latitude/longitude coordinates")
    if hasattr(f, "magnitude"):
        f_arr = np.asarray(f.magnitude, dtype=np.float64)
    elif _is_dataarray_like(f):
        f_arr = np.asarray(f.values, dtype=np.float64)
    else:
        f_arr = np.asarray(f, dtype=np.float64)
    if parallel_scale is None and meridional_scale is None:
        ps, ms = _get_scale_factors(f)
        parallel_scale = ps if ps is not None else parallel_scale
        meridional_scale = ms if ms is not None else meridional_scale
    if parallel_scale is not None and meridional_scale is not None:
        grad_u, grad_v = geospatial_gradient(
            f,
            dx=dx,
            dy=dy,
            x_dim=x_dim,
            y_dim=y_dim,
            parallel_scale=parallel_scale,
            meridional_scale=meridional_scale,
            latitude=latitude,
            longitude=longitude,
            crs=crs,
        )
        x_axis = x_dim % f_arr.ndim
        y_axis = y_dim % f_arr.ndim
        term_x = first_derivative(grad_u, delta=dx, axis=x_axis)
        term_y = first_derivative(grad_v, delta=dy, axis=y_axis)
        return term_x + term_y
    x_axis = x_dim % f_arr.ndim
    y_axis = y_dim % f_arr.ndim
    return second_derivative(f, delta=dx, axis=x_axis) + second_derivative(f, delta=dy, axis=y_axis)


# ============================================================================
# Severe weather composite parameters
# ============================================================================

def significant_tornado_parameter(sbcape, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter (fixed-layer STP).

    Uses surface-based parcel CAPE and LCL height (not mixed-layer).
    Matches MetPy's ``significant_tornado`` formula exactly.

    Parameters
    ----------
    sbcape : Quantity (J/kg)
        Surface-based CAPE.
    lcl_height : Quantity (m)
        Surface-based LCL height AGL.
    srh_0_1km : Quantity (m^2/s^2)
        0-1 km storm-relative helicity.
    bulk_shear_0_6km : Quantity (m/s)
        0-6 km bulk wind shear magnitude.

    Returns
    -------
    Quantity (dimensionless)
    """
    cape = _as_float(_strip(sbcape, "J/kg"))
    lcl = _as_float(_strip(lcl_height, "m"))
    srh = _as_float(_strip(srh_0_1km, "m**2/s**2"))
    shear = _as_float(_strip(bulk_shear_0_6km, "m/s"))
    return np.asarray([_calc.significant_tornado_parameter(cape, lcl, srh, shear)]) * units.dimensionless


def supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff):
    """Supercell Composite Parameter (SCP).

    Parameters
    ----------
    mucape : Quantity (J/kg)
    srh_eff : Quantity (m^2/s^2)
    bulk_shear_eff : Quantity (m/s)

    Returns
    -------
    Quantity (dimensionless)
    """
    cape = _as_float(_strip(mucape, "J/kg"))
    srh = _as_float(_strip(srh_eff, "m**2/s**2"))
    shear = _as_float(_strip(bulk_shear_eff, "m/s"))
    return np.asarray([_calc.supercell_composite_parameter(cape, srh, shear)]) * units.dimensionless


def critical_angle(*args):
    """Critical angle between storm-relative inflow and 0-500m shear.

    MetPy form: ``critical_angle(pressure, u, v, height, u_storm, v_storm)``

    Computes the angle between the 10m storm-relative inflow vector and
    the 0-500m shear vector (MetPy-exact algorithm).

    Returns
    -------
    Quantity (degree)
    """
    if len(args) != 6:
        raise TypeError(
            "critical_angle expects 6 positional args: "
            "(pressure, u, v, height, u_storm, v_storm)"
        )
    # Detect profile form: if 1st arg is array-like with >1 element
    first = np.asarray(args[0].magnitude if hasattr(args[0], "magnitude") else args[0])
    if first.ndim >= 1 and first.size > 1:
        # MetPy profile form: (pressure, u, v, height, u_storm, v_storm)
        p_prof, u_prof, v_prof, h_prof, storm_u, storm_v = args
        su = _scalar_strip(storm_u, "m/s")
        sv = _scalar_strip(storm_v, "m/s")

        # 0-500m bulk shear (MetPy uses bulk_shear for this)
        shr_u, shr_v = bulk_shear(p_prof, u_prof, v_prof, height=h_prof,
                                   depth=500.0 * units.m)
        shr_u_val = float(shr_u.magnitude if hasattr(shr_u, "magnitude") else shr_u)
        shr_v_val = float(shr_v.magnitude if hasattr(shr_v, "magnitude") else shr_v)

        # Storm motion relative to surface wind (MetPy convention: u_storm - u[0])
        u_sfc = float(_as_1d(_strip(u_prof, "m/s"))[0])
        v_sfc = float(_as_1d(_strip(v_prof, "m/s"))[0])
        inflow_u = su - u_sfc
        inflow_v = sv - v_sfc

        # Angle between shear vector and inflow vector
        vshr = np.array([shr_u_val, shr_v_val])
        vsm = np.array([inflow_u, inflow_v])
        mag_product = np.linalg.norm(vshr) * np.linalg.norm(vsm)
        if mag_product < 1e-10:
            return 0.0 * units.degree
        cos_angle = np.clip(np.dot(vshr, vsm) / mag_product, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        return angle_deg * units.degree
    else:
        # Direct scalar form: (storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m)
        storm_u, storm_v, u_sfc_q, v_sfc_q, u_500_q, v_500_q = args
        su = _scalar_strip(storm_u, "m/s")
        sv = _scalar_strip(storm_v, "m/s")
        u_sfc = _scalar_strip(u_sfc_q, "m/s")
        v_sfc = _scalar_strip(v_sfc_q, "m/s")
        u_500 = _scalar_strip(u_500_q, "m/s")
        v_500 = _scalar_strip(v_500_q, "m/s")
    return _attach(
        _calc.critical_angle(su, sv, u_sfc, v_sfc, u_500, v_500),
        "degree",
    )


def boyden_index(z1000, z700, t700):
    """Boyden Index.

    Parameters
    ----------
    z1000 : Quantity (m)
        1000 hPa geopotential height.
    z700 : Quantity (m)
        700 hPa geopotential height.
    t700 : Quantity (temperature)
        700 hPa temperature.

    Returns
    -------
    Quantity (dimensionless)
    """
    return _calc.boyden_index(
        _as_float(_strip(z1000, "m")),
        _as_float(_strip(z700, "m")),
        _as_float(_strip(t700, "degC")),
    ) * units.dimensionless


def bulk_richardson_number(cape, shear_0_6km):
    """Bulk Richardson Number.

    Parameters
    ----------
    cape : Quantity (J/kg)
    shear_0_6km : Quantity (m/s)

    Returns
    -------
    Quantity (dimensionless)
    """
    return _calc.bulk_richardson_number(
        _as_float(_strip(cape, "J/kg")),
        _as_float(_strip(shear_0_6km, "m/s")),
    ) * units.dimensionless


def convective_inhibition_depth(pressure, temperature, dewpoint):
    """Convective inhibition depth.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    return _calc.convective_inhibition_depth(p, t, td) * units.hPa


def dendritic_growth_zone(temperature, pressure):
    """Dendritic growth zone bounds.

    Parameters
    ----------
    temperature : array Quantity (temperature)
    pressure : array Quantity (pressure)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (hPa))
        Bottom and top pressure of the dendritic growth zone.
    """
    t = _as_1d(_strip(temperature, "degC"))
    p = _as_1d(_strip(pressure, "hPa"))
    bot, top = _calc.dendritic_growth_zone(t, p)
    return bot * units.hPa, top * units.hPa


def fosberg_fire_weather_index(temperature, relative_humidity, wind_speed_val):
    """Fosberg Fire Weather Index.

    Parameters
    ----------
    temperature : Quantity (temperature, Fahrenheit)
    relative_humidity : Quantity (percent) or float
    wind_speed_val : Quantity (speed, mph)

    Returns
    -------
    Quantity (dimensionless)
    """
    t = _as_float(_strip(temperature, "degF"))
    rh = _as_float(_rh_to_percent(relative_humidity))
    ws = _as_float(_strip(wind_speed_val, "mph"))
    return _calc.fosberg_fire_weather_index(t, rh, ws) * units.dimensionless


def freezing_rain_composite(temperature, pressure, precip_type):
    """Freezing rain composite index.

    Parameters
    ----------
    temperature : array Quantity (temperature)
    pressure : array Quantity (pressure)
    precip_type : int
        Precipitation type flag.

    Returns
    -------
    Quantity (dimensionless)
    """
    t = _as_1d(_strip(temperature, "degC"))
    p = _as_1d(_strip(pressure, "hPa"))
    return _calc.freezing_rain_composite(t, p, int(precip_type)) * units.dimensionless


def haines_index(t_950, t_850, td_850):
    """Haines Index (fire weather).

    Parameters
    ----------
    t_950 : Quantity (temperature)
    t_850 : Quantity (temperature)
    td_850 : Quantity (temperature)

    Returns
    -------
    int
    """
    return _calc.haines_index(
        _as_float(_strip(t_950, "degC")),
        _as_float(_strip(t_850, "degC")),
        _as_float(_strip(td_850, "degC")),
    )


def hot_dry_windy(temperature, relative_humidity, wind_speed_val, vpd=0.0):
    """Hot-Dry-Windy Index.

    Parameters
    ----------
    temperature : Quantity (temperature)
    relative_humidity : float or Quantity (percent)
    wind_speed_val : Quantity (m/s)
    vpd : float
        Vapor pressure deficit (hPa). If 0, computed internally.

    Returns
    -------
    Quantity (dimensionless)
    """
    t = _as_float(_strip(temperature, "degC"))
    rh = _as_float(_rh_to_percent(relative_humidity))
    ws = _as_float(_strip(wind_speed_val, "m/s"))
    return _calc.hot_dry_windy(t, rh, ws, float(vpd)) * units.dimensionless


def warm_nose_check(temperature, pressure):
    """Check for a warm nose (melting layer above freezing aloft).

    Parameters
    ----------
    temperature : array Quantity (temperature)
    pressure : array Quantity (pressure)

    Returns
    -------
    bool
    """
    t = _as_1d(_strip(temperature, "degC"))
    p = _as_1d(_strip(pressure, "hPa"))
    return _calc.warm_nose_check(t, p)


def galvez_davison_index(pressure, temperature, mixing_ratio, surface_pressure, vertical_dim=0):
    """Calculate the Galvez-Davison Index."""
    cp_d = 1004.6662184201462 * units("joule / kelvin / kilogram")
    pressure_hpa = np.asarray(pressure.to("hPa").magnitude, dtype=np.float64)
    temperature_k = np.asarray(temperature.to("kelvin").magnitude, dtype=np.float64)
    mixing_ratio_nd = np.asarray(mixing_ratio.to("dimensionless").magnitude, dtype=np.float64)
    theta_k = np.asarray(
        potential_temperature(pressure, temperature).to("kelvin").magnitude,
        dtype=np.float64,
    )

    if np.any(np.max(pressure_hpa, axis=vertical_dim) < 950.0):
        raise ValueError(
            "Data not provided for 950hPa or higher pressure. "
            "GDI requires 950hPa temperature and mixing ratio data.\n"
            f"Max provided pressures:\n{np.max(pressure, axis=0)}"
        )

    def _interp_to_levels(field):
        field_axis = np.moveaxis(np.asarray(field, dtype=np.float64), vertical_dim, 0)
        pressure_axis = np.moveaxis(pressure_hpa, vertical_dim, 0)
        flat_field = field_axis.reshape(field_axis.shape[0], -1)
        flat_pressure = pressure_axis.reshape(pressure_axis.shape[0], -1)
        out = np.empty((4, flat_field.shape[1]), dtype=np.float64)
        targets = np.array([950.0, 850.0, 700.0, 500.0], dtype=np.float64)
        for column in range(flat_field.shape[1]):
            p_col = flat_pressure[:, column]
            f_col = flat_field[:, column]
            valid = np.isfinite(p_col) & np.isfinite(f_col)
            if np.count_nonzero(valid) < 2:
                out[:, column] = np.nan
                continue
            p_valid = p_col[valid]
            f_valid = f_col[valid]
            if p_valid[0] > p_valid[-1]:
                p_valid = p_valid[::-1]
                f_valid = f_valid[::-1]
            out[:, column] = np.interp(targets, p_valid, f_valid)
        return out.reshape((4,) + field_axis.shape[1:])

    temps = _interp_to_levels(temperature_k) * units.kelvin
    mixrs = _interp_to_levels(mixing_ratio_nd) * units.dimensionless
    thetas = _interp_to_levels(theta_k) * units.kelvin
    t950, t850, t700, t500 = temps
    r950, r850, r700, r500 = mixrs
    th950, th850, th700, th500 = thetas

    l_0 = units.Quantity(2.69e6, "J/kg")
    alpha = units.Quantity(-10, "K")
    eptp_a = th950 * np.exp(l_0 * r950 / (cp_d * t850))
    eptp_b = (
        (th850 + th700) / 2
        * np.exp(l_0 * (r850 + r700) / 2 / (cp_d * t850))
        + alpha
    )
    eptp_c = th500 * np.exp(l_0 * r500 / (cp_d * t850)) + alpha

    beta = units.Quantity(303, "K")
    l_e = eptp_a - beta
    m_e = eptp_c - beta
    gamma = units.Quantity(6.5e-2, "1/K^2")
    column_buoyancy_index = np.atleast_1d(gamma * l_e * m_e)
    column_buoyancy_index[l_e <= 0] = 0

    tau = units.Quantity(263.15, "K")
    t_diff = t500 - tau
    mu = units.Quantity(-7, "1/K")
    mid_tropospheric_warming_index = np.atleast_1d(mu * t_diff)
    mid_tropospheric_warming_index[t_diff <= 0] = 0

    s = t950 - t700
    d = eptp_b - eptp_a
    inv_sum = s + d
    sigma = units.Quantity(1.5, "1/K")
    inversion_index = np.atleast_1d(sigma * inv_sum)
    inversion_index[inv_sum >= 0] = 0

    terrain_correction = (
        18 - 9000 / (np.asarray(surface_pressure.to("hPa").magnitude, dtype=np.float64) - 500)
    ) * units.dimensionless
    gdi = (
        column_buoyancy_index
        + mid_tropospheric_warming_index
        + inversion_index
        + terrain_correction
    )
    if np.size(gdi) == 1:
        return gdi[0]
    return gdi


# ============================================================================
# Atmo (standard atmosphere, comfort indices)
# ============================================================================

def pressure_to_height_std(pressure):
    """Convert pressure to height using the US Standard Atmosphere 1976.

    Parameters
    ----------
    pressure : Quantity (pressure)

    Returns
    -------
    Quantity (m)
    """
    pressure = pressure if hasattr(pressure, "to") else np.asarray(pressure, dtype=np.float64) * units.hPa
    p0 = 1013.25 * units.hectopascal
    t0 = 288.0 * units.kelvin
    gamma = 6.5 * units.kelvin / units.kilometer
    rd = 287.04749097718457 * units("joule / kelvin / kilogram")
    g = 9.80665 * units("meter / second ** 2")
    exponent = (rd * gamma / g).to("dimensionless").magnitude
    return (t0 / gamma) * (1 - (pressure / p0).to("dimensionless") ** exponent)


def height_to_pressure_std(height):
    """Convert height to pressure using the US Standard Atmosphere 1976.

    Parameters
    ----------
    height : Quantity (m)

    Returns
    -------
    Quantity (hPa)
    """
    height = height if hasattr(height, "to") else np.asarray(height, dtype=np.float64) * units.meter
    p0 = 1013.25 * units.hectopascal
    t0 = 288.0 * units.kelvin
    gamma = 6.5 * units.kelvin / units.kilometer
    rd = 287.04749097718457 * units("joule / kelvin / kilogram")
    g = 9.80665 * units("meter / second ** 2")
    exponent = (g / (rd * gamma)).to("dimensionless").magnitude
    return p0 * (1 - (gamma / t0) * height) ** exponent


def altimeter_to_station_pressure(altimeter_value, height):
    """Convert altimeter setting to station pressure.

    Parameters
    ----------
    altimeter : Quantity (pressure)
    elevation : Quantity (m)

    Returns
    -------
    Quantity (hPa)
    """
    altimeter_value = (
        altimeter_value
        if hasattr(altimeter_value, "to")
        else np.asarray(altimeter_value, dtype=np.float64) * units.hPa
    )
    height = height if hasattr(height, "to") else np.asarray(height, dtype=np.float64) * units.meter
    p0 = 1013.25 * units.hectopascal
    t0 = 288.0 * units.kelvin
    gamma = 6.5 * units.kelvin / units.kilometer
    rd = 287.04749097718457 * units("joule / kelvin / kilogram")
    g = 9.80665 * units("meter / second ** 2")
    n_value = (rd * gamma / g).to_base_units()
    return (
        (
            altimeter_value ** n_value
            - ((p0.to(altimeter_value.units) ** n_value * gamma * height) / t0)
        )
        ** (1 / n_value)
        + units.Quantity(0.3, "hPa")
    )


def station_to_altimeter_pressure(station_pressure, elevation):
    """Convert station pressure to altimeter setting.

    Parameters
    ----------
    station_pressure : Quantity (pressure)
    elevation : Quantity (m)

    Returns
    -------
    Quantity (hPa)
    """
    s = _as_float(_strip(station_pressure, "hPa"))
    e = _as_float(_strip(elevation, "m"))
    return _calc.station_to_altimeter_pressure(s, e) * units.hPa


def altimeter_to_sea_level_pressure(altimeter_value, height, temperature):
    """Convert altimeter setting to sea-level pressure.

    Parameters
    ----------
    altimeter : Quantity (pressure)
    elevation : Quantity (m)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
    """
    altimeter_value = (
        altimeter_value
        if hasattr(altimeter_value, "to")
        else np.asarray(altimeter_value, dtype=np.float64) * units.hPa
    )
    height = height if hasattr(height, "to") else np.asarray(height, dtype=np.float64) * units.meter
    temperature = (
        temperature
        if hasattr(temperature, "to")
        else np.asarray(temperature, dtype=np.float64) * units.degC
    )
    station_pressure = altimeter_to_station_pressure(altimeter_value, height)
    rd = 287.04749097718457 * units("joule / kelvin / kilogram")
    g = 9.80665 * units("meter / second ** 2")
    scale_height = rd * temperature.to("kelvin") / g
    return station_pressure * np.exp(height / scale_height)


def sigma_to_pressure(sigma, pressure_sfc, pressure_top):
    """Convert a sigma coordinate to pressure.

    Parameters
    ----------
    sigma : float (dimensionless, 0 to 1)
    psfc : Quantity (pressure)
    ptop : Quantity (pressure)

    Returns
    -------
    Quantity (hPa)
    """
    sigma_arr = (
        np.asarray(sigma.to("dimensionless").magnitude, dtype=np.float64)
        if hasattr(sigma, "to")
        else np.asarray(sigma, dtype=np.float64)
    )
    pressure_sfc = (
        pressure_sfc
        if hasattr(pressure_sfc, "to")
        else np.asarray(pressure_sfc, dtype=np.float64) * units.hPa
    )
    pressure_top = (
        pressure_top
        if hasattr(pressure_top, "to")
        else np.asarray(pressure_top, dtype=np.float64) * units.hPa
    )
    if np.any(sigma_arr < 0) or np.any(sigma_arr > 1):
        raise ValueError("Sigma values should be bounded by 0 and 1")
    if np.any(np.asarray(pressure_sfc.magnitude, dtype=np.float64) < 0) or np.any(
        np.asarray(pressure_top.magnitude, dtype=np.float64) < 0
    ):
        raise ValueError("Pressure input should be non-negative")
    return sigma_arr * (pressure_sfc - pressure_top) + pressure_top


def heat_index(temperature, relative_humidity):
    """Heat index (NWS Rothfusz regression).

    Parameters
    ----------
    temperature : Quantity (temperature)
    relative_humidity : float or Quantity (percent)

    Returns
    -------
    Quantity (degC)
    """
    result = _vec_call(_calc.heat_index, _strip(temperature, "degC"), _rh_to_percent(relative_humidity))
    return np.atleast_1d(result) * units.degC


def windchill(temperature, wind_speed_val):
    """Wind chill index (NWS formula).

    Parameters
    ----------
    temperature : Quantity (temperature)
    wind_speed_val : Quantity (m/s)

    Returns
    -------
    Quantity (degC)
    """
    result = _vec_call(_calc.windchill, _strip(temperature, "degC"), _strip(wind_speed_val, "m/s"))
    return result * units.degC


def apparent_temperature(temperature, relative_humidity, wind_speed_val):
    """Apparent temperature combining heat index and wind chill.

    Parameters
    ----------
    temperature : Quantity (temperature)
    relative_humidity : float or Quantity (percent)
    wind_speed_val : Quantity (m/s)

    Returns
    -------
    Quantity (degC)
    """
    result = _vec_call(
        _calc.apparent_temperature,
        _strip(temperature, "degC"),
        _rh_to_percent(relative_humidity),
        _strip(wind_speed_val, "m/s"),
    )
    return np.atleast_1d(result) * units.degC


# ============================================================================
# Smooth / spatial derivatives
# ============================================================================

def smooth_gaussian(scalar_grid, n):
    """2-D Gaussian smoothing.

    Parameters
    ----------
    data : 2-D array
    sigma : float (grid-point units)

    Returns
    -------
    2-D ndarray
    """
    from scipy.ndimage import gaussian_filter

    n = max(int(round(n)), 2)
    sigma = n / (2 * np.pi)
    num_axes = len(scalar_grid.shape)
    sigma_seq = [sigma if i > num_axes - 3 else 0 for i in range(num_axes)]
    data_units = getattr(scalar_grid, "units", None)
    data = getattr(scalar_grid, "magnitude", scalar_grid)
    filter_args = {"sigma": sigma_seq, "truncate": 2 * np.sqrt(2)}
    if hasattr(data, "mask"):
        smoothed = gaussian_filter(data.data, **filter_args)
        result = np.ma.array(smoothed, mask=data.mask)
    else:
        result = gaussian_filter(data, **filter_args)
    return result * data_units if data_units is not None else result


def smooth_rectangular(scalar_grid, size, passes=1):
    """Rectangular (box) smoothing.

    Parameters
    ----------
    data : 2-D array
    size : int (kernel side length)
    passes : int, optional
        Number of times to apply the filter (default 1).

    Returns
    -------
    2-D ndarray
    """
    return smooth_window(scalar_grid, np.ones(size), passes=passes)


def smooth_circular(scalar_grid, radius, passes=1):
    """Circular (disk) smoothing.

    Parameters
    ----------
    data : 2-D array
    radius : float (grid-point units)
    passes : int, optional
        Number of times to apply the filter (default 1).

    Returns
    -------
    2-D ndarray
    """
    radius = int(radius)
    size = 2 * radius + 1
    x, y = np.mgrid[:size, :size]
    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    circle = distance <= radius
    return smooth_window(scalar_grid, circle, passes=passes)


def smooth_n_point(scalar_grid, n=5, passes=1):
    """N-point smoother (5 or 9).

    Parameters
    ----------
    data : 2-D array
    n : int (5 or 9)
    passes : int, optional
        Number of times to apply the filter (default 1).

    Returns
    -------
    2-D ndarray
    """
    n = int(n)
    if n == 9:
        weights = np.array(
            [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
        )
    elif n == 5:
        weights = np.array(
            [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]]
        )
    else:
        raise ValueError(
            "The number of points to use in the smoothing calculation must be either 5 or 9."
        )
    return smooth_window(
        scalar_grid,
        window=weights,
        passes=passes,
        normalize_weights=False,
    )


def smooth_window(scalar_grid, window, passes=1, normalize_weights=True):
    """Generic 2-D convolution with a user-supplied kernel.

    Parameters
    ----------
    data : 2-D array
    window : 2-D array (kernel)
    passes : int, optional
        Number of times to apply the filter (default 1).
    normalize_weights : bool, optional
        Whether to normalize the kernel weights before applying
        (default True).

    Returns
    -------
    2-D ndarray
    """
    from itertools import product

    def _pad(window_size):
        return (window_size - 1) // 2

    def _zero_to_none(value):
        return value if value != 0 else None

    def _offset(pad, offset):
        return slice(_zero_to_none(pad + offset), _zero_to_none(-pad + offset))

    def _trailing_dims(indexer):
        return (Ellipsis,) + tuple(indexer)

    weights = np.asarray(getattr(window, "magnitude", window), dtype=np.float64)
    if any((size % 2 == 0) for size in weights.shape):
        raise ValueError("The shape of the smoothing window must be odd in all dimensions.")
    if normalize_weights:
        weights = weights / np.sum(weights)

    inner_full_index = _trailing_dims(_offset(_pad(n), 0) for n in weights.shape)
    weight_indexes = tuple(product(*(range(n) for n in weights.shape)))

    def offset_full_index(weight_index):
        return _trailing_dims(
            _offset(_pad(n), weight_index[i] - _pad(n))
            for i, n in enumerate(weights.shape)
        )

    data_units = getattr(scalar_grid, "units", None)
    unit_str = str(data_units) if data_units is not None else None
    if unit_str is None:
        unit_str = getattr(getattr(scalar_grid, "attrs", None), "get", lambda *_: None)("units")
    data = np.array(getattr(scalar_grid, "magnitude", scalar_grid))
    for _ in range(int(passes)):
        data[inner_full_index] = sum(
            weights[index] * data[offset_full_index(index)] for index in weight_indexes
        )
    return _wrap_result_like(scalar_grid, data, unit_str=unit_str)


def _gradient_axes_and_positions(f, axes, coordinates, deltas):
    data = np.asarray(
        f.values if hasattr(f, "values") else getattr(f, "magnitude", f),
        dtype=np.float64,
    )
    if axes is None:
        resolved_axes = tuple(range(data.ndim))
    elif isinstance(axes, (int, str)):
        resolved_axes = (axes,)
    else:
        resolved_axes = tuple(axes)

    mapped_axes = []
    for axis in resolved_axes:
        if isinstance(axis, str):
            if not hasattr(f, "dims"):
                raise TypeError("String axes require an xarray.DataArray input")
            mapped_axes.append(f.dims.index(axis))
        else:
            mapped_axes.append(int(axis) % data.ndim)
    mapped_axes = tuple(mapped_axes)

    axes_given = axes is not None

    def _check_length(positions):
        if axes_given and len(positions) < len(mapped_axes):
            raise ValueError('Length of "coordinates" or "deltas" cannot be less than that of "axes".')
        if not axes_given and len(positions) != len(mapped_axes):
            raise ValueError(
                'Length of "coordinates" or "deltas" must match the number of dimensions of '
                '"f" when "axes" is not given.'
            )

    if deltas is not None:
        if coordinates is not None:
            raise ValueError('Cannot specify both "coordinates" and "deltas".')
        _check_length(deltas)
        return data, mapped_axes, tuple(deltas), "delta"
    if coordinates is not None:
        _check_length(coordinates)
        return data, mapped_axes, tuple(coordinates), "coordinate"
    if hasattr(f, "dims") and hasattr(f, "coords"):
        coord_positions = []
        for axis in mapped_axes:
            dim = f.dims[axis]
            coord_positions.append(f.coords[dim] if dim in f.coords else np.arange(data.shape[axis]))
        return data, mapped_axes, tuple(coord_positions), "coordinate"
    raise ValueError(
        'Must specify either "coordinates" or "deltas" for value positions when "f" is not '
        "a DataArray."
    )


def _gradient_spacing(position, axis_size, mode):
    unit_obj = None
    if mode == "delta":
        if hasattr(position, "to"):
            quantity = position.to_base_units()
            values = np.asarray(quantity.magnitude, dtype=np.float64)
            unit_obj = quantity.units
        else:
            values = np.asarray(position, dtype=np.float64)
        if values.ndim == 0:
            return float(values), unit_obj
        flat = values.ravel()
        if flat.size == axis_size - 1:
            return np.concatenate(([0.0], np.cumsum(flat))), unit_obj
        if flat.size == axis_size:
            return flat, unit_obj
        raise ValueError("Delta array length must be axis length or axis length minus one.")

    if hasattr(position, "to"):
        quantity = position.to_base_units()
        return np.asarray(quantity.magnitude, dtype=np.float64), quantity.units
    if hasattr(position, "attrs") and "units" in getattr(position, "attrs", {}):
        unit_obj = units(position.attrs["units"])
    values = position.values if hasattr(position, "values") else position
    return np.asarray(values, dtype=np.float64), unit_obj


def _data_unit(data):
    if hasattr(data, "units"):
        return data.units
    try:
        unit_attr = data.metpy.units
        if str(unit_attr) != "dimensionless":
            return unit_attr
    except Exception:
        pass
    attrs = getattr(data, "attrs", {})
    if isinstance(attrs, dict) and "units" in attrs:
        return units(attrs["units"])
    return None


def _wrap_derivative_like(template, values, denom_unit=None, power=1):
    numerator_unit = _data_unit(template)
    result_unit = None
    if numerator_unit is not None and denom_unit is not None:
        result_unit = numerator_unit / (denom_unit ** power)
    elif numerator_unit is not None:
        result_unit = numerator_unit
    elif denom_unit is not None:
        result_unit = 1 / (denom_unit ** power)

    if hasattr(template, "dims") and xr is not None:
        result = xr.DataArray(
            np.asarray(values, dtype=np.float64),
            dims=template.dims,
            coords=template.coords,
            attrs=getattr(template, "attrs", {}).copy(),
        )
        if result_unit is not None:
            result.attrs["units"] = str(result_unit)
        return result
    if result_unit is not None:
        return np.asarray(values, dtype=np.float64) * result_unit
    return np.asarray(values, dtype=np.float64)


def gradient(f, axes=None, coordinates=None, deltas=None):
    """Calculate the gradient of a scalar field.

    Parameters
    ----------
    f : array-like or Quantity
        Scalar field.
    **kwargs : dict
        Accepts coordinates, axes, deltas for MetPy compatibility.
        When deltas is provided, uses native gradient_x/gradient_y for 2-D.
        Otherwise falls back to numpy.gradient.

    Returns
    -------
    list of arrays
        One array per dimension.
    """
    data, axes, positions, mode = _gradient_axes_and_positions(f, axes, coordinates, deltas)
    derivatives = []
    for axis, position in zip(axes, positions):
        spacing, spacing_unit = _gradient_spacing(position, data.shape[axis], mode)
        derivative = np.gradient(data, spacing, axis=axis)
        derivatives.append(_wrap_derivative_like(f, derivative, spacing_unit, power=1))
    return tuple(derivatives)


def gradient_x(data, dx):
    """Partial derivative df/dx.

    Parameters
    ----------
    data : 2-D array
    dx : Quantity (m) or float

    Returns
    -------
    2-D array Quantity (data_units / m)
    """
    has_units = hasattr(data, "units")
    d_arr = np.asarray(data.magnitude if has_units else data, dtype=np.float64)
    dx_val = _mean_spacing(dx, "m")
    result = np.asarray(_calc.gradient_x(d_arr, dx_val))
    if has_units:
        return result * (data.units / units.m)
    return result


def gradient_y(data, dy):
    """Partial derivative df/dy.

    Parameters
    ----------
    data : 2-D array
    dy : Quantity (m) or float

    Returns
    -------
    2-D array Quantity (data_units / m)
    """
    has_units = hasattr(data, "units")
    d_arr = np.asarray(data.magnitude if has_units else data, dtype=np.float64)
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.gradient_y(d_arr, dy_val))
    if has_units:
        return result * (data.units / units.m)
    return result


def laplacian(f, axes=None, coordinates=None, deltas=None):
    """Laplacian (d2f/dx2 + d2f/dy2).

    Parameters
    ----------
    data : 2-D array
    dx, dy : Quantity (m) or float

    Returns
    -------
    2-D array Quantity (data_units / m^2)
    """
    data, axes, positions, mode = _gradient_axes_and_positions(f, axes, coordinates, deltas)
    derivatives = []
    for axis, position in zip(axes, positions):
        spacing, spacing_unit = _gradient_spacing(position, data.shape[axis], mode)
        first = np.gradient(data, spacing, axis=axis)
        second = np.gradient(first, spacing, axis=axis)
        derivatives.append(_wrap_derivative_like(f, second, spacing_unit, power=2))
    total = derivatives[0]
    for derivative in derivatives[1:]:
        total = total + derivative
    return total


def first_derivative(data, axis_spacing=None, axis=0, x=None, delta=None):
    """First derivative along a chosen axis.

    Parameters
    ----------
    data : 2-D array
    axis_spacing : Quantity (m) or float
    axis : int (0=x, 1=y)

    Returns
    -------
    2-D array
    """
    has_units = hasattr(data, "units")
    d_arr = np.asarray(data.magnitude if has_units else data, dtype=np.float64)
    if delta is not None:
        axis_spacing = delta
    elif x is not None and axis_spacing is None:
        x_arr = x.to("m").magnitude if hasattr(x, "to") else x
        x_arr = np.asarray(x_arr, dtype=np.float64)
        ax = axis % d_arr.ndim
        if x_arr.ndim > 0 and x_arr.shape[ax if x_arr.ndim > ax else 0] == d_arr.shape[ax]:
            axis_spacing = np.diff(x_arr, axis=ax if x_arr.ndim > ax else 0)
        else:
            axis_spacing = x
    if axis_spacing is None:
        raise TypeError("first_derivative requires axis spacing via axis_spacing, x, or delta")
    spacing_arr = np.asarray(
        axis_spacing.to("m").magnitude if hasattr(axis_spacing, "to") else axis_spacing,
        dtype=np.float64,
    )
    if spacing_arr.size > 1:
        result = _first_derivative_variable(d_arr, spacing_arr, int(axis))
    else:
        ds = float(spacing_arr.ravel()[0])
        result = np.asarray(_calc.first_derivative(d_arr, ds, int(axis)))
    if has_units:
        return result * (data.units / units.m)
    if hasattr(axis_spacing, "magnitude"):
        return result / units.m
    return result


def second_derivative(data, axis_spacing=None, axis=0, x=None, delta=None):
    """Second derivative along a chosen axis.

    Parameters
    ----------
    data : 2-D array
    axis_spacing : Quantity (m) or float
    axis : int (0=x, 1=y)

    Returns
    -------
    2-D array
    """
    has_units = hasattr(data, "units")
    d_arr = np.asarray(data.magnitude if has_units else data, dtype=np.float64)
    if delta is not None:
        axis_spacing = delta
    elif x is not None and axis_spacing is None:
        axis_spacing = x
    if axis_spacing is None:
        raise TypeError("second_derivative requires axis spacing via axis_spacing, x, or delta")
    ds = _mean_spacing(axis_spacing, "m")
    result = np.asarray(_calc.second_derivative(d_arr, ds, int(axis)))
    if has_units:
        return result * (data.units / units.m ** 2)
    if hasattr(axis_spacing, "magnitude"):
        return result / units.m ** 2
    return result


def lat_lon_grid_deltas(longitude, latitude, x_dim=-1, y_dim=-2, geod=None):
    """Physical grid spacings (dx, dy) in meters from lat/lon grids.

    Parameters
    ----------
    lats : 2-D array (degrees)
    lons : 2-D array (degrees)

    Returns
    -------
    tuple of (2-D array Quantity (m), 2-D array Quantity (m))
    """
    from pyproj import Geod

    def _make_take(ndims, slice_dim):
        return lambda indexer: tuple(
            indexer if slice_dim % ndims == i else slice(None) for i in range(ndims)
        )

    if latitude.ndim != longitude.ndim:
        raise ValueError("Latitude and longitude must have the same number of dimensions.")
    if latitude.ndim < 2:
        longitude, latitude = np.meshgrid(longitude, latitude)

    try:
        longitude = np.asarray(longitude.to("degrees").magnitude, dtype=np.float64)
        latitude = np.asarray(latitude.to("degrees").magnitude, dtype=np.float64)
    except AttributeError:
        longitude = np.asarray(longitude, dtype=np.float64)
        latitude = np.asarray(latitude, dtype=np.float64)

    take_y = _make_take(latitude.ndim, y_dim)
    take_x = _make_take(latitude.ndim, x_dim)
    geod_obj = Geod(ellps="sphere") if geod is None else geod

    forward_az, _, dy = geod_obj.inv(
        longitude[take_y(slice(None, -1))],
        latitude[take_y(slice(None, -1))],
        longitude[take_y(slice(1, None))],
        latitude[take_y(slice(1, None))],
    )
    dy[(forward_az < -90.0) | (forward_az > 90.0)] *= -1

    forward_az, _, dx = geod_obj.inv(
        longitude[take_x(slice(None, -1))],
        latitude[take_x(slice(None, -1))],
        longitude[take_x(slice(1, None))],
        latitude[take_x(slice(1, None))],
    )
    dx[(forward_az < 0.0) | (forward_az > 180.0)] *= -1
    return units.Quantity(dx, "meter"), units.Quantity(dy, "meter")


# ============================================================================
# Utils
# ============================================================================

_UND = "UND"
_UND_ANGLE = -999.0
_DIR_STRS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
    _UND,
]
_MAX_DEGREE_ANGLE = units.Quantity(360, "degree")
_BASE_DEGREE_MULTIPLIER = units.Quantity(22.5, "degree")
_DIR_DICT = {
    dir_str: index * _BASE_DEGREE_MULTIPLIER
    for index, dir_str in enumerate(_DIR_STRS)
}
_DIR_DICT[_UND] = units.Quantity(np.nan, "degree")


def _clean_direction_local(dir_list, preprocess=False):
    if preprocess:
        return [_UND if not isinstance(direction, str) else direction for direction in dir_list]
    return [_UND if direction not in _DIR_STRS else direction for direction in dir_list]


def _abbreviate_direction_local(ext_dir_str):
    return (
        ext_dir_str.upper()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("NORTH", "N")
        .replace("EAST", "E")
        .replace("SOUTH", "S")
        .replace("WEST", "W")
    )


def _unabbreviate_direction_local(abb_dir_str):
    return (
        abb_dir_str.upper()
        .replace(_UND, "Undefined ")
        .replace("N", "North ")
        .replace("E", "East ")
        .replace("S", "South ")
        .replace("W", "West ")
        .replace(" ,", ",")
    ).strip()


def _make_take_local(ndims, slice_dim):
    return lambda indexer: tuple(
        indexer if slice_dim % ndims == i else slice(None) for i in range(ndims)
    )


def _broadcast_indices_local(indices, shape, axis):
    ret = []
    ndim = len(shape)
    for dim in range(ndim):
        if dim == axis:
            ret.append(indices)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)


def _next_non_masked_element_local(a, idx):
    import numpy.ma as ma

    try:
        next_idx = idx + a[idx:].mask.argmin()
        if ma.is_masked(a[next_idx]):
            return None, None
        return next_idx, a[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, a[idx]


def _neighbor_inds_local(y, x):
    from itertools import product

    for dx, dy in product((-1, 0, 1), repeat=2):
        yield y + dy, x + dx


def _find_uf_local(uf, item):
    while (next_item := uf[item]) != item:
        uf[item] = uf[next_item]
        item = next_item
    return item


def angle_to_direction(input_angle, full=False, level=3):
    """Convert a meteorological angle to a cardinal direction string.

    Parameters
    ----------
    degrees : float or Quantity (degree)
        Angle in degrees clockwise from north.
    level : int, optional
        Number of compass points: 8, 16 (default), or 32.
    full : bool, optional
        If True, return full word names (e.g. "North" instead of "N").

    Returns
    -------
    str
    """
    from operator import itemgetter

    try:
        origin_units = input_angle.units
        input_angle = input_angle.m
    except AttributeError:
        origin_units = units.degree

    if not hasattr(input_angle, "__len__") or isinstance(input_angle, str):
        input_angle = [input_angle]
        scalar = True
    else:
        scalar = False

    np_input_angle = np.array(input_angle).astype(float)
    origshape = np_input_angle.shape
    ndarray = len(origshape) > 1
    input_angle = units.Quantity(np_input_angle, origin_units)
    input_angle[input_angle < 0] = np.nan
    norm_angles = input_angle % _MAX_DEGREE_ANGLE

    if level == 3:
        nskip = 1
    elif level == 2:
        nskip = 2
    elif level == 1:
        nskip = 4
    else:
        raise ValueError("Level of complexity cannot be less than 1 or greater than 3!")

    angle_dict = {
        index * _BASE_DEGREE_MULTIPLIER.m * nskip: dir_str
        for index, dir_str in enumerate(_DIR_STRS[::nskip])
    }
    angle_dict[_MAX_DEGREE_ANGLE.m] = "N"
    angle_dict[_UND_ANGLE] = _UND

    multiplier = np.round((norm_angles / _BASE_DEGREE_MULTIPLIER / nskip) - 0.001).m
    round_angles = multiplier * _BASE_DEGREE_MULTIPLIER.m * nskip
    round_angles[np.where(np.isnan(round_angles))] = _UND_ANGLE
    if ndarray:
        round_angles = round_angles.flatten()
    dir_str_arr = itemgetter(*round_angles)(angle_dict)
    if full:
        dir_str_arr = ",".join(dir_str_arr)
        dir_str_arr = _unabbreviate_direction_local(dir_str_arr)
        dir_str_arr = dir_str_arr.split(",")
        if scalar:
            return dir_str_arr[0]
        return np.array(dir_str_arr).reshape(origshape)
    if scalar:
        return dir_str_arr
    return np.array(dir_str_arr).reshape(origshape)


def parse_angle(input_dir):
    """Parse a cardinal direction string to degrees.

    Parameters
    ----------
    direction : str

    Returns
    -------
    float or None
        Degrees (meteorological convention), or None if unrecognised.
    """
    from operator import itemgetter

    if isinstance(input_dir, str):
        abb_dir = _clean_direction_local([_abbreviate_direction_local(input_dir)])[0]
        return _DIR_DICT[abb_dir]
    if hasattr(input_dir, "__len__"):
        input_dir_str = ",".join(_clean_direction_local(input_dir, preprocess=True))
        abb_dir_str = _abbreviate_direction_local(input_dir_str)
        abb_dirs = _clean_direction_local(abb_dir_str.split(","))
        return units.Quantity.from_list(itemgetter(*abb_dirs)(_DIR_DICT))
    return units.Quantity(np.nan, "degree")


def find_bounding_indices(arr, values, axis, from_below=True):
    """Find two indices that bracket a target value.

    Parameters
    ----------
    values : array-like
    target : float

    Returns
    -------
    tuple of (int, int) or None
    """
    if hasattr(arr, "to"):
        arr_units = arr.units
        arr = np.asarray(arr.magnitude, dtype=np.float64)
        values = np.atleast_1d(np.asarray(values.to(arr_units).magnitude, dtype=np.float64))
    else:
        arr = np.asarray(arr, dtype=np.float64)
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))

    indices_shape = list(arr.shape)
    indices_shape[axis] = len(values)
    indices = np.empty(indices_shape, dtype=int)
    good = np.empty(indices_shape, dtype=bool)
    take = _make_take_local(arr.ndim, axis)

    for level_index, value in enumerate(values):
        switches = np.abs(np.diff((arr <= value).astype(int), axis=axis))
        good_search = np.any(switches, axis=axis)
        if from_below:
            index = switches.argmax(axis=axis) + 1
        else:
            arr_slice = [slice(None)] * arr.ndim
            arr_slice[axis] = slice(None, None, -1)
            index = arr.shape[axis] - 1 - switches[tuple(arr_slice)].argmax(axis=axis)
        index[~good_search] = 0
        store_slice = take(level_index)
        indices[store_slice] = index
        good[store_slice] = good_search

    above = _broadcast_indices_local(indices, arr.shape, axis)
    below = _broadcast_indices_local(indices - 1, arr.shape, axis)
    return above, below, good


def nearest_intersection_idx(a, b):
    """Find the index nearest to where two series cross.

    Parameters
    ----------
    x : array-like
    y1, y2 : array-like

    Returns
    -------
    int or None
    """
    a_arr = np.asarray(getattr(a, "magnitude", a), dtype=np.float64)
    b_arr = np.asarray(getattr(b, "magnitude", b), dtype=np.float64)
    difference = a_arr - b_arr
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))
    return sign_change_idx


def resample_nn_1d(a, centers):
    """Nearest-neighbour 1-D resampling.

    Parameters
    ----------
    x : array-like
    xp : array-like
    fp : array-like

    Returns
    -------
    ndarray
    """
    a_arr = np.asarray(getattr(a, "magnitude", a), dtype=np.float64)
    centers_arr = np.asarray(getattr(centers, "magnitude", centers), dtype=np.float64)
    indices = []
    for center in centers_arr:
        index = np.abs(a_arr - center).argmin()
        if index not in indices:
            indices.append(index)
    return indices


def find_peaks(data, maxima=True, iqr_ratio=2):
    """Find peaks (or troughs) in a 1-D array, filtered by IQR.

    A local extremum is any point greater (or less, for troughs) than
    both its neighbours.  Only those extrema that stand out by at least
    ``iqr_ratio * IQR`` above (or below) the median are kept.

    Parameters
    ----------
    data : array-like
        1-D data array.
    maxima : bool, optional
        If True (default), find maxima; if False, find minima.
    iqr_ratio : float, optional
        IQR multiplier threshold.  Default 0.0 (no filtering).

    Returns
    -------
    numpy.ndarray of int
        Indices of qualifying peaks.
    """
    import itertools

    peaks = peak_persistence(data, maxima=maxima)
    q1, q3 = np.percentile([peak[-1] for peak in peaks], (25, 75))
    thresh = q3 + iqr_ratio * (q3 - q1)
    return map(
        list,
        zip(
            *(item[0] for item in itertools.takewhile(lambda item: item[1] > thresh, peaks)),
            strict=True,
        ),
    )


def peak_persistence(data, maxima=True):
    """Topological persistence-based peak detection.

    Ranks peaks (or troughs) by their "persistence" -- the height
    difference between a peak and the higher of the two saddle points
    that bound it.

    Parameters
    ----------
    data : array-like
        1-D data array.
    maxima : bool, optional
        If True (default), detect peaks; if False, detect troughs.

    Returns
    -------
    list of (int, float)
        (index, persistence) pairs sorted by descending persistence.
    """
    arr = np.asarray(getattr(data, "magnitude", data), dtype=np.float64)
    points = sorted(
        (item for item in np.ndenumerate(arr) if not np.isnan(item[1])),
        key=lambda item: item[1],
        reverse=maxima,
    )
    per = {points[0][0]: np.inf}
    peaks = {}
    for pt, val in points:
        already_done = {
            _find_uf_local(peaks, neighbor)
            for neighbor in _neighbor_inds_local(*pt)
            if neighbor in peaks
        }
        if already_done:
            biggest, *others = sorted(already_done, key=lambda item: arr[item], reverse=maxima)
            peaks[pt] = biggest
            for neighbor in others:
                peaks[neighbor] = biggest
                if arr[neighbor] != val:
                    per[neighbor] = abs(arr[neighbor] - val)
        else:
            peaks[pt] = pt
    return sorted(per.items(), key=lambda item: item[1], reverse=True)


def azimuth_range_to_lat_lon(azimuths, ranges, center_lon, center_lat, geod=None):
    """Convert radar azimuth/range to latitude/longitude.

    Uses great-circle (spherical earth) forward projection.

    Parameters
    ----------
    azimuths : array-like
        Azimuth angles in degrees clockwise from north.
    ranges : array-like
        Range values in meters from the radar.
    center_lat : float
        Radar site latitude in degrees.
    center_lon : float
        Radar site longitude in degrees.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (latitudes, longitudes) arrays.
    """
    import warnings
    from pyproj import Geod

    geod_obj = Geod(ellps="sphere") if geod is None else geod
    try:
        ranges = ranges.to("meters").magnitude
    except AttributeError:
        warnings.warn("Range values are not a Pint-Quantity, assuming values are in meters.")
    try:
        azimuths = azimuths.to("degrees").magnitude
    except AttributeError:
        warnings.warn("Azimuth values are not a Pint-Quantity, assuming values are in degrees.")
    rng2d, az2d = np.meshgrid(ranges, azimuths)
    lats = np.full(az2d.shape, center_lat)
    lons = np.full(az2d.shape, center_lon)
    lon, lat, _ = geod_obj.fwd(lons, lats, az2d, rng2d)
    return lon, lat


def find_intersections(x, a, b, direction="all", log_x=False):
    """Calculate the best estimate of intersection."""
    x_unit = getattr(x, "units", None)
    y_unit = getattr(a, "units", getattr(b, "units", None))
    x_arr = np.asarray(getattr(x, "magnitude", x), dtype=np.float64)
    a_arr = np.asarray(getattr(a, "magnitude", a), dtype=np.float64)
    b_arr = np.asarray(getattr(b, "magnitude", b), dtype=np.float64)

    x_work = np.log(x_arr) if log_x else x_arr
    nearest_idx = nearest_intersection_idx(a_arr, b_arr)
    next_idx = nearest_idx + 1
    sign_change = np.sign(a_arr[next_idx] - b_arr[next_idx])

    _, x0 = _next_non_masked_element_local(x_work, nearest_idx)
    _, x1 = _next_non_masked_element_local(x_work, next_idx)
    _, a0 = _next_non_masked_element_local(a_arr, nearest_idx)
    _, a1 = _next_non_masked_element_local(a_arr, next_idx)
    _, b0 = _next_non_masked_element_local(b_arr, nearest_idx)
    _, b1 = _next_non_masked_element_local(b_arr, next_idx)

    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    if len(intersect_x) == 0:
        if x_unit is not None:
            intersect_x = intersect_x * x_unit
        if y_unit is not None:
            intersect_y = intersect_y * y_unit
        return intersect_x, intersect_y

    if log_x:
        intersect_x = np.exp(intersect_x)

    duplicate_mask = np.ediff1d(intersect_x, to_end=1) != 0
    if direction == "increasing":
        mask = sign_change > 0
    elif direction == "decreasing":
        mask = sign_change < 0
    elif direction == "all":
        mask = duplicate_mask
    else:
        raise ValueError(f"Unknown option for direction: {direction}")

    intersect_x = intersect_x[mask]
    intersect_y = intersect_y[mask]
    if direction != "all":
        intersect_x = intersect_x[duplicate_mask[mask]]
        intersect_y = intersect_y[duplicate_mask[mask]]

    if x_unit is not None:
        intersect_x = intersect_x * x_unit
    if y_unit is not None:
        intersect_y = intersect_y * y_unit
    return intersect_x, intersect_y


def advection_3d(scalar, u, v, w, dx, dy, dz):
    """Advection of a scalar field by a 3-D wind.

    Extends 2-D advection to include the vertical term:
    ``-u(ds/dx) - v(ds/dy) - w(ds/dz)``

    Parameters
    ----------
    scalar : 3-D array Quantity, flattened [nz*ny*nx]
    u, v, w : 3-D array Quantity (m/s), flattened [nz*ny*nx]
    dx, dy : Quantity (m) -- horizontal grid spacings
    dz : Quantity (m) -- vertical grid spacing

    Returns
    -------
    1-D array Quantity (scalar_units / s)
    """
    has_units = hasattr(scalar, "units")
    s_unit_str = _safe_unit_str(scalar.units) if has_units else "dimensionless"
    s_arr = _as_1d(_strip(scalar, "")) if has_units else _as_1d(scalar)
    u_arr = _as_1d(_strip(u, "m/s")) if hasattr(u, "magnitude") else _as_1d(u)
    v_arr = _as_1d(_strip(v, "m/s")) if hasattr(v, "magnitude") else _as_1d(v)
    w_arr = _as_1d(_strip(w, "m/s")) if hasattr(w, "magnitude") else _as_1d(w)
    dx_val = _mean_spacing(dx, "m") if hasattr(dx, "magnitude") else float(np.asarray(dx).mean())
    dy_val = _mean_spacing(dy, "m") if hasattr(dy, "magnitude") else float(np.asarray(dy).mean())
    dz_val = _as_float(_strip(dz, "m")) if hasattr(dz, "magnitude") else float(dz)

    # Infer nx, ny, nz from the 3D array shape if possible
    arr = np.asarray(scalar.magnitude if has_units else scalar)
    if arr.ndim == 3:
        nz, ny, nx = arr.shape
    else:
        raise ValueError(
            "scalar must be a 3-D array (nz, ny, nx) for advection_3d; "
            f"got {arr.ndim}-D"
        )

    result = np.asarray(_calc.advection_3d(
        s_arr, u_arr, v_arr, w_arr,
        nx, ny, nz, dx_val, dy_val, dz_val,
    ))
    return result.reshape(nz, ny, nx) * units(f"({s_unit_str}) / s")


# ============================================================================
# Grid composite kernels (Rust-parallel, no Pint overhead)
# ============================================================================

def _grid_strip(arr):
    """Strip Pint units and return contiguous float64 array."""
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    return np.ascontiguousarray(arr, dtype=np.float64)


def _scalar_strip(val, target_unit=None):
    """Strip Pint unit from a scalar, optionally converting first."""
    if val is None:
        return None
    if hasattr(val, "magnitude"):
        if target_unit is not None:
            val = val.to(target_unit)
        return float(val.magnitude)
    return float(val)


def _grid_flatten_3d(arr):
    """Flatten a 3D array [nz, ny, nx] → 1D, return (flat, nx, ny, nz)."""
    a = _grid_strip(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected 3-D array (nz, ny, nx), got {a.ndim}-D")
    nz, ny, nx = a.shape
    return a.ravel(), nx, ny, nz


def _grid_flatten_2d(arr):
    """Flatten a 2D array [ny, nx] → 1D, return (flat, nx, ny)."""
    a = _grid_strip(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D array (ny, nx), got {a.ndim}-D")
    ny, nx = a.shape
    return a.ravel(), nx, ny


def compute_cape_cin(pressure_3d, temperature_c_3d, qvapor_3d,
                     height_agl_3d, psfc, t2, q2,
                     parcel_type="surface", top_m=None):
    """CAPE/CIN for every grid point (parallelized Rust).

    3-D inputs: shape (nz, ny, nx) — pressure (Pa), temperature (C),
    mixing ratio (kg/kg), height AGL (m).
    2-D inputs: shape (ny, nx) — surface pressure (Pa), T2m (K), Q2m (kg/kg).

    Returns (cape, cin, lcl_height, lfc_height) each shaped (ny, nx).
    """
    if _BACKEND == "gpu":
        cape, cin, lcl, lfc = _gpu_to_numpy(
            _load_gpu_calc().compute_cape_cin(
                _grid_strip(pressure_3d),
                _grid_strip(temperature_c_3d),
                _grid_strip(qvapor_3d),
                _grid_strip(height_agl_3d),
                _grid_strip(psfc),
                _grid_strip(t2),
                _grid_strip(q2),
                parcel_type=parcel_type,
                top_m=_scalar_strip(top_m, "m"),
            )
        )
        return (
            np.asarray(cape) * units("J/kg"),
            np.asarray(cin) * units("J/kg"),
            np.asarray(lcl) * units.m,
            np.asarray(lfc) * units.m,
        )
    p3, nx, ny, nz = _grid_flatten_3d(pressure_3d)
    t3 = _grid_strip(temperature_c_3d).ravel()
    q3 = _grid_strip(qvapor_3d).ravel()
    h3 = _grid_strip(height_agl_3d).ravel()
    ps = _grid_strip(psfc).ravel()
    t2v = _grid_strip(t2).ravel()
    q2v = _grid_strip(q2).ravel()
    cape, cin, lcl, lfc = _calc.compute_cape_cin(
        p3, t3, q3, h3, ps, t2v, q2v,
        nx, ny, nz, parcel_type, _scalar_strip(top_m, "m"),
    )
    shape = (ny, nx)
    return (
        np.asarray(cape).reshape(shape) * units("J/kg"),
        np.asarray(cin).reshape(shape) * units("J/kg"),
        np.asarray(lcl).reshape(shape) * units.m,
        np.asarray(lfc).reshape(shape) * units.m,
    )


def compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0):
    """Storm-relative helicity for every grid point.

    3-D inputs: shape (nz, ny, nx) — u/v wind (m/s), height AGL (m).
    top_m: integration depth in meters (default 1000 = 0-1 km).

    Returns SRH shaped (ny, nx) in m^2/s^2.
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(
            _load_gpu_calc().compute_srh(
                _grid_strip(u_3d),
                _grid_strip(v_3d),
                _grid_strip(height_agl_3d),
                top_m=_scalar_strip(top_m, "m"),
            )
        )
        return np.asarray(result) * units("m**2/s**2")
    u3, nx, ny, nz = _grid_flatten_3d(u_3d)
    v3 = _grid_strip(v_3d).ravel()
    h3 = _grid_strip(height_agl_3d).ravel()
    result = _calc.compute_srh(u3, v3, h3, nx, ny, nz, _scalar_strip(top_m, "m"))
    return np.asarray(result).reshape(ny, nx) * units("m**2/s**2")


def compute_shear(u_3d, v_3d, height_agl_3d, bottom_m=0.0, top_m=6000.0):
    """Bulk wind shear for every grid point.

    3-D inputs: shape (nz, ny, nx) — u/v wind (m/s), height AGL (m).
    bottom_m, top_m: shear layer bounds in meters AGL.

    Returns shear magnitude shaped (ny, nx) in m/s.
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(
            _load_gpu_calc().compute_shear(
                _grid_strip(u_3d),
                _grid_strip(v_3d),
                _grid_strip(height_agl_3d),
                bottom_m=_scalar_strip(bottom_m, "m"),
                top_m=_scalar_strip(top_m, "m"),
            )
        )
        return np.asarray(result) * units("m/s")
    u3, nx, ny, nz = _grid_flatten_3d(u_3d)
    v3 = _grid_strip(v_3d).ravel()
    h3 = _grid_strip(height_agl_3d).ravel()
    result = _calc.compute_shear(
        u3, v3, h3, nx, ny, nz, _scalar_strip(bottom_m, "m"), _scalar_strip(top_m, "m"),
    )
    return np.asarray(result).reshape(ny, nx) * units("m/s")


def compute_lapse_rate(temperature_c_3d, qvapor_3d, height_agl_3d,
                       bottom_km=0.0, top_km=3.0):
    """Environmental lapse rate for every grid point (C/km).

    3-D inputs: shape (nz, ny, nx).
    bottom_km, top_km: layer bounds in km AGL.

    Returns lapse rate shaped (ny, nx) in C/km.
    """
    t3, nx, ny, nz = _grid_flatten_3d(temperature_c_3d)
    q3 = _grid_strip(qvapor_3d).ravel()
    h3 = _grid_strip(height_agl_3d).ravel()
    result = _calc.compute_lapse_rate(
        t3, q3, h3, nx, ny, nz, _scalar_strip(bottom_km, "km"), _scalar_strip(top_km, "km"),
    )
    return np.asarray(result).reshape(ny, nx) * units("degC/km")


def compute_pw(qvapor_3d, pressure_3d):
    """Precipitable water for every grid point (mm).

    3-D inputs: shape (nz, ny, nx) — mixing ratio (kg/kg), pressure (Pa).

    Returns PW shaped (ny, nx) in mm.
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(
            _load_gpu_calc().compute_pw(
                _grid_strip(qvapor_3d),
                _grid_strip(pressure_3d),
            )
        )
        return np.asarray(result) * units.mm
    q3, nx, ny, nz = _grid_flatten_3d(qvapor_3d)
    p3 = _grid_strip(pressure_3d).ravel()
    result = _calc.compute_pw(q3, p3, nx, ny, nz)
    return np.asarray(result).reshape(ny, nx) * units.mm


def compute_stp(cape, lcl_height, srh_1km, shear_6km):
    """Significant Tornado Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    - cape: MLCAPE (J/kg)
    - lcl_height: LCL height (m AGL)
    - srh_1km: 0-1 km SRH (m^2/s^2)
    - shear_6km: 0-6 km bulk shear (m/s)

    Returns STP shaped (ny, nx), dimensionless.
    """
    c, nx, ny = _grid_flatten_2d(cape)
    l = _grid_strip(lcl_height).ravel()
    s = _grid_strip(srh_1km).ravel()
    sh = _grid_strip(shear_6km).ravel()
    result = _calc.compute_stp(c, l, s, sh)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_scp(mucape, srh_3km, shear_6km):
    """Supercell Composite Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    - mucape: MUCAPE (J/kg)
    - srh_3km: 0-3 km SRH (m^2/s^2)
    - shear_6km: 0-6 km bulk shear (m/s)

    Returns SCP shaped (ny, nx), dimensionless.
    """
    c, nx, ny = _grid_flatten_2d(mucape)
    s = _grid_strip(srh_3km).ravel()
    sh = _grid_strip(shear_6km).ravel()
    result = _calc.compute_scp(c, s, sh)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_ehi(cape, srh):
    """Energy-Helicity Index on pre-computed 2-D fields.

    EHI = (CAPE * SRH) / 160000.

    All inputs: shape (ny, nx).

    Returns EHI shaped (ny, nx), dimensionless.
    """
    c, nx, ny = _grid_flatten_2d(cape)
    s = _grid_strip(srh).ravel()
    result = _calc.compute_ehi(c, s)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg):
    """Significant Hail Parameter (SHIP) on 2-D fields.

    All inputs: shape (ny, nx).
    - cape: MUCAPE (J/kg)
    - shear06: 0-6 km bulk shear (m/s)
    - t500: 500 hPa temperature (C)
    - lr_700_500: 700-500 hPa lapse rate (C/km)
    - mixing_ratio_gkg: Mixing ratio (g/kg)

    Returns SHIP shaped (ny, nx), dimensionless.
    """
    c, nx, ny = _grid_flatten_2d(cape)
    sh = _grid_strip(shear06).ravel()
    t5 = _grid_strip(t500).ravel()
    lr = _grid_strip(lr_700_500).ravel()
    mr = _grid_strip(mixing_ratio_gkg).ravel()
    result = _calc.significant_hail_parameter(c, sh, t5, lr, mr, nx, ny)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_dcp(dcape, mu_cape, shear06, mu_mixing_ratio):
    """Derecho Composite Parameter (DCP) on 2-D fields.

    DCP = (DCAPE/980) * (MUCAPE/2000) * (SHEAR_06/20) * (MU_MR/11).

    All inputs: shape (ny, nx).

    Returns DCP shaped (ny, nx), dimensionless.
    """
    d, nx, ny = _grid_flatten_2d(dcape)
    mc = _grid_strip(mu_cape).ravel()
    sh = _grid_strip(shear06).ravel()
    mr = _grid_strip(mu_mixing_ratio).ravel()
    result = _calc.derecho_composite_parameter(d, mc, sh, mr, nx, ny)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_grid_scp(mu_cape, srh, shear_06, mu_cin):
    """Enhanced Supercell Composite with CIN term on 2-D fields.

    SCP = (MUCAPE/1000) * (SRH/50) * (SHEAR_06/40) * CIN_term.

    All inputs: shape (ny, nx).

    Returns SCP shaped (ny, nx), dimensionless.
    """
    mc, nx, ny = _grid_flatten_2d(mu_cape)
    s = _grid_strip(srh).ravel()
    sh = _grid_strip(shear_06).ravel()
    ci = _grid_strip(mu_cin).ravel()
    result = _calc.grid_supercell_composite_parameter(mc, s, sh, ci, nx, ny)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def compute_grid_critical_angle(u_storm, v_storm, u_shear, v_shear):
    """Critical angle on 2-D fields.

    Returns angle in degrees (0-180). Values near 90 favor tornadogenesis.

    All inputs: shape (ny, nx).
    """
    us, nx, ny = _grid_flatten_2d(u_storm)
    vs = _grid_strip(v_storm).ravel()
    ush = _grid_strip(u_shear).ravel()
    vsh = _grid_strip(v_shear).ravel()
    result = _calc.grid_critical_angle(us, vs, ush, vsh, nx, ny)
    return np.asarray(result).reshape(ny, nx) * units.degree


def composite_reflectivity(refl_3d):
    """Composite reflectivity (column max) from a 3-D reflectivity field.

    Input: shape (nz, ny, nx) — reflectivity in dBZ.

    Returns composite reflectivity shaped (ny, nx) in dBZ.
    """
    r3, nx, ny, nz = _grid_flatten_3d(refl_3d)
    result = _calc.composite_reflectivity_from_refl(r3, nx, ny, nz)
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


def composite_reflectivity_from_hydrometeors(pressure_3d, temperature_c_3d,
                                             qrain_3d, qsnow_3d, qgraup_3d):
    """Composite reflectivity from hydrometeor mixing ratios.

    All 3-D inputs: shape (nz, ny, nx).
    - pressure_3d: Pa
    - temperature_c_3d: Celsius
    - qrain_3d, qsnow_3d, qgraup_3d: kg/kg

    Returns composite reflectivity shaped (ny, nx) in dBZ.
    """
    if _BACKEND == "gpu":
        result = _gpu_to_numpy(
            _load_gpu_calc().composite_reflectivity_from_hydrometeors(
                _grid_strip(pressure_3d),
                _grid_strip(temperature_c_3d),
                _grid_strip(qrain_3d),
                _grid_strip(qsnow_3d),
                _grid_strip(qgraup_3d),
            )
        )
        return np.asarray(result) * units.dimensionless
    p3, nx, ny, nz = _grid_flatten_3d(pressure_3d)
    t3 = _grid_strip(temperature_c_3d).ravel()
    qr = _grid_strip(qrain_3d).ravel()
    qs = _grid_strip(qsnow_3d).ravel()
    qg = _grid_strip(qgraup_3d).ravel()
    result = _calc.composite_reflectivity_from_hydrometeors(
        p3, t3, qr, qs, qg, nx, ny, nz,
    )
    return np.asarray(result).reshape(ny, nx) * units.dimensionless


# ============================================================================
# ============================================================================
# xarray Dataset wrappers
# ============================================================================

def parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint):
    """Calculate parcel profile and return as xarray Dataset with LCL inserted.

    Wraps :func:`parcel_profile_with_lcl` and returns the result as an
    ``xarray.Dataset`` with pressure as the ``isobaric`` coordinate and
    data variables for ambient temperature, ambient dewpoint, and parcel
    temperature.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr

    p, ambient_temperature, ambient_dew_point, parcel_temperature = parcel_profile_with_lcl(
        pressure,
        temperature,
        dewpoint,
    )
    return xr.Dataset(
        {
            "ambient_temperature": (
                ("isobaric",),
                ambient_temperature,
                {"standard_name": "air_temperature"},
            ),
            "ambient_dew_point": (
                ("isobaric",),
                ambient_dew_point,
                {"standard_name": "dew_point_temperature"},
            ),
            "parcel_temperature": (
                ("isobaric",),
                parcel_temperature,
                {"long_name": "air_temperature_of_lifted_parcel"},
            ),
        },
        coords={
            "isobaric": (
                "isobaric",
                p.magnitude,
                {"units": str(p.units), "standard_name": "air_pressure"},
            )
        },
    )


def isentropic_interpolation_as_dataset(levels, temperature, *args,
                                         max_iters=50, eps=1e-6,
                                         bottom_up_search=True, pressure=None):
    """Interpolate to isentropic surfaces and return as xarray Dataset.

    Wraps :func:`isentropic_interpolation` and packages the result as an
    ``xarray.Dataset`` with ``isentropic_level`` as the vertical coordinate.

    Parameters
    ----------
    levels : array Quantity (K)
        Desired theta surfaces.
    temperature : xarray.DataArray
        Temperature array with a vertical (pressure) dimension.
    *args : xarray.DataArray
        Additional fields to interpolate.
    max_iters : int, optional
    eps : float, optional
    bottom_up_search : bool, optional
    pressure : xarray.DataArray, optional
        Pressure array if vertical coordinate is not pressure.

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr

    all_args = xr.broadcast(temperature, *args)
    if pressure is None:
        pressure = all_args[0].metpy.vertical

    pressure_units = getattr(getattr(pressure, "metpy", None), "units", None)
    if pressure_units is None:
        _, pressure_units = _as_array_with_unit(pressure)
    if units.get_dimensionality(pressure_units) != units.get_dimensionality("[pressure]"):
        raise ValueError(
            "The pressure array/vertical coordinate for the passed in data does not appear "
            "to be pressure. Pass pressure explicitly with proper units."
        )

    vertical_dim_num = all_args[0].metpy.find_axis_number("vertical")
    vertical_dim_name = all_args[0].dims[vertical_dim_num]
    ret = isentropic_interpolation(
        levels,
        pressure,
        all_args[0].metpy.unit_array,
        *(arg.metpy.unit_array for arg in all_args[1:]),
        vertical_dim=vertical_dim_num,
        temperature_out=True,
        max_iters=max_iters,
        eps=eps,
        bottom_up_search=bottom_up_search,
    )

    new_coords = {
        "isentropic_level": xr.DataArray(
            levels.magnitude,
            dims=("isentropic_level",),
            coords={"isentropic_level": levels.magnitude},
            name="isentropic_level",
            attrs={"units": str(levels.units), "positive": "up"},
        ),
        **{
            key: value
            for key, value in all_args[0].coords.items()
            if key != vertical_dim_name
        },
    }
    new_dims = [
        dim if dim != vertical_dim_name else "isentropic_level"
        for dim in all_args[0].dims
    ]

    return xr.Dataset(
        {
            "pressure": (
                new_dims,
                ret[0],
                {"standard_name": "air_pressure"},
            ),
            "temperature": (
                new_dims,
                ret[1],
                {"standard_name": "air_temperature"},
            ),
            **{
                (all_args[i].name or f"field_{i - 1}"): (
                    new_dims,
                    ret[i + 1],
                    all_args[i].attrs,
                )
                for i in range(1, len(all_args))
            },
        },
        coords=new_coords,
    )


def zoom_xarray(input_field, zoom, output=None, order=3, mode="constant",
                cval=0.0, prefilter=True):
    """Zoom/interpolate an xarray DataArray using scipy.ndimage.zoom.

    Parameters
    ----------
    input_field : xarray.DataArray
        2-D field to zoom.
    zoom : float or sequence of float
        Zoom factor(s) for each axis.
    output, order, mode, cval, prefilter
        Passed directly to :func:`scipy.ndimage.zoom`.

    Returns
    -------
    xarray.DataArray
        Zoomed field with scaled coordinates.
    """
    from scipy.ndimage import zoom as scipy_zoom

    field = input_field.metpy.dequantify() if hasattr(input_field, "metpy") else input_field
    zoomed_data = scipy_zoom(
        field.data,
        zoom,
        output=output,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    if not np.iterable(zoom):
        zoom = tuple(zoom for _ in field.dims)

    zoomed_dim_coords = {}
    for dim_name, dim_zoom in zip(field.dims, zoom, strict=False):
        if dim_name in field.coords:
            zoomed_dim_coords[dim_name] = scipy_zoom(
                field[dim_name].data,
                dim_zoom,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )
    if hasattr(field, "metpy_crs"):
        zoomed_dim_coords["metpy_crs"] = field.metpy_crs
    return xr.DataArray(
        zoomed_data,
        dims=field.dims,
        coords=zoomed_dim_coords,
        attrs=field.attrs,
    )


# ---------------------------------------------------------------------------
# Interpolation functions (MetPy compatibility)
# ---------------------------------------------------------------------------

def _idw_fallback(obs_x, obs_y, obs_values, grid_x, grid_y, radius,
                  min_neighbors, kind_int, kappa, gamma):
    """Pure-Python fallback for inverse-distance interpolation using cKDTree."""
    from scipy.spatial import cKDTree
    obs_points = np.column_stack([obs_x, obs_y])
    query_points = np.column_stack([grid_x, grid_y])
    tree = cKDTree(obs_points)
    result = np.full(len(grid_x), np.nan)
    for i, qp in enumerate(query_points):
        idx = tree.query_ball_point(qp, radius)
        if len(idx) < min_neighbors:
            continue
        dists = np.sqrt(np.sum((obs_points[idx] - qp) ** 2, axis=1))
        dists = np.maximum(dists, 1e-12)
        if kind_int == 0:  # cressman
            weights = (radius ** 2 - dists ** 2) / (radius ** 2 + dists ** 2)
        else:  # barnes
            weights = np.exp(-dists ** 2 / (kappa if kappa else radius ** 2))
        weights = np.maximum(weights, 0.0)
        wsum = weights.sum()
        if wsum > 0:
            result[i] = np.sum(weights * np.asarray(obs_values)[idx]) / wsum
    return result


def _call_idw(obs_x, obs_y, obs_values, grid_x, grid_y, r,
              gamma, kappa, min_neighbors, kind):
    """Call Rust IDW if available, otherwise fall back to Python."""
    kind_int = 0 if kind == 'cressman' else 1
    _gamma = gamma if gamma is not None else 1.0
    _kappa = kappa if kappa is not None else (r ** 2)
    try:
        from metrust._metrust import interpolate as _interp
        return _interp.inverse_distance_to_points(
            np.asarray(obs_x, dtype=np.float64).ravel(),
            np.asarray(obs_y, dtype=np.float64).ravel(),
            np.asarray(obs_values, dtype=np.float64).ravel(),
            np.asarray(grid_x, dtype=np.float64).ravel(),
            np.asarray(grid_y, dtype=np.float64).ravel(),
            float(r), int(min_neighbors), kind_int, float(_kappa), float(_gamma),
        )
    except AttributeError:
        return _idw_fallback(
            np.asarray(obs_x, dtype=np.float64).ravel(),
            np.asarray(obs_y, dtype=np.float64).ravel(),
            np.asarray(obs_values, dtype=np.float64).ravel(),
            np.asarray(grid_x, dtype=np.float64).ravel(),
            np.asarray(grid_y, dtype=np.float64).ravel(),
            float(r), int(min_neighbors), kind_int, float(_kappa), float(_gamma),
        )


def inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r,
                              gamma=None, kappa=None, min_neighbors=3,
                              kind='cressman'):
    """Interpolate using inverse-distance weighting to a grid.

    Parameters match :func:`metpy.interpolate.inverse_distance_to_grid`.
    """
    xp = np.asarray(xp, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    variable = np.asarray(variable, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64)
    grid_y = np.asarray(grid_y, dtype=np.float64)
    target_shape = grid_x.shape
    result = _call_idw(xp, yp, variable, grid_x.ravel(), grid_y.ravel(),
                       r, gamma, kappa, min_neighbors, kind)
    return np.asarray(result).reshape(target_shape)


def inverse_distance_to_points(points, values, xi, r,
                                gamma=None, kappa=None, min_neighbors=3,
                                kind='cressman'):
    """Interpolate using inverse-distance weighting to arbitrary points.

    Parameters match :func:`metpy.interpolate.inverse_distance_to_points`.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    return np.asarray(_call_idw(
        points[:, 0], points[:, 1], values,
        xi[:, 0], xi[:, 1],
        r, gamma, kappa, min_neighbors, kind,
    ))


def remove_nan_observations(x, y, z):
    """Remove all observations where any of x, y, or z is NaN."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    return x[mask], y[mask], z[mask]


def remove_observations_below_value(x, y, z, val=0):
    """Remove observations where z < val."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = z >= val
    return x[mask], y[mask], z[mask]


def remove_repeat_coordinates(x, y, z):
    """Remove duplicate (x, y) coordinate pairs, keeping the first."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    coords = np.column_stack([x, y])
    _, idx = np.unique(coords, axis=0, return_index=True)
    idx = np.sort(idx)
    return x[idx], y[idx], z[idx]


def interpolate_nans_1d(x, y, kind='linear'):
    """Interpolate NaN values in a 1D array using scipy interp1d."""
    from scipy.interpolate import interp1d
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~np.isnan(y)
    if mask.all() or not mask.any():
        return y.copy()
    f = interp1d(x[mask], y[mask], kind=kind, bounds_error=False,
                 fill_value=np.nan)
    out = y.copy()
    out[~mask] = f(x[~mask])
    return out


def interpolate_to_grid(x, y, z, interp_type='linear', hres=50000,
                         minimum_neighbors=3, search_radius=None,
                         gamma=None, kappa=None, rbf_func='linear',
                         rbf_smooth=0):
    """Interpolate observations to a grid.

    Parameters match :func:`metpy.interpolate.interpolate_to_grid`.
    Supports interp_type = 'linear', 'cressman', 'barnes', 'rbf'.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_x_1d = np.arange(x_min, x_max + hres, hres)
    grid_y_1d = np.arange(y_min, y_max + hres, hres)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    if interp_type in ('cressman', 'barnes'):
        r = search_radius if search_radius is not None else hres * 5
        img = inverse_distance_to_grid(
            x, y, z, grid_x, grid_y, r,
            gamma=gamma, kappa=kappa,
            min_neighbors=minimum_neighbors, kind=interp_type,
        )
    elif interp_type == 'rbf':
        from scipy.interpolate import Rbf
        rbf = Rbf(x, y, z, function=rbf_func, smooth=rbf_smooth)
        img = rbf(grid_x, grid_y)
    else:  # 'linear' or other scipy methods
        from scipy.interpolate import griddata
        img = griddata(np.column_stack([x, y]), z,
                       (grid_x, grid_y), method=interp_type)
    return grid_x, grid_y, img


def interpolate_to_points(points, values, xi, interp_type='linear',
                           minimum_neighbors=3, search_radius=None,
                           gamma=None, kappa=None, rbf_func='linear',
                           rbf_smooth=0):
    """Interpolate observations to arbitrary points.

    Parameters match :func:`metpy.interpolate.interpolate_to_points`.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)

    if interp_type in ('cressman', 'barnes'):
        r = search_radius if search_radius is not None else 100000.0
        return inverse_distance_to_points(
            points, values, xi, r,
            gamma=gamma, kappa=kappa,
            min_neighbors=minimum_neighbors, kind=interp_type,
        )
    elif interp_type == 'rbf':
        from scipy.interpolate import Rbf
        rbf = Rbf(points[:, 0], points[:, 1], values,
                  function=rbf_func, smooth=rbf_smooth)
        return rbf(xi[:, 0], xi[:, 1])
    else:
        from scipy.interpolate import griddata
        return griddata(points, values, xi, method=interp_type)


def natural_neighbor_to_grid(xp, yp, variable, grid_x, grid_y):
    """Interpolate using natural-neighbor-like method to a grid.

    Uses scipy griddata with ``method='cubic'`` (falls back to 'nearest'
    if cubic fails).  True Voronoi-based natural neighbor would require
    MetPy's Delaunay code.
    """
    xp = np.asarray(xp, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    variable = np.asarray(variable, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64)
    grid_y = np.asarray(grid_y, dtype=np.float64)
    from scipy.interpolate import griddata
    pts = np.column_stack([xp, yp])
    try:
        result = griddata(pts, variable, (grid_x, grid_y), method='cubic')
    except Exception:
        result = griddata(pts, variable, (grid_x, grid_y), method='nearest')
    return result


def natural_neighbor_to_points(points, values, xi):
    """Interpolate using natural-neighbor-like method to arbitrary points.

    Uses scipy griddata with ``method='cubic'`` (falls back to 'nearest').
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    from scipy.interpolate import griddata
    try:
        return griddata(points, values, xi, method='cubic')
    except Exception:
        return griddata(points, values, xi, method='nearest')


def interpolate_to_isosurface(level_var, interp_var, level,
                               bottom_up_search=True):
    """Interpolate a variable to an isosurface of another variable.

    Thin wrapper around :func:`isentropic_interpolation` for the common
    case of extracting a single level.
    """
    level_var = np.asarray(level_var, dtype=np.float64)
    interp_var = np.asarray(interp_var, dtype=np.float64)
    level = float(level)
    nz = level_var.shape[0]
    shape_2d = level_var.shape[1:]
    result = np.full(shape_2d, np.nan)
    rng = range(nz - 1) if bottom_up_search else range(nz - 2, -1, -1)
    for k in rng:
        k_above = k + 1 if bottom_up_search else k - 1
        if k_above < 0 or k_above >= nz:
            continue
        below = level_var[k]
        above = level_var[k_above]
        mask_lower = np.minimum(below, above) <= level
        mask_upper = np.maximum(below, above) >= level
        mask = mask_lower & mask_upper & np.isnan(result)
        if not mask.any():
            continue
        denom = above[mask] - below[mask]
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        frac = (level - below[mask]) / denom
        result[mask] = (interp_var[k][mask] +
                        frac * (interp_var[k_above][mask] - interp_var[k][mask]))
    return result


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan,
                   return_list_always=False):
    """Interpolate 1D data along an axis.

    Parameters match :func:`metpy.interpolate.interpolate_1d`.
    """
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    results = []
    for a in args:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            results.append(np.interp(x, xp, a, left=fill_value,
                                     right=fill_value))
        else:
            # Move interpolation axis to front, interp along it, move back
            a_moved = np.moveaxis(a, axis, 0)
            shape_rest = a_moved.shape[1:]
            out_shape = (len(x),) + shape_rest
            out = np.full(out_shape, fill_value)
            for idx in np.ndindex(shape_rest):
                out[(slice(None),) + idx] = np.interp(
                    x, xp, a_moved[(slice(None),) + idx],
                    left=fill_value, right=fill_value,
                )
            results.append(np.moveaxis(out, 0, axis))
    if len(results) == 1 and not return_list_always:
        return results[0]
    return results


def log_interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan):
    """Interpolate in log-space along an axis.

    Parameters match :func:`metpy.interpolate.log_interpolate_1d`.
    """
    log_x = np.log(np.asarray(x, dtype=np.float64))
    log_xp = np.log(np.asarray(xp, dtype=np.float64))
    return interpolate_1d(log_x, log_xp, *args, axis=axis,
                          fill_value=fill_value, return_list_always=True)


def cross_section(data, start, end, steps=100, interp_type='linear'):
    """Extract a cross-section from gridded data.

    Forwards to :func:`metpy.interpolate.cross_section` if available.
    """
    try:
        from metpy.interpolate import cross_section as _mp_cross_section
        return _mp_cross_section(data, start, end, steps=steps,
                                 interp_type=interp_type)
    except ImportError:
        raise NotImplementedError(
            "cross_section requires metpy with xarray/cartopy support. "
            "Install metpy to use this function."
        )


def interpolate_to_slice(data, points, interp_type='linear'):
    """Interpolate data to a slice along a set of points.

    Forwards to :func:`metpy.interpolate.interpolate_to_slice` if available.
    """
    try:
        from metpy.interpolate import interpolate_to_slice as _mp_slice
        return _mp_slice(data, points, interp_type=interp_type)
    except ImportError:
        raise NotImplementedError(
            "interpolate_to_slice requires metpy with xarray/cartopy support. "
            "Install metpy to use this function."
        )


def geodesic(crs, start, end, steps):
    """Calculate points along a geodesic between two points.

    Forwards to :func:`metpy.interpolate.geodesic` if available, otherwise
    uses pyproj.Geod for the calculation.
    """
    try:
        from metpy.interpolate import geodesic as _mp_geodesic
        return _mp_geodesic(crs, start, end, steps)
    except ImportError:
        pass
    try:
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        lon_start, lat_start = start
        lon_end, lat_end = end
        pts = g.npts(lon_start, lat_start, lon_end, lat_end, steps)
        lons = [lon_start] + [p[0] for p in pts] + [lon_end]
        lats = [lat_start] + [p[1] for p in pts] + [lat_end]
        return np.array(list(zip(lons, lats)))
    except ImportError:
        raise NotImplementedError(
            "geodesic requires either metpy or pyproj. "
            "Install one of them to use this function."
        )


# __all__ -- explicit public API
# ============================================================================

__all__ = [
    "get_backend",
    "set_backend",
    "use_backend",
    # thermo -- existing
    "potential_temperature",
    "equivalent_potential_temperature",
    "saturation_vapor_pressure",
    "saturation_mixing_ratio",
    "wet_bulb_temperature",
    "lfc",
    "el",
    "lcl",
    "dewpoint_from_relative_humidity",
    "relative_humidity_from_dewpoint",
    "virtual_temperature",
    "virtual_temperature_from_dewpoint",
    "cape_cin",
    "mixing_ratio",
    "showalter_index",
    "k_index",
    "total_totals",
    "downdraft_cape",
    "cross_totals",
    "vertical_totals",
    "sweat_index",
    "brunt_vaisala_frequency",
    "brunt_vaisala_period",
    "brunt_vaisala_frequency_squared",
    "precipitable_water",
    "parcel_profile_with_lcl",
    "moist_air_gas_constant",
    "moist_air_specific_heat_pressure",
    "moist_air_poisson_exponent",
    "water_latent_heat_vaporization",
    "water_latent_heat_melting",
    "water_latent_heat_sublimation",
    "relative_humidity_wet_psychrometric",
    "weighted_continuous_average",
    "get_perturbation",
    "add_height_to_pressure",
    "add_pressure_to_height",
    "thickness_hydrostatic",
    "specific_humidity_from_mixing_ratio",
    "thickness_hydrostatic_from_relative_humidity",
    "vapor_pressure",
    # thermo -- new
    "ccl",
    "lifted_index",
    "density",
    "dewpoint",
    "dewpoint_from_specific_humidity",
    "dry_lapse",
    "dry_static_energy",
    "exner_function",
    "find_intersections",
    "geopotential_to_height",
    "get_layer",
    "get_layer_heights",
    "height_to_geopotential",
    "isentropic_interpolation",
    "mean_pressure_weighted",
    "mixed_layer",
    "mixed_layer_cape_cin",
    "mixing_ratio_from_relative_humidity",
    "mixing_ratio_from_specific_humidity",
    "moist_lapse",
    "moist_static_energy",
    "montgomery_streamfunction",
    "most_unstable_cape_cin",
    "parcel_profile",
    "reduce_point_density",
    "relative_humidity_from_mixing_ratio",
    "relative_humidity_from_specific_humidity",
    "saturation_equivalent_potential_temperature",
    "scale_height",
    "specific_humidity_from_dewpoint",
    "static_stability",
    "surface_based_cape_cin",
    "temperature_from_potential_temperature",
    "vertical_velocity",
    "vertical_velocity_pressure",
    "virtual_potential_temperature",
    "wet_bulb_potential_temperature",
    "get_mixed_layer_parcel",
    "get_most_unstable_parcel",
    "psychrometric_vapor_pressure",
    "frost_point",
    # aliases
    "mixed_parcel",
    "most_unstable_parcel",
    "psychrometric_vapor_pressure_wet",
    # wind
    "wind_speed",
    "wind_direction",
    "wind_components",
    "bulk_shear",
    "mean_wind",
    "storm_relative_helicity",
    "bunkers_storm_motion",
    "corfidi_storm_motion",
    "friction_velocity",
    "tke",
    "gradient_richardson_number",
    # kinematics -- existing
    "divergence",
    "vorticity",
    "absolute_vorticity",
    "advection",
    "frontogenesis",
    "geostrophic_wind",
    "ageostrophic_wind",
    "potential_vorticity_baroclinic",
    "potential_vorticity_barotropic",
    "normal_component",
    "tangential_component",
    "unit_vectors_from_cross_section",
    "vector_derivative",
    # kinematics -- new
    "absolute_momentum",
    "coriolis_parameter",
    "cross_section_components",
    "curvature_vorticity",
    "inertial_advective_wind",
    "kinematic_flux",
    "q_vector",
    "shear_vorticity",
    "shearing_deformation",
    "stretching_deformation",
    "total_deformation",
    "geospatial_gradient",
    "geospatial_laplacian",
    # severe
    "significant_tornado_parameter",
    "supercell_composite_parameter",
    "critical_angle",
    "boyden_index",
    "bulk_richardson_number",
    "convective_inhibition_depth",
    "dendritic_growth_zone",
    "fosberg_fire_weather_index",
    "freezing_rain_composite",
    "haines_index",
    "hot_dry_windy",
    "warm_nose_check",
    "galvez_davison_index",
    # grid composites
    "compute_cape_cin",
    "compute_srh",
    "compute_shear",
    "compute_lapse_rate",
    "compute_pw",
    "compute_stp",
    "compute_scp",
    "compute_ehi",
    "compute_ship",
    "compute_dcp",
    "compute_grid_scp",
    "compute_grid_critical_angle",
    "composite_reflectivity",
    "composite_reflectivity_from_hydrometeors",
    # atmo
    "pressure_to_height_std",
    "height_to_pressure_std",
    "altimeter_to_station_pressure",
    "station_to_altimeter_pressure",
    "altimeter_to_sea_level_pressure",
    "sigma_to_pressure",
    "heat_index",
    "windchill",
    "apparent_temperature",
    # smooth
    "smooth_gaussian",
    "smooth_rectangular",
    "smooth_circular",
    "smooth_n_point",
    "smooth_window",
    "gradient",
    "gradient_x",
    "gradient_y",
    "laplacian",
    "first_derivative",
    "second_derivative",
    "lat_lon_grid_deltas",
    # utils
    "angle_to_direction",
    "parse_angle",
    "find_bounding_indices",
    "nearest_intersection_idx",
    "resample_nn_1d",
    "find_peaks",
    "peak_persistence",
    "azimuth_range_to_lat_lon",
    "advection_3d",
    # exceptions
    "InvalidSoundingError",
    # xarray dataset wrappers
    "parcel_profile_with_lcl_as_dataset",
    "isentropic_interpolation_as_dataset",
    "zoom_xarray",
    # interpolation
    "inverse_distance_to_grid",
    "inverse_distance_to_points",
    "remove_nan_observations",
    "remove_observations_below_value",
    "remove_repeat_coordinates",
    "interpolate_nans_1d",
    "interpolate_to_grid",
    "interpolate_to_points",
    "natural_neighbor_to_grid",
    "natural_neighbor_to_points",
    "interpolate_to_isosurface",
    "interpolate_1d",
    "log_interpolate_1d",
    "cross_section",
    "interpolate_to_slice",
    "geodesic",
]

_COMPAT_ALIASES = {
    "significant_tornado": "significant_tornado_parameter",
    "supercell_composite": "supercell_composite_parameter",
    "total_totals_index": "total_totals",
}

__all__.extend(sorted(_COMPAT_ALIASES))
__all__ = sorted(set(__all__))


def __getattr__(name):
    if name in _COMPAT_ALIASES:
        return globals()[_COMPAT_ALIASES[name]]
    raise AttributeError(f"module 'metrust.calc' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(__all__))


_METPY_SIGNATURE_HOOK = None


def _apply_metpy_signatures():
    """Mirror MetPy signatures for shared public wrappers when MetPy is available."""
    _metpy_calc = sys.modules.get("metpy.calc")
    if _metpy_calc is None:
        return

    for name in __all__:
        metpy_obj = getattr(_metpy_calc, name, None)
        if metpy_obj is None or not callable(metpy_obj):
            continue

        target_name = _COMPAT_ALIASES.get(name, name)
        wrapper = globals().get(target_name)
        if wrapper is None or not callable(wrapper):
            continue

        try:
            wrapper.__signature__ = inspect.signature(metpy_obj)
        except (TypeError, ValueError):
            continue


class _MetPyCalcSignatureHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Apply MetPy signature mirroring when metpy.calc imports after metrust.calc."""

    def __init__(self):
        self._wrapped_loader = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "metpy.calc":
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        self._wrapped_loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        if self._wrapped_loader is not None and hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module):
        if self._wrapped_loader is None:
            raise ImportError("metpy.calc signature hook missing wrapped loader")
        self._wrapped_loader.exec_module(module)
        _apply_metpy_signatures()
        _remove_metpy_signature_hook()


def _install_metpy_signature_hook():
    global _METPY_SIGNATURE_HOOK
    if _METPY_SIGNATURE_HOOK is not None or "metpy.calc" in sys.modules:
        return
    _METPY_SIGNATURE_HOOK = _MetPyCalcSignatureHook()
    sys.meta_path.insert(0, _METPY_SIGNATURE_HOOK)


def _remove_metpy_signature_hook():
    global _METPY_SIGNATURE_HOOK
    hook = _METPY_SIGNATURE_HOOK
    if hook is None:
        return
    try:
        sys.meta_path.remove(hook)
    except ValueError:
        pass
    _METPY_SIGNATURE_HOOK = None


_apply_metpy_signatures()
_install_metpy_signature_hook()
