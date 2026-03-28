"""metrust.calc -- Drop-in replacement for metpy.calc

Every public function accepts and returns Pint Quantity objects, matching
the MetPy API exactly.  Internally, units are stripped to the convention
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
    return val


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

    return lat_lon_grid_deltas(lat_arr, lon_arr)


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


def _svp_with_phase(t_c, phase):
    """SVP in hPa with explicit phase selection."""
    t_k = t_c + _ZEROCNK
    if phase == "ice":
        return _svp_ice_pa(t_k) / 100.0
    # auto: ice below T0
    if t_k > _T0:
        return _calc.saturation_vapor_pressure(t_c)  # liquid, already hPa
    return _svp_ice_pa(t_k) / 100.0

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
    p = _as_float(p_raw)
    t = _as_float(t_raw)
    _EPS = 0.6219569100577033
    es = _svp_with_phase(t, phase)
    return _attach(max((_EPS * es / (p - es)), 0.0) / 1000.0, "kg/kg")


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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    result = np.asarray(_calc.lfc(p, t, td), dtype=np.float64).ravel()
    if result.size >= 2:
        return result[0] * units.hPa, result[1] * units.degC
    return result[0] * units.hPa, np.nan * units.degC


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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    result = np.asarray(_calc.el(p, t, td), dtype=np.float64).ravel()
    if result.size >= 2:
        return result[0] * units.hPa, result[1] * units.degC
    return result[0] * units.hPa, np.nan * units.degC


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    p_lcl, t_lcl = _calc.lcl(p, t, td)
    return p_lcl * units.hPa, t_lcl * units.degC


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
        # MetPy calling convention: 4th arg is a parcel temperature profile.
        # Integrate CAPE/CIN directly using the provided parcel profile,
        # matching MetPy's integration: g * (Tv_p - Tv_e) / Tv_e * dz
        t_parcel = _as_1d(_strip(fourth, "degC"))
        # Compute height from hypsometric equation using environment
        h_calc = np.zeros(len(p))
        for i in range(1, len(p)):
            if p[i] <= 0 or p[i-1] <= 0:
                h_calc[i] = h_calc[i-1]
                continue
            tv_mean = (_calc.virtual_temp(t[i-1], p[i-1], td[i-1])
                       + _calc.virtual_temp(t[i], p[i], td[i])) / 2.0 + 273.15
            h_calc[i] = h_calc[i-1] + (287.04749 * tv_mean / 9.80665) * np.log(p[i-1] / p[i])

        # Integrate CAPE/CIN: trapezoidal rule
        # Two-pass approach matching MetPy:
        #   Pass 1: Find all buoyancy values to locate LFC and EL
        #   Pass 2: Integrate CAPE between LFC-EL, CIN between LCL-LFC
        #
        # CIN is the negative area where the parcel is cooler than the
        # environment.  For surface-based parcels, the parcel may be
        # positively buoyant near the surface (superadiabatic layer),
        # then negatively buoyant (the cap/CIN), then positively buoyant
        # again above the LFC.  We need to capture that middle negative
        # layer, not just stop at the first positive.

        # Find LCL (where T_parcel ≈ Td, i.e. parcel becomes saturated)
        lcl_idx = 0
        for i in range(1, len(p)):
            if t_parcel[i] <= t_parcel[0] - 1.0:  # parcel has cooled — above LCL
                lcl_idx = i
                break

        # Compute buoyancy at each level
        buoyancy = np.zeros(len(p))
        for i in range(len(p)):
            if p[i] <= 0:
                continue
            tv_e = _calc.virtual_temp(t[i], p[i], td[i]) + 273.15
            tv_p = _calc.virtual_temp(t_parcel[i], p[i], t_parcel[i]) + 273.15
            if tv_e > 0:
                buoyancy[i] = (tv_p - tv_e) / tv_e

        # Find LFC: first crossing from negative to positive buoyancy above LCL.
        # If the parcel is positively buoyant from the start (no cap), LFC = 0.
        lfc_idx = None
        for i in range(1, len(p)):
            if buoyancy[i] > 0 and buoyancy[i-1] <= 0:
                lfc_idx = i
                break  # first crossing is the LFC

        # If no negative-to-positive crossing found, check if the parcel is
        # positively buoyant everywhere (no cap at all) — LFC is the surface.
        if lfc_idx is None and any(buoyancy[i] > 0 for i in range(1, len(p))):
            lfc_idx = 0

        # Find EL: last crossing from positive to negative after LFC
        el_idx = len(p) - 1
        if lfc_idx is not None:
            for i in range(lfc_idx + 1, len(p)):
                if buoyancy[i] <= 0 and buoyancy[i-1] > 0:
                    el_idx = i

        cape_val = 0.0
        cin_val = 0.0
        for i in range(1, len(p)):
            if p[i] <= 0:
                continue
            tv_e_lo = _calc.virtual_temp(t[i-1], p[i-1], td[i-1]) + 273.15
            tv_e_hi = _calc.virtual_temp(t[i], p[i], td[i]) + 273.15
            tv_p_lo = _calc.virtual_temp(t_parcel[i-1], p[i-1], t_parcel[i-1]) + 273.15
            tv_p_hi = _calc.virtual_temp(t_parcel[i], p[i], t_parcel[i]) + 273.15
            dz = h_calc[i] - h_calc[i-1]
            if abs(dz) < 1e-6 or tv_e_lo <= 0 or tv_e_hi <= 0:
                continue
            buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo
            buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi
            val = 9.80665 * (buoy_lo + buoy_hi) / 2.0 * dz

            if lfc_idx is not None and i <= el_idx:
                if val > 0 and i >= lfc_idx:
                    cape_val += val
                elif val < 0 and i <= lfc_idx:
                    cin_val += val

        # MetPy convention: if there is no LFC (no positive buoyancy / no
        # free convection), CIN is zero — there is no energy barrier when
        # there is nothing to convect into.
        if lfc_idx is None or cape_val <= 0:
            cin_val = 0.0
            cape_val = 0.0

        return cape_val * units("J/kg"), cin_val * units("J/kg")
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
    if metpy_profile_form:
        return cape_val * units("J/kg"), cin_val * units("J/kg")
    return (
        cape_val * units("J/kg"),
        cin_val * units("J/kg"),
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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    return _calc.showalter_index(p, t, td) * units.delta_degC


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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    return _calc.downdraft_cape(p, t, td) * units("J/kg")


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


def sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500):
    """SWEAT Index.

    Parameters
    ----------
    t850, td850, t500 : Quantity (temperature) in Celsius
    dd850, dd500 : Quantity (degrees) -- wind direction
    ff850, ff500 : Quantity (speed) in knots

    Returns
    -------
    Quantity (dimensionless)
    """
    return _calc.sweat_index(
        _as_float(_strip(t850, "degC")),
        _as_float(_strip(td850, "degC")),
        _as_float(_strip(t500, "degC")),
        _as_float(_strip(dd850, "degree")),
        _as_float(_strip(dd500, "degree")),
        _as_float(_strip(ff850, "knot")),
        _as_float(_strip(ff500, "knot")),
    ) * units.dimensionless


def brunt_vaisala_frequency(height, potential_temp):
    """Brunt-Vaisala frequency at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (1/s)
    """
    z = _as_1d(_strip(height, "m"))
    theta = _as_1d(_strip(potential_temp, "K"))
    result = np.array(_calc.brunt_vaisala_frequency(z, theta))
    return result * units("1/s")


def brunt_vaisala_period(height, potential_temp):
    """Brunt-Vaisala period at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (s)
    """
    z = _as_1d(_strip(height, "m"))
    theta = _as_1d(_strip(potential_temp, "K"))
    result = np.array(_calc.brunt_vaisala_period(z, theta))
    return result * units.s


def brunt_vaisala_frequency_squared(height, potential_temp):
    """Brunt-Vaisala frequency squared (N^2) at each level.

    Parameters
    ----------
    height : array Quantity (m)
    potential_temp : array Quantity (K)

    Returns
    -------
    array Quantity (1/s^2)
    """
    z = _as_1d(_strip(height, "m"))
    theta = _as_1d(_strip(potential_temp, "K"))
    result = np.array(_calc.brunt_vaisala_frequency_squared(z, theta))
    return result * units("1/s**2")


def precipitable_water(pressure, dewpoint):
    """Precipitable water from pressure and dewpoint profiles.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    Quantity (mm)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    td = _as_1d(_strip(dewpoint, "degC"))
    return _calc.precipitable_water(p, td) * units.mm


def parcel_profile_with_lcl(pressure, t_surface, td_surface):
    """Parcel temperature profile with the LCL level inserted.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    t_surface : Quantity (temperature)
    td_surface : Quantity (temperature)

    Returns
    -------
    tuple of (array Quantity (hPa), array Quantity (degC))
        Pressure levels (with LCL inserted) and parcel temperatures.
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(t_surface, "degC"))
    td = _as_float(_strip(td_surface, "degC"))
    p_out, t_out = _calc.parcel_profile_with_lcl(p, t, td)
    return np.array(p_out) * units.hPa, np.array(t_out) * units.degC


def moist_air_gas_constant(mixing_ratio_kgkg):
    """Gas constant for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : Quantity (kg/kg) or float
        Mixing ratio in kg/kg.

    Returns
    -------
    Quantity (J/(kg*K))
    """
    result = _vec_call(_calc.moist_air_gas_constant, _strip(mixing_ratio_kgkg, "kg/kg") if hasattr(mixing_ratio_kgkg, "magnitude") else mixing_ratio_kgkg)
    return result * units("J/(kg*K)")


def moist_air_specific_heat_pressure(mixing_ratio_kgkg):
    """Specific heat at constant pressure for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : Quantity (kg/kg) or float

    Returns
    -------
    Quantity (J/(kg*K))
    """
    result = _vec_call(_calc.moist_air_specific_heat_pressure, _strip(mixing_ratio_kgkg, "kg/kg") if hasattr(mixing_ratio_kgkg, "magnitude") else mixing_ratio_kgkg)
    return result * units("J/(kg*K)")


def moist_air_poisson_exponent(mixing_ratio_kgkg):
    """Poisson exponent (kappa) for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : Quantity (kg/kg) or float

    Returns
    -------
    Quantity (dimensionless)
    """
    result = _vec_call(_calc.moist_air_poisson_exponent, _strip(mixing_ratio_kgkg, "kg/kg") if hasattr(mixing_ratio_kgkg, "magnitude") else mixing_ratio_kgkg)
    return result * units.dimensionless


def water_latent_heat_vaporization(temperature):
    """Latent heat of vaporization (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    result = _vec_call(_calc.water_latent_heat_vaporization, _strip(temperature, "degC"))
    return result * units("J/kg")


def water_latent_heat_melting(temperature):
    """Latent heat of melting (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    result = _vec_call(_calc.water_latent_heat_melting, _strip(temperature, "degC"))
    return result * units("J/kg")


def water_latent_heat_sublimation(temperature):
    """Latent heat of sublimation (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    result = _vec_call(_calc.water_latent_heat_sublimation, _strip(temperature, "degC"))
    return result * units("J/kg")


def relative_humidity_wet_psychrometric(temperature, wet_bulb, pressure):
    """Relative humidity from dry-bulb, wet-bulb, and pressure.

    Parameters
    ----------
    temperature : Quantity (temperature)
    wet_bulb : Quantity (temperature)
    pressure : Quantity (pressure)

    Returns
    -------
    Quantity (percent)
    """
    result = _vec_call(_calc.relative_humidity_wet_psychrometric, _strip(temperature, "degC"), _strip(wet_bulb, "degC"), _strip(pressure, "hPa"))
    return result * units.percent


def weighted_continuous_average(values, weights):
    """Trapezoidal weighted average over a coordinate.

    Parameters
    ----------
    values : array-like
    weights : array-like

    Returns
    -------
    float
    """
    v = _as_1d(_strip(values, "")) if hasattr(values, "magnitude") else _as_1d(values)
    w = _as_1d(_strip(weights, "")) if hasattr(weights, "magnitude") else _as_1d(weights)
    return _calc.weighted_continuous_average(v, w)


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


def add_height_to_pressure(pressure, delta_height):
    """New pressure after ascending/descending by a height increment.

    Parameters
    ----------
    pressure : Quantity (pressure)
    delta_height : Quantity (length)

    Returns
    -------
    Quantity (hPa)
    """
    result = _vec_call(_calc.add_height_to_pressure, _strip(pressure, "hPa"), _strip(delta_height, "m"))
    return result * units.hPa


def add_pressure_to_height(height, delta_pressure):
    """New height after a pressure increment.

    Parameters
    ----------
    height : Quantity (length)
    delta_pressure : Quantity (pressure)

    Returns
    -------
    Quantity (m)
    """
    result = _vec_call(_calc.add_pressure_to_height, _strip(height, "m"), _strip(delta_pressure, "hPa"))
    return result * units.m


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
                                                 relative_humidity):
    """Hypsometric thickness from pressure, temperature, and relative humidity profiles.

    Computes layer thickness using virtual temperature derived from RH.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature, Celsius)
    relative_humidity : array Quantity (dimensionless 0-1, or percent 0-100)

    Returns
    -------
    Quantity (m)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    # Handle RH: Rust expects percent (0-100).
    # If pint Quantity, try converting to percent first.
    if hasattr(relative_humidity, "magnitude"):
        try:
            rh = _as_1d(relative_humidity.to("percent").magnitude)
        except Exception:
            # dimensionless ratio 0-1 -> convert to percent
            rh = _as_1d(relative_humidity.magnitude) * 100.0
    else:
        rh_arr = np.asarray(relative_humidity, dtype=np.float64)
        # Heuristic: if max <= 1.0, treat as ratio
        if rh_arr.max() <= 1.0:
            rh = _as_1d(rh_arr * 100.0)
        else:
            rh = _as_1d(rh_arr)
    return _calc.thickness_hydrostatic_from_relative_humidity(p, t, rh) * units.m


def ccl(pressure, temperature, dewpoint):
    """Convective Condensation Level.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC)) or None
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    result = _calc.ccl(p, t, td)
    if result is None:
        return None
    return result[0] * units.hPa, result[1] * units.degC


def lifted_index(pressure, temperature, dewpoint):
    """Lifted Index.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)

    Returns
    -------
    Quantity (delta_degK)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    return _calc.lifted_index(p, t, td) * units.delta_degC


def density(pressure, temperature, mixing_ratio):
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
    p_raw = _strip(pressure, "hPa")
    t_raw = _strip(temperature, "degC")
    w_raw = _strip(mixing_ratio, "g/kg") if _can_convert(mixing_ratio, "g/kg") else np.asarray(_strip(mixing_ratio, "kg/kg"), dtype=np.float64) * 1000.0
    vals, shape, is_arr = _prep(p_raw, t_raw, w_raw)
    if is_arr:
        result = np.asarray(_calc.density_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.density(vals[0], vals[1], vals[2])
    return result * units("kg/m**3")


def dewpoint(vapor_pressure_val):
    """Dewpoint from vapor pressure.

    Parameters
    ----------
    vapor_pressure_val : Quantity (pressure, hPa)

    Returns
    -------
    Quantity (degC)
    """
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
    result = _vec_call(_calc.dry_static_energy, _strip(height, "m"), _strip(temperature, "K"))
    return result * units("J/kg")


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
    result = _vec_call(_calc.geopotential_to_height, _strip(geopotential, "m**2/s**2") if hasattr(geopotential, "magnitude") else geopotential)
    return result * units.m


def get_layer(pressure, *args, p_bottom=None, p_top=None,
              bottom=None, depth=None, interpolate=True):
    """Extract one or more fields from a sounding layer.

    Supports both the original form ``get_layer(pressure, values, p_bottom, p_top)``
    and the MetPy-style form ``get_layer(pressure, values..., bottom=..., depth=...)``.
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
        p_bottom = bottom if bottom is not None else pressure[0]
    if p_top is None:
        if depth is None:
            raise TypeError("get_layer requires either p_top or depth")
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
        if p_result is None:
            p_result = np.asarray(p_out) * units.hPa
        v_result = np.asarray(v_out)
        if v_unit is not None:
            v_result = v_result * v_unit
        value_results.append(v_result)

    if len(value_results) == 1:
        return p_result, value_results[0]
    return (p_result, *value_results)


def get_layer_heights(pressure, heights, p_bottom, p_top):
    """Extract layer heights between two pressures.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    heights : array Quantity (m)
    p_bottom : Quantity (pressure)
    p_top : Quantity (pressure)

    Returns
    -------
    tuple of (array Quantity (hPa), array Quantity (m))
    """
    p = _as_1d(_strip(pressure, "hPa"))
    h = _as_1d(_strip(heights, "m"))
    pb = _as_float(_strip(p_bottom, "hPa"))
    pt = _as_float(_strip(p_top, "hPa"))
    p_out, h_out = _calc.get_layer_heights(p, h, pb, pt)
    return np.asarray(p_out) * units.hPa, np.asarray(h_out) * units.m


def height_to_geopotential(height):
    """Convert height to geopotential.

    Parameters
    ----------
    height : Quantity (m)

    Returns
    -------
    Quantity (m^2/s^2)
    """
    result = _vec_call(_calc.height_to_geopotential, _strip(height, "m"))
    return result * units("m**2/s**2")


def isentropic_interpolation(theta_levels, pressure_3d, temperature_3d,
                              fields, nx=None, ny=None, nz=None):
    """Interpolate fields to isentropic surfaces.

    Parameters
    ----------
    theta_levels : array (K)
    pressure_3d : 3-D array (hPa), shape (nz, ny, nx)
    temperature_3d : 3-D array (K), shape (nz, ny, nx)
    fields : list of 3-D arrays, each shape (nz, ny, nx)
    nx, ny, nz : int, optional
        Grid dimensions; inferred from pressure_3d if not given.

    Returns
    -------
    list of 1-D arrays
    """
    theta = _as_1d(_strip(theta_levels, "K")) if hasattr(theta_levels, "magnitude") else _as_1d(theta_levels)
    p_arr = np.asarray(pressure_3d.magnitude if hasattr(pressure_3d, "magnitude") else pressure_3d, dtype=np.float64)
    t_arr = np.asarray(temperature_3d.magnitude if hasattr(temperature_3d, "magnitude") else temperature_3d, dtype=np.float64)
    if p_arr.ndim == 3:
        _nz, _ny, _nx = p_arr.shape
        if nx is None: nx = _nx
        if ny is None: ny = _ny
        if nz is None: nz = _nz
    p_flat = np.ascontiguousarray(p_arr.ravel())
    t_flat = np.ascontiguousarray(t_arr.ravel())
    field_list = []
    for f in fields:
        fa = np.asarray(f.magnitude if hasattr(f, "magnitude") else f, dtype=np.float64)
        field_list.append(np.ascontiguousarray(fa.ravel()))
    result = _calc.isentropic_interpolation(theta, p_flat, t_flat, field_list, nx, ny, nz)
    return [np.asarray(r) for r in result]


def mean_pressure_weighted(pressure, values):
    """Pressure-weighted mean of a quantity.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    values : array Quantity or array-like

    Returns
    -------
    float
    """
    p = _as_1d(_strip(pressure, "hPa"))
    v = _as_1d(_strip(values, "")) if hasattr(values, "magnitude") else _as_1d(values)
    return _calc.mean_pressure_weighted(p, v)


def mixed_layer(pressure, *args, height=None, bottom=None, depth=100.0, interpolate=True):
    """Mixed-layer mean of one or more profiles.

    Supports both the original form ``mixed_layer(pressure, values, depth=...)``
    and the MetPy-style form ``mixed_layer(pressure, values..., bottom=..., depth=...)``.
    """
    if not args:
        raise TypeError("mixed_layer requires at least one value profile")

    p_layer = pressure
    value_layers = args
    if bottom is not None and abs(_as_float(_strip(bottom, "hPa")) - _as_float(_strip(pressure[0], "hPa"))) > 1e-6:
        extracted = get_layer(pressure, *args, bottom=bottom, depth=depth, interpolate=interpolate)
        p_layer, *value_layers = extracted

    p = _as_1d(_strip(p_layer, "hPa"))
    d = _as_float(_strip(depth, "hPa")) if hasattr(depth, "magnitude") else float(depth)
    results = []
    for values in value_layers:
        has_units = hasattr(values, "units")
        if has_units:
            unit_obj = values.units
            v = _as_1d(np.asarray(values.magnitude, dtype=np.float64))
        elif hasattr(values, "values"):
            unit_obj = None
            v = _as_1d(np.asarray(values.values, dtype=np.float64))
        else:
            unit_obj = None
            v = _as_1d(values)
        mixed = _calc.mixed_layer(p, v, d)
        results.append(mixed * unit_obj if unit_obj is not None else mixed)

    if len(results) == 1:
        return results[0]
    return tuple(results)


def mixed_layer_cape_cin(pressure, temperature, dewpoint, depth=100.0):
    """Mixed-layer CAPE and CIN.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    depth : float
        Mixed-layer depth in hPa (default 100).

    Returns
    -------
    tuple of (Quantity (J/kg), Quantity (J/kg))
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    d = _as_float(_strip(depth, "hPa")) if hasattr(depth, "magnitude") else float(depth)
    cape_val, cin_val = _calc.mixed_layer_cape_cin(p, t, td, d)
    return cape_val * units("J/kg"), cin_val * units("J/kg")


def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity):
    """Mixing ratio from pressure, temperature, and relative humidity.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    relative_humidity : Quantity (percent or dimensionless)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
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


def moist_lapse(pressure, t_start, reference_pressure=None):
    """Moist adiabatic lapse rate temperature profile.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    t_start : Quantity (temperature)

    Returns
    -------
    array Quantity (degC)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(t_start, "degC"))
    if reference_pressure is None:
        result = np.asarray(_calc.moist_lapse(p, t))
    else:
        ref_p = _as_float(_strip(reference_pressure, "hPa"))
        if np.isclose(ref_p, p[0]):
            p_run = np.ascontiguousarray(p)
            reverse = False
        elif np.isclose(ref_p, p[-1]):
            p_run = np.ascontiguousarray(p[::-1])
            reverse = True
        else:
            raise NotImplementedError(
                "moist_lapse currently supports reference_pressure only when it matches "
                "the first or last pressure level"
            )
        result = np.asarray(_calc.moist_lapse(p_run, t))
        if reverse:
            result = result[::-1]
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
    result = _vec_call(_calc.moist_static_energy, _strip(height, "m"), _strip(temperature, "K"), _strip(specific_humidity, "kg/kg") if hasattr(specific_humidity, "magnitude") else specific_humidity)
    return result * units("J/kg")


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
        # 2-arg MetPy form: (height, temperature)
        h_val = _strip(height_or_theta, "m")
        t_val = _strip(temperature_or_pressure, "K")
        # M = c_p * T + g * z  (MetPy-exact constants)
        cp = 1004.6662184201462  # J/(kg*K), matches MetPy Cp_d
        g = 9.80665  # m/s^2
        h_arr = np.asarray(h_val, dtype=np.float64)
        t_arr = np.asarray(t_val, dtype=np.float64)
        result = (cp * t_arr + g * h_arr) / 1000.0  # Convert to kJ/kg (MetPy convention)
        if hasattr(height_or_theta, "coords") and hasattr(height_or_theta, "dims"):
            import xarray as xr
            return xr.DataArray(result, coords=height_or_theta.coords,
                                dims=height_or_theta.dims,
                                attrs={"units": "kJ/kg"})
        return result * units("kJ/kg")
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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    d = _scalar_strip(depth, "hPa") if hasattr(depth, "magnitude") else float(depth)
    # Find MU parcel within the specified depth, then compute CAPE from it
    mu_p, mu_t, mu_td = _calc.get_most_unstable_parcel(p, t, td, d)
    # Find index of MU parcel and compute CAPE from that level up
    mu_idx = int(np.argmin(np.abs(p - mu_p)))
    cape_val, cin_val = _calc.surface_based_cape_cin(p[mu_idx:], t[mu_idx:], td[mu_idx:])
    return cape_val * units("J/kg"), cin_val * units("J/kg")


def parcel_profile(pressure, temperature, dewpoint):
    """Parcel temperature profile.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    array Quantity (degC)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    result = np.asarray(_calc.parcel_profile(p, t, td))
    return result * units.degC


def reduce_point_density(lats, lons, radius):
    """Reduce point density by removing points too close together.

    Parameters
    ----------
    lats : array (degrees)
    lons : array (degrees)
    radius : float (degrees)

    Returns
    -------
    list of bool
    """
    lat_arr = _as_1d(_strip(lats, "degree")) if hasattr(lats, "magnitude") else _as_1d(lats)
    lon_arr = _as_1d(_strip(lons, "degree")) if hasattr(lons, "magnitude") else _as_1d(lons)
    r = _as_float(_strip(radius, "degree")) if hasattr(radius, "magnitude") else float(radius)
    return _calc.reduce_point_density(lat_arr, lon_arr, r)


def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio_val):
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
    p_raw = _strip(pressure, "hPa")
    t_raw = _strip(temperature, "degC")
    w_raw = _strip(mixing_ratio_val, "g/kg") if _can_convert(mixing_ratio_val, "g/kg") else np.asarray(_strip(mixing_ratio_val, "kg/kg"), dtype=np.float64) * 1000.0
    vals, shape, is_arr = _prep(p_raw, t_raw, w_raw)
    if is_arr:
        result = np.asarray(_calc.relative_humidity_from_mixing_ratio_array(vals[0], vals[1], vals[2])).reshape(shape) / 100.0
    else:
        result = _calc.relative_humidity_from_mixing_ratio(vals[0], vals[1], vals[2]) / 100.0
    return _attach(result, "")


def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
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


def scale_height(temperature):
    """Atmospheric scale height.

    Parameters
    ----------
    temperature : Quantity (K)

    Returns
    -------
    Quantity (m)
    """
    result = _vec_call(_calc.scale_height, _strip(temperature, "K"))
    return result * units.m


def specific_humidity_from_dewpoint(pressure, dewpoint_val):
    """Specific humidity from pressure and dewpoint.

    Parameters
    ----------
    pressure : Quantity (pressure)
    dewpoint_val : Quantity (temperature)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(dewpoint_val, "degC"))
    if is_arr:
        result = np.asarray(_calc.specific_humidity_from_dewpoint_array(vals[0], vals[1])).reshape(shape)
    else:
        result = _calc.specific_humidity_from_dewpoint(vals[0], vals[1])
    return result * units("kg/kg")


def static_stability(pressure, temperature):
    """Static stability.

    Parameters
    ----------
    pressure : array Quantity (pressure, hPa)
    temperature : array Quantity (K)

    Returns
    -------
    array Quantity (K/Pa)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t_k = _as_1d(_strip(temperature, "K"))
    result = np.asarray(_calc.static_stability(p, t_k))
    return result * units("K/Pa")


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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    cape_val, cin_val = _calc.surface_based_cape_cin(p, t, td)
    return cape_val * units("J/kg"), cin_val * units("J/kg")


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
    """Wet-bulb potential temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (K)
    """
    vals, shape, is_arr = _prep(_strip(pressure, "hPa"), _strip(temperature, "degC"), _strip(dewpoint, "degC"))
    if is_arr:
        result = np.asarray(_calc.wet_bulb_potential_temperature_array(vals[0], vals[1], vals[2])).reshape(shape)
    else:
        result = _calc.wet_bulb_potential_temperature(vals[0], vals[1], vals[2])
    return result * units.K


def get_mixed_layer_parcel(pressure, temperature, dewpoint, depth=100.0):
    """Get mixed-layer parcel properties.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    depth : float
        Mixed-layer depth in hPa (default 100).

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC), Quantity (degC))
        Parcel pressure, temperature, and dewpoint.
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    d = _as_float(_strip(depth, "hPa")) if hasattr(depth, "magnitude") else float(depth)
    pp, tp, tdp = _calc.get_mixed_layer_parcel(p, t, td, d)
    return pp * units.hPa, tp * units.degC, tdp * units.degC


def get_most_unstable_parcel(pressure, temperature, dewpoint,
                             height=None, bottom=None, depth=300.0):
    """Get most-unstable parcel properties.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    height : ignored (MetPy compat)
    bottom : ignored (MetPy compat)
    depth : Quantity or float
        Search depth in hPa (default 300).

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC), Quantity (degC), int)
        Parcel pressure, temperature, dewpoint, and source index.
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    d = _scalar_strip(depth, "hPa") if hasattr(depth, "magnitude") else float(depth)
    pp, tp, tdp = _calc.get_most_unstable_parcel(p, t, td, d)
    # Find the source index (level closest to the returned parcel pressure)
    idx = int(np.argmin(np.abs(p - pp)))
    return pp * units.hPa, tp * units.degC, tdp * units.degC, idx


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
    """Mixed-layer parcel (MetPy-compatible).

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    parcel_start_pressure : ignored (MetPy compat)
    height : ignored (MetPy compat)
    bottom : ignored (MetPy compat)
    depth : Quantity or float
        Mixing depth in hPa (default 100).
    interpolate : ignored (MetPy compat)

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC), Quantity (degC))
    """
    d = _scalar_strip(depth, "hPa") if hasattr(depth, "magnitude") else float(depth)
    return get_mixed_layer_parcel(pressure, temperature, dewpoint, d)


def most_unstable_parcel(pressure, temperature, dewpoint, height=None,
                         bottom=None, depth=300):
    """Alias for :func:`get_most_unstable_parcel` (MetPy-compatible)."""
    return get_most_unstable_parcel(pressure, temperature, dewpoint,
                                    height=height, bottom=bottom, depth=depth)


def psychrometric_vapor_pressure_wet(temperature, wet_bulb, pressure):
    """Alias for :func:`psychrometric_vapor_pressure`."""
    return psychrometric_vapor_pressure(temperature, wet_bulb, pressure)


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


def wind_direction(u, v):
    """Meteorological wind direction from (u, v).

    Parameters
    ----------
    u, v : array Quantity (m/s)

    Returns
    -------
    array Quantity (degree)
    """
    orig_shape = np.asarray(_strip(u, "m/s")).shape
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.wind_direction(u_arr, v_arr))
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
        (ru, rv), _, _ = bunkers_storm_motion(u, v, height_a)
        if storm_u is None:
            storm_u = ru
        if storm_v is None:
            storm_v = rv
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
    return (
        (_attach(ru, "m/s"), _attach(rv, "m/s")),
        (_attach(lu, "m/s"), _attach(lv, "m/s")),
        (_attach(mu, "m/s"), _attach(mv, "m/s")),
    )


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
    d = np.asarray(delta, dtype=np.float64).copy()
    # Replace near-zero spacings with NaN to avoid division by zero (e.g., at poles)
    d[np.abs(d) < 1.0] = np.nan
    n = arr.shape[axis]

    # Expand delta to match field dimensions if needed
    if d.ndim == 1 and d.size == n - 1:
        # Standard case: spacing between adjacent levels
        pass
    elif d.ndim == 2 and d.shape[axis] == n - 1:
        pass
    elif d.ndim == 2 and d.shape[axis] == n:
        # Average adjacent to get n-1 spacings
        d = (np.take(d, range(d.shape[axis] - 1), axis=axis)
             + np.take(d, range(1, d.shape[axis]), axis=axis)) / 2.0
    elif d.size == 1:
        return np.gradient(arr, float(d.ravel()[0]), axis=axis)
    else:
        return np.gradient(arr, float(np.mean(d)), axis=axis)

    result = np.empty_like(arr)
    # Interior: centered differences
    slc_c = [slice(None)] * arr.ndim
    slc_p = [slice(None)] * arr.ndim
    slc_m = [slice(None)] * arr.ndim
    slc_c[axis] = slice(1, -1)
    slc_p[axis] = slice(2, None)
    slc_m[axis] = slice(None, -2)

    # d_fwd[i] = spacing from i to i+1, d_bwd[i] = spacing from i-1 to i
    slc_df = [slice(None)] * d.ndim
    slc_db = [slice(None)] * d.ndim
    slc_df[axis] = slice(1, None)    # d[1:]
    slc_db[axis] = slice(None, -1)   # d[:-1]

    d_fwd = d[tuple(slc_df)]
    d_bwd = d[tuple(slc_db)]

    # Broadcast delta to match field shape
    if d_fwd.ndim < arr.ndim:
        shape = [1] * arr.ndim
        shape[axis] = d_fwd.shape[0] if d_fwd.ndim > 0 else 1
        d_fwd = d_fwd.reshape(shape)
        d_bwd = d_bwd.reshape(shape)

    result[tuple(slc_c)] = (arr[tuple(slc_p)] - arr[tuple(slc_m)]) / (d_fwd + d_bwd)

    # Boundaries: forward/backward differences
    slc_0 = [slice(None)] * arr.ndim
    slc_1 = [slice(None)] * arr.ndim
    slc_0[axis] = 0
    slc_1[axis] = 1
    d0 = np.take(d, 0, axis=axis)
    if d0.ndim < arr[tuple(slc_0)].ndim:
        d0 = np.expand_dims(d0, axis=axis) if d0.ndim > 0 else d0
    result[tuple(slc_0)] = (arr[tuple(slc_1)] - arr[tuple(slc_0)]) / d0

    slc_n1 = [slice(None)] * arr.ndim
    slc_n2 = [slice(None)] * arr.ndim
    slc_n1[axis] = -1
    slc_n2[axis] = -2
    dn = np.take(d, -1, axis=axis)
    if dn.ndim < arr[tuple(slc_n1)].ndim:
        dn = np.expand_dims(dn, axis=axis) if dn.ndim > 0 else dn
    result[tuple(slc_n1)] = (arr[tuple(slc_n1)] - arr[tuple(slc_n2)]) / dn

    return result


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
                       longitude=None, x_dim=-1, y_dim=-2, crs=None):
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
    dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=lat_source, longitude=longitude)
    if lat_source is None:
        lat_source, _ = _infer_lat_lon(u, latitude=latitude, longitude=longitude)
    if dx is None or dy is None or lat_source is None:
        raise TypeError("absolute_vorticity requires latitude plus dx/dy or inferable coordinates")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lat_source, "degree") if hasattr(lat_source, "magnitude") else _flat(lat_source)
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.absolute_vorticity(u_f, v_f, lats_f, dx_val, dy_val))
    return result * units("1/s")


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


def frontogenesis(theta, u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                  parallel_scale=None, meridional_scale=None,
                  latitude=None, longitude=None, crs=None):
    """2-D Petterssen frontogenesis function.

    Parameters
    ----------
    theta : 2-D array Quantity (K)
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (K/m/s)
    """
    dx, dy = _resolve_dx_dy(theta, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if (dx is None or dy is None) and u is not None:
        dx, dy = _resolve_dx_dy(u, dx=dx, dy=dy, latitude=latitude, longitude=longitude)
    if dx is None or dy is None:
        raise TypeError("frontogenesis requires dx/dy or inferable latitude/longitude coordinates")
    t_arr = np.asarray(_strip(theta, "K"), dtype=np.float64)
    u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
    v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
    dx_m = np.asarray(dx.to("m").magnitude if hasattr(dx, "to") else dx, dtype=np.float64)
    dy_m = np.asarray(dy.to("m").magnitude if hasattr(dy, "to") else dy, dtype=np.float64)

    if _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
        # Variable-spacing: compute frontogenesis with full 2D dx/dy
        dtdx = _first_derivative_variable(t_arr, dx_m, axis=-1)
        dtdy = _first_derivative_variable(t_arr, dy_m, axis=-2)
        dudx = _first_derivative_variable(u_arr, dx_m, axis=-1)
        dvdy = _first_derivative_variable(v_arr, dy_m, axis=-2)
        dudy = _first_derivative_variable(u_arr, dy_m, axis=-2)
        dvdx = _first_derivative_variable(v_arr, dx_m, axis=-1)
        mag_t = np.sqrt(dtdx**2 + dtdy**2)
        mag_t = np.where(mag_t < 1e-30, np.nan, mag_t)
        result = -0.5 * mag_t * (
            (dtdx**2 * dudx + dtdy**2 * dvdy + (dtdx * dtdy) * (dvdx + dudy)) / (mag_t**2)
        )
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
        lat_2d = lat_arr if lat_arr.ndim == 2 else np.broadcast_to(lat_arr[:, None], h_arr.shape)
        f = 2.0 * omega * np.sin(np.deg2rad(lat_2d))
        f = np.where(np.abs(f) < 1e-10, np.nan, f)
        dzdx = _first_derivative_variable(h_arr, dx_m, axis=-1)
        dzdy = _first_derivative_variable(h_arr, dy_m, axis=-2)
        u_g = -(g / f) * dzdy
        v_g = (g / f) * dzdx
        return _wrap_result_like(heights, u_g, "m/s"), _wrap_result_like(heights, v_g, "m/s")

    h_f = _as_2d(heights, "m")
    lats_f = _as_2d(latitude, "degree") if hasattr(latitude, "magnitude") else _as_2d(latitude)
    dx_val = float(dx_m.mean()) if dx_m.ndim > 0 else float(dx_m)
    dy_val = float(dy_m.mean()) if dy_m.ndim > 0 else float(dy_m)
    u_g, v_g = _calc.geostrophic_wind(h_f, lats_f, dx_val, dy_val)
    ms = units("m/s")
    return np.asarray(u_g) * ms, np.asarray(v_g) * ms


def ageostrophic_wind(u, v, heights, lats, dx, dy):
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
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    h_f = _flat(heights, "m")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    # Compute geostrophic wind first, then ageostrophic = total - geostrophic
    u_g, v_g = _calc.geostrophic_wind(h_f, lats_f, dx_val, dy_val)
    u_g_flat = np.ascontiguousarray(np.asarray(u_g).ravel(), dtype=np.float64)
    v_g_flat = np.ascontiguousarray(np.asarray(v_g).ravel(), dtype=np.float64)
    u_flat = np.ascontiguousarray(u_f.ravel(), dtype=np.float64)
    v_flat = np.ascontiguousarray(v_f.ravel(), dtype=np.float64)
    ua, va = _calc.ageostrophic_wind(u_flat, v_flat, u_g_flat, v_g_flat)
    ms = units("m/s")
    orig_shape = np.asarray(_strip(u, "m/s")).shape
    return np.asarray(ua).reshape(orig_shape) * ms, np.asarray(va).reshape(orig_shape) * ms


def potential_vorticity_baroclinic(potential_temp, pressure, *args, dx=None, dy=None,
                                   latitude=None, x_dim=-1, y_dim=-2, vertical_dim=-3,
                                   longitude=None, crs=None):
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
        pt_arr = np.asarray(_strip(potential_temp, "K"), dtype=np.float64)
        u_arr = np.asarray(_strip(u, "m/s"), dtype=np.float64)
        v_arr = np.asarray(_strip(v, "m/s"), dtype=np.float64)
        lat_arr = np.asarray(_strip(latitude, "degree"), dtype=np.float64) if hasattr(latitude, "magnitude") else np.asarray(latitude, dtype=np.float64)
        p_arr = np.asarray(_strip(pressure, "Pa"), dtype=np.float64) if hasattr(pressure, "magnitude") else np.asarray(pressure, dtype=np.float64)
        pt_arr = np.moveaxis(pt_arr, vertical_dim, 0)
        u_arr = np.moveaxis(u_arr, vertical_dim, 0)
        v_arr = np.moveaxis(v_arr, vertical_dim, 0)
        if lat_arr.ndim == pt_arr.ndim:
            lat_arr = np.moveaxis(lat_arr, vertical_dim, 0)
        if p_arr.ndim == pt_arr.ndim:
            p_levels = np.moveaxis(p_arr, vertical_dim, 0)[..., 0, 0]
        else:
            p_levels = p_arr.reshape(-1)
        dx_val = _mean_spacing(dx, "m")
        dy_val = _mean_spacing(dy, "m")
        result = np.full(pt_arr.shape, np.nan, dtype=np.float64)
        for idx in range(1, pt_arr.shape[0] - 1):
            lat_slice = lat_arr[idx] if lat_arr.ndim == pt_arr.ndim else lat_arr
            result[idx] = np.asarray(_calc.potential_vorticity_baroclinic(
                np.ascontiguousarray(pt_arr[idx]),
                np.asarray([p_levels[idx - 1], p_levels[idx + 1]], dtype=np.float64),
                np.ascontiguousarray(pt_arr[idx - 1]),
                np.ascontiguousarray(pt_arr[idx + 1]),
                np.ascontiguousarray(u_arr[idx]),
                np.ascontiguousarray(v_arr[idx]),
                np.ascontiguousarray(lat_slice),
                dx_val,
                dy_val,
            )).reshape(pt_arr.shape[-2:])
        result = np.moveaxis(result, 0, vertical_dim)
        template = potential_temp if hasattr(potential_temp, "coords") and hasattr(potential_temp, "dims") else u
        return _wrap_result_like(template, result, "K*m**2/(kg*s)")
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


def potential_vorticity_barotropic(heights, u, v, lats, dx, dy):
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
    h_f = _flat(heights, "m")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.potential_vorticity_barotropic(
        h_f, u_f, v_f, lats_f, dx_val, dy_val,
    ))
    return result * units("1/(m*s)")


def normal_component(u, v, start, end):
    """Normal (perpendicular) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    array Quantity (m/s)
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.array(_calc.normal_component(u_arr, v_arr, start, end))
    return result * units("m/s")


def tangential_component(u, v, start, end):
    """Tangential (parallel) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    array Quantity (m/s)
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.array(_calc.tangential_component(u_arr, v_arr, start, end))
    return result * units("m/s")


def unit_vectors_from_cross_section(start, end):
    """Tangent and normal unit vectors for a cross-section line.

    Parameters
    ----------
    start, end : tuple of (lat, lon) in degrees

    Returns
    -------
    tuple of ((east, north), (east, north))
        Tangent and normal unit vector components.
    """
    return _calc.unit_vectors_from_cross_section(start, end)


def vector_derivative(u, v, dx, dy):
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
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    dudx, dudy, dvdx, dvdy = _calc.vector_derivative(u_f, v_f, dx_val, dy_val)
    inv_s = units("1/s")
    return (
        np.asarray(dudx) * inv_s,
        np.asarray(dudy) * inv_s,
        np.asarray(dvdx) * inv_s,
        np.asarray(dvdy) * inv_s,
    )


def absolute_momentum(u, lats, y_distances):
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
    result = _vec_call(_calc.coriolis_parameter, lat_stripped)
    return _attach(result, "1/s")


def cross_section_components(u, v, start_lat, start_lon, end_lat, end_lon):
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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    slat = _as_float(_strip(start_lat, "degree")) if hasattr(start_lat, "magnitude") else float(start_lat)
    slon = _as_float(_strip(start_lon, "degree")) if hasattr(start_lon, "magnitude") else float(start_lon)
    elat = _as_float(_strip(end_lat, "degree")) if hasattr(end_lat, "magnitude") else float(end_lat)
    elon = _as_float(_strip(end_lon, "degree")) if hasattr(end_lon, "magnitude") else float(end_lon)
    par, perp = _calc.cross_section_components(u_arr, v_arr, slat, slon, elat, elon)
    ms = units("m/s")
    return np.asarray(par) * ms, np.asarray(perp) * ms


def curvature_vorticity(u, v, dx, dy):
    """Curvature vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.curvature_vorticity(u_2d, v_2d, dx_val, dy_val))
    return result * units("1/s")


def inertial_advective_wind(u, v, u_geo, v_geo, dx, dy):
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
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    ug_2d = _as_2d(u_geo, "m/s")
    vg_2d = _as_2d(v_geo, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    u_ia, v_ia = _calc.inertial_advective_wind(u_2d, v_2d, ug_2d, vg_2d, dx_val, dy_val)
    ms = units("m/s")
    return np.asarray(u_ia) * ms, np.asarray(v_ia) * ms


def kinematic_flux(v_component, scalar):
    """Kinematic flux (element-wise product).

    Parameters
    ----------
    v_component : array Quantity (m/s)
    scalar : array Quantity or array-like

    Returns
    -------
    array (product units)
    """
    v_arr = _as_1d(_strip(v_component, "m/s"))
    s_arr = _as_1d(_strip(scalar, "")) if hasattr(scalar, "magnitude") else _as_1d(scalar)
    result = np.asarray(_calc.kinematic_flux(v_arr, s_arr))
    return result


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
    t_2d = _as_2d(temperature, "K") if _can_convert(temperature, "K") else _as_2d(temperature, "degC")
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    p_val = _as_float(_strip(pressure, "hPa"))
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    q1, q2 = _calc.q_vector(t_2d, u_2d, v_2d, p_val, dx_val, dy_val)
    return _wrap_result_like(u, q1), _wrap_result_like(v, q2)


def shear_vorticity(u, v, dx, dy):
    """Shear vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.shear_vorticity(u_2d, v_2d, dx_val, dy_val))
    return result * units("1/s")


def shearing_deformation(u, v, dx, dy):
    """Shearing deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.shearing_deformation(u_2d, v_2d, dx_val, dy_val))
    return result * units("1/s")


def stretching_deformation(u, v, dx, dy):
    """Stretching deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.stretching_deformation(u_2d, v_2d, dx_val, dy_val))
    return result * units("1/s")


def total_deformation(u, v, dx, dy):
    """Total deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.total_deformation(u_2d, v_2d, dx_val, dy_val))
    return result * units("1/s")


def geospatial_gradient(data, lats, lons):
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
    has_units = hasattr(data, "units")
    d_2d = _as_2d(data) if not has_units else _as_2d_raw(data)
    lat_2d = _as_2d(lats, "degree") if hasattr(lats, "magnitude") else _as_2d_raw(lats)
    lon_2d = _as_2d(lons, "degree") if hasattr(lons, "magnitude") else _as_2d_raw(lons)
    dfdx, dfdy = _calc.geospatial_gradient(d_2d, lat_2d, lon_2d)
    dfdx = np.asarray(dfdx)
    dfdy = np.asarray(dfdy)
    if has_units:
        return dfdx * (data.units / units.m), dfdy * (data.units / units.m)
    return dfdx, dfdy


def geospatial_laplacian(data, lats, lons):
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
    has_units = hasattr(data, "units")
    d_2d = _as_2d(data) if not has_units else _as_2d_raw(data)
    lat_2d = _as_2d(lats, "degree") if hasattr(lats, "magnitude") else _as_2d_raw(lats)
    lon_2d = _as_2d(lons, "degree") if hasattr(lons, "magnitude") else _as_2d_raw(lons)
    result = np.asarray(_calc.geospatial_laplacian(d_2d, lat_2d, lon_2d))
    if has_units:
        return result * (data.units / units.m ** 2)
    return result


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
    return _calc.significant_tornado_parameter(cape, lcl, srh, shear) * units.dimensionless


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
    return _calc.supercell_composite_parameter(cape, srh, shear) * units.dimensionless


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
    rh = _as_float(_strip(relative_humidity, "percent")) if hasattr(relative_humidity, "magnitude") else float(relative_humidity)
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
    rh = _as_float(_strip(relative_humidity, "percent")) if hasattr(relative_humidity, "magnitude") else float(relative_humidity)
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


def galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst):
    """Galvez-Davison Index (tropical thunderstorm potential).

    Parameters
    ----------
    t950, t850, t700, t500 : Quantity (temperature)
    td950, td850, td700 : Quantity (temperature)
    sst : Quantity (temperature)

    Returns
    -------
    Quantity (dimensionless)
    """
    return _calc.galvez_davison_index(
        _as_float(_strip(t950, "degC")),
        _as_float(_strip(t850, "degC")),
        _as_float(_strip(t700, "degC")),
        _as_float(_strip(t500, "degC")),
        _as_float(_strip(td950, "degC")),
        _as_float(_strip(td850, "degC")),
        _as_float(_strip(td700, "degC")),
        _as_float(_strip(sst, "degC")),
    ) * units.dimensionless


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
    p = _as_float(_strip(pressure, "hPa"))
    return _calc.pressure_to_height_std(p) * units.m


def height_to_pressure_std(height):
    """Convert height to pressure using the US Standard Atmosphere 1976.

    Parameters
    ----------
    height : Quantity (m)

    Returns
    -------
    Quantity (hPa)
    """
    h = _as_float(_strip(height, "m"))
    return _calc.height_to_pressure_std(h) * units.hPa


def altimeter_to_station_pressure(altimeter, elevation):
    """Convert altimeter setting to station pressure.

    Parameters
    ----------
    altimeter : Quantity (pressure)
    elevation : Quantity (m)

    Returns
    -------
    Quantity (hPa)
    """
    a = _as_float(_strip(altimeter, "hPa"))
    e = _as_float(_strip(elevation, "m"))
    return _calc.altimeter_to_station_pressure(a, e) * units.hPa


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


def altimeter_to_sea_level_pressure(altimeter, elevation, temperature):
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
    a = _as_float(_strip(altimeter, "hPa"))
    e = _as_float(_strip(elevation, "m"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.altimeter_to_sea_level_pressure(a, e, t) * units.hPa


def sigma_to_pressure(sigma, psfc, ptop):
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
    ps = _as_float(_strip(psfc, "hPa"))
    pt = _as_float(_strip(ptop, "hPa"))
    return _calc.sigma_to_pressure(float(sigma), ps, pt) * units.hPa


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
    result = _vec_call(_calc.heat_index, _strip(temperature, "degC"), _strip(relative_humidity, "percent") if hasattr(relative_humidity, "magnitude") else relative_humidity)
    return result * units.degC


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
    result = _vec_call(_calc.apparent_temperature, _strip(temperature, "degC"), _strip(relative_humidity, "percent") if hasattr(relative_humidity, "magnitude") else relative_humidity, _strip(wind_speed_val, "m/s"))
    return result * units.degC


# ============================================================================
# Smooth / spatial derivatives
# ============================================================================

def smooth_gaussian(data, sigma):
    """2-D Gaussian smoothing.

    Parameters
    ----------
    data : 2-D array
    sigma : float (grid-point units)

    Returns
    -------
    2-D ndarray
    """
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_gaussian(arr, float(sigma))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_rectangular(data, size, passes=1):
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
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_rectangular(arr, int(size), int(passes))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_circular(data, radius, passes=1):
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
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_circular(arr, float(radius), int(passes))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_n_point(data, n, passes=1):
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
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_n_point(arr, int(n), int(passes))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_window(data, window, passes=1, normalize_weights=True):
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
    d_arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                       dtype=np.float64)
    w_arr = np.asarray(window.magnitude if hasattr(window, "magnitude") else window,
                       dtype=np.float64)
    result = _calc.smooth_window(d_arr, w_arr, int(passes), bool(normalize_weights))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def gradient(f, **kwargs):
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
    has_units = hasattr(f, "units")
    data = np.asarray(f.magnitude if has_units else f, dtype=np.float64)
    deltas = kwargs.get("deltas", None)

    if data.ndim == 2 and deltas is not None and len(deltas) >= 2:
        dy_val = _as_float(_strip(deltas[0], "m")) if hasattr(deltas[0], "magnitude") else float(deltas[0])
        dx_val = _as_float(_strip(deltas[1], "m")) if hasattr(deltas[1], "magnitude") else float(deltas[1])
        gx = np.asarray(_calc.gradient_x(data, dx_val))
        gy = np.asarray(_calc.gradient_y(data, dy_val))
        if has_units:
            return [gy * (f.units / units.m), gx * (f.units / units.m)]
        return [gy, gx]

    # General fallback: numpy.gradient
    axes = kwargs.get("axes", None)
    if deltas is not None:
        spacing = [float(d.magnitude) if hasattr(d, "magnitude") else float(d) for d in deltas]
        result = np.gradient(data, *spacing)
    elif axes is not None:
        result = np.gradient(data, axis=axes)
    else:
        result = np.gradient(data)

    # np.gradient returns ndarray for 1-D, list or tuple for N-D
    if isinstance(result, np.ndarray):
        result = [result]
    else:
        result = list(result)
    if has_units:
        return [r * f.units for r in result]
    return result


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


def laplacian(data, dx, dy):
    """Laplacian (d2f/dx2 + d2f/dy2).

    Parameters
    ----------
    data : 2-D array
    dx, dy : Quantity (m) or float

    Returns
    -------
    2-D array Quantity (data_units / m^2)
    """
    has_units = hasattr(data, "units")
    d_arr = np.asarray(data.magnitude if has_units else data, dtype=np.float64)
    dx_val = _mean_spacing(dx, "m")
    dy_val = _mean_spacing(dy, "m")
    result = np.asarray(_calc.laplacian(d_arr, dx_val, dy_val))
    if has_units:
        return result * (data.units / units.m ** 2)
    return result


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
        axis_spacing = x
    if axis_spacing is None:
        raise TypeError("first_derivative requires axis spacing via axis_spacing, x, or delta")
    ds = _mean_spacing(axis_spacing, "m")
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
    lon_arr = np.asarray(longitude.magnitude if hasattr(longitude, "magnitude") else longitude,
                         dtype=np.float64)
    lat_arr = np.asarray(latitude.magnitude if hasattr(latitude, "magnitude") else latitude,
                         dtype=np.float64)
    if np.nanmax(np.abs(lon_arr)) <= 90 and np.nanmax(np.abs(lat_arr)) > 90:
        lon_arr, lat_arr = lat_arr, lon_arr
    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr)
    dx_abs, dy_abs = _calc.lat_lon_grid_deltas(lat_arr, lon_arr)
    dx_out = np.asarray(dx_abs, dtype=np.float64)
    dy_out = np.asarray(dy_abs, dtype=np.float64)

    # Apply sign based on coordinate direction (MetPy convention):
    # dx > 0 when longitude increases, dy > 0 when latitude increases
    ny, nx = lat_arr.shape
    # dx sign: based on longitude difference (column-wise)
    if nx > 1:
        lon_sign = np.sign(lon_arr[:, 1:] - lon_arr[:, :-1])
        # Pad to match dx shape (which may be ny x nx or ny x nx-1)
        if dx_out.shape[-1] == nx - 1:
            dx_out = dx_out * lon_sign
        elif dx_out.shape[-1] == nx:
            lon_sign_full = np.ones((ny, nx))
            lon_sign_full[:, 1:] = lon_sign
            lon_sign_full[:, 0] = lon_sign[:, 0]
            dx_out = dx_out * lon_sign_full

    # dy sign: based on latitude difference (row-wise)
    if ny > 1:
        lat_sign = np.sign(lat_arr[1:, :] - lat_arr[:-1, :])
        if dy_out.shape[0] == ny - 1:
            dy_out = dy_out * lat_sign
        elif dy_out.shape[0] == ny:
            lat_sign_full = np.ones((ny, nx))
            lat_sign_full[1:, :] = lat_sign
            lat_sign_full[0, :] = lat_sign[0, :]
            dy_out = dy_out * lat_sign_full

    return dx_out * units.m, dy_out * units.m


# ============================================================================
# Utils
# ============================================================================

def angle_to_direction(degrees, level=16, full=False):
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
    d = _as_float(_strip(degrees, "degree")) if hasattr(degrees, "magnitude") else float(degrees)
    return _calc.angle_to_direction(d, int(level), bool(full))


def parse_angle(direction):
    """Parse a cardinal direction string to degrees.

    Parameters
    ----------
    direction : str

    Returns
    -------
    float or None
        Degrees (meteorological convention), or None if unrecognised.
    """
    result = _calc.parse_angle(direction)
    if result is not None:
        return result * units.degree
    return None


def find_bounding_indices(values, target):
    """Find two indices that bracket a target value.

    Parameters
    ----------
    values : array-like
    target : float

    Returns
    -------
    tuple of (int, int) or None
    """
    v = _as_1d(_strip(values, "")) if hasattr(values, "magnitude") else _as_1d(values)
    t = _as_float(_strip(target, "")) if hasattr(target, "magnitude") else float(target)
    return _calc.find_bounding_indices(v, t)


def nearest_intersection_idx(x, y1, y2):
    """Find the index nearest to where two series cross.

    Parameters
    ----------
    x : array-like
    y1, y2 : array-like

    Returns
    -------
    int or None
    """
    x_arr = _as_1d(_strip(x, "")) if hasattr(x, "magnitude") else _as_1d(x)
    y1_arr = _as_1d(_strip(y1, "")) if hasattr(y1, "magnitude") else _as_1d(y1)
    y2_arr = _as_1d(_strip(y2, "")) if hasattr(y2, "magnitude") else _as_1d(y2)
    return _calc.nearest_intersection_idx(x_arr, y1_arr, y2_arr)


def resample_nn_1d(x, xp, fp):
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
    x_arr = _as_1d(_strip(x, "")) if hasattr(x, "magnitude") else _as_1d(x)
    xp_arr = _as_1d(_strip(xp, "")) if hasattr(xp, "magnitude") else _as_1d(xp)
    fp_arr = _as_1d(_strip(fp, "")) if hasattr(fp, "magnitude") else _as_1d(fp)
    result = np.asarray(_calc.resample_nn_1d(x_arr, xp_arr, fp_arr))
    if hasattr(fp, "units"):
        return result * fp.units
    return result


def find_peaks(data, maxima=True, iqr_ratio=0.0):
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
    d = _as_1d(_strip(data, "")) if hasattr(data, "magnitude") else _as_1d(data)
    return np.asarray(_calc.find_peaks(d, bool(maxima), float(iqr_ratio)))


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
    d = _as_1d(_strip(data, "")) if hasattr(data, "magnitude") else _as_1d(data)
    return _calc.peak_persistence(d, bool(maxima))


def azimuth_range_to_lat_lon(azimuths, ranges, center_lat, center_lon):
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
    az = _as_1d(_strip(azimuths, "degree")) if hasattr(azimuths, "magnitude") else _as_1d(azimuths)
    rng = _as_1d(_strip(ranges, "m")) if hasattr(ranges, "magnitude") else _as_1d(ranges)
    clat = _as_float(_strip(center_lat, "degree")) if hasattr(center_lat, "magnitude") else float(center_lat)
    clon = _as_float(_strip(center_lon, "degree")) if hasattr(center_lon, "magnitude") else float(center_lon)
    lats, lons = _calc.azimuth_range_to_lat_lon(az, rng, clat, clon)
    return np.asarray(lats), np.asarray(lons)


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
    # Extract surface values (first element = highest pressure)
    t_arr = np.asarray(temperature.magnitude if hasattr(temperature, "magnitude") else temperature, dtype=np.float64)
    td_arr = np.asarray(dewpoint.magnitude if hasattr(dewpoint, "magnitude") else dewpoint, dtype=np.float64)
    t_units_obj = temperature.units if hasattr(temperature, "units") else units.degC
    t_sfc = t_arr[0] * t_units_obj
    td_sfc = td_arr[0] * t_units_obj
    p_out, t_parcel = parcel_profile_with_lcl(pressure, t_sfc, td_sfc)
    p_mag = p_out.magnitude if hasattr(p_out, "magnitude") else np.asarray(p_out)
    p_units = str(p_out.units) if hasattr(p_out, "units") else "hPa"
    # Interpolate ambient T and Td onto the new pressure grid (with LCL inserted)
    p_orig = np.asarray(pressure.magnitude if hasattr(pressure, "magnitude") else pressure, dtype=np.float64)
    t_orig = t_arr
    td_orig = td_arr
    p_new = np.asarray(p_mag, dtype=np.float64)
    t_interp = np.interp(p_new, p_orig[::-1], t_orig[::-1])[::-1] if p_orig[0] > p_orig[-1] else np.interp(p_new, p_orig, t_orig)
    td_interp = np.interp(p_new, p_orig[::-1], td_orig[::-1])[::-1] if p_orig[0] > p_orig[-1] else np.interp(p_new, p_orig, td_orig)
    t_parc = np.asarray(t_parcel.magnitude if hasattr(t_parcel, "magnitude") else t_parcel, dtype=np.float64)
    t_units = str(temperature.units) if hasattr(temperature, "units") else "degC"
    coord = xr.Variable("isobaric", p_new, attrs={"units": p_units})
    return xr.Dataset(
        {
            "ambient_temperature": xr.Variable("isobaric", t_interp, attrs={"units": t_units}),
            "ambient_dew_point": xr.Variable("isobaric", td_interp, attrs={"units": t_units}),
            "parcel_temperature": xr.Variable("isobaric", t_parc, attrs={"units": t_units}),
        },
        coords={"isobaric": coord},
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
    # Extract pressure from the vertical coordinate if not provided
    temp_arr = temperature
    if hasattr(temp_arr, "metpy"):
        temp_arr = temp_arr.metpy.dequantify()
    t_vals = np.asarray(temp_arr.values, dtype=np.float64)
    if pressure is not None:
        p_vals = np.asarray(pressure.values if hasattr(pressure, "values") else pressure, dtype=np.float64)
    else:
        # Try to find the vertical coordinate
        for dim in temp_arr.dims:
            coord = temp_arr.coords[dim]
            u = str(getattr(coord, "units", getattr(coord.attrs.get("units", ""), "__str__", lambda: "")()))
            if "Pa" in u or "hPa" in u or "millibar" in u or dim in ("isobaric", "level", "pressure"):
                p_vals = np.asarray(coord.values, dtype=np.float64)
                break
        else:
            p_vals = np.asarray(temp_arr.coords[temp_arr.dims[0]].values, dtype=np.float64)
    theta_levs = np.asarray(levels.magnitude if hasattr(levels, "magnitude") else levels, dtype=np.float64)
    # Build 3D pressure array if needed
    shape = t_vals.shape
    if p_vals.ndim == 1 and t_vals.ndim == 3:
        p_3d = np.broadcast_to(p_vals[:, None, None], shape).copy()
    elif p_vals.ndim == 1 and t_vals.ndim == 1:
        p_3d = p_vals
    else:
        p_3d = p_vals
    extra = [np.asarray(a.values if hasattr(a, "values") else a, dtype=np.float64) for a in args]
    result = isentropic_interpolation(theta_levs, p_3d, t_vals, extra)
    # Package as Dataset, preserving spatial dimensions for 3D output
    n_theta = len(theta_levs)
    theta_coord = xr.Variable("isentropic_level", theta_levs, attrs={"units": "K", "positive": "up"})

    # Determine output spatial dims from the input temperature array
    has_spatial = hasattr(temperature, "dims") and len(temperature.dims) >= 2
    if has_spatial:
        # Input is (vertical, ..spatial_dims..). Output is (isentropic_level, ..spatial_dims..)
        spatial_dims = temperature.dims[1:]  # e.g., ("y", "x")
        spatial_coords = {k: temperature.coords[k] for k in temperature.coords
                          if k not in (temperature.dims[0],) and k != "isentropic_level"}
        out_dims = ("isentropic_level",) + spatial_dims
        spatial_shape = tuple(temperature.sizes[d] for d in spatial_dims)
    else:
        spatial_dims = ()
        spatial_coords = {}
        out_dims = ("isentropic_level",)
        spatial_shape = ()

    ds_vars = {}
    all_coords = {"isentropic_level": theta_coord}
    all_coords.update(spatial_coords)

    if isinstance(result, (list, tuple)) and len(result) >= 2:
        target_shape = (n_theta,) + spatial_shape
        p_arr = np.asarray(result[0])
        t_arr = np.asarray(result[1])
        if p_arr.shape != target_shape and spatial_shape:
            try:
                p_arr = p_arr.reshape(target_shape)
                t_arr = t_arr.reshape(target_shape)
            except ValueError:
                pass
        ds_vars["pressure"] = xr.Variable(out_dims, p_arr, attrs={"units": "hPa"})
        ds_vars["temperature"] = xr.Variable(out_dims, t_arr, attrs={"units": "K"})
        for i, a in enumerate(args):
            name = getattr(a, "name", None) or f"field_{i}"
            if 2 + i < len(result):
                f_arr = np.asarray(result[2 + i])
                if f_arr.shape != target_shape and spatial_shape:
                    try:
                        f_arr = f_arr.reshape(target_shape)
                    except ValueError:
                        pass
                ds_vars[name] = xr.Variable(out_dims, f_arr)
    return xr.Dataset(ds_vars, coords=all_coords)


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
    import xarray as xr
    from scipy.ndimage import zoom as _zoom
    data = input_field.values
    if hasattr(input_field, "metpy"):
        data = input_field.metpy.dequantify().values
    zoomed = _zoom(np.asarray(data, dtype=np.float64), zoom, output=output,
                   order=order, mode=mode, cval=cval, prefilter=prefilter)
    zoom_factors = np.atleast_1d(zoom)
    if zoom_factors.size == 1:
        zoom_factors = np.full(data.ndim, zoom_factors[0])
    new_coords = {}
    for i, dim in enumerate(input_field.dims):
        if dim in input_field.coords:
            old = np.asarray(input_field.coords[dim].values, dtype=np.float64)
            new_len = zoomed.shape[i]
            new_coords[dim] = np.linspace(old[0], old[-1], new_len)
    result = xr.DataArray(zoomed, dims=input_field.dims, coords=new_coords,
                          attrs=input_field.attrs.copy())
    return result


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
