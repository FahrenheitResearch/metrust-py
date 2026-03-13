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
    shapes = [a.shape for a in arrays if a.ndim > 0]
    orig_shape = np.broadcast_shapes(*shapes)
    flat = [np.ascontiguousarray(np.broadcast_to(a, orig_shape).ravel()) for a in arrays]
    return flat, orig_shape, True


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


def lfc(pressure, temperature, dewpoint):
    """Level of Free Convection.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
        LFC pressure.
    """
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    result = _calc.lfc(p, t, td)
    return result * units.hPa


def el(pressure, temperature, dewpoint):
    """Equilibrium Level.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
        EL pressure.
    """
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    result = _calc.el(p, t, td)
    return result * units.hPa


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
        t = _as_float(_strip(temperature, "degC"))
        w = _as_float(_strip(pressure_or_mixing_ratio, "kg/kg")) if hasattr(pressure_or_mixing_ratio, "magnitude") else float(pressure_or_mixing_ratio)
        eps = molecular_weight_ratio
        t_k = t + 273.15
        tv_k = t_k * (1.0 + w / eps) / (1.0 + w)
        return _attach(tv_k - 273.15, "degC")
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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.virtual_temperature_from_dewpoint(t, td, p) * units.degC


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
    h = _as_1d(_strip(parcel_profile_or_height, "m"))
    ps = _as_float(_strip(psfc, "hPa"))
    t2 = _as_float(_strip(t2m, "degC"))
    td2 = _as_float(_strip(td2m, "degC"))
    cape_val, cin_val, h_lcl, h_lfc = _calc.cape_cin(
        p, t, td, h, ps, t2, td2, parcel_type,
        float(ml_depth), float(mu_depth), top_m,
    )
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
    e = _as_float(_strip(partial_press_or_pressure, "Pa"))
    p = _as_float(_strip(total_press_or_temperature, "Pa"))
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
    if len(args) != 5:
        raise TypeError("k_index expects either (pressure, temperature, dewpoint) or 5 scalar level values")
    t850, td850, t700, td700, t500 = args
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
    if len(args) != 3:
        raise TypeError("total_totals expects either (pressure, temperature, dewpoint) or 3 scalar level values")
    t850, td850, t500 = args
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
    if len(args) != 2:
        raise TypeError("cross_totals expects either (pressure, temperature, dewpoint) or (td850, t500)")
    td850, t500 = args
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
    if len(args) != 2:
        raise TypeError("vertical_totals expects either (pressure, temperature) or (t850, t500)")
    t850, t500 = args
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
    w = _as_float(_strip(mixing_ratio_kgkg, "kg/kg")) if hasattr(mixing_ratio_kgkg, "magnitude") else float(mixing_ratio_kgkg)
    return _calc.moist_air_gas_constant(w) * units("J/(kg*K)")


def moist_air_specific_heat_pressure(mixing_ratio_kgkg):
    """Specific heat at constant pressure for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : Quantity (kg/kg) or float

    Returns
    -------
    Quantity (J/(kg*K))
    """
    w = _as_float(_strip(mixing_ratio_kgkg, "kg/kg")) if hasattr(mixing_ratio_kgkg, "magnitude") else float(mixing_ratio_kgkg)
    return _calc.moist_air_specific_heat_pressure(w) * units("J/(kg*K)")


def moist_air_poisson_exponent(mixing_ratio_kgkg):
    """Poisson exponent (kappa) for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : Quantity (kg/kg) or float

    Returns
    -------
    Quantity (dimensionless)
    """
    w = _as_float(_strip(mixing_ratio_kgkg, "kg/kg")) if hasattr(mixing_ratio_kgkg, "magnitude") else float(mixing_ratio_kgkg)
    return _calc.moist_air_poisson_exponent(w) * units.dimensionless


def water_latent_heat_vaporization(temperature):
    """Latent heat of vaporization (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    t = _as_float(_strip(temperature, "degC"))
    return _calc.water_latent_heat_vaporization(t) * units("J/kg")


def water_latent_heat_melting(temperature):
    """Latent heat of melting (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    t = _as_float(_strip(temperature, "degC"))
    return _calc.water_latent_heat_melting(t) * units("J/kg")


def water_latent_heat_sublimation(temperature):
    """Latent heat of sublimation (temperature-dependent).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (J/kg)
    """
    t = _as_float(_strip(temperature, "degC"))
    return _calc.water_latent_heat_sublimation(t) * units("J/kg")


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
    t = _as_float(_strip(temperature, "degC"))
    tw = _as_float(_strip(wet_bulb, "degC"))
    p = _as_float(_strip(pressure, "hPa"))
    return _calc.relative_humidity_wet_psychrometric(t, tw, p) * units.percent


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
    p = _as_float(_strip(pressure, "hPa"))
    dh = _as_float(_strip(delta_height, "m"))
    return _calc.add_height_to_pressure(p, dh) * units.hPa


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
    h = _as_float(_strip(height, "m"))
    dp = _as_float(_strip(delta_pressure, "hPa"))
    return _calc.add_pressure_to_height(h, dp) * units.m


def thickness_hydrostatic(p_bottom, p_top, t_mean):
    """Hypsometric thickness between two pressure levels.

    Parameters
    ----------
    p_bottom : Quantity (pressure)
    p_top : Quantity (pressure)
    t_mean : Quantity (temperature, K)

    Returns
    -------
    Quantity (m)
    """
    pb = _as_float(_strip(p_bottom, "hPa"))
    pt = _as_float(_strip(p_top, "hPa"))
    tm = _as_float(_strip(t_mean, "K"))
    return _calc.thickness_hydrostatic(pb, pt, tm) * units.m


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
        pressure_pa = _attach(_as_float(_strip(pressure_or_dewpoint, "Pa")), "Pa")
        mixing_ratio_val = _attach(_as_float(_strip(mixing_ratio, "kg/kg")), "kg/kg")
        return pressure_pa * mixing_ratio_val / (molecular_weight_ratio + mixing_ratio_val)
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
    w = _as_float(_strip(mixing_ratio, "kg/kg"))
    return _calc.specific_humidity_from_mixing_ratio(w) * units("kg/kg")


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
    e = _as_float(_strip(vapor_pressure_val, "hPa"))
    return _calc.dewpoint(e) * units.degC


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
    p = _as_float(_strip(pressure, "hPa"))
    q = _as_float(_strip(specific_humidity, "kg/kg")) if hasattr(specific_humidity, "magnitude") else float(specific_humidity)
    return _calc.dewpoint_from_specific_humidity(p, q) * units.degC


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
    z = _as_float(_strip(height, "m"))
    t_k = _as_float(_strip(temperature, "K"))
    return _calc.dry_static_energy(z, t_k) * units("J/kg")


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
    gp = _as_float(_strip(geopotential, "m**2/s**2")) if hasattr(geopotential, "magnitude") else float(geopotential)
    return _calc.geopotential_to_height(gp) * units.m


def get_layer(pressure, values, p_bottom, p_top):
    """Extract a layer from a sounding between two pressures.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    values : array Quantity
    p_bottom : Quantity (pressure)
    p_top : Quantity (pressure)

    Returns
    -------
    tuple of (array Quantity (hPa), array Quantity)
    """
    p = _as_1d(_strip(pressure, "hPa"))
    has_units = hasattr(values, "units")
    v_unit = values.units if has_units else None
    v = _as_1d(_strip(values, "")) if has_units else _as_1d(values)
    pb = _as_float(_strip(p_bottom, "hPa"))
    pt = _as_float(_strip(p_top, "hPa"))
    p_out, v_out = _calc.get_layer(p, v, pb, pt)
    p_result = np.asarray(p_out) * units.hPa
    v_result = np.asarray(v_out)
    if v_unit is not None:
        v_result = v_result * v_unit
    return p_result, v_result


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
    h = _as_float(_strip(height, "m"))
    return _calc.height_to_geopotential(h) * units("m**2/s**2")


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


def mixed_layer(pressure, values, depth=100.0):
    """Mixed-layer mean of a quantity.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    values : array Quantity or array-like
    depth : float
        Depth in hPa (default 100).

    Returns
    -------
    float
    """
    p = _as_1d(_strip(pressure, "hPa"))
    v = _as_1d(_strip(values, "")) if hasattr(values, "magnitude") else _as_1d(values)
    d = _as_float(_strip(depth, "hPa")) if hasattr(depth, "magnitude") else float(depth)
    return _calc.mixed_layer(p, v, d)


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    rh = _rh_to_percent(relative_humidity)
    # Rust returns g/kg; MetPy convention is dimensionless kg/kg
    return _attach(_calc.mixing_ratio_from_relative_humidity(p, t, rh) / 1000.0, "kg/kg")


def mixing_ratio_from_specific_humidity(specific_humidity):
    """Mixing ratio from specific humidity.

    Parameters
    ----------
    specific_humidity : Quantity (kg/kg)

    Returns
    -------
    Quantity (dimensionless, kg/kg)
    """
    q = _as_float(_strip(specific_humidity, "kg/kg")) if hasattr(specific_humidity, "magnitude") else float(specific_humidity)
    # Rust returns g/kg; MetPy convention is dimensionless kg/kg
    return _attach(_calc.mixing_ratio_from_specific_humidity(q) / 1000.0, "kg/kg")


def moist_lapse(pressure, t_start):
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
    result = np.asarray(_calc.moist_lapse(p, t))
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
    z = _as_float(_strip(height, "m"))
    t_k = _as_float(_strip(temperature, "K"))
    q = _as_float(_strip(specific_humidity, "kg/kg")) if hasattr(specific_humidity, "magnitude") else float(specific_humidity)
    return _calc.moist_static_energy(z, t_k, q) * units("J/kg")


def montgomery_streamfunction(theta, pressure, temperature, height):
    """Montgomery streamfunction.

    Parameters
    ----------
    theta : Quantity (K)
    pressure : Quantity (pressure)
    temperature : Quantity (K)
    height : Quantity (m)

    Returns
    -------
    Quantity (J/kg)
    """
    th = _as_float(_strip(theta, "K"))
    p = _as_float(_strip(pressure, "hPa"))
    t_k = _as_float(_strip(temperature, "K"))
    z = _as_float(_strip(height, "m"))
    return _calc.montgomery_streamfunction(th, p, t_k, z) * units("J/kg")


def most_unstable_cape_cin(pressure, temperature, dewpoint):
    """Most-unstable CAPE and CIN.

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
    cape_val, cin_val = _calc.most_unstable_cape_cin(p, t, td)
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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    w = _as_float(_strip(mixing_ratio_val, "g/kg")) if _can_convert(mixing_ratio_val, "g/kg") else _as_float(_strip(mixing_ratio_val, "kg/kg")) * 1000.0
    # Rust returns percent; MetPy returns dimensionless fraction
    return _attach(_calc.relative_humidity_from_mixing_ratio(p, t, w) / 100.0, "")


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    q = _as_float(_strip(specific_humidity, "kg/kg")) if hasattr(specific_humidity, "magnitude") else float(specific_humidity)
    # Rust returns percent; MetPy returns dimensionless fraction
    return _attach(_calc.relative_humidity_from_specific_humidity(p, t, q) / 100.0, "")


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.saturation_equivalent_potential_temperature(p, t) * units.K


def scale_height(temperature):
    """Atmospheric scale height.

    Parameters
    ----------
    temperature : Quantity (K)

    Returns
    -------
    Quantity (m)
    """
    t_k = _as_float(_strip(temperature, "K"))
    return _calc.scale_height(t_k) * units.m


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
    p = _as_float(_strip(pressure, "hPa"))
    td = _as_float(_strip(dewpoint_val, "degC"))
    return _calc.specific_humidity_from_dewpoint(p, td) * units("kg/kg")


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
    p = _as_float(_strip(pressure, "hPa"))
    th = _as_float(_strip(theta, "K"))
    return _calc.temperature_from_potential_temperature(p, th) * units.K


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
    o = _as_float(_strip(omega, "Pa/s"))
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.vertical_velocity(o, p, t) * units("m/s")


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
    w_val = _as_float(_strip(w, "m/s"))
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.vertical_velocity_pressure(w_val, p, t) * units("Pa/s")


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    w = _as_float(_strip(mixing_ratio_val, "g/kg")) if _can_convert(mixing_ratio_val, "g/kg") else _as_float(_strip(mixing_ratio_val, "kg/kg")) * 1000.0
    return _calc.virtual_potential_temperature(p, t, w) * units.K


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.wet_bulb_potential_temperature(p, t, td) * units.K


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


def get_most_unstable_parcel(pressure, temperature, dewpoint, depth=300.0):
    """Get most-unstable parcel properties.

    Parameters
    ----------
    pressure : array Quantity (pressure)
    temperature : array Quantity (temperature)
    dewpoint : array Quantity (temperature)
    depth : float
        Search depth in hPa (default 300).

    Returns
    -------
    tuple of (Quantity (hPa), Quantity (degC), Quantity (degC))
        Parcel pressure, temperature, and dewpoint.
    """
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    d = _as_float(_strip(depth, "hPa")) if hasattr(depth, "magnitude") else float(depth)
    pp, tp, tdp = _calc.get_most_unstable_parcel(p, t, td, d)
    return pp * units.hPa, tp * units.degC, tdp * units.degC


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
    t = _as_float(_strip(temperature, "degC"))
    tw = _as_float(_strip(wet_bulb, "degC"))
    p = _as_float(_strip(pressure, "hPa"))
    return _calc.psychrometric_vapor_pressure(t, tw, p) * units.hPa


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
    t = _as_float(_strip(temperature, "degC"))
    rh = _rh_to_percent(relative_humidity)
    return _calc.frost_point(t, rh) * units.degC


# ============================================================================
# Aliases
# ============================================================================

def mixed_parcel(pressure, temperature, dewpoint, depth=100):
    """Alias for :func:`get_mixed_layer_parcel`."""
    return get_mixed_layer_parcel(pressure, temperature, dewpoint, depth)


def most_unstable_parcel(pressure, temperature, dewpoint, depth=300):
    """Alias for :func:`get_most_unstable_parcel`."""
    return get_most_unstable_parcel(pressure, temperature, dewpoint, depth)


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


def bulk_shear(pressure_or_u, u_or_v, v_or_height, height=None, bottom=None, depth=None, top=None):
    """Bulk wind shear over a height layer.

    Parameters
    ----------
    u, v : array Quantity (m/s)
        Wind component profiles.
    height : array Quantity (m)
        Height profile.
    bottom : Quantity (m)
        Bottom of the layer.
    top : Quantity (m)
        Top of the layer.

    Returns
    -------
    tuple of (Quantity (m/s), Quantity (m/s))
        Shear u and v components.
    """
    u_arr = _as_1d(_strip(pressure_or_u, "m/s"))
    v_arr = _as_1d(_strip(u_or_v, "m/s"))
    h_arr = _as_1d(_strip(v_or_height, "m"))
    bot = _as_float(_strip(height, "m"))
    top_src = top if top is not None else bottom
    top_val = _as_float(_strip(top_src, "m"))
    su, sv = _calc.bulk_shear(u_arr, v_arr, h_arr, bot, top_val)
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


def storm_relative_helicity(*args, bottom=None, storm_u=None, storm_v=None):
    """Storm-relative helicity.

    Parameters
    ----------
    u, v : array Quantity (m/s)
    height : array Quantity (m)
    depth : Quantity (m)
    storm_u, storm_v : Quantity (m/s)

    Returns
    -------
    tuple of (Quantity (m^2/s^2), Quantity (m^2/s^2), Quantity (m^2/s^2))
        Positive, negative, and total SRH.
    """
    if len(args) != 6:
        raise TypeError(
            "storm_relative_helicity expects either (height, u, v, depth, *, bottom, storm_u, storm_v) "
            "or legacy positional (u, v, height, depth, storm_u, storm_v)"
        )
    u, v, height, depth, storm_u, storm_v = args
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    d = _as_float(_strip(depth, "m"))
    su = _as_float(_strip(storm_u, "m/s"))
    sv = _as_float(_strip(storm_v, "m/s"))
    pos, neg, total = _calc.storm_relative_helicity(u_arr, v_arr, h_arr, d, su, sv)
    return _attach(pos, "m**2/s**2"), _attach(neg, "m**2/s**2"), _attach(total, "m**2/s**2")


def bunkers_storm_motion(pressure_or_u, u_or_v, v_or_height, height=None):
    """Bunkers storm motion (right-mover, left-mover, mean wind).

    Parameters
    ----------
    u, v : array Quantity (m/s)
    height : array Quantity (m)

    Returns
    -------
    tuple of 3 tuples, each (Quantity (m/s), Quantity (m/s))
        (right_u, right_v), (left_u, left_v), (mean_u, mean_v)
    """
    u_arr = _as_1d(_strip(pressure_or_u, "m/s"))
    v_arr = _as_1d(_strip(u_or_v, "m/s"))
    h_arr = _as_1d(_strip(v_or_height, "m"))
    (ru, rv), (lu, lv), (mu, mv) = _calc.bunkers_storm_motion(u_arr, v_arr, h_arr)
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


def divergence(u, v, dx, dy):
    """Horizontal divergence on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.divergence(u_f, v_f, dx_val, dy_val))
    return result * units("1/s")


def vorticity(u, v, dx, dy):
    """Relative vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array Quantity (m/s)
    dx, dy : Quantity (m)

    Returns
    -------
    2-D array Quantity (1/s)
    """
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.vorticity(u_f, v_f, dx_val, dy_val))
    return result * units("1/s")


def absolute_vorticity(u, v, lats, dx, dy):
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
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.absolute_vorticity(u_f, v_f, lats_f, dx_val, dy_val))
    return result * units("1/s")


def advection(scalar, u, v, dx, dy):
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
    has_units = hasattr(scalar, "units")
    s_unit = scalar.units if has_units else units.dimensionless
    s_f = _flat(scalar)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.advection(s_f, u_f, v_f, dx_val, dy_val))
    return result * (s_unit / units.s)


def frontogenesis(theta, u, v, dx, dy):
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
    t_f = _flat(theta, "K")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.frontogenesis(t_f, u_f, v_f, dx_val, dy_val))
    return result * units("K/m/s")


def geostrophic_wind(heights, lats, dx, dy):
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
    h_f = _flat(heights, "m")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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


def potential_vorticity_baroclinic(potential_temp, pressure, theta_below,
                                   theta_above, u, v, lats, dx, dy):
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
    ny, nx = _grid_shape(potential_temp)
    pt_f = _flat(potential_temp, "K")
    tb_f = _flat(theta_below, "K")
    ta_f = _flat(theta_above, "K")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    # Rust expects Pa for the pressure pair
    if hasattr(pressure, "magnitude"):
        p_arr = np.asarray(pressure.to("Pa").magnitude, dtype=np.float64)
    else:
        p_arr = np.asarray(pressure, dtype=np.float64)
    result = np.asarray(_calc.potential_vorticity_baroclinic(
        pt_f, p_arr, tb_f, ta_f, u_f, v_f, lats_f, dx_val, dy_val,
    ))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    lat = _as_float(_strip(latitude, "degree")) if hasattr(latitude, "magnitude") else float(latitude)
    return _calc.coriolis_parameter(lat) * units("1/s")


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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    if dx is None or dy is None:
        raise TypeError("q_vector requires dx and dy grid spacings")
    t_2d = _as_2d(temperature, "K") if _can_convert(temperature, "K") else _as_2d(temperature, "degC")
    u_2d = _as_2d(u, "m/s")
    v_2d = _as_2d(v, "m/s")
    p_val = _as_float(_strip(pressure, "hPa"))
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    q1, q2 = _calc.q_vector(t_2d, u_2d, v_2d, p_val, dx_val, dy_val)
    return np.asarray(q1), np.asarray(q2)


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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
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

def significant_tornado_parameter(mlcape, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter (STP).

    Parameters
    ----------
    mlcape : Quantity (J/kg)
    lcl_height : Quantity (m)
    srh_0_1km : Quantity (m^2/s^2)
    bulk_shear_0_6km : Quantity (m/s)

    Returns
    -------
    Quantity (dimensionless)
    """
    cape = _as_float(_strip(mlcape, "J/kg"))
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

    Parameters
    ----------
    storm_u, storm_v : Quantity (m/s)
    u_sfc, v_sfc : Quantity (m/s)
    u_500m, v_500m : Quantity (m/s)

    Returns
    -------
    Quantity (degree)
    """
    if len(args) != 6:
        raise TypeError(
            "critical_angle expects either (pressure, u, v, height, u_storm, v_storm) "
            "or legacy positional (storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m)"
        )
    storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m = args
    return _attach(
        _calc.critical_angle(
            _as_float(_strip(storm_u, "m/s")),
            _as_float(_strip(storm_v, "m/s")),
            _as_float(_strip(u_sfc, "m/s")),
            _as_float(_strip(v_sfc, "m/s")),
            _as_float(_strip(u_500m, "m/s")),
            _as_float(_strip(v_500m, "m/s")),
        ),
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
    t = _as_float(_strip(temperature, "degC"))
    rh = _as_float(_strip(relative_humidity, "percent")) if hasattr(relative_humidity, "magnitude") else float(relative_humidity)
    return _calc.heat_index(t, rh) * units.degC


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
    t = _as_float(_strip(temperature, "degC"))
    ws = _as_float(_strip(wind_speed_val, "m/s"))
    return _calc.windchill(t, ws) * units.degC


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
    t = _as_float(_strip(temperature, "degC"))
    rh = _as_float(_strip(relative_humidity, "percent")) if hasattr(relative_humidity, "magnitude") else float(relative_humidity)
    ws = _as_float(_strip(wind_speed_val, "m/s"))
    return _calc.apparent_temperature(t, rh, ws) * units.degC


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
    dx_val = _as_float(_strip(dx, "m"))
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
    dy_val = _as_float(_strip(dy, "m"))
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
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.asarray(_calc.laplacian(d_arr, dx_val, dy_val))
    if has_units:
        return result * (data.units / units.m ** 2)
    return result


def first_derivative(data, axis_spacing, axis=0):
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
    ds = _as_float(_strip(axis_spacing, "m"))
    result = np.asarray(_calc.first_derivative(d_arr, ds, int(axis)))
    if has_units:
        return result * (data.units / units.m)
    return result


def second_derivative(data, axis_spacing, axis=0):
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
    ds = _as_float(_strip(axis_spacing, "m"))
    result = np.asarray(_calc.second_derivative(d_arr, ds, int(axis)))
    if has_units:
        return result * (data.units / units.m ** 2)
    return result


def lat_lon_grid_deltas(lats, lons):
    """Physical grid spacings (dx, dy) in meters from lat/lon grids.

    Parameters
    ----------
    lats : 2-D array (degrees)
    lons : 2-D array (degrees)

    Returns
    -------
    tuple of (2-D array Quantity (m), 2-D array Quantity (m))
    """
    lat_arr = np.asarray(lats.magnitude if hasattr(lats, "magnitude") else lats,
                         dtype=np.float64)
    lon_arr = np.asarray(lons.magnitude if hasattr(lons, "magnitude") else lons,
                         dtype=np.float64)
    dx, dy = _calc.lat_lon_grid_deltas(lat_arr, lon_arr)
    return np.asarray(dx) * units.m, np.asarray(dy) * units.m


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
    s_unit = scalar.units if has_units else units.dimensionless
    s_arr = _as_1d(_strip(scalar, "")) if has_units else _as_1d(scalar)
    u_arr = _as_1d(_strip(u, "m/s")) if hasattr(u, "magnitude") else _as_1d(u)
    v_arr = _as_1d(_strip(v, "m/s")) if hasattr(v, "magnitude") else _as_1d(v)
    w_arr = _as_1d(_strip(w, "m/s")) if hasattr(w, "magnitude") else _as_1d(w)
    dx_val = _as_float(_strip(dx, "m")) if hasattr(dx, "magnitude") else float(dx)
    dy_val = _as_float(_strip(dy, "m")) if hasattr(dy, "magnitude") else float(dy)
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
    return result.reshape(nz, ny, nx) * (s_unit / units.s)


# ============================================================================
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
