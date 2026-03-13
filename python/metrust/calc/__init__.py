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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.potential_temperature(p, t) * units.K


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.equivalent_potential_temperature(p, t, td) * units.K


def saturation_vapor_pressure(temperature):
    """Saturation vapor pressure (Bolton 1980).

    Parameters
    ----------
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
    """
    t = _as_float(_strip(temperature, "degC"))
    return _calc.saturation_vapor_pressure(t) * units.hPa


def saturation_mixing_ratio(pressure, temperature):
    """Saturation mixing ratio.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (g/kg)
    """
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.saturation_mixing_ratio(p, t) * units("g/kg")


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
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.wet_bulb_temperature(p, t, td) * units.degC


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
    t = _as_float(_strip(temperature, "degC"))
    rh = _as_float(_strip(relative_humidity, "percent")) if hasattr(relative_humidity, "magnitude") else float(relative_humidity)
    return _calc.dewpoint_from_relative_humidity(t, rh) * units.degC


def relative_humidity_from_dewpoint(temperature, dewpoint):
    """Relative humidity from temperature and dewpoint.

    Parameters
    ----------
    temperature : Quantity (temperature)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (percent)
    """
    t = _as_float(_strip(temperature, "degC"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.relative_humidity_from_dewpoint(t, td) * units.percent


def virtual_temperature(temperature, pressure, dewpoint):
    """Virtual temperature.

    Parameters
    ----------
    temperature : Quantity (temperature)
    pressure : Quantity (pressure)
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (degC)
    """
    t = _as_float(_strip(temperature, "degC"))
    p = _as_float(_strip(pressure, "hPa"))
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.virtual_temperature(t, p, td) * units.degC


def cape_cin(pressure, temperature, dewpoint, height,
             psfc, t2m, td2m, parcel_type="sb",
             ml_depth=100.0, mu_depth=300.0, top_m=None):
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
    p = _as_1d(_strip(pressure, "hPa"))
    t = _as_1d(_strip(temperature, "degC"))
    td = _as_1d(_strip(dewpoint, "degC"))
    h = _as_1d(_strip(height, "m"))
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


def mixing_ratio(pressure, temperature):
    """Mixing ratio from pressure and temperature.

    Parameters
    ----------
    pressure : Quantity (pressure)
    temperature : Quantity (temperature)

    Returns
    -------
    Quantity (g/kg)
    """
    p = _as_float(_strip(pressure, "hPa"))
    t = _as_float(_strip(temperature, "degC"))
    return _calc.mixing_ratio(p, t) * units("g/kg")


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


def k_index(t850, td850, t700, td700, t500):
    """K-Index.

    All temperatures in Celsius (or Quantity).

    Returns
    -------
    Quantity (delta_degC)
    """
    return _calc.k_index(
        _as_float(_strip(t850, "degC")),
        _as_float(_strip(td850, "degC")),
        _as_float(_strip(t700, "degC")),
        _as_float(_strip(td700, "degC")),
        _as_float(_strip(t500, "degC")),
    ) * units.delta_degC


def total_totals(t850, td850, t500):
    """Total Totals Index.

    Returns
    -------
    Quantity (delta_degC)
    """
    return _calc.total_totals(
        _as_float(_strip(t850, "degC")),
        _as_float(_strip(td850, "degC")),
        _as_float(_strip(t500, "degC")),
    ) * units.delta_degC


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


def cross_totals(td850, t500):
    """Cross Totals: Td850 - T500.

    Returns
    -------
    Quantity (delta_degC)
    """
    return _calc.cross_totals(
        _as_float(_strip(td850, "degC")),
        _as_float(_strip(t500, "degC")),
    ) * units.delta_degC


def vertical_totals(t850, t500):
    """Vertical Totals: T850 - T500.

    Returns
    -------
    Quantity (delta_degC)
    """
    return _calc.vertical_totals(
        _as_float(_strip(t850, "degC")),
        _as_float(_strip(t500, "degC")),
    ) * units.delta_degC


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


def vapor_pressure(dewpoint):
    """Vapor pressure from dewpoint temperature.

    Parameters
    ----------
    dewpoint : Quantity (temperature)

    Returns
    -------
    Quantity (hPa)
    """
    td = _as_float(_strip(dewpoint, "degC"))
    return _calc.vapor_pressure(td) * units.hPa


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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.wind_speed(u_arr, v_arr))
    return result * units("m/s")


def wind_direction(u, v):
    """Meteorological wind direction from (u, v).

    Parameters
    ----------
    u, v : array Quantity (m/s)

    Returns
    -------
    array Quantity (degree)
    """
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    result = np.asarray(_calc.wind_direction(u_arr, v_arr))
    return result * units.degree


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
    spd = _as_1d(_strip(speed, "m/s"))
    dirn = _as_1d(_strip(direction, "degree"))
    u, v = _calc.wind_components(spd, dirn)
    return np.asarray(u) * units("m/s"), np.asarray(v) * units("m/s")


def bulk_shear(u, v, height, bottom, top):
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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    bot = _as_float(_strip(bottom, "m"))
    top_val = _as_float(_strip(top, "m"))
    su, sv = _calc.bulk_shear(u_arr, v_arr, h_arr, bot, top_val)
    return su * units("m/s"), sv * units("m/s")


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


def storm_relative_helicity(u, v, height, depth, storm_u, storm_v):
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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    d = _as_float(_strip(depth, "m"))
    su = _as_float(_strip(storm_u, "m/s"))
    sv = _as_float(_strip(storm_v, "m/s"))
    pos, neg, total = _calc.storm_relative_helicity(u_arr, v_arr, h_arr, d, su, sv)
    srh_unit = units("m**2/s**2")
    return pos * srh_unit, neg * srh_unit, total * srh_unit


def bunkers_storm_motion(u, v, height):
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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    (ru, rv), (lu, lv), (mu, mv) = _calc.bunkers_storm_motion(u_arr, v_arr, h_arr)
    ms = units("m/s")
    return (ru * ms, rv * ms), (lu * ms, lv * ms), (mu * ms, mv * ms)


def corfidi_storm_motion(u, v, height, u_850, v_850):
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
    u_arr = _as_1d(_strip(u, "m/s"))
    v_arr = _as_1d(_strip(v, "m/s"))
    h_arr = _as_1d(_strip(height, "m"))
    u8 = _as_float(_strip(u_850, "m/s"))
    v8 = _as_float(_strip(v_850, "m/s"))
    (uu, uv), (du, dv) = _calc.corfidi_storm_motion(u_arr, v_arr, h_arr, u8, v8)
    ms = units("m/s")
    return (uu * ms, uv * ms), (du * ms, dv * ms)


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
    """Flatten a 2-D Quantity to a contiguous float64 1-D array."""
    if unit and hasattr(data, "magnitude"):
        arr = np.asarray(data.to(unit).magnitude, dtype=np.float64)
    elif hasattr(data, "magnitude"):
        arr = np.asarray(data.magnitude, dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    return np.ascontiguousarray(arr.ravel())


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
    ny, nx = _grid_shape(u)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.divergence(u_f, v_f, nx, ny, dx_val, dy_val))
    return result.reshape(ny, nx) * units("1/s")


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
    ny, nx = _grid_shape(u)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.vorticity(u_f, v_f, nx, ny, dx_val, dy_val))
    return result.reshape(ny, nx) * units("1/s")


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
    ny, nx = _grid_shape(u)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.absolute_vorticity(u_f, v_f, lats_f, nx, ny, dx_val, dy_val))
    return result.reshape(ny, nx) * units("1/s")


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
    ny, nx = _grid_shape(scalar)
    has_units = hasattr(scalar, "units")
    s_unit = scalar.units if has_units else units.dimensionless
    s_f = _flat(scalar)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.advection(s_f, u_f, v_f, nx, ny, dx_val, dy_val))
    return result.reshape(ny, nx) * (s_unit / units.s)


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
    ny, nx = _grid_shape(theta)
    t_f = _flat(theta, "K")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.frontogenesis(t_f, u_f, v_f, nx, ny, dx_val, dy_val))
    return result.reshape(ny, nx) * units("K/m/s")


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
    ny, nx = _grid_shape(heights)
    h_f = _flat(heights, "m")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    u_g, v_g = _calc.geostrophic_wind(h_f, lats_f, nx, ny, dx_val, dy_val)
    ms = units("m/s")
    return np.array(u_g).reshape(ny, nx) * ms, np.array(v_g).reshape(ny, nx) * ms


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
    ny, nx = _grid_shape(u)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    h_f = _flat(heights, "m")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    ua, va = _calc.ageostrophic_wind(u_f, v_f, h_f, lats_f, nx, ny, dx_val, dy_val)
    ms = units("m/s")
    return np.array(ua).reshape(ny, nx) * ms, np.array(va).reshape(ny, nx) * ms


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
    result = np.array(_calc.potential_vorticity_baroclinic(
        pt_f, p_arr, tb_f, ta_f, u_f, v_f, lats_f, nx, ny, dx_val, dy_val,
    ))
    return result.reshape(ny, nx) * units("K*m**2/(kg*s)")


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
    ny, nx = _grid_shape(heights)
    h_f = _flat(heights, "m")
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    lats_f = _flat(lats, "degree") if hasattr(lats, "magnitude") else _flat(lats)
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    result = np.array(_calc.potential_vorticity_barotropic(
        h_f, u_f, v_f, lats_f, nx, ny, dx_val, dy_val,
    ))
    return result.reshape(ny, nx) * units("1/(m*s)")


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
    ny, nx = _grid_shape(u)
    u_f = _flat(u, "m/s")
    v_f = _flat(v, "m/s")
    dx_val = _as_float(_strip(dx, "m"))
    dy_val = _as_float(_strip(dy, "m"))
    dudx, dudy, dvdx, dvdy = _calc.vector_derivative(u_f, v_f, nx, ny, dx_val, dy_val)
    inv_s = units("1/s")
    return (
        np.array(dudx).reshape(ny, nx) * inv_s,
        np.array(dudy).reshape(ny, nx) * inv_s,
        np.array(dvdx).reshape(ny, nx) * inv_s,
        np.array(dvdy).reshape(ny, nx) * inv_s,
    )


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


def critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_500m, v_500m):
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
    return _calc.critical_angle(
        _as_float(_strip(storm_u, "m/s")),
        _as_float(_strip(storm_v, "m/s")),
        _as_float(_strip(u_sfc, "m/s")),
        _as_float(_strip(v_sfc, "m/s")),
        _as_float(_strip(u_500m, "m/s")),
        _as_float(_strip(v_500m, "m/s")),
    ) * units.degree


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


def smooth_rectangular(data, size):
    """Rectangular (box) smoothing.

    Parameters
    ----------
    data : 2-D array
    size : int (kernel side length)

    Returns
    -------
    2-D ndarray
    """
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_rectangular(arr, int(size))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_circular(data, radius):
    """Circular (disk) smoothing.

    Parameters
    ----------
    data : 2-D array
    radius : float (grid-point units)

    Returns
    -------
    2-D ndarray
    """
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_circular(arr, float(radius))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_n_point(data, n):
    """N-point smoother (5 or 9).

    Parameters
    ----------
    data : 2-D array
    n : int (5 or 9)

    Returns
    -------
    2-D ndarray
    """
    arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                     dtype=np.float64)
    result = _calc.smooth_n_point(arr, int(n))
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


def smooth_window(data, window):
    """Generic 2-D convolution with a user-supplied kernel.

    Parameters
    ----------
    data : 2-D array
    window : 2-D array (kernel)

    Returns
    -------
    2-D ndarray
    """
    d_arr = np.asarray(data.magnitude if hasattr(data, "magnitude") else data,
                       dtype=np.float64)
    w_arr = np.asarray(window.magnitude if hasattr(window, "magnitude") else window,
                       dtype=np.float64)
    result = _calc.smooth_window(d_arr, w_arr)
    if hasattr(data, "units"):
        return np.asarray(result) * data.units
    return np.asarray(result)


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

def angle_to_direction(degrees):
    """Convert a meteorological angle to a 16-point cardinal direction string.

    Parameters
    ----------
    degrees : float or Quantity (degree)

    Returns
    -------
    str
    """
    d = _as_float(_strip(degrees, "degree")) if hasattr(degrees, "magnitude") else float(degrees)
    return _calc.angle_to_direction(d)


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


# ============================================================================
# __all__ -- explicit public API
# ============================================================================

__all__ = [
    # thermo
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
    "vapor_pressure",
    # wind
    "wind_speed",
    "wind_direction",
    "wind_components",
    "bulk_shear",
    "mean_wind",
    "storm_relative_helicity",
    "bunkers_storm_motion",
    "corfidi_storm_motion",
    # kinematics
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
]
