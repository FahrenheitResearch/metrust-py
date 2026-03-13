from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import pytest

import metrust.calc as mcalc
from metrust.units import units as mr_units

mpcalc = pytest.importorskip("metpy.calc")
mp_units = pytest.importorskip("metpy.units").units
xr = pytest.importorskip("xarray")
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


@pytest.fixture(scope="module")
def sounding_profile():
    col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]
    sounding_data = pd.read_fwf(
        get_test_data("20110522_OUN_12Z.txt", as_file_obj=False),
        skiprows=7,
        usecols=[0, 1, 2, 3, 6, 7],
        names=col_names,
    )
    sounding_data = sounding_data.dropna(
        subset=("temperature", "dewpoint", "direction", "speed"),
        how="all",
    ).reset_index(drop=True)

    pres = sounding_data["pressure"].values * mp_units.hPa
    temp = sounding_data["temperature"].values * mp_units.degC
    dewpoint = sounding_data["dewpoint"].values * mp_units.degC
    height = sounding_data["height"].values * mp_units.meter
    speed = sounding_data["speed"].values * mp_units.knots
    direction = sounding_data["direction"].values * mp_units.degrees
    u, v = mcalc.wind_components(speed, direction)
    return pres, temp, dewpoint, height, u, v


def test_uses_application_registry():
    assert mr_units is pint.get_application_registry()

    spec_humidity = mcalc.specific_humidity_from_dewpoint(
        1000 * mp_units.hPa,
        20 * mp_units.degC,
    )
    converted = (spec_humidity * 1000) * mp_units.g / mp_units.kg
    assert converted.units == mp_units.g / mp_units.kg


def test_sounding_index_forms(sounding_profile):
    pres, temp, dewpoint, _, _, _ = sounding_profile

    prof = mcalc.parcel_profile(pres, temp[0], dewpoint[0])
    cape, cin = mcalc.cape_cin(pres, temp, dewpoint, prof)
    lfc_p, lfc_t = mcalc.lfc(pres, temp, dewpoint, prof)
    el_p, el_t = mcalc.el(pres, temp, dewpoint, prof)

    assert cape.units == mp_units("J/kg")
    assert cin.units == mp_units("J/kg")
    assert lfc_p.units == mp_units.hPa
    assert lfc_t.units == mp_units.degC
    assert el_p.units == mp_units.hPa
    assert el_t.units == mp_units.degC

    assert mcalc.cross_totals(pres, temp, dewpoint).units == mp_units.delta_degC
    assert mcalc.vertical_totals(pres, temp).units == mp_units.delta_degC
    assert mcalc.total_totals(pres, temp, dewpoint).units == mp_units.delta_degC
    assert mcalc.k_index(pres, temp, dewpoint).units == mp_units.delta_degC


def test_layer_and_mixed_layer_forms(sounding_profile):
    pres, temp, dewpoint, _, _, _ = sounding_profile

    p_layer, t_layer, td_layer = mcalc.get_layer(
        pres,
        temp,
        dewpoint,
        bottom=700 * mp_units.hPa,
        depth=200 * mp_units.hPa,
        interpolate=True,
    )
    mixed_t, mixed_td = mcalc.mixed_layer(pres, temp, dewpoint, depth=50 * mp_units.hPa)
    parcel_p, parcel_t, parcel_td = mcalc.mixed_parcel(
        pres,
        temp,
        dewpoint,
        bottom=pres[0],
        depth=50 * mp_units.hPa,
        interpolate=True,
    )
    mu_p, mu_t, mu_td, idx = mcalc.most_unstable_parcel(
        pres,
        temp,
        dewpoint,
        depth=50 * mp_units.hPa,
    )

    assert p_layer.shape == t_layer.shape == td_layer.shape
    assert mixed_t.units == mp_units.degC
    assert mixed_td.units == mp_units.degC
    assert parcel_p.units == mp_units.hPa
    assert parcel_t.units == mp_units.degC
    assert parcel_td.units == mp_units.degC
    assert mu_p.units == mp_units.hPa
    assert mu_t.units == mp_units.degC
    assert mu_td.units == mp_units.degC
    assert isinstance(idx, int)


def test_moist_lapse_reference_pressure(sounding_profile):
    pres, temp, dewpoint, _, _, _ = sounding_profile

    p_layer, t_layer, td_layer = mcalc.get_layer(
        pres,
        temp,
        dewpoint,
        bottom=700 * mp_units.hPa,
        depth=200 * mp_units.hPa,
        interpolate=True,
    )
    start_p = p_layer[-1]
    start_wb = mcalc.wet_bulb_temperature(start_p, t_layer[-1], td_layer[-1])
    down_pressure = pres[pres >= start_p].to(mp_units.hPa)
    trace = mcalc.moist_lapse(down_pressure, start_wb, reference_pressure=start_p)

    assert trace.shape == down_pressure.shape
    assert trace.units == mp_units.degC


def test_grid_deltas_can_be_inferred_from_coords():
    lat2d = np.array(
        [
            [35.0, 35.0, 35.0, 35.0],
            [35.5, 35.5, 35.5, 35.5],
            [36.0, 36.0, 36.0, 36.0],
        ]
    )
    lon2d = np.array(
        [
            [-98.0, -97.5, -97.0, -96.5],
            [-98.0, -97.5, -97.0, -96.5],
            [-98.0, -97.5, -97.0, -96.5],
        ]
    )
    coords = {
        "latitude": (("y", "x"), lat2d),
        "longitude": (("y", "x"), lon2d),
    }
    u = xr.DataArray(np.full((3, 4), 10.0), dims=("y", "x"), coords=coords)
    v = xr.DataArray(np.full((3, 4), 5.0), dims=("y", "x"), coords=coords)
    scalar = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x"), coords=coords)

    vort = mcalc.vorticity(u, v)
    adv = mcalc.advection(scalar, u, v)

    assert vort.shape == (3, 4)
    assert adv.shape == (3, 4)
    assert vort.units == mp_units("1/s")
    assert adv.units == mp_units("1/s")
