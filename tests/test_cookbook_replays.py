from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


def _compare_quantity(actual, expected):
    actual_units = str(actual.units)
    expected_units = str(expected.units)
    if (
        "degree_Celsius" in actual_units
        and "delta_degree_Celsius" in expected_units
    ) or (
        "delta_degree_Celsius" in actual_units
        and "degree_Celsius" in expected_units
    ):
        return (
            np.asarray(actual.magnitude, dtype=np.float64),
            np.asarray(expected.magnitude, dtype=np.float64),
        )

    try:
        return (
            np.asarray(actual.to(expected.units).magnitude, dtype=np.float64),
            np.asarray(expected.magnitude, dtype=np.float64),
        )
    except Exception:
        return (
            np.asarray(actual.to_base_units().magnitude, dtype=np.float64),
            np.asarray(expected.to_base_units().magnitude, dtype=np.float64),
        )


def _assert_close(actual, expected, atol):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            key_atol = atol.get(key, 1e-6) if isinstance(atol, dict) else atol
            _assert_close(actual[key], expected[key], key_atol)
        return

    if isinstance(actual, xr.Dataset) and isinstance(expected, xr.Dataset):
        assert dict(actual.sizes) == dict(expected.sizes)
        assert set(actual.coords) == set(expected.coords)
        assert set(actual.data_vars) == set(expected.data_vars)
        for name in expected.coords:
            np.testing.assert_allclose(
                np.asarray(actual[name].values, dtype=np.float64),
                np.asarray(expected[name].values, dtype=np.float64),
                atol=atol,
                rtol=atol,
                equal_nan=True,
            )
        for name in expected.data_vars:
            _assert_close(actual[name].data, expected[name].data, atol)
        return

    if isinstance(actual, xr.DataArray) and isinstance(expected, xr.DataArray):
        assert actual.dims == expected.dims
        _assert_close(actual.data, expected.data, atol)
        return

    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_close(actual_item, expected_item, atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
    else:
        if hasattr(actual, "magnitude"):
            actual_arr = np.asarray(actual.magnitude, dtype=np.float64)
        else:
            actual_arr = np.asarray(actual, dtype=np.float64)
        if hasattr(expected, "magnitude"):
            expected_arr = np.asarray(expected.magnitude, dtype=np.float64)
        else:
            expected_arr = np.asarray(expected, dtype=np.float64)

    assert actual_arr.shape == expected_arr.shape
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


def _load_sounding():
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
    return {
        "pressure": sounding_data["pressure"].values * units.hPa,
        "height": sounding_data["height"].values * units.meter,
        "temperature": sounding_data["temperature"].values * units.degC,
        "dewpoint": sounding_data["dewpoint"].values * units.degC,
        "speed": sounding_data["speed"].values * units.knots,
        "direction": sounding_data["direction"].values * units.degrees,
    }


def _run_sounding_workflow(calc):
    sounding = _load_sounding()
    pressure = sounding["pressure"]
    height = sounding["height"]
    temperature = sounding["temperature"]
    dewpoint = sounding["dewpoint"]
    speed = sounding["speed"]
    direction = sounding["direction"]

    u, v = calc.wind_components(speed, direction)
    parcel = calc.parcel_profile(pressure, temperature[0], dewpoint[0])
    cape, cin = calc.cape_cin(pressure, temperature, dewpoint, parcel)
    lcl_pressure, _ = calc.lcl(pressure[0], temperature[0], dewpoint[0])
    lfc_pressure, _ = calc.lfc(pressure, temperature, dewpoint)
    el_pressure, _ = calc.el(pressure, temperature, dewpoint, parcel)
    mlcape, mlcin = calc.mixed_layer_cape_cin(pressure, temperature, dewpoint, depth=50 * units.hPa)
    mucape, mucin = calc.most_unstable_cape_cin(pressure, temperature, dewpoint)
    (storm_u, storm_v), _, _ = calc.bunkers_storm_motion(pressure, u, v, height)
    _, _, srh_1km = calc.storm_relative_helicity(
        height,
        u,
        v,
        depth=1 * units.km,
        storm_u=storm_u,
        storm_v=storm_v,
    )
    shear_u, shear_v = calc.bulk_shear(pressure, u, v, height=height, depth=6 * units.km)
    bulk_shear = calc.wind_speed(shear_u, shear_v)

    return {
        "cape": cape,
        "cin": cin,
        "lcl_pressure": lcl_pressure,
        "lfc_pressure": lfc_pressure,
        "el_pressure": el_pressure,
        "mlcape": mlcape,
        "mlcin": mlcin,
        "mucape": mucape,
        "mucin": mucin,
        "storm_motion": (storm_u, storm_v),
        "srh_1km": srh_1km,
        "bulk_shear": bulk_shear,
    }


def _build_grid_fields():
    lat = np.linspace(34.0, 36.0, 3)
    lon = np.linspace(-99.0, -95.0, 4)
    xx, yy = np.meshgrid(lon, lat)
    u = xr.DataArray(
        12.0 + 0.6 * (xx + 99.0) + 0.8 * (yy - 34.0),
        dims=("latitude", "longitude"),
        coords={
            "latitude": ("latitude", lat, {"units": "degrees_north"}),
            "longitude": ("longitude", lon, {"units": "degrees_east"}),
        },
        attrs={"units": "m/s"},
        name="u_wind",
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude").metpy.quantify()
    v = xr.DataArray(
        8.0 + 0.5 * (yy - 34.0) - 0.4 * (xx + 99.0),
        dims=("latitude", "longitude"),
        coords=u.coords,
        attrs={"units": "m/s"},
        name="v_wind",
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude").metpy.quantify()
    theta = xr.DataArray(
        300.0 + 1.8 * (xx + 99.0) - 1.1 * (yy - 34.0),
        dims=("latitude", "longitude"),
        coords=u.coords,
        attrs={"units": "K"},
        name="potential_temperature",
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude").metpy.quantify()
    return u, v, theta


def _run_grid_workflow(calc):
    u, v, theta = _build_grid_fields()
    absolute_vorticity = calc.vorticity(u, v)
    smoothed_vorticity = calc.smooth_n_point(absolute_vorticity, 9, 1)
    divergence = calc.divergence(u, v)
    frontogenesis = calc.frontogenesis(theta, u, v)
    return {
        "absolute_vorticity": absolute_vorticity,
        "smoothed_vorticity": smoothed_vorticity,
        "divergence": divergence,
        "frontogenesis": frontogenesis,
    }


def _run_xarray_workflow(calc):
    pressure = np.array([1000.0, 900.0, 800.0, 700.0]) * units.hPa
    temperature_profile = np.array([20.0, 15.0, 10.0, 5.0]) * units.degC
    dewpoint_profile = np.array([18.0, 10.0, 5.0, -2.0]) * units.degC
    levels = np.array([285.0, 295.0]) * units.kelvin

    temperature_2d = np.array(
        [
            [280.0, 281.0],
            [290.0, 291.0],
            [300.0, 301.0],
            [310.0, 311.0],
        ]
    )
    u_field = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    temperature_da = xr.DataArray(
        temperature_2d,
        dims=("isobaric", "x"),
        coords={
            "isobaric": ("isobaric", pressure.magnitude, {"units": "hPa"}),
            "x": [0, 1],
        },
        attrs={"units": "K"},
        name="temperature",
    ).metpy.quantify()
    u_da = xr.DataArray(
        u_field,
        dims=("isobaric", "x"),
        coords={
            "isobaric": ("isobaric", pressure.magnitude, {"units": "hPa"}),
            "x": [0, 1],
        },
        attrs={"units": "m/s"},
        name="u_wind",
    ).metpy.quantify()
    return {
        "parcel_profile_with_lcl": calc.parcel_profile_with_lcl_as_dataset(
            pressure,
            temperature_profile,
            dewpoint_profile,
        ),
        "isentropic_interpolation": calc.isentropic_interpolation_as_dataset(
            levels,
            temperature_da,
            u_da,
        ),
    }


def test_cookbook_sounding_workflow_replay():
    _assert_close(
        _run_sounding_workflow(mcalc),
        _run_sounding_workflow(mpcalc),
        {
            "cape": 1.0,
            "cin": 1.0,
            "lcl_pressure": 5e-2,
            "lfc_pressure": 5e-2,
            "el_pressure": 5e-2,
            "mlcape": 10.0,
            "mlcin": 10.0,
            "mucape": 10.0,
            "mucin": 10.0,
            "storm_motion": 5e-2,
            "srh_1km": 5e-1,
            "bulk_shear": 5e-2,
        },
    )


def test_cookbook_grid_workflow_replay():
    _assert_close(_run_grid_workflow(mcalc), _run_grid_workflow(mpcalc), 2e-6)


def test_cookbook_xarray_workflow_replay():
    _assert_close(
        _run_xarray_workflow(mcalc),
        _run_xarray_workflow(mpcalc),
        {
            "parcel_profile_with_lcl": 3e-2,
            "isentropic_interpolation": 1e-6,
        },
    )
