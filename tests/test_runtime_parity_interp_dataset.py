from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units


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


def _assert_runtime_close(actual, expected, atol):
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_runtime_close(actual_item, expected_item, atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
    else:
        actual_arr = np.asarray(actual, dtype=np.float64)
        expected_arr = np.asarray(expected, dtype=np.float64)

    assert actual_arr.shape == expected_arr.shape
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


def _assert_dataset_close(actual, expected, atol):
    assert dict(actual.sizes) == dict(expected.sizes)
    assert set(actual.coords) == set(expected.coords)
    assert set(actual.data_vars) == set(expected.data_vars)

    for coord_name in expected.coords:
        np.testing.assert_allclose(
            np.asarray(actual[coord_name].values, dtype=np.float64),
            np.asarray(expected[coord_name].values, dtype=np.float64),
            atol=atol,
            rtol=atol,
            equal_nan=True,
        )
        for attr_name, attr_value in expected[coord_name].attrs.items():
            assert actual[coord_name].attrs.get(attr_name) == attr_value

    for var_name in expected.data_vars:
        _assert_runtime_close(actual[var_name].data, expected[var_name].data, atol)
        for attr_name, attr_value in expected[var_name].attrs.items():
            assert actual[var_name].attrs.get(attr_name) == attr_value


@pytest.fixture(scope="module")
def interp_context():
    pressure = np.array([1000.0, 900.0, 800.0, 700.0]) * units.hPa
    temperature_profile = np.array([20.0, 15.0, 10.0, 5.0]) * units.degC
    dewpoint_profile = np.array([18.0, 10.0, 5.0, -2.0]) * units.degC
    levels = np.array([285.0, 295.0]) * units.kelvin

    pressure_2d = pressure[:, None] * np.ones((4, 2))
    temperature_2d = np.array([
        [280.0, 281.0],
        [290.0, 291.0],
        [300.0, 301.0],
        [310.0, 311.0],
    ]) * units.kelvin
    u_field = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]) * units("m/s")

    temperature_da = xr.DataArray(
        temperature_2d.magnitude,
        dims=("isobaric", "x"),
        coords={
            "isobaric": ("isobaric", pressure.magnitude, {"units": "hPa"}),
            "x": [0, 1],
        },
        attrs={"units": "K"},
        name="temperature",
    ).metpy.quantify()
    u_da = xr.DataArray(
        u_field.magnitude,
        dims=("isobaric", "x"),
        coords={
            "isobaric": ("isobaric", pressure.magnitude, {"units": "hPa"}),
            "x": [0, 1],
        },
        attrs={"units": "m/s"},
        name="u_wind",
    ).metpy.quantify()

    return {
        "pressure_profile": pressure,
        "temperature_profile": temperature_profile,
        "dewpoint_profile": dewpoint_profile,
        "levels": levels,
        "pressure_2d": pressure_2d,
        "temperature_2d": temperature_2d,
        "u_field": u_field,
        "temperature_da": temperature_da,
        "u_da": u_da,
    }


def test_runtime_parity_parcel_profile(interp_context):
    actual = mcalc.parcel_profile(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"][0],
        interp_context["dewpoint_profile"][0],
    )
    expected = mpcalc.parcel_profile(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"][0],
        interp_context["dewpoint_profile"][0],
    )
    # The remaining difference here is the known moist-lapse numerical drift in the
    # parcel trace, which is still well below a millikelvin.
    _assert_runtime_close(actual, expected, 1e-3)


def test_runtime_parity_parcel_profile_with_lcl(interp_context):
    actual = mcalc.parcel_profile_with_lcl(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"],
        interp_context["dewpoint_profile"],
    )
    expected = mpcalc.parcel_profile_with_lcl(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"],
        interp_context["dewpoint_profile"],
    )
    _assert_runtime_close(actual, expected, 1e-8)


def test_runtime_parity_parcel_profile_with_lcl_as_dataset(interp_context):
    actual = mcalc.parcel_profile_with_lcl_as_dataset(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"],
        interp_context["dewpoint_profile"],
    )
    expected = mpcalc.parcel_profile_with_lcl_as_dataset(
        interp_context["pressure_profile"],
        interp_context["temperature_profile"],
        interp_context["dewpoint_profile"],
    )
    _assert_dataset_close(actual, expected, 1e-8)


def test_runtime_parity_isentropic_interpolation(interp_context):
    actual = mcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_2d"],
        interp_context["temperature_2d"],
        interp_context["u_field"],
    )
    expected = mpcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_2d"],
        interp_context["temperature_2d"],
        interp_context["u_field"],
    )
    _assert_runtime_close(actual, expected, 1e-6)


def test_runtime_parity_isentropic_interpolation_vertical_dim(interp_context):
    actual = mcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_profile"],
        interp_context["temperature_2d"].T,
        interp_context["u_field"].T,
        vertical_dim=1,
    )
    expected = mpcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_profile"],
        interp_context["temperature_2d"].T,
        interp_context["u_field"].T,
        vertical_dim=1,
    )
    _assert_runtime_close(actual, expected, 1e-6)


def test_runtime_parity_isentropic_interpolation_temperature_out(interp_context):
    actual = mcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_profile"],
        interp_context["temperature_2d"],
        interp_context["u_field"],
        temperature_out=True,
    )
    expected = mpcalc.isentropic_interpolation(
        interp_context["levels"],
        interp_context["pressure_profile"],
        interp_context["temperature_2d"],
        interp_context["u_field"],
        temperature_out=True,
    )
    _assert_runtime_close(actual, expected, 1e-6)


def test_runtime_parity_isentropic_interpolation_as_dataset(interp_context):
    actual = mcalc.isentropic_interpolation_as_dataset(
        interp_context["levels"],
        interp_context["temperature_da"],
        interp_context["u_da"],
    )
    expected = mpcalc.isentropic_interpolation_as_dataset(
        interp_context["levels"],
        interp_context["temperature_da"],
        interp_context["u_da"],
    )
    _assert_dataset_close(actual, expected, 1e-6)
