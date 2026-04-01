from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


@pytest.fixture(scope="module")
def sounding():
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

    pressure = sounding_data["pressure"].values * units.hPa
    temperature = sounding_data["temperature"].values * units.degC
    dewpoint = sounding_data["dewpoint"].values * units.degC
    height = sounding_data["height"].values * units.meter
    speed = sounding_data["speed"].values * units.knots
    direction = sounding_data["direction"].values * units.degrees
    return {
        "p": pressure,
        "t": temperature,
        "td": dewpoint,
        "height": height,
        "speed": speed,
        "direction": direction,
    }


def _compare_quantity(actual, expected):
    actual_units = str(actual.units)
    expected_units = str(expected.units)
    if (
        "degree_Celsius" in actual_units and "delta_degree_Celsius" in expected_units
    ) or (
        "delta_degree_Celsius" in actual_units and "degree_Celsius" in expected_units
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
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        atol=atol,
        rtol=atol,
        equal_nan=True,
    )


def test_lfc_runtime_parity(sounding):
    actual = mcalc.lfc(sounding["p"], sounding["t"], sounding["td"])
    expected = mpcalc.lfc(sounding["p"], sounding["t"], sounding["td"])
    # Native LFC now stays off MetPy; the remaining drift is dominated by the
    # native LCL/moist-lapse profile insertion and stays well below 0.03 hPa/C.
    _assert_runtime_close(actual, expected, atol=3e-2)


def test_lfc_runtime_parity_with_parcel_profile(sounding):
    parcel_profile = mpcalc.parcel_profile(sounding["p"], sounding["t"][0], sounding["td"][0])
    actual = mcalc.lfc(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        parcel_temperature_profile=parcel_profile,
    )
    expected = mpcalc.lfc(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        parcel_temperature_profile=parcel_profile,
    )
    _assert_runtime_close(actual, expected, atol=1e-5)


def test_el_runtime_parity(sounding):
    actual = mcalc.el(sounding["p"], sounding["t"], sounding["td"])
    expected = mpcalc.el(sounding["p"], sounding["t"], sounding["td"])
    _assert_runtime_close(actual, expected, atol=3e-2)


def test_el_runtime_parity_with_parcel_profile(sounding):
    parcel_profile = mpcalc.parcel_profile(sounding["p"], sounding["t"][0], sounding["td"][0])
    actual = mcalc.el(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        parcel_temperature_profile=parcel_profile,
    )
    expected = mpcalc.el(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        parcel_temperature_profile=parcel_profile,
    )
    _assert_runtime_close(actual, expected, atol=1e-5)


def test_cape_cin_runtime_parity(sounding):
    parcel_profile = mpcalc.parcel_profile(sounding["p"], sounding["t"][0], sounding["td"][0])
    actual = mcalc.cape_cin(sounding["p"], sounding["t"], sounding["td"], parcel_profile)
    expected = mpcalc.cape_cin(sounding["p"], sounding["t"], sounding["td"], parcel_profile)
    _assert_runtime_close(actual, expected, atol=5e-3)


def test_downdraft_cape_runtime_parity(sounding):
    actual = mcalc.downdraft_cape(sounding["p"], sounding["t"], sounding["td"])
    expected = mpcalc.downdraft_cape(sounding["p"], sounding["t"], sounding["td"])
    _assert_runtime_close(actual, expected, atol=5e-1)


def test_sweat_index_runtime_parity(sounding):
    actual = mcalc.sweat_index(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        sounding["speed"],
        sounding["direction"],
        vertical_dim=0,
    )
    expected = mpcalc.sweat_index(
        sounding["p"],
        sounding["t"],
        sounding["td"],
        sounding["speed"],
        sounding["direction"],
        vertical_dim=0,
    )
    _assert_runtime_close(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("func_name", "atol"),
    [
        ("brunt_vaisala_frequency", 1e-8),
        ("brunt_vaisala_period", 1e-6),
        ("brunt_vaisala_frequency_squared", 1e-10),
    ],
)
def test_brunt_vaisala_runtime_parity(sounding, func_name, atol):
    theta = mpcalc.potential_temperature(sounding["p"], sounding["t"])
    actual = getattr(mcalc, func_name)(sounding["height"], theta, vertical_dim=0)
    expected = getattr(mpcalc, func_name)(sounding["height"], theta, vertical_dim=0)
    _assert_runtime_close(actual, expected, atol=atol)


def test_parcel_profile_with_lcl_runtime_parity(sounding):
    actual = mcalc.parcel_profile_with_lcl(sounding["p"], sounding["t"], sounding["td"])
    expected = mpcalc.parcel_profile_with_lcl(sounding["p"], sounding["t"], sounding["td"])
    _assert_runtime_close(actual, expected, atol=3e-2)


@pytest.mark.parametrize(
    ("func_name", "temperature", "atol"),
    [
        ("water_latent_heat_vaporization", 20 * units.degC, 1e-8),
        ("water_latent_heat_melting", -15 * units.degC, 1e-8),
        ("water_latent_heat_sublimation", -15 * units.degC, 1e-8),
    ],
)
def test_water_latent_heat_runtime_parity(func_name, temperature, atol):
    actual = getattr(mcalc, func_name)(temperature)
    expected = getattr(mpcalc, func_name)(temperature)
    _assert_runtime_close(actual, expected, atol=atol)
