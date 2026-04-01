from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


def _compare_quantity(actual, expected):
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
    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
    else:
        actual_arr = np.asarray(actual)
        expected_arr = np.asarray(expected)
    assert actual_arr.shape == expected_arr.shape
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def sounding_context():
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
        "temperature": sounding_data["temperature"].values * units.degC,
        "dewpoint": sounding_data["dewpoint"].values * units.degC,
    }


def test_runtime_parity_mixed_layer_cape_cin(sounding_context):
    actual = mcalc.mixed_layer_cape_cin(
        sounding_context["pressure"],
        sounding_context["temperature"],
        sounding_context["dewpoint"],
        depth=50 * units.hPa,
    )
    expected = mpcalc.mixed_layer_cape_cin(
        sounding_context["pressure"],
        sounding_context["temperature"],
        sounding_context["dewpoint"],
        depth=50 * units.hPa,
    )
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    _assert_runtime_close(actual[0], expected[0], 10.0)
    _assert_runtime_close(actual[1], expected[1], 10.0)


def test_runtime_parity_reduce_point_density():
    points = np.array([0.0, 0.5, 2.0, 2.4, 5.0]) * units.meter
    priority = np.array([0.2, 0.9, 0.1, 0.8, 0.5])
    actual = mcalc.reduce_point_density(points, 1.0 * units.meter, priority=priority)
    expected = mpcalc.reduce_point_density(points, 1.0 * units.meter, priority=priority)
    np.testing.assert_array_equal(actual, expected)


def test_runtime_parity_get_perturbation():
    values = np.array([290.0, 292.0, 288.0, 294.0]) * units.kelvin
    actual = mcalc.get_perturbation(values)
    expected = mpcalc.get_perturbation(values)
    _assert_runtime_close(actual, expected, 1e-12)


def test_invalid_sounding_error_is_metpy_compatible():
    assert issubclass(mcalc.InvalidSoundingError, Exception)
    assert mcalc.InvalidSoundingError.__name__ == mpcalc.InvalidSoundingError.__name__
