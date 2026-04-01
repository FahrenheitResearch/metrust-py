from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


@pytest.fixture(scope="module")
def wind_profile_context():
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
    u, v = mpcalc.wind_components(speed, direction)
    theta = mpcalc.potential_temperature(pressure, temperature)

    return {
        "pressure": pressure,
        "temperature": temperature,
        "dewpoint": dewpoint,
        "height": height,
        "speed": speed,
        "direction": direction,
        "u": u,
        "v": v,
        "theta": theta,
    }


@pytest.fixture(scope="module")
def turbulence_context():
    rng = np.random.default_rng(0)
    u = (8 + rng.normal(0, 1.5, (3, 256))) * units("m/s")
    v = (2 + rng.normal(0, 1.0, (3, 256))) * units("m/s")
    w = rng.normal(0, 0.35, (3, 256)) * units("m/s")
    up = u - u.mean(axis=1, keepdims=True)
    vp = v - v.mean(axis=1, keepdims=True)
    wp = w - w.mean(axis=1, keepdims=True)
    return {"u": u, "v": v, "w": w, "up": up, "vp": vp, "wp": wp}


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


def _assert_runtime_close(actual, expected, atol=1e-10):
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_runtime_close(actual_item, expected_item, atol=atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
    else:
        actual_arr = np.asarray(actual, dtype=np.float64)
        expected_arr = np.asarray(expected, dtype=np.float64)

    assert actual_arr.shape == expected_arr.shape
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


def test_wet_bulb_potential_temperature_runtime_parity(wind_profile_context):
    actual = mcalc.wet_bulb_potential_temperature(
        wind_profile_context["pressure"][:5],
        wind_profile_context["temperature"][:5],
        wind_profile_context["dewpoint"][:5],
    )
    expected = mpcalc.wet_bulb_potential_temperature(
        wind_profile_context["pressure"][:5],
        wind_profile_context["temperature"][:5],
        wind_profile_context["dewpoint"][:5],
    )
    _assert_runtime_close(actual, expected)


def test_mixed_parcel_runtime_parity(wind_profile_context):
    kwargs = {"depth": 150 * units.hPa}
    actual = mcalc.mixed_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    expected = mpcalc.mixed_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    _assert_runtime_close(actual, expected)


def test_most_unstable_parcel_runtime_parity(wind_profile_context):
    kwargs = {"depth": 300 * units.hPa}
    actual = mcalc.most_unstable_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    expected = mpcalc.most_unstable_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    _assert_runtime_close(actual, expected)


def test_psychrometric_vapor_pressure_wet_runtime_parity():
    kwargs = {"psychrometer_coefficient": 7e-4 / units.kelvin}
    actual = mcalc.psychrometric_vapor_pressure_wet(
        958 * units.hPa,
        25 * units.degC,
        12 * units.degC,
        **kwargs,
    )
    expected = mpcalc.psychrometric_vapor_pressure_wet(
        958 * units.hPa,
        25 * units.degC,
        12 * units.degC,
        **kwargs,
    )
    _assert_runtime_close(actual, expected)


def test_storm_relative_helicity_runtime_parity(wind_profile_context):
    kwargs = {
        "bottom": 500 * units.meter,
        "storm_u": 5 * units("m/s"),
        "storm_v": -2 * units("m/s"),
    }
    actual = mcalc.storm_relative_helicity(
        wind_profile_context["height"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        3 * units.kilometer,
        **kwargs,
    )
    expected = mpcalc.storm_relative_helicity(
        wind_profile_context["height"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        3 * units.kilometer,
        **kwargs,
    )
    _assert_runtime_close(actual, expected)


def test_corfidi_storm_motion_runtime_parity(wind_profile_context):
    kwargs = {
        "u_llj": wind_profile_context["u"][1],
        "v_llj": wind_profile_context["v"][1],
    }
    actual = mcalc.corfidi_storm_motion(
        wind_profile_context["pressure"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        **kwargs,
    )
    expected = mpcalc.corfidi_storm_motion(
        wind_profile_context["pressure"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        **kwargs,
    )
    _assert_runtime_close(actual, expected)


def test_friction_velocity_runtime_parity(turbulence_context):
    actual = mcalc.friction_velocity(
        turbulence_context["u"],
        turbulence_context["w"],
        v=turbulence_context["v"],
        axis=1,
    )
    expected = mpcalc.friction_velocity(
        turbulence_context["u"],
        turbulence_context["w"],
        v=turbulence_context["v"],
        axis=1,
    )
    _assert_runtime_close(actual, expected)


def test_tke_runtime_parity(turbulence_context):
    actual = mcalc.tke(
        turbulence_context["up"],
        turbulence_context["vp"],
        turbulence_context["wp"],
        perturbation=True,
        axis=1,
    )
    expected = mpcalc.tke(
        turbulence_context["up"],
        turbulence_context["vp"],
        turbulence_context["wp"],
        perturbation=True,
        axis=1,
    )
    _assert_runtime_close(actual, expected)


def test_gradient_richardson_number_runtime_parity(wind_profile_context):
    height_2d = np.column_stack(
        [wind_profile_context["height"][:20].magnitude, wind_profile_context["height"][:20].magnitude]
    ) * wind_profile_context["height"].units
    theta_2d = np.column_stack(
        [wind_profile_context["theta"][:20].magnitude, wind_profile_context["theta"][:20].magnitude + 1.0]
    ) * wind_profile_context["theta"].units
    u_2d = np.column_stack(
        [wind_profile_context["u"][:20].magnitude, wind_profile_context["u"][:20].magnitude]
    ) * wind_profile_context["u"].units
    v_2d = np.column_stack(
        [wind_profile_context["v"][:20].magnitude, wind_profile_context["v"][:20].magnitude]
    ) * wind_profile_context["v"].units

    actual = mcalc.gradient_richardson_number(
        height_2d,
        theta_2d,
        u_2d,
        v_2d,
        vertical_dim=0,
    )
    expected = mpcalc.gradient_richardson_number(
        height_2d,
        theta_2d,
        u_2d,
        v_2d,
        vertical_dim=0,
    )
    _assert_runtime_close(actual, expected)
