from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


@pytest.fixture(scope="module")
def thermo_context():
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
    height = sounding_data["height"].values * units.meter
    temperature = sounding_data["temperature"].values * units.degC
    dewpoint = sounding_data["dewpoint"].values * units.degC

    rh_profile = np.linspace(0.55, 0.9, 10) * units.dimensionless
    parcel_profile = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0])

    return {
        "p": pressure,
        "h": height,
        "t": temperature,
        "td": dewpoint,
        "rh": rh_profile,
        "parcel_profile": parcel_profile,
    }


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
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        atol=atol,
        rtol=atol,
        equal_nan=True,
    )


def _case_relative_humidity_wet_psychrometric(ctx):
    return (900 * units.hPa, 20 * units.degC, 17 * units.degC), {}


def _case_weighted_continuous_average(ctx):
    return (ctx["p"], ctx["t"]), {"bottom": 900 * units.hPa, "depth": 150 * units.hPa}


def _case_add_height_to_pressure(ctx):
    return (850 * units.hPa, 150 * units.m), {}


def _case_add_pressure_to_height(ctx):
    return (1500 * units.m, 25 * units.hPa), {}


def _case_thickness_hydrostatic_from_relative_humidity(ctx):
    return (ctx["p"][:10], ctx["t"][:10], ctx["rh"]), {}


def _case_ccl(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_density(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.01 * units("kg/kg")), {}


def _case_dry_static_energy(ctx):
    return (1500 * units.m, 290 * units.kelvin), {}


def _case_geopotential_to_height(ctx):
    return (15000 * units("m^2/s^2"),), {}


def _case_get_layer_heights(ctx):
    return (ctx["h"], 2500 * units.m, ctx["t"]), {"bottom": 500 * units.m}


def _case_height_to_geopotential(ctx):
    return (1500 * units.m,), {}


def _case_moist_air_gas_constant(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_air_specific_heat_pressure(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_air_poisson_exponent(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_static_energy(ctx):
    return (1500 * units.m, 290 * units.kelvin, 0.008 * units("kg/kg")), {}


def _case_montgomery_streamfunction(ctx):
    return (1500 * units.m, 290 * units.kelvin), {}


def _case_most_unstable_cape_cin(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_saturation_equivalent_potential_temperature(ctx):
    return (850 * units.hPa, 20 * units.degC), {}


def _case_scale_height(ctx):
    return (20 * units.degC, -50 * units.degC), {}


def _case_specific_humidity_from_dewpoint(ctx):
    return (900 * units.hPa, 15 * units.degC), {}


def _case_surface_based_cape_cin(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_lifted_index(ctx):
    return (ctx["p"], ctx["t"], ctx["parcel_profile"]), {}


THERMO_LAYER_CASES = [
    pytest.param(
        "relative_humidity_wet_psychrometric",
        _case_relative_humidity_wet_psychrometric,
        1e-12,
        id="relative_humidity_wet_psychrometric",
    ),
    pytest.param(
        "weighted_continuous_average",
        _case_weighted_continuous_average,
        1e-12,
        id="weighted_continuous_average",
    ),
    pytest.param("add_height_to_pressure", _case_add_height_to_pressure, 1e-3, id="add_height_to_pressure"),
    pytest.param("add_pressure_to_height", _case_add_pressure_to_height, 1e-5, id="add_pressure_to_height"),
    pytest.param(
        "thickness_hydrostatic_from_relative_humidity",
        _case_thickness_hydrostatic_from_relative_humidity,
        0.1,
        id="thickness_hydrostatic_from_relative_humidity",
    ),
    pytest.param("ccl", _case_ccl, 1e-9, id="ccl"),
    pytest.param("density", _case_density, 1e-12, id="density"),
    pytest.param("dry_static_energy", _case_dry_static_energy, 1e-12, id="dry_static_energy"),
    pytest.param("geopotential_to_height", _case_geopotential_to_height, 1e-12, id="geopotential_to_height"),
    pytest.param("get_layer_heights", _case_get_layer_heights, 1e-12, id="get_layer_heights"),
    pytest.param("height_to_geopotential", _case_height_to_geopotential, 1e-12, id="height_to_geopotential"),
    pytest.param("moist_air_gas_constant", _case_moist_air_gas_constant, 1e-12, id="moist_air_gas_constant"),
    pytest.param(
        "moist_air_specific_heat_pressure",
        _case_moist_air_specific_heat_pressure,
        1e-12,
        id="moist_air_specific_heat_pressure",
    ),
    pytest.param(
        "moist_air_poisson_exponent",
        _case_moist_air_poisson_exponent,
        1e-12,
        id="moist_air_poisson_exponent",
    ),
    pytest.param("moist_static_energy", _case_moist_static_energy, 1e-12, id="moist_static_energy"),
    pytest.param("montgomery_streamfunction", _case_montgomery_streamfunction, 1e-12, id="montgomery_streamfunction"),
    pytest.param("most_unstable_cape_cin", _case_most_unstable_cape_cin, 10.0, id="most_unstable_cape_cin"),
    pytest.param(
        "saturation_equivalent_potential_temperature",
        _case_saturation_equivalent_potential_temperature,
        1e-12,
        id="saturation_equivalent_potential_temperature",
    ),
    pytest.param("scale_height", _case_scale_height, 1e-12, id="scale_height"),
    pytest.param(
        "specific_humidity_from_dewpoint",
        _case_specific_humidity_from_dewpoint,
        1e-12,
        id="specific_humidity_from_dewpoint",
    ),
    pytest.param("surface_based_cape_cin", _case_surface_based_cape_cin, 5.0, id="surface_based_cape_cin"),
    pytest.param("lifted_index", _case_lifted_index, 1e-12, id="lifted_index"),
]


@pytest.mark.parametrize(("name", "builder", "atol"), THERMO_LAYER_CASES)
def test_runtime_parity_thermo_layers(thermo_context, name, builder, atol):
    args, kwargs = builder(thermo_context)
    actual = getattr(mcalc, name)(*args, **kwargs)
    expected = getattr(mpcalc, name)(*args, **kwargs)
    _assert_runtime_close(actual, expected, atol)
