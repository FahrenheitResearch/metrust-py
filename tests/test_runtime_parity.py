from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units
get_test_data = pytest.importorskip("metpy.cbook").get_test_data


@pytest.fixture(scope="module")
def parity_context():
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

    x = np.linspace(0, 400000, 5) * units.m
    y = np.linspace(0, 300000, 4) * units.m
    dx = np.diff(x) * np.ones((4, 4))
    dy = np.diff(y)[:, None] * np.ones((3, 5))
    xx, yy = np.meshgrid(x.m, y.m)
    u_grid = ((xx / 100000.0) + 2.0 * (yy / 100000.0)) * units("m/s")
    v_grid = ((yy / 100000.0) - (xx / 150000.0)) * units("m/s")
    theta = (300.0 + 0.5 * (xx / 100000.0) - 0.75 * (yy / 100000.0)) * units.kelvin
    height_field = (5600.0 + 10.0 * (xx / 100000.0) - 20.0 * (yy / 100000.0)) * units.meter
    scalar = (280.0 + 2.0 * (xx / 100000.0) + 1.5 * (yy / 100000.0)) * units.kelvin
    latitude = np.full_like(xx, 35.0) * units.degrees

    return {
        "p": pressure,
        "t": temperature,
        "td": dewpoint,
        "height": height,
        "speed": speed,
        "direction": direction,
        "u": u,
        "v": v,
        "dx": dx,
        "dy": dy,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "theta": theta,
        "height_field": height_field,
        "scalar": scalar,
        "latitude": latitude,
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
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


def _case_potential_temperature(ctx):
    return (850 * units.hPa, 20 * units.degC), {}


def _case_equivalent_potential_temperature(ctx):
    return (850 * units.hPa, 20 * units.degC, 15 * units.degC), {}


def _case_saturation_vapor_pressure(ctx):
    return (20 * units.degC,), {}


def _case_saturation_mixing_ratio(ctx):
    return (850 * units.hPa, 10 * units.degC), {}


def _case_wet_bulb_temperature(ctx):
    return (900 * units.hPa, 22 * units.degC, 17 * units.degC), {}


def _case_lcl(ctx):
    return (950 * units.hPa, 21 * units.degC, 18 * units.degC), {}


def _case_dewpoint_from_relative_humidity(ctx):
    return (20 * units.degC, 0.7), {}


def _case_relative_humidity_from_dewpoint(ctx):
    return (20 * units.degC, 15 * units.degC), {}


def _case_virtual_temperature(ctx):
    return (300 * units.kelvin, 0.01 * units.dimensionless), {}


def _case_virtual_temperature_from_dewpoint(ctx):
    return (900 * units.hPa, 20 * units.degC, 15 * units.degC), {}


def _case_mixing_ratio(ctx):
    return (10 * units.hPa, 900 * units.hPa), {}


def _case_showalter_index(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_k_index(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_cross_totals(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_vertical_totals(ctx):
    return (ctx["p"], ctx["t"]), {}


def _case_total_totals_index(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_precipitable_water(ctx):
    return (ctx["p"], ctx["td"]), {"bottom": 950 * units.hPa, "top": 700 * units.hPa}


def _case_vapor_pressure(ctx):
    return (900 * units.hPa, 0.01 * units.dimensionless), {}


def _case_specific_humidity_from_mixing_ratio(ctx):
    return (0.01 * units.dimensionless,), {}


def _case_thickness_hydrostatic(ctx):
    return (ctx["p"][:10], ctx["t"][:10]), {}


def _case_dewpoint(ctx):
    return (10 * units.hPa,), {}


def _case_dewpoint_from_specific_humidity(ctx):
    return (900 * units.hPa, 0.008 * units.dimensionless), {}


def _case_dry_lapse(ctx):
    return (ctx["p"][:8], ctx["t"][0]), {}


def _case_exner_function(ctx):
    return (900 * units.hPa,), {}


def _case_get_layer(ctx):
    return (
        ctx["p"],
        ctx["t"],
        ctx["td"],
    ), {"bottom": 900 * units.hPa, "depth": 150 * units.hPa, "interpolate": True}


def _case_mean_pressure_weighted(ctx):
    return (ctx["p"], ctx["t"]), {"bottom": 900 * units.hPa, "depth": 150 * units.hPa}


def _case_mixed_layer(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {"depth": 100 * units.hPa}


def _case_mixing_ratio_from_relative_humidity(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.7), {}


def _case_mixing_ratio_from_specific_humidity(ctx):
    return (0.008 * units.dimensionless,), {}


def _case_moist_lapse(ctx):
    return (ctx["p"][:8], ctx["t"][0]), {}


def _case_parcel_profile(ctx):
    return (ctx["p"][:15], ctx["t"][0], ctx["td"][0]), {}


def _case_relative_humidity_from_mixing_ratio(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.01 * units.dimensionless), {}


def _case_relative_humidity_from_specific_humidity(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.008 * units.dimensionless), {}


def _case_static_stability(ctx):
    return (ctx["p"][:12], ctx["t"][:12]), {}


def _case_temperature_from_potential_temperature(ctx):
    return (900 * units.hPa, 300 * units.kelvin), {}


def _case_vertical_velocity(ctx):
    return (-0.2 * units("Pa/s"), 700 * units.hPa, -5 * units.degC), {}


def _case_vertical_velocity_pressure(ctx):
    return (0.5 * units("m/s"), 700 * units.hPa, -5 * units.degC), {}


def _case_virtual_potential_temperature(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.01 * units.dimensionless), {}


def _case_wind_speed(ctx):
    return (ctx["u"][:5], ctx["v"][:5]), {}


def _case_wind_direction(ctx):
    return (ctx["u"][:5], ctx["v"][:5]), {}


def _case_wind_components(ctx):
    return (ctx["speed"][:5], ctx["direction"][:5]), {}


def _case_bulk_shear(ctx):
    return (ctx["p"], ctx["u"], ctx["v"]), {"height": ctx["height"], "depth": 6000 * units.m}


def _case_bunkers_storm_motion(ctx):
    return (ctx["p"], ctx["u"], ctx["v"], ctx["height"]), {}


def _case_divergence(ctx):
    return (ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_vorticity(ctx):
    return (ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_advection(ctx):
    return (ctx["scalar"], ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_frontogenesis(ctx):
    return (ctx["theta"], ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_geostrophic_wind(ctx):
    return (ctx["height_field"],), {"dx": ctx["dx"], "dy": ctx["dy"], "latitude": 35 * units.degrees}


def _case_q_vector(ctx):
    return (ctx["u_grid"], ctx["v_grid"], ctx["scalar"], 700 * units.hPa), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_shearing_deformation(ctx):
    return (ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_stretching_deformation(ctx):
    return (ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_total_deformation(ctx):
    return (ctx["u_grid"], ctx["v_grid"]), {"dx": ctx["dx"], "dy": ctx["dy"]}


def _case_first_derivative(ctx):
    return (ctx["scalar"],), {"axis": 1, "delta": ctx["dx"]}


def _case_second_derivative(ctx):
    return (ctx["scalar"],), {"axis": 1, "delta": ctx["dx"]}


def _case_significant_tornado(ctx):
    return (
        1500 * units("J/kg"),
        800 * units.m,
        150 * units("m^2/s^2"),
        20 * units("m/s"),
    ), {}


def _case_supercell_composite(ctx):
    return (
        2500 * units("J/kg"),
        200 * units("m^2/s^2"),
        25 * units("m/s"),
    ), {}


def _case_critical_angle(ctx):
    return (
        np.array([1000, 925, 850, 700, 500]) * units.hPa,
        np.array([10, 20, 25, 30, 35]) * units.knots,
        np.array([5, 10, 15, 20, 25]) * units.knots,
        np.array([0, 750, 1500, 3000, 5500]) * units.m,
        15 * units.knots,
        10 * units.knots,
    ), {}


def _case_heat_index(ctx):
    return (32 * units.degC, 70 * units.percent), {}


def _case_windchill(ctx):
    return (0 * units.degC, 20 * units("mile/hour")), {}


def _case_apparent_temperature(ctx):
    return (30 * units.degC, 70 * units.percent, 10 * units("m/s")), {}


RUNTIME_CASES = [
    pytest.param("potential_temperature", _case_potential_temperature, 1e-9, id="potential_temperature"),
    pytest.param("equivalent_potential_temperature", _case_equivalent_potential_temperature, 1e-6, id="equivalent_potential_temperature"),
    pytest.param("saturation_vapor_pressure", _case_saturation_vapor_pressure, 1e-6, id="saturation_vapor_pressure"),
    pytest.param("saturation_mixing_ratio", _case_saturation_mixing_ratio, 1e-8, id="saturation_mixing_ratio"),
    pytest.param("wet_bulb_temperature", _case_wet_bulb_temperature, 5e-2, id="wet_bulb_temperature"),
    pytest.param("lcl", _case_lcl, 5e-2, id="lcl"),
    pytest.param("dewpoint_from_relative_humidity", _case_dewpoint_from_relative_humidity, 1e-9, id="dewpoint_from_relative_humidity"),
    pytest.param("relative_humidity_from_dewpoint", _case_relative_humidity_from_dewpoint, 1e-9, id="relative_humidity_from_dewpoint"),
    pytest.param("virtual_temperature", _case_virtual_temperature, 1e-9, id="virtual_temperature"),
    pytest.param("virtual_temperature_from_dewpoint", _case_virtual_temperature_from_dewpoint, 5e-2, id="virtual_temperature_from_dewpoint"),
    pytest.param("mixing_ratio", _case_mixing_ratio, 1e-9, id="mixing_ratio"),
    pytest.param("showalter_index", _case_showalter_index, 2e-2, id="showalter_index"),
    pytest.param("k_index", _case_k_index, 1e-6, id="k_index"),
    pytest.param("cross_totals", _case_cross_totals, 1e-9, id="cross_totals"),
    pytest.param("vertical_totals", _case_vertical_totals, 1e-9, id="vertical_totals"),
    pytest.param("total_totals_index", _case_total_totals_index, 1e-9, id="total_totals_index"),
    pytest.param("precipitable_water", _case_precipitable_water, 1e-3, id="precipitable_water"),
    pytest.param("vapor_pressure", _case_vapor_pressure, 1e-9, id="vapor_pressure"),
    pytest.param("specific_humidity_from_mixing_ratio", _case_specific_humidity_from_mixing_ratio, 1e-9, id="specific_humidity_from_mixing_ratio"),
    pytest.param("thickness_hydrostatic", _case_thickness_hydrostatic, 1e-2, id="thickness_hydrostatic"),
    pytest.param("dewpoint", _case_dewpoint, 1e-6, id="dewpoint"),
    pytest.param("dewpoint_from_specific_humidity", _case_dewpoint_from_specific_humidity, 1e-6, id="dewpoint_from_specific_humidity"),
    pytest.param("dry_lapse", _case_dry_lapse, 1e-6, id="dry_lapse"),
    pytest.param("exner_function", _case_exner_function, 1e-9, id="exner_function"),
    pytest.param("get_layer", _case_get_layer, 1e-9, id="get_layer"),
    pytest.param("mean_pressure_weighted", _case_mean_pressure_weighted, 1e-9, id="mean_pressure_weighted"),
    pytest.param("mixed_layer", _case_mixed_layer, 1e-9, id="mixed_layer"),
    pytest.param("mixing_ratio_from_relative_humidity", _case_mixing_ratio_from_relative_humidity, 1e-4, id="mixing_ratio_from_relative_humidity"),
    pytest.param("mixing_ratio_from_specific_humidity", _case_mixing_ratio_from_specific_humidity, 1e-9, id="mixing_ratio_from_specific_humidity"),
    pytest.param("moist_lapse", _case_moist_lapse, 1e-6, id="moist_lapse"),
    pytest.param("parcel_profile", _case_parcel_profile, 2e-3, id="parcel_profile"),
    pytest.param("relative_humidity_from_mixing_ratio", _case_relative_humidity_from_mixing_ratio, 7e-3, id="relative_humidity_from_mixing_ratio"),
    pytest.param("relative_humidity_from_specific_humidity", _case_relative_humidity_from_specific_humidity, 7e-3, id="relative_humidity_from_specific_humidity"),
    pytest.param("static_stability", _case_static_stability, 2e-6, id="static_stability"),
    pytest.param("temperature_from_potential_temperature", _case_temperature_from_potential_temperature, 1e-9, id="temperature_from_potential_temperature"),
    pytest.param("vertical_velocity", _case_vertical_velocity, 1e-9, id="vertical_velocity"),
    pytest.param("vertical_velocity_pressure", _case_vertical_velocity_pressure, 1e-9, id="vertical_velocity_pressure"),
    pytest.param("virtual_potential_temperature", _case_virtual_potential_temperature, 3e-2, id="virtual_potential_temperature"),
    pytest.param("wind_speed", _case_wind_speed, 1e-9, id="wind_speed"),
    pytest.param("wind_direction", _case_wind_direction, 1e-9, id="wind_direction"),
    pytest.param("wind_components", _case_wind_components, 1e-9, id="wind_components"),
    pytest.param("bulk_shear", _case_bulk_shear, 3e-2, id="bulk_shear"),
    pytest.param("bunkers_storm_motion", _case_bunkers_storm_motion, 3e-2, id="bunkers_storm_motion"),
    pytest.param("divergence", _case_divergence, 1e-9, id="divergence"),
    pytest.param("vorticity", _case_vorticity, 1e-9, id="vorticity"),
    pytest.param("advection", _case_advection, 1e-9, id="advection"),
    pytest.param("frontogenesis", _case_frontogenesis, 1e-9, id="frontogenesis"),
    pytest.param("geostrophic_wind", _case_geostrophic_wind, 1e-5, id="geostrophic_wind"),
    pytest.param("q_vector", _case_q_vector, 1e-9, id="q_vector"),
    pytest.param("shearing_deformation", _case_shearing_deformation, 1e-9, id="shearing_deformation"),
    pytest.param("stretching_deformation", _case_stretching_deformation, 1e-9, id="stretching_deformation"),
    pytest.param("total_deformation", _case_total_deformation, 1e-9, id="total_deformation"),
    pytest.param("first_derivative", _case_first_derivative, 1e-9, id="first_derivative"),
    pytest.param("second_derivative", _case_second_derivative, 1e-9, id="second_derivative"),
    pytest.param("significant_tornado", _case_significant_tornado, 1e-9, id="significant_tornado"),
    pytest.param("supercell_composite", _case_supercell_composite, 1e-9, id="supercell_composite"),
    pytest.param("critical_angle", _case_critical_angle, 1e-9, id="critical_angle"),
    pytest.param("heat_index", _case_heat_index, 1e-9, id="heat_index"),
    pytest.param("windchill", _case_windchill, 1e-9, id="windchill"),
    pytest.param("apparent_temperature", _case_apparent_temperature, 1e-9, id="apparent_temperature"),
]


@pytest.mark.parametrize(("name", "builder", "atol"), RUNTIME_CASES)
def test_runtime_parity_against_metpy(parity_context, name, builder, atol):
    args, kwargs = builder(parity_context)
    actual = getattr(mcalc, name)(*args, **kwargs)
    expected = getattr(mpcalc, name)(*args, **kwargs)
    _assert_runtime_close(actual, expected, atol)
