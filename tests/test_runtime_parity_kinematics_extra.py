from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units


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
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_runtime_close(actual_item, expected_item, atol)
        return

    if isinstance(actual, xr.DataArray) and isinstance(expected, xr.DataArray):
        assert actual.dims == expected.dims
        _assert_runtime_close(actual.data, expected.data, atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
    else:
        actual_arr = np.asarray(actual, dtype=np.float64)
        expected_arr = np.asarray(expected, dtype=np.float64)

    assert actual_arr.shape == expected_arr.shape
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def kinematics_context():
    x = np.linspace(0, 300000, 4) * units.m
    y = np.linspace(0, 200000, 3) * units.m
    xx, yy = np.meshgrid(x.magnitude, y.magnitude)
    latitude = np.full_like(xx, 35.0) * units.degrees
    u = ((xx / 100000.0) + 2.0 * (yy / 100000.0)) * units("m/s")
    v = ((yy / 100000.0) - (xx / 150000.0)) * units("m/s")
    height = (5600.0 + 10.0 * (xx / 100000.0) - 20.0 * (yy / 100000.0)) * units.meter
    scalar = (280.0 + 2.0 * (xx / 100000.0) + 1.5 * (yy / 100000.0)) * units.kelvin

    pressure_levels = np.array([900.0, 800.0, 700.0]) * units.hPa
    pressure_3d = pressure_levels[:, None, None] * np.ones((3, *u.shape))
    theta = np.stack([
        300.0 + 0.01 * xx + 0.02 * yy,
        305.0 + 0.01 * xx + 0.02 * yy,
        310.0 + 0.01 * xx + 0.02 * yy,
    ]) * units.kelvin
    u3 = np.stack([u.magnitude, u.magnitude + 1.0, u.magnitude + 2.0]) * units("m/s")
    v3 = np.stack([v.magnitude, v.magnitude + 0.5, v.magnitude + 1.0]) * units("m/s")

    cross = xr.Dataset(
        {
            "u": (("isobaric", "index"), np.array([[10, 11, 12, 13], [14, 15, 16, 17]]), {"units": "m/s"}),
            "v": (("isobaric", "index"), np.array([[5, 6, 7, 8], [9, 10, 11, 12]]), {"units": "m/s"}),
        },
        coords={
            "isobaric": ("isobaric", np.array([900.0, 800.0]), {"units": "hPa"}),
            "index": ("index", np.arange(4)),
            "latitude": ("index", np.array([35.0, 35.1, 35.2, 35.3]), {"units": "degrees_north"}),
            "longitude": ("index", np.array([-97.0, -96.9, -96.8, -96.7]), {"units": "degrees_east"}),
        },
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude").metpy.quantify()

    u_cs = cross["u"]
    v_cs = cross["v"]
    u_geostrophic, v_geostrophic = mpcalc.geostrophic_wind(
        height,
        dx=100000 * units.m,
        dy=100000 * units.m,
        latitude=latitude,
    )

    return {
        "u": u,
        "v": v,
        "height": height,
        "scalar": scalar,
        "latitude": latitude,
        "pressure_3d": pressure_3d,
        "theta": theta,
        "u3": u3,
        "v3": v3,
        "u_cs": u_cs,
        "v_cs": v_cs,
        "u_geostrophic": u_geostrophic,
        "v_geostrophic": v_geostrophic,
        "vel_series": np.array([1.0, 2.0, 3.0, 4.0]) * units("m/s"),
        "scalar_series": np.array([4.0, 5.0, 6.0, 7.0]) * units.kelvin,
    }


def _case_absolute_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_ageostrophic_wind(ctx):
    return (
        ctx["height"],
        ctx["u"],
        ctx["v"],
    ), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_potential_vorticity_baroclinic(ctx):
    return (
        ctx["theta"],
        ctx["pressure_3d"],
        ctx["u3"],
        ctx["v3"],
    ), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"], "vertical_dim": 0}


def _case_potential_vorticity_barotropic(ctx):
    return (
        ctx["height"],
        ctx["u"],
        ctx["v"],
    ), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_normal_component(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_tangential_component(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_unit_vectors_from_cross_section(ctx):
    return (ctx["u_cs"],), {}


def _case_vector_derivative(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_absolute_momentum(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_cross_section_components(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_curvature_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_coriolis_parameter(ctx):
    return (ctx["latitude"],), {}


def _case_inertial_advective_wind(ctx):
    return (
        ctx["u"],
        ctx["v"],
        ctx["u_geostrophic"],
        ctx["v_geostrophic"],
    ), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_kinematic_flux(ctx):
    return (ctx["vel_series"], ctx["scalar_series"]), {}


def _case_shear_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_geospatial_gradient(ctx):
    return (ctx["height"],), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_geospatial_laplacian(ctx):
    return (ctx["height"],), {"dx": 100000 * units.m, "dy": 100000 * units.m}


KINEMATICS_CASES = [
    pytest.param("absolute_vorticity", _case_absolute_vorticity, 1e-9, id="absolute_vorticity"),
    pytest.param("ageostrophic_wind", _case_ageostrophic_wind, 1e-6, id="ageostrophic_wind"),
    pytest.param(
        "potential_vorticity_baroclinic",
        _case_potential_vorticity_baroclinic,
        1e-6,
        id="potential_vorticity_baroclinic",
    ),
    pytest.param(
        "potential_vorticity_barotropic",
        _case_potential_vorticity_barotropic,
        1e-9,
        id="potential_vorticity_barotropic",
    ),
    pytest.param("normal_component", _case_normal_component, 1e-9, id="normal_component"),
    pytest.param("tangential_component", _case_tangential_component, 1e-9, id="tangential_component"),
    pytest.param(
        "unit_vectors_from_cross_section",
        _case_unit_vectors_from_cross_section,
        1e-9,
        id="unit_vectors_from_cross_section",
    ),
    pytest.param("vector_derivative", _case_vector_derivative, 1e-9, id="vector_derivative"),
    pytest.param("absolute_momentum", _case_absolute_momentum, 1e-6, id="absolute_momentum"),
    pytest.param(
        "cross_section_components",
        _case_cross_section_components,
        1e-9,
        id="cross_section_components",
    ),
    pytest.param("curvature_vorticity", _case_curvature_vorticity, 1e-8, id="curvature_vorticity"),
    pytest.param("coriolis_parameter", _case_coriolis_parameter, 1e-12, id="coriolis_parameter"),
    pytest.param(
        "inertial_advective_wind",
        _case_inertial_advective_wind,
        1e-6,
        id="inertial_advective_wind",
    ),
    pytest.param("kinematic_flux", _case_kinematic_flux, 1e-12, id="kinematic_flux"),
    pytest.param("shear_vorticity", _case_shear_vorticity, 1e-8, id="shear_vorticity"),
    pytest.param("geospatial_gradient", _case_geospatial_gradient, 1e-9, id="geospatial_gradient"),
    pytest.param("geospatial_laplacian", _case_geospatial_laplacian, 1e-9, id="geospatial_laplacian"),
]


@pytest.mark.parametrize(("name", "builder", "atol"), KINEMATICS_CASES)
def test_runtime_parity_kinematics_extra(kinematics_context, name, builder, atol):
    args, kwargs = builder(kinematics_context)
    actual = getattr(mcalc, name)(*args, **kwargs)
    expected = getattr(mpcalc, name)(*args, **kwargs)
    _assert_runtime_close(actual, expected, atol)
