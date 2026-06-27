from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import pytest

import metrust
import metrust.calc as mcalc
import metrust.interpolate as minterp
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
    assert lfc_t.units == mp_units.kelvin
    assert el_p.units == mp_units.hPa
    assert el_t.units == mp_units.kelvin

    assert mcalc.cross_totals(pres, temp, dewpoint).units == mp_units.delta_degC
    assert mcalc.vertical_totals(pres, temp).units == mp_units.delta_degC
    assert mcalc.total_totals(pres, temp, dewpoint).units == mp_units.delta_degC
    assert mcalc.k_index(pres, temp, dewpoint).units == mp_units.degC


def test_package_exposes_version():
    assert isinstance(metrust.__version__, str)
    assert metrust.__version__


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


def test_get_layer_exact_boundary_parity(sounding_profile):
    pres, temp, dewpoint, _, _, _ = sounding_profile

    actual = mcalc.get_layer(
        pres,
        temp,
        dewpoint,
        bottom=700 * mp_units.hPa,
        depth=200 * mp_units.hPa,
        interpolate=True,
    )
    expected = mpcalc.get_layer(
        pres,
        temp,
        dewpoint,
        bottom=700 * mp_units.hPa,
        depth=200 * mp_units.hPa,
        interpolate=True,
    )

    for actual_item, expected_item in zip(actual, expected):
        assert actual_item.shape == expected_item.shape
        np.testing.assert_allclose(
            actual_item.to(expected_item.units).magnitude,
            expected_item.magnitude,
            atol=1e-9,
            rtol=1e-9,
        )


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
    # xarray-wrapped results store units in .attrs, not as Pint .units
    vort_unit = getattr(vort, "units", None) or vort.attrs.get("units", "")
    adv_unit = getattr(adv, "units", None) or adv.attrs.get("units", "")
    assert "1/s" in str(vort_unit) or "second" in str(vort_unit)
    assert "1/s" in str(adv_unit) or "second" in str(adv_unit)


def test_qvector_and_frontogenesis_infer_dxdy_from_coords():
    lat2d = np.array(
        [
            [35.0, 35.0, 35.0],
            [35.5, 35.5, 35.5],
            [36.0, 36.0, 36.0],
        ]
    )
    lon2d = np.array(
        [
            [-98.0, -97.5, -97.0],
            [-98.0, -97.5, -97.0],
            [-98.0, -97.5, -97.0],
        ]
    )
    coords = {
        "latitude": (("y", "x"), lat2d),
        "longitude": (("y", "x"), lon2d),
    }
    u = xr.DataArray(np.full((3, 3), 12.0), dims=("y", "x"), coords=coords)
    v = xr.DataArray(np.full((3, 3), 6.0), dims=("y", "x"), coords=coords)
    temperature = xr.DataArray(np.full((3, 3), 290.0), dims=("y", "x"), coords=coords)

    qx, qy = mcalc.q_vector(u, v, temperature, 850 * mp_units.hPa)
    qdiv = mcalc.divergence(qx, qy)
    theta = mcalc.potential_temperature(850 * mp_units.hPa, temperature * mp_units.degC)
    fronto = mcalc.frontogenesis(theta, u, v)

    assert qx.dims == ("y", "x")
    assert qy.dims == ("y", "x")
    assert qdiv.dims == ("y", "x")
    assert qdiv.attrs["units"] == "meter / kilogram / second"
    assert fronto.shape == (3, 3)


def test_geostrophic_coriolis_and_first_derivative_metpy_forms():
    lats = np.linspace(35, 37, 3)
    lons = np.linspace(-98, -96, 3)
    lon2d, lat2d = np.meshgrid(lons, lats)
    dx, dy = mcalc.lat_lon_grid_deltas(lons * mp_units.degrees, lats * mp_units.degrees)
    heights = np.array(
        [
            [5600.0, 5610.0, 5620.0],
            [5590.0, 5600.0, 5610.0],
            [5580.0, 5590.0, 5600.0],
        ]
    ) * mp_units.meter

    coriolis = mcalc.coriolis_parameter(lat2d * mp_units.degrees)
    u_geo, v_geo = mcalc.geostrophic_wind(heights, dx=dx, dy=dy, latitude=lat2d * mp_units.degrees)
    fd = mcalc.first_derivative(np.arange(9.0).reshape(3, 3), axis=1, delta=dx)

    assert coriolis.shape == (3, 3)
    assert u_geo.shape == v_geo.shape == (3, 3)
    assert fd.units == 1 / mp_units.m


def test_baroclinic_pv_metpy_signature_preserves_xarray_coords():
    lat2d = np.array([[35.0, 35.0], [36.0, 36.0]])
    lon2d = np.array([[-98.0, -97.0], [-98.0, -97.0]])
    vertical = xr.DataArray(
        np.array([850.0, 700.0, 500.0]),
        dims=("vertical",),
        attrs={"units": "hPa"},
    )
    coords = {
        "vertical": vertical,
        "latitude": (("y", "x"), lat2d),
        "longitude": (("y", "x"), lon2d),
    }
    theta = xr.DataArray(
        np.broadcast_to(np.array([300.0, 305.0, 310.0])[:, None, None], (3, 2, 2)),
        dims=("vertical", "y", "x"),
        coords=coords,
    )
    u = xr.DataArray(np.ones((3, 2, 2)), dims=("vertical", "y", "x"), coords=coords)
    v = xr.DataArray(np.ones((3, 2, 2)), dims=("vertical", "y", "x"), coords=coords)

    pv = mcalc.potential_vorticity_baroclinic(theta, vertical * mp_units.hPa, u, v)
    pv_700 = pv.metpy.sel(vertical=700 * mp_units.hPa)

    assert pv.dims == ("vertical", "y", "x")
    assert pv.attrs["units"] == "K*m**2/(kg*s)"
    assert pv_700.shape == (2, 2)


def test_advection_handles_compound_scalar_units():
    scalar = np.full((3, 4), 1.0) / mp_units.s
    u = np.full((3, 4), 10.0) * mp_units("m/s")
    v = np.full((3, 4), 5.0) * mp_units("m/s")

    adv = mcalc.advection(scalar, u, v, dx=100000 * mp_units.m, dy=100000 * mp_units.m)

    assert adv.shape == (3, 4)
    assert adv.units == 1 / (mp_units.s ** 2)


def test_common_keyword_forms_match_metpy():
    u = np.array([1.0, 0.0]) * mp_units("m/s")
    v = np.array([0.0, -10.0]) * mp_units("m/s")

    mr_wdir = mcalc.wind_direction(u, v, convention="to")
    mp_wdir = mpcalc.wind_direction(u, v, convention="to")
    np.testing.assert_allclose(mr_wdir.to("degree").m, mp_wdir.to("degree").m, atol=1e-12)

    pressure = np.array([1000.0, 950.0, 900.0, 850.0, 800.0]) * mp_units.hPa
    dewpoint = np.array([20.0, 16.0, 12.0, 7.0, 2.0]) * mp_units.degC
    mr_pw = mcalc.precipitable_water(
        pressure,
        dewpoint,
        bottom=950 * mp_units.hPa,
        top=800 * mp_units.hPa,
    )
    mp_pw = mpcalc.precipitable_water(
        pressure,
        dewpoint,
        bottom=950 * mp_units.hPa,
        top=800 * mp_units.hPa,
    )
    np.testing.assert_allclose(mr_pw.to("mm").m, mp_pw.to("mm").m, atol=1e-3)

    theta = np.array(
        [
            [300.0, 301.0, 302.0],
            [300.5, 301.5, 302.5],
            [301.0, 302.0, 303.0],
        ]
    ) * mp_units.K
    u_grid = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    ) * mp_units("m/s")
    v_grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5],
            [-1.0, -1.0, -1.0],
        ]
    ) * mp_units("m/s")
    mr_fg = mcalc.frontogenesis(
        potential_temperature=theta,
        u=u_grid,
        v=v_grid,
        dx=1000 * mp_units.m,
        dy=1000 * mp_units.m,
    )
    mp_fg = mpcalc.frontogenesis(
        theta,
        u_grid,
        v_grid,
        dx=1000 * mp_units.m,
        dy=1000 * mp_units.m,
    )
    np.testing.assert_allclose(mr_fg.to("K/m/s").m, mp_fg.to("K/m/s").m, atol=1e-12)

    mr_adv = mcalc.advection(theta, u=u_grid, v=v_grid, dx=1000 * mp_units.m, dy=1000 * mp_units.m)
    mp_adv = mpcalc.advection(theta, u=u_grid, v=v_grid, dx=1000 * mp_units.m, dy=1000 * mp_units.m)
    np.testing.assert_allclose(mr_adv.to(mp_adv.units).m, mp_adv.m, atol=1e-12)


def test_interpolate_preserves_quantity_units_and_metpy_return_shape():
    x = np.array([925.0, 700.0, 500.0]) * mp_units.hPa
    xp = np.array([1000.0, 925.0, 850.0, 700.0, 500.0]) * mp_units.hPa
    temperature = np.array([31.0, 25.0, 19.0, 7.0, -10.0]) * mp_units.degC
    height = np.array([100.0, 750.0, 1500.0, 3000.0, 5600.0]) * mp_units.m

    mr_one = minterp.log_interpolate_1d(x, xp, temperature)
    mp_one = pytest.importorskip("metpy.interpolate").log_interpolate_1d(x, xp, temperature)
    assert hasattr(mr_one, "units")
    assert mr_one.units == mp_one.units
    np.testing.assert_allclose(mr_one.to(mp_one.units).m, mp_one.m, atol=1e-12)

    mr_many = minterp.log_interpolate_1d(x, xp, temperature, height)
    mp_many = pytest.importorskip("metpy.interpolate").log_interpolate_1d(x, xp, temperature, height)
    assert isinstance(mr_many, list)
    assert len(mr_many) == len(mp_many) == 2
    for actual, expected in zip(mr_many, mp_many):
        assert actual.units == expected.units
        np.testing.assert_allclose(actual.to(expected.units).m, expected.m, atol=1e-12)


def test_apparent_temperature_family_preserves_temperature_units():
    temp_f = 95 * mp_units.degF
    rh = 55 * mp_units.percent
    speed = 25 * mp_units("mile/hour")

    mr_hi = mcalc.heat_index(temp_f, rh, mask_undefined=False)
    mp_hi = mpcalc.heat_index(temp_f, rh, mask_undefined=False)
    assert mr_hi.units == mp_hi.units == mp_units.degF
    np.testing.assert_allclose(mr_hi.m, mp_hi.m, atol=0.25)

    mr_wc = mcalc.windchill(20 * mp_units.degF, speed, mask_undefined=False)
    mp_wc = mpcalc.windchill(20 * mp_units.degF, speed, mask_undefined=False)
    assert mr_wc.units == mp_wc.units == mp_units.degF
    np.testing.assert_allclose(mr_wc.m, mp_wc.m, atol=0.25)

    mr_at = mcalc.apparent_temperature(temp_f, rh, speed, mask_undefined=False)
    mp_at = mpcalc.apparent_temperature(temp_f, rh, speed, mask_undefined=False)
    assert mr_at.units == mp_at.units == mp_units.degF
    np.testing.assert_allclose(mr_at.m, mp_at.m, atol=0.25)


def test_metpy_signature_edge_cases_match():
    pressure = np.array([1000.0, 900.0, 800.0]) * mp_units.hPa
    temperature = np.array([20.0, 12.0, 4.0]) * mp_units.degC
    specific_humidity = np.array([0.012, 0.009, 0.006]) * mp_units("kg/kg")

    mr_dewpoint = mcalc.dewpoint_from_specific_humidity(
        pressure,
        temperature,
        specific_humidity,
    )
    mp_dewpoint = mpcalc.dewpoint_from_specific_humidity(
        pressure,
        temperature,
        specific_humidity,
    )
    np.testing.assert_allclose(mr_dewpoint.to("degC").m, mp_dewpoint.to("degC").m, atol=1e-10)

    field = np.arange(9.0).reshape(3, 3) * mp_units.K
    mr_derivative = mcalc.first_derivative(field, axis=-1, delta=1000 * mp_units.m)
    mp_derivative = mpcalc.first_derivative(field, axis=-1, delta=1000 * mp_units.m)
    np.testing.assert_allclose(mr_derivative.to("K/m").m, mp_derivative.to("K/m").m, atol=1e-12)

    mr_x, mr_y = mcalc.find_intersections(
        np.arange(5.0),
        np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        np.ones(5),
    )
    mp_x, mp_y = mpcalc.find_intersections(
        np.arange(5.0),
        np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        np.ones(5),
    )
    assert mr_x.units == mp_x.units == mp_units.dimensionless
    assert mr_y.units == mp_y.units == mp_units.dimensionless
    np.testing.assert_allclose(mr_x.m, mp_x.m, atol=1e-12)
    np.testing.assert_allclose(mr_y.m, mp_y.m, atol=1e-12)

    smoothed = mcalc.smooth_gaussian(np.arange(25.0).reshape(5, 5), 2)
    windowed = mcalc.smooth_window(np.arange(25.0).reshape(5, 5), np.ones((3, 3)))
    assert smoothed.units == mp_units.dimensionless
    assert windowed.units == mp_units.dimensionless


def test_cross_section_xarray_components_match_metpy_unit_surface():
    cross = xr.DataArray(
        np.arange(4.0),
        dims=("index",),
        coords={
            "index": np.arange(4),
            "latitude": ("index", np.linspace(35, 36, 4)),
            "longitude": ("index", np.linspace(-99, -98, 4)),
        },
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude")
    data_x = cross.copy(data=np.ones(4))
    data_x.attrs["units"] = "m/s"
    data_y = cross.copy(data=np.ones(4) * 2)
    data_y.attrs["units"] = "m/s"

    mr_tangential, mr_normal = mcalc.cross_section_components(data_x, data_y)
    mp_tangential, mp_normal = mpcalc.cross_section_components(data_x, data_y)

    for actual, expected in ((mr_tangential, mp_tangential), (mr_normal, mp_normal)):
        assert actual.attrs["units"] == expected.attrs["units"] == "m/s"
        assert getattr(actual.data, "units", None) == getattr(expected.data, "units", None)
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-12)

    mr_abs = mcalc.absolute_momentum(data_x, data_x)
    mp_abs = mpcalc.absolute_momentum(data_x, data_x)
    assert getattr(mr_abs.data, "units", None) == getattr(mp_abs.data, "units", None)
    np.testing.assert_allclose(mr_abs.values, mp_abs.values, atol=1e-12)


def test_plain_relative_humidity_arrays_match_metpy():
    temperature = np.array([20.0, 20.0]) * mp_units.degC
    pressure = np.array([1000.0, 1000.0]) * mp_units.hPa
    rh_plain = np.array([0.75, 0.80])

    mr_td = mcalc.dewpoint_from_relative_humidity(temperature, rh_plain)
    mp_td = mpcalc.dewpoint_from_relative_humidity(temperature, rh_plain)
    np.testing.assert_allclose(mr_td.to("degC").m, mp_td.to("degC").m, atol=1e-10)

    mr_w = mcalc.mixing_ratio_from_relative_humidity(
        pressure,
        temperature,
        rh_plain,
        phase="liquid",
    )
    mp_w = mpcalc.mixing_ratio_from_relative_humidity(
        pressure,
        temperature,
        rh_plain,
        phase="liquid",
    )
    np.testing.assert_allclose(mr_w.to("kg/kg").m, mp_w.to("kg/kg").m, rtol=1e-2)

    cold_pressure = np.array([800.0, 700.0]) * mp_units.hPa
    cold_temperature = np.array([-10.0, -20.0]) * mp_units.degC
    cold_rh = np.array([0.75, 0.80])
    mr_w_solid = mcalc.mixing_ratio_from_relative_humidity(
        cold_pressure,
        cold_temperature,
        cold_rh,
        phase="solid",
    )
    mp_w_solid = mpcalc.mixing_ratio_from_relative_humidity(
        cold_pressure,
        cold_temperature,
        cold_rh,
        phase="solid",
    )
    np.testing.assert_allclose(mr_w_solid.to("kg/kg").m, mp_w_solid.to("kg/kg").m, rtol=1e-2)

    rh_roundtrip = mcalc.relative_humidity_from_mixing_ratio(
        cold_pressure,
        cold_temperature,
        mr_w_solid,
        phase="solid",
    )
    np.testing.assert_allclose(rh_roundtrip.m, cold_rh, atol=1e-10)


def test_moist_lapse_internal_reference_pressure_matches_metpy():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0, 500.0]) * mp_units.hPa
    reference_pressure = 875 * mp_units.hPa

    mr_trace = mcalc.moist_lapse(
        pressure,
        10 * mp_units.degC,
        reference_pressure=reference_pressure,
    )
    mp_trace = mpcalc.moist_lapse(
        pressure,
        10 * mp_units.degC,
        reference_pressure=reference_pressure,
    )

    np.testing.assert_allclose(mr_trace.to("degC").m, mp_trace.to("degC").m, atol=2e-5)


def test_variable_spacing_derivatives_match_metpy():
    x = np.array([0.0, 1000.0, 3000.0, 6000.0])
    y = np.array([0.0, 500.0, 2000.0])
    xx, yy = np.meshgrid(x, y)
    dx = np.tile(np.diff(x), (len(y), 1)) * mp_units.m
    dy = np.tile(np.diff(y)[:, None], (1, len(x))) * mp_units.m

    field = ((xx / 1000.0) ** 2) * mp_units.K
    mr_fd = mcalc.first_derivative(field, axis=1, delta=dx)
    mp_fd = mpcalc.first_derivative(field, axis=1, delta=dx)
    np.testing.assert_allclose(mr_fd.to("K/m").m, mp_fd.to("K/m").m, atol=1e-12)

    u = ((yy / 1000.0) ** 2) * mp_units("m/s")
    v = ((xx / 1000.0) ** 2) * mp_units("m/s")
    mr_div = mcalc.divergence(u, v, dx=dx, dy=dy)
    mp_div = mpcalc.divergence(u, v, dx=dx, dy=dy)
    np.testing.assert_allclose(mr_div.to("1/s").m, mp_div.to("1/s").m, atol=1e-12)

    mr_vort = mcalc.vorticity(u, v, dx=dx, dy=dy)
    mp_vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)
    np.testing.assert_allclose(mr_vort.to("1/s").m, mp_vort.to("1/s").m, atol=1e-12)

    theta = (300.0 + xx / 1000.0) * mp_units.K
    u_front = (xx / 1000.0) * mp_units("m/s")
    v_front = (-yy / 1000.0) * mp_units("m/s")
    mr_fg = mcalc.frontogenesis(theta, u_front, v_front, dx=dx, dy=dy)
    mp_fg = mpcalc.frontogenesis(theta, u_front, v_front, dx=dx, dy=dy)
    np.testing.assert_allclose(mr_fg.to("K/m/s").m, mp_fg.to("K/m/s").m, atol=1e-12)

    mr_tdef = mcalc.total_deformation(u_front, v_front, dx=dx, dy=dy)
    mp_tdef = mpcalc.total_deformation(u_front, v_front, dx=dx, dy=dy)
    np.testing.assert_allclose(mr_tdef.to("1/s").m, mp_tdef.to("1/s").m, atol=1e-12)
