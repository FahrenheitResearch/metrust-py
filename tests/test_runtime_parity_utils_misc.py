from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

import metrust.calc as mcalc

mpcalc = pytest.importorskip("metpy.calc")
units = pytest.importorskip("metpy.units").units


def _compare_quantity(actual, expected):
    try:
        actual_mag = np.asarray(actual.to(expected.units).magnitude, dtype=np.float64)
        expected_mag = np.asarray(expected.magnitude, dtype=np.float64)
    except Exception:
        actual_mag = np.asarray(actual.to_base_units().magnitude, dtype=np.float64)
        expected_mag = np.asarray(expected.to_base_units().magnitude, dtype=np.float64)
    return actual_mag, expected_mag


def _assert_close(actual, expected, atol=1e-9):
    if isinstance(actual, xr.DataArray) and isinstance(expected, xr.DataArray):
        assert actual.dims == expected.dims
        np.testing.assert_allclose(actual.values, expected.values, atol=atol, rtol=atol, equal_nan=True)
        for dim in actual.dims:
            np.testing.assert_allclose(actual[dim].values, expected[dim].values, atol=atol, rtol=atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = _compare_quantity(actual, expected)
        np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)
        return

    actual_arr = np.asarray(getattr(actual, "magnitude", actual))
    expected_arr = np.asarray(getattr(expected, "magnitude", expected))
    if actual_arr.dtype.kind in {"U", "S", "O"} or expected_arr.dtype.kind in {"U", "S", "O"}:
        np.testing.assert_array_equal(actual_arr, expected_arr)
        return
    np.testing.assert_allclose(actual_arr, expected_arr, atol=atol, rtol=atol, equal_nan=True)


def test_galvez_davison_and_standard_atmosphere_runtime_parity():
    pressure = np.array([1000, 950, 850, 700, 500]) * units.hPa
    temperature = np.array([28, 24, 18, 8, -5]) * units.degC
    mixing_ratio = np.array([15, 13, 9, 4, 1]) * units("g/kg")

    _assert_close(
        mcalc.galvez_davison_index(pressure, temperature, mixing_ratio, pressure[0]),
        mpcalc.galvez_davison_index(pressure, temperature, mixing_ratio, pressure[0]),
    )
    _assert_close(
        mcalc.pressure_to_height_std(np.array([1000, 900]) * units.hPa),
        mpcalc.pressure_to_height_std(np.array([1000, 900]) * units.hPa),
    )
    _assert_close(
        mcalc.height_to_pressure_std(np.array([0, 1000]) * units.m),
        mpcalc.height_to_pressure_std(np.array([0, 1000]) * units.m),
    )
    _assert_close(
        mcalc.altimeter_to_station_pressure(29.92 * units.inHg, 350 * units.m),
        mpcalc.altimeter_to_station_pressure(29.92 * units.inHg, 350 * units.m),
    )
    _assert_close(
        mcalc.altimeter_to_sea_level_pressure(29.92 * units.inHg, 350 * units.m, 20 * units.degC),
        mpcalc.altimeter_to_sea_level_pressure(29.92 * units.inHg, 350 * units.m, 20 * units.degC),
    )
    _assert_close(
        mcalc.sigma_to_pressure(np.linspace(0, 1, 5), 1000 * units.hPa, 100 * units.hPa),
        mpcalc.sigma_to_pressure(np.linspace(0, 1, 5), 1000 * units.hPa, 100 * units.hPa),
    )


def test_smoothing_runtime_parity():
    field = np.arange(25.0).reshape(5, 5)
    window = np.ones((3, 3))

    _assert_close(mcalc.smooth_gaussian(field, 2), mpcalc.smooth_gaussian(field, 2), atol=1e-12)
    _assert_close(
        mcalc.smooth_rectangular(field, 3, passes=2),
        mpcalc.smooth_rectangular(field, 3, passes=2),
        atol=1e-12,
    )
    _assert_close(
        mcalc.smooth_circular(field, 1, passes=2),
        mpcalc.smooth_circular(field, 1, passes=2),
        atol=1e-12,
    )
    _assert_close(
        mcalc.smooth_n_point(field, 5, passes=2),
        mpcalc.smooth_n_point(field, 5, passes=2),
        atol=1e-12,
    )
    _assert_close(
        mcalc.smooth_window(field, window, passes=2),
        mpcalc.smooth_window(field, window, passes=2),
        atol=1e-12,
    )


def test_gradient_and_laplacian_runtime_parity():
    field = np.arange(12.0).reshape(3, 4) * units.kelvin
    coordinates = (
        np.array([0.0, 2.0, 4.0]) * units.m,
        np.array([0.0, 5.0, 10.0, 15.0]) * units.m,
    )

    actual_gradient = mcalc.gradient(field, coordinates=coordinates)
    expected_gradient = mpcalc.gradient(field, coordinates=coordinates)
    assert isinstance(actual_gradient, tuple)
    assert len(actual_gradient) == len(expected_gradient) == 2
    for actual, expected in zip(actual_gradient, expected_gradient):
        _assert_close(actual, expected, atol=1e-12)

    _assert_close(
        mcalc.laplacian(field, coordinates=coordinates),
        mpcalc.laplacian(field, coordinates=coordinates),
        atol=1e-12,
    )


def test_lat_lon_grid_deltas_runtime_parity():
    longitude = np.linspace(-100, -98, 4) * units.deg
    latitude = np.linspace(35, 36, 3) * units.deg

    actual_dx, actual_dy = mcalc.lat_lon_grid_deltas(longitude, latitude)
    expected_dx, expected_dy = mpcalc.lat_lon_grid_deltas(longitude, latitude)
    _assert_close(actual_dx, expected_dx, atol=1e-6)
    _assert_close(actual_dy, expected_dy, atol=1e-6)


def test_angle_helpers_runtime_parity():
    _assert_close(
        mcalc.angle_to_direction(np.array([0, 45, 225]) * units.deg, level=3, full=True),
        mpcalc.angle_to_direction(np.array([0, 45, 225]) * units.deg, level=3, full=True),
    )
    _assert_close(
        mcalc.parse_angle(["north", "SW"]),
        mpcalc.parse_angle(["north", "SW"]),
    )


def test_find_bounding_indices_runtime_parity():
    arr = np.array([[1000.0, 900.0], [800.0, 700.0]])
    actual_above, actual_below, actual_good = mcalc.find_bounding_indices(arr, [850.0], axis=0)
    expected_above, expected_below, expected_good = mpcalc.find_bounding_indices(arr, [850.0], axis=0)

    assert len(actual_above) == len(expected_above)
    assert len(actual_below) == len(expected_below)
    for actual, expected in zip(actual_above, expected_above):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_below, expected_below):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_good, expected_good)


def test_intersection_helpers_runtime_parity():
    x = np.array([0.0, 1.0, 2.0])
    a = np.array([0.0, 2.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    actual_x, actual_y = mcalc.find_intersections(x, a, b)
    expected_x, expected_y = mpcalc.find_intersections(x, a, b)
    _assert_close(actual_x, expected_x, atol=1e-12)
    _assert_close(actual_y, expected_y, atol=1e-12)
    _assert_close(
        mcalc.nearest_intersection_idx(a, b),
        mpcalc.nearest_intersection_idx(a, b),
    )


def test_peak_and_resample_helpers_runtime_parity():
    field = np.array(
        [
            [5.0, 1.0, 5.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [4.0, 1.0, 6.0, 1.0],
            [1.0, 3.0, 1.0, 4.0],
        ]
    )

    actual_persistence = mcalc.peak_persistence(field)
    expected_persistence = mpcalc.peak_persistence(field)
    assert len(actual_persistence) == len(expected_persistence)
    for (actual_idx, actual_value), (expected_idx, expected_value) in zip(
        actual_persistence,
        expected_persistence,
    ):
        assert actual_idx == expected_idx
        if np.isinf(expected_value):
            assert np.isinf(actual_value)
        else:
            assert actual_value == pytest.approx(expected_value)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual_peaks = list(mcalc.find_peaks(field, iqr_ratio=0))
        expected_peaks = list(mpcalc.find_peaks(field, iqr_ratio=0))
    assert actual_peaks == expected_peaks

    assert mcalc.resample_nn_1d(np.array([0, 1, 2, 3]), np.array([0.1, 2.1])) == mpcalc.resample_nn_1d(
        np.array([0, 1, 2, 3]),
        np.array([0.1, 2.1]),
    )


def test_azimuth_range_to_lat_lon_runtime_parity():
    azimuths = np.array([0.0, 90.0]) * units.deg
    ranges = np.array([0.0, 1000.0]) * units.m

    actual_lon, actual_lat = mcalc.azimuth_range_to_lat_lon(azimuths, ranges, -97.5, 35.4)
    expected_lon, expected_lat = mpcalc.azimuth_range_to_lat_lon(azimuths, ranges, -97.5, 35.4)
    _assert_close(actual_lon, expected_lon, atol=1e-9)
    _assert_close(actual_lat, expected_lat, atol=1e-9)


def test_zoom_xarray_runtime_parity():
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [10.0, 20.0, 30.0]},
    )
    _assert_close(mcalc.zoom_xarray(data, 2), mpcalc.zoom_xarray(data, 2), atol=1e-12)
