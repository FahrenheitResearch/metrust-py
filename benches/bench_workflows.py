#!/usr/bin/env python3
"""End-to-end workflow replay benchmarks for metrust vs MetPy.

This harness benchmarks three real workflow shapes that already have replay
parity tests in the suite:

- sounding analysis
- gridded diagnostics
- xarray-heavy dataset helpers

It verifies output parity once before timing, then reports p50 latency and
MetPy->metrust speedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import metrust.calc as mcalc
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units


WARMUP_CALLS = 3
REPEATS = 7
TARGET_SECONDS = 0.2


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
        actual_arr = np.asarray(actual, dtype=np.float64)
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


def _run_sounding_workflow(calc, sounding):
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


def _run_grid_workflow(calc, grid_fields):
    u, v, theta = grid_fields
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


def _build_xarray_inputs():
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
        "pressure": pressure,
        "temperature_profile": temperature_profile,
        "dewpoint_profile": dewpoint_profile,
        "levels": levels,
        "temperature_da": temperature_da,
        "u_da": u_da,
    }


def _run_xarray_workflow(calc, ctx):
    return {
        "parcel_profile_with_lcl": calc.parcel_profile_with_lcl_as_dataset(
            ctx["pressure"],
            ctx["temperature_profile"],
            ctx["dewpoint_profile"],
        ),
        "isentropic_interpolation": calc.isentropic_interpolation_as_dataset(
            ctx["levels"],
            ctx["temperature_da"],
            ctx["u_da"],
        ),
    }


WORKFLOWS = (
    {
        "name": "Cookbook sounding replay",
        "builder": _load_sounding,
        "runner": _run_sounding_workflow,
        "atol": {
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
    },
    {
        "name": "Cookbook grid diagnostics replay",
        "builder": _build_grid_fields,
        "runner": _run_grid_workflow,
        "atol": 2e-6,
    },
    {
        "name": "Cookbook xarray replay",
        "builder": _build_xarray_inputs,
        "runner": _run_xarray_workflow,
        "atol": {
            "parcel_profile_with_lcl": 3e-2,
            "isentropic_interpolation": 1e-6,
        },
    },
)


def _auto_iterations(func, target_seconds=TARGET_SECONDS):
    elapsed = timeit.timeit(func, number=1)
    if elapsed <= 0:
        elapsed = 1e-7
    return max(1, int(target_seconds / elapsed))


def _bench_ms(func):
    for _ in range(WARMUP_CALLS):
        func()
    number = _auto_iterations(func)
    samples = timeit.repeat(func, number=number, repeat=REPEATS)
    per_call_ms = sorted((sample / number) * 1000.0 for sample in samples)
    return {
        "iterations": number,
        "p50_ms": per_call_ms[len(per_call_ms) // 2],
        "samples_ms": per_call_ms,
    }


def run_workflow_benchmarks():
    results = []
    for spec in WORKFLOWS:
        ctx = spec["builder"]()
        metrust_value = spec["runner"](mcalc, ctx)
        metpy_value = spec["runner"](mpcalc, ctx)
        _assert_close(metrust_value, metpy_value, spec["atol"])

        metrust_bench = _bench_ms(lambda ctx=ctx, spec=spec: spec["runner"](mcalc, ctx))
        metpy_bench = _bench_ms(lambda ctx=ctx, spec=spec: spec["runner"](mpcalc, ctx))
        speedup = (
            metpy_bench["p50_ms"] / metrust_bench["p50_ms"]
            if metrust_bench["p50_ms"] > 0
            else float("nan")
        )
        results.append(
            {
                "workflow": spec["name"],
                "metrust_p50_ms": metrust_bench["p50_ms"],
                "metpy_p50_ms": metpy_bench["p50_ms"],
                "speedup": speedup,
                "metrust_iterations": metrust_bench["iterations"],
                "metpy_iterations": metpy_bench["iterations"],
            }
        )
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "results": results,
    }


def _print_table(payload):
    print(f"Platform: {payload['platform']}")
    print(f"Python:   {payload['python']}")
    print()
    print("| Workflow | metrust p50 | MetPy p50 | Speedup |")
    print("|---|---:|---:|---:|")
    for row in payload["results"]:
        print(
            f"| {row['workflow']} | {row['metrust_p50_ms']:.2f} ms | "
            f"{row['metpy_p50_ms']:.2f} ms | {row['speedup']:.2f}x |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--json-file",
        default="workflow_bench_results.json",
        help="Path for JSON output when --json is set.",
    )
    args = parser.parse_args()

    payload = run_workflow_benchmarks()
    _print_table(payload)
    if args.json:
        json_path = Path(args.json_file)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
