from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import metrust.calc as mcalc
from metrust.units import units


EXPECTED_DELEGATIONS = set()

NATIVE_NONDELEGATING_FUNCTIONS = {
    "cape_cin",
    "downdraft_cape",
    "lfc",
    "el",
    "parcel_profile_with_lcl",
    "potential_vorticity_baroclinic",
    "geospatial_laplacian",
}


def _profile_inputs():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0]) * units.hPa
    temperature = np.array([25.0, 20.0, 15.0, 5.0, -15.0, -40.0]) * units.degC
    dewpoint = np.array([20.0, 15.0, 10.0, -5.0, -25.0, -50.0]) * units.degC
    return pressure, temperature, dewpoint


def _pv_inputs():
    theta = np.stack(
        [
            np.full((3, 4), 300.0),
            np.full((3, 4), 305.0),
            np.full((3, 4), 310.0),
        ]
    ) * units.kelvin
    pressure = np.array([900.0, 800.0, 700.0]) * units.hPa
    u = np.stack(
        [
            np.full((3, 4), 10.0),
            np.full((3, 4), 12.0),
            np.full((3, 4), 14.0),
        ]
    ) * units("m/s")
    v = np.stack(
        [
            np.full((3, 4), 5.0),
            np.full((3, 4), 6.0),
            np.full((3, 4), 7.0),
        ]
    ) * units("m/s")
    latitude = np.full((3, 4), 35.0) * units.degrees
    return (theta, pressure, u, v), {
        "dx": 100000.0 * units.m,
        "dy": 100000.0 * units.m,
        "latitude": latitude,
    }


def _laplacian_inputs():
    field = np.arange(12.0, dtype=np.float64).reshape(3, 4) * units.kelvin
    return (field,), {"dx": 100000.0 * units.m, "dy": 100000.0 * units.m}


def _cape_cin_inputs():
    pressure, temperature, dewpoint = _profile_inputs()
    parcel_profile = mcalc.parcel_profile(pressure, temperature[0], dewpoint[0])
    return (pressure, temperature, dewpoint, parcel_profile), {}


def _delegation_inputs():
    return {}


def test_optional_metpy_calc_delegation_ledger_is_current():
    names = [entry["function"] for entry in mcalc.METPY_OPTIONAL_CALC_DELEGATIONS]
    assert set(names) == EXPECTED_DELEGATIONS
    assert len(names) == len(set(names))
    assert mcalc.METPY_COMPATIBILITY_TARGET == {
        "metpy": "1.7.1",
        "python": ("3.10", "3.11", "3.12", "3.13"),
    }


@pytest.mark.parametrize("name", sorted(EXPECTED_DELEGATIONS))
def test_optional_metpy_calc_delegations_use_metpy_when_available(monkeypatch, name):
    mpcalc = pytest.importorskip("metpy.calc")
    sentinel = object()

    def _stub(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(mpcalc, name, _stub)
    args, kwargs = _delegation_inputs()[name]
    result = getattr(mcalc, name)(*args, **kwargs)
    assert result is sentinel


@pytest.mark.parametrize("name", sorted(NATIVE_NONDELEGATING_FUNCTIONS))
def test_native_calc_paths_do_not_delegate_when_metpy_is_available(monkeypatch, name):
    mpcalc = pytest.importorskip("metpy.calc")

    def _boom(*args, **kwargs):
        raise AssertionError(f"{name} should stay on the native metrust path")

    monkeypatch.setattr(mpcalc, name, _boom)
    args, kwargs = {
        "cape_cin": _cape_cin_inputs(),
        "downdraft_cape": (_profile_inputs(), {}),
        "lfc": (_profile_inputs(), {}),
        "el": (_profile_inputs(), {}),
        "parcel_profile_with_lcl": (_profile_inputs(), {}),
        "potential_vorticity_baroclinic": _pv_inputs(),
        "geospatial_laplacian": _laplacian_inputs(),
    }[name]
    getattr(mcalc, name)(*args, **kwargs)


def test_optional_metpy_calc_delegations_fall_back_without_metpy():
    script = textwrap.dedent(
        """
        import builtins
        import numpy as np
        import sys

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "metpy" or name.startswith("metpy."):
                raise ImportError("blocked by fallback test")
            return real_import(name, *args, **kwargs)

        for mod in list(sys.modules):
            if mod == "metpy" or mod.startswith("metpy."):
                sys.modules.pop(mod)
        builtins.__import__ = blocked_import

        import metrust.calc as mcalc
        from metrust.units import units

        pressure = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0]) * units.hPa
        temperature = np.array([25.0, 20.0, 15.0, 5.0, -15.0, -40.0]) * units.degC
        dewpoint = np.array([20.0, 15.0, 10.0, -5.0, -25.0, -50.0]) * units.degC
        parcel_profile = mcalc.parcel_profile(pressure, temperature[0], dewpoint[0])

        lfc_pressure, _ = mcalc.lfc(pressure, temperature, dewpoint)
        el_pressure, _ = mcalc.el(pressure, temperature, dewpoint)
        cape, cin = mcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile)
        dcape, down_pressure, down_trace = mcalc.downdraft_cape(pressure, temperature, dewpoint)
        p_out, t_out, td_out, parcel_out = mcalc.parcel_profile_with_lcl(pressure, temperature, dewpoint)

        theta = np.stack([
            np.full((3, 4), 300.0),
            np.full((3, 4), 305.0),
            np.full((3, 4), 310.0),
        ]) * units.kelvin
        pressure_levels = np.array([900.0, 800.0, 700.0]) * units.hPa
        u = np.stack([
            np.full((3, 4), 10.0),
            np.full((3, 4), 12.0),
            np.full((3, 4), 14.0),
        ]) * units("m/s")
        v = np.stack([
            np.full((3, 4), 5.0),
            np.full((3, 4), 6.0),
            np.full((3, 4), 7.0),
        ]) * units("m/s")
        latitude = np.full((3, 4), 35.0) * units.degrees
        pv = mcalc.potential_vorticity_baroclinic(
            theta,
            pressure_levels,
            u,
            v,
            dx=100000.0 * units.m,
            dy=100000.0 * units.m,
            latitude=latitude,
        )

        field = np.arange(12.0, dtype=np.float64).reshape(3, 4) * units.kelvin
        lap = mcalc.geospatial_laplacian(field, dx=100000.0 * units.m, dy=100000.0 * units.m)

        assert hasattr(lfc_pressure, "to")
        assert hasattr(el_pressure, "to")
        assert np.ndim(cape.magnitude) == 0
        assert np.ndim(cin.magnitude) == 0
        assert hasattr(dcape, "to")
        assert down_pressure.shape == down_trace.shape
        assert p_out.shape == t_out.shape == td_out.shape == parcel_out.shape
        assert pv.shape == theta.shape
        assert lap.shape == field.shape
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
