import types

import numpy as np
import pytest

import metrust
import metrust.calc as mcalc
from metrust.units import units


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    monkeypatch.setattr(mcalc, "_GPU_CALC", None)
    mcalc.set_backend("cpu")
    yield
    monkeypatch.setattr(mcalc, "_GPU_CALC", None)
    mcalc.set_backend("cpu")


def test_backend_exports_and_defaults():
    assert mcalc.get_backend() == "cpu"
    assert metrust.get_backend() == "cpu"
    with pytest.raises(ValueError, match="cpu' or 'gpu"):
        mcalc.set_backend("nope")


def test_set_backend_gpu_requires_metcu(monkeypatch):
    monkeypatch.setattr(
        mcalc.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'metcu'")),
    )
    with pytest.raises(ImportError, match="met-cu"):
        mcalc.set_backend("gpu")
    assert mcalc.get_backend() == "cpu"


def test_use_backend_restores_previous_backend(monkeypatch):
    fake_gpu = types.SimpleNamespace(
        potential_temperature=lambda pressure, temperature: np.array(301.0)
    )
    monkeypatch.setattr(mcalc, "_load_gpu_calc", lambda: fake_gpu)

    assert mcalc.get_backend() == "cpu"
    with mcalc.use_backend("gpu"):
        assert mcalc.get_backend() == "gpu"
    assert mcalc.get_backend() == "cpu"


def test_gpu_backend_dispatches_scalar_thermo(monkeypatch):
    calls = {}

    def fake_potential_temperature(pressure, temperature):
        calls["args"] = (pressure, temperature)
        return np.array(302.5)

    fake_gpu = types.SimpleNamespace(potential_temperature=fake_potential_temperature)
    monkeypatch.setattr(mcalc, "_load_gpu_calc", lambda: fake_gpu)

    mcalc.set_backend("gpu")
    theta = mcalc.potential_temperature(1000 * units.hPa, 25 * units.degC)

    assert calls["args"] == (1000, 25)
    assert theta.units == units.K
    assert theta.to("K").m == pytest.approx(302.5)


def test_gpu_backend_dispatches_grid_tuple_function(monkeypatch):
    calls = {}

    def fake_compute_cape_cin(pressure_3d, temperature_c_3d, qvapor_3d,
                              height_agl_3d, psfc, t2, q2,
                              parcel_type="surface", top_m=None):
        calls["shapes"] = (
            pressure_3d.shape,
            temperature_c_3d.shape,
            qvapor_3d.shape,
            height_agl_3d.shape,
            psfc.shape,
            t2.shape,
            q2.shape,
        )
        calls["options"] = (parcel_type, top_m)
        shape = pressure_3d.shape[1:]
        return (
            np.full(shape, 1200.0),
            np.full(shape, -40.0),
            np.full(shape, 900.0),
            np.full(shape, 1500.0),
        )

    fake_gpu = types.SimpleNamespace(compute_cape_cin=fake_compute_cape_cin)
    monkeypatch.setattr(mcalc, "_load_gpu_calc", lambda: fake_gpu)

    p3 = np.full((4, 2, 3), 90000.0) * units.Pa
    t3 = np.full((4, 2, 3), 18.0) * units.degC
    q3 = np.full((4, 2, 3), 0.012)
    h3 = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[500.0, 500.0, 500.0], [500.0, 500.0, 500.0]],
            [[1500.0, 1500.0, 1500.0], [1500.0, 1500.0, 1500.0]],
            [[3000.0, 3000.0, 3000.0], [3000.0, 3000.0, 3000.0]],
        ]
    ) * units.m
    psfc = np.full((2, 3), 95000.0) * units.Pa
    t2 = np.full((2, 3), 295.0) * units.K
    q2 = np.full((2, 3), 0.014)

    mcalc.set_backend("gpu")
    cape, cin, lcl, lfc = mcalc.compute_cape_cin(
        p3, t3, q3, h3, psfc, t2, q2, parcel_type="mu", top_m=1500 * units.m
    )

    assert calls["shapes"] == (
        (4, 2, 3),
        (4, 2, 3),
        (4, 2, 3),
        (4, 2, 3),
        (2, 3),
        (2, 3),
        (2, 3),
    )
    assert calls["options"] == ("mu", 1500.0)
    assert cape.units == units("J/kg")
    assert cin.units == units("J/kg")
    assert lcl.units == units.m
    assert lfc.units == units.m
    assert cape.shape == (2, 3)
    assert float(cape[0, 0].m) == pytest.approx(1200.0)
    assert float(cin[0, 0].m) == pytest.approx(-40.0)


def test_gpu_backend_falls_back_for_projection_aware_vorticity(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("GPU dispatch should not run for projection-aware vorticity")

    fake_gpu = types.SimpleNamespace(vorticity=should_not_run)
    monkeypatch.setattr(mcalc, "_load_gpu_calc", lambda: fake_gpu)

    u = np.array([[5.0, 6.0], [7.0, 8.0]]) * units("m/s")
    v = np.array([[1.0, 1.5], [2.0, 2.5]]) * units("m/s")

    mcalc.set_backend("gpu")
    vort = mcalc.vorticity(
        u,
        v,
        dx=1000 * units.m,
        dy=1000 * units.m,
        latitude=np.array([[35.0, 35.0], [36.0, 36.0]]),
    )

    assert vort.units == units("1/s")
    assert vort.shape == (2, 2)
