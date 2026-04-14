"""Standalone tests for metrust -- no MetPy dependency required.

These tests verify that metrust's native Rust-backed functionality works
correctly without importing or comparing against MetPy.
"""

import numpy as np
import pytest


# ── 1. test_basic_import ─────────────────────────────────────────────

def test_basic_import():
    """All core submodules should be importable."""
    import metrust.calc
    import metrust.io
    import metrust.constants
    import metrust.units

    # Sanity: each module object is truthy and has a name
    assert metrust.calc.__name__ == "metrust.calc"
    assert metrust.io.__name__ == "metrust.io"
    assert metrust.constants.__name__ == "metrust.constants"
    assert metrust.units.__name__ == "metrust.units"


# ── 2. test_pint_unit_handling ───────────────────────────────────────

def test_pint_unit_handling():
    """Offset temperatures (degC) work with metrust's unit registry."""
    from metrust.units import units

    # Scalar offset temperature
    t = 20 * units.degC
    assert t.magnitude == 20
    assert t.units == units.degC

    # Array offset temperature
    temps = [25, 20, 15] * units.degC
    assert temps.units == units.degC
    assert temps.to("degC").m.tolist() == [25, 20, 15]

    # Conversion to Kelvin (autoconvert_offset_to_baseunit)
    t_k = t.to("K")
    assert t_k.m == pytest.approx(293.15, abs=0.01)

    # Pressure units
    p = 1000 * units.hPa
    p_pa = p.to("Pa")
    assert p_pa.m == pytest.approx(100000.0)


# ── 3. test_potential_temperature ────────────────────────────────────

def test_potential_temperature():
    """potential_temperature(1000 hPa, 25 degC) ~ 298.15 K."""
    import metrust.calc as mcalc
    from metrust.units import units

    theta = mcalc.potential_temperature(1000 * units.hPa, 25 * units.degC)

    # Result should be in Kelvin
    assert theta.units == units.K

    # At 1000 hPa the potential temperature equals the absolute temperature
    # 25 C = 298.15 K, so theta should be very close to 298.15 K
    assert theta.to("K").m == pytest.approx(298.15, abs=1.0)


# ── 4. test_saturation_vapor_pressure ────────────────────────────────

def test_saturation_vapor_pressure():
    """saturation_vapor_pressure at 20 degC is ~2338 Pa (Bolton 1980)."""
    import metrust.calc as mcalc
    from metrust.units import units

    svp = mcalc.saturation_vapor_pressure(20 * units.degC)

    # Result should be in Pa
    assert svp.units == units.Pa

    # Well-known value: ~2338 Pa at 20 C (within 1 %)
    assert svp.to("Pa").m == pytest.approx(2338.0, rel=0.01)


# ── 5. test_cape_cin ─────────────────────────────────────────────────

def test_cape_cin():
    """CAPE/CIN with a simple sounding returns reasonable values."""
    import metrust.calc as mcalc
    from metrust.units import units

    # Construct a simple sounding with corresponding heights
    pressure = np.array([1000, 925, 850, 700, 500, 300]) * units.hPa
    temperature = np.array([30, 25, 20, 10, -10, -40]) * units.degC
    dewpoint = np.array([25, 20, 15, 0, -20, -45]) * units.degC
    height = np.array([0, 750, 1500, 3000, 5500, 9000]) * units.m

    # Use the native Rust path (psfc / t2m / td2m interface)
    cape, cin, h_lcl, h_lfc = mcalc.cape_cin(
        pressure, temperature, dewpoint, height,
        psfc=1000 * units.hPa,
        t2m=30 * units.degC,
        td2m=25 * units.degC,
        parcel_type="sb",
    )

    # Units check
    assert cape.units == units("J/kg")
    assert cin.units == units("J/kg")

    # CAPE should be non-negative and reasonably large for this moist sounding
    assert cape.m >= 0
    # CIN should be non-positive (inhibition)
    assert cin.m <= 0


def test_compute_ecape_smoke():
    """Grid ECAPE returns the expected field family with units and shapes."""
    import metrust.calc as mcalc
    from metrust.units import units

    pressure = np.array([95000, 90000, 85000, 70000, 50000, 30000], dtype=float)
    temperature = np.array([26, 22, 18, 8, -10, -38], dtype=float)
    qvapor = np.array([0.016, 0.013, 0.010, 0.005, 0.0015, 0.0003], dtype=float)
    height = np.array([150, 800, 1500, 3000, 5600, 9200], dtype=float)
    u = np.array([6, 9, 12, 18, 26, 33], dtype=float)
    v = np.array([2, 5, 8, 13, 20, 28], dtype=float)

    p3 = pressure[:, None, None] * np.ones((6, 2, 2)) * units.Pa
    t3 = temperature[:, None, None] * np.ones((6, 2, 2)) * units.degC
    q3 = qvapor[:, None, None] * np.ones((6, 2, 2))
    h3 = height[:, None, None] * np.ones((6, 2, 2)) * units.m
    u3 = u[:, None, None] * np.ones((6, 2, 2)) * units("m/s")
    v3 = v[:, None, None] * np.ones((6, 2, 2)) * units("m/s")

    ecape, ncape, cape, cin, lfc, el = mcalc.compute_ecape(
        p3,
        t3,
        q3,
        h3,
        u3,
        v3,
        np.full((2, 2), 100000.0) * units.Pa,
        np.full((2, 2), 303.15) * units.K,
        np.full((2, 2), 0.018),
        np.full((2, 2), 5.0) * units("m/s"),
        np.full((2, 2), 1.5) * units("m/s"),
        parcel_type="ml",
        storm_motion_type="bunkers_rm",
    )

    for field in (ecape, ncape, cape, cin):
        assert field.shape == (2, 2)
        assert field.units == units("J/kg")
        assert np.all(np.isfinite(field.m))

    for field in (lfc, el):
        assert field.shape == (2, 2)
        assert field.units == units.m
        assert np.all(np.isfinite(field.m))


# ── 6. test_wind_functions ───────────────────────────────────────────

def test_compute_ecape_converts_pint_units_at_boundary():
    """ECAPE converts Pint inputs to its documented kernel units."""
    import metrust.calc as mcalc
    from metrust.units import units

    pressure = np.array([95000, 90000, 85000, 70000, 50000, 30000], dtype=float)
    temperature = np.array([26, 22, 18, 8, -10, -38], dtype=float)
    qvapor = np.array([0.016, 0.013, 0.010, 0.005, 0.0015, 0.0003], dtype=float)
    height = np.array([150, 800, 1500, 3000, 5600, 9200], dtype=float)
    u = np.array([6, 9, 12, 18, 26, 33], dtype=float)
    v = np.array([2, 5, 8, 13, 20, 28], dtype=float)

    p_native = pressure[:, None, None] * np.ones((6, 1, 1)) * units.Pa
    t_native = temperature[:, None, None] * np.ones((6, 1, 1)) * units.degC
    q_native = qvapor[:, None, None] * np.ones((6, 1, 1))
    h_native = height[:, None, None] * np.ones((6, 1, 1)) * units.m
    u_native = u[:, None, None] * np.ones((6, 1, 1)) * units("m/s")
    v_native = v[:, None, None] * np.ones((6, 1, 1)) * units("m/s")

    native = mcalc.compute_ecape(
        p_native, t_native, q_native, h_native, u_native, v_native,
        np.array([[100000.0]]) * units.Pa,
        np.array([[303.15]]) * units.K,
        np.array([[0.018]]),
        np.array([[5.0]]) * units("m/s"),
        np.array([[1.5]]) * units("m/s"),
        parcel_type="ml",
        storm_motion_type="bunkers_rm",
    )

    converted = mcalc.compute_ecape(
        (pressure / 100.0)[:, None, None] * np.ones((6, 1, 1)) * units.hPa,
        (temperature + 273.15)[:, None, None] * np.ones((6, 1, 1)) * units.K,
        (qvapor * 1000.0)[:, None, None] * np.ones((6, 1, 1)) * units("g/kg"),
        h_native,
        u_native,
        v_native,
        np.array([[1000.0]]) * units.hPa,
        np.array([[30.0]]) * units.degC,
        np.array([[18.0]]) * units("g/kg"),
        np.array([[5.0]]) * units("m/s"),
        np.array([[1.5]]) * units("m/s"),
        parcel_type="ml",
        storm_motion_type="bunkers_rm",
    )

    for lhs, rhs in zip(native, converted):
        np.testing.assert_allclose(lhs.m, rhs.m, rtol=0, atol=1.0e-8)


def test_compute_ecape_with_failure_mask_flags_zero_filled_columns():
    """The debug ECAPE helper exposes columns that silently zero-fill."""
    import metrust.calc as mcalc

    nan3 = np.full((2, 1, 1), np.nan)
    ecape, ncape, cape, cin, lfc, el, failure_mask = (
        mcalc.compute_ecape_with_failure_mask(
            nan3,
            nan3,
            nan3,
            nan3,
            nan3,
            nan3,
            np.array([[100000.0]]),
            np.array([[300.0]]),
            np.array([[0.014]]),
            np.array([[4.0]]),
            np.array([[1.0]]),
        )
    )

    assert failure_mask.shape == (1, 1)
    assert failure_mask.dtype == np.bool_
    assert failure_mask[0, 0]
    for field in (ecape, ncape, cape, cin, lfc, el):
        assert field.m[0, 0] == 0.0


def test_wind_functions():
    """wind_speed, wind_direction, and wind_components round-trip."""
    import metrust.calc as mcalc
    from metrust.units import units

    u = np.array([5.0, -10.0, 0.0]) * units("m/s")
    v = np.array([5.0, 0.0, -15.0]) * units("m/s")

    # wind_speed
    speed = mcalc.wind_speed(u, v)
    assert speed.units == units("m/s")
    expected_speed = np.sqrt(np.array([50.0, 100.0, 225.0]))
    np.testing.assert_allclose(speed.m, expected_speed, atol=0.01)

    # wind_direction
    wdir = mcalc.wind_direction(u, v)
    assert wdir.units == units.degree
    assert wdir.m.shape == (3,)

    # wind_components: round-trip from speed + direction
    u_rt, v_rt = mcalc.wind_components(speed, wdir)
    assert u_rt.units == units("m/s")
    assert v_rt.units == units("m/s")
    np.testing.assert_allclose(u_rt.m, u.m, atol=0.1)
    np.testing.assert_allclose(v_rt.m, v.m, atol=0.1)

    # Specific case: pure south wind (v = -10, u = 0) -> 0 or 360 deg (north)
    wdir_south = mcalc.wind_direction(
        np.array([0.0]) * units("m/s"),
        np.array([-10.0]) * units("m/s"),
    )
    # Both 0 and 360 are valid for due-north wind direction
    assert wdir_south.m[0] % 360 == pytest.approx(0.0, abs=1.0)


# ── 7. test_native_functions_exist ───────────────────────────────────

def test_native_functions_exist():
    """Key functions that should exist as callables in metrust.calc."""
    import metrust.calc as mcalc

    native_names = [
        "ccl",
        "coriolis_parameter",
        "density",
        "dewpoint",
        "divergence",
        "dry_lapse",
        "first_derivative",
        "frontogenesis",
        "heat_index",
        "isentropic_interpolation",
        "lat_lon_grid_deltas",
        "mixed_layer",
        "mixed_layer_cape_cin",
        "mixed_parcel",
        "moist_lapse",
        "most_unstable_cape_cin",
        "most_unstable_parcel",
        "parcel_profile",
        "q_vector",
        "shearing_deformation",
        "significant_tornado_parameter",
        "smooth_gaussian",
        "static_stability",
        "stretching_deformation",
        "supercell_composite_parameter",
        "surface_based_cape_cin",
        "total_deformation",
        "virtual_potential_temperature",
        "virtual_temperature_from_dewpoint",
        "vorticity",
        "wet_bulb_potential_temperature",
        "windchill",
    ]

    missing = []
    for name in native_names:
        obj = getattr(mcalc, name, None)
        if obj is None or not callable(obj):
            missing.append(name)

    assert missing == [], f"Missing or non-callable in metrust.calc: {missing}"


# ── 8. test_no_metpy_required ────────────────────────────────────────

def test_no_metpy_required():
    """Core calc module must work without MetPy."""
    import sys
    # Remove metpy from sys.modules if present to ensure clean test
    metpy_modules = [k for k in sys.modules if k.startswith("metpy")]
    saved = {k: sys.modules.pop(k) for k in metpy_modules}
    try:
        # Re-import calc to verify it loads without metpy
        import importlib
        import metrust.calc
        importlib.reload(metrust.calc)
        # Verify key functions work
        from metrust.units import units
        result = metrust.calc.potential_temperature(1000.0 * units.hPa, 25.0 * units.degC)
        assert result.magnitude > 0
        # Verify metpy was NOT imported as side effect
        for mod in sys.modules:
            assert not mod.startswith("metpy"), f"metpy module imported: {mod}"
    finally:
        # Restore metpy modules
        sys.modules.update(saved)


# ── 8b. test_calc_all_callable ─────────────────────────────────────

def test_calc_all_callable():
    """Every name in metrust.calc.__all__ must be a callable or exception."""
    import metrust.calc as mcalc
    missing = []
    for name in mcalc.__all__:
        obj = getattr(mcalc, name, None)
        if obj is None:
            missing.append(name)
        elif not (callable(obj) or (isinstance(obj, type) and issubclass(obj, Exception))):
            missing.append(f"{name} (not callable)")
    assert missing == [], f"Missing or non-callable in __all__: {missing}"


# ── 9. test_io_exports ──────────────────────────────────────────────

def test_io_exports():
    """Key I/O classes should be accessible from metrust.io."""
    import metrust.io as mio

    expected = ["Level3File", "Metar", "StationLookup", "GiniFile", "GempakGrid"]

    missing = []
    for name in expected:
        obj = getattr(mio, name, None)
        if obj is None:
            missing.append(name)

    assert missing == [], f"Missing in metrust.io: {missing}"


# ── 10. test_constants ──────────────────────────────────────────────

def test_constants():
    """Key physical constants exist and have sane values."""
    import metrust.constants as const

    # Dry-air gas constant  ~287 J/(kg K)
    assert hasattr(const, "Rd"), "Missing constant: Rd"
    assert 280 < const.Rd < 300

    # Specific heat of dry air at constant pressure  ~1004 J/(kg K)
    assert hasattr(const, "Cp_d"), "Missing constant: Cp_d"
    assert 990 < const.Cp_d < 1020

    # Gravitational acceleration  ~9.81 m/s^2
    assert hasattr(const, "g"), "Missing constant: g"
    assert 9.7 < const.g < 9.9

    # Molecular weight ratio (Mw/Md)  ~0.622
    assert hasattr(const, "epsilon"), "Missing constant: epsilon"
    assert 0.60 < const.epsilon < 0.65

    # Latent heat of vaporization  ~2.5e6 J/kg
    assert hasattr(const, "Lv"), "Missing constant: Lv"
    assert 2.4e6 < const.Lv < 2.6e6
