# Verification

Current verification is split into two layers:

1. Rust implementation correctness in the workspace
2. Python compatibility checks for the MetPy-facing API

## Current Automated Checks

As of 2026-03-13, the repo passes:

- `cargo test --workspace`
- `python -m pytest tests/test_python_compat.py -q`

The Rust workspace covers the core native implementation, including thermodynamics, wind, kinematics, smoothing, interpolation, and the current GEMPAK/GINI test fixtures. The Python compatibility test file is intentionally small and targeted: it checks the issues most likely to break MetPy-style usage.

## Python Compatibility Coverage

`tests/test_python_compat.py` currently verifies:

- Offset-temperature Pint quantities such as `20 * units.degC`
- Wrapper return units for key thermodynamic functions
- MetPy-style function calls like `cape_cin(..., parcel_profile=...)`
- Public `metrust.io` exports, including fallback-only surfaces
- `metrust.plots` and `metrust.xarray` forwarding behavior

These assertions are the authoritative CI gate for the Python compatibility layer.

## Exploratory Comparison Scripts

The repo also contains the older `tests/verify_*.py` scripts:

- `tests/verify_thermo.py`
- `tests/verify_wind.py`
- `tests/verify_kinematics.py`
- `tests/verify_severe_atmo.py`
- `tests/verify_smooth_interp.py`
- `tests/verify_constants.py`
- `tests/verify_units.py`
- `tests/verify_edge_cases.py`

Those scripts remain useful for deeper spot checks and investigation, but they are not a substitute for assert-based automated tests. Treat them as reference tooling rather than the source of truth for CI status.

## Known Limits of Verification

Passing checks here does not mean full package-level parity with MetPy.

- `metrust.plots`, `metrust.xarray`, and `metrust.io.Level2File` forward to MetPy when installed. Core calc functions are 100% native.
- Several calculations are designed for close agreement, not bit-for-bit identity
- `moist_lapse` still needs more scrutiny before being considered high-confidence parity work
- Plotting and xarray support are compatibility shims, not native reimplementations

## How To Run

Recommended local setup:

```bash
python -m pip install -e .
python -m pip install pytest numpy pint
```

Run the automated checks:

```bash
cargo test --workspace
python -m pytest tests/test_python_compat.py -q
```

Run the optional exploratory scripts:

```bash
python tests/verify_thermo.py
python tests/verify_wind.py
python tests/verify_kinematics.py
python tests/verify_severe_atmo.py
python tests/verify_smooth_interp.py
python tests/verify_constants.py
python tests/verify_units.py
python tests/verify_edge_cases.py
```
