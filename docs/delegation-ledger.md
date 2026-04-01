# Optional MetPy Delegation Ledger

`metrust.calc` is Rust-first by default, but a small parity-sensitive subset can
delegate to `metpy.calc` when MetPy is installed. This page is the explicit
ledger for that behavior.

Current compatibility target:

- MetPy `1.7.1`
- Python `3.10` to `3.13`
- CI modes: `metrust-only`, `metrust + MetPy`, `metrust + MetPy + xarray`

## Calc Delegations

| Function | Delegates When | Local Fallback |
|---|---|---|
| `lfc` | Quantity profile inputs, especially MetPy's more complex `which=` handling | metrust native profile intersection solver |
| `el` | Quantity profile inputs, especially MetPy's more complex `which=` handling | metrust native profile intersection solver |
| `cape_cin` | MetPy parcel-profile form where the 4th positional argument is temperature-like | metrust native CAPE/CIN integration |
| `downdraft_cape` | Quantity profile inputs when MetPy is available | metrust native DCAPE layer selection and integration |
| `parcel_profile_with_lcl` | Quantity profile inputs in the MetPy profile-returning form | metrust native interpolation and parcel-trace construction |
| `potential_vorticity_baroclinic` | Quantity/DataArray inputs in the MetPy-style baroclinic-PV form | metrust native PV computation with local dx/dy and latitude handling |
| `geospatial_laplacian` | Quantity/DataArray geospatial-laplacian inputs when MetPy is available | metrust native geospatial gradient and derivative path |

## CI Guarantees

- `tests/test_delegation_ledger.py` verifies the ledger stays in sync with the code.
- The same test file verifies delegated functions still run with MetPy blocked.
- Differential CI runs the relevant suites against pinned MetPy `1.7.1`.
