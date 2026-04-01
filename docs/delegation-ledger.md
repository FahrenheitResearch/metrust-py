# Optional MetPy Delegation Ledger

`metrust.calc` is Rust-first by default. This page is the explicit ledger for
any remaining `metpy.calc` delegation behavior.

Current compatibility target:

- MetPy `1.7.1`
- Python `3.10` to `3.13`
- CI modes: `metrust-only`, `metrust + MetPy`, `metrust + MetPy + xarray`

## Calc Delegations

There are currently no optional `metrust.calc` delegations. The shared
`metpy.calc` surface now stays on native metrust implementations even when
MetPy is installed.

## CI Guarantees

- `tests/test_delegation_ledger.py` verifies the ledger stays in sync with the code.
- The same test file verifies native calc paths still do not call into MetPy when it is installed.
- Differential CI runs the relevant suites against pinned MetPy `1.7.1`.
