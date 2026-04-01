# Installation

## From PyPI (recommended)

Pre-built wheels are available for **Python 3.10+** on Windows, macOS, and Linux:

```bash
pip install metrust
```

This installs the compiled Rust extension along with the required Python
dependencies. No Rust toolchain is needed when installing from a wheel.

## From source

Building from source requires a **Rust toolchain** (1.70+) and **Python 3.10+**.

```bash
git clone https://github.com/FahrenheitResearch/metrust-py.git
cd metrust-py
pip install -e .
```

The editable install uses [maturin](https://github.com/PyO3/maturin) as the
build backend. Maturin compiles the Rust crate and links the resulting shared
library into the Python package. On first build this may take a minute or two
depending on your hardware.

!!! note "Rust toolchain"
    Install Rust via [rustup](https://rustup.rs/) if you do not already have it.
    After installation, confirm with `rustc --version`. Any stable release from
    1.70 onward should work.

If you need to trigger a manual rebuild after editing Rust code:

```bash
maturin develop --release
```

## Optional: MetPy integration

The core `metrust.calc` module is fully native Rust and does **not** require
MetPy. However, a few convenience surfaces forward to MetPy when it is
installed:

- `metrust.plots` -- delegates to `metpy.plots`
- `metrust.xarray` -- delegates to the MetPy xarray accessor
- `metrust.io.Level2File` -- delegates to MetPy's Level-II radar reader

To enable these, install MetPy alongside metrust:

```bash
pip install metpy
```

MetPy is not listed as a dependency and will never be pulled in automatically.
Everything in `metrust.calc`, `metrust.units`, and the native I/O readers
works without it.

## Dependencies

The following packages are installed automatically with `metrust`:

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `numpy` | >= 1.20 | Array backend for all calculations |
| `pint`  | >= 0.20 | Physical units and quantity support |

No other runtime dependencies are required.

## Verifying the installation

After installing, confirm that the package loads and the native extension is
linked correctly:

```bash
python -c "import metrust; print(metrust.__version__)"
```

This should print the installed version (e.g., `0.1.8`). If the import fails
with a missing shared library error, the Rust extension was not compiled -- see
the [from source](#from-source) instructions above.

A quick smoke test of the calculation layer:

```python
from metrust.calc import potential_temperature
from metrust.units import units

theta = potential_temperature(850 * units.hPa, 25 * units.degC)
print(theta)  # ~298.9 K
```

## Troubleshooting

**`pip install metrust` fails with "no matching distribution"**
:   Pre-built wheels may not be available for your platform or Python version.
    Ensure you are running Python 3.10 or later (`python --version`) and that
    your pip is up to date (`pip install --upgrade pip`). If no wheel exists
    for your platform, pip will attempt a source build, which requires the Rust
    toolchain (see above).

**Source build fails at the Rust compilation step**
:   Make sure `rustc` and `cargo` are on your `PATH`. Run `rustup update` to
    ensure you have a recent stable toolchain. On Linux you may also need
    `python3-dev` (or `python3-devel`) for the Python headers.

**`import metrust` raises `ImportError: dynamic module does not define init function`**
:   This usually means the compiled extension was built for a different Python
    version. Reinstall with `pip install --force-reinstall metrust` or rebuild
    from source with the correct interpreter active.
