"""metrust.plots -- Plotting utilities.

Note: MetPy's plots module relies heavily on matplotlib.  metrust provides
rendering primitives (skew-T, hodograph, colormaps, contours) but the actual
rendering happens in Rust.  The Python interface to these is provided as
utility functions rather than matplotlib-integrated classes.

Users who need full matplotlib integration should use metpy.plots directly
and feed it data computed by metrust.calc.
"""


def __getattr__(name):
    raise AttributeError(
        f"metrust.plots.{name} is not available. "
        f"metrust performs rendering natively in Rust. "
        f"For matplotlib-based plots, use metpy.plots with data from metrust.calc."
    )
