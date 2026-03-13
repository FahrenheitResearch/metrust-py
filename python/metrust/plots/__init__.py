"""metrust.plots -- forwarding to MetPy's plotting module.

Plotting is MetPy's domain. This module provides access to MetPy's plotting
classes (StationPlot, SkewT, Hodograph, etc.) when MetPy is installed.

Install MetPy separately if you need plotting:

    pip install metpy
"""


def __getattr__(name):
    # Lazy-import metpy.plots only when an attribute is actually requested
    try:
        import metpy.plots as _metpy_plots  # type: ignore
        if hasattr(_metpy_plots, name):
            return getattr(_metpy_plots, name)
    except ImportError:
        pass
    raise ImportError(
        f"metrust.plots requires MetPy for matplotlib integration. "
        f"Install it with: pip install metpy"
    )


def __dir__():
    try:
        import metpy.plots as _metpy_plots  # type: ignore
        names = [n for n in dir(_metpy_plots) if not n.startswith("_")]
    except ImportError:
        names = []
    return sorted(set(globals()).union(names))
