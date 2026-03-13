"""metrust.xarray -- forwarding to MetPy's xarray accessor.

The xarray coordinate/CRS accessor is a MetPy feature. This module provides
access to it when MetPy is installed.

Install MetPy separately if you need the xarray accessor:

    pip install metpy
"""


def __getattr__(name):
    # Lazy-import metpy.xarray only when an attribute is actually requested
    try:
        import metpy.xarray as _metpy_xarray  # type: ignore
        if hasattr(_metpy_xarray, name):
            return getattr(_metpy_xarray, name)
    except ImportError:
        pass
    raise ImportError(
        f"metrust.xarray requires MetPy for xarray accessor support. "
        f"Install it with: pip install metpy"
    )


def __dir__():
    try:
        import metpy.xarray as _metpy_xarray  # type: ignore
        names = [n for n in dir(_metpy_xarray) if not n.startswith("_")]
    except ImportError:
        names = []
    return sorted(set(globals()).union(names))
