"""metrust.constants -- Drop-in replacement for metpy.constants

All physical constants are re-exported from the Rust metrust engine as plain
float values using SI base units unless otherwise noted.  The Rust module
exposes every constant that MetPy provides, including both long descriptive
names (e.g. ``earth_gravity``) and short aliases (e.g. ``g``).

Usage::

    from metrust.constants import earth_gravity, Rd, epsilon
    from metrust import constants as mpconsts
    print(mpconsts.g)  # 9.80665
"""
from metrust._metrust import constants as _constants
import sys as _sys

# Transparently re-export every attribute from the Rust constants submodule
# so that ``from metrust.constants import <name>`` and tab-completion both
# work exactly as they do with metpy.constants.
_this = _sys.modules[__name__]

for _name in dir(_constants):
    if not _name.startswith('_'):
        setattr(_this, _name, getattr(_constants, _name))

# Clean up helper variables so they don't pollute the namespace
del _this, _sys


def __getattr__(name):
    """Fall through to the Rust module for any attribute not yet copied."""
    try:
        return getattr(_constants, name)
    except AttributeError:
        raise AttributeError(
            f"module 'metrust.constants' has no attribute {name!r}"
        ) from None
