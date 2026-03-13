"""Unit handling for metrust, built on Pint."""

import numpy as np
import pint

# Use Pint's application registry so metrust and MetPy share the same unit
# system when both are imported in the same process.
units = pint.get_application_registry()
units.autoconvert_offset_to_baseunit = True

# Ensure common meteorological aliases are registered.  degC, degF, and hPa
# are usually available in modern Pint, but we guard against older
# versions that may lack the convenience aliases.
try:
    units.degC
except (pint.errors.UndefinedUnitError, AttributeError):
    units.define("degC = kelvin; offset: 273.15 = degree_Celsius")

try:
    units.degF
except (pint.errors.UndefinedUnitError, AttributeError):
    units.define("degF = 5/9 * kelvin; offset: 255.372 = degree_Fahrenheit")

try:
    units.hPa
except (pint.errors.UndefinedUnitError, AttributeError):
    units.define("hPa = 100 * pascal = hectopascal")

try:
    units.knot
except (pint.errors.UndefinedUnitError, AttributeError):
    units.define("knot = 0.514444 * meter / second")


# ---------------------------------------------------------------------------
# Internal helpers -- not part of the public API
# ---------------------------------------------------------------------------

def _strip(quantity, target_unit):
    """Strip pint units, converting to *target_unit* first.

    If *quantity* is already a plain float / ndarray (no ``.magnitude``),
    it is returned unchanged -- this lets callers pass raw numbers when
    they know the units are already correct.
    """
    if hasattr(quantity, "magnitude"):
        return quantity.to(target_unit).magnitude
    return quantity


def _strip_or_none(quantity, target_unit):
    """Like ``_strip`` but passes ``None`` through unchanged."""
    if quantity is None:
        return None
    return _strip(quantity, target_unit)


def _attach(value, unit_str):
    """Attach pint units to a bare value."""
    return units.Quantity(value, unit_str)


def _as_float(v):
    """Ensure scalar -- extract .item() from 0-d numpy arrays."""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return float(v.item())
    return float(v)


def _as_1d(v):
    """Ensure a contiguous float64 1-D numpy array."""
    return np.ascontiguousarray(v, dtype=np.float64).ravel()
