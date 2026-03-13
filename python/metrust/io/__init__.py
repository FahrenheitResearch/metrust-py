"""metrust.io -- Drop-in replacement for metpy.io

Provides Level3File, Metar, StationInfo, StationLookup, and METAR parsing
functions backed by the Rust metrust engine.
"""
from metrust._metrust import io as _io

# ── Classes ──────────────────────────────────────────────────────────────

Level3File = _io.Level3File
Metar = _io.Metar
StationInfo = _io.StationInfo
StationLookup = _io.StationLookup

# ── Functions ────────────────────────────────────────────────────────────

def parse_metar(text):
    """Parse a single METAR observation string.

    Parameters
    ----------
    text : str
        Raw METAR text (may start with "METAR" or "SPECI").

    Returns
    -------
    Metar
        Parsed observation with attributes for station, wind, visibility,
        temperature, dewpoint, altimeter, sky cover, and weather phenomena.
    """
    return _io.parse_metar(text)


def parse_metar_file(content):
    """Parse a multi-line string containing one METAR per line.

    Blank lines and lines starting with '#' are skipped.  Returns all
    successfully parsed METARs (silently ignoring unparseable lines).

    Parameters
    ----------
    content : str
        Multi-line string with one METAR per line.

    Returns
    -------
    list of Metar
    """
    return _io.parse_metar_file(content)


__all__ = [
    "Level3File",
    "Metar",
    "StationInfo",
    "StationLookup",
    "parse_metar",
    "parse_metar_file",
]
