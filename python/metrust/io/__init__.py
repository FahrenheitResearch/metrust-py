"""metrust.io -- Rust-native meteorological file I/O."""

from metrust._metrust import io as _io

_RUST_EXPORTS = [
    "Level3File",
    "Metar",
    "StationInfo",
    "StationLookup",
    "GiniFile",
    "GempakGrid",
    "GempakGridRecord",
    "GempakSounding",
    "GempakSoundingStation",
    "SoundingData",
    "GempakSurface",
    "GempakSurfaceStation",
    "SurfaceObs",
    "SurfaceBulletinFeature",
    "parse_metar",
    "parse_metar_file",
    "parse_wpc_surface_bulletin",
    "is_precip_mode",
]

for _name in _RUST_EXPORTS:
    if hasattr(_io, _name):
        globals()[_name] = getattr(_io, _name)


def __getattr__(name):
    if hasattr(_io, name):
        return getattr(_io, name)
    # Level2File is not yet implemented in Rust -- lazy-load from MetPy
    if name == "Level2File":
        try:
            import metpy.io as _metpy_io  # type: ignore
            if hasattr(_metpy_io, "Level2File"):
                globals()["Level2File"] = _metpy_io.Level2File
                return _metpy_io.Level2File
        except ImportError:
            pass
        raise ImportError(
            "Level2File is not yet implemented natively. "
            "Install MetPy for Level2 radar support: pip install metpy"
        )
    raise AttributeError(f"module 'metrust.io' has no attribute {name!r}")


__all__ = sorted(
    set(_RUST_EXPORTS).union({"Level2File"})
)


def __dir__():
    return sorted(set(globals()).union(__all__))
