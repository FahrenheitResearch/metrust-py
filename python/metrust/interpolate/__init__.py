"""metrust.interpolate -- Drop-in replacement for metpy.interpolate

Provides 1-D interpolation, log-pressure interpolation, gridding
(IDW, natural-neighbor), isosurface extraction, cross-section slicing,
and observation-filtering utilities backed by the Rust metrust engine.

All functions accept plain numpy arrays *or* pint Quantity objects.
When a Quantity is passed its ``.magnitude`` is extracted automatically.
"""
import numpy as np
from metrust._metrust import interpolate as _interp


# ── Internal helpers ─────────────────────────────────────────────────────

def _mag(x):
    """Extract the magnitude from a pint Quantity, or return x unchanged."""
    return x.magnitude if hasattr(x, 'magnitude') else x


def _f64(x):
    """Convert *x* to a contiguous float64 numpy array, stripping units."""
    return np.asarray(_mag(x), dtype=np.float64)


# ── 1-D interpolation ───────────────────────────────────────────────────

def interpolate_1d(x, xp, fp):
    """Piecewise linear interpolation (like numpy.interp).

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the interpolant.
    xp : array-like
        Monotonically increasing breakpoints.
    fp : array-like
        Values at each breakpoint (same length as *xp*).

    Returns
    -------
    numpy.ndarray
    """
    return _interp.interpolate_1d(_f64(x), _f64(xp), _f64(fp))


def log_interpolate_1d(x, xp, *args):
    """Interpolation in log-pressure space (MetPy compatible).

    Performs linear interpolation in ``ln(x)`` space, which is the
    correct approach for pressure-coordinate meteorological data.
    *xp* may be ascending or descending (typical for pressure levels).

    Parameters
    ----------
    x : array-like
        Target coordinate values (e.g. pressure levels to interpolate to).
    xp : array-like
        Source coordinate values (e.g. observed pressure levels).
    *args : array-like
        One or more data arrays to interpolate (same length as *xp*).

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Single array when one data array is given, tuple otherwise.
    """
    x_arr = _f64(x)
    xp_arr = _f64(xp)
    results = []
    for fp in args:
        results.append(_interp.log_interpolate_1d(x_arr, xp_arr, _f64(fp)))
    return tuple(results) if len(results) > 1 else results[0]


# ── NaN handling ─────────────────────────────────────────────────────────

def interpolate_nans_1d(values):
    """Fill NaN gaps in a 1-D array by linear interpolation.

    Edge NaNs are filled with the nearest valid value.  If all values
    are NaN the array is returned unchanged.

    Parameters
    ----------
    values : array-like
        1-D array possibly containing NaN entries.

    Returns
    -------
    numpy.ndarray
        Array with NaN gaps filled.
    """
    return _interp.interpolate_nans_1d(_f64(values))


# ── Isosurface ───────────────────────────────────────────────────────────

def interpolate_to_isosurface(values_3d, surface_values, target,
                              levels, nx, ny, nz):
    """Interpolate a 3-D field to an isosurface of another 3-D field.

    For each ``(i, j)`` column the function walks upward through the
    *nz* levels, finds where *surface_values* crosses *target*, and
    linearly interpolates the corresponding value from *values_3d*.

    Parameters
    ----------
    values_3d : array-like
        Flattened 3-D data in ``[k*ny*nx + j*nx + i]`` order.
    surface_values : array-like
        Flattened 3-D surface field (same shape as *values_3d*).
    target : float
        Isosurface value to find.
    levels : array-like
        1-D array of length *nz* giving the coordinate at each level.
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    numpy.ndarray
        2-D result of length ``nx * ny``.  Columns without a crossing
        are filled with ``NaN``.
    """
    return _interp.interpolate_to_isosurface(
        _f64(values_3d), _f64(surface_values),
        float(_mag(target)), _f64(levels),
        int(nx), int(ny), int(nz),
    )


# ── Gridding (convenience) ──────────────────────────────────────────────

def interpolate_to_grid(x, y, z, interp_type='linear', hres=50000,
                        minimum_neighbors=3, search_radius=None,
                        rbf_func='linear'):
    """Interpolate scattered observations to a regular grid.

    This is a convenience wrapper that dispatches to either IDW or
    natural-neighbor gridding depending on *interp_type*.

    Parameters
    ----------
    x : array-like
        Longitudes (or x-coordinates) of observations.
    y : array-like
        Latitudes (or y-coordinates) of observations.
    z : array-like
        Observation values.
    interp_type : str, optional
        Interpolation method: ``'linear'`` (default, uses IDW with
        power=1), ``'natural_neighbor'``, or ``'idw'``.
    hres : float, optional
        Approximate horizontal grid spacing (metres).  Used to infer
        the output grid resolution.
    minimum_neighbors : int, optional
        Minimum number of neighbours required for an IDW estimate.
    search_radius : float or None, optional
        Search radius in degrees for IDW.  If *None* a default of 5
        degrees is used.
    rbf_func : str, optional
        Ignored (present for MetPy API compatibility).

    Returns
    -------
    numpy.ndarray
    """
    x_arr = _f64(x)
    y_arr = _f64(y)
    z_arr = _f64(z)
    return _interp.interpolate_to_grid(
        x_arr, y_arr, z_arr, int(hres), int(minimum_neighbors),
    )


# ── Vertical cross-section slice ─────────────────────────────────────────

def interpolate_to_slice(values_3d, levels, lat_slice, lon_slice,
                         src_lats, src_lons, nx, ny, nz):
    """Extract a vertical cross-section from 3-D gridded data.

    Bilinear interpolation in the horizontal plane at each path point
    and level.

    Parameters
    ----------
    values_3d : array-like
        Flattened 3-D data ``[nz, ny, nx]`` in level-major order.
    levels : array-like
        1-D coordinate array of length *nz*.
    lat_slice, lon_slice : array-like
        Latitudes and longitudes along the cross-section path.
    src_lats : array-like
        Source grid latitudes (length *ny*).
    src_lons : array-like
        Source grid longitudes (length *nx*).
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    list of list of float
        Shape ``[n_points][nz]``.
    """
    return _interp.interpolate_to_slice(
        _f64(values_3d), _f64(levels),
        _f64(lat_slice), _f64(lon_slice),
        _f64(src_lats), _f64(src_lons),
        int(nx), int(ny), int(nz),
    )


# ── Observation filtering ───────────────────────────────────────────────

def remove_nan_observations(lats, lons, values):
    """Remove observations where the value is NaN.

    Parameters
    ----------
    lats, lons, values : array-like
        Equal-length arrays of observation locations and values.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Filtered ``(lats, lons, values)`` with NaN entries dropped.
    """
    return _interp.remove_nan_observations(
        _f64(lats), _f64(lons), _f64(values),
    )


def remove_observations_below_value(lats, lons, values, threshold):
    """Remove observations where the value is below *threshold*.

    Parameters
    ----------
    lats, lons, values : array-like
        Equal-length arrays of observation locations and values.
    threshold : float
        Minimum value to keep.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Filtered ``(lats, lons, values)``.
    """
    return _interp.remove_observations_below_value(
        _f64(lats), _f64(lons), _f64(values), float(_mag(threshold)),
    )


def remove_repeat_coordinates(lats, lons, values):
    """Remove observations with duplicate (lat, lon), keeping the first.

    Parameters
    ----------
    lats, lons, values : array-like
        Equal-length arrays of observation locations and values.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        De-duplicated ``(lats, lons, values)``.
    """
    return _interp.remove_repeat_coordinates(
        _f64(lats), _f64(lons), _f64(values),
    )


# ── IDW interpolation ───────────────────────────────────────────────────

def inverse_distance_to_grid(lats, lons, values, target_grid,
                             power=2.0, min_neighbors=3,
                             search_radius=5.0):
    """Inverse-distance-weighted interpolation to a regular grid.

    Parameters
    ----------
    lats, lons, values : array-like
        Source observation locations and values.
    target_grid : GridSpec
        Target grid specification (from the Rust engine).
    power : float, optional
        Distance weighting exponent (default 2).
    min_neighbors : int, optional
        Minimum number of neighbours within the search radius (default 3).
    search_radius : float, optional
        Search radius in degrees (default 5).

    Returns
    -------
    numpy.ndarray
        Gridded values (row-major, ``ny * nx``).
    """
    return _interp.inverse_distance_to_grid(
        _f64(lats), _f64(lons), _f64(values),
        target_grid,
        float(power), int(min_neighbors), float(search_radius),
    )


def inverse_distance_to_points(src_lats, src_lons, src_values,
                               target_lats, target_lons,
                               power=2.0, min_neighbors=3,
                               search_radius=5.0):
    """Inverse-distance-weighted interpolation to arbitrary points.

    Parameters
    ----------
    src_lats, src_lons, src_values : array-like
        Source observation locations and values.
    target_lats, target_lons : array-like
        Target point locations.
    power : float, optional
        Distance weighting exponent (default 2).
    min_neighbors : int, optional
        Minimum number of neighbours within the search radius (default 3).
    search_radius : float, optional
        Search radius in degrees (default 5).

    Returns
    -------
    numpy.ndarray
    """
    return _interp.inverse_distance_to_points(
        _f64(src_lats), _f64(src_lons), _f64(src_values),
        _f64(target_lats), _f64(target_lons),
        float(power), int(min_neighbors), float(search_radius),
    )


# ── Natural-neighbor interpolation ───────────────────────────────────────

def natural_neighbor_to_grid(lats, lons, values, target_grid):
    """Approximate natural-neighbor (Sibson) interpolation to a grid.

    Uses the K nearest source points (K = min(12, n)) with inverse-
    distance-squared weights to approximate Sibson weights.

    Parameters
    ----------
    lats, lons, values : array-like
        Source observation locations and values.
    target_grid : GridSpec
        Target grid specification (from the Rust engine).

    Returns
    -------
    numpy.ndarray
        Gridded values (row-major, ``ny * nx``).
    """
    return _interp.natural_neighbor_to_grid(
        _f64(lats), _f64(lons), _f64(values), target_grid,
    )


def natural_neighbor_to_points(src_lats, src_lons, src_values,
                               target_lats, target_lons):
    """Approximate natural-neighbor interpolation to arbitrary points.

    Parameters
    ----------
    src_lats, src_lons, src_values : array-like
        Source observation locations and values.
    target_lats, target_lons : array-like
        Target point locations.

    Returns
    -------
    numpy.ndarray
    """
    return _interp.natural_neighbor_to_points(
        _f64(src_lats), _f64(src_lons), _f64(src_values),
        _f64(target_lats), _f64(target_lons),
    )


# ── Great-circle path ───────────────────────────────────────────────────

def geodesic(start, end, n_points):
    """Compute equally-spaced points along a great-circle path.

    Parameters
    ----------
    start : tuple of (float, float)
        Starting ``(lat, lon)`` in degrees.
    end : tuple of (float, float)
        Ending ``(lat, lon)`` in degrees.
    n_points : int
        Number of points (must be >= 2), including start and end.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(lats, lons)`` arrays each of length *n_points*.
    """
    return _interp.geodesic(
        (float(_mag(start[0])), float(_mag(start[1]))),
        (float(_mag(end[0])), float(_mag(end[1]))),
        int(n_points),
    )


def interpolate_to_points(src_lats, src_lons, src_values,
                          target_lats, target_lons,
                          interp_type='linear'):
    """Interpolate scattered data to arbitrary target points.

    A convenience dispatcher that selects between inverse-distance
    weighting and natural-neighbor interpolation.

    Parameters
    ----------
    src_lats, src_lons, src_values : array-like
        Source observation locations and values.
    target_lats, target_lons : array-like
        Target point locations.
    interp_type : str, optional
        Interpolation method: ``'linear'`` (default) or ``'idw'`` for
        inverse-distance weighting, ``'natural_neighbor'`` (or ``'nn'``)
        for natural-neighbor interpolation.

    Returns
    -------
    numpy.ndarray
    """
    return _interp.interpolate_to_points_dispatch(
        _f64(src_lats), _f64(src_lons), _f64(src_values),
        _f64(target_lats), _f64(target_lons),
        str(interp_type),
    )


__all__ = [
    # 1-D interpolation
    "interpolate_1d",
    "log_interpolate_1d",
    "interpolate_nans_1d",
    # 3-D / isosurface
    "interpolate_to_isosurface",
    "interpolate_to_slice",
    # Gridding convenience
    "interpolate_to_grid",
    # Point interpolation
    "interpolate_to_points",
    # IDW
    "inverse_distance_to_grid",
    "inverse_distance_to_points",
    # Natural neighbor
    "natural_neighbor_to_grid",
    "natural_neighbor_to_points",
    # Observation filtering
    "remove_nan_observations",
    "remove_observations_below_value",
    "remove_repeat_coordinates",
    # Geodesic
    "geodesic",
]
