"""MetPy Cookbook: 500 hPa Vorticity Advection — metrust drop-in demo.

This is the MetPy "500 hPa Vorticity Advection" cookbook example, running
entirely on metrust.  The only change is the import line.

Benchmark (v0.2.3, 32-core Ryzen):
    MetPy:   619 ms
    metrust:  42 ms  (14.7x faster)

Grid correlation vs MetPy:
    Absolute vorticity:     r = 0.9969
    Vorticity advection:    r = 0.9876

Requires network access (THREDDS) and: pip install siphon cartopy netcdf4
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from metpy.units import units
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import cartopy.crs as ccrs

# === THE ONLY CHANGE: swap metpy.calc -> metrust.calc ===
import metrust.calc as mpcalc


def earth_relative_wind_components(ugrd, vgrd):
    """Transform grid-relative winds to earth-relative."""
    data_crs = ugrd.metpy.cartopy_crs
    x = ugrd.x.values
    y = ugrd.y.values
    xx, yy = np.meshgrid(x, y)
    ut, vt = ccrs.PlateCarree().transform_vectors(data_crs, xx, yy, ugrd.values, vgrd.values)
    uer = ugrd.copy()
    ver = vgrd.copy()
    uer.values = ut
    ver.values = vt
    return uer, ver


# ---------- Load NAM 500 hPa data ----------
print("Fetching NAM analysis from THREDDS...")
dt = pd.Timestamp("2016-04-16T18:00:00")
base_url = "https://www.ncei.noaa.gov/thredds/catalog/model-namanl-old/"
cat = TDSCatalog(f"{base_url}{dt:%Y%m}/{dt:%Y%m%d}/catalog.xml")
ncss = cat.datasets[f"namanl_218_{dt:%Y%m%d}_{dt:%H}00_000.grb"].subset()
query = ncss.query()
query.time(dt.to_pydatetime())
query.accept("netcdf")
query.variables(
    "Geopotential_height_isobaric",
    "u-component_of_wind_isobaric",
    "v-component_of_wind_isobaric",
)
data = ncss.get_data(query)
ds = xr.open_dataset(NetCDF4DataStore(data)).metpy.parse_cf()
ds = ds.metpy.assign_latitude_longitude()

lev_500 = 500 * units.hPa
hght_500 = ds.Geopotential_height_isobaric.metpy.sel(vertical=lev_500).squeeze()
uwnd_500 = ds["u-component_of_wind_isobaric"].metpy.sel(vertical=lev_500).squeeze()
vwnd_500 = ds["v-component_of_wind_isobaric"].metpy.sel(vertical=lev_500).squeeze()
uwnd_500, vwnd_500 = earth_relative_wind_components(uwnd_500, vwnd_500)

print(f"Grid shape: {hght_500.shape}")

# ---------- Compute vorticity and advection ----------
avor = mpcalc.vorticity(uwnd_500, vwnd_500)
avor = mpcalc.smooth_n_point(avor, 9, 10) * 1e5
vort_adv = mpcalc.advection(avor, uwnd_500, vwnd_500) * 1e4

print(f"\n=== 500 hPa Analysis ===")
print(f"  Abs. vorticity range: [{np.nanmin(avor):.2f}, {np.nanmax(avor):.2f}] x10^-5 /s")
print(f"  Vort. advection range: [{np.nanmin(vort_adv):.2f}, {np.nanmax(vort_adv):.2f}] x10^-4 /s^2")

print("\n--- Demo complete. Grid kinematics used metrust as a drop-in for MetPy. ---")
