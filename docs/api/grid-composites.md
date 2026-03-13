# Grid Composites API Reference

Grid composite functions are the flagship capability of metrust. They process
entire 3-D model grids -- thousands to millions of vertical columns -- in a
single call, parallelized across all CPU cores via Rust and
[rayon](https://github.com/rayon-rs/rayon). MetPy has no equivalent; producing
the same fields with MetPy requires writing Python loops over every grid column
and is orders of magnitude slower.

All functions live in `metrust.calc`:

```python
from metrust.calc import (
    compute_cape_cin, compute_srh, compute_shear,
    compute_lapse_rate, compute_pw,
    compute_stp, compute_scp, compute_ehi, compute_ship,
    compute_dcp, compute_grid_scp, compute_grid_critical_angle,
    composite_reflectivity, composite_reflectivity_from_hydrometeors,
)
```

## Conventions

| Convention | Details |
|---|---|
| **3-D array ordering** | `(nz, ny, nx)` -- vertical levels first, then latitude, then longitude. This matches the native layout of HRRR, RAP, NAM, and WRF output. |
| **2-D array ordering** | `(ny, nx)` |
| **Pint quantities** | Every array and scalar argument accepts either a bare NumPy array / Python float or a `pint.Quantity`. When a Pint quantity is passed, units are stripped automatically (and converted to the expected unit for scalar args such as `top_m`). |
| **Return values** | All return values carry Pint units. You can strip them with `.magnitude` or use them directly in arithmetic and plotting. |
| **dtype** | Inputs are coerced to `float64` contiguous arrays internally. |

---

## 3-D Field to 2-D Result (Rust-parallel)

These functions accept full 3-D model grids and reduce each vertical column in
parallel to produce a 2-D output field.

---

### `compute_cape_cin`

Convective Available Potential Energy and Convective Inhibition for every grid
column, computed in parallel. Returns four 2-D fields: CAPE, CIN, LCL height,
and LFC height.

```python
compute_cape_cin(
    pressure_3d,
    temperature_c_3d,
    qvapor_3d,
    height_agl_3d,
    psfc,
    t2,
    q2,
    parcel_type="surface",
    top_m=None,
)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `pressure_3d` | `(nz, ny, nx)` | Pa | 3-D pressure field |
| `temperature_c_3d` | `(nz, ny, nx)` | degrees Celsius | 3-D temperature field |
| `qvapor_3d` | `(nz, ny, nx)` | kg/kg | 3-D water vapor mixing ratio |
| `height_agl_3d` | `(nz, ny, nx)` | m | 3-D height above ground level |
| `psfc` | `(ny, nx)` | Pa | Surface pressure |
| `t2` | `(ny, nx)` | K | 2-meter temperature |
| `q2` | `(ny, nx)` | kg/kg | 2-meter water vapor mixing ratio |
| `parcel_type` | scalar | -- | `"surface"`, `"mixed_layer"`, or `"most_unstable"` |
| `top_m` | scalar or `None` | m | Integration cap in meters AGL. `None` integrates through the full column. Accepts Pint: `top_m=3000 * units.m`. |

**Returns**

A tuple of four 2-D arrays, each shaped `(ny, nx)`:

| Index | Field | Units |
|---|---|---|
| 0 | CAPE | J/kg |
| 1 | CIN | J/kg |
| 2 | LCL height | m |
| 3 | LFC height | m |

**Example**

```python
import xarray as xr
from metrust.calc import compute_cape_cin

ds = xr.open_dataset("hrrr.t00z.wrfnatf12.grib2", engine="cfgrib")

cape, cin, lcl_hgt, lfc_hgt = compute_cape_cin(
    pressure_3d=ds["P"].values + ds["PB"].values,   # full pressure, Pa
    temperature_c_3d=ds["T"].values - 273.15,        # convert K -> C
    qvapor_3d=ds["QVAPOR"].values,                   # kg/kg
    height_agl_3d=ds["height_agl"].values,            # m AGL
    psfc=ds["PSFC"].values,                           # Pa
    t2=ds["T2"].values,                               # K
    q2=ds["Q2"].values,                               # kg/kg
    parcel_type="mixed_layer",
)

print(cape.shape)  # (ny, nx)
print(cape.max())  # e.g. 3500 J/kg
```

---

### `compute_srh`

Storm-Relative Helicity integrated from the surface to `top_m` for every grid
column. SRH quantifies the potential for rotating updrafts.

```python
compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `u_3d` | `(nz, ny, nx)` | m/s | 3-D u-component of wind |
| `v_3d` | `(nz, ny, nx)` | m/s | 3-D v-component of wind |
| `height_agl_3d` | `(nz, ny, nx)` | m | 3-D height above ground level |
| `top_m` | scalar | m | Integration depth in meters AGL (default `1000.0` = 0--1 km). Use `3000.0` for 0--3 km SRH. Accepts Pint: `top_m=1 * units.km`. |

**Returns**

`ndarray` shaped `(ny, nx)` in m^2/s^2.

**Example**

```python
from metrust.calc import compute_srh

srh_1km = compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0)
srh_3km = compute_srh(u_3d, v_3d, height_agl_3d, top_m=3000.0)
```

---

### `compute_shear`

Bulk wind shear magnitude between two height layers for every grid column.

```python
compute_shear(u_3d, v_3d, height_agl_3d, bottom_m=0.0, top_m=6000.0)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `u_3d` | `(nz, ny, nx)` | m/s | 3-D u-component of wind |
| `v_3d` | `(nz, ny, nx)` | m/s | 3-D v-component of wind |
| `height_agl_3d` | `(nz, ny, nx)` | m | 3-D height above ground level |
| `bottom_m` | scalar | m | Bottom of the shear layer in meters AGL (default `0.0`). Accepts Pint. |
| `top_m` | scalar | m | Top of the shear layer in meters AGL (default `6000.0`). Accepts Pint. |

**Returns**

`ndarray` shaped `(ny, nx)` in m/s (shear magnitude).

**Example**

```python
from metrust.calc import compute_shear
from metpy.units import units

shear_06 = compute_shear(u_3d, v_3d, height_agl_3d)
shear_01 = compute_shear(u_3d, v_3d, height_agl_3d, top_m=1 * units.km)
```

---

### `compute_lapse_rate`

Environmental lapse rate between two height layers for every grid column.

```python
compute_lapse_rate(
    temperature_c_3d,
    qvapor_3d,
    height_agl_3d,
    bottom_km=0.0,
    top_km=3.0,
)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `temperature_c_3d` | `(nz, ny, nx)` | degrees Celsius | 3-D temperature field |
| `qvapor_3d` | `(nz, ny, nx)` | kg/kg | 3-D water vapor mixing ratio |
| `height_agl_3d` | `(nz, ny, nx)` | m | 3-D height above ground level |
| `bottom_km` | scalar | km | Bottom of the layer in km AGL (default `0.0`). Accepts Pint. |
| `top_km` | scalar | km | Top of the layer in km AGL (default `3.0`). Accepts Pint. |

**Returns**

`ndarray` shaped `(ny, nx)` in degC/km.

**Example**

```python
from metrust.calc import compute_lapse_rate

lr_03 = compute_lapse_rate(temp_c_3d, qvapor_3d, height_agl_3d)
lr_700_500 = compute_lapse_rate(
    temp_c_3d, qvapor_3d, height_agl_3d,
    bottom_km=3.0, top_km=5.5,
)
```

---

### `compute_pw`

Precipitable water (column-integrated water vapor) for every grid column.

```python
compute_pw(qvapor_3d, pressure_3d)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `qvapor_3d` | `(nz, ny, nx)` | kg/kg | 3-D water vapor mixing ratio |
| `pressure_3d` | `(nz, ny, nx)` | Pa | 3-D pressure field |

**Returns**

`ndarray` shaped `(ny, nx)` in mm.

**Example**

```python
from metrust.calc import compute_pw

pw = compute_pw(qvapor_3d, pressure_3d)
print(pw.max())  # e.g. 55.3 mm
```

---

## 2-D Composite Parameters

These functions combine pre-computed 2-D fields (the outputs of the 3-D
functions above, or fields read directly from model output) into severe-weather
composite indices. All inputs and outputs are shaped `(ny, nx)`.

---

### `compute_stp`

Significant Tornado Parameter (fixed-layer formulation).

```python
compute_stp(cape, lcl_height, srh_1km, shear_6km)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `cape` | `(ny, nx)` | J/kg | MLCAPE (Mixed-Layer CAPE) |
| `lcl_height` | `(ny, nx)` | m AGL | LCL height above ground |
| `srh_1km` | `(ny, nx)` | m^2/s^2 | 0--1 km storm-relative helicity |
| `shear_6km` | `(ny, nx)` | m/s | 0--6 km bulk wind shear |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless. Values above 1 indicate an
environment favorable for significant tornadoes; values above 4--5 indicate a
particularly dangerous setup.

**Example**

```python
from metrust.calc import compute_stp

stp = compute_stp(cape, lcl_hgt, srh_1km, shear_06)
```

---

### `compute_scp`

Supercell Composite Parameter.

```python
compute_scp(mucape, srh_3km, shear_6km)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `mucape` | `(ny, nx)` | J/kg | Most-Unstable CAPE |
| `srh_3km` | `(ny, nx)` | m^2/s^2 | 0--3 km storm-relative helicity |
| `shear_6km` | `(ny, nx)` | m/s | 0--6 km bulk wind shear |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless. Values above 1 favor supercell
development.

**Example**

```python
from metrust.calc import compute_scp

scp = compute_scp(mucape, srh_3km, shear_06)
```

---

### `compute_ehi`

Energy-Helicity Index.

```
EHI = (CAPE * SRH) / 160,000
```

```python
compute_ehi(cape, srh)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `cape` | `(ny, nx)` | J/kg | CAPE |
| `srh` | `(ny, nx)` | m^2/s^2 | Storm-relative helicity |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless. Values above 1 suggest an
environment capable of producing significant tornadoes.

**Example**

```python
from metrust.calc import compute_ehi

ehi = compute_ehi(cape, srh_1km)
```

---

### `compute_ship`

Significant Hail Parameter. Identifies environments favorable for significant
(2-inch+) hail.

```python
compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `cape` | `(ny, nx)` | J/kg | MUCAPE |
| `shear06` | `(ny, nx)` | m/s | 0--6 km bulk wind shear |
| `t500` | `(ny, nx)` | degrees Celsius | 500 hPa temperature |
| `lr_700_500` | `(ny, nx)` | degC/km | 700--500 hPa lapse rate |
| `mixing_ratio_gkg` | `(ny, nx)` | g/kg | Low-level mixing ratio |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless. Values above 1 favor significant
hail; values above 4 indicate extreme hail environments.

**Example**

```python
from metrust.calc import compute_ship

ship = compute_ship(mucape, shear_06, t_500, lr_700_500, mr_gkg)
```

---

### `compute_dcp`

Derecho Composite Parameter. Identifies environments favorable for
long-lived damaging wind events (derechos).

```
DCP = (DCAPE / 980) * (MUCAPE / 2000) * (SHEAR_06 / 20) * (MU_MR / 11)
```

```python
compute_dcp(dcape, mu_cape, shear06, mu_mixing_ratio)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `dcape` | `(ny, nx)` | J/kg | Downdraft CAPE |
| `mu_cape` | `(ny, nx)` | J/kg | Most-Unstable CAPE |
| `shear06` | `(ny, nx)` | m/s | 0--6 km bulk wind shear |
| `mu_mixing_ratio` | `(ny, nx)` | g/kg | Most-unstable parcel mixing ratio |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless. Values above 2 favor derecho-type
events.

**Example**

```python
from metrust.calc import compute_dcp

dcp = compute_dcp(dcape, mu_cape, shear_06, mu_mr)
```

---

### `compute_grid_scp`

Enhanced Supercell Composite Parameter with a CIN term. This variant adds CIN
as a modifier to the standard SCP formulation, penalizing environments where
convective initiation is inhibited.

```
SCP = (MUCAPE / 1000) * (SRH / 50) * (SHEAR_06 / 40) * CIN_term
```

```python
compute_grid_scp(mu_cape, srh, shear_06, mu_cin)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `mu_cape` | `(ny, nx)` | J/kg | Most-Unstable CAPE |
| `srh` | `(ny, nx)` | m^2/s^2 | Storm-relative helicity (0--3 km) |
| `shear_06` | `(ny, nx)` | m/s | 0--6 km bulk wind shear |
| `mu_cin` | `(ny, nx)` | J/kg | Most-Unstable CIN |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless.

**Example**

```python
from metrust.calc import compute_grid_scp

scp_cin = compute_grid_scp(mu_cape, srh_3km, shear_06, mu_cin)
```

---

### `compute_grid_critical_angle`

Critical angle between the storm-relative inflow vector and the low-level
shear vector at every grid point. Values near 90 degrees are most favorable
for tornadogenesis.

```python
compute_grid_critical_angle(u_storm, v_storm, u_shear, v_shear)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `u_storm` | `(ny, nx)` | m/s | Storm-motion u-component |
| `v_storm` | `(ny, nx)` | m/s | Storm-motion v-component |
| `u_shear` | `(ny, nx)` | m/s | Low-level shear u-component |
| `v_shear` | `(ny, nx)` | m/s | Low-level shear v-component |

**Returns**

`ndarray` shaped `(ny, nx)` in degrees (0--180).

**Example**

```python
from metrust.calc import compute_grid_critical_angle

crit_angle = compute_grid_critical_angle(u_storm, v_storm, u_shr, v_shr)
```

---

## Reflectivity

---

### `composite_reflectivity`

Column-maximum reflectivity from a 3-D reflectivity field (e.g., the `REFL_10CM`
variable in WRF/HRRR output).

```python
composite_reflectivity(refl_3d)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `refl_3d` | `(nz, ny, nx)` | dBZ | 3-D reflectivity |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless (dBZ values).

**Example**

```python
from metrust.calc import composite_reflectivity

comp_refl = composite_reflectivity(ds["REFL_10CM"].values)
```

---

### `composite_reflectivity_from_hydrometeors`

Compute composite reflectivity from raw hydrometeor mixing ratios when a
reflectivity field is not directly available in the model output. The
function synthesizes reflectivity from rain, snow, and graupel mixing
ratios using Smith (1984) Z--M relationships, then returns the column
maximum.

```python
composite_reflectivity_from_hydrometeors(
    pressure_3d,
    temperature_c_3d,
    qrain_3d,
    qsnow_3d,
    qgraup_3d,
)
```

**Parameters**

| Parameter | Shape | Units | Description |
|---|---|---|---|
| `pressure_3d` | `(nz, ny, nx)` | Pa | 3-D pressure field |
| `temperature_c_3d` | `(nz, ny, nx)` | degrees Celsius | 3-D temperature field |
| `qrain_3d` | `(nz, ny, nx)` | kg/kg | Rain mixing ratio |
| `qsnow_3d` | `(nz, ny, nx)` | kg/kg | Snow mixing ratio |
| `qgraup_3d` | `(nz, ny, nx)` | kg/kg | Graupel mixing ratio |

**Returns**

`ndarray` shaped `(ny, nx)`, dimensionless (dBZ values).

**Example**

```python
from metrust.calc import composite_reflectivity_from_hydrometeors

comp_refl = composite_reflectivity_from_hydrometeors(
    pressure_3d=ds["P"].values + ds["PB"].values,
    temperature_c_3d=ds["T"].values - 273.15,
    qrain_3d=ds["QRAIN"].values,
    qsnow_3d=ds["QSNOW"].values,
    qgraup_3d=ds["QGRAUP"].values,
)
```

---

## Pint Unit Handling

All functions in this module accept Pint quantities transparently. Arrays are
stripped of their units via `.magnitude` before being passed to the Rust layer.
Scalar keyword arguments (such as `top_m`, `bottom_km`) are converted to the
expected unit before stripping. This means all of the following work
identically:

```python
from metpy.units import units

# Bare float -- caller is responsible for correct unit
srh = compute_srh(u_3d, v_3d, height_agl_3d, top_m=3000.0)

# Pint quantity in the expected unit
srh = compute_srh(u_3d, v_3d, height_agl_3d, top_m=3000.0 * units.m)

# Pint quantity in a different unit -- converted automatically
srh = compute_srh(u_3d, v_3d, height_agl_3d, top_m=3 * units.km)
```

Return values always carry Pint units. Strip them for plotting or when passing
to libraries that do not understand Pint:

```python
cape_values = cape.magnitude          # bare ndarray
cape_values = cape.m                  # shorthand
```

---

## End-to-End Example: Computing a Severe Weather Composite from HRRR Data

This example demonstrates the full workflow: read HRRR model output, compute
the 3-D-to-2-D derived fields in parallel, combine them into an STP map, and
plot the result.

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.units import units
from metrust.calc import (
    compute_cape_cin,
    compute_srh,
    compute_shear,
    compute_stp,
)

# ---- 1. Load HRRR data ----

ds = xr.open_dataset("hrrr.t18z.wrfnatf06.grib2", engine="cfgrib")

# Extract 3-D fields (nz, ny, nx)
pressure_3d   = ds["P"].values + ds["PB"].values      # Pa
temperature_c = ds["T"].values - 273.15                # K -> C
qvapor_3d     = ds["QVAPOR"].values                    # kg/kg
height_agl_3d = ds["height_agl"].values                # m AGL
u_3d          = ds["U"].values                         # m/s
v_3d          = ds["V"].values                         # m/s

# Extract 2-D surface fields (ny, nx)
psfc = ds["PSFC"].values   # Pa
t2   = ds["T2"].values     # K
q2   = ds["Q2"].values     # kg/kg

# Latitude/longitude for plotting
lats = ds["XLAT"].values
lons = ds["XLONG"].values

# ---- 2. Compute 3-D -> 2-D fields (Rust-parallel) ----

# CAPE, CIN, LCL height, LFC height -- mixed-layer parcel
cape, cin, lcl_hgt, lfc_hgt = compute_cape_cin(
    pressure_3d, temperature_c, qvapor_3d, height_agl_3d,
    psfc, t2, q2,
    parcel_type="mixed_layer",
)

# 0-1 km storm-relative helicity
srh_1km = compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0)

# 0-6 km bulk wind shear
shear_06 = compute_shear(u_3d, v_3d, height_agl_3d, top_m=6000.0)

# ---- 3. Compute the Significant Tornado Parameter ----

stp = compute_stp(cape, lcl_hgt, srh_1km, shear_06)

# ---- 4. Plot ----

fig, ax = plt.subplots(
    subplot_kw={"projection": ccrs.LambertConformal()},
    figsize=(12, 8),
)
ax.set_extent([-105, -85, 30, 45], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE)

cf = ax.pcolormesh(
    lons, lats, stp.magnitude,
    transform=ccrs.PlateCarree(),
    cmap="RdYlGn_r",
    vmin=0, vmax=8,
)
plt.colorbar(cf, ax=ax, label="Significant Tornado Parameter")
ax.set_title("HRRR 06-h Forecast -- STP")
plt.tight_layout()
plt.savefig("stp_map.png", dpi=150)
plt.show()
```

### What this computes, step by step

1. **`compute_cape_cin`** lifts a mixed-layer parcel at every grid column
   through the full 3-D thermodynamic profile. For a standard 1059x1799 HRRR
   grid with 50 vertical levels, this processes roughly 95 million data points.
   On a modern 8-core machine the Rust backend completes this in seconds rather
   than the minutes required by a pure-Python column loop.

2. **`compute_srh`** integrates helicity in the 0--1 km layer using the
   storm-relative wind profile at each column.

3. **`compute_shear`** computes the wind vector difference between the surface
   and 6 km AGL, returning the magnitude.

4. **`compute_stp`** combines the four resulting 2-D fields into the STP
   composite index using the standard SPC formulation.

### Performance note

On a 1059x1799 HRRR grid (50 levels, ~95M points), typical wall-clock times on
an 8-core machine:

| Function | Approximate time |
|---|---|
| `compute_cape_cin` | 2--5 s |
| `compute_srh` | < 1 s |
| `compute_shear` | < 1 s |
| `compute_stp` | < 0.1 s |
| **Total** | **3--7 s** |

The equivalent pure-Python implementation (looping over 1.9 million columns)
takes 10--30 minutes depending on the thermodynamic calculation.
