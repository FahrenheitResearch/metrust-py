# Your First Grid Analysis

This tutorial walks you through processing 2-D and 3-D weather model grids
with metrust. By the end you will know how to apply thermodynamic functions
across entire grids, compute wind-field kinematics, run full-column severe
weather diagnostics on a 3-D domain, and smooth noisy output fields -- all
without writing a single Python loop.

No meteorology background is assumed. Every concept is explained as it comes
up.

---

## What is a weather grid?

Numerical Weather Prediction (NWP) models simulate the atmosphere on a
three-dimensional grid of boxes, tracking temperature, moisture, pressure,
wind, and dozens of other variables at every point.  The result is a snapshot
of the entire atmosphere stored as a stack of 2-D arrays, one per vertical
level.  The models you will encounter most often:

| Model | Grid size (approx.) | Spacing | Update frequency |
|-------|---------------------|---------|------------------|
| HRRR  | 1059 x 1799 x 50 levels | 3 km | Hourly |
| NAM   | 614 x 428 x 60 levels | 12 km | Every 6 hours |
| GFS   | 721 x 1440 x 34 levels | ~13 km (0.25 deg) | Every 6 hours |

Think of a 3-D grid as having a sounding at every grid point -- each vertical
column contains a pressure, temperature, moisture, and wind profile, just like
a weather balloon measures, but available everywhere simultaneously.

The catch: many useful quantities -- CAPE, storm-relative helicity, bulk
shear -- require integrating through each column individually.  An HRRR grid
has over 1.9 million columns.  Looping over them in Python would take tens of
minutes.  metrust passes the full arrays to Rust, which uses
[rayon](https://docs.rs/rayon/) to process all columns in parallel across
every CPU core.

---

## Creating synthetic data

Real model data comes in multi-gigabyte GRIB2 or NetCDF files.  For this
tutorial we will build a small but realistic synthetic grid so you can follow
along without downloading anything.

```python
import numpy as np
from metrust.units import units

ny, nx = 50, 80   # 50 rows (lat), 80 columns (lon)
nz = 10            # 10 vertical levels

rng = np.random.default_rng(42)

# --- Pressure: 1000 hPa (surface) to 200 hPa (jet stream) ---
p_levels_hpa = np.linspace(1000, 200, nz)
pressure_3d = p_levels_hpa[:, None, None] * np.ones((nz, ny, nx))

# --- Temperature: warm at surface / cold aloft, warm south / cold north ---
t_profile = np.linspace(30, -55, nz)            # base profile (degC)
lat_gradient = np.linspace(6, -6, ny)            # +6 C south, -6 C north
temperature_2d = t_profile[0] + lat_gradient[:, None] * np.ones((ny, nx))
temperature_3d = (
    t_profile[:, None, None]
    + lat_gradient[None, :, None] * np.ones((nz, ny, nx))
)
temperature_3d += rng.normal(0, 0.5, temperature_3d.shape)

# --- Dewpoint: moist near surface / dry aloft, moister in the east ---
td_profile = np.linspace(22, -65, nz)
moisture_gradient = np.linspace(-4, 4, nx)       # drier west, moister east
dewpoint_3d = (
    td_profile[:, None, None]
    + moisture_gradient[None, None, :] * np.ones((nz, ny, nx))
)
dewpoint_3d += rng.normal(0, 0.3, dewpoint_3d.shape)
dewpoint_3d = np.minimum(dewpoint_3d, temperature_3d - 0.5)

# --- Height AGL: simple linear approximation ---
height_3d = np.linspace(0, 14000, nz)[:, None, None] * np.ones((nz, ny, nx))

# --- Wind: increasing + veering with height (classic severe-weather profile) ---
u_3d = np.linspace(3, 30, nz)[:, None, None] * np.ones((nz, ny, nx))
v_3d = np.linspace(8, -5, nz)[:, None, None] * np.ones((nz, ny, nx))
u_3d += rng.normal(0, 1.5, u_3d.shape)
v_3d += rng.normal(0, 1.5, v_3d.shape)
```

The patterns baked into this data mimic a classic Great Plains severe weather
setup: warm, moist air in the southeast corner, cooler and drier air to the
northwest, and wind shear that increases with height.

!!! tip "Shape convention"
    3-D arrays in metrust are always **(nz, ny, nx)** -- vertical levels
    first, then latitude rows, then longitude columns.  2-D arrays are
    **(ny, nx)**.  This matches the native layout of HRRR, NAM, and WRF
    output.

---

## 2-D calculations -- scalar functions on grids

Every thermodynamic function in metrust accepts arrays of any shape.  Pass a
2-D grid in, get a 2-D grid out.  No loops, no reshaping.

### Potential temperature

Potential temperature is what the air temperature would be if you moved a
parcel to a reference pressure of 1000 hPa without adding or removing heat.
It lets you compare air at different altitudes on equal footing.

```python
from metrust.calc import potential_temperature

# Pick one level -- say, 850 hPa (index 2 in our 10-level grid)
p_850 = pressure_3d[2, :, :] * units.hPa
t_850 = temperature_3d[2, :, :] * units.degC

theta = potential_temperature(p_850, t_850)

print(f"Input shape:  {t_850.shape}")
print(f"Output shape: {theta.shape}")
print(f"Theta range:  {theta.min():.1f} to {theta.max():.1f}")
```

The output has the exact same shape as the input -- (50, 80).  metrust
preserved the grid dimensions automatically.

### Dewpoint from relative humidity

Suppose you have temperature and relative humidity grids and need dewpoint.

```python
from metrust.calc import dewpoint_from_relative_humidity

# Create a synthetic RH field (higher in the east, lower in the west)
rh_2d = np.linspace(40, 85, nx)[None, :] * np.ones((ny, nx))
rh_2d += rng.normal(0, 3, rh_2d.shape)
rh_2d = np.clip(rh_2d, 5, 100)

td_from_rh = dewpoint_from_relative_humidity(
    temperature_2d * units.degC,
    rh_2d / 100.0,  # metrust expects 0-1 for RH
)

print(f"Dewpoint shape: {td_from_rh.shape}")
print(f"Dewpoint range: {td_from_rh.min():.1f} to {td_from_rh.max():.1f}")
```

### Wind speed and direction

Meteorologists talk about wind in two ways: components (u, v) for math, and
speed/direction for communication.  `wind_speed` and `wind_direction` convert
between them.

```python
from metrust.calc import wind_speed, wind_direction

# Use the surface level (index 0)
u_sfc = u_3d[0, :, :] * units("m/s")
v_sfc = v_3d[0, :, :] * units("m/s")

speed = wind_speed(u_sfc, v_sfc)
direction = wind_direction(u_sfc, v_sfc)

print(f"Wind speed shape: {speed.shape}")
print(f"Speed range:  {speed.min():.1f} to {speed.max():.1f}")
print(f"Direction range: {direction.min():.0f} to {direction.max():.0f}")
```

`wind_direction` returns the meteorological convention: the direction the wind
is blowing *from*, measured clockwise from north.  A direction of 270 means
the wind comes from the west.

!!! note "Shape preservation"
    Every function shown above took a (50, 80) input and returned a (50, 80)
    output.  This is true for all scalar and element-wise functions in
    `metrust.calc`.  You never need to manually iterate over grid points.

---

## Kinematics on grids -- divergence and vorticity

Kinematics tells us how the wind field is stretching, rotating, and
converging.  Two quantities matter most:

**Divergence** measures whether air is spreading apart (positive divergence)
or piling together (negative divergence, called *convergence*).  Convergence
forces air upward -- when surface air has nowhere to go but up, cumulus towers
build.  This is one of the primary triggers for thunderstorm development.

**Vorticity** measures the spin of the air.  Positive vorticity (Northern
Hemisphere) means counterclockwise rotation, associated with low-pressure
systems and storm-scale mesocyclones.

Both require knowing the grid spacing in physical distance.

```python
from metrust.calc import divergence, vorticity

# Grid spacing: assume a 10 km grid
dx = 10000 * units.m
dy = 10000 * units.m

# Use wind at 850 hPa (index 2)
u_850 = u_3d[2, :, :] * units("m/s")
v_850 = v_3d[2, :, :] * units("m/s")

div = divergence(u_850, v_850, dx, dy)
vort = vorticity(u_850, v_850, dx, dy)

print(f"Divergence shape: {div.shape}")
print(f"Divergence range: {div.min():.2e} to {div.max():.2e}")
print(f"Vorticity range:  {vort.min():.2e} to {vort.max():.2e}")
```

The units of both fields are 1/s (per second).  Typical synoptic-scale values
are on the order of 10^-5 to 10^-4 s^-1.

!!! info "Why dx and dy matter"
    Divergence and vorticity are computed using finite differences --
    essentially measuring how much u changes across neighboring grid points in
    the x-direction, and how much v changes in the y-direction.  The grid
    spacing converts those index-based differences into physical rates of
    change.  Using the wrong dx/dy produces values that are off by orders of
    magnitude.

---

## 3-D grid composites -- the real power

This is where metrust shines.  The `compute_*` family of functions takes
full 3-D grids, processes every vertical column in parallel via Rust, and
returns 2-D result maps.

### CAPE and CIN

CAPE (Convective Available Potential Energy) measures how much energy is
available to fuel thunderstorm updrafts.  CIN (Convective Inhibition) is the
energy barrier that must be overcome before a storm can form -- think of it as
a lid on the atmosphere.

Computing CAPE requires lifting a parcel through the full column at each
grid point -- the most expensive operation in convective analysis.

```python
from metrust.calc import compute_cape_cin

# The grid composites expect raw arrays in specific units:
#   pressure in Pa, temperature in Celsius, moisture in kg/kg,
#   height in meters, surface temperature in Kelvin.

# Convert our hPa pressure to Pa
pressure_pa = pressure_3d * 100.0

# Surface fields (2-D)
psfc = np.full((ny, nx), 100000.0)               # surface pressure, Pa
t2 = (temperature_3d[0, :, :] + 273.15)          # 2-m temperature, K
q2 = np.full((ny, nx), 0.015)                     # 2-m mixing ratio, kg/kg

# Convert dewpoint_3d to a crude mixing ratio field for the 3-D moisture input
# (In real data this comes directly from the model output)
qvapor_3d = np.clip(0.016 * np.exp(-np.linspace(0, 3, nz)), 0.0001, 0.025)
qvapor_3d = qvapor_3d[:, None, None] * np.ones((nz, ny, nx))
qvapor_3d += rng.normal(0, 0.001, qvapor_3d.shape)
qvapor_3d = np.clip(qvapor_3d, 0.0001, 0.025)

cape, cin, lcl_height, lfc_height = compute_cape_cin(
    pressure_pa,
    temperature_3d,
    qvapor_3d,
    height_3d,
    psfc,
    t2,
    q2,
    parcel_type="surface",
)

print(f"CAPE shape: {cape.shape}")
print(f"CAPE range: {cape.min():.0f} to {cape.max():.0f}")
print(f"CIN range:  {cin.min():.0f} to {cin.max():.0f}")
```

That single call processed all 4,000 columns (50 x 80) in parallel.  Every
returned array is shaped (ny, nx) with Pint units attached.

| Return value | What it means | Units |
|---|---|---|
| `cape` | Energy available for updrafts | J/kg |
| `cin` | Energy barrier to storm initiation | J/kg |
| `lcl_height` | Cloud base height | m AGL |
| `lfc_height` | Height where storms become self-sustaining | m AGL |

### Storm-relative helicity

Storm-relative helicity (SRH) measures how much the low-level wind rotates
relative to a storm's motion.  High SRH means the storm's updraft is
ingesting air that already has a corkscrew spin, which favors supercell
thunderstorms and tornadoes.

```python
from metrust.calc import compute_srh

srh_1km = compute_srh(u_3d, v_3d, height_3d, top_m=1000.0)
srh_3km = compute_srh(u_3d, v_3d, height_3d, top_m=3000.0)

print(f"0-1 km SRH range: {srh_1km.min():.0f} to {srh_1km.max():.0f}")
print(f"0-3 km SRH range: {srh_3km.min():.0f} to {srh_3km.max():.0f}")
```

### Significant Tornado Parameter

Composite indices combine multiple ingredients into a single number.  The
Significant Tornado Parameter (STP) is the primary tool the Storm Prediction
Center uses to identify environments capable of producing EF2+ tornadoes.
It multiplies normalized CAPE, LCL height, 0-1 km SRH, and 0-6 km shear.

```python
from metrust.calc import compute_shear, compute_stp

shear_06 = compute_shear(u_3d, v_3d, height_3d, bottom_m=0.0, top_m=6000.0)

stp = compute_stp(cape, lcl_height, srh_1km, shear_06)

print(f"STP range: {stp.min():.2f} to {stp.max():.2f}")
```

An STP of 1 means all four ingredients are simultaneously at their
"significant tornado" threshold values.  Higher values indicate more
dangerous setups.

### Supercell Composite Parameter

SCP identifies environments likely to produce supercell thunderstorms.  It
combines CAPE, 0-3 km SRH, and 0-6 km shear.

```python
from metrust.calc import compute_scp

scp = compute_scp(cape, srh_3km, shear_06)

print(f"SCP range: {scp.min():.2f} to {scp.max():.2f}")
```

!!! tip "What do the output maps look like?"
    Each of these 2-D arrays is a map.  In a real analysis you would plot
    them with matplotlib's `pcolormesh` -- warm colors where CAPE or STP is
    high, cool colors where it is low.  The spatial patterns reveal where the
    atmosphere is primed for severe weather.  In our synthetic data, the
    warm, moist southeast corner of the grid should show the highest values.

---

## Smoothing noisy fields

Real model output and especially derived fields can be noisy.  Grid-scale
fluctuations sometimes obscure the larger pattern you care about.
`smooth_gaussian` applies a Gaussian filter to a 2-D field, controlled by a
single parameter: sigma, the standard deviation of the kernel in grid points.

```python
from metrust.calc import smooth_gaussian

# Our CAPE field has some grid-scale noise from the noisy input data
cape_raw = cape.magnitude  # strip Pint units for raw array

# Smooth with sigma=2 grid points
cape_smooth = smooth_gaussian(cape_raw, sigma=2.0)

print(f"Raw CAPE std dev:      {np.std(cape_raw):.1f}")
print(f"Smoothed CAPE std dev: {np.std(cape_smooth):.1f}")
```

A larger sigma produces more smoothing.  For a 3 km HRRR grid, sigma=2
smooths over roughly 6 km -- enough to remove grid-scale noise while
preserving mesoscale features.  Before smoothing, the field might show sharp
cell-to-cell jumps; after smoothing, the broad pattern emerges: high CAPE in
the warm/moist sector, low CAPE to the northwest.

!!! info "Other smoothing options"
    metrust also provides `smooth_rectangular` (box average),
    `smooth_circular` (disk filter), and `smooth_n_point` (classic 5-point
    or 9-point stencils).  All run in compiled Rust and accept the `passes`
    parameter for repeated application.

---

## Performance note

Everything in this tutorial ran on a small 50 x 80 grid, finishing in
milliseconds.  The same code scales to production-sized grids with no changes.

On a full HRRR domain (1059 x 1799, 50 levels, ~95 million 3-D data points),
typical wall-clock times on an 8-core machine:

| Operation | Approximate time |
|---|---|
| `compute_cape_cin` (1.9M columns) | 2--5 s |
| `compute_srh` (x2 depths) | < 1 s |
| `compute_shear` | < 1 s |
| `compute_stp` / `compute_scp` | < 0.1 s each |
| `smooth_gaussian` on a 2-D field | < 0.1 s |
| `divergence` / `vorticity` | < 0.1 s |
| **Full analysis pipeline** | **3--7 s** |

The equivalent pure-Python approach (looping over 1.9 million columns calling
MetPy's `cape_cin` once per column) takes over 50 minutes.  metrust
eliminates the loop: the Rust backend distributes columns across all cores via
rayon, and the GIL is released so nothing in Python blocks the work.  The 2-D
functions also run in compiled Rust -- a single FFI call, no per-element
Python overhead.

---

## Recap

Here is what you learned and the functions you used:

| What you did | Function | Input shape | Output shape |
|---|---|---|---|
| Potential temperature on a grid | `potential_temperature` | (ny, nx) | (ny, nx) |
| Dewpoint from RH on a grid | `dewpoint_from_relative_humidity` | (ny, nx) | (ny, nx) |
| Wind speed and direction | `wind_speed`, `wind_direction` | (ny, nx) | (ny, nx) |
| Divergence and vorticity | `divergence`, `vorticity` | (ny, nx) | (ny, nx) |
| CAPE/CIN across a 3-D domain | `compute_cape_cin` | (nz, ny, nx) | (ny, nx) |
| Storm-relative helicity | `compute_srh` | (nz, ny, nx) | (ny, nx) |
| Bulk wind shear | `compute_shear` | (nz, ny, nx) | (ny, nx) |
| Significant Tornado Parameter | `compute_stp` | (ny, nx) x4 | (ny, nx) |
| Supercell Composite Parameter | `compute_scp` | (ny, nx) x3 | (ny, nx) |
| Gaussian smoothing | `smooth_gaussian` | (ny, nx) | (ny, nx) |

The workflow is always the same: pass full arrays in, get full arrays out.
No column loops, no manual indexing, no intermediate data management.

---

## Next steps

- **[Your First Sounding](first-sounding.md)** -- Explains the meteorology
  behind CAPE, shear, helicity, and composite parameters using a single
  vertical profile.
- **[Grid Composites API](../api/grid-composites.md)** -- Full docs for every
  `compute_*` function, including `compute_lapse_rate`, `compute_pw`,
  `compute_ship`, `compute_dcp`, and reflectivity composites.
- **[Kinematics API](../api/kinematics.md)** -- Divergence, vorticity,
  advection, frontogenesis, deformation, and potential vorticity.
- **[Smoothing & Interpolation API](../api/smoothing.md)** -- All smoothing
  filters and finite-difference calculus operators.
- **[Array Support](../guides/arrays.md)** -- How metrust dispatches between
  scalars, 1-D arrays, and 2-D grids.
- **[Performance](../performance.md)** -- Detailed benchmarks comparing
  metrust to MetPy across scalar, array, and grid workloads.
