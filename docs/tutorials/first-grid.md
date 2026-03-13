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

Numerical Weather Prediction (NWP) models divide the atmosphere into a
three-dimensional grid of boxes. At every box, the model tracks temperature,
moisture, pressure, wind, and dozens of other variables as they evolve forward
in time. The output is a snapshot of the entire atmosphere stored as a stack
of 2-D arrays -- one per vertical level, one per variable.

The models you will encounter most often:

| Model | Grid size (approx.) | Spacing | Update frequency |
|-------|---------------------|---------|------------------|
| HRRR  | 1059 x 1799 x 50 levels | 3 km | Hourly |
| NAM   | 614 x 428 x 60 levels | 12 km | Every 6 hours |
| GFS   | 721 x 1440 x 34 levels | ~13 km (0.25 deg) | Every 6 hours |

Think of a 3-D grid as having **a sounding at every single grid point**.
Each vertical column contains a pressure, temperature, moisture, and wind
profile -- exactly what a weather balloon measures, but available everywhere
simultaneously. A single HRRR forecast has over 1.9 million columns. That
is an enormous amount of data, and many useful quantities (CAPE, helicity,
bulk shear) require integrating through every column individually.

The catch: looping over 1.9 million columns in Python would take tens of
minutes. metrust solves this by passing the full arrays to Rust, which uses
[rayon](https://docs.rs/rayon/) to process all columns in parallel across
every CPU core. Your code stays simple -- one function call, one result --
while the heavy lifting happens at native speed.

---

## Creating synthetic data

Real model data arrives in multi-gigabyte GRIB2 or NetCDF files. For this
tutorial we will build a small but realistic synthetic grid so you can follow
along without downloading anything.

```python
import numpy as np
from metrust.units import units

ny, nx = 50, 80   # 50 rows (latitude), 80 columns (longitude)
nz = 10            # 10 vertical levels

rng = np.random.default_rng(42)  # fixed seed for reproducibility
```

### Pressure levels

```python
# Surface (1000 hPa) to jet stream (200 hPa)
p_levels_hpa = np.linspace(1000, 200, nz)
pressure_3d = p_levels_hpa[:, None, None] * np.ones((nz, ny, nx))
```

Each slice `pressure_3d[k, :, :]` is a 2-D field at a single pressure level.
Level 0 is the surface (1000 hPa); level 9 is the upper troposphere (200 hPa).

### Temperature

```python
# Base profile: 30 C at the surface, -55 C at the top
t_profile = np.linspace(30, -55, nz)

# North-south gradient: +6 C in the south, -6 C in the north
lat_gradient = np.linspace(6, -6, ny)

# Build the 3-D field and add realistic noise
temperature_3d = (
    t_profile[:, None, None]
    + lat_gradient[None, :, None] * np.ones((nz, ny, nx))
)
temperature_3d += rng.normal(0, 0.5, temperature_3d.shape)

# Keep a 2-D surface temperature grid for later
temperature_2d = temperature_3d[0, :, :]
```

Our grid mimics a classic Great Plains pattern: warm in the south, cold in
the north, cooling rapidly with altitude.

### Dewpoint

```python
td_profile = np.linspace(22, -65, nz)
moisture_gradient = np.linspace(-4, 4, nx)  # drier west, moister east

dewpoint_3d = (
    td_profile[:, None, None]
    + moisture_gradient[None, None, :] * np.ones((nz, ny, nx))
)
dewpoint_3d += rng.normal(0, 0.3, dewpoint_3d.shape)
dewpoint_3d = np.minimum(dewpoint_3d, temperature_3d - 0.5)  # Td <= T always
```

### Height, wind, and moisture

```python
# Height AGL: 0 m at the surface to 14 km at the top
height_3d = np.linspace(0, 14000, nz)[:, None, None] * np.ones((nz, ny, nx))

# Wind: increasing and veering with height (classic severe-weather shear)
u_3d = np.linspace(3, 30, nz)[:, None, None] * np.ones((nz, ny, nx))
v_3d = np.linspace(8, -5, nz)[:, None, None] * np.ones((nz, ny, nx))
u_3d += rng.normal(0, 1.5, u_3d.shape)
v_3d += rng.normal(0, 1.5, v_3d.shape)

# Water vapor mixing ratio: high near the surface, dry aloft
qvapor_3d = np.clip(0.016 * np.exp(-np.linspace(0, 3, nz)), 0.0001, 0.025)
qvapor_3d = qvapor_3d[:, None, None] * np.ones((nz, ny, nx))
qvapor_3d += rng.normal(0, 0.001, qvapor_3d.shape)
qvapor_3d = np.clip(qvapor_3d, 0.0001, 0.025)
```

!!! tip "Shape convention"
    3-D arrays in metrust are always **(nz, ny, nx)** -- vertical levels
    first, then latitude rows, then longitude columns. 2-D arrays are
    **(ny, nx)**. This matches the native layout of HRRR, NAM, and WRF
    output.

---

## 2-D calculations -- scalar functions on grids

Every thermodynamic function in metrust accepts arrays of any shape. Pass a
2-D grid in, get a 2-D grid out. No loops, no reshaping.

### Potential temperature

Potential temperature is what the air temperature would be if you moved a
parcel to a reference pressure of 1000 hPa without adding or removing heat.
It lets you compare air at different altitudes on equal footing.

```python
from metrust.calc import potential_temperature

# Pick the 850 hPa level (index 2 in our 10-level grid)
p_850 = pressure_3d[2, :, :] * units.hPa
t_850 = temperature_3d[2, :, :] * units.degC

theta = potential_temperature(p_850, t_850)

print(f"Input shape:  {t_850.shape}")    # (50, 80)
print(f"Output shape: {theta.shape}")    # (50, 80)
print(f"Theta range:  {theta.min():.1f} to {theta.max():.1f}")
```

The output has the exact same shape as the input -- (50, 80). metrust
preserved the grid dimensions automatically.

### Dewpoint from relative humidity

Suppose you have temperature and relative humidity grids and need dewpoint.

```python
from metrust.calc import dewpoint_from_relative_humidity

# Synthetic RH field: drier in the west (40%), moister in the east (85%)
rh_2d = np.linspace(40, 85, nx)[None, :] * np.ones((ny, nx))
rh_2d += rng.normal(0, 3, rh_2d.shape)
rh_2d = np.clip(rh_2d, 5, 100)

td_from_rh = dewpoint_from_relative_humidity(
    temperature_2d * units.degC,
    rh_2d / 100.0,  # metrust expects fractional RH (0-1)
)

print(f"Dewpoint shape: {td_from_rh.shape}")    # (50, 80)
print(f"Dewpoint range: {td_from_rh.min():.1f} to {td_from_rh.max():.1f}")
```

### Wind speed and direction

Meteorologists store wind as u/v components for math, but communicate it as
speed and direction. `wind_speed` and `wind_direction` convert between the
two representations.

```python
from metrust.calc import wind_speed, wind_direction

# Surface-level wind
u_sfc = u_3d[0, :, :] * units("m/s")
v_sfc = v_3d[0, :, :] * units("m/s")

speed = wind_speed(u_sfc, v_sfc)
direction = wind_direction(u_sfc, v_sfc)

print(f"Speed shape:     {speed.shape}")        # (50, 80)
print(f"Speed range:     {speed.min():.1f} to {speed.max():.1f}")
print(f"Direction range: {direction.min():.0f} to {direction.max():.0f}")
```

`wind_direction` returns the meteorological convention: the direction the
wind is blowing *from*, measured clockwise from north. A direction of 270
means the wind comes from the west.

!!! note "Shape preservation"
    Every function shown above took a (50, 80) input and returned a (50, 80)
    output. This is true for all scalar and element-wise functions in
    `metrust.calc`. You never need to manually iterate over grid points.

---

## Kinematics on grids -- divergence and vorticity

Kinematics tells us how the wind field is stretching, rotating, and
converging. Two quantities matter most for severe weather:

**Divergence** measures whether air is spreading apart (positive divergence)
or piling together (negative divergence, called *convergence*). When air
converges at the surface, it has nowhere to go but up, building the cumulus
towers that become thunderstorms. Surface convergence is one of the primary
triggers for storm initiation. Aloft, the opposite is true: divergence at
upper levels acts as a vacuum, pulling air upward and strengthening updrafts.

**Vorticity** measures the spin of the air. In the Northern Hemisphere,
positive vorticity means counterclockwise rotation, associated with
low-pressure systems and storm-scale mesocyclones. Strong low-level
vorticity is one of the ingredients that produces tornadoes.

Both require knowing the physical grid spacing so that differences between
neighboring grid points can be converted into real-world rates of change.

```python
from metrust.calc import divergence, vorticity

# Grid spacing: assume a 10 km grid
dx = 10_000 * units.m
dy = 10_000 * units.m

# Use wind at 850 hPa (index 2)
u_850 = u_3d[2, :, :] * units("m/s")
v_850 = v_3d[2, :, :] * units("m/s")

div = divergence(u_850, v_850, dx, dy)
vort = vorticity(u_850, v_850, dx, dy)

print(f"Divergence shape: {div.shape}")       # (50, 80)
print(f"Divergence range: {div.min():.2e} to {div.max():.2e}")
print(f"Vorticity range:  {vort.min():.2e} to {vort.max():.2e}")
```

The units of both fields are 1/s (per second). Typical synoptic-scale values
are on the order of 10^-5 to 10^-4 s^-1.

!!! info "Why dx and dy matter"
    Divergence and vorticity are computed using finite differences --
    essentially measuring how much u changes across neighboring grid points
    in the x-direction, and how much v changes in the y-direction. The grid
    spacing converts those index-based differences into physical rates of
    change. Using the wrong dx/dy produces values that are off by orders of
    magnitude. For real model data, you can compute dx and dy from the
    lat/lon grid using `metrust.calc.lat_lon_grid_deltas`.

---

## 3-D grid composites -- the real power

This is where metrust shines. The `compute_*` family of functions takes
full 3-D grids, processes every vertical column in parallel via Rust, and
returns 2-D result maps. Each function turns a 3-D problem into a 2-D
answer in a single call.

### CAPE and CIN

CAPE (Convective Available Potential Energy) measures the total energy
available to fuel a thunderstorm's updraft. Higher CAPE means stronger
updrafts, which means larger hail, heavier rain, and more violent storms.
CIN (Convective Inhibition) is the energy barrier that must be overcome
before a storm can form -- think of it as a lid on the atmosphere.

Computing CAPE requires lifting a theoretical air parcel through the
entire column at each grid point, comparing its temperature against the
environment at every level. This is the most computationally expensive
operation in convective analysis.

```python
from metrust.calc import compute_cape_cin

# Grid composites expect raw arrays in specific units:
#   pressure in Pa, temperature in Celsius, moisture in kg/kg,
#   height in meters, surface temperature in Kelvin.

pressure_pa = pressure_3d * 100.0                    # hPa -> Pa

# Surface fields (2-D)
psfc = np.full((ny, nx), 100_000.0)                  # surface pressure, Pa
t2 = temperature_3d[0, :, :] + 273.15               # 2-m temperature, K
q2 = np.full((ny, nx), 0.015)                        # 2-m mixing ratio, kg/kg

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

print(f"CAPE shape: {cape.shape}")      # (50, 80) -- 3-D input, 2-D output
print(f"CAPE range: {cape.min():.0f} to {cape.max():.0f} J/kg")
print(f"CIN range:  {cin.min():.0f} to {cin.max():.0f} J/kg")
```

That single call processed all 4,000 columns (50 x 80) in parallel. Every
returned array is shaped (ny, nx):

| Return value  | What it means                               | Units |
|---------------|---------------------------------------------|-------|
| `cape`        | Energy available for updrafts                | J/kg  |
| `cin`         | Energy barrier to storm initiation           | J/kg  |
| `lcl_height`  | Cloud base height above ground               | m AGL |
| `lfc_height`  | Height where storms become self-sustaining   | m AGL |

In our synthetic data, the warm, moist southeast corner of the grid should
show the highest CAPE values -- that is where the atmosphere has the most
fuel for storms.

!!! tip "Parcel types"
    The `parcel_type` parameter controls which air parcel is lifted.
    `"surface"` lifts the parcel from the lowest level. `"mixed_layer"`
    averages the lowest 100 hPa before lifting -- this is what the Storm
    Prediction Center primarily uses. `"most_unstable"` searches for the
    level with the most energy. Operational forecasters look at all three.

### Storm-relative helicity

Storm-relative helicity (SRH) measures how much the low-level wind rotates
relative to a storm's motion. Picture the air spiraling into the storm like
water swirling into a drain. High SRH means the storm's updraft is
ingesting air that already has a corkscrew spin, which it tilts into the
vertical to create a mesocyclone -- the rotating updraft of a supercell
thunderstorm.

```python
from metrust.calc import compute_srh

srh_1km = compute_srh(u_3d, v_3d, height_3d, top_m=1000.0)
srh_3km = compute_srh(u_3d, v_3d, height_3d, top_m=3000.0)

print(f"0-1 km SRH shape: {srh_1km.shape}")    # (50, 80)
print(f"0-1 km SRH range: {srh_1km.min():.0f} to {srh_1km.max():.0f} m^2/s^2")
print(f"0-3 km SRH range: {srh_3km.min():.0f} to {srh_3km.max():.0f} m^2/s^2")
```

The 0-1 km layer is especially critical: tornadoes are a near-surface
phenomenon, so strong low-level SRH signals that the storm can produce
rotation right down to the ground.

### Significant Tornado Parameter

Composite indices combine multiple ingredients into a single number. The
Significant Tornado Parameter (STP) is the primary tool the Storm
Prediction Center uses to identify environments capable of producing EF2+
tornadoes. It multiplies normalized CAPE, LCL height, 0-1 km SRH, and
0-6 km bulk wind shear.

```python
from metrust.calc import compute_shear, compute_stp

# 0-6 km bulk wind shear at every column
shear_06 = compute_shear(u_3d, v_3d, height_3d, bottom_m=0.0, top_m=6000.0)

# Combine everything into STP
stp = compute_stp(cape, lcl_height, srh_1km, shear_06)

print(f"STP shape: {stp.shape}")            # (50, 80)
print(f"STP range: {stp.min():.2f} to {stp.max():.2f}")
```

An STP of 1.0 means all four ingredients are simultaneously at their
"significant tornado" threshold values. Higher values indicate a more
dangerous setup. Values above 4 signal an environment where violent
tornadoes become a real possibility.

### Supercell Composite Parameter

SCP identifies environments likely to produce supercell thunderstorms --
rotating storms that are responsible for the vast majority of tornadoes,
giant hail, and damaging winds. It combines CAPE, 0-3 km SRH, and 0-6 km
shear.

```python
from metrust.calc import compute_scp

scp = compute_scp(cape, srh_3km, shear_06)

print(f"SCP shape: {scp.shape}")            # (50, 80)
print(f"SCP range: {scp.min():.2f} to {scp.max():.2f}")
```

!!! note "What do the output maps look like?"
    Each of these 2-D arrays is a map. In a real analysis you would plot
    them with matplotlib's `pcolormesh` -- warm colors where CAPE or STP is
    high, cool colors where it is low. The spatial patterns reveal where the
    atmosphere is primed for severe weather. In our synthetic data, the
    warm, moist southeast corner of the grid should show the highest values,
    while the cool, dry northwest corner stays near zero.

---

## Smoothing noisy fields

Real model output and especially derived fields can be noisy. Grid-scale
fluctuations sometimes obscure the larger pattern you care about.
`smooth_gaussian` applies a Gaussian filter to a 2-D field, controlled by a
single parameter: **sigma**, the standard deviation of the kernel in grid
points.

```python
from metrust.calc import smooth_gaussian

# Our CAPE field has some grid-scale noise from the random perturbations
cape_raw = cape.magnitude   # strip Pint units for the raw array

# Smooth with sigma = 2 grid points
cape_smooth = smooth_gaussian(cape_raw, sigma=2.0)

print(f"Raw CAPE std dev:      {np.std(cape_raw):.1f}")
print(f"Smoothed CAPE std dev: {np.std(cape_smooth):.1f}")
```

A larger sigma produces more smoothing. For a 3 km HRRR grid, sigma=2
smooths over roughly 6 km -- enough to remove grid-scale noise while
preserving mesoscale features like outflow boundaries and drylines. Before
smoothing, the CAPE field might show sharp cell-to-cell jumps; after
smoothing, the broad pattern emerges cleanly: high CAPE in the warm/moist
sector, low CAPE to the northwest.

!!! info "Other smoothing options"
    metrust also provides `smooth_rectangular` (box average),
    `smooth_circular` (disk filter), and `smooth_n_point` (classic 5-point
    or 9-point stencils). All run in compiled Rust and accept the `passes`
    parameter for repeated application. Multiple passes of a box filter
    approximate a Gaussian, so `smooth_rectangular(data, size=5, passes=3)`
    is another common choice.

---

## Performance note

Everything in this tutorial ran on a small 50 x 80 grid and finished in
milliseconds. The same code scales to production-sized grids with no changes
-- you do not need to rewrite anything when you move from a tutorial dataset
to a real HRRR forecast.

On a full HRRR domain (1059 x 1799, 50 levels, ~95 million 3-D data
points), typical wall-clock times on an 8-core machine:

| Operation | Approximate time |
|---|---|
| `compute_cape_cin` (1.9M columns) | 2--5 s |
| `compute_srh` (x2 depths) | < 1 s |
| `compute_shear` | < 1 s |
| `compute_stp` / `compute_scp` | < 0.1 s each |
| `smooth_gaussian` on a 2-D field | < 0.1 s |
| `divergence` / `vorticity` | < 0.1 s |
| **Full analysis pipeline** | **3--7 s** |

The equivalent pure-Python approach -- looping over 1.9 million columns
calling MetPy's `cape_cin` once per column -- takes over 50 minutes on the
same hardware. metrust eliminates the loop entirely: the Rust backend
distributes columns across all CPU cores via rayon, and the Python GIL is
released so nothing blocks the work. The 2-D functions (divergence,
vorticity, smoothing) also run in compiled Rust -- a single FFI call per
grid, no per-element Python overhead.

!!! tip "Scaling beyond a single machine"
    The parallelism is per-process (all cores on the current machine).
    If you need to process hundreds of forecast hours, run each time step
    as a separate process or use a task queue. Each process independently
    saturates all available cores.

---

## Recap

Here is what you learned and the functions you used:

| What you did | Function | Input | Output |
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

- **[Your First Sounding](first-sounding.md)** -- Walks through the
  meteorology behind CAPE, shear, helicity, and composite parameters
  using a single vertical profile. If the 3-D concepts in this tutorial
  felt abstract, that tutorial grounds them in a single column.
- **[Grid Composites API](../api/grid-composites.md)** -- Full reference
  for every `compute_*` function, including `compute_lapse_rate`,
  `compute_pw`, `compute_ship`, `compute_dcp`, and reflectivity composites.
- **[Kinematics API](../api/kinematics.md)** -- Divergence, vorticity,
  advection, frontogenesis, deformation, and potential vorticity.
- **[Smoothing API](../api/smoothing.md)** -- All smoothing filters and
  finite-difference calculus operators.
- **[Array Support](../guides/arrays.md)** -- How metrust dispatches
  between scalars, 1-D arrays, and 2-D grids.
- **[Performance](../performance.md)** -- Detailed benchmarks comparing
  metrust to MetPy across scalar, array, and grid workloads.
