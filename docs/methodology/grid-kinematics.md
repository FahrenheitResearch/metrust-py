# Grid Kinematics: Methodology and Algorithms

This document describes the 2D grid kinematics algorithms used in metrust,
covering the Rust engine (`crates/wx-math/src/dynamics.rs`,
`crates/wx-math/src/gridmath.rs`) and the Python wrapper
(`python/metrust/calc/__init__.py`). The Python layer handles unit
conversions, variable-spacing grids, and spherical metric tensor
corrections before delegating to the Rust engine for uniform-spacing
computation.

---

## 1. Finite Differences

All Rust grid functions use **flattened row-major** arrays. A 2D grid of
dimensions `(ny, nx)` is stored in a 1D array where the index of point
`(j, i)` is:

```
k = j * nx + i
```

Here `j` is the row (y-index, increasing southward in most GRIB data) and
`i` is the column (x-index, increasing eastward).

### gradient_x: partial f / partial x

Three-case stencil with uniform spacing `dx`:

| Region | Formula |
|---|---|
| Interior (`1 <= i <= nx-2`) | `(f[j,i+1] - f[j,i-1]) / (2*dx)` |
| Left boundary (`i = 0`) | `(f[j,1] - f[j,0]) / dx` |
| Right boundary (`i = nx-1`) | `(f[j,nx-1] - f[j,nx-2]) / dx` |

The interior uses second-order centered differences. The boundaries use
first-order one-sided (forward or backward) differences. When `nx < 2`
the gradient is zero.

In array-index notation:

```
Interior:   df/dx[j*nx+i] = (values[j*nx+(i+1)] - values[j*nx+(i-1)]) * (1 / (2*dx))
Left:       df/dx[j*nx+0] = (values[j*nx+1] - values[j*nx+0]) * (1 / dx)
Right:      df/dx[j*nx+(nx-1)] = (values[j*nx+(nx-1)] - values[j*nx+(nx-2)]) * (1 / dx)
```

### gradient_y: partial f / partial y

Same scheme along the y-axis:

| Region | Formula |
|---|---|
| Interior (`1 <= j <= ny-2`) | `(f[j+1,i] - f[j-1,i]) / (2*dy)` |
| Top boundary (`j = 0`) | `(f[1,i] - f[0,i]) / dy` |
| Bottom boundary (`j = ny-1`) | `(f[ny-1,i] - f[ny-2,i]) / dy` |

Pre-computed reciprocals `inv_2dx = 1/(2*dx)` and `inv_dx = 1/dx` avoid
repeated division.

### Laplacian

The Laplacian uses three-point centered second derivatives:

```
d2f/dx2 = (f[j,i+1] - 2*f[j,i] + f[j,i-1]) / dx^2     (interior)
d2f/dy2 = (f[j+1,i] - 2*f[j,i] + f[j-1,i]) / dy^2     (interior)
nabla^2 f = d2f/dx2 + d2f/dy2
```

Boundary stencils use the same three-point pattern shifted to one side:

```
d2f/dx2[j,0]     = (f[j,2] - 2*f[j,1] + f[j,0]) / dx^2      (forward)
d2f/dx2[j,nx-1]  = (f[j,nx-1] - 2*f[j,nx-2] + f[j,nx-3]) / dx^2  (backward)
```

This is exact for quadratic fields (the test confirms `nabla^2(x^2+y^2) = 4`
everywhere, including boundaries).

---

## 2. Divergence

Horizontal divergence is defined as:

```
div = du/dx + dv/dy
```

The Rust implementation delegates directly to `gradient_x` and `gradient_y`:

```rust
pub fn divergence(u, v, nx, ny, dx, dy) -> Vec<f64> {
    let dudx = gradient_x(u, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);
    dudx.iter().zip(dvdy.iter()).map(|(a, b)| a + b).collect()
}
```

For a linearly expanding flow `u = alpha*x, v = beta*y`, the divergence is
exactly `alpha + beta` at all interior points.

---

## 3. Vorticity

Relative vorticity (vertical component of the curl):

```
zeta = dv/dx - du/dy
```

Implemented as:

```rust
pub fn vorticity(u, v, nx, ny, dx, dy) -> Vec<f64> {
    let dvdx = gradient_x(v, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);
    dvdx.iter().zip(dudy.iter()).map(|(a, b)| a - b).collect()
}
```

For solid-body rotation `u = -omega*y, v = omega*x`, vorticity is `2*omega`
at all interior points. For an irrotational field `u = x, v = y`, vorticity
is zero.

**Absolute vorticity** adds the Coriolis parameter:

```
eta = zeta + f,    where f = 2*Omega*sin(lat)
```

with `Omega = 7.2921e-5 rad/s`.

---

## 4. Spherical Metric Tensor Corrections

This is the most important section for understanding why metrust's Python
layer exists and why the uniform-spacing Rust path is not sufficient for
lat/lon grids.

### Why plain finite differences fail on lat/lon grids

On a regular lat/lon grid, the physical east-west distance between adjacent
grid points varies with latitude:

```
dx_physical = R * cos(lat) * dlon
dy_physical = R * dlat                (approximately constant)
```

At 60N, `cos(60) = 0.5`, so the physical dx is half what it is at the
equator. Plain finite differences using a constant `dx` in meters implicitly
assume a Cartesian grid and will produce errors proportional to the
variation in `cos(lat)` across the domain.

More fundamentally, wind is a **vector** field. On a curved manifold,
derivatives of vector fields require metric tensor corrections that scalar
derivatives do not. The divergence and vorticity of a vector field `(u, v)`
on a surface with non-trivial metric involve scale factor derivatives that
produce additional terms beyond the simple `du/dx + dv/dy` form.

### MetPy's approach: parallel_scale and meridional_scale

MetPy's `vector_derivative` function uses two scale factors from the map
projection:

- **parallel_scale** (`ps`): ratio of true distance to coordinate distance
  along parallels (east-west). For lat/lon: `ps = 1/cos(lat)`.
- **meridional_scale** (`ms`): ratio along meridians (north-south). For
  lat/lon: `ms = 1`.

These are obtained from `pyproj.Proj.get_factors()`, which returns the
Tissot indicatrix parameters for any map projection at any point. For
projections like Lambert Conformal, both `ps` and `ms` vary across the grid
and differ from unity.

### The correction formulas

Given the Cartesian (flat-grid) derivatives `du/dx`, `du/dy`, `dv/dx`,
`dv/dy` computed via finite differences, the corrected vector derivatives
are:

```
dx_correction = (ms / ps) * dp/dy
dy_correction = (ps / ms) * dm/dx

du/dx_corr = ps * du/dx - v * dx_correction
du/dy_corr = ms * du/dy + v * dy_correction
dv/dx_corr = ps * dv/dx + u * dx_correction
dv/dy_corr = ms * dv/dy - u * dy_correction
```

where:
- `dp/dy` = derivative of parallel_scale with respect to y (computed via
  `_first_derivative_variable` along axis=-2)
- `dm/dx` = derivative of meridional_scale with respect to x (computed via
  `_first_derivative_variable` along axis=-1)

Corrected divergence and vorticity are then:

```
divergence = du/dx_corr + dv/dy_corr
vorticity  = dv/dx_corr - du/dy_corr
```

### The u*tan(lat)/R term in spherical vorticity

For a pure lat/lon grid (`ps = 1/cos(lat)`, `ms = 1`):

```
dp/dy = d(1/cos(lat))/dy = sin(lat)/cos^2(lat) * (1/R)
      = tan(lat) * ps / R

dx_correction = (ms/ps) * dp/dy = cos(lat) * tan(lat)/(R*cos^2(lat))
              = tan(lat) / (R * cos(lat))
```

The `u * dx_correction` term that appears in `dv/dx_corr` (and
`-v * dx_correction` in `du/dx_corr`) is precisely the `u*tan(lat)/R`
spherical correction term that textbooks derive from the full spherical
vorticity equation.

### Scale factor computation from CRS

The Python function `_get_scale_factors(data)` attempts to extract scale
factors automatically:

1. **Primary path**: access `data.metpy.cartopy_crs`, then call
   `pyproj.Proj(crs).get_factors(lon, lat)` to get `parallel_scale` and
   `meridional_scale` arrays shaped `(ny, nx)`.

2. **Fallback** (when pyproj fails or no CRS is available): assume lat/lon
   and compute `ps = 1/cos(lat)`, `ms = 1`.

This means Lambert Conformal, Mercator, Polar Stereographic, and other
projections are handled correctly when the xarray DataArray carries CRS
metadata via `metpy.parse_cf()`.

### Pole singularity

At the geographic poles, `cos(lat) = 0`, so `ps = 1/cos(lat) = infinity`
and `dx_physical = 0`. The variable-spacing code handles this by replacing
near-zero spacings (`|dx| < 1.0` meter) with NaN:

```python
d[np.abs(d) < 1.0] = np.nan
```

This propagates NaN into the derivatives at those points rather than
producing infinities or division-by-zero errors.

---

## 5. Signed Grid Deltas

### The problem

The Rust `lat_lon_grid_deltas` function uses haversine distances, which are
always **positive** (haversine computes great-circle distance). But finite
differences need **signed** spacings: if latitude decreases from row 0 to
row ny-1 (common in GRIB data, where row 0 is the northernmost), then `dy`
must be negative so that `df/dy` has the correct sign.

### The Python solution

`lat_lon_grid_deltas()` in the Python layer takes the unsigned distances
from the Rust engine and applies signs based on the coordinate direction:

```python
# dx sign: based on longitude difference (column-wise)
lon_sign = np.sign(lon_arr[:, 1:] - lon_arr[:, :-1])
dx_out = dx_out * lon_sign

# dy sign: based on latitude difference (row-wise)
lat_sign = np.sign(lat_arr[1:, :] - lat_arr[:-1, :])
dy_out = dy_out * lat_sign
```

For a standard GRIB2 file where latitude decreases from north to south,
`lat_sign` is negative, making `dy` negative. This means
`df/dy = (f[south] - f[north]) / (2*dy)` correctly reflects that moving in
the positive y-index direction is moving southward (negative physical y).

### Why this matters

If `dy` were always positive (unsigned haversine), then vorticity and
divergence would have the wrong sign whenever latitude decreases with
increasing row index -- which is most real-world data. Early development
used unsigned distances, which produced correlation coefficients near
`-0.0002` against MetPy for vorticity and divergence on GFS lat/lon data.
Adding the sign convention fixed this to `corr > 0.999`.

---

## 6. Variable vs. Uniform Spacing: Three-Tier Dispatch

The Python wrapper implements a three-tier dispatch for derivative
computation:

### Tier 1: CRS + scale factors (full spherical correction)

When the input data carries CRS metadata (via `data.metpy.cartopy_crs`) or
the caller provides `parallel_scale`/`meridional_scale`, the code uses
`_vector_derivative_corrected()`. This computes all four corrected partial
derivatives using variable-spacing finite differences and metric tensor
corrections. This is the **most accurate** path.

**When used**: any xarray DataArray with MetPy CRS metadata, or when the
caller explicitly passes scale factors.

### Tier 2: Variable-spacing Python (no scale factors)

When `dx` and `dy` are 2D arrays (varying across the grid) but no scale
factors are available, the code uses `_first_derivative_variable()` -- a
pure-numpy implementation of centered differences with per-point spacing.

This handles non-uniform grids (e.g., rotated lat/lon, or hand-constructed
grids) without the full metric tensor correction.

**When used**: `lat_lon_grid_deltas()` output on a flat-Earth assumption,
or any 2D `dx`/`dy` arrays.

### Tier 3: Uniform-spacing Rust (fastest)

When `dx` and `dy` are scalars (or effectively uniform), the code calls the
Rust engine directly: `_calc.divergence()`, `_calc.vorticity()`, etc.

This is the fastest path -- pure Rust with no Python overhead -- but only
correct for projected grids where the spacing is truly uniform (e.g., HRRR
on Lambert Conformal with pre-computed constant dx/dy, or any grid where
the caller has already accounted for projection effects).

**When used**: Lambert Conformal data with constant grid spacing, or any
case where the caller passes scalar `dx`/`dy`.

### Detection logic

```python
if ps is not None and ms is not None:
    # Tier 1: full metric correction
elif _is_variable_spacing(dx) or _is_variable_spacing(dy) or dx_m.ndim >= 2:
    # Tier 2: variable spacing, no metric correction
else:
    # Tier 3: uniform Rust path
```

`_is_variable_spacing()` returns True when a 2D array's values vary by more
than 5% relative to their mean.

---

## 7. Advection

Advection of a scalar field `s` by wind `(u, v)`:

```
advection = -u * ds/dx - v * ds/dy
```

The Rust implementation:

```rust
pub fn advection(scalar, u, v, nx, ny, dx, dy) -> Vec<f64> {
    let dsdx = gradient_x(scalar, nx, ny, dx);
    let dsdy = gradient_y(scalar, nx, ny, dy);
    for k in 0..n {
        out[k] = -u[k] * dsdx[k] - v[k] * dsdy[k];
    }
    out
}
```

### Unit handling for compound units

The Python wrapper must produce output units of `[scalar_units] / s`. For
simple scalars (e.g., temperature in K), the output is `K/s`. For compound
scalars like vorticity (`1/s`), the output is `1/s^2`.

The unit construction uses **string-based** construction to avoid a Pint
cross-registry bug:

```python
try:
    s_u = units.Unit(str(scalar.units))
except Exception:
    s_u = units.dimensionless
out_unit = str(s_u / units.s)
```

**The cross-registry Pint bug**: when MetPy and metrust each create their
own `pint.UnitRegistry`, Pint objects from one registry cannot be combined
with objects from the other. Attempting `metpy_quantity.units / metrust_units.s`
raises a `DimensionalityError`. The fix is to convert the unit to a string
first (`str(scalar.units)`), then parse it in metrust's own registry. This
decouples the two registries while preserving the unit semantics.

### Temperature and moisture advection

`temperature_advection` and `moisture_advection` are thin wrappers around
`advection` -- they exist for API compatibility with MetPy.

---

## 8. Frontogenesis, Deformation, Geostrophic Wind

### Frontogenesis (2D Petterssen)

The rate of change of the magnitude of the potential temperature gradient:

```
F = -(1/|grad(theta)|) * [
    (dtheta/dx)^2 * (du/dx) +
    (dtheta/dy)^2 * (dv/dy) +
    (dtheta/dx)(dtheta/dy)(dv/dx + du/dy)
]
```

When `|grad(theta)| < 1e-20`, frontogenesis is set to zero to avoid
division by zero.

### Deformation

Three components:

- **Stretching deformation**: `du/dx - dv/dy`
- **Shearing deformation**: `dv/dx + du/dy`
- **Total deformation**: `sqrt(stretching^2 + shearing^2)`

These are computed by reusing `gradient_x` and `gradient_y`.

### Geostrophic wind

From a geopotential height field `Z` and latitude `lat`:

```
u_g = -(g/f) * dZ/dy
v_g =  (g/f) * dZ/dx
```

where `f = 2*Omega*sin(lat)` and `g = 9.80665 m/s^2`.

Near the equator (`|f| < 1e-10`), geostrophic balance breaks down and the
wind is set to zero.

### Curvature and shear vorticity

The relative vorticity is decomposed into curvature and shear components:

- **Curvature vorticity** arises from streamline curvature:
  `zeta_c = u*(dpsi/dx) + v*(dpsi/dy)` where `psi = atan2(v, u)`.
- **Shear vorticity** is the remainder: `zeta_s = zeta - zeta_c`.

For solid-body rotation, each contributes half the total vorticity.

---

## 9. Parallelism

### Current state: sequential Rust, no Rayon in dynamics.rs

The gradient, divergence, vorticity, and advection functions in
`dynamics.rs` use simple sequential `for j in 0..ny { for i in 0..nx }`
loops. They do **not** currently use Rayon parallelism.

This is a deliberate trade-off: for typical NWP grid sizes (1059x1799 for
HRRR, 720x361 for GFS), the gradient computation is memory-bandwidth-bound
rather than compute-bound. The sequential loops achieve good cache locality
by iterating in row-major order, and the operations per point are trivial
(one subtraction, one multiply).

### Where Rayon is used

Rayon parallelism is used in computationally heavier modules:

- **composite.rs**: CAPE/CIN, SRH, shear, STP, EHI, SCP -- these involve
  per-grid-point vertical profile integration (O(nz) per point), making
  them compute-bound. Each grid point is independent, so `into_par_iter()`
  over the flattened 2D indices provides near-linear speedup.
- **interpolate.rs**: Cressman/Barnes interpolation -- O(n_obs) per grid
  point, also embarrassingly parallel.

### The Python variable-spacing fallback

When the variable-spacing path (Tier 2) is taken, derivatives are computed
in pure numpy rather than Rust. This uses numpy's vectorized array
operations, which are internally parallelized via BLAS/LAPACK threads for
large arrays. The overhead of crossing the Python-Rust boundary with
per-point variable spacing would negate any Rayon benefit, so numpy is the
pragmatic choice.

The variable-spacing code uses sliced array operations:

```python
# Centered differences via numpy slicing (vectorized, no Python loop)
result[1:-1] = (arr[2:] - arr[:-2]) / (d_fwd + d_bwd)
# Boundaries
result[0] = (arr[1] - arr[0]) / d[0]
result[-1] = (arr[-1] - arr[-2]) / d[-1]
```

---

## Appendix: Lessons Learned

### The vort/div inversion case (corr = -0.0002)

During initial development, vorticity and divergence on GFS lat/lon data
showed near-zero (or slightly negative) correlation with MetPy reference
values. The root cause was twofold:

1. **Unsigned dy**: the Rust `lat_lon_grid_deltas` returns haversine
   distances, which are always positive. GFS data has latitude decreasing
   with increasing row index (north to south). Without negating `dy`, the
   y-derivative had the wrong sign, which inverted both vorticity and
   divergence.

2. **Missing metric corrections**: even with signed `dy`, lat/lon grids
   require the `u*tan(lat)/R` spherical correction terms. Without these,
   errors scale with `tan(lat)` and become large at high latitudes.

The fix had two parts:
- Apply latitude/longitude sign to the haversine distances in the Python
  `lat_lon_grid_deltas()` wrapper.
- Implement the full `_vector_derivative_corrected()` path with scale
  factor derivatives.

After both fixes, correlation against MetPy exceeded 0.999 across all
tested domains and variables.

### The Pint cross-registry bug

When computing vorticity advection (advecting a quantity with units `1/s`),
the output unit should be `1/s^2`. Constructing this via
`scalar.units / units.s` fails when `scalar` comes from MetPy's registry
and `units` is metrust's registry. The fix -- `units.Unit(str(scalar.units))`
-- converts through a string representation, avoiding any direct
cross-registry arithmetic.

### Curl of gradient = 0 as a verification tool

The identity `curl(grad(phi)) = 0` for any smooth scalar field provides an
excellent end-to-end test of the finite difference implementation. The test
suite verifies this for both quadratic (`phi = x^2 + 3xy + y^2`) and cubic
(`phi = x^3 + y^3 + xy^2`) fields. For the quadratic case, the finite
difference is exact. For the cubic, it is exact at interior points where
centered differences cancel the third-order terms symmetrically.
