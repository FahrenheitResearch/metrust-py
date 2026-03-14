# Wind Calculations: Methodology and Algorithms

This document describes the exact algorithms used in metrust's wind calculations, traced directly from the Rust source code. All code references are from the `rustmet` repository.

---

## 1. Wind Speed and Direction

### Source files
- `crates/wx-math/src/dynamics.rs` (primary, profile-based)
- `crates/rustmet-core/src/grib2/ops.rs` (grid field version)

### Direction Convention

Metrust uses the **meteorological convention**: wind direction is the compass bearing the wind is blowing **from**.

- 0 degrees = from the **north**
- 90 degrees = from the **east**
- 180 degrees = from the **south**
- 270 degrees = from the **west**

This means a "southerly" wind (blowing from south toward north) has `u = 0, v > 0` and direction = 180 degrees.

### Wind Speed

```
speed = sqrt(u^2 + v^2)
```

```rust
// dynamics.rs
pub fn wind_speed(u: &[f64], v: &[f64]) -> Vec<f64> {
    u.iter()
        .zip(v.iter())
        .map(|(ui, vi)| (ui * ui + vi * vi).sqrt())
        .collect()
}
```

### Wind Direction (dynamics.rs)

The primary implementation uses `atan2(u, v)` (note: u is the first argument to `atan2`, v is the second), then adds 180 degrees:

```
dir = (atan2(u, v) * 180 / pi) + 180
dir = dir mod 360
```

```rust
// dynamics.rs
pub fn wind_direction(u: &[f64], v: &[f64]) -> Vec<f64> {
    u.iter()
        .zip(v.iter())
        .map(|(ui, vi)| {
            let spd = (ui * ui + vi * vi).sqrt();
            if spd < 1e-10 {
                0.0
            } else {
                let dir = (ui.atan2(*vi) * 180.0 / PI) + 180.0;
                dir % 360.0
            }
        })
        .collect()
}
```

The `atan2(u, v)` call (with u as the y-argument and v as the x-argument) gives the angle of the wind vector relative to the v-axis (north). Adding 180 degrees flips it from the "blowing toward" direction to the "blowing from" direction.

Calm winds (speed < 1e-10 m/s) are assigned direction 0.

### Wind Direction (grib2/ops.rs variant)

The GRIB2 operations module uses a different but mathematically equivalent formula:

```
dir = 270 - atan2(v, u) * (180/pi)
```

```rust
// grib2/ops.rs
let mut dir = 270.0 - vv.atan2(uu).to_degrees();
if dir < 0.0 { dir += 360.0; }
if dir >= 360.0 { dir -= 360.0; }
```

Here `atan2(v, u)` returns the standard math angle (counterclockwise from the +x axis), and `270 - angle` converts from math convention to meteorological convention. Both implementations produce identical results.

---

## 2. Wind Components

### Speed/Direction to u/v Conversion

```
u = -speed * sin(direction)
v = -speed * cos(direction)
```

Where direction is in radians (converted from degrees).

```rust
// dynamics.rs
pub fn wind_components(speed: &[f64], direction: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let u: Vec<f64> = speed.iter().zip(direction.iter())
        .map(|(s, d)| -s * (d * PI / 180.0).sin())
        .collect();
    let v: Vec<f64> = speed.iter().zip(direction.iter())
        .map(|(s, d)| -s * (d * PI / 180.0).cos())
        .collect();
    (u, v)
}
```

### Sign Conventions

| Component | Positive means wind blows... | Example |
|-----------|------------------------------|---------|
| u         | from west to east            | Westerly wind: u > 0 |
| v         | from south to north          | Southerly wind: v > 0 |

The negative signs in the conversion formulas account for the fact that meteorological direction is the direction the wind is **from**, while the components describe the direction the wind is **going to**:

- A north wind (direction = 0 degrees): `u = -spd * sin(0) = 0`, `v = -spd * cos(0) = -spd` (blows southward, so v < 0).
- A west wind (direction = 270 degrees): `u = -spd * sin(270) = +spd`, `v = -spd * cos(270) = 0` (blows eastward, so u > 0).

### Sounding Wind Components (wx-sounding)

The sounding module converts from direction (degrees) and speed (knots) to u/v (m/s) in a single step:

```rust
// wx-sounding/src/derived.rs
fn wind_components(wdir: &[f64], wspd: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let kt_to_ms = 0.51444;
    for i in 0..wdir.len() {
        let spd_ms = wspd[i] * kt_to_ms;
        let dir_rad = wdir[i].to_radians();
        u.push(-spd_ms * dir_rad.sin());
        v.push(-spd_ms * dir_rad.cos());
    }
    (u, v)
}
```

---

## 3. Bulk Shear

### Source files
- `crates/wx-math/src/composite.rs` (`compute_shear` for 3D grids)
- `crates/wx-sounding/src/derived.rs` (`compute_bulk_shear` for single columns)
- `crates/wx-ui/src/panels/hodograph.rs` (`bulk_shear` for UI display)

### Algorithm

Bulk shear is the vector difference of wind at two height levels:

```
shear_u = u(top) - u(bottom)
shear_v = v(top) - v(bottom)
shear_magnitude = sqrt(shear_u^2 + shear_v^2)
```

### Height Interpolation

Both the top and bottom wind values are obtained by **linear interpolation** to the exact requested height AGL. The interpolation function (`interp_at_height`) works on ordered height profiles:

```rust
// composite.rs / derived.rs
fn interp_at_height(target_h: f64, heights: &[f64], values: &[f64]) -> f64 {
    if target_h <= heights[0] {
        return values[0];   // Clamp to surface
    }
    if target_h >= heights[heights.len() - 1] {
        return values[values.len() - 1];   // Clamp to top
    }
    for k in 0..heights.len() - 1 {
        if heights[k] <= target_h && heights[k + 1] >= target_h {
            let frac = (target_h - heights[k]) / (heights[k + 1] - heights[k]);
            return values[k] + frac * (values[k + 1] - values[k]);
        }
    }
    // fallback
}
```

Key behavior: if `target_h <= heights[0]`, the function returns the surface value `values[0]`. This means requesting shear from `bottom=0` when the first height level is above 0 AGL will use the first available level as the bottom wind. This is intentional -- the first level in a surface-first profile represents the surface observation.

### Sounding Bulk Shear (derived.rs)

```rust
fn compute_bulk_shear(h_agl: &[f64], u_prof: &[f64], v_prof: &[f64],
                      bottom_m: f64, top_m: f64) -> f64 {
    let u_bot = interp_at_height(bottom_m, h_agl, u_prof);
    let v_bot = interp_at_height(bottom_m, h_agl, v_prof);
    let u_top = interp_at_height(top_m, h_agl, u_prof);
    let v_top = interp_at_height(top_m, h_agl, v_prof);

    let du = u_top - u_bot;
    let dv = v_top - v_bot;
    let shear_ms = (du * du + dv * dv).sqrt();
    shear_ms / 0.51444   // Convert m/s back to knots for output
}
```

Note: The sounding version returns the result in **knots** (divides m/s by 0.51444), while the composite/grid version (`compute_shear`) returns **m/s**.

### 3D Grid Bulk Shear (composite.rs)

The grid version operates column-by-column in parallel via rayon. Each column is extracted, ordered surface-upward if needed, and then the same interpolation + vector difference is applied:

```rust
let u_bot = interp_at_height(bottom_m, &h_prof, &u_prof);
let v_bot = interp_at_height(bottom_m, &h_prof, &v_prof);
let u_top = interp_at_height(top_m, &h_prof, &u_prof);
let v_top = interp_at_height(top_m, &h_prof, &v_prof);
let du = u_top - u_bot;
let dv = v_top - v_bot;
(du * du + dv * dv).sqrt()   // Returns m/s
```

### Historical Note: `bottom=0` vs `bottom=h[0]`

When `bottom_m = 0.0` is passed and the profile's first height is above 0 AGL (e.g., 10m for a model's 10-meter wind level), `interp_at_height` clamps to `values[0]`. This means `bulk_shear(0, 6000)` uses the first available level as the surface, which is correct for sounding data but worth noting for model data where the lowest level may be 10m AGL. The standard calls use `bottom_m = 0.0` for both 0-1 km and 0-6 km shear.

---

## 4. Mean Wind

Metrust uses two distinct approaches for computing mean wind depending on context.

### Simple Height-Weighted Average (composite.rs, derived.rs)

Used in the SRH calculation and the composite grid Bunkers computation. This is a **trapezoidal height-weighted average**: the wind at the midpoint of each layer is weighted by the layer thickness `dz`.

```
mean_u = (sum of u_mid * dz) / (sum of dz)
mean_v = (sum of v_mid * dz) / (sum of dz)
```

Where `u_mid = 0.5 * (u[k] + u[k+1])` and `dz = h[k+1] - h[k]`.

```rust
// composite.rs :: compute_srh_column
for k in 0..nz - 1 {
    if heights[k] >= mean_depth { break; }
    let h_bot = heights[k];
    let h_top = heights[k + 1].min(mean_depth);
    let dz = h_top - h_bot;
    if dz <= 0.0 { continue; }
    let u_mid = 0.5 * (u_prof[k] + u_prof[k + 1]);
    let v_mid = 0.5 * (v_prof[k] + v_prof[k + 1]);
    sum_u += u_mid * dz;
    sum_v += v_mid * dz;
    sum_dz += dz;
}
let mean_u = sum_u / sum_dz;
let mean_v = sum_v / sum_dz;
```

This properly handles the top boundary by clamping `h_top` to `mean_depth` (e.g., 6000m), so partial layers at the boundary are correctly weighted.

### Simple Level Average (thermo.rs Bunkers)

The `bunkers_storm_motion` function in `thermo.rs` uses a simpler approach -- an unweighted average of all data levels below 6 km:

```rust
// thermo.rs :: bunkers_storm_motion
for i in 0..z.len() {
    if z[i] <= 6000.0 {
        sum_u += u[i];
        sum_v += v[i];
        count += 1.0;
    }
}
let u_mean = sum_u / count;
let v_mean = sum_v / count;
```

This treats every sounding level equally regardless of spacing, which can bias the result toward densely-sampled layers.

### Interpolated Sample Average (hodograph UI)

The hodograph UI panel uses yet another approach: it interpolates the wind profile at evenly-spaced 250m intervals and averages those interpolated values:

```rust
// wx-ui/panels/hodograph.rs
fn mean_wind(h: &[f64], u: &[f64], v: &[f64], h0: f64, h1: f64) -> Option<(f64, f64)> {
    let n = ((h1 - h0) / 250.0).ceil() as usize;
    for i in 0..=n {
        let hh = h0 + (h1 - h0) * i as f64 / n as f64;
        // interpolate u and v at hh, accumulate
    }
    Some((su / c as f64, sv / c as f64))
}
```

### Where Each Is Used

| Method | Where | Context |
|--------|-------|---------|
| Height-weighted trapezoidal | `composite.rs`, `derived.rs` | SRH computation, grid-based Bunkers |
| Simple level average | `thermo.rs` | Python-exposed `bunkers_storm_motion` |
| 250m interpolated samples | `wx-ui/panels/hodograph.rs` | Hodograph display panel |

---

## 5. Bunkers Storm Motion

This is the most critical wind algorithm, and the one with the most implementation variants across the codebase.

### Reference

Bunkers, M.J., et al. (2000): "Predicting Supercell Motion Using a New Hodograph Technique." *Weather and Forecasting*, 15, 61-79.

### Overview

Bunkers storm motion estimates the movement of right-moving and left-moving supercells by:
1. Computing the mean wind in the 0-6 km layer
2. Computing a shear vector
3. Deviating 7.5 m/s perpendicular to that shear vector

The right-mover deviates to the right of the shear; the left-mover to the left.

### Implementation A: thermo.rs (Python-exposed via metrust-py)

This is the simplest implementation. It uses:
- **Simple level average** for the 0-6 km mean wind
- **0-6 km bulk shear** (highest level <= 6 km minus surface) for the shear vector
- No pressure weighting

```rust
pub fn bunkers_storm_motion(
    _p: &[f64], u: &[f64], v: &[f64], z: &[f64],
) -> ((f64, f64), (f64, f64)) {
    // Mean wind: simple average of all levels <= 6000m
    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut count = 0.0;
    for i in 0..z.len() {
        if z[i] <= 6000.0 {
            sum_u += u[i];
            sum_v += v[i];
            count += 1.0;
        }
    }
    let u_mean = sum_u / count;
    let v_mean = sum_v / count;

    // Shear: last level <= 6000m minus surface
    let (u_shr, v_shr) = {
        let mut u6 = u[0];
        let mut v6 = v[0];
        for i in 0..z.len() {
            if z[i] <= 6000.0 {
                u6 = u[i];
                v6 = v[i];
            }
        }
        (u6 - u[0], v6 - v[0])
    };

    let shear_mag = (u_shr * u_shr + v_shr * v_shr).sqrt();
    let d = 7.5; // m/s deviation

    // Perpendicular rotation
    let u_perp = -v_shr / shear_mag * d;
    let v_perp =  u_shr / shear_mag * d;

    let u_rm = u_mean + u_perp;
    let v_rm = v_mean + v_perp;
    let u_lm = u_mean - u_perp;
    let v_lm = v_mean - v_perp;

    ((u_rm, v_rm), (u_lm, v_lm))
}
```

Note: the pressure array `_p` is accepted but **not used** in this implementation.

#### Perpendicular Rotation (thermo.rs)

The perpendicular vector to the shear `(u_shr, v_shr)` is computed as:

```
u_perp = -v_shr / |shear| * 7.5
v_perp =  u_shr / |shear| * 7.5
```

This rotates the shear vector 90 degrees **counterclockwise**:
- The rotation `(-v, u)` is counterclockwise in standard math coordinates.
- In meteorological coordinates (where v is northward, u is eastward), this means the right-mover deviates to the **right** of the shear vector when viewed along the shear direction.

### Implementation B: composite.rs / derived.rs (SRH-embedded Bunkers)

The SRH computation includes an inline Bunkers calculation that uses:
- **Height-weighted trapezoidal** mean wind for the 0-6 km layer
- **Interpolated** shear: `interp_at_height(6000)` minus `u_prof[0]`

```rust
// Shear vector
let u_sfc = u_prof[0];
let v_sfc = v_prof[0];
let u_6km = interp_at_height(mean_depth, heights, u_prof);
let v_6km = interp_at_height(mean_depth, heights, v_prof);
let shear_u = u_6km - u_sfc;
let shear_v = v_6km - v_sfc;

// Perpendicular deviation
let (dev_u, dev_v) = if shear_mag > 0.1 {
    let scale = 7.5 / shear_mag;
    (shear_v * scale, -shear_u * scale)
} else {
    (0.0, 0.0)
};
```

#### Perpendicular Rotation (composite.rs)

```
dev_u =  shear_v / |shear| * 7.5
dev_v = -shear_u / |shear| * 7.5
```

This is the rotation `(v, -u)` -- a 90-degree **clockwise** rotation.

**This is the opposite sign from thermo.rs.** The comment in `composite.rs` says "rotate shear 90 degrees clockwise". Both claim to produce the right-mover, but they use different rotation directions.

### Implementation C: Hodograph UI (wx-ui/panels/hodograph.rs)

Uses 250m-sampled mean wind and interpolated shear, with the same clockwise rotation as composite.rs:

```rust
fn bunkers(h: &[f64], u: &[f64], v: &[f64]) -> Option<(f64, f64)> {
    let (mu, mv) = mean_wind(h, u, v, 0.0, 6000.0)?;
    let (u0, v0) = (interp_at_h(h, u, 0.0)?, interp_at_h(h, v, 0.0)?);
    let (u6, v6) = (interp_at_h(h, u, 6000.0)?, interp_at_h(h, v, 6000.0)?);
    let (su, sv) = (u6 - u0, v6 - v0);
    let sm = (su * su + sv * sv).sqrt();
    // Right-mover: (sv, -su) -- clockwise rotation
    Some((mu + sv / sm * 7.5, mv - su / sm * 7.5))
}
```

Note: this works in **knots** (the hodograph data is in knots), and 7.5 is applied as 7.5 knots, not m/s.

### Implementation D: Rendering Hodograph (rustmet-core/src/render/hodograph.rs)

Also uses the clockwise `(dv, -du)` rotation, but converts the 7.5 m/s deviation to knots first:

```rust
let dev = 7.5 * 1.94384; // m/s to knots
let perp_u = dv / shear_mag * dev;
let perp_v = -du / shear_mag * dev;
```

### The Rotation Sign Issue

There are **two different rotation conventions** in the codebase:

| Implementation | Rotation | Formula | Convention |
|---------------|----------|---------|------------|
| `thermo.rs` | `(-v_shr, u_shr)` | Counterclockwise | **Differs from others** |
| `composite.rs` | `(v_shr, -u_shr)` | Clockwise | MetPy convention |
| `hodograph.rs` (UI) | `(sv, -su)` | Clockwise | MetPy convention |
| `hodograph.rs` (render) | `(dv, -du)` | Clockwise | MetPy convention |

The **MetPy convention** (and SHARPpy convention) uses the clockwise rotation `(shear_v, -shear_u)` to get the right-mover. This is what `composite.rs` and both hodograph implementations use.

The `thermo.rs` implementation uses `(-shear_v, shear_u)` which is the **opposite** direction. This means `thermo.rs` produces the right-mover at the position where the others would put the left-mover, and vice versa.

This is likely a sign bug in `thermo.rs`. The function returns `((u_rm, v_rm), (u_lm, v_lm))` but with the perpendicular reversed, the labels are swapped relative to the meteorological standard.

### Comparison with MetPy's Bunkers

MetPy uses a more sophisticated algorithm:
- **Pressure-weighted continuous averaging** (WCA) via trapezoidal integration in pressure: `mean = integral(A dp) / integral(dp)`, with three separate layer means:
  - Surface to 6 km: overall mean wind
  - Surface to 0.5 km: tail of shear (low-level)
  - 5.5 km to 6.0 km: head of shear (upper-level)
- The shear vector is `head_mean - tail_mean` (not the simple 0-6 km bulk shear)
- Boundary values are interpolated in height before integration

Metrust's implementations are all closer to the **SHARPpy approach**: simple 0-6 km mean wind and 0-6 km bulk shear, without pressure weighting or the three-layer decomposition.

### Comparison with SHARPpy

SHARPpy uses:
- Simple level average for mean wind (like `thermo.rs`)
- 0-6 km bulk shear directly (surface to 6 km interpolated)
- The same 7.5 m/s deviation magnitude

Metrust's `thermo.rs` closely mirrors SHARPpy's approach, while `composite.rs` adds height-weighted averaging.

---

## 6. Storm-Relative Helicity (SRH)

### Source files
- `crates/wx-math/src/composite.rs` (`compute_srh`, `compute_srh_column`)
- `crates/wx-sounding/src/derived.rs` (`compute_srh_column`)
- `crates/wx-ui/src/panels/hodograph.rs` (`srh`)

### Mathematical Definition

SRH integrates the cross product of the storm-relative wind with the wind shear over a depth:

```
SRH = integral from 0 to h of (V - C) x (dV/dz) dz
```

Where `V` is the wind vector, `C` is the storm motion vector.

### Discrete Implementation (composite.rs / derived.rs)

The main implementation sums layer-by-layer contributions:

```rust
for k in 0..nz - 1 {
    if heights[k] >= top_m { break; }

    // Get layer top/bottom winds (with interpolation at boundary)
    let sr_u_bot = u_bot - storm_u;
    let sr_v_bot = v_bot - storm_v;
    let sr_u_top = u_top_val - storm_u;
    let sr_v_top = v_top_val - storm_v;

    let du = u_top_val - u_bot;
    let dv = v_top_val - v_bot;
    let avg_sr_u = 0.5 * (sr_u_bot + sr_u_top);
    let avg_sr_v = 0.5 * (sr_v_bot + sr_v_top);

    srh += avg_sr_u * dv - avg_sr_v * du;
}
```

The formula for each layer is:

```
SRH_layer = avg_sr_u * dv - avg_sr_v * du
```

Where:
- `avg_sr_u = 0.5 * (sr_u[k] + sr_u[k+1])` -- average storm-relative u
- `avg_sr_v = 0.5 * (sr_v[k] + sr_v[k+1])` -- average storm-relative v
- `du = u[k+1] - u[k]` -- wind change across the layer
- `dv = v[k+1] - v[k]` -- wind change across the layer

This is equivalent to the trapezoidal approximation of the cross-product integral.

### MetPy-style Cross Product

An equivalent formulation (used by MetPy) is:

```
SRH_layer = sr_u[k+1] * sr_v[k] - sr_u[k] * sr_v[k+1]
```

The metrust formula `avg_sr_u * dv - avg_sr_v * du` expands to:

```
= 0.5*(sr_u_bot + sr_u_top)*(v_top - v_bot) - 0.5*(sr_v_bot + sr_v_top)*(u_top - u_bot)
```

Since `sr_u = u - storm_u`, substituting and simplifying shows this is equivalent to the cross-product form. Both compute the signed area of the parallelogram formed by consecutive storm-relative wind vectors.

### Height Interpolation at Boundary

When the depth boundary (e.g., 1000m or 3000m) falls between two data levels, the wind is linearly interpolated:

```rust
let (u_top_val, v_top_val) = if h_top < heights[k + 1] {
    let frac = (h_top - heights[k]) / (heights[k + 1] - heights[k]);
    (
        u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]),
        v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]),
    )
} else {
    (u_prof[k + 1], v_prof[k + 1])
};
```

### Sign Convention

- **Positive SRH** indicates cyclonic (counterclockwise) curvature of the storm-relative hodograph -- favorable for right-moving supercells in the Northern Hemisphere.
- **Negative SRH** indicates anticyclonic curvature -- favorable for left-movers.

### Hodograph UI SRH (slightly different formula)

The `wx-ui/panels/hodograph.rs` version uses a different discrete formula:

```rust
val += (sr2u - sr1u) * (sr2v + sr1v) - (sr2v - sr1v) * (sr2u + sr1u);
// ...
val * 0.5 * kt_to_ms * kt_to_ms
```

This expands to:

```
SRH_layer = 0.5 * [(du_sr)(sum_sr_v) - (dv_sr)(sum_sr_u)]
```

Where `du_sr = sr_u[k+1] - sr_u[k]`, `sum_sr_v = sr_v[k+1] + sr_v[k]`, etc. Expanding:

```
= 0.5 * [sr_u[k+1]*sr_v[k+1] + sr_u[k+1]*sr_v[k] - sr_u[k]*sr_v[k+1] - sr_u[k]*sr_v[k]
       - sr_v[k+1]*sr_u[k+1] - sr_v[k+1]*sr_u[k] + sr_v[k]*sr_u[k+1] + sr_v[k]*sr_u[k]]
= 0.5 * [2*sr_u[k+1]*sr_v[k] - 2*sr_u[k]*sr_v[k+1]]
= sr_u[k+1]*sr_v[k] - sr_u[k]*sr_v[k+1]
```

This is exactly the MetPy cross-product form. The factor of `0.5` in the code is absorbed because the intermediate `val` accumulates the expanded form before the final multiplication.

The hodograph UI version also includes the `kt_to_ms^2` conversion since it operates on wind in knots but outputs SRH in m^2/s^2.

### Storm Motion for SRH

All SRH implementations compute Bunkers right-mover storm motion inline (as described in Section 5) rather than accepting it as an input parameter. The storm motion used is always the **right-mover**.

---

## 7. Critical Angle

### Source file
- `crates/wx-math/src/composite.rs`

### Definition

The critical angle is the angle between:
1. The **storm-relative inflow vector** at the surface
2. The **0-500m bulk shear vector**

Values near 90 degrees are associated with enhanced tornado potential (Esterheld and Giuliano, 2008).

### Algorithm

```rust
pub fn critical_angle(
    u_storm: &[f64], v_storm: &[f64],
    u_shear: &[f64], v_shear: &[f64],
    nx: usize, ny: usize,
) -> Vec<f64> {
    // Storm-relative inflow: negative of storm motion
    let inflow_u = -u_storm[i];
    let inflow_v = -v_storm[i];
    let shear_u = u_shear[i];
    let shear_v = v_shear[i];

    let dot = inflow_u * shear_u + inflow_v * shear_v;
    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

    let cos_angle = (dot / (mag_inflow * mag_shear)).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}
```

### Formula

```
inflow = (-u_storm, -v_storm)
angle = arccos( (inflow . shear) / (|inflow| * |shear|) )
```

The result is in degrees, range [0, 180].

### Inputs

- `u_storm, v_storm`: Storm motion components in m/s (typically from Bunkers right-mover)
- `u_shear, v_shear`: 0-500m bulk shear vector components in m/s

The shear inputs should be computed from `bulk_shear()` or `compute_shear()`, not from raw level differences. This ensures consistent interpolation to the 0m and 500m levels.

### Storm-Relative Inflow Definition

The storm-relative inflow is defined as `(-u_storm, -v_storm)` -- the negative of the storm motion. This represents the direction from which air flows into the storm in the storm's reference frame. The surface wind itself is not used; the assumption is that the storm-relative inflow at the surface is dominated by the storm's own motion.

### Edge Cases

- If either `|inflow| < 0.01` or `|shear| < 0.01` m/s, the function returns `NaN` to avoid division by zero.
- The `clamp(-1.0, 1.0)` on `cos_angle` guards against floating-point errors outside the valid range of `acos`.

---

## 8. Corfidi Storm Motion

### Source file
- `crates/wx-math/src/thermo.rs`

### Purpose

Corfidi vectors estimate the motion of mesoscale convective systems (MCSs), which often move differently from individual supercells. The method produces two vectors:
- **Upshear (Corfidi-U)**: propagation-dominant MCS motion (back-building or training)
- **Downshear (Corfidi-D)**: advection-dominant MCS motion

### Algorithm

#### Step 1: Cloud-Layer Mean Wind (850-300 hPa)

A simple level average of all sounding levels with pressure between 850 and 300 hPa:

```rust
for i in 0..p.len() {
    if p[i] <= 850.0 && p[i] >= 300.0 {
        sum_u_cl += u[i];
        sum_v_cl += v[i];
        count_cl += 1.0;
    }
}
let u_cl = sum_u_cl / count_cl;
let v_cl = sum_v_cl / count_cl;
```

This is an **unweighted** average of all data levels in the 850-300 hPa layer. No pressure weighting or height weighting is applied.

#### Step 2: Low-Level Jet (LLJ) Detection

The LLJ is defined as the maximum wind speed in the lowest 1500m AGL:

```rust
let mut max_spd = 0.0_f64;
let mut u_llj = u[0];
let mut v_llj = v[0];
for i in 0..z.len() {
    if z[i] > 1500.0 { break; }
    let spd = (u[i] * u[i] + v[i] * v[i]).sqrt();
    if spd > max_spd {
        max_spd = spd;
        u_llj = u[i];
        v_llj = v[i];
    }
}
```

The search terminates (via `break`) at the first level above 1500m, relying on the data being ordered surface-first with increasing height. If no level exceeds the surface wind, the surface wind itself is used as the LLJ.

#### Step 3: Corfidi Vectors

```
Upshear:    V_up   = V_cl - V_llj
Downshear:  V_down = V_cl + V_up = 2*V_cl - V_llj
```

```rust
let u_up = u_cl - u_llj;
let v_up = v_cl - v_llj;
let u_down = u_cl + u_up;   // = 2*u_cl - u_llj
let v_down = v_cl + v_up;
```

### Physical Interpretation

- The **upshear vector** represents the propagation component: new cells form on the upshear (upwind) flank of the MCS, displaced opposite to the LLJ.
- The **downshear vector** adds the cloud-layer advection to this propagation, giving the net MCS motion for systems where advection dominates.

### Returns

```rust
((u_up, v_up), (u_down, v_down))
```

Returns `((0, 0), (0, 0))` if no data levels fall within the 850-300 hPa range.

---

## Summary of Implementation Differences

| Feature | thermo.rs | composite.rs | hodograph UI | MetPy |
|---------|-----------|-------------|-------------|-------|
| Mean wind method | Simple level avg | Height-weighted trapz | 250m interpolated | Pressure-weighted trapz |
| Shear vector | 0-6km bulk | Interpolated 0-6km | Interpolated 0-6km | Head-tail (5.5-6 vs 0-0.5 km) |
| Bunkers rotation | `(-v, u)` CCW | `(v, -u)` CW | `(v, -u)` CW | `(v, -u)` CW |
| Deviation magnitude | 7.5 m/s | 7.5 m/s | 7.5 kt | 7.5 m/s |
| Pressure used | No | No | No | Yes (WCA integral) |
| Units | m/s | m/s | knots | m/s |
