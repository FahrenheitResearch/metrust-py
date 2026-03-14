# Severe Weather Parameter Calculations

This document describes the severe weather parameter calculations implemented in
metrust. All functions live in the `wx-math` crate, split between two modules:

- **`thermo.rs`** -- single-column thermodynamic functions (CAPE/CIN, parcel
  lifting, stability indices, DCAPE, Galvez-Davison Index, Bunkers storm motion).
- **`composite.rs`** -- grid-level composites that iterate over every column in a
  3D field using `rayon::par_iter` (STP, SCP, SHIP, bulk shear, SRH, lapse rates,
  fire weather indices, winter weather diagnostics).

Source paths (relative to the repository root):

```
crates/wx-math/src/thermo.rs
crates/wx-math/src/composite.rs
```

---

## 1. Significant Tornado Parameter (STP)

**Function:** `composite::compute_stp`

### Formula

```
STP = cape_term * lcl_term * srh_term * shear_term
```

where:

| Term | Definition | Normalization |
|------|-----------|---------------|
| `cape_term` | `max(SBCAPE / 1500, 0)` | 1500 J/kg |
| `lcl_term`  | `clamp((2000 - LCL_height) / 1000, 0, 2)` | 1000 m; capped at 2.0 |
| `srh_term`  | `max(SRH_0-1km / 150, 0)` | 150 m^2/s^2 |
| `shear_term` | `clamp(SHEAR_0-6km / 20, 0, 1.5)` | 20 m/s; capped at 1.5 |

### Units

- SBCAPE: J/kg
- LCL height: meters AGL
- SRH (0-1 km): m^2/s^2
- Bulk shear (0-6 km): m/s
- STP: dimensionless

### Operational Thresholds

| STP Value | Interpretation |
|-----------|---------------|
| < 1       | Significant tornadoes unlikely |
| 1 -- 4    | Conditional tornado environment; watch for mesocyclones |
| 4 -- 8    | Favorable for strong (EF2+) tornadoes |
| > 8       | Particularly dangerous situation; violent tornadoes possible |

---

## 2. Supercell Composite Parameter (SCP)

Two versions are implemented.

### Simple SCP

**Function:** `composite::compute_scp`

```
SCP = (MUCAPE / 1000) * (SRH_3km / 50) * (SHEAR_6km / 40)
```

All terms are floored at 0.

### Enhanced SCP

**Function:** `composite::supercell_composite_parameter`

```
SCP = (MUCAPE / 1000) * (SRH_3km / 50) * (SHEAR_6km / 40) * CIN_term
```

The CIN adjustment:

```
CIN_term = 1.0              if MUCIN > -40 J/kg
CIN_term = -40 / MUCIN      otherwise (reduces SCP when CIN is strong)
```

### Units

- MUCAPE: J/kg
- SRH (0-3 km): m^2/s^2
- Bulk shear (0-6 km): m/s
- MUCIN: J/kg (negative values)
- SCP: dimensionless

### Operational Thresholds

| SCP Value | Interpretation |
|-----------|---------------|
| < 1       | Supercell development unlikely |
| 1 -- 4    | Marginally favorable for supercells |
| 4 -- 10   | Favorable for sustained supercells |
| > 10      | Strongly favorable; long-lived supercells likely |

---

## 3. Significant Hail Parameter (SHIP)

**Function:** `composite::significant_hail_parameter`

### Formula

```
SHIP = (MUCAPE * MR * LR_700_500 * (-T500) * SHEAR_06) / 42,000,000
```

When MUCAPE < 1300 J/kg, an additional scaling is applied:

```
SHIP = SHIP * (MUCAPE / 1300)
```

All component values are floored at 0.

### Components

| Symbol | Description | Units |
|--------|-------------|-------|
| `MUCAPE` | Most-Unstable CAPE | J/kg |
| `MR` | Low-level mixing ratio | g/kg |
| `LR_700_500` | 700-500 hPa lapse rate | C/km |
| `-T500` | Negated 500 hPa temperature (positive when cold aloft) | C |
| `SHEAR_06` | 0-6 km bulk shear magnitude | m/s |

### Operational Thresholds

| SHIP Value | Interpretation |
|------------|---------------|
| < 0.5      | Significant hail unlikely |
| 0.5 -- 1.0 | Marginally favorable for large hail |
| 1.0 -- 2.0 | Favorable for significant hail (>=2 inch) |
| > 2.0      | Very favorable; giant hail possible |

---

## 4. Bulk Richardson Number (BRN)

**Function:** `composite::bulk_richardson_number`

### Formula

```
BRN = CAPE / (0.5 * shear_06^2)
```

When `0.5 * shear_06^2 < 0.1`, the function returns `NaN` (near-zero shear
makes BRN undefined).

### Units

- CAPE: J/kg
- shear_06: m/s (0-6 km bulk shear magnitude)
- BRN: dimensionless

### Operational Thresholds

| BRN Value | Interpretation |
|-----------|---------------|
| < 10      | Strongly shear-dominated; may be too hostile for sustained updrafts |
| 10 -- 45  | Favors supercells (balance of buoyancy and shear) |
| > 45      | Buoyancy-dominated; multicell/pulse storms more likely |

---

## 5. Critical Angle

**Function:** `composite::critical_angle`

The angle between the storm-relative inflow vector and the 0-500 m shear vector.
Values near 90 degrees favor low-level mesocyclone development and
tornadogenesis.

### Convention

The inflow vector is defined as:

```
inflow_u = -u_storm
inflow_v = -v_storm
```

This is the *negated storm motion*, representing the direction from which air
flows toward the storm. Note the sign convention: `u_storm - u[0]` is used for
the storm-relative wind, **not** `u[0] - u_storm`. The grid-level function
receives pre-computed storm motion and shear vectors.

The 0-500 m shear vector is passed separately as `(u_shear, v_shear)`.

### Computation

```
cos(angle) = dot(inflow, shear) / (|inflow| * |shear|)
critical_angle = acos(cos_angle)    [degrees, 0-180]
```

Returns `NaN` if either vector magnitude is less than 0.01 m/s.

### Operational Thresholds

| Critical Angle | Interpretation |
|----------------|---------------|
| < 60 deg       | Low-level rotation unlikely |
| 60 -- 120 deg  | Favorable geometry for mesocyclone/tornado |
| ~90 deg        | Optimal; maximizes streamwise vorticity tilting |
| > 120 deg      | Unfavorable orientation |

> **Cross-reference:** See also `wind.md` for detailed documentation of
> storm-relative wind calculations and the Bunkers storm motion method.

---

## 6. Stability Indices

### 6.1 K-Index

**Function:** `composite::k_index`

```
KI = (T850 - T500) + Td850 - (T700 - Td700)
```

All temperatures in Celsius. Combines low-level moisture (Td850), mid-level
lapse rate (T850 - T500), and mid-level moisture (T700 - Td700).

| K-Index | Thunderstorm Probability |
|---------|--------------------------|
| < 20    | None |
| 20 -- 25 | Isolated thunderstorms |
| 26 -- 30 | Widely scattered thunderstorms |
| 31 -- 35 | Scattered thunderstorms |
| > 35    | Numerous thunderstorms |

### 6.2 Total Totals (TT)

**Function:** `composite::total_totals`

```
TT = VT + CT = (T850 - T500) + (Td850 - T500)
```

Helper functions also expose the components separately:

- **Vertical Totals (VT):** `T850 - T500` (`composite::vertical_totals`)
- **Cross Totals (CT):** `Td850 - T500` (`composite::cross_totals`)

| Total Totals | Interpretation |
|-------------|----------------|
| < 44        | Thunderstorms unlikely |
| 44 -- 50    | Thunderstorms likely |
| 50 -- 55    | Severe thunderstorms possible |
| > 55        | Severe thunderstorms likely; tornadoes possible |

### 6.3 Showalter Index (SI)

**Function:** `composite::showalter_index`

Lifts the 850 hPa parcel to 500 hPa:

```
SI = T_env(500) - T_parcel(500)
```

The parcel is lifted dry-adiabatically to its LCL, then moist-adiabatically
to 500 hPa using the Wobus function (`satlift`).

| Showalter Index | Interpretation |
|----------------|----------------|
| > 3            | Stable; no significant convection |
| 1 -- 3         | Marginal instability |
| -3 -- 1        | Moderate instability; thunderstorms likely |
| < -3           | Extreme instability; severe thunderstorms likely |

### 6.4 Lifted Index (LI)

**Functions:** `thermo::lifted_index`, `composite::lifted_index`

Lifts the surface parcel to 500 hPa:

```
LI = T_env(500) - T_parcel(500)
```

Implementation is identical to the Showalter Index except the starting parcel
originates at the surface rather than 850 hPa.

| Lifted Index | Interpretation |
|-------------|----------------|
| > 0         | Stable |
| 0 to -3     | Marginally unstable |
| -3 to -6    | Moderately unstable |
| -6 to -9    | Very unstable |
| < -9        | Extremely unstable |

### 6.5 SWEAT Index

**Function:** `composite::sweat_index`

```
SWEAT = 12*Td850 + 20*(TT - 49) + 2*f850 + f500 + 125*(sin(d500 - d850) + 0.2)
```

Conditional rules:

- Term 1: set to 0 if `Td850 <= 0`
- Term 2: set to 0 if `TT <= 49`
- Term 5 (shear term): set to 0 unless all of:
  - `130 <= wdir850 <= 250`
  - `210 <= wdir500 <= 310`
  - `(wdir500 - wdir850) > 0`
  - `wspd850 >= 15 knots`
  - `wspd500 >= 15 knots`

The result is floored at 0.

| Symbol | Description | Units |
|--------|-------------|-------|
| `Td850` | 850 hPa dewpoint | Celsius |
| `TT` | Total Totals index | dimensionless |
| `f850`, `f500` | Wind speed at 850/500 hPa | knots |
| `d850`, `d500` | Wind direction at 850/500 hPa | degrees |

| SWEAT Value | Interpretation |
|-------------|---------------|
| < 150       | Non-severe thunderstorms |
| 150 -- 300  | Severe thunderstorms possible |
| 300 -- 400  | Severe thunderstorms likely |
| > 400       | Tornadoes possible |

### 6.6 Galvez-Davison Index (GDI)

**Function:** `thermo::galvez_davison_index`

Designed for tropical convection potential where mid-latitude indices
(like K-Index) perform poorly.

```
GDI = CBI + II - MWI
```

where:

- **CBI (Column Buoyancy Index):**
  `CBI = mean(theta_e_950, theta_e_850) - theta_e_700`
  Uses Bolton equivalent potential temperature at 950, 850, and 700 hPa.

- **MWI (Mid-level Warming Index):**
  `MWI = (T500_K - 243.15) * 1.5`
  Scaled departure of 500 hPa temperature from -30 C reference.

- **II (Inflow Index):**
  `II = max(SST_C - 25, 0) * 5`
  Sea surface temperature influence; activates only when SST > 25 C.

| GDI Value | Interpretation |
|-----------|---------------|
| < 0       | Convection suppressed |
| 0 -- 25   | Marginal tropical convection |
| 25 -- 45  | Scattered convection likely |
| > 45      | Widespread deep convection likely |

---

## 7. Fire Weather Indices

### 7.1 Fosberg Fire Weather Index (FFWI)

**Function:** `composite::fosberg_fire_weather_index`

Combines temperature, relative humidity, and wind speed into a single fire
danger metric.

#### Equilibrium Moisture Content (EMC)

```
if RH <= 10:  EMC = 0.03229 + 0.281073*RH - 0.000578*RH*T_F
if RH <= 50:  EMC = 2.22749 + 0.160107*RH - 0.01478*T_F
if RH >  50:  EMC = 21.0606 + 0.005565*RH^2 - 0.00035*RH*T_F - 0.483199*RH
```

Then scale: `m = max(EMC / 30, 0)`

#### Moisture Damping

```
eta = 1 - 2*m + 1.5*m^2 - 0.5*m^3
```

#### Final Index

```
FFWI = clamp(eta * sqrt(1 + wspd_mph^2) * 10/3, 0, 100)
```

| Input | Units |
|-------|-------|
| `T_F` | Fahrenheit |
| `RH` | Percent (0-100) |
| `wspd_mph` | Miles per hour |

| FFWI Value | Interpretation |
|------------|---------------|
| < 25       | Low fire danger |
| 25 -- 50   | Moderate fire danger |
| 50 -- 75   | High fire danger; red flag potential |
| > 75       | Extreme fire danger |

### 7.2 Haines Index

**Function:** `composite::haines_index`

Low-elevation variant using 950 and 850 hPa levels. Returns an integer 2-6.

| Component | Criterion | Score |
|-----------|-----------|-------|
| A (stability: T950 - T850) | <= 3 C | 1 |
| | 4 -- 7 C | 2 |
| | > 7 C | 3 |
| B (moisture: T850 - Td850) | <= 5 C | 1 |
| | 6 -- 9 C | 2 |
| | > 9 C | 3 |

```
Haines = A + B    (range: 2 -- 6)
```

| Haines | Interpretation |
|--------|---------------|
| 2 -- 3 | Very low fire growth potential |
| 4      | Low fire growth potential |
| 5      | Moderate fire growth potential |
| 6      | High fire growth potential |

### 7.3 Hot-Dry-Windy Index (HDW)

**Function:** `composite::hot_dry_windy`

```
HDW = VPD * wind_speed
```

If the caller provides a pre-computed VPD (vapor pressure deficit), it is used
directly. Otherwise VPD is computed as:

```
VPD = es(T_C) - ea
ea  = es(T_C) * (RH / 100)
```

where `es` is the SHARPpy polynomial saturation vapor pressure (`vappres`).

| Input | Units |
|-------|-------|
| `T_C` | Celsius |
| `RH` | Percent (0-100) |
| `wspd_ms` | m/s |
| `VPD` | hPa |

| HDW Value | Interpretation |
|-----------|---------------|
| < 100     | Low fire weather concern |
| 100 -- 400 | Moderate; monitor conditions |
| 400 -- 800 | High; significant fire weather |
| > 800     | Extreme fire weather |

---

## 8. Other Severe Parameters

### 8.1 Downdraft CAPE (DCAPE)

**Function:** `thermo::downdraft_cape`

Quantifies the potential energy available for convective downdrafts.

#### Algorithm

1. Find the level of minimum theta-e in the lowest 400 hPa.
2. From that level, descend moist-adiabatically to the surface using RK4
   integration (`moist_lapse`).
3. Integrate negative buoyancy (parcel colder than environment) downward:

```
DCAPE = sum( Rd * |Tv_parcel - Tv_env| * |ln(p[i]) - ln(p[i+1])| )
```

Only layers where the descending parcel is cooler than the environment
contribute. Virtual temperature corrections are applied to both parcel and
environment.

| DCAPE Value | Interpretation |
|-------------|---------------|
| < 200 J/kg  | Weak downdraft potential |
| 200 -- 800  | Moderate downdraft potential |
| 800 -- 1200 | Strong downdrafts; damaging winds possible |
| > 1200      | Extreme downdraft potential |

### 8.2 Freezing Rain Composite

**Function:** `composite::freezing_rain_composite`

Returns a scaled value 0-1 representing freezing rain likelihood.

#### Requirements

- Surface temperature must be <= 0 C (otherwise returns 0).
- A warm layer (T > 0 C) must exist above the surface.

#### Computation

```
depth_factor     = clamp(warm_depth / 100, 0, 1)    -- warm layer thickness in hPa
intensity_factor = clamp(warm_intensity / (warm_depth * 3), 0, 1)
base_score       = depth_factor * intensity_factor
```

where `warm_intensity = sum(T * dp)` over the warm layer.

A precipitation-type multiplier is applied:

- Freezing rain (type 4): multiply by 1.0
- Other types: multiply by 0.5

### 8.3 Dendritic Growth Zone (DGZ)

**Function:** `composite::dendritic_growth_zone`

Returns `(p_top, p_bottom)` in hPa bounding the layer where temperature is
between -12 C and -18 C. This is the optimal temperature range for dendritic
ice crystal growth, which maximizes snow production efficiency.

Crossing boundaries are interpolated linearly between model levels. Returns
`(NaN, NaN)` if the profile never enters the -12 to -18 C range.

### 8.4 Convective Inhibition Depth

**Function:** `composite::convective_inhibition_depth`

The pressure depth (hPa) from the surface to the LFC (or top of model if no
LFC exists) over which the lifted parcel is negatively buoyant.

```
CIN_depth = P_surface - P_LFC    [hPa]
```

The parcel is lifted dry-adiabatically to the LCL, then moist-adiabatically
via the Wobus/satlift method. The first level above the LCL where the
parcel virtual temperature exceeds the environment virtual temperature
marks the LFC.

### 8.5 Warm Nose Check

**Function:** `composite::warm_nose_check`

A boolean check for the presence of an elevated warm layer (warm nose) that
can produce freezing rain or ice pellets.

Returns `true` if the temperature profile (surface-first, decreasing pressure)
transitions from below freezing to above freezing at any point above the
surface. The algorithm:

1. Scan upward from the surface.
2. Mark when a below-freezing level is found.
3. If a subsequent level is above freezing, a warm nose exists.

---

## 9. Grid Composites -- Parallel Computation Architecture

All grid-level functions in `composite.rs` share a common pattern for
processing 3D model fields.

### Data Layout

3D fields are stored as flattened arrays in `[nz][ny][nx]` order (C-contiguous,
level-major). The index for level `k`, row `j`, column `i` is:

```
flat_index = k * ny * nx + j * nx + i
```

2D fields (e.g., surface pressure, 2-meter temperature) are `[ny][nx]`:

```
flat_index = j * nx + i
```

### Column Extraction

The helper `extract_column` pulls a vertical profile at grid point `(j, i)`:

```rust
fn extract_column(data: &[f64], nz: usize, ny: usize, nx: usize, j: usize, i: usize) -> Vec<f64> {
    let mut col = Vec::with_capacity(nz);
    for k in 0..nz {
        col.push(data[k * ny * nx + j * nx + i]);
    }
    col
}
```

### Parallel Iteration

Every grid function uses `rayon::par_iter` over the 2D index space:

```rust
let results: Vec<T> = (0..ny * nx)
    .into_par_iter()
    .map(|idx| {
        let j = idx / nx;
        let i = idx % nx;
        // extract columns, compute parameter, return scalar
    })
    .collect();
```

This distributes columns across all available CPU cores. Each column is
processed independently -- there are no horizontal dependencies in the
severe weather calculations.

### Profile Orientation

Model levels may arrive in either bottom-up or top-down order. All functions
check the first and last elements and reverse profiles if needed to ensure
a consistent ordering:

- **CAPE/CIN, Stability Indices:** surface-first, decreasing pressure (high
  pressure at index 0).
- **SRH, Bulk Shear, Lapse Rate:** surface-first, increasing height (low
  heights at index 0).

### Key Grid Functions

| Function | Inputs (3D) | Output (2D) | Description |
|----------|-------------|-------------|-------------|
| `compute_cape_cin` | P, T, Q, H_AGL, Psfc, T2, Q2 | CAPE, CIN, LCL, LFC | CAPE/CIN with parcel selection (sb/ml/mu) |
| `compute_srh` | U, V, H_AGL | SRH | Storm-relative helicity using Bunkers motion |
| `compute_shear` | U, V, H_AGL | shear magnitude | Bulk wind shear between two height levels |
| `compute_stp` | CAPE, LCL, SRH_1km, shear_6km | STP | Significant Tornado Parameter |
| `compute_scp` | MUCAPE, SRH_3km, shear_6km | SCP | Supercell Composite Parameter |
| `compute_ehi` | CAPE, SRH | EHI | Energy-Helicity Index = CAPE*SRH/160000 |
| `compute_lapse_rate` | T, Q, H_AGL | LR (C/km) | Lapse rate between two heights |
| `compute_pw` | Q, P | PW (mm) | Precipitable water via (1/g)*integral(Q*dp) |
| `significant_hail_parameter` | CAPE, shear, T500, LR, MR | SHIP | Significant Hail Parameter |
| `supercell_composite_parameter` | MUCAPE, SRH, shear, MUCIN | SCP | Enhanced SCP with CIN term |

### Unit Conventions in `compute_cape_cin`

The function auto-detects input units and converts internally:

- Pressure > 2000 is assumed Pa and divided by 100 to get hPa.
- Temperature > 150 is assumed Kelvin and converted to Celsius.
- Dewpoint is capped at temperature (`Td <= T`).
- Surface data is prepended to profiles before parcel selection and integration.

### CAPE Integration Method

CAPE is computed via the density-weighted buoyancy integral:

```
CAPE = sum( Rd * (Tv_parcel - Tv_env) * ln(P_bottom / P_top) )
```

for layers where the parcel is warmer than the environment (positive buoyancy).
CIN accumulates the same expression where the parcel is cooler.

Two passes are made:

1. **Geometric scan** to find LFC and EL by locating buoyancy sign changes.
2. **Integration pass** with sub-stepping (layers > 10 hPa thick are subdivided)
   for accuracy.

Below the LCL, the parcel follows a dry adiabat with constant mixing ratio.
Above the LCL, the parcel follows a moist adiabat computed via the Wobus
function (`satlift`). Virtual temperature corrections are applied to both
parcel and environment throughout.

---

## Appendix: Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Dry air gas constant | Rd | 287.058 | J/(kg K) |
| Water vapor gas constant | Rv | 461.5 | J/(kg K) |
| Specific heat at constant pressure | Cp | 1005.7 | J/(kg K) |
| Gravitational acceleration | g | 9.80665 | m/s^2 |
| Rd/Cp | kappa | 0.28571426 | dimensionless |
| 0 Celsius in Kelvin | -- | 273.15 | K |
| Rd/Rv (molecular weight ratio) | epsilon | 0.62197 | dimensionless |
| Latent heat of vaporization | Lv | 2.501 x 10^6 | J/kg |
