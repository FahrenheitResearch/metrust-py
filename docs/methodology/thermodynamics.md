# Thermodynamic Algorithms in metrust

> **Source file:** `crates/wx-math/src/thermo.rs`
>
> This document describes the exact algorithms, formulas, constants, and implementation
> details used in metrust's thermodynamic calculations. It is intended as a reference for
> anyone reimplementing these routines in another language (Julia, Fortran, Python, etc.).

---

## 1. Physical Constants

All constants are defined at module scope in `thermo.rs` and are used throughout
every thermodynamic calculation. The values are MetPy-exact (matching the
`metpy.constants` module and SHARPpy conventions).

| Symbol   | Constant Name  | Value              | Units       | Description                                |
|----------|---------------|--------------------|--------------|--------------------------------------------|
| $R_d$    | `RD`          | 287.058            | J/(kg K)     | Specific gas constant for dry air          |
| $R_v$    | `RV`          | 461.5              | J/(kg K)     | Specific gas constant for water vapor      |
| $c_p$    | `CP`          | 1005.7             | J/(kg K)     | Specific heat of dry air at constant pressure |
| $g$      | `G`           | 9.80665            | m/s^2        | Standard gravitational acceleration        |
| $\kappa$ | `ROCP`        | 0.28571426         | dimensionless| $R_d / c_p$ (Poisson exponent)             |
| $T_0$    | `ZEROCNK`     | 273.15             | K            | 0 degrees Celsius in Kelvin                        |
| $\varepsilon$ | `EPS`    | 0.62197            | dimensionless| $R_d/R_v = M_w/M_d$ (molecular weight ratio) |
| $L_v$    | `LV`          | 2.501 x 10^6       | J/kg         | Latent heat of vaporization at 0 degrees C         |
| $\Gamma_s$ | `LAPSE_STD` | 0.0065             | K/m          | Standard atmosphere lapse rate             |
| $p_{0s}$ | `P0_STD`      | 1013.25            | hPa          | Standard sea-level pressure                |
| $T_{0s}$ | `T0_STD`      | 288.15             | K            | Standard sea-level temperature             |
|          | `MISSING`     | -9999.0            |              | Sentinel for missing data                  |

```rust
pub const RD: f64 = 287.058;
pub const RV: f64 = 461.5;
pub const CP: f64 = 1005.7;
pub const G: f64 = 9.80665;
pub const ROCP: f64 = 0.28571426;  // Rd/Cp
pub const ZEROCNK: f64 = 273.15;
pub const EPS: f64 = 0.62197;      // Rd/Rv
pub const LV: f64 = 2.501e6;
pub const LAPSE_STD: f64 = 0.0065;
pub const P0_STD: f64 = 1013.25;
pub const T0_STD: f64 = 288.15;
```

**Note on `ROCP`:** The value 0.28571426 is $2/7$ to float64 precision, which equals
$R_d/c_p = 287.058/1005.7 \approx 0.28539$... The code uses the rounded $2/7$ value
consistent with SHARPpy and many operational codes rather than the ratio of the declared
constants. This is a deliberate choice for compatibility.

---

## 2. Potential Temperature

### Poisson's Equation

$$\theta = T \left(\frac{1000}{p}\right)^{R_d/c_p}$$

where $T$ is in Kelvin, $p$ is in hPa, and $R_d/c_p$ = `ROCP` = 0.28571426.

```rust
pub fn potential_temperature(p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    t_k * (1000.0 / p_hpa).powf(ROCP)
}
```

**Units:** Input temperature in Celsius, pressure in hPa. Output in Kelvin.

### Inverse Poisson (Temperature from Theta)

$$T = \theta \left(\frac{p}{1000}\right)^{R_d/c_p}$$

```rust
pub fn temperature_from_potential_temperature(p_hpa: f64, theta_k: f64) -> f64 {
    theta_k * (p_hpa / 1000.0).powf(ROCP)
}
```

### Virtual Potential Temperature

$$\theta_v = \theta \cdot (1 + 0.61 \, w)$$

where $w$ is the mixing ratio in kg/kg.

```rust
pub fn virtual_potential_temperature(p_hpa: f64, t_c: f64, w_gkg: f64) -> f64 {
    let theta = potential_temperature(p_hpa, t_c);
    let w = w_gkg / 1000.0;
    theta * (1.0 + 0.61 * w)
}
```

### Exner Function

$$\Pi = \left(\frac{p}{p_0}\right)^{R_d/c_p}$$

```rust
pub fn exner_function(p_hpa: f64) -> f64 {
    (p_hpa / 1000.0).powf(ROCP)
}
```

---

## 3. Equivalent Potential Temperature

The codebase contains **two** equivalent potential temperature implementations:

### 3a. SHARPpy-legacy `thetae()` (used in CAPE integration)

This function lifts the parcel to its LCL, computes potential temperature at the LCL,
then applies a simplified exponential correction for latent heat release.

$$\theta_e = \theta_{LCL} \cdot \exp\!\left(\frac{L_c \cdot r}{c_p \cdot T_{LCL}}\right)$$

where:
- $\theta_{LCL} = (T_{LCL} + 273.15) \cdot (1000/p_{LCL})^\kappa$
- $r$ = mixing ratio at original level (kg/kg)
- $L_c = 2500 - 2.37 \cdot T_{LCL}$ (kJ/kg, temperature-dependent latent heat)
- The exponential argument is $L_c \cdot 1000 \cdot r / (c_p \cdot T_{LCL,K})$

```rust
pub fn thetae(p: f64, t: f64, td: f64) -> f64 {
    let (p_lcl, t_lcl) = drylift(p, t, td);
    let theta = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
    let r = mixratio(p, td) / 1000.0;       // kg/kg
    let lc = 2500.0 - 2.37 * t_lcl;          // kJ/kg
    let te_k = theta * ((lc * 1000.0 * r) / (CP * (t_lcl + ZEROCNK))).exp();
    te_k - ZEROCNK  // returns Celsius
}
```

**Output:** Celsius (note: unusual convention; most references give theta_e in Kelvin).

### 3b. Bolton (1980) `equivalent_potential_temperature()` (MetPy-compatible)

This implements Bolton (1980) equation 39 exactly as MetPy does:

**Step 1 -- Bolton LCL Temperature** (Bolton eq. 15):

$$T_{LCL} = \frac{56 + \cfrac{1}{\cfrac{1}{T_d - 56} + \cfrac{\ln(T/T_d)}{800}}}{1}$$

(all temperatures in Kelvin)

**Step 2 -- Vapor pressure and mixing ratio:**

$$e = 6.112 \cdot \exp\!\left(\frac{17.67 \cdot T_d}{T_d + 243.5}\right), \qquad r = \frac{\varepsilon \cdot e}{p - e}$$

**Step 3 -- Dry-leg potential temperature:**

$$\theta_{DL} = T \cdot \left(\frac{1000}{p - e}\right)^\kappa \cdot \left(\frac{T}{T_{LCL}}\right)^{0.28 \, r}$$

**Step 4 -- Final theta_e:**

$$\theta_e = \theta_{DL} \cdot \exp\!\left[\left(\frac{3036}{T_{LCL}} - 1.78\right) \cdot r \cdot (1 + 0.448 \, r)\right]$$

```rust
pub fn equivalent_potential_temperature(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let td_k = td_c + ZEROCNK;
    let t_lcl = 56.0 + 1.0 / (1.0 / (td_k - 56.0) + (t_k / td_k).ln() / 800.0);
    let e = saturation_vapor_pressure(td_c);
    let r = EPS * e / (p_hpa - e);  // kg/kg
    let theta_dl = t_k * (1000.0 / (p_hpa - e)).powf(ROCP) * (t_k / t_lcl).powf(0.28 * r);
    theta_dl * ((3036.0 / t_lcl - 1.78) * r * (1.0 + 0.448 * r)).exp()
}
```

**Output:** Kelvin.

---

## 4. Saturation Vapor Pressure

The codebase contains **two** saturation vapor pressure formulas:

### 4a. SHARPpy 8th-order polynomial `vappres()` (Eschner)

Used by the SHARPpy-heritage functions (`mixratio`, `satlift`, `wobf`-based CAPE).

$$e_s(T) = \frac{6.1078}{P(T)^8}$$

where $P(T)$ is a nested polynomial in temperature (Celsius):

```rust
pub fn vappres(t: f64) -> f64 {
    let pol = t * (1.1112018e-17 + (t * -3.0994571e-20));
    let pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)));
    let pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)));
    let pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)));
    let pol = 0.99999683 + (t * (-9.082695e-03 + pol));
    6.1078 / pol.powi(8)
}
```

This is a rational approximation that avoids the transcendental `exp()` call.

### 4b. Bolton (1980) formula `saturation_vapor_pressure()`

Used by the MetPy-compatible functions (`equivalent_potential_temperature`, `saturation_mixing_ratio`, `dewpoint_from_rh`, etc.).

$$e_s(T) = 6.112 \cdot \exp\!\left(\frac{17.67 \cdot T}{T + 243.5}\right)$$

where $T$ is in Celsius.

```rust
pub fn saturation_vapor_pressure(t_c: f64) -> f64 {
    6.112 * ((17.67 * t_c) / (t_c + 243.5)).exp()
}
```

### Ice/liquid phase handling

Neither formula distinguishes between ice and liquid phases. Both compute saturation
vapor pressure over **liquid water** at all temperatures. The frost point function
uses a separate Magnus formula over ice:

$$e_i(T) = 6.112 \cdot \exp\!\left(\frac{22.46 \cdot T}{T + 272.62}\right)$$

---

## 5. Mixing Ratio, Specific Humidity, and Vapor Pressure

### Mixing ratio from pressure and temperature (SHARPpy)

Includes a **Wexler enhancement factor** for non-ideal gas behavior:

$$x = 0.02 \cdot (T - 12.5 + 7500/p)$$

$$f_w = 1 + 0.0000045 \cdot p + 0.0014 \cdot x^2$$

$$e_s^* = f_w \cdot e_s(T)$$

$$w = 621.97 \cdot \frac{e_s^*}{p - e_s^*} \quad \text{(g/kg)}$$

```rust
pub fn mixratio(p: f64, t: f64) -> f64 {
    let x = 0.02 * (t - 12.5 + (7500.0 / p));
    let wfw = 1.0 + (0.0000045 * p) + (0.0014 * x * x);
    let fwesw = wfw * vappres(t);
    621.97 * (fwesw / (p - fwesw))
}
```

### Saturation mixing ratio (Bolton-based)

$$w_s = \frac{\varepsilon \cdot e_s}{p - e_s} \cdot 1000 \quad \text{(g/kg)}$$

Clamped to non-negative values.

```rust
pub fn saturation_mixing_ratio(p_hpa: f64, t_c: f64) -> f64 {
    let es = saturation_vapor_pressure(t_c);
    (EPS * es / (p_hpa - es) * 1000.0).max(0.0)
}
```

### Conversion chain

The full conversion chain between moisture variables:

**Vapor pressure <-> Dewpoint** (Bolton, invertible):

$$e = 6.112 \cdot \exp\!\left(\frac{17.67 \cdot T_d}{T_d + 243.5}\right)$$

$$T_d = \frac{243.5 \cdot \ln(e/6.112)}{17.67 - \ln(e/6.112)}$$

**Mixing ratio <-> Specific humidity:**

$$q = \frac{w}{1 + w}, \qquad w = \frac{q}{1 - q}$$

where $w$ is in kg/kg. The code converts between g/kg and kg/kg as needed:

```rust
pub fn specific_humidity(p_hpa: f64, w_gkg: f64) -> f64 {
    let w = w_gkg / 1000.0;
    w / (1.0 + w)
}

pub fn mixing_ratio_from_specific_humidity(q: f64) -> f64 {
    (q / (1.0 - q)) * 1000.0  // returns g/kg
}
```

**Specific humidity from dewpoint:**

$$e = e_s(T_d), \qquad w = \frac{\varepsilon \cdot e}{p - e}, \qquad q = \frac{w}{1 + w}$$

**Dewpoint from specific humidity:**

$$w = \frac{q}{1 - q}, \qquad e = \frac{w \cdot p}{\varepsilon + w}, \qquad T_d = \text{dewpoint}(e)$$

**Vapor pressure from mixing ratio:**

$$e = \frac{w \cdot p}{\varepsilon + w}$$

**Relative humidity:**

$$RH = \frac{e_s(T_d)}{e_s(T)} \cdot 100\%$$

### Temperature at a given mixing ratio (SHARPpy)

An empirical formula from SHARPpy `params.py`:

$$x = \log_{10}\!\left(\frac{w \cdot p}{622 + w}\right)$$

$$T = 10^{c_1 x + c_2} - c_3 + c_4 \cdot (10^{c_5 x} - c_6)^2 - 273.15$$

where: $c_1 = 0.0498646455$, $c_2 = 2.4082965$, $c_3 = 7.07475$, $c_4 = 38.9114$, $c_5 = 0.0915$, $c_6 = 1.2035$.

---

## 6. Virtual Temperature

$$T_v = T_K \cdot (1 + 0.61 \, w) - 273.15$$

where $w$ is mixing ratio in kg/kg (computed from dewpoint and pressure via `mixratio`).

```rust
pub fn virtual_temp(t: f64, p: f64, td: f64) -> f64 {
    let w = mixratio(p, td) / 1000.0;
    let tk = t + ZEROCNK;
    let vt = tk * (1.0 + 0.61 * w);
    vt - ZEROCNK
}
```

**Inputs and output all in Celsius.** The 0.61 factor is $(R_v/R_d - 1) \approx (461.5/287.058 - 1) = 0.608$, rounded to 0.61.

---

## 7. Wet Bulb Temperature

The wet bulb temperature is computed via **Normand's construction** -- not a direct iterative solver on a single equation, but a two-step physical process:

1. **Lift the parcel dry-adiabatically to its LCL** (via `drylift`)
2. **Descend moist-adiabatically** from the LCL back to the original pressure (via `satlift`)

The moist descent uses the Wobus function and Newton-Raphson iteration (see section on `satlift` below).

```rust
pub fn wet_bulb_temperature(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    let (p_lcl, t_lcl) = drylift(p_hpa, t_c, td_c);
    let theta_c = t_lcl + ZEROCNK;
    let theta_sfc = theta_c * ((1000.0 / p_lcl).powf(ROCP));
    let theta_start_c = theta_sfc - ZEROCNK;
    let thetam = theta_start_c - wobf(theta_start_c) + wobf(t_lcl);
    satlift(p_hpa, thetam)
}
```

### The `satlift` function (Newton-Raphson saturated parcel descent/ascent)

Given a pressure $p$ and a "saturation potential temperature" $\theta_m$ (Celsius), find
the temperature of a saturated parcel at that pressure.

**Algorithm:** 7 iterations of Newton-Raphson with adaptive step sizing.

**Initial guess:**

$$T_1 = (\theta_m + 273.15) \cdot (p/1000)^\kappa - 273.15$$

**Error function:**

$$e_1 = \text{wobf}(T_1) - \text{wobf}(\theta_m)$$

**Newton-Raphson update (per iteration):**

$$T_2 = T_1 - e_1 \cdot \text{rate}$$

$$e_2 = \frac{T_2 + 273.15}{(p/1000)^\kappa} - 273.15$$

$$e_2 \mathrel{+}= \text{wobf}(T_2) - \text{wobf}(e_2) - \theta_m$$

$$\text{rate} = \frac{T_2 - T_1}{e_2 - e_1}$$

**Convergence criterion:** $|e_1| < 0.001$ (early exit). Maximum 7 iterations.

**Final answer:** $T_1 - e_1 \cdot \text{rate}$

```rust
pub fn satlift(p: f64, thetam: f64) -> f64 {
    if p >= 1000.0 { return thetam; }
    let pwrp = (p / 1000.0_f64).powf(ROCP);
    let mut t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK;
    let mut e1 = wobf(t1) - wobf(thetam);
    let mut rate = 1.0;
    for _ in 0..7 {
        if e1.abs() < 0.001 { break; }
        let t2 = t1 - (e1 * rate);
        let mut e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK;
        e2 += wobf(t2) - wobf(e2) - thetam;
        rate = (t2 - t1) / (e2 - e1);
        t1 = t2;
        e1 = e2;
    }
    t1 - e1 * rate
}
```

### The Wobus function `wobf()`

A polynomial approximation for computing moist adiabats, originally from SHARPpy.
Split into two branches at $T = 20$ degrees C:

**For $t' = T - 20 \leq 0$:**

$$\text{npol} = 1 + t'(-8.8417\text{e-3} + t'(1.4714\text{e-4} + t'(-9.672\text{e-7} + t'(-3.261\text{e-8} + t'(-3.860\text{e-10})))))$$

$$\text{wobf} = \frac{15.13}{\text{npol}^4}$$

**For $t' > 0$:**

$$\text{ppol} = 1 + t'(3.618\text{e-3} + t'(-1.360\text{e-5} + t'(4.962\text{e-7} + t'(-6.106\text{e-9} + t'(3.940\text{e-11} + t'(-1.259\text{e-13} + t'(1.669\text{e-16})))))))$$

$$\text{wobf} = \frac{29.93}{\text{ppol}^4} + 0.96 \cdot t' - 14.8$$

---

## 8. LCL (Lifted Condensation Level)

### Analytical approximation

The LCL temperature is computed using an empirical polynomial fit (from SHARPpy):

$$\Delta T = s \cdot (1.2185 + 0.001278 \cdot T + s \cdot (-0.00219 + 1.173\text{e-5} \cdot s - 5.2\text{e-6} \cdot T))$$

$$T_{LCL} = T - \Delta T$$

where $s = T - T_d$ is the dewpoint depression (both in Celsius).

```rust
pub fn lcltemp(t: f64, td: f64) -> f64 {
    let s = t - td;
    let dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s - 0.0000052 * t));
    t - dlt
}
```

### LCL Pressure (`drylift`)

Once $T_{LCL}$ is known, the LCL pressure is found by inverting the dry adiabat:

$$p_{LCL} = 1000 \cdot \left(\frac{T_{LCL} + 273.15}{(T + 273.15) \cdot (1000/p)^\kappa}\right)^{1/\kappa}$$

This follows from requiring:

$$\theta = (T + 273.15) \cdot (1000/p)^\kappa = (T_{LCL} + 273.15) \cdot (1000/p_{LCL})^\kappa$$

```rust
pub fn drylift(p: f64, t: f64, td: f64) -> (f64, f64) {
    let t_lcl = lcltemp(t, td);
    let p_lcl = 1000.0
        * (((t_lcl + ZEROCNK) / ((t + ZEROCNK) * ((1000.0 / p).powf(ROCP)))))
            .powf(1.0 / ROCP);
    (p_lcl, t_lcl)
}
```

**Returns:** `(p_lcl, t_lcl)` in (hPa, Celsius).

---

## 9. Moist Adiabatic Lapse Rate and RK4 Integration

### Moist lapse rate formula

The saturated adiabatic lapse rate $dT/dp$ is from Bakhshaii & Stull (2013):

$$\frac{dT}{dp} = \frac{1}{p} \cdot \frac{R_d T + L_v r_s}{c_p + \dfrac{L_v^2 \, r_s \, \varepsilon}{R_d \, T^2}}$$

where:
- $T$ is in Kelvin
- $p$ is in hPa (the 100x factor from Pa cancels between numerator and denominator treatment)
- $r_s = \varepsilon \cdot e_s / (p - e_s)$ is the saturation mixing ratio (kg/kg)
- $e_s$ uses the Bolton formula

```rust
fn moist_lapse_rate(p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let es = saturation_vapor_pressure(t_c);
    let rs = EPS * es / (p_hpa - es);
    let numerator = (RD * t_k + LV * rs) / p_hpa;
    let denominator = CP + (LV * LV * rs * EPS) / (RD * t_k * t_k);
    numerator / denominator
}
```

**Result:** K/hPa (positive when ascending, since pressure decreases upward).

### RK4 integration (`moist_lapse`)

The temperature along a moist adiabat is integrated using **4th-order Runge-Kutta** in
pressure coordinates.

**Step size selection:** Between any two pressure levels, the interval is subdivided into
$n$ sub-steps:

$$n = \max\!\left(4,\; \left\lfloor \frac{|\Delta p|}{5} \right\rfloor + 1\right)$$

So the sub-step size is at most 5 hPa, with a minimum of 4 steps per layer.

**RK4 scheme per sub-step:**

$$h = \Delta p / n$$

$$k_1 = h \cdot f(p, T)$$
$$k_2 = h \cdot f(p + h/2, \; T + k_1/2)$$
$$k_3 = h \cdot f(p + h/2, \; T + k_2/2)$$
$$k_4 = h \cdot f(p + h, \; T + k_3)$$
$$T \mathrel{+}= \frac{k_1 + 2k_2 + 2k_3 + k_4}{6}$$

where $f(p, T) = \text{moist\_lapse\_rate}(p, T)$.

```rust
pub fn moist_lapse(p: &[f64], t_start_c: f64) -> Vec<f64> {
    let mut t = t_start_c;
    for i in 1..p.len() {
        let dp = p[i] - p[i - 1];
        let n_steps = ((dp.abs() / 5.0) as usize).max(4);
        let h = dp / n_steps as f64;
        let mut p_c = p[i - 1];
        for _ in 0..n_steps {
            let k1 = h * moist_lapse_rate(p_c, t);
            let k2 = h * moist_lapse_rate(p_c + h / 2.0, t + k1 / 2.0);
            let k3 = h * moist_lapse_rate(p_c + h / 2.0, t + k2 / 2.0);
            let k4 = h * moist_lapse_rate(p_c + h, t + k3);
            t += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            p_c += h;
        }
    }
    // ...
}
```

### Full parcel profile (`parcel_profile`)

Combines dry and moist adiabats:

1. **Below LCL:** Dry adiabat: $T(p) = T_{sfc,K} \cdot (p / p_{sfc})^\kappa - 273.15$
2. **At and above LCL:** Moist adiabat via `moist_lapse()` starting from $(p_{LCL}, T_{LCL})$

The LCL is inserted as the first point in the moist pressure array to ensure the
moist adiabat starts exactly at the LCL temperature.

---

## 10. CAPE/CIN Integration

This is the core convective parameter computation. The codebase contains **two** CAPE/CIN
implementations that differ in their moist adiabat calculation but share the same
integration formula.

### 10.1 Integration formula (both implementations)

Both use the **pressure-log integral** form:

$$\text{CAPE} = \sum_{\text{layers}} R_d \cdot (T_{v,\text{parcel}} - T_{v,\text{env}}) \cdot \ln\!\left(\frac{p_1}{p_2}\right) \quad \text{where buoyancy} > 0$$

$$\text{CIN} = \sum_{\text{layers}} R_d \cdot (T_{v,\text{parcel}} - T_{v,\text{env}}) \cdot \ln\!\left(\frac{p_1}{p_2}\right) \quad \text{where buoyancy} < 0$$

This is derived from the hydrostatic equation. Starting from the geometric form
$\text{CAPE} = \int g \cdot \frac{T_{v,p} - T_{v,e}}{T_{v,e}} \, dz$ and substituting
$dz = -\frac{R_d T_v}{g} \cdot \frac{dp}{p}$, one obtains:

$$\text{CAPE} = \int_{p_{LFC}}^{p_{EL}} R_d \cdot (T_{v,p} - T_{v,e}) \cdot \frac{dp}{p} = \int R_d \cdot \Delta T_v \cdot d(\ln p)$$

The code approximates the integral as a sum over discrete layers:

```rust
let val = RD * (tv_parcel - tv_env) * (p1 / p2).ln();
if val > 0.0 {
    cape += val;
} else {
    cin += val;
}
```

### 10.2 Differences between the old SHARPpy formula and MetPy's formula

Both formulas compute $R_d \cdot \Delta T_v \cdot \ln(p_1/p_2)$. The key difference is
in **how the parcel temperature** above the LCL is computed:

| Aspect | SHARPpy (`satlift`/`wobf`) | MetPy (`moist_lapse` / RK4) |
|--------|----------------------------|------------------------------|
| Moist adiabat | Wobus polynomial + 7-iter Newton-Raphson | Bakhshaii-Stull dT/dp + RK4 integration |
| Accuracy | ~0.1-0.5 K typical error at upper levels | ~0.01 K (4th-order convergence) |
| Speed | Very fast (no exp/ln in inner loop) | Slower (transcendental functions per sub-step) |
| Heritage | SHARPpy / NSHARP / original NWS code | MetPy / modern formulation |

**Why the Wobus function was replaced with RK4 moist_lapse:**

The Wobus polynomial (`wobf`) is a compact approximation of the difference between the
dry and moist adiabats, developed in the 1970s for computational efficiency on hardware
where transcendental functions were expensive. It works well in the lower-to-mid
troposphere but introduces errors of 0.5+ K at pressures below ~300 hPa. The RK4
integration of the exact Bakhshaii-Stull lapse rate formula is more accurate and
produces CAPE values that match MetPy to within 1-2 J/kg for typical soundings.

The legacy `satlift`/`wobf` path is retained for the `cape_cin_core` and
`cape_cin_from_parcel` functions. The modern `parcel_profile` function uses the RK4 path.

### 10.3 `cape_cin_core` -- full implementation with unit detection

This is the primary entry point for grid-based CAPE/CIN computation. It includes:

#### Unit standardization
- If `psfc > 2000`, pressure is assumed Pa and divided by 100
- If `t2m > 150`, temperatures are assumed Kelvin and converted to Celsius
- Dewpoint is clamped: $T_d \leq T$

#### Parcel selection
Three parcel types are supported:
- **`"sb"`** (Surface-Based): Uses 2-meter T, Td at surface pressure
- **`"ml"`** (Mixed Layer): Averages theta, Td over the lowest `ml_depth` hPa using a 1-2-1 weighting scheme (surface and top boundary weight 1, inner levels weight 2)
- **`"mu"`** (Most Unstable): Finds the level of maximum $\theta_e$ in the lowest `mu_depth` hPa

#### LCL computation
Analytic LCL via `drylift()` (see section 8).

#### Theta-M computation (saturation potential temperature)
The key parameter for tracking the parcel along the moist adiabat:

$$\theta_K = (T_{LCL} + 273.15) \cdot (1000 / p_{LCL})^\kappa$$

$$\theta_C = \theta_K - 273.15$$

$$\theta_m = \theta_C - \text{wobf}(\theta_C) + \text{wobf}(T_{LCL})$$

This $\theta_m$ is passed to `satlift(p, thetam)` at every pressure level to get the
parcel temperature along the moist adiabat.

#### Pass 1: Geometric scan for LFC and EL

Before integration, the code scans all levels above the LCL to find where buoyancy
crosses zero:

- **LFC detection:** First level where buoyancy changes from negative to positive. The
  exact crossing pressure is found by linear interpolation:

$$\text{frac} = \frac{0 - B_{prev}}{B_{curr} - B_{prev}}, \qquad p_{LFC} = p_{prev} + \text{frac} \cdot (p_{curr} - p_{prev})$$

- **EL detection:** The last level where buoyancy changes from positive to negative,
  interpolated the same way.

- If LFC is below the LCL, it is set to the LCL pressure.

#### Pass 2: Integration (sub-stepped)

**Below LCL (dry adiabat, CIN only):**

The parcel follows the dry adiabat:

$$T_{parcel} = \theta_{start,K} \cdot (p_{mid} / 1000)^\kappa - 273.15$$

Virtual temperature uses the parcel's constant mixing ratio:

$$T_{v,parcel} = (T_{parcel} + 273.15) \cdot (1 + 0.61 \cdot r_{parcel}/1000) - 273.15$$

Only negative buoyancy (CIN) is accumulated in this region.

**Above LCL (moist adiabat, CAPE and CIN):**

The integration is **sub-stepped**: any layer thicker than 10 hPa is split into smaller
sub-layers:

$$n_{steps} = \left\lfloor \frac{\Delta p}{10} \right\rfloor + 1 \quad \text{if } \Delta p > 10, \text{ else } 1$$

At each sub-layer midpoint:
- Environment: interpolated from profile using log-pressure interpolation
- Parcel: `satlift(p_mid, thetam)` with virtual temperature correction (parcel Td = parcel T since saturated)

```rust
let t_parc = satlift(p_mid, thetam);
let tv_parc = virtual_temp(t_parc, p_mid, t_parc);  // Td = T (saturated)
let val = RD * (tv_parc - tv_env) * (p1 / p2).ln();
```

#### Virtual temperature correction

Both parcel and environment use virtual temperature:

- **Environment:** $T_{v,env} = \text{virtual\_temp}(T_{env}, p, T_{d,env})$ -- uses the environmental dewpoint
- **Parcel (below LCL):** Uses the parcel's original mixing ratio (constant during dry ascent)
- **Parcel (above LCL):** Assumes saturation: $T_{d,parcel} = T_{parcel}$, so $w = w_s(T_{parcel}, p)$

#### Height computation

Heights are provided as an input `height_agl` profile. LCL height and LFC height are
obtained by interpolating pressure to height via `get_height_at_pres()` (linear
interpolation in pressure space).

The hypsometric equation is used separately in `thickness_hypsometric` and
`brunt_vaisala_frequency`, but **not** in the CAPE integration itself (which uses
the $R_d \Delta T_v \ln(p_1/p_2)$ form directly).

### 10.4 `cape_cin_from_parcel` -- simplified version

A simpler CAPE/CIN computation that:
- Takes pre-selected parcel parameters `(p_start, t_start, td_start)`
- Does NOT sub-step layers (one evaluation per model layer pair)
- Uses midpoint averaging: $p_{mid} = (p_1 + p_2)/2$, $T_{mid} = (T_1 + T_2)/2$
- Integrates from surface to top of sounding (no explicit LFC/EL bounds)

```rust
fn cape_cin_from_parcel(
    p: &[f64], t: &[f64], td: &[f64],
    p_start: f64, t_start: f64, td_start: f64,
) -> (f64, f64) {
    // ... same formula: RD * (tv_parcel - tv_env) * ln(p1/p2)
}
```

### 10.5 Downdraft CAPE (DCAPE)

Integrates negative buoyancy from the level of minimum $\theta_e$ (in the lowest 400 hPa)
downward to the surface, following a **moist adiabat** (via RK4 `moist_lapse`):

$$\text{DCAPE} = \sum R_d \cdot |T_{v,parcel} - T_{v,env}| \cdot \Delta(\ln p) \quad \text{where } T_{v,parcel} < T_{v,env}$$

---

## 11. Isentropic Interpolation

### Algorithm

The isentropic interpolation function maps 3D model data from pressure levels to
constant potential temperature (isentropic) surfaces.

**Step 1 -- Compute theta at every 3D grid point:**

$$\theta_{k,j,i} = T_{k,j,i} \cdot \left(\frac{1000}{p_{k,j,i}}\right)^{R_d/c_p}$$

where $T$ is in Kelvin and $p$ is in hPa.

**Step 2 -- Column-by-column interpolation:**

For each grid column $(j, i)$ and each target $\theta$ level:

1. Search vertically for the bracketing levels where $\theta_k \leq \theta_{target} \leq \theta_{k+1}$ (or the reverse, to handle non-monotonic profiles).
2. Compute the interpolation fraction:

$$f = \frac{\theta_{target} - \theta_k}{\theta_{k+1} - \theta_k}$$

3. Interpolate all fields linearly:

$$X_{target} = X_k + f \cdot (X_{k+1} - X_k)$$

for $X \in \{p, T, \text{field}_0, \text{field}_1, \ldots\}$.

```rust
let dth = th_hi - th_lo;
if dth.abs() < 1e-10 { continue; }
let frac = (target_theta - th_lo) / dth;
output[0][out_idx] = col_p[k] + frac * (col_p[k + 1] - col_p[k]);
output[1][out_idx] = col_t[k] + frac * (col_t[k + 1] - col_t[k]);
```

### Handling non-monotonic profiles

The search condition checks both orderings:

```rust
if (th_lo <= target_theta && th_hi >= target_theta)
    || (th_lo >= target_theta && th_hi <= target_theta)
```

This handles inversions where $\theta$ temporarily decreases with height. The first
matching layer (from the bottom up) is used.

### Output format

Returns a `Vec<Vec<f64>>` where:
- `output[0]` = pressure on theta surfaces (hPa)
- `output[1]` = temperature on theta surfaces (K)
- `output[2+f]` = interpolated field $f$ on theta surfaces

All output arrays are flattened `[n_theta][ny][nx]`.

**Note:** This implementation uses simple linear interpolation in theta-space. It does
**not** use the Newton-solver / $T = a \ln(p) + b$ approach that some implementations use.
NaN values are preserved for columns where the target theta level is outside the data range.

---

## 12. Additional Functions

### Dry adiabat

$$T(p) = T_{sfc,K} \cdot \left(\frac{p}{p_{sfc}}\right)^{R_d/c_p} - 273.15$$

### Hypsometric thickness

$$\Delta z = \frac{R_d \cdot \overline{T}_K}{g} \cdot \ln\!\left(\frac{p_{bottom}}{p_{top}}\right)$$

### Standard atmosphere

**Pressure to height (troposphere):**

$$z = \frac{T_0}{\Gamma} \cdot \left[1 - \left(\frac{p}{p_0}\right)^{R_d \Gamma / g}\right]$$

**Height to pressure:**

$$p = p_0 \cdot \left(1 - \frac{\Gamma \cdot z}{T_0}\right)^{g / (R_d \Gamma)}$$

### Air density

$$\rho = \frac{p}{R_d \cdot T_v} = \frac{100 \cdot p_{hPa}}{R_d \cdot T_K \cdot (1 + 0.61 \, w)}$$

### Brunt-Vaisala frequency

$$N^2 = \frac{g}{\theta} \cdot \frac{d\theta}{dz}$$

Heights approximated via the hypsometric equation between levels. $N = \sqrt{N^2}$ where
$N^2 > 0$, else $N = 0$.

### Static energy

$$\text{DSE} = c_p T + g z$$

$$\text{MSE} = c_p T + g z + L_v q$$

---

## References

- **Bolton, D. (1980).** "The Computation of Equivalent Potential Temperature." *Monthly Weather Review*, 108(7), 1046-1053.
- **Bakhshaii, A. and Stull, R. (2013).** "Saturated Pseudoadiabats -- A Noniterative Approximation." *Journal of Applied Meteorology and Climatology*, 52(1), 5-15.
- **Rothfusz, L.P. (1990).** "The Heat Index Equation." NWS Southern Region Technical Attachment, SR 90-23.
- **Steadman, R.G. (1984).** "A Universal Scale of Apparent Temperature." *Journal of Climate and Applied Meteorology*, 23, 1674-1687.
- **Bunkers, M.J. et al. (2000).** "Predicting Supercell Motion Using a New Hodograph Technique." *Weather and Forecasting*, 15(1), 61-79.
- **SHARPpy:** Blumberg, W.G. et al. Sounding/Hodograph Analysis and Research Program in Python. https://github.com/sharppy/SHARPpy
- **MetPy:** May, R.M. et al. MetPy: A Meteorological Python Library. https://github.com/Unidata/MetPy
