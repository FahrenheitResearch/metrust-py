# Moisture and Humidity Calculations

This document describes the moisture and humidity functions implemented in `crates/wx-math/src/thermo.rs` and `crates/wx-math/src/composite.rs`. All functions are pure math with no external dependencies, ported from SHARPpy-derived implementations.

## Physical Constants

| Symbol | Name | Value | Units |
|--------|------|-------|-------|
| R_d | Dry air gas constant | 287.058 | J/(kg K) |
| R_v | Water vapor gas constant | 461.5 | J/(kg K) |
| c_p | Specific heat at constant pressure | 1005.7 | J/(kg K) |
| g | Gravitational acceleration | 9.80665 | m/s^2 |
| epsilon | Molecular weight ratio (R_d / R_v) | 0.62197 | dimensionless |
| L_v | Latent heat of vaporization | 2.501 x 10^6 | J/kg |
| T_0 | Zero Celsius in Kelvin | 273.15 | K |

---

## 1. Saturation Vapor Pressure

Two independent implementations exist, each used in different code paths.

### 1a. Eschner 8th-Order Polynomial (`vappres`)

Used by the SHARPpy-derived functions (`mixratio`, `virtual_temp`, `thetae`, `temp_at_mixrat`).

```
e_s(T) = 6.1078 / P(T)^8
```

where P(T) is the polynomial:

```
P(T) = 0.99999683
     + T * (-9.082695e-03
     + T * (7.8736169e-05
     + T * (-6.111796e-07
     + T * (4.3884180e-09
     + T * (-2.988388e-11
     + T * (2.1874425e-13
     + T * (-1.789232e-15
     + T * (1.1112018e-17
     + T * (-3.0994571e-20)))))))))
```

Input: T in Celsius. Output: e_s in hPa.

This is a high-precision polynomial fit to the Clausius-Clapeyron equation. It does **not** distinguish between liquid and ice phases -- all calculations assume saturation over liquid water regardless of temperature. The approximation is accurate across standard meteorological temperature ranges (roughly -40 C to +50 C).

### 1b. Bolton (1980) Formula (`saturation_vapor_pressure`)

Used by the Bolton-based functions (`dewpoint_from_rh`, `rh_from_dewpoint`, `saturation_mixing_ratio`, `equivalent_potential_temperature`, `frost_point`, `psychrometric_vapor_pressure`).

```
e_s(T) = 6.112 * exp(17.67 * T / (T + 243.5))
```

Input: T in Celsius. Output: e_s in hPa.

This is the Bolton (1980) empirical fit, which is itself a form of the August-Roche-Magnus equation. The coefficients 17.67 and 243.5 are Bolton's recommended values. Like the polynomial version, this assumes saturation over **liquid water only**. Valid range is approximately -35 C to +35 C with errors under 0.3%.

### Ice Phase Handling

Neither implementation provides a separate ice saturation curve. All saturation calculations use the liquid-water formula. This is standard practice in operational meteorology (SHARPpy, GEMPAK, etc.) because mixed-phase processes and supercooled water make the ice/liquid distinction ambiguous in most sounding analysis applications.

---

## 2. Vapor Pressure

### From Dewpoint

The vapor pressure of moist air equals the saturation vapor pressure evaluated at the dewpoint temperature:

```
e = e_s(T_d)
```

Implemented as `vapor_pressure_from_dewpoint(td_c)`, which simply calls `saturation_vapor_pressure(td_c)`.

### Inverse: Dewpoint from Vapor Pressure

The Bolton formula is algebraically inverted to recover dewpoint from a known vapor pressure:

```
T_d = 243.5 * ln(e / 6.112) / (17.67 - ln(e / 6.112))
```

Implemented as `dewpoint(vapor_pressure_hpa)`. A guard returns approximately absolute zero (-273.15 C) when `e <= 0`.

---

## 3. Mixing Ratio

### SHARPpy Version (`mixratio`)

```
w = 621.97 * (f_w * e_s(T)) / (p - f_w * e_s(T))
```

where the **Wexler enhancement factor** accounts for non-ideal gas behavior:

```
x = 0.02 * (T - 12.5 + 7500 / p)
f_w = 1.0 + 4.5e-6 * p + 1.4e-3 * x^2
```

- Inputs: p in hPa, T in Celsius (the temperature at which saturation is evaluated, typically the dewpoint).
- Output: w in **g/kg**.
- The factor 621.97 = 1000 * epsilon (converting the kg/kg ratio to g/kg).

The Wexler enhancement factor is a correction for the fact that water vapor in air does not behave as an ideal gas. At typical surface pressures and temperatures, f_w is close to 1.004.

### Bolton Version (`saturation_mixing_ratio`)

```
w_s = max(0, 1000 * epsilon * e_s(T) / (p - e_s(T)))
```

Uses the Bolton saturation vapor pressure. Output clamped to non-negative. Units: **g/kg**.

---

## 4. Saturation Mixing Ratio

```
w_s(T, p) = 1000 * epsilon * e_s(T) / (p - e_s(T))
```

Implemented as `saturation_mixing_ratio(p_hpa, t_c)` using the Bolton `saturation_vapor_pressure`. Result is in **g/kg**, clamped to >= 0.

This is the mixing ratio of air that is fully saturated at temperature T and pressure p.

---

## 5. Relative Humidity

### From Dewpoint and Temperature

```
RH = e_s(T_d) / e_s(T) * 100
```

Implemented as `rh_from_dewpoint(t_c, td_c)`. Both saturation vapor pressures use the Bolton formula. Output is in **percent** (0-100).

### From Mixing Ratio

```
RH = (w / w_s) * 100
```

Implemented as `relative_humidity_from_mixing_ratio(p_hpa, t_c, w_gkg)`. Both w and w_s are in g/kg.

### From Specific Humidity

Converts q to w first, then uses the mixing ratio form:

```
w = q / (1 - q)     [kg/kg -> kg/kg]
RH = (w * 1000 / w_s) * 100
```

Implemented as `relative_humidity_from_specific_humidity(p_hpa, t_c, q)`.

### Convention

All RH functions use the **percent** convention (0-100), not the fraction convention (0-1).

---

## 6. Dewpoint from Relative Humidity

Uses the Bolton formula inverted:

```
e = (RH / 100) * e_s(T)
T_d = 243.5 * ln(e / 6.112) / (17.67 - ln(e / 6.112))
```

Implemented as `dewpoint_from_rh(t_c, rh)` where RH is in percent (0-100).

This is algebraically equivalent to the Magnus formula inversion. The coefficients (17.67, 243.5, 6.112) are from Bolton (1980).

---

## 7. Specific Humidity

### From Mixing Ratio

```
q = w / (1 + w)
```

where w is in kg/kg. Implemented as `specific_humidity(p_hpa, w_gkg)` -- note the input is in g/kg but is converted internally:

```rust
let w = w_gkg / 1000.0;  // g/kg -> kg/kg
w / (1.0 + w)             // result in kg/kg
```

Output: **kg/kg**. The pressure parameter is accepted but unused (it is not needed for this conversion).

### Inverse: Mixing Ratio from Specific Humidity

```
w = q / (1 - q)
```

Implemented as `mixing_ratio_from_specific_humidity(q)`. Input q in kg/kg, output w in **g/kg** (multiplied by 1000).

### From Dewpoint

```
e = e_s(T_d)
w = epsilon * e / (p - e)     [kg/kg]
q = w / (1 + w)
```

Implemented as `specific_humidity_from_dewpoint(p_hpa, td_c)`. Output in kg/kg.

### Inverse: Dewpoint from Specific Humidity

```
w = q / (1 - q)               [kg/kg]
e = w * p / (epsilon + w)      [hPa]
T_d = dewpoint(e)
```

Implemented as `dewpoint_from_specific_humidity(p_hpa, q)`.

---

## 8. Virtual Temperature

Virtual temperature accounts for the buoyancy effect of water vapor, which is lighter than dry air (M_w = 18.015 vs M_d = 28.964).

### Formula Used

```
T_v = T_K * (1 + 0.61 * w) - 273.15
```

where T_K is temperature in Kelvin and w is mixing ratio in **kg/kg**. Implemented as `virtual_temp(t, p, td)`:

```rust
let w = mixratio(p, td) / 1000.0;   // g/kg -> kg/kg
let tk = t + 273.15;
let vt = tk * (1.0 + 0.61 * w);
vt - 273.15
```

The coefficient 0.61 is the approximation of `(1/epsilon - 1)` = `(R_v/R_d - 1)` = `(461.5/287.058 - 1)` = 0.608, rounded to 0.61.

### Why This Matters

Virtual temperature is used in two critical calculations:

1. **CAPE integration**: The buoyancy term `(T_v,parcel - T_v,env)` drives the integral. Using T instead of T_v underestimates CAPE, especially in the moist lower troposphere where moisture content is high.

2. **Hypsometric equation**: Layer thickness depends on mean virtual temperature. The function `thickness_hypsometric` computes `dz = (R_d * T_v) / g * ln(p_bottom / p_top)`. Using T_v ensures that moist layers are correctly computed as thicker than dry layers at the same temperature.

---

## 9. Equivalent Potential Temperature

Two implementations exist.

### 9a. SHARPpy Wobus Approximation (`thetae`)

```
theta_e = theta * exp(L_c * 1000 * r / (c_p * T_LCL_K)) - 273.15
```

where:
- theta = potential temperature at the LCL: `(T_LCL + 273.15) * (1000 / p_LCL)^(R_d/c_p)`
- r = mixing ratio in kg/kg (from `mixratio(p, td) / 1000`)
- L_c = `2500 - 2.37 * T_LCL` (temperature-dependent latent heat, kJ/kg)
- T_LCL computed via `drylift`

Output is in **Celsius** (nonstandard; most references use Kelvin).

### 9b. Bolton (1980) Formula (`equivalent_potential_temperature`)

The full Bolton (1980) equation 39, matching MetPy's implementation:

```
T_LCL = 56 + 1 / (1/(T_d_K - 56) + ln(T_K / T_d_K) / 800)

e = e_s(T_d)
r = epsilon * e / (p - e)           [kg/kg]

theta_DL = T_K * (1000 / (p - e))^kappa * (T_K / T_LCL)^(0.28 * r)

theta_e = theta_DL * exp((3036 / T_LCL - 1.78) * r * (1 + 0.448 * r))
```

where kappa = R_d / c_p = 0.28571426.

Output is in **Kelvin**.

The Bolton LCL temperature formula (equation 15) provides T_LCL directly without iteration. The exponential term captures the latent heat release. The factor `(T_K / T_LCL)^(0.28r)` is Bolton's correction for the effect of moisture on the dry adiabatic ascent.

---

## 10. Precipitable Water

Implemented in `composite.rs` as `compute_pw`.

### Formula

```
PW = (1/g) * integral from surface to top { q * dp }
```

Discretized using the trapezoidal rule:

```
PW = (1/g) * sum over k { 0.5 * (q_k + q_{k+1}) * |p_k - p_{k+1}| }
```

- q: water vapor mixing ratio in **kg/kg** (clamped to >= 0 at each level)
- p: full pressure in **Pa**
- g = 9.80665 m/s^2
- Output: **kg/m^2**, which equals **mm** of liquid water

The profile is sorted so that pressure decreases upward (surface first) before integration. The computation is parallelized over horizontal grid points using Rayon.

---

## 11. Wet Bulb Temperature

### Method: Normand's Rule (Analytic, Not Iterative)

Implemented as `wet_bulb_temperature(p_hpa, t_c, td_c)`. Despite the docstring saying "iterative," the actual implementation uses Normand's construction, which is a three-step analytic process:

1. **Lift to LCL**: Compute the LCL pressure and temperature by dry-adiabatic ascent from (p, T, T_d) using `drylift`.

2. **Compute thetam**: Determine the moist-adiabatic equivalent potential temperature parameter at the LCL:
   ```
   theta_K = (T_LCL + 273.15) * (1000 / p_LCL)^kappa
   theta_C = theta_K - 273.15
   thetam = theta_C - wobf(theta_C) + wobf(T_LCL)
   ```

3. **Descend moist adiabat**: Follow the saturated adiabat from the LCL back down to the original pressure using `satlift(p, thetam)`.

The `satlift` function uses **Newton-Raphson iteration** with up to **7 iterations** and a convergence threshold of 0.001 C. The Wobus polynomial (`wobf`) provides the thermodynamic relationship between temperature and the moist adiabat.

The wet bulb temperature is the temperature at which the moist adiabat from the LCL intersects the original pressure level. By definition, T_d <= T_w <= T.

---

## 12. Frost Point

The frost point is the temperature at which air becomes saturated with respect to **ice** rather than liquid water.

### Formula

```
e = (RH / 100) * e_s,water(T)
```

The saturation vapor pressure over water is computed using the Bolton formula. Then the **Magnus formula over ice** is inverted:

```
e_s,ice(T) = 6.112 * exp(22.46 * T / (T + 272.62))
```

Solving for T:

```
T_frost = 272.62 * ln(e / 6.112) / (22.46 - ln(e / 6.112))
```

Implemented as `frost_point(t_c, rh)`.

### Difference from Dewpoint Below 0 C

Below 0 C, the saturation vapor pressure over ice is **lower** than over supercooled liquid water. This means:
- The frost point is always **warmer** than the dewpoint when both are below 0 C.
- The difference grows as temperature decreases (roughly 1-2 C at -20 C, up to 4-5 C at -40 C).
- For cloud and precipitation processes involving ice nucleation, the frost point is the physically relevant quantity.

The ice Magnus coefficients (22.46, 272.62) differ from the liquid-water Bolton coefficients (17.67, 243.5), reflecting the higher enthalpy of sublimation compared to evaporation.

---

## 13. Psychrometric Calculations

The psychrometric equation relates vapor pressure to the dry-bulb and wet-bulb temperatures measured by an aspirated psychrometer.

### Formula

```
e = e_s(T_w) - A * p * (T - T_w)
```

where:
- e_s(T_w): saturation vapor pressure at the wet-bulb temperature (Bolton formula)
- A = 6.6 x 10^-4 C^-1: psychrometric constant for an aspirated (Assmann-type) psychrometer
- p: station pressure in hPa
- T: dry-bulb temperature in Celsius
- T_w: wet-bulb temperature in Celsius

Implemented as `psychrometric_vapor_pressure(t_c, tw_c, p_hpa)`. Output: vapor pressure in hPa.

The psychrometric constant A depends on the ventilation rate. The value 6.6 x 10^-4 applies to aspirated instruments with forced ventilation. For natural ventilation (sling psychrometer), a larger value (~8 x 10^-4) is typical but is **not** implemented.

---

## Relationship Diagram

The following shows how the moisture variables connect to each other. Arrows indicate "computed from."

```
                        Temperature (T)
                             |
                    +--------+--------+
                    |                 |
                    v                 v
        Saturation Vapor         Potential
        Pressure: e_s(T)      Temperature: theta
                    |                 |
          +---------+---------+       |
          |         |         |       v
          v         v         v    Equiv. Potential
     Sat. Mixing  Frost    Bolton    Temp: theta_e
     Ratio: w_s   Point    T_LCL       ^
          |                   |         |
          |    Dewpoint (T_d) |    Mixing Ratio (w)
          |         |        |         ^
          |    +----+----+   |         |
          |    |         |   |    +----+----+
          |    v         v   |    |         |
          |  Vapor    Mixing |  Specific  Relative
          | Pressure  Ratio  |  Humidity  Humidity
          |  e=e_s(Td) (w)   |    (q)      (RH)
          |    |         |   |    ^ |       ^ |
          |    |    +----+---+    | |       | |
          |    |    |    |        | v       | v
          |    v    v    v        +---------+-+
          |   Psychrometric      q = w/(1+w)
          |   Vapor Pressure     w = q/(1-q)
          |   e = e_s(Tw) -      RH = w/w_s
          |     A*p*(T-Tw)       w = RH*w_s
          |
          +-------> Precipitable Water
          |         PW = (1/g) * integral(q dp)
          |
          +-------> Virtual Temperature
          |         T_v = T(1 + 0.61w)
          |              |
          |              +---> CAPE, Hypsometric Eq.
          |
          +-------> Wet Bulb Temperature
                    T_w via Normand's Rule:
                    Dry lift -> LCL -> Moist descent
```

---

## Numerical Precision Notes

1. **Polynomial vs. exponential saturation vapor pressure**: The Eschner polynomial (`vappres`) and Bolton exponential (`saturation_vapor_pressure`) agree to within ~0.1% across the range -40 C to +40 C. They are not interchangeable at the bit level, and different code paths use different versions for consistency with their upstream implementations (SHARPpy vs. MetPy/Bolton).

2. **Newton-Raphson convergence**: The `satlift` function uses a convergence threshold of 0.001 C with a maximum of 7 iterations. In practice, convergence is achieved in 3-5 iterations for typical atmospheric conditions.

3. **Wexler enhancement factor**: The correction in `mixratio` is typically 1.003-1.005 and matters at the sub-percent level. It is included for fidelity with the SHARPpy implementation.

4. **Floating-point guards**: The `dewpoint` function guards against `e <= 0` (returns -273.15 C). The `saturation_mixing_ratio` function clamps output to >= 0 to prevent negative values when `e_s > p` (physically impossible but numerically possible at extreme extrapolations).

5. **Unit discipline**: Mixing ratio functions return g/kg in most cases, but specific humidity is always kg/kg. Internal calculations convert between the two. Virtual temperature calculations always convert mixing ratio to kg/kg before applying the `1 + 0.61w` factor. Careless mixing of g/kg and kg/kg is a common source of factor-of-1000 errors.

6. **Precipitable water integration**: Uses trapezoidal rule with model-level spacing. Accuracy depends on vertical resolution; typical NWP grids (40-80 levels) provide adequate resolution. Negative mixing ratios (numerical artifacts) are clamped to zero before integration.

7. **Temperature range of validity**: The Bolton formula is most accurate for -35 C to +35 C. The Eschner polynomial is fitted across a wider range. Neither should be trusted below about -80 C (stratospheric conditions at very high altitudes).
