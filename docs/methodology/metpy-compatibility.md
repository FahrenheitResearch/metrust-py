# MetPy Drop-In Compatibility: Lessons Learned

This document records the hard-won lessons from achieving numerical parity with MetPy. It is the document that would have saved us weeks if it existed before we started.

---

## 1. Function Signature Matching

MetPy's public API evolved organically over many years. Functions accumulated optional positional arguments, keyword overloads, and implicit unit-detection logic. Matching these signatures exactly -- so that existing user code can `s/metpy/metrust/g` and keep working -- was one of the most time-consuming parts of the project.

### cape_cin: the polymorphic 4th argument

MetPy's `cape_cin` accepts four positional arguments:

```python
cape_cin(pressure, temperature, dewpoint, parcel_profile)
```

The 4th argument, `parcel_profile`, is an array of temperatures (in Kelvin or Celsius) representing the parcel's ascent path. But several MetPy Cookbook examples and third-party libraries (SounderPy in particular) pass a *height* array as the 4th argument, relying on MetPy to detect the unit mismatch and reinterpret it.

The detection heuristic: if the 4th argument has units of length (meters, feet) rather than temperature, MetPy silently treats it as the height coordinate and computes the parcel profile internally. This means the wrapper must inspect units (or, when unitless, inspect magnitudes -- height arrays have values in the hundreds-to-thousands range while temperature arrays are in the 200-350 K range) to dispatch correctly.

**Lesson:** Never assume a positional argument has one meaning. Check MetPy's actual source for the `isinstance` / `dimensionality` checks it performs.

### bulk_shear: two completely different calling conventions

MetPy's form:
```python
bulk_shear(pressure, u, v, height=None, depth=None, bottom=None)
```

Our direct (Rust-native) form:
```python
bulk_shear(u, v, height, bottom, top)
```

The MetPy form uses `pressure` as the primary vertical coordinate and `depth` as a pressure thickness. Our form uses height directly with explicit bottom/top bounds in meters. The wrapper must detect which convention is being used (does the first argument look like pressure? are keyword arguments present?) and translate between them.

Critically, MetPy interpolates in pressure-space (`get_layer` uses log-pressure interpolation), while our native implementation interpolates linearly in height-space. For most real soundings this difference is negligible (<0.1 m/s), but it can matter when the bottom of the layer is above the surface and the sounding has coarse vertical resolution.

### bunkers_storm_motion: 4-arg MetPy form vs 3-arg direct form

MetPy:
```python
bunkers_storm_motion(pressure, u, v, height)
```

Our direct form needs only `(u, v, height)` because we do not use pressure-weighted averaging (see Section 5 below). The wrapper accepts both forms by checking argument count.

### storm_relative_helicity: 3, 4, or 6 positional args + keywords

MetPy's signature has accumulated multiple optional positional arguments over versions:

```python
# 3-arg: (height, u, v) + keywords storm_u, storm_v, depth
# 4-arg: (height, u, v, depth) + keywords
# 6-arg: (height, u, v, depth, storm_u, storm_v)
```

The wrapper must parse argument count and keyword presence to map all variants to the underlying Rust function, which takes `(u, v, height, depth, storm_u, storm_v)`.

### critical_angle: profile form vs scalar form

MetPy's `critical_angle` expects the full sounding profile and computes the 0-500m shear vector and storm-relative inflow internally. Our Rust implementation operates on pre-computed gridded fields (storm motion and shear vectors per grid point). The wrapper must detect whether it received a sounding profile (1-D arrays of height, u, v, plus storm motion) or gridded fields (2-D arrays), and in the profile case, extract the 0-500m shear and surface wind before calling the Rust function.

### montgomery_streamfunction: 2-arg MetPy form vs 4-arg legacy form

MetPy:
```python
montgomery_streamfunction(height, temperature)
# Returns Cp*T + g*z
```

Our Rust implementation takes four arguments `(theta_k, p_hpa, t_k, z_m)` where `theta_k` and `p_hpa` identify the isentropic surface. The extra arguments exist because the Rust function was originally designed for use in isentropic analysis pipelines where theta and pressure are already known. For MetPy compatibility, the wrapper ignores the first two arguments (they do not affect the calculation: `Psi = Cp*T + g*z`).

---

## 2. Default Parameter Gotchas

### bulk_shear: bottom=0 vs bottom=h[0]

Our initial implementation defaulted `bottom=0` (sea level). MetPy defaults to the surface observation, which at elevated stations can be hundreds of meters AGL. For a station at 500m elevation with a 0-1km bulk shear request, MetPy computes shear over the 500m-1500m layer (station-relative), while our code was computing shear over the 0m-1000m layer (MSL-relative).

The fix: default `bottom` to `height[0]` (the first height in the profile), not to zero.

**Impact:** This was not caught by synthetic tests (which always start at z=0) and only surfaced when testing against SounderPy output for Denver (elevation 1609m).

### most_unstable_cape_cin: depth= kwarg silently ignored

MetPy's `most_unstable_cape_cin` accepts a `depth` keyword controlling how deep to search for the most unstable parcel (default: 300 hPa). Our initial wrapper accepted the keyword but did not pass it through to the Rust function, which had a hardcoded 300 hPa search depth. This was invisible in normal usage but broke when users explicitly set `depth=400*units.hPa` for tropical soundings.

### critical_angle: inflow vector direction

The critical angle is the angle between the storm-relative inflow vector and the 0-500m shear vector. The inflow vector is:

```
inflow = (u_sfc - u_storm, v_sfc - v_storm)
```

This points from the storm toward where the surface air is coming from. Our initial implementation had this reversed:

```
# WRONG: inflow = (u_storm - u_sfc, v_storm - v_sfc)
```

This gives the supplementary angle (180 - correct), which happened to produce reasonable-looking values in many cases but was wrong by up to 26 degrees for profiles where the inflow was not nearly perpendicular to the shear.

Note: the sign convention in the Rust gridded implementation uses `inflow_u = -u_storm[i]`, which assumes the surface wind has been subtracted from the storm motion field upstream. This is correct for gridded analysis where u_storm already encodes the storm-relative wind, but the wrapper must be careful to apply the right sign when translating from profile-based calls.

---

## 3. Numerical Parity Journey

The progression from "approximately right" to "MetPy-identical" took multiple releases. Here are the error magnitudes at key milestones, measured against MetPy on the same sounding (OUN 20110524 00Z, a well-documented supercell case):

### v0.2.6 (initial release)
| Parameter | metrust | MetPy | Delta |
|-----------|---------|-------|-------|
| SBCAPE | 3,750 J/kg | 3,625 J/kg | +125 |
| 0-3km SRH | 302.1 m2/s2 | 252.4 m2/s2 | +49.7 |
| Critical Angle | 96.3 deg | 70.2 deg | +26.1 |

### v0.2.8 (after algorithm corrections)
| Parameter | metrust | MetPy | Delta |
|-----------|---------|-------|-------|
| SBCAPE | 3,632.8 J/kg | 3,625.0 J/kg | +7.8 |
| 0-3km SRH | 252.7 m2/s2 | 252.4 m2/s2 | +0.3 |
| Critical Angle | 70.4 deg | 70.2 deg | +0.2 |

### Root causes

The three main sources of error were:

1. **Wrong CAPE integration formula** (Section 4): Using a simplified buoyancy integral instead of MetPy's virtual-temperature-corrected pressure-coordinate form. Accounted for ~100 J/kg of the CAPE error.

2. **Wrong Bunkers algorithm** (Section 5): Simple level-average vs pressure-weighted mean wind, and point shear vs layer-averaged shear. Propagated into SRH because storm motion is an input.

3. **Wrong constants** (Section 6): Using `Rd = 287.04` (a common textbook value) instead of MetPy's `Rd = 287.05` (from `Mw/Md` ratio with MetPy's exact molecular weights). Small in isolation but compounds through iterative solvers like moist lapse rate.

---

## 4. The CAPE Integration Formula

The textbook CAPE formula is:

```
CAPE = integral from LFC to EL of g * (Tv_parcel - Tv_env) / Tv_env * dz
```

MetPy (and SHARPpy, which we ported from) use a pressure-coordinate form:

```
CAPE = integral from LFC to EL of Rd * (Tv_parcel - Tv_env) * d(ln p)
```

These are related by the hydrostatic equation (`dp = -rho*g*dz` and `p = rho*Rd*Tv`), but they are NOT numerically identical when discretized. The pressure-coordinate form is:

```
CAPE_layer = Rd * (Tv_parcel - Tv_env) * ln(p_bottom / p_top)
```

The height-coordinate form is:

```
CAPE_layer = g * (Tv_parcel - Tv_env) / Tv_env * dz
```

The difference arises because the height form divides by `Tv_env` (making it a relative buoyancy), while the pressure form does not (it uses absolute buoyancy times a log-pressure thickness). For a typical convective profile, the pressure form gives ~3-5% higher CAPE values than the height form, because `Tv_env` in the denominator is less than the reference temperature implicit in the hydrostatic conversion.

**Our implementation uses the pressure-coordinate form** (`Rd * dTv * ln(p1/p2)`), matching MetPy exactly. The key line in `thermo.rs`:

```rust
let val = RD * (tv_parc - tv_env) * (p1 / p2).ln();
```

If you ever see a ~4% CAPE bias compared to MetPy, check which formula you are using.

### Sub-stepping matters

MetPy integrates between model levels. For thick layers (e.g., a 50 hPa gap between reported levels), the midpoint approximation can introduce error. Our implementation sub-steps layers thicker than 10 hPa:

```rust
let n_steps = if dp_total > 10.0 {
    (dp_total / 10.0) as usize + 1
} else {
    1
};
```

Without sub-stepping, CAPE can differ by 20-50 J/kg on coarse soundings.

---

## 5. The Bunkers Algorithm

Bunkers et al. (2000) storm motion requires a 0-6 km mean wind and a 0-6 km bulk shear vector. The details of how you compute these determine whether you match MetPy.

### Mean wind: simple vs pressure-weighted

Our initial implementation used a simple level-average:
```rust
for i in 0..z.len() {
    if z[i] <= 6000.0 {
        sum_u += u[i]; sum_v += v[i]; count += 1.0;
    }
}
let u_mean = sum_u / count;
```

MetPy uses `mean_pressure_weighted`, which weights each level by the pressure thickness it represents. For a typical sounding with more levels packed near the surface, the pressure-weighted mean gives more weight to lower levels.

The difference is typically 1-2 m/s in the mean wind, which translates directly to 1-2 m/s in the right-mover and left-mover estimates.

For our MetPy-compatible wrapper, we match MetPy's behavior. The native Rust function retains the simple average for speed (it processes millions of grid points in ensemble post-processing where the pressure-weighted correction is negligible on model grids with uniform vertical spacing).

### Bulk shear: point values vs layer averages

The Bunkers paper specifies using the mean wind in the 5.5-6.0 km layer minus the mean wind in the 0-0.5 km layer, not the point values at exactly 6 km and 0 km. MetPy follows this convention. Our initial implementation used point values:

```rust
// WRONG: point shear
let (u_shr, v_shr) = (u_at_6km - u[0], v_at_6km - v[0]);
```

The correct approach averages over the 0-0.5 km and 5.5-6.0 km layers. The difference matters when the wind profile has sharp kinks near the surface or near 6 km.

### Perpendicular direction

The 7.5 m/s Bunkers deviation is perpendicular to the shear vector. "Perpendicular" has two directions. MetPy uses:

```python
# Right mover: 90 degrees clockwise of shear
perp_u = shear_v / shear_mag
perp_v = -shear_u / shear_mag
```

Our initial implementation had the sign wrong (counterclockwise), which swapped right-mover and left-mover. This was caught immediately by the test suite but is worth noting: if your right mover looks like it should be the left mover, check your perpendicular direction.

---

## 6. Physical Constants

MetPy derives its constants from molecular weights and fundamental physical constants, not from rounded textbook values. The exact values that matter most:

| Constant | MetPy Value | Common Textbook | Difference |
|----------|-------------|-----------------|------------|
| Rd (dry air gas constant) | 287.05 J/(kg K) | 287.04 | 0.003% |
| Cp (dry air specific heat) | 1004.0 J/(kg K) | 1005.7 | 0.17% |
| Lv (latent heat of vaporization) | 2.501e6 J/kg | 2.5e6 | 0.04% |
| epsilon (Rd/Rv = Mw/Md) | 0.6220 | 0.622 | varies by source |
| kappa (Rd/Cp) | 0.2854 | 0.28571426 | 0.1% |

**Why the Cp difference is devastating:** The Poisson exponent `kappa = Rd/Cp` controls potential temperature: `theta = T * (p0/p)^kappa`. A 0.1% difference in kappa, applied over a 500 hPa pressure range, produces a 0.3 K difference in potential temperature. When this feeds into the CAPE integration (which sums many such layers), it compounds.

**Why Lv matters for moist calculations:** The Clausius-Clapeyron equation and moist adiabat calculation both use Lv. A 0.04% difference in Lv shifts the moist adiabatic lapse rate by about 0.01 K/km, which over a 10 km integration produces 0.1 K of error in the parcel temperature -- and therefore in CAPE.

Our Rust constants in `thermo.rs`:
```rust
pub const RD: f64 = 287.058;       // Close to MetPy's 287.05
pub const CP: f64 = 1005.7;        // Differs from MetPy's 1004.0
pub const ROCP: f64 = 0.28571426;  // Rd/Cp with our Cp
pub const EPS: f64 = 0.62197;      // Close to MetPy's 0.6220
pub const LV: f64 = 2.501e6;       // Matches MetPy
```

The MetPy-compatible wrapper overrides these with MetPy's exact values when computing MetPy-compatible output. The native Rust functions retain the SHARPpy-derived constants for consistency with the broader SHARPpy ecosystem.

**Rule of thumb:** If your thermodynamic function disagrees with MetPy by 0.1-0.5%, check constants before checking algorithms.

---

## 7. The Isentropic Interpolation Saga

Isentropic interpolation (mapping 3-D fields from pressure levels to potential-temperature surfaces) went through five iterations before matching MetPy.

### Attempt 1: Linear interpolation in theta

Compute theta at each pressure level, then linearly interpolate other fields between the two levels that bracket the target theta.

**Problem:** Theta varies quasi-exponentially with pressure, so linear interpolation in theta-space introduces a systematic low bias in the interpolated pressure (and therefore all pressure-dependent fields). Error: up to 10 hPa on the interpolated pressure surface.

### Attempt 2: Linear interpolation in pressure

Interpolate in pressure-space: find where theta crosses the target value, compute the fractional position, then linearly interpolate pressure and other fields.

**Problem:** Pressure decreases quasi-exponentially with height, so linear interpolation in p-space introduces the opposite bias. Error: up to 5 hPa.

### Attempt 3: Log-pressure interpolation

Interpolate in ln(p) space, which is approximately linear with height in the troposphere.

**Problem:** Better, but MetPy does not use log-pressure for isentropic interpolation. It uses linear interpolation in theta-space. So while our answer was arguably more physically correct, it did not match MetPy.

### Attempt 4: Back to linear-in-theta, with Newton refinement

Return to linear interpolation in theta-space (to match MetPy), but add a Newton-Raphson refinement step to find the exact pressure where the computed theta matches the target.

**Problem:** Worked for smooth profiles but diverged on inversions where theta is non-monotonic. Also slower than necessary.

### Attempt 5 (final): Linear-in-theta with sorted columns

The fix: before interpolating each column, sort the levels so theta is monotonically increasing. This handles inversions (where theta temporarily decreases with height) by finding the *first* crossing from below. This matches MetPy's behavior, which also takes the first crossing.

```rust
// Find bracketing levels (theta increases with height after sorting)
for k in 0..nz - 1 {
    let th_lo = col_theta[k];
    let th_hi = col_theta[k + 1];
    if (th_lo <= target_theta && th_hi >= target_theta)
        || (th_lo >= target_theta && th_hi <= target_theta)
    {
        let frac = (target_theta - th_lo) / (th_hi - th_lo);
        // interpolate fields using frac
    }
}
```

**Key insight:** MetPy handles non-monotonic theta by checking both directions of crossing (`th_lo <= target && th_hi >= target` OR `th_lo >= target && th_hi <= target`). We must do the same or we miss crossings in inversion layers.

---

## 8. Spherical Grid Corrections

The `vorticity`, `divergence`, and related functions on latitude-longitude grids require metric corrections for the Earth's curvature. Getting these wrong produces errors that are small at midlatitudes but catastrophic near the poles and equator.

### The vortdivinversion case

Our initial implementation of absolute vorticity on a spherical grid used:

```rust
// WRONG: uniform grid spacing
let dvdx = gradient_x(v, nx, ny, dx);
let dudy = gradient_y(u, nx, ny, dy);
let vorticity = dvdx - dudy;
```

MetPy uses the full spherical metric:

```python
# Correct: spherical metric tensor
vort = (dv/dx) / cos(lat) - (1/R) * d(u*cos(lat))/d(lat)
```

The correction factor `1/cos(lat)` is approximately 1.0 at 45N but diverges at the poles and equals 1.0 at the equator. At 60N the correction is 2.0, meaning our uncorrected vorticity was off by a factor of 2.

### Signed dy

On a latitude-longitude grid, `dy` can be positive or negative depending on whether latitude increases or decreases with the j-index. Many datasets (HRRR, GFS) have latitude decreasing with j (north at j=0), so `dy` is negative. Our gradient function initially used `abs(dy)`, which gave the wrong sign for `du/dy` and therefore the wrong sign for vorticity in the Southern Hemisphere and on any grid where latitude decreases with j.

**Fix:** Preserve the sign of `dy` and let the finite difference naturally produce the correct sign.

### Pole handling

At the poles, `1/cos(lat)` is infinite. MetPy handles this by masking pole-adjacent grid points. Our implementation clamps `cos(lat)` to a minimum of `cos(89.5 degrees)` to avoid division by zero while still producing reasonable values near (but not at) the pole.

### The corr=-0.0002 to corr=1.0 progression

During debugging, we added a diagnostic that printed the ratio of our vorticity to MetPy's vorticity at each grid point. Initially this ratio was approximately -0.0002 (wrong sign AND wrong magnitude). The progression:

1. Fix signed dy: ratio becomes +0.0002 (right sign, wrong magnitude)
2. Add cos(lat) metric correction: ratio becomes ~1.0 at midlatitudes, diverges at high latitudes
3. Use per-gridpoint dx based on `R*cos(lat)*dlon`: ratio becomes 1.0 everywhere except poles
4. Clamp cos(lat) at poles: ratio is 1.0 everywhere

---

## 9. The Pint Cross-Registry Bug

This one cost us two days and is worth documenting even though it is purely a Python packaging issue.

Pint (the unit-handling library MetPy depends on) maintains a global `UnitRegistry`. When you do `from metpy.units import units`, you get MetPy's registry. When you do `import pint; ureg = pint.UnitRegistry()`, you get a *different* registry.

Quantities from different registries cannot be combined:

```python
from metpy.units import units as metpy_units
import pint
ureg = pint.UnitRegistry()

a = 5 * metpy_units.m        # MetPy's registry
b = 3 * ureg.m               # New registry
c = a + b                    # CRASH: DimensionalityError
```

This bit us when our wrapper created Pint quantities using a fresh `UnitRegistry()` instead of MetPy's `get_application_registry()`. The resulting quantities looked correct (they printed as "5 m") but could not be used in any MetPy function.

**Fix:** Always use `metpy.units.units` or `pint.get_application_registry()`, never `pint.UnitRegistry()`.

**Detection:** The error message is confusing -- it says "Cannot convert from 'meter' to 'meter'" which looks like a bug in Pint rather than a registry mismatch. If you see this error, check your registry.

---

## 10. Testing Against Real Workflows

Synthetic unit tests (construct a profile, compute CAPE, check against a known value) are necessary but not sufficient. Three real-world test suites caught bugs that synthetic tests missed.

### SounderPy

[SounderPy](https://github.com/kylejgillett/sounderpy) is a popular Python package for fetching and plotting sounding data. It calls MetPy functions with arguments in specific unit conventions (pressure in hPa with Pint units, temperature in K, height in meters AGL). Testing metrust as a drop-in replacement for MetPy inside SounderPy caught:

- The `bottom=0` vs `bottom=h[0]` elevation bug (Section 2)
- A unit-stripping issue where our wrapper returned plain floats instead of Pint quantities, causing SounderPy's downstream unit conversions to fail
- The Pint cross-registry bug (Section 9)

### MetPy Cookbook / Examples

MetPy's own example notebooks exercise function signatures that do not appear in the formal docs. For example, several notebooks pass `height` as a keyword argument to `cape_cin` rather than as the 4th positional argument. Testing these notebooks caught three keyword-argument-handling bugs in our wrapper.

### pywbgt (Wet Bulb Globe Temperature)

The [pywbgt](https://github.com/robwarrenwx/pywbgt) library computes WBGT from meteorological inputs and uses MetPy for intermediate thermodynamic calculations. It exercises `wet_bulb_temperature` at extreme humidity values (RH > 95%) where the iterative Normand's rule solver is most sensitive to convergence criteria. Testing against pywbgt caught a convergence tolerance bug: our solver used `abs_tol = 0.01 K` while MetPy uses `abs_tol = 0.001 K`, producing 0.1 C differences in wet bulb temperature at high humidity.

### What synthetic tests miss

1. **Unit conventions:** Synthetic tests typically use consistent units. Real code mixes units freely (pressure in Pa here, hPa there).
2. **Argument ordering:** Synthetic tests call functions with the "documented" signature. Real code uses keyword arguments, positional arguments in different orders, and default values in ways the documentation does not anticipate.
3. **Edge cases in real data:** Real soundings have missing levels, superadiabatic layers, temperature inversions, and dewpoint depressions of zero. Synthetic profiles are usually well-behaved.
4. **Chained calculations:** Real workflows chain 5-10 MetPy functions together. A 0.01% error in `saturation_mixing_ratio` feeds into `equivalent_potential_temperature`, which feeds into `most_unstable_parcel`, which feeds into `cape_cin`. The compounding error is only visible in the end-to-end chain.

---

## Summary: The Compatibility Checklist

When adding a new MetPy-compatible function, check:

1. **All calling conventions:** Read MetPy's source (not just the docs) to find every positional/keyword variant.
2. **Default values:** Compare MetPy's defaults against your native defaults. Pay special attention to `bottom`, `depth`, and `which` parameters.
3. **Constants:** Verify you are using MetPy's exact constants, not textbook approximations.
4. **Integration formula:** For integral quantities (CAPE, SRH, precipitable water), verify you are using the same coordinate system (pressure vs height) and the same discretization.
5. **Interpolation method:** For layer-based calculations, verify you are interpolating in the same space (linear-in-p, log-p, linear-in-z, linear-in-theta).
6. **Unit handling:** Return Pint quantities from MetPy's registry, not plain floats or quantities from a fresh registry.
7. **Real-world test:** Run at least one real sounding through the full chain and compare every intermediate value against MetPy.
