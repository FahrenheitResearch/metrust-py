# Units and Constants Reference

metrust uses [Pint](https://pint.readthedocs.io/) for physical unit handling
and exposes a full set of meteorological constants computed in Rust.

---

## Unit Registry

### `metrust.units.units`

metrust creates its own `pint.UnitRegistry` instance, separate from any
registry that MetPy may create. This is the registry you should use for all
metrust operations.

```python
from metrust.units import units
```

The registry is configured with `autoconvert_offset_to_baseunit=True`, which
means offset units like `degC` and `degF` are automatically converted to
their base unit (kelvin) during arithmetic. This matches the behavior that
MetPy users expect.

### Common Unit Patterns

Attach units to values by multiplying with the registry:

```python
from metrust.units import units

# Scalars
p = 850 * units.hPa
T = 20 * units.degC
Td = 15 * units.degF
ws = 25 * units.knot

# Arrays
import numpy as np
pressure = np.array([1000, 925, 850, 700, 500]) * units.hPa
temperature = np.array([25, 20, 15, 5, -15]) * units.degC
```

### Meteorological Aliases

The registry pre-registers the following aliases that are commonly used in
meteorology. These are guaranteed to be available even on older Pint versions:

| Alias | Definition | Notes |
|---|---|---|
| `units.degC` | degree Celsius (offset from kelvin) | `273.15 K` offset |
| `units.degF` | degree Fahrenheit (offset from kelvin) | `255.372 K` offset |
| `units.hPa` | hectopascal = 100 Pa | Standard pressure unit |
| `units.knot` | 0.514444 m/s | Wind speed unit |

All standard Pint units remain available as well (`units.meter`, `units.kelvin`,
`units.pascal`, `units.kg`, etc.).

### Registry Compatibility Note

metrust uses its own `pint.UnitRegistry`, which is a **separate instance**
from MetPy's `metpy.units.units` registry. Pint does not allow mixing
quantities from different registries in the same operation. If you are using
both metrust and MetPy in the same script, be consistent about which
registry you use for a given computation:

```python
# Do this -- pick one registry per computation
from metrust.units import units
p = 850 * units.hPa
T = 20 * units.degC
theta = metrust.calc.potential_temperature(p, T)

# Do NOT mix registries in one call
from metrust.units import units as mu
from metpy.units import units as mpu
p = 850 * mpu.hPa       # MetPy registry
T = 20 * mu.degC         # metrust registry
# theta = metrust.calc.potential_temperature(p, T)  # raises DimensionalityError
```

---

## Internal Helper Functions

These functions are not part of the public API but are documented here for
contributors working on metrust internals. They live in `metrust.units` and
are used throughout `metrust.calc` to bridge between Pint quantities and the
raw floats that the Rust backend expects.

### `_strip(quantity, target_unit)`

Strip Pint units from a quantity, converting to `target_unit` first. If the
input is already a plain float or numpy array (no `.magnitude` attribute),
it is returned unchanged. This lets callers pass raw numbers when they
already know the units are correct.

```python
from metrust.units import _strip, units

_strip(850 * units.hPa, units.Pa)      # 85000.0
_strip(20 * units.degC, units.kelvin)   # 293.15
_strip(42.0, units.Pa)                  # 42.0 (no conversion, already bare)
```

### `_strip_or_none(quantity, target_unit)`

Like `_strip`, but passes `None` through unchanged. Used for optional
parameters.

```python
from metrust.units import _strip_or_none, units

_strip_or_none(850 * units.hPa, units.Pa)  # 85000.0
_strip_or_none(None, units.Pa)             # None
```

### `_attach(value, unit_str)`

Attach Pint units to a bare numeric value. Wraps
`units.Quantity(value, unit_str)`.

```python
from metrust.units import _attach

result = _attach(293.15, "kelvin")  # Quantity(293.15, 'kelvin')
```

### `_as_float(v)`

Ensure a value is a Python `float`. Extracts `.item()` from 0-d numpy
arrays, which commonly arise from scalar Rust computations.

```python
from metrust.units import _as_float
import numpy as np

_as_float(np.array(3.14))  # 3.14 (plain float, not 0-d array)
_as_float(3.14)            # 3.14
```

### `_as_1d(v)`

Ensure the input is a contiguous float64 1-D numpy array. Used to
prepare data before passing it across the PyO3 boundary to Rust.

```python
from metrust.units import _as_1d
import numpy as np

_as_1d([1, 2, 3])                          # array([1., 2., 3.], dtype=float64)
_as_1d(np.array([[1, 2], [3, 4]]))         # array([1., 2., 3., 4.], dtype=float64)
```

---

## Constants

### `metrust.constants`

All physical constants are implemented in Rust and exposed as plain `float`
values in SI base units (unless otherwise noted). The module mirrors MetPy's
`metpy.constants`, providing both long descriptive names and short aliases.

```python
from metrust.constants import earth_gravity, Rd, epsilon
from metrust import constants as mpconsts

print(mpconsts.g)   # 9.80665
print(mpconsts.Rd)  # 287.04749...
```

Sources follow the same references as MetPy: CODATA 2018, the U.S. Standard
Atmosphere (1976), and the WMO International Meteorological Tables.

### Universal Constants

| Name | Alias | Value | Unit | Description |
|---|---|---|---|---|
| `R` | -- | 8.314462618 | J mol^-1 K^-1 | Universal gas constant |
| `stefan_boltzmann` | -- | 5.670374419e-8 | W m^-2 K^-4 | Stefan-Boltzmann constant |
| `gravitational_constant` | -- | 6.6743e-11 | m^3 kg^-1 s^-2 | Newtonian gravitational constant |

### Earth Constants

| Name | Alias | Value | Unit | Description |
|---|---|---|---|---|
| `earth_avg_radius` | `Re` | 6371008.7714 | m | Mean radius of the Earth |
| `noaa_mean_earth_radius` | -- | 6371008.7714 | m | Same as `earth_avg_radius` |
| `earth_gravity` | `g` | 9.80665 | m s^-2 | Standard gravitational acceleration |
| `earth_gravitational_acceleration` | -- | 9.80665 | m s^-2 | Same as `earth_gravity` |
| `omega` | -- | 7.292115e-5 | rad s^-1 | Angular velocity of Earth's rotation |
| `earth_avg_density` | -- | 5515.0 | kg m^-3 | Mean density of the Earth |
| `earth_max_declination` | -- | 23.45 | degrees | Maximum solar declination angle |
| `GM` | -- | 3.986005e14 | m^3 s^-2 | Geocentric gravitational constant |
| `earth_mass` | -- | 5.972e24 | kg | Mass of the Earth |
| `earth_orbit_eccentricity` | -- | 0.0167 | -- | Eccentricity of Earth's orbit |
| `earth_sfc_avg_dist_sun` | -- | 1.496e11 | m | Mean Earth-Sun distance |
| `earth_solar_irradiance` | -- | 1360.8 | W m^-2 | Total solar irradiance |

### Dry Air Constants

| Name | Alias | Value | Unit | Description |
|---|---|---|---|---|
| `dry_air_gas_constant` | `Rd` | 287.048 | J kg^-1 K^-1 | Specific gas constant for dry air (R / Md) |
| `dry_air_spec_heat_press` | `Cp_d` | 1004.67 | J kg^-1 K^-1 | Specific heat at constant pressure (Rd / kappa) |
| `dry_air_spec_heat_vol` | `Cv_d` | 717.62 | J kg^-1 K^-1 | Specific heat at constant volume (Cp_d - Rd) |
| `poisson_exponent_dry_air` | `kappa` | 2/7 | -- | Poisson constant for dry air |
| `molecular_weight_dry_air` | -- | 0.02896546 | kg mol^-1 | Mean molecular weight of dry air |
| `dry_air_molecular_weight` | -- | 0.02896546 | kg mol^-1 | Same as `molecular_weight_dry_air` |
| `dry_air_density_stp` | `rho_d_stp` | 1.225 | kg m^-3 | Density of dry air at STP (P_stp / (Rd * T_stp)) |
| `epsilon` | -- | 0.6220 | -- | Ratio Mw / Md (also equals Rd / Rv) |
| `dry_air_spec_heat_ratio` | -- | 1.4 | -- | Cp_d / Cv_d |
| `dry_adiabatic_lapse_rate` | -- | 0.00976 | K m^-1 | g / Cp_d |

### Water and Moist Thermodynamics Constants

| Name | Alias | Value | Unit | Description |
|---|---|---|---|---|
| `water_gas_constant` | `Rv` | 461.52 | J kg^-1 K^-1 | Specific gas constant for water vapour (R / Mw) |
| `water_specific_heat_vapor` | `Cp_v` | 1860.08 | J kg^-1 K^-1 | Specific heat of water vapour at constant pressure |
| `Cv_v` | -- | 1398.55 | J kg^-1 K^-1 | Specific heat of water vapour at constant volume |
| `rho_l` | -- | 999.97 | kg m^-3 | Density of liquid water at 0 C |
| `density_ice` | `rho_i` | 917.0 | kg m^-3 | Density of ice |
| `water_heat_vaporization` | `Lv` | 2500840 | J kg^-1 | Latent heat of vaporisation at 0 C |
| `water_heat_fusion` | `Lf` | 333700 | J kg^-1 | Latent heat of fusion at 0 C |
| `water_heat_sublimation` | `Ls` | 2834540 | J kg^-1 | Latent heat of sublimation at 0 C |
| `water_specific_heat_liquid` | `Cp_l` | 4219.4 | J kg^-1 K^-1 | Specific heat of liquid water at 0 C |
| `ice_specific_heat` | `Cp_i` | 2090.0 | J kg^-1 K^-1 | Specific heat of ice |
| `water_triple_point_temperature` | `T0` | 273.16 | K | Triple point temperature of water |
| `T_freeze` | -- | 273.15 | K | Freezing point of water |
| `sat_pressure_0c` | -- | 611.2 | Pa | Saturation vapour pressure at 0 C |
| `wv_specific_heat_ratio` | -- | 1.33 | -- | Cp_v / Cv_v |
| `molecular_weight_water` | -- | 0.018015268 | kg mol^-1 | Mean molecular weight of water |
| `water_molecular_weight` | -- | 0.018015268 | kg mol^-1 | Same as `molecular_weight_water` |

### Standard Atmosphere Constants

| Name | Alias | Value | Unit | Description |
|---|---|---|---|---|
| `pot_temp_ref_press` | `P0` | 100000 | Pa | Reference pressure for potential temperature (1000 hPa) |
| `P_stp` | -- | 101325 | Pa | Standard atmospheric pressure at sea level |
| `T_stp` | -- | 288.15 | K | Standard temperature at sea level |

### Usage Examples

```python
from metrust import constants as c
from metrust.units import units

# Use constants directly (plain floats, SI units)
theta = 300  # K
p = 85000    # Pa
T = theta * (p / c.P0) ** c.kappa

# Combine with Pint units
p_hpa = 850 * units.hPa
# Convert to Pa for use with constants
p_pa = p_hpa.to(units.Pa).magnitude

# Coriolis parameter at 45 N
import math
f = 2 * c.omega * math.sin(math.radians(45))
```

### Derivation Chain

The derived constants follow MetPy's derivation chain:

- `Rd = R / Md` (universal gas constant / molecular weight of dry air)
- `Rv = R / Mw` (universal gas constant / molecular weight of water)
- `kappa = 2/7` (Poisson constant, exact fraction)
- `Cp_d = Rd / kappa` (specific heat at constant pressure for dry air)
- `Cv_d = Cp_d - Rd` (specific heat at constant volume for dry air)
- `epsilon = Mw / Md` (molecular weight ratio, also equals `Rd / Rv`)
- `dry_adiabatic_lapse_rate = g / Cp_d`
- `rho_d_stp = P_stp / (Rd * T_stp)`
