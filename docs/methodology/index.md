# Methodology and Implementation Notes

These documents describe the exact algorithms, formulas, and implementation decisions in metrust. They are written for:

- **Developers** reimplementing these calculations in other languages
- **Researchers** who need to understand exactly what metrust computes
- **Contributors** who want to add new functions or fix numerical issues
- **Anyone** who hit a numerical discrepancy and wants to understand why

Each document includes the mathematical formulation, the specific numerical method chosen, and (where applicable) a comparison against MetPy's reference output.

---

## The Numerical Parity Story

metrust started with SHARPpy-heritage algorithms — Wobus polynomial approximations for saturation vapor pressure, simplified CAPE formulas, and basic storm-motion estimates. These were "roughly correct" but could diverge from MetPy by tens of J/kg on CAPE or fractions of a degree on critical angle.

Over a series of systematic replacements, metrust moved to MetPy-exact algorithms:

- **Moist lapse rate** replaced with a fourth-order Runge-Kutta (RK4) integration of the full moist-adiabatic ODE, matching MetPy closely on the supported reference-pressure paths.
- **CAPE integration** switched from a simplified buoyancy sum to MetPy's exact formulation: `g * dTv / Tv * dz`, integrated over each layer.
- **Bunkers storm motion** moved to a pressure-weighted mean wind, matching the Bunkers et al. (2000) method that MetPy implements.
- **Isentropic interpolation** adopted a Newton solver on the exact theta equation rather than a lookup-table approach.

The final parity numbers against MetPy on a representative set of soundings:

| Parameter      | Residual          |
|----------------|-------------------|
| CAPE           | +4 J/kg           |
| SRH            | +0.3 m^2/s^2      |
| Critical angle | +0.2 deg          |
| Montgomery correction | 1.0        |

These residuals are within the noise floor of floating-point differences between Python/NumPy and Rust, and are dominated by interpolation-order effects at level boundaries.

---

## Document Index

| Document | Description |
|----------|-------------|
| [Thermodynamics](thermodynamics.md) | CAPE, moist lapse rate, isentropic interpolation |
| [Wind](wind.md) | Bunkers storm motion, SRH, critical angle, bulk shear |
| [Grid Kinematics](grid-kinematics.md) | Finite differences, spherical corrections, divergence/vorticity |
| [Smoothing](smoothing.md) | SAT algorithm, Gaussian smoothing, NaN handling |
| [Severe Weather](severe-weather.md) | STP, SCP, stability indices, grid composites |
| [Interpolation](interpolation.md) | Barnes/Cressman IDW, natural neighbor |
| [Moisture](moisture.md) | Saturation vapor pressure, humidity conversions |
| [Units and Pint](units.md) | Application registry, cross-registry safety |
| [MetPy Compatibility](metpy-compat.md) | Signature matching, default parameters, lessons learned |
| [Parallelism](parallelism.md) | Rayon, GIL release, SAT, grid composites |
| [Architecture](architecture.md) | Crate structure, PyO3 bridge, wrapper patterns |
| [I/O Formats](io-formats.md) | Level III, METAR, GEMPAK, GINI |

---

## Key References

- **Bolton (1980)** — "The Computation of Equivalent Potential Temperature." *Monthly Weather Review*, 108, 1046-1053.
- **Bunkers et al. (2000)** — "Predicting Supercell Motion Using a New Hodograph Technique." *Weather and Forecasting*, 15, 61-79.
- **Koch et al. (1983)** — "An Interactive Barnes Objective Map Analysis Scheme for Use with Satellite and Conventional Data." *Journal of Climate and Applied Meteorology*, 22, 1487-1503.
- **Holton (2004)** — *An Introduction to Dynamic Meteorology*, 4th ed. Elsevier Academic Press. (Weighted continuous average formulation.)
- **Doswell & Rasmussen (1994)** — "The Effect of Neglecting the Virtual Temperature Correction on CAPE Calculations." *Weather and Forecasting*, 9, 625-629.
