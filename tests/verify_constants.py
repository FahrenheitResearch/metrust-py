"""
Verify every MetPy constant and print its SI base-unit value.

Run with: python tests/verify_constants.py

Output is consumed by the companion Rust integration test
(metrust/tests/verify_constants_metpy.rs) to ensure the two
libraries agree to at least 6 significant figures.
"""

import metpy.constants as mpconst
from metpy.units import units

# ── Discover every Quantity attribute in metpy.constants ──────────────
all_attrs = sorted(
    a
    for a in dir(mpconst)
    if not a.startswith("_") and type(getattr(mpconst, a)).__name__ == "Quantity"
)

print("=" * 72)
print("MetPy constants — SI base-unit magnitudes")
print("=" * 72)

for name in all_attrs:
    val = getattr(mpconst, name)
    try:
        si = val.to_base_units()
        mag = si.magnitude
        unit_str = str(si.units)
        print(f"{name:40s} = {mag: .15e}  [{unit_str}]")
    except Exception as exc:
        print(f"{name:40s} = ERROR ({exc})")

# ── Explicit table used by Rust tests ─────────────────────────────────
# Each entry: (name, MetPy constant, optional unit to convert *before*
#              calling .to_base_units()).
constants_to_check = [
    # Earth
    ("earth_gravity",           mpconst.earth_gravity,           None),
    ("earth_avg_radius",        mpconst.earth_avg_radius,        None),
    ("earth_max_declination",   mpconst.earth_max_declination,   "degrees"),
    ("earth_sfc_avg_dist_sun",  mpconst.earth_sfc_avg_dist_sun,  None),
    ("earth_avg_angular_vel",   mpconst.earth_avg_angular_vel,   None),
    ("earth_mass",              mpconst.earth_mass,              None),
    ("earth_orbit_eccentricity",mpconst.earth_orbit_eccentricity,None),
    ("earth_solar_irradiance",  mpconst.earth_solar_irradiance,  None),

    # Dry air
    ("dry_air_gas_constant",    mpconst.dry_air_gas_constant,    None),
    ("dry_air_spec_heat_press", mpconst.dry_air_spec_heat_press, None),
    ("dry_air_spec_heat_vol",   mpconst.dry_air_spec_heat_vol,   None),
    ("dry_air_spec_heat_ratio", mpconst.dry_air_spec_heat_ratio, None),
    ("dry_air_molecular_weight",mpconst.dry_air_molecular_weight,None),
    ("dry_air_density_stp",     mpconst.dry_air_density_stp,     None),
    ("dry_adiabatic_lapse_rate",mpconst.dry_adiabatic_lapse_rate,None),

    # Water / moist thermodynamics
    ("water_gas_constant",         mpconst.water_gas_constant,          None),
    ("water_heat_vaporization",    mpconst.water_heat_vaporization,     None),
    ("water_heat_fusion",          mpconst.water_heat_fusion,           None),
    ("water_heat_sublimation",     mpconst.water_heat_sublimation,      None),
    ("water_molecular_weight",     mpconst.water_molecular_weight,      None),
    ("water_specific_heat",        mpconst.water_specific_heat,         None),
    ("density_water",              mpconst.density_water,               None),
    ("density_ice",                mpconst.density_ice,                 None),
    ("ice_specific_heat",          mpconst.ice_specific_heat,           None),
    ("sat_pressure_0c",            mpconst.sat_pressure_0c,             None),
    ("water_triple_point_temperature", mpconst.water_triple_point_temperature, None),
    ("wv_specific_heat_press",     mpconst.wv_specific_heat_press,      None),
    ("wv_specific_heat_vol",       mpconst.wv_specific_heat_vol,        None),
    ("wv_specific_heat_ratio",     mpconst.wv_specific_heat_ratio,      None),

    # Universal
    ("R",                       mpconst.R,                       None),
    ("G",                       mpconst.G,                       None),
    ("GM",                      mpconst.GM,                      None),

    # Derived / dimensionless
    ("epsilon",                 mpconst.epsilon,                 None),
    ("kappa",                   mpconst.kappa,                   None),

    # Reference values
    ("P0",                      mpconst.P0,                      None),
    ("T0",                      mpconst.T0,                      None),

    # Short aliases (should duplicate longer names)
    ("g",                       mpconst.g,                       None),
    ("Rd",                      mpconst.Rd,                      None),
    ("Rv",                      mpconst.Rv,                      None),
    ("Cp_d",                    mpconst.Cp_d,                    None),
    ("Cv_d",                    mpconst.Cv_d,                    None),
    ("Lv",                      mpconst.Lv,                      None),
    ("Lf",                      mpconst.Lf,                      None),
    ("Ls",                      mpconst.Ls,                      None),
    ("Re",                      mpconst.Re,                      None),
    ("Md",                      mpconst.Md,                      None),
    ("Mw",                      mpconst.Mw,                      None),
    ("omega",                   mpconst.omega,                   None),
    ("S",                       mpconst.S,                       None),
    ("d",                       mpconst.d,                       None),
    ("me",                      mpconst.me,                      None),
    ("delta",                   mpconst.delta,                   None),
    ("Cp_l",                    mpconst.Cp_l,                    None),
    ("Cp_v",                    mpconst.Cp_v,                    None),
    ("Cp_i",                    mpconst.Cp_i,                    None),
    ("Cv_v",                    mpconst.Cv_v,                    None),
    ("rho_d",                   mpconst.rho_d,                   None),
    ("rho_l",                   mpconst.rho_l,                   None),
    ("rho_i",                   mpconst.rho_i,                   None),
    ("gamma_d",                 mpconst.gamma_d,                 None),
    ("poisson_exponent",        mpconst.poisson_exponent,        None),
    ("molecular_weight_ratio",  mpconst.molecular_weight_ratio,  None),
    ("pot_temp_ref_press",      mpconst.pot_temp_ref_press,      None),
]

print()
print("=" * 72)
print("Explicit constant table for Rust test verification")
print("=" * 72)

for name, val, convert_unit in constants_to_check:
    try:
        if convert_unit:
            v = val.to(convert_unit)
            mag = v.magnitude
            unit_str = convert_unit
        else:
            si = val.to_base_units()
            mag = si.magnitude
            unit_str = str(si.units)
        print(f"{name:40s} = {mag: .15e}  [{unit_str}]")
    except Exception as exc:
        print(f"{name:40s} = ERROR ({exc})")
