"""
Verification script: compute MetPy reference values for kinematics/dynamics functions.

These reference values are used by the Rust integration test
`verify_kinematics_metpy.rs` in the metrust crate to confirm that metrust
produces numerically identical results.

Both MetPy and metrust use centered finite differences in the interior and
forward/backward differences at boundaries, so the results should match
to machine precision at interior points.

Note on frontogenesis:
  MetPy uses the form: F = (1/2)|grad(theta)|[D*cos(2*beta) - delta]
  metrust uses:        F = -1/|grad(theta)| * [(dtheta/dx)^2*(du/dx) + ...]
  These are algebraically equivalent (both are the Petterssen formulation),
  but MetPy's output is in K/(m*s) while metrust also outputs K/(m*s).

Note on baroclinic PV:
  MetPy uses the full Bluestein (1993) eq 4.5.93 which includes horizontal
  tilting terms (du/dp * dtheta/dy, dv/dp * dtheta/dx). metrust uses a
  simplified single-level form: PV = -g*(f+zeta)*(dtheta/dp). These will
  differ when there are horizontal theta gradients AND vertical wind shear.
  We test with uniform theta fields so tilting terms vanish.
"""

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

# ─────────────────────────────────────────────────────────────
# Grid setup: 5x5 uniform grid
# ─────────────────────────────────────────────────────────────

NX, NY = 5, 5
dx = 100_000.0 * units.m   # 100 km
dy = 100_000.0 * units.m

# Wind fields -- linearly varying (m/s)
u_raw = np.array([
    [10, 12, 14, 16, 18],
    [11, 13, 15, 17, 19],
    [12, 14, 16, 18, 20],
    [13, 15, 17, 19, 21],
    [14, 16, 18, 20, 22],
], dtype=float)

v_raw = np.array([
    [5,  6,  7,  8,  9],
    [6,  7,  8,  9, 10],
    [7,  8,  9, 10, 11],
    [8,  9, 10, 11, 12],
    [9, 10, 11, 12, 13],
], dtype=float)

u = u_raw * units('m/s')
v = v_raw * units('m/s')

# Temperature / potential temperature field (K)
temperature_raw = np.array([
    [290, 291, 292, 293, 294],
    [291, 292, 293, 294, 295],
    [292, 293, 294, 295, 296],
    [293, 294, 295, 296, 297],
    [294, 295, 296, 297, 298],
], dtype=float)
temperature = temperature_raw * units.K

# Latitude field for Coriolis calculations -- 45N everywhere
lat_raw = np.full((NY, NX), 45.0)
latitude = lat_raw * units.degrees

# Geopotential height field (m) -- linearly varying
height_raw = np.array([
    [5500, 5510, 5520, 5530, 5540],
    [5520, 5530, 5540, 5550, 5560],
    [5540, 5550, 5560, 5570, 5580],
    [5560, 5570, 5580, 5590, 5600],
    [5580, 5590, 5600, 5610, 5620],
], dtype=float)
height = height_raw * units.m

print("=" * 70)
print("MetPy Kinematics Reference Values")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. Divergence
# ─────────────────────────────────────────────────────────────
div = mpcalc.divergence(u, v, dx=dx, dy=dy)
div_vals = div.magnitude
print("\n--- DIVERGENCE (1/s) ---")
print(f"Full grid (row-major, flattened):")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {div_vals[j, i]:.15e}")
print(f"Center [2,2] = {div_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 2. Vorticity
# ─────────────────────────────────────────────────────────────
vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)
vort_vals = vort.magnitude
print("\n--- VORTICITY (1/s) ---")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {vort_vals[j, i]:.15e}")
print(f"Center [2,2] = {vort_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 3. Absolute vorticity
# ─────────────────────────────────────────────────────────────
abs_vort = mpcalc.absolute_vorticity(u, v, dx=dx, dy=dy, latitude=latitude)
abs_vort_vals = abs_vort.magnitude
print("\n--- ABSOLUTE VORTICITY (1/s) ---")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {abs_vort_vals[j, i]:.15e}")
print(f"Center [2,2] = {abs_vort_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 4. Advection
# ─────────────────────────────────────────────────────────────
adv = mpcalc.advection(temperature, u=u, v=v, dx=dx, dy=dy)
adv_vals = adv.magnitude
print("\n--- ADVECTION (K/s) ---")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {adv_vals[j, i]:.15e}")
print(f"Center [2,2] = {adv_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 5. Frontogenesis
# ─────────────────────────────────────────────────────────────
fronto = mpcalc.frontogenesis(temperature, u, v, dx=dx, dy=dy)
fronto_vals = fronto.magnitude
print("\n--- FRONTOGENESIS (K/m/s) ---")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {fronto_vals[j, i]:.15e}")
print(f"Center [2,2] = {fronto_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 6. Geostrophic wind
# ─────────────────────────────────────────────────────────────
u_geo, v_geo = mpcalc.geostrophic_wind(height, dx=dx, dy=dy, latitude=latitude)
u_geo_vals = u_geo.magnitude
v_geo_vals = v_geo.magnitude
print("\n--- GEOSTROPHIC WIND (m/s) ---")
print("u_geo:")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {u_geo_vals[j, i]:.15e}")
print("v_geo:")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {v_geo_vals[j, i]:.15e}")
print(f"u_geo center [2,2] = {u_geo_vals[2, 2]:.15e}")
print(f"v_geo center [2,2] = {v_geo_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 7. Barotropic potential vorticity
# ─────────────────────────────────────────────────────────────
pv_bt = mpcalc.potential_vorticity_barotropic(height, u, v, dx=dx, dy=dy, latitude=latitude)
pv_bt_vals = pv_bt.magnitude
print("\n--- BAROTROPIC PV (1/(m*s)) ---")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {pv_bt_vals[j, i]:.15e}")
print(f"Center [2,2] = {pv_bt_vals[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# 8. Baroclinic potential vorticity
#    MetPy requires 3D arrays (pressure, y, x).
#    We use 3 levels. The middle level is the one we compare.
#    Use uniform theta on each level so the tilting terms vanish
#    and the result matches metrust's simplified form.
# ─────────────────────────────────────────────────────────────
# Three pressure levels (Pa)
p_levels = np.array([85000.0, 70000.0, 50000.0]) * units.Pa
# Shape: (3, 5, 5)
pressure_3d = np.broadcast_to(p_levels[:, None, None].magnitude,
                                (3, NY, NX)) * units.Pa

# Potential temperature: uniform on each level (no horizontal gradient)
# This makes tilting terms zero, isolating the (f+zeta)*dtheta/dp term
theta_below_val = 298.0
theta_mid_val = 302.0
theta_above_val = 308.0
theta_3d_raw = np.zeros((3, NY, NX))
theta_3d_raw[0, :, :] = theta_below_val
theta_3d_raw[1, :, :] = theta_mid_val
theta_3d_raw[2, :, :] = theta_above_val
theta_3d = theta_3d_raw * units.K

# Wind: same on all 3 levels (no vertical shear => du/dp = dv/dp = 0)
u_3d_raw = np.stack([u_raw, u_raw, u_raw])  # shape (3,5,5)
v_3d_raw = np.stack([v_raw, v_raw, v_raw])
u_3d = u_3d_raw * units('m/s')
v_3d = v_3d_raw * units('m/s')

# Latitude broadcast to 3D
latitude_3d = np.broadcast_to(lat_raw, (3, NY, NX)) * units.degrees

pv_bc = mpcalc.potential_vorticity_baroclinic(
    theta_3d, pressure_3d, u_3d, v_3d,
    dx=dx, dy=dy, latitude=latitude_3d
)

# Extract the middle level
pv_bc_mid = pv_bc[1, :, :].magnitude
print("\n--- BAROCLINIC PV (PVU) at middle level (700 hPa) ---")
print("NOTE: MetPy uses full Bluestein form; metrust uses simplified form.")
print("With uniform theta per level and no vertical wind shear, they should match.")
for j in range(NY):
    for i in range(NX):
        print(f"  [{j},{i}] = {pv_bc_mid[j, i]:.15e}")
print(f"Center [2,2] = {pv_bc_mid[2, 2]:.15e}")

# ─────────────────────────────────────────────────────────────
# Also print the metrust-compatible inputs for baroclinic PV
# ─────────────────────────────────────────────────────────────
print("\n--- BAROCLINIC PV INPUTS (for metrust) ---")
print(f"theta_below = {theta_below_val}")
print(f"theta_above = {theta_above_val}")
print(f"p_below (Pa) = {p_levels[0].magnitude}")
print(f"p_above (Pa) = {p_levels[2].magnitude}")
print("NOTE: metrust uses [p_below, p_above] = [85000, 50000] Pa")
print("      and theta_below/theta_above arrays at those levels")

# Compute what metrust should give:
# PV = -g * (f + zeta) * dtheta/dp
# where dtheta/dp = (theta_above - theta_below) / (p_above - p_below)
g = 9.80665
omega = 7.2921e-5
f_45 = 2 * omega * np.sin(np.deg2rad(45.0))
dp = p_levels[2].magnitude - p_levels[0].magnitude  # 50000 - 85000 = -35000
dtheta_dp = (theta_above_val - theta_below_val) / dp  # (308 - 298) / (-35000)

# For uniform wind, relative vorticity = 0 at interior
expected_pv_uniform = -g * f_45 * dtheta_dp
print(f"\nExpected PV (uniform wind, interior): {expected_pv_uniform:.15e}")

# Now compute MetPy's dtheta/dp at the middle level using centered diff
# MetPy: dtheta/dp at level 1 = (theta[2] - theta[0]) / (p[2] - p[0])
metpy_dtheta_dp = (theta_above_val - theta_below_val) / dp
print(f"MetPy dtheta/dp = {metpy_dtheta_dp:.15e}")
print(f"f(45N) = {f_45:.15e}")

# ─────────────────────────────────────────────────────────────
# Summary of ALL center values for easy test construction
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: Center [2,2] values and all grid values")
print("=" * 70)

# Print flattened row-major arrays for all key outputs
def print_flat(name, arr):
    """Print a 2D array as a flattened row-major list."""
    flat = arr.flatten()
    print(f"\n{name} (row-major flat, {len(flat)} values):")
    for k, val in enumerate(flat):
        j, i = divmod(k, NX)
        print(f"  flat[{k:2d}] = [{j},{i}] = {val:.15e}")

print_flat("DIVERGENCE", div_vals)
print_flat("VORTICITY", vort_vals)
print_flat("ABSOLUTE_VORTICITY", abs_vort_vals)
print_flat("ADVECTION", adv_vals)
print_flat("FRONTOGENESIS", fronto_vals)
print_flat("U_GEO", u_geo_vals)
print_flat("V_GEO", v_geo_vals)
print_flat("BAROTROPIC_PV", pv_bt_vals)
print_flat("BAROCLINIC_PV_MID", pv_bc_mid)

# ─────────────────────────────────────────────────────────────
# Compact format for Rust test embedding
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPACT FORMAT: Rust-ready arrays")
print("=" * 70)

def print_rust_array(name, arr):
    flat = arr.flatten()
    vals = ", ".join(f"{v:.15e}" for v in flat)
    print(f"\nconst {name}: [f64; {len(flat)}] = [{vals}];")

print_rust_array("EXPECTED_DIV", div_vals)
print_rust_array("EXPECTED_VORT", vort_vals)
print_rust_array("EXPECTED_ABS_VORT", abs_vort_vals)
print_rust_array("EXPECTED_ADV", adv_vals)
print_rust_array("EXPECTED_FRONTO", fronto_vals)
print_rust_array("EXPECTED_U_GEO", u_geo_vals)
print_rust_array("EXPECTED_V_GEO", v_geo_vals)
print_rust_array("EXPECTED_PV_BT", pv_bt_vals)
print_rust_array("EXPECTED_PV_BC_MID", pv_bc_mid)
