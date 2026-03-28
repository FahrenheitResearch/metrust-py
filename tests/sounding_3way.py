"""Three-way sounding comparison: SharpJS vs MetPy vs metrust."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from rusbie import Herbie

H = Herbie("2026-03-27 23:00", model="hrrr", product="prs", fxx=0, verbose=False)

ds_t = H.xarray(":TMP:.*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
ds_d = H.xarray(":DPT:.*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
ds_u = H.xarray(":UGRD:.*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
ds_v = H.xarray(":VGRD:.*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
ds_h = H.xarray(":HGT:.*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})

for name in ["ds_t", "ds_d", "ds_u", "ds_v", "ds_h"]:
    obj = locals()[name]
    if isinstance(obj, list):
        locals()[name] = obj[0]

ds_t, ds_d, ds_u, ds_v, ds_h = [x[0] if isinstance(x, list) else x for x in [ds_t, ds_d, ds_u, ds_v, ds_h]]

lat_target, lon_target = 32.5462, -89.12
lon_360 = lon_target % 360
lat_vals = ds_t.latitude.values
lon_vals = ds_t.longitude.values
dist = (lat_vals - lat_target)**2 + (lon_vals - lon_360)**2
j, i = np.unravel_index(np.nanargmin(dist), dist.shape)

pres_coord = [c for c in ds_t.coords if "isobaric" in c.lower()][0]
plevs = ds_t[pres_coord].values
p_hpa = plevs / 100.0 if plevs.max() > 2000 else plevs

t_k = ds_t[list(ds_t.data_vars)[0]].values[:, j, i]
td_k = ds_d[list(ds_d.data_vars)[0]].values[:, j, i]
u_ms = ds_u[list(ds_u.data_vars)[0]].values[:, j, i]
v_ms = ds_v[list(ds_v.data_vars)[0]].values[:, j, i]
h_m = ds_h[list(ds_h.data_vars)[0]].values[:, j, i]

sort_idx = np.argsort(p_hpa)[::-1]
p = p_hpa[sort_idx].astype(np.float64)
t_c = (t_k[sort_idx] - 273.15).astype(np.float64)
td_c = (td_k[sort_idx] - 273.15).astype(np.float64)
u = u_ms[sort_idx].astype(np.float64)
v = v_ms[sort_idx].astype(np.float64)
h = h_m[sort_idx].astype(np.float64)

print(f"Nearest grid point: {lat_vals[j,i]:.4f}N {abs(lon_vals[j,i]-360):.2f}W")
print(f"Sfc: T={t_c[0]:.2f}C Td={td_c[0]:.2f}C P={p[0]:.1f}hPa Z={h[0]:.0f}m")
print(f"Levels: {len(p)}")
print()

# ─── MetPy ───
import metpy.calc as mpcalc
from metpy.units import units as mpu

p_mp = p * mpu.hPa
t_mp = t_c * mpu.degC
td_mp = td_c * mpu.degC
h_mp = h * mpu.meter
u_mp = u * mpu("m/s")
v_mp = v * mpu("m/s")

pp_sb = mpcalc.parcel_profile(p_mp, t_mp[0], td_mp[0])
mp_sbcape, mp_sbcin = mpcalc.cape_cin(p_mp, t_mp, td_mp, pp_sb)
mp_lcl_p, mp_lcl_t = mpcalc.lcl(p_mp[0], t_mp[0], td_mp[0])
mp_lfc_p, mp_lfc_t = mpcalc.lfc(p_mp, t_mp, td_mp, pp_sb)
mp_el_p, mp_el_t = mpcalc.el(p_mp, t_mp, td_mp, pp_sb)

ml_t, ml_td = mpcalc.mixed_layer(p_mp, t_mp, td_mp, depth=100 * mpu.hPa)
pp_ml = mpcalc.parcel_profile(p_mp, ml_t, ml_td)
mp_mlcape, mp_mlcin = mpcalc.cape_cin(p_mp, t_mp, td_mp, pp_ml)

mu_idx = np.argmax(mpcalc.equivalent_potential_temperature(p_mp[:20], t_mp[:20], td_mp[:20]).magnitude)
try:
    pp_mu = mpcalc.parcel_profile(p_mp[mu_idx:], t_mp[mu_idx], td_mp[mu_idx])
    pp_mu_arr = np.full(len(p), np.nan)
    pp_mu_arr[mu_idx:] = pp_mu.magnitude
    mp_mucape, mp_mucin = mpcalc.cape_cin(p_mp, t_mp, td_mp, pp_mu_arr * pp_mu.units)
except:
    mp_mucape = mp_sbcape  # fallback — MU is often same as SB
    mp_mucin = mp_sbcin

mp_li = mpcalc.lifted_index(p_mp, t_mp, pp_sb)
mp_pw = mpcalc.precipitable_water(p_mp, td_mp)
mp_ki = mpcalc.k_index(p_mp, t_mp, td_mp)
try:
    mp_tt = mpcalc.total_totals(p_mp, t_mp, td_mp)
except AttributeError:
    mp_vt = mpcalc.vertical_totals(p_mp, t_mp)
    mp_ct = mpcalc.cross_totals(p_mp, t_mp, td_mp)
    mp_tt_val = mp_vt.magnitude + mp_ct.magnitude
    mp_tt = mp_tt_val * mpu.delta_degC
else:
    mp_ct = mpcalc.cross_totals(p_mp, t_mp, td_mp)
    mp_vt = mpcalc.vertical_totals(p_mp, t_mp)

# Bunkers
rm, lm, mw = mpcalc.bunkers_storm_motion(p_mp, u_mp, v_mp, h_mp)
mp_rm_spd = np.sqrt(rm[0].magnitude**2 + rm[1].magnitude**2) * 1.944
mp_rm_dir = (270 - np.degrees(np.arctan2(rm[1].magnitude, rm[0].magnitude))) % 360

# Shear
def mp_shear(depth):
    try:
        bu, bv = mpcalc.bulk_shear(p_mp, u_mp, v_mp, height=h_mp, depth=depth * mpu.meter)
        return np.sqrt(bu.magnitude**2 + bv.magnitude**2) * 1.944
    except: return 0

# SRH
def mp_srh(depth):
    try:
        pos, neg, tot = mpcalc.storm_relative_helicity(h_mp, u_mp, v_mp, depth=depth * mpu.meter)
        return tot.magnitude, pos.magnitude, neg.magnitude
    except: return 0, 0, 0

# Lapse rates
def lr(pb, pt):
    ib = np.argmin(np.abs(p - pb))
    it = np.argmin(np.abs(p - pt))
    dT = t_c[it] - t_c[ib]
    dZ = (h[it] - h[ib]) / 1000.0
    return -dT / dZ if dZ != 0 else 0

# ─── metrust ───
from metrust.calc import cape_cin as mr_cape_cin, parcel_profile as mr_parcel_profile
from metrust.calc import lcl as mr_lcl, precipitable_water as mr_pw
from metrust.calc import k_index as mr_ki, total_totals as mr_tt, lifted_index as mr_li
from metrust.units import units as mru

p_mr = p * mru.hPa
t_mr = t_c * mru.degC
td_mr = td_c * mru.degC

# Use the Rust cape_cin_core directly (not the Python parcel_profile wrapper)
from metrust._metrust import calc as _calc
_rust_result = _calc.cape_cin(p, t_c, td_c, h - h[0], float(p[0]), float(t_c[0]), float(td_c[0]), "sb", 100.0, 300.0, None)
mr_cape_val, mr_cin_val = _rust_result[0], _rust_result[1]
# Wrap in units for display
mr_cape = mr_cape_val * mru("J/kg")
mr_cin = mr_cin_val * mru("J/kg")
mr_lcl_p, mr_lcl_t = mr_lcl(p_mr[0], t_mr[0], td_mr[0])
mr_pwat = mr_pw(p_mr, td_mr)
mr_k = mr_ki(p_mr, t_mr, td_mr)
mr_t = mr_tt(p_mr, t_mr, td_mr)

# SharpJS reference values
sjs = {
    "sbcape": 1254.83, "sbcin": -7.62, "sb_lcl": 905.92, "sb_lfc": 900.0, "sb_el": 200.0,
    "mlcape": 782.51, "mlcin": -1.73, "ml_lcl": 884.0,
    "mucape": 1254.83, "mucin": -7.62,
    "li": -3.37, "pwat": 1.23, "ki": 17.88, "tt": 43.97, "ct": 20.40, "vt": 23.58,
    "rm_dir": 337.3, "rm_spd": 24.80,
}

print("=" * 78)
print("  PARCELS            SharpJS      MetPy     metrust")
print("-" * 78)
print(f"  SBCAPE (J/kg)    {sjs['sbcape']:9.1f}  {mp_sbcape.magnitude:9.1f}  {mr_cape.magnitude:9.1f}")
print(f"  SBCIN  (J/kg)    {sjs['sbcin']:9.2f}  {mp_sbcin.magnitude:9.2f}  {mr_cin.magnitude:9.2f}")
print(f"  SB LCL (hPa)    {sjs['sb_lcl']:9.2f}  {mp_lcl_p.magnitude:9.2f}  {mr_lcl_p.magnitude:9.2f}")
try:
    print(f"  SB LFC (hPa)    {sjs['sb_lfc']:9.2f}  {mp_lfc_p.magnitude:9.2f}")
except: print(f"  SB LFC (hPa)    {sjs['sb_lfc']:9.2f}       nan")
try:
    print(f"  SB EL  (hPa)    {sjs['sb_el']:9.2f}  {mp_el_p.magnitude:9.2f}")
except: print(f"  SB EL  (hPa)    {sjs['sb_el']:9.2f}       nan")
print(f"  MLCAPE (J/kg)    {sjs['mlcape']:9.1f}  {mp_mlcape.magnitude:9.1f}")
print(f"  MLCIN  (J/kg)    {sjs['mlcin']:9.2f}  {mp_mlcin.magnitude:9.2f}")
print(f"  MUCAPE (J/kg)    {sjs['mucape']:9.1f}  {mp_mucape.magnitude:9.1f}")
print(f"  LI               {sjs['li']:9.2f}  {mp_li.magnitude[0]:9.2f}")
print()

print("  INDICES            SharpJS      MetPy     metrust")
print("-" * 78)
print(f"  K-Index          {sjs['ki']:9.2f}  {mp_ki.magnitude:9.2f}  {mr_k.magnitude:9.2f}")
print(f"  Total Totals     {sjs['tt']:9.2f}  {mp_tt.magnitude:9.2f}  {mr_t.magnitude:9.2f}")
print(f"  Cross Totals     {sjs['ct']:9.2f}  {mp_ct.magnitude:9.2f}")
print(f"  Vert Totals      {sjs['vt']:9.2f}  {mp_vt.magnitude:9.2f}")
print(f"  PWAT (in)        {sjs['pwat']:9.2f}  {mp_pw.to('inch').magnitude:9.2f}  {(mr_pwat.magnitude/25.4 if mr_pwat.magnitude > 5 else mr_pwat.magnitude):9.2f}")
print()

print("  SHEAR (kts)        SharpJS      MetPy")
print("-" * 78)
for depth, label, ref in [(500,"0-500m",5.54),(1000,"0-1km",5.83),(3000,"0-3km",12.38),(6000,"0-6km",18.35)]:
    ms = mp_shear(depth)
    print(f"  {label:15s}  {ref:9.2f}  {ms:9.2f}  ({ms-ref:+.2f})")
print()

print("  SRH (m2/s2)        SharpJS      MetPy")
print("-" * 78)
for depth, label, ref in [(500,"0-500m",3.73),(1000,"0-1km",3.70),(3000,"0-3km",33.04)]:
    tot, pos, neg = mp_srh(depth)
    print(f"  {label:15s}  {ref:9.2f}  {tot:9.2f}  ({tot-ref:+.2f})")
print()

print("  STORM MOTION       SharpJS      MetPy")
print("-" * 78)
print(f"  Bunkers RM       {sjs['rm_dir']:5.1f}/{sjs['rm_spd']:5.2f}  {mp_rm_dir:5.1f}/{mp_rm_spd:5.2f}")
print()

print("  LAPSE RATES        SharpJS      MetPy")
print("-" * 78)
for (pb,pt), label, ref in [((1000,850),"1000-850",9.48),((925,700),"925-700",6.36),
                              ((850,500),"850-500",5.84),((700,500),"700-500",6.14),((500,300),"500-300",7.75)]:
    v = lr(pb, pt)
    print(f"  {label:15s}  {ref:9.2f}  {v:9.2f}  ({v-ref:+.2f})")
print("=" * 78)
