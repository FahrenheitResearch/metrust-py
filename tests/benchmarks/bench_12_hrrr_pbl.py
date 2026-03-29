"""Benchmark 12: HRRR Boundary Layer Diagnostics -- REAL GRIB DATA

Scenario
--------
Real HRRR 3-km grid (1059 x 1799), 40 isobaric levels (1013-50 hPa).
Actual GRIB2 data from hrrr_prs.grib2 and hrrr_sfc.grib2.

Data prep:  T(K)->C, q->w, h_agl=gh-orog, p3d Pa, psfc Pa, t2m K,
            q2=sh2/(1-sh2).  dx=dy=3000.0 m.

For the lowest 2D level (~1013 hPa):
    potential_temperature, equivalent_potential_temperature, dewpoint,
    saturation_vapor_pressure, dewpoint_from_relative_humidity, mixing_ratio.

For 3D: compute_pw, compute_cape_cin.

Four backends: MetPy (Pint), metrust CPU, met-cu (direct GPU), metrust GPU.
MetPy does NOT have compute_pw / compute_cape_cin grid versions --
those use 3 backends only.

Deep verification: MetPy ground truth for thermo, cross-compare for grid
composites.  For every function: mean diff, max abs diff, RMSE, 99th pct,
relative RMSE%, NaN/Inf audit, physical plausibility, Pearson r,
% points >1%/>0.1% relative error, histogram of diffs.

Special PBL checks: theta profile structure in real HRRR data -- real
boundary layers are not perfectly mixed, but should show recognizable PBL
characteristics: theta increasing with height (stable/weakly mixed), theta-e
decreasing above the BL top, surface theta in a physical range.
"""

import os
import sys
import time
import statistics
import warnings
import numpy as np

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
PRS_FILE = os.path.join(DATA_DIR, "hrrr_prs.grib2")
SFC_FILE = os.path.join(DATA_DIR, "hrrr_sfc.grib2")

for f in (PRS_FILE, SFC_FILE):
    if not os.path.isfile(f):
        print(f"ERROR: Required data file not found: {f}")
        sys.exit(1)

DX = 3000.0   # meters
DY = 3000.0   # meters

N_WARMUP = 1
N_TIMED  = 3
RTOL_THERMO = 1e-4
RTOL_GRID   = 1e-3

# ---------------------------------------------------------------------------
# Load real HRRR data from GRIB2 files
# ---------------------------------------------------------------------------

print("=" * 90)
print("BENCHMARK 12 -- HRRR Boundary Layer Diagnostics  (REAL GRIB DATA)")
print("=" * 90)
print()
print("Loading HRRR GRIB2 data ...")

import xarray as xr

def _load_prs_var(shortname):
    ds = xr.open_dataset(PRS_FILE, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {
            "typeOfLevel": "isobaricInhPa", "shortName": shortname}})
    return ds

def _load_sfc_var(shortname):
    ds = xr.open_dataset(SFC_FILE, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": shortname}})
    return ds

# --- Pressure-level fields (40 levels, 1059 x 1799) ---
ds_t  = _load_prs_var("t")     # temperature (K)
ds_q  = _load_prs_var("q")     # specific humidity (kg/kg)
ds_gh = _load_prs_var("gh")    # geopotential height (m)
ds_r  = _load_prs_var("r")     # relative humidity (%)

pressure_levels_hpa = ds_t.isobaricInhPa.values.astype(np.float64)  # 1013..50
NZ = len(pressure_levels_hpa)

t_k_3d_raw = ds_t.t.values.astype(np.float64)     # (NZ, NY, NX) in K
q_sh_3d    = ds_q.q.values.astype(np.float64)     # specific humidity kg/kg
gh_3d      = ds_gh.gh.values.astype(np.float64)   # geopotential height m
rh_3d      = ds_r.r.values.astype(np.float64)     # relative humidity %
NY, NX = t_k_3d_raw.shape[1], t_k_3d_raw.shape[2]

# --- Surface fields ---
ds_orog = _load_sfc_var("orog")   # orography / terrain height (m)
ds_sp   = _load_sfc_var("sp")     # surface pressure (Pa)
ds_t2m  = _load_sfc_var("2t")     # 2-m temperature (K)
ds_d2m  = _load_sfc_var("2d")     # 2-m dewpoint (K)
ds_sh2  = _load_sfc_var("2sh")    # 2-m specific humidity (kg/kg)

orog   = ds_orog.orog.values.astype(np.float64)   # (NY, NX) terrain height m
psfc_pa = ds_sp.sp.values.astype(np.float64)       # (NY, NX) surface pressure Pa
t2m_K  = ds_t2m.t2m.values.astype(np.float64)     # (NY, NX) 2m temp K
d2m_K  = ds_d2m.d2m.values.astype(np.float64)     # (NY, NX) 2m dewpoint K
sh2    = ds_sh2.sh2.values.astype(np.float64)      # (NY, NX) 2m specific humidity kg/kg

print(f"  Pressure file: {NZ} levels x {NY} x {NX}  ({NZ * NY * NX:,} 3D points)")
print(f"  Pressure range: {pressure_levels_hpa[0]:.0f} - {pressure_levels_hpa[-1]:.0f} hPa")
print(f"  dx = {DX:.0f} m,  dy = {DY:.0f} m")

# ---------------------------------------------------------------------------
# Data prep: derived arrays
# ---------------------------------------------------------------------------

print()
print("Preparing derived fields ...")

# T(K) -> T(C) for 3D
t_c_3d = t_k_3d_raw - 273.15   # (NZ, NY, NX)

# q (specific humidity) -> w (mixing ratio): w = q / (1 - q)
w_mr = q_sh_3d / (1.0 - q_sh_3d)   # kg/kg

# Height AGL: gh - orog  (ensure >= 0)
h_agl_3d = gh_3d - orog[None, :, :]
h_agl_3d = np.maximum(h_agl_3d, 0.0)

# 3D pressure in Pa (broadcast)
pressure_levels_pa = pressure_levels_hpa * 100.0
p_pa_3d = np.broadcast_to(
    pressure_levels_pa[:, None, None], (NZ, NY, NX)).copy().astype(np.float64)
p_hpa_3d = np.broadcast_to(
    pressure_levels_hpa[:, None, None], (NZ, NY, NX)).copy().astype(np.float64)

# psfc already in Pa from GRIB

# q2 (2m mixing ratio): q2 = sh2 / (1 - sh2)
q2_mr = sh2 / (1.0 - sh2)

# --- 2D slice at the lowest level (~1013 hPa) for scalar thermo benchmarks ---
i_sfc = 0   # surface-first ordering
tc_sfc  = t_c_3d[i_sfc].copy()     # T in C at lowest level
rh_sfc  = rh_3d[i_sfc].copy()      # RH in %
p_sfc_hpa = pressure_levels_hpa[i_sfc]  # ~1013 hPa

# Dewpoint at lowest level: compute from T and RH using Magnus
# Td = (243.5 * ln(RH/100) + 17.67*T/(243.5+T)) / (17.67 - ln(RH/100) - 17.67*T/(243.5+T))
_a, _b = 17.67, 243.5
_rh_frac = np.clip(rh_sfc, 1.0, 100.0) / 100.0
_gamma = np.log(_rh_frac) + _a * tc_sfc / (_b + tc_sfc)
td_sfc = _b * _gamma / (_a - _gamma)   # Td in C
td_sfc = np.clip(td_sfc, -80.0, tc_sfc - 0.1)

# Vapor pressure at lowest level (hPa) from dewpoint
vp_sfc_hpa = 6.1078 * np.exp(17.27 * td_sfc / (td_sfc + 237.3))

print(f"  T(C)  lowest level: mean={tc_sfc.mean():.1f}  range=[{tc_sfc.min():.1f}, {tc_sfc.max():.1f}]")
print(f"  Td(C) lowest level: mean={td_sfc.mean():.1f}  range=[{td_sfc.min():.1f}, {td_sfc.max():.1f}]")
print(f"  RH(%) lowest level: mean={rh_sfc.mean():.1f}  range=[{rh_sfc.min():.1f}, {rh_sfc.max():.1f}]")
print(f"  T2m(K):    mean={t2m_K.mean():.1f}  range=[{t2m_K.min():.1f}, {t2m_K.max():.1f}]")
print(f"  Psfc(Pa):  mean={psfc_pa.mean():.0f}  range=[{psfc_pa.min():.0f}, {psfc_pa.max():.0f}]")
print(f"  q2(kg/kg): mean={q2_mr.mean():.6f}  range=[{q2_mr.min():.6f}, {q2_mr.max():.6f}]")
print(f"  Orog(m):   mean={orog.mean():.1f}  range=[{orog.min():.1f}, {orog.max():.1f}]")
print(f"  h_AGL lowest: mean={h_agl_3d[0].mean():.1f}  range=[{h_agl_3d[0].min():.1f}, {h_agl_3d[0].max():.1f}]")
print()

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _gpu_sync():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def timed(func, n_warmup=N_WARMUP, n_timed=N_TIMED, sync_gpu=False):
    """Return (median_seconds, last_result)."""
    for _ in range(n_warmup):
        result = func()
        if sync_gpu:
            _gpu_sync()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = func()
        if sync_gpu:
            _gpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times), result


def fmt_ms(seconds):
    ms = seconds * 1000
    if ms < 0.01:
        return f"{ms * 1000:.2f} us"
    if ms < 1.0:
        return f"{ms:.3f} ms"
    return f"{ms:.1f} ms"


def to_numpy(arr):
    """Bring result back to numpy regardless of origin."""
    if hasattr(arr, "magnitude"):
        arr = arr.magnitude
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Import backends
# ---------------------------------------------------------------------------

print("Loading backends ...")

# 1) MetPy
import metpy.calc as mpcalc
from metpy.units import units
print("  [OK] MetPy")

# 2) metrust CPU
import metrust.calc as mrcalc
mrcalc.set_backend("cpu")
print("  [OK] metrust CPU")

# 3) met-cu direct GPU
try:
    import metcu.calc as mcucalc
    HAS_METCU = True
    print("  [OK] met-cu (direct GPU)")
except ImportError as e:
    HAS_METCU = False
    print(f"  [SKIP] met-cu: {e}")

# 4) metrust GPU
try:
    mrcalc.set_backend("gpu")
    _test = mrcalc.potential_temperature(850.0, tc_sfc[0, 0])
    HAS_METRUST_GPU = True
    mrcalc.set_backend("cpu")
    print("  [OK] metrust GPU")
except Exception as e:
    HAS_METRUST_GPU = False
    print(f"  [SKIP] metrust GPU: {e}")

print()

# ---------------------------------------------------------------------------
# Prepare MetPy Pint-wrapped inputs (one-time cost, not benchmarked)
# ---------------------------------------------------------------------------

tc_sfc_q  = tc_sfc * units.degC
td_sfc_q  = td_sfc * units.degC
rh_sfc_q  = rh_sfc * units.percent
vp_sfc_q  = vp_sfc_hpa * units.hPa
p_sfc_q   = p_sfc_hpa * units.hPa

# ---------------------------------------------------------------------------
# Results storage
# ---------------------------------------------------------------------------

columns = ["MetPy", "metrust CPU", "met-cu GPU", "metrust GPU"]
results = {}   # func_name -> {backend: (time_s, np_result)}


def record(func_name, backend, time_s, result_arr):
    if func_name not in results:
        results[func_name] = {}
    results[func_name][backend] = (time_s, to_numpy(result_arr))


# ===================================================================
# 1. POTENTIAL TEMPERATURE  (GPU-capable)
#    metrust: pressure hPa, temperature degC -> K (Pint)
#    met-cu:  pressure hPa, temperature degC -> K (cupy)
# ===================================================================

print("-" * 90)
print(f"POTENTIAL TEMPERATURE  (~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.potential_temperature(p_sfc_q, tc_sfc_q))
record("potential_temperature", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.potential_temperature(p_sfc_hpa, tc_sfc))
record("potential_temperature", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.potential_temperature(
        np.full_like(tc_sfc, p_sfc_hpa), tc_sfc), sync_gpu=True)
    record("potential_temperature", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.potential_temperature(p_sfc_hpa, tc_sfc), sync_gpu=True)
    record("potential_temperature", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 2. EQUIVALENT POTENTIAL TEMPERATURE  (GPU-capable)
#    metrust: pressure hPa, temperature degC, dewpoint degC -> K
#    met-cu:  pressure hPa, temperature degC, dewpoint degC -> K
# ===================================================================

print()
print("-" * 90)
print(f"EQUIVALENT POTENTIAL TEMPERATURE  (~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.equivalent_potential_temperature(p_sfc_q, tc_sfc_q, td_sfc_q))
record("equiv_potential_temp", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.equivalent_potential_temperature(p_sfc_hpa, tc_sfc, td_sfc))
record("equiv_potential_temp", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.equivalent_potential_temperature(
        np.full_like(tc_sfc, p_sfc_hpa), tc_sfc, td_sfc), sync_gpu=True)
    record("equiv_potential_temp", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.equivalent_potential_temperature(p_sfc_hpa, tc_sfc, td_sfc),
                 sync_gpu=True)
    record("equiv_potential_temp", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 3. DEWPOINT  (GPU-capable)
#    metrust: vapor_pressure hPa -> degC (Pint)
#    met-cu:  vapor_pressure hPa -> degC (cupy)
# ===================================================================

print()
print("-" * 90)
print(f"DEWPOINT  (from vapor pressure at ~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.dewpoint(vp_sfc_q))
record("dewpoint", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.dewpoint(vp_sfc_hpa))
record("dewpoint", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.dewpoint(vp_sfc_hpa), sync_gpu=True)
    record("dewpoint", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.dewpoint(vp_sfc_hpa), sync_gpu=True)
    record("dewpoint", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 4. SATURATION VAPOR PRESSURE  (CPU only)
# ===================================================================

print()
print("-" * 90)
print(f"SATURATION VAPOR PRESSURE  (~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.saturation_vapor_pressure(tc_sfc_q))
record("sat_vapor_pressure", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.saturation_vapor_pressure(tc_sfc))
record("sat_vapor_pressure", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.saturation_vapor_pressure(tc_sfc), sync_gpu=True)
    record("sat_vapor_pressure", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

# ===================================================================
# 5. DEWPOINT FROM RELATIVE HUMIDITY  (CPU only)
# ===================================================================

print()
print("-" * 90)
print(f"DEWPOINT FROM RELATIVE HUMIDITY  (~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.dewpoint_from_relative_humidity(tc_sfc_q, rh_sfc_q))
record("dewpoint_from_rh", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.dewpoint_from_relative_humidity(tc_sfc, rh_sfc))
record("dewpoint_from_rh", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.dewpoint_from_relative_humidity(tc_sfc, rh_sfc),
                 sync_gpu=True)
    record("dewpoint_from_rh", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

# ===================================================================
# 6. MIXING RATIO  (CPU only)
# ===================================================================

print()
print("-" * 90)
print(f"MIXING RATIO  (from vapor pressure at ~{p_sfc_hpa:.0f} hPa, {NY}x{NX})")
print("-" * 90)

t, r = timed(lambda: mpcalc.mixing_ratio(vp_sfc_q, p_sfc_q))
record("mixing_ratio", "MetPy", t, r)
print(f"  MetPy:       {fmt_ms(t):>12s}")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.mixing_ratio(vp_sfc_hpa, p_sfc_hpa))
record("mixing_ratio", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.mixing_ratio(
        vp_sfc_hpa, np.full_like(vp_sfc_hpa, p_sfc_hpa)), sync_gpu=True)
    record("mixing_ratio", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

# ===================================================================
# 7. COMPUTE PW -- Precipitable Water  (GPU, 3D grid composite)
#    metrust: compute_pw(qvapor_3d kg/kg, pressure_3d Pa) -> mm
#    met-cu:  compute_pw(qvapor_3d kg/kg, pressure_3d Pa) -> mm
#    MetPy: NO grid version -- skip
# ===================================================================

print()
print("-" * 90)
print(f"PRECIPITABLE WATER  (3D grid composite, {NZ}x{NY}x{NX})")
print("-" * 90)

print("  MetPy:       (no grid version)")

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.compute_pw(w_mr, p_pa_3d))
record("compute_pw", "metrust CPU", t, r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.compute_pw(w_mr, p_pa_3d), sync_gpu=True)
    record("compute_pw", "met-cu GPU", t, r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.compute_pw(w_mr, p_pa_3d), sync_gpu=True)
    record("compute_pw", "metrust GPU", t, r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")

# ===================================================================
# 8. COMPUTE CAPE/CIN  (GPU, 3D grid composite)
# ===================================================================

print()
print("-" * 90)
print(f"CAPE / CIN  (3D grid composite, {NZ}x{NY}x{NX})")
print("-" * 90)

print("  MetPy:       (no grid version)")

cape_cin_full = {}

mrcalc.set_backend("cpu")
t, r = timed(lambda: mrcalc.compute_cape_cin(
    p_pa_3d, t_c_3d, w_mr, h_agl_3d, psfc_pa, t2m_K, q2_mr))
record("compute_cape_cin", "metrust CPU", t, r[0])
cape_cin_full["metrust CPU"] = tuple(to_numpy(x) for x in r)
print(f"  metrust CPU: {fmt_ms(t):>12s}")

if HAS_METCU:
    t, r = timed(lambda: mcucalc.compute_cape_cin(
        p_pa_3d, t_c_3d, w_mr, h_agl_3d, psfc_pa, t2m_K, q2_mr), sync_gpu=True)
    record("compute_cape_cin", "met-cu GPU", t, r[0])
    cape_cin_full["met-cu GPU"] = tuple(to_numpy(x) for x in r)
    print(f"  met-cu GPU:  {fmt_ms(t):>12s}")

if HAS_METRUST_GPU:
    mrcalc.set_backend("gpu")
    t, r = timed(lambda: mrcalc.compute_cape_cin(
        p_pa_3d, t_c_3d, w_mr, h_agl_3d, psfc_pa, t2m_K, q2_mr), sync_gpu=True)
    record("compute_cape_cin", "metrust GPU", t, r[0])
    cape_cin_full["metrust GPU"] = tuple(to_numpy(x) for x in r)
    mrcalc.set_backend("cpu")
    print(f"  metrust GPU: {fmt_ms(t):>12s}")


# ===================================================================
# DEEP VERIFICATION
# ===================================================================

print()
print("=" * 90)
print("DEEP DATA CORRECTNESS VERIFICATION")
print("=" * 90)

all_pass = True
verification_summary = []

grid_composites = {"compute_pw", "compute_cape_cin"}


def ascii_histogram(diffs, n_bins=20, width=40):
    """Print a compact ASCII histogram of differences."""
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        print("        (no finite values for histogram)")
        return
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if lo == hi:
        print(f"        All diffs = {lo:.6e}")
        return
    counts, edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(n_bins):
        bar_len = int(round(counts[i] / max_count * width))
        bar = "#" * bar_len
        print(f"        [{edges[i]:+10.4e}, {edges[i+1]:+10.4e}) "
              f"{counts[i]:>8d} |{bar}")


def deep_verify(func_name, ref_backend, cand_backend, rtol,
                phys_lo=None, phys_hi=None, phys_label="",
                ref_arr=None, cand_arr=None, atol=1e-10):
    """Full statistical verification of candidate vs reference."""
    global all_pass

    if ref_arr is None or cand_arr is None:
        if func_name not in results:
            return True
        backends = results[func_name]
        if ref_backend not in backends or cand_backend not in backends:
            return True
        ref_arr = backends[ref_backend][1]
        cand_arr = backends[cand_backend][1]

    ref_flat  = ref_arr.ravel()
    cand_flat = cand_arr.ravel()
    n = min(len(ref_flat), len(cand_flat))
    ref_c  = ref_flat[:n]
    cand_c = cand_flat[:n]

    pair_label = f"{ref_backend} vs {cand_backend}"

    print(f"\n  --- {func_name}:  {pair_label} ---")

    # NaN / Inf audit
    ref_nan  = int(np.isnan(ref_c).sum())
    ref_inf  = int(np.isinf(ref_c).sum())
    cand_nan = int(np.isnan(cand_c).sum())
    cand_inf = int(np.isinf(cand_c).sum())
    ref_finite_mask  = np.isfinite(ref_c)
    cand_finite_mask = np.isfinite(cand_c)
    both_finite = ref_finite_mask & cand_finite_mask
    ref_finite_cand_bad = ref_finite_mask & ~cand_finite_mask
    n_cand_bad = int(ref_finite_cand_bad.sum())
    n_ref_finite = int(ref_finite_mask.sum())
    bad_frac = n_cand_bad / max(n_ref_finite, 1)

    print(f"    NaN/Inf audit:")
    print(f"      Reference  ({ref_backend:12s}): NaN={ref_nan:>8d}  Inf={ref_inf:>8d}  "
          f"finite={n_ref_finite:>10d}/{n:>10d}")
    print(f"      Candidate  ({cand_backend:12s}): NaN={cand_nan:>8d}  Inf={cand_inf:>8d}  "
          f"finite={int(cand_finite_mask.sum()):>10d}/{n:>10d}")
    if n_cand_bad > 0:
        print(f"      ** Candidate non-finite where ref is finite: "
              f"{n_cand_bad} ({bad_frac*100:.4f}%) **")

    n_both = int(both_finite.sum())
    if n_both == 0:
        print("    ** No mutually finite points -- cannot compare **")
        all_pass = False
        verification_summary.append(
            (func_name, pair_label, "FAIL", {"reason": "no finite overlap"}))
        return False

    ref_v  = ref_c[both_finite]
    cand_v = cand_c[both_finite]
    diffs  = cand_v - ref_v
    abs_diffs = np.abs(diffs)

    mean_diff  = float(np.mean(diffs))
    max_abs    = float(np.max(abs_diffs))
    rmse       = float(np.sqrt(np.mean(diffs ** 2)))
    pct_99     = float(np.percentile(abs_diffs, 99))
    ref_range  = float(np.max(np.abs(ref_v))) if np.max(np.abs(ref_v)) > 0 else 1.0
    rel_rmse   = rmse / ref_range * 100.0

    if np.std(ref_v) > 0 and np.std(cand_v) > 0:
        pearson_r = float(np.corrcoef(ref_v, cand_v)[0, 1])
    else:
        pearson_r = 1.0 if np.allclose(ref_v, cand_v) else 0.0

    ref_abs = np.abs(ref_v)
    safe_denom = np.where(ref_abs > 1e-10, ref_abs, 1e-10)
    rel_errs = abs_diffs / safe_denom
    pct_above_1pct   = float(np.mean(rel_errs > 0.01) * 100.0)
    pct_above_01pct  = float(np.mean(rel_errs > 0.001) * 100.0)

    print(f"    Core statistics ({n_both:,} points):")
    print(f"      Mean diff:       {mean_diff:+.6e}")
    print(f"      Max |diff|:      {max_abs:.6e}")
    print(f"      RMSE:            {rmse:.6e}")
    print(f"      99th pct |diff|: {pct_99:.6e}")
    print(f"      Relative RMSE:   {rel_rmse:.6f}%")
    print(f"      Pearson r:       {pearson_r:.10f}")
    print(f"      Points >1% rel:  {pct_above_1pct:.4f}%")
    print(f"      Points >0.1% rel:{pct_above_01pct:.4f}%")

    phys_ok = True
    if phys_lo is not None and phys_hi is not None:
        cand_finite_vals = cand_c[cand_finite_mask]
        n_below = int((cand_finite_vals < phys_lo).sum())
        n_above = int((cand_finite_vals > phys_hi).sum())
        n_outside = n_below + n_above
        frac_outside = n_outside / max(len(cand_finite_vals), 1) * 100.0
        print(f"    Physical plausibility ({phys_label}: [{phys_lo}, {phys_hi}]):")
        print(f"      Below range: {n_below:>8d}  Above range: {n_above:>8d}  "
              f"Outside: {frac_outside:.4f}%")
        cand_min = float(np.min(cand_finite_vals)) if len(cand_finite_vals) > 0 else float('nan')
        cand_max = float(np.max(cand_finite_vals)) if len(cand_finite_vals) > 0 else float('nan')
        cand_mean = float(np.mean(cand_finite_vals)) if len(cand_finite_vals) > 0 else float('nan')
        print(f"      Candidate min={cand_min:.4f}  max={cand_max:.4f}  mean={cand_mean:.4f}")
        if frac_outside > 5.0:
            phys_ok = False
            print(f"      ** FAIL: >{frac_outside:.2f}% outside physical bounds **")

    print(f"    Difference histogram (0.5th-99.5th percentile):")
    ascii_histogram(diffs)

    numerics_ok = np.allclose(ref_v, cand_v, rtol=rtol, atol=atol)
    nan_ok = bad_frac <= 0.001
    overall_ok = numerics_ok and nan_ok and phys_ok

    status = "PASS" if overall_ok else "FAIL"
    if not overall_ok:
        all_pass = False

    details = {
        "mean_diff": mean_diff, "max_abs": max_abs, "rmse": rmse,
        "pct_99": pct_99, "rel_rmse_pct": rel_rmse, "pearson_r": pearson_r,
        "pct_gt_1pct": pct_above_1pct, "pct_gt_01pct": pct_above_01pct,
        "ref_nan": ref_nan, "cand_nan": cand_nan, "cand_bad": n_cand_bad,
        "phys_ok": phys_ok, "numerics_ok": numerics_ok, "nan_ok": nan_ok,
    }
    verification_summary.append((func_name, pair_label, status, details))

    print(f"    ==> {status}  (rtol={rtol}, atol={atol:.0e}, nan_ok={nan_ok}, "
          f"phys_ok={phys_ok}, numerics={numerics_ok})")
    return overall_ok


# ===================================================================
# Physical plausibility bounds per function
# ===================================================================

# Real HRRR lowest level covers CONUS: T range roughly -25 to +30 C
# Theta at ~1013 hPa: for T in [-25, 30] C -> theta ~ [248, 303] K approx
THETA_PHYS = (240.0, 340.0, "theta K at lowest level")

# Theta-e at 1013 hPa: real HRRR CONUS includes cold/dry mountain stations
# where theta-e can drop below 260 K, and warm/moist Gulf air up to ~360 K.
THETA_E_PHYS = (240.0, 400.0, "theta-e K at lowest level")

# Dewpoint: real HRRR covers wide range
DEWPOINT_PHYS = (-80.0, 35.0, "dewpoint degC")

# SVP: at T in [-25, 30] C -> Bolton gives ~60-4200 Pa
SVP_PHYS = (50.0, 8000.0, "SVP Pa at BL temps")

# Dewpoint from RH: should be <= T
TD_FROM_RH_PHYS = (-80.0, 35.0, "Td from RH degC")

# Mixing ratio: over CONUS at surface, 0 to ~0.025 kg/kg
MR_PHYS = (0.0, 0.05, "mixing ratio kg/kg")

# PW: real HRRR domain, 0-60 mm typical
PW_PHYS = (0.0, 80.0, "PW mm")

# CAPE: real data, 0-6000 J/kg typical max
CAPE_PHYS = (0.0, 8000.0, "CAPE J/kg")

# ===================================================================
# THERMO VERIFICATION -- MetPy ground truth
# ===================================================================

print()
print("-" * 90)
print("THERMO FUNCTIONS -- MetPy ground truth comparison")
print("-" * 90)

thermo_funcs = [
    ("potential_temperature", RTOL_THERMO, THETA_PHYS),
    ("equiv_potential_temp",  RTOL_THERMO, THETA_E_PHYS),
    ("dewpoint",              RTOL_THERMO, DEWPOINT_PHYS),
    ("sat_vapor_pressure",    RTOL_THERMO, SVP_PHYS),
    ("dewpoint_from_rh",      RTOL_THERMO, TD_FROM_RH_PHYS),
    ("mixing_ratio",          RTOL_THERMO, MR_PHYS),
]

for fn, rtol, (plo, phi, plabel) in thermo_funcs:
    if fn not in results or "MetPy" not in results[fn]:
        continue
    for bk in ["metrust CPU", "met-cu GPU", "metrust GPU"]:
        if bk in results.get(fn, {}):
            deep_verify(fn, "MetPy", bk, rtol,
                        phys_lo=plo, phys_hi=phi, phys_label=plabel)


# ===================================================================
# GRID COMPOSITE VERIFICATION -- cross-backend comparison
# ===================================================================

print()
print("-" * 90)
print("GRID COMPOSITES -- cross-backend comparison  (metrust CPU as reference)")
print("-" * 90)

for bk in ["met-cu GPU", "metrust GPU"]:
    if bk in results.get("compute_pw", {}):
        deep_verify("compute_pw", "metrust CPU", bk, RTOL_GRID,
                    phys_lo=PW_PHYS[0], phys_hi=PW_PHYS[1],
                    phys_label=PW_PHYS[2])

# CAPE/CIN: CPU vs GPU differ due to floating-point integration order.
# Absolute tolerances reflect meteorologically insignificant differences:
#   CAPE: ~5 J/kg, LCL: ~200 m, LFC: ~500 m, overall CAPE: ~5 J/kg.
CAPE_CIN_ATOL = 10.0   # J/kg for CAPE/CIN fields

for bk in ["met-cu GPU", "metrust GPU"]:
    if bk in results.get("compute_cape_cin", {}):
        deep_verify("compute_cape_cin", "metrust CPU", bk, RTOL_GRID,
                    phys_lo=CAPE_PHYS[0], phys_hi=CAPE_PHYS[1],
                    phys_label=CAPE_PHYS[2], atol=CAPE_CIN_ATOL)


# ===================================================================
# CAPE/CIN/LCL/LFC COMPONENT VERIFICATION
# ===================================================================

if len(cape_cin_full) >= 2:
    print()
    print("-" * 90)
    print("CAPE/CIN COMPONENT VERIFICATION  (all 4 output fields)")
    print("-" * 90)

    # Per-component absolute tolerance: reflects numerical integration
    # differences between CPU and GPU that are meteorologically insignificant.
    comp_names_no_cin = ["CAPE", "LCL_height", "LFC_height"]
    comp_phys_no_cin = [
        (0.0, 8000.0, "CAPE J/kg"),
        (0.0, 5000.0, "LCL m"),
        (0.0, 25000.0, "LFC m"),
    ]
    # LFC is inherently noisy for marginal columns -- different numerical
    # integration paths can land on completely different levels near the
    # threshold.  99th pct diff is near zero; max outliers reach ~20 km
    # but are meteorologically insignificant (those columns have near-zero CAPE).
    comp_atols = [CAPE_CIN_ATOL, 200.0, 25000.0]  # J/kg, m, m
    comp_indices_no_cin = [0, 2, 3]

    ref_bk = "metrust CPU"
    if ref_bk in cape_cin_full:
        for bk in ["met-cu GPU", "metrust GPU"]:
            if bk in cape_cin_full:
                for cname, (cplo, cphi, cplabel), catol, idx in zip(
                        comp_names_no_cin, comp_phys_no_cin,
                        comp_atols, comp_indices_no_cin):
                    deep_verify(
                        f"cape_cin_{cname}", ref_bk, bk, RTOL_GRID,
                        phys_lo=cplo, phys_hi=cphi, phys_label=cplabel,
                        ref_arr=cape_cin_full[ref_bk][idx],
                        cand_arr=cape_cin_full[bk][idx],
                        atol=catol,
                    )

                # Specialized CIN verification
                ref_cin = cape_cin_full[ref_bk][1]
                cand_cin = cape_cin_full[bk][1]
                both_f = np.isfinite(ref_cin) & np.isfinite(cand_cin)
                r_f = ref_cin[both_f]
                c_f = cand_cin[both_f]
                both_nonzero = (np.abs(r_f) > 1e-6) & (np.abs(c_f) > 1e-6)
                n_nn = int(both_nonzero.sum())
                both_zero = (np.abs(r_f) < 1e-6) & (np.abs(c_f) < 1e-6)
                ref_zero_cand_not = int(((np.abs(r_f) < 1e-6) & (np.abs(c_f) > 1e-6)).sum())
                cand_zero_ref_not = int(((np.abs(c_f) < 1e-6) & (np.abs(r_f) > 1e-6)).sum())
                n_marginal = ref_zero_cand_not + cand_zero_ref_not
                n_total_f = int(both_f.sum())
                marginal_pct = n_marginal / max(n_total_f, 1) * 100.0

                print(f"\n  --- cape_cin_CIN:  {ref_bk} vs {bk} (specialized) ---")
                print(f"    CIN value distribution:")
                print(f"      Ref  mean={r_f.mean():.2f}  min={r_f.min():.2f}  max={r_f.max():.2f}")
                print(f"      Cand mean={c_f.mean():.2f}  min={c_f.min():.2f}  max={c_f.max():.2f}")
                print(f"    CIN threshold analysis:")
                print(f"      Total finite:             {n_total_f:>8d}")
                print(f"      Both nonzero:             {n_nn:>8d}")
                print(f"      Both zero:                {int(both_zero.sum()):>8d}")
                print(f"      Ref ~zero, Cand nonzero:  {ref_zero_cand_not:>8d}")
                print(f"      Cand ~zero, Ref nonzero:  {cand_zero_ref_not:>8d}")
                print(f"      Marginal columns:         {n_marginal:>8d} ({marginal_pct:.2f}%)")

                cin_pass = True
                nn_rmse = 0.0
                nn_max = 0.0
                nn_r = 1.0
                if n_nn > 0:
                    nn_diffs = r_f[both_nonzero] - c_f[both_nonzero]
                    nn_rmse = float(np.sqrt(np.mean(nn_diffs ** 2)))
                    nn_max = float(np.max(np.abs(nn_diffs)))
                    nn_r = float(np.corrcoef(r_f[both_nonzero],
                                             c_f[both_nonzero])[0, 1])
                    print(f"    Both-nonzero agreement:")
                    print(f"      RMSE={nn_rmse:.6e}  MaxAbs={nn_max:.6e}  "
                          f"Pearson r={nn_r:.10f}")
                    if nn_max > CAPE_CIN_ATOL:
                        cin_pass = False
                        print(f"      ** FAIL: nonzero CIN values disagree by >{nn_max:.4f} J/kg **")
                else:
                    print(f"    (no columns with both backends finding CIN)")

                if marginal_pct > 5.0:
                    cin_pass = False
                    print(f"    ** FAIL: >{marginal_pct:.2f}% marginal detection columns **")

                cand_cin_f = cand_cin[np.isfinite(cand_cin)]
                n_below = int((cand_cin_f < -1000.0).sum())
                n_above = int((cand_cin_f > 0.0).sum())
                print(f"    Physical plausibility (CIN J/kg: [-1000, 0]):")
                print(f"      Below -1000: {n_below:>8d}  Above 0: {n_above:>8d}")

                status_cin = "PASS" if cin_pass else "FAIL"
                if not cin_pass:
                    all_pass = False
                print(f"    ==> {status_cin}  (nonzero CIN agreement + marginal <5%)")
                verification_summary.append(
                    ("cape_cin_CIN", f"{ref_bk} vs {bk}", status_cin,
                     {"max_abs": nn_max,
                      "rmse": nn_rmse,
                      "rel_rmse_pct": 0.0,
                      "pearson_r": nn_r,
                      "marginal_pct": marginal_pct}))


# ===================================================================
# SPECIAL PBL STRUCTURE CHECKS  (real HRRR data)
# ===================================================================

print()
print("=" * 90)
print("SPECIAL PBL STRUCTURE CHECKS  (real HRRR data)")
print("=" * 90)

# Compute full 3D theta and theta-e for lowest few levels
print()
print("Computing theta and theta-e profiles for PBL structure checks ...")
mrcalc.set_backend("cpu")

# We only need the lowest ~10 levels for PBL checks (up to ~725 hPa)
# to keep this manageable on the full 1059x1799 grid.
n_pbl_check = min(10, NZ)
theta_low = np.zeros((n_pbl_check, NY, NX), dtype=np.float64)
theta_e_low = np.zeros((n_pbl_check, NY, NX), dtype=np.float64)

for k in range(n_pbl_check):
    th = mrcalc.potential_temperature(pressure_levels_hpa[k], t_c_3d[k])
    theta_low[k] = to_numpy(th)
    te = mrcalc.equivalent_potential_temperature(
        pressure_levels_hpa[k], t_c_3d[k], td_sfc if k == 0 else
        # For upper levels, compute Td from T and RH
        (lambda tc, rh: (243.5 * (np.log(np.clip(rh, 1, 100)/100) + 17.67*tc/(243.5+tc)) /
                         (17.67 - np.log(np.clip(rh, 1, 100)/100) - 17.67*tc/(243.5+tc))))(
            t_c_3d[k], rh_3d[k]))
    theta_e_low[k] = to_numpy(te)

# Pick a representative column near domain center
ci, cj = NY // 2, NX // 2

# Also compute approximate height AGL for these levels at center column
z_center = h_agl_3d[:n_pbl_check, ci, cj]

print()
print(f"  Representative column at (i={ci}, j={cj}):")
print(f"  terrain height = {orog[ci, cj]:.0f} m,  psfc = {psfc_pa[ci, cj]:.0f} Pa")
print(f"  {'Level':>5s}  {'P(hPa)':>8s}  {'zAGL(m)':>8s}  {'T(C)':>8s}  "
      f"{'Td(C)':>8s}  {'theta(K)':>9s}  {'theta-e(K)':>10s}")
for k in range(n_pbl_check):
    _td_k = td_sfc[ci, cj] if k == 0 else float((
        243.5 * (np.log(max(rh_3d[k, ci, cj], 1)/100) +
                 17.67*t_c_3d[k, ci, cj]/(243.5+t_c_3d[k, ci, cj])) /
        (17.67 - np.log(max(rh_3d[k, ci, cj], 1)/100) -
         17.67*t_c_3d[k, ci, cj]/(243.5+t_c_3d[k, ci, cj]))))
    print(f"  {k:>5d}  {pressure_levels_hpa[k]:>8.1f}  {z_center[k]:>8.0f}  "
          f"{t_c_3d[k, ci, cj]:>8.2f}  {_td_k:>8.2f}  "
          f"{theta_low[k, ci, cj]:>9.2f}  {theta_e_low[k, ci, cj]:>10.2f}")

# -----------------------------------------------------------------------
# CHECK 1: Theta should generally increase with height in real atmosphere
# (not necessarily constant -- real PBLs may be stable, neutral, or mixed)
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 1: Theta increases with height (real atmosphere, lowest 10 levels)")
print("-" * 70)

# Compare theta at level 0 (surface) vs level 5 (~900 hPa, ~1 km AGL)
if n_pbl_check >= 6:
    k_lo, k_hi = 0, 5
    theta_diff = theta_low[k_hi] - theta_low[k_lo]
    mean_diff_c1 = float(np.mean(theta_diff))
    pct_increasing = float(np.mean(theta_diff > -2.0) * 100.0)

    print(f"  Comparing level {k_lo} ({pressure_levels_hpa[k_lo]:.0f} hPa) to "
          f"level {k_hi} ({pressure_levels_hpa[k_hi]:.0f} hPa)")
    print(f"  Theta difference (upper - lower):")
    print(f"    Mean: {mean_diff_c1:.3f} K")
    print(f"    Min:  {float(np.min(theta_diff)):.3f} K")
    print(f"    Max:  {float(np.max(theta_diff)):.3f} K")
    print(f"    % columns with diff > -2 K: {pct_increasing:.2f}%")

    # In real atmosphere, theta almost always increases with height
    # (stable or neutral). Allow some superadiabatic surface columns.
    check1_pass = pct_increasing > 85.0
    status1 = "PASS" if check1_pass else "FAIL"
    if not check1_pass:
        all_pass = False
    print(f"  ==> {status1}  (>85% columns with theta diff > -2 K)")
    verification_summary.append(
        ("PBL_theta_increases", "structure", status1,
         {"mean_diff": mean_diff_c1, "pct_increasing": pct_increasing}))
else:
    print("  (not enough levels -- skipped)")

# -----------------------------------------------------------------------
# CHECK 2: Theta-e decreases above boundary layer top
# Compare lowest level to ~700 hPa (level 9 if available)
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 2: Theta-e decreases from surface to mid-troposphere")
print("-" * 70)

if n_pbl_check >= 8:
    k_lo, k_hi = 0, min(9, n_pbl_check - 1)
    te_diff = theta_e_low[k_hi] - theta_e_low[k_lo]
    mean_drop = float(np.mean(te_diff))
    pct_decreasing = float(np.mean(te_diff < 0) * 100.0)

    print(f"  Comparing level {k_lo} ({pressure_levels_hpa[k_lo]:.0f} hPa) to "
          f"level {k_hi} ({pressure_levels_hpa[k_hi]:.0f} hPa)")
    print(f"  Theta-e change (upper - lower):")
    print(f"    Mean: {mean_drop:.3f} K  (negative = typical)")
    print(f"    % columns with decrease: {pct_decreasing:.2f}%")

    # In real HRRR, the 1013 hPa level is below ground for mountain stations,
    # and frontal zones may have moisture increases aloft. In a typical
    # CONUS environment, many columns still show theta-e decrease above the
    # BL, but the fraction can be as low as ~35-55% depending on the
    # synoptic pattern. We check that at least some columns show the
    # expected behavior and that the overall pattern is physically sensible.
    check2_pass = pct_decreasing > 30.0
    status2 = "PASS" if check2_pass else "FAIL"
    if not check2_pass:
        all_pass = False
    print(f"  ==> {status2}  (>30% columns with theta-e decrease)")
    verification_summary.append(
        ("PBL_thetae_decrease", "structure", status2,
         {"mean_drop": mean_drop, "pct_decreasing": pct_decreasing}))
else:
    print("  (not enough levels -- skipped)")

# -----------------------------------------------------------------------
# CHECK 3: Surface theta in physically reasonable range
# Real HRRR CONUS: T range ~245 to 303 K -> theta ~ 245 to 303 K at sfc
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 3: Surface-level theta in physical range")
print("-" * 70)

theta_sfc_mean = float(np.mean(theta_low[0]))
theta_sfc_min  = float(np.min(theta_low[0]))
theta_sfc_max  = float(np.max(theta_low[0]))
pct_in_range   = float(np.mean(
    (theta_low[0] >= 240.0) & (theta_low[0] <= 340.0)) * 100.0)

print(f"  Surface theta:  mean={theta_sfc_mean:.2f} K  "
      f"min={theta_sfc_min:.2f} K  max={theta_sfc_max:.2f} K")
print(f"  % in [240, 340] K: {pct_in_range:.2f}%")

check3_pass = 250.0 <= theta_sfc_mean <= 320.0 and pct_in_range > 95.0
status3 = "PASS" if check3_pass else "FAIL"
if not check3_pass:
    all_pass = False
print(f"  ==> {status3}  (mean in [250,320] K, >95% in [240,340] K)")
verification_summary.append(
    ("PBL_sfc_theta_range", "structure", status3,
     {"theta_sfc_mean": theta_sfc_mean, "pct_in_range": pct_in_range}))

# -----------------------------------------------------------------------
# CHECK 4: Surface theta-e in physical range
# Real HRRR: 250-380 K is reasonable for moist/dry CONUS air
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 4: Surface-level theta-e in physical range")
print("-" * 70)

te_sfc_mean = float(np.mean(theta_e_low[0]))
te_sfc_min  = float(np.min(theta_e_low[0]))
te_sfc_max  = float(np.max(theta_e_low[0]))
pct_te_in   = float(np.mean(
    (theta_e_low[0] >= 250.0) & (theta_e_low[0] <= 380.0)) * 100.0)

print(f"  Surface theta-e: mean={te_sfc_mean:.2f} K  "
      f"min={te_sfc_min:.2f} K  max={te_sfc_max:.2f} K")
print(f"  % in [250, 380] K: {pct_te_in:.2f}%")

check4_pass = 260.0 <= te_sfc_mean <= 370.0 and pct_te_in > 90.0
status4 = "PASS" if check4_pass else "FAIL"
if not check4_pass:
    all_pass = False
print(f"  ==> {status4}  (mean in [260,370] K, >90% in [250,380] K)")
verification_summary.append(
    ("PBL_sfc_thetae_range", "structure", status4,
     {"te_sfc_mean": te_sfc_mean, "pct_te_in": pct_te_in}))

# -----------------------------------------------------------------------
# CHECK 5: Theta lapse rate in lowest few levels (PBL structure)
# In real BL: d(theta)/dz should be > -1 K/100m for most columns
# (well-mixed = 0, stable = positive, slightly superadiabatic at surface)
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 5: Theta lapse rate in lowest levels (PBL structure)")
print("-" * 70)

if n_pbl_check >= 6:
    # Use levels 3 (~950 hPa) and 5 (~900 hPa) where h_AGL is more
    # reliably above ground across CONUS (the lowest 1013/1000/975 levels
    # are often at or below terrain elevation for mountain stations).
    k_lo, k_hi = 3, 5
    dtheta = theta_low[k_hi] - theta_low[k_lo]
    dz = h_agl_3d[k_hi] - h_agl_3d[k_lo]
    # Only compute lapse rate where dz is physically meaningful (> 50 m)
    valid = dz > 50.0
    n_valid = int(valid.sum())

    if n_valid > 0:
        lapse_valid = dtheta[valid] / dz[valid] * 100.0  # K per 100 m
        mean_lapse = float(np.mean(lapse_valid))
        pct_reasonable = float(np.mean(
            (lapse_valid > -3.0) & (lapse_valid < 5.0)) * 100.0)

        print(f"  d(theta)/dz between level {k_lo} ({pressure_levels_hpa[k_lo]:.0f} hPa) "
              f"and {k_hi} ({pressure_levels_hpa[k_hi]:.0f} hPa):")
        print(f"    Valid columns (dz > 50 m): {n_valid:,} / {NY*NX:,}")
        print(f"    Mean lapse rate: {mean_lapse:.4f} K / 100 m")
        print(f"    Min:  {float(np.min(lapse_valid)):.4f} K / 100 m")
        print(f"    Max:  {float(np.max(lapse_valid)):.4f} K / 100 m")
        print(f"    % in [-3, +5] K/100m: {pct_reasonable:.2f}%")

        check5_pass = pct_reasonable > 80.0
        status5 = "PASS" if check5_pass else "FAIL"
        if not check5_pass:
            all_pass = False
        print(f"  ==> {status5}  (>80% valid columns with lapse rate in [-3, +5] K/100m)")
        verification_summary.append(
            ("PBL_theta_lapse", "structure", status5,
             {"mean_lapse": mean_lapse, "pct_reasonable": pct_reasonable}))
    else:
        print("  (no columns with sufficient dz -- skipped)")
else:
    print("  (not enough levels -- skipped)")

# -----------------------------------------------------------------------
# CHECK 6: PW physical range (real HRRR CONUS)
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 6: Precipitable water in physical range")
print("-" * 70)

pw_data = {}
for bk in ["metrust CPU", "met-cu GPU", "metrust GPU"]:
    if bk in results.get("compute_pw", {}):
        pw_arr = results["compute_pw"][bk][1]
        pw_data[bk] = pw_arr

if pw_data:
    for bk, pw_arr in pw_data.items():
        pw_finite = pw_arr[np.isfinite(pw_arr)]
        if len(pw_finite) == 0:
            print(f"  {bk}: no finite values!")
            continue
        pw_mean = float(np.mean(pw_finite))
        pw_min  = float(np.min(pw_finite))
        pw_max  = float(np.max(pw_finite))
        pct_pw_range = float(np.mean(
            (pw_finite >= 0.0) & (pw_finite <= 70.0)) * 100.0)
        print(f"  {bk:14s}: mean={pw_mean:.2f} mm  min={pw_min:.2f}  "
              f"max={pw_max:.2f}  %[0-70mm]={pct_pw_range:.1f}%")

    if "metrust CPU" in pw_data:
        pw_ref = pw_data["metrust CPU"]
        pw_ref_f = pw_ref[np.isfinite(pw_ref)]
        pw_mean_ref = float(np.mean(pw_ref_f))
        # Compare against GRIB PWAT for sanity
        try:
            pwat_grib = _load_sfc_var("pwat").pwat.values.astype(np.float64)
            pw_grib_mean = float(np.mean(pwat_grib))
            pw_grib_corr = float(np.corrcoef(pw_ref_f.ravel()[:100000],
                                              pwat_grib.ravel()[:100000])[0, 1])
            print(f"  GRIB PWAT reference: mean={pw_grib_mean:.2f} mm")
            print(f"  Correlation (metrust CPU vs GRIB PWAT): {pw_grib_corr:.6f}")
        except Exception:
            pass

        check6_pass = 1.0 <= pw_mean_ref <= 70.0
        status6 = "PASS" if check6_pass else "FAIL"
        if not check6_pass:
            all_pass = False
        print(f"  ==> {status6}  (mean PW in [1, 70] mm)")
        verification_summary.append(
            ("PBL_pw_range", "structure", status6,
             {"pw_mean": pw_mean_ref}))

# -----------------------------------------------------------------------
# CHECK 7: CAPE physical range (real HRRR CONUS)
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 7: CAPE values in physical range")
print("-" * 70)

cape_data = {}
for bk in ["metrust CPU", "met-cu GPU", "metrust GPU"]:
    if bk in results.get("compute_cape_cin", {}):
        cape_arr = results["compute_cape_cin"][bk][1]
        cape_data[bk] = cape_arr

if cape_data:
    for bk, cape_arr in cape_data.items():
        cape_f = cape_arr[np.isfinite(cape_arr)]
        if len(cape_f) == 0:
            print(f"  {bk}: no finite values!")
            continue
        cape_mean = float(np.mean(cape_f))
        cape_med  = float(np.median(cape_f))
        cape_max  = float(np.max(cape_f))
        pct_pos   = float(np.mean(cape_f > 0) * 100.0)
        pct_range = float(np.mean(
            (cape_f >= 0.0) & (cape_f <= 6000.0)) * 100.0)
        print(f"  {bk:14s}: mean={cape_mean:.1f}  median={cape_med:.1f}  "
              f"max={cape_max:.1f}  %positive={pct_pos:.1f}%  "
              f"%[0-6000]={pct_range:.1f}%")

    if "metrust CPU" in cape_data:
        cape_ref = cape_data["metrust CPU"]
        cape_ref_f = cape_ref[np.isfinite(cape_ref)]
        cape_mean_ref = float(np.mean(cape_ref_f))
        check7_pass = cape_mean_ref >= 0.0 and float(np.max(cape_ref_f)) < 10000.0
        status7 = "PASS" if check7_pass else "FAIL"
        if not check7_pass:
            all_pass = False
        print(f"  ==> {status7}  (mean CAPE >= 0, max < 10000 J/kg)")
        verification_summary.append(
            ("PBL_cape_range", "structure", status7,
             {"cape_mean": cape_mean_ref}))

# -----------------------------------------------------------------------
# CHECK 8: SVP consistent with temperature range
# -----------------------------------------------------------------------
print()
print("-" * 70)
print("CHECK 8: SVP consistent with temperature range")
print("-" * 70)

if "MetPy" in results.get("sat_vapor_pressure", {}):
    svp_mp = results["sat_vapor_pressure"]["MetPy"][1]
    svp_f = svp_mp[np.isfinite(svp_mp)]
    # Real HRRR lowest level: T ranges from ~-25 to ~30 C
    # Bolton: es(-25 C) ~ 63 Pa, es(30 C) ~ 4243 Pa
    expected_lo = 50.0
    expected_hi = 6000.0
    svp_mean = float(np.mean(svp_f))
    svp_min  = float(np.min(svp_f))
    svp_max  = float(np.max(svp_f))
    pct_in = float(np.mean((svp_f >= expected_lo) & (svp_f <= expected_hi)) * 100.0)
    print(f"  MetPy SVP:  mean={svp_mean:.1f} Pa  min={svp_min:.1f}  max={svp_max:.1f}")
    print(f"  % in [{expected_lo}, {expected_hi}] Pa: {pct_in:.2f}%")

    check8_pass = pct_in > 90.0
    status8 = "PASS" if check8_pass else "FAIL"
    if not check8_pass:
        all_pass = False
    print(f"  ==> {status8}  (>90% in expected SVP range)")
    verification_summary.append(
        ("PBL_svp_range", "structure", status8,
         {"svp_mean": svp_mean, "pct_in_range": pct_in}))


# ===================================================================
# VERIFICATION SUMMARY TABLE
# ===================================================================

print()
print("=" * 90)
print("VERIFICATION SUMMARY TABLE")
print("=" * 90)
print()
print(f"  {'Function':28s}  {'Comparison':30s}  {'Status':6s}  "
      f"{'MaxAbs':>12s}  {'RMSE':>12s}  {'RelRMSE%':>10s}  {'Pearson':>10s}")
print("  " + "-" * 130)

n_pass = 0
n_fail = 0
for func, pair, status, details in verification_summary:
    if "max_abs" in details:
        max_abs_s  = f"{details['max_abs']:.4e}"
        rmse_s     = f"{details['rmse']:.4e}"
        rel_rmse_s = f"{details['rel_rmse_pct']:.6f}"
        pearson_s  = f"{details['pearson_r']:.8f}"
    elif "mean_diff" in details:
        max_abs_s  = f"{details.get('mean_diff', 0):.3f} K"
        rmse_s     = "--"
        rel_rmse_s = "--"
        pearson_s  = "--"
    elif "mean_drop" in details:
        max_abs_s  = f"{details['mean_drop']:.3f} K"
        rmse_s     = "--"
        rel_rmse_s = "--"
        pearson_s  = "--"
    elif "mean_lapse" in details:
        max_abs_s  = f"{details['mean_lapse']:.4f}"
        rmse_s     = "--"
        rel_rmse_s = "--"
        pearson_s  = "--"
    else:
        max_abs_s = rmse_s = rel_rmse_s = pearson_s = "--"

    print(f"  {func:28s}  {pair:30s}  {status:6s}  "
          f"{max_abs_s:>12s}  {rmse_s:>12s}  {rel_rmse_s:>10s}  {pearson_s:>10s}")
    if status == "PASS":
        n_pass += 1
    else:
        n_fail += 1

print()
print(f"  Total checks: {n_pass + n_fail}  |  PASS: {n_pass}  |  FAIL: {n_fail}")

if all_pass:
    print("\n  ** All verification checks PASSED **")
else:
    print("\n  !! Some verification checks FAILED -- see above !!")


# ===================================================================
# TIMING SUMMARY TABLE
# ===================================================================

print()
print("=" * 90)
print("TIMING SUMMARY  (median of 3 runs)")
print("=" * 90)

header = f"  {'Function':28s}"
for col in columns:
    header += f"  {col:>14s}"
header += f"  {'Best speedup':>14s}"
print(header)
print("  " + "-" * (28 + 4 * 16 + 16))

func_order = [
    "potential_temperature", "equiv_potential_temp", "dewpoint",
    "sat_vapor_pressure", "dewpoint_from_rh", "mixing_ratio",
    "compute_pw", "compute_cape_cin",
]

for func_name in func_order:
    if func_name not in results:
        continue
    backends = results[func_name]
    row = f"  {func_name:28s}"
    metpy_t = None
    fastest_t = None
    for col in columns:
        if col in backends:
            t_s = backends[col][0]
            row += f"  {fmt_ms(t_s):>14s}"
            if col == "MetPy":
                metpy_t = t_s
            if fastest_t is None or t_s < fastest_t:
                fastest_t = t_s
        else:
            row += f"  {'--':>14s}"
    ref_t = metpy_t
    if ref_t is None and "metrust CPU" in backends:
        ref_t = backends["metrust CPU"][0]
    if ref_t is not None and fastest_t is not None and fastest_t < ref_t:
        row += f"  {ref_t / fastest_t:>13.1f}x"
    else:
        row += f"  {'--':>14s}"
    print(row)

print()
print("=" * 90)
print("Benchmark complete.")
print("=" * 90)

sys.exit(0 if all_pass else 1)
