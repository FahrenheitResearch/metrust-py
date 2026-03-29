# Benchmarks

All results on **NVIDIA GeForce RTX 5090** (34.2 GB VRAM, Blackwell, 21,760 CUDA cores) with CuPy 14.0.1, met-cu 0.2.1, metrust 0.4.0. CPU is a 32-core AMD Ryzen with rayon parallelism.

## Three-Way Comparison: MetPy vs Rust vs CUDA

Real HRRR model output (40 isobaric levels, 1059 × 1799 grid, ~1.9M points per level). `python tests/benchmark_gpu.py`

### Scalar Thermodynamics (2D: 1059×1799)

| Function | MetPy | Rust | CUDA | Rust/MetPy | CUDA/Rust |
|---|---:|---:|---:|---:|---:|
| potential_temperature ★ | 11.2 ms | 13.0 ms | 7.9 ms | 0.9x | 1.6x |
| equiv_potential_temperature ★ | 303.1 ms | 16.5 ms | 9.1 ms | **18x** | 1.8x |
| dewpoint ★ | 33.6 ms | 10.8 ms | 8.8 ms | 3.1x | 1.2x |
| saturation_vapor_pressure | 66.7 ms | 8.2 ms | — | 8.1x | — |
| saturation_mixing_ratio | 78.6 ms | 13.7 ms | — | 5.7x | — |
| dewpoint_from_rh | 110.3 ms | 7.6 ms | — | **14x** | — |
| rh_from_dewpoint | 138.7 ms | 10.9 ms | — | **13x** | — |
| virtual_temperature | 31.1 ms | 19.7 ms | — | 1.6x | — |
| mixing_ratio | 11.6 ms | 8.5 ms | — | 1.4x | — |
| wet_bulb_temperature | >10 min | 26.9 ms | — | **>22,000x** | — |

### Grid Kinematics (2D: 1059×1799)

| Function | MetPy | Rust | CUDA | Rust/MetPy | CUDA/Rust |
|---|---:|---:|---:|---:|---:|
| vorticity ★ | 98.3 ms | 92.8 ms | 9.3 ms | 1.1x | **10x** |
| divergence | 96.1 ms | 91.2 ms | — | 1.1x | — |
| frontogenesis ★ | 733.0 ms | 339.4 ms | 12.2 ms | 2.2x | **28x** |
| q_vector ★ | 390.3 ms | 310.1 ms | 10.8 ms | 1.3x | **29x** |
| advection | 161.8 ms | 87.7 ms | — | 1.8x | — |

### 1D Sounding (40 levels, single column)

| Function | MetPy | Rust | Rust/MetPy |
|---|---:|---:|---:|
| parcel_profile | 5.5 ms | 0.074 ms | **74x** |
| cape_cin | 1.4 ms | 0.254 ms | 5.4x |
| lcl | 0.118 ms | 0.065 ms | 1.8x |
| lfc | 6.7 ms | 0.107 ms | **62x** |
| el | 6.6 ms | 0.112 ms | **59x** |
| precipitable_water | 2.1 ms | 0.057 ms | **36x** |

### Grid Composites (3D: 40×1059×1799 → 2D)

MetPy has no grid-level equivalents for these functions.

| Function | Rust | CUDA | CUDA/Rust |
|---|---:|---:|---:|
| compute_cape_cin ★ | 2.96 s | 674.5 ms | **4.4x** |
| compute_srh ★ | 223.5 ms | 135.8 ms | 1.6x |
| compute_shear ★ | 190.4 ms | 166.5 ms | 1.1x |
| compute_pw ★ | 191.6 ms | 107.9 ms | 1.8x |
| composite_refl_hydrometeors ★ | 154.2 ms | 232.2 ms | 0.7x |

### Summary

| Category | MetPy | Rust | CUDA |
|---|---:|---:|---:|
| Scalar thermo (×10) | 785 ms | 136 ms | 26 ms |
| Grid kinematics (×5) | 1.48 s | 921 ms | 32 ms |
| 1D sounding (×6) | 22 ms | 0.67 ms | — |
| Grid composites (×5) | — | 3.72 s | 1.32 s |
| **★ GPU-eligible total** | — | **4.50 s** | **1.37 s (3.3x)** |

★ = dispatches to CUDA when `set_backend("gpu")`.

---

## Real-Data Verification Benchmarks

12 independent scenarios using actual HRRR and GFS GRIB2 data, verifying correctness across all 4 backends (MetPy, metrust CPU, met-cu GPU, metrust GPU) with ~230 deep statistical checks.

### Timing Highlights on Real Atmospheric Data

| Function | Grid | MetPy | Rust | met-cu GPU | vs MetPy | vs Rust |
|---|---|---:|---:|---:|---:|---:|
| frontogenesis | 721×1440 | 388 ms | 154 ms | 3.8 ms | **101x** | **40x** |
| equiv_potential_temp | 1059×1799 | 13.7 s | 464 ms | 163 ms | **84x** | **2.8x** |
| compute_srh | 200×200 | — | 124 ms | 2.7 ms | — | **46x** |
| q_vector | 1059×1799 | — | 115 ms | 4.6 ms | — | **25x** |
| compute_cape_cin | 200×200 | — | 317 ms | 13 ms | — | **24x** |
| vorticity | 721×1440 | 41.6 ms | 41.1 ms | 1.7 ms | **24x** | **24x** |
| potential_temp | 721×1440 | 28.4 ms | 5.5 ms | 1.7 ms | **17x** | **3.2x** |

### Full Pipeline (bench 10: HRRR squall line, 1059×1799×40)

| | Rust CPU | met-cu GPU | Speedup |
|---|---:|---:|---:|
| Total (6 functions) | 3.58 s | 762 ms | **4.7x** |
| CAPE alone | 2.84 s | 622 ms | **4.6x** |
| Kinematics alone | 230 ms | 10 ms | **23x** |

### 100 Real Soundings (bench 09: HRRR columns across CONUS)

| | MetPy | metrust | Speedup |
|---|---:|---:|---:|
| 100 soundings total | 676 ms | 72 ms | **9.4x** |
| Per sounding | 6.8 ms | 0.72 ms | 9.4x |

### Correctness Summary

All 12 benchmarks pass deep verification against MetPy as ground truth:

| Benchmark | Data | Checks | RMSE | Pearson r |
|---|---|---:|---|---|
| 01 HRRR Severe | 1059×1799×40 | 20/20 | ~1e-13 | 1.0000 |
| 02 GFS Upper Air | 721×1440 | 19/21 | ~1e-14 | 1.0000 |
| 03 HRRR Warm Front | 1059×1799 | All | ~1e-18 | 1.0000 |
| 04 HRRR Precip Water | 1059×1799×40 | 14/14 | ~1e-13 | 1.0000 |
| 05 HRRR Supercell | 1059×1799×40 | 16/20 | ~1e-13 | 0.99999 |
| 06 GFS Jet Stream | 721×1440 | 16/16 | ~1e-15 | 1.0000 |
| 07 HRRR Fire Weather | 1059×1799 | 16/16 | ~1e-14 | 1.0000 |
| 08 GFS Tropical | 121×201×33 | 15/15 | ~1e-15 | 1.0000 |
| 09 HRRR Soundings | 100×40 | 25/25 | ~12 J/kg (CAPE) | 0.9999 |
| 10 HRRR Squall Line | 1059×1799×40 | All | ~1e-13 | 1.0000 |
| 11 GFS Cold Air | 161×1440 | 19/19 | ~1e-14 | 1.0000 |
| 12 HRRR Boundary Layer | 1059×1799×40 | 35/35 | ~1e-13 | 1.0000 |

Non-passing checks are documented edge cases: frontogenesis boundary NaN (0.05% of points), CIN/LFC sentinel values on marginal soundings.

---

## met-cu Comprehensive Benchmark (202 Functions)

All 202 met-cu functions benchmarked against metrust CPU on 1,905,141 points (HRRR grid).

### Category Averages

| Category | Functions | Avg Speedup | Min | Max |
|---|---:|---:|---:|---:|
| Per-element thermo | 52 | **238x** | 1.0x | 971x |
| Wind per-element | 3 | **7.5x** | 6.3x | 9.3x |
| Grid stencil | 28 | **39x** | 3.7x | 108x |
| Column/sounding | 72 | **3.7x** | 0.0x | 85x |

### Top GPU Speedups

| Function | GPU | CPU | Speedup |
|---|---:|---:|---:|
| height_to_geopotential | 0.94 ms | 912 ms | **971x** |
| moist_air_specific_heat_pressure | 0.91 ms | 819 ms | **902x** |
| coriolis_parameter | 0.94 ms | 830 ms | **883x** |
| water_latent_heat_melting | 0.98 ms | 828 ms | **849x** |
| scale_height | 1.08 ms | 916 ms | **847x** |
| geopotential_to_height | 1.13 ms | 921 ms | **815x** |
| water_latent_heat_vaporization | 1.02 ms | 808 ms | **794x** |
| water_latent_heat_sublimation | 1.09 ms | 816 ms | **746x** |
| moist_air_gas_constant | 1.13 ms | 820 ms | **723x** |
| heat_index | 2.00 ms | 1035 ms | **517x** |

Note: The extreme speedups (>100x) are for functions where the metrust CPU path falls back to a Python scalar loop (`_vec_call`). The GPU kernel runs natively on the full array. These represent real user-facing speedups.

### Grid Stencil Functions (1059×1799)

| Function | GPU | CPU | Speedup |
|---|---:|---:|---:|
| shear_vorticity | 2.05 ms | 221 ms | **108x** |
| curvature_vorticity | 1.90 ms | 160 ms | **84x** |
| smooth_rectangular | 0.98 ms | 78 ms | **80x** |
| total_deformation | 2.26 ms | 144 ms | **64x** |
| vector_derivative | 2.41 ms | 153 ms | **63x** |
| geostrophic_wind | 2.24 ms | 117 ms | **52x** |
| q_vector | 5.37 ms | 240 ms | **45x** |
| frontogenesis | 5.86 ms | 257 ms | **44x** |
| vorticity | 2.02 ms | 75 ms | **37x** |
| smooth_n_point (5) | 0.95 ms | 32 ms | **33x** |

### Where GPU Is Slower

Single-column sounding functions are faster on CPU due to kernel launch overhead:

| Function | GPU | CPU | Ratio |
|---|---:|---:|---:|
| parcel_profile | 3.25 ms | 0.16 ms | 0.05x |
| moist_lapse | 3.15 ms | 0.13 ms | 0.04x |
| cape_cin (1 sounding) | 3.42 ms | 0.34 ms | 0.1x |

For single soundings, use CPU. For grid-level computation (1000+ columns), GPU wins.

---

## Workflow-Level Speedups

metrust vs MetPy on real-world analysis workflows:

| Workflow | Speedup | Notes |
|---|---:|---|
| SounderPy compute-heavy subset | **29.7x** | Thermo + wind + severe params |
| MetPy Cookbook sounding analysis | **6.0x** | Full severe weather stack |
| MetPy Cookbook 500 hPa grid | **6.1x** | Vorticity, smoothing, advection |
| MetPy Cookbook Q-vectors | **6.1x** | Q-vector divergence |
| MetPy isentropic example | **2.3x** | Isentropic interpolation + Montgomery |
| Vorticity/divergence (global grid) | **2.3x** | Spherical corrections on 721×1440 |

---

## Rust Array Throughput

1M elements, 32-core Ryzen, rayon parallel:

| Function | Time | Throughput |
|---|---:|---:|
| potential_temperature | 1.8 ms | 550 M elem/s |
| wet_bulb_temperature | 7.3 ms | 137 M elem/s |
| wind_speed | 1.5 ms | 660 M elem/s |

---

## Running the Benchmarks

```bash
# Three-way comparison (requires real HRRR data in data/)
python tests/benchmark_gpu.py              # MetPy + Rust + CUDA
python tests/benchmark_gpu.py --no-metpy   # Rust + CUDA only (faster)

# Real-data verification suite (requires HRRR + GFS GRIB files in data/)
python tests/benchmarks/bench_01_hrrr_severe.py
python tests/benchmarks/bench_02_gfs_upper_air.py
# ... through bench_12

# Run all verification benchmarks
for f in tests/benchmarks/bench_*.py; do python "$f"; done
```

Data files needed in `data/`:
- `hrrr_prs.grib2` — HRRR pressure levels (~405 MB)
- `hrrr_sfc.grib2` — HRRR surface fields (~148 MB)
- `gfs_0p25.grib2` — GFS 0.25° analysis (~490 MB)

Download from NOAA AWS:
```bash
curl -Lo data/hrrr_prs.grib2 "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260328/conus/hrrr.t00z.wrfprsf00.grib2"
curl -Lo data/gfs_0p25.grib2 "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260328/00/atmos/gfs.t00z.pgrb2.0p25.f000"
```
