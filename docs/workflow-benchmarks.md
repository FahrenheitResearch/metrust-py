# Workflow Benchmarks

This page tracks end-to-end replay benchmarks for real `metpy.calc` workflow
shapes rather than isolated function timings. The goal is to answer the
question users actually care about:

`Can I swap the import in a real workflow and get the same result faster?`

The benchmark harness lives in `benches/bench_workflows.py` and reuses the
same workflow shapes covered by `tests/test_cookbook_replays.py`:

- sounding analysis
- gridded diagnostics
- xarray-heavy dataset helpers

The harness verifies output parity once before it measures time, then reports
p50 latency for both libraries.

## How To Run

```bash
python benches/bench_workflows.py
python benches/bench_workflows.py --json
```

The JSON mode writes `workflow_bench_results.json` by default.

## Current Snapshot

The table below is updated from a local run of `python benches/bench_workflows.py`
on Windows 11 (`10.0.26200`) with Python `3.13.7`. These numbers are
workflow-level p50 timings, not microbenchmarks.

| Workflow | metrust p50 | MetPy p50 | Speedup |
|---|---:|---:|---:|
| Cookbook sounding replay | 10.38 ms | 30.79 ms | 2.97x |
| Cookbook grid diagnostics replay | 2.59 ms | 22.73 ms | 8.79x |
| Cookbook xarray replay | 2.51 ms | 3.66 ms | 1.46x |

## Why These Benchmarks Matter

- They exercise chained calculations instead of isolated math kernels.
- They reflect the import-swap story users actually try first.
- They complement the replay tests by proving both correctness and runtime
  behavior on the same workflow surface.

For lower-level benchmarking details, see [Performance](performance.md).
