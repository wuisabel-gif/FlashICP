# 011 — Build the CPU/CUDA benchmark suite

**Status:** Planned
**Priority:** P1
**Labels:** `area:benchmark`, `area:cuda`, `priority:p1`
**Depends on:** 006, 007, 009

## Goal

Make performance a reproducible measurement of the registration pipeline rather
than a collection of isolated timing claims.

## Scope

Benchmark load, voxelization, normal estimation, correspondence,
residual/Jacobian construction, reduction, solve, and total ICP. Provide CPU and
CUDA modes, warm-up iterations, configurable repetitions, and synchronized stage
boundaries. State whether host-device transfers and one-time grid allocation are
included; report both useful end-to-end timing and separable kernel timing when
possible.

Support representative point counts around 10k, 50k, 100k, 250k, and 500k when
data exists. Emit CSV/JSON containing point count, method, voxel/radius options,
latency, FPS, speedup, accuracy, correspondence count, device, CUDA version,
compiler, build type, and timing protocol. Use percentiles, including P95, for
per-frame runs. Keep the existing real ZED/Orin numbers as historical evidence,
not as KITTI or end-to-end values.

## Acceptance criteria

- Re-running one documented command regenerates a machine-readable result.
- CUDA timing synchronizes before samples are read and includes warm-up policy.
- CPU and CUDA results are paired with the same input and accuracy check.
- No table contains placeholders presented as measurements.
- The benchmark reports when CUDA is unavailable instead of silently labeling CPU
  work as GPU work.
