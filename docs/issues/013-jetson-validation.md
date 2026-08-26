# 013 — Validate and document Jetson deployment

**Status:** Planned
**Priority:** P1
**Labels:** `area:cuda`, `area:benchmark`, `priority:p1`
**Depends on:** 011

## Goal

Demonstrate the supported CUDA path on NVIDIA Jetson AGX Orin with reproducible
configuration and honest measurements.

## Scope

Document and, when hardware is available, run the full odometry and benchmark
commands on Orin. Record JetPack, CUDA, driver, GPU/power mode, compiler, CMake
configuration, dataset/sequence, points per frame, method, options, warm-up,
and timing protocol. Make CMake architecture selection explicit and retain a
convenient `sm_87` configuration.

Preserve the existing run log showing that the first sort implementation was
slower than CPU and the later atomic hash improved the voxel stage. Add new
results rather than rewriting history. Include CPU-only and CUDA-disabled
behavior in the deployment instructions.

## Acceptance criteria

- A clean documented build runs the current supported CUDA test and odometry
  command on Orin when hardware is available.
- Logs identify exact platform and workload metadata and contain no fabricated
  values.
- CUDA failures and unavailable hardware are reported as such, not converted to
  successful benchmark rows.
- The CPU fallback remains buildable on non-NVIDIA hosts.
