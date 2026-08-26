# 006 — Implement CUDA point-to-point ICP

**Status:** Partial, CUDA unverified  
**Priority:** P0  
**Labels:** `area:cuda`, `area:core`, `priority:p0`  
**Depends on:** 004, 005

## Goal

Complete the CUDA point-to-point path while retaining the CPU implementation as
the reference and keeping device details behind the public API.

## Scope

Start with a correctness-first implementation: upload or stage generic clouds,
transform source points on the device, use the existing spatial-hash
correspondence as appropriate, reduce accepted residuals on the device, and
compose a host-solved rigid increment. Reuse allocations once the numerical path
is correct; measure before changing the existing hash or voxel kernels. The
checked-in first cut intentionally keeps the small Horn solve on the host.

Define what timing includes: host-to-device transfer, grid construction, query,
reduction, solve, and synchronization. Check every launch and synchronize at
stage boundaries used for timing.

## Acceptance criteria

- CUDA registration recovers the same synthetic transforms as CPU within the
  tolerances from issue 005.
- CUDA and CPU report comparable match counts and final residuals on the same
  input.
- Empty/invalid/insufficient-match cases return the same documented failure
  classes rather than aborting or returning a garbage transform.
- CPU-only builds still compile and run without CUDA.
- A CUDA correctness command or test records device, CUDA version, and whether
  transfers are included in the measurement.
