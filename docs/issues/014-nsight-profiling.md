# 014 — Profile important CUDA stages before optimizing

**Status:** Planned
**Priority:** P1
**Labels:** `area:cuda`, `area:benchmark`, `priority:p1`
**Depends on:** 011, 013

## Goal

Use Nsight evidence to guide optimization of the actual ICP workload.

## Scope

Profile voxelization, hash construction, correspondence, transform, residual /
Jacobian, reduction, and host-device synchronization with representative ZED and
KITTI-sized clouds where available. Investigate occupancy, global memory traffic
and coalescing, atomic contention, warp divergence, launch overhead, transfers,
synchronization, and shared-memory opportunities.

For each meaningful change, add a short record under `docs/performance/` with
problem, measurement, change, before/after timing, accuracy impact, and tradeoff.
Do not optimize solely against a tiny cloud or report a kernel-only gain as an
end-to-end registration gain.

## Acceptance criteria

- A checked-in profiling procedure names the Nsight tool, command, workload, and
  synchronization policy.
- At least one profile identifies a measured bottleneck before a corresponding
  optimization is merged.
- Optimized results pass the CPU/GPU correctness suite.
- Before/after numbers identify platform, input, build, and timing scope.
- Failed or neutral optimizations may be recorded as engineering evidence.
