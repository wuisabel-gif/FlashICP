# 007 — Add point-to-plane ICP

**Status:** Planned
**Priority:** P0
**Labels:** `area:core`, `area:cuda`, `area:odometry`, `priority:p0`
**Depends on:** 005, 006

## Goal

Make point-to-plane the primary robotics registration method while retaining
point-to-point as a baseline and fallback.

## Scope

Implement target normal estimation with a straightforward documented method
first. For each valid correspondence, compute the point-to-plane residual and
SE(3) Jacobian, accumulate the 6x6 normal equations and 6x1 right-hand side,
and solve for the incremental twist. Provide a stable small-system solver and
reject singular or poorly conditioned systems.

Implement the CPU path first, then the CUDA per-correspondence accumulation and
parallel reduction. Expose `--method point-to-point` and
`--method point-to-plane` (or equivalent API configuration). Make normal
orientation and source/target frame conventions explicit.

## Acceptance criteria

- CPU point-to-plane reduces residual on a deterministic planar/structured
  fixture and recovers a known small motion within documented tolerances.
- CUDA and CPU normal-equation results and final poses agree within tolerances.
- The result reports normal-estimation and degenerate-system failures.
- Options include maximum iterations, radius, voxel size, convergence tolerance,
  and minimum correspondences.
- Point-to-point remains selectable and its tests continue to pass.
