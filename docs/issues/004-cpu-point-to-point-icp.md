# 004 — Implement CPU point-to-point ICP

**Status:** Implemented  
**Priority:** P0  
**Labels:** `area:core`, `area:odometry`, `priority:p0`  
**Depends on:** 002, 003

## Goal

Implement a complete, readable CPU reference registration path before optimizing
CUDA. This is the correctness oracle for later GPU work.

## Scope

For each iteration, transform source points with the current estimate, find
nearest target points within the configured radius, reject invalid pairs, solve
the rigid point-to-point least-squares problem, compose the SE(3) update, and test
convergence. Use a robust, dependency-light 3D rigid solver (for example a
well-tested Horn/Kabsch implementation) and handle reflection/degenerate cases.

Support maximum iterations, correspondence radius, convergence tolerance,
voxel size, and minimum correspondence count. Report no-correspondence,
insufficient-correspondence, degenerate-system, invalid-input, and non-converged
outcomes distinctly.

The initial implementation may use brute-force CPU correspondence and host-side
linear algebra. It should be simple enough to inspect against a known transform.

## Acceptance criteria

- A synthetic cloud transformed by a known SE(3) pose is recovered within stated
  translation and rotation tolerances from a reasonable initial guess.
- The result reports iterations, final residual, and correspondence count.
- Empty, nonfinite, degenerate, and insufficient-overlap inputs return explicit
  failure statuses.
- The CPU path works with `-DUSE_CUDA=OFF` and preserves AUV cloud loading.
- A transform convention and composition order are covered by a unit test.
