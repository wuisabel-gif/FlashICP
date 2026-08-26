# 015 — Add optional Rerun visualization

**Status:** Planned
**Priority:** P2
**Labels:** `area:integration`, `priority:p2`
**Depends on:** 009, 010

## Goal

Provide an optional visualization layer for debugging and demonstrations without
making Rerun part of the registration algorithm or required core dependency.

## Scope

Visualize source, target, transformed source, selected correspondences, frame
axes, estimated trajectory, KITTI ground truth when available, registration
error, and ICP iteration/convergence information. Choose a narrow integration
boundary, such as a JSON/CSV/cloud event export consumed by a small optional
viewer, or an optional Rerun SDK target. Keep headless odometry and benchmarks
working when Rerun is not installed.

Avoid logging every full-resolution intermediate cloud by default; make sampling
and output volume configurable.

## Acceptance criteria

- The standalone core and CLI build without Rerun.
- An optional documented build/run produces a view for a small deterministic
  registration and an odometry trajectory.
- Estimated and ground-truth frames use the same documented convention as the
  evaluator.
- Visualization failure cannot change registration results or exit status for a
  headless run.
