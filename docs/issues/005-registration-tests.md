# 005 — Add registration correctness and failure-mode tests

**Status:** Partial  
**Priority:** P0  
**Labels:** `area:core`, `priority:p0`  
**Depends on:** 004

## Goal

Protect the CPU oracle and provide the fixtures needed to judge the CUDA
implementation. Tests must cover useful failures, not only a successful toy
alignment.

## Scope

Add deterministic tests for:

- known translation and rotation;
- Gaussian noise;
- outliers and robust rejection policy;
- partial overlap and reduced point density;
- empty clouds and too few matches;
- extreme initial guesses;
- NaN/Inf input;
- collinear, coplanar, duplicate, and otherwise degenerate geometry.

Add keyed voxel-output checks rather than only aggregate centroid checks. For
CPU/GPU correspondence, compare match validity and distance within a defined
float tolerance while allowing documented nearest-neighbor ties.

Keep generated datasets small and deterministic. Do not require CUDA for the
core suite; add GPU agreement tests conditionally when a device is available.

## Acceptance criteria

- Tests are registered with the build system and run by the normal CPU CI job.
- Every failure test asserts a specific status or invariant, not merely “did not
  crash.”
- CPU/GPU registration comparison defines translation, rotation, residual, and
  correspondence tolerances in one place.
- A failing CUDA device test is distinguishable from a skipped test.
