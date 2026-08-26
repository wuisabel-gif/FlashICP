# 017 — Add an optional GTSAM frontend example

**Status:** Planned  
**Priority:** P2  
**Labels:** `area:integration`, `priority:p2`  
**Depends on:** 009, 010

## Goal

Show how FlashICP can provide relative pose constraints to a factor-graph backend
without turning the project into a GTSAM wrapper.

## Scope

Add a small example or separate optional target that converts a successful
FlashICP relative SE(3) result into `BetweenFactor<Pose3>` input. Document the
noise model boundary, transform convention, failure handling, and ownership:
FlashICP performs registration; GTSAM performs estimation/optimization.

GTSAM must be discovered optionally and must not be required for the core,
KITTI loader, odometry CLI, tests, or CUDA benchmark. Do not add loop closure,
map management, or an embedded optimizer to the core API.

## Acceptance criteria

- The example compiles only when GTSAM is available and is skipped cleanly
  otherwise.
- A deterministic relative transform fixture verifies the conversion to the
  chosen GTSAM pose convention.
- Registration failure is not converted into a valid graph factor.
- README and build instructions label this as an integration example, not as a
  FlashICP backend feature.
