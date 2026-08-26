# 010 — Add KITTI trajectory evaluation

**Status:** Planned  
**Priority:** P0  
**Labels:** `area:odometry`, `area:benchmark`, `priority:p0`  
**Depends on:** 008, 009

## Goal

Quantify LiDAR odometry against KITTI ground truth without hard-coded or
fabricated values.

## Scope

Parse the official KITTI `poses/<sequence>.txt` 3x4 pose format when supplied or
when the documented dataset root makes it discoverable. Align frame indices and
coordinate conventions explicitly; report missing labels instead of guessing.

At minimum compute per-frame and aggregate translation error and rotation error.
Add KITTI-style relative pose error over documented segment lengths where
practical, plus absolute trajectory error when the alignment convention is
clear. Also report total frames, successful/failed registrations, average
correspondences, average iterations, mean/p95 latency, and the number of frames
with no ground truth.

Write stable CSV and/or JSON schemas with units and field definitions. Keep
registration metrics separate from trajectory metrics so a good local ICP result
cannot be confused with good global odometry.

## Acceptance criteria

- A synthetic ground-truth/estimate pair produces known translation and rotation
  metrics.
- KITTI pose parsing and frame indexing are tested with a small fixture.
- Evaluation fails clearly on incompatible lengths or undocumented frame
  conventions; it never fabricates a score.
- The odometry command can emit machine-readable per-frame and aggregate results.
- Human-readable output identifies sequence, frames, method, device, and metric
  definitions.
