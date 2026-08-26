# FlashICP issue briefs

These files are copy/paste-ready issue specifications for the roadmap in
[`../roadmap.md`](../roadmap.md). They are deliberately narrower than the full
handoff: each issue has one outcome, explicit dependencies, and acceptance
criteria that can be checked in a pull request.

| Issue | Title | Status |
|---|---|---|
| [001](001-audit-and-baseline.md) | Audit and baseline | Complete |
| [002](002-generic-core-api.md) | Introduce the generic core API | Implemented |
| [003](003-primitives-and-ci.md) | Harden primitives and CI | Partial |
| [004](004-cpu-point-to-point-icp.md) | Implement CPU point-to-point ICP | Implemented |
| [005](005-registration-tests.md) | Add registration correctness tests | Partial |
| [006](006-cuda-point-to-point-icp.md) | Implement CUDA point-to-point ICP | Partial, CUDA unverified |
| [007](007-point-to-plane-icp.md) | Add point-to-plane ICP | Planned |
| [008](008-kitti-loader.md) | Add the KITTI Odometry loader | Planned |
| [009](009-lidar-odometry-cli.md) | Build sequential LiDAR odometry | Planned |
| [010](010-trajectory-evaluation.md) | Add trajectory evaluation | Planned |
| [011](011-benchmark-suite.md) | Build the CPU/CUDA benchmark suite | Planned |
| [012](012-robustness-experiments.md) | Add reproducible robustness experiments | Planned |
| [013](013-jetson-validation.md) | Validate and document Jetson deployment | Planned |
| [014](014-nsight-profiling.md) | Profile before optimizing | Planned |
| [015](015-rerun-visualization.md) | Add optional Rerun visualization | Planned |
| [016](016-ros2-wrapper.md) | Add an optional ROS 2 wrapper | Planned |
| [017](017-gtsam-example.md) | Add an optional GTSAM example | Planned |

Suggested labels are `area:core`, `area:cuda`, `area:data`, `area:odometry`,
`area:benchmark`, `area:integration`, and `priority:p0` through `priority:p2`.
The status in this directory is a planning aid; GitHub issue state remains the
source of truth once these briefs are copied into the tracker.
