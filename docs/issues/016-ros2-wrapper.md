# 016 — Add an optional ROS 2 wrapper

**Status:** Planned
**Priority:** P2
**Labels:** `area:integration`, `area:odometry`, `priority:p2`
**Depends on:** 009

## Goal

Expose the stable standalone registration/odometry core to ROS 2 without making
ROS a runtime requirement for FlashICP.

## Scope

Add an isolated package such as `ros/flashicp_ros`. Subscribe to
`sensor_msgs/msg/PointCloud2`, convert XYZ fields into the generic PointCloud
representation, and publish explicitly documented `nav_msgs/msg/Odometry`,
optional aligned `sensor_msgs/msg/PointCloud2`, and `geometry_msgs/msg/TransformStamped` /
TF outputs. Define timestamp, frame-id, queue, QoS, failure, and reset behavior.

Reuse core options and result metrics rather than duplicating ICP logic. Make
`ament`/ROS discovery conditional and keep the existing offline SQLite extractor
separate from this runtime node.

## Acceptance criteria

- The core library builds and tests on a machine with no ROS installation.
- A ROS 2 workspace can build the wrapper when ROS dependencies are present.
- A small PointCloud2 conversion test covers fields, endianness/step, NaN
  filtering, timestamps, and frame IDs.
- Published pose direction and TF tree are documented and exercised with a test
  or recorded example.
- Node failures are observable and do not publish stale valid-looking poses.
