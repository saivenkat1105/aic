# AIC Policy Master Context

Use this as the compact bootstrap context for new chats. It is policy-focused and intentionally excludes backend internals, evaluation-container plumbing, Gazebo plugin details, and Zenoh access-control mechanics.

## Mission

- Build a ROS 2 policy for flexible cable insertion in the Intrinsic AI for Industry Challenge.
- Qualification tasks:
  - Trials 1-2: insert grasped `SFP_MODULE` into target `SFP_PORT`.
  - Trial 3: insert grasped `SC_PLUG` into target `SC_PORT`.
- Robot starts near target with one cable plug already grasped.
- Board/component poses are randomized; target details arrive in `Task`.

## Policy Contract

- Submitted node must be a ROS 2 Lifecycle node named `aic_model`.
- Recommended path: subclass `aic_model.Policy` and implement:

```python
insert_cable(task, get_observation, move_robot, send_feedback) -> bool
```

- Policy class name must match final module name:
  - `policy:=my_pkg.MyPolicy` requires class `MyPolicy`.
- Keep module import fast; heavy top-level imports count before lifecycle configuration.
- Load checkpoints/models in policy `__init__`, which runs during configure.

## Lifecycle Requirements

- `unconfigured`
  - Initial state.
  - No robot command publishing.
- `configure -> configured`
  - Must succeed within `60 s`.
  - Load policy/model here.
  - Still no robot command publishing.
  - `/insert_cable` goals must be rejected.
- `activate -> active`
  - Must succeed within `60 s`.
  - `/insert_cable` goals accepted.
  - Goals must be cancellable.
- Active goal must finish within `task.time_limit`.
- `deactivate -> configured` within `60 s`.
- `cleanup -> unconfigured` within `60 s`.
- `shutdown` within `60 s`; no command publishers should remain.

## Strict I/O Map

### Composite Policy Input

- `/observations`
  - `aic_model_interfaces/msg/Observation`
  - Synchronized packet at up to `20 Hz`.
  - Contains all cameras, camera info, wrist wrench, joint states, controller state.

### Raw Sensor/State Topics

- `/left_camera/image` - `sensor_msgs/msg/Image`
- `/left_camera/camera_info` - `sensor_msgs/msg/CameraInfo`
- `/center_camera/image` - `sensor_msgs/msg/Image`
- `/center_camera/camera_info` - `sensor_msgs/msg/CameraInfo`
- `/right_camera/image` - `sensor_msgs/msg/Image`
- `/right_camera/camera_info` - `sensor_msgs/msg/CameraInfo`
- `/fts_broadcaster/wrench` - `geometry_msgs/msg/WrenchStamped`
- `/joint_states` - `sensor_msgs/msg/JointState`
- `/gripper_state` - `sensor_msgs/msg/JointState`
- `/tf` - `tf2_msgs/msg/TFMessage`
- `/tf_static` - `tf2_msgs/msg/TFMessage`
- `/aic_controller/controller_state` - `aic_control_interfaces/msg/ControllerState`

### Command Topics

- `/aic_controller/pose_commands`
  - `aic_control_interfaces/msg/MotionUpdate`
  - Cartesian pose/velocity commands.
- `/aic_controller/joint_commands`
  - `aic_control_interfaces/msg/JointMotionUpdate`
  - Joint position/velocity commands.

### Controller Mode Service

- `/aic_controller/change_target_mode`
  - `aic_control_interfaces/srv/ChangeTargetMode`
  - `MODE_CARTESIAN = 1`
  - `MODE_JOINT = 2`
- Modes are mutually exclusive. Wrong-mode commands are ignored.
- Provided `move_robot()` helper switches mode automatically.

## Observation Message

```text
left_image, left_camera_info
center_image, center_camera_info
right_image, right_camera_info
wrist_wrench
joint_states
controller_state
```

`ControllerState` includes:

- `tcp_pose`
- `tcp_velocity`
- `reference_tcp_pose`
- `tcp_error[6]`
- `reference_joint_state`
- `target_mode`
- `fts_tare_offset`

## Task / Action Server

- Action: `/insert_cable`
- Type: `aic_task_interfaces/action/InsertCable`

Goal:

```text
Task task
```

`Task` fields:

- `id`
- `cable_type`
- `cable_name`
- `plug_type`
- `plug_name`
- `port_type`
- `port_name`
- `target_module_name`
- `time_limit`

Result:

```text
bool success
string message
```

Feedback:

```text
string message
```

Expected behavior:

- Reject goals unless lifecycle state is active.
- Accept only one active goal at a time.
- Run insertion policy.
- Publish feedback optionally via `send_feedback(str)`.
- Return `True` only when policy considers task complete.
- Return before `task.time_limit`.

## Control Modalities

### Cartesian Impedance: `MotionUpdate`

Command topic: `/aic_controller/pose_commands`

Math:

```text
tau = J^T [Kp (x_des - x) + Kd (xdot_des - xdot) + W_f] + tau_null
```

Required/programmatic fields:

- `header.frame_id`
  - `base_link`: absolute Cartesian target.
  - `gripper/tcp`: target offset or twist in TCP frame.
- `pose`
  - Used when `trajectory_generation_mode.mode = MODE_POSITION`.
- `velocity`
  - Used when `trajectory_generation_mode.mode = MODE_VELOCITY`.
- `target_stiffness`
  - `float64[36]`, row-major `6x6` Cartesian stiffness.
- `target_damping`
  - `float64[36]`, row-major `6x6` Cartesian damping.
- `feedforward_wrench_at_tip`
  - `geometry_msgs/Wrench`, external TCP force/torque term `W_f`.
- `wrench_feedback_gains_at_tip`
  - `float64[6]`, each gain should be in `[0, 0.95]`.
- `trajectory_generation_mode`
  - `MODE_VELOCITY = 1`
  - `MODE_POSITION = 2`

### Joint Impedance: `JointMotionUpdate`

Command topic: `/aic_controller/joint_commands`

Math:

```text
tau = Kp (q_des - q) + Kd (qdot_des - qdot) + tau_f
```

Required/programmatic fields:

- `target_state`
  - `trajectory_msgs/JointTrajectoryPoint`
  - `positions` used in position mode.
  - `velocities` used in velocity mode.
- `target_stiffness`
  - `float64[]`, per commanded joint.
- `target_damping`
  - `float64[]`, per commanded joint.
- `target_feedforward_torque`
  - `float64[]`, joint-space feedforward effort.
- `trajectory_generation_mode`
  - `MODE_VELOCITY = 1`
  - `MODE_POSITION = 2`

Key difference:

- Cartesian control regulates TCP pose/twist through `J^T` and accepts TCP wrench feedforward.
- Joint control regulates joint targets directly and accepts joint torque feedforward.
- `feedforward_wrench_at_tip` exists only in `MotionUpdate`, not `JointMotionUpdate`.

## Practical Policy Notes

- Observations arrive around camera rate: `20 Hz`.
- Controller accepts policy commands roughly `10-30 Hz`.
- Use ROS/sim-time-aware helpers:
  - `self.time_now()`
  - `self.sleep_for(seconds)`
- Avoid wall-clock-only logic for task timing.
- `Policy.set_pose_target()` is convenient but not neutral:
  - default stiffness `[90, 90, 90, 50, 50, 50]`
  - default damping `[50, 50, 50, 20, 20, 20]`
  - wrench feedback gains `[0.5, 0.5, 0.5, 0, 0, 0]`

## Scoring Metrics

Per trial max: `100`. Qualification has 3 trials, max `300`.

- Tier 1: Model validity, `0/1`
  - Lifecycle compliance.
  - Action response.
  - Valid robot commands.
- Jerk / Smoothness: `0-6`
  - Time-weighted average linear jerk magnitude.
  - Computed only while TCP speed `> 0.01 m/s`.
  - `0 m/s^3 -> 6 pts`; `>= 50 m/s^3 -> 0 pts`.
  - Awarded only if plug ends near/inside target region.
- Task Time: `0-12`
  - Elapsed task duration.
  - `<= 5 s -> 12 pts`; `>= 60 s -> 0 pts`.
  - Awarded only if plug ends near/inside target region.
- Trajectory Efficiency: `0-6`
  - Cumulative TCP path length.
  - Shorter/direct paths score higher.
- Convergence / Insertion: up to `75`
  - Correct full insertion: `75`.
  - Wrong port insertion: `-12`.
  - Partial insertion: `38-50`, based on depth inside port region with `5 mm` x-y tolerance.
  - Proximity: `0-25`, based on final plug distance to target port.
- Force Safety: `0 to -12`
  - Penalty if force exceeds `20 N` for more than `1 s`.
- Collisions: `0 to -24`
  - Penalty for robot-link contact with off-limit environment/task-board surfaces.

## Default Strategy Bias

- Prefer Cartesian velocity or small Cartesian pose updates near insertion.
- Use compliant stiffness/damping during contact.
- Monitor `wrist_wrench`, `tcp_error`, `tcp_velocity`, and final approach alignment.
- Optimize in this order: valid lifecycle/action behavior, safe convergence, insertion depth, then speed/smoothness.
