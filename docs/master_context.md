# AIC Master Context Document

Generated from the current workspace on 2026-04-26.

This document is the working reference for future development in this repo. It is grounded in the requested docs first, then cross-checked against the actual ROS interfaces, controller code, engine, adapter, Docker files, and baseline policies.

## Canonical Sources Used

- `docs/overview.md`
- `docs/challenge_rules.md`
- `docs/getting_started.md`
- `docs/scene_description.md`
- `docs/glossary.md`
- `docs/aic_interfaces.md`
- `docs/aic_controller.md`
- `docs/custom_dockerfile.md`
- `docs/build_eval.md`
- `docs/participant_utilities.md`
- `docs/scoring.md`
- `docs/qualification_phase.md`
- `README.md`
- `aic_engine/README.md`
- `aic_model/aic_model/aic_model.py`
- `aic_model/aic_model/policy.py`
- `aic_adapter/src/aic_adapter.cpp`
- `aic_controller/src/aic_controller.cpp`
- `aic_controller/src/actions/cartesian_impedance_action.cpp`
- `aic_controller/src/actions/joint_impedance_action.cpp`
- `aic_bringup/config/aic_ros2_controllers.yaml`
- `docker/aic_model/Dockerfile`
- `docker/aic_eval/Dockerfile`
- `docker/docker-compose.yaml`

## Project Overview & Goals

The AI for Industry Challenge is a sim-to-real robotics competition centered on dexterous cable insertion for electronics assembly.

For qualification, the robot is a UR5e with:

- Robotiq Hand-E gripper
- ATI AXIA80-M20 force/torque sensor
- Three wrist-mounted RGB cameras

The manipulation problem is contact-rich and precision-sensitive:

- a flexible fiber cable is already grasped by the robot
- one plug end must be inserted into the specified target port
- the other cable end remains free
- board pose and component locations are randomized across trials

Qualification-phase task pattern:

- Trials 1-2: insert `SFP_MODULE` into the specified `SFP_PORT`
- Trial 3: insert `SC_PLUG` into the specified `SC_PORT`

The evaluation objective is not just binary insertion. The score combines:

- model validity
- successful insertion or near-port convergence
- smoothness
- task duration
- path efficiency
- force safety
- off-limit contact avoidance

Current authoritative per-trial scoring:

- Tier 1: `0` or `1`
- Tier 2:
  - smoothness `0..6`
  - duration `0..12`
  - efficiency `0..6`
  - force penalty `0..-12`
  - off-limit contact penalty `0..-24`
- Tier 3:
  - correct insertion `75`
  - wrong-port insertion `-12`
  - partial insertion / proximity up to `50`

Maximum per trial: `100`.

## The Golden Rules (Strict Constraints)

### 1. Use only official interfaces

Our model may interact only through the official ROS 2 topics, services, and actions.

Do not use backend or simulation-control interfaces to gain hidden state or manipulate the world.

Explicitly prohibited areas include:

- `/scoring`
- `/gazebo`
- `/gz_server`
- spawn/despawn/delete entity services
- simulation control and reset paths
- `/clock`, `/model`, `/world_stats`, `/pause_physics`
- changing lifecycle states or parameters of evaluation nodes
- tampering with engine, scoring, or logging infrastructure

Training is allowed to use ground-truth data. Evaluation is not.

### 2. `aic_model` lifecycle contract is mandatory

The submitted container must start a ROS 2 Lifecycle node named `aic_model`.

Required behavior:

- Discoverable within `30 s` of container start
- Start in `unconfigured`
  - no robot-command topics published
- `configure` must succeed within `60 s`
  - model loading belongs here
  - still no robot-command publishing
  - `/insert_cable` goals must be rejected here
- `activate` must succeed within `60 s`
  - `/insert_cable` goals must now be accepted
  - goals must be cancellable
- Each goal must complete within `Task.time_limit`
- `deactivate` back to `configured` within `60 s`
- `cleanup` back to `unconfigured` within `60 s`
- `shutdown` must succeed within `60 s`
  - no robot-command topics published
  - no robot-command publishers present in the graph

Practical implication: anything expensive in policy startup must finish inside the configure window.

### 3. Middleware is fixed: `rmw_zenoh_cpp`

Official evaluation uses:

- ROS 2 Kilted Kaiju
- `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
- Zenoh router-based connectivity

A custom participant image must:

- connect to the model router
- authenticate as user `model`
- use the password provided at runtime

The provided reference container also disables Zenoh shared memory for container-to-container operation.

### 4. Zenoh ACL is part of the anti-cheat boundary

The evaluation system uses Zenoh ACLs so the model identity cannot access cheating interfaces such as `gz_server` state services.

Treat ACL failures as expected behavior, not bugs to work around.

### 5. F/T tare is not ours during evaluation

`/aic_controller/tare_force_torque_sensor` exists for training and teleop, but it is disabled during evaluation. The evaluation system tares the sensor automatically before the cable is spawned.

## Architecture Map (Immutable vs. Mutable)

### Immutable: evaluation component (`aic_eval`)

These packages define the organizer-owned runtime. We may read them locally, but changing them does not change official evaluation behavior.

| Component | Role | Touch in official eval? |
| --- | --- | --- |
| `aic_engine` | Trial orchestration, lifecycle validation, goal dispatch, scoring run control | No |
| `aic_bringup` | Launches Gazebo, bridges, controller, engine | No |
| `aic_controller` | 500 Hz low-level control, clamping, interpolation, impedance, gravity comp | No |
| `aic_adapter` | Synchronizes cameras + state into `/observations` | No |
| `aic_gazebo` | Gazebo plugins for contacts, scoring, cable behavior, reset | No |
| `aic_scoring` | Tier 1/2/3 score computation | No |
| `aic_description` / `aic_assets` | URDF/SDF/assets for world, robot, board, cable | No |
| `docker/aic_eval` | Reference eval container and Zenoh router setup | No |

Boundary rule: we interact with this side only through allowed ROS interfaces.

### Mutable: participant model component (`aic_model`)

This is our workspace and our submission surface.

| Area | What belongs here |
| --- | --- |
| `aic_model/` | Provided lifecycle/action wrapper if we use the recommended path |
| custom policy package(s) | Vision, state estimation, control logic, learned models |
| model weights/config | Packaged assets needed at runtime |
| `docker/aic_model` or custom model Dockerfile | Submission image build |
| training utilities | Data collection, offline training, simulator-side experiments |

Recommended default: keep using the provided `aic_model` framework and swap in our own policy class.

## Interface Reference Guide

### Preferred high-level input: `/observations`

The organizer-side `aic_adapter` republishes a synchronized `aic_model_interfaces/msg/Observation` at camera rate, effectively up to `20 Hz`.

`Observation` contains:

- `left_image`, `center_image`, `right_image`
- corresponding `camera_info`
- `wrist_wrench`
- `joint_states`
- `controller_state`

Important implementation detail from `aic_adapter`:

- the three camera images are only published together when their timestamps align within `1 ms`
- the adapter then attaches the nearest older joint-state, controller-state, and wrench samples
- joint order is normalized to 7 joints: 6 arm joints + left gripper finger joint

### Raw sensor and state topics

| Topic | Type | Notes |
| --- | --- | --- |
| `/left_camera/image` | `sensor_msgs/msg/Image` | Rectified left wrist camera |
| `/left_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Left camera calibration |
| `/center_camera/image` | `sensor_msgs/msg/Image` | Rectified center wrist camera |
| `/center_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Center camera calibration |
| `/right_camera/image` | `sensor_msgs/msg/Image` | Rectified right wrist camera |
| `/right_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Right camera calibration |
| `/fts_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | Wrist force/torque |
| `/joint_states` | `sensor_msgs/msg/JointState` | Arm + gripper joint state |
| `/gripper_state` | `sensor_msgs/msg/JointState` | Gripper state topic |
| `/tf` | `tf2_msgs/msg/TFMessage` | Dynamic TF |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Static TF |
| `/aic_controller/controller_state` | `aic_control_interfaces/msg/ControllerState` | TCP pose/velocity, reference pose, TCP error, reference joint command, target mode, F/T tare offset |

### Actuators and controller services

| Interface | Type | Meaning |
| --- | --- | --- |
| `/aic_controller/pose_commands` | `aic_control_interfaces/msg/MotionUpdate` | Cartesian command input |
| `/aic_controller/joint_commands` | `aic_control_interfaces/msg/JointMotionUpdate` | Joint-space command input |
| `/aic_controller/change_target_mode` | `aic_control_interfaces/srv/ChangeTargetMode` | Switch controller between Cartesian and joint target modes |
| `/aic_controller/tare_force_torque_sensor` | `std_srvs/srv/Trigger` | Training-only tare service; disabled during eval |

Controller mode is mutually exclusive:

- `MODE_CARTESIAN = 1`
- `MODE_JOINT = 2`

If the controller is in Cartesian mode, it ignores joint commands. If it is in joint mode, it ignores Cartesian commands.

The provided `aic_model` wrapper automatically switches target mode the first time we publish a command of a different type.

### `MotionUpdate` essentials

Fields that matter most:

- `header.frame_id`: must be `base_link` or `gripper/tcp`
- `pose`: used when `trajectory_generation_mode.mode == MODE_POSITION`
- `velocity`: used when `trajectory_generation_mode.mode == MODE_VELOCITY`
- `target_stiffness`: `6x6` row-major matrix
- `target_damping`: `6x6` row-major matrix
- `feedforward_wrench_at_tip`
- `wrench_feedback_gains_at_tip`: each gain should stay in `[0, 0.95]`

### `JointMotionUpdate` essentials

Fields that matter most:

- `target_state.positions` for position mode
- `target_state.velocities` for velocity mode
- per-joint `target_stiffness`
- per-joint `target_damping`
- optional `target_feedforward_torque`

### Task trigger: `/insert_cable`

The task trigger is the ROS 2 action:

- `/insert_cable` of type `aic_task_interfaces/action/InsertCable`

The goal carries a `Task`:

- `id`
- `cable_type`
- `cable_name`
- `plug_type`
- `plug_name`
- `port_type`
- `port_name`
- `target_module_name`
- `time_limit`

Expected behavior:

- reject goals unless `aic_model` is active
- allow only one active goal at a time
- support cancellation
- return success/failure before `task.time_limit`

In the provided Python framework, this arrives as `Policy.insert_cable(task, get_observation, move_robot, send_feedback)`.

## Coordinate Frames & Control Math

### Critical frames

| Frame | Meaning |
| --- | --- |
| `base_link` | Robot base frame; global command frame for absolute poses |
| `gripper/tcp` | Tool center point at the gripper pinch point; key insertion frame |
| `ati/tool_link` | Force/torque sensor frame for tare offset reporting |
| `task_board/...` | Ground-truth task-board frames available only when enabled for development |
| `aic_world` | World frame used internally by scoring |

### Command-frame semantics

#### Cartesian position mode

- `frame_id = base_link`: target pose is absolute in base coordinates
- `frame_id = gripper/tcp`: target pose is interpreted as an offset from the current TCP pose

#### Cartesian velocity mode

- `frame_id = gripper/tcp`: twist is a body-frame TCP command
- `frame_id = base_link`: twist is a world-frame command

The controller internally transforms as needed so the impedance law still runs in base-frame coordinates.

### Controller pipeline

The controller runs at `500 Hz`. The intended policy command rate is much lower, roughly `10-30 Hz`.

Pipeline:

1. Clamp incoming target to limits
2. Interpolate low-rate targets into smooth high-rate references
3. Apply impedance control
4. Add gravity compensation
5. Send joint efforts to hardware

### Cartesian impedance law

The controller documentation and source implement the usual Jacobian-transpose Cartesian impedance structure:

`tau = J^T (Kp * pose_error + Kd * vel_error + wrench_term) + tau_null`

Source-verified additions:

- optional pose-error integral term
- wrench clamping
- nullspace torque
- optional joint-limit avoidance torque
- final per-joint torque clamping to URDF effort limits

### Joint impedance law

Joint-space control is standard per-joint impedance:

`tau = Kp * position_error + Kd * velocity_error + feedforward_torque`

Final torques are clamped to joint effort limits.

### Practical controller limits and defaults

From `aic_bringup/config/aic_ros2_controllers.yaml`:

- default target mode: `cartesian`
- translational velocity clamp: `[-0.25, 0.25] m/s` per axis
- rotational velocity clamp: `2.0 rad/s`
- cartesian workspace clamp: currently very loose at `[-5, 5] m` per axis
- tracking-error reset:
  - timeout `2.0 s`
  - min translation error to trigger reset logic: `0.2 m`
  - min translation change: `0.01 m`
  - min angular change: `0.01 rad`

Cartesian impedance defaults:

- stiffness diag: `[75, 75, 75, 75, 75, 75]`
- damping diag: `[35, 35, 35, 35, 35, 35]`
- maximum wrench after feedback law: `[10, 10, 10, 10, 10, 10]`
- feedforward wrench clamp: force `[-40, 40] N`, torque `[-5, 5] Nm`
- feedforward wrench slew limit: `[500, 500, 500, 25, 25, 25]` per second

Joint impedance defaults:

- stiffness: `[100, 100, 100, 50, 50, 50]`
- damping: `[40, 40, 40, 15, 15, 15]`

### One subtle but important helper behavior

The convenience helper `Policy.set_pose_target()` does not send a pure pose command. It also sets:

- translational wrench feedback gains to `[0.5, 0.5, 0.5]`
- default stiffness `[90, 90, 90, 50, 50, 50]`
- default damping `[50, 50, 50, 20, 20, 20]`

If we want exact control over force behavior, we should build `MotionUpdate` manually instead of assuming the helper is neutral.

## Development Workflow Summary

### Recommended eval-side workflow

For local development, the recommended path is still the prebuilt eval container.

Basic pattern:

1. Create/pull the `aic_eval` distrobox container
2. Start eval first, because it brings up the Zenoh router
3. Start the model from the Pixi workspace outside the container

For a headless remote machine, prefer:

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  gazebo_gui:=false \
  launch_rviz:=false \
  ground_truth:=false \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true
```

If you need longer bringup/debug time, increase:

- `model_discovery_timeout_seconds`

### Running our policy headlessly

Typical model launch:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=<your_python_module.YourPolicyClass>
```

Important: `aic_engine` expects to find `aic_model` within `30 s` by default.

### Foxglove Bridge for a no-GUI setup

The eval image includes `foxglove_bridge`.

Repo helper:

```bash
scripts/start_foxglove.sh
```

It launches the bridge on port `8765`.

For a remote server workflow, forward that port over SSH and use Foxglove instead of Gazebo/RViz.

### Training / debugging mode

For local experimentation:

- set `ground_truth:=true` when you explicitly want TF ground truth
- set `start_aic_engine:=false` when you want free-form scene exploration
- tare the F/T sensor before teleop/data collection
  
  
## Source-Backed Implementation Notes

### `aic_model` wrapper behavior

The provided wrapper:

- is a ROS 2 lifecycle node named `aic_model`
- dynamically imports the policy module from the `policy` parameter
- instantiates the policy during `on_configure()`
- subscribes to `/observations`
- creates a TF buffer/listener
- hosts the `/insert_cable` action server
- creates lifecycle publishers for the two controller command topics

Goal handling behavior:

- inactive lifecycle state => reject goal
- one active goal at a time
- cancellation is supported
- policy execution runs in a Python thread

### Baseline policies

`WaveArm`

- minimal reference for the policy API
- repeatedly sends Cartesian pose targets

`CheatCode`

- uses TF ground truth directly
- excellent for debugging geometry and insertion sequencing
- not legal as an evaluation strategy

`RunACT`

- proof-of-concept learned policy
- consumes three cameras plus a 26D robot state vector
- emits Cartesian twist commands
- downloads weights from Hugging Face at runtime

Important implication: `RunACT` is an integration example, not a submission-hardened packaging pattern. A real submission should not rely on runtime downloads during configure/activation.

### Observation construction matters

The adapter publishes observations at image cadence, not controller cadence.

That means:

- cameras are the timing anchor
- controller state and wrench are nearest-earlier samples
- policies should think in roughly `20 Hz` observation updates even though the robot controller runs at `500 Hz`

This is a major architecture fact for controller design and learned policy rollout timing.

## Watchouts / Doc Mismatches

### 1. Scoring ranges: `docs/scoring.md` is the authoritative one

The current scoring implementation and `docs/scoring.md` agree on:

- smoothness `0..6`
- duration `0..12`
- efficiency `0..6`

Some secondary docs still show older `0..5`, `0..10`, `0..5` ranges.

For future work, trust:

- `docs/scoring.md`
- `aic_scoring/src/ScoringTier2.cc`

### 2. Router env var naming mismatch

There is a repo inconsistency:

- `docs/custom_dockerfile.md` says `AIC_MODEL_ROUTER_ADDR`
- `docker/aic_model/Dockerfile` and `docker/docker-compose.yaml` use `AIC_ROUTER_ADDR`

If we ever build a fully custom entrypoint, the safest approach is to support both names.

## Working Assumptions For Our Future Development

These are the assumptions that best fit this repo and should stay our default unless proven wrong:

1. Use the provided `aic_model` lifecycle wrapper unless a hard requirement forces a custom node.
2. Treat `/observations` plus `/tf` as the main policy interface.
3. Use Cartesian control as the default insertion/control mode, switching to joint control only when it clearly helps.
4. Keep all evaluation-time dependencies self-contained in the image; do not rely on network downloads.
5. Debug visually with Foxglove and numerically with `/aic_controller/controller_state`, `/fts_broadcaster/wrench`, and the command topics.
6. Never let local convenience features like `ground_truth:=true` leak into evaluation assumptions.

## Bottom Line

The repo is built around a very clean separation:

- organizer-owned eval side provides synchronized sensing, 500 Hz low-level control, task orchestration, and scoring
- participant-owned model side is responsible only for turning allowed observations into valid robot commands under strict lifecycle and middleware constraints

If we stay inside that boundary, respect the lifecycle timing, and treat controller/frame semantics carefully, we can move fast here without building on sand.
