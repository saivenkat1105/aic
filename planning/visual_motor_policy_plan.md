# Visual-Motor Policy Implementation Plan

## Goal

Build a legal AIC visual-motor policy that moves the grasped plug close to the correct target port using only official evaluation-time observations.

The target is not full insertion yet. The target is positive Tier 3 proximity score while preserving smoothness, short path length, and low force.

## Policy Architecture

Implement a staged policy:

```text
/observations + Task
      |
      v
image preprocessing
      |
      v
target port/keypoint detector
      |
      v
visual-servo controller
      |
      v
safe Cartesian velocity command
      |
      v
/aic_controller/pose_commands
```

The detector is internal and legal because it uses official camera images. Do not use ground-truth port or plug TF during evaluation.

## Inputs

Use:

- `Observation.left_image`, `center_image`, `right_image`
- `Observation.controller_state.tcp_pose`
- `Observation.controller_state.tcp_velocity`
- `Observation.controller_state.tcp_error`
- `Observation.joint_states`
- `Observation.wrist_wrench`
- `Task.plug_type`
- `Task.port_type`
- `Task.port_name`
- `Task.target_module_name`

Do not use:

- `/scoring`
- `/gazebo`
- `/gz_server`
- ground-truth task-board or cable TF during evaluation
- simulator model or world state
- evaluation lifecycle or scoring internals

## Implementation Phases

1. Create the policy shell.
   - Subclass `aic_model.Policy`.
   - Implement `insert_cable(task, get_observation, move_robot, send_feedback)`.
   - Use sim-time-aware `self.time_now()` and `self.sleep_for()`.
   - Run at roughly `10-20 Hz`.

2. Implement perception.
   - Start with a simple target detector trained from ground-truth-labeled Gazebo images.
   - Predict target port entrance keypoint in image coordinates.
   - Prefer center camera first; use left and right cameras as fallback or for confidence voting.
   - Output: `(u, v, confidence, target_type)`.

3. Implement visual servo.
   - Convert image keypoint error into Cartesian velocity.
   - Use conservative gains.
   - Command in `base_link` initially for consistency with ACT training.
   - Clamp translation velocity to safe values, for example `0.01-0.06 m/s`.
   - Clamp angular velocity conservatively.
   - Smooth commands with an exponential moving average.

4. Implement approach stages.
   - `WAIT_OBS`: wait for first valid observation.
   - `SEARCH`: small safe scan if detector confidence is low.
   - `CENTER`: reduce image-space target error.
   - `APPROACH`: move toward port while keeping target centered.
   - `HOLD`: stop near port entrance and return success.
   - `ABORT_SAFE`: stop or retreat on excessive force or timeout.

5. Implement safety.
   - Stop or retreat if force magnitude exceeds a conservative threshold.
   - Avoid aggressive descent.
   - Return before `task.time_limit`.
   - Publish zero or near-zero velocity before returning.

6. Record data while running.
   - Save every observation/action pair.
   - Save detector outputs as metadata.
   - Save task fields and final score.
   - Keep the dataset usable for ACT and Diffusion later.

## Motion Command

Use `MotionUpdate` in velocity mode:

- `header.frame_id = "base_link"`
- `trajectory_generation_mode.mode = MODE_VELOCITY`
- `velocity.linear.{x,y,z}` from visual servo
- `velocity.angular.{x,y,z}` small or zero initially
- moderate stiffness/damping
- zero feedforward wrench
- zero or low wrench feedback gains

## Detector Training Data

Use a training-only labeling script:

- Run Gazebo with `ground_truth:=true`.
- Use target port TF and camera calibration to project target entrance into image pixels.
- Store images and labels.
- Randomize board pose, target rail positions, lighting if available, and approach offsets.
- Include negative or low-confidence frames where the target is partially occluded or off-center.

## Acceptance Tests

- Policy starts, configures, activates, and completes `/insert_cable`.
- With `ground_truth:=false`, it uses no forbidden topics.
- It produces smooth nonzero commands only while active.
- It gets positive Tier 3 proximity in at least one local Gazebo run.
- It avoids force and off-limit contact penalties.
- Recorded data can be loaded into a LeRobot-style dataset.

## Future ACT Use

Train ACT later on the same recorded data:

- ACT inputs: images, robot state, task text or task fields.
- ACT output: 6D Cartesian twist.
- Do not provide detector keypoints as ACT inputs unless the deployed ACT wrapper also computes them internally.
- Use detector keypoints and ground-truth projections only for auxiliary training, debugging, filtering, or metadata.

## Blind Spots

- If the detector fails silently, the controller may confidently move to the wrong place. Always use confidence thresholds and timeout behavior.
- Positive Tier 3 depends on final plug-port distance, not just TCP distance. The controller should learn or estimate the plug offset from TCP.
- A policy that stops near the port can score more than a policy that attempts insertion and causes force or contact penalties.
- The first version should be boring, slow, and safe. Speed can be optimized after proximity is reliable.

