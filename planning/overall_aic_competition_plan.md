# Overall AIC Competition Plan

## Summary

The immediate goal is to move beyond the existing lifecycle-valid 3-point policy and build a legal policy that gets the plug close enough to the correct target port to earn positive Tier 3 proximity score and unlock Tier 2 duration, smoothness, and efficiency points.

The fastest path is a staged system: first a visual-motor baseline, then model-agnostic data generation, then ACT training, with IsaacLab validated in parallel but not blocking Gazebo progress.

## Strategy

- Keep official evaluation compliance absolute: submitted policies may use only `/observations`, task fields, official robot transforms, controller state, wrench, and command topics.
- Use Gazebo as the scoring source of truth because official evaluation is Gazebo-based.
- Use IsaacLab as a data and randomization multiplier once validated, not as the primary proof of performance.
- Prioritize proximity first, partial insertion second, full insertion third.
- Use ACT as the first learned policy because it is fast, lightweight, and already integrated with LeRobot/AIC.
- Defer Diffusion and VLA work until the proximity baseline and ACT data pipeline are working.

## Work Plan

1. Build `VisualMotorApproachPolicy`.
   - Detect target port from official wrist camera images.
   - Servo the robot toward the target port using Cartesian velocity commands.
   - Stop near the port entrance without forcing insertion.
   - Monitor wrench and retreat or stop on high contact force.

2. Record model-agnostic training data.
   - Store raw RGB images, robot state, wrench, task fields, and 6D Cartesian twist actions.
   - Store detector outputs and ground-truth labels only as metadata or training labels.
   - Keep ACT, Diffusion, and VLA training possible from the same dataset.

3. Validate IsaacLab in parallel.
   - Hard cap setup and debugging time.
   - Require headless resettable rollouts, camera observations, and action recording.
   - Use IsaacLab data only after Gazebo validation confirms transfer.

4. Train ACT after the visual-motor baseline works.
   - Train first on approach behavior.
   - Add noisy `CheatCode` and failure-recovery data for better robustness.
   - Deploy through `RunACTOffline` style local checkpoint loading.

5. Iterate with DAgger.
   - Run learned policy in Gazebo with `ground_truth:=false`.
   - Collect failures.
   - Correct with teacher, script, or human intervention.
   - Retrain and compare scoring.

## Dataset Design

Each episode should contain:

- `observation.images.left_camera`
- `observation.images.center_camera`
- `observation.images.right_camera`
- `observation.state`: TCP pose, TCP velocity, TCP error, joint positions
- `observation.wrench`: force/torque
- `task`: cable type, plug type, port type, target module, port name, text prompt
- `action`: 6D Cartesian twist in `base_link`
- `metadata`: seed, sim source, score, target type, detector keypoints, optional ground-truth labels

## Success Criteria

- Baseline policy gets positive Tier 3 proximity on at least one trial.
- Stronger baseline gets positive Tier 3 on all three trials.
- No force penalty, no off-limit contact penalty.
- ACT training data can be reused without changing schema.
- First learned ACT policy matches or beats the visual-motor baseline in Gazebo.

## Assumptions

- Existing lifecycle-valid node already passes Tier 1.
- Official scoring rewards proximity even without insertion.
- Internal perception outputs are legal if computed from official camera images.
- Ground-truth labels are allowed for training but not as evaluation-time inputs.

