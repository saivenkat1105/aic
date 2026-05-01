# SFP Visual Proximity Baseline Workflow

## Purpose

This document defines the first implementation workflow for a legal AIC visual
baseline that targets modest total score through the Tier 3 proximity branch.
The goal is not insertion. The goal is to move the grasped SFP plug tip close to
the correct SFP port, stop before contact, activate Tier 2 scoring, and produce
useful trajectory data for later ACT fine-tuning.

The baseline is intentionally conservative:

- SFP first, covering qualification trials 1 and 2 before adding SC.
- No partial insertion and no intentional contact.
- A 15-20 mm no-contact standoff for the first scoring runs.
- Candidate detection plus deterministic target selection.
- Tri-camera keypoint triangulation for plug-space control.
- Training-only use of ground-truth TF for label generation and teacher
  demonstrations.
- Training/local-evaluation use of scoring modules as episode-level metrics.
- Evaluation-time use of only official observations and task fields.

## Scoring Objective

The target score range is 50-100 points total across all qualification trials,
not per trial. This is realistic for a no-contact proximity baseline if at least
some trials end with positive Tier 3 proximity and avoid penalties.

Pure no-contact proximity cannot reach the full per-trial score because it does
not claim partial insertion or successful insertion. It can still be useful:

- Tier 3 proximity gives up to 25 points if the plug is near the target port
  but not inserted.
- Tier 2 smoothness, duration, and efficiency are only awarded when Tier 3 is
  positive.
- Avoiding off-limit contact and excessive force is more important than chasing
  a small additional proximity gain in the first version.

The policy should return success when it has held a stable no-contact standoff
near the selected target port. It should not descend through the port entrance
plane.

## Challenge Compliance

Competition-facing behavior must use only official evaluation-time inputs:

- `Observation.left_image`, `Observation.center_image`,
  `Observation.right_image`
- corresponding `CameraInfo` messages
- `Observation.controller_state.tcp_pose`
- `Observation.controller_state.tcp_velocity`
- `Observation.controller_state.tcp_error`
- `Observation.joint_states`
- `Observation.wrist_wrench`
- `Task.plug_type`
- `Task.plug_name`
- `Task.port_type`
- `Task.port_name`
- `Task.target_module_name`
- `Task.time_limit`

Do not use these during evaluation:

- `/scoring`
- `/gazebo`
- `/gz_server`
- `/scoring/tf`
- ground-truth task-board TF
- ground-truth cable or plug TF
- simulator model/world state
- scoring or evaluation internals

Training utilities may use `ground_truth:=true` and TF-derived labels or teacher
actions, but those labels/actions must become model weights, validation
artifacts, demonstrations, or metadata. They must not become evaluation-time
state inputs.

Training and local evaluation utilities may also run the scoring modules to
compute episode-level metrics. These scores may be used for dataset filtering,
checkpoint selection, controller threshold tuning, and experiment comparison.
They must not be subscribed to, queried, or used by the deployed policy during
evaluation.

## Board And Task Facts Used By The Baseline

For SFP trials:

- The robot starts near the task with the SFP plug already grasped.
- The target is an SFP port on a NIC card in board Zone 1.
- A NIC card has two relevant SFP ports:
  - `sfp_port_0`
  - `sfp_port_1`
- The task identifies the target semantically through:
  - `Task.target_module_name`, for example `nic_card_mount_0`
  - `Task.port_name`, for example `sfp_port_0`
  - `Task.port_type`, expected to be `sfp`
- The target is expected to be visible from the wrist cameras at task start.
- Board pose, NIC rail position, and NIC card yaw/translation may vary.

The detector must not merely find "an SFP port." It must produce enough
candidate structure for the runtime selector to choose the task-specified module
and port.

## High-Level Architecture

```text
Task + Observation
      |
      v
image preprocessing
      |
      v
SFP candidate detector
      |
      v
candidate grouping + task selector
      |
      v
tri-camera keypoint triangulation
      |
      v
plug-tip standoff target
      |
      v
safe Cartesian velocity controller
      |
      v
/aic_controller/pose_commands
```

The runtime policy should remain inspectable. It should expose internal status
through logs and feedback strings, especially:

- number of detected SFP candidates
- selected module candidate
- selected port name
- detector confidence
- triangulation reprojection error
- estimated standoff distance
- safety-stop reason, if any

## Component 1: Training-Only Proximity Teacher

Create a training-only `ProximityTeacher` derived from the existing
`CheatCode` policy pattern. This teacher uses ground-truth TF to generate safe
approach trajectories and labels. It must never be used as an evaluation policy.

Teacher purpose:

- collect images and labels from the full approach distribution, not only the
  initial camera pose
- generate ACT-style demonstration actions for later fine-tuning
- deliberately create small offset and recovery cases near the target
- stop at a no-contact standoff instead of inserting

Teacher inputs:

- `Task.target_module_name`, `Task.port_name`, and `Task.plug_name`
- training-only target port TF, for example
  `task_board/<target_module_name>/<port_name>_link`
- training-only port entrance TF, for example
  `task_board/<target_module_name>/<port_name>_link_entrance`
- training-only plug-tip TF, for example `<cable_name>/<plug_name>_link`
- official observations, robot state, and wrench for recorded data

Teacher behavior:

1. Wait for target port, entrance, plug-tip, and TCP transforms.
2. Compute the port approach axis from the port link and entrance link.
3. Compute a no-contact standoff target 15-20 mm outside the entrance.
4. Move smoothly from the start pose toward the standoff target.
5. Hold at standoff long enough to record stable near-port observations.
6. Publish zero velocity or a stable hold command before returning.

The teacher must remove or disable the original `CheatCode` insertion descent
loop. It should not intentionally cross the entrance plane, use contact as a
signal, or chase partial insertion.

Trajectory variants:

- nominal approach to the standoff target
- small lateral offsets around the approach axis
- high and low standoff offsets that remain outside the port
- recovery motions that begin slightly off-axis and return to standoff
- mild viewpoint/occlusion variation caused by the plug and gripper moving
  near the target

The offset/recovery data is important. A perfect straight-line teacher creates
an overly clean distribution and does not teach the detector or ACT policy what
to do when the deployed visual estimate is a few millimeters or centimeters
wrong.

## Component 2: Training Data Generation

Create a training-only labeling workflow that runs local Gazebo with
`ground_truth:=true` and the `ProximityTeacher`.

Inputs:

- `/observations` images and camera info
- task fields
- training-only TF frames for target modules and ports
- training-only plug-tip and target distance metadata
- teacher commands/actions
- robot/TCP state, joint state, and wrench
- optional post-episode score summaries from the scoring modules

Labels per frame:

- image timestamp
- camera name: left, center, right
- task fields
- all visible SFP candidate modules
- for each visible candidate module:
  - module identity label for training/debugging only
  - approximate module bounding box or keypoints
  - `sfp_port_0_link` image keypoint
  - `sfp_port_0_link_entrance` image keypoint
  - `sfp_port_1_link` image keypoint
  - `sfp_port_1_link_entrance` image keypoint
  - visibility flag for each keypoint
  - occlusion or low-confidence flag when projection is valid but visual
    evidence is poor
- selected-task target flag:
  - true for `Task.target_module_name + Task.port_name`
  - false for distractor SFP ports
- teacher trajectory metadata:
  - `teacher_mode`
  - `trajectory_variant`
  - `standoff_distance_m`
  - `target_port_frame`
  - `target_entrance_frame`
  - `selected_candidate`
  - `plug_tip_distance_m`
  - `termination_reason`
- action/demo data:
  - commanded 6D Cartesian twist or pose delta
  - command frame
  - teacher target pose or standoff target as metadata
  - safety gate state
- scoring metrics, stored only after an episode or rollout:
  - total score
  - Tier 2 smoothness, duration, efficiency, and penalties
  - Tier 3 proximity or insertion score
  - force penalty flag
  - off-limit contact penalty flag
  - final scorer-reported outcome string, if available

Projection method:

1. Get target port and entrance TF in the training environment.
2. Transform each 3D point into each camera frame using known robot/camera
   geometry and the observation timestamp.
3. Project with the corresponding `CameraInfo`.
4. Reject points behind the camera or outside the image.
5. Store projected labels plus enough metadata to debug projection drift.

Initial data budget:

- Small sweep first: approximately 1k-2k labeled SFP frames collected across
  complete teacher approach trajectories.
- Include board yaw/pose variation, NIC rail translation, both SFP ports,
  target/distractor modules, near-port views, mild gripper/plug occlusion,
  nominal approaches, and offset/recovery examples.
- Expand only after the label overlays and first detector metrics look sane.
- ALways tare the sensors using the recommended methid before starting a new trial.

Artifacts:

- saved images or compressed image references
- JSONL or Parquet label file
- overlay images for manual inspection
- dataset manifest with seed/config/task details
- ACT-compatible demonstration records or a clear conversion path to the
  LeRobot dataset layout

## Component 3: SFP Candidate Detector

Use a small PyTorch heatmap/keypoint model with OpenCV preprocessing. This uses
the dependency surface already present through the repo/LeRobot/ACT stack and
avoids adding a heavy detector runtime dependency for v1.

Model inputs:

- one RGB camera image at a time
- optional task conditioning in v1 through runtime selection rather than direct
  model conditioning

Model outputs:

- heatmaps for SFP port/body/entrance keypoints
- grouping or embedding features that associate port keypoints with a NIC
  candidate
- confidence per candidate keypoint
- optional module bounding box/keypoint heatmaps

Runtime detector output schema:

```text
SfpCandidate:
  camera_name
  module_candidate_id
  module_score
  port_name              # "sfp_port_0" or "sfp_port_1"
  port_score
  port_link_px           # image keypoint for port body/link
  port_entrance_px       # image keypoint for entrance
  approach_axis_px       # optional image-space axis cue
  visibility_score
```

The model should detect all visible SFP candidates, not just the current task
target. This is important for wrong-target rejection and future ACT data quality.

Validation metrics:

- keypoint reprojection error in pixels
- candidate recall for visible SFP ports
- correct grouping of `sfp_port_0` and `sfp_port_1` within each NIC
- correct selected target after task-field selection
- false-positive rate on non-target board regions
- robustness on early, mid-approach, and near-port frames

## Component 4: Task-Based Candidate Selector

Runtime selection should be deterministic and conservative.

Inputs:

- SFP candidate lists from left, center, and right camera images
- `Task.target_module_name`
- `Task.port_name`
- `Task.port_type`
- current policy state and previous selected candidate

Selection behavior:

1. If `Task.port_type != "sfp"`, reject for v1 and return safe success/failure
   without moving aggressively.
2. Group candidates into NIC/module candidates in each camera.
3. Estimate module ordering using image position and learned module-group
   identity confidence.
4. Select the module candidate corresponding to `Task.target_module_name`.
5. Select `Task.port_name` within that module.
6. Require stable selection over several frames before moving.
7. If multiple candidates are plausible, hold position or abort safely.

The exact mapping from `target_module_name` to visual module candidate should be
validated with labeled data before deployment. Do not assume the text
`nic_card_mount_0` is visually visible. It must be inferred from board layout,
rail order, and training labels.

## Component 5: Tri-Camera 3D Estimation

Use the three wrist cameras to estimate a metric target instead of relying only
on center-camera pixel error.

Inputs:

- selected `port_link_px` and `port_entrance_px` from at least two cameras
- `CameraInfo` intrinsics
- static camera extrinsics relative to the robot wrist
- current TCP pose from `Observation.controller_state.tcp_pose`

Outputs:

- `p_base_port_link`
- `p_base_port_entrance`
- estimated approach axis:
  - `axis = normalize(p_base_port_entrance - p_base_port_link)` or the reverse,
    whichever matches the training-time convention
- reprojection error
- triangulation confidence

Required gates:

- at least two camera observations for each keypoint
- reprojection error below threshold
- triangulated point near the expected working volume
- stable target estimate across several frames

The controller should not move forward if triangulation is unstable. It may
perform only very small centering or hold commands while waiting for stable
estimates.

## Component 6: Plug-Tip Standoff Target

Scoring is based on plug-to-port distance, not TCP-to-port distance. The policy
must control the estimated SFP plug tip.

Plug-tip estimate:

```text
T_base_sfp_tip_est = T_base_tcp * T_tcp_sfp_tip_nominal
```

Use the nominal SFP grasp offset from the task/config as the first estimate.
The qualification docs mention small grasp deviations, so the controller should
keep a conservative standoff until this is validated.

No-contact target:

```text
p_standoff = p_base_port_entrance + approach_axis * standoff_distance
```

Initial standoff:

- 15-20 mm for first local scoring runs.
- Do not tune below this until logs confirm no contact, low wrench, and stable
  detector/triangulation estimates.

Future insertion-ready posture:

- The final held pose should place the plug tip near the approach axis, with the
  plug positioned such that a later controller could slide along the approach
  axis.
- V1 does not execute that slide.

## Component 7: Proximity Policy State Machine

Use `aic_model.Policy` and implement
`insert_cable(task, get_observation, move_robot, send_feedback)`.

States:

```text
WAIT_OBS
  Wait for a valid synchronized observation.

DETECT
  Run SFP detector on camera images.
  Reject low-confidence or ambiguous candidates.

SELECT_TARGET
  Select task-specified module and port.
  Require stable selection across frames.

TRIANGULATE
  Estimate port link, entrance, and approach axis in base frame.
  Reject high reprojection error or unstable target.

APPROACH_STANDOFF
  Move estimated plug tip toward standoff target.
  Use conservative velocity clamps and smoothing.

HOLD_SUCCESS
  Hold near standoff with low wrench and stable estimate.
  Publish zero velocity, return success.

ABORT_SAFE
  Stop or retreat slightly, publish zero velocity, return safely.
```

Commanding:

- Use `MotionUpdate` in velocity mode.
- `header.frame_id = "base_link"` for v1.
- Translational velocity from plug-tip error to standoff target.
- Angular velocity zero or near-zero for v1 unless coarse alignment proves
  necessary.
- Exponential moving average on commanded twist.
- Publish zero velocity before returning from the action.

Initial conservative velocity limits:

- lateral/vertical velocity: low enough to avoid overshoot near the board
- forward approach velocity: only enabled when target selection and
  triangulation are stable
- hard clamp all linear components
- hard clamp angular components near zero

Exact numeric gains should be tuned in local Gazebo after the first detector
and triangulation logs are available.

## Safety And Contact Avoidance

The baseline should prefer no score over negative score.

Abort or retreat on:

- force magnitude above a conservative threshold well below the scoring penalty
  threshold
- sudden force derivative spike
- TCP tracking error spike
- low detector confidence after movement starts
- target selection changing between candidates
- triangulation reprojection error above threshold
- estimated plug tip crossing the no-contact boundary
- task timeout approaching

Retreat behavior:

- send zero velocity immediately
- optionally move a small distance away from the estimated port approach axis
- return before `Task.time_limit`

Do not:

- descend into the port entrance
- intentionally touch the task board
- continue moving when target identity is ambiguous
- use scoring contacts or force penalties as policy feedback

## Runtime Packaging

Keep heavy imports inside policy initialization, following the existing
`RunACTOffline` pattern. Avoid top-level imports that slow model discovery.

Recommended runtime assets:

- SFP detector checkpoint
- detector config JSON
- normalization statistics
- camera/triangulation config if not derivable from observation/TF
- policy thresholds config

Recommended environment variable:

```text
AIC_SFP_DETECTOR_PATH=/absolute/path/to/sfp_detector_snapshot
```

The policy should fail clearly during configure if required assets are missing.
It should not download weights at runtime in the submitted container.

The runtime policy package should not include any code path that subscribes to
scoring topics or consumes scoring outputs as observations. Scoring integration
belongs in training/evaluation scripts and experiment analysis only.

## Implementation Discipline

Before implementing any new or nontrivial component, search the repo for an
existing implementation or reusable pattern. This is mandatory for:

- policy classes and `aic_model.Policy` integration
- `MotionUpdate` construction and controller command helpers
- TF lookup and projection utilities
- camera image conversion and preprocessing
- LeRobot dataset recording or conversion
- scoring/evaluation scripts
- logging, overlays, and Foxglove/debug artifacts

Prefer adapting existing repo code over creating duplicate abstractions. Known
starting points include:

- `aic_example_policies/.../CheatCode.py` for training-only TF teacher logic
- `aic_model/aic_model/policies/RunACTOffline.py` for runtime model loading,
  image tensor conversion, and velocity-mode command construction
- `aic_utils/lerobot_robot_aic` for ACT-compatible robot/data workflows
- `aic_scoring` and `docs/scoring.md` for score interpretation
- `aic_adapter` for observation timing and synchronization assumptions

Also prefer open-source PyTorch, torchvision, Hugging Face, or LeRobot
components whenever they fit the problem and can be packaged safely. Do not
reinvent a detector, dataset format, trainer, model checkpoint loader, or
augmentation pipeline if a stable open-source implementation is already
available and compatible with the AIC runtime constraints.

Open-source dependency rule:

- acceptable for training: broader Hugging Face/PyTorch tooling if it improves
  iteration speed
- acceptable for deployment: only dependencies packaged locally, import within
  lifecycle budgets, and avoid runtime downloads
- avoid adding a new dependency when the existing torch/torchvision/OpenCV stack
  can solve the problem cleanly

## Validation Workflow

1. ProximityTeacher sanity:
   - run training Gazebo with `ground_truth:=true`
   - confirm the teacher reaches 15-20 mm standoff
   - confirm it never executes an insertion descent
   - confirm zero or hold command before action completion

2. Label projection sanity:
   - run training Gazebo with `ground_truth:=true`
   - generate labels
   - inspect overlay images from early, mid-approach, and near-port frames
   - verify `sfp_port_0/1` and entrance labels match actual visible ports

3. Demo/action sanity:
   - verify recorded teacher actions use the same command frame planned for
     deployment
   - verify nominal and perturbed trajectories are both present
   - verify recovery data moves back toward the approach axis instead of only
     replaying perfect approaches

4. Detector offline validation:
   - train on the small sweep
   - check held-out keypoint error
   - check candidate recall and wrong-target confusion
   - inspect false positives on board clutter
   - evaluate separately on start, mid-approach, and near-port frames

5. Triangulation validation:
   - triangulate from detector labels first
   - compare against training-only ground truth
   - then triangulate from model predictions
   - inspect reprojection error and 3D stability

6. Dry policy run with frozen motion:
   - run detector and selector live
   - publish logs/feedback only
   - confirm stable target identity from task fields

7. Live no-contact policy run:
   - enable conservative velocity
   - stop at 15-20 mm standoff
   - verify no off-limit contact and low wrench

8. Scoring run:
   - run with `ground_truth:=false`
   - verify no forbidden topics
   - record final score and logs
   - tune thresholds or standoff only after reviewing failures
   - use scoring outputs as run-level metrics, not as policy observations
   - compare score against contact/force logs before accepting a tuned setting

9. Score-driven experiment selection:
   - rank detector/controller checkpoints by held-out rollout score
   - reject any checkpoint with off-limit contact or excessive-force penalties
   - prefer robust positive Tier 3 over higher single-run score
   - keep separate train/tune and held-out seed sets to reduce local overfit

## ACT Data Strategy

This baseline should produce data that is useful for later ACT fine-tuning.

Record for ACT:

- three RGB images
- TCP pose, TCP velocity, TCP error
- joint states
- wrist wrench
- task fields
- commanded 6D Cartesian twist
- detector outputs as metadata
- selected candidate and confidence as metadata
- `teacher_mode`
- `trajectory_variant`
- `standoff_distance_m`
- `target_port_frame`
- `target_entrance_frame`
- `plug_tip_distance_m` as training metadata only
- post-episode score summary as training metadata only
- termination reason

Do not feed detector keypoints into ACT unless the deployed ACT wrapper also
computes those keypoints internally. Otherwise the learned policy would depend
on a training-only feature.

Do not record evaluation-only or forbidden sources into ACT observations.
Training-only ground-truth labels and scoring summaries may be stored as
metadata for debugging, filtering, sample weighting, auxiliary losses, or
checkpoint selection.

Score use should remain episode-level or dataset-level:

- keep or upweight high-score no-contact demonstrations
- downweight or remove episodes with force or off-limit contact penalties
- flag wrong-target and zero-proximity failures for analysis
- select ACT checkpoints by held-out scoring rollouts

Do not train ACT with score, contact state, scoring TF, or scoring topics as
input features unless an equivalent legal signal is computed internally from
official observations at deployment.

## SFP-First Scope Boundaries

Included in v1:

- training-only ProximityTeacher data generation
- nominal and perturbed teacher trajectories
- SFP candidate detection
- SFP target selection
- tri-camera triangulation
- SFP plug-tip standoff control
- no-contact safety gates
- local Gazebo scoring validation for SFP trials
- score-based experiment comparison and dataset filtering

Deferred:

- SC detector and trial 3 support
- partial insertion
- learned ACT deployment
- Isaac Lab data expansion
- diffusion or VLA policies
- contact-rich insertion controller

SC should reuse the same architecture later with an SC-specific detector and
port geometry.

## Open Risks

Wrong module identity:

- `target_module_name` is not printed on the image.
- The model and selector must infer module identity from layout and training
  labels.
- Ambiguous module identity should cause hold/abort, not motion.

Teacher distribution bias:

- A perfect TF-based teacher can make the dataset too clean.
- Include perturbed starts and recovery motions so later ACT training sees
  realistic visual-policy errors.
- Do not let the learned policy depend on training-only TF values.

Plug-tip error:

- The nominal SFP plug-tip offset may be wrong by a few millimeters.
- Conservative standoff reduces contact risk but may reduce proximity score.

Triangulation noise:

- Small port keypoint errors can create large depth errors.
- Require multi-frame smoothing and reprojection gates.

Target visibility:

- The target is expected to be visible at start, but the plug/gripper may
  partially occlude it during approach.
- The controller should keep the last stable target only briefly and stop if
  confidence does not recover.

Scoring threshold:

- A 15-20 mm standoff may be too conservative for positive proximity in some
  initial-distance configurations.
- Reduce standoff only after confirming no contact and low force in local runs.

Score overfitting:

- Local scoring is useful as the scoreboard, but optimizing it too directly can
  overfit to specific seeds or scene quirks.
- Use held-out randomized runs and reject any score improvement that increases
  contact or force risk.
- Never let scoring outputs leak into the deployed policy observation space.

## Review Checklist Before Implementation

- Search the codebase for existing implementations before adding any new
  nontrivial component.
- Prefer open-source Hugging Face, PyTorch, torchvision, or LeRobot components
  when they are compatible with packaging and evaluation constraints.
- Confirm SFP-first is still the desired v1 scope.
- Confirm no-contact proximity remains more important than higher per-trial
  score.
- Confirm `ProximityTeacher` is training-only and never submitted.
- Confirm the teacher collects labels and demonstrations along full approach
  trajectories, not only the initial pose.
- Confirm PyTorch heatmap detector is acceptable over adding YOLO/Ultralytics.
- Confirm tri-camera triangulation is worth the added complexity.
- Confirm initial standoff remains 15-20 mm.
- Confirm scoring modules are used only for training metrics, filtering,
  checkpoint selection, and threshold tuning.
- Confirm data artifacts can be written under a project-owned path and excluded
  from submission if needed.
