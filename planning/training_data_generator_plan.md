# Training Data Generator Plan

Last updated: 2026-05-06

## Objective
Run ProximityTeacher data generation with split architecture:
1. Generator is control-plane only.
2. `training_frame_sink.py` handles frame write + online lossless WebP conversion + bin purge + async visibility-label post-processing.
3. All `ros2` and `python3` commands use `pixi run -- ...`.
4. Use `enter_new_eval.sh` once to create the first eval runtime, then `enter_eval.sh` for all additional terminals.
5. Episode teardown follows organizer sequencing: deactivate `aic_model` -> delete `cable_0`/`task_board` -> reset joints -> next episode re-activates model before action.
6. Lifecycle transitions use retry + long timeout guardrails to prevent transient service stalls from failing episodes.
7. Each episode now spawns a random number of NIC mounts (`1..5`), randomizes each mount translation + yaw within organizer limits, and selects exactly one target port for the task while recording all mount/port locations.
8. Each frame capture now records all metadata required for C3/C4/C5 supervision, including per-frame camera calibration, controller kinematics terms (`tcp_velocity`, `tcp_error`), and training-only GT port transforms.
9. Visibility/occlusion tags are generated as a required post-processing step from stored GT geometry + camera projection consistency.
10. High-throughput mode keeps observation callback minimal and shifts heavy work to worker/post-processing queues.

## Exact Start Procedure (Eval Container Only)
Run in this exact order.

Terminal 1 (host: create fresh eval runtime once):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_new_eval.sh
```

Terminal 2 (host: join runtime for sim bringup):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh
```
Then run:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=false \
  spawn_task_board:=false \
  spawn_cable:=false \
  gazebo_gui:=false \
  launch_rviz:=false
```

Terminal 3 (host: join runtime for xacro service):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh
```
Then run:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

pkill -f xacro_expander.py || true
python3 /home/user/aic/aic_utils/aic_training_utils/scripts/xacro_expander.py
```

Terminal 4 (host: join runtime for ProximityTeacher model):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh
```
Then run:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

pixi run -- ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_model.policies.ProximityTeacher
```

Terminal 5 (host: join runtime for training frame sink):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh
```
Then run:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/training_frame_sink.py --ros-args \
  -p observation_qos:=sensor_data \
  -p max_pending_frames:=128 \
  -p write_workers:=2 \
  -p convert_workers:=1 \
  -p postprocess_workers:=1 \
  -p postprocess_visibility_labels:=true \
  -p postprocess_labels_filename:=labels_visibility_occlusion.jsonl \
  -p defer_gt_to_postprocess:=true \
  -p drop_convert_when_busy:=true \
  -p wait_for_postprocess_on_stop:=false \
  -p keep_every_nth_bin:=10 \
  -p images_output_subdir:=images_debug
```

Terminal 6 (host: join runtime for generator):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh
```
Then run:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

/home/user/aic/scripts/run_proximity_generator_eval.sh --ros-args \
  -p num_episodes:=100 \
  -p seed:=467 \
  -p output_root:=/home/user/training_data/visual_motor_policy/testing_data \
  -p use_frame_sink:=true \
  -p frame_sink_service_ns:=/training_frame_sink \
  -p frame_sink_require_webp_done:=true \
  -p frame_sink_flush_timeout_s:=120.0 \
  -p deactivate_model_between_episodes:=true \
  -p lifecycle_transition_timeout_s:=60.0 \
  -p lifecycle_transition_retries:=3 \
  -p reset_joints_after_episode:=true
```

## Postprocess Ownership
1. In split mode (`use_frame_sink:=true`), `training_frame_sink.py` owns image conversion and post-processing.
2. Generator-side `postprocess_webp_after_episode` is ignored in split mode by design.
3. Post-processing runs asynchronously after each episode stop and does not block the next episode by default.
4. In throughput mode (`defer_gt_to_postprocess:=true`), per-frame GT aliases are hydrated during post-processing to reduce callback overhead.
5. In throughput mode (`drop_convert_when_busy:=true`), conversion jobs may be skipped under heavy load to protect observation ingestion rate.

## Build Step (Only After Interface/Script Changes)
In one joined eval terminal:
```bash
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

pixi run -- colcon build --packages-up-to aic_training_interfaces aic_training_utils
source install/setup.bash
```

## Expected Output Per Episode
- `scene.json`
- `task.json`
- `frames.jsonl`
- `labels_visibility_occlusion.jsonl`
- `result.json`
- `score_summary.json`
- `images_debug/{left,center,right}/*.webp`
- `images/{left,center,right}/*.bin` only if retained or conversion fails

`scene.json` now includes:
- all spawned NIC mounts with configured pose randomization values
- world-pose estimates for each spawned NIC mount
- world-pose estimates for both SFP ports on each spawned NIC mount
- boolean markers for `is_task_target_module` and `is_task_target_port`

`frames.jsonl` must now include (required for C3/C4/C5 training):
- frame identity: `episode_id`, `frame_idx`, `obs_stamp`
- task identity: `task_id`, `target_module_name`, `port_name`, `port_type`, `plug_type`
- per-camera image packet (`left/center/right`):
  - existing image metadata (`path`, `stamp`, `frame_id`, `width`, `height`, `encoding`, `step`)
  - full per-frame `CameraInfo`: `distortion_model`, `d`, `k`, `r`, `p`, `binning_x`, `binning_y`, `roi`
- controller state:
  - existing `tcp_pose`
  - required additions `tcp_velocity` and `tcp_error`
- robot and force state: `joint_states`, `wrench`
- training-only GT transforms per frame:
  - `t_base_target_port_link_gt`
  - `t_base_target_port_entrance_gt`
  - optional full visible-port GT list for distractor-aware supervision

Guaranteed per-frame aliases (top-level):
- `tcp_velocity`
- `tcp_error`
- `left_camera_info`
- `center_camera_info`
- `right_camera_info`
- `t_base_target_port_link_gt`
- `t_base_target_port_entrance_gt`

`labels_visibility_occlusion.jsonl` must be produced per episode during post-processing:
- per camera, per labeled keypoint (`port_link`, `port_entrance`, auxiliary landmarks):
  - `visibility`: `visible|occluded|out_of_fov|truncated`
  - projection diagnostics: projected pixel, in-bounds flag, positive-depth flag
  - optional reprojection residual field for quality auditing

Throughput-mode note:
- GT aliases (`t_base_target_port_link_gt`, `t_base_target_port_entrance_gt`) may appear as `null` during live write and then be filled by post-processing.

## Verification Commands
Run these after generator starts:
```bash
pixi run -- ros2 service info /training_frame_sink/start_episode_capture
pixi run -- ros2 service info /training_frame_sink/stop_episode_capture
pixi run -- ros2 service info /training_frame_sink/capture_status
pixi run -- ros2 service call /training_frame_sink/capture_status aic_training_interfaces/srv/CaptureStatus "{episode_id: ''}"
```

After at least one episode:
```bash
RUN_DIR=$(ls -dt /home/user/training_data/visual_motor_policy/updated_training_data/run_* | head -n 1)
EP_DIR=$(ls -dt "$RUN_DIR"/episodes/episode_* | head -n 1)
head -n 1 "$EP_DIR/frames.jsonl" | jq '{
  frame_idx,
  has_tcp_velocity: (.tcp_velocity != null),
  has_tcp_error: (.tcp_error != null),
  has_left_camera_info: (.left_camera_info != null),
  has_center_camera_info: (.center_camera_info != null),
  has_right_camera_info: (.right_camera_info != null),
  has_gt_port_link: (.t_base_target_port_link_gt != null),
  has_gt_port_entrance: (.t_base_target_port_entrance_gt != null)
}'
test -f "$EP_DIR/labels_visibility_occlusion.jsonl" && echo "labels file exists"
pixi run -- ros2 service call /training_frame_sink/capture_status aic_training_interfaces/srv/CaptureStatus "{episode_id: ''}"
```

## Hard Guardrails
1. Always start with `enter_new_eval.sh` once, then only `enter_eval.sh` for additional terminals.
2. Use `pixi run --` for every `ros2` and `python3` command.
3. Keep exactly one `/expand_xacro` and one `/aic_model`.
4. Keep frame sink running before generator starts.
5. Use eval-sourced shells (`/ws_aic/install/setup.bash`) in all terminals.
6. Keep `reset_joints_after_episode:=true` and do not bypass model deactivation if you want organizer-equivalent reset behavior.
7. If lifecycle services are unstable in a specific run, temporary fallback is `-p deactivate_model_between_episodes:=false` (reset still runs, but this is less organizer-equivalent).

## Change Log
- 2026-05-04: Updated plan to enforce `enter_new_eval.sh` first, `enter_eval.sh` for additional terminals, and `pixi run --` for all `ros2`/`python3` commands.
- 2026-05-04: Updated runtime behavior to organizer-aligned per-episode reset ordering (model deactivate, entity delete, joint reset, re-activate on next episode).
- 2026-05-04: Added lifecycle transition robustness (`lifecycle_transition_timeout_s`, `lifecycle_transition_retries`) and explicit `deactivate_model_between_episodes` switch.
- 2026-05-06: Updated generator to randomize NIC mount count per episode (`1..5`), keep one task target port, and store all mount/port locations for distractor-aware training.
- 2026-05-06: Updated plan to require per-frame `CameraInfo`, `tcp_velocity`, `tcp_error`, and training-only GT `port_link`/`port_entrance` transforms in `frames.jsonl`.
- 2026-05-06: Added required post-processing output `labels_visibility_occlusion.jsonl` and made visibility/occlusion tagging a mandatory data-prep step for C3/C4/C5.
- 2026-05-06: Locked exact startup order (6 terminals), clarified sink ownership of async post-processing, and added explicit runtime verification commands for per-frame fields.
- 2026-05-06: Added high-throughput sink settings (`observation_qos:=sensor_data`, deferred GT hydration, and drop-on-convert-backlog policy) to maximize observation ingestion rate under disk constraints.
