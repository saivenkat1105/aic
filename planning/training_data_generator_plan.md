# Training Data Generator Plan

Last updated: 2026-05-04

## Objective
Build a baseline training data generator for `ProximityTeacher` that:

1. Randomizes scene configuration per episode.
2. Spawns/deletes entities without `aic_engine` orchestration.
3. Runs `/insert_cable` with `aic_model.policies.ProximityTeacher`.
4. Records per-frame observations, actions, and metadata.
5. Captures scoring streams and writes per-episode score artifacts.
6. Produces reproducible experiment outputs for future ACT training.

## Baseline Workflow
1. Enter `aic_eval` container:
   - `bash /home/user/aic/scripts/enter_eval.sh`
2. In `aic_eval` shell #1, launch training Gazebo stack with:
   - `ground_truth:=true`
   - `start_aic_engine:=false`
   - headless options as needed
3. In `aic_eval` shell #2, run one fresh `xacro_expander.py` service with:
   - `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
   - `source /ws_aic/install/setup.bash`
4. In host shell #1, run one `aic_model` with `aic_model.policies.ProximityTeacher`.
5. In eval shell #3, run generator node for `N` episodes:
   - sample scene params
   - delete previous entities
   - expand xacros and spawn `task_board` + `cable_0`
   - tare FT sensor
   - activate lifecycle model if needed
   - send task goal
   - record per-frame data until result/timeout
   - compute and save per-episode score summary
6. After run completes, optionally post-process `.bin` images into WebP/PNG for debugging (offline).
7. Repeat.

### Baseline Run Commands (Clean Version)
Terminal 1 (host: enter eval container):
```bash
bash /home/user/aic/scripts/enter_eval.sh
```

Terminal 2 (`aic_eval` shell #1: bringup, headless):
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
source /ws_aic/install/setup.bash

/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=false \
  spawn_task_board:=false \
  spawn_cable:=false \
  gazebo_gui:=false \
  launch_rviz:=false
```

Terminal 3 (`aic_eval` shell #2: reset + start xacro service):
```bash
export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval


export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
source /ws_aic/install/setup.bash

pkill -f xacro_expander.py || true
sleep 1
ros2 service list | grep /expand_xacro || true

python3 /home/user/aic/aic_utils/aic_training_utils/scripts/xacro_expander.py
```

Terminal 4 (host pixi: ProximityTeacher policy):
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
pixi run -- ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_model.policies.ProximityTeacher
```

Terminal 5 (`aic_eval` shell #3: generator with preflight):
```bash

export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval


source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
python3 -c "from aic_engine_interfaces.srv import ResetJoints; print('ResetJoints import OK')"
/home/user/aic/scripts/run_proximity_generator_eval.sh --ros-args \
  -p num_episodes:=5 \
  -p seed:=42 \
  -p output_root:=/home/user/training_data/visual_motor_policy/debug
```

Async per-episode image postprocess mode (convert to lossless WebP in background and delete `.bin` while next episodes run):
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
/home/user/aic/scripts/run_proximity_generator_eval.sh --ros-args \
  -p num_episodes:=10 \
  -p seed:=42 \
  -p output_root:=/home/user/training_data/visual_motor_policy/debug \
  -p postprocess_webp_after_episode:=true \
  -p postprocess_delete_bin_after_webp:=true \
  -p postprocess_workers:=2 \
  -p postprocess_output_subdir:=images_debug
```

Post-processing (host pixi: convert saved `.bin` to debug images):
```bash
# Convert all episodes to lossless WebP
pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/bin_to_webp_converter.py \
  --run-dir /home/user/training_data/visual_motor_policy/debug/run_<timestamp> \
  --mode all \
  --format webp \
  --lossless

# Convert every 10th episode only
pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/bin_to_webp_converter.py \
  --run-dir /home/user/training_data/visual_motor_policy/debug/run_<timestamp> \
  --mode every_n \
  --every-n 10 \
  --format webp \
  --lossless
```

If `aic_description` lookup fails, verify xacro server environment:
```bash
pgrep -af xacro_expander.py
for p in $(pgrep -f xacro_expander.py); do
  echo "PID=$p"
  tr '\0' '\n' < /proc/$p/environ | grep '^AMENT_PREFIX_PATH='
done
```
Expected:
`AMENT_PREFIX_PATH=/ws_aic/install:/opt/ros/kilted`

### Operational Guardrails
- Keep exactly one `aic_model` process active; duplicate `/aic_model` nodes can break lifecycle service calls.
- Keep exactly one `/expand_xacro` provider active.
- Prefer headless bringup (`gazebo_gui:=false`, `launch_rviz:=false`) for long dataset runs.
- For generator runs that require joint reset via `ResetJoints`, run inside eval runtime and source `/ws_aic/install/setup.bash`.
- Do not run generator from host `pixi run -- python3 ...` for this workflow.
- Use `/home/user/aic/scripts/run_proximity_generator_eval.sh` to enforce preflight import checks and fail fast.
- Do not run `xacro_expander.py` with pixi. Run it in `aic_eval` after sourcing `/ws_aic/install/setup.bash`, otherwise `aic_description` is not discoverable.
- Keep online data generation as `.bin` only for throughput. Convert debug images offline.

## Data To Record
### Frame-level
- Tri-camera images + camera infos
- joint states
- controller state
- wrist wrench
- task fields
- latest command action (`pose_commands` or `joint_commands`)
- image quality metrics (brightness/contrast estimate)
- action latency estimate

### Episode-level
- sampled scene manifest
- task used
- action result message and status
- timing, frame count
- contact events and force peaks
- insertion events
- TF-derived distances (initial/final plug-to-port)
- score summary fields
- run provenance (`seed`, `git commit`, generator version)

## Scoring Artifact Notes
- Generator subscribes to scoring streams (`/scoring/tf`, `/scoring/tf_static`, `/scoring/insertion_event`, contacts, wrench).
- Baseline writes `score_summary.json` per episode with a local score estimate and raw scoring signals.
- These score artifacts are for training analytics and checkpoint selection.

## Output Layout
- `<output_root>/run_<timestamp>/manifest.json`
- `<output_root>/run_<timestamp>/episodes/episode_000001/`
  - `task.json`
  - `scene.json`
  - `frames.jsonl`
  - `score_summary.json`
  - `result.json`
  - `images/{left,center,right}/*.bin`
  - `images_debug/{left,center,right}/*.webp` (optional, generated by post-process converter)

## Async Postprocess Parameters
- `postprocess_webp_after_episode` (bool, default `false`):
  - If `true`, schedule background conversion right after each episode finishes writing.
- `postprocess_delete_bin_after_webp` (bool, default `true`):
  - If `true`, delete per-frame `.bin` after successful `.webp` write.
- `postprocess_workers` (int, default `2`):
  - Number of background postprocess threads.
- `postprocess_output_subdir` (string, default `images_debug`):
  - Per-episode folder for converted images.

Behavior:
- Episode `k+1` starts immediately while episode `k` image conversion/deletion runs in background.
- Generator waits for all queued background jobs before exiting.

## Change Log
- 2026-05-03: Created baseline plan document and fixed required filename `training_data_generator_plan.md`.
- 2026-05-03: Added baseline implementation script `aic_utils/aic_training_utils/scripts/proximity_data_generator.py`.
- 2026-05-03: Updated `aic_training_utils` install/dependency metadata for the new generator script.
- 2026-05-03: Added baseline run commands and documented per-episode output layout.
- 2026-05-03: Added episode-level fault handling so failed episodes still emit result artifacts and run continues.
- 2026-05-03: Validation update: added missing Pixi runtime deps `ros-kilted-ros-gz-interfaces` and `ros-kilted-simulation-interfaces`.
- 2026-05-03: Validation fix: switched TF listener to `spin_thread=False` to avoid shutdown thread exception during smoke tests.
- 2026-05-03: Validation finding: generator entrypoint correctly fails fast when `/expand_xacro` is unavailable (`Required service unavailable: /expand_xacro`).
- 2026-05-03: Validation finding: workspace Pixi environment currently does not include `aic_training_utils`/`aic_bringup` as discoverable ROS packages; end-to-end launch requires eval-side bringup context.
- 2026-05-03: Validation fix: corrected `controller_state.target_mode` parsing to `target_mode.mode` in `proximity_data_generator.py` to prevent callback crash.
- 2026-05-03: Validation fix: replaced deprecated `get_logger().warn(...)` with `get_logger().warning(...)`.
- 2026-05-03: Validation result: completed full one-episode run with eval-side bringup + eval-side xacro expander + host-side `aic_model`; artifacts produced (`manifest.json`, `scene.json`, `task.json`, `frames.jsonl`, `result.json`, `score_summary.json`, image bins).
- 2026-05-03: Validation result: completed sequential 2-episode run (`num_episodes:=2`) without generator crashes; both episodes wrote full artifacts and `success:true` in `result.json`.
- 2026-05-03: Validation note: `/expand_xacro` must be provided by a process using `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and in the eval-side sourced environment to resolve `aic_description` package paths correctly.
- 2026-05-03: Documentation fix: added explicit commands to enter `aic_eval` (`scripts/enter_eval.sh` or `distrobox enter -r aic_eval`) before bringup steps.
- 2026-05-03: Documentation fix: switched generator/xacro commands to `pixi run -- python3 ...` to avoid missing `numpy` in system Python.
- 2026-05-03: Documentation correction: reverted `xacro_expander.py` to eval-side Python with `/ws_aic/install/setup.bash`; pixi-side xacro service cannot resolve `aic_description`.
- 2026-05-03: Documentation update: added explicit "kill all xacro expander services and restart fresh" commands before each run, plus verification commands for `/expand_xacro`.
- 2026-05-03: Documentation update: replaced run section with latest validated full-loop sequence and added concrete `AMENT_PREFIX_PATH` verification for xacro server (`/ws_aic/install:/opt/ros/kilted` expected).
- 2026-05-03: Added offline post-processing script `bin_to_webp_converter.py` to convert saved frame `.bin` images into lossless WebP/PNG (`all`, `every_n`, `episode_list` modes).
- 2026-05-03: Simplified run instructions to reduce redundant terminal steps and separated debug image generation from live data collection.
- 2026-05-03: Added async per-episode postprocess mode in generator: background lossless WebP conversion + optional `.bin` deletion while subsequent episodes continue.
- 2026-05-04: Switched generator runtime instructions to eval-side execution with `/ws_aic/install/setup.bash` and added preflight `aic_engine_interfaces` import guardrail via `scripts/run_proximity_generator_eval.sh`.
