# Training Data Generator Plan

Last updated: 2026-05-03

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
   - or `distrobox enter -r aic_eval`
2. In the `aic_eval` shell, launch training Gazebo stack with:
   - `ground_truth:=true`
   - `start_aic_engine:=false`
   - headless options as needed
3. In the same eval shell, export:
   - `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
   - `ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'` (baseline stability mode)
4. In the same eval shell, run `xacro_expander.py` (so `/expand_xacro` resolves eval-side package shares such as `aic_description`).
5. Launch `aic_model` with policy `aic_model.policies.ProximityTeacher` from host pixi shell (single instance only).
6. Run generator node for `N` episodes from eval shell:
   - sample scene params
   - delete previous entities
   - expand xacros and spawn `task_board` + `cable_0`
   - tare FT sensor
   - activate lifecycle model if needed
   - send task goal
   - record per-frame data until result/timeout
   - compute and save per-episode score summary
7. Repeat.

### Baseline Run Commands (Latest Validated Loop)
Terminal 1 (host: fresh eval entry):
```bash
bash /home/user/aic/scripts/enter_eval.sh
```

Terminal 2 (`aic_eval`: bringup, headless):
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

Terminal 3 (`aic_eval`: reset + start xacro service):
```bash
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

Terminal 5 (host pixi: generator):
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/proximity_data_generator.py --ros-args \
  -p num_episodes:=5 \
  -p seed:=42 \
  -p output_root:=/home/user/training_data/visual_motor_policy/debug
```

Pre-run verification (recommended in `aic_eval`):
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
source /ws_aic/install/setup.bash

ros2 service list | grep /expand_xacro
ros2 service info /expand_xacro
```

If `aic_description` lookup still fails, verify xacro server environment:
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
- Use `pixi run -- python3 ...` for host-side Python scripts to guarantee `numpy` and ROS Python dependencies are available.
- Do not run `xacro_expander.py` with pixi. Run it in `aic_eval` after sourcing `/ws_aic/install/setup.bash`, otherwise `aic_description` is not discoverable.

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
