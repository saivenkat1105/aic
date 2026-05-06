# Training Data Generator Plan

Last updated: 2026-05-04

## Objective
Run ProximityTeacher data generation with split architecture:
1. Generator is control-plane only.
2. `training_frame_sink.py` handles frame write + online lossless WebP conversion + bin purge.
3. All `ros2` and `python3` commands use `pixi run -- ...`.
4. Use `enter_new_eval.sh` once to create the first eval runtime, then `enter_eval.sh` for all additional terminals.
5. Episode teardown follows organizer sequencing: deactivate `aic_model` -> delete `cable_0`/`task_board` -> reset joints -> next episode re-activates model before action.
6. Lifecycle transitions use retry + long timeout guardrails to prevent transient service stalls from failing episodes.
7. Each episode now spawns a random number of NIC mounts (`1..5`), randomizes each mount pose within organizer limits, and selects exactly one target port for the task while recording all mount/port locations.

## Exact Setup (Eval Container Only)
Terminal 1 (host: create fresh first eval runtime):
```bash
AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_new_eval.sh
```

Terminal 2 (host: join existing eval runtime for bringup):
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

Terminal 3 (host: join existing eval runtime for xacro service):
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

Terminal 4 (host: join existing eval runtime for model):
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

Terminal 5 (host: join existing eval runtime for frame sink):
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
  -p max_pending_frames:=128 \
  -p write_workers:=2 \
  -p convert_workers:=4 \
  -p keep_every_nth_bin:=0 \
  -p images_output_subdir:=images_debug
```

Terminal 6 (host: join existing eval runtime for generator):
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
  -p num_episodes:=500 \
  -p seed:=22000123 \
  -p output_root:=/home/user/training_data/visual_motor_policy/training_dataset \
  -p use_frame_sink:=true \
  -p frame_sink_service_ns:=/training_frame_sink \
  -p frame_sink_require_webp_done:=true \
  -p frame_sink_flush_timeout_s:=120.0 \
  -p deactivate_model_between_episodes:=true \
  -p lifecycle_transition_timeout_s:=60.0 \
  -p lifecycle_transition_retries:=3 \
  -p reset_joints_after_episode:=true
```

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
- `result.json`
- `score_summary.json`
- `images_debug/{left,center,right}/*.webp`
- `images/{left,center,right}/*.bin` only if retained or conversion fails

`scene.json` now includes:
- all spawned NIC mounts with configured pose randomization values
- world-pose estimates for each spawned NIC mount
- world-pose estimates for both SFP ports on each spawned NIC mount
- boolean markers for `is_task_target_module` and `is_task_target_port`

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
