# Training Campaign Runbook (ProximityTeacher Baseline)
Last updated: 2026-05-03

## 1. Purpose
This runbook defines a strict, repeatable process to generate high-volume training data using `ProximityTeacher` without `aic_engine` orchestration, while keeping data generation isolated from ongoing development experiments.

## 2. Scope and Non-Scope
In scope:
1. Scene randomization by respawn/delete flow.
2. Per-frame data recording.
3. Episode success recording.
4. Raw scoring-signal recording only (not official final score).
5. Offline image conversion for debugging.

Out of scope:
1. Official organizer-equivalent Tier2/Tier3 score computation.
2. Parallelization/multi-sim scale-out.
3. ACT training itself.

## 3. Known Truths (Must Not Be Violated)
1. `score_summary.json` from current generator is a local estimate and not organizer-equivalent.
2. If `xacro_expander.py` is not running in eval environment (`/ws_aic/install/setup.bash`), `aic_description` resolution can fail.
3. Data collection must run with exactly one `/aic_model` process and one `/expand_xacro` provider.

## 4. Campaign Targets
1. Pilot quality check: 200 episodes.
2. Minimum baseline: 1000 episodes.
3. Strong baseline target: 3000 to 5000 episodes.
4. Recommended final target for first serious ACT baseline: 5000 episodes.

## 5. Folder and Naming Policy
1. Output root must be campaign-specific:
`/home/user/training_data/visual_motor_policy/campaign_<YYYYMMDD>_<commit_or_tag>/`
2. Each run uses auto-created `run_<timestamp>/`.
3. Never mix debug test runs and production campaign runs in the same campaign folder.

## 6. Environment Isolation Policy
1. Create a dedicated git worktree pinned to a specific commit for data collection.
2. Do not run feature development from this frozen worktree.
3. Keep `pixi.lock` fixed during campaign.
4. Use dedicated terminal/tmux session for campaign only.
5. Do not update dependencies mid-campaign.

## 7. Pre-Flight Checklist (Hard Gate)
Pass all before starting:
1. `RMW_IMPLEMENTATION=rmw_zenoh_cpp` is set in all relevant terminals.
2. `ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'` is set.
3. Exactly one `/expand_xacro` service exists.
4. Exactly one `/aic_model` node exists.
5. `xacro_expander.py` process has `AMENT_PREFIX_PATH=/ws_aic/install:/opt/ros/kilted`.
6. Generator output root path exists and is writable.
7. Disk space is sufficient for target episode count.
8. Previous stale entities/processes are cleaned.

## 8. Runtime Terminals and Commands
Terminal A (enter eval):
1. `bash /home/user/aic/scripts/enter_eval.sh`

Terminal B (eval bringup):
1. `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`
2. `export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'`
3. `source /ws_aic/install/setup.bash`
4. `/entrypoint.sh ground_truth:=true start_aic_engine:=false spawn_task_board:=false spawn_cable:=false gazebo_gui:=false launch_rviz:=false`

Terminal C (eval xacro):
1. `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`
2. `export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'`
3. `source /ws_aic/install/setup.bash`
4. `pkill -f xacro_expander.py || true`
5. `python3 /home/user/aic/aic_utils/aic_training_utils/scripts/xacro_expander.py`

Terminal D (host model):
1. `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`
2. `export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'`
3. `pixi run -- ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_model.policies.ProximityTeacher`

Terminal E (host generator):
1. `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`
2. `export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'`
3. `pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/proximity_data_generator.py --ros-args -p num_episodes:=<N> -p seed:=<S> -p output_root:=<CAMPAIGN_ROOT>`

## 9. Episode-Level Acceptance Criteria
Every episode directory must include:
1. `scene.json`
2. `task.json`
3. `frames.jsonl`
4. `result.json`
5. `images/left/*.bin`
6. `images/center/*.bin`
7. `images/right/*.bin`

Episode is valid if:
1. `result.json` exists.
2. `frames.jsonl` has at least one frame.
3. Image entries in `frames.jsonl` resolve to existing `.bin` files.
4. No JSON parse errors in artifacts.

## 10. Score Handling Policy (Temporary Correctness Mode)
1. Do not use local `score_summary.json` as final quality metric.
2. Treat score as diagnostics only.
3. Use these campaign metrics instead:
   1. Episode success rate.
   2. Median frame count.
   3. Port-visibility rate in sampled frames.
   4. Failure reason distribution from `result.json`.

## 11. Port Visibility Quality Protocol (Step 3 Feasibility)
1. Sample at least 50 episodes per 1000 episodes.
2. For each sampled episode, inspect multi-view frames around approach and insertion.
3. Mark as usable if port is visible in at least one camera in at least one key phase frame.
4. If usability rate falls below 85%, pause campaign and adjust randomization/camera strategy.

## 12. Failure Triage (Symptom -> Cause -> Action)
1. `aic_description not found` -> xacro not in eval-sourced env -> restart Terminal C with `/ws_aic/install/setup.bash`.
2. `/expand_xacro` missing -> xacro service down -> relaunch Terminal C.
3. Service timeouts for `/aic_model/get_state` -> duplicate or stale model process -> ensure single model process only.
4. Very low frame counts repeatedly -> early action exits -> inspect `result.json` messages and scene randomization bounds.
5. All local scores stuck at 1 -> expected with current local estimate path when Tier3 signals are incomplete -> ignore as ranking signal.

## 13. Debug Image Conversion (Offline Only)
1. Keep live run storage in `.bin` for throughput.
2. Convert after run:
   1. `pixi run -- python3 /home/user/aic/aic_utils/aic_training_utils/scripts/bin_to_webp_converter.py --run-dir <RUN_DIR> --mode all --format webp --lossless`
3. For large runs, convert subset only:
   1. `--mode every_n --every-n <K>`

## 14. Campaign Completion Gate
Campaign passes when all are true:
1. Target episode count reached.
2. Artifact completeness > 99%.
3. Success/failure distribution is plausible (not pathological all-fail or trivial all-success).
4. Visibility protocol passes threshold.
5. Run manifest records seeds, commit SHA, and command parameters.

## 15. What To Change Later (Not In This Baseline)
1. Replace local score summary with organizer-aligned scoring integration path.
2. Add parallel campaign workers only after single-run stability is proven.
3. Add automatic dataset QC reports.

## 16. Operator Checklist (Dumb-Agent Mode)
1. Start five terminals exactly as defined.
2. Run pre-flight checks.
3. Launch one pilot run (200 episodes).
4. Verify artifact completeness.
5. Run baseline block (1000 episodes).
6. Run remaining blocks until 3000 to 5000.
7. Run offline WebP conversion for sampled episodes.
8. Archive manifests and logs.
