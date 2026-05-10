# C3 Port Detector (Step 3) Implementation

This document describes the implemented Step 3 (`C3`) pipeline:
- detect all visible SFP ports in each camera image (`left`, `center`, `right`)
- output candidate detections and keypoints for C4 semantic selection

Rule compliance note:
- `challenge_rules.md` is not present in this workspace, so competition-rule compliance is unverified here.

## 1. What Was Implemented

### New script
- `aic_utils/aic_training_utils/scripts/port_detector_c3.py`

### Package wiring
- `aic_utils/aic_training_utils/CMakeLists.txt` updated to install `port_detector_c3.py`

### What the script does
Single CLI with 3 subcommands:
1. `inspect`: inspect dataset coverage and projected all-port labels.
2. `train`: train a `torchvision` Keypoint R-CNN detector.
3. `predict-episode`: run inference on an episode and export C3-style JSONL.
4. `preview-labels`: draw auto-generated labels (bbox + keypoints) on random samples for visual QA.
   - Task target port is highlighted with a stronger yellow overlay and `TARGET ...` label.

Common logging flags for all subcommands:
- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--log-every-n <int>` for periodic progress logs in long loops (set `0` to disable periodic logs)

## 2. Model Choice (Simple + Existing)

We use pretrained `torchvision` Keypoint R-CNN (`keypointrcnn_resnet50_fpn`) and fine-tune it.

Why:
- already aligned with your plan (`visual_proximity_components_3_5_plan.md`)
- standard PyTorch/torchvision model, easy to train and debug
- predicts both bbox + keypoints, useful for C5 later

Detected class:
- single foreground class: `sfp_port` (plus background)

Predicted keypoints per port instance:
1. `port_link`
2. `port_entrance`

## 3. Data and Labeling Strategy

## 3.1 Training/Test roots used

Training episodes:
- `/home/user/training_data/visual_motor_policy/updated_training_data/run_20260506_102554/episodes`

Testing episodes:
- `/home/user/training_data/visual_motor_policy/testing_data/run_20260506_193523/episodes`

## 3.2 How labels are created (all ports, not only task target)

Your current `labels_visibility_occlusion.jsonl` focuses on task target labels.  
For C3 we need all visible port candidates.

The script generates all-port supervision directly from per-frame GT:
- `frames.jsonl -> training_gt.all_ports_gt_base`
- per-camera transform: `training_gt.base_to_camera_optical[cam]`
- intrinsics: `camera_info[cam]`

For each frame-camera:
1. project each GT `t_base_port_link_gt` and `t_base_port_entrance_gt` to pixel `uv`
2. compute keypoint visibility flags (`0/1/2` style)
3. build a compact bbox around visible keypoints (`bbox_margin_px`)
4. create one training instance per visible port

This yields labels for all ports in all 3 cameras and matches your C3 objective.

## 3.3 Runtime image format support

The script supports both:
- `images_debug/*/*.webp`
- `images/*/*.bin` with `rgb8` (or `bgr8`) metadata in `frames.jsonl`

Use `--prefer-bin` for inference to mimic evaluation-time `rgb8` arrays.

## 4. CLI Usage

Run with Pixi environment:

```bash
pixi run -- python3 aic_utils/aic_training_utils/scripts/port_detector_c3.py inspect \
  --episodes-root /home/user/training_data/visual_motor_policy/updated_training_data/run_20260506_102554/episodes
```

```bash
pixi run -- python3 aic_utils/aic_training_utils/scripts/port_detector_c3.py preview-labels \
  --episodes-root /home/user/training_data/visual_motor_policy/updated_training_data/run_20260506_102554/episodes \
  --output-dir /home/user/training_data/visual_motor_policy/c3_runs/label_preview_run_001 \
  --num-samples 80 \
  --camera all \
  --seed 7 \
  --log-level INFO \
  --log-every-n 10
```

```bash
pixi run -- python3 aic_utils/aic_training_utils/scripts/port_detector_c3.py train \
  --train-episodes-root /home/user/training_data/visual_motor_policy/updated_training_data/run_20260506_102554/episodes \
  --val-episodes-root /home/user/training_data/visual_motor_policy/testing_data/run_20260506_193523/episodes \
  --output-dir /home/user/training_data/visual_motor_policy/c3_runs/run_001 \
  --epochs 24 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --num-workers 8 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --amp
```

```bash
pixi run -- python3 aic_utils/aic_training_utils/scripts/port_detector_c3.py predict-episode \
  --checkpoint /home/user/training_data/visual_motor_policy/c3_runs/run_001/best.pt \
  --episode-dir /home/user/training_data/visual_motor_policy/testing_data/run_20260506_193523/episodes/episode_000001 \
  --output-jsonl /home/user/training_data/visual_motor_policy/c3_runs/run_001/episode_000001_predictions.jsonl \
  --prefer-bin \
  --score-thresh 0.35
```

## 5. Output Contract for C4 Handoff

`predict-episode` writes JSONL rows:
- `frame_idx`, `obs_stamp`, `task`
- `camera_packets` for `left/center/right`
- each camera packet contains `detections[]`

Each detection contains:
- `score`
- `bbox_xyxy`
- `port_link_uv_px`
- `port_entrance_uv_px`

This is the C3 candidate layer C4 needs.

## 6. Hyperparameters (Defined in Code)

Defaults in script:
- `epochs = 24`
- `batch_size = 2`
- `eval_batch_size = 2`
- `num_workers = 8`
- `lr = 2e-4`
- `weight_decay = 1e-4`
- `lr_step_size = 8`
- `lr_gamma = 0.5`
- `bbox_margin_px = 14.0`
- `score_thresh = 0.35`
- `match_px = 24.0`
- `val_ratio = 0.10` (if separate val root not provided)
- `seed = 7`
- `max_train_samples = 0` (0 = no cap)
- `max_val_samples = 0` (0 = no cap)
- `amp = off by default` (enable with `--amp`)

Hardware guidance (RTX A4000 + 32GB RAM):
- start with `batch_size=2`, `--amp`
- if stable VRAM headroom exists, try `batch_size=3` then `4`
- increase `num_workers` to `10-12` if CPU is not saturated

## 7. Code Walkthrough

Core blocks in `port_detector_c3.py`:

1. **Projection + labeling**
- `_transform_point_base_to_camera`
- `_intrinsics_from_camera_info`
- `_project_point`
- `_build_port_instance`
- `build_samples_index`
- `run_preview_labels` (visual QA overlay utility)

2. **Dataset loader**
- `PortDataset`
- loads WebP or RGB8 `.bin` directly
- returns tensors + detection targets for Keypoint R-CNN

3. **Model**
- `build_model`
- pretrained Keypoint R-CNN with predictor heads replaced for:
  - `num_classes=2` (`background`, `sfp_port`)
  - `num_keypoints=2` (`port_link`, `port_entrance`)

4. **Training + validation**
- `train_one_epoch`
- `evaluate_detector` (simple precision/recall + pixel errors)
- checkpoints:
  - `best.pt` (best recall)
  - `last.pt`
- metrics history in `history.jsonl`

5. **Inference**
- `_predict_image`
- `run_predict_episode`
- outputs C3-style per-camera candidate packets

## 8. What to Label for Better Accuracy

Current pipeline auto-labels port keypoints from GT transforms.  
To improve robustness and C4 behavior, add:

1. **Module-level labels**
- NIC module bbox/keypoints per candidate module
- helps C4 module identity stability

2. **Port identity labels**
- explicit `sfp_port_0` vs `sfp_port_1` class/attribute per detection
- reduces swap errors

3. **Visibility/occlusion quality**
- explicit occlusion severity for keypoints
- not just in/out-of-view

4. **Hard negatives**
- dummy mounts/ports and confusing textures
- near-lookalike geometry

5. **Temporal association IDs**
- consistent track IDs across frames for the same module/port
- helps C4 lock/hold state transitions

## 9. Extra Information Needed for Robust, Accurate C3

For strong evaluation performance, collect or verify:

1. Camera calibration drift checks by run/date.
2. Per-camera motion blur and exposure variation statistics.
3. Domain randomization coverage:
- board pose
- mount translations/yaw
- cable pose/occlusion
- lighting and material changes
4. Failure slices:
- partial visibility
- one-camera-only visibility
- edge-of-frame truncated keypoints
- high distractor density
5. Label-noise audit:
- projected GT vs image overlays sampled by episode phase.

## 10. Known Simplifications (Intentional)

This implementation is intentionally simple:
- single-class detector (`sfp_port`) for C3
- no temporal model in C3
- no heavy custom augmentation stack
- no C4/C5 integration logic inside this script

This keeps Step 3 clear, modular, and easy to run/debug in terminal.
