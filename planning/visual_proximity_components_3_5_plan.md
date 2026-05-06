# Visual Proximity Baseline Implementation Plan (Components 3-5)

## 1. Title and Scope

This document defines a decision-complete implementation plan for Components 3, 4, and 5 of the `visual_proximity_baseline_workflow`.

Scope:
- Component 3 (`C3`): detect all visible SFP port candidates in each camera image.
- Component 4 (`C4`): select the task-correct `(target_module_name, port_name)` from candidate sets.
- Component 5 (`C5`): estimate selected target port 3D location and approach axis with respect to the robot.

Final objective:
- Produce robust, safety-gated target localization with respect to `base_link` and `gripper/tcp` so downstream proximity motion can reliably move toward the requested SFP port.

Out of scope (this document):
- Contact-rich insertion controller.
- SC trial implementation details (except interface compatibility).
- End-to-end policy training from scratch.

---

## 2. Critical Review

### Why this decomposition is correct

The decomposition `C3 -> C4 -> C5` separates failure modes cleanly:
- `C3` owns perception quality (what is visible and where in pixels).
- `C4` owns semantics (which module/port the task requires).
- `C5` owns geometry (where the selected target is in robot coordinates).

This prevents hidden coupling where wrong semantic selection can be masked by numerically stable triangulation.

### Main risks

1. Semantic ID drift
- `nic_card_mount_i` misassignment under occlusion, perspective distortion, and distractors.

2. Port identity swap
- confusion between `sfp_port_0` and `sfp_port_1` on the same NIC face.

3. Calibration and timing mismatch
- per-frame keypoints must be tied to the correct camera intrinsics/extrinsics and robot pose timestamp.

4. Monocular depth ambiguity
- one-camera estimates can cause wrong-depth approach unless strictly gated.

5. Training/evaluation boundary leakage
- training-only TF and score internals must never become evaluation-time policy inputs.

### Core mitigation strategy

- Use explicit contracts between components.
- Enforce conservative lock/hold/abort gates.
- Make `C5` the single source of truth for fused 3D geometry.
- Log enough metadata per frame to diagnose all 3 layers independently.

---

## 3. Locked Decisions

These decisions are fixed for v1 unless explicitly revised:

1. `C3` model baseline
- Use pretrained `torchvision` Keypoint R-CNN fine-tune.

2. Dataset strategy
- Use distractor-rich dataset (including dummy mounts/ports).
 - Include mandatory preprocessing-generated visibility/occlusion tags from stored GT transforms.

3. Camera metadata granularity
- Store full `CameraInfo` every frame for all cameras.

4. 3D ownership boundary
- `C5` is the only component that emits fused 3D port geometry.

5. Monocular behavior
- Allow controlled monocular motion only to recover multi-camera visibility.
- Do not permit normal forward approach on weak monocular geometry.

6. Dummy object labels
- Use explicit dummy classes (not background-only).

7. Runtime interface style
- Internal dataclass/JSON contracts first; map to ROS standard messages later.

8. Gate philosophy
- Conservative safety-first thresholds for initial integration.

9. Graph grounding
- Regenerate and use graph context as part of implementation workflow (`graphify update .` before code-level integration work).

---

## 4. Component Plans

## 4.1 Component 3 (C3): SFP Candidate Detector

### Objective

For each camera (`left`, `center`, `right`), detect all visible NIC module candidates and both SFP ports (`sfp_port_0`, `sfp_port_1`) with keypoints:
- `port_link`
- `port_entrance`

### Model

- Architecture: Keypoint R-CNN with pretrained backbone.
- Inputs: RGB frame per camera.
- Outputs per module candidate:
  - module bbox and confidence
  - keypoints for both ports and entrances
  - per-keypoint confidence and uncertainty proxy
  - optional auxiliary landmarks for downstream semantic mapping

### Label requirements

Per frame-camera sample:
- module bbox
- `sfp_port_0_link` and `sfp_port_0_entrance`
- `sfp_port_1_link` and `sfp_port_1_entrance`
- visibility labels for each keypoint
- dummy class annotations for distractor mounts/ports

Visibility/occlusion labels are not hand-entered by default:
- they must be generated in preprocessing from stored per-frame GT transforms + per-frame camera projection checks.

### Runtime output schema (owned by C3)

```json
{
  "obs_stamp": 0.0,
  "model_id": "c3_kprcnn_v1",
  "calibration_id": "calib_hash_or_version",
  "camera_packets": [
    {
      "camera_name": "left",
      "image_stamp": 0.0,
      "frame_id": "left_camera/optical",
      "width": 1152,
      "height": 1024,
      "distortion_model": "plumb_bob",
      "k": [0.0],
      "d": [0.0],
      "r": [0.0],
      "p": [0.0],
      "module_candidates": [
        {
          "track_id": 0,
          "track_age": 0,
          "track_state": "new|confirmed|lost",
          "module_bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
          "module_score": 0.0,
          "module_uncertainty": 0.0,
          "auxiliary_landmarks": [
            {
              "landmark_type": "board_edge|mount_corner|heatsink_corner",
              "uv_px": [0.0, 0.0],
              "confidence": 0.0,
              "visibility": "visible|occluded|out_of_fov|truncated"
            }
          ],
          "ports": [
            {
              "port_name_pred": "sfp_port_0|sfp_port_1|unknown",
              "port_link_uv_px": [0.0, 0.0],
              "port_link_conf": 0.0,
              "port_link_sigma_px": [0.0, 0.0],
              "port_link_visibility": "visible|occluded|out_of_fov|truncated",
              "port_entrance_uv_px": [0.0, 0.0],
              "port_entrance_conf": 0.0,
              "port_entrance_sigma_px": [0.0, 0.0],
              "port_entrance_visibility": "visible|occluded|out_of_fov|truncated",
              "occlusion_flag": false
            }
          ]
        }
      ]
    }
  ],
  "diagnostics": {
    "latency_ms_total_3cam": 0.0,
    "dropped_camera_count": 0
  }
}
```

### C3 acceptance gates

- Visible-keypoint recall >= 95%.
- Port swap error (`port_0` vs `port_1`) <= 3%.
- Mean keypoint error <= 8 px and p95 <= 14 px.
- False candidate inflation controlled on distractor scenes.
- Total 3-camera inference <= 120 ms on target hardware.

---

## 4.2 Component 4 (C4): Task-Based Semantic Selector

### Objective

Given all C3 candidates plus task fields:
- `target_module_name` (e.g., `nic_card_mount_2`)
- `port_name` (`sfp_port_0` or `sfp_port_1`)

select the task-correct candidate deterministically and safely.

### Strategy

Hybrid deterministic semantic assignment:
- track candidates temporally
- use board/mount geometry priors
- fuse cross-camera evidence
- resolve identity with deterministic cost-based assignment
- consume preprocessing-generated visibility/occlusion tags as reliability priors during selector lock/hold decisions.

### Selector state machine

- `OBSERVE`
- `LOCKING`
- `LOCKED`
- `HOLD`
- `ABORT_SAFE`

### Conservative default gates

- lock requires stable `(module, port)` for `N_lock` consecutive frames.
- require >=2 camera support for selected keypoints.
- enforce semantic margin (best vs second best).
- enter `HOLD` on ambiguity or rapid ID flip.
- enter `ABORT_SAFE` if hold persists past timeout.

### Runtime output schema (owned by C4)

```json
{
  "obs_stamp": 0.0,
  "task_id": "task_001",
  "selector_state": "OBSERVE|LOCKING|LOCKED|HOLD|ABORT_SAFE",
  "lock_id": "selector_lock_uuid",
  "lock_age_frames": 0,
  "target_module_name": "nic_card_mount_0",
  "target_port_name": "sfp_port_0",
  "semantic_confidence": 0.0,
  "id_margin": 0.0,
  "selected_measurements": [
    {
      "camera_name": "left",
      "image_stamp": 0.0,
      "frame_id": "left_camera/optical",
      "module_track_id": 0,
      "port_link_uv_px": [0.0, 0.0],
      "port_link_conf": 0.0,
      "port_link_sigma_px": [0.0, 0.0],
      "port_link_visibility": "visible|occluded|out_of_fov|truncated",
      "port_entrance_uv_px": [0.0, 0.0],
      "port_entrance_conf": 0.0,
      "port_entrance_sigma_px": [0.0, 0.0],
      "port_entrance_visibility": "visible|occluded|out_of_fov|truncated"
    }
  ],
  "diagnostics": {
    "id_switch_count_last_1s": 0,
    "gate_fail_reasons": []
  }
}
```

### C4 acceptance gates

- semantic target selection accuracy >= 98% on labeled validation set.
- low target-ID flip rate during approach windows.
- no forward-approach permission when state is `HOLD` or `ABORT_SAFE`.

---

## 4.3 Component 5 (C5): Triangulated 3D Port Estimation

### Objective

From C4-selected 2D measurements and camera calibration, produce stable 3D estimates of:
- `p_base_port_link`
- `p_base_port_entrance`
- approach axis and standoff target

### Geometry architecture

Primary mode:
- weighted multi-view triangulation using >=2 cameras.

Monocular recovery mode:
- allow limited motion aimed at reacquiring second camera view.
- do not allow normal forward approach on weak monocular depth.

Training/validation dependency:
- C5 validation must use preprocessing-generated visibility/occlusion tags to partition metrics by visibility regime (visible, partially occluded, out-of-fov/truncated).

### Ownership rule

`C5` is the only component publishing fused 3D target geometry and covariance.

### Runtime output schema (owned by C5)

```json
{
  "obs_stamp": 0.0,
  "task_id": "task_001",
  "status_code": "VALID_3CAM|VALID_2CAM|REACQUIRE_1CAM|STALE_HOLD|INVALID",
  "frame_id_base": "base_link",
  "frame_id_tcp": "gripper/tcp",
  "target_module_name": "nic_card_mount_0",
  "target_port_name": "sfp_port_0",
  "p_base_port_link_m": [0.0, 0.0, 0.0],
  "p_base_port_entrance_m": [0.0, 0.0, 0.0],
  "u_base_port_outward": [0.0, 0.0, 1.0],
  "u_base_port_inward": [0.0, 0.0, -1.0],
  "p_tcp_port_link_m": [0.0, 0.0, 0.0],
  "p_tcp_port_entrance_m": [0.0, 0.0, 0.0],
  "p_base_standoff_m": [0.0, 0.0, 0.0],
  "sigma_link_3x3_m2": [0.0],
  "sigma_entrance_3x3_m2": [0.0],
  "sigma_axis_rad": 0.0,
  "reproj_rmse_px": 0.0,
  "ray_angle_deg": 0.0,
  "cameras_used": ["left", "center"],
  "confidence_0_1": 0.0
}
```

### C5 acceptance gates

- 2+ camera mode must meet tight geometric error and jitter targets against training-time GT validation.
- monocular recovery mode must not trigger unsafe forward movement.
- robust recovery to 2-camera valid mode within configured timeout on occlusion scenarios.

---

## 5. Cross-Component Interfaces

### Canonical `C3 -> C4` contract

Required fields:
- `obs_stamp`, `calibration_id`
- per camera: `image_stamp`, `frame_id`, `width`, `height`, intrinsics arrays
- candidate tracks (`track_id`, `track_state`, `track_age`)
- per-port keypoints and uncertainties
- visibility and occlusion flags

### Canonical `C4 -> C5` contract

Required fields:
- semantic selection: `selector_state`, `target_module_name`, `target_port_name`, `semantic_confidence`, `lock_id`, `lock_age_frames`
- selected per-camera 2D measurements with uncertainty
- no fused 3D in this message

### Ownership boundary (hard invariant)

- `C3`: 2D perception authority.
- `C4`: semantic selection authority.
- `C5`: fused 3D geometry authority.

No component outside `C5` may publish final `p_base_port_*` geometry for control.

---

## 6. Dataset Metadata Specification

This section defines exact metadata to populate in the new distractor-rich dataset.

## 6.1 Per-frame fields

1. Frame identity:
- `episode_id`
- `frame_idx`
- `obs_stamp`

2. Task context:
- `task_id`
- `target_module_name`
- `target_port_name`
- `port_type`
- `plug_type`

3. Camera packets (for each of left/center/right):
- `image_path`
- `image_stamp`
- `frame_id`
- `width`, `height`, `encoding`
- full `camera_info`:
  - `distortion_model`, `d`, `k`, `r`, `p`, `binning_x`, `binning_y`, `roi`

4. Robot/controller state:
- `tcp_pose`
- `tcp_velocity`
- `tcp_error`
- `joint_states`
- `wrist_wrench`

5. C3 outputs:
- all candidate tracks and per-port keypoint outputs
- visibility and uncertainty per keypoint

6. C4 outputs:
- selector state
- semantic confidence
- selected camera measurements
- gate fail reasons (if any)

7. Training-only GT (when available):
- `t_base_target_port_link_gt`
- `t_base_target_port_entrance_gt`
- GT projected keypoints per camera for audit

8. Preprocessing labels (required for C3/C4/C5 supervision):
- `visibility` and `occlusion` tags for each supervised keypoint
- projection validity flags (in-bounds, positive depth)
- optional per-keypoint reprojection diagnostics

## 6.2 Per-episode fields

- `scene.json` (board and component placements)
- calibration snapshot reference
- episode-level calibration hash/version
- `labels_visibility_occlusion.jsonl` (generated preprocessing artifact)
- outcome summary (`success/failure`, reason)
- score summary (training analytics only)

## 6.3 Per-run fields

- run-level manifest
- calibration file copies or immutable references
- data schema version
- git commit hash
- generator config
- preprocessing config/version (visibility/occlusion labeling rules)

## 6.4 Required Preprocessing Pipeline (Must-Do)

1. Input artifacts:
- `frames.jsonl` with per-frame `CameraInfo`, `tcp_pose`, `tcp_velocity`, `tcp_error`, and GT target transforms.

2. Per-frame projection pass:
- project GT `port_link` and `port_entrance` into each camera using per-frame calibration metadata.
- compute `in_bounds`, `positive_depth`, and view consistency fields.

3. Label generation:
- assign keypoint `visibility`/`occlusion` tags (`visible|occluded|out_of_fov|truncated`).
- write standardized output `labels_visibility_occlusion.jsonl`.

4. Quality audit:
- run overlay checks on sampled frames across early/mid/near-port phases.
- fail preprocessing if schema or projection validity checks regress.

---

## 7. Validation and Acceptance

## 7.1 Offline validation matrix

| Layer | Validation target | Key metrics | Pass criteria |
|---|---|---|---|
| C3 | 2D detection/keypoints on distractor-rich set | recall, keypoint error, port swap rate | gates in Section 4.1 |
| C4 | semantic correctness under occlusion/distractors | module/port accuracy, ID switch rate | gates in Section 4.2 |
| C5 | geometric accuracy and stability | 3D error, reprojection RMSE, jitter | gates in Section 4.3 |
| C3+C4+C5 | contract consistency | schema completeness, timestamp consistency, calibration linkage | zero contract violations |

## 7.2 Integrated simulation validation

Test categories:
1. Nominal visibility, no distractor ambiguity.
2. High distractor density.
3. Partial occlusion during approach.
4. One-camera temporary visibility with recovery.
5. Near-port clutter and perspective skew.

Required behavior:
- no unsafe forward motion on weak geometry.
- deterministic hold/abort on semantic ambiguity.
- stable convergence to standoff targets when valid.

## 7.3 Failure handling requirements

- every abort path must return explicit reason code.
- C4 ambiguity must propagate to C5 as non-advance states.
- C5 invalid geometry must propagate to control gating immediately.

---

## 8. Blind Spots and Mitigations

1. Dataset too clean
- Risk: overestimates robustness.
- Mitigation: enforce occlusion and distractor coverage quotas.

2. Hidden calibration drift
- Risk: triangulation bias.
- Mitigation: per-frame camera info + calibration hash + reprojection audits.

3. Silent semantic errors
- Risk: robot approaches wrong mount with plausible geometry.
- Mitigation: explicit semantic confidence margins and ID switch monitoring.

4. Monocular over-trust
- Risk: depth error causes unsafe near-board moves.
- Mitigation: monocular mode restricted to reacquisition maneuvers and conservative velocity clamps.

5. Boundary leakage
- Risk: training-only signals influencing evaluation-time behavior.
- Mitigation: explicit separation of training metadata and runtime policy inputs.

---

## 9. Quality Checks Before Finalization

1. Section consistency:
- verify `C3`, `C4`, and `C5` field names are identical where shared.

2. Safety invariants:
- verify explicit “no unsafe forward motion on weak geometry”.

3. Legality boundaries:
- verify training-only vs evaluation-legal data paths are explicitly separated.

4. Repo alignment:
- verify frame names, task fields, and message conventions match repository artifacts.

---

## 10. Assumptions

1. Target document path and filename are fixed:
- `/home/user/aic/planning/visual_proximity_components_3_5_plan.md`

2. Camera calibration values are stable across datasets:
- still record per-frame `CameraInfo` for traceability and auditability.

3. Distractor-rich dataset will include dummy mounts/ports with explicit labels.

4. This is a planning artifact:
- implementation follows this spec in subsequent code changes.
