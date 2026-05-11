#!/usr/bin/env python3
"""Simple C3 SFP port detector (PyTorch + torchvision Keypoint R-CNN).

This script implements Step 3 from the visual proximity plan:
- detect all visible SFP ports in each camera image (left, center, right)
- output per-port keypoints for handoff to C4

Design goals:
- minimal moving parts
- easy CLI usage from terminal
- no ROS runtime required for training/inference

The script supports:
1) inspect dataset
2) train model
3) run prediction on episode data

Expected dataset format is the current training_data layout with:
- episodes/episode_*/frames.jsonl
- per-frame `training_gt.all_ports_gt_base`
- per-frame camera calibration and base->camera transforms
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from PIL import Image, ImageDraw
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python deps (Pillow / PyTorch / torchvision). "
        "Run with Pixi env and install if needed:\n"
        "  pixi add --pypi pillow torch torchvision\n"
        "  pixi run -- python3 aic_utils/aic_training_utils/scripts/port_detector_c3.py --help"
    ) from exc

try:
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
except Exception:  # pragma: no cover - for older torchvision versions
    KeypointRCNN_ResNet50_FPN_Weights = None


CAMERAS: tuple[str, str, str] = ("left", "center", "right")
KEYPOINT_NAMES: tuple[str, str] = ("port_link", "port_entrance")
MODEL_ID = "c3_keypointrcnn_v1"
LOGGER = logging.getLogger("port_detector_c3")


@dataclass(frozen=True)
class SampleRecord:
    episode_dir: Path
    episode_id: str
    frame_idx: int
    obs_stamp: float
    camera_name: str
    image_meta: dict[str, Any]
    camera_info: dict[str, Any]
    base_to_camera_optical: dict[str, Any] | None
    instances: list[dict[str, Any]]


def _seed_everything(seed: int) -> None:
    LOGGER.info("Seeding RNGs with seed=%d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_triplet_csv(raw: str, *, name: str) -> tuple[float, float, float]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values, got: {raw}")
    try:
        vals = tuple(float(x) for x in parts)
    except Exception as exc:
        raise ValueError(f"{name} contains non-numeric values: {raw}") from exc
    return vals  # type: ignore[return-value]


def _normalized_allocation(total: int, weights: tuple[float, float, float]) -> list[int]:
    if total <= 0:
        return [0, 0, 0]
    wsum = float(weights[0] + weights[1] + weights[2])
    if wsum <= 0.0:
        raise ValueError("allocation weights must sum to > 0")
    raw = [total * (w / wsum) for w in weights]
    base = [int(np.floor(x)) for x in raw]
    rem = total - sum(base)
    if rem > 0:
        frac_order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
        for i in frac_order[:rem]:
            base[i] += 1
    return base


def _sample_episode_frame_indices(
    *,
    n_frames: int,
    max_frames: int,
    policy: str,
    quota_split: tuple[float, float, float],
    bucket_split: tuple[float, float, float],
    rng: random.Random,
) -> list[int]:
    if max_frames <= 0 or n_frames <= max_frames:
        return list(range(n_frames))

    if policy == "uniform":
        return sorted(rng.sample(list(range(n_frames)), k=max_frames))

    if policy != "early_quota":
        raise ValueError(f"Unsupported episode_sampling_policy: {policy}")

    if min(quota_split) < 0.0:
        raise ValueError(f"episode_quota_split must be non-negative, got {quota_split}")
    if min(bucket_split) <= 0.0:
        raise ValueError(f"episode_bucket_split must be >0 per bucket, got {bucket_split}")
    if abs(sum(bucket_split) - 1.0) > 1e-6:
        raise ValueError(f"episode_bucket_split must sum to 1.0, got {bucket_split}")

    bucket_sizes = _normalized_allocation(n_frames, bucket_split)
    b0 = list(range(0, bucket_sizes[0]))
    b1 = list(range(bucket_sizes[0], bucket_sizes[0] + bucket_sizes[1]))
    b2 = list(range(bucket_sizes[0] + bucket_sizes[1], n_frames))
    buckets = [b0, b1, b2]

    targets = _normalized_allocation(max_frames, quota_split)
    picks: list[list[int]] = [[], [], []]

    for i in range(3):
        k = min(targets[i], len(buckets[i]))
        if k > 0:
            picks[i] = sorted(rng.sample(buckets[i], k=k))

    # Redistribute shortfalls: earlier buckets first, then later buckets.
    used = sum(len(x) for x in picks)
    shortfall = max_frames - used
    if shortfall > 0:
        for src_order in ([0, 1, 2], [2, 1, 0]):
            if shortfall <= 0:
                break
            for bi in src_order:
                if shortfall <= 0:
                    break
                pool = sorted(set(buckets[bi]) - set(picks[bi]))
                if not pool:
                    continue
                k = min(shortfall, len(pool))
                extra = rng.sample(pool, k=k)
                picks[bi].extend(extra)
                shortfall -= k

    chosen = sorted(picks[0] + picks[1] + picks[2])
    if len(chosen) > max_frames:
        chosen = chosen[:max_frames]
    return chosen


def _load_rgb_image(episode_dir: Path, image_meta: dict[str, Any], prefer_bin: bool) -> np.ndarray:
    width = int(image_meta.get("width", 0))
    height = int(image_meta.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError("Image width/height missing or invalid in frame metadata.")

    debug_rel = image_meta.get("path")
    bin_rel = image_meta.get("bin_path")
    encoding = str(image_meta.get("encoding", "rgb8")).lower()
    step = int(image_meta.get("step", width * 3))

    candidates: list[tuple[str, Path | None]] = []
    if prefer_bin:
        candidates.append(("bin", episode_dir / str(bin_rel) if bin_rel else None))
        candidates.append(("debug", episode_dir / str(debug_rel) if debug_rel else None))
    else:
        candidates.append(("debug", episode_dir / str(debug_rel) if debug_rel else None))
        candidates.append(("bin", episode_dir / str(bin_rel) if bin_rel else None))

    errors: list[str] = []
    for source, path in candidates:
        if path is None or not path.exists():
            continue
        try:
            if source == "debug":
                with Image.open(path) as im:
                    return np.array(im.convert("RGB"), dtype=np.uint8)
            raw = path.read_bytes()
            if encoding not in ("rgb8", "bgr8"):
                raise RuntimeError(f"Unsupported bin encoding: {encoding}")
            expected = height * step
            if len(raw) < expected:
                raise RuntimeError(
                    f"Binary image too short: got={len(raw)} bytes expected>={expected}"
                )
            arr = np.frombuffer(raw[: expected], dtype=np.uint8).reshape(height, step)
            arr = arr[:, : width * 3].reshape(height, width, 3)
            if encoding == "bgr8":
                arr = arr[..., ::-1]
            return arr.copy()
        except Exception as exc:
            errors.append(f"{source}:{path}:{exc}")
            continue

    if errors:
        raise RuntimeError("Failed to decode image candidates: " + " | ".join(errors))
    raise FileNotFoundError(
        f"No image found for frame image meta (debug={debug_rel}, bin={bin_rel}) under {episode_dir}"
    )


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> list[list[float]] | None:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n <= 1e-12:
        return None
    s = 2.0 / n
    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s
    return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]


def _intrinsics_from_camera_info(camera_info: dict[str, Any]) -> tuple[float, float, float, float]:
    k = camera_info.get("k", [])
    p = camera_info.get("p", [])
    fx = fy = cx = cy = 0.0
    if isinstance(k, list) and len(k) >= 9:
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
    if (fx <= 0.0 or fy <= 0.0) and isinstance(p, list) and len(p) >= 12:
        fx = float(p[0])
        fy = float(p[5])
        cx = float(p[2])
        cy = float(p[6])
    return fx, fy, cx, cy


def _transform_point_base_to_camera(
    p_base: tuple[float, float, float], t_base_to_camera: dict[str, Any]
) -> tuple[float, float, float] | None:
    if not isinstance(t_base_to_camera, dict):
        return None
    t = t_base_to_camera.get("translation", {})
    q = t_base_to_camera.get("rotation", {})
    try:
        tx = float(t["x"])
        ty = float(t["y"])
        tz = float(t["z"])
        qx = float(q["x"])
        qy = float(q["y"])
        qz = float(q["z"])
        qw = float(q["w"])
    except Exception:
        return None
    rot = _quat_to_rot(qx, qy, qz, qw)
    if rot is None:
        return None
    px = float(p_base[0]) - tx
    py = float(p_base[1]) - ty
    pz = float(p_base[2]) - tz
    # p_cam = R^T * (p_base - t_base_to_camera)
    rx = rot[0][0] * px + rot[1][0] * py + rot[2][0] * pz
    ry = rot[0][1] * px + rot[1][1] * py + rot[2][1] * pz
    rz = rot[0][2] * px + rot[1][2] * py + rot[2][2] * pz
    return (rx, ry, rz)


def _point_from_transform(transform: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(transform, dict):
        return None
    t = transform.get("translation", {})
    if not isinstance(t, dict):
        return None
    try:
        return (float(t["x"]), float(t["y"]), float(t["z"]))
    except Exception:
        return None


def _project_point(
    *,
    p_base: tuple[float, float, float] | None,
    t_base_to_camera: dict[str, Any] | None,
    camera_info: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    if p_base is None or not isinstance(t_base_to_camera, dict):
        return {
            "visible": False,
            "visibility_flag": 0,
            "uv": None,
            "reason": "missing_point_or_camera_tf",
            "depth_m": None,
        }

    p_cam = _transform_point_base_to_camera(p_base, t_base_to_camera)
    if p_cam is None:
        return {
            "visible": False,
            "visibility_flag": 0,
            "uv": None,
            "reason": "invalid_camera_tf",
            "depth_m": None,
        }

    x, y, z = p_cam
    if z <= 1e-6:
        return {
            "visible": False,
            "visibility_flag": 0,
            "uv": None,
            "reason": "negative_depth",
            "depth_m": float(z),
        }

    fx, fy, cx, cy = _intrinsics_from_camera_info(camera_info)
    if fx <= 0.0 or fy <= 0.0:
        return {
            "visible": False,
            "visibility_flag": 0,
            "uv": None,
            "reason": "invalid_intrinsics",
            "depth_m": float(z),
        }

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    in_bounds = 0.0 <= u < float(width) and 0.0 <= v < float(height)
    edge_margin = 2.0
    near_edge = (
        in_bounds
        and (
            u < edge_margin
            or v < edge_margin
            or u >= float(width) - edge_margin
            or v >= float(height) - edge_margin
        )
    )
    if not in_bounds:
        visibility_flag = 0
        visible = False
    elif near_edge:
        visibility_flag = 1
        visible = True
    else:
        visibility_flag = 2
        visible = True

    return {
        "visible": bool(visible),
        "visibility_flag": int(visibility_flag),
        "uv": (float(u), float(v)),
        "reason": None,
        "depth_m": float(z),
    }


def _build_port_instance(
    *,
    port_gt: dict[str, Any],
    t_base_to_camera: dict[str, Any] | None,
    camera_info: dict[str, Any],
    width: int,
    height: int,
    bbox_margin_px: float,
    require_both_keypoints: bool,
) -> dict[str, Any] | None:
    link_t = port_gt.get("t_base_port_link_gt")
    entrance_t = port_gt.get("t_base_port_entrance_gt")
    link_point = _point_from_transform(link_t)
    entrance_point = _point_from_transform(entrance_t)

    link_proj = _project_point(
        p_base=link_point,
        t_base_to_camera=t_base_to_camera,
        camera_info=camera_info,
        width=width,
        height=height,
    )
    entrance_proj = _project_point(
        p_base=entrance_point,
        t_base_to_camera=t_base_to_camera,
        camera_info=camera_info,
        width=width,
        height=height,
    )

    keypoints: list[list[float]] = []
    for proj in (link_proj, entrance_proj):
        uv = proj["uv"]
        if uv is None:
            keypoints.append([0.0, 0.0, 0.0])
        else:
            keypoints.append([float(uv[0]), float(uv[1]), float(proj["visibility_flag"])])

    visible_uv: list[tuple[float, float]] = []
    for proj in (link_proj, entrance_proj):
        if proj["visible"] and proj["uv"] is not None:
            visible_uv.append(proj["uv"])

    if require_both_keypoints and len(visible_uv) < 2:
        return None
    if len(visible_uv) == 0:
        return None

    if len(visible_uv) == 1:
        cx, cy = visible_uv[0]
        x1 = cx - bbox_margin_px
        y1 = cy - bbox_margin_px
        x2 = cx + bbox_margin_px
        y2 = cy + bbox_margin_px
    else:
        xs = [p[0] for p in visible_uv]
        ys = [p[1] for p in visible_uv]
        x1 = min(xs) - bbox_margin_px
        y1 = min(ys) - bbox_margin_px
        x2 = max(xs) + bbox_margin_px
        y2 = max(ys) + bbox_margin_px

    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "keypoints": keypoints,
        "module_name": str(port_gt.get("module_name", "")),
        "port_name": str(port_gt.get("port_name", "")),
        "is_task_target_port": bool(port_gt.get("is_task_target_port", False)),
    }


def build_samples_index(
    *,
    episodes_root: Path,
    bbox_margin_px: float,
    require_both_keypoints: bool,
    episode_tail_fraction: float = 1.0,
    episode_max_frames: int = 0,
    episode_sampling_policy: str = "early_quota",
    episode_quota_split: tuple[float, float, float] = (70.0, 20.0, 10.0),
    episode_bucket_split: tuple[float, float, float] = (0.5, 0.3, 0.2),
    episode_sampling_seed: int = 7,
) -> list[SampleRecord]:
    if episode_tail_fraction <= 0.0 or episode_tail_fraction > 1.0:
        raise ValueError(
            f"episode_tail_fraction must be in (0.0, 1.0], got {episode_tail_fraction}"
        )
    LOGGER.info(
        "Indexing samples from %s (bbox_margin_px=%.2f require_both_keypoints=%s episode_tail_fraction=%.3f episode_max_frames=%d episode_sampling_policy=%s)",
        episodes_root,
        bbox_margin_px,
        require_both_keypoints,
        episode_tail_fraction,
        int(episode_max_frames),
        episode_sampling_policy,
    )
    samples: list[SampleRecord] = []
    episode_dirs = sorted(
        [p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    )
    LOGGER.info("Found %d episode directories", len(episode_dirs))
    rng = random.Random(int(episode_sampling_seed))
    total_frames_seen = 0
    total_frames_used = 0
    total_frames_after_subsampling = 0
    for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
        frames_path = episode_dir / "frames.jsonl"
        if not frames_path.is_file():
            continue
        frames: list[dict[str, Any]] = []
        with frames_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                frames.append(json.loads(line))
        if len(frames) == 0:
            continue
        original_episode_frames_n = len(frames)
        total_frames_seen += original_episode_frames_n
        if episode_tail_fraction < 1.0:
            keep_n = max(1, int(np.ceil(len(frames) * episode_tail_fraction)))
            start_idx = len(frames) - keep_n
            frames = frames[start_idx:]
        after_tail_n = len(frames)
        total_frames_used += after_tail_n

        before_subsample_n = after_tail_n
        picks_by_bucket = [0, 0, 0]
        bucket_sizes = [0, 0, 0]
        if int(episode_max_frames) > 0 and len(frames) > int(episode_max_frames):
            selected_idx = _sample_episode_frame_indices(
                n_frames=len(frames),
                max_frames=int(episode_max_frames),
                policy=str(episode_sampling_policy),
                quota_split=episode_quota_split,
                bucket_split=episode_bucket_split,
                rng=rng,
            )

            # Stats/logging breakdown for early/mid/late buckets.
            bucket_sizes = _normalized_allocation(len(frames), episode_bucket_split)
            b0_end = bucket_sizes[0]
            b1_end = bucket_sizes[0] + bucket_sizes[1]
            for idx in selected_idx:
                if idx < b0_end:
                    picks_by_bucket[0] += 1
                elif idx < b1_end:
                    picks_by_bucket[1] += 1
                else:
                    picks_by_bucket[2] += 1
            frames = [frames[i] for i in selected_idx]

        total_frames_after_subsampling += len(frames)
        # if int(episode_max_frames) > 0:
        #     LOGGER.info(
        #         "Episode %s frame sampling: original=%d after_tail=%d selected=%d bucket_sizes=%s bucket_picks=%s",
        #         episode_dir.name,
        #         original_episode_frames_n,
        #         after_tail_n,
        #         len(frames),
        #         bucket_sizes,
        #         picks_by_bucket,
        #         )

        for frame in frames:
            training_gt = frame.get("training_gt", {})
            all_ports_gt = training_gt.get("all_ports_gt_base", [])
            if not isinstance(all_ports_gt, list):
                all_ports_gt = []
            camera_info_all = frame.get("camera_info", {})
            images = frame.get("images", {})
            cam_tfs = training_gt.get("base_to_camera_optical", {})

            frame_idx = int(frame.get("frame_idx", 0))
            obs_stamp = float(frame.get("obs_stamp", 0.0))
            episode_id = str(frame.get("episode_id", episode_dir.name))

            for cam in CAMERAS:
                image_meta = images.get(cam)
                if not isinstance(image_meta, dict):
                    continue
                width = int(image_meta.get("width", 0))
                height = int(image_meta.get("height", 0))
                if width <= 0 or height <= 0:
                    continue
                camera_info = camera_info_all.get(cam, {})
                t_base_to_camera = cam_tfs.get(cam)
                instances: list[dict[str, Any]] = []
                for port_gt in all_ports_gt:
                    if not isinstance(port_gt, dict):
                        continue
                    instance = _build_port_instance(
                        port_gt=port_gt,
                        t_base_to_camera=t_base_to_camera,
                        camera_info=camera_info,
                        width=width,
                        height=height,
                        bbox_margin_px=bbox_margin_px,
                        require_both_keypoints=require_both_keypoints,
                    )
                    if instance is not None:
                        instances.append(instance)

                samples.append(
                    SampleRecord(
                        episode_dir=episode_dir,
                        episode_id=episode_id,
                        frame_idx=frame_idx,
                        obs_stamp=obs_stamp,
                        camera_name=cam,
                        image_meta=image_meta,
                        camera_info=camera_info,
                        base_to_camera_optical=t_base_to_camera,
                        instances=instances,
                    )
                )
        if episode_idx % 50 == 0:
            LOGGER.info(
                "Indexed %d/%d episodes; accumulated samples=%d frames_used=%d/%d frames_after_subsampling=%d",
                episode_idx,
                len(episode_dirs),
                len(samples),
                total_frames_used,
                total_frames_seen,
                total_frames_after_subsampling,
            )
    total_instances = sum(len(s.instances) for s in samples)
    reduction_ratio = (1.0 - (float(total_frames_after_subsampling) / float(total_frames_used))) if total_frames_used > 0 else 0.0
    LOGGER.info(
        "Finished indexing: samples=%d total_port_instances=%d avg_instances_per_sample=%.3f frames_used=%d/%d frames_after_subsampling=%d reduction_ratio=%.3f",
        len(samples),
        total_instances,
        (float(total_instances) / float(len(samples))) if samples else 0.0,
        total_frames_used,
        total_frames_seen,
        total_frames_after_subsampling,
        reduction_ratio,
    )
    return samples


class PortDataset(Dataset):
    def __init__(self, samples: list[SampleRecord], prefer_bin: bool):
        self.samples = samples
        self.prefer_bin = prefer_bin

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], SampleRecord]:
        sample = self.samples[idx]
        img_np = _load_rgb_image(sample.episode_dir, sample.image_meta, self.prefer_bin)
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        boxes = [inst["bbox_xyxy"] for inst in sample.instances]
        keypoints = [inst["keypoints"] for inst in sample.instances]
        labels = [1 for _ in sample.instances]  # single class: sfp_port

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            keypoints_t = torch.zeros((0, 2, 3), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            keypoints_t = torch.tensor(keypoints, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            iscrowd_t = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "keypoints": keypoints_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }
        return img, target, sample


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], SampleRecord]]) -> tuple[Any, Any, Any]:
    images, targets, samples = zip(*batch)
    return list(images), list(targets), list(samples)


def build_model(num_classes: int = 2, num_keypoints: int = 2) -> torch.nn.Module:
    # Use pretrained backbone weights while keeping custom predictor heads.
    if KeypointRCNN_ResNet50_FPN_Weights is not None:
        model = keypointrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
        )
    else:  # pragma: no cover
        model = keypointrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    k_in = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(k_in, num_keypoints)
    return model


def _targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    out: list[dict[str, torch.Tensor]] = []
    for target in targets:
        out.append({k: v.to(device) for k, v in target.items()})
    return out


def _make_grad_scaler(device: torch.device, use_amp: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except Exception:  # pragma: no cover - older torch fallback
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(device: torch.device, use_amp: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    except Exception:  # pragma: no cover - older torch fallback
        return torch.cuda.amp.autocast(enabled=use_amp)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    epoch_idx: int,
    log_every_n_steps: int,
) -> dict[str, float]:
    model.train()
    scaler = _make_grad_scaler(device, use_amp)
    running: dict[str, float] = {}
    steps = 0
    t0 = time.time()
    total_steps = len(loader)
    LOGGER.info("Epoch %d: training started (batches=%d)", epoch_idx, total_steps)

    for images, targets, _samples in loader:
        images = [img.to(device) for img in images]
        targets = _targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v.detach().cpu().item())
        running["loss_total"] = running.get("loss_total", 0.0) + float(loss.detach().cpu().item())
        steps += 1
        if log_every_n_steps > 0 and steps % log_every_n_steps == 0:
            LOGGER.info(
                "Epoch %d: train step %d/%d batch_loss=%.6f",
                epoch_idx,
                steps,
                total_steps,
                float(loss.detach().cpu().item()),
            )

    if steps == 0:
        return {"loss_total": float("nan")}
    for k in list(running.keys()):
        running[k] /= float(steps)
    running["epoch_time_s"] = float(time.time() - t0)
    running["epoch"] = float(epoch_idx)
    LOGGER.info(
        "Epoch %d: training finished loss_total=%.6f epoch_time_s=%.2f",
        epoch_idx,
        running.get("loss_total", float("nan")),
        running["epoch_time_s"],
    )
    return running


def evaluate_detector(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float,
    match_px: float,
) -> dict[str, float]:
    LOGGER.info(
        "Running evaluation (score_thresh=%.3f match_px=%.2f batches=%d)",
        score_thresh,
        match_px,
        len(loader),
    )
    model.eval()
    total_gt = 0
    total_pred = 0
    tp = 0
    link_errs: list[float] = []
    entrance_errs: list[float] = []

    with torch.no_grad():
        for images, targets, _samples in loader:
            images_d = [img.to(device) for img in images]
            preds = model(images_d)

            for pred, target in zip(preds, targets):
                gt_kps = target["keypoints"].cpu().numpy()
                gt_vis_link = gt_kps[:, 0, 2] > 0 if gt_kps.size else np.zeros((0,), dtype=bool)
                valid_gt_idx = np.where(gt_vis_link)[0]
                total_gt += int(len(valid_gt_idx))

                pred_scores = pred.get("scores", torch.zeros((0,), device=device)).detach().cpu().numpy()
                keep_pred_idx = np.where(pred_scores >= score_thresh)[0]
                pred_kps = pred.get("keypoints", torch.zeros((0, 2, 3), device=device)).detach().cpu().numpy()
                total_pred += int(len(keep_pred_idx))

                if len(valid_gt_idx) == 0 or len(keep_pred_idx) == 0:
                    continue

                unmatched_pred = set(int(i) for i in keep_pred_idx.tolist())
                for gt_i in valid_gt_idx:
                    gt_link = gt_kps[gt_i, 0, :2]
                    best_pred_i = None
                    best_dist = float("inf")
                    for pred_i in list(unmatched_pred):
                        pred_link = pred_kps[pred_i, 0, :2]
                        dist = float(np.linalg.norm(pred_link - gt_link))
                        if dist < best_dist:
                            best_dist = dist
                            best_pred_i = pred_i
                    if best_pred_i is None:
                        continue
                    if best_dist <= match_px:
                        tp += 1
                        unmatched_pred.remove(best_pred_i)
                        gt_entrance = gt_kps[gt_i, 1, :2]
                        pred_entrance = pred_kps[best_pred_i, 1, :2]
                        link_errs.append(best_dist)
                        entrance_errs.append(float(np.linalg.norm(pred_entrance - gt_entrance)))

    precision = float(tp / total_pred) if total_pred > 0 else 0.0
    recall = float(tp / total_gt) if total_gt > 0 else 0.0
    mean_link_err = float(np.mean(link_errs)) if link_errs else float("nan")
    mean_entrance_err = float(np.mean(entrance_errs)) if entrance_errs else float("nan")
    metrics = {
        "tp": float(tp),
        "total_gt": float(total_gt),
        "total_pred": float(total_pred),
        "precision": precision,
        "recall": recall,
        "mean_link_error_px": mean_link_err,
        "mean_entrance_error_px": mean_entrance_err,
    }
    LOGGER.info(
        "Evaluation complete: precision=%.4f recall=%.4f mean_link_err_px=%s",
        metrics["precision"],
        metrics["recall"],
        "nan" if np.isnan(metrics["mean_link_error_px"]) else f"{metrics['mean_link_error_px']:.3f}",
    )
    return metrics


def _split_train_val(samples: list[SampleRecord], val_ratio: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    idx = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    val_idx = set(idx[:n_val])
    train_samples = [samples[i] for i in idx if i not in val_idx]
    val_samples = [samples[i] for i in idx if i in val_idx]
    return train_samples, val_samples


def _default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _resolve_run_args_dir(args: argparse.Namespace) -> Path:
    # Prefer explicit output locations first.
    out_dir = getattr(args, "output_dir", "")
    if isinstance(out_dir, str) and out_dir.strip():
        return Path(out_dir)

    out_jsonl = getattr(args, "output_jsonl", "")
    if isinstance(out_jsonl, str) and out_jsonl.strip():
        return Path(out_jsonl).expanduser().resolve().parent

    # Fallbacks for commands without output-dir style args.
    episodes_root = getattr(args, "episodes_root", "")
    if isinstance(episodes_root, str) and episodes_root.strip():
        return Path(episodes_root)

    episode_dir = getattr(args, "episode_dir", "")
    if isinstance(episode_dir, str) and episode_dir.strip():
        return Path(episode_dir)

    return Path.cwd()


def _write_run_args_json(args: argparse.Namespace) -> Path | None:
    try:
        target_dir = _resolve_run_args_dir(args)
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"{args.cmd}_run_args_{ts}.json"
        out = target_dir / fname
        payload = {
            "cmd": str(args.cmd),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "args": _safe_cli_args(args),
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out
    except Exception as exc:
        LOGGER.warning("Failed to write run args JSON: %s", exc)
        return None



def _sample_has_any_image_asset(sample: SampleRecord) -> bool:
    meta = sample.image_meta if isinstance(sample.image_meta, dict) else {}
    debug_rel = meta.get("path")
    bin_rel = meta.get("bin_path")
    if debug_rel and (sample.episode_dir / str(debug_rel)).exists():
        return True
    if bin_rel and (sample.episode_dir / str(bin_rel)).exists():
        return True
    return False


def _filter_samples_missing_assets(
    *,
    samples: list[SampleRecord],
    context: str,
) -> tuple[list[SampleRecord], int]:
    kept: list[SampleRecord] = []
    skipped = 0
    for sample in samples:
        if _sample_has_any_image_asset(sample):
            kept.append(sample)
            continue
        skipped += 1
        if skipped <= 5:
            meta = sample.image_meta if isinstance(sample.image_meta, dict) else {}
            LOGGER.warning(
                "%s skipping sample with missing assets episode=%s frame=%d cam=%s debug=%s bin=%s",
                context,
                sample.episode_id,
                sample.frame_idx,
                sample.camera_name,
                meta.get("path", ""),
                meta.get("bin_path", ""),
            )
    if skipped > 0:
        LOGGER.warning("%s skipped samples with missing assets: %d", context, skipped)
    return kept, skipped

def run_inspect(args: argparse.Namespace) -> None:
    LOGGER.info("Running inspect on episodes_root=%s", args.episodes_root)
    episodes_root = Path(args.episodes_root)
    samples = build_samples_index(
        episodes_root=episodes_root,
        bbox_margin_px=args.bbox_margin_px,
        require_both_keypoints=args.require_both_keypoints,
    )
    n_ports = sum(len(s.instances) for s in samples)
    by_cam = {cam: 0 for cam in CAMERAS}
    by_cam_ports = {cam: 0 for cam in CAMERAS}
    for s in samples:
        by_cam[s.camera_name] += 1
        by_cam_ports[s.camera_name] += len(s.instances)

    summary = {
        "episodes_root": str(episodes_root),
        "num_samples_frame_camera": len(samples),
        "num_port_instances": n_ports,
        "samples_per_camera": by_cam,
        "port_instances_per_camera": by_cam_ports,
        "avg_ports_per_sample": (n_ports / len(samples)) if samples else 0.0,
        "require_both_keypoints": bool(args.require_both_keypoints),
        "bbox_margin_px": float(args.bbox_margin_px),
    }
    LOGGER.info(
        "Inspect summary: samples=%d total_port_instances=%d avg_ports_per_sample=%.3f",
        summary["num_samples_frame_camera"],
        summary["num_port_instances"],
        summary["avg_ports_per_sample"],
    )
    print(json.dumps(summary, indent=2))


def run_train(args: argparse.Namespace) -> None:
    LOGGER.info("Starting train command")
    _seed_everything(args.seed)
    device = _default_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    LOGGER.info("Device=%s amp=%s", device, use_amp)

    train_root = Path(args.train_episodes_root)
    val_root = Path(args.val_episodes_root) if args.val_episodes_root else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quota_split = _parse_triplet_csv(args.episode_quota_split, name="episode_quota_split")
    if abs(sum(quota_split) - 100.0) > 1e-6:
        raise ValueError(f"episode_quota_split must sum to 100, got {quota_split}")
    bucket_split = _parse_triplet_csv(args.episode_bucket_split, name="episode_bucket_split")
    if abs(sum(bucket_split) - 1.0) > 1e-6:
        raise ValueError(f"episode_bucket_split must sum to 1.0, got {bucket_split}")
    sampling_seed = args.seed if int(args.episode_sampling_seed) < 0 else int(args.episode_sampling_seed)

    all_train_samples = build_samples_index(
        episodes_root=train_root,
        bbox_margin_px=args.bbox_margin_px,
        require_both_keypoints=args.require_both_keypoints,
        episode_tail_fraction=args.episode_tail_fraction,
        episode_max_frames=int(args.episode_max_frames),
        episode_sampling_policy=str(args.episode_sampling_policy),
        episode_quota_split=quota_split,
        episode_bucket_split=bucket_split,
        episode_sampling_seed=sampling_seed,
    )
    if len(all_train_samples) == 0:
        raise RuntimeError(f"No samples found in {train_root}")
    LOGGER.info("Initial train sample count=%d", len(all_train_samples))

    if val_root is None:
        train_samples, val_samples = _split_train_val(
            all_train_samples, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        train_samples = all_train_samples
        val_samples = build_samples_index(
            episodes_root=val_root,
            bbox_margin_px=args.bbox_margin_px,
            require_both_keypoints=args.require_both_keypoints,
            episode_tail_fraction=args.episode_tail_fraction,
            episode_max_frames=0,
        )

    train_samples, skipped_missing_train = _filter_samples_missing_assets(
        samples=train_samples,
        context="train",
    )
    val_samples, skipped_missing_val = _filter_samples_missing_assets(
        samples=val_samples,
        context="val",
    )

    if len(train_samples) == 0:
        raise RuntimeError("No train samples left after filtering missing image assets.")
    if len(val_samples) == 0:
        raise RuntimeError("No val samples left after filtering missing image assets.")

    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    LOGGER.info(
        "Using train_samples=%d val_samples=%d batch_size=%d eval_batch_size=%d skipped_missing_train=%d skipped_missing_val=%d",
        len(train_samples),
        len(val_samples),
        args.batch_size,
        args.eval_batch_size,
        skipped_missing_train,
        skipped_missing_val,
    )

    train_ds = PortDataset(train_samples, prefer_bin=args.prefer_bin_for_train)
    val_ds = PortDataset(val_samples, prefer_bin=args.prefer_bin_for_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2, num_keypoints=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    history: list[dict[str, float]] = []
    best_recall = -1.0
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"
    history_path = out_dir / "history.jsonl"
    LOGGER.info("Checkpoints will be written to %s", out_dir)

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, args.epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            epoch_idx=epoch,
            log_every_n_steps=args.log_every_n,
        )
        eval_metrics = evaluate_detector(
            model=model,
            loader=val_loader,
            device=device,
            score_thresh=args.score_thresh,
            match_px=args.match_px,
        )
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train/{k}": float(v) for k, v in train_metrics.items()},
            **{f"val/{k}": float(v) for k, v in eval_metrics.items()},
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(row) + "\n")

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row.get("train/loss_total"),
                    "val_recall": row.get("val/recall"),
                    "val_precision": row.get("val/precision"),
                    "val_link_err_px": row.get("val/mean_link_error_px"),
                }
            )
        )

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "args": _safe_cli_args(args),
            "epoch": epoch,
            "model_id": MODEL_ID,
            "keypoint_names": list(KEYPOINT_NAMES),
            "class_names": ["background", "sfp_port"],
        }
        torch.save(ckpt_payload, last_ckpt)

        recall = float(eval_metrics.get("recall", 0.0))
        if recall > best_recall:
            best_recall = recall
            torch.save(ckpt_payload, best_ckpt)
            LOGGER.info("New best checkpoint at epoch %d with recall=%.4f", epoch, best_recall)

    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": str(out_dir),
                "best_ckpt": str(best_ckpt),
                "last_ckpt": str(last_ckpt),
                "history_jsonl": str(history_path),
                "best_val_recall": best_recall,
                "device": str(device),
                "amp": use_amp,
                "skipped_missing_train_samples": int(skipped_missing_train),
                "skipped_missing_val_samples": int(skipped_missing_val),
            },
            indent=2,
        )
    )


def run_preview_labels(args: argparse.Namespace) -> None:
    LOGGER.info(
        "Running preview-labels on %s -> %s (num_samples=%d camera=%s prefer_bin=%s episode_tail_fraction=%.3f)",
        args.episodes_root,
        args.output_dir,
        args.num_samples,
        args.camera,
        args.prefer_bin,
        args.episode_tail_fraction,
    )
    episodes_root = Path(args.episodes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples_index(
        episodes_root=episodes_root,
        bbox_margin_px=args.bbox_margin_px,
        require_both_keypoints=args.require_both_keypoints,
        episode_tail_fraction=args.episode_tail_fraction,
    )
    if args.camera != "all":
        samples = [s for s in samples if s.camera_name == args.camera]
    if not args.include_empty_samples:
        samples = [s for s in samples if len(s.instances) > 0]
    if len(samples) == 0:
        raise RuntimeError("No samples available after filters. Try --include-empty-samples.")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(samples))
    selected = rng.sample(samples, k=k)
    LOGGER.info("Selected %d samples for preview rendering", len(selected))

    colors = [
        (255, 64, 64),
        (64, 220, 64),
        (64, 160, 255),
        (255, 200, 64),
        (220, 64, 220),
        (64, 220, 220),
    ]
    kp_radius = max(2, int(args.keypoint_radius_px))
    bbox_width = max(1, int(args.bbox_line_width_px))

    manifest_path = output_dir / "preview_manifest.jsonl"
    saved = 0
    with manifest_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(selected):
            img_np = _load_rgb_image(
                episode_dir=sample.episode_dir,
                image_meta=sample.image_meta,
                prefer_bin=args.prefer_bin,
            )
            canvas = Image.fromarray(img_np, mode="RGB")
            draw = ImageDraw.Draw(canvas)

            non_target_instances = [inst for inst in sample.instances if not inst.get("is_task_target_port", False)]
            target_instances = [inst for inst in sample.instances if inst.get("is_task_target_port", False)]

            for inst_i, inst in enumerate(non_target_instances):
                color = colors[inst_i % len(colors)]
                x1, y1, x2, y2 = inst["bbox_xyxy"]
                draw.rectangle(
                    [(float(x1), float(y1)), (float(x2), float(y2))],
                    outline=color,
                    width=bbox_width,
                )
                label = f"{inst.get('module_name', '')}:{inst.get('port_name', '')}"
                draw.text((float(x1) + 3.0, float(y1) + 3.0), label, fill=color)

                kps = inst.get("keypoints", [])
                if len(kps) >= 2:
                    u0, v0, vis0 = kps[0]
                    u1, v1, vis1 = kps[1]
                    if vis0 > 0:
                        draw.ellipse(
                            [(u0 - kp_radius, v0 - kp_radius), (u0 + kp_radius, v0 + kp_radius)],
                            fill=(255, 255, 255),
                            outline=color,
                            width=1,
                        )
                    if vis1 > 0:
                        draw.ellipse(
                            [(u1 - kp_radius, v1 - kp_radius), (u1 + kp_radius, v1 + kp_radius)],
                            fill=color,
                            outline=(255, 255, 255),
                            width=1,
                        )
                    if vis0 > 0 and vis1 > 0:
                        draw.line([(u0, v0), (u1, v1)], fill=color, width=2)

            # Draw task target ports last so they stay visible on top.
            for inst in target_instances:
                target_color = (255, 230, 0)  # bright yellow
                target_text_color = (0, 0, 0)
                x1, y1, x2, y2 = inst["bbox_xyxy"]
                draw.rectangle(
                    [(float(x1), float(y1)), (float(x2), float(y2))],
                    outline=target_color,
                    width=max(2, bbox_width + 2),
                )
                label = f"TARGET {inst.get('module_name', '')}:{inst.get('port_name', '')}"
                draw.text((float(x1) + 3.0, float(y1) + 3.0), label, fill=target_text_color)

                kps = inst.get("keypoints", [])
                if len(kps) >= 2:
                    u0, v0, vis0 = kps[0]
                    u1, v1, vis1 = kps[1]
                    r = kp_radius + 2
                    if vis0 > 0:
                        draw.ellipse(
                            [(u0 - r, v0 - r), (u0 + r, v0 + r)],
                            fill=(255, 255, 255),
                            outline=target_color,
                            width=2,
                        )
                    if vis1 > 0:
                        draw.ellipse(
                            [(u1 - r, v1 - r), (u1 + r, v1 + r)],
                            fill=target_color,
                            outline=(255, 255, 255),
                            width=2,
                        )
                    if vis0 > 0 and vis1 > 0:
                        draw.line([(u0, v0), (u1, v1)], fill=target_color, width=3)

            out_name = (
                f"{i:04d}_{sample.episode_id}_f{sample.frame_idx:06d}_{sample.camera_name}.png"
            )
            out_path = output_dir / out_name
            canvas.save(out_path)
            saved += 1

            record = {
                "preview_path": str(out_path),
                "episode_id": sample.episode_id,
                "episode_dir": str(sample.episode_dir),
                "frame_idx": sample.frame_idx,
                "obs_stamp": sample.obs_stamp,
                "camera_name": sample.camera_name,
                "image_meta": sample.image_meta,
                "num_instances": len(sample.instances),
                "num_target_instances": len(target_instances),
            }
            fout.write(json.dumps(record) + "\n")
            if args.log_every_n > 0 and (i + 1) % args.log_every_n == 0:
                LOGGER.info(
                    "Preview progress: %d/%d saved (latest=%s instances=%d)",
                    i + 1,
                    len(selected),
                    out_path.name,
                    len(sample.instances),
                )

    print(
        json.dumps(
            {
                "status": "done",
                "episodes_root": str(episodes_root),
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "saved_images": saved,
                "camera_filter": args.camera,
                "include_empty_samples": bool(args.include_empty_samples),
            },
            indent=2,
        )
    )


def _predict_image(
    *,
    model: torch.nn.Module,
    device: torch.device,
    img_np: np.ndarray,
    score_thresh: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    model.eval()
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        pred = model([img_t])[0]
    boxes = pred.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu().numpy()
    scores = pred.get("scores", torch.zeros((0,), device=device)).detach().cpu().numpy()
    keypoints = pred.get("keypoints", torch.zeros((0, 2, 3), device=device)).detach().cpu().numpy()

    detections: list[dict[str, Any]] = []
    max_keep = max(1, int(max_detections))
    kept = 0
    for i in range(len(scores)):
        score = float(scores[i])
        if score < score_thresh:
            continue
        if kept >= max_keep:
            break
        box = boxes[i].tolist()
        kp = keypoints[i]
        link_uv = [float(kp[0][0]), float(kp[0][1])]
        entrance_uv = [float(kp[1][0]), float(kp[1][1])]
        detections.append(
            {
                "score": score,
                "bbox_xyxy": [float(v) for v in box],
                "port_link_uv_px": link_uv,
                "port_entrance_uv_px": entrance_uv,
            }
        )
        kept += 1
    return detections


def _load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading checkpoint from %s on device=%s", checkpoint_path, device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch fallback
        payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=2, num_keypoints=2).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def run_predict_episode(args: argparse.Namespace) -> None:
    LOGGER.info(
        "Running predict-episode checkpoint=%s episode_dir=%s output=%s",
        args.checkpoint,
        args.episode_dir,
        args.output_jsonl,
    )
    device = _default_device(args.device)
    model = _load_model_checkpoint(Path(args.checkpoint), device=device)
    episode_dir = Path(args.episode_dir)
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        raise RuntimeError(f"Missing frames file: {frames_path}")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with frames_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            frame_idx = int(frame.get("frame_idx", 0))
            obs_stamp = float(frame.get("obs_stamp", 0.0))
            images = frame.get("images", {})
            camera_packets: list[dict[str, Any]] = []

            for cam in CAMERAS:
                image_meta = images.get(cam)
                if not isinstance(image_meta, dict):
                    continue
                img_np = _load_rgb_image(
                    episode_dir=episode_dir,
                    image_meta=image_meta,
                    prefer_bin=args.prefer_bin,
                )
                detections = _predict_image(
                    model=model,
                    device=device,
                    img_np=img_np,
                    score_thresh=args.score_thresh,
                    max_detections=args.max_detections,
                )
                camera_packets.append(
                    {
                        "camera_name": cam,
                        "image_stamp": float(image_meta.get("stamp", obs_stamp)),
                        "frame_id": str(image_meta.get("frame_id", "")),
                        "width": int(image_meta.get("width", img_np.shape[1])),
                        "height": int(image_meta.get("height", img_np.shape[0])),
                        "encoding": str(image_meta.get("encoding", "rgb8")),
                        "detections": detections,
                    }
                )

            out_row = {
                "model_id": MODEL_ID,
                "frame_idx": frame_idx,
                "obs_stamp": obs_stamp,
                "task": frame.get("task", {}),
                "camera_packets": camera_packets,
            }
            fout.write(json.dumps(out_row) + "\n")
            frame_count += 1
            if args.log_every_n > 0 and frame_count % args.log_every_n == 0:
                LOGGER.info("Prediction progress: frames=%d (latest frame_idx=%d)", frame_count, frame_idx)

    LOGGER.info("Prediction complete: frames_written=%d", frame_count)

    print(
        json.dumps(
            {
                "status": "done",
                "episode_dir": str(episode_dir),
                "output_jsonl": str(out_path),
                "checkpoint": str(args.checkpoint),
                "prefer_bin": bool(args.prefer_bin),
            },
            indent=2,
        )
    )


def run_visualize_predictions(args: argparse.Namespace) -> None:
    LOGGER.info(
        "Running visualize-predictions episode_dir=%s pred_jsonl=%s output_dir=%s",
        args.episode_dir,
        args.pred_jsonl,
        args.output_dir,
    )
    episode_dir = Path(args.episode_dir)
    pred_jsonl = Path(args.pred_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pred_jsonl.is_file():
        raise RuntimeError(f"Missing prediction jsonl: {pred_jsonl}")
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        raise RuntimeError(f"Missing frames file: {frames_path}")

    frame_index: dict[int, dict[str, Any]] = {}
    with frames_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            frame_index[int(frame.get("frame_idx", 0))] = frame

    candidates: list[dict[str, Any]] = []
    with pred_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_idx = int(row.get("frame_idx", 0))
            frame = frame_index.get(frame_idx)
            if frame is None:
                continue
            for packet in row.get("camera_packets", []):
                if not isinstance(packet, dict):
                    continue
                cam = str(packet.get("camera_name", ""))
                if cam not in CAMERAS:
                    continue
                if args.camera != "all" and cam != args.camera:
                    continue
                dets = packet.get("detections", [])
                if not isinstance(dets, list):
                    dets = []
                if not args.include_empty and len(dets) == 0:
                    continue
                candidates.append(
                    {
                        "frame_idx": frame_idx,
                        "camera_name": cam,
                        "frame": frame,
                        "detections": dets,
                    }
                )

    if len(candidates) == 0:
        raise RuntimeError("No prediction entries available after filters.")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(candidates))
    selected = rng.sample(candidates, k=k)
    LOGGER.info("Selected %d/%d prediction entries for visualization", k, len(candidates))

    colors = [
        (255, 64, 64),
        (64, 220, 64),
        (64, 160, 255),
        (255, 200, 64),
        (220, 64, 220),
        (64, 220, 220),
    ]
    bbox_width = max(1, int(args.bbox_line_width_px))
    kp_radius = max(2, int(args.keypoint_radius_px))

    manifest_path = output_dir / "visualize_predictions_manifest.jsonl"
    saved = 0
    with manifest_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(selected):
            frame = item["frame"]
            frame_idx = int(item["frame_idx"])
            cam = str(item["camera_name"])
            image_meta = frame.get("images", {}).get(cam, {})
            if not isinstance(image_meta, dict):
                continue
            img_np = _load_rgb_image(
                episode_dir=episode_dir,
                image_meta=image_meta,
                prefer_bin=args.prefer_bin,
            )
            canvas = Image.fromarray(img_np, mode="RGB")
            draw = ImageDraw.Draw(canvas)

            drawn = 0
            for det_i, det in enumerate(item["detections"]):
                if not isinstance(det, dict):
                    continue
                score = float(det.get("score", 0.0))
                if score < args.score_thresh:
                    continue
                bbox = det.get("bbox_xyxy", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                color = colors[det_i % len(colors)]
                x1, y1, x2, y2 = [float(v) for v in bbox]
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=color,
                    width=bbox_width,
                )
                draw.text((x1 + 3.0, y1 + 3.0), f"{score:.2f}", fill=color)

                link_uv = det.get("port_link_uv_px", [])
                ent_uv = det.get("port_entrance_uv_px", [])
                if isinstance(link_uv, list) and len(link_uv) == 2:
                    u0, v0 = float(link_uv[0]), float(link_uv[1])
                    draw.ellipse(
                        [(u0 - kp_radius, v0 - kp_radius), (u0 + kp_radius, v0 + kp_radius)],
                        fill=(255, 255, 255),
                        outline=color,
                        width=1,
                    )
                if isinstance(ent_uv, list) and len(ent_uv) == 2:
                    u1, v1 = float(ent_uv[0]), float(ent_uv[1])
                    draw.ellipse(
                        [(u1 - kp_radius, v1 - kp_radius), (u1 + kp_radius, v1 + kp_radius)],
                        fill=color,
                        outline=(255, 255, 255),
                        width=1,
                    )
                if (
                    isinstance(link_uv, list)
                    and len(link_uv) == 2
                    and isinstance(ent_uv, list)
                    and len(ent_uv) == 2
                ):
                    draw.line(
                        [(float(link_uv[0]), float(link_uv[1])), (float(ent_uv[0]), float(ent_uv[1]))],
                        fill=color,
                        width=2,
                    )
                drawn += 1

            out_name = f"{i:04d}_f{frame_idx:06d}_{cam}.png"
            out_path = output_dir / out_name
            canvas.save(out_path)
            saved += 1

            fout.write(
                json.dumps(
                    {
                        "preview_path": str(out_path),
                        "frame_idx": frame_idx,
                        "camera_name": cam,
                        "num_detections_total": len(item["detections"]),
                        "num_detections_drawn": drawn,
                    }
                )
                + "\n"
            )
            if args.log_every_n > 0 and (i + 1) % args.log_every_n == 0:
                LOGGER.info("Visualization progress: %d/%d", i + 1, len(selected))

    print(
        json.dumps(
            {
                "status": "done",
                "episode_dir": str(episode_dir),
                "pred_jsonl": str(pred_jsonl),
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "candidates": len(candidates),
                "saved_images": saved,
                "camera_filter": args.camera,
                "score_thresh": float(args.score_thresh),
            },
            indent=2,
        )
    )


def run_sample_test_predictions(args: argparse.Namespace) -> None:
    LOGGER.info(
        "Running sample-test-predictions episodes_root=%s checkpoint=%s n=%d",
        args.episodes_root,
        args.checkpoint,
        args.num_samples,
    )
    episodes_root = Path(args.episodes_root)
    if not episodes_root.is_dir():
        raise RuntimeError(f"Episodes root not found: {episodes_root}")
    device = _default_device(args.device)
    model = _load_model_checkpoint(Path(args.checkpoint), device=device)

    samples = build_samples_index(
        episodes_root=episodes_root,
        bbox_margin_px=args.bbox_margin_px,
        require_both_keypoints=args.require_both_keypoints,
        episode_tail_fraction=args.episode_tail_fraction,
    )
    if args.camera != "all":
        samples = [s for s in samples if s.camera_name == args.camera]
    if not args.include_empty_gt:
        samples = [s for s in samples if len(s.instances) > 0]
    if len(samples) == 0:
        raise RuntimeError("No test samples available after filters.")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(samples))
    selected = rng.sample(samples, k=k)
    LOGGER.info("Selected %d/%d test samples", k, len(samples))

    out_dir = Path(args.output_dir) if args.output_dir else (episodes_root / "images")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / args.overlay_subdir
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "sample_test_predictions.jsonl"
    colors_pred = (64, 220, 64)
    colors_gt = (255, 230, 0)

    saved = 0
    with result_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(selected):
            img_np = _load_rgb_image(
                episode_dir=sample.episode_dir,
                image_meta=sample.image_meta,
                prefer_bin=args.prefer_bin,
            )
            preds = _predict_image(
                model=model,
                device=device,
                img_np=img_np,
                score_thresh=args.score_thresh,
                max_detections=args.max_detections,
            )
            row = {
                "sample_idx": i,
                "episode_id": sample.episode_id,
                "episode_dir": str(sample.episode_dir),
                "frame_idx": sample.frame_idx,
                "obs_stamp": sample.obs_stamp,
                "camera_name": sample.camera_name,
                "image_meta": sample.image_meta,
                "gt_instances": sample.instances,
                "pred_instances": preds,
            }

            if args.save_overlays:
                canvas = Image.fromarray(img_np, mode="RGB")
                draw = ImageDraw.Draw(canvas)

                # Draw GT boxes/keypoints
                for gt in sample.instances:
                    x1, y1, x2, y2 = [float(v) for v in gt["bbox_xyxy"]]
                    draw.rectangle([(x1, y1), (x2, y2)], outline=colors_gt, width=3)
                    draw.text((x1 + 3.0, y1 + 3.0), "GT", fill=colors_gt)
                    kps = gt.get("keypoints", [])
                    if isinstance(kps, list) and len(kps) >= 2:
                        u0, v0, vis0 = kps[0]
                        u1, v1, vis1 = kps[1]
                        if vis0 > 0:
                            draw.ellipse(
                                [(u0 - 4, v0 - 4), (u0 + 4, v0 + 4)],
                                fill=(255, 255, 255),
                                outline=colors_gt,
                                width=1,
                            )
                        if vis1 > 0:
                            draw.ellipse(
                                [(u1 - 4, v1 - 4), (u1 + 4, v1 + 4)],
                                fill=colors_gt,
                                outline=(255, 255, 255),
                                width=1,
                            )

                # Draw predicted boxes/keypoints
                for pred in preds:
                    x1, y1, x2, y2 = [float(v) for v in pred["bbox_xyxy"]]
                    score = float(pred.get("score", 0.0))
                    draw.rectangle([(x1, y1), (x2, y2)], outline=colors_pred, width=3)
                    draw.text((x1 + 3.0, y1 + 16.0), f"PRED {score:.2f}", fill=colors_pred)
                    link_uv = pred.get("port_link_uv_px", [])
                    ent_uv = pred.get("port_entrance_uv_px", [])
                    if isinstance(link_uv, list) and len(link_uv) == 2:
                        u0, v0 = float(link_uv[0]), float(link_uv[1])
                        draw.ellipse(
                            [(u0 - 4, v0 - 4), (u0 + 4, v0 + 4)],
                            fill=(255, 255, 255),
                            outline=colors_pred,
                            width=1,
                        )
                    if isinstance(ent_uv, list) and len(ent_uv) == 2:
                        u1, v1 = float(ent_uv[0]), float(ent_uv[1])
                        draw.ellipse(
                            [(u1 - 4, v1 - 4), (u1 + 4, v1 + 4)],
                            fill=colors_pred,
                            outline=(255, 255, 255),
                            width=1,
                        )
                    if (
                        isinstance(link_uv, list)
                        and len(link_uv) == 2
                        and isinstance(ent_uv, list)
                        and len(ent_uv) == 2
                    ):
                        draw.line(
                            [(float(link_uv[0]), float(link_uv[1])), (float(ent_uv[0]), float(ent_uv[1]))],
                            fill=colors_pred,
                            width=2,
                        )

                overlay_name = (
                    f"{i:04d}_{sample.episode_id}_f{sample.frame_idx:06d}_{sample.camera_name}.webp"
                )
                overlay_path = overlays_dir / overlay_name
                canvas.save(overlay_path, format="WEBP", lossless=True)
                row["overlay_path"] = str(overlay_path)

            fout.write(json.dumps(row) + "\n")
            saved += 1
            if args.log_every_n > 0 and saved % args.log_every_n == 0:
                LOGGER.info("sample-test-predictions progress: %d/%d", saved, len(selected))

    print(
        json.dumps(
            {
                "status": "done",
                "episodes_root": str(episodes_root),
                "output_dir": str(out_dir),
                "results_jsonl": str(result_path),
                "save_overlays": bool(args.save_overlays),
                "overlays_dir": str(overlays_dir) if args.save_overlays else "",
                "num_saved": saved,
                "episode_tail_fraction": float(args.episode_tail_fraction),
            },
            indent=2,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C3 SFP port detector CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    def add_common_log_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging verbosity.",
        )
        subparser.add_argument(
            "--log-every-n",
            type=int,
            default=25,
            help="Progress log cadence for long loops (0 disables periodic progress logs).",
        )

    p_inspect = subparsers.add_parser("inspect", help="Inspect dataset and projected labels.")
    add_common_log_args(p_inspect)
    p_inspect.add_argument("--episodes-root", required=True, help="Path to episodes directory.")
    p_inspect.add_argument("--bbox-margin-px", type=float, default=14.0)
    p_inspect.add_argument(
        "--require-both-keypoints",
        action="store_true",
        help="Only keep labels where both link and entrance are visible.",
    )
    p_inspect.set_defaults(func=run_inspect)

    p_train = subparsers.add_parser("train", help="Train C3 keypoint detector.")
    add_common_log_args(p_train)
    p_train.add_argument("--train-episodes-root", required=True)
    p_train.add_argument(
        "--val-episodes-root",
        default="",
        help="Optional separate validation episodes root. If omitted, split train set.",
    )
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--eval-batch-size", type=int, default=16)
    p_train.add_argument("--num-workers", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--lr-step-size", type=int, default=8)
    p_train.add_argument("--lr-gamma", type=float, default=0.5)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.add_argument(
        "--episode-tail-fraction",
        type=float,
        default=1.0,
        help="Use only the final fraction of frames from each episode during training/validation indexing.",
    )
    p_train.add_argument("--bbox-margin-px", type=float, default=14.0)
    p_train.add_argument("--score-thresh", type=float, default=0.35)
    p_train.add_argument("--match-px", type=float, default=24.0)
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument(
        "--episode-max-frames",
        type=int,
        default=0,
        help="Per-episode frame cap after episode-tail filtering. 0 means disabled.",
    )
    p_train.add_argument(
        "--episode-sampling-policy",
        choices=["uniform", "early_quota"],
        default="early_quota",
        help="Per-episode frame subsampling policy when episode-max-frames > 0.",
    )
    p_train.add_argument(
        "--episode-quota-split",
        default="70,20,10",
        help="Early/mid/late quota percentages for early_quota policy.",
    )
    p_train.add_argument(
        "--episode-bucket-split",
        default="0.5,0.3,0.2",
        help="Early/mid/late timeline bucket fractions; must sum to 1.0.",
    )
    p_train.add_argument(
        "--episode-sampling-seed",
        type=int,
        default=-1,
        help="RNG seed for per-episode frame sampling. -1 means reuse --seed.",
    )
    p_train.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap for quick debugging. 0 means use all samples.",
    )
    p_train.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Optional cap for quick debugging. 0 means use all samples.",
    )
    p_train.add_argument("--device", default="auto", help="auto|cuda|cpu|cuda:0")
    p_train.add_argument("--amp", action="store_true")
    p_train.add_argument("--prefer-bin-for-train", action="store_true")
    p_train.add_argument("--prefer-bin-for-eval", action="store_true")
    p_train.add_argument(
        "--require-both-keypoints",
        action="store_true",
        help="Only keep labels where both link and entrance are visible.",
    )
    p_train.set_defaults(func=run_train)

    p_preview = subparsers.add_parser(
        "preview-labels",
        help="Overlay auto-generated labels on random samples for visual QA.",
    )
    add_common_log_args(p_preview)
    p_preview.add_argument("--episodes-root", required=True)
    p_preview.add_argument("--output-dir", required=True)
    p_preview.add_argument("--num-samples", type=int, default=40)
    p_preview.add_argument("--seed", type=int, default=7)
    p_preview.add_argument("--camera", choices=["all", "left", "center", "right"], default="all")
    p_preview.add_argument("--prefer-bin", action="store_true")
    p_preview.add_argument("--bbox-margin-px", type=float, default=14.0)
    p_preview.add_argument(
        "--episode-tail-fraction",
        type=float,
        default=1.0,
        help="Use only the final fraction of frames from each episode when building preview samples.",
    )
    p_preview.add_argument("--keypoint-radius-px", type=int, default=4)
    p_preview.add_argument("--bbox-line-width-px", type=int, default=3)
    p_preview.add_argument("--include-empty-samples", action="store_true")
    p_preview.add_argument(
        "--require-both-keypoints",
        action="store_true",
        help="Only keep labels where both link and entrance are visible.",
    )
    p_preview.set_defaults(func=run_preview_labels)

    p_predict = subparsers.add_parser(
        "predict-episode",
        help="Run detector on an episode and write C3-like JSONL output.",
    )
    add_common_log_args(p_predict)
    p_predict.add_argument("--checkpoint", required=True)
    p_predict.add_argument("--episode-dir", required=True)
    p_predict.add_argument("--output-jsonl", required=True)
    p_predict.add_argument("--score-thresh", type=float, default=0.85)
    p_predict.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Maximum number of predictions kept per image after thresholding.",
    )
    p_predict.add_argument("--prefer-bin", action="store_true")
    p_predict.add_argument("--device", default="auto", help="auto|cuda|cpu|cuda:0")
    p_predict.set_defaults(func=run_predict_episode)

    p_viz = subparsers.add_parser(
        "visualize-predictions",
        help="Overlay predicted bounding boxes/keypoints on sampled frame-camera entries.",
    )
    add_common_log_args(p_viz)
    p_viz.add_argument("--episode-dir", required=True)
    p_viz.add_argument("--pred-jsonl", required=True)
    p_viz.add_argument("--output-dir", required=True)
    p_viz.add_argument("--num-samples", type=int, default=40)
    p_viz.add_argument("--seed", type=int, default=7)
    p_viz.add_argument("--camera", choices=["all", "left", "center", "right"], default="all")
    p_viz.add_argument("--score-thresh", type=float, default=0.35)
    p_viz.add_argument("--bbox-line-width-px", type=int, default=3)
    p_viz.add_argument("--keypoint-radius-px", type=int, default=4)
    p_viz.add_argument("--include-empty", action="store_true")
    p_viz.add_argument("--prefer-bin", action="store_true")
    p_viz.set_defaults(func=run_visualize_predictions)

    p_sample_test = subparsers.add_parser(
        "sample-test-predictions",
        help="Sample N testing images, predict ports, and optionally save GT+pred overlay WEBP images.",
    )
    add_common_log_args(p_sample_test)
    p_sample_test.add_argument("--episodes-root", required=True)
    p_sample_test.add_argument("--checkpoint", required=True)
    p_sample_test.add_argument("--num-samples", type=int, required=True)
    p_sample_test.add_argument("--output-dir", default="")
    p_sample_test.add_argument("--camera", choices=["all", "left", "center", "right"], default="all")
    p_sample_test.add_argument("--seed", type=int, default=7)
    p_sample_test.add_argument("--score-thresh", type=float, default=0.35)
    p_sample_test.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Maximum number of predictions kept per image after thresholding.",
    )
    p_sample_test.add_argument("--bbox-margin-px", type=float, default=14.0)
    p_sample_test.add_argument("--episode-tail-fraction", type=float, default=1.0)
    p_sample_test.add_argument("--overlay-subdir", default="images")
    p_sample_test.add_argument("--save-overlays", action="store_true")
    p_sample_test.add_argument("--include-empty-gt", action="store_true")
    p_sample_test.add_argument("--require-both-keypoints", action="store_true")
    p_sample_test.add_argument("--prefer-bin", action="store_true")
    p_sample_test.add_argument("--device", default="auto", help="auto|cuda|cpu|cuda:0")
    p_sample_test.set_defaults(func=run_sample_test_predictions)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.info("Command=%s", args.cmd)
    LOGGER.debug("Parsed args=%s", _safe_cli_args(args))
    if hasattr(args, "val_episodes_root"):
        args.val_episodes_root = args.val_episodes_root or ""
    run_args_path = _write_run_args_json(args)
    if run_args_path is not None:
        LOGGER.info("Saved run args: %s", run_args_path)
    args.func(args)


if __name__ == "__main__":
    main()
