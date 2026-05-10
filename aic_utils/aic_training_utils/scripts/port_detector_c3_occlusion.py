#!/usr/bin/env python3
"""Occlusion-aware C3 detector wrapper (fast image-only mode).

This script keeps the original C3 training/inference flow but adds an
on-the-fly label-cleaning stage for manipulator occlusions in existing RGB data.

Implemented fast method:
- Bottom-strip black-mass gate (default: bottom 25%)
- Local keypoint-neighborhood black-ratio check near projected ports
- Per-camera drop only; no cross-camera coupling
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Pillow dependency. Run with Pixi environment."
    ) from exc

import port_detector_c3 as base

LOGGER = logging.getLogger("port_detector_c3_occlusion")


@dataclass(frozen=True)
class OcclusionFilterConfig:
    enabled: bool = True
    black_rgb_max: int = 55
    black_hsv_v_max: int = 70
    black_hsv_s_max: int = 90
    drop_rule: str = "either_occluded"  # both_occluded | either_occluded
    log_occlusion_stats: bool = True
    prefer_bin_for_occlusion: bool = False

    # Fast image-only gate + local patch settings.
    use_fast_bottom_patch_occlusion: bool = True
    bottom_strip_fraction: float = 0.25
    bottom_black_ratio_threshold: float = 0.01
    kp_patch_radius_px: int = 8
    kp_black_ratio_threshold: float = 0.45

    # Backward-compat args retained for CLI compatibility (currently unused).
    black_mask_dilate_k: int = 1
    gripper_roi_w_px: int = 220
    gripper_roi_h_px: int = 220


ACTIVE_OCCLUSION_CONFIG = OcclusionFilterConfig()


@dataclass(frozen=True)
class OcclusionContext:
    width: int
    height: int
    bottom_strip_xyxy: tuple[int, int, int, int]
    bottom_black_ratio: float
    gate_triggered: bool


def _to_bool_arg(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _rgb_to_hsv_sv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Returns S, V in [0,1]. H is intentionally omitted because black detection
    # only needs S and V constraints.
    x = rgb.astype(np.float32) / 255.0
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    s = np.zeros_like(maxc)
    nz = maxc > 1e-6
    s[nz] = delta[nz] / maxc[nz]
    v = maxc
    return s, v


def _black_mask(rgb: np.ndarray, cfg: OcclusionFilterConfig) -> np.ndarray:
    max_rgb = np.max(rgb, axis=2)
    rgb_gate = max_rgb <= int(cfg.black_rgb_max)
    s, v = _rgb_to_hsv_sv(rgb)
    hsv_gate = (v * 255.0 <= float(cfg.black_hsv_v_max)) & (s * 255.0 <= float(cfg.black_hsv_s_max))
    return rgb_gate & hsv_gate


def _bottom_strip_bounds(height: int, width: int, frac: float) -> tuple[int, int, int, int]:
    frac = _clamp01(frac)
    strip_h = max(1, int(round(height * frac)))
    y1 = max(0, height - strip_h)
    y2 = height
    return (0, y1, width, y2)


def _patch_bounds(
    *,
    cx: float,
    cy: float,
    radius: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    r = max(1, int(radius))
    x1 = max(0, int(round(cx)) - r)
    y1 = max(0, int(round(cy)) - r)
    x2 = min(width, int(round(cx)) + r + 1)
    y2 = min(height, int(round(cy)) + r + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _build_occlusion_context(
    *,
    width: int,
    height: int,
    rgb: np.ndarray,
    cfg: OcclusionFilterConfig,
) -> tuple[OcclusionContext | None, str | None]:
    if not cfg.enabled:
        return None, "disabled"
    if not cfg.use_fast_bottom_patch_occlusion:
        return None, "fast_mode_disabled"

    x1, y1, x2, y2 = _bottom_strip_bounds(
        height=height,
        width=width,
        frac=cfg.bottom_strip_fraction,
    )
    strip = rgb[y1:y2, x1:x2]
    if strip.size == 0:
        return None, "bottom_strip_empty"

    strip_black = _black_mask(strip, cfg)
    ratio = float(np.mean(strip_black)) if strip_black.size > 0 else 0.0
    gate_triggered = ratio >= float(cfg.bottom_black_ratio_threshold)
    ctx = OcclusionContext(
        width=width,
        height=height,
        bottom_strip_xyxy=(x1, y1, x2, y2),
        bottom_black_ratio=ratio,
        gate_triggered=gate_triggered,
    )
    return ctx, None


def _keypoint_black_ratio(
    *,
    rgb: np.ndarray,
    u: float,
    v: float,
    cfg: OcclusionFilterConfig,
    width: int,
    height: int,
) -> tuple[float | None, tuple[int, int, int, int] | None]:
    pb = _patch_bounds(
        cx=float(u),
        cy=float(v),
        radius=cfg.kp_patch_radius_px,
        width=width,
        height=height,
    )
    if pb is None:
        return None, None
    x1, y1, x2, y2 = pb
    patch = rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return None, pb
    ratio = float(np.mean(_black_mask(patch, cfg)))
    return ratio, pb


def _apply_occlusion_to_instance(
    *,
    instance: dict[str, Any],
    ctx: OcclusionContext | None,
    cfg: OcclusionFilterConfig,
    rgb: np.ndarray | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    decision = {
        "module_name": str(instance.get("module_name", "")),
        "port_name": str(instance.get("port_name", "")),
        "drop": False,
        "drop_reason": "kept",
        "link_occluded": False,
        "entrance_occluded": False,
        "occlusion_filter_applied": bool(ctx is not None),
        "gate_triggered": bool(ctx.gate_triggered) if ctx is not None else False,
        "bottom_black_ratio": None if ctx is None else float(ctx.bottom_black_ratio),
        "link_patch_black_ratio": None,
        "entrance_patch_black_ratio": None,
        "link_patch_xyxy": None,
        "entrance_patch_xyxy": None,
    }

    if ctx is None or rgb is None:
        decision["drop_reason"] = "occlusion_ctx_unavailable"
        return instance, decision

    if not ctx.gate_triggered:
        decision["drop_reason"] = "gate_not_triggered"
        return instance, decision

    out = dict(instance)
    keypoints = [list(kp) for kp in instance.get("keypoints", [])]
    if len(keypoints) < 2:
        decision["drop_reason"] = "missing_keypoints"
        return out, decision

    width, height = ctx.width, ctx.height
    link_occ = False
    entrance_occ = False

    for idx in (0, 1):
        u, v, vis = keypoints[idx]
        if float(vis) <= 0.0:
            continue

        ratio, patch_xyxy = _keypoint_black_ratio(
            rgb=rgb,
            u=float(u),
            v=float(v),
            cfg=cfg,
            width=width,
            height=height,
        )
        if idx == 0:
            decision["link_patch_black_ratio"] = ratio
            decision["link_patch_xyxy"] = None if patch_xyxy is None else list(patch_xyxy)
        else:
            decision["entrance_patch_black_ratio"] = ratio
            decision["entrance_patch_xyxy"] = None if patch_xyxy is None else list(patch_xyxy)

        if ratio is None:
            continue

        if ratio >= float(cfg.kp_black_ratio_threshold):
            keypoints[idx][2] = 0.0
            if idx == 0:
                link_occ = True
            else:
                entrance_occ = True

    decision["link_occluded"] = link_occ
    decision["entrance_occluded"] = entrance_occ

    if cfg.drop_rule == "both_occluded":
        drop = link_occ and entrance_occ
    else:
        drop = link_occ or entrance_occ

    if drop:
        decision["drop"] = True
        decision["drop_reason"] = cfg.drop_rule
        return None, decision

    out["keypoints"] = keypoints
    if link_occ or entrance_occ:
        decision["drop_reason"] = "partial_occlusion_kept"
    return out, decision


def _frame_iter(
    *,
    episodes_root: Path,
    episode_tail_fraction: float,
):
    episode_dirs = sorted(
        [p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    )
    LOGGER.info("Indexing episodes from %s (count=%d, tail_fraction=%.3f)", episodes_root, len(episode_dirs), episode_tail_fraction)
    for episode_dir in episode_dirs:
        frames_path = episode_dir / "frames.jsonl"
        if not frames_path.is_file():
            LOGGER.warning("Skipping %s (missing frames.jsonl)", episode_dir)
            continue

        # Two-pass streaming so we can keep only tail without loading entire file into memory.
        total = 0
        with frames_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    total += 1
        if total == 0:
            LOGGER.warning("Skipping %s (empty frames.jsonl)", episode_dir)
            continue

        keep_n = total
        if episode_tail_fraction < 1.0:
            keep_n = max(1, int(np.ceil(total * episode_tail_fraction)))

        q: deque[dict[str, Any]] = deque(maxlen=keep_n)
        with frames_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                q.append(json.loads(line))

        LOGGER.info("Episode %s: total_frames=%d kept_tail=%d", episode_dir.name, total, len(q))
        for frame in q:
            yield episode_dir, frame


def build_samples_index_occlusion(
    *,
    episodes_root: Path,
    bbox_margin_px: float,
    require_both_keypoints: bool,
    episode_tail_fraction: float = 1.0,
    collect_decisions: bool = False,
    log_every_n: int = 25,
) -> tuple[list[base.SampleRecord], list[dict[str, Any]]]:
    cfg = ACTIVE_OCCLUSION_CONFIG
    samples: list[base.SampleRecord] = []
    decisions: list[dict[str, Any]] = []

    total_ports = 0
    kept_ports = 0
    dropped_ports = 0
    missing_ctx = 0
    frame_count = 0
    camera_samples = 0
    gate_triggered_samples = 0
    log_every_n = max(1, int(log_every_n))

    for episode_dir, frame in _frame_iter(
        episodes_root=episodes_root,
        episode_tail_fraction=episode_tail_fraction,
    ):
        frame_count += 1
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

        for cam in base.CAMERAS:
            image_meta = images.get(cam)
            if not isinstance(image_meta, dict):
                continue
            width = int(image_meta.get("width", 0))
            height = int(image_meta.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            camera_info = camera_info_all.get(cam, {})
            t_base_to_camera = cam_tfs.get(cam)

            prelim: list[dict[str, Any]] = []
            cam_decisions: list[dict[str, Any]] = []
            for port_gt in all_ports_gt:
                if not isinstance(port_gt, dict):
                    continue
                total_ports += 1
                inst = base._build_port_instance(
                    port_gt=port_gt,
                    t_base_to_camera=t_base_to_camera,
                    camera_info=camera_info,
                    width=width,
                    height=height,
                    bbox_margin_px=bbox_margin_px,
                    require_both_keypoints=require_both_keypoints,
                )
                if inst is None:
                    cam_decisions.append(
                        {
                            "module_name": str(port_gt.get("module_name", "")),
                            "port_name": str(port_gt.get("port_name", "")),
                            "drop": True,
                            "drop_reason": "not_visible_pre_occlusion",
                            "link_occluded": False,
                            "entrance_occluded": False,
                            "occlusion_filter_applied": False,
                            "gate_triggered": False,
                            "bottom_black_ratio": None,
                            "link_patch_black_ratio": None,
                            "entrance_patch_black_ratio": None,
                            "link_patch_xyxy": None,
                            "entrance_patch_xyxy": None,
                        }
                    )
                    continue
                prelim.append(inst)

            occlusion_ctx: OcclusionContext | None = None
            occlusion_err: str | None = None
            rgb: np.ndarray | None = None
            if cfg.enabled and prelim:
                rgb = base._load_rgb_image(
                    episode_dir=episode_dir,
                    image_meta=image_meta,
                    prefer_bin=cfg.prefer_bin_for_occlusion,
                )
                occlusion_ctx, occlusion_err = _build_occlusion_context(
                    width=width,
                    height=height,
                    rgb=rgb,
                    cfg=cfg,
                )
                if occlusion_ctx is None:
                    missing_ctx += 1
                elif occlusion_ctx.gate_triggered:
                    gate_triggered_samples += 1

            kept_instances: list[dict[str, Any]] = []
            for inst in prelim:
                out, decision = _apply_occlusion_to_instance(
                    instance=inst,
                    ctx=occlusion_ctx,
                    cfg=cfg,
                    rgb=rgb,
                )
                if occlusion_ctx is None and occlusion_err:
                    decision["drop_reason"] = occlusion_err
                if out is None:
                    dropped_ports += 1
                else:
                    kept_ports += 1
                    kept_instances.append(out)
                cam_decisions.append(decision)

            samples.append(
                base.SampleRecord(
                    episode_dir=episode_dir,
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    obs_stamp=obs_stamp,
                    camera_name=cam,
                    image_meta=image_meta,
                    camera_info=camera_info,
                    base_to_camera_optical=t_base_to_camera,
                    instances=kept_instances,
                )
            )

            camera_samples += 1
            if camera_samples % log_every_n == 0:
                LOGGER.info(
                    "Indexing progress: frames=%d camera_samples=%d total_ports=%d kept_ports=%d dropped_ports=%d gate_triggered_samples=%d",
                    frame_count,
                    camera_samples,
                    total_ports,
                    kept_ports,
                    dropped_ports,
                    gate_triggered_samples,
                )

            if collect_decisions:
                decisions.append(
                    {
                        "episode_dir": str(episode_dir),
                        "episode_id": episode_id,
                        "frame_idx": frame_idx,
                        "obs_stamp": obs_stamp,
                        "camera_name": cam,
                        "image_meta": image_meta,
                        "decisions": cam_decisions,
                        "bottom_strip": None if occlusion_ctx is None else {
                            "xyxy": list(occlusion_ctx.bottom_strip_xyxy),
                            "black_ratio": float(occlusion_ctx.bottom_black_ratio),
                            "gate_triggered": bool(occlusion_ctx.gate_triggered),
                        },
                    }
                )

    if cfg.log_occlusion_stats:
        LOGGER.info(
            "Occlusion filter stats: enabled=%s total_ports=%d kept_ports=%d dropped_ports=%d missing_ctx=%d gate_triggered_samples=%d",
            cfg.enabled,
            total_ports,
            kept_ports,
            dropped_ports,
            missing_ctx,
            gate_triggered_samples,
        )

    return samples, decisions


def _patched_build_samples_index(
    *,
    episodes_root: Path,
    bbox_margin_px: float,
    require_both_keypoints: bool,
    episode_tail_fraction: float = 1.0,
):
    samples, _ = build_samples_index_occlusion(
        episodes_root=episodes_root,
        bbox_margin_px=bbox_margin_px,
        require_both_keypoints=require_both_keypoints,
        episode_tail_fraction=episode_tail_fraction,
        collect_decisions=False,
        log_every_n=int(getattr(base, "CLI_LOG_EVERY_N", 25)),
    )
    return samples


def _args_to_cfg(args: argparse.Namespace) -> OcclusionFilterConfig:
    enabled_raw = getattr(args, "occlusion_filter_enabled", True)
    return OcclusionFilterConfig(
        enabled=_to_bool_arg(enabled_raw),
        black_rgb_max=int(getattr(args, "black_rgb_max", 55)),
        black_hsv_v_max=int(getattr(args, "black_hsv_v_max", 70)),
        black_hsv_s_max=int(getattr(args, "black_hsv_s_max", 90)),
        drop_rule=str(getattr(args, "drop_rule", "either_occluded")),
        log_occlusion_stats=_to_bool_arg(getattr(args, "log_occlusion_stats", True)),
        prefer_bin_for_occlusion=_to_bool_arg(getattr(args, "prefer_bin_for_occlusion", False)),
        use_fast_bottom_patch_occlusion=_to_bool_arg(getattr(args, "use_fast_bottom_patch_occlusion", True)),
        bottom_strip_fraction=float(getattr(args, "bottom_strip_fraction", 0.25)),
        bottom_black_ratio_threshold=float(getattr(args, "bottom_black_ratio_threshold", 0.01)),
        kp_patch_radius_px=int(getattr(args, "kp_patch_radius_px", 8)),
        kp_black_ratio_threshold=float(getattr(args, "kp_black_ratio_threshold", 0.45)),
        black_mask_dilate_k=int(getattr(args, "black_mask_dilate_k", 1)),
        gripper_roi_w_px=int(getattr(args, "gripper_roi_w_px", 220)),
        gripper_roi_h_px=int(getattr(args, "gripper_roi_h_px", 220)),
    )


def _set_active_cfg_from_args(args: argparse.Namespace) -> None:
    global ACTIVE_OCCLUSION_CONFIG
    ACTIVE_OCCLUSION_CONFIG = _args_to_cfg(args)
    LOGGER.info("Active occlusion config: %s", ACTIVE_OCCLUSION_CONFIG)


def _wrap_base_func(fn):
    def _wrapped(args: argparse.Namespace):
        _set_active_cfg_from_args(args)
        setattr(base, "CLI_LOG_EVERY_N", int(getattr(args, "log_every_n", 25)))
        return fn(args)

    return _wrapped


def _add_occlusion_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--occlusion-filter-enabled", default="true")

    subparser.add_argument("--use-fast-bottom-patch-occlusion", default="true")
    subparser.add_argument("--bottom-strip-fraction", type=float, default=0.25)
    subparser.add_argument("--bottom-black-ratio-threshold", type=float, default=0.01)
    subparser.add_argument("--kp-patch-radius-px", type=int, default=8)
    subparser.add_argument("--kp-black-ratio-threshold", type=float, default=0.45)

    subparser.add_argument("--black-rgb-max", type=int, default=55)
    subparser.add_argument("--black-hsv-v-max", type=int, default=70)
    subparser.add_argument("--black-hsv-s-max", type=int, default=90)

    # Retained for compatibility with prior CLI calls; ignored in fast mode.
    subparser.add_argument("--black-mask-dilate-k", type=int, default=1)
    subparser.add_argument("--gripper-roi-w-px", type=int, default=220)
    subparser.add_argument("--gripper-roi-h-px", type=int, default=220)

    subparser.add_argument(
        "--drop-rule",
        choices=["both_occluded", "either_occluded"],
        default="either_occluded",
    )
    subparser.add_argument("--log-occlusion-stats", default="true")
    subparser.add_argument(
        "--prefer-bin-for-occlusion",
        default="false",
        help="Use .bin instead of debug image when reading occlusion image data.",
    )


def run_preview_occlusion(args: argparse.Namespace) -> None:
    _set_active_cfg_from_args(args)

    episodes_root = Path(args.episodes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, decisions = build_samples_index_occlusion(
        episodes_root=episodes_root,
        bbox_margin_px=float(args.bbox_margin_px),
        require_both_keypoints=bool(args.require_both_keypoints),
        episode_tail_fraction=float(args.episode_tail_fraction),
        collect_decisions=True,
        log_every_n=int(getattr(args, "log_every_n", 25)),
    )

    if args.camera != "all":
        decisions = [d for d in decisions if d["camera_name"] == args.camera]

    if not decisions:
        raise RuntimeError("No occlusion decisions available after filters.")

    rng = np.random.default_rng(int(args.seed))
    k = min(int(args.num_samples), len(decisions))
    idx = rng.choice(len(decisions), size=k, replace=False)
    selected = [decisions[int(i)] for i in idx]

    manifest_path = output_dir / "preview_occlusion_manifest.jsonl"
    saved = 0

    with manifest_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(selected):
            episode_dir = Path(row["episode_dir"])
            image_meta = row["image_meta"]
            rgb = base._load_rgb_image(
                episode_dir=episode_dir,
                image_meta=image_meta,
                prefer_bin=_to_bool_arg(args.prefer_bin_for_occlusion),
            )
            canvas = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(canvas)

            bottom = row.get("bottom_strip")
            if isinstance(bottom, dict):
                xyxy = bottom.get("xyxy", [])
                if isinstance(xyxy, list) and len(xyxy) == 4:
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 165, 0), width=2)

                    # Visualize black points in bottom strip sparsely.
                    cfg = ACTIVE_OCCLUSION_CONFIG
                    strip = rgb[y1:y2, x1:x2]
                    if strip.size > 0:
                        b = _black_mask(strip, cfg)
                        ys, xs = np.where(b)
                        step = max(1, int(args.mask_point_stride_px))
                        for x, y in zip(xs[::step], ys[::step]):
                            draw.point((int(x1 + x), int(y1 + y)), fill=(0, 255, 255))

                ratio = float(bottom.get("black_ratio", 0.0))
                gate = bool(bottom.get("gate_triggered", False))
                draw.text((8, 8), f"bottom_black_ratio={ratio:.3f} gate={gate}", fill=(255, 200, 0))

            for d_i, d in enumerate(row["decisions"]):
                name = f"{d.get('module_name','')}:{d.get('port_name','')}"
                txt = f"{name} {'DROP' if d.get('drop') else 'KEEP'} {d.get('drop_reason','')}"
                color = (255, 64, 64) if d.get("drop") else (64, 220, 64)
                draw.text((8, 28 + d_i * 14), txt, fill=color)

                for patch_key, patch_color in (("link_patch_xyxy", (64, 128, 255)), ("entrance_patch_xyxy", (255, 255, 64))):
                    patch = d.get(patch_key)
                    if isinstance(patch, list) and len(patch) == 4:
                        px1, py1, px2, py2 = [int(v) for v in patch]
                        draw.rectangle([(px1, py1), (px2, py2)], outline=patch_color, width=2)

            out_name = f"{i:04d}_{row['episode_id']}_f{int(row['frame_idx']):06d}_{row['camera_name']}.png"
            out_path = output_dir / out_name
            canvas.save(out_path)
            saved += 1

            manifest_row = {
                "preview_path": str(out_path),
                "episode_id": row["episode_id"],
                "frame_idx": int(row["frame_idx"]),
                "camera_name": row["camera_name"],
                "num_decisions": len(row["decisions"]),
                "num_drop": sum(1 for d in row["decisions"] if d.get("drop")),
            }
            fout.write(json.dumps(manifest_row) + "\n")

    print(
        json.dumps(
            {
                "status": "done",
                "episodes_root": str(episodes_root),
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "saved_images": saved,
                "indexed_samples": len(samples),
            },
            indent=2,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = base.build_arg_parser()

    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break
    if subparsers_action is None:
        raise RuntimeError("Failed to locate subparsers in base parser.")

    for cmd_name, sub in subparsers_action.choices.items():
        if cmd_name in {
            "inspect",
            "train",
            "preview-labels",
            "sample-test-predictions",
        }:
            _add_occlusion_args(sub)
            sub.set_defaults(func=_wrap_base_func(getattr(base, f"run_{cmd_name.replace('-', '_')}", sub.get_default("func"))))

    p_preview_occ = subparsers_action.add_parser(
        "preview-occlusion",
        help="Visual QA for fast bottom-strip + local-patch occlusion filtering decisions.",
    )
    p_preview_occ.add_argument("--episodes-root", required=True)
    p_preview_occ.add_argument("--output-dir", required=True)
    p_preview_occ.add_argument("--num-samples", type=int, default=40)
    p_preview_occ.add_argument("--seed", type=int, default=7)
    p_preview_occ.add_argument("--camera", choices=["all", "left", "center", "right"], default="all")
    p_preview_occ.add_argument("--bbox-margin-px", type=float, default=14.0)
    p_preview_occ.add_argument("--episode-tail-fraction", type=float, default=0.5)
    p_preview_occ.add_argument("--require-both-keypoints", action="store_true")
    p_preview_occ.add_argument("--mask-point-stride-px", type=int, default=40)
    p_preview_occ.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p_preview_occ.add_argument(
        "--log-every-n",
        type=int,
        default=25,
        help="Progress log cadence for long loops.",
    )
    _add_occlusion_args(p_preview_occ)
    p_preview_occ.set_defaults(func=run_preview_occlusion)

    # Keep base functions for subcommands we didn't wrap.
    for cmd_name, sub in subparsers_action.choices.items():
        fn = sub.get_default("func")
        if fn is None:
            continue
        if cmd_name in {"inspect", "train", "preview-labels", "sample-test-predictions", "preview-occlusion"}:
            continue
        sub.set_defaults(func=fn)

    return parser


def _patch_base_build_samples_index() -> None:
    base.build_samples_index = _patched_build_samples_index


def main() -> None:
    _patch_base_build_samples_index()
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(getattr(args, "log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    LOGGER.info("Command=%s", args.cmd)
    args.func(args)


if __name__ == "__main__":
    main()
