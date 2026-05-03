#!/usr/bin/env python3

# Copyright (C) 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0

"""Offline converter for training run frame .bin images to WebP/PNG debug images.

This tool is intentionally post-processing only. It keeps data generation
throughput unaffected by doing image encoding after runs complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8", "mono8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert episode frame .bin images into debug-friendly images."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run_<timestamp> directory containing episodes/ and manifest.json.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "every_n", "episode_list"],
        default="all",
        help="Episode selection mode for conversion.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=10,
        help="Convert every Nth episode when --mode every_n (1-based indexing).",
    )
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma-separated episode numbers when --mode episode_list (example: 1,5,10).",
    )
    parser.add_argument(
        "--format",
        choices=["webp", "png"],
        default="webp",
        help="Output debug image format.",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use lossless encoding for WebP output.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Image quality (used for lossy WebP).",
    )
    parser.add_argument(
        "--output-subdir",
        default="images_debug",
        help="Per-episode output subdirectory for converted images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files.",
    )
    return parser.parse_args()


def parse_episode_list(text: str) -> set[int]:
    selected: set[int] = set()
    if not text.strip():
        return selected
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        selected.add(int(token))
    return selected


def list_episode_dirs(run_dir: Path) -> list[Path]:
    episodes_root = run_dir / "episodes"
    if not episodes_root.is_dir():
        raise RuntimeError(f"episodes directory not found: {episodes_root}")
    return sorted([p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])


def episode_number(episode_dir: Path) -> int:
    suffix = episode_dir.name.replace("episode_", "")
    return int(suffix)


def should_convert_episode(ep_num: int, args: argparse.Namespace, selected_eps: set[int]) -> bool:
    if args.mode == "all":
        return True
    if args.mode == "every_n":
        return ep_num % args.every_n == 0
    return ep_num in selected_eps


def decode_ros_image(raw: bytes, width: int, height: int, step: int, encoding: str) -> Image.Image:
    arr = np.frombuffer(raw, dtype=np.uint8)
    if encoding == "mono8":
        rows = arr.reshape(height, step)[:, :width]
        return Image.fromarray(rows, mode="L")

    if encoding in {"rgb8", "bgr8"}:
        rows = arr.reshape(height, step)[:, : width * 3]
        rgb = rows.reshape(height, width, 3)
        if encoding == "bgr8":
            rgb = rgb[..., ::-1]
        return Image.fromarray(rgb, mode="RGB")

    if encoding in {"rgba8", "bgra8"}:
        rows = arr.reshape(height, step)[:, : width * 4]
        rgba = rows.reshape(height, width, 4)
        if encoding == "bgra8":
            rgba = rgba[..., [2, 1, 0, 3]]
        return Image.fromarray(rgba, mode="RGBA")

    raise ValueError(f"unsupported encoding: {encoding}")


def convert_episode(
    episode_dir: Path,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        report["episodes_skipped_missing_frames"] += 1
        return

    converted = 0
    skipped_unsupported = 0
    skipped_missing_bin = 0

    with open(frames_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            frame_idx = int(frame.get("frame_idx", 0))
            images = frame.get("images", {})
            for camera in ("left", "center", "right"):
                meta = images.get(camera)
                if not isinstance(meta, dict):
                    continue

                rel = Path(str(meta.get("path", "")))
                bin_path = episode_dir / rel
                if not bin_path.is_file():
                    skipped_missing_bin += 1
                    continue

                width = int(meta.get("width", 0))
                height = int(meta.get("height", 0))
                step = int(meta.get("step", 0))
                encoding = str(meta.get("encoding", "")).lower()
                if encoding not in SUPPORTED_ENCODINGS:
                    skipped_unsupported += 1
                    report["unsupported_encodings"][encoding] = (
                        report["unsupported_encodings"].get(encoding, 0) + 1
                    )
                    continue

                out_dir = episode_dir / args.output_subdir / camera
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{frame_idx:06d}.{args.format}"
                if out_path.exists() and not args.overwrite:
                    continue

                raw = bin_path.read_bytes()
                image = decode_ros_image(raw, width=width, height=height, step=step, encoding=encoding)

                if args.format == "webp":
                    save_kwargs: dict[str, Any] = {"format": "WEBP"}
                    if args.lossless:
                        save_kwargs["lossless"] = True
                    else:
                        save_kwargs["quality"] = int(args.quality)
                    image.save(out_path, **save_kwargs)
                else:
                    image.save(out_path, format="PNG")

                converted += 1

    report["episodes_converted"] += 1
    report["frames_converted"] += converted
    report["skipped_unsupported_encoding"] += skipped_unsupported
    report["skipped_missing_bin"] += skipped_missing_bin


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"run directory not found: {run_dir}")

    if args.mode == "every_n" and args.every_n < 1:
        raise RuntimeError("--every-n must be >= 1")

    selected_eps = parse_episode_list(args.episodes)
    if args.mode == "episode_list" and not selected_eps:
        raise RuntimeError("--episodes must be provided when --mode episode_list")

    episodes = list_episode_dirs(run_dir)
    report: dict[str, Any] = {
        "tool": "bin_to_webp_converter",
        "run_dir": str(run_dir),
        "mode": args.mode,
        "every_n": int(args.every_n),
        "episodes_arg": args.episodes,
        "output_format": args.format,
        "lossless": bool(args.lossless),
        "quality": int(args.quality),
        "output_subdir": args.output_subdir,
        "overwrite": bool(args.overwrite),
        "episodes_total": len(episodes),
        "episodes_selected": 0,
        "episodes_converted": 0,
        "episodes_skipped_missing_frames": 0,
        "frames_converted": 0,
        "skipped_unsupported_encoding": 0,
        "skipped_missing_bin": 0,
        "unsupported_encodings": {},
    }

    for ep_dir in episodes:
        ep_num = episode_number(ep_dir)
        if not should_convert_episode(ep_num, args=args, selected_eps=selected_eps):
            continue
        report["episodes_selected"] += 1
        convert_episode(ep_dir, args=args, report=report)

    out_report = run_dir / f"postprocess_{args.format}_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

