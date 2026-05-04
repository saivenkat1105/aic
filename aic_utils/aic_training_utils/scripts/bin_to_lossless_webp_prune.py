#!/usr/bin/env python3

# Copyright (C) 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0

"""Convert frame .bin images to lossless WebP and prune .bin files.

This script recursively scans a user-provided subfolder for episode directories
containing frames.jsonl. For each frame image entry, it:
1) decodes .bin image payload using ROS image metadata from frames.jsonl
2) writes a lossless WebP into images_debug/<camera>/<frame>.webp
3) deletes .bin files except every Nth file as requested
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8", "mono8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert .bin frame images to lossless WebP and delete "
            "all .bin files except every Nth file."
        )
    )
    parser.add_argument(
        "--subfolder",
        required=True,
        help=(
            "Root subfolder to scan recursively (for example: "
            "/home/user/training_data/visual_motor_policy/campaign_20260503_abc123)."
        ),
    )
    parser.add_argument(
        "--keep-every-nth-bin",
        type=int,
        default=10,
        help=(
            "Keep every Nth .bin image file and delete the rest after successful "
            "conversion (1-based indexing within each camera stream)."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default="images_debug",
        help="Per-episode output subdirectory for WebP files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing WebP files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write/delete files; print report only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes. 0 means auto (cpu_count-1, minimum 1). "
            "Use 1 to force single-process mode."
        ),
    )
    parser.add_argument(
        "--log-every-folders",
        type=int,
        default=5,
        help="Print a log line every N processed episode folders.",
    )
    return parser.parse_args()


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


def find_episode_dirs(root: Path) -> list[Path]:
    episodes: list[Path] = []
    for frames in root.rglob("frames.jsonl"):
        ep = frames.parent
        if ep.is_dir():
            episodes.append(ep)
    # Stable deterministic ordering
    return sorted(set(episodes))


def camera_index(counters: dict[str, int], camera: str) -> int:
    counters[camera] = counters.get(camera, 0) + 1
    return counters[camera]


def process_episode(
    episode_dir: Path,
    args: argparse.Namespace,
    ) -> dict[str, Any]:
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        return {
            "episode_dir": str(episode_dir),
            "processed": False,
            "missing_frames": True,
            "webp_converted": 0,
            "bin_missing": 0,
            "unsupported_images": 0,
            "bin_deleted": 0,
            "bin_kept": 0,
            "unsupported_encodings": {},
        }

    cam_counts: dict[str, int] = {}
    converted = 0
    missing_bin = 0
    unsupported = 0
    deleted = 0
    kept = 0
    unsupported_encodings: dict[str, int] = {}

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

                stream_idx = camera_index(cam_counts, camera)
                rel = Path(str(meta.get("path", "")))
                bin_path = episode_dir / rel
                if not bin_path.is_file():
                    missing_bin += 1
                    continue

                width = int(meta.get("width", 0))
                height = int(meta.get("height", 0))
                step = int(meta.get("step", 0))
                encoding = str(meta.get("encoding", "")).lower()
                if encoding not in SUPPORTED_ENCODINGS:
                    unsupported += 1
                    unsupported_encodings[encoding] = unsupported_encodings.get(encoding, 0) + 1
                    continue

                out_dir = episode_dir / args.output_subdir / camera
                out_path = out_dir / f"{frame_idx:06d}.webp"
                if args.overwrite or not out_path.exists():
                    if not args.dry_run:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        raw = bin_path.read_bytes()
                        image = decode_ros_image(
                            raw,
                            width=width,
                            height=height,
                            step=step,
                            encoding=encoding,
                        )
                        image.save(out_path, format="WEBP", lossless=True)
                    converted += 1

                keep_this_bin = (stream_idx % args.keep_every_nth_bin) == 0
                if keep_this_bin:
                    kept += 1
                else:
                    if not args.dry_run:
                        bin_path.unlink(missing_ok=True)
                    deleted += 1

    return {
        "episode_dir": str(episode_dir),
        "processed": True,
        "missing_frames": False,
        "webp_converted": converted,
        "bin_missing": missing_bin,
        "unsupported_images": unsupported,
        "bin_deleted": deleted,
        "bin_kept": kept,
        "unsupported_encodings": unsupported_encodings,
    }


def _progress(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------] 0/0 (0.0%)"
    ratio = min(max(done / total, 0.0), 1.0)
    fill = int(ratio * width)
    bar = "#" * fill + "-" * (width - fill)
    return f"[{bar}] {done}/{total} ({ratio * 100.0:5.1f}%)"


def _emit_progress(done: int, total: int) -> None:
    msg = _progress(done, total)
    if sys.stdout.isatty():
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        # In non-tty environments (logs/redirect), print line-by-line.
        print(msg, flush=True)


def _resolve_workers(requested: int) -> int:
    if requested >= 1:
        return requested
    import os

    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def main() -> None:
    args = parse_args()
    if args.keep_every_nth_bin < 1:
        raise RuntimeError("--keep-every-nth-bin must be >= 1")

    root = Path(args.subfolder).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"subfolder not found: {root}")

    episodes = find_episode_dirs(root)
    total_episodes = len(episodes)
    workers = _resolve_workers(args.workers)
    report: dict[str, Any] = {
        "tool": "bin_to_lossless_webp_prune",
        "subfolder": str(root),
        "episodes_found": total_episodes,
        "episodes_processed": 0,
        "episodes_skipped_missing_frames": 0,
        "keep_every_nth_bin": int(args.keep_every_nth_bin),
        "output_subdir": args.output_subdir,
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "workers": workers,
        "log_every_folders": int(args.log_every_folders),
        "webp_converted": 0,
        "bin_deleted": 0,
        "bin_kept": 0,
        "bin_missing": 0,
        "unsupported_images": 0,
        "unsupported_encodings": {},
    }

    if total_episodes == 0:
        _emit_progress(0, 0)
    elif workers == 1:
        done = 0
        for episode_dir in episodes:
            result = process_episode(episode_dir, args=args)
            done += 1
            if result["missing_frames"]:
                report["episodes_skipped_missing_frames"] += 1
            if result["processed"]:
                report["episodes_processed"] += 1
            report["webp_converted"] += int(result["webp_converted"])
            report["bin_missing"] += int(result["bin_missing"])
            report["unsupported_images"] += int(result["unsupported_images"])
            report["bin_deleted"] += int(result["bin_deleted"])
            report["bin_kept"] += int(result["bin_kept"])
            for enc, cnt in result["unsupported_encodings"].items():
                report["unsupported_encodings"][enc] = report["unsupported_encodings"].get(enc, 0) + int(cnt)
            _emit_progress(done, total_episodes)
            if args.log_every_folders > 0 and done % args.log_every_folders == 0:
                print(
                    f"[log] processed {done} folders, cumulative bin_deleted={report['bin_deleted']}",
                    flush=True,
                )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_episode, episode_dir, args) for episode_dir in episodes]
            for fut in as_completed(futures):
                result = fut.result()
                done += 1
                if result["missing_frames"]:
                    report["episodes_skipped_missing_frames"] += 1
                if result["processed"]:
                    report["episodes_processed"] += 1
                report["webp_converted"] += int(result["webp_converted"])
                report["bin_missing"] += int(result["bin_missing"])
                report["unsupported_images"] += int(result["unsupported_images"])
                report["bin_deleted"] += int(result["bin_deleted"])
                report["bin_kept"] += int(result["bin_kept"])
                for enc, cnt in result["unsupported_encodings"].items():
                    report["unsupported_encodings"][enc] = report["unsupported_encodings"].get(enc, 0) + int(cnt)
                _emit_progress(done, total_episodes)
                if args.log_every_folders > 0 and done % args.log_every_folders == 0:
                    print(
                        f"[log] processed {done} folders, cumulative bin_deleted={report['bin_deleted']}",
                        flush=True,
                    )

    out_report = root / "postprocess_lossless_webp_prune_report.json"
    if not args.dry_run:
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
