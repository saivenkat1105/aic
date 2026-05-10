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
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
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
        "--source-subdir",
        default="images",
        help=(
            "Per-episode source subdirectory for raw .bin images. Used as a "
            "fallback when frames.jsonl already points at converted debug images."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files.",
    )
    parser.add_argument(
        "--delete-bin-after-convert",
        action="store_true",
        help="Delete each source .bin after its converted output exists.",
    )
    parser.add_argument(
        "--keep-bins-every-nth-episode",
        type=int,
        default=0,
        help=(
            "When deleting source .bin files, keep all .bin files for every Nth "
            "episode after conversion. 0 means keep none."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted/deleted without writing or deleting files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 means auto (cpu_count-1, minimum 1).",
    )
    parser.add_argument(
        "--log-every-episodes",
        type=int,
        default=1,
        help="Print a progress log after every N completed episode folders. Use 0 to disable.",
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


def resolve_bin_path(episode_dir: Path, frame_idx: int, camera: str, meta: dict[str, Any], args: argparse.Namespace) -> tuple[Path, bool]:
    rel = Path(str(meta.get("path", "")))
    metadata_path = episode_dir / rel
    if metadata_path.suffix == ".bin" and metadata_path.is_file():
        return metadata_path, False

    fallback = episode_dir / args.source_subdir / camera / f"{frame_idx:06d}.bin"
    return fallback, True


def save_image(image: Image.Image, out_path: Path, args: argparse.Namespace) -> None:
    if args.format == "webp":
        save_kwargs: dict[str, Any] = {"format": "WEBP"}
        if args.lossless:
            save_kwargs["lossless"] = True
        else:
            save_kwargs["quality"] = int(args.quality)
        image.save(out_path, **save_kwargs)
    else:
        image.save(out_path, format="PNG")


def should_keep_episode_bins(episode_dir: Path, args: argparse.Namespace) -> bool:
    return (
        args.keep_bins_every_nth_episode > 0
        and episode_number(episode_dir) % args.keep_bins_every_nth_episode == 0
    )


def convert_episode(
    episode_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.is_file():
        return {
            "episode_dir": str(episode_dir),
            "episodes_converted": 0,
            "episodes_skipped_missing_frames": 1,
            "frames_converted": 0,
            "skipped_existing_output": 0,
            "skipped_unsupported_encoding": 0,
            "skipped_missing_bin": 0,
            "fallback_bin_paths": 0,
            "deleted_bins": 0,
            "kept_bins": 0,
            "unsupported_encodings": {},
        }

    keep_episode_bins = should_keep_episode_bins(episode_dir, args)
    converted = 0
    skipped_existing_output = 0
    skipped_unsupported = 0
    skipped_missing_bin = 0
    fallback_bin_paths = 0
    deleted_bins = 0
    kept_bins = 0
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

                bin_path, used_fallback = resolve_bin_path(
                    episode_dir,
                    frame_idx=frame_idx,
                    camera=camera,
                    meta=meta,
                    args=args,
                )
                if used_fallback:
                    fallback_bin_paths += 1
                if not bin_path.is_file():
                    skipped_missing_bin += 1
                    continue

                width = int(meta.get("width", 0))
                height = int(meta.get("height", 0))
                step = int(meta.get("step", 0))
                encoding = str(meta.get("encoding", "")).lower()
                if encoding not in SUPPORTED_ENCODINGS:
                    skipped_unsupported += 1
                    unsupported_encodings[encoding] = unsupported_encodings.get(encoding, 0) + 1
                    continue

                out_dir = episode_dir / args.output_subdir / camera
                out_path = out_dir / f"{frame_idx:06d}.{args.format}"
                if out_path.exists() and not args.overwrite:
                    skipped_existing_output += 1
                    if args.delete_bin_after_convert:
                        if keep_episode_bins:
                            kept_bins += 1
                        else:
                            if not args.dry_run:
                                bin_path.unlink(missing_ok=True)
                            deleted_bins += 1
                    continue

                if not args.dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    raw = bin_path.read_bytes()
                    image = decode_ros_image(raw, width=width, height=height, step=step, encoding=encoding)
                    save_image(image, out_path, args=args)

                converted += 1
                if args.delete_bin_after_convert:
                    if keep_episode_bins:
                        kept_bins += 1
                    else:
                        if not args.dry_run:
                            bin_path.unlink(missing_ok=True)
                        deleted_bins += 1

    return {
        "episode_dir": str(episode_dir),
        "episodes_converted": 1,
        "episodes_skipped_missing_frames": 0,
        "frames_converted": converted,
        "skipped_existing_output": skipped_existing_output,
        "skipped_unsupported_encoding": skipped_unsupported,
        "skipped_missing_bin": skipped_missing_bin,
        "fallback_bin_paths": fallback_bin_paths,
        "deleted_bins": deleted_bins,
        "kept_bins": kept_bins,
        "unsupported_encodings": unsupported_encodings,
    }


def resolve_workers(requested: int) -> int:
    if requested >= 1:
        return requested
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def add_episode_report(report: dict[str, Any], episode_report: dict[str, Any]) -> None:
    for key in (
        "episodes_converted",
        "episodes_skipped_missing_frames",
        "frames_converted",
        "skipped_existing_output",
        "skipped_unsupported_encoding",
        "skipped_missing_bin",
        "fallback_bin_paths",
        "deleted_bins",
        "kept_bins",
    ):
        report[key] += int(episode_report[key])

    for enc, count in episode_report["unsupported_encodings"].items():
        report["unsupported_encodings"][enc] = report["unsupported_encodings"].get(enc, 0) + int(count)


def log_episode_done(
    done: int,
    total: int,
    episode_report: dict[str, Any],
    report: dict[str, Any],
    started_at: float,
    log_every_episodes: int,
) -> None:
    if log_every_episodes <= 0:
        return
    if done % log_every_episodes != 0 and done != total:
        return

    elapsed = max(time.monotonic() - started_at, 0.001)
    eps_per_sec = done / elapsed
    remaining = max(total - done, 0)
    eta_sec = remaining / eps_per_sec if eps_per_sec > 0 else 0.0
    episode_name = Path(str(episode_report["episode_dir"])).name
    status = "skipped_missing_frames" if episode_report["episodes_skipped_missing_frames"] else "processed"
    print(
        (
            f"[episode] {done}/{total} {episode_name} {status}: "
            f"converted={episode_report['frames_converted']} "
            f"deleted_bins={episode_report['deleted_bins']} "
            f"kept_bins={episode_report['kept_bins']} "
            f"missing_bins={episode_report['skipped_missing_bin']} "
            f"existing_outputs={episode_report['skipped_existing_output']} | "
            f"totals converted={report['frames_converted']} "
            f"deleted_bins={report['deleted_bins']} "
            f"kept_bins={report['kept_bins']} "
            f"missing_bins={report['skipped_missing_bin']} | "
            f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s"
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"run directory not found: {run_dir}")

    if args.mode == "every_n" and args.every_n < 1:
        raise RuntimeError("--every-n must be >= 1")
    if args.keep_bins_every_nth_episode < 0:
        raise RuntimeError("--keep-bins-every-nth-episode must be >= 0")

    selected_eps = parse_episode_list(args.episodes)
    if args.mode == "episode_list" and not selected_eps:
        raise RuntimeError("--episodes must be provided when --mode episode_list")

    episodes = list_episode_dirs(run_dir)
    workers = resolve_workers(args.workers)
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
        "source_subdir": args.source_subdir,
        "overwrite": bool(args.overwrite),
        "delete_bin_after_convert": bool(args.delete_bin_after_convert),
        "keep_bins_every_nth_episode": int(args.keep_bins_every_nth_episode),
        "dry_run": bool(args.dry_run),
        "workers": workers,
        "episodes_total": len(episodes),
        "episodes_selected": 0,
        "episodes_converted": 0,
        "episodes_skipped_missing_frames": 0,
        "frames_converted": 0,
        "skipped_existing_output": 0,
        "skipped_unsupported_encoding": 0,
        "skipped_missing_bin": 0,
        "fallback_bin_paths": 0,
        "deleted_bins": 0,
        "kept_bins": 0,
        "unsupported_encodings": {},
    }

    selected_episode_dirs = []
    for ep_dir in episodes:
        ep_num = episode_number(ep_dir)
        if not should_convert_episode(ep_num, args=args, selected_eps=selected_eps):
            continue
        report["episodes_selected"] += 1
        selected_episode_dirs.append(ep_dir)

    started_at = time.monotonic()
    done = 0
    if workers == 1:
        for ep_dir in selected_episode_dirs:
            episode_report = convert_episode(ep_dir, args=args)
            done += 1
            add_episode_report(report, episode_report)
            log_episode_done(
                done,
                len(selected_episode_dirs),
                episode_report,
                report,
                started_at,
                args.log_every_episodes,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(convert_episode, ep_dir, args) for ep_dir in selected_episode_dirs]
            for future in as_completed(futures):
                episode_report = future.result()
                done += 1
                add_episode_report(report, episode_report)
                log_episode_done(
                    done,
                    len(selected_episode_dirs),
                    episode_report,
                    report,
                    started_at,
                    args.log_every_episodes,
                )

    out_report = run_dir / f"postprocess_{args.format}_report.json"
    if not args.dry_run:
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
