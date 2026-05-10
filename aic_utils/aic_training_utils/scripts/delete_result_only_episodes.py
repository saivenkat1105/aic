#!/usr/bin/env python3

# Copyright (C) 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0

"""Delete empty/failed episode folders that only contain results.json.

The script is dry-run by default. Pass --delete to actually remove matching
episode directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and optionally delete episode folders that only contain results.json."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        help="Path to a run_<timestamp> directory containing an episodes/ subdirectory.",
    )
    group.add_argument(
        "--episodes-root",
        help="Path directly to a directory containing episode_* folders.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matching episode directories. Without this, only report candidates.",
    )
    parser.add_argument(
        "--allow-empty-dirs",
        action="store_true",
        help=(
            "Also treat episodes as candidates when they contain results.json plus "
            "empty directories. By default, any subdirectory prevents deletion."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress after every N episode folders scanned. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_episodes_root(args: argparse.Namespace) -> Path:
    if args.run_dir:
        episodes_root = Path(args.run_dir).expanduser().resolve() / "episodes"
    else:
        episodes_root = Path(args.episodes_root).expanduser().resolve()

    if not episodes_root.is_dir():
        raise RuntimeError(f"episodes directory not found: {episodes_root}")
    return episodes_root


def list_episode_dirs(episodes_root: Path) -> list[Path]:
    return sorted(
        p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )


def is_empty_dir(path: Path) -> bool:
    return not any(path.iterdir())


def classify_episode(episode_dir: Path, allow_empty_dirs: bool) -> tuple[bool, str]:
    results_path = episode_dir / "result.json"
    if not results_path.is_file():
        return False, "missing_results_json"

    files = [p for p in episode_dir.rglob("*") if p.is_file()]
    extra_files = [p for p in files if p != results_path]
    if extra_files:
        return False, "has_other_files"

    dirs = [p for p in episode_dir.rglob("*") if p.is_dir()]
    if dirs and not allow_empty_dirs:
        return False, "has_directories"

    nonempty_dirs = [p for p in dirs if not is_empty_dir(p)]
    if nonempty_dirs:
        return False, "has_nonempty_directories"

    return True, "result_only"


def main() -> None:
    args = parse_args()
    episodes_root = resolve_episodes_root(args)
    episodes = list_episode_dirs(episodes_root)

    scanned = 0
    deleted = 0
    candidates: list[str] = []
    skipped_reasons: dict[str, int] = {}

    for episode_dir in episodes:
        scanned += 1
        should_delete, reason = classify_episode(
            episode_dir,
            allow_empty_dirs=bool(args.allow_empty_dirs),
        )
        if should_delete:
            candidates.append(str(episode_dir))
            if args.delete:
                shutil.rmtree(episode_dir)
                deleted += 1
        else:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        if args.log_every > 0 and scanned % args.log_every == 0:
            print(
                (
                    f"[scan] {scanned}/{len(episodes)} "
                    f"candidates={len(candidates)} deleted={deleted}"
                ),
                flush=True,
            )

    report: dict[str, Any] = {
        "tool": "delete_result_only_episodes",
        "episodes_root": str(episodes_root),
        "delete": bool(args.delete),
        "allow_empty_dirs": bool(args.allow_empty_dirs),
        "episodes_scanned": scanned,
        "candidates": len(candidates),
        "deleted": deleted,
        "skipped_reasons": skipped_reasons,
        "candidate_episode_dirs": candidates,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
