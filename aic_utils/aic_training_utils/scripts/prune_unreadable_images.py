#!/usr/bin/env python3
"""Delete unreadable image files under a directory tree.

Default mode is dry-run (no deletion). Use --apply to delete files.
Supports parallel per-episode scanning for speed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Pillow dependency. Run with Pixi environment."
    ) from exc

LOGGER = logging.getLogger("prune_unreadable_images")


@dataclass
class EpisodeResult:
    episode: str
    scanned: int
    unreadable: int
    deleted: int
    delete_errors: int
    examples: list[dict[str, str]]


def _iter_candidates(root: Path, exts: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts:
            yield path


def _is_readable_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            im2.load()
        return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _scan_one_episode(
    episode_dir: str,
    exts: list[str],
    apply: bool,
    max_errors_to_print: int,
) -> EpisodeResult:
    root = Path(episode_dir)
    extset = set(exts)
    scanned = 0
    unreadable = 0
    deleted = 0
    delete_errors = 0
    examples: list[dict[str, str]] = []

    for path in _iter_candidates(root, extset):
        scanned += 1
        ok, err = _is_readable_image(path)
        if ok:
            continue

        unreadable += 1
        if len(examples) < max_errors_to_print:
            examples.append({"path": str(path), "error": str(err or "unknown")})

        if apply:
            try:
                path.unlink()
                deleted += 1
            except Exception as exc:  # pragma: no cover
                delete_errors += 1
                if len(examples) < max_errors_to_print:
                    examples.append({"path": str(path), "error": f"delete_failed: {exc}"})

    return EpisodeResult(
        episode=root.name,
        scanned=scanned,
        unreadable=unreadable,
        deleted=deleted,
        delete_errors=delete_errors,
        examples=examples,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune unreadable image files under a root folder.")
    p.add_argument("--root", required=True, help="Folder to scan recursively.")
    p.add_argument(
        "--extensions",
        default="webp,png,jpg,jpeg,bmp,tif,tiff",
        help="Comma-separated extensions (without dots) to check.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete unreadable files. Without this flag, only report (dry-run).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel worker processes for per-episode scanning.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--max-errors-to-print",
        type=int,
        default=20,
        help="Maximum unreadable file examples to print overall.",
    )
    return p


def _episode_dirs_or_root(root: Path) -> list[Path]:
    eps = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if eps:
        return eps
    return [root]


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f"Root directory not found: {root}")

    ext_tokens = [t.strip().lower().lstrip(".") for t in str(args.extensions).split(",") if t.strip()]
    if not ext_tokens:
        raise RuntimeError("No extensions provided.")
    exts = [f".{e}" for e in ext_tokens]

    episode_dirs = _episode_dirs_or_root(root)
    LOGGER.info(
        "Starting scan root=%s episodes=%d workers=%d apply=%s",
        root,
        len(episode_dirs),
        int(args.workers),
        bool(args.apply),
    )

    total_scanned = 0
    total_unreadable = 0
    total_deleted = 0
    total_delete_errors = 0
    printed_examples = 0

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {
            ex.submit(
                _scan_one_episode,
                str(ep),
                exts,
                bool(args.apply),
                int(args.max_errors_to_print),
            ): ep
            for ep in episode_dirs
        }

        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            total_scanned += res.scanned
            total_unreadable += res.unreadable
            total_deleted += res.deleted
            total_delete_errors += res.delete_errors

            LOGGER.info(
                "Episode scanned %d/%d episode=%s scanned=%d unreadable=%d deleted=%d delete_errors=%d",
                done,
                len(episode_dirs),
                res.episode,
                res.scanned,
                res.unreadable,
                res.deleted,
                res.delete_errors,
            )

            for row in res.examples:
                if printed_examples >= int(args.max_errors_to_print):
                    break
                LOGGER.warning("Unreadable: %s | %s", row["path"], row["error"])
                printed_examples += 1

    summary = {
        "status": "done",
        "root": str(root),
        "episodes_scanned": len(episode_dirs),
        "workers": int(args.workers),
        "apply": bool(args.apply),
        "extensions": sorted(exts),
        "scanned": total_scanned,
        "unreadable": total_unreadable,
        "deleted": total_deleted,
        "delete_errors": total_delete_errors,
        "printed_unreadable_examples": printed_examples,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
