#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime environment issue
    raise SystemExit(
        "PyYAML is required for tracking scripts. Run through Pixi or install pyyaml."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_LOG = ROOT / "experiment_log.yaml"
SUBMISSION_LOG = ROOT / "submission_log.yaml"
DEFAULT_DOCKERFILE = ROOT / "docker" / "my_policy" / "Dockerfile"
TRIAL_KEYS = ["trial_1", "trial_2", "trial_3"]
TIER2_CATEGORY_MAP = {
    "smoothness": "trajectory smoothness",
    "duration": "duration",
    "efficiency": "trajectory efficiency",
    "force": "insertion force",
    "contacts": "contacts",
}


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)


def ensure_log(path: Path, root_key: str) -> dict[str, Any]:
    if not path.exists():
        data = {root_key: []}
        dump_yaml(path, data)
        return data
    data = load_yaml(path)
    data.setdefault(root_key, [])
    return data


def next_id(entries: list[dict[str, Any]], prefix: str = "") -> str:
    max_id = 0
    for entry in entries:
        raw = str(entry.get("id", ""))
        number_part = raw[len(prefix) :] if prefix and raw.startswith(prefix) else raw
        if number_part.isdigit():
            max_id = max(max_id, int(number_part))
    return f"{prefix}{max_id + 1:03d}"


def score_value(node: dict[str, Any], *keys: str) -> float:
    value: Any = node
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_scoring_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scoring file not found: {path}")
    scoring = load_yaml(path)
    result: dict[str, Any] = {
        "total": float(scoring.get("total", 0.0) or 0.0),
        "scores": {},
        "tier2_details": {},
    }
    for trial_key in TRIAL_KEYS:
        trial_node = scoring.get(trial_key, {}) or {}
        trial_summary = {
            "total": (
                score_value(trial_node, "tier_1", "score")
                + score_value(trial_node, "tier_2", "score")
                + score_value(trial_node, "tier_3", "score")
            ),
            "tier1": score_value(trial_node, "tier_1", "score"),
            "tier2": score_value(trial_node, "tier_2", "score"),
            "tier3": score_value(trial_node, "tier_3", "score"),
        }
        categories = ((trial_node.get("tier_2") or {}).get("categories") or {}) if isinstance(trial_node, dict) else {}
        tier2_details = {
            public_name: score_value(categories, internal_name, "score")
            for public_name, internal_name in TIER2_CATEGORY_MAP.items()
        }
        result["scores"][trial_key] = trial_summary
        result["tier2_details"][trial_key] = tier2_details
    return result


def append_experiment_entry(args: argparse.Namespace) -> int:
    log = ensure_log(EXPERIMENT_LOG, "experiments")
    parsed_scores = parse_scoring_yaml(Path(args.scoring))
    experiment_id = next_id(log["experiments"])
    entry = {
        "id": experiment_id,
        "timestamp": args.timestamp or iso_timestamp(),
        "policy": args.policy,
        "policy_path": f"aic_model/aic_model/policies/{args.policy}.py",
        "git_commit": args.commit,
        "git_branch": args.branch,
        "description": args.description,
        "scores": {
            "total": parsed_scores["total"],
            **parsed_scores["scores"],
        },
        "tier2_details": parsed_scores["tier2_details"],
        "notes": args.notes or "",
    }
    log["experiments"].append(entry)
    dump_yaml(EXPERIMENT_LOG, log)
    print(experiment_id)
    return 0


def append_submission_entry(args: argparse.Namespace) -> int:
    log = ensure_log(SUBMISSION_LOG, "submissions")
    parsed_scores = parse_scoring_yaml(Path(args.scoring))
    submission_id = next_id(log["submissions"], prefix="S")
    entry = {
        "id": submission_id,
        "timestamp": args.timestamp or iso_timestamp(),
        "policy": args.policy,
        "git_branch": args.branch,
        "git_commit": args.commit,
        "docker_tag": args.tag,
        "ecr_uri": args.ecr_uri or "",
        "description": args.description,
        "local_scores": {
            "total": parsed_scores["total"],
            **{
                trial_key: parsed_scores["scores"][trial_key]["total"]
                for trial_key in TRIAL_KEYS
            },
        },
        "official_scores": {
            "total": None,
            "trial_1": None,
            "trial_2": None,
            "trial_3": None,
            "status": "pending",
        },
        "notes": args.notes or "",
    }
    log["submissions"].append(entry)
    dump_yaml(SUBMISSION_LOG, log)
    print(submission_id)
    return 0


def update_dockerfile_policy(args: argparse.Namespace) -> int:
    dockerfile = Path(args.dockerfile)
    contents = dockerfile.read_text(encoding="utf-8")
    new_cmd = (
        f'CMD ["--ros-args", "-p", "policy:=aic_model.policies.{args.policy}", '
        f'"-p", "use_sim_time:=true"]'
    )
    updated, count = re.subn(
        r'^CMD \["--ros-args".*policy:=.*$',
        new_cmd,
        contents,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"Unable to update Dockerfile CMD in {dockerfile}")
    dockerfile.write_text(updated, encoding="utf-8")
    return 0


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def build_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def border(left: str, mid: str, right: str, fill: str = "═") -> str:
        return left + mid.join(fill * (width + 2) for width in widths) + right

    def render_row(row: list[str]) -> str:
        return "║ " + " ║ ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " ║"

    output = [
        border("╔", "╦", "╗"),
        render_row(headers),
        border("╠", "╬", "╣"),
    ]
    output.extend(render_row(row) for row in rows)
    output.append(border("╚", "╩", "╝"))
    return "\n".join(output)


def experiment_rows(entries: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        scores = entry.get("scores", {})
        rows.append(
            [
                str(entry.get("id", "")),
                str(entry.get("policy", "")),
                str(entry.get("git_commit", ""))[:7],
                format_float((scores.get("trial_1") or {}).get("total")),
                format_float((scores.get("trial_2") or {}).get("total")),
                format_float((scores.get("trial_3") or {}).get("total")),
                format_float(scores.get("total")),
            ]
        )
    return rows


def submission_rows(entries: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        local_scores = entry.get("local_scores", {})
        official_scores = entry.get("official_scores", {})
        rows.append(
            [
                str(entry.get("id", "")),
                str(entry.get("policy", "")),
                str(entry.get("git_branch", "")),
                format_float(local_scores.get("trial_1")),
                format_float(local_scores.get("trial_2")),
                format_float(local_scores.get("trial_3")),
                format_float(local_scores.get("total")),
                str(official_scores.get("status", "")),
            ]
        )
    return rows


def show_detail(entry: dict[str, Any], mode: str) -> str:
    if mode == "experiments":
        return yaml.safe_dump(entry, sort_keys=False, allow_unicode=False).rstrip()
    return yaml.safe_dump(entry, sort_keys=False, allow_unicode=False).rstrip()


def show_scores(args: argparse.Namespace) -> int:
    mode = "submissions" if args.submissions else "experiments"
    log_path = SUBMISSION_LOG if args.submissions else EXPERIMENT_LOG
    log = ensure_log(log_path, mode)
    entries = log.get(mode, [])
    if not entries:
        print(f"No {mode} recorded yet.")
        return 0

    if args.detail:
        for entry in entries:
            if str(entry.get("id")) == args.detail:
                print(show_detail(entry, mode))
                return 0
        raise SystemExit(f"Could not find {mode[:-1]} with id {args.detail}")

    if args.submissions:
        headers = [
            "ID",
            "Policy",
            "Branch",
            "Local T1",
            "Local T2",
            "Local T3",
            "Local Total",
            "Status",
        ]
        rows = submission_rows(entries)
    else:
        headers = ["ID", "Policy", "Commit", "Trial 1", "Trial 2", "Trial 3", "Total"]
        rows = experiment_rows(entries)

    print(build_table(headers, rows))

    if args.submissions:
        completed = [
            entry
            for entry in entries
            if entry.get("official_scores", {}).get("total") is not None
        ]
        if completed:
            best = max(completed, key=lambda item: float(item["official_scores"]["total"]))
            print(
                "Best official: "
                f"{best['id']} {best['policy']} — {format_float(best['official_scores']['total'])}"
            )
        else:
            best = max(entries, key=lambda item: float(item["local_scores"]["total"]))
            print(
                "Best local verify: "
                f"{best['id']} {best['policy']} — {format_float(best['local_scores']['total'])}"
            )
    else:
        best = max(entries, key=lambda item: float(item["scores"]["total"]))
        print(
            f"Best: #{best['id']} {best['policy']} — {format_float(best['scores']['total'])}/300"
        )
    return 0


def dump_scoring_json(args: argparse.Namespace) -> int:
    print(json.dumps(parse_scoring_yaml(Path(args.scoring)), indent=2, sort_keys=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment tracking utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exp = subparsers.add_parser("append-experiment", help="Append an experiment entry.")
    exp.add_argument("--policy", required=True)
    exp.add_argument("--description", required=True)
    exp.add_argument("--scoring", required=True)
    exp.add_argument("--commit", required=True)
    exp.add_argument("--branch", required=True)
    exp.add_argument("--timestamp")
    exp.add_argument("--notes")
    exp.set_defaults(func=append_experiment_entry)

    sub = subparsers.add_parser("append-submission", help="Append a submission entry.")
    sub.add_argument("--policy", required=True)
    sub.add_argument("--tag", required=True)
    sub.add_argument("--description", required=True)
    sub.add_argument("--scoring", required=True)
    sub.add_argument("--commit", required=True)
    sub.add_argument("--branch", required=True)
    sub.add_argument("--ecr-uri")
    sub.add_argument("--timestamp")
    sub.add_argument("--notes")
    sub.set_defaults(func=append_submission_entry)

    scores = subparsers.add_parser("show-scores", help="Render the score dashboard.")
    scores.add_argument("--detail")
    scores.add_argument("--submissions", action="store_true")
    scores.set_defaults(func=show_scores)

    docker = subparsers.add_parser("set-dockerfile-policy", help="Update Dockerfile CMD.")
    docker.add_argument("--policy", required=True)
    docker.add_argument("--dockerfile", default=str(DEFAULT_DOCKERFILE))
    docker.set_defaults(func=update_dockerfile_policy)

    parse = subparsers.add_parser("parse-scoring", help="Print parsed scoring as JSON.")
    parse.add_argument("--scoring", required=True)
    parse.set_defaults(func=dump_scoring_json)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
