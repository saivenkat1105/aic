#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
BACKUP_REPO="${CODEX_BACKUP_REPO:-$HOME/aic/my_aic_setup/codex_sessions/codex_sessions}"
BACKUP_DIR="$BACKUP_REPO/codex"

usage() {
  cat <<'USAGE'
Usage:
  scripts/backup_codex_sessions.sh

Environment:
  CODEX_HOME            Codex config directory. Default: $HOME/.codex
  CODEX_BACKUP_REPO     Git repo to receive the backup.
                        Default: $HOME/aic/my_aic_setup/codex_sessions/codex_sessions

This copies:
  - $CODEX_HOME/sessions/
  - $CODEX_HOME/session_index.jsonl
  - $CODEX_HOME/config.toml
  - $CODEX_HOME/memories/
  - $CODEX_HOME/skills/
  - $CODEX_HOME/version.json

It intentionally does not copy auth.json, SQLite state, caches, logs, tmp data,
or other files that commonly contain credentials or machine-local state.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$CODEX_HOME/sessions" ]]; then
  echo "Missing Codex sessions directory: $CODEX_HOME/sessions" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

rsync -a --delete "$CODEX_HOME/sessions/" "$BACKUP_DIR/sessions/"

if [[ -f "$CODEX_HOME/session_index.jsonl" ]]; then
  cp "$CODEX_HOME/session_index.jsonl" "$BACKUP_DIR/session_index.jsonl"
fi

if [[ -f "$CODEX_HOME/config.toml" ]]; then
  cp "$CODEX_HOME/config.toml" "$BACKUP_DIR/config.toml"
fi

if [[ -d "$CODEX_HOME/memories" ]]; then
  rsync -a --delete "$CODEX_HOME/memories/" "$BACKUP_DIR/memories/"
fi

if [[ -d "$CODEX_HOME/skills" ]]; then
  rsync -a --delete \
    --exclude '.system/' \
    "$CODEX_HOME/skills/" "$BACKUP_DIR/skills/"
fi

if [[ -f "$CODEX_HOME/version.json" ]]; then
  cp "$CODEX_HOME/version.json" "$BACKUP_DIR/version.json"
fi

cat >"$BACKUP_REPO/README.md" <<'README'
# Codex Session Backup

This repository stores selected local Codex state copied from `~/.codex`.

Included:
- `codex/sessions/`
- `codex/session_index.jsonl`
- `codex/config.toml`
- `codex/memories/`
- `codex/skills/`
- `codex/version.json`

Excluded on purpose:
- `auth.json`
- SQLite state and WAL/SHM files
- logs
- caches
- tmp directories
- bundled `.system` skills

Keep this repository private. Session transcripts may contain prompts, code,
paths, credentials pasted into chat, or other sensitive project details.
README

{
  echo "Backup created at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Source: $CODEX_HOME"
  echo "Destination: $BACKUP_DIR"
} >"$BACKUP_DIR/MANIFEST.txt"

echo "Codex state copied to: $BACKUP_DIR"
echo "Review and commit manually from: $BACKUP_REPO"
