#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
preview_pid=""
markers_pid=""

cleanup() {
    if [[ -n "${markers_pid}" ]]; then
        echo "Stopping scene marker publisher..."
        kill "${markers_pid}" 2>/dev/null || true
        wait "${markers_pid}" 2>/dev/null || true
    fi
    if [[ -n "${preview_pid}" ]]; then
        echo "Stopping Foxglove preview publisher..."
        kill "${preview_pid}" 2>/dev/null || true
        wait "${preview_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting low-bandwidth Foxglove stack..."
echo "Starting compressed camera preview publisher in the background..."
"${script_dir}/start_foxglove_preview.sh" &
preview_pid=$!



sleep 2

echo "Starting Foxglove Bridge..."
"${script_dir}/start_foxglove.sh"
