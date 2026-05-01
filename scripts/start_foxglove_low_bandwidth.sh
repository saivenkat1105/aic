#!/bin/bash
set -euo pipefail

preview_pid=""

cleanup() {
    if [[ -n "${preview_pid}" ]]; then
        echo "Stopping Foxglove preview publisher..."
        kill "${preview_pid}" 2>/dev/null || true
        wait "${preview_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting low-bandwidth Foxglove stack..."
echo "Starting compressed camera preview publisher in the background..."
scripts/start_foxglove_preview.sh &
preview_pid=$!

sleep 2

echo "Starting Foxglove Bridge..."
scripts/start_foxglove.sh
