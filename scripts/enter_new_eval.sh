#!/bin/bash
set -euo pipefail

IMAGE="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
NAME="${AIC_EVAL_CONTAINER_NAME:-aic_eval}"
STATE_FILE="${AIC_EVAL_STATE_FILE:-/tmp/aic_eval_container_name}"

echo "Setting Docker as the container manager..."
export DBX_CONTAINER_MANAGER=docker

echo "Stopping local AIC helper processes and connections..."
pkill -f "ros2 run aic_model aic_model" >/dev/null 2>&1 || true
pkill -f "rmw_zenohd" >/dev/null 2>&1 || true
pkill -f "publish_low_bandwidth_previews.py" >/dev/null 2>&1 || true
pkill -f "publish_scene_markers.py" >/dev/null 2>&1 || true

echo "Stopping running AIC-related containers..."
mapfile -t aic_containers < <(docker ps --format '{{.Names}}' | grep -E '(^aic_|^aic-|aic_eval|aic_model)' || true)
if [[ ${#aic_containers[@]} -gt 0 ]]; then
  docker stop "${aic_containers[@]}" >/dev/null 2>&1 || true
fi

echo "Removing existing '${NAME}' instances..."
docker stop "${NAME}" >/dev/null 2>&1 || true


echo "Pulling eval image: ${IMAGE}"
docker pull "${IMAGE}"

echo "Creating fresh distrobox container: ${NAME}"
# Note: Remove the --nvidia flag below if you do NOT have an NVIDIA GPU
distrobox create -r --nvidia -i "${IMAGE}" "${NAME}"

echo "Capturing eval container name in ${STATE_FILE}"
printf '%s\n' "${NAME}" > "${STATE_FILE}"

echo "Entering '${NAME}'..."
distrobox enter -r "${NAME}"
