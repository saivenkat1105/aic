#!/bin/bash
set -euo pipefail

export DBX_CONTAINER_MANAGER=docker

STATE_FILE="${AIC_EVAL_STATE_FILE:-/tmp/aic_eval_container_name}"
NAME="${AIC_EVAL_CONTAINER_NAME:-}"

if [[ -z "${NAME}" && -f "${STATE_FILE}" ]]; then
  NAME="$(<"${STATE_FILE}")"
fi

if [[ -z "${NAME}" ]]; then
  NAME="aic_eval"
fi

echo "Entering eval container: ${NAME}"
distrobox enter -r "${NAME}"
