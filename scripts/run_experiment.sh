#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_NAME="${1:-}"
DESCRIPTION="${2:-}"

if [[ -z "${POLICY_NAME}" || -z "${DESCRIPTION}" ]]; then
  echo "Usage: ./scripts/run_experiment.sh <policy_name> <description>"
  exit 1
fi

POLICY_FILE="${ROOT_DIR}/aic_model/aic_model/policies/${POLICY_NAME}.py"
if [[ ! -f "${POLICY_FILE}" ]]; then
  echo "Policy file not found: ${POLICY_FILE}"
  exit 1
fi

if ! command -v distrobox >/dev/null 2>&1; then
  echo "distrobox is required for local evaluation."
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required for local evaluation."
  exit 1
fi

export DBX_CONTAINER_MANAGER="${DBX_CONTAINER_MANAGER:-docker}"
EVAL_IMAGE="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
EVAL_CONTAINER="${AIC_EVAL_CONTAINER:-aic_eval}"
DISCOVERY_TIMEOUT="${AIC_MODEL_DISCOVERY_TIMEOUT_SECONDS:-120}"
MODEL_CONFIGURE_TIMEOUT="${AIC_MODEL_CONFIGURE_TIMEOUT_SECONDS:-120}"
MAX_WAIT_SECONDS="${AIC_EXPERIMENT_MAX_WAIT_SECONDS:-900}"
POLL_INTERVAL_SECONDS="${AIC_EXPERIMENT_POLL_INTERVAL_SECONDS:-2}"
RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RESULTS_DIR="${AIC_RESULTS_DIR:-$HOME/aic_results}"
RUN_RESULTS_DIR="${RESULTS_DIR}/runs/${RUN_TIMESTAMP}_${POLICY_NAME}"
SCORING_FILE="${RUN_RESULTS_DIR}/scoring.yaml"
EVAL_LOG="${RUN_RESULTS_DIR}/eval.log"
MODEL_LOG="${RUN_RESULTS_DIR}/model.log"

mkdir -p "${RUN_RESULTS_DIR}"

if command -v docker >/dev/null 2>&1; then
  echo "Pulling eval image ${EVAL_IMAGE}..."
  docker pull "${EVAL_IMAGE}"
fi

if ! distrobox list | grep -q "${EVAL_CONTAINER}"; then
  echo "Creating distrobox container '${EVAL_CONTAINER}'..."
  distrobox create -r --nvidia -i "${EVAL_IMAGE}" "${EVAL_CONTAINER}"
fi

cleanup() {
  if [[ -n "${MODEL_PID:-}" ]] && kill -0 "${MODEL_PID}" >/dev/null 2>&1; then
    kill "${MODEL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${EVAL_PID:-}" ]] && kill -0 "${EVAL_PID}" >/dev/null 2>&1; then
    kill "${EVAL_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Starting evaluation container..."
distrobox enter -r "${EVAL_CONTAINER}" -- env \
  AIC_RESULTS_DIR="${RUN_RESULTS_DIR}" \
  /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  model_discovery_timeout_seconds:="${DISCOVERY_TIMEOUT}" \
  model_configure_timeout_seconds:="${MODEL_CONFIGURE_TIMEOUT}" \
  >"${EVAL_LOG}" 2>&1 &
EVAL_PID=$!

sleep 5

echo "Starting policy ${POLICY_NAME}..."
(
  cd "${ROOT_DIR}"
  pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:="aic_model.policies.${POLICY_NAME}"
) >"${MODEL_LOG}" 2>&1 &
MODEL_PID=$!

echo "Waiting for scoring output at ${SCORING_FILE}"
START_TS="$(date +%s)"
MODEL_EXIT_NOTED=0

while true; do
  if [[ -f "${SCORING_FILE}" ]]; then
    echo "Detected scoring file. Finalizing run..."
    EVAL_STATUS=0
    break
  fi

  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED > MAX_WAIT_SECONDS )); then
    echo "Timed out after ${MAX_WAIT_SECONDS}s waiting for scoring output."
    echo "Eval log: ${EVAL_LOG}"
    echo "Model log: ${MODEL_LOG}"
    EVAL_STATUS=124
    break
  fi

  if ! kill -0 "${MODEL_PID}" >/dev/null 2>&1 && [[ "${MODEL_EXIT_NOTED}" -eq 0 ]]; then
    MODEL_EXIT_NOTED=1
    echo "Policy process exited; waiting for evaluation to finish scoring..."
  fi

  if ! kill -0 "${EVAL_PID}" >/dev/null 2>&1; then
    set +e
    wait "${EVAL_PID}"
    EVAL_STATUS=$?
    set -e
    break
  fi

  sleep "${POLL_INTERVAL_SECONDS}"
done

if kill -0 "${MODEL_PID}" >/dev/null 2>&1; then
  kill "${MODEL_PID}" >/dev/null 2>&1 || true
fi
wait "${MODEL_PID}" || true

if kill -0 "${EVAL_PID}" >/dev/null 2>&1; then
  kill "${EVAL_PID}" >/dev/null 2>&1 || true
fi
wait "${EVAL_PID}" || true

if [[ ${EVAL_STATUS} -ne 0 ]]; then
  echo "Evaluation process failed. See ${EVAL_LOG}"
  exit "${EVAL_STATUS}"
fi

if [[ ! -f "${SCORING_FILE}" ]]; then
  echo "Expected scoring file not found: ${SCORING_FILE}"
  echo "Eval log: ${EVAL_LOG}"
  exit 1
fi

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
GIT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"

EXPERIMENT_ID="$(
  cd "${ROOT_DIR}" && \
  pixi run python scripts/tracking_tools.py append-experiment \
    --policy "${POLICY_NAME}" \
    --description "${DESCRIPTION}" \
    --scoring "${SCORING_FILE}" \
    --commit "${GIT_COMMIT}" \
    --branch "${GIT_BRANCH}" \
    --timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
)"

ARCHIVE_DIR="${RESULTS_DIR}/experiments/${GIT_COMMIT}/${EXPERIMENT_ID}_${POLICY_NAME}_${RUN_TIMESTAMP}"
mkdir -p "${ARCHIVE_DIR}"
cp "${SCORING_FILE}" "${ARCHIVE_DIR}/scoring.yaml"
cp "${EVAL_LOG}" "${ARCHIVE_DIR}/eval.log"
cp "${MODEL_LOG}" "${ARCHIVE_DIR}/model.log"

echo
echo "Recorded experiment ${EXPERIMENT_ID}"
echo "Archived raw results to ${ARCHIVE_DIR}"
echo
cd "${ROOT_DIR}"
pixi run python scripts/show_scores.py
