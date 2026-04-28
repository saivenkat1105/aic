#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_NAME="${1:-}"
SUBMISSION_TAG="${2:-}"
DESCRIPTION="${3:-}"

if [[ -z "${POLICY_NAME}" || -z "${SUBMISSION_TAG}" || -z "${DESCRIPTION}" ]]; then
  echo "Usage: ./scripts/submit.sh <policy_name> <tag> <description>"
  exit 1
fi

POLICY_FILE="${ROOT_DIR}/aic_model/aic_model/policies/${POLICY_NAME}.py"
if [[ ! -f "${POLICY_FILE}" ]]; then
  echo "Policy file not found: ${POLICY_FILE}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for submission verification."
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required for submission verification."
  exit 1
fi

if [[ -n "$(git -C "${ROOT_DIR}" status --short)" ]]; then
  echo "Working tree is not clean. Commit or stash your changes before running submit.sh."
  exit 1
fi

ORIGINAL_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"
SUBMISSION_BRANCH="submit/${SUBMISSION_TAG}"
if git -C "${ROOT_DIR}" show-ref --verify --quiet "refs/heads/${SUBMISSION_BRANCH}"; then
  echo "Branch already exists: ${SUBMISSION_BRANCH}"
  exit 1
fi
RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
TEMP_DIR="${ROOT_DIR}/.tmp_submit_${SUBMISSION_TAG}_${RUN_TIMESTAMP}"
SCORING_FILE="${TEMP_DIR}/scoring.yaml"
mkdir -p "${TEMP_DIR}"

cleanup() {
  if [[ -n "${ORIGINAL_BRANCH:-}" ]]; then
    CURRENT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ -n "${CURRENT_BRANCH}" && "${CURRENT_BRANCH}" != "${ORIGINAL_BRANCH}" ]]; then
      git -C "${ROOT_DIR}" checkout "${ORIGINAL_BRANCH}" >/dev/null 2>&1 || true
    fi
  fi
  docker compose -f "${ROOT_DIR}/docker/docker-compose.yaml" down --remove-orphans >/dev/null 2>&1 || true
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

git -C "${ROOT_DIR}" checkout -b "${SUBMISSION_BRANCH}"

cd "${ROOT_DIR}"
pixi run python scripts/tracking_tools.py set-dockerfile-policy --policy "${POLICY_NAME}"

git add docker/my_policy/Dockerfile
git commit -m "submission: ${SUBMISSION_TAG} (${POLICY_NAME})"

docker compose -f docker/docker-compose.yaml build model

set +e
docker compose -f docker/docker-compose.yaml up --abort-on-container-exit --exit-code-from eval
VERIFY_STATUS=$?
set -e

EVAL_CONTAINER_ID="$(docker compose -f docker/docker-compose.yaml ps -q eval)"
if [[ -n "${EVAL_CONTAINER_ID}" ]]; then
  docker cp "${EVAL_CONTAINER_ID}:/root/aic_results/scoring.yaml" "${SCORING_FILE}" >/dev/null 2>&1 || true
fi

if [[ ! -f "${SCORING_FILE}" ]]; then
  echo "Unable to retrieve scoring.yaml from the eval container."
  exit 1
fi

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
ECR_URI_BASE="${AIC_ECR_URI_BASE:-}"
FULL_ECR_URI=""
if [[ -n "${ECR_URI_BASE}" ]]; then
  FULL_ECR_URI="${ECR_URI_BASE}:${SUBMISSION_TAG}"
fi

git -C "${ROOT_DIR}" checkout "${ORIGINAL_BRANCH}"

SUBMISSION_CMD=(
  pixi run python scripts/tracking_tools.py append-submission
  --policy "${POLICY_NAME}"
  --tag "${SUBMISSION_TAG}"
  --description "${DESCRIPTION}"
  --scoring "${SCORING_FILE}"
  --commit "${GIT_COMMIT}"
  --branch "${SUBMISSION_BRANCH}"
  --timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
)
if [[ -n "${FULL_ECR_URI}" ]]; then
  SUBMISSION_CMD+=(--ecr-uri "${FULL_ECR_URI}")
fi
SUBMISSION_ID="$(
  cd "${ROOT_DIR}" && \
  "${SUBMISSION_CMD[@]}"
)"

echo
echo "Recorded submission ${SUBMISSION_ID} in submission_log.yaml on ${ORIGINAL_BRANCH}"
echo "Submission branch: ${SUBMISSION_BRANCH}"
echo "Verify status: ${VERIFY_STATUS}"
echo
if [[ -n "${FULL_ECR_URI}" ]]; then
  echo "Next commands:"
  echo "  docker tag my-solution:v1 ${FULL_ECR_URI}"
  echo "  docker push ${FULL_ECR_URI}"
else
  echo "Set AIC_ECR_URI_BASE to print the exact tag/push commands automatically."
  echo "Example: export AIC_ECR_URI_BASE=973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>"
fi

if [[ ${VERIFY_STATUS} -ne 0 ]]; then
  exit "${VERIFY_STATUS}"
fi
