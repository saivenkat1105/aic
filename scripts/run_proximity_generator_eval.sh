#!/bin/bash
set -euo pipefail

# Run the training data generator inside eval runtime with required overlay.
# This guarantees aic_engine_interfaces (ResetJoints) is importable.

if [[ ! -f /ws_aic/install/setup.bash ]]; then
  echo "[ERROR] /ws_aic/install/setup.bash not found."
  echo "You are likely not inside a valid aic_eval runtime."
  echo "Recreate and enter a fresh eval container, then rerun:"
  echo "  AIC_EVAL_CONTAINER_NAME=aic_eval_training bash /home/user/aic/scripts/enter_eval.sh"
  exit 1
fi

# setup.bash may reference optional vars (for example COLCON_TRACE) that are
# unset under strict nounset shells (common in tmux workflows).
set +u
source /ws_aic/install/setup.bash
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

echo "[INFO] Running preflight import check for ResetJoints..."
python3 -c "from aic_engine_interfaces.srv import ResetJoints; print('ResetJoints import OK')"

echo "[INFO] Starting proximity_data_generator.py in eval runtime..."
exec python3 /home/user/aic/aic_utils/aic_training_utils/scripts/proximity_data_generator.py "$@"
