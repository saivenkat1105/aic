#!/bin/bash
set -euo pipefail

echo "Starting low-bandwidth Foxglove camera previews..."
echo "Publishing compressed preview topics under /foxglove/*."

docker exec -it aic_eval bash -c "source /opt/ros/kilted/setup.bash && \
source /ws_aic/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_zenoh_cpp && \
export PYTHONPATH=/home/user/aic/aic_utils/aic_foxglove:\${PYTHONPATH:-} && \
python3 -m aic_foxglove.publish_low_bandwidth_previews --ros-args \
  -p preview_width:=576 \
  -p preview_height:=512 \
  -p jpeg_quality:=70 \
  -p fps:=5.0"

echo "Foxglove preview publisher stopped."
