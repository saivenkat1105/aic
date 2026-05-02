#!/bin/bash

echo "Starting Foxglove Bridge as Root to bypass permissions..."
echo "Applying ROS 2 overlays and Zenoh middleware..."

echo "Preparing aic_assets package paths for Foxglove mesh asset URIs..."
docker exec aic_eval bash -c "
set -e
asset_share=/ws_aic/install/share/aic_assets
if [ ! -d \"\${asset_share}/models\" ]; then
  ln -sfn /home/user/aic/aic_assets \"\${asset_share}\"
fi

cd \"\${asset_share}/models\"
alias_model() {
  source_dir=\"\$1\"
  alias_dir=\"\$2\"
  if [ -e \"\${alias_dir}\" ]; then
    return
  fi
  if [ -e \"\${source_dir}\" ]; then
    ln -s \"\${source_dir}\" \"\${alias_dir}\"
  fi
}

alias_model 'Task Board Base' Task_Board_Base
alias_model 'NIC Card Mount' NIC_Card_Mount
alias_model 'NIC Card' NIC_Card
alias_model 'SC Port' SC_Port
alias_model 'LC Mount' LC_Mount
alias_model 'SFP Mount' SFP_Mount
alias_model 'SC Mount' SC_Mount
alias_model 'SC Plug' SC_Plug
alias_model 'LC Plug' LC_Plug
alias_model 'SFP Module' SFP_Module
"

echo "Stopping any existing Foxglove Bridge..."
docker exec aic_eval bash -c "pkill -f foxglove_bridge || true"
sleep 1

# Run the Foxglove bridge
docker exec -it aic_eval bash -c "source /opt/ros/kilted/setup.bash && \
source /ws_aic/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_zenoh_cpp && \
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p 'asset_uri_allowlist:=[\"^package://.*\"]'"

echo "Bridge closed."
