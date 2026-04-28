#!/bin/bash

echo "Starting Foxglove Bridge as Root to bypass permissions..."
echo "Applying ROS 2 overlays and Zenoh middleware..."

# Symlink aic_assets into the ROS install path so package:// URIs resolve correctly
# This is the same mechanism that makes the robot's ur_description meshes work.
echo "Linking aic_assets into ROS install path..."
docker exec aic_eval bash -c "ln -sfn /home/user/aic/aic_assets /ws_aic/install/share/aic_assets"

# Run the Foxglove bridge
docker exec -it aic_eval bash -c "source /opt/ros/kilted/setup.bash && \
source /ws_aic/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_zenoh_cpp && \
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p 'asset_uri_allowlist:=[\"^package://.*\"]'"

echo "Bridge closed."