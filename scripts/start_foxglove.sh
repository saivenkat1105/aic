#!/bin/bash

echo "Starting Foxglove Bridge as Root to bypass permissions..."
echo "Applying ROS 2 overlays and Zenoh middleware..."

# Run the injection command
docker exec -it aic_eval bash -c "source /opt/ros/kilted/setup.bash && \
source /ws_aic/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_zenoh_cpp && \
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p asset_uri_allowlist:=\"['^package://.*']\""

echo "Bridge closed."