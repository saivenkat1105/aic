#!/bin/bash

echo "Starting Scene Markers Publisher..."
echo "This will publish 3D meshes of the taskboard, ports, and cables to Foxglove."

# Run the publish_scene_markers.py script inside the eval container
# using the Zenoh middleware so Foxglove Bridge can discover the topic.
docker exec -it aic_eval bash -c "source /opt/ros/kilted/setup.bash && \
source /ws_aic/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_zenoh_cpp && \
python3 /home/user/aic/aic_utils/aic_foxglove/aic_foxglove/publish_scene_markers.py"

echo "Scene markers publisher stopped."
