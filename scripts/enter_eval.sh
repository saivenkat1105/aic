#!/bin/bash
sudo chmod a+rw /dev/dri/*

echo "Setting Docker as the container manager..."
export DBX_CONTAINER_MANAGER=docker

echo "Pulling the latest image..."
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest

# Check if the container already exists. If not, create it.
if ! distrobox list | grep -q "aic_eval"; then
    echo "Container 'aic_eval' not found. Creating it now..."
    # Note: Remove the --nvidia flag below if you do NOT have an NVIDIA GPU
    distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
else
    echo "Container 'aic_eval' already exists. Skipping creation."
fi

echo "Entering container and starting the simulation engine..."
# The '--' tells distrobox to pass the following command to the container's shell
distrobox enter -r aic_eval 