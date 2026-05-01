#!/bin/bash


# The '--' tells distrobox to pass the following command to the container's shell
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true model_discovery_timeout_seconds:=120