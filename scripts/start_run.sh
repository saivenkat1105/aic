#!/bin/bash
set -euo pipefail

rand_float() {
    local min="$1"
    local max="$2"
    awk -v min="$min" -v max="$max" -v seed="$RANDOM" 'BEGIN {
        srand(seed);
        printf "%.4f", min + rand() * (max - min);
    }'
}

rand_bool() {
    if (( RANDOM % 2 )); then
        echo "true"
    else
        echo "false"
    fi
}

build_random_training_args() {
    local -n out_args="$1"
    local task_kind="${AIC_TRAINING_TASK_KIND:-random}"

    if [[ "$task_kind" == "random" ]]; then
        if (( RANDOM % 2 )); then
            task_kind="sfp"
        else
            task_kind="sc"
        fi
    fi

    local board_x board_y board_yaw
    board_x="$(rand_float 0.10 0.22)"
    board_y="$(rand_float -0.26 -0.14)"
    board_yaw="$(rand_float 2.95 3.33)"

    out_args=(
        "ground_truth:=true"
        "spawn_task_board:=true"
        "spawn_cable:=true"
        "gazebo_gui:=false"
        "launch_rviz:=false"
        "start_aic_engine:=false"
        "attach_cable_to_gripper:=true"
        "task_board_x:=${board_x}"
        "task_board_y:=${board_y}"
        "task_board_z:=1.14"
        "task_board_roll:=0.0"
        "task_board_pitch:=0.0"
        "task_board_yaw:=${board_yaw}"
    )

    if [[ "$task_kind" == "sfp" ]]; then
        local nic_idx nic_translation nic_yaw sfp_mount_idx sc_mount_idx
        nic_idx=$(( RANDOM % 5 ))
        nic_translation="$(rand_float -0.0215 0.0234)"
        nic_yaw="$(rand_float -0.1745 0.1745)"
        sfp_mount_idx=$(( RANDOM % 2 ))
        sc_mount_idx=$(( RANDOM % 2 ))

        out_args+=(
            "cable_type:=sfp_sc_cable"
            "nic_card_mount_${nic_idx}_present:=true"
            "nic_card_mount_${nic_idx}_translation:=${nic_translation}"
            "nic_card_mount_${nic_idx}_yaw:=${nic_yaw}"
            "sfp_mount_rail_${sfp_mount_idx}_present:=true"
            "sfp_mount_rail_${sfp_mount_idx}_translation:=$(rand_float -0.09425 0.09425)"
            "sc_mount_rail_${sc_mount_idx}_present:=true"
            "sc_mount_rail_${sc_mount_idx}_translation:=$(rand_float -0.09425 0.09425)"
        )
    elif [[ "$task_kind" == "sc" ]]; then
        local sc_idx sc_translation nic_idx sfp_mount_idx sc_mount_idx
        sc_idx=$(( RANDOM % 2 ))
        sc_translation="$(rand_float -0.06 0.055)"
        nic_idx=$(( RANDOM % 5 ))
        sfp_mount_idx=$(( RANDOM % 2 ))
        sc_mount_idx=$(( RANDOM % 2 ))

        out_args+=(
            "cable_type:=sfp_sc_cable_reversed"
            "cable_z:=1.508"
            "sc_port_${sc_idx}_present:=true"
            "sc_port_${sc_idx}_translation:=${sc_translation}"
            "sc_port_${sc_idx}_yaw:=0.0"
            "nic_card_mount_${nic_idx}_present:=$(rand_bool)"
            "sfp_mount_rail_${sfp_mount_idx}_present:=true"
            "sfp_mount_rail_${sfp_mount_idx}_translation:=$(rand_float -0.09425 0.09425)"
            "sc_mount_rail_${sc_mount_idx}_present:=true"
            "sc_mount_rail_${sc_mount_idx}_translation:=$(rand_float -0.09425 0.09425)"
        )
    else
        echo "Unknown AIC_TRAINING_TASK_KIND='${task_kind}'. Use 'random', 'sfp', or 'sc'." >&2
        exit 2
    fi
}

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

launch_args=()
build_random_training_args launch_args

echo "Starting randomized training sample..."
printf '  %s\n' "${launch_args[@]}"
echo "The spawned world will be exported inside the container at /tmp/aic.sdf."

echo "Entering container and starting the simulation engine..."
# The '--' tells distrobox to pass the following command to the container's shell
distrobox enter -r aic_eval -- /entrypoint.sh "${launch_args[@]}"
