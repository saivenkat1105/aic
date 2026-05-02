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
    local board_x board_y board_yaw
    local nic_rail_min="-0.0215"
    local nic_rail_max="0.0234"
    local sc_rail_min="-0.0600"
    local sc_rail_max="0.0550"
    local mount_rail_min="-0.09425"
    local mount_rail_max="0.09425"

    if [[ "$task_kind" == "random" ]]; then
        if (( RANDOM % 2 )); then
            task_kind="sfp"
        else
            task_kind="sc"
        fi
    fi

    # Mirror the organizer sample-config envelope instead of using a wider
    # custom range. The shipped trials keep roll/pitch fixed at 0, z fixed
    # at 1.14, and vary the board inside this observed x/y/yaw box.
    board_x="$(rand_float 0.15 0.17)"
    board_y="$(rand_float -0.20 0.00)"
    board_yaw="$(rand_float 3.00 3.1415)"

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
        nic_translation="$(rand_float "$nic_rail_min" "$nic_rail_max")"
        nic_yaw="$(rand_float -0.1745 0.1745)"
        sfp_mount_idx=$(( RANDOM % 2 ))
        sc_mount_idx=$(( RANDOM % 2 ))

        out_args+=(
            "cable_type:=sfp_sc_cable"
            "nic_card_mount_${nic_idx}_present:=true"
            "nic_card_mount_${nic_idx}_translation:=${nic_translation}"
            "nic_card_mount_${nic_idx}_yaw:=${nic_yaw}"
            "sfp_mount_rail_${sfp_mount_idx}_present:=true"
            "sfp_mount_rail_${sfp_mount_idx}_translation:=$(rand_float "$mount_rail_min" "$mount_rail_max")"
            "sc_mount_rail_${sc_mount_idx}_present:=true"
            "sc_mount_rail_${sc_mount_idx}_translation:=$(rand_float "$mount_rail_min" "$mount_rail_max")"
        )
    elif [[ "$task_kind" == "sc" ]]; then
        local sc_idx sc_translation nic_idx sfp_mount_idx sc_mount_idx other_sc_idx
        sc_idx=$(( RANDOM % 2 ))
        sc_translation="$(rand_float "$sc_rail_min" "$sc_rail_max")"
        nic_idx=$(( RANDOM % 5 ))
        sfp_mount_idx=$(( RANDOM % 2 ))
        sc_mount_idx=$(( RANDOM % 2 ))
        other_sc_idx=$(( 1 - sc_idx ))

        out_args+=(
            "cable_type:=sfp_sc_cable_reversed"
            "cable_z:=1.508"
            "sc_port_${sc_idx}_present:=true"
            "sc_port_${sc_idx}_translation:=${sc_translation}"
            "sc_port_${sc_idx}_yaw:=0.0"
            "sc_port_${other_sc_idx}_present:=$(rand_bool)"
            "sc_port_${other_sc_idx}_translation:=$(rand_float "$sc_rail_min" "$sc_rail_max")"
            "sc_port_${other_sc_idx}_yaw:=0.0"
            "nic_card_mount_${nic_idx}_present:=$(rand_bool)"
            "sfp_mount_rail_${sfp_mount_idx}_present:=true"
            "sfp_mount_rail_${sfp_mount_idx}_translation:=$(rand_float "$mount_rail_min" "$mount_rail_max")"
            "sc_mount_rail_${sc_mount_idx}_present:=true"
            "sc_mount_rail_${sc_mount_idx}_translation:=$(rand_float "$mount_rail_min" "$mount_rail_max")"
        )
    else
        echo "Unknown AIC_TRAINING_TASK_KIND='${task_kind}'. Use 'random', 'sfp', or 'sc'." >&2
        exit 2
    fi
}

launch_args=()
build_random_training_args launch_args

echo "Starting randomized training sample..."
printf '  %s\n' "${launch_args[@]}"
echo "The spawned world will be exported inside the container at /tmp/aic.sdf."

echo "Starting the simulation engine..."
/entrypoint.sh "${launch_args[@]}"
