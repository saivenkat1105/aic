#!/usr/bin/env python3

# Copyright (C) 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0

"""Baseline ProximityTeacher training data generator.

This script runs a single-worker episode loop:
1) sample randomized task-board scene
2) delete + respawn entities
3) tare FT sensor
4) trigger ProximityTeacher via /insert_cable
5) capture per-frame data and scoring streams
6) write per-episode artifacts (including local score summary)

Assumes simulation is already running in training mode:
- ground_truth:=true
- start_aic_engine:=false
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate
from aic_engine_interfaces.srv import ResetJoints
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.action import InsertCable
from aic_training_interfaces.srv import (
    CaptureStatus,
    ExpandXacro,
    StartEpisodeCapture,
    StopEpisodeCapture,
)
try:
    from controller_manager_msgs.srv import SwitchController
except ImportError:
    SwitchController = None
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from simulation_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener
from PIL import Image


@dataclasses.dataclass
class EpisodeSpec:
    episode_index: int
    seed: int
    task_board_pose: dict[str, float]
    nic_mounts: list[dict[str, Any]]
    sfp_rail_0_translation: float
    sfp_rail_1_translation: float
    port_name: str
    target_module_name: str
    cable_type: str


class ProximityDataGenerator(Node):
    def __init__(self) -> None:
        super().__init__("proximity_data_generator")

        self.declare_parameter("num_episodes", 10)
        self.declare_parameter("seed", 42)
        self.declare_parameter("output_root", str(Path.home() / "aic_training_data"))
        self.declare_parameter("task_time_limit_s", 90)
        self.declare_parameter("model_node_name", "aic_model")
        self.declare_parameter("reset_joints_after_episode", True)
        self.declare_parameter("pre_action_settle_s", 0.1)
        self.declare_parameter("post_action_settle_s", 0.1)
        self.declare_parameter("postprocess_webp_after_episode", False)
        self.declare_parameter("postprocess_delete_bin_after_webp", True)
        self.declare_parameter("postprocess_workers", 2)
        self.declare_parameter("postprocess_output_subdir", "images_debug")
        self.declare_parameter("use_frame_sink", True)
        self.declare_parameter("frame_sink_service_ns", "/training_frame_sink")
        self.declare_parameter("frame_sink_flush_timeout_s", 120.0)
        self.declare_parameter("frame_sink_require_webp_done", True)
        self.declare_parameter("frame_sink_enable_cameras", ["left", "center", "right"])
        self.declare_parameter("use_controller_switch_for_reset", True)
        self.declare_parameter("switch_controller_service", "/controller_manager/switch_controller")
        self.declare_parameter("aic_controller_name", "aic_controller")
        self.declare_parameter("deactivate_model_between_episodes", True)
        self.declare_parameter("lifecycle_transition_timeout_s", 60.0)
        self.declare_parameter("lifecycle_transition_retries", 3)
        self.declare_parameter(
            "home_joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.declare_parameter(
            "home_joint_positions",
            [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
        )

        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.seed = int(self.get_parameter("seed").value)
        self.output_root = Path(str(self.get_parameter("output_root").value)).expanduser()
        self.task_time_limit_s = int(self.get_parameter("task_time_limit_s").value)
        self.model_node_name = str(self.get_parameter("model_node_name").value)
        self.reset_joints_after_episode = bool(
            self.get_parameter("reset_joints_after_episode").value
        )
        self.pre_action_settle_s = float(self.get_parameter("pre_action_settle_s").value)
        self.post_action_settle_s = float(self.get_parameter("post_action_settle_s").value)
        if self.pre_action_settle_s < 0.0:
            self.pre_action_settle_s = 0.0
        if self.post_action_settle_s < 0.0:
            self.post_action_settle_s = 0.0
        self.postprocess_webp_after_episode = bool(
            self.get_parameter("postprocess_webp_after_episode").value
        )
        self.postprocess_delete_bin_after_webp = bool(
            self.get_parameter("postprocess_delete_bin_after_webp").value
        )
        self.postprocess_workers = int(self.get_parameter("postprocess_workers").value)
        self.postprocess_output_subdir = str(
            self.get_parameter("postprocess_output_subdir").value
        )
        self.use_frame_sink = bool(self.get_parameter("use_frame_sink").value)
        self.frame_sink_service_ns = str(self.get_parameter("frame_sink_service_ns").value).rstrip("/")
        self.frame_sink_flush_timeout_s = float(
            self.get_parameter("frame_sink_flush_timeout_s").value
        )
        self.frame_sink_require_webp_done = bool(
            self.get_parameter("frame_sink_require_webp_done").value
        )
        self.frame_sink_enable_cameras = [
            str(v) for v in self.get_parameter("frame_sink_enable_cameras").value
        ]
        self.use_controller_switch_for_reset = bool(
            self.get_parameter("use_controller_switch_for_reset").value
        )
        self.switch_controller_service = str(
            self.get_parameter("switch_controller_service").value
        )
        self.aic_controller_name = str(self.get_parameter("aic_controller_name").value)
        self.deactivate_model_between_episodes = bool(
            self.get_parameter("deactivate_model_between_episodes").value
        )
        self.lifecycle_transition_timeout_s = float(
            self.get_parameter("lifecycle_transition_timeout_s").value
        )
        self.lifecycle_transition_retries = int(
            self.get_parameter("lifecycle_transition_retries").value
        )
        if self.lifecycle_transition_retries < 1:
            self.lifecycle_transition_retries = 1
        if self.use_controller_switch_for_reset and SwitchController is None:
            raise RuntimeError(
                "use_controller_switch_for_reset is true but "
                "controller_manager_msgs/SwitchController is not available in this runtime"
            )
        if self.postprocess_workers < 1:
            self.postprocess_workers = 1
        self.home_joint_names = [
            str(v) for v in self.get_parameter("home_joint_names").value
        ]
        self.home_joint_positions = [
            float(v) for v in self.get_parameter("home_joint_positions").value
        ]
        if len(self.home_joint_names) != len(self.home_joint_positions):
            raise RuntimeError(
                "home_joint_names and home_joint_positions must have the same length"
            )

        random.seed(self.seed)

        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / f"run_{run_stamp}"
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = Buffer()
        # Keep TF processing on the node's regular spin path to avoid a
        # background thread lifecycle race during shutdown.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.expand_xacro_client = self.create_client(ExpandXacro, "/expand_xacro")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/gz_server/spawn_entity")
        self.delete_entity_client = self.create_client(DeleteEntity, "/gz_server/delete_entity")
        self.tare_client = self.create_client(Trigger, "/aic_controller/tare_force_torque_sensor")
        self.reset_joints_client = self.create_client(ResetJoints, "/scoring/reset_joints")
        self.get_state_client = self.create_client(GetState, f"/{self.model_node_name}/get_state")
        self.change_state_client = self.create_client(
            ChangeState, f"/{self.model_node_name}/change_state"
        )
        self.insert_cable_client = ActionClient(self, InsertCable, "/insert_cable")
        self.start_capture_client = self.create_client(
            StartEpisodeCapture, f"{self.frame_sink_service_ns}/start_episode_capture"
        )
        self.stop_capture_client = self.create_client(
            StopEpisodeCapture, f"{self.frame_sink_service_ns}/stop_episode_capture"
        )
        self.capture_status_client = self.create_client(
            CaptureStatus, f"{self.frame_sink_service_ns}/capture_status"
        )
        self.switch_controller_client = None
        if self.use_controller_switch_for_reset and SwitchController is not None:
            self.switch_controller_client = self.create_client(
                SwitchController, self.switch_controller_service
            )

        self.create_subscription(Observation, "/observations", self._on_observation, 10)
        self.create_subscription(MotionUpdate, "/aic_controller/pose_commands", self._on_pose_cmd, 50)
        self.create_subscription(
            JointMotionUpdate, "/aic_controller/joint_commands", self._on_joint_cmd, 50
        )
        self.create_subscription(String, "/scoring/insertion_event", self._on_insertion_event, 50)
        self.create_subscription(Contacts, "/aic/gazebo/contacts/off_limit", self._on_contacts, 50)
        self.create_subscription(
            TFMessage, "/scoring/tf", self._on_scoring_tf, 100
        )
        self.create_subscription(
            TFMessage, "/scoring/tf_static", self._on_scoring_tf_static, 10
        )

        self.current_episode: dict[str, Any] | None = None
        self.latest_pose_cmd: dict[str, Any] | None = None
        self.latest_joint_cmd: dict[str, Any] | None = None
        self.prev_episode_first_tcp_xyz: tuple[float, float, float] | None = None
        self.prev_episode_index: int | None = None
        self.postprocess_executor: ThreadPoolExecutor | None = None
        if self.use_frame_sink and self.postprocess_webp_after_episode:
            self.get_logger().warn(
                "postprocess_webp_after_episode is ignored when use_frame_sink=true; "
                "frame sink owns async image/postprocess pipeline."
            )
            self.postprocess_webp_after_episode = False
        self.postprocess_futures: list[Future] = []
        if self.postprocess_webp_after_episode:
            self.postprocess_executor = ThreadPoolExecutor(
                max_workers=self.postprocess_workers,
                thread_name_prefix="episode_postprocess",
            )

    # ------------------------ ROS callbacks ------------------------

    def _on_pose_cmd(self, msg: MotionUpdate) -> None:
        self.latest_pose_cmd = {
            "stamp": self._stamp_to_sec(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "mode": int(msg.trajectory_generation_mode.mode),
            "velocity": {
                "linear": {
                    "x": float(msg.velocity.linear.x),
                    "y": float(msg.velocity.linear.y),
                    "z": float(msg.velocity.linear.z),
                },
                "angular": {
                    "x": float(msg.velocity.angular.x),
                    "y": float(msg.velocity.angular.y),
                    "z": float(msg.velocity.angular.z),
                },
            },
            "pose": {
                "position": {
                    "x": float(msg.pose.position.x),
                    "y": float(msg.pose.position.y),
                    "z": float(msg.pose.position.z),
                },
                "orientation": {
                    "x": float(msg.pose.orientation.x),
                    "y": float(msg.pose.orientation.y),
                    "z": float(msg.pose.orientation.z),
                    "w": float(msg.pose.orientation.w),
                },
            },
        }

    def _on_joint_cmd(self, msg: JointMotionUpdate) -> None:
        self.latest_joint_cmd = {
            "stamp": self._stamp_to_sec(msg.header.stamp),
            "mode": int(msg.trajectory_generation_mode.mode),
            "positions": [float(v) for v in msg.target_state.positions],
            "velocities": [float(v) for v in msg.target_state.velocities],
            "stiffness": [float(v) for v in msg.target_stiffness],
            "damping": [float(v) for v in msg.target_damping],
            "feedforward_torque": [float(v) for v in msg.target_feedforward_torque],
        }

    def _on_insertion_event(self, msg: String) -> None:
        if self.current_episode is None:
            return
        self.current_episode["insertion_events"].append({
            "t": time.time(),
            "event": msg.data,
        })

    def _on_contacts(self, msg: Contacts) -> None:
        if self.current_episode is None:
            return
        if len(msg.contacts) > 0:
            self.current_episode["contact_events"] += 1

    def _on_scoring_tf(self, msg: TFMessage) -> None:
        if self.current_episode is None:
            return
        self.current_episode["scoring_tf_count"] += len(msg.transforms)

    def _on_scoring_tf_static(self, msg: TFMessage) -> None:
        if self.current_episode is None:
            return
        self.current_episode["scoring_tf_static_count"] += len(msg.transforms)

    def _on_observation(self, msg: Observation) -> None:
        if self.current_episode is None:
            return

        ep = self.current_episode
        frame_idx = ep["frame_count"]

        vx = float(msg.controller_state.tcp_velocity.linear.x)
        vy = float(msg.controller_state.tcp_velocity.linear.y)
        vz = float(msg.controller_state.tcp_velocity.linear.z)
        speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))

        # Path metric for local efficiency estimate.
        tcp_p = msg.controller_state.tcp_pose.position
        tcp_xyz = (float(tcp_p.x), float(tcp_p.y), float(tcp_p.z))
        if ep["last_tcp_xyz"] is not None:
            prev = ep["last_tcp_xyz"]
            ep["path_length_m"] += float(
                math.sqrt(
                    (tcp_xyz[0] - prev[0]) ** 2
                    + (tcp_xyz[1] - prev[1]) ** 2
                    + (tcp_xyz[2] - prev[2]) ** 2
                )
            )
        ep["last_tcp_xyz"] = tcp_xyz

        ep["vel_samples"].append(
            {
                "t": self._stamp_to_sec(msg.center_image.header.stamp),
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": speed,
            }
        )

        if frame_idx == 0:
            ep["first_tcp_xyz"] = tcp_xyz
        if frame_idx == 0 and self.prev_episode_first_tcp_xyz is not None:
            prev = self.prev_episode_first_tcp_xyz
            dx = tcp_xyz[0] - prev[0]
            dy = tcp_xyz[1] - prev[1]
            dz = tcp_xyz[2] - prev[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            prev_ep = (
                str(self.prev_episode_index)
                if self.prev_episode_index is not None
                else "unknown"
            )
            self.get_logger().info(
                "Reset check transition: "
                f"prev_episode={prev_ep} prev_first_tcp=({prev[0]:.6f}, {prev[1]:.6f}, {prev[2]:.6f}) "
                f"current_episode={ep.get('episode_index', 'unknown')} "
                f"current_first_tcp=({tcp_xyz[0]:.6f}, {tcp_xyz[1]:.6f}, {tcp_xyz[2]:.6f}) "
                f"delta=({dx:.6f}, {dy:.6f}, {dz:.6f}) dist={dist:.6f}m"
            )

        ep["frame_count"] += 1

    # ------------------------ Episode lifecycle ------------------------

    def run(self) -> None:
        self._wait_for_readiness()

        self._write_run_manifest()

        for i in range(self.num_episodes):
            episode_seed = self.seed + i
            spec = self._sample_episode_spec(i, episode_seed)
            self.get_logger().info(
                f"Starting episode {i + 1}/{self.num_episodes} with seed={episode_seed}"
            )

            episode_dir = self.episodes_dir / f"episode_{i + 1:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            sink_started = False
            sink_frame_count = None
            try:
                self._delete_entity("cable_0")
                self._delete_entity("task_board")

                scene = self._spawn_scene(spec)
                self._tare_ft_sensor()

                task = self._make_task(spec)
                task_json = self._task_to_dict(task)

                with open(episode_dir / "scene.json", "w", encoding="utf-8") as f:
                    json.dump(scene, f, indent=2)
                with open(episode_dir / "task.json", "w", encoding="utf-8") as f:
                    json.dump(task_json, f, indent=2)

                # Match organizer flow: model should be active for task execution.
                self._ensure_model_active()

                self.current_episode = {
                    "episode_index": i + 1,
                    "episode_dir": str(episode_dir),
                    "seed": episode_seed,
                    "task": task_json,
                    "scene": scene,
                    "frame_count": 0,
                    "path_length_m": 0.0,
                    "first_tcp_xyz": None,
                    "last_tcp_xyz": None,
                    "vel_samples": [],
                    "insertion_events": [],
                    "contact_events": 0,
                    "scoring_tf_count": 0,
                    "scoring_tf_static_count": 0,
                    "start_wall_time": time.time(),
                }

                start_dist_m = self._lookup_plug_port_distance(task)
                if start_dist_m is not None:
                    self.current_episode["start_plug_port_distance_m"] = start_dist_m

                if self.use_frame_sink:
                    self._start_episode_capture(episode_id=f"episode_{i + 1:06d}", episode_dir=episode_dir)
                    sink_started = True

                self._capture_window(self.pre_action_settle_s, "pre_action")
                result = self._run_insert_cable(task)
                self._capture_window(self.post_action_settle_s, "post_action")

                if self.use_frame_sink and sink_started:
                    sink_frame_count = self._stop_episode_capture(episode_id=f"episode_{i + 1:06d}")
                    sink_started = False

                end_dist_m = self._lookup_plug_port_distance(task)
                if end_dist_m is not None:
                    self.current_episode["end_plug_port_distance_m"] = end_dist_m

                duration_s = max(0.0, time.time() - self.current_episode["start_wall_time"])

                score_summary = self._estimate_score(
                    task=task,
                    duration_s=duration_s,
                    path_length_m=float(self.current_episode["path_length_m"]),
                    vel_samples=self.current_episode["vel_samples"],
                    insertion_events=self.current_episode["insertion_events"],
                    start_dist_m=self.current_episode.get("start_plug_port_distance_m"),
                    end_dist_m=self.current_episode.get("end_plug_port_distance_m"),
                    contact_events=int(self.current_episode["contact_events"]),
                )

                result_artifact = {
                    "goal_status": result.get("status"),
                    "success": result.get("success"),
                    "message": result.get("message"),
                    "duration_s": duration_s,
                    "frame_count": int(
                        sink_frame_count
                        if sink_frame_count is not None
                        else self.current_episode["frame_count"]
                    ),
                    "path_length_m": float(self.current_episode["path_length_m"]),
                }

                with open(episode_dir / "result.json", "w", encoding="utf-8") as f:
                    json.dump(result_artifact, f, indent=2)

                with open(episode_dir / "score_summary.json", "w", encoding="utf-8") as f:
                    json.dump(score_summary, f, indent=2)

                self.get_logger().info(
                    f"Episode {i + 1} done: success={result_artifact['success']} "
                    f"score_total={score_summary['total_score_estimate']:.3f} "
                    f"frames={result_artifact['frame_count']}"
                )
            except Exception as exc:
                self.get_logger().error(f"Episode {i + 1} failed: {exc}")
                with open(episode_dir / "result.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "goal_status": "episode_error",
                            "success": False,
                            "message": str(exc),
                            "duration_s": None,
                            "frame_count": int(
                                self.current_episode["frame_count"]
                                if self.current_episode is not None
                                else 0
                            ),
                        },
                        f,
                        indent=2,
                    )
            finally:
                if self.current_episode is not None:
                    self.prev_episode_first_tcp_xyz = self.current_episode.get("first_tcp_xyz")
                    self.prev_episode_index = self.current_episode.get("episode_index")
                if self.use_frame_sink and sink_started:
                    try:
                        self._stop_episode_capture(episode_id=f"episode_{i + 1:06d}")
                    except Exception as exc:
                        self.get_logger().error(f"StopEpisodeCapture failed during cleanup: {exc}")
                if self.postprocess_webp_after_episode:
                    self._schedule_episode_postprocess(episode_dir)
                if self.deactivate_model_between_episodes:
                    try:
                        self._deactivate_model_if_active()
                    except Exception as exc:
                        self.get_logger().error(f"Model deactivate after episode failed: {exc}")
                # Match organizer flow by cleaning up entities before homing reset.
                self._delete_entity("cable_0")
                self._delete_entity("task_board")
                if self.reset_joints_after_episode:
                    try:
                        self._reset_joints_to_home()
                    except Exception as exc:
                        self.get_logger().error(f"Joint reset after episode failed: {exc}")
                self.current_episode = None

        self._wait_for_postprocess_jobs()
        self.get_logger().info("All episodes completed.")

    # ------------------------ Scene + task ------------------------

    def _sample_episode_spec(self, episode_index: int, seed: int) -> EpisodeSpec:
        rng = random.Random(seed)
        # Organizer docs for qualification specify random NIC translation on rail
        # plus random yaw offset; keep roll/pitch fixed to avoid cross-rail clashes.
        nic_yaw_limit_rad = math.radians(10.0)
        present_count = rng.randint(1, 5)
        present_indices = sorted(rng.sample(range(5), present_count))
        nic_mounts: list[dict[str, Any]] = []
        for idx in present_indices:
            nic_mounts.append(
                {
                    "index": idx,
                    "translation": rng.uniform(-0.0215, 0.0234),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rng.uniform(-nic_yaw_limit_rad, nic_yaw_limit_rad),
                }
            )
        target_mount = rng.choice(nic_mounts)
        port_name = rng.choice(["sfp_port_0", "sfp_port_1"])
        return EpisodeSpec(
            episode_index=episode_index,
            seed=seed,
            task_board_pose={
                "x": 0.15 + rng.uniform(-0.02, 0.02),
                "y": -0.2 + rng.uniform(-0.02, 0.02),
                "z": 1.14,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 3.1415 + rng.uniform(-0.15, 0.15),
            },
            nic_mounts=nic_mounts,
            sfp_rail_0_translation=rng.uniform(-0.05, 0.05),
            sfp_rail_1_translation=rng.uniform(-0.05, 0.05),
            port_name=port_name,
            target_module_name=f"nic_card_mount_{int(target_mount['index'])}",
            cable_type="sfp_sc_cable",
        )

    def _spawn_scene(self, spec: EpisodeSpec) -> dict[str, Any]:
        tb_args = [
            f"x:={spec.task_board_pose['x']}",
            f"y:={spec.task_board_pose['y']}",
            f"z:={spec.task_board_pose['z']}",
            f"roll:={spec.task_board_pose['roll']}",
            f"pitch:={spec.task_board_pose['pitch']}",
            f"yaw:={spec.task_board_pose['yaw']}",
            "ground_truth:=true",
            "sfp_mount_rail_0_present:=true",
            "sfp_mount_rail_1_present:=true",
            f"sfp_mount_rail_0_translation:={spec.sfp_rail_0_translation}",
            f"sfp_mount_rail_1_translation:={spec.sfp_rail_1_translation}",
        ]

        mounts_by_index = {
            int(mount["index"]): mount for mount in spec.nic_mounts
        }
        for i in range(5):
            mount = mounts_by_index.get(i)
            present = "true" if mount is not None else "false"
            tb_args.append(f"nic_card_mount_{i}_present:={present}")
            if mount is not None:
                tb_args.append(
                    f"nic_card_mount_{i}_translation:={float(mount['translation'])}"
                )
                tb_args.append(f"nic_card_mount_{i}_roll:={float(mount['roll'])}")
                tb_args.append(f"nic_card_mount_{i}_pitch:={float(mount['pitch'])}")
                tb_args.append(f"nic_card_mount_{i}_yaw:={float(mount['yaw'])}")

        tb_xml = self._expand_xacro("aic_description", "urdf/task_board.urdf.xacro", tb_args)

        cable_args = ["attach_cable_to_gripper:=true", f"cable_type:={spec.cable_type}"]
        cable_xml = self._expand_xacro("aic_description", "urdf/cable.sdf.xacro", cable_args)

        self._spawn_entity(
            name="task_board",
            xml=tb_xml,
            pose=spec.task_board_pose,
        )

        gripper_tf = self._lookup_transform("world", "gripper/tcp")
        if gripper_tf is None:
            raise RuntimeError("Could not look up world <- gripper/tcp for cable spawn")

        cable_roll = 0.4432
        cable_pitch = -0.4838
        cable_yaw = 1.3303
        cable_pose = {
            "x": float(gripper_tf.translation.x + 0.0),
            "y": float(gripper_tf.translation.y + 0.015385),
            "z": float(gripper_tf.translation.z + 0.04245),
            "roll": cable_roll,
            "pitch": cable_pitch,
            "yaw": cable_yaw,
        }

        self._spawn_entity(name="cable_0", xml=cable_xml, pose=cable_pose)

        mount_locations: list[dict[str, Any]] = []
        port_locations: list[dict[str, Any]] = []
        for mount in spec.nic_mounts:
            idx = int(mount["index"])
            module_name = f"nic_card_mount_{idx}"
            mount_frame = f"task_board/{module_name}/nic_card_mount_link"
            mount_tf = self._lookup_transform("world", mount_frame)
            mount_locations.append(
                {
                    "module_name": module_name,
                    "index": idx,
                    "configured": {
                        "translation": float(mount["translation"]),
                        "orientation_rpy_rad": {
                            "roll": float(mount["roll"]),
                            "pitch": float(mount["pitch"]),
                            "yaw": float(mount["yaw"]),
                        },
                    },
                    "world_pose_estimate": self._transform_to_dict(mount_tf),
                    "is_task_target_module": module_name == spec.target_module_name,
                }
            )

            for port_name in ("sfp_port_0", "sfp_port_1"):
                port_frame = f"task_board/{module_name}/{port_name}_link"
                port_tf = self._lookup_transform("world", port_frame)
                port_locations.append(
                    {
                        "module_name": module_name,
                        "port_name": port_name,
                        "frame": port_frame,
                        "world_pose_estimate": self._transform_to_dict(port_tf),
                        "is_task_target_port": (
                            module_name == spec.target_module_name
                            and port_name == spec.port_name
                        ),
                    }
                )

        return {
            "task_board": {
                "pose": spec.task_board_pose,
                "nic_mount_count": len(spec.nic_mounts),
                "nic_mounts": mount_locations,
                "ports": port_locations,
                "sfp_mount_rail_0_translation": spec.sfp_rail_0_translation,
                "sfp_mount_rail_1_translation": spec.sfp_rail_1_translation,
            },
            "cable": {
                "name": "cable_0",
                "type": spec.cable_type,
                "pose": cable_pose,
                "attach_cable_to_gripper": True,
            },
        }

    def _make_task(self, spec: EpisodeSpec) -> InsertCable.Goal().task.__class__:
        task = InsertCable.Goal().task
        task.id = f"episode_{spec.episode_index + 1:06d}"
        task.cable_type = "sfp_sc"
        task.cable_name = "cable_0"
        task.plug_type = "sfp"
        task.plug_name = "sfp_tip"
        task.port_type = "sfp"
        task.port_name = spec.port_name
        task.target_module_name = spec.target_module_name
        task.time_limit = int(self.task_time_limit_s)
        return task

    @staticmethod
    def _transform_to_dict(transform) -> dict[str, Any] | None:
        if transform is None:
            return None
        return {
            "translation": {
                "x": float(transform.translation.x),
                "y": float(transform.translation.y),
                "z": float(transform.translation.z),
            },
            "rotation": {
                "x": float(transform.rotation.x),
                "y": float(transform.rotation.y),
                "z": float(transform.rotation.z),
                "w": float(transform.rotation.w),
            },
        }

    # ------------------------ Scoring estimate ------------------------

    def _estimate_score(
        self,
        *,
        task,
        duration_s: float,
        path_length_m: float,
        vel_samples: list[dict[str, float]],
        insertion_events: list[dict[str, Any]],
        start_dist_m: float | None,
        end_dist_m: float | None,
        contact_events: int,
    ) -> dict[str, Any]:
        tier1 = 1.0
        target_ns = f"task_board/{task.target_module_name}/{task.port_name}"
        events = [str(e["event"]) for e in insertion_events]

        correct_insert = any(target_ns in ev for ev in events)
        wrong_insert = (not correct_insert) and any(len(ev) > 0 for ev in events)

        if correct_insert:
            tier3 = 75.0
            tier3_reason = "correct insertion event"
        elif wrong_insert:
            tier3 = -12.0
            tier3_reason = "wrong-port insertion event"
        else:
            if start_dist_m is not None and end_dist_m is not None and start_dist_m > 1e-6:
                max_dist = 0.5 * start_dist_m
                prox = (max_dist - end_dist_m) / max(max_dist, 1e-6)
                tier3 = float(np.clip(prox * 25.0, 0.0, 25.0))
                tier3_reason = "proximity estimate"
            else:
                tier3 = 0.0
                tier3_reason = "missing distance estimate"

        # Tier 2 gates on Tier 3 > 0 in official scoring.
        if tier3 > 0.0:
            duration_score = float(
                np.clip((60.0 - duration_s) / (60.0 - 5.0), 0.0, 1.0) * 12.0
            )

            if start_dist_m is None:
                min_path = 0.0
            else:
                min_path = float(start_dist_m)
            max_path = 1.0 + min_path
            eff_score = float(
                np.clip((max_path - path_length_m) / max(max_path - min_path, 1e-6), 0.0, 1.0)
                * 6.0
            )

            jerk_avg = self._estimate_jerk_magnitude(vel_samples)
            smooth_score = float(np.clip((50.0 - jerk_avg) / 50.0, 0.0, 1.0) * 6.0)
        else:
            duration_score = 0.0
            eff_score = 0.0
            smooth_score = 0.0
            jerk_avg = 0.0

        contact_penalty = -24.0 if contact_events > 0 else 0.0
        force_penalty = 0.0

        tier2_total = smooth_score + duration_score + eff_score + contact_penalty + force_penalty
        total = tier1 + tier2_total + tier3

        return {
            "score_type": "local_estimate",
            "notes": (
                "Local training estimate from recorded streams. "
                "Not guaranteed to exactly match organizer-side scoring."
            ),
            "tier1": tier1,
            "tier2": {
                "smoothness": smooth_score,
                "duration": duration_score,
                "efficiency": eff_score,
                "force_penalty": force_penalty,
                "off_limit_contact_penalty": contact_penalty,
                "jerk_avg_mps3": jerk_avg,
            },
            "tier3": {
                "score": tier3,
                "reason": tier3_reason,
                "start_plug_port_distance_m": start_dist_m,
                "end_plug_port_distance_m": end_dist_m,
                "insertion_events": events,
            },
            "total_score_estimate": total,
            "raw_scoring_signals": {
                "contact_events": contact_events,
                "insertion_event_count": len(events),
            },
        }

    def _estimate_jerk_magnitude(self, vel_samples: list[dict[str, float]]) -> float:
        if len(vel_samples) < 3:
            return 0.0

        accelerations: list[tuple[float, float, float, float]] = []
        for i in range(1, len(vel_samples)):
            s0 = vel_samples[i - 1]
            s1 = vel_samples[i]
            dt = s1["t"] - s0["t"]
            if dt <= 1e-6:
                continue
            ax = (s1["vx"] - s0["vx"]) / dt
            ay = (s1["vy"] - s0["vy"]) / dt
            az = (s1["vz"] - s0["vz"]) / dt
            accelerations.append((s1["t"], ax, ay, az))

        if len(accelerations) < 2:
            return 0.0

        jerk_sum = 0.0
        jerk_count = 0
        for i in range(1, len(accelerations)):
            t0, ax0, ay0, az0 = accelerations[i - 1]
            t1, ax1, ay1, az1 = accelerations[i]
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            jx = (ax1 - ax0) / dt
            jy = (ay1 - ay0) / dt
            jz = (az1 - az0) / dt
            jerk_mag = math.sqrt(jx * jx + jy * jy + jz * jz)
            jerk_sum += jerk_mag
            jerk_count += 1

        if jerk_count == 0:
            return 0.0
        return jerk_sum / float(jerk_count)

    # ------------------------ ROS helpers ------------------------

    def _wait_for_readiness(self) -> None:
        required = [
            (self.expand_xacro_client, "/expand_xacro"),
            (self.spawn_entity_client, "/gz_server/spawn_entity"),
            (self.delete_entity_client, "/gz_server/delete_entity"),
            (self.tare_client, "/aic_controller/tare_force_torque_sensor"),
            (self.reset_joints_client, "/scoring/reset_joints"),
            (self.get_state_client, f"/{self.model_node_name}/get_state"),
            (self.change_state_client, f"/{self.model_node_name}/change_state"),
        ]
        if self.use_frame_sink:
            required.extend(
                [
                    (self.start_capture_client, f"{self.frame_sink_service_ns}/start_episode_capture"),
                    (self.stop_capture_client, f"{self.frame_sink_service_ns}/stop_episode_capture"),
                    (self.capture_status_client, f"{self.frame_sink_service_ns}/capture_status"),
                ]
            )
        if self.switch_controller_client is not None:
            required.append(
                (self.switch_controller_client, self.switch_controller_service)
            )
        for client, name in required:
            self.get_logger().info(f"Waiting for service {name}...")
            if not client.wait_for_service(timeout_sec=30.0):
                raise RuntimeError(f"Required service unavailable: {name}")

        self.get_logger().info("Waiting for /insert_cable action server...")
        if not self.insert_cable_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("/insert_cable action server unavailable")

    def _ensure_model_active(self) -> None:
        state_id = self._get_model_state_id()

        if state_id == int(State.PRIMARY_STATE_UNCONFIGURED):
            self._transition_model_state(
                transition_id=int(Transition.TRANSITION_CONFIGURE),
                transition_name="configure",
            )
            state_id = self._get_model_state_id()

        if state_id == int(State.PRIMARY_STATE_INACTIVE):
            self._transition_model_state(
                transition_id=int(Transition.TRANSITION_ACTIVATE),
                transition_name="activate",
            )

    def _deactivate_model_if_active(self) -> None:
        state_id = self._get_model_state_id()
        if state_id == int(State.PRIMARY_STATE_ACTIVE):
            self._transition_model_state(
                transition_id=int(Transition.TRANSITION_DEACTIVATE),
                transition_name="deactivate",
            )

    def _get_model_state_id(self) -> int:
        last_err: Exception | None = None
        for attempt in range(1, self.lifecycle_transition_retries + 1):
            try:
                state_resp = self._call_service(
                    self.get_state_client,
                    GetState.Request(),
                    timeout_s=self.lifecycle_transition_timeout_s,
                )
                return int(state_resp.current_state.id)
            except Exception as exc:
                last_err = exc
                self.get_logger().warn(
                    f"GetState attempt {attempt}/{self.lifecycle_transition_retries} failed: {exc}"
                )
                time.sleep(0.5)
        raise RuntimeError(f"Failed to query model lifecycle state: {last_err}")

    def _transition_model_state(self, *, transition_id: int, transition_name: str) -> None:
        req = ChangeState.Request()
        req.transition.id = int(transition_id)

        last_err: Exception | None = None
        for attempt in range(1, self.lifecycle_transition_retries + 1):
            try:
                resp = self._call_service(
                    self.change_state_client,
                    req,
                    timeout_s=self.lifecycle_transition_timeout_s,
                )
            except Exception as exc:
                last_err = exc
                self.get_logger().warn(
                    f"Model {transition_name} attempt {attempt}/{self.lifecycle_transition_retries} "
                    f"timed out or failed: {exc}"
                )
                time.sleep(0.5)
                continue

            if resp.success:
                return

            last_err = RuntimeError(f"Model {transition_name} returned success=false")
            self.get_logger().warn(
                f"Model {transition_name} attempt {attempt}/{self.lifecycle_transition_retries} "
                "returned success=false"
            )
            time.sleep(0.5)

        raise RuntimeError(f"Failed to {transition_name} model lifecycle node: {last_err}")

    def _run_insert_cable(self, task) -> dict[str, Any]:
        goal = InsertCable.Goal()
        goal.task = task

        feedback_msgs: list[str] = []

        def _feedback_cb(feedback_msg) -> None:
            if feedback_msg and feedback_msg.feedback:
                feedback_msgs.append(str(feedback_msg.feedback.message))

        send_future = self.insert_cable_client.send_goal_async(goal, feedback_callback=_feedback_cb)
        self._spin_until_future(send_future, timeout_s=20.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {
                "status": "goal_rejected",
                "success": False,
                "message": "insert_cable goal rejected",
                "feedback": feedback_msgs,
            }

        result_future = goal_handle.get_result_async()
        self._spin_until_future(result_future, timeout_s=float(self.task_time_limit_s + 30))
        result = result_future.result()
        if result is None:
            return {
                "status": "timeout",
                "success": False,
                "message": "insert_cable result timeout",
                "feedback": feedback_msgs,
            }

        return {
            "status": int(result.status),
            "success": bool(result.result.success),
            "message": str(result.result.message),
            "feedback": feedback_msgs,
        }

    def _start_episode_capture(self, episode_id: str, episode_dir: Path) -> None:
        req = StartEpisodeCapture.Request()
        req.episode_id = str(episode_id)
        req.run_dir = str(self.run_dir)
        req.episode_dir = str(episode_dir)
        req.task_json_path = str(episode_dir / "task.json")
        req.scene_json_path = str(episode_dir / "scene.json")
        req.enable_cameras = list(self.frame_sink_enable_cameras)
        resp = self._call_service(self.start_capture_client, req, timeout_s=20.0)
        if not resp.success:
            raise RuntimeError(f"StartEpisodeCapture failed: {resp.message}")

    def _stop_episode_capture(self, episode_id: str) -> int:
        req = StopEpisodeCapture.Request()
        req.episode_id = str(episode_id)
        req.flush_timeout_s = float(self.frame_sink_flush_timeout_s)
        req.require_webp_done = bool(self.frame_sink_require_webp_done)
        resp = self._call_service(
            self.stop_capture_client,
            req,
            timeout_s=max(20.0, self.frame_sink_flush_timeout_s + 10.0),
        )
        if not resp.success:
            self.get_logger().warn(
                "StopEpisodeCapture incomplete: "
                f"message={resp.message} pending_write={resp.pending_write} "
                f"pending_convert={resp.pending_convert}"
            )
        return int(resp.written_frames)

    def _call_service(self, client, request, timeout_s: float = 10.0):
        future = client.call_async(request)
        self._spin_until_future(future, timeout_s=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError(f"Service call failed: {client.srv_name}")
        return response

    def _spin_until_future(self, future, timeout_s: float) -> None:
        start = time.time()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout_s:
                raise RuntimeError("Timeout waiting for future completion")

    def _capture_window(self, duration_s: float, label: str) -> None:
        if duration_s <= 0.0:
            return
        start_count = 0
        if self.current_episode is not None:
            start_count = int(self.current_episode.get("frame_count", 0))
        end_t = time.time() + duration_s
        while rclpy.ok() and time.time() < end_t:
            rclpy.spin_once(self, timeout_sec=0.05)
        end_count = start_count
        if self.current_episode is not None:
            end_count = int(self.current_episode.get("frame_count", 0))
        self.get_logger().info(
            f"Capture window [{label}] {duration_s:.2f}s frames_added={end_count - start_count}"
        )

    def _expand_xacro(
        self, package_name: str, relative_path: str, xacro_arguments: list[str]
    ) -> str:
        req = ExpandXacro.Request()
        req.package_name = package_name
        req.relative_path = relative_path
        req.xacro_arguments = list(xacro_arguments)
        resp = self._call_service(self.expand_xacro_client, req, timeout_s=15.0)
        if not resp.success:
            raise RuntimeError(f"expand_xacro failed: {resp.message}")
        return str(resp.xml)

    def _spawn_entity(self, *, name: str, xml: str, pose: dict[str, float]) -> None:
        req = SpawnEntity.Request()
        req.name = name
        req.allow_renaming = True
        if hasattr(req, "uri"):
            req.uri = ""
        if hasattr(req, "resource_string"):
            req.resource_string = xml
        elif hasattr(req, "entity_resource"):
            req.entity_resource.uri = ""
            req.entity_resource.resource_string = xml
        else:
            raise RuntimeError("SpawnEntity request has unsupported resource fields")

        if hasattr(req, "entity_namespace"):
            req.entity_namespace = ""

        qx, qy, qz, qw = self._rpy_to_quat(
            pose["roll"], pose["pitch"], pose["yaw"]
        )
        req.initial_pose = PoseStamped()
        req.initial_pose.header.frame_id = "world"
        req.initial_pose.pose.position.x = float(pose["x"])
        req.initial_pose.pose.position.y = float(pose["y"])
        req.initial_pose.pose.position.z = float(pose["z"])
        req.initial_pose.pose.orientation.x = float(qx)
        req.initial_pose.pose.orientation.y = float(qy)
        req.initial_pose.pose.orientation.z = float(qz)
        req.initial_pose.pose.orientation.w = float(qw)

        resp = self._call_service(self.spawn_entity_client, req, timeout_s=15.0)

        if hasattr(resp, "result") and hasattr(resp.result, "result"):
            # simulation_interfaces/msg/Result::RESULT_OK == 1
            if int(resp.result.result) != 1:
                err = getattr(resp.result, "error_message", "unknown")
                raise RuntimeError(f"SpawnEntity failed for {name}: {err}")

    def _delete_entity(self, name: str) -> None:
        req = DeleteEntity.Request()
        req.entity = name
        try:
            resp = self._call_service(self.delete_entity_client, req, timeout_s=10.0)
        except Exception as exc:
            self.get_logger().warn(f"Delete entity call failed for {name}: {exc}")
            return

        if hasattr(resp, "result") and hasattr(resp.result, "result"):
            # ignore not-found failures to keep loop simple
            if int(resp.result.result) != 1:
                err = getattr(resp.result, "error_message", "unknown")
                self.get_logger().info(f"Delete entity non-OK for {name}: {err}")

    def _tare_ft_sensor(self) -> None:
        req = Trigger.Request()
        resp = self._call_service(self.tare_client, req, timeout_s=10.0)
        if not resp.success:
            raise RuntimeError(f"FT tare failed: {resp.message}")

    def _reset_joints_to_home(self) -> None:
        if self.switch_controller_client is not None:
            self._switch_controller(
                activate=[],
                deactivate=[self.aic_controller_name],
            )
        try:
            req = ResetJoints.Request()
            req.joint_names = list(self.home_joint_names)
            req.initial_positions = list(self.home_joint_positions)
            resp = self._call_service(self.reset_joints_client, req, timeout_s=10.0)
            if not resp.success:
                raise RuntimeError(f"ResetJoints failed: {resp.message}")
        finally:
            if self.switch_controller_client is not None:
                self._switch_controller(
                    activate=[self.aic_controller_name],
                    deactivate=[],
                )

    def _switch_controller(self, *, activate: list[str], deactivate: list[str]) -> None:
        if self.switch_controller_client is None or SwitchController is None:
            raise RuntimeError("SwitchController client unavailable")
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        if hasattr(req, "strictness"):
            req.strictness = int(getattr(SwitchController.Request, "BEST_EFFORT", 1))
        if hasattr(req, "activate_asap"):
            req.activate_asap = True
        resp = self._call_service(self.switch_controller_client, req, timeout_s=10.0)
        if not getattr(resp, "ok", False):
            raise RuntimeError(
                "SwitchController failed "
                f"(activate={activate}, deactivate={deactivate})"
            )

    # ------------------------ Geometry ------------------------

    def _lookup_transform(self, target_frame: str, source_frame: str):
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return tf.transform
        except TransformException as exc:
            self.get_logger().warning(
                f"TF lookup failed {source_frame} -> {target_frame}: {exc}"
            )
            return None

    def _lookup_plug_port_distance(self, task) -> float | None:
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"

        plug_tf = self._lookup_transform("base_link", plug_frame)
        port_tf = self._lookup_transform("base_link", port_frame)
        if plug_tf is None or port_tf is None:
            return None

        dx = float(plug_tf.translation.x - port_tf.translation.x)
        dy = float(plug_tf.translation.y - port_tf.translation.y)
        dz = float(plug_tf.translation.z - port_tf.translation.z)
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    @staticmethod
    def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    # ------------------------ Serialization ------------------------

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _pose_to_dict(pose) -> dict[str, Any]:
        return {
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }

    @staticmethod
    def _task_to_dict(task) -> dict[str, Any]:
        return {
            "id": task.id,
            "cable_type": task.cable_type,
            "cable_name": task.cable_name,
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "time_limit": int(task.time_limit),
        }

    def _write_image(self, episode_dir: Path, camera: str, frame_idx: int, image_msg) -> dict[str, Any]:
        rel = Path("images") / camera / f"{frame_idx:06d}.bin"
        out_path = episode_dir / rel
        with open(out_path, "wb") as f:
            f.write(bytes(image_msg.data))

        return {
            "path": str(rel),
            "width": int(image_msg.width),
            "height": int(image_msg.height),
            "encoding": str(image_msg.encoding),
            "step": int(image_msg.step),
            "is_bigendian": int(image_msg.is_bigendian),
            "stamp": self._stamp_to_sec(image_msg.header.stamp),
            "frame_id": str(image_msg.header.frame_id),
        }

    @staticmethod
    def _image_stats(image_msg) -> dict[str, float]:
        arr = np.frombuffer(image_msg.data, dtype=np.uint8)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "p01": 0.0, "p99": 0.0}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p01": float(np.percentile(arr, 1)),
            "p99": float(np.percentile(arr, 99)),
        }

    # ------------------------ Async post-processing ------------------------

    @staticmethod
    def _decode_image_from_meta(raw: bytes, meta: dict[str, Any]) -> Image.Image:
        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        step = int(meta.get("step", 0))
        encoding = str(meta.get("encoding", "")).lower()
        arr = np.frombuffer(raw, dtype=np.uint8)
        if encoding == "mono8":
            rows = arr.reshape(height, step)[:, :width]
            return Image.fromarray(rows, mode="L")
        if encoding in {"rgb8", "bgr8"}:
            rows = arr.reshape(height, step)[:, : width * 3]
            rgb = rows.reshape(height, width, 3)
            if encoding == "bgr8":
                rgb = rgb[..., ::-1]
            return Image.fromarray(rgb, mode="RGB")
        if encoding in {"rgba8", "bgra8"}:
            rows = arr.reshape(height, step)[:, : width * 4]
            rgba = rows.reshape(height, width, 4)
            if encoding == "bgra8":
                rgba = rgba[..., [2, 1, 0, 3]]
            return Image.fromarray(rgba, mode="RGBA")
        raise RuntimeError(f"Unsupported encoding: {encoding}")

    @classmethod
    def _postprocess_episode_images(
        cls,
        episode_dir: Path,
        output_subdir: str,
        delete_bin_after_webp: bool,
    ) -> dict[str, int]:
        frames_path = episode_dir / "frames.jsonl"
        if not frames_path.is_file():
            return {"converted": 0, "deleted": 0, "missing_bin": 0, "unsupported": 0}

        converted = 0
        deleted = 0
        missing_bin = 0
        unsupported = 0
        with open(frames_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                frame_idx = int(frame.get("frame_idx", 0))
                images = frame.get("images", {})
                for camera in ("left", "center", "right"):
                    meta = images.get(camera)
                    if not isinstance(meta, dict):
                        continue
                    rel = Path(str(meta.get("path", "")))
                    bin_path = episode_dir / rel
                    if not bin_path.is_file():
                        missing_bin += 1
                        continue
                    out_dir = episode_dir / output_subdir / camera
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{frame_idx:06d}.webp"
                    raw = bin_path.read_bytes()
                    try:
                        image = cls._decode_image_from_meta(raw, meta)
                    except Exception:
                        unsupported += 1
                        continue
                    image.save(out_path, format="WEBP", lossless=True)
                    converted += 1
                    if delete_bin_after_webp:
                        try:
                            bin_path.unlink()
                            deleted += 1
                        except FileNotFoundError:
                            pass
        return {
            "converted": converted,
            "deleted": deleted,
            "missing_bin": missing_bin,
            "unsupported": unsupported,
        }

    def _schedule_episode_postprocess(self, episode_dir: Path) -> None:
        if self.postprocess_executor is None:
            return
        future = self.postprocess_executor.submit(
            self._postprocess_episode_images,
            episode_dir,
            self.postprocess_output_subdir,
            self.postprocess_delete_bin_after_webp,
        )
        self.postprocess_futures.append(future)
        self.get_logger().info(f"Scheduled background postprocess for {episode_dir.name}")

    def _wait_for_postprocess_jobs(self) -> None:
        if self.postprocess_executor is None:
            return
        total = len(self.postprocess_futures)
        if total == 0:
            self.postprocess_executor.shutdown(wait=True)
            self.postprocess_executor = None
            return
        self.get_logger().info(f"Waiting for {total} background postprocess job(s)...")
        converted = 0
        deleted = 0
        missing = 0
        unsupported = 0
        for future in self.postprocess_futures:
            try:
                result = future.result()
            except Exception as exc:
                self.get_logger().error(f"Background postprocess failed: {exc}")
                continue
            converted += int(result.get("converted", 0))
            deleted += int(result.get("deleted", 0))
            missing += int(result.get("missing_bin", 0))
            unsupported += int(result.get("unsupported", 0))
        self.postprocess_executor.shutdown(wait=True)
        self.postprocess_executor = None
        self.postprocess_futures.clear()
        self.get_logger().info(
            "Background postprocess complete: "
            f"converted={converted}, deleted_bin={deleted}, missing_bin={missing}, "
            f"unsupported={unsupported}"
        )

    def _write_run_manifest(self) -> None:
        commit = os.environ.get("AIC_GIT_COMMIT", "unknown")
        manifest = {
            "generator": "proximity_data_generator",
            "version": "v2",
            "frame_schema_version": "visual_proximity_v1",
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "task_time_limit_s": self.task_time_limit_s,
            "use_frame_sink": self.use_frame_sink,
            "frame_sink_service_ns": self.frame_sink_service_ns,
            "frame_sink_require_webp_done": self.frame_sink_require_webp_done,
            "frame_sink_flush_timeout_s": self.frame_sink_flush_timeout_s,
            "postprocess_webp_after_episode": self.postprocess_webp_after_episode,
            "postprocess_delete_bin_after_webp": self.postprocess_delete_bin_after_webp,
            "postprocess_workers": self.postprocess_workers,
            "postprocess_output_subdir": self.postprocess_output_subdir,
            "output_dir": str(self.run_dir),
            "git_commit": commit,
            "notes": "Baseline single-worker ProximityTeacher data generation run",
        }
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def main() -> None:
    rclpy.init()
    node = ProximityDataGenerator()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
