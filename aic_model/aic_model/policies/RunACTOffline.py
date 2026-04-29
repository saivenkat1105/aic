#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import os
from pathlib import Path
from typing import Dict

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench
from rclpy.node import Node


class RunACTOffline(Policy):
    REQUIRED_POLICY_FILES = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    )

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        import cv2
        import draccus
        import numpy as np
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from safetensors.torch import load_file

        self._cv2 = cv2
        self._np = np
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_path = self._get_policy_path()

        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            config_dict.pop("type", None)

        config = draccus.decode(ACTConfig, config_dict)

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(
            f"ACT Offline Policy loaded on {self.device} from {policy_path}"
        )

        stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }

        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))
        self.image_scaling = 0.25

        self.get_logger().info("Normalization statistics loaded successfully.")

    @classmethod
    def _get_policy_path(cls) -> Path:
        policy_path_env = os.environ.get("AIC_ACT_POLICY_PATH")
        if not policy_path_env:
            raise RuntimeError(
                "AIC_ACT_POLICY_PATH must point to a local ACT policy snapshot. "
                "Download it once with: pixi run hf download grkw/aic_act_policy "
                "--include config.json --include model.safetensors "
                '--include "*.safetensors" --local-dir /home/user/aic/.models/aic_act_policy'
            )

        policy_path = Path(policy_path_env).expanduser().resolve()
        if not policy_path.is_dir():
            raise FileNotFoundError(
                f"AIC_ACT_POLICY_PATH does not exist or is not a directory: {policy_path}"
            )

        missing_files = [
            file_name
            for file_name in cls.REQUIRED_POLICY_FILES
            if not (policy_path / file_name).is_file()
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"AIC_ACT_POLICY_PATH is missing required file(s): {missing}. "
                f"Checked directory: {policy_path}"
            )

        return policy_path

    def _img_to_tensor(self, raw_img, mean, std):
        img_np = self._np.frombuffer(raw_img.data, dtype=self._np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        if self.image_scaling != 1.0:
            img_np = self._cv2.resize(
                img_np,
                None,
                fx=self.image_scaling,
                fy=self.image_scaling,
                interpolation=self._cv2.INTER_AREA,
            )

        tensor = (
            self._torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, object]:
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        state_np = self._np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs_msg.controller_state.tcp_error,
                *obs_msg.joint_states.position[:7],
            ],
            dtype=self._np.float32,
        )

        raw_state_tensor = (
            self._torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACTOffline.insert_cable() enter. Task: {task}")

        start_time = self.time_now()

        while (self.time_now() - start_time).nanoseconds / 1e9 < 30.0:
            loop_start = self.time_now()
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                self.sleep_for(0.05)
                continue

            with self._torch.inference_mode():
                normalized_action = self.policy.select_action(
                    self.prepare_observations(observation_msg)
                )

            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Action: {action}")

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            move_robot(motion_update=self.set_cartesian_twist_target(twist))
            send_feedback("in progress...")

            elapsed = (self.time_now() - loop_start).nanoseconds / 1e9
            self.sleep_for(max(0.0, 0.25 - elapsed))

        self.get_logger().info("RunACTOffline.insert_cable() exiting...")
        return True

    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = self._np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = self._np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return motion_update_msg
