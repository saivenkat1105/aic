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

"""Training-only no-contact proximity teacher.

This policy intentionally uses ground-truth TF and is not legal for evaluation.
It is meant to collect approach demonstrations and perception labels near the
target port while stopping before insertion/contact.
"""

import math
import os

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class ProximityTeacher(Policy):
    """Move the grasped plug tip to a safe standoff using training-only TF.

    The teacher follows the existing CheatCode alignment pattern, but it does
    not run the insertion descent. It uses the target port and plug-tip TFs only
    for data generation with `ground_truth:=true`.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task = None
        self.standoff_m = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_STANDOFF_M", 0.018
        )
        self.approach_duration_s = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_APPROACH_S", 5.0
        )
        self.hold_duration_s = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_HOLD_S", 2.0
        )
        self.waypoint_hold_s = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_WAYPOINT_HOLD_S", 0.5
        )
        self.lateral_offset_m = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_LATERAL_OFFSET_M", 0.012
        )
        self.standoff_delta_m = self._get_float_env(
            "AIC_PROXIMITY_TEACHER_STANDOFF_DELTA_M", 0.008
        )
        self.rate_hz = self._get_float_env("AIC_PROXIMITY_TEACHER_RATE_HZ", 20.0)
        self.variant = os.environ.get(
            "AIC_PROXIMITY_TEACHER_VARIANT", "nominal"
        ).strip()

        if self.standoff_m <= 0.0:
            raise ValueError("AIC_PROXIMITY_TEACHER_STANDOFF_M must be positive")
        if self.rate_hz <= 0.0:
            raise ValueError("AIC_PROXIMITY_TEACHER_RATE_HZ must be positive")

        self.get_logger().warn(
            "ProximityTeacher uses ground-truth TF and is training-only. "
            "Do not submit this policy for evaluation."
        )
        self.get_logger().info(
            "ProximityTeacher config: "
            f"variant={self.variant}, standoff={self.standoff_m:.4f} m, "
            f"approach={self.approach_duration_s:.2f} s, "
            f"hold={self.hold_duration_s:.2f} s, rate={self.rate_hz:.1f} Hz"
        )

    @staticmethod
    def _get_float_env(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return float(value)

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> "
                        f"'{target_frame}'... run with ground_truth:=true."
                    )
                attempt += 1
                self.sleep_for(0.1)

        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        ).transform

    @staticmethod
    def _translation(transform: Transform) -> np.ndarray:
        return np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quaternion_wxyz(transform: Transform) -> tuple[float, float, float, float]:
        return (
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
        )

    @staticmethod
    def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9 or not math.isfinite(norm):
            return fallback.astype(np.float64)
        return vec / norm

    def _lateral_basis(self, approach_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        lateral_a = np.cross(approach_axis, world_x)
        if np.linalg.norm(lateral_a) < 1e-6:
            lateral_a = np.cross(approach_axis, world_y)
        lateral_a = self._normalize(lateral_a, world_y)
        lateral_b = self._normalize(np.cross(approach_axis, lateral_a), world_x)
        return lateral_a, lateral_b

    def _target_for_variant(
        self,
        entrance: np.ndarray,
        approach_axis: np.ndarray,
        variant: str,
    ) -> np.ndarray:
        lateral_a, lateral_b = self._lateral_basis(approach_axis)
        standoff = self.standoff_m
        lateral = np.zeros(3, dtype=np.float64)

        if variant in ("offset_left", "left"):
            lateral = self.lateral_offset_m * lateral_a
        elif variant in ("offset_right", "right"):
            lateral = -self.lateral_offset_m * lateral_a
        elif variant in ("offset_up", "up"):
            lateral = self.lateral_offset_m * lateral_b
        elif variant in ("offset_down", "down"):
            lateral = -self.lateral_offset_m * lateral_b
        elif variant in ("high_standoff", "high"):
            standoff += self.standoff_delta_m
        elif variant in ("low_standoff", "low"):
            standoff = max(0.012, standoff - self.standoff_delta_m)
        elif variant in ("nominal", ""):
            pass
        else:
            self.get_logger().warn(
                f"Unknown ProximityTeacher variant '{variant}', using nominal target."
            )

        return entrance + approach_axis * standoff + lateral

    def _calc_gripper_pose_for_tip_target(
        self,
        port_transform: Transform,
        target_tip: np.ndarray,
        slerp_fraction: float,
        position_fraction: float,
    ) -> Pose:
        """Return a TCP pose that moves the ground-truth plug tip to target_tip."""
        plug_tf = self._lookup_transform(
            "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link"
        )
        gripper_tf = self._lookup_transform("base_link", "gripper/tcp")

        q_port = self._quaternion_wxyz(port_transform)
        q_plug = self._quaternion_wxyz(plug_tf)
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        q_gripper = self._quaternion_wxyz(gripper_tf)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = self._translation(gripper_tf)
        plug_xyz = self._translation(plug_tf)
        plug_tip_gripper_offset = gripper_xyz - plug_xyz
        target_gripper_xyz = target_tip + plug_tip_gripper_offset

        blend_xyz = (
            position_fraction * target_gripper_xyz
            + (1.0 - position_fraction) * gripper_xyz
        )

        return Pose(
            position=Point(
                x=float(blend_xyz[0]),
                y=float(blend_xyz[1]),
                z=float(blend_xyz[2]),
            ),
            orientation=Quaternion(
                w=float(q_gripper_slerp[0]),
                x=float(q_gripper_slerp[1]),
                y=float(q_gripper_slerp[2]),
                z=float(q_gripper_slerp[3]),
            ),
        )

    def _move_to_tip_target(
        self,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        target_tip: np.ndarray,
        duration_s: float,
        label: str,
    ) -> bool:
        dt = 1.0 / self.rate_hz
        steps = max(1, int(duration_s * self.rate_hz))

        for step in range(steps):
            fraction = (step + 1) / steps
            try:
                pose = self._calc_gripper_pose_for_tip_target(
                    port_transform=port_transform,
                    target_tip=target_tip,
                    slerp_fraction=fraction,
                    position_fraction=fraction,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(
                    f"TF lookup failed while moving to {label}: {ex}"
                )
                return False
            self.sleep_for(dt)

        return True

    def _hold_tip_target(
        self,
        move_robot: MoveRobotCallback,
        port_transform: Transform,
        target_tip: np.ndarray,
        duration_s: float,
        label: str,
    ) -> bool:
        dt = 1.0 / self.rate_hz
        steps = max(1, int(duration_s * self.rate_hz))

        for _ in range(steps):
            try:
                pose = self._calc_gripper_pose_for_tip_target(
                    port_transform=port_transform,
                    target_tip=target_tip,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed while holding {label}: {ex}")
                return False
            self.sleep_for(dt)

        return True

    def _trajectory_targets(
        self,
        entrance: np.ndarray,
        approach_axis: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float, float]]:
        final_target = self._target_for_variant(entrance, approach_axis, "nominal")

        if self.variant in ("nominal", ""):
            return [
                ("nominal", final_target, self.approach_duration_s, self.hold_duration_s)
            ]

        offset_target = self._target_for_variant(entrance, approach_axis, self.variant)
        return [
            (
                self.variant,
                offset_target,
                self.approach_duration_s,
                self.waypoint_hold_s,
            ),
            (
                "recovery_to_nominal",
                final_target,
                max(1.0, self.approach_duration_s * 0.5),
                self.hold_duration_s,
            ),
        ]

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"ProximityTeacher.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        )
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, entrance_frame, cable_tip_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_transform = self._lookup_transform("base_link", port_frame)
            entrance_transform = self._lookup_transform("base_link", entrance_frame)
        except TransformException as ex:
            self.get_logger().error(f"Could not look up proximity target TFs: {ex}")
            return False

        port_xyz = self._translation(port_transform)
        entrance_xyz = self._translation(entrance_transform)
        approach_axis = self._normalize(
            entrance_xyz - port_xyz, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        self.get_logger().info(
            "ProximityTeacher target: "
            f"port_frame={port_frame}, entrance_frame={entrance_frame}, "
            f"axis=({approach_axis[0]:.4f}, {approach_axis[1]:.4f}, "
            f"{approach_axis[2]:.4f})"
        )

        for label, target_tip, move_duration, hold_duration in self._trajectory_targets(
            entrance_xyz, approach_axis
        ):
            send_feedback(
                f"teacher_mode=proximity trajectory_variant={label} "
                f"standoff_distance_m={self.standoff_m:.4f}"
            )
            self.get_logger().info(
                f"Moving to {label} tip target: "
                f"({target_tip[0]:.4f}, {target_tip[1]:.4f}, {target_tip[2]:.4f})"
            )
            if not self._move_to_tip_target(
                move_robot=move_robot,
                port_transform=port_transform,
                target_tip=target_tip,
                duration_s=move_duration,
                label=label,
            ):
                return False
            if not self._hold_tip_target(
                move_robot=move_robot,
                port_transform=port_transform,
                target_tip=target_tip,
                duration_s=hold_duration,
                label=label,
            ):
                return False

        self.get_logger().info("ProximityTeacher held no-contact standoff target.")
        send_feedback("teacher_mode=proximity termination_reason=standoff_complete")
        return True
