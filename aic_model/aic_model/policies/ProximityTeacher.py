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

Component 1 implementation:
- derived from CheatCode alignment behavior
- uses training-only ground-truth TF
- stops at a 20 mm standoff from the port entrance
- never executes insertion descent
"""

import math
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class ProximityTeacher(Policy):
    """Training-only teacher that reaches no-contact standoff from target port."""

    TF_WAIT_TIMEOUT_S = 10.0
    APPROACH_DURATION_S = 5.0
    APPROACH_STEPS = 100
    STANDOFF_M = 0.020
    HOLD_DURATION_S = 5.0
    MAX_INTEGRATOR_WINDUP = 0.05
    INTEGRATOR_GAIN = 0.15

    def __init__(self, parent_node):
        self._task = None
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        super().__init__(parent_node)
        self.get_logger().warn(
            "ProximityTeacher uses ground-truth TF and is training-only. "
            "Do not submit this policy for evaluation."
        )

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = TF_WAIT_TIMEOUT_S
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
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... "
                        "run with ground_truth:=true."
                    )
                attempt += 1
                self.sleep_for(0.1)

        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _lookup_tf(self, target_frame: str, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        ).transform

    @staticmethod
    def _xyz(transform) -> tuple[float, float, float]:
        return (
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        )

    @staticmethod
    def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
        if norm < 1e-9:
            return (0.0, 0.0, 1.0)
        return (vec[0] / norm, vec[1] / norm, vec[2] / norm)

    def _calc_gripper_pose_to_tip_target(
        self,
        target_tip_xyz: tuple[float, float, float],
        port_transform,
        slerp_fraction: float,
        position_fraction: float,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )

        plug_tf = self._lookup_tf(
            "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link"
        )
        gripper_tf = self._lookup_tf("base_link", "gripper/tcp")

        q_plug = (
            plug_tf.rotation.w,
            plug_tf.rotation.x,
            plug_tf.rotation.y,
            plug_tf.rotation.z,
        )
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        q_gripper = (
            gripper_tf.rotation.w,
            gripper_tf.rotation.x,
            gripper_tf.rotation.y,
            gripper_tf.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        plug_xyz = self._xyz(plug_tf)
        gripper_xyz = self._xyz(gripper_tf)
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        target_tcp_xyz = (
            target_tip_xyz[0] + plug_tip_gripper_offset[0],
            target_tip_xyz[1] + plug_tip_gripper_offset[1],
            target_tip_xyz[2] + plug_tip_gripper_offset[2],
        )

        tip_x_error = target_tip_xyz[0] - plug_xyz[0]
        tip_y_error = target_tip_xyz[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(
                np.clip(
                    self._tip_x_error_integrator + tip_x_error,
                    -self.MAX_INTEGRATOR_WINDUP,
                    self.MAX_INTEGRATOR_WINDUP,
                )
            )
            self._tip_y_error_integrator = float(
                np.clip(
                    self._tip_y_error_integrator + tip_y_error,
                    -self.MAX_INTEGRATOR_WINDUP,
                    self.MAX_INTEGRATOR_WINDUP,
                )
            )

        target_tcp_xyz = (
            target_tcp_xyz[0] + self.INTEGRATOR_GAIN * self._tip_x_error_integrator,
            target_tcp_xyz[1] + self.INTEGRATOR_GAIN * self._tip_y_error_integrator,
            target_tcp_xyz[2],
        )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3f} "
            f"xy_error: {tip_x_error:.4f} {tip_y_error:.4f} "
            f"integrators: {self._tip_x_error_integrator:.4f}, {self._tip_y_error_integrator:.4f}"
        )

        blend_xyz = (
            position_fraction * target_tcp_xyz[0]
            + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_tcp_xyz[1]
            + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_tcp_xyz[2]
            + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"ProximityTeacher.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        )
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, entrance_frame, cable_tip_frame, "gripper/tcp"]:
            self.get_logger().info(f"TF precheck: base_link <- {frame}")
            if not self._wait_for_tf("base_link", frame):
                send_feedback(f"proximity_teacher_failed: missing_tf={frame}")
                return False

        try:
            port_tf = self._lookup_tf("base_link", port_frame)
            entrance_tf = self._lookup_tf("base_link", entrance_frame)
        except TransformException as ex:
            self.get_logger().error(f"Could not look up required transforms: {ex}")
            send_feedback(f"proximity_teacher_failed: tf_lookup_error={ex}")
            return False

        port_xyz = self._xyz(port_tf)
        entrance_xyz = self._xyz(entrance_tf)
        approach_axis = self._normalize(
            (
                entrance_xyz[0] - port_xyz[0],
                entrance_xyz[1] - port_xyz[1],
                entrance_xyz[2] - port_xyz[2],
            )
        )

        target_tip_xyz = (
            entrance_xyz[0] + self.STANDOFF_M * approach_axis[0],
            entrance_xyz[1] + self.STANDOFF_M * approach_axis[1],
            entrance_xyz[2] + self.STANDOFF_M * approach_axis[2],
        )

        send_feedback(
            "teacher_mode=proximity "
            f"standoff_distance_m={self.STANDOFF_M:.3f} "
            f"target_port_frame={port_frame}"
        )

        self.get_logger().info(
            "Target tip standoff and axis: "
            f"({target_tip_xyz[0]:.4f}, {target_tip_xyz[1]:.4f}, {target_tip_xyz[2]:.4f})"
            f" axis=({approach_axis[0]:.4f}, {approach_axis[1]:.4f}, {approach_axis[2]:.4f})"
        )

        for t in range(self.APPROACH_STEPS):
            interp_fraction = float(t + 1) / float(self.APPROACH_STEPS)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose_to_tip_target(
                        target_tip_xyz=target_tip_xyz,
                        port_transform=port_tf,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        reset_xy_integrator=(t == 0),
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during approach: {ex}")
                send_feedback(f"proximity_teacher_warn: approach_tf_lookup_failed={ex}")
            self.sleep_for(self.APPROACH_DURATION_S / float(self.APPROACH_STEPS))

        hold_steps = max(1, int(self.HOLD_DURATION_S / 0.05))
        for _ in range(hold_steps):
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose_to_tip_target(
                        target_tip_xyz=target_tip_xyz,
                        port_transform=port_tf,
                        slerp_fraction=1.0,
                        position_fraction=1.0,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during hold: {ex}")
                send_feedback(f"proximity_teacher_warn: hold_tf_lookup_failed={ex}")
            self.sleep_for(0.05)

        self.get_logger().info(
            "ProximityTeacher reached 20 mm standoff target without insertion descent."
        )
        return True
