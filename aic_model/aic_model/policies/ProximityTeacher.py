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

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
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
from trajectory_msgs.msg import JointTrajectoryPoint
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
        self.target_alpha = self._get_float_env("AIC_PROXIMITY_TARGET_ALPHA", 0.25)
        self.max_step_m = self._get_float_env("AIC_PROXIMITY_MAX_STEP_M", 0.0025)
        self.max_target_jump_m = self._get_float_env("AIC_PROXIMITY_MAX_TARGET_JUMP_M", 0.02)
        self.plug_drift_abort_m = self._get_float_env(
            "AIC_PROXIMITY_PLUG_DRIFT_ABORT_M", 0.018
        )
        self.plug_drift_hold_s = self._get_float_env("AIC_PROXIMITY_PLUG_DRIFT_HOLD_S", 0.25)
        self.gripper_open_abort_rad = self._get_float_env(
            "AIC_PROXIMITY_GRIPPER_OPEN_ABORT_RAD", 0.004
        )
        self.soft_force_n = self._get_float_env("AIC_PROXIMITY_FORCE_SOFT_N", 12.0)
        self.backoff_force_n = self._get_float_env("AIC_PROXIMITY_FORCE_BACKOFF_N", 18.0)
        self.hard_force_n = self._get_float_env("AIC_PROXIMITY_FORCE_HARD_N", 22.0)
        self.backoff_duration_s = self._get_float_env(
            "AIC_PROXIMITY_BACKOFF_DURATION_S", 0.4
        )
        self.backoff_distance_m = self._get_float_env(
            "AIC_PROXIMITY_BACKOFF_DISTANCE_M", 0.008
        )
        self.backoff_cooldown_s = self._get_float_env(
            "AIC_PROXIMITY_BACKOFF_COOLDOWN_S", 0.6
        )
        self.abort_to_timeout = os.environ.get(
            "AIC_PROXIMITY_ABORT_TO_TIMEOUT", "true"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.gripper_tighten_enabled = os.environ.get(
            "AIC_PROXIMITY_GRIPPER_TIGHTEN_ENABLED", "true"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.gripper_tighten_interval_s = self._get_float_env(
            "AIC_PROXIMITY_GRIPPER_TIGHTEN_INTERVAL_S", 1.5
        )
        self.gripper_tighten_delta = self._get_float_env(
            "AIC_PROXIMITY_GRIPPER_TIGHTEN_DELTA_RAD", 0.0006
        )
        self.variant = os.environ.get(
            "AIC_PROXIMITY_TEACHER_VARIANT", "nominal"
        ).strip()
        self._filtered_tip_target = None
        self._last_valid_target_tip = None
        self._baseline_plug_offset = None
        self._plug_drift_started_at = None
        self._baseline_gripper_joint = None
        self._backoff_cooldown_until = None
        self._last_gripper_tighten_time = None
        self._last_failure_reason = ""

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

    @staticmethod
    def _force_mag(observation) -> float:
        if observation is None:
            return 0.0
        f = observation.wrist_wrench.wrench.force
        return float(math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))

    def _reset_runtime_guards(self):
        self._filtered_tip_target = None
        self._last_valid_target_tip = None
        self._baseline_plug_offset = None
        self._plug_drift_started_at = None
        self._baseline_gripper_joint = None
        self._backoff_cooldown_until = None
        self._last_gripper_tighten_time = None
        self._last_failure_reason = ""

    def _maybe_tighten_gripper(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> None:
        if not self.gripper_tighten_enabled:
            return
        now = self.time_now()
        if (
            self._last_gripper_tighten_time is not None
            and now < self._last_gripper_tighten_time + Duration(seconds=self.gripper_tighten_interval_s)
        ):
            return
        obs = get_observation()
        if obs is None or len(obs.joint_states.position) < 6:
            return

        # aic_controller in this setup expects 6 arm joints for JointMotionUpdate.
        # Do not publish 7-joint vectors here; they get rejected.
        positions = list(obs.joint_states.position[:6])
        # Best-effort "tighten" pulse while staying 6-DOF: hold current arm position.
        # (No separate gripper joint command channel is exposed in this controller mode.)
        joint_cmd = JointMotionUpdate()
        joint_cmd.target_state = JointTrajectoryPoint(positions=positions)
        joint_cmd.target_stiffness = [120.0, 120.0, 120.0, 80.0, 80.0, 80.0]
        joint_cmd.target_damping = [25.0, 25.0, 25.0, 18.0, 18.0, 18.0]
        joint_cmd.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        joint_cmd.target_feedforward_torque = [0.0] * 6
        move_robot(joint_motion_update=joint_cmd)
        self._last_gripper_tighten_time = now

    def _linger_until_timeout(
        self,
        deadline: Time,
        move_robot: MoveRobotCallback,
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
        reason: str,
    ) -> bool:
        send_feedback(
            f"teacher_mode=proximity state=holding_until_timeout reason={reason}"
        )
        dt = max(0.05, 1.0 / self.rate_hz)
        while self.time_now() < deadline:
            self._command_current_hold(move_robot)
            self._maybe_tighten_gripper(get_observation, move_robot)
            self.sleep_for(dt)
        send_feedback("teacher_mode=proximity message=task_completed timeout_seconds=65")
        return True

    def _stabilize_tip_target(self, raw_tip: np.ndarray) -> np.ndarray:
        if self._last_valid_target_tip is None:
            self._last_valid_target_tip = raw_tip.copy()
        jump = float(np.linalg.norm(raw_tip - self._last_valid_target_tip))
        if jump <= self.max_target_jump_m:
            self._last_valid_target_tip = raw_tip.copy()
        target = self._last_valid_target_tip.copy()

        if self._filtered_tip_target is None:
            self._filtered_tip_target = target
            return self._filtered_tip_target

        alpha = float(np.clip(self.target_alpha, 0.01, 1.0))
        candidate = alpha * target + (1.0 - alpha) * self._filtered_tip_target
        delta = candidate - self._filtered_tip_target
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.max_step_m and delta_norm > 1e-9:
            delta = delta * (self.max_step_m / delta_norm)
        self._filtered_tip_target = self._filtered_tip_target + delta
        return self._filtered_tip_target

    def _command_current_hold(self, move_robot: MoveRobotCallback):
        try:
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
            pose = Pose(
                position=Point(
                    x=float(gripper_tf.translation.x),
                    y=float(gripper_tf.translation.y),
                    z=float(gripper_tf.translation.z),
                ),
                orientation=Quaternion(
                    w=float(gripper_tf.rotation.w),
                    x=float(gripper_tf.rotation.x),
                    y=float(gripper_tf.rotation.y),
                    z=float(gripper_tf.rotation.z),
                ),
            )
            self.set_pose_target(move_robot=move_robot, pose=pose)
        except TransformException:
            pass

    def _maybe_abort_on_loose_plug(self, get_observation: GetObservationCallback) -> bool:
        try:
            plug_tf = self._lookup_transform(
                "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link"
            )
            gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
        except TransformException:
            return False

        offset = self._translation(gripper_tf) - self._translation(plug_tf)
        if self._baseline_plug_offset is None:
            self._baseline_plug_offset = offset
            return False

        drift = float(np.linalg.norm(offset - self._baseline_plug_offset))
        now = self.time_now()
        if drift > self.plug_drift_abort_m:
            if self._plug_drift_started_at is None:
                self._plug_drift_started_at = now
            elif now >= self._plug_drift_started_at + Duration(seconds=self.plug_drift_hold_s):
                self.get_logger().warn(
                    f"Loose plug detected: offset drift {drift:.4f} m exceeded limit."
                )
                self._last_failure_reason = "plug_drift_loose"
                return True
        else:
            self._plug_drift_started_at = None

        obs = get_observation()
        if obs is None or not obs.joint_states.position:
            return False
        gripper_joint = float(obs.joint_states.position[-1])
        if self._baseline_gripper_joint is None:
            self._baseline_gripper_joint = gripper_joint
            return False
        if gripper_joint > self._baseline_gripper_joint + self.gripper_open_abort_rad:
            self.get_logger().warn(
                "Gripper opening detected during approach; aborting to preserve plug."
            )
            self._last_failure_reason = "gripper_open_loose"
            return True
        return False

    def _force_scale_and_backoff(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        deadline: Time,
    ) -> tuple[bool, float]:
        obs = get_observation()
        force_n = self._force_mag(obs)
        if force_n >= self.hard_force_n:
            send_feedback(
                f"teacher_mode=proximity termination_reason=hard_force force_n={force_n:.2f}"
            )
            self.get_logger().warn(f"Hard force abort at {force_n:.2f} N")
            self._last_failure_reason = "hard_force"
            return False, 0.0

        now = self.time_now()
        cooldown_active = (
            self._backoff_cooldown_until is not None and now < self._backoff_cooldown_until
        )
        if force_n >= self.backoff_force_n and not cooldown_active and obs is not None:
            send_feedback(f"teacher_mode=proximity state=force_backoff force_n={force_n:.2f}")
            self._command_current_hold(move_robot)
            self.sleep_for(0.2)
            force_vec = np.array(
                [
                    obs.wrist_wrench.wrench.force.x,
                    obs.wrist_wrench.wrench.force.y,
                    obs.wrist_wrench.wrench.force.z,
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(force_vec))
            if norm > 1e-6:
                direction = -force_vec / norm
                steps = max(1, int(self.backoff_duration_s * self.rate_hz))
                step_dist = self.backoff_distance_m / steps
                dt = 1.0 / self.rate_hz
                for _ in range(steps):
                    if self.time_now() >= deadline:
                        return False, 0.0
                    try:
                        gripper_tf = self._lookup_transform("base_link", "gripper/tcp")
                        pos = self._translation(gripper_tf) + direction * step_dist
                        pose = Pose(
                            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                            orientation=Quaternion(
                                w=float(gripper_tf.rotation.w),
                                x=float(gripper_tf.rotation.x),
                                y=float(gripper_tf.rotation.y),
                                z=float(gripper_tf.rotation.z),
                            ),
                        )
                        self.set_pose_target(move_robot=move_robot, pose=pose)
                    except TransformException:
                        break
                    self.sleep_for(dt)
            self._backoff_cooldown_until = now + Duration(seconds=self.backoff_cooldown_s)

        if force_n <= self.soft_force_n:
            return True, 1.0
        if self.backoff_force_n <= self.soft_force_n:
            return True, 0.4
        ratio = (force_n - self.soft_force_n) / (self.backoff_force_n - self.soft_force_n)
        return True, float(np.clip(1.0 - 0.6 * ratio, 0.4, 1.0))

    def _calc_gripper_pose_for_tip_target(
        self,
        port_frame: str,
        target_tip: np.ndarray,
        slerp_fraction: float,
        position_fraction: float,
    ) -> Pose:
        """Return a TCP pose that moves the ground-truth plug tip to target_tip."""
        port_transform = self._lookup_transform("base_link", port_frame)
        plug_tf = self._lookup_transform(
            "base_link", f"{self._task.cable_name}/{self._task.plug_name}_link"
        )
        gripper_tf = self._lookup_transform("base_link", "gripper/tcp")

        q_port = self._quaternion_wxyz(port_transform)
        q_plug = self._quaternion_wxyz(plug_tf)
        # Correct inverse for unit quaternion in (w, x, y, z) convention.
        q_plug_inv = (q_plug[0], -q_plug[1], -q_plug[2], -q_plug[3])
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
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
        port_frame: str,
        target_tip: np.ndarray,
        duration_s: float,
        label: str,
        deadline: Time,
    ) -> bool:
        dt = 1.0 / self.rate_hz
        steps = max(1, int(duration_s * self.rate_hz))

        for step in range(steps):
            if self.time_now() >= deadline:
                self.get_logger().warn(
                    f"ProximityTeacher timed out while moving to {label}."
                )
                self._last_failure_reason = "timeout_move"
                return False
            if self._maybe_abort_on_loose_plug(get_observation):
                send_feedback("teacher_mode=proximity termination_reason=plug_loose")
                return False
            ok, speed_scale = self._force_scale_and_backoff(
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                deadline=deadline,
            )
            if not ok:
                return False
            fraction = (step + 1) / steps
            try:
                target_tip_step = self._stabilize_tip_target(target_tip)
                pose = self._calc_gripper_pose_for_tip_target(
                    port_frame=port_frame,
                    target_tip=target_tip_step,
                    slerp_fraction=fraction,
                    position_fraction=fraction,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                self._maybe_tighten_gripper(get_observation, move_robot)
            except TransformException as ex:
                self.get_logger().warn(
                    f"TF lookup failed while moving to {label}: {ex}"
                )
                self._last_failure_reason = "tf_move"
                return False
            self.sleep_for(dt / max(0.4, speed_scale))

        return True

    def _hold_tip_target(
        self,
        move_robot: MoveRobotCallback,
        get_observation: GetObservationCallback,
        send_feedback: SendFeedbackCallback,
        port_frame: str,
        target_tip: np.ndarray,
        duration_s: float,
        label: str,
        deadline: Time,
    ) -> bool:
        dt = 1.0 / self.rate_hz
        steps = max(1, int(duration_s * self.rate_hz))

        for _ in range(steps):
            if self.time_now() >= deadline:
                self.get_logger().warn(
                    f"ProximityTeacher timed out while holding {label}."
                )
                self._last_failure_reason = "timeout_hold"
                return False
            if self._maybe_abort_on_loose_plug(get_observation):
                send_feedback("teacher_mode=proximity termination_reason=plug_loose")
                return False
            ok, speed_scale = self._force_scale_and_backoff(
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                deadline=deadline,
            )
            if not ok:
                return False
            try:
                target_tip_step = self._stabilize_tip_target(target_tip)
                pose = self._calc_gripper_pose_for_tip_target(
                    port_frame=port_frame,
                    target_tip=target_tip_step,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
                self._maybe_tighten_gripper(get_observation, move_robot)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed while holding {label}: {ex}")
                self._last_failure_reason = "tf_hold"
                return False
            self.sleep_for(dt / max(0.4, speed_scale))

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
        self._reset_runtime_guards()

        start_time = self.time_now()
        runtime_limit_s = min(float(task.time_limit), 65.0)
        deadline = start_time + Duration(seconds=runtime_limit_s)

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
                get_observation=get_observation,
                send_feedback=send_feedback,
                port_frame=port_frame,
                target_tip=target_tip,
                duration_s=move_duration,
                label=label,
                deadline=deadline,
            ):
                self._command_current_hold(move_robot)
                if self.abort_to_timeout:
                    reason = self._last_failure_reason or "move_failed"
                    return self._linger_until_timeout(
                        deadline=deadline,
                        move_robot=move_robot,
                        get_observation=get_observation,
                        send_feedback=send_feedback,
                        reason=reason,
                    )
                return False
            if not self._hold_tip_target(
                move_robot=move_robot,
                get_observation=get_observation,
                send_feedback=send_feedback,
                port_frame=port_frame,
                target_tip=target_tip,
                duration_s=hold_duration,
                label=label,
                deadline=deadline,
            ):
                self._command_current_hold(move_robot)
                if self.abort_to_timeout:
                    reason = self._last_failure_reason or "hold_failed"
                    return self._linger_until_timeout(
                        deadline=deadline,
                        move_robot=move_robot,
                        get_observation=get_observation,
                        send_feedback=send_feedback,
                        reason=reason,
                    )
                return False

        self._command_current_hold(move_robot)
        self.get_logger().info(
            "ProximityTeacher reached standoff target. Holding until runtime limit."
        )
        return self._linger_until_timeout(
            deadline=deadline,
            move_robot=move_robot,
            get_observation=get_observation,
            send_feedback=send_feedback,
            reason="standoff_complete_waiting_for_runtime_limit",
        )
