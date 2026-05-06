#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import queue
import traceback
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from aic_model_interfaces.msg import Observation
from aic_training_interfaces.srv import CaptureStatus, StartEpisodeCapture, StopEpisodeCapture
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from PIL import Image


class TrainingFrameSink(Node):
    def __init__(self) -> None:
        super().__init__("training_frame_sink")

        self.declare_parameter("max_pending_frames", 64)
        self.declare_parameter("write_workers", 2)
        self.declare_parameter("convert_workers", 2)
        self.declare_parameter("images_output_subdir", "images_debug")
        self.declare_parameter("keep_every_nth_bin", 0)
        self.declare_parameter("stats_log_period_s", 5.0)
        self.declare_parameter("postprocess_workers", 1)
        self.declare_parameter("postprocess_visibility_labels", True)
        self.declare_parameter("postprocess_labels_filename", "labels_visibility_occlusion.jsonl")
        self.declare_parameter("wait_for_postprocess_on_stop", False)
        self.declare_parameter("frame_schema_version", "visual_proximity_v1")

        self.max_pending_frames = int(self.get_parameter("max_pending_frames").value)
        self.write_workers = max(1, int(self.get_parameter("write_workers").value))
        self.convert_workers = max(1, int(self.get_parameter("convert_workers").value))
        self.images_output_subdir = str(self.get_parameter("images_output_subdir").value)
        self.keep_every_nth_bin = int(self.get_parameter("keep_every_nth_bin").value)
        self.stats_log_period_s = float(self.get_parameter("stats_log_period_s").value)
        self.postprocess_workers = max(1, int(self.get_parameter("postprocess_workers").value))
        self.postprocess_visibility_labels = bool(
            self.get_parameter("postprocess_visibility_labels").value
        )
        self.postprocess_labels_filename = str(
            self.get_parameter("postprocess_labels_filename").value
        )
        self.wait_for_postprocess_on_stop = bool(
            self.get_parameter("wait_for_postprocess_on_stop").value
        )
        self.frame_schema_version = str(self.get_parameter("frame_schema_version").value)

        self.active = False
        self.active_episode_id = ""
        self.run_dir: Path | None = None
        self.episode_dir: Path | None = None
        self.enabled_cameras = {"left", "center", "right"}
        self.task_context: dict[str, Any] = {}
        self.scene_context: dict[str, Any] = {}
        self.target_port_link_frame = ""
        self.target_port_entrance_frame = ""
        self.static_gt_cache: dict[str, Any] | None = None
        self.frame_idx = 0
        self.frames_file = None
        self.frames_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.obs_rx = 0
        self.written_frames = 0
        self.converted_images = 0
        self.purged_bins = 0
        self.dropped_frames = 0
        self.postprocessed_episodes = 0
        self.postprocess_failures = 0
        self.pending_postprocess = 0

        self.write_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.max_pending_frames)
        self.convert_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.max_pending_frames * 6)
        self.postprocess_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.workers: list[threading.Thread] = []
        self._postprocess_done = threading.Event()
        self._postprocess_done.set()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self._start_workers()

        self.create_subscription(Observation, "/observations", self._on_observation, 10)
        self.create_service(StartEpisodeCapture, "/training_frame_sink/start_episode_capture", self._start_episode)
        self.create_service(StopEpisodeCapture, "/training_frame_sink/stop_episode_capture", self._stop_episode)
        self.create_service(CaptureStatus, "/training_frame_sink/capture_status", self._capture_status)
        self.create_timer(self.stats_log_period_s, self._log_stats)

    def _start_workers(self) -> None:
        for i in range(self.write_workers):
            t = threading.Thread(target=self._write_worker, name=f"write_worker_{i}", daemon=True)
            t.start()
            self.workers.append(t)
        for i in range(self.convert_workers):
            t = threading.Thread(target=self._convert_worker, name=f"convert_worker_{i}", daemon=True)
            t.start()
            self.workers.append(t)
        for i in range(self.postprocess_workers):
            t = threading.Thread(target=self._postprocess_worker, name=f"postprocess_worker_{i}", daemon=True)
            t.start()
            self.workers.append(t)

    def _start_episode(self, req: StartEpisodeCapture.Request, resp: StartEpisodeCapture.Response):
        with self.state_lock:
            if self.active:
                resp.success = False
                resp.message = f"capture already active for episode {self.active_episode_id}"
                return resp

            self.active_episode_id = str(req.episode_id)
            self.run_dir = Path(req.run_dir)
            self.episode_dir = Path(req.episode_dir)
            self.enabled_cameras = set(req.enable_cameras) if req.enable_cameras else {"left", "center", "right"}
            self.task_context = self._safe_load_json(req.task_json_path)
            self.scene_context = self._safe_load_json(req.scene_json_path)
            self.target_port_link_frame = self._target_port_link_frame_from_task(self.task_context)
            self.target_port_entrance_frame = self._target_port_entrance_frame_from_task(
                self.task_context
            )
            self.static_gt_cache = None
            self.frame_idx = 0
            self.written_frames = 0
            self.converted_images = 0
            self.purged_bins = 0
            self.dropped_frames = 0
            self.obs_rx = 0

            (self.episode_dir / "images" / "left").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "images" / "center").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "images" / "right").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / self.images_output_subdir / "left").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / self.images_output_subdir / "center").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / self.images_output_subdir / "right").mkdir(parents=True, exist_ok=True)
            self.frames_file = open(self.episode_dir / "frames.jsonl", "w", encoding="utf-8")
            self.active = True

        resp.success = True
        resp.message = f"started capture for {self.active_episode_id}"
        return resp

    def _stop_episode(self, req: StopEpisodeCapture.Request, resp: StopEpisodeCapture.Response):
        start_t = time.time()
        timeout_s = max(0.0, float(req.flush_timeout_s))

        with self.state_lock:
            if not self.active:
                resp.success = False
                resp.message = "capture not active"
                return resp
            if req.episode_id and req.episode_id != self.active_episode_id:
                resp.success = False
                resp.message = (
                    f"episode_id mismatch active={self.active_episode_id} request={req.episode_id}"
                )
                return resp
            self.active = False

        while time.time() - start_t <= timeout_s:
            if self.write_q.empty() and (self.convert_q.empty() or not req.require_webp_done):
                break
            time.sleep(0.05)

        with self.state_lock:
            if self.frames_file is not None:
                self.frames_file.close()
                self.frames_file = None
            ep = self.active_episode_id
            ep_dir = self.episode_dir
            self.active_episode_id = ""
            self.episode_dir = None

        if self.postprocess_visibility_labels and ep_dir is not None:
            self._postprocess_done.clear()
            self.postprocess_q.put(
                {
                    "episode_id": ep,
                    "episode_dir": ep_dir,
                    "task": dict(self.task_context),
                }
            )
            with self.state_lock:
                self.pending_postprocess += 1

        if self.wait_for_postprocess_on_stop:
            while time.time() - start_t <= timeout_s:
                with self.state_lock:
                    pending_post = int(self.pending_postprocess)
                if pending_post == 0:
                    break
                time.sleep(0.05)

        resp.written_frames = int(self.written_frames)
        resp.converted_images = int(self.converted_images)
        resp.purged_bins = int(self.purged_bins)
        resp.dropped_frames = int(self.dropped_frames)
        resp.pending_write = int(self.write_q.qsize())
        resp.pending_convert = int(self.convert_q.qsize())
        post_ok = True
        if self.wait_for_postprocess_on_stop:
            with self.state_lock:
                post_ok = self.pending_postprocess == 0
        resp.success = (
            resp.pending_write == 0
            and (resp.pending_convert == 0 or not req.require_webp_done)
            and post_ok
        )
        resp.message = (
            f"stopped capture for {ep}; "
            f"postprocess_pending={self.pending_postprocess}"
        )
        return resp

    def _capture_status(self, req: CaptureStatus.Request, resp: CaptureStatus.Response):
        _ = req
        with self.state_lock:
            resp.active = bool(self.active)
            resp.active_episode_id = self.active_episode_id
            resp.written_frames = int(self.written_frames)
            resp.converted_images = int(self.converted_images)
            resp.purged_bins = int(self.purged_bins)
            resp.dropped_frames = int(self.dropped_frames)
        resp.queue_depth = int(self.write_q.qsize())
        resp.convert_queue_depth = int(self.convert_q.qsize())
        resp.message = (
            f"ok postprocess_pending={self.pending_postprocess} "
            f"postprocessed={self.postprocessed_episodes} "
            f"postprocess_failures={self.postprocess_failures}"
        )
        return resp

    def _on_observation(self, msg: Observation) -> None:
        with self.state_lock:
            if not self.active or self.episode_dir is None:
                return
            frame_idx = self.frame_idx
            self.frame_idx += 1
            self.obs_rx += 1
            episode_dir = self.episode_dir
            episode_id = self.active_episode_id

        static_gt = self._ensure_static_gt(msg)

        frame = {
            "schema_version": self.frame_schema_version,
            "frame_idx": frame_idx,
            "episode_id": episode_id,
            "obs_stamp": self._stamp_to_sec(msg.center_image.header.stamp),
            "task": {
                "task_id": str(self.task_context.get("id", "")),
                "target_module_name": str(self.task_context.get("target_module_name", "")),
                "port_name": str(self.task_context.get("port_name", "")),
                "port_type": str(self.task_context.get("port_type", "")),
                "plug_type": str(self.task_context.get("plug_type", "")),
            },
            "wrench": {
                "fx": float(msg.wrist_wrench.wrench.force.x),
                "fy": float(msg.wrist_wrench.wrench.force.y),
                "fz": float(msg.wrist_wrench.wrench.force.z),
                "tx": float(msg.wrist_wrench.wrench.torque.x),
                "ty": float(msg.wrist_wrench.wrench.torque.y),
                "tz": float(msg.wrist_wrench.wrench.torque.z),
            },
            "joint_states": {
                "name": list(msg.joint_states.name),
                "position": [float(v) for v in msg.joint_states.position],
                "velocity": [float(v) for v in msg.joint_states.velocity],
                "effort": [float(v) for v in msg.joint_states.effort],
            },
            "controller_state": {
                "target_mode": int(msg.controller_state.target_mode.mode),
                "tcp_pose": self._pose_to_dict(msg.controller_state.tcp_pose),
                "tcp_velocity": self._twist_to_dict(msg.controller_state.tcp_velocity),
                "tcp_error": [float(v) for v in msg.controller_state.tcp_error],
            },
            "images": {},
            "camera_info": {
                "left": self._camera_info_to_dict(msg.left_camera_info),
                "center": self._camera_info_to_dict(msg.center_camera_info),
                "right": self._camera_info_to_dict(msg.right_camera_info),
            },
            "training_gt": static_gt,
        }
        # Flat aliases kept for downstream tooling that expects top-level fields.
        frame["tcp_velocity"] = frame["controller_state"]["tcp_velocity"]
        frame["tcp_error"] = frame["controller_state"]["tcp_error"]
        frame["left_camera_info"] = frame["camera_info"]["left"]
        frame["center_camera_info"] = frame["camera_info"]["center"]
        frame["right_camera_info"] = frame["camera_info"]["right"]
        frame["t_base_target_port_link_gt"] = static_gt.get("t_base_target_port_link_gt")
        frame["t_base_target_port_entrance_gt"] = static_gt.get("t_base_target_port_entrance_gt")
        cams = {
            "left": msg.left_image,
            "center": msg.center_image,
            "right": msg.right_image,
        }
        jobs = []
        for cam, imsg in cams.items():
            if cam not in self.enabled_cameras:
                continue
            rel_bin = Path("images") / cam / f"{frame_idx:06d}.bin"
            rel_webp = Path(self.images_output_subdir) / cam / f"{frame_idx:06d}.webp"
            meta = {
                "path": rel_webp.as_posix(),
                "bin_path": rel_bin.as_posix(),
                "width": int(imsg.width),
                "height": int(imsg.height),
                "encoding": str(imsg.encoding),
                "step": int(imsg.step),
                "is_bigendian": int(imsg.is_bigendian),
                "stamp": self._stamp_to_sec(imsg.header.stamp),
                "frame_id": str(imsg.header.frame_id),
            }
            frame["images"][cam] = meta
            jobs.append(
                {
                    "episode_dir": episode_dir,
                    "frame_idx": frame_idx,
                    "camera": cam,
                    "raw": bytes(imsg.data),
                    "meta": meta,
                }
            )

        try:
            self.write_q.put_nowait({"frame": frame, "jobs": jobs})
        except queue.Full:
            try:
                _ = self.write_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.write_q.put_nowait({"frame": frame, "jobs": jobs})
            except queue.Full:
                with self.state_lock:
                    self.dropped_frames += 1
                return
            with self.state_lock:
                self.dropped_frames += 1

    def _write_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                item = self.write_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                frame = item["frame"]
                jobs = item["jobs"]
                for job in jobs:
                    bin_path = Path(job["episode_dir"]) / Path(job["meta"]["bin_path"])
                    bin_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(bin_path, "wb") as f:
                        f.write(job["raw"])
                    self.convert_q.put(job)
                with self.frames_lock:
                    if self.frames_file is not None:
                        self.frames_file.write(json.dumps(frame) + "\n")
                with self.state_lock:
                    self.written_frames += 1
            finally:
                self.write_q.task_done()

    def _convert_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                job = self.convert_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                episode_dir = Path(job["episode_dir"])
                meta = job["meta"]
                raw = job["raw"]
                bin_path = episode_dir / Path(meta["bin_path"])
                webp_path = episode_dir / Path(meta["path"])
                webp_path.parent.mkdir(parents=True, exist_ok=True)
                image = self._decode_image_from_meta(raw, meta)
                image.save(webp_path, format="WEBP", lossless=True)
                with self.state_lock:
                    self.converted_images += 1
                keep_bin = self.keep_every_nth_bin > 0 and (
                    (int(job["frame_idx"]) % self.keep_every_nth_bin) == 0
                )
                if not keep_bin and bin_path.exists():
                    bin_path.unlink()
                    with self.state_lock:
                        self.purged_bins += 1
            except Exception as exc:
                self.get_logger().warn(f"WebP conversion failed for {job.get('meta', {}).get('bin_path')}: {exc}")
            finally:
                self.convert_q.task_done()

    def _postprocess_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                item = self.postprocess_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                episode_dir = Path(item["episode_dir"])
                task = dict(item.get("task", {}))
                out_path = episode_dir / self.postprocess_labels_filename
                self._generate_visibility_occlusion_labels(
                    frames_path=episode_dir / "frames.jsonl",
                    out_path=out_path,
                    task=task,
                )
                with self.state_lock:
                    self.postprocessed_episodes += 1
            except Exception as exc:
                with self.state_lock:
                    self.postprocess_failures += 1
                self.get_logger().warn(
                    "Visibility/occlusion postprocess failed for "
                    f"{item.get('episode_id', 'unknown')}: {exc}\n{traceback.format_exc()}"
                )
            finally:
                with self.state_lock:
                    self.pending_postprocess = max(0, self.pending_postprocess - 1)
                    if self.pending_postprocess == 0:
                        self._postprocess_done.set()
                self.postprocess_q.task_done()

    def _log_stats(self) -> None:
        with self.state_lock:
            self.get_logger().info(
                "frame_sink stats: "
                f"active={self.active} episode={self.active_episode_id or '-'} "
                f"obs_rx={self.obs_rx} queued={self.write_q.qsize()} "
                f"written={self.written_frames} converted={self.converted_images} "
                f"purged={self.purged_bins} dropped={self.dropped_frames} "
                f"convert_queued={self.convert_q.qsize()} "
                f"postprocess_pending={self.pending_postprocess} "
                f"postprocessed={self.postprocessed_episodes} "
                f"postprocess_failures={self.postprocess_failures}"
            )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _safe_load_json(path: str) -> dict[str, Any]:
        if not path:
            return {}
        p = Path(path)
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _target_port_link_frame_from_task(task: dict[str, Any]) -> str:
        module = str(task.get("target_module_name", ""))
        port_name = str(task.get("port_name", ""))
        if not module or not port_name:
            return ""
        return f"task_board/{module}/{port_name}_link"

    @staticmethod
    def _target_port_entrance_frame_from_task(task: dict[str, Any]) -> str:
        module = str(task.get("target_module_name", ""))
        port_name = str(task.get("port_name", ""))
        if not module or not port_name:
            return ""
        return f"task_board/{module}/{port_name}_link_entrance"

    def _ensure_static_gt(self, msg: Observation) -> dict[str, Any]:
        if self.static_gt_cache is not None:
            return self.static_gt_cache

        base_to_port_link = self._lookup_transform_dict("base_link", self.target_port_link_frame)
        base_to_port_entrance = self._lookup_transform_dict(
            "base_link", self.target_port_entrance_frame
        )

        cam_frames = {
            "left": str(msg.left_image.header.frame_id),
            "center": str(msg.center_image.header.frame_id),
            "right": str(msg.right_image.header.frame_id),
        }
        base_to_camera_optical: dict[str, Any] = {}
        for cam, frame_id in cam_frames.items():
            if not frame_id:
                continue
            base_to_camera_optical[cam] = self._lookup_transform_dict("base_link", frame_id)

        all_ports_gt = []
        scene_ports = self.scene_context.get("task_board", {}).get("ports", [])
        if isinstance(scene_ports, list):
            for p in scene_ports:
                if not isinstance(p, dict):
                    continue
                module_name = str(p.get("module_name", ""))
                port_name = str(p.get("port_name", ""))
                port_link_frame = f"task_board/{module_name}/{port_name}_link"
                port_entrance_frame = f"{port_link_frame}_entrance"
                all_ports_gt.append(
                    {
                        "module_name": module_name,
                        "port_name": port_name,
                        "is_task_target_port": bool(p.get("is_task_target_port", False)),
                        "t_base_port_link_gt": self._lookup_transform_dict(
                            "base_link", port_link_frame
                        ),
                        "t_base_port_entrance_gt": self._lookup_transform_dict(
                            "base_link", port_entrance_frame
                        ),
                    }
                )

        candidate = {
            "t_base_target_port_link_gt": base_to_port_link,
            "t_base_target_port_entrance_gt": base_to_port_entrance,
            "base_to_camera_optical": base_to_camera_optical,
            "all_ports_gt_base": all_ports_gt,
        }
        camera_ready = all(
            cam in base_to_camera_optical and base_to_camera_optical[cam] is not None
            for cam in ("left", "center", "right")
        )
        if (
            base_to_port_link is not None
            and base_to_port_entrance is not None
            and camera_ready
        ):
            self.static_gt_cache = candidate
        return candidate

    def _lookup_transform_dict(self, target_frame: str, source_frame: str) -> dict[str, Any] | None:
        if not source_frame:
            return None
        try:
            tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException:
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return {
            "target_frame": target_frame,
            "source_frame": source_frame,
            "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
            "rotation": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
        }

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
    def _twist_to_dict(twist) -> dict[str, Any]:
        return {
            "linear": {
                "x": float(twist.linear.x),
                "y": float(twist.linear.y),
                "z": float(twist.linear.z),
            },
            "angular": {
                "x": float(twist.angular.x),
                "y": float(twist.angular.y),
                "z": float(twist.angular.z),
            },
        }

    @staticmethod
    def _camera_info_to_dict(camera_info) -> dict[str, Any]:
        return {
            "distortion_model": str(camera_info.distortion_model),
            "d": [float(v) for v in camera_info.d],
            "k": [float(v) for v in camera_info.k],
            "r": [float(v) for v in camera_info.r],
            "p": [float(v) for v in camera_info.p],
            "binning_x": int(camera_info.binning_x),
            "binning_y": int(camera_info.binning_y),
            "roi": {
                "x_offset": int(camera_info.roi.x_offset),
                "y_offset": int(camera_info.roi.y_offset),
                "height": int(camera_info.roi.height),
                "width": int(camera_info.roi.width),
                "do_rectify": bool(camera_info.roi.do_rectify),
            },
        }

    @staticmethod
    def _decode_image_from_meta(raw: bytes, meta: dict[str, Any]) -> Image.Image:
        width = int(meta["width"])
        height = int(meta["height"])
        encoding = str(meta["encoding"]).lower()
        if encoding == "rgb8":
            return Image.frombytes("RGB", (width, height), raw)
        if encoding == "bgr8":
            return Image.frombytes("RGB", (width, height), raw, "raw", "BGR")
        if encoding == "mono8":
            return Image.frombytes("L", (width, height), raw)
        if encoding == "rgba8":
            return Image.frombytes("RGBA", (width, height), raw)
        if encoding == "bgra8":
            return Image.frombytes("RGBA", (width, height), raw, "raw", "BGRA")
        raise RuntimeError(f"Unsupported encoding for conversion: {encoding}")

    @classmethod
    def _generate_visibility_occlusion_labels(
        cls,
        *,
        frames_path: Path,
        out_path: Path,
        task: dict[str, Any],
    ) -> None:
        if not frames_path.is_file():
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with frames_path.open("r", encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                labels = {
                    "frame_idx": int(frame.get("frame_idx", 0)),
                    "obs_stamp": float(frame.get("obs_stamp", 0.0)),
                    "task_id": str(task.get("id", "")),
                    "target_module_name": str(task.get("target_module_name", "")),
                    "target_port_name": str(task.get("port_name", "")),
                    "labels": {},
                }
                training_gt = frame.get("training_gt", {})
                t_port_link = training_gt.get("t_base_target_port_link_gt")
                t_port_entrance = training_gt.get("t_base_target_port_entrance_gt")
                p_port_link = cls._point_from_transform(t_port_link)
                p_port_entrance = cls._point_from_transform(t_port_entrance)
                camera_info = frame.get("camera_info", {})
                images = frame.get("images", {})
                cam_tfs = training_gt.get("base_to_camera_optical", {})

                for cam in ("left", "center", "right"):
                    info = camera_info.get(cam, {})
                    im_meta = images.get(cam, {})
                    cam_tf = cam_tfs.get(cam)
                    width = int(im_meta.get("width", 0))
                    height = int(im_meta.get("height", 0))
                    labels["labels"][cam] = {
                        "port_link": cls._project_label_point(
                            p_base=p_port_link,
                            t_base_to_camera=cam_tf,
                            camera_info=info,
                            width=width,
                            height=height,
                        ),
                        "port_entrance": cls._project_label_point(
                            p_base=p_port_entrance,
                            t_base_to_camera=cam_tf,
                            camera_info=info,
                            width=width,
                            height=height,
                        ),
                    }
                fout.write(json.dumps(labels) + "\n")

    @staticmethod
    def _point_from_transform(transform: dict[str, Any] | None) -> tuple[float, float, float] | None:
        if not isinstance(transform, dict):
            return None
        t = transform.get("translation", {})
        if not isinstance(t, dict):
            return None
        try:
            return (float(t["x"]), float(t["y"]), float(t["z"]))
        except Exception:
            return None

    @classmethod
    def _project_label_point(
        cls,
        *,
        p_base: tuple[float, float, float] | None,
        t_base_to_camera: dict[str, Any] | None,
        camera_info: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        if p_base is None:
            return cls._label_unavailable("missing_point")
        if not isinstance(t_base_to_camera, dict):
            return cls._label_unavailable("missing_camera_tf")
        if width <= 0 or height <= 0:
            return cls._label_unavailable("invalid_image_shape")

        p_cam = cls._transform_point_base_to_camera(p_base, t_base_to_camera)
        if p_cam is None:
            return cls._label_unavailable("invalid_camera_tf")

        x, y, z = p_cam
        if z <= 1e-6:
            return {
                "visibility": "out_of_fov",
                "uv_px": [None, None],
                "in_bounds": False,
                "positive_depth": False,
                "depth_m": float(z),
            }

        fx, fy, cx, cy = cls._intrinsics_from_camera_info(camera_info)
        if fx <= 0.0 or fy <= 0.0:
            return cls._label_unavailable("invalid_intrinsics")

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        in_bounds = 0.0 <= u < float(width) and 0.0 <= v < float(height)
        edge_margin_px = 3.0
        near_edge = (
            in_bounds
            and (
                u < edge_margin_px
                or v < edge_margin_px
                or u >= float(width) - edge_margin_px
                or v >= float(height) - edge_margin_px
            )
        )
        visibility = "visible"
        if not in_bounds:
            visibility = "out_of_fov"
        elif near_edge:
            visibility = "truncated"

        return {
            "visibility": visibility,
            "uv_px": [float(u), float(v)],
            "in_bounds": bool(in_bounds),
            "positive_depth": True,
            "depth_m": float(z),
            "projection_source": "gt_projection_only",
        }

    @staticmethod
    def _label_unavailable(reason: str) -> dict[str, Any]:
        return {
            "visibility": "out_of_fov",
            "uv_px": [None, None],
            "in_bounds": False,
            "positive_depth": False,
            "depth_m": None,
            "reason": reason,
        }

    @staticmethod
    def _intrinsics_from_camera_info(camera_info: dict[str, Any]) -> tuple[float, float, float, float]:
        k = camera_info.get("k", [])
        p = camera_info.get("p", [])
        fx = fy = cx = cy = 0.0
        if isinstance(k, list) and len(k) >= 9:
            fx = float(k[0])
            fy = float(k[4])
            cx = float(k[2])
            cy = float(k[5])
        if (fx <= 0.0 or fy <= 0.0) and isinstance(p, list) and len(p) >= 12:
            fx = float(p[0])
            fy = float(p[5])
            cx = float(p[2])
            cy = float(p[6])
        return fx, fy, cx, cy

    @classmethod
    def _transform_point_base_to_camera(
        cls,
        p_base: tuple[float, float, float],
        t_base_to_camera: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        t = t_base_to_camera.get("translation", {})
        q = t_base_to_camera.get("rotation", {})
        try:
            tx = float(t["x"])
            ty = float(t["y"])
            tz = float(t["z"])
            qx = float(q["x"])
            qy = float(q["y"])
            qz = float(q["z"])
            qw = float(q["w"])
        except Exception:
            return None

        rot = cls._quat_to_rot(qx, qy, qz, qw)
        if rot is None:
            return None

        px = float(p_base[0]) - tx
        py = float(p_base[1]) - ty
        pz = float(p_base[2]) - tz
        # inverse rotation: R^T * (p_base - t)
        rx = rot[0][0] * px + rot[1][0] * py + rot[2][0] * pz
        ry = rot[0][1] * px + rot[1][1] * py + rot[2][1] * pz
        rz = rot[0][2] * px + rot[1][2] * py + rot[2][2] * pz
        return (rx, ry, rz)

    @staticmethod
    def _quat_to_rot(
        qx: float, qy: float, qz: float, qw: float
    ) -> list[list[float]] | None:
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if n <= 1e-12:
            return None
        qx /= n
        qy /= n
        qz /= n
        qw /= n
        return [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]


def main() -> None:
    rclpy.init()
    node = TrainingFrameSink()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown_event.set()
        # Best-effort flush for background episode post-processing jobs.
        flush_deadline = time.time() + 3.0
        while time.time() < flush_deadline:
            with node.state_lock:
                if node.pending_postprocess == 0:
                    break
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
