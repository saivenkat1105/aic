#!/usr/bin/env python3

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from aic_model_interfaces.msg import Observation
from aic_training_interfaces.srv import CaptureStatus, StartEpisodeCapture, StopEpisodeCapture
from rclpy.node import Node
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

        self.max_pending_frames = int(self.get_parameter("max_pending_frames").value)
        self.write_workers = max(1, int(self.get_parameter("write_workers").value))
        self.convert_workers = max(1, int(self.get_parameter("convert_workers").value))
        self.images_output_subdir = str(self.get_parameter("images_output_subdir").value)
        self.keep_every_nth_bin = int(self.get_parameter("keep_every_nth_bin").value)
        self.stats_log_period_s = float(self.get_parameter("stats_log_period_s").value)

        self.active = False
        self.active_episode_id = ""
        self.run_dir: Path | None = None
        self.episode_dir: Path | None = None
        self.enabled_cameras = {"left", "center", "right"}
        self.frame_idx = 0
        self.frames_file = None
        self.frames_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.obs_rx = 0
        self.written_frames = 0
        self.converted_images = 0
        self.purged_bins = 0
        self.dropped_frames = 0

        self.write_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.max_pending_frames)
        self.convert_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.max_pending_frames * 6)
        self.shutdown_event = threading.Event()
        self.workers: list[threading.Thread] = []
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
            self.active_episode_id = ""

        resp.written_frames = int(self.written_frames)
        resp.converted_images = int(self.converted_images)
        resp.purged_bins = int(self.purged_bins)
        resp.dropped_frames = int(self.dropped_frames)
        resp.pending_write = int(self.write_q.qsize())
        resp.pending_convert = int(self.convert_q.qsize())
        resp.success = resp.pending_write == 0 and (resp.pending_convert == 0 or not req.require_webp_done)
        resp.message = f"stopped capture for {ep}"
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
        resp.message = "ok"
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

        frame = {
            "frame_idx": frame_idx,
            "episode_id": episode_id,
            "obs_stamp": self._stamp_to_sec(msg.center_image.header.stamp),
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
            },
            "images": {},
        }
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

    def _log_stats(self) -> None:
        with self.state_lock:
            self.get_logger().info(
                "frame_sink stats: "
                f"active={self.active} episode={self.active_episode_id or '-'} "
                f"obs_rx={self.obs_rx} queued={self.write_q.qsize()} "
                f"written={self.written_frames} converted={self.converted_images} "
                f"purged={self.purged_bins} dropped={self.dropped_frames} "
                f"convert_queued={self.convert_q.qsize()}"
            )

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


def main() -> None:
    rclpy.init()
    node = TrainingFrameSink()
    try:
        rclpy.spin(node)
    finally:
        node.shutdown_event.set()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
