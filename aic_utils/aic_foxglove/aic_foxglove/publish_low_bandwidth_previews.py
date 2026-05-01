import copy
import time
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class FoxglovePreviewPublisher(Node):
    def __init__(self):
        super().__init__("foxglove_preview_publisher")

        self.declare_parameter("preview_width", 576)
        self.declare_parameter("preview_height", 512)
        self.declare_parameter("jpeg_quality", 70)
        self.declare_parameter("fps", 5.0)

        self.preview_width = self.get_parameter("preview_width").value
        self.preview_height = self.get_parameter("preview_height").value
        self.jpeg_quality = self.get_parameter("jpeg_quality").value
        self.fps = self.get_parameter("fps").value
        self.min_period_s = 1.0 / self.fps if self.fps > 0 else 0.0

        self.camera_infos: Dict[str, CameraInfo] = {}
        self.last_publish_time: Dict[str, float] = {}

        self.cameras = ("left", "center", "right")
        self.image_subs = []
        self.info_subs = []
        self.image_pubs = {}
        self.info_pubs = {}

        for camera in self.cameras:
            camera_ns = f"{camera}_camera"
            self.image_pubs[camera] = self.create_publisher(
                CompressedImage,
                f"/foxglove/{camera_ns}/image/compressed",
                10,
            )
            self.info_pubs[camera] = self.create_publisher(
                CameraInfo,
                f"/foxglove/{camera_ns}/camera_info",
                10,
            )
            self.image_subs.append(
                self.create_subscription(
                    Image,
                    f"/{camera_ns}/image",
                    lambda msg, camera=camera: self.image_callback(camera, msg),
                    qos_profile_sensor_data,
                )
            )
            self.info_subs.append(
                self.create_subscription(
                    CameraInfo,
                    f"/{camera_ns}/camera_info",
                    lambda msg, camera=camera: self.info_callback(camera, msg),
                    qos_profile_sensor_data,
                )
            )

        self.get_logger().info(
            "Foxglove preview publisher started: "
            f"{self.preview_width}x{self.preview_height}, "
            f"jpeg_quality={self.jpeg_quality}, fps={self.fps}"
        )

    def info_callback(self, camera: str, msg: CameraInfo) -> None:
        self.camera_infos[camera] = msg

    def image_callback(self, camera: str, msg: Image) -> None:
        now = time.monotonic()
        last_publish = self.last_publish_time.get(camera, 0.0)
        if now - last_publish < self.min_period_s:
            return

        try:
            image = self.ros_image_to_cv2(msg)
            preview = cv2.resize(
                image,
                (self.preview_width, self.preview_height),
                interpolation=cv2.INTER_AREA,
            )
            jpeg_input = self.to_bgr_for_jpeg(preview, msg.encoding)
            success, encoded = cv2.imencode(
                ".jpg",
                jpeg_input,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not success:
                self.get_logger().warning(
                    f"Failed to JPEG-compress {camera} preview",
                    throttle_duration_sec=5.0,
                )
                return

            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "rgb8; jpeg compressed rgb8"
            compressed_msg.data = encoded.tobytes()
            self.image_pubs[camera].publish(compressed_msg)

            scaled_info = self.scaled_camera_info(camera, msg)
            if scaled_info is not None:
                self.info_pubs[camera].publish(scaled_info)

            self.last_publish_time[camera] = now
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to publish {camera} preview: {exc}",
                throttle_duration_sec=5.0,
            )

    def ros_image_to_cv2(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8"):
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

        channels = 1 if msg.encoding == "mono8" else 4 if "a" in msg.encoding else 3
        expected_step = msg.width * channels
        if msg.step < expected_step:
            raise ValueError(
                f"Image step {msg.step} is too small for {msg.width}x{channels}"
            )

        image = np.frombuffer(msg.data, dtype=np.uint8)
        image = image.reshape((msg.height, msg.step))
        image = image[:, :expected_step]

        if channels == 1:
            return image.reshape((msg.height, msg.width))
        return image.reshape((msg.height, msg.width, channels))

    def to_bgr_for_jpeg(self, image: np.ndarray, encoding: str) -> np.ndarray:
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def scaled_camera_info(self, camera: str, image_msg: Image) -> Optional[CameraInfo]:
        info = self.camera_infos.get(camera)
        if info is None:
            return None

        scaled = copy.deepcopy(info)
        scale_x = self.preview_width / float(image_msg.width)
        scale_y = self.preview_height / float(image_msg.height)

        scaled.header = image_msg.header
        scaled.width = self.preview_width
        scaled.height = self.preview_height

        scaled.k[0] *= scale_x
        scaled.k[2] *= scale_x
        scaled.k[4] *= scale_y
        scaled.k[5] *= scale_y

        scaled.p[0] *= scale_x
        scaled.p[2] *= scale_x
        scaled.p[3] *= scale_x
        scaled.p[5] *= scale_y
        scaled.p[6] *= scale_y
        scaled.p[7] *= scale_y

        return scaled


def main(args=None):
    rclpy.init(args=args)
    node = FoxglovePreviewPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
