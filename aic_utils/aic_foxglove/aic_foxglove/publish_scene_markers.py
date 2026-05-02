import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
import yaml
from tf2_ros import TransformListener, Buffer


class SceneMarkerPublisher(Node):
    def __init__(self):
        super().__init__('publish_scene_markers')

        # We publish to a scene_markers topic
        self.marker_pub = self.create_publisher(MarkerArray, '/scene_markers', 10)

        # TF listener to discover active frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Exact frame-to-mesh mappings
        self.exact_frame_meshes = {}

        # Suffix-based mappings for dynamically named frames (e.g., cable_0/sc_plug_link)
        self.suffix_meshes = {
            '/sc_plug_link': 'package://aic_assets/models/SC_Plug/sc_plug_visual.glb',
            '/lc_plug_link': 'package://aic_assets/models/LC_Plug/lc_plug_visual.glb',
            '/sfp_module_link': 'package://aic_assets/models/SFP_Module/sfp_module_visual.glb',
        }

        # Publish at 1 Hz (Foxglove caches meshes, so we don't need high frequency)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self._logged_frames = False
        self.get_logger().info("Scene marker publisher started.")

    def publish_markers(self):
        try:
            # Get all current frames in the TF tree
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
            if not frames_yaml:
                self.get_logger().info("No TF frames yet...", throttle_duration_sec=5.0)
                return
            frames_dict = yaml.safe_load(frames_yaml)
            if not frames_dict:
                self.get_logger().info("TF frames dict empty...", throttle_duration_sec=5.0)
                return

            frame_names = list(frames_dict.keys())
        except Exception as e:
            self.get_logger().warning(f"Failed to parse TF frames: {e}")
            return

        # Log discovered frames once for debugging
        if not self._logged_frames:
            self.get_logger().info(f"Discovered {len(frame_names)} TF frames: {frame_names}")
            self._logged_frames = True

        marker_array = MarkerArray()
        marker_id = 0
        matched_frames = []

        for frame in frame_names:
            mesh_uri = None

            # Check exact matches first
            if frame in self.exact_frame_meshes:
                mesh_uri = self.exact_frame_meshes[frame]

            # Then check suffix matches (for dynamic prefixes like cable_0/)
            if mesh_uri is None:
                for suffix, uri in self.suffix_meshes.items():
                    if frame.endswith(suffix):
                        mesh_uri = uri
                        break

            if mesh_uri is None:
                continue

            matched_frames.append(f"{frame} -> {mesh_uri.split('/')[-1]}")

            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "scene_objects"
            marker.id = marker_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD

            # Pose is identity - the TF frame itself defines position/orientation
            marker.pose = Pose()
            marker.pose.orientation.w = 1.0

            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.mesh_resource = mesh_uri
            marker.mesh_use_embedded_materials = True

            marker_array.markers.append(marker)
            marker_id += 1

        if marker_array.markers:
            if not hasattr(self, '_logged_matches'):
                self.get_logger().info(f"Publishing {len(marker_array.markers)} markers: {matched_frames}")
                self._logged_matches = True
            self.marker_pub.publish(marker_array)
        else:
            self.get_logger().info("No matching frames found for scene markers.", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = SceneMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
