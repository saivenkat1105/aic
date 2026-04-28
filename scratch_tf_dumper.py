import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
import time

class TFTreeDumper(Node):
    def __init__(self):
        super().__init__('tf_dumper')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(2.0, self.dump_frames)

    def dump_frames(self):
        frames = self.tf_buffer.all_frames_as_yaml()
        with open("/home/user/aic/scratch_frames.yaml", "w") as f:
            f.write(frames)
        self.get_logger().info("Dumped frames to /home/user/aic/scratch_frames.yaml")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = TFTreeDumper()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
