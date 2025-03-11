#!/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud
from custom_interfaces.msg import Camera

from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
from cv_bridge import CvBridge


class FrankaSImPointCloudNode(Node):
    def __init__(self):
        super().__init__('point_cloud_node')
        self.declare_parameter('segmentation_mask', 252)  # Default value is integer
        self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().integer_value
        self.get_logger().info(f"segmentation_mask: {self.segmentation_mask}")


        # Publisher
        self.point_cloud_pub = self.create_publisher(PointCloud, 'pointcloud', 10)

        # Subscribers
        self.camera_subscriber = self.create_subscription(Camera, "/camera", self.camera_callback, 10)
        
        # CV Bridge
        self.bridge = CvBridge()

    def camera_callback(self, camera_msg
                 ):
        """
        Process synchronized messages to compute and publish a `PointCloud`.
        """
        try:
            # depth and segmentation images to numpy arrays
            depth_image = self.bridge.imgmsg_to_cv2(camera_msg.depth_image, desired_encoding="passthrough")
            segmentation_image = self.bridge.imgmsg_to_cv2(camera_msg.segmentation_map, desired_encoding="passthrough")
            self.get_logger().debug(f"Depth image shape: {depth_image.shape}, Segmentation image shape: {segmentation_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        try:

            depth_image = depth_image.astype(np.float32)

            height, width = depth_image.shape
            fx = width / (2.0 * np.tan(np.radians(75) / 2.0))  # For 60° FOV
            fy = height / (2.0 * np.tan(np.radians(75) / 2.0))  
            cx = width / 2.0
            cy = height / 2.0

            # 3D point cloud in the camera frame
            x_indices, y_indices = np.arange(0, width), np.arange(0, height)
            xx, yy = np.meshgrid(x_indices, y_indices)

            z = depth_image
            x = (xx - cx) * z / fx
            y = (yy - cy) * z / fy

            points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
            
            # segmentation mask
            unique_labels = np.unique(segmentation_image)
            self.get_logger().info(f"Unique labels in segmentation image: {unique_labels}")
            self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().integer_value
            mask = segmentation_image.flatten() == self.segmentation_mask
            points_camera = points_camera[mask]

            if points_camera.size == 0:
                self.get_logger().warn("No points passed the segmentation mask.")
                return

            # # points to the world frame
            # camera_position = np.array([
            #     camera_pose_msg.pose.position.x,
            #     camera_pose_msg.pose.position.y,
            #     camera_pose_msg.pose.position.z
            # ])
            # camera_orientation = R.from_quat([
            #     camera_pose_msg.pose.orientation.x,
            #     camera_pose_msg.pose.orientation.y,
            #     camera_pose_msg.pose.orientation.z,
            #     camera_pose_msg.pose.orientation.w
            # ])

            # T_camera_to_world = np.eye(4)
            # T_camera_to_world[:3, :3] = camera_orientation.as_matrix()
            # T_camera_to_world[:3, 3] = camera_position

            # points_camera_h = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))  # Homogeneous coordinates
            # points_world = (T_camera_to_world @ points_camera_h.T).T[:, :3]

            # self.get_logger().info(f"Point cloud shape before segmentation mask published by point_cloud_node: {points_camera.shape}")
            # `PointCloud` message
            cloud_msg = PointCloud()
            cloud_msg.header.stamp = camera_msg.header.stamp
            cloud_msg.header.frame_id = "camera_frame"

            for point in points_camera:
                point_msg = Point32(x=point[0], y=point[1], z=point[2])
                cloud_msg.points.append(point_msg)

            self.point_cloud_pub.publish(cloud_msg)
            self.get_logger().info(f"Point cloud shape after segmentation mask: {points_camera.shape}")

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FrankaSImPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
