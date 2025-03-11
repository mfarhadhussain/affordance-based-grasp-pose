#!/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Float32MultiArray 
import numpy as np 
import trimesh
import pybullet as p
import sys

from custom_interfaces.msg import Object  
# Append the deep-learning model path
dl_model_path = '/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds'
sys.path.append(dl_model_path)
from utils.visualization import create_gripper_marker

class AffordancePosePlot(Node): 
    def __init__(self): 
        super().__init__('affordance_pose_plot_node')
        
        # Declare a ROS parameter for the threshold with a default value of 0.03
        self.declare_parameter("threshold", 0.03)  # 0.03 is the official git repo (3DAPNet)
        self.threshold = self.get_parameter("threshold").value

        # Subscribe to the custom object message
        self.create_subscription(Object, '/object/data', self.callback, 0)
        
        # Color codes for visualization
        self.color_code_1 = np.array([0, 0, 255])  # Affordance region color (blue)
        self.color_code_2 = np.array([0, 255, 0])  # Gripper pose color (green)
        

    def callback(self, obj_msg):
        """
        Callback for the custom Object message.
        Extracts the downsampled point cloud, affordance label, and pose data
        from the message and sends them for visualization.
        """
        
        pcd_msg = obj_msg.downsampled_pcd
        affordance_msg = obj_msg.affordance_label
        pose_msg = obj_msg.pose
        
        pcd = np.array([[point.x, point.y, point.z] for point in pcd_msg.points])
        affordance_label = np.array(affordance_msg.data)
        poses = np.array(pose_msg.data).reshape(-1, 7)
        
        self.affordance_pose_plotter(poses, pcd, affordance_label)
        
    def affordance_pose_plotter(self, poses, pcd, affordance_label):
        """
        Visualizes point cloud with affordance regions and gripper poses.

        Parameters:
        - poses: Array of poses [N, 7] (quaternion + position)
        - pcd: Point cloud as a NumPy array (Nx3)
        - affordance_label: Affordance values for each point (Nx1)
        """
        colors = affordance_label[:, None] * self.color_code_1
        point_cloud = trimesh.points.PointCloud(pcd, colors=colors)

        scene = trimesh.Scene()
        scene.add_geometry(point_cloud)

        for pose in poses:
            orientation = pose[:4]  # [qw, qx, qy, qz]
            position = pose[4:]     # [x, y, z]
        
            # Get rotation matrix from quaternion using pybullet
            r_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            T = np.eye(4)  
            T[:3, :3] = r_matrix  
            T[:3, 3] = position
        
            # Calculate gripper position
            gripper_Position = (T @ np.array([0., 0., 6.59999996e-02, 1.]))[:3]
            self.threshold = self.get_parameter("threshold").value
            threshold = self.threshold
        
            distance = np.linalg.norm(point_cloud.vertices - gripper_Position, axis=1)
            
            if np.any(distance <= threshold):
                gripper_marker = create_gripper_marker(color=self.color_code_2)
                gripper_marker.apply_transform(T)  
                scene.add_geometry(gripper_marker)
        
        scene.show(block=False)
        
def main(args=None):
    rclpy.init(args=args)
    node = AffordancePosePlot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
