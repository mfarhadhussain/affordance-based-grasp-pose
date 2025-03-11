#!/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from custom_interfaces.msg import Object  
import numpy as np
import torch
import sys

# Append path to load your model
sys.path.append('/home/mdfh/ros2_ws/src/franka_common/franka_common/')
from load_model import load_model

class ModelNode(Node):
    def __init__(self):
        super().__init__('model_node')

        self.declare_parameter('task_description', 'grasp')
        self.task_description = self.get_parameter('task_description').get_parameter_value().string_value
        self.get_logger().info(f"Task Description: {self.task_description}")
        
        # Subscriber
        self.create_subscription(PointCloud, 'pointcloud', self.process_point_cloud, 0)
        
        # Publisher 
        self.object_pub = self.create_publisher(Object, '/object/data', 0)
        
        self.model = load_model()
        self.GUIDE_W = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def process_point_cloud(self, msg):
        """
        Callback for processing incoming point cloud messages.
        It samples or filters the point cloud, runs model inference, and
        publishes a synchronized custom Object message.
        """
        try:
            points = np.array([[point.x, point.y, point.z] for point in msg.points])

            if points.shape[0] < 2048:
                self.get_logger().error("Not enough points in the point cloud. Skipping...")
                return
            
            # model input size is 2028
            if points.shape[0] > 2048:
                indices = np.random.choice(points.shape[0], 2048, replace=False)
                points = points[indices]

            point_cloud_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.task_description = self.get_parameter('task_description').get_parameter_value().string_value
            self.get_logger().info(f"Task Description: {self.task_description}")

            with torch.no_grad():
                model_output = self.model.detect_and_sample(
                    point_cloud_tensor,
                    self.task_description,
                    1000,
                    guide_w=self.GUIDE_W
                )
        
            affordance_label = model_output[0]  
            poses = model_output[1]             
            
            object_msg = Object()
            object_msg.header.stamp = self.get_clock().now().to_msg()
            object_msg.header.frame_id = "camera_frame"  
    
            object_msg.affordance_label = Float32MultiArray(data=affordance_label.flatten().tolist())
            object_msg.pose = Float32MultiArray(data=poses.flatten().tolist())
            
           
            for point in points:
                pt = Point32(x=point[0], y=point[1], z=point[2])
                object_msg.downsampled_pcd.points.append(pt)
            
        
            self.object_pub.publish(object_msg)
            self.get_logger().info("Published object message with affordance label and pose.")

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
