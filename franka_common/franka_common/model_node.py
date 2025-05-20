#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import pybullet as p

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from custom_interfaces.msg import Object

# Attempt to import extract_summary_vector and load_model from franka_common
try:
    from franka_common.meaningful_pose import extract_summary_vector
    from franka_common.load_model import load_model
except ImportError:
    model_path = os.path.expanduser('~/ros2_ws/src/franka_common/franka_common')
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from meaningful_pose import extract_summary_vector
    from load_model import load_model


class ModelNode(Node):
    def __init__(self):
        super().__init__('model_node')

        # Parameters
        self.declare_parameter('task_description', 'grasp to pour')
        self.declare_parameter('threshold', 0.05)
        self.add_on_set_parameters_callback(self._on_param_update)

        self.desc = self.get_parameter('task_description').get_parameter_value().string_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value

        # Point‐cloud size limits
        self.num_pts_max = 2048
        self.num_pts_min = 1024
        self.num_samples  = 1000
        self.guide_w      = 0.75

        # Pinned CPU buffer for zero‐copy transfer
        self.pin_buffer = torch.empty((self.num_pts_max, 3),
                                      dtype=torch.float32).pin_memory()

        # Load model
        self.model = load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.get_logger().info(f'Model running on: {self.device}')

        # QoS
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Subscriber / Publisher
        self.create_subscription(PointCloud, 'pointcloud', self.process_point_cloud, qos)
        self.object_pub = self.create_publisher(Object, 'object_aff_pose_points', qos)

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'task_description' and p.type_ == p.Type.STRING:
                self.desc = p.value
                self.get_logger().info(f'task_description updated to: {self.desc}')
            elif p.name == 'threshold' and p.type_ == p.Type.DOUBLE:
                self.threshold = p.value
                self.get_logger().info(f'threshold updated to: {self.threshold}')
        return SetParametersResult(successful=True)

    def process_point_cloud(self, msg: PointCloud):
        # --- 1) Build Nx3 numpy array
        pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.points], dtype=np.float32)
        n = pts.shape[0]
        if n < self.num_pts_min:
            return
        if n > self.num_pts_max:
            idx = np.random.choice(n, self.num_pts_max, replace=False)
            pts = pts[idx]
            n = self.num_pts_max

        # --- 2) Copy into pinned CPU buffer & non-blocking to GPU
        self.pin_buffer[:n].copy_(torch.from_numpy(pts))
        valid_pc = self.pin_buffer[:n] \
                        .unsqueeze(0)   \
                        .to(self.device, non_blocking=True)

        # --- 3) Center the point cloud
        centroid = valid_pc.mean(dim=1, keepdim=True)
        pc_centered = valid_pc - centroid
        self.get_logger().debug(f'pc_centered shape: {pc_centered.shape}')

        # --- 4) Inference
        with torch.no_grad():
            label_t, poses_t = self.model.detect_and_sample(
                pc_centered, self.desc, self.num_samples, guide_w=self.guide_w
            )

        # --- 5) Shift poses back to world frame
        if poses_t.ndim == 2 and poses_t.shape[1] >= 7:
            poses_t[:, 4:7] += centroid.squeeze(0).cpu().numpy()
        elif poses_t.ndim == 3 and poses_t.shape[2] >= 7:
            poses_t[:, :, 4:7] += centroid

        labels = (label_t.flatten() * 1.0).tolist()

        # --- 6) Filter good poses by distance threshold
        pts_cpu = valid_pc.squeeze(0).cpu().numpy()
        all_quatpos = poses_t.reshape(-1, 7)
        good_poses = []
        for qp in all_quatpos:
            quat, pos = qp[:4], qp[4:]
            # pybullet: (x,y,z,w)
            q_pb = np.roll(quat, -1)
            R = np.array(p.getMatrixFromQuaternion(q_pb)).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3] = R; T[:3, 3] = pos
            gripper_origin = T @ np.array([0, 0, 0.065, 1.0])
            dists = np.linalg.norm(pts_cpu - gripper_origin[:3], axis=1)
            if np.min(dists) <= self.threshold:
                good_poses.append(qp.tolist())

        # --- 7) Compute single summary target if any
        single_target = []
        if good_poses:
            try:
                summary = extract_summary_vector(np.array(good_poses),
                                                 use_geometric_median=True)
                single_target = summary.flatten().tolist()
            except Exception as e:
                self.get_logger().warning(f"Summary‐pose failed: {e}")

        # --- 8) Build and publish the Object message
        obj = Object()
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.header.frame_id = msg.header.frame_id
        obj.model_points.header = msg.header

        for x, y, z in pts:
            obj.model_points.points.append(Point32(x=float(x), y=float(y), z=float(z)))

        obj.affordance_label.data   = labels
        obj.pose.data               = sum(good_poses, [])
        obj.single_target_pose.data = single_target

        self.object_pub.publish(obj)
        self.get_logger().info(
            f"Published Object: {n} pts, {len(good_poses)} valid poses, "
            f"{'with' if single_target else 'without'} summary."
        )

        # --- 9) Cleanup to avoid fragmentation
        del valid_pc, label_t, poses_t
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def destroy_node(self):
        super().destroy_node()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    rclpy.init()
    node = ModelNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

