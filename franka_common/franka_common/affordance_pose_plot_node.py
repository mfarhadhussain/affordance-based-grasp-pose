#!/usr/bin/env python3
"""
CPU-only Affordance-Pose Plot Node for ROS2
Visualizes affordance regions and gripper poses filtered by proximity.
"""
import os
# ─── FORCE CPU ONLY ───────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import numpy as np
import trimesh
import pybullet as p

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from custom_interfaces.msg import Object

# ─── INLINE CPU-ONLY GRIPPER MARKER ────────────────────────────────────────────────
def create_gripper_marker(color=[0, 255, 0], tube_radius=0.002, sections=6):
    """
    Create a 3D mesh visualizing a parallel yaw gripper.
    Four cylinders concatenated and colored.
    """
    # Finger left
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[0.041, 0.0, 0.066], [0.041, 0.0, 0.112]]
    )
    # Finger right
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-0.041, 0.0, 0.066], [-0.041, 0.0, 0.112]]
    )
    # Base back
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[0, 0, 0], [0, 0, 0.066]]
    )
    # Base front
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-0.041, 0, 0.066], [0.041, 0, 0.066]]
    )

    gripper = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    gripper.visual.face_colors = color
    return gripper

# ─── IMPORT SUMMARY‐VECTOR ONLY ───────────────────────────────────────────────────
try:
    from franka_common.meaningful_pose import extract_summary_vector
except ImportError:
    # fallback if installed differently
    model_path = os.path.expanduser('~/ros2_ws/src/franka_common/franka_common')
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from meaningful_pose import extract_summary_vector


class AffordancePosePlot(Node):
    def __init__(self):
        super().__init__('affordance_pose_plot_node')
        self.get_logger().info('Initializing AffordancePosePlotNode (CPU only)...')

        # Dynamic threshold parameter
        self.threshold = self.declare_parameter('threshold', 0.05).value
        self.add_on_set_parameters_callback(self._on_param_update)

        # Pre-create a reusable marker mesh
        self.gripper_marker = create_gripper_marker(color=[255, 165, 0])

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Object, 'object_aff_pose_points',
                                 self._object_callback, qos)

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'threshold' and p.type_ == p.Type.DOUBLE:
                self.threshold = p.value
                self.get_logger().info(f'Updated threshold: {self.threshold}')
        return SetParametersResult(successful=True)

    def _object_callback(self, msg: Object):
        # Point cloud as Nx3
        pts = np.array([[pt.x, pt.y, pt.z]
                        for pt in msg.model_points.points],
                       dtype=np.float32)
        labels = np.array(msg.affordance_label.data, dtype=np.float32)

        # All candidate poses [M×7]
        poses = np.array(msg.pose.data, dtype=np.float32).reshape(-1, 7)
        # Optional single-target summary [1×7]
        st = np.array(msg.single_target_pose.data, dtype=np.float32)
        single_target = st.reshape(-1, 7)[0] if st.size else None

        self._plot_aff_pose(pts, labels, poses, single_target)

    def _plot_aff_pose(self, pcd, labels, poses, single_target_pose=None):
        # Color by affordance (blue intensity)
        colors = (labels[:, None] * np.array([0, 0, 255], dtype=np.uint8))
        cloud = trimesh.points.PointCloud(vertices=pcd, colors=colors)
        scene = trimesh.Scene([cloud])

        # Draw each candidate gripper pose
        for quat_xyzw, pos_xyz in zip(poses[:, :4], poses[:, 4:]):
            # pybullet expects [x,y,z,w]
            q = np.roll(quat_xyzw, -1)
            R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, pos_xyz
            marker = self.gripper_marker.copy()
            marker.apply_transform(T)
            # scene.add_geometry(marker)

        # Highlight the single-target pose (orange)
        if single_target_pose is not None:
            quat_xyzw, pos_xyz = single_target_pose[:4], single_target_pose[4:]
            q = np.roll(quat_xyzw, -1)
            R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, pos_xyz
            highlight = create_gripper_marker(color=[0, 255, 0])
            highlight.apply_transform(T)
            scene.add_geometry(highlight)

        # Show non-blocking
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


if __name__ == '__main__':
    main()









# #!/usr/bin/env python3
# """
# Optimized Affordance-Pose Plot Node for ROS2
# Visualizes affordance regions and gripper poses filtered by proximity.
# """
# import os
# import sys
# from pathlib import Path

# DL_MODEL_DIR = Path.home() / 'open_vocab_ws' / 'src' / 'dl_model' / 'Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds'
# if str(DL_MODEL_DIR) not in sys.path:
#     sys.path.insert(0, str(DL_MODEL_DIR))

# try:
#     import rclpy
#     from rclpy.node import Node
#     from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
#     from custom_interfaces.msg import Object
# except ImportError as e:
#     sys.stderr.write(f"[ERROR] ROS2 imports failed: {e}\n")
#     sys.exit(1)

# import numpy as np
# import trimesh
# import pybullet as p
# from utils.visualization import create_gripper_marker


# try:
#     from franka_common.meaningful_pose import extract_summary_vector
# except ImportError:
#     import sys
#     model_path = os.path.expanduser('~/ros2_ws/src/franka_common/franka_common')
#     if model_path not in sys.path:
#         sys.path.insert(0, model_path)
#     from meaningful_pose import extract_summary_vector


# class AffordancePosePlot(Node):
#     def __init__(self):
#         super().__init__('affordance_pose_plot_node')
#         self.get_logger().info('Initializing AffordancePosePlotNode...')

#         # Declare and cache dynamic threshold parameter
#         self.threshold = self.declare_parameter('threshold', 0.05).value
#         self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
#         self.add_on_set_parameters_callback(self._on_param_update)

#         # Pre-allocate reusable marker
#         self.gripper_marker = create_gripper_marker(color=[0, 255, 0])

#         # QoS
#         qos = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
#             reliability=QoSReliabilityPolicy.RELIABLE
#         )

#         # Subscription to Object messages
#         self.create_subscription(
#             Object,
#             'object_aff_pose_points',
#             self._object_callback,
#             qos
#         )

#     def _on_param_update(self, params):
#         for p in params:
#             if p.name == 'threshold' and p.type_ == p.Type.DOUBLE:
#                 self.threshold = p.value
#                 self.get_logger().info(f'Updated threshold: {self.threshold}')
#         return rclpy.parameter.SetParametersResult(successful=True)

#     def _object_callback(self, msg: Object):
#         # Convert downsampled cloud to Nx3 array
#         pts = np.array([[pt.x, pt.y, pt.z] for pt in msg.model_points.points], dtype=np.float32)
#         labels = np.array(msg.affordance_label.data, dtype=np.float32)

#         # Stack poses into [N,7] orientation first four and remaining translation 
#         poses = np.array(msg.pose.data, dtype=np.float32).reshape(-1, 7)
#         single_target_pose = np.array(msg.single_target_pose.data, dtype=np.float32).reshape(-1, 7)
#         self._plot_aff_pose(pts, labels, poses, single_target_pose=single_target_pose[0] if single_target_pose.size else None)
#         # self._plot(pts, labels, poses) 
        
#     def _plot_aff_pose(self, pcd: np.ndarray, labels: np.ndarray, poses: np.ndarray, single_target_pose=None):
#         # Color points by affordance
#         colors = (labels[:, None] * np.array([0, 0, 255], dtype=np.uint8))
#         cloud = trimesh.points.PointCloud(vertices=pcd, colors=colors)
#         scene = trimesh.Scene([cloud])
        
#         # Process each gripper pose
#         for quat_xyzw, pos_xyz in zip(poses[:, :4], poses[:, 4:]):
#             # pybullet expects [x, y, z, w], convert if needed
#             q = np.roll(quat_xyzw, -1) if quat_xyzw.shape[0] == 4 else quat_xyzw
#             R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
#             T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos_xyz
#             marker = self.gripper_marker.copy()
#             marker.apply_transform(T)
#             # scene.add_geometry(marker) 
            
#         if single_target_pose is not None:
#             quat_xyzw, pos_xyz = single_target_pose[:4], single_target_pose[4:]
#             q = np.roll(quat_xyzw, -1)  # convert to [x, y, z, w]
#             R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
#             T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos_xyz

#             # Orange gripper marker
#             informative_marker = create_gripper_marker(color=[0, 255, 0])
#             informative_marker.apply_transform(T)
#             scene.add_geometry(informative_marker)
#         # Non-blocking show for interactive use
#         scene.show(block=False)    
        

#     # def _plot(self, pcd: np.ndarray, labels: np.ndarray, poses: np.ndarray):
#     #     # Color points by affordance
#     #     colors = (labels[:, None] * np.array([0, 0, 255], dtype=np.uint8))
#     #     cloud = trimesh.points.PointCloud(vertices=pcd, colors=colors)
#     #     scene = trimesh.Scene([cloud])
        
#     #     valid_poses = []
#     #     # Process each gripper pose
#     #     for quat_xyzw, pos_xyz in zip(poses[:, :4], poses[:, 4:]):
#     #         # pybullet expects [x, y, z, w], convert if needed
#     #         q = np.roll(quat_xyzw, -1) if quat_xyzw.shape[0] == 4 else quat_xyzw
#     #         R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
#     #         T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos_xyz

#     #         # gripper origin offset (z offset in world)
#     #         gripper_pt = T @ np.array([0, 0, 0.065, 1.0])
#     #         distances = np.linalg.norm(pcd - gripper_pt[:3], axis=1)

#     #         if np.any(distances <= self.threshold):
#     #             valid_poses.append(np.concatenate([quat_xyzw, pos_xyz]))
#     #             marker = self.gripper_marker.copy()
#     #             marker.apply_transform(T)
#     #             scene.add_geometry(marker)  
                
#     #     if len(valid_poses) > 0:
#     #         valid_poses_np = np.array(valid_poses)
#     #         try:
#     #             summary_pose = extract_summary_vector(valid_poses_np, use_geometric_median=True)
#     #             quat_xyzw = summary_pose[:4]
#     #             pos_xyz = summary_pose[4:]
#     #             q = np.roll(quat_xyzw, -1)  # convert to [x, y, z, w]
#     #             R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
#     #             T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos_xyz

#     #             # Orange gripper marker
#     #             informative_marker = create_gripper_marker(color=[255, 165, 0])
#     #             informative_marker.apply_transform(T)
#     #             scene.add_geometry(informative_marker)
#     #         except Exception as e:
#     #             self.get_logger().warn(f"Failed to compute or visualize informative pose: {e}")
#     #     else:
#     #         self.get_logger().info("No valid poses within threshold. Skipping informative pose.")    

#     #     # Non-blocking show for interactive use
#     #     scene.show(block=False)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AffordancePosePlot()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
