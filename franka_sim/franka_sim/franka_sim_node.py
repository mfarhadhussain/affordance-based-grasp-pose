#!/bin/env python3
"""ROS2 node: publishes PyBullet camera images and robot state with live display (optional)."""
import sys
import numpy as np
import pybullet as p
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from custom_interfaces.msg import Camera, Robot
import cv2

# Ensure local PybulletFranka is importable
sys.path.append('/home/mdfh/ros2_ws/src/franka_sim/franka_sim/')
from pybullet_franka import PybulletFranka

class PybulletCamera(object):
    def __init__(self, camera_offset=np.array([-0.05, -0.01, 0.05], np.float32),
                 camera_distance=0.75, width=720, height=720,
                 fov=75, near=0.01, far=100):
        self.camera_offset = camera_offset
        self.camera_distance = camera_distance
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        # Precompute projection
        aspect = width / height
        self.proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        # Pre-allocate buffers for getCameraImage
        # Create marker once
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0,1,0,1])
        self.marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)

    def _capture(self, view):
        _, _, rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            view, self.proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        return rgb[:,:,:3], depth, (seg.astype(np.uint8) * 255)

    def add_visual_markers(self, cam_pos, cam_target):
        p.addUserDebugLine(cam_pos, cam_target, [1,0,0], 2, 0.1)
        p.addUserDebugText('Camera', cam_pos, [0.5,1,0.5], 1.2, 0.1)

    def get_camera_image_end_effector(self, ee_pos, ee_orn):
        off = p.getQuaternionFromEuler([0,-np.pi/2,0])
        cam_orn = p.multiplyTransforms([0]*3, ee_orn, [0]*3, off)[1]
        Rm = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3,3)
        cam_pos = ee_pos + Rm.dot(self.camera_offset)
        forward = Rm[:,0]
        cam_target = cam_pos + self.camera_distance * forward
        view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), Rm[:,2].tolist())
        rgb, depth, seg = self._capture(view)
        p.resetBasePositionAndOrientation(self.marker, cam_pos.tolist(), cam_orn)
        self.add_visual_markers(cam_pos.tolist(), cam_target.tolist())
        return rgb, depth, seg

    def get_camera_image_fixed(self):
        cam_pos = [0.0001,0.0001,0.60]
        target = [0,0,0]
        view = p.computeViewMatrix(cam_pos, target, [0,0,1])
        rgb, depth, seg = self._capture(view)
        p.resetBasePositionAndOrientation(self.marker, cam_pos, [0,0,0,1])
        self.add_visual_markers(cam_pos, target)
        return rgb, depth, seg

class FrankaSimCamera(Node):
    def __init__(self):
        super().__init__('franka_sim_camera')
        self.declare_parameter('end_effector', True)
        self.add_on_set_parameters_callback(self._on_param_update)
        self.end_effector = self.get_parameter('end_effector').value
        self.cam_pub = self.create_publisher(Camera, 'camera', 10)
        self.robot_pub = self.create_publisher(Robot, 'robot', 10)
        self.bridge = CvBridge()
        self.cam_msg = Camera()
        self.robot_msg = Robot()
        self.robot = PybulletFranka()
        self.camera = PybulletCamera()
        try:
            cv2.namedWindow('EndEffectorCamera', cv2.WINDOW_NORMAL)
            self.show = True
        except cv2.error as e:
            self.get_logger().warn(f'OpenCV GUI unavailable, disabling display: {e}')
            self.show = False
        self.create_timer(0.1, self._timer)
        self.get_logger().info('FrankaSimCamera node started.')

    def _on_param_update(self, params):
        for p in params:
            if p.name=='end_effector' and p.type_==p.Type.BOOL:
                self.end_effector = p.value
        return SetParametersResult(successful=True)

    def _timer(self):
        self.robot.update_scene()
        self._publish_robot()
        self._publish_cam()

    def _publish_robot(self):
        ee = p.getLinkState(self.robot.robot_id, self.robot.end_effector_index)
        now = self.get_clock().now().to_msg()
        hdr = self.robot_msg.end_effector_pose.header
        hdr.stamp=now; hdr.frame_id='end_effector'
        pos,orn=ee[0],ee[1]
        pe=self.robot_msg.end_effector_pose.pose
        pe.position.x,pe.position.y,pe.position.z=pos
        pe.orientation.x,pe.orientation.y,pe.orientation.z,pe.orientation.w=orn
        hdrb=self.robot_msg.robot_base_pose.header
        hdrb.stamp=now; hdrb.frame_id='base'
        bp,bo=self.robot.robot_base_position,self.robot.robot_base_orientation
        pb=self.robot_msg.robot_base_pose.pose
        pb.position.x,pb.position.y,pb.position.z=bp
        pb.orientation.x,pb.orientation.y,pb.orientation.z,pb.orientation.w=bo
        js=self.robot_msg.joint_state
        js.header.stamp=now; js.header.frame_id='joints'
        pl,vl,el=self.robot.get_robot_joints_state()
        if isinstance(el[0],(list,tuple)): el=[e for sub in el for e in sub]
        js.position,js.velocity,js.effort=pl,vl,el
        self.robot_pub.publish(self.robot_msg)

    def _publish_cam(self):
        ee=p.getLinkState(self.robot.robot_id,self.robot.end_effector_index)
        if self.end_effector:
            rgb,depth,seg=self.camera.get_camera_image_end_effector(
                np.array(ee[0],np.float32),ee[1])
        else:
            rgb,depth,seg=self.camera.get_camera_image_fixed()
        if self.show:
            cv2.imshow('EndEffectorCamera',rgb); cv2.waitKey(1)
        try:
            self.cam_msg.rgb_image=self.bridge.cv2_to_imgmsg(rgb,'rgb8')
            self.cam_msg.depth_image=self.bridge.cv2_to_imgmsg(depth,'32FC1')
            self.cam_msg.segmentation_map=self.bridge.cv2_to_imgmsg(seg,'mono8')
        except Exception as e:
            self.get_logger().error(f'Error converting images: {e}'); return
        h=self.cam_msg.header; h.stamp=self.get_clock().now().to_msg();h.frame_id='camera'
        self.cam_pub.publish(self.cam_msg)

    def destroy_node(self):
        super().destroy_node()
        if self.show: cv2.destroyAllWindows()


def main():
    rclpy.init()
    node=FrankaSimCamera()
    try: rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown(); p.disconnect()

if __name__=='__main__': main()

# #!/bin/env python3
# import pybullet as p
# import rclpy
# from rclpy.node import Node
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# import math
# import sys
# sys.path.append('/home/mdfh/ros2_ws/src/franka_sim/franka_sim/')
# from pybullet_franka import PybulletFranka
# from custom_interfaces.msg import Camera, Robot

# #---------------------------------------------------------------------
# # PybulletCamera class to compute camera images.
# # 
# # 1. get_camera_image_end_effector: camera view based on the robot's
# #    end-effector pose.
# # 2. get_camera_image_fixed: a fixed view (e.g., looking down at the origin
# #    from a height of 0.75).
# #---------------------------------------------------------------------
# class PybulletCamera(object):
#     def __init__(self, camera_offset=np.array([-0.05, -0.01, 0.05]), camera_distance=0.75,
#                  width=720, height=720, fov=75, near=0.01, far=100):
#         self.camera_offset = camera_offset
#         self.camera_distance = camera_distance
#         self.width = width
#         self.height = height
#         self.fov = fov
#         self.near = near
#         self.far = far

#         # Create a small visible sphere to mark the camera position.
#         self.camera_visual_shape_id = p.createVisualShape(
#             shapeType=p.GEOM_SPHERE,
#             radius=0.01,
#             rgbaColor=[0, 1, 0, 1]
#         )
#         self.camera_body_id = p.createMultiBody(
#             baseMass=0,
#             baseVisualShapeIndex=self.camera_visual_shape_id,
#             basePosition=[0, 0, 0]
#         )

#     def add_visual_markers(self, camera_pos, camera_target):
#         p.addUserDebugLine(camera_pos.tolist(), camera_target.tolist(),
#                            lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0.1)
#         p.addUserDebugText("Camera", camera_pos.tolist(),
#                            textColorRGB=[0.5, 1, 0.5], lifeTime=0.1, textSize=1.2)

#     def get_camera_image_end_effector(self, ee_pos, ee_orn):
#         # offset rotation of -90° about the y-axis.
#         offset_quat = p.getQuaternionFromEuler([0, -np.pi/2, 0])
#         # Combine the end effector's orientation with the offset.
#         camera_orn = p.multiplyTransforms([0, 0, 0], ee_orn,
#                                           [0, 0, 0], offset_quat)[1]
#         rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
#         camera_pos = ee_pos + rot_matrix.dot(self.camera_offset)
#         forward_vec = rot_matrix.dot(np.array([1, 0, 0]))
#         camera_target = camera_pos + self.camera_distance * forward_vec
#         up_vec = rot_matrix.dot(np.array([0, 0, 1]))

#         view_matrix = p.computeViewMatrix(
#             cameraEyePosition=camera_pos.tolist(),
#             cameraTargetPosition=camera_target.tolist(),
#             cameraUpVector=up_vec.tolist()
#         )
#         aspect = self.width / self.height
#         projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
#         width, height, rgb, depth, seg = p.getCameraImage(
#             width=self.width,
#             height=self.height,
#             viewMatrix=view_matrix,
#             projectionMatrix=projection_matrix
#         )
 
#         depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
#         rgb = rgb[:, :, :3]
#         seg = (seg * 255).astype(np.uint8)

#         p.resetBasePositionAndOrientation(self.camera_body_id, camera_pos.tolist(), camera_orn)
#         self.add_visual_markers(camera_pos, camera_target)
#         return width, height, rgb, depth, seg

#     def get_camera_image_fixed(self):
#         camera_pos = np.array([0.0001, 0.0001, 0.60])
#         camera_target = np.array([0.0, 0.0, 0.0])
#         up_vec = np.array([0.0, 0.0, 1.0])
#         view_matrix = p.computeViewMatrix(
#             cameraEyePosition=camera_pos.tolist(),
#             cameraTargetPosition=camera_target.tolist(),
#             cameraUpVector=up_vec.tolist()
#         )
#         aspect = self.width / self.height
#         projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
#         width, height, rgb, depth, seg = p.getCameraImage(
#             width=self.width,
#             height=self.height,
#             viewMatrix=view_matrix,
#             projectionMatrix=projection_matrix
#         )
#         depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
#         rgb = rgb[:, :, :3]
#         seg = (seg * 255).astype(np.uint8)

#         p.resetBasePositionAndOrientation(self.camera_body_id, camera_pos.tolist(), [0, 0, 0, 1])
#         self.add_visual_markers(camera_pos, camera_target)
#         return width, height, rgb, depth, seg

# #---------------------------------------------------------------------
# # Combined node: publishes both robot state and camera images.
# # The camera mode is selected via the ROS parameter "end_effector".
# #---------------------------------------------------------------------
# class FrankaSimCamera(Node):
#     def __init__(self):
#         super().__init__("franka_sim_camera")
        
#         # parameter for camera mode.
#         self.declare_parameter('end_effector', False)
#         self.end_effector_mode = self.get_parameter('end_effector').value
        
#         # publishers 
#         self.camera_publisher = self.create_publisher(Camera, "camera", 10)
#         self.robot_state_pose_publisher = self.create_publisher(Robot, "robot", 10)
        
#         self.bridge = CvBridge()
#         timer_period = 0.1  # seconds 
#         self.timer = self.create_timer(timer_period, self.timer_callback)
        
#         self.robot = PybulletFranka()
#         self.custom_camera = PybulletCamera()
        
#         self.get_logger().info("FrankaSimCamera node started.")

#     def timer_callback(self):
#         self.robot.update_scene()
#         self.publish_robot_state()
#         self.publish_camera_image()

#     def publish_robot_state(self):
#         end_effector_state = p.getLinkState(self.robot.robot_id, self.robot.end_effector_index)
#         robot_msg = Robot()
#         current_time = self.get_clock().now().to_msg()
        
#         #end effector pose.
#         robot_msg.end_effector_pose.header.stamp = current_time
#         robot_msg.end_effector_pose.header.frame_id = "end_effector"
#         robot_msg.end_effector_pose.pose.position.x = end_effector_state[0][0]
#         robot_msg.end_effector_pose.pose.position.y = end_effector_state[0][1]
#         robot_msg.end_effector_pose.pose.position.z = end_effector_state[0][2]
#         robot_msg.end_effector_pose.pose.orientation.x = end_effector_state[1][0]
#         robot_msg.end_effector_pose.pose.orientation.y = end_effector_state[1][1]
#         robot_msg.end_effector_pose.pose.orientation.z = end_effector_state[1][2]
#         robot_msg.end_effector_pose.pose.orientation.w = end_effector_state[1][3]
        
#         # robot base pose.
#         robot_msg.robot_base_pose.header.stamp = current_time
#         robot_msg.robot_base_pose.header.frame_id = "base"
#         robot_msg.robot_base_pose.pose.position.x = self.robot.robot_base_position[0]
#         robot_msg.robot_base_pose.pose.position.y = self.robot.robot_base_position[1]
#         robot_msg.robot_base_pose.pose.position.z = self.robot.robot_base_position[2]
#         robot_msg.robot_base_pose.pose.orientation.x = self.robot.robot_base_orientation[0]
#         robot_msg.robot_base_pose.pose.orientation.y = self.robot.robot_base_orientation[1]
#         robot_msg.robot_base_pose.pose.orientation.z = self.robot.robot_base_orientation[2]
#         robot_msg.robot_base_pose.pose.orientation.w = self.robot.robot_base_orientation[3]
        
#         # joint states.
#         positions, velocities, torques = self.robot.get_robot_joints_state()
#         if isinstance(torques[0], (tuple, list)):
#             torques = [t for torque_tuple in torques for t in torque_tuple]
            
#         robot_msg.joint_state.header.stamp = current_time
#         robot_msg.joint_state.header.frame_id = "joints"
#         robot_msg.joint_state.position = positions
#         robot_msg.joint_state.velocity = velocities
#         robot_msg.joint_state.effort = torques
        
#         self.robot_state_pose_publisher.publish(robot_msg)

#     def publish_camera_image(self):
#         self.end_effector_mode = self.get_parameter('end_effector').value
        
#         if self.end_effector_mode:
            
#             end_effector_state = p.getLinkState(self.robot.robot_id, self.robot.end_effector_index)
#             ee_pos = np.array(end_effector_state[0])
#             ee_orn = end_effector_state[1]
#             width, height, rgb, depth, seg = self.custom_camera.get_camera_image_end_effector(ee_pos, ee_orn)
#         else:
#             width, height, rgb, depth, seg = self.custom_camera.get_camera_image_fixed()
        
#         try:
#             rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="8UC3")
#             depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
#             seg_msg = self.bridge.cv2_to_imgmsg(seg, encoding="8UC1")
#         except Exception as e:
#             self.get_logger().error(f"Error converting images: {e}")
#             return
        
#         camera_msg = Camera()
#         current_time = self.get_clock().now().to_msg()
#         camera_msg.header.stamp = current_time
#         camera_msg.header.frame_id = "camera"
#         camera_msg.rgb_image = rgb_msg
#         camera_msg.depth_image = depth_msg
#         camera_msg.segmentation_map = seg_msg
#         camera_msg.depth_scale = 0.0  
        
#         self.camera_publisher.publish(camera_msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = FrankaSimCamera()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         p.disconnect()

# if __name__=="__main__":
#     main()
