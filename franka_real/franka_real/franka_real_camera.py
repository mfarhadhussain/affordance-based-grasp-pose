#!/bin/env python3
import rclpy
from rclpy.node import Node
import threading
import time
import cv2
import numpy as np
import traceback

from sensor_msgs.msg import Image, PointCloud
from std_msgs.msg import Header
from custom_interfaces.msg import Camera 
from geometry_msgs.msg import Point32 

import pyrealsense2 as rs
from cv_bridge import CvBridge

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Import Open3D for point cloud visualization
import open3d as o3d


# -----------------------------------------------------------------------------
# RealSense Pipeline
# -----------------------------------------------------------------------------
class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print("[SharedRealSense] Failed to start pipeline:", e)
            raise
        self.align = rs.align(rs.stream.color)
        self.latest_frames = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if frames:
                    aligned_frames = self.align.process(frames)
                    with self.lock:
                        self.latest_frames = aligned_frames
            except Exception as e:
                print("[RealSense] Exception in _update:", e)
                traceback.print_exc()
            time.sleep(0.01)

    def get_frames(self):
        with self.lock:
            return self.latest_frames

    def stop(self):
        self.running = False
        try:
            self.pipeline.stop()
        except Exception as e:
            print("[RealSense] Exception during stop:", e)


realsense = RealSense()

# -----------------------------------------------------------------------------
# Realsense Publisher Node
# -----------------------------------------------------------------------------
class RealsensePublisher(Node):
    def __init__(self): 
        super().__init__('realsense_publisher')
        
        self.declare_parameter('segmentation_mask', 1)  # Default value is integer
        self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().integer_value
        self.get_logger().info(f"segmentation_mask: {self.segmentation_mask}")
        
        # Detectron2 configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold for segmentation
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        
        # Publishers for Camera images and PointCloud
        self.realsense_publisher_ = self.create_publisher(Camera, 'camera', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud, 'pointcloud', 10)
        
        # Timer callback for processing frames
        self.timer = self.create_timer(0.1, self.timer_callback)
        # cv_bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        # Create Realsense pointcloud object
        self.pc = rs.pointcloud()

        try:
            profile = realsense.pipeline.get_active_profile()
            sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = sensor.get_depth_scale() if sensor else 1.0
            self.get_logger().info(f"[INFO] Depth Scale: {self.depth_scale}")
        except Exception as e:
            self.get_logger().error(f"[ERROR] Could not get depth scale: {e}")
            self.depth_scale = 1.0
        # Debug flag to show image windows and point cloud visualizer
        self.declare_parameter('debug', 1)  
        self.debug = self.get_parameter('debug').get_parameter_value().integer_value
        self.get_logger().info(f"debug: {self.debug}")
        
    def depth2PointCloud(self, depth, rgb, depth_scale, clip_distance_max):
    
        intrinsics = depth.profile.as_video_stream_profile().intrinsics
        depth = np.asanyarray(depth.get_data()) * depth_scale # 1000 mm => 0.001 meters
        rgb = np.asanyarray(rgb.get_data())
        rows,cols  = depth.shape

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        r = r.astype(float)
        c = c.astype(float)

        # valid = (depth > 0) & (depth < clip_distance_max) #remove from the depth image all values above a given value (meters).
        # valid = np.ravel(valid)
        z = depth 
        x =  z * (c - intrinsics.ppx) / intrinsics.fx
        y =  z * (r - intrinsics.ppy) / intrinsics.fy
    
        z = np.ravel(z)
        x = np.ravel(x)
        y = np.ravel(y)
        
        r = np.ravel(rgb[:,:,0])
        g = np.ravel(rgb[:,:,1])
        b = np.ravel(rgb[:,:,2])
        
        pointsxyzrgb = np.dstack((x, y, z, r, g, b))
        pointsxyzrgb = pointsxyzrgb.reshape(-1,6)

        return pointsxyzrgb
            
    def debug_show_images(self, color_image, depth_image, seg_map):
        """
        Display RGB, depth, and segmentation images.
        """
        cv2.imshow("RGB Image", color_image)
        depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)
        cv2.imshow("Depth Image", depth_display)
        seg_display = cv2.applyColorMap(cv2.convertScaleAbs(seg_map, alpha=10), cv2.COLORMAP_JET)
        cv2.imshow("Segmentation Map", seg_display)
        cv2.waitKey(1)

    def timer_callback(self):
        frames = realsense.get_frames()
        if frames is None:
            return

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # --- Detectron2 segmentation ---
        outputs = self.predictor(color_image)
        seg_map = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        if "instances" in outputs and outputs["instances"].has("pred_masks"):
            masks = outputs["instances"].pred_masks.cpu().numpy()  # shape: (N, H, W)
            for idx, mask in enumerate(masks):
                seg_map[mask] = idx + 1 
                
        self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().integer_value
        self.get_logger().info(f"segmentation_mask: {self.segmentation_mask}")
        
        unique_labels = np.unique(seg_map.flatten())
        self.get_logger().info(f"Unique labels in segmentation image: {unique_labels}")
        mask = seg_map.flatten() == self.segmentation_mask
        
        self.debug = self.get_parameter('debug').get_parameter_value().integer_value
        # self.get_logger().info(f"debug: {self.debug}")
        if self.debug==1:
            self.debug_show_images(color_image, depth_image, seg_map)
        
        # --- Publish Camera Data ---
        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
            seg_msg = self.bridge.cv2_to_imgmsg(seg_map, encoding="mono8")

            msg = Camera()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "Realsense Camera"
            msg.rgb_image = rgb_msg
            msg.depth_image = depth_msg
            msg.segmentation_map = seg_msg
            msg.depth_scale = float(self.depth_scale)
            self.realsense_publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error("Failed to publish RealsenseData: " + str(e))
            traceback.print_exc()

        # --- Compute and Publish Point Cloud ---
        try:
            pointsxyzrgb = self.depth2PointCloud(depth_frame, color_frame, self.depth_scale, clip_distance_max=3)
            points_list = np.array(pointsxyzrgb[:, 0:3]) 
            
            points_list = points_list[mask]
            
            point_cloud = PointCloud()
            point_cloud.header.stamp = self.get_clock().now().to_msg()
            point_cloud.header.frame_id = "Realsense Camera"
            
            for point in points_list:
                point_msg = Point32(x=point[0], y=point[1], z=point[2])
                point_cloud.points.append(point_msg)

            self.pointcloud_publisher.publish(point_cloud)

        except Exception as e:
            self.get_logger().error("Failed to publish PointCloud: " + str(e))
            traceback.print_exc()

    def destroy_node(self):
        # Optionally close the Open3D visualizer when shutting down
        if self.pc_vis is not None:
            self.pc_vis.destroy_window()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealsensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        realsense.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
