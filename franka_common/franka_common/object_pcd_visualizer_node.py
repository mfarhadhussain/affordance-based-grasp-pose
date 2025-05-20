#!/usr/bin/env python3
"""
Optimized Object PointCloud Visualizer (PointCloud msg) for ROS2
- Uses numpy.fromiter to avoid intermediate Python lists
- Downsamples to MAX_POINTS for smooth rendering
- Preallocates a matplotlib scatter for efficient updates
"""
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Maximum points to display for performance
MAX_POINTS = 2048

class ObjectPCDVisualizer(Node):
    def __init__(self):
        super().__init__('object_pcd_visualizer')
        self.get_logger().info('Initializing ObjectPCDVisualizer...')

        # QoS: best-effort for fast streaming
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(
            PointCloud,
            'pointcloud',
            self._on_pointcloud,
            qos
        )

        # Matplotlib setup
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = self.ax.scatter([], [], [], s=2, c='r')
        self.ax.set_title('3D Point Cloud')
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')

    def _on_pointcloud(self, msg: PointCloud):
        n_pts = len(msg.points)
        if n_pts == 0:
            self.get_logger().warn('Received empty point cloud')
            return

        # Create a flat iterator of floats: x,y,z for each point
        count = 3 * n_pts
        data = np.fromiter(
            (coord for pt in msg.points for coord in (pt.x, pt.y, pt.z)),
            dtype=np.float32,
            count=count
        )
        pts = data.reshape(n_pts, 3)

        # Downsample if too many points
        if n_pts > MAX_POINTS:
            idx = np.random.choice(n_pts, MAX_POINTS, replace=False)
            pts = pts[idx]

        self._update_plot(pts)
        self.get_logger().info(f'Displayed {pts.shape[0]} points')

    def _update_plot(self, pts: np.ndarray):
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        # In-place update of scatter data
        self.scatter._offsets3d = (xs, ys, zs)

        # Dynamic axis limits
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        self.ax.set_xlim(mins[0], maxs[0])
        self.ax.set_ylim(mins[1], maxs[1])
        self.ax.set_zlim(mins[2], maxs[2])

        # Draw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPCDVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
