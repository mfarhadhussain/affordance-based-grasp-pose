import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from custom_interfaces.msg import Object

class ObjectPCDVisualizer(Node):
    def __init__(self):
        super().__init__('object_pcd_visualizer')

        # Subscriber
        self.point_cloud_sub = self.create_subscription(
            PointCloud,
            "pointcloud",
            self.callback,
            0
        )

        # Matplotlib figure 
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()  
        
        # # go plotter
        # self.go_fig = go.FigureWidget()

    def callback(self, msg):
        """
        Callback function to process and visualize incoming point cloud data.
        """
        
        try:
            pcd_msg = msg 
            points = np.array([[point.x, point.y, point.z] for point in pcd_msg.points])

            if points.size == 0:
                self.get_logger().warn("Received empty point cloud.")
                return

            self.pcd_plotter_using_matplotlib(pcd=points)
            # self.pcd_plotter_using_go(pcd=points)

            self.get_logger().info(f"Visualized point cloud with {points.shape[0]} points.")

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")
            
            
    def pcd_plotter_using_matplotlib(self, pcd: np.array): 
        self.ax.clear()
        self.ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='r', marker='o', s=1)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Point Cloud')

        # Dynamic limits adjustement
        self.ax.set_xlim([np.min(pcd[:, 0]), np.max(pcd[:, 0])])
        self.ax.set_ylim([np.min(pcd[:, 1]), np.max(pcd[:, 1])])
        self.ax.set_zlim([np.min(pcd[:, 2]), np.max(pcd[:, 2])])
        plt.axis("off")
        plt.draw()
        plt.pause(0.1)  
            
            
    def pcd_plotter_using_go(self, pcd: np.array):
        
        self.go_fig.data = []
        self.go_fig.add_trace(    
                go.Scatter3d(
                x = pcd[:, 0],
                y = pcd[:, 1],
                z = pcd[:, 2],
                mode = 'markers',
                marker = dict(size=2, color='blue'),
                name = 'Point cloud of object (downsampled to 2048 points)'
            )
        )
        
        self.go_fig.update_layout(
            title = '3D Point Cloud of Object',
            scene = dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            legend=dict(x=0, y=1),
        )
    
        self.go_fig.show()
        
            


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
