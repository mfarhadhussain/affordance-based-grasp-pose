# Cmaera.msg
# Custom message to encapsulate RealSense/Pybullet camera data

std_msgs/Header header
sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/Image segmentation_map
float32 depth_scale


