from setuptools import find_packages, setup
import os 
from glob import glob 

package_name = 'franka_common'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.launch.py'))
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mdfh',
    maintainer_email='farhadh@iisc.ac.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "model_node=franka_common.model:main",
            "object_pcd_visualizer_node=franka_common.object_pcd_visualizer:main",
            "affordance_pose_plot_node=franka_common.affordance_pose_plot:main",
        ],
    },
)
