from setuptools import find_packages, setup

package_name = 'franka_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "franka_sim_node=franka_sim.franka_sim:main",
            "franka_sim_point_cloud_node=franka_sim.franka_sim_point_cloud:main",
        ],
    },
)
