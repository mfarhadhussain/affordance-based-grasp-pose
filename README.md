# Language-Conditioned Affordance-Based Grasp Pose Learning from 3D Point Clouds

<p align="center">
  <img src="images/affordance_mug.drawio.png" width="40%" />
</p>

## Overview

Poject implements an end-to-end, language-conditioned robotic grasping system using 3D point clouds and diffusion models. It detects task-relevant affordances in objects and generates 6-DoF grasp poses for various manipulation tasks such as lifting, pouring, wrapping, and containing.

## Key Features

- 🎯 **Language-Conditioned**: Task descriptions guide affordance detection.
- 🔄 **Diffusion-Based Poses**: Generates diverse grasp candidates via denoising diffusion models.
- 🌐 **Multi-Modal Fusion**: Combines point cloud geometry with text semantics.
- 🤖 **Sim-to-Real Ready**: PyBullet simulation + ROS 2 for real robot deployment.

## Quick Start

### Prerequisites
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Python**: 3.10+
- **GPU**: NVIDIA with CUDA (recommended)

### Installation

```bash
# 1. Create workspace and clone repo
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/mfarhadhussain/affordance-based-grasp-pose.git
cd affordance-based-grasp-pose && git submodule update --init --recursive

# 2. Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/affordance-based-grasp-pose/requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Build ROS 2 packages
colcon build --symlink-install
source install/setup.bash
```

## Usage

**Simulation Demo:**
```bash
ros2 launch franka_sim franka_sim.launch.py # Terminal 1
ros2 run franka_common model_node --ros-args --param task_description:="grasp to lift" # Terminal 2
ros2 run franka_common affordance_pose_plot_node # Terminal 3
```

**Real Robot:**
```bash
ros2 run franka_real franka_real_cam_node # Terminal 1
ros2 run franka_common model_node --ros-args --param task_description:="grasp to pour" # Terminal 2
```

## Results & Visualizations

The 3DAPNet model successfully generalizes object affordances and 6-DoF grasp poses across simulations and real-world scenes. 

- **Simulation vs. Real Scene:** Affordance detection on the test dataset is qualitatively better than on the self-occluded PyBullet simulation datasets. However, generated poses in the PyBullet simulation are more accurate.
- **Real-World Generalization:** Evaluated using an Intel RealSense camera and Franka Panda robot, the system accurately detects affordances and provides corresponding poses for tasks like picking, pouring, wrapping, and containing.

### Task Demonstrations

#### Contain
| <img src="images/contain/Screenshot%20from%202025-05-19%2013-19-32.png" height="200" /> | <img src="images/contain/Screenshot%20from%202025-05-19%2013-19-54.png" height="200" /> |
|:---:|:---:|

#### Lift
| <img src="images/lift/Screenshot%20from%202025-05-19%2013-17-16.png" height="200" /> | <img src="images/lift/Screenshot%20from%202025-05-19%2013-17-52.png" height="200" /> |
|:---:|:---:|

#### Pick
| <img src="images/pick/Screenshot%20from%202025-05-19%2013-16-13.png" height="200" /> | <img src="images/pick/Screenshot%20from%202025-05-19%2013-16-38.png" height="200" /> |
|:---:|:---:|

#### Wrap Grasp
| <img src="images/wrap_grasp/Screenshot%20from%202025-05-19%2013-13-09.png" height="200" /> | <img src="images/wrap_grasp/Screenshot%20from%202025-05-19%2013-13-29.png" height="200" /> |
|:---:|:---:|

### Network Components

| Context Block & Affordance Net |  Grasp Net | 
|:---:|:---:|
| <img src="images/context block.png" width=81% /> | <img src="images/affordance_block.png" width=81% /> |

## Citation

```bibtex
@thesis{3d_lang_cond_af_pose_2025,
  title={Language-Conditioned Affordance-Based Grasp Pose Learning from 3D Point Clouds for Robotic Skill Generalization},
  author={Md Farhad Hussain and Dr. Ravi Prakash},
  school={Robert Bosch Centre for Cyber-Physical Systems, Indian Institute of Science, Bengaluru, India},
  year={2025}
}
```
