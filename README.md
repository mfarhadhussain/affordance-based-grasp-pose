# Language-Conditioned Affordance-Based Grasp Pose Learning from 3D Point Clouds for Robotic Skill Generalization

## Overview

This repository implements an end-to-end pipeline for task-oriented robotic grasping:
1. **Affordance Detection**  
   Detects functional parts (e.g. handles) in 3D point clouds  based on task.
2. **Grasp Pose Generation**  
   Computes 6-DoF poses for a Franka Emika Panda (or any compatible manipulator) to perform task-constrained grasps.
3. **Simulation & Real-World Execution**  
   - **PyBullet** simulation with URDF models  
   - **ROS 2** integration for easy sim to real  on-hardware execution


