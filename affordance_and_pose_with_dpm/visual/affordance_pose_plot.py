import open3d as o3d
import numpy as np
import yaml
import random
from torch.utils.data import DataLoader


def create_gripper_marker(tube_radius=0.002, sections=6):
    """Create a gripper marker using Open3D cylinders."""
    cylinders = []
    segments = [
        ([0.041, 0, 0.066], [0.041, 0, 0.112]),
        ([-0.041, 0, 0.066], [-0.041, 0, 0.112]),
        ([0, 0, 0], [0, 0, 0.066]),
        ([-0.041, 0, 0.066], [0.041, 0, 0.066])
    ]
    for start, end in segments:
        start_np = np.array(start)
        end_np = np.array(end)
        height = np.linalg.norm(end_np - start_np)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=tube_radius, height=height, resolution=sections)
        cylinder.compute_vertex_normals()
        midpoint = (start_np + end_np) / 2.0
        direction = end_np - start_np
        cylinder.translate(midpoint)
        if np.linalg.norm(direction) > 1e-6:
            rot_matrix = o3d.geometry.get_rotation_matrix_from_xyz((
                np.arctan2(direction[1], direction[0]),
                np.arccos(direction[2] / np.linalg.norm(direction)),
                0))
            cylinder.rotate(rot_matrix, center=midpoint)
        cylinders.append(cylinder)
    return cylinders


def visualize_point_cloud_with_grasp(points, affordance_mask, grasp_pose, thresold=0.03):
    """
    Visualize a point cloud with a grasp marker if the gripper tip is within thresold.
    
    Parameters
    ----------
    points : np.ndarray
        Nx3 array of 3D points.
    affordance_mask : np.ndarray
        Nx1 array (0/1) indicating affordance.
    grasp_pose : np.ndarray
        (7, 1) array with [x, y, z, q1, q2, q3, w] (quaternion in [q1, q2, q3, w] order).
    """
    if points.shape[1] != 3:
        raise ValueError("Points array must have shape (N, 3)")
    if affordance_mask.shape[0] != points.shape[0]:
        raise ValueError("affordance_mask must have the same length as points")
    if grasp_pose.shape != (7, 1):
        raise ValueError("Grasp pose must have shape (7, 1)")

    # Create point cloud and assign colors.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[affordance_mask.flatten() == 1] = [0, 0, 1]
    colors[affordance_mask.flatten() == 0] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Extract grasp pose: position and quaternion.
    grasp_position = grasp_pose[:3, 0]
    grasp_quaternion = grasp_pose[3:, 0]  # [q1, q2, q3, w]
    quat = np.array([grasp_quaternion[3], grasp_quaternion[0],
                     grasp_quaternion[1], grasp_quaternion[2]])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)

    # Build homogeneous transformation matrix.
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = grasp_position

    # Compute gripper tip location.
    tip = (T @ np.array([0., 0., 6.6e-02, 1.])).flatten()[:3]

    # Display only if the tip is within 3cm of the point cloud.
    if thresold is not None and np.min(np.linalg.norm(points - tip, axis=1)) <= thresold:
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        grasp_frame.rotate(R, center=(0, 0, 0))
        grasp_frame.translate(grasp_position)

        gripper_meshes = create_gripper_marker()
        for mesh in gripper_meshes:
            mesh.paint_uniform_color([0, 1, 0])
            mesh.rotate(R, center=(0, 0, 0))
            mesh.translate(grasp_position)

        o3d.visualization.draw_geometries([pcd, grasp_frame] + gripper_meshes)
    elif thresold is None:
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        grasp_frame.rotate(R, center=(0, 0, 0))
        grasp_frame.translate(grasp_position)

        gripper_meshes = create_gripper_marker()
        for mesh in gripper_meshes:
            mesh.paint_uniform_color([0, 1, 0])
            mesh.rotate(R, center=(0, 0, 0))
            mesh.translate(grasp_position)

        o3d.visualization.draw_geometries([pcd, grasp_frame] + gripper_meshes)
    else:
        print("Grasp invalid: gripper tip is too far from the point cloud.")


def main():
    import os 
    import sys 
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.three_dap.joint_ap_datasets import JointAPDatsets


    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config_file", type=str, help="Path of configuration file")
    args = parser.parse_args() 

    with open(args.config_file, "r") as f: 
        config = yaml.safe_load(f)
        
    data_file_path = config["dataset"]["data_file_path"]
    batch_size = config["training"]["batch_size"]

    train_dataset = JointAPDatsets(data_file_path=data_file_path, mode="train")
    # test_dataset = JointAPDatsets(data_file_path=data_file_path, mode="test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=4, shuffle=False, pin_memory=False)

    for _, _, centroid, _, pcd, text, a, p in train_loader:
        idx = random.choice(range(pcd.shape[0]))
        points = pcd[idx, :, :]
        affordance_mask = a[idx, :, :]
        grasp_pose = p[idx, :, :]

        print(f"Shape:\nPCD: {points.shape},\nA: {affordance_mask.shape}, Total_number of affordable point: {affordance_mask.sum()}"
              f"P: {grasp_pose.shape}, Centroid: {centroid.shape}")
        
        print(grasp_pose.min(), grasp_pose.max(), points, grasp_pose)
        visualize_point_cloud_with_grasp(points.numpy(),
                                         affordance_mask.numpy(),
                                         grasp_pose.numpy(), thresold=0.03)
        break


if __name__ == "__main__":
    main()
