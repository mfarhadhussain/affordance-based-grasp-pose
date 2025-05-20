import trimesh
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import argparse
from utils.visualization import create_gripper_marker
import pybullet as p 

color_code_1 = np.array([0, 0, 255])    # color code for affordance region
color_code_2 = np.array([0, 255, 0])    # color code for gripper pose
num_pose = 100 # number of poses to visualize per each object-affordance pair


# def parse_args():
#     parser = argparse.ArgumentParser(description="Visualize")
#     parser.add_argument("--result_file", help="result file")
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    # args = parse_args()
    # result_file = args.result_file
    result_file = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/result_pn_bn2.pkl"
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    for i in range(len(result)):
        if result[i]['semantic class'] == 'Mug':
            
            shape_index = i
            shape = result[shape_index]

            for affordance in shape['affordance']:
                # if affordance != "contain":
                #     continue
                colors = np.transpose(shape['result'][affordance][0]) * color_code_1
                point_cloud = trimesh.points.PointCloud(shape['full_shape']['coordinate'], colors=colors)
                print(f"Affordance: {affordance}")
                
                scene = trimesh.Scene()
                scene.add_geometry(point_cloud)
                poses = shape['result'][affordance][1][:num_pose]
                for pose in poses:
                    orientation = pose[:4]  # [qw, qx, qy, qz]
                    position = pose[4:]  # [x, y, z]
                    
                    # r_matrix = R.from_quat(orientation).as_matrix()
                    r_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
                    
                    T = np.eye(4)  
                    T[:3, :3] = r_matrix  
                    T[:3, 3] = position
                    
                    gripper_Position = (T @ np.array([0., 0., 6.59999996e-02, 1.]))[:3]
                    distance = np.linalg.norm(gripper_Position - point_cloud.vertices, axis=1)
                    thresold  = 0.005
                    
                    if np.any(distance <= thresold):
                        gripper_marker = create_gripper_marker(color=color_code_2)
                        gripper_marker.apply_transform(T)  
                        scene.add_geometry(gripper_marker)
                        
                # scene.show(line_settings={'point size': 10})
                
                
                
        
                # # rotation = np.concatenate((R.from_quat(T[:, :4]).as_matrix(), np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
                # rotation = np.concatenate(np.array(p.getMatrixFromQuaternion(T[:, :4])).reshape(3,3), np.zeros((num_pose, 1, 3), dtype=np.float32), axis=1)
                # translation = np.expand_dims(np.concatenate((T[:, 4:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
                # T = np.concatenate((rotation, translation), axis=2)
                # poses = [create_gripper_marker(color=color_code_2).apply_transform(t) for t in T
                #          if np.min(np.linalg.norm(point_cloud.vertices - (t @ np.array([0., 0., 6.59999996e-02, 1.]))[:3], axis=1)) <= 0.005] # this line is used to get reliable poses only
                # scene = trimesh.Scene([point_cloud, poses])
                # # scene.show(line_settings={'point size': 10})
                scene.show()
                
                
