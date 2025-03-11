import pybullet as p 
import pybullet_data
import time
import numpy as np
import cv2
import os 
import random 

#
import sys
sys.path.append('/home/mdfh/ros2_ws/src/franka_sim/franka_sim/')
from urdf_model import URDFModel


#########################################################################
# PybulletFranka Class: Sets up the robot, scene, and GUI for model placement.
#########################################################################

class PybulletFranka(object):
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.63],
                                   physicsClientId=self.physics_client)
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.0, 0.0, -0.63],
                                   physicsClientId=self.physics_client)

        self.robot_base_position = [-0.5, -0.35, 0.0]
        self.robot_base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                   basePosition=self.robot_base_position,
                                   baseOrientation=self.robot_base_orientation,
                                   useFixedBase=True,
                                   physicsClientId=self.physics_client)

        self.end_effector_index = 11
        self.initial_joint_state = [0.031, -0.271, -0.207, -1.228, 0.092, 1.186, -2.155, 0.00, 0.00]
        self._initial_robot_joints_state()

        # (Assume URDFModel is defined in your module; otherwise, comment this out)
        self.urdf_model = URDFModel()  
        self.dict_of_model_added_to_scene = {}
        self._model_user_interface()

        # Reset debug visualizer camera (this is independent of the virtual camera).
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=[0.0, 0.0, 0.0])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    def _initial_robot_joints_state(self):
        for joint_index, joint_state in enumerate(self.initial_joint_state):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=joint_index,
                              targetValue=joint_state,
                              physicsClientId=self.physics_client)

    def get_robot_joints_state(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        torques = [state[2] for state in joint_states]
        return positions, velocities, torques

    #####################################
    # GUI for placing/removing objects
    ####################################
    def _model_user_interface(self):
        self.pose_required = p.addUserDebugParameter("Pose Required (0/1)", 0, 1, 0)
        self.default_object = p.addUserDebugParameter(f"MUG (pybullet URDF)", 0, 1, 0)
        
        self.model_pose = {
            "x": p.addUserDebugParameter("X position", -0.2, 0.3, 0.0),
            "y": p.addUserDebugParameter("Y position", -0.2, 0.3, 0.0),
            "z": p.addUserDebugParameter("Z position", 0.0, 0.5, 0.2),
            "roll": p.addUserDebugParameter("Roll", -3.4, 3.4, 0.0),
            "pitch": p.addUserDebugParameter("Pitch", -3.4, 3.4, 0.0),
            "yaw": p.addUserDebugParameter("Yaw", -3.4, 3.4, 0.0)
        }
        three_dap_objects = [
            "Clock", "Earphone", "Vase", "Bowl", "Bag", "Faucet", "Bottle", "Mug", "Hat", 
            "drill", "sugar", # these is not seen by model
            "cup", # mug is in the 3dap dataset
        ]

        self.models = {
            model_name: p.addUserDebugParameter(f"Add {model_name}", 0, 1, 0)
            for model_name in self.urdf_model.list_of_models()
            if any(m_name.lower() in model_name.lower() for m_name in three_dap_objects)
        }

    def update_scene(self):
        """
        Dynamically add or remove objects from the scene based on user input.
        """
        pose_required = p.readUserDebugParameter(self.pose_required)
        
        # # this for DEFAULT OBJECT 
        if int(p.readUserDebugParameter(self.default_object)) == 1 and  self.default_object not in list(self.dict_of_model_added_to_scene.keys()):
            if int(pose_required) == 1:
                position = [p.readUserDebugParameter(self.model_pose["x"]),
                            p.readUserDebugParameter(self.model_pose["y"]),
                            p.readUserDebugParameter(self.model_pose["z"])]
                orientation = p.getQuaternionFromEuler([p.readUserDebugParameter(self.model_pose["roll"]),
                            p.readUserDebugParameter(self.model_pose["pitch"]),
                            p.readUserDebugParameter(self.model_pose["yaw"])])
            else: 
                position =  [
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.15),
                0.140
                ]
                orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            default_object_id_ = p.loadURDF(
                os.path.join(pybullet_data.getDataPath(), "objects/mug.urdf"), 
                basePosition=position,
                baseOrientation=orientation
                )
            self.dict_of_model_added_to_scene[self.default_object] = default_object_id_
            
        if int(p.readUserDebugParameter(self.default_object)) == 0 and self.default_object in list(self.dict_of_model_added_to_scene.keys()):
            p.removeBody(
                self.dict_of_model_added_to_scene[self.default_object]
            )
            del self.dict_of_model_added_to_scene[self.default_object]
        
        
        
        model_names = []
        for model_name in list(self.models.keys()):
            if int(p.readUserDebugParameter(self.models[model_name])) == 1:
                model_names.append(model_name)
                    
        models_to_be_added = [model_name for model_name in model_names
                              if model_name not in list(self.dict_of_model_added_to_scene.keys())]
        models_to_be_removed = [model_name for model_name in list(self.dict_of_model_added_to_scene.keys())
                                if model_name not in model_names]

        if models_to_be_removed:
            for model_name in models_to_be_removed:
                if model_name == self.default_object:
                    continue
                p.removeBody(self.dict_of_model_added_to_scene[model_name])
                del self.dict_of_model_added_to_scene[model_name]

        if int(pose_required) == 1 and len(models_to_be_added) == 1:
            # When placing a single object with a defined pose.
            position = [p.readUserDebugParameter(self.model_pose["x"]),
                        p.readUserDebugParameter(self.model_pose["y"]),
                        p.readUserDebugParameter(self.model_pose["z"])]
            orientation = [p.readUserDebugParameter(self.model_pose["roll"]),
                           p.readUserDebugParameter(self.model_pose["pitch"]),
                           p.readUserDebugParameter(self.model_pose["yaw"])]
            model_id = self.urdf_model.add_model(models_to_be_added[0],
                                                 position=position,
                                                 orientation=orientation)
            self.dict_of_model_added_to_scene[models_to_be_added[0]] = model_id
            models_to_be_added = []
            
        if models_to_be_added:
            for model_name in models_to_be_added:
                model_id = self.urdf_model.add_model(model_name)
                self.dict_of_model_added_to_scene[model_name] = model_id

        p.stepSimulation()


#########################################################################
# Main Simulation Loop
#########################################################################

def main():
    sim_franka = PybulletFranka()
    while True:
        p.stepSimulation()
        sim_franka.update_scene()
        
        # --- Update the camera attached at the end effector ---
        _, _, rgb, _, _ = sim_franka.camera.update(sim_franka.robot_id, sim_franka.end_effector_index)
        
        # Process and display the camera image using OpenCV.
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((720, 720, 3))
        rgb_bgr = cv2.cvtColor(rgb_array[:, :, :3], cv2.COLOR_RGB2BGR)
        cv2.imshow("End Effector Camera", rgb_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
        
    cv2.destroyAllWindows()
    p.disconnect()

if __name__ == "__main__":
    main()