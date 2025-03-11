import pybullet as p 
import random

#
import urdf_models.models_data as md



class URDFModel(object): 
    def __init__(self): 
        self.models_lib = md.model_lib()
        # print(self.models_lib.model_name_list)
        # print(self.models_lib['yellow_bowl'])
        # print(self.models_lib.random) 
        self._list_of_models = self.models_lib.model_name_list
        
    def list_of_models(self):
        return self._list_of_models
        
    def add_model(self, model_name: str=None, position=None, orientation=None): 
        try:
            if model_name is None:
                model_path = self.models_lib.random
            else: 
                model_path = self.models_lib[model_name]
                
            if position is None:
                position =  [
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.15),
                0.140
                ]
                
            if orientation is None: 
                orientation = p.getQuaternionFromEuler([0, 0, 0])
            else: 
                orientation = p.getQuaternionFromEuler(orientation)
                
            model_id = p.loadURDF(
                model_path,
                basePosition=position,
                baseOrientation=orientation,
                )
            
            return model_id
        except Exception as e: 
            print(e)   
        
    def remove_model(self, model_id):
        p.removeBody(model_id)
        
        
    
        
        
        
        
        
        
