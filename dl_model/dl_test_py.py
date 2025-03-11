# import sys
# import os
# import torch 
# from types import SimpleNamespace
# import numpy as np 


# dl_model_path = '/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds'
# sys.path.append(dl_model_path)

# from utils.builder import *
# from utils.utils import *
# from utils.trainer import *


# model_weight_path = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/current_model.t7"
# device = "cuda" if torch.cuda.is_available() else "cpu"


# class Config:
#     def __init__(self):
#         self.seed = 1
#         # self.log_dir = "/log"
#         self.log_dir = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log"
#         self.scheduler = None

#         self.model = {
#             'type': 'detectiondiffusion',
#             'device': "cuda" if torch.cuda.is_available() else "cpu",
#             'background_text': 'none',
#             'betas': [1e-4, 0.02],
#             'n_T': 1000,
#             'drop_prob': 0.1,
#             'weights_init': 'default_init',
#         }
        
#         self.optimizer = {
#             'type': 'adam',
#             'lr': 1e-3,
#             'betas': (0.9, 0.999),
#             'eps': 1e-08,
#             'weight_decay': 1e-5,
#         }
        
        
#         self.training_cfg = {
#             "model": self.model, 
#             "batch_size": 32,
#             # epoch=200,
#             "epoch": 50,
#             "gpu": '0',
#             "workflow": {
#                         "train":1,
#                         },
#             "bn_momentum": PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),  
#         }
        
#         self.data = SimpleNamespace(
#             data_path="/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/dataset/data.pkl",
#             # data_path="/kaggle/input/affordance-datas/datas/data.pkl"
#         )
        
        
# def load_model(model_weight_path=model_weight_path):
#     cfg = Config()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = build_model(cfg).to(device)
    
#     checkpoint = torch.load(model_weight_path, weights_only=True)
    
#     new_checkpoint = {}
#     for key, value in checkpoint.items():
#         if key.startswith('module.'):  
#             new_key = key[len('module.'): ]  
#             new_checkpoint[new_key] = value
#         else:
#             new_checkpoint[key] = value

#     model.load_state_dict(new_checkpoint)
    
#     return model
     

# # ## TRAINING             
# # # cfg = Config()
# # # model = build_model(cfg).to(device)

# model = load_model() # loaded model with trained weight
# dataset_dict = build_dataset(cfg)  
# loader_dict = build_loader(cfg, dataset_dict) 
# optim_dict = build_optimizer(cfg, model)

# os.makedirs(cfg.log_dir, exist_ok=True)
# logger = IOStream(os.path.join(cfg.log_dir, 'run.log'))
# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg["gpu"]
# num_gpu = len(cfg.training_cfg["gpu"].split(','))
# logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg["gpu"]))

# training = dict(
#     model=model,
#     dataset_dict=dataset_dict,
#     loader_dict=loader_dict,
#     optim_dict=optim_dict,
#     logger=logger
# ) 

# # task_trainer = Trainer(cfg, training) 
# # task_trainer.run()




# # inference 
# model.eval()
# # EXAMPLE TEST
# # xyz = torch.rand((2048, 3)).unsqueeze(0).float().to(device)
# # text = "pick"
# # GUIDE_W = 0.5
# # with torch.inference_mode():
# #     result = model.detect_and_sample(xyz, text, 1000, guide_w=GUIDE_W)
    
# # affordance_lable = result[0]
# # poses = result[1]
# # poses = torch.from_numpy(poses)
# # base_pose = torch.median(poses, dim=0).values
# # print("Base Pose (Median):", base_pose)
# # print(f"Shape: pose: {poses.shape}\naffordance_label: {affordance_lable.shape}")
# # print(f"affordance: {affordance_lable}")
# # print(f"poses: {poses}")



# import pickle as pkl
# import random
# from scipy.spatial.transform import Rotation as R

# data_path = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/dataset/data.pkl"
# with open(data_path, "rb") as f: 
#     data = pkl.load(f)
#     random.shuffle(data)

# # Example of extracting the first data point for visualization
# for data_point in data:
#     for affordance in data_point["affordance"]:
#         for pose in data_point["pose"][affordance]:
#             new_data_dict = {
#                 "shape_id": data_point["shape_id"],
#                 "semantic class": data_point["semantic class"],
#                 "coordinate": data_point["full_shape"]["coordinate"],   # full point cloud coordinates
#                 "affordance": affordance,
#                 "affordance label": data_point["full_shape"]["label"][affordance],
#                 "rotation": R.from_matrix(pose[:3, :3]).as_quat(),
#                 "translation": pose[:3, 3]
#             }
#             break
#         break
#     break
# print(new_data_dict)



#### CHECKING RESULT FILE (INFERENCE FILE #####
import pickle as pkl
import random
from scipy.spatial.transform import Rotation as R


with open(
    "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/result.pkl", "rb"
    ) as f: 
    result_datas = pkl.load(f)
    
for result in result_datas:
    print(result)
    break