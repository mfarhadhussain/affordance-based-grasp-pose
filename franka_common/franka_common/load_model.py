import sys
import os
import torch 
from types import SimpleNamespace
import numpy as np 

dl_model_path = '/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds'
sys.path.append(dl_model_path)

from utils.builder import *
from utils.utils import *
from utils.trainer import *


model_weight_path = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/current_model.t7"
device = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    def __init__(self):
        self.seed = 1
        # self.log_dir = "/log"
        self.log_dir = "/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log"
        self.scheduler = None

        self.model = {
            'type': 'detectiondiffusion',
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'background_text': 'none',
            'betas': [1e-4, 0.02],
            'n_T': 1000,
            'drop_prob': 0.1,
            'weights_init': 'default_init',
        }
        
        self.optimizer = {
            'type': 'adam',
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 1e-5,
        }
        
        
        self.training_cfg = {
            "model": self.model, 
            "batch_size": 32,
            # epoch=200,
            "epoch": 50,
            "gpu": '0',
            "workflow": {
                        "train":1,
                        },
            "bn_momentum": None,  
        }
        
        self.data = SimpleNamespace(
            data_path="/home/mdfh/open_vocab_ws/src/dl_model/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/dataset/data.pkl",
            # data_path="/kaggle/input/affordance-datas/datas/data.pkl"
        )
        
        
def load_model(model_weight_path=model_weight_path, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    
    checkpoint = torch.load(model_weight_path, map_location=device)
    
    new_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):  
            new_key = key[len('module.'): ]  
            new_checkpoint[new_key] = value
        else:
            new_checkpoint[key] = value

    model.load_state_dict(new_checkpoint)
    
    return model


     
if __name__=="__main__": 
    model = load_model(model_weight_path=model_weight_path)
    print("Model is running on: ", model.device)