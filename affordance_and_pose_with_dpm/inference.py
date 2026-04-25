"""
    INPUT:
        Point cloud: (B, N, 3), N>256
        task description: (B, )
    OUTPUT: 
        affordance: (B, N, 1)
        grasp pose: (B, 7, 1)

    Before passing the point to model it must be normalized as 
        centroid = np.median(Point_cloud, axis=0)
        coords_centered = Point_cloud - centroid  
        scale = np.max(np.linalg.norm(coords_centered, axis=1)) + 1e-6
        coords_norm = coords_centered / scale          # (N,3) 

        and this coords_norm will be passed to network and then network will give 
            "grasp pose" and "affordance" 
        To use this grasp pose for end task it must be transformed back to original point_cloud space 

            trans_centered = pose[:3,3]*scale + centroid         # (3,)
            quat = pose[:3,:3])    # (4,) # its already normalized 
    
"""
#!/usr/bin/env python3
# import warnings
# warnings.filterwarnings("error", category=UserWarning)

import os
import sys
import argparse
import yaml
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random 
import torch.nn.functional as F 

# project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from utils import get_device, get_weights_file_path, latest_weights_file_path
from data.three_dap.joint_ap_datasets import JointAPDatsets
from text_encoder.open_clip_encoder import OpenCLIPPerTokenEncoder
from point_encoder.pointnext_c64 import PointNeXtC64
from context.context import PointTextEncoderDecoder 
from affordance_pose.pose import PoseNetLocal
from affordance_pose.affordance import AffordanceNet
from ddpm.bg_ddpm import JointAffordancePose 

from ddpm.scheduler import (
    create_schedule, 
)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
args = parser.parse_args()


cfg = EasyDict(yaml.safe_load(open(args.config_file)))

device = get_device()
torch.cuda.manual_seed_all(cfg.training.seed)
random.seed(cfg.training.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# data loaders
loader_args = dict(
    num_workers=cfg.training.num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

test_loader = DataLoader(
    JointAPDatsets(cfg.dataset.data_file_path, mode="test"),
    batch_size=cfg.training.batch_size,
    shuffle=False,
    **loader_args,
)

# models
point_model = PointNeXtC64(
    cfg["pointnext-s_c64"].cfg,
    cfg["pointnext-s_c64"].ckpt,
    device=device
).to(device)

text_model = OpenCLIPPerTokenEncoder(device=device).to(device)

ctxt_model = PointTextEncoderDecoder(
    point_dim=cfg.xyz.feats_dim,
    text_dim=cfg.text.feats_dim,
    sample_ratios=(0.5, 0.25, 0.125),
    ks=(16, 32, 64),
    num_heads=4,
    use_mp=False,
    random_start=True,
    jitter=True,
    jitter_strength=1e-3,
    token_level=True
).to(device=device) 

if cfg.model.mh_context.preload:
    key = (latest_weights_file_path if cfg.model.mh_context.preload == "latest"
            else get_weights_file_path)
    ckpt = key(cfg.model.mh_context.model_folder, cfg.model.mh_context.model_basename)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        ctxt_model.load_state_dict(state["model_state_dict"]) 

affordance_model  = AffordanceNet().to(device=device)

if cfg.model.affordance.preload:
    getter = (latest_weights_file_path
                if cfg.model.affordance.preload == "latest"
                else get_weights_file_path)
    ckpt = getter(
        cfg.model.affordance.model_folder,
        cfg.model.affordance.model_basename
    )
    if ckpt:
        ck = torch.load(ckpt, map_location=device)
        affordance_model.load_state_dict(ck["model_state_dict"])

pose_model = PoseNetLocal(
    feat_dim=cfg.xyz.feats_dim, M_min=4, M_max=32,
    k_max=32, r_min=0.05, r_max=0.2
).to(device) 

if cfg.model.pose.preload:
    getter = (latest_weights_file_path
                if cfg.model.pose.preload == "latest"
                else get_weights_file_path)
    ckpt = getter(
        cfg.model.pose.model_folder,
        cfg.model.pose.model_basename
    )
    if ckpt:
        ck = torch.load(ckpt, map_location=device)
        pose_model.load_state_dict(ck["model_state_dict"]) 


T = cfg.diffusion.T
sched = create_schedule(
    schedule_type=cfg.diffusion.schedule.type,
    T=T,
    device=device,
    s=cfg.diffusion.schedule.s,
    beta_start=cfg.diffusion.schedule.beta_start,
    beta_end=cfg.diffusion.schedule.beta_end,
)
betas      = sched.betas
alphas     = sched.alphas
alpha_bar  = sched.alpha_bar  


jap = JointAffordancePose(
            point_encoder=point_model,
            text_encoder=text_model,
            context_net=ctxt_model,
            mask_denoiser=affordance_model,
            pose_denoiser=pose_model,
            betas=betas, 
            alpha=alphas,
            alpha_bar=alpha_bar
        )


T = cfg.diffusion.T

# inference per batch 
def inference(xyz, text): 
    xyz = xyz.to(device)
    centroid = torch.median(xyz, dim=1, keepdim=True)[0]  # (B, 1, 3)
    coords_centered = xyz - centroid
    scale = torch.amax(torch.linalg.norm(coords_centered, dim=2), dim=1, keepdim=True).view(-1, 1, 1) + 1e-6
    coords_norm = coords_centered / scale
    descriptions_v = text

    B = coords_norm.shape[0]
    with torch.no_grad():

        # sample affordance mask + confidence
        ctx_c, ctx_u, a_pred, conf_pred = jap.sample_mask(
            timesteps=T,
            points=coords_norm,
            descriptions=descriptions_v,
            guidance=cfg.diffusion.guidance_scale,
            strategy = "ddpm",
        )
        # sample pose
        p_pred = jap.sample_pose(
            timesteps=T,
            points=coords_norm,
            ctx_c=ctx_c,
            ctx_u=ctx_u,
            mask=a_pred.bool(),
            conf=conf_pred,
            guidance=cfg.diffusion.guidance_scale
        ) 

        "p_pred is shape of (B, 7, 1)"
        centroid = centroid.transpose(1, 2)
        trans = p_pred[:, 0:3, :]*scale + centroid       
        quat = p_pred[:, 3:7, :]  
        print(f"Pose pred shape: {p_pred.shape}")
        print(f"trans shape: {trans.shape}")
        print(f"quat shape: {quat.shape}")


        # output = torch.cat([trans, quat], dim=1)
    return xyz, a_pred, trans, quat

if __name__=="__main__": 
    import pickle as pkl
    import numpy as np  
    from scipy.spatial.transform import Rotation as R

    B, N = 2, 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.rand(B, N, 3, device=device)
    descriptions = ["a handle", "A black cap."] 

    data_file_path = "/home/farhad/affordance_and_pose/data/three_dap/full_shape_release.pkl"
    with open(data_file_path, "rb") as f: 
        dataset = pkl.load(f) 

    for data_point in dataset:
        for affordance in data_point["affordance"]:
            for pose in data_point["pose"][affordance]:
                coords = np.array(data_point["full_shape"]["coordinate"])  
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = pose[:3, :3]
                transform_matrix[:3, 3] = pose[:3, 3]

                quat = R.from_matrix(pose[:3,:3]).as_quat()    # (4,)
                position = pose[:3,3]
                new_data_dict = {
                    "shape_id": data_point["shape_id"],
                    "semantic class": data_point["semantic class"],
                    "coordinate": coords, 
                    "affordance": affordance,
                    "affordance label": data_point["full_shape"]["label"][affordance],
                    "translation": position,
                    "rotation": quat
                } 

                xyz_pred, a_pred, trans_pred, quat_pred = inference(torch.from_numpy(new_data_dict["coordinate"]).view(1, -1, 3), new_data_dict["affordance"])
                print(f"shape: \n\t{a_pred.shape} \t\n{trans_pred.shape} \t\n {quat_pred}")
                print(f"a_pred: {a_pred}\n\ttrans: {trans_pred} \n\t rot: {quat_pred}")





    # xyz, a_pred, g_pose = inference(xyz, text=descriptions) 

    # print(f"shape: \n\t{a_pred.shape} \t\n{g_pose.shape}")
    # print(f"a_pred: {a_pred}\n\ng_pose: {g_pose}")



