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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random 
import torch.nn.functional as F 

# project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from utils import get_device, get_weights_file_path, latest_weights_file_path
from data.three_dap.joint_ap_datasets import JointAPDatsets
from text_encoder.open_clip_encoder import OpenCLIPPerTokenEncoder, OpenCLIPTextEncoder
from point_encoder.pointnext_c64 import PointNeXtC64
from context.context import PointTextEncoderDecoder 
from affordance_pose.pose import PoseNetLocal
from affordance_pose.affordance import AffordanceNet
from ddpm.bg_ddpm import JointAffordancePose 


from ddpm.scheduler import (
    create_schedule, 
)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.config_file)))

    device = get_device()
    torch.cuda.manual_seed_all(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    from utils import suggest_radii
    
    # data loaders
    loader_args = dict(
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(
        JointAPDatsets(cfg.dataset.data_file_path, mode="train"),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        **loader_args,
    )
    val_loader = DataLoader(
        JointAPDatsets(cfg.dataset.data_file_path, mode="val"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        **loader_args,
    )

    r_min, r_max = 0.001, 0.025
    num_batches = len(train_loader) 
    # r_min, r_max = suggest_radii(
    # train_loader,
    # k=32,                 
    # batches_to_scan=num_batches  
    # )

    # models
    point_model = PointNeXtC64(
        cfg["pointnext-s_c64"].cfg,
        cfg["pointnext-s_c64"].ckpt,
        device=device
    ).to(device)

    text_model = OpenCLIPTextEncoder(device=device).to(device)

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
        token_level=False
    ).to(device=device) 


    if cfg.model.mh_context.preload:
        key = (latest_weights_file_path if cfg.model.mh_context.preload == "latest"
               else get_weights_file_path)
        ckpt = key(cfg.model.mh_context.model_folder, cfg.model.mh_context.model_basename)
        if ckpt:
            state = torch.load(ckpt, map_location=device)
            ctxt_model.load_state_dict(state["model_state_dict"]) 
            print("loaded the trained weight for context")
            
    affordance_model  = AffordanceNet().to(device=device)
    pose_model = PoseNetLocal(
        feat_dim=cfg.xyz.feats_dim, M_min=8, M_max=64,
        k_max=32, r_min=r_min, r_max=r_max
    ).to(device)


    #####
    # diffusion schedule
    T          = cfg.diffusion.T
    sched      = create_schedule(
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

    # optimizer & losses 
        # optimizer & losses 
    # ——————————————————————————

    opt_aff  = torch.optim.Adam(
        affordance_model.parameters(),
        lr=cfg.training.lr_aff
    )
    opt_pose = torch.optim.Adam(
        pose_model.parameters(),
        lr=cfg.training.lr_pose
    )
    joint_params = None
    if cfg.training.multi_optimizer:
        optimizers = {"aff": opt_aff, "pose": opt_pose}
    else:
        # one joint optimizer over both nets
        joint_params = list(affordance_model.parameters()) + list(pose_model.parameters())
        optimizers = {
            "joint": torch.optim.Adam(
                joint_params,
                lr=cfg.training.lr_joint
            )
        }

    # scheduler setup
    sched_aff  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_aff,  T_max=cfg.training.num_epochs)
    sched_pose = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pose, T_max=cfg.training.num_epochs)
    if cfg.training.multi_scheduler:
        sched_aff  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_aff,  T_max=cfg.training.num_epochs)
        sched_pose = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pose, T_max=cfg.training.num_epochs)
        schedulers = {"aff": sched_aff, "pose": sched_pose}
    else:
        # single scheduler on the joint optimizer
        schedulers = {
            "joint": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers["joint"], T_max=cfg.training.num_epochs
            )
        }

    writer = SummaryWriter(log_dir=cfg.training.experiment_name)




    # optionally preload affordance weights
    if os.path.exists(cfg.model.affordance.model_folder):
        print(f"The path {cfg.model.affordance.model_folder} exists.")
    else:
        print(f"The path {cfg.model.affordance.model_folder} does not exist.")
        os.makedirs(cfg.model.affordance.model_folder)
        print(f"The path {cfg.model.affordance.model_folder} created.")

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
            # opt_aff.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
        else:
            start_epoch, global_step = 0, 0
    else:
        start_epoch, global_step = 0, 0 


    # optionally preload pose weights
    if os.path.exists(cfg.model.pose.model_folder):
        print(f"The path {cfg.model.pose.model_folder} exists.")
    else:
        print(f"The path {cfg.model.pose.model_folder} does not exist.")
        os.makedirs(cfg.model.pose.model_folder)
        print(f"The path {cfg.model.pose.model_folder} created.")

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
            # opt_pose.load_state_dict(ck["optimizer_state_dict"])
    #     else:
    #         start_epoch, global_step = 0, 0
    # else:
    #     start_epoch, global_step = 0, 0 

    for epoch in range(start_epoch, cfg.training.num_epochs):
        
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        # pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Train]")
        epoch_loss_a = 0.0
        epoch_loss_p = 0.0

        jap.train()
        for sid, sematic_class, centroid, scale, xyz, text, a_gt, p_gt in pbar:
            xyz   = xyz.to(device)                         # (B,N,3)
            a_gt  = a_gt.to(device)                        # (B,N,1) binary mask
            p_gt  = p_gt.to(device)                        # (B,D_pose)

            
            null_token = cfg.diffusion.null_token
            descriptions = [
                null_token if random.random() < cfg.training.drop_prob  else desc
                for desc in text
            ]
            B = xyz.size(0)
            t = torch.randint(0, T, (B,), dtype=torch.float, device=device) 
            
            ctx = jap.encode_ctx(xyz, descriptions)

            # print(f"shape: {xyz.shape}, {a_gt.shape}, {p_gt.shape}, {ctx.shape}")
            l_a = jap.loss_mask(
                x0=a_gt,
                points=xyz,
                ctx=ctx,
                t=t
            )

        
            # loss for pose (Gaussian, use mask GT and no conf)
            l_p = jap.loss_pose(
                x0=p_gt,
                points=xyz,
                ctx=ctx,
                mask=a_gt.bool(),
                t=t,
                conf=None,
            )


            if cfg.training.multi_optimizer:
                # separate optimizers
                optimizers["aff"].zero_grad()
                l_a.backward()      
                torch.nn.utils.clip_grad_norm_(jap.mask_denoiser.parameters(), max_norm=1.0) 
                optimizers["aff"].step()

                optimizers["pose"].zero_grad()
                l_p.backward()
                torch.nn.utils.clip_grad_norm_(jap.pose_denoiser.parameters(), max_norm=1.0)
                optimizers["pose"].step()
            else:
                optimizers["joint"].zero_grad()
                (l_a + l_p).backward()
                torch.nn.utils.clip_grad_norm_(joint_params, max_norm=1.0)
                optimizers["joint"].step()
            
            epoch_loss_a += l_a.item()
            epoch_loss_p += l_p.item()
            global_step += 1
            pbar.set_postfix(loss_a=l_a.item(), loss_p=l_p.item())
            writer.add_scalar("Train/Batch_Loss_A", l_a.item(), global_step)
            writer.add_scalar("Train/Batch_Loss_P", l_p.item(), global_step)


######################################
        #     opt_aff.zero_grad() 
        #     l_a.backward()
        #     opt_aff.step() 
        #     epoch_loss_a += l_a.item()
        #     global_step += 1
        #     pbar.set_postfix(loss_a=l_a.item())
        #     writer.add_scalar("Train/Batch_Loss_A", l_a.item(), global_step)

        # avg_a = epoch_loss_a / len(train_loader)
        # print(f"Epoch {epoch} — Aff Loss: {avg_a:.4f}") 
        # writer.add_scalar("Train/Epoch_Loss_A", avg_a, epoch)

        # # save checkpoints every 10 epochs
        # if (epoch + 5) % 1 == 0 or (epoch + 1) == cfg.training.num_epochs:
        #     aff_path = get_weights_file_path(
        #         model_folder=cfg.model.affordance.model_folder,
        #         model_basename=cfg.model.affordance.model_basename,
        #         epoch=f"{epoch:03d}"
        #     )
            
        #     torch.save({
        #         "epoch": epoch,
        #         "model_state_dict": affordance_model.state_dict(),
        #         "optimizer_state_dict": opt_aff.state_dict(),
        #         "global_step": global_step
        #     }, aff_path)

        # sched_aff.step()
        # torch.cuda.empty_cache()
        
        # continue
#####################################

        avg_a = epoch_loss_a / len(train_loader)
        avg_p = epoch_loss_p / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss_A", avg_a, epoch)
        writer.add_scalar("Train/Epoch_Loss_P", avg_p, epoch)
        print(f"Epoch {epoch} — Aff Loss: {avg_a:.4f}, Pose Loss: {avg_p:.4f}") 

        # jap.eval()
        # total_aff_loss, samples = 0.0, 0
        # total_pos_loss = 0
        # total_val_loss = 0
        # tp = fp = fn = 0
        # iou_sum = 0.0
        # pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        # with torch.no_grad():
        #     for _, _, _, _, xyz_v, text_v, a_gt_v, p_gt_v  in pbar_val:
        #         xyz_v = xyz_v.to(device)
        #         a_gt_v     = a_gt_v.to(device)                  # (B, N, 1)
        #         p_gt_v    = p_gt_v.to(device) 
        #         descriptions_v = text_v

        #         B = xyz_v.shape[0]
                
        #         # sample affordance mask + confidence
        #         ctx_c, ctx_u, a_pred, conf_pred = jap.sample_affordance_ddim(
        #             T=T,
        #             points=xyz_v,
        #             desc=descriptions_v,
        #             guidance=cfg.diffusion.guidance_scale,
        #         )


        #         # ctx_c, ctx_u, a_pred, conf_pred = jap.sample_affordance_ddpm(
        #         #     T=T,
        #         #     points=xyz_v,
        #         #     desc=descriptions_v,
        #         #     guidance=cfg.diffusion.guidance_scale,
        #         # )


        #         # sample pose
        #         p_pred = jap.sample_pose_ddim(
        #             T=T,
        #             points=xyz_v,
        #             ctx_c=ctx_c,
        #             ctx_u=ctx_u,
        #             mask=a_pred.bool(),
        #             conf=None,
        #             guidance=cfg.diffusion.guidance_scale, 
        #             num_steps=cfg.diffusion.ddim_step
        #         ) 
                    
        #         a_pred_squeezed = a_pred.squeeze(-1)
        #         a_gt_squeezed = a_gt_v.squeeze(-1)

        #         # affordance loss loss (BCE)
        #         aff_loss = F.binary_cross_entropy(a_pred_squeezed, a_gt_squeezed, reduction='mean')
        #         total_aff_loss += aff_loss.item()*B
        #         samples += B
        #         inter = (a_pred_squeezed.bool() & a_gt_squeezed.bool()).sum(dim=1).float()
        #         union = (a_pred_squeezed.bool() | a_gt_squeezed.bool()).sum(dim=1).float()
        #         iou_sum += (inter / (union + 1e-6)).sum().item()
        #         tp += inter.sum().item()
        #         fp += ((a_pred_squeezed.bool() == 1) & ( a_gt_squeezed.bool() == 0)).sum().item()
        #         fn += ((a_pred_squeezed.bool() == 0) & ( a_gt_squeezed.bool() == 1)).sum().item()

                
        #         # Reshape poses to (B, 7)
        #         pose_pred = p_pred.view(B, 7)
        #         pose_gt_v   = p_gt_v.view(B, 7) 

        #         # Position loss (MSE)
        #         pos_loss = F.mse_loss(pose_pred, pose_gt_v, reduction='mean')
        #         loss = aff_loss + pos_loss 

        #         # Accumulate losses
        #         total_pos_loss += pos_loss.item() * B
        #         total_val_loss += loss.item() * B
            
        #     avg_aff_val_loss = total_aff_loss / samples
        #     avg_pos_loss   = total_pos_loss  / samples
        #     avg_total_loss = total_val_loss  / samples

        #     precision = tp / (tp + fp + 1e-6)
        #     recall = tp / (tp + fn + 1e-6)
        #     f1 = 2 * precision * recall / (precision + recall + 1e-6)

        #     N = a_gt_v.size(1)
        #     total_elems = samples * N
        #     # true negatives = total - TP - FP - FN
        #     tn = total_elems - tp - fp - fn
        #     hamming = (tp + tn) / total_elems
        #     miou = iou_sum / samples

        #     writer.add_scalars("Val", {
        #         "Affordance BCE Loss": avg_aff_val_loss,
        #         "Precision": precision,
        #         "Recall": recall,
        #         "F1": f1,
        #         "Hamming": hamming,
        #         "mIoU": miou,
        #     }, epoch)



        #     # Log to TensorBoard
        #     writer.add_scalars("Val", {
        #         "Total/Loss":       avg_total_loss,
        #         "Mask/Loss":        avg_aff_val_loss,
        #         "Pose/Position":    avg_pos_loss,
        #     }, epoch)


            # save checkpoints every 10 epochs
        if (epoch + 1) % 1 == 0 or (epoch + 1) == cfg.training.num_epochs:
            aff_path = get_weights_file_path(
                model_folder=cfg.model.affordance.model_folder,
                model_basename=cfg.model.affordance.model_basename,
                epoch=f"{epoch:03d}"
            )
            pose_path = os.path.join(
                cfg.model.pose.model_folder,
                f"{cfg.model.pose.model_basename}_{epoch:03d}.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": affordance_model.state_dict(),
                "optimizer_state_dict": opt_aff.state_dict(),
                "global_step": global_step
            }, aff_path)

            torch.save({
                "epoch": epoch,
                "model_state_dict": pose_model.state_dict(),
                "optimizer_state_dict": opt_pose.state_dict(),
                "global_step": global_step
            }, pose_path)
            print(f"Saved affordance checkpoint: {aff_path}")
            print(f"Saved pose checkpoint: {pose_path}")

        # step schedulers
        for sch in schedulers.values():
            sch.step()
        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()