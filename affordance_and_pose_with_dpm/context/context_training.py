#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from easydict import EasyDict

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from utils import get_device, get_weights_file_path, latest_weights_file_path
from data.three_dap.joint_ap_datasets import JointAPDatsets
from text_encoder.open_clip_encoder import OpenCLIPPerTokenEncoder, OpenCLIPTextEncoder
from point_encoder.pointnext_c64 import PointNeXtC64
from context import PointTextEncoderDecoder, FinetuningPointText


def train_step(xyz, text, targets, point_model, text_model,
               ctxt_model, fine_head, optimizer,
               loss_fn, grad_clip):
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        p_feats = point_model(xyz)
        t_feats = text_model(text)
    ctxt = ctxt_model(xyz, p_feats, t_feats)
    logits = fine_head(ctxt)
    loss = loss_fn(logits, targets)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(ctxt_model.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(fine_head.parameters(), grad_clip)
    optimizer.step()
    return loss


def eval_step(xyz, text, targets, point_model, text_model,
              ctxt_model, fine_head, loss_fn, threshold):
    p_feats = point_model(xyz)
    t_feats = text_model(text)
    ctxt = ctxt_model(xyz, p_feats, t_feats)
    logits = fine_head(ctxt)
    loss = loss_fn(logits, targets)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()
    return loss, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.config_file)))

    device = get_device()
    torch.manual_seed(cfg.training.seed)
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
        JointAPDatsets(cfg.dataset.data_file_path, mode="test"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        **loader_args,
    )

    point_model = PointNeXtC64(
        cfg["pointnext-s_c64"].cfg,
        cfg["pointnext-s_c64"].ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to(device).eval()
    text_model = OpenCLIPTextEncoder(device="cuda" if torch.cuda.is_available() else "cpu").to(device).eval()

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
    ).to(device)


    fine_head = FinetuningPointText(
        point_feats_dim=cfg.xyz.feats_dim,
    ).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(ctxt_model.parameters()) + list(fine_head.parameters()),
        lr=cfg.training.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs,
    )
   
    writer = SummaryWriter(log_dir=cfg.training.experiment_name)
    os.makedirs(cfg.model.model_folder, exist_ok=True)

    start_epoch, global_step = 0, 0
    if cfg.model.preload:
        key = (latest_weights_file_path if cfg.model.preload == "latest"
               else get_weights_file_path)
        ckpt = key(cfg.model.model_folder, cfg.model.model_basename)
        if ckpt:
            state = torch.load(ckpt, map_location=device)
            ctxt_model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = state["epoch"] + 1
            global_step = state["global_step"]

    for epoch in range(start_epoch, cfg.training.num_epochs):
        ctxt_model.train(); fine_head.train()
        total_loss, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for *_, xyz, text, targets, _ in pbar:
            xyz = xyz.to(device, non_blocking=True)
            targets = targets.squeeze(-1)
            targets = targets.to(device, non_blocking=True).float()
            loss = train_step(
                xyz, text, targets,
                point_model, text_model,
                ctxt_model, fine_head,
                optimizer, 
                loss_fn, cfg.training.grad_clip,
            )
            batch = xyz.size(0)
            total_loss += loss.item() * batch
            count += batch
            global_step += 1
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total_loss / count
        writer.add_scalar("Train/Epoch_Loss", avg_train, epoch)

        ctxt_model.eval(); fine_head.eval()
        val_loss, samples = 0.0, 0
        tp = fp = fn = 0
        iou_sum = 0.0

        for *_, xyz, text, targets, _ in val_loader:
            xyz = xyz.to(device, non_blocking=True)
            targets = targets.squeeze(-1)
            targets = targets.to(device, non_blocking=True).float()
            loss, preds = eval_step(
                xyz, text, targets,
                point_model, text_model,
                ctxt_model, fine_head,
                loss_fn, cfg.training.threshold,
            )
            batch = xyz.size(0)
            val_loss += loss.item() * batch
            samples += batch
            inter = (preds & targets.long()).sum(dim=1).float()
            union = (preds | targets.long()).sum(dim=1).float()
            iou_sum += (inter / (union + 1e-6)).sum().item()
            tp += inter.sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()

        avg_val = val_loss / samples
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        N = targets.size(1)
        total_elems = samples * N
        # true negatives = total - TP - FP - FN
        tn = total_elems - tp - fp - fn
        hamming = (tp + tn) / total_elems
        miou = iou_sum / samples

        writer.add_scalars("Val", {
            "Loss": avg_val,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Hamming": hamming,
            "mIoU": miou,
        }, epoch)

        print(f"[{epoch:03d}] TrainL={avg_train:.4f} "
              f"ValL={avg_val:.4f} P={precision:.4f} "
              f"R={recall:.4f} F1={f1:.4f} "
              f"H={hamming:.4f} mIoU={miou:.4f}")

        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            fname = get_weights_file_path(
                cfg.model.model_folder,
                cfg.model.model_basename,
                epoch=f"{epoch:03d}"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": ctxt_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            }, fname)
        scheduler.step()
        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()
