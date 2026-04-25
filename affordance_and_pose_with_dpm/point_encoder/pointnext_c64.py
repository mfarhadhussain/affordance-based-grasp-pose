#!/usr/bin/env python3
import os, sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(__file__)
PNX_ROOT = os.path.abspath(os.path.join(HERE, "..", "PointNeXt"))
if PNX_ROOT not in sys.path:
    sys.path.insert(0, PNX_ROOT)
    
from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg

class PointNeXtC64(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str = None,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        freeze=True,
        recursive_cfg: bool = False
    ):
        super().__init__()

        self.cfg = EasyConfig()
        self.cfg.load(cfg_path, recursive=recursive_cfg)
        self.model = build_model_from_cfg(self.cfg.model)

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            weights = ckpt.get("model", ckpt)
            # print("Loaded weight keys:", weights.keys())
            self.model.load_state_dict(weights, strict=False) 
            if freeze:
                for p in self.model.parameters(): 
                    p.requires_grad = False

        self.model.to(device)
        self.device = device

        # backbone encoder for feature extraction
        self.backbone = self.model.encoder

        # Input channels (3 coords + 4 extras)
        self.in_ch = self.cfg.model.encoder_args.in_channels

    @staticmethod
    def estimate_normals(pts: np.ndarray, k: int = 16) -> np.ndarray:
        """
        PCA-based normal estimation on k-NN.
        pts: (N,3) array.
        Returns normals: (N,3).
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)
        _, idx = nbrs.kneighbors(pts)
        normals = np.zeros_like(pts)
        for i, neigh_idx in enumerate(idx):
            neigh = pts[neigh_idx[1:]]
            cov = np.cov((neigh - pts[i]).T)
            _, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        return normals

    @staticmethod
    def compute_height(pts: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Height above lowest point along axis.
        pts: (N,3) array.
        Returns heights: (N,1).
        """
        c = pts[:, axis]
        h = (c - c.min()).reshape(-1, 1)
        return h

    def _build_input(self, x: torch.Tensor):
        """
        Creates the 7-channel input and splits into coords and features.
        Returns:
          p0: (B, N, 3) coord tensor
          f0: (B, 7, N) feature tensor
        """
        B, N, C = x.shape
        if C == 3:
            pts = x.cpu().numpy().reshape(B, N, 3)
            normals = [self.estimate_normals(pts[b], k=16) for b in range(B)]
            heights = [self.compute_height(pts[b], axis=1) for b in range(B)]
            normals = np.stack(normals, axis=0)  # (B, N, 3)
            heights = np.stack(heights, axis=0)  # (B, N, 1)
            all7 = np.concatenate([pts, normals, heights], axis=2)  # (B, N, 7)
            x7 = torch.from_numpy(all7).to(self.device).float()
        elif C == self.in_ch:
            x7 = x.to(self.device).float()
        else:
            raise ValueError(f"Expected input channels=3 or {self.in_ch}, got {C}")

        p0 = x7[..., :3].contiguous()         # (B, N, 3)
        f0 = x7.permute(0, 2, 1).contiguous()  # (B, 7, N)
        return p0, f0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure feature extraction: runs encoder + decoder (minus final head) and returns
        the fine per-point features used by segmentation.
        Input:
          x: (B, N, 3 or 7)
        Returns:
          mid_feats: (B, C_mid, N)
        """
        p0, f0 = self._build_input(x)
        # Encoder produces coordinate and feature lists
        p_list, f_list = self.backbone.forward_seg_feat(p0, f0)
        # Dummy category for decoder context (unused in feature props)
        cls0 = torch.zeros(p0.size(0), 1, device=self.device, dtype=torch.long)
        # Decoder returns mid-level features per point
        mid_feats = self.model.decoder(p_list, f_list, cls0)
        mid_feats = mid_feats.permute(0, 2, 1)
        return mid_feats

    def extract_segmentation_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    cfg_path = "/home/farhad/m-project/affordance_and_pose/PointNeXt/cfgs/shapenetpart/pointnext-s_c64.yaml"
    ckpt_path = "/home/farhad/m-project/affordance_and_pose/checkpoints/pointnext/shapenetpart-train-pointnext-s_c64-ngpus4-seed7798-20220822-024210-ZcJ8JwCgc7yysEBWzkyAaE_ckpt_best.pth"
    model = PointNeXtC64(cfg_path, ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_xyz = torch.rand(2, 1024, 3).to(device)
    feats = model(dummy_xyz)
    print("Segmentation features shape:", feats.shape)
