#!/usr/bin/env python3
"""Local-patch equivariant grasp-pose denoiser with adaptive sampling and global context."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_anchors(
    mask: torch.Tensor,
    xyz: torch.Tensor,
    M_min: int,
    M_max: int
) -> torch.Tensor:
    """Adaptive anchor sampling (random + padding)"""
    B, N, _ = xyz.shape
    density = mask.view(B, -1).float().sum(1) / N
    M_sample = (density * M_max).clamp(min=M_min, max=M_max).long()
    anchors = torch.zeros(B, M_max, dtype=torch.long, device=xyz.device)
    for b in range(B):
        idx = mask[b].view(-1).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.arange(N, device=xyz.device)
        Mb = M_sample[b].item()
        if idx.numel() >= Mb:
            sel = idx[torch.randperm(idx.numel(), device=xyz.device)[:Mb]]
        else:
            extra = idx[torch.randint(idx.numel(), (Mb - idx.numel(),), device=xyz.device)]
            sel = torch.cat([idx, extra], dim=0)
        # pad to M_max
        pad_n = M_max - sel.numel()
        if pad_n > 0:
            pad = sel[-1].repeat(pad_n)
            sel = torch.cat([sel, pad], dim=0)
        anchors[b] = sel
    return anchors


import torch

# def radius_neighbors(
#     xyz: torch.Tensor,
#     anchors: torch.Tensor,
#     k_max: int,
#     r_min: float,
#     r_max: float,
#     valid: torch.Tensor
# ) -> torch.Tensor:
#     """Radius-based neighbor indices per anchor, robust to empty masks."""
#     B, N, _ = xyz.shape
#     M = anchors.size(1)

#     # dynamic radius per batch
#     density = valid.view(B, -1).float().sum(1) / N
#     r_batch = r_min + density * (r_max - r_min)

#     neigh = torch.zeros(B, M, k_max, dtype=torch.long, device=xyz.device)
#     for b in range(B):
#         pts = xyz[b]                         # (N,3)
#         cent = pts[anchors[b]]               # (M,3)
#         d    = torch.cdist(cent, pts)        # (M,N)
#         r    = r_batch[b].item()
#         mask = d <= r                        # (M,N)

#         for i in range(M):
#             valid_idx = mask[i].nonzero(as_tuple=False).flatten()  # LongTensor

#             if valid_idx.numel() == 0:
#                 # No neighbors: fully random fallback
#                 sel = torch.randint(0, N, (k_max,), device=xyz.device)
#             elif valid_idx.numel() >= k_max:
#                 # enough neighbors: sample without replacement
#                 perm = torch.randperm(valid_idx.numel(), device=xyz.device)
#                 sel  = valid_idx[perm[:k_max]]
#             else:
#                 # some neighbors but fewer than k_max: pad with repeats
#                 need = k_max - valid_idx.numel()
#                 # pick extras uniformly from the valid set
#                 extras = valid_idx[torch.randint(
#                     0, valid_idx.numel(), (need,), device=xyz.device
#                 )]
#                 sel = torch.cat([valid_idx, extras], dim=0)

#             neigh[b, i] = sel

#     return neigh

import torch

def knn_neighbors(
    xyz: torch.Tensor,
    anchors: torch.Tensor,
    k_max: int
) -> torch.Tensor:
    """
    For each anchor point, return the indices of its k_max nearest neighbors.
    Args:
      xyz:     (B, N, 3) point clouds
      anchors: (B, M)   anchor indices into N
      k_max:   int      number of neighbors to sample
    Returns:
      neigh:   (B, M, k_max) long indices in [0..N)
    """
    B, N, _ = xyz.shape
    M = anchors.size(1)

    # allocate output
    neigh = torch.zeros(B, M, k_max, dtype=torch.long, device=xyz.device)

    # compute for each batch
    for b in range(B):
        points = xyz[b]              # (N,3)
        cent   = points[anchors[b]]  # (M,3)
        # full distance matrix (M,N)
        dists  = torch.cdist(cent, points)  
        # get k smallest distances
        _, idx = torch.topk(dists, k_max, dim=1, largest=False, sorted=False)  
        # idx is (M, k_max)
        neigh[b] = idx

    return neigh




class EGNNPatch(nn.Module):  # noqa: D204
    def __init__(self, dim: int):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * dim + 1, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU()
        )
        self.phi_x = nn.Linear(dim, 1, bias=False)
        self.phi_h = nn.Sequential(
            nn.Linear(dim + dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Bm, k, d = h.shape
        xi = x.unsqueeze(2)
        xj = x.unsqueeze(1)
        d2 = (xi - xj).pow(2).sum(-1, keepdim=True)
        hi = h.unsqueeze(2).expand(-1, -1, k, -1)
        hj = h.unsqueeze(1).expand(-1, k, -1, -1)
        e = torch.cat([hi, hj, d2], dim=-1)
        m = self.phi_e(e)
        dx = (xi - xj) * self.phi_x(m)
        x = x + dx.sum(2) / k
        agg = m.sum(2) / k
        h = h + self.phi_h(torch.cat([h, agg], dim=-1))
        return x, self.norm(h)


class LocalAttn(nn.Module):  # noqa: D204
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor
    ) -> torch.Tensor:
        B, M, d = q.shape
        k = kv.size(2)
        qf = q.reshape(B * M, 1, d)
        kvf = kv.reshape(B * M, k, d)
        y, _ = self.attn(self.ln1(qf), kvf, kvf)
        res = qf + y
        out = res + self.ff(self.ln2(res))
        return out.reshape(B, M, d)


class FiLMRes(nn.Module):  # noqa: D204
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        return x + self.fc(F.gelu(self.ln(x * gamma + beta)))


class PoseNetLocal(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        M_min: int,
        M_max: int,
        k_max: int,
        r_min: float,
        r_max: float,
        use_conf: bool = False,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.M_min, self.M_max = M_min, M_max
        self.k_max = k_max
        self.r_min, self.r_max = r_min, r_max
        self.use_conf = use_conf
        self.patch_enc = EGNNPatch(feat_dim)
        
        # Stack of local attention + FiLMRes blocks
        self.local_attn_blocks = nn.ModuleList([
            LocalAttn(feat_dim, heads, dropout) for _ in range(layers)
        ])
        self.film_blocks = nn.ModuleList([
            FiLMRes(feat_dim) for _ in range(layers)
        ])

        self.pose_proj = nn.Linear(7, M_max * feat_dim)

        # FiLM MLP outputs gamma and beta for all layers at once
        self.film = nn.Sequential(
            nn.Linear(1 + int(use_conf), 128), nn.SiLU(),
            nn.Linear(128, 2 * layers * feat_dim)
        )

        self.head = nn.Linear(feat_dim, 7)

    def forward(
        self,
        z_t: torch.Tensor,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = xyz.shape
        # 1) sample anchors
        anchors = sample_anchors(mask, xyz, self.M_min, self.M_max)

        # 2) knn neighbors
        nn_idx = knn_neighbors(xyz, anchors, self.k_max)

        # 3) gather local patches
        idx_b = torch.arange(B, device=xyz.device)[:, None, None]
        xyz_n = xyz[idx_b, nn_idx]
        anc_xyz = xyz[idx_b[:, :, 0], anchors]
        xyz_local = xyz_n - anc_xyz.unsqueeze(2)
        feat_n = feat[idx_b, nn_idx]

        # 4) patch encode
        Bm = B * self.M_max
        x_flat = xyz_local.reshape(Bm, self.k_max, 3)
        f_flat = feat_n.reshape(Bm, self.k_max, self.feat_dim)
        _, f_enc = self.patch_enc(x_flat, f_flat)
        f_enc = f_enc.reshape(B, self.M_max, self.k_max, self.feat_dim)

        # 5) initial pose projection to query tokens
        z = z_t.squeeze(-1)
        Q = self.pose_proj(z).view(B, self.M_max, self.feat_dim)

        # 6) prepare FiLM conditioning vector
        cond = [t.unsqueeze(1)]
        if self.use_conf:
            c = conf.unsqueeze(1) if conf is not None else t.new_zeros(B, 1)
            cond.append(c)
        film_out = self.film(torch.cat(cond, dim=1))  # (B, 2 * layers * feat_dim)
        film_out = film_out.view(B, len(self.film_blocks), 2, self.feat_dim)  # (B, layers, 2, feat_dim)

        # 7) stacked local attention + FiLMRes blocks
        for i, (attn_blk, film_blk) in enumerate(zip(self.local_attn_blocks, self.film_blocks)):
            Q = attn_blk(Q, f_enc)  # local attention
            gamma, beta = film_out[:, i].unbind(1)  # (B, feat_dim) each
            Q = film_blk(Q, gamma.unsqueeze(1), beta.unsqueeze(1))  # FiLMRes modulation

        # 8) global token & final head
        Q_global = Q.mean(dim=1)  # (B, feat_dim)
        out = self.head(Q_global)  # (B, 7)
        quat = F.normalize(out[:, 3:], dim=-1)
        pose = torch.cat([out[:, :3], quat], dim=1)
        return pose.unsqueeze(-1)


if __name__ == '__main__':
    B, N, C = 2, 2048, 64
    xyz = torch.randn(B, N, 3)
    feat = torch.randn(B, N, C)
    mask = (torch.rand(B, N, 1) > 0.7)
    z_t = torch.randn(B, 7, 1)
    t = torch.randint(0, 1000, (B,), dtype=torch.float32)

    net = PoseNetLocal(
        feat_dim=C,
        M_min=4, M_max=32,
        k_max=32,
        r_min=0.05, r_max=0.2,
        use_conf=False,
        layers=3
    )
    
    out = net(z_t, xyz, feat, mask, t)
    print('output:', out.shape)  # (B,7,1)
