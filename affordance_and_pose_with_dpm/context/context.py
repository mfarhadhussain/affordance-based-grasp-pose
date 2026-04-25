#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch_cluster import fps as fps_cluster, knn
from torch_geometric.nn import knn_interpolate 

def _sanitize(t: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(t, nan=val, posinf=val, neginf=val)

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        D: int,
        k: int,
        sample_ratio: float,
        feature_fps: bool = False,
        num_heads: int = 4,
        use_mp: bool = False,
        random_start: bool = True,
        jitter: bool = True,
        jitter_strength: float = 1e-3,
    ):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.k = k
        self.sample_ratio = float(sample_ratio)
        self.feature_fps = feature_fps
        self.random_start = random_start
        self.jitter = jitter
        self.jitter_strength = jitter_strength
        self.use_mp = use_mp

        # cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=C_in, num_heads=num_heads,
            kdim=D, vdim=D, batch_first=True
        )

        # patch MLP + pool
        self.mlp = nn.Sequential(
            nn.Linear(C_in, C_in, bias=False),
            nn.ReLU(inplace=True)
        )

        # residual FFN
        self.norm1 = nn.LayerNorm(C_in)
        self.ffn = nn.Sequential(
            nn.Linear(C_in, C_in*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C_in*4, C_in, bias=False)
        )
        
        self.norm2 = nn.LayerNorm(C_in) 

        # expand channels if needed
        self.expand = nn.Identity() if C_out == C_in else nn.Linear(C_in, C_out)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor, text: torch.Tensor, is_token_level: bool = True):
        B, N, C = feats.shape
        xyz = _sanitize(xyz)
        feats = _sanitize(feats)
        device = xyz.device

        # 1) number of samples
        S = max(int(N * self.sample_ratio), 1)
        S = min(S, N)

        # 2) flatten tensors safely
        xyz_flat =  xyz.contiguous().view(B * N, 3)
        feats_flat = feats.contiguous().view(B * N, C).view(B * N, C)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        fps_criteria_flat = xyz_flat
        if self.feature_fps: 
            fps_criteria_flat = feats_flat 
        if self.jitter:
            fps_criteria_flat = fps_criteria_flat + torch.randn_like(fps_criteria_flat) * self.jitter_strength
            fps_criteria_flat = _sanitize(fps_criteria_flat)


        # 1) FPS sampling via torch_cluster
        ratio = self.sample_ratio
        idx_flat = fps_cluster(xyz_flat, batch_idx, ratio=ratio, random_start=False)

        new_xyz_flat = xyz_flat[idx_flat]     # (B*S, 3)
        new_xyz = new_xyz_flat.view(B, S, 3)  # (B, S, 3)
        new_feats_flat = feats_flat[idx_flat]
        new_feats      = new_feats_flat.view(B, S, C) 

        new_fps_criteria_flat = fps_criteria_flat[idx_flat]

        batch_new = torch.arange(B, device=device).repeat_interleave(S)

        # 2) KNN grouping via torch_cluster.knn 
        row, col = knn(fps_criteria_flat, new_fps_criteria_flat, self.k, batch_idx, batch_new) 


        # grouped neighbor features: (B, S, k, C)
        grouped = feats_flat[row].view(B, S, self.k, C)
        # queries: (B*S, k, C)
        q = grouped.view(B * S, self.k, C)
        q = _sanitize(q)
        

        # 7) safely construct kv
        if is_token_level:
            L = text.size(1)
            kv = text.unsqueeze(1).expand(-1, S, -1, -1).reshape(B*S, L, -1)
        else:
            kv = text.unsqueeze(1).expand(-1, S, -1).reshape(B*S, 1, -1)

        kv = _sanitize(kv)

        # 8) stable cross-attention
        # Corrected snippet:
        if self.use_mp:
            with torch.cuda.amp.autocast(enabled=True):
                attn_out, _ = self.attn(q.half(), kv.half(), kv.half(), need_weights=False)
            attn_out = attn_out.float()
        else:
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)


        # 9) safe pooling
        patch = self.mlp(attn_out)
        pooled, _ = patch.max(dim=1)
        pooled = _sanitize(pooled)

        # 10) stable residual + FFN
        x = new_feats + pooled.view(B, S, C)
        x = self.norm1(_sanitize(x))
        x_ffn = self.ffn(x)
        x_ffn = _sanitize(x_ffn)
        new_feats = self.norm2(_sanitize(x + x_ffn))

        # 11) expand channels safely
        return new_xyz, self.expand(new_feats)


class PointTextEncoderDecoder(nn.Module):
    def __init__(
        self,
        point_dim: int,
        text_dim: int,
        sample_ratios=(0.5, 0.25, 0.125),
        ks=(16, 32, 64),
        num_heads: int = 4,
        use_mp: bool = False,
        random_start: bool = True,
        jitter: bool = True,
        jitter_strength: float = 1e-3,
        token_level: bool = True
    ):
        super().__init__()
        C = point_dim
        dims = [C, 2*C, 4*C, 8*C]
        L = len(sample_ratios)

        # spatial FPS for all but last layer, feature FPS for last
        feature_schedule = [False] * (L - 1) + [True]

        self.enc_blocks = nn.ModuleList([
            CrossAttentionBlock(
                C_in=dims[i], C_out=dims[i+1], D=text_dim,
                k=ks[i], sample_ratio=sample_ratios[i],
                feature_fps=feature_schedule[i],
                num_heads=num_heads, use_mp=use_mp,
                random_start=random_start,
                jitter=jitter, jitter_strength=jitter_strength
            )
            for i in range(L)
        ])

        self.bottleneck = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(inplace=True)
        )

        dec_ins  = [8*C, 4*C, 2*C]
        dec_outs = [4*C, 2*C,   C]
        self.dec_blocks = nn.ModuleList()
        prev = dims[-1]
        for s_dim, o_dim in zip(dec_ins, dec_outs):
            self.dec_blocks.append(nn.Sequential(
                nn.Linear(prev + s_dim, o_dim),
                nn.ReLU(inplace=True)
            ))
            prev = o_dim

        self.final_proj = nn.Identity()
        self.token_level = token_level

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor, text: torch.Tensor):
        B, N, _ = feats.shape
        orig_xyz = xyz
        skips = []
        cur_xyz, cur_feats = xyz, feats

        # encoder
        for enc in self.enc_blocks:
            new_xyz, new_feats = enc(
                cur_xyz, cur_feats, text,
                is_token_level=self.token_level
            )
            skips.append((new_xyz, new_feats))
            cur_xyz, cur_feats = new_xyz, new_feats

        # bottleneck
        x = self.bottleneck(cur_feats)

        # decoder with skip connections
        for (skip_xyz, skip_feats), dec in zip(reversed(skips), self.dec_blocks):
            B2, S2, _ = skip_feats.shape
            x_flat = x.reshape(B2 * x.shape[1], -1)
            cur_xyz_flat = cur_xyz.reshape(B2 * x.shape[1], 3)
            batch_cur = torch.arange(B2, device=xyz.device).repeat_interleave(x.shape[1])
            skip_xyz_flat = skip_xyz.reshape(B2 * S2, 3)
            batch_skip    = torch.arange(B2, device=xyz.device).repeat_interleave(S2)

            up = knn_interpolate(
                x_flat, cur_xyz_flat,
                skip_xyz_flat, batch_cur, batch_skip, k=3
            ).reshape(B2, S2, -1)

            x = dec(torch.cat([up, skip_feats], dim=-1))
            cur_xyz, cur_feats = skip_xyz, x

        # final upsample to original N
        x_flat = x.reshape(B * cur_feats.shape[1], -1)
        cur_xyz_flat = cur_xyz.reshape(B * cur_feats.shape[1], 3)
        batch_cur = torch.arange(B, device=orig_xyz.device).repeat_interleave(cur_feats.shape[1])
        orig_xyz_flat = orig_xyz.reshape(B * N, 3)
        batch_orig    = torch.arange(B, device=orig_xyz.device).repeat_interleave(N)

        out_flat = knn_interpolate(
            x_flat, cur_xyz_flat,
            orig_xyz_flat, batch_cur, batch_orig, k=3
        )
        return self.final_proj(out_flat.reshape(B, N, -1))


class FinetuningPointText(nn.Module):
    """Binary segmentation head on point–text features."""
    def __init__(self, point_feats_dim: int):
        super().__init__()
        hidden = max(point_feats_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Conv1d(point_feats_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, point_text_feat: torch.Tensor) -> torch.Tensor:
        x = point_text_feat.transpose(1, 2)
        return self.net(x).squeeze(1)


if __name__ == "__main__":
    # smoke test
    B, N, C, L, D = 2, 1024, 64, 16, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.rand(B, N, 3, device=device)
    feats = torch.rand(B, N, C, device=device)
    ctx_tokens = torch.rand(B, D, device=device)

    model = PointTextEncoderDecoder(
        point_dim=C,
        text_dim=D,
        sample_ratios=(0.5, 0.25, 0.125),
        ks=(16, 32, 64),
        num_heads=4,
        use_mp=False,
        random_start=True,
        jitter=True,
        jitter_strength=1e-3,
        token_level=False
    ).to(device)

    out = model(xyz, feats, ctx_tokens)
    print("Output shape (B,N,C):", out.shape)  
