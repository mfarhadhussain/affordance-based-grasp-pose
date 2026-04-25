import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps as fps_cluster, knn
from torch_geometric.nn import knn_interpolate
import math 

# ───────────────────────── drop‑path ──────────────────────────
def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p == 0. or not training:
        return x
    keep = 1 - p
    mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                      device=x.device, dtype=x.dtype) < keep
    return x.div(keep) * mask

# ──────────────────── PointNeXt INV block ─────────────────────
class InvBlock(nn.Module):
    def __init__(self, c: int, exp: int = 4, layer_scale: float = 1e-6, drop: float = 0.):
        super().__init__()
        h = c * exp
        self.net = nn.Sequential(
            nn.Conv1d(c, h, 1, bias=False), nn.BatchNorm1d(h), nn.GELU(),
            nn.Conv1d(h, h, 1, groups=h, bias=False), nn.BatchNorm1d(h), nn.GELU(),
            nn.Conv1d(h, c, 1, bias=False), nn.BatchNorm1d(c)
        )
        self.gamma = nn.Parameter(layer_scale * torch.ones(c))
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x) * self.gamma.unsqueeze(-1)
        y = drop_path(y, self.drop, self.training)
        return x + y

# ────────────────────────── FiLM ──────────────────────────────
class FiLM(nn.Module):
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * d)
        )

    def forward(self, t: torch.Tensor):
        gamma, beta = self.mlp(t[:, None]).chunk(2, dim=-1)
        return gamma, beta

# ───────────────────────── Fusion ─────────────────────────────
class Fusion(nn.Module):
    def __init__(self, ca: int, cp: int, cout: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv1d(ca + cp, cout, 1, bias=False),
            nn.BatchNorm1d(cout), nn.GELU()
        )
        self.proj_p = nn.Identity() if cp == cout else nn.Conv1d(cp, cout, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, fa: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        fused = self.mix(torch.cat([fa, fp], dim=1))
        return self.alpha * fused + (1 - self.alpha) * self.proj_p(fp)

# ──────────────── Set‑Abstraction layer (KNN + FPS) ────────────────
class SA(nn.Module):
    def __init__(self, ca: int, cp: int, cout: int, k: int, drop: float):
        super().__init__()
        self.k = k
        self.fuse = Fusion(ca, cp, cout)
        self.film = FiLM(cout)
        self.block = InvBlock(cout, drop=drop)

    def forward(self, xyz: torch.Tensor, fa: torch.Tensor, fp: torch.Tensor, t: torch.Tensor):
        B, N, _ = xyz.shape
        xyz_flat = xyz.reshape(B * N, 3)
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)

        cent_idx = fps_cluster(xyz_flat, batch, ratio=0.5)
        cent_xyz = xyz_flat[cent_idx].view(B, -1, 3)
        batch_cent = batch[cent_idx]

        idx, col = knn(xyz_flat, cent_xyz.view(-1, 3), self.k, batch, batch_cent) 
        grouped_idx = idx.view(B, -1, self.k)

        def gather(feat):  # (B, C, N)
            feat_flat = feat.permute(0, 2, 1).reshape(B * N, -1)
            gathered = feat_flat[idx]
            return gathered.view(B, -1, self.k, feat.shape[1]).permute(0, 3, 1, 2).max(dim=-1)[0]

        fa_pool = gather(fa)
        fp_pool = gather(fp)

        x = self.fuse(fa_pool, fp_pool)
        gamma, beta = self.film(t)
        x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        return cent_xyz, self.block(x)

# ──────────────────── Feature‑Propagation ─────────────────────
class FP(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(cin + cskip, cout, 1, bias=False),
            nn.BatchNorm1d(cout), nn.GELU()
        )

    def forward(self, xyz_src, xyz_tgt, feat_src, feat_skip):
        B, C, N = feat_src.shape
        xyz_src_flat = xyz_src.view(B * xyz_src.shape[1], 3)
        xyz_tgt_flat = xyz_tgt.view(B * xyz_tgt.shape[1], 3)
        feat_src_flat = feat_src.permute(0, 2, 1).reshape(B * N, C)

        batch_src = torch.arange(B, device=xyz_src.device).repeat_interleave(xyz_src.shape[1])
        batch_tgt = torch.arange(B, device=xyz_src.device).repeat_interleave(xyz_tgt.shape[1])

        up = knn_interpolate(
            feat_src_flat, xyz_src_flat, xyz_tgt_flat,
            batch_src, batch_tgt, k=3
        ).view(B, xyz_tgt.shape[1], C).permute(0, 2, 1)

        return self.mlp(torch.cat([up, feat_skip], dim=1))

# ──────────────────── Main AffordanceNet ──────────────────────
class AffordanceNet(nn.Module):
    def __init__(
        self, widths = (64, 128, 256),
        per_dim: int = 64, k: int = 32, drop_depth: float = 0.1
    ):
        super().__init__()
        ca, cp = 1, per_dim
        self.encoder = nn.ModuleList()
        total = len(widths)
        for i, w in enumerate(widths):
            drop = drop_depth * i / max(1, total - 1)
            self.encoder.append(SA(ca, cp, w, k, drop))
            ca = cp = w

        dec_out = (128, 64, 32)[:total]
        dec_in = widths[::-1]
        skips = list(widths[::-1][1:]) + [per_dim]
        self.decoder = nn.ModuleList([
            FP(cin, skip, cout) for cin, skip, cout in zip(dec_in, skips, dec_out)
        ])

        self.head = nn.Sequential(
            nn.Conv1d(dec_out[-1], 16, 1), nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(16, 1, 1)#, nn.Sigmoid()
        )

    def forward(self, afford: torch.Tensor, xyz: torch.Tensor, ppf: torch.Tensor, t: torch.Tensor):
        # print(">>> afford before permute:", afford.shape)
        # if afford.dim() == 4 and afford.size(-1) == 1:
        #     afford = afford.squeeze(-1)   # [B,C,1,1] → [B,C,1]
        # fa = afford.permute(0, 2, 1)

        fa = afford.permute(0, 2, 1)
        fp = ppf.permute(0, 2, 1)
        xyz_s, feat_s = [xyz], []

        for sa in self.encoder:
            xyz, fp = sa(xyz, fa, fp, t)
            fa = fp
            xyz_s.append(xyz)
            feat_s.append(fp)

        feat = fp
        for fpunit, xyz_skip, skip in zip(
            self.decoder, xyz_s[-2::-1], feat_s[-2::-1] + [ppf.permute(0, 2, 1)]
        ):
            feat = fpunit(xyz, xyz_skip, feat, skip)
            xyz = xyz_skip

        return self.head(feat).permute(0, 2, 1)

# ───────────────────────── Test ───────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, C = 8, 2048, 64
    net = AffordanceNet().to(device)
    out = net(
        torch.rand(B, N, 1).to(device),
        torch.randn(B, N, 3).to(device),
        torch.randn(B, N, C).to(device),
        torch.randint(0, 1000, (B,), dtype=torch.float).to(device)
    )
    print(out.shape)  # (B, N, 1)

