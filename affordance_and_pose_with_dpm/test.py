#!/usr/bin/env python3
"""
Unit tests and sanity checks for sample_anchors, radius_neighbors, knn_neighbors, and PoseNetLocal.
"""

import torch
import sys
import os

# allow imports from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

# Adjust this import based on where your code lives
from affordance_pose.pose import (
    sample_anchors,
    PoseNetLocal,
    knn_neighbors
)

def test_sample_anchors():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N = 4, 1024
    M_min, M_max = 4, 32

    xyz = torch.randn(B, N, 3, device=device)
    for density in [0.0, 0.2, 0.8, 1.0]:
        mask = (torch.rand(B, N, 1, device=device) < density)
        anchors = sample_anchors(mask, xyz, M_min, M_max)
        assert anchors.shape == (B, M_max), f"Wrong shape: {anchors.shape}"
        assert anchors.dtype == torch.long
        assert anchors.min() >= 0 and anchors.max() < N, "Anchor index OOB"
    print("✅ sample_anchors passed all tests.")

def test_radius_neighbors():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, C = 4, 512, 16
    M_max = 32
    k_max = 8
    r_min, r_max = 0.0, 0.05

    xyz = torch.randn(B, N, 3, device=device)
    anchors = torch.randint(0, N, (B, M_max), device=device)
    for valid_frac in [0.0, 0.5, 1.0]:
        valid = (torch.rand(B, N, 1, device=device) < valid_frac)
        neigh = radius_neighbors(xyz, anchors, k_max, r_min, r_max, valid)
        assert neigh.shape == (B, M_max, k_max), f"Wrong shape: {neigh.shape}"
        assert neigh.dtype == torch.long
        assert neigh.min() >= 0 and neigh.max() < N, "Neighbor index OOB"
    print("✅ radius_neighbors passed all tests.")

def test_knn_neighbors():
    torch.manual_seed(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, C = 4, 512, 3
    M = 32
    k_max = 16

    xyz = torch.randn(B, N, C, device=device)
    anchors = torch.randint(0, N, (B, M), device=device)
    neigh = knn_neighbors(xyz, anchors, k_max)
    assert isinstance(neigh, torch.Tensor), "Output must be a tensor"
    assert neigh.dtype == torch.long, f"Expected dtype torch.long, got {neigh.dtype}"
    assert neigh.shape == (B, M, k_max), f"Expected shape {(B, M, k_max)}, got {neigh.shape}"
    assert neigh.min() >= 0 and neigh.max() < N, "Neighbor indices out of range"
    print("✅ knn_neighbors passed all tests.")

def test_posenetlocal_forward():
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, C = 2, 2048, 64
    M_min, M_max, k_max = 4, 32, 16
    r_min, r_max = 0.05, 0.2

    xyz  = torch.randn(B, N, 3, device=device)
    feat = torch.randn(B, N, C, device=device)
    mask = (torch.rand(B, N, 1, device=device) > 0.7)
    z_t  = torch.randn(B, 7, 1, device=device)
    t    = torch.randint(0, 1000, (B,), dtype=torch.float32, device=device)

    net = PoseNetLocal(
        feat_dim=C,
        M_min=M_min, M_max=M_max,
        k_max=k_max,
        r_min=r_min, r_max=r_max,
        use_conf=False, layers=3, heads=4, dropout=0.1
    ).to(device)

    out = net(z_t, xyz, feat, mask, t)
    assert out.shape == (B, 7, 1), f"Expected output (B,7,1), got {out.shape}"
    print("✅ PoseNetLocal forward pass succeeded.")

if __name__ == "__main__":
    print("Running PoseNetLocal sanity tests...")
    test_sample_anchors()
    # test_radius_neighbors()
    test_knn_neighbors()
    test_posenetlocal_forward()
    print("🎉 All PoseNetLocal tests passed!")
