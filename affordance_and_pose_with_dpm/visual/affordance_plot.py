import open3d as o3d
import numpy as np
from typing import Optional

def visualize_affordance(
    points: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask:   Optional[np.ndarray] = None,
    offset:    float = 3.0
):
    """
    Visualize point cloud affordance masks, with clear separation when both are given.

    If only gt_mask is given, shows one cloud colored by gt_mask.
    If only pred_mask is given, shows one cloud colored by pred_mask.
    If both are given, shows two clouds side-by-side:
      - Left (shifted -offset/2):  GT
      - Right (shifted +offset/2): Pred

    Colors: mask==1 → blue, mask==0 → red.
    """
    N = len(points)
    assert points.shape[1] == 3
    if gt_mask is None and pred_mask is None:
        raise ValueError("At least one of pred_mask or gt_mask must be provided.")
    if gt_mask is not None:
        assert len(gt_mask) == N
    if pred_mask is not None:
        assert len(pred_mask) == N

    def make_pcd(pts, mask):
        m = mask.flatten()
        colors = np.zeros((N,3))
        colors[m == 1] = [0, 0, 1]
        colors[m == 0] = [1, 0, 0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    geometries = []

    # only one mask → no shifts
    if (gt_mask is None) ^ (pred_mask is None):
        mask = gt_mask if gt_mask is not None else pred_mask
        pcd = make_pcd(points, mask)
        geometries.append(pcd)
        title = "Affordance Mask"

    else:
        # both masks → shift left/right
        half = offset / 2.0

        # GT on left
        pts_gt = points.copy()
        pts_gt[:,0] -= half
        pcd_gt = make_pcd(pts_gt, gt_mask)
        geometries.append(pcd_gt)
        frame_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        frame_gt.translate([-half,0,0])
        geometries.append(frame_gt)

        # Pred on right
        pts_pr = points.copy()
        pts_pr[:,0] += half
        pcd_pr = make_pcd(pts_pr, pred_mask)
        geometries.append(pcd_pr)
        frame_pr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        frame_pr.translate([+half,0,0])
        geometries.append(frame_pr)

        title = "GT (left) vs. Pred (right)"

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=800, height=600
    )


def generate_random_data(N=5000, seed=42):
    """Generate random 3D points and two random binary masks."""
    np.random.seed(seed)
    points = np.random.uniform(-1, 1, size=(N, 3)).astype(np.float32)
    # Ground truth: 30% of points are ‘1’
    gt_mask = (np.random.rand(N, 1) < 0.3).astype(np.uint8)
    # Prediction: 25% of points are ‘1’, shifted randomly
    pred_mask = (np.random.rand(N, 1) < 0.25).astype(np.uint8)
    return points, gt_mask, pred_mask

if __name__ == "__main__":
    pts, gt, pr = generate_random_data(N=10000)

    # print("1) Visualizing only ground-truth mask...")
    
    # visualize_affordance(pts, gt_mask=gt)

    # input("Press Enter to continue to prediction-only view…\n")

    # print("2) Visualizing only prediction mask...")
    # # visualize_overlay(pts, pred_mask=pr)
    # visualize_affordance(pts, pred_mask=pr)

    # input("Press Enter to continue to overlay view…\n")

    print("3) Visualizing overlay of GT vs. Pred with legend...")
    # visualize_overlay(pts, pred_mask=pr, gt_mask=gt)
    visualize_affordance(pts, pred_mask=pr, gt_mask=gt)

    print("Done.")
