import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

# ----------------------------
# Quaternion utilities
# ----------------------------

def normalize_quaternions(quats):
    """Normalize all quaternions to unit norm."""
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)

def karcher_mean_quaternion(quaternions, max_iters=100, tol=1e-9):
    """Compute the Karcher (intrinsic) mean of unit quaternions."""
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]
    mean = quaternions[0]
    for _ in range(max_iters):
        rot_mean = R.from_quat(mean)
        delta = np.zeros(3)
        for q in quaternions:
            r = R.from_quat(q)
            relative = r * rot_mean.inv()
            delta += relative.as_rotvec()
        delta /= len(quaternions)
        if np.linalg.norm(delta) < tol:
            break
        mean = (R.from_rotvec(delta) * rot_mean).as_quat()
    return mean

# ----------------------------
# Position utilities
# ----------------------------

def geometric_median(X, eps=1e-5, max_iter=500):
    """Compute the geometric median (L1 minimizer) of 2D array X."""
    y = X.mean(axis=0)
    for _ in range(max_iter):
        D = np.linalg.norm(X - y, axis=1)
        nonzeros = (D != 0)
        if not np.any(nonzeros):
            return y
        W = 1 / D[nonzeros]
        T = (X[nonzeros] * W[:, np.newaxis]).sum(axis=0) / W.sum()
        if np.linalg.norm(y - T) < eps:
            return T
        y = T
    return y

# ----------------------------
# Main summarization function
# ----------------------------

def extract_summary_vector(data, use_geometric_median=True):
    """
    data: np.ndarray of shape (N, 7) with columns [q1, q2, q3, w, x, y, z]
    Returns: 1x7 summary vector [q1, q2, q3, w, x, y, z]
    """
    assert data.shape[1] == 7, "Input must be N x 7"
    
    # Step 1: Split quaternion and position
    quats = data[:, :4]
    positions = data[:, 4:]

    # Step 2: Normalize quaternions
    quats = normalize_quaternions(quats)

    # Step 3: Karcher mean for quaternions
    quat_mean = karcher_mean_quaternion(quats)

    # Step 4: Position summary using mean or geometric median
    if use_geometric_median:
        position_summary = geometric_median(positions)
    else:
        position_summary = positions.mean(axis=0)

    # Step 5: Concatenate and return
    final_vector = np.concatenate([quat_mean, position_summary])
    return final_vector

# ----------------------------
# Test example
# ----------------------------

if __name__ == "__main__":
    # Simulate 10 x 7 pose vectors: [q1, q2, q3, w, x, y, z]
    np.random.seed(42)
    quats = np.random.randn(10, 4)
    positions = np.random.randn(10, 3) * 10 + np.array([5, 5, 5])  # Center around (5,5,5)
    data = np.hstack([quats, positions])

    # Get summary vector
    summary_vector = extract_summary_vector(data, use_geometric_median=False)
    print("Summary vector using MEAN position:\n", summary_vector)

    summary_vector_med = extract_summary_vector(data, use_geometric_median=True)
    print("\nSummary vector using GEOMETRIC MEDIAN position:\n", summary_vector_med)
