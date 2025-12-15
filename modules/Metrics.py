import torch
import open3d as o3d
import numpy as np

def estimate_normals_open3d(pc: torch.Tensor, radius: float = 0.5) -> tuple:
    """
    Estimates normals using Open3D.
    Returns points and normals as torch tensors.
    """
    pc_np = pc.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=75))
    pcd.orient_normals_consistent_tangent_plane(100)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return (
        torch.tensor(points, dtype=torch.float32, device=pc.device),
        torch.tensor(normals, dtype=torch.float32, device=pc.device)
    )

def point_to_plane_distance(src: torch.Tensor, ref: torch.Tensor, radius: float = 0.5) -> torch.Tensor:
    """
    Computes the point-to-plane distance from src to ref using normals on ref.
    """
    ref_pts, ref_normals = estimate_normals_open3d(ref, radius)

    dist = torch.cdist(src, ref_pts, p=2)  # (N1, N2)
    nn_indices = torch.argmin(dist, dim=1)

    nearest_ref_pts = ref_pts[nn_indices]      # (N1, 3)
    nearest_normals = ref_normals[nn_indices]  # (N1, 3)
    diff = src - nearest_ref_pts

    ptp = torch.abs(torch.sum(diff * nearest_normals, dim=1))  # (N1,)
    return ptp.mean()

def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Chamfer Distance between two point clouds.
    """
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)
    dist = torch.norm(diff, dim=2)

    min_dist_pc1, _ = dist.min(dim=1)
    min_dist_pc2, _ = dist.min(dim=0)

    return min_dist_pc1.mean() + min_dist_pc2.mean()

def occupancy_grid_iou(pc1: torch.Tensor, pc2: torch.Tensor, voxel_size=(0.25, 0.25, 0.25)) -> torch.Tensor:
    """
    Computes IoU between two voxelized point clouds.
    """
    def to_voxel_indices(points, voxel_size):
        min_coords = points.min(dim=0).values
        voxel_indices = ((points - min_coords) / torch.tensor(voxel_size, device=points.device)).floor().long()
        return voxel_indices

    vox1 = to_voxel_indices(pc1, voxel_size)
    vox2 = to_voxel_indices(pc2, voxel_size)

    set1 = set(map(tuple, vox1.cpu().numpy()))
    set2 = set(map(tuple, vox2.cpu().numpy()))

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    iou = len(intersection) / (len(union) + 1e-6)

    return torch.tensor(iou, dtype=torch.float32, device=pc1.device)

def _voxel_hist(
    points: torch.Tensor,
    resolution: int,
    eps: float = 1e-6,
    min_coord: torch.Tensor = None,
    max_coord: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute normalized occupancy histogram for a batch of point clouds.

    Args:
        points: (B, N, 3) or (N, 3) point cloud tensor.
        resolution: voxels per axis.
        eps: small constant for numerical stability.
        min_coord: optional (B,1,3) tensor of minimum bounds.
        max_coord: optional (B,1,3) tensor of maximum bounds.

    Returns:
        hist: (B, V) tensor where V = resolution**3, rows sum to 1.
    """
    # Handle single-cloud inputs
    if points.dim() == 2:
        points = points.unsqueeze(0)

    B, P, _ = points.shape

    # Compute bounds if not provided
    if min_coord is None or max_coord is None:
        min_coord = points.amin(dim=1, keepdim=True)
        max_coord = points.amax(dim=1, keepdim=True)

    # Normalize to [0,1]
    normed = (points - min_coord) / (max_coord - min_coord + eps)
    # Compute voxel indices
    idx = (normed * resolution).floor().long().clamp(0, resolution - 1)
    flat = idx[..., 0] * (resolution**2) + idx[..., 1] * resolution + idx[..., 2]

    # Build histograms
    hists = []
    for b in range(B):
        counts = torch.bincount(flat[b], minlength=resolution**3)
        hists.append(counts.float())
    hist = torch.stack(hists, dim=0)
    # Normalize
    hist = hist / (hist.sum(dim=1, keepdim=True) + eps)

    return hist


def soft_iou(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    resolution: int = 32,
    eps: float = 1e-6
    ) -> torch.Tensor:
    """
    Compute Soft IoU between two point clouds based on occupancy histograms.

    Args:
        pred_points: (B, N, 3) or (N, 3) predicted points.
        gt_points: (B, M, 3) or (M, 3) ground-truth points.
        resolution: number of voxels per axis.
        eps: small constant to avoid division by zero.

    Returns:
        torch.Tensor: scalar Soft IoU in [0,1], averaged over batch.
    """
    # Compute histograms
    hist_pred = _voxel_hist(pred_points, resolution, eps)
    hist_gt = _voxel_hist(gt_points, resolution, eps)

    # Compute numerator and denominator
    intersection = torch.sum(torch.min(hist_pred, hist_gt), dim=1)
    union = torch.sum(torch.max(hist_pred, hist_gt), dim=1) + eps

    # Soft IoU per batch
    soft_iou_vals = intersection / union

    # Return mean over batch (or single value if no batch)
    return soft_iou_vals.mean()