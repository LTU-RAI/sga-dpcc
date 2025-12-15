import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelReconstructionLoss(nn.Module):
    def __init__(self, grid_size=64, normalize=True, padding=0.05):
        """
        Args:
            grid_size (int): Number of voxels per axis (grid_size^3 total).
            normalize (bool): Whether to normalize point clouds to [-1, 1]^3 before voxelizing.
            padding (float): Extra margin added during normalization.
        """
        super().__init__()
        self.grid_size = grid_size
        self.normalize = normalize
        self.padding = padding
        self.bce = nn.BCELoss()

    def normalize_points(self, points, mask):
        """
        Normalize each point cloud to fit into [-1, 1]^3 based on its bbox.
        Args:
            points: (B, N, 3)
            mask: (B, N) or None
        Returns:
            normalized: (B, N, 3)
        """
        B, N, _ = points.shape
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
            points_masked = points * mask_exp
            mask_exp_f = mask_exp.float()  # Convert to float
            min_coords = (points_masked + (1.0 - mask_exp_f) * 1e9).min(dim=1).values
            max_coords = (points_masked + (1.0 - mask_exp_f) * -1e9).max(dim=1).values
        else:
            min_coords = points.min(dim=1).values  # (B, 3)
            max_coords = points.max(dim=1).values  # (B, 3)

        center = (min_coords + max_coords) / 2
        scale = (max_coords - min_coords).max(dim=1).values.unsqueeze(-1) / 2 + self.padding  # (B, 1)
        normalized = (points - center.unsqueeze(1)) / scale.unsqueeze(1).clamp(min=1e-6)
        return normalized.clamp(-1, 1)

    def voxelize(self, points, mask=None):
        """
        Convert a batch of point clouds to voxel occupancy.
        Args:
            points: (B, N, 3)
            mask: (B, N) or None
        Returns:
            voxels: (B, grid, grid, grid), float in [0,1]
        """
        B, N, _ = points.shape
        grid = self.grid_size

        # Normalize to [-1, 1] cube if enabled
        if self.normalize:
            points = self.normalize_points(points, mask)

        # Map [-1, 1] → [0, grid-1]
        coords = ((points + 1) / 2 * (grid - 1)).long()
        coords = coords.clamp(0, grid - 1)

        voxels = torch.zeros(B, grid, grid, grid, device=points.device)
        for b in range(B):
            valid_mask = mask[b] if mask is not None else torch.ones(N, device=points.device, dtype=torch.bool)
            inds = coords[b][valid_mask]
            voxels[b, inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
        return voxels

    def forward(self, pred_points, target_points, target_mask=None):
        """
        Args:
            pred_points: (B, N_out, 3)
            target_points: (B, N_in, 3)
            target_mask: (B, N_in) or None
        Returns:
            loss: scalar
        """
        vox_pred = self.voxelize(pred_points)         # shape: (B, grid, grid, grid)
        vox_target = self.voxelize(target_points, target_mask)
        loss = self.bce(vox_pred, vox_target)
        loss = loss * 1.0   # ensure it's a tensor with grad_fn
        loss = (vox_pred - vox_target).pow(2).mean()  # MSE loss (has grad)
        return loss
