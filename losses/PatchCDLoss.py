import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F


class ChamferDistanceLossWithMask(nn.Module):
    def __init__(self):
        super(ChamferDistanceLossWithMask, self).__init__()

    @autocast(device_type='cuda', enabled=True)
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                target_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the bidirectional Chamfer Distance between two batches of point clouds.
        Assumes that pred (the reconstruction) always has full points (i.e. no padding) and
        uses the provided target_mask for the target point cloud.
        
        Args:
            pred: (B, N_out, 3) predicted points (decoder output).
            target: (B, N_in, 3) ground truth points.
            pred_mask: (B, N_pred) Boolean mask for predicted points. This is overridden to all ones.
            target_mask: (B, N_in) Boolean mask (True for valid points in target).
            
        Returns:
            loss: Mean Chamfer distance over the batch.
        """
        B = pred.shape[0]
        total_loss = 0.0
        # Override pred_mask to be all ones because decoder always outputs full points.
        
        for i in range(B):
            # Select only valid target points.
            p_valid = pred[i]  # (N_out, 3); all predicted points are valid.
            t_valid = target[i][target_mask[i].bool()]  # (n_valid, 3)
            # If no valid target points, skip this sample.
            if t_valid.shape[0] == 0:
                continue
            # Compute pairwise Euclidean distances.
            diff = p_valid.unsqueeze(1) - t_valid.unsqueeze(0)  # (N_out, n_valid, 3)
            dist = torch.norm(diff, dim=2)  # (N_out, n_valid)
            
            # For each predicted point, find the minimum distance to any target point.
            min_dist_p, _ = torch.min(dist, dim=1)  # (N_out,)
            # For each target point, find the minimum distance to any predicted point.
            min_dist_t, _ = torch.min(dist, dim=0)  # (n_valid,)
            
            loss_i = min_dist_p.mean() + min_dist_t.mean()
            total_loss += loss_i
        
        return total_loss / B

# ---------------- Test Function ----------------
def test_chamfer_loss_with_mask():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N_out, N_in = 2, 2048, 1000
    pred = torch.randn(B, N_out, 3, device=device)
    target = torch.randn(B, N_in, 3, device=device)
    # For predicted points, we'll use a dummy mask (but it will be overridden).
    # For target, assume the last 200 points are padded.
    target_mask = torch.ones(B, N_in, dtype=torch.bool, device=device)
    target_mask[:, 800:] = False
    
    loss_fn = ChamferDistanceLossWithMask()
    loss_value = loss_fn(pred, target, target_mask)
    print(f"Masked Chamfer Loss: {loss_value.item():.6f}")


import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssistedChamferDistanceLoss(nn.Module):
    def __init__(self,
                 coarse_weight=[1.0, 0.01, 20],
                 lambda_mask_final_schedule=[1.0, 0.01, 20],
                 lambda_mask_coarse_schedule=[1.0, 0.01, 10],
                 repulsion_weight=[1.0, 0.01, 20],
                 repulsion_threshold=0.1,
                 density_weight=[1.0, 0.1, 20],
                 grid_size=32,
                 mask_threshold=0.5,
                 max_points=2048,
                 verbose=False):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.lambda_mask_final_schedule = lambda_mask_final_schedule
        self.lambda_mask_coarse_schedule = lambda_mask_coarse_schedule
        self.repulsion_weight = repulsion_weight
        self.repulsion_threshold = repulsion_threshold
        self.density_weight = density_weight
        self.grid_size = grid_size
        self.mask_threshold = mask_threshold
        self.min_valid_points = max_points // 4
        self.verbose = verbose

    def _get_current_value(self, schedule, epoch):
        start, end, lifetime = schedule
        if epoch <= lifetime and lifetime > 0:
            return start - (start - end) * (epoch / lifetime)
        elif epoch > lifetime and lifetime > 0:
            return end
        else:
            return 0

    def chamfer_loss(self, pred_points, target_points, pred_mask=None):
        if pred_mask is not None:
            valid_idx = (pred_mask.squeeze(-1) > self.mask_threshold)
            valid_pred = pred_points[valid_idx]
            if valid_pred.shape[0] == 0:
                valid_pred = pred_points
        else:
            valid_pred = pred_points

        diff = valid_pred.unsqueeze(1) - target_points.unsqueeze(0)
        dist = torch.norm(diff, dim=2)
        min_dist_pred, _ = torch.min(dist, dim=1)
        min_dist_target, _ = torch.min(dist, dim=0)
        return min_dist_pred.mean() + min_dist_target.mean()

    def repulsion_loss(self, pred_points, threshold):
        N = pred_points.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=pred_points.device)
        dists = torch.cdist(pred_points, pred_points, p=2)
        diag = torch.eye(N, device=pred_points.device) * 1e9
        dists = dists + diag
        min_dists, _ = torch.min(dists, dim=1)
        loss = torch.clamp(threshold - min_dists, min=0.0).mean()
        return loss

    def voxel_density_loss(self, pred_points, target_points, grid_size=None):
        """
        Computes a voxel-based density loss after normalizing the input points to [-1, 1].
        This normalization is done per sample by computing the min and max from the union of
        predicted and target points. The occupancy grid is then computed in a fixed volume
        of [-1, 1]^3.
        
        Args:
            pred_points: (N, 3) tensor of predicted points.
            target_points: (M, 3) tensor of target points.
            grid_size: Number of voxels per axis. Defaults to self.grid_size.
            
        Returns:
            Scalar density loss.
        """
        if grid_size is None:
            grid_size = self.grid_size

        # Combine points to compute the dynamic range.
        all_points = torch.cat([pred_points, target_points], dim=0)
        low = all_points.min(dim=0).values
        high = all_points.max(dim=0).values
        # Normalize both predicted and target points to [-1, 1]
        normalized_pred = 2 * (pred_points - low) / ((high - low).clamp(min=1e-6)) - 1
        normalized_target = 2 * (target_points - low) / ((high - low).clamp(min=1e-6)) - 1

        # In the normalized space, the fixed range is [-1, 1] so the bin size is:
        bin_size = 2.0 / grid_size  # since 1 - (-1) = 2

        def occupancy_grid(points):
            # Map normalized points in [-1, 1] to voxel indices in [0, grid_size-1]
            idx = (((points + 1) / 2) * grid_size).clamp(0, grid_size - 1).long()  # (N, 3)
            # Flatten 3D indices to a single index.
            flat_idx = idx[:, 0] * (grid_size ** 2) + idx[:, 1] * grid_size + idx[:, 2]
            counts = torch.bincount(flat_idx, minlength=grid_size ** 3).float()
            return counts.reshape(grid_size, grid_size, grid_size)

        pred_grid = occupancy_grid(normalized_pred)
        target_grid = occupancy_grid(normalized_target)

        # Avoid zeros by adding a small constant and then normalize.
        pred_grid = pred_grid + 1e-6
        target_grid = target_grid + 1e-6
        pred_grid = pred_grid / pred_grid.sum()
        target_grid = target_grid / target_grid.sum()

        loss = torch.mean((pred_grid - target_grid) ** 2)
        return loss


    @staticmethod
    def farthest_point_sampling(points, n_samples):
        device = points.device
        N = points.shape[0]
        indices = torch.zeros(n_samples, dtype=torch.long, device=device)
        indices[0] = torch.randint(0, N, (1,), device=device)
        distances = torch.full((N,), float('inf'), device=device)
        for i in range(1, n_samples):
            new_point = points[indices[i - 1]].unsqueeze(0)  # (1,3)
            dist = torch.norm(points - new_point, dim=1)
            distances = torch.minimum(distances, dist)
            indices[i] = torch.argmax(distances)
        return indices

    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, 
                pred: torch.Tensor,       # (B, N_final, 3)
                coarse: torch.Tensor,     # (B, N_coarse, 3)
                coarse_conf: torch.Tensor,  # (B, N_coarse, 1)
                final_conf: torch.Tensor,   # (B, N_final, 1)
                target: torch.Tensor,     # (B, N_target, 3)
                target_mask: torch.Tensor,  # (B, N_target)
                epoch: int):
        current_lambda_mask_final = self._get_current_value(self.lambda_mask_final_schedule, epoch)
        current_lambda_mask_coarse = self._get_current_value(self.lambda_mask_coarse_schedule, epoch)
        current_coarse_weight = self._get_current_value(self.coarse_weight, epoch)
        current_repulsion_weight = self._get_current_value(self.repulsion_weight, epoch)
        current_density_weight = self._get_current_value(self.density_weight, epoch)

        B = pred.shape[0]
        total_fine_loss = total_coarse_loss = total_mask_loss = total_repulsion_loss = total_density_loss = 0.0
        total_weight = 0.0
        
        # Variables to accumulate timing information (in seconds).
        time_fine = time_coarse = time_mask = time_rep = time_density = 0.0

        for i in range(B):
            t_valid = target[i][target_mask[i].bool()]
            n_valid = t_valid.shape[0]
            if n_valid == 0:
                continue

            # Fine Loss Timing
            start = time.perf_counter()
            p_fine = pred[i]
            fine_loss = self.chamfer_loss(p_fine, t_valid, pred_mask=final_conf[i])
            time_fine += time.perf_counter() - start

            # Coarse Loss Timing (using FPS)
            start = time.perf_counter()
            p_coarse = coarse[i]
            num_coarse = p_coarse.shape[0]
            if t_valid.shape[0] > num_coarse:
                idx = self.farthest_point_sampling(t_valid, num_coarse)
                t_valid_coarse = t_valid[idx]
            else:
                t_valid_coarse = t_valid
            coarse_loss = self.chamfer_loss(p_coarse, t_valid_coarse, pred_mask=coarse_conf[i])
            time_coarse += time.perf_counter() - start

            # Mask Loss Timing
            start = time.perf_counter()
            loss_mask_final = F.binary_cross_entropy_with_logits(final_conf[i].squeeze(-1), target_mask[i].float())
            target_mask_coarse = F.interpolate(target_mask[i].unsqueeze(0).unsqueeze(0).float(),
                                               size=coarse_conf[i].shape[0],
                                               mode='linear', align_corners=False)
            target_mask_coarse = target_mask_coarse.squeeze(0).squeeze(0)
            loss_mask_coarse = F.binary_cross_entropy_with_logits(coarse_conf[i].squeeze(-1), target_mask_coarse)
            mask_loss = current_lambda_mask_final * loss_mask_final + current_lambda_mask_coarse * loss_mask_coarse
            time_mask += time.perf_counter() - start

            # Repulsion Loss Timing
            if current_repulsion_weight > 0:
                start = time.perf_counter()
                rep_loss = self.repulsion_loss(p_fine, self.repulsion_threshold)
                time_rep += time.perf_counter() - start
            else:
                rep_loss = torch.tensor(0.0, device=pred.device)
                time_rep += 0.0
            # Density Loss Timing
            if current_density_weight > 0:
                start = time.perf_counter()
                dens_loss = self.voxel_density_loss(p_fine, t_valid)
                time_density += time.perf_counter() - start
            else:
                dens_loss = torch.tensor(0.0, device=pred.device)
                time_density += 0.0
            # Compute weight based on number of valid points.
            weight = 1.0 if n_valid > self.min_valid_points else n_valid / self.min_valid_points

            total_fine_loss += weight * fine_loss
            total_coarse_loss += weight * coarse_loss
            total_mask_loss += weight * mask_loss
            total_repulsion_loss += weight * rep_loss
            total_density_loss += weight * dens_loss
            total_weight += weight

        if total_weight == 0.0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        avg_fine_loss = total_fine_loss / total_weight
        avg_coarse_loss = total_coarse_loss / total_weight
        avg_mask_loss = total_mask_loss / total_weight
        avg_repulsion_loss = total_repulsion_loss / total_weight
        avg_density_loss = total_density_loss / total_weight

        total_loss = (avg_fine_loss +
                      current_coarse_weight * avg_coarse_loss +
                      avg_mask_loss +
                      current_repulsion_weight * avg_repulsion_loss +
                      current_density_weight * avg_density_loss)

        if self.verbose:
            # Print average time per sample (in milliseconds) for each loss term.
            avg_time_fine = (time_fine / B) * 1000
            avg_time_coarse = (time_coarse / B) * 1000
            avg_time_mask = (time_mask / B) * 1000
            avg_time_rep = (time_rep / B) * 1000
            avg_time_density = (time_density / B) * 1000
            print(f"Loss Timing per sample (ms): Fine: {avg_time_fine:.2f}, Coarse: {avg_time_coarse:.2f}, "
                  f"Mask: {avg_time_mask:.2f}, Repulsion: {avg_time_rep:.2f}, Density: {avg_time_density:.2f}")

        return (total_loss, avg_fine_loss, 
                current_coarse_weight * avg_coarse_loss, 
                avg_mask_loss, 
                current_repulsion_weight * avg_repulsion_loss,
                current_density_weight * avg_density_loss)


# ---------------- Test Function ----------------
def test_chamfer_loss_with_mask_and_repulsion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N_out, N_in = 2, 2048, 1000
    pred = torch.randn(B, N_out, 3, device=device)
    target = torch.randn(B, N_in, 3, device=device)
    # Assume last 200 target points are padded.
    target_mask = torch.ones(B, N_in, dtype=torch.bool, device=device)
    target_mask[:, 800:] = False
    
    loss_fn = AssistedChamferDistanceLoss(repulsion_weight=0.1, repulsion_threshold=0.1)
    loss_value = loss_fn(pred, target, target_mask)
    print(f"Chamfer+Repulsion Loss: {loss_value.item():.6f}")

# Uncomment below to run the test.
# if __name__ == "__main__":
#     test_chamfer_loss_with_mask_and_repulsion()
