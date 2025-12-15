import torch
import torch.nn as nn
from torch.amp import autocast

class EarthMoverDistanceLossWithMask(nn.Module):
    def __init__(self, eps=0.1, max_iter=50, max_points_loss=1024, use_local=True):
        """
        Args:
            eps (float): Base entropy regularization parameter.
            max_iter (int): Number of Sinkhorn iterations.
            max_points_loss (int): Maximum number of valid points per cloud to use in the loss.
            use_local (bool): If True, subtract the centroid from each cloud before computing cost.
        """
        super(EarthMoverDistanceLossWithMask, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.max_points_loss = max_points_loss
        self.use_local = use_local

    @autocast(device_type='cuda', enabled=True)
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pred_mask: torch.Tensor, target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes an approximate Earth Mover's Distance (EMD) using log-domain Sinkhorn iterations.
        
        If target_mask is not provided, a mask of ones matching the target shape is used.
        
        Args:
            pred: (B, N, 3) predicted points.
            target: (B, M, 3) ground truth points.
            pred_mask: (B, N) Boolean mask for valid points in pred.
            target_mask: (B, M) Boolean mask for valid points in target (if None, will be created).
            
        Returns:
            loss: mean approximate EMD over the batch.
        """
        # If target_mask is None, create one that marks all target points as valid.
        if target_mask is None:
            target_mask = torch.ones(target.shape[0], target.shape[1], dtype=torch.bool, device=target.device)
            
        B = pred.shape[0]
        total_loss = 0.0
        
        for i in range(B):
            # Filter valid points.
            p_valid = pred[i]   # shape: (n_valid, 3)
            t_valid = target[i][target_mask[i].bool()] # shape: (m_valid, 3)
            
            if p_valid.shape[0] == 0 or t_valid.shape[0] == 0:
                continue
            
            if self.use_local:
                p_center = p_valid.mean(dim=0, keepdim=True)
                t_center = t_valid.mean(dim=0, keepdim=True)
                p_valid = p_valid - p_center
                t_valid = t_valid - t_center
            
            if p_valid.shape[0] > self.max_points_loss:
                indices = torch.randperm(p_valid.shape[0])[:self.max_points_loss]
                p_valid = p_valid[indices]
            if t_valid.shape[0] > self.max_points_loss:
                indices = torch.randperm(t_valid.shape[0])[:self.max_points_loss]
                t_valid = t_valid[indices]
                
            n_valid = p_valid.shape[0]
            m_valid = t_valid.shape[0]
            
            mu = torch.full((n_valid,), 1.0 / n_valid, device=pred.device)
            nu = torch.full((m_valid,), 1.0 / m_valid, device=pred.device)
            
            C = torch.cdist(p_valid, t_valid, p=2) ** 2
            
            scale = torch.median(C).item()
            if scale < 1e-3:
                scale = 1e-3
            eps_scaled = self.eps * scale
            
            log_mu = torch.log(mu + 1e-8)
            log_nu = torch.log(nu + 1e-8)
            log_K = -C / eps_scaled
            
            log_u = torch.zeros_like(mu)
            log_v = torch.zeros_like(nu)
            
            for _ in range(self.max_iter):
                log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
                log_v = log_nu - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)
            
            log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
            T = torch.exp(log_T)
            
            loss_i = torch.sum(T * C)
            total_loss += loss_i
        
        return total_loss / B

# ---------------- Test Function ----------------
def test_emd_loss_with_mask():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, M = 2, 1000, 1200
    pred = torch.randn(B, N, 3, device=device)
    target = torch.randn(B, M, 3, device=device)
    pred_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    pred_mask[:, 800:] = False
    # Do not provide target_mask; it will be created inside the function.
    loss_fn = EarthMoverDistanceLossWithMask(eps=0.1, max_iter=50, max_points_loss=512, use_local=True)
    loss_value = loss_fn(pred, target, pred_mask)
    print(f"Masked EMD Loss: {loss_value.item():.6f}")

# Uncomment below to run the test.
if __name__ == "__main__":
    test_emd_loss_with_mask()
