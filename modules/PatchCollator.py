import torch

class PatchCollator:
    def __init__(self, max_allowed_points):
        """
        Initializes the PatchCollator with the maximum allowed points per patch.

        Args:
            max_allowed_points (int): Maximum number of points allowed per patch.
        """
        self.max_allowed_points = max_allowed_points

    def collate(self, batch):
        """
        Collates a batch of patch dictionaries.

        Each sample in the batch is a dictionary with key 'patches', where each patch dictionary contains:
          - 'points': (N_i, 4) tensor (we only use columns 0:3: x,y,z)
          - 'semantic_id': int
          - 'center': (3,) list or tensor
          - 'extent': (3,) list or tensor
          - 'orientation': (3, 3) list or tensor

        The function:
          - Flattens all patches across the batch.
          - For each patch, if the number of points > max_allowed_points, uniformly samples down.
          - Pads each patch along the point dimension to the maximum number of points in the batch.
          - Creates a mask for valid points.

        Returns a dictionary with:
           'points': (B_total, max_points, 3) tensor,
           'mask':   (B_total, max_points) tensor,
           'semantic': (B_total,) tensor,
           'center': (B_total, 3) tensor,
           'extent': (B_total, 3) tensor,
           'orientation': (B_total, 3, 3) tensor
        """
        all_patches = []
        for sample in batch:
            all_patches.extend(sample['patches'])

        points_list = []
        mask_list = []
        semantic_list = []
        center_list = []
        extent_list = []
        orientation_list = []

        for patch in all_patches:
            p = patch['points']
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32)
            p = p[:, :3]
            if p.shape[0] > self.max_allowed_points:
                idx = torch.randperm(p.shape[0])[:self.max_allowed_points]
                p = p[idx]
            points_list.append(p)
            semantic_list.append(patch['semantic_id'])
            mask_list.append(torch.ones(p.shape[0], dtype=torch.bool))

            center_list.append(torch.tensor(patch['center'], dtype=torch.float32))
            extent_list.append(torch.tensor(patch['extent'], dtype=torch.float32))
            orientation_list.append(torch.tensor(patch['orientation'], dtype=torch.float32))

        padded_points = []
        padded_masks = []
        for p, m in zip(points_list, mask_list):
            n = p.shape[0]
            if n < self.max_allowed_points:
                pad = torch.zeros((self.max_allowed_points - n, p.shape[1]), dtype=p.dtype)
                p = torch.cat([p, pad], dim=0)
                m = torch.cat([m, torch.zeros(self.max_allowed_points - n, dtype=torch.bool)], dim=0)
            padded_points.append(p)
            padded_masks.append(m)
        
        batch_points = torch.stack(padded_points, dim=0)
        batch_mask = torch.stack(padded_masks, dim=0)
        batch_semantic = torch.tensor(semantic_list, dtype=torch.long)
        batch_center = torch.stack(center_list, dim=0)
        batch_extent = torch.stack(extent_list, dim=0)
        batch_orientation = torch.stack(orientation_list, dim=0)

        return {
            'points': batch_points,
            'mask': batch_mask,
            'semantic': batch_semantic,
            'center': batch_center,
            'extent': batch_extent,
            'orientation': batch_orientation
        }
