import torch
import numpy as np

class PatchExtractor:
    def __init__(self, target_layers=[1, 2, 3, 4], max_points=2048, terrain_grid_dim=16, min_cell_points=64, overlap_ratio=0.05): #10,64 / 16,64
        """
        Initializes the PatchExtractor.
        
        Args:
            target_layers (list): List of scene graph layers to consider.
            max_points (int): Maximum number of points per patch.
            terrain_grid_dim (int): Number of subdivisions (per axis) for terrain patches.
            min_cell_points (int): Minimum points required in a grid cell to create a patch.
        """
        self.target_layers = target_layers
        self.max_points = max_points
        self.terrain_grid_dim = terrain_grid_dim
        self.min_cell_points = min_cell_points
        self.overlap_ratio = overlap_ratio

    def extract_patch(self, point_cloud, node, gamma=1.0):
        """
        Extracts a patch for a given node using bounding box filtering.
        After extraction, it limits the number of points to self.max_points.
        
        Args:
            point_cloud (torch.Tensor): (N, 4) tensor of (x, y, z, semantic_class).
            node (dict): Must contain 'center', 'extent', 'orientation', 'semantic_id', 'layer', and 'num_points'.
            gamma (float): Factor to inflate the bounding box.
        
        Returns:
            torch.Tensor: Patch for the node.
        """
        semantic_class = node.get('semantic_id', None)
        if semantic_class is not None:
            sem_points = point_cloud[point_cloud[:, 3] == semantic_class]
        else:
            sem_points = point_cloud

        center = torch.tensor(node['center'], dtype=torch.float32, device=point_cloud.device)
        extent = torch.tensor(node['extent'], dtype=torch.float32, device=point_cloud.device)
        orientation = torch.tensor(node['orientation'], dtype=torch.float32, device=point_cloud.device)
        half_extent = extent / 2.0

        ## Inflate the bounding box by gamma.
        inflated_half_extent = half_extent * (1 + gamma)
        local_pts = torch.matmul((sem_points[:, :3] - center), orientation)
        mask = (
            (local_pts[:, 0] >= -inflated_half_extent[0]) & (local_pts[:, 0] <= inflated_half_extent[0]) &
            (local_pts[:, 1] >= -inflated_half_extent[1]) & (local_pts[:, 1] <= inflated_half_extent[1]) &
            (local_pts[:, 2] >= -inflated_half_extent[2]) & (local_pts[:, 2] <= inflated_half_extent[2])
        )
        patch = sem_points[mask]

        ## If too many points, uniformly sample down to max_points.
        if patch.shape[0] > self.max_points:
            indices = torch.randperm(patch.shape[0])[:self.max_points]
            patch = patch[indices]
        return patch

    def extract_terrain_patch(self, point_cloud, node, overlap_ratio=0.05):
        """
        Subdivides the terrain (layer 1) node into a grid of smaller patches based on the x-y extent.
        The grid cells now overlap by 10%.
        
        Args:
            point_cloud (torch.Tensor): (N, 4) tensor of (x, y, z, semantic_class).
            node (dict): Node information for the terrain patch.
            
        Returns:
            list[dict]: List of patch dictionaries extracted from the terrain.
        """
        terrain_points = point_cloud[point_cloud[:, 3] == node['semantic_id']]
        if terrain_points.shape[0] == 0:
            return []
        
        ## Project to 2D (x, y) for grid subdivision.
        pts_xy = terrain_points[:, :2]
        min_xy = torch.min(pts_xy, dim=0).values
        max_xy = torch.max(pts_xy, dim=0).values
        
        grid_dim = self.terrain_grid_dim
        cell_size = (max_xy - min_xy) / grid_dim
        
        ## Define overlap ratio (10%) and compute stride as 90% of the cell_size.
        ## overlap_ratio = 0.1 #0.05
        stride = cell_size * (1 - overlap_ratio)
        
        patches = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                # Cell lower bound is placed using the stride,
                # which makes neighboring cells overlap by 10%.
                offset = torch.tensor([i, j], device=terrain_points.device, dtype=cell_size.dtype)
                cell_min = min_xy + offset * stride
                cell_max = cell_min + cell_size
                # Optionally, you can clip cell_max to max_xy if you want to restrict to overall bounds.
                cell_mask = (
                    (pts_xy[:, 0] >= cell_min[0]) & (pts_xy[:, 0] < cell_max[0]) &
                    (pts_xy[:, 1] >= cell_min[1]) & (pts_xy[:, 1] < cell_max[1])
                )
                cell_points = terrain_points[cell_mask]
                if cell_points.shape[0] < self.min_cell_points:
                    continue  # Skip cells with too few points.
                
                if cell_points.shape[0] > self.max_points:
                    # Use weighted sampling here (or uniform) as needed.
                    distances = torch.norm(cell_points, dim=1)
                    weights = distances / torch.sum(distances)
                    indices = torch.multinomial(weights, self.max_points, replacement=False)
                    cell_points = cell_points[indices]

                patch_center = cell_points[:, :3].mean(dim=0).tolist()
                cell_min_3d = torch.min(cell_points[:, :3], dim=0).values
                cell_max_3d = torch.max(cell_points[:, :3], dim=0).values
                patch_extent = (cell_max_3d - cell_min_3d).tolist()
                
                patch_info = {
                    'semantic_id': node['semantic_id'],
                    'center': patch_center,
                    'extent': patch_extent,
                    'orientation': node['orientation'],  # Orientation remains as defined in the node.
                    'layer': node['layer'],
                    'points': cell_points,
                    'num_points': cell_points.shape[0]
                }
                patches.append(patch_info)
        return patches

    def extract_patches(self, point_cloud, nodes, gamma=1.0):
        """
        Extracts patches from the point cloud for nodes in target_layers.
        For terrain nodes (layer 1), subdivide into multiple patches;
        for other layers, process each node individually.
        
        Args:
            point_cloud (torch.Tensor): (N, 4) tensor with (x, y, z, semantic_class).
            nodes (list[dict]): List of node dictionaries.
            gamma (float): Parameter for bounding box inflation.
            
        Returns:
            list[dict]: Each dictionary corresponds to a patch with keys:
                'semantic_id', 'center', 'extent', 'orientation', 'layer', 'points', and 'num_points'.
        """
        patches = []
        ## Filter nodes to only those in target layers.
        nodes = [node for node in nodes if node['layer'] in self.target_layers]
        for node in nodes:
            if node['layer'] == 1:
                ## For every terrain type, subdivide into smaller patches.
                terrain_patches = self.extract_terrain_patch(point_cloud, node, self.overlap_ratio)
                patches.extend(terrain_patches)
            else:
                patch = self.extract_patch(point_cloud, node, gamma)
                patch_info = {
                    'semantic_id': node['semantic_id'],
                    'center': node['center'],
                    'extent': node['extent'],
                    'orientation': node['orientation'],
                    'layer': node['layer'],
                    'points': patch,
                    'num_points': patch.shape[0]
                }
                patches.append(patch_info)
        return patches
