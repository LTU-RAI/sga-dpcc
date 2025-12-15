import torch, torch.nn as nn, torch.nn.functional as F  # type: ignore
import os, sys, time, yaml, pickle, argparse, numpy as np   # type: ignore
import open3d as o3d    # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sgadpcc_utils.LaserScan import SemLaserScan
from modules.SSG import SSG
from modules.PatchExtractor import PatchExtractor
from losses import PatchCDLoss
from models.PatchAutoEncoder import PatchEncoderTransformerWithSem, PatchDecoderFolding, PatchAutoencoderTransformer
from modules.PatchCollator import PatchCollator
from copy import deepcopy


class Data():
    def __init__(self, root: str, config_path: str, from_file: bool=False, max_range: float=50.0, verbose: bool=False, autoencoder_cfg_path: str=None):
        """
            DataHandler class to handle the loading and processing of the dataset.
            Inherits from SemLaserScan and SSG classes.
            Args:
                root (str): Root directory of the dataset.
                config_path (str): Path to the configuration file.
                max_range (float): Maximum range for filtering points.
                verbose (bool): Verbose mode for debugging.
            Returns:
                None
        """
        self.root = root
        self.verbose = verbose
        self.from_file = from_file
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.kitti_config = yaml.safe_load(open(self.config['dataset']['semantic-kitti-conf'], 'r'))
        self.autoencoder_cfg_path = autoencoder_cfg_path
        self.max_range = max_range

        ## Initialize the Semantic Laser Scan.
        self.sem_scan = SemLaserScan(sem_color_dict=self.kitti_config['color_map'])
        
        ## Initialize the Semantic Scene Graph.
        self.ssg = SSG(vocabulary=self.config['vocabulary'], verbose=False)

        return
    
    ## Reset the DataHandler.
    def reset(self) -> None:
        """
            Reset the DataHandler.
            Args:
                None
            Returns:
                None
        """
        self.sem_scan.reset()
        self.ssg.reset()
        return

    ## Get the Semantic Point Cloud.
    def get_sem_scan(self, sequence:str, scan_id: str) -> None:
        """
            Load the semantic scan from the file.
        """
        self.sem_scan.reset()
        velodyne_path = os.path.join(self.root, 'sequences', sequence, 'velodyne', scan_id + '.bin')
        label_path = os.path.join(self.root, 'sequences', sequence, 'labels', scan_id + '.label')
        self.sem_scan.open_scan(velodyne_path)
        self.sem_scan.open_label(label_path)
        self.sem_scan.map_semantic_classes(self.kitti_config['learning_map'], 
                                            self.kitti_config['learning_map_inv'])
        if self.verbose: print(f"Loaded semantic scan from {velodyne_path} and {label_path}.")
        return

    ## Get the SSG.
    def get_ssg(self, from_file: bool, sequence: str, scan_id: str) -> None:
        """
            Load SSG from file or generate it from the scan.
            Args:
                from_file (bool): If True, load SSG from file.
                scan_id (str): Scan ID.
            Returns:
                ssg (SSG): Semantic Scene Graph object.
        """
        self.ssg.reset()
        ssg_path = os.path.join(self.root, 'sequences', sequence, 
                                'graph', scan_id + '.pickle')
        if from_file:
            self.ssg.load_ssg(ssg_path)
            if self.verbose: print(f"Loaded SSG from {ssg_path}.")
        else:
            try:
                self.ssg.generate_ssg(scan_id=scan_id, sem_scan=self.sem_scan)
                if self.verbose: print(f"Generated SSG from {scan_id}.")
            except Exception as e: print(f"Semantic scan not found, try calling 'get sem_scan' first: {e}")
        return
    

    def get_data(self, sequence: str, scan_id: str) -> None:
        """
            Load the semantic scan and SSG.
            Args:
                sequence (str): Sequence ID.
                scan_id (str): Scan ID.
            Returns:
                None
        """
        # tic = time.time()
        self.get_sem_scan(sequence=sequence, scan_id=scan_id)
        # toc = time.time()
        # print(f"Time taken for loading scan: {toc - tic:.2f} seconds")
        # tic = time.time()
        self.get_ssg(from_file=self.from_file, sequence=sequence, scan_id=scan_id)
        # toc = time.time()
        # print(f"Time taken for generating the SSG: {toc - tic:.2f} seconds")
        return


class DPCC():
    def __init__(self, data: Data, selected_layer: int, overlap_ratio: float= 0.05, 
                 terrain_grid: int= 16, min_cell_pts: int = 64, device: str="cuda", autoencoder_cfg_path: str=None):
        ## Inherit all attributes from the provided DataHandler instance.
        self.__dict__.update(data.__dict__)
        """
            Initialize the Processor class which processes a specified layer.
            Args:
                root (str): Root directory of the dataset.
                config (dict): Configuration dictionary.
                kitti_config (dict): KITTI configuration dictionary.
                max_range (float): Maximum range for filtering points.
        """
        self.selected_layer = selected_layer
        self.device = torch.device(device)
        self.terrain_grid = terrain_grid
        self.min_cell_pts = min_cell_pts
        self.overlap_ratio = overlap_ratio

        ## Get autoencoder configuration.
        if self.autoencoder_cfg_path is None:
            self.autoencoder_cfg_path = self.config['autoencoder']['config']
            self.autoencoder_cfg = yaml.safe_load(open(self.autoencoder_cfg_path, 'r'))
        else:
            self.autoencoder_cfg = yaml.safe_load(open(self.autoencoder_cfg_path, 'r'))
        self.layer_cfg = self.autoencoder_cfg[self.selected_layer]
        self.max_points = self.layer_cfg['max_points']

        ## Initialize the Patch Collator
        self.collator = PatchCollator(max_allowed_points=self.max_points)

        ## Initialize the Patch Extractor
        self.patch_extractor = PatchExtractor(target_layers=[self.selected_layer], max_points=self.max_points,
                                                  terrain_grid_dim=self.terrain_grid, 
                                                  min_cell_points=self.min_cell_pts,
                                                    overlap_ratio=self.overlap_ratio)

        if self.verbose: print(f"Processor initialized with root: {self.root}, config: {self.config}, ")

        ## Initialize the Autoencoder.
        self.autoencoder = self.init_autoencoder()

    ## Get Autoencoder.
    def init_autoencoder(self) -> nn.Module:
        """
            Get the autoencoder model.
            Args:
                layer_cfg (dict): Layer configuration.
            Returns:
                autoencoder (nn.Module): Autoencoder model.
        """
        enc_cfg = self.layer_cfg['encoder']
        dec_cfg = self.layer_cfg['decoder']
        encoder = PatchEncoderTransformerWithSem(
            input_dim=enc_cfg['input_dim'],
            encoder_blocks=enc_cfg['encoder_blocks'],
            feature_dim=enc_cfg['feature_dim'],
            latent_dim=enc_cfg['latent_dim'],
            pos_enc_dim=enc_cfg['pos_enc_dim'],
            sem_dim=enc_cfg['sem_dim'],
            num_classes=enc_cfg['num_classes'],
            dropout_prob=enc_cfg['dropout_prob']
        ).to(self.device)
        encoder.eval()
        decoder = PatchDecoderFolding(
            latent_dim=dec_cfg['latent_dim'],
            coarse_points=dec_cfg['coarse_points'],
            feature_dim=dec_cfg['feature_dim'],
            upsample_factors=dec_cfg['upsample_factors'],
            decoder_hidden_dims=dec_cfg['decoder_hidden_dims'],
            output_dim=dec_cfg['output_dim']
        ).to(self.device)
        decoder.eval()
        autoencoder = PatchAutoencoderTransformer(encoder, decoder).to(self.device)
        autoencoder.eval()
        if self.verbose: print(f'Number of parameters in autoencoder: {sum(p.numel() for p in autoencoder.parameters())}')
        return autoencoder
    
    ## Load Checkpoint.
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
            Load the checkpoint for the autoencoder.
            Args:
                checkpoint_path (str): Path to the checkpoint file.
            Returns:
                None
        """
        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "")
                    new_state_dict[new_key] = v
                self.autoencoder.load_state_dict(new_state_dict, strict=False)
                if self.verbose: print(f"Loaded autoencoder checkpoint from {checkpoint_path}.")
            else: print(f"Checkpoint: {checkpoint_path} not found; testing with randomly initialized weights.")
        except Exception as e: print(f"Makle sure autoencoder is initialized: {e}")
        return 
    
    ## Preprocess the scan.
    def tensorfy_scan(self, sem_scan: SemLaserScan) -> torch.tensor:
        """
            Preprocess the scan by filtering points by range and remapping semantic classes.
            Args:
                scan (SemLaserScan): Semantic Laser Scan object.
            Returns:
                scan (torch.tensor): Preprocessed scan tensor.
        """
        ## Convert to learning labels.
        sem_scan.learning_map(learning_map=self.kitti_config['learning_map'])
        sem_scan.sem_label = np.vectorize(lambda x: self.kitti_config['learning_map'][x])(sem_scan.sem_label)
        sem_scan = np.concatenate((sem_scan.points,  np.expand_dims(sem_scan.sem_label, axis=1)), axis=1, dtype=np.float32)
        sem_scan = sem_scan[np.linalg.norm(sem_scan[:, :3], axis=1) <= self.max_range]
        ## Convert to tensor.
        tensor = torch.tensor(sem_scan, dtype=torch.float32)
        return tensor
    
    ## Get graph nodes.
    def get_graph_nodes(self, ssg) -> list:
        """
            Get graph nodes from the SSG.
            Args:
                ssg (SSG): Semantic Scene Graph object.
            Returns:
                nodes (list): List of nodes in the graph.
        """
        ## Remap semantic classes.
    
        for node, data in ssg.ssg.nodes(data=True):
            semantic_id = data.get('semantic_id', None)            
            if semantic_id is not None:
                data['semantic_id'] = self.kitti_config['learning_map'][semantic_id]
        nodes = [data for node, data in ssg.ssg.nodes(data=True)]
        if self.verbose: print(f"Extracted {len(nodes)} nodes from SSG.")
        return nodes
    
    ## Get Patches.
    def get_patches(self, tensor_scan: torch.tensor, nodes: list, scan_id: str) -> list:
        patches = self.patch_extractor.extract_patches(tensor_scan, nodes, gamma=0.0)
        if self.verbose: print(f"Extracted {len(patches)} patches from sem_scan.")
        ## Filter patches by desired layers (if necessary).
        patches = [patch for patch in patches if patch.get('layer') in [self.selected_layer]]
        # patches = [patch for patch in patches if patch.get('points').shape[0] > 256]
        if self.verbose: print(f"Filtered patches count: {len(patches)}")
        ## Wrap the filtered patches into a dict as expected by the collate function.
        patches_dict = [{'patches': patches, 'scan_id': scan_id}]
        return patches_dict
    
    ## Collate Patches into a Batch.
    def collate_patches(self, patches_dict: list) -> dict:
        """
            Collate patches into a batch.
            Args:
                patches_dict (list): List of patches.
            Returns:
                batch (dict): Batched patches.
        """
        batch = self.collator.collate(patches_dict)
        if self.verbose: print(f"Batched patches shape: {batch['points'].shape}")
        return batch
    
    ## Forward Pass Through the Encoder.
    def encoder(self, batch: dict) -> tuple:
        """
            Forward pass through the encoder.
            Args:
                batch (dict): Batch of patches.
            Returns:
                latent (torch.tensor): Latent representation.
                center (torch.tensor): Center of the points.
        """
        batch_points = batch['points'].to(self.device)   # (B_total, max_points, 3)
        batch_mask = batch['mask'].to(self.device)         # (B_total, max_points)
        batch_semantic = batch['semantic'].to(self.device) # (B_total,)
        batch_center = batch['center'].to(self.device)         # (B_total, 3)
        tic = time.time()
        # if batch_mask is not None:
        #     valid = batch_mask.unsqueeze(-1).float()
        #     center = (batch_points * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        # else:
        #     center = torch.zeros(batch_points.shape[0], 1, 3, device=batch_points.device)
        x_local = batch_points - batch_center.unsqueeze(1)  # (B_total, max_points, 3)
        latent = self.autoencoder.encoder(x_local, batch_semantic, batch_mask)
        toc = time.time()
        if self.verbose: print(f"Forward pass took {toc - tic:.6f} seconds for {batch_points.shape[0]} patches.")
        if self.verbose: print("Latent shape:", latent.shape)  # Expected: (B_total, latent_dim)
        return latent 
    
    ## Forward Pass Through the Decoder.
    def decoder(self, latent: torch.tensor, batch) -> tuple:
        extent = batch['extent'].to(self.device)  # (B_total, 3)
        orientation = batch['orientation'].to(self.device)  # (B_total, 3, 3)
        center = batch['center'].to(self.device).unsqueeze(1)  # (B_total, 3)
        coarse_pts_local, coarse_mask, fine_mask, fine_pts_local = self.autoencoder.decoder(latent, extent, orientation)
        ## Convert back to global coordinates.
        coarse_pts = coarse_pts_local + center
        fine_pts = fine_pts_local + center

        if self.verbose: print("Reconstruction shape:", fine_pts.shape)  # Expected: (B_total, max_points, 3)
        if self.verbose: print("Latent shape:", latent.shape)  # Expected: (B_total, latent_dim)
        return coarse_pts, coarse_mask, fine_mask, fine_pts
    
    def decode_only(self, ssg: SSG):
        """
        Decode all patch latents for selected_layer in the SSG.
        For terrain (layer 1) nodes, center/extent/orientation are lists (one per patch).
        For other nodes, they are single values.
        """
        nodes = [data for node, data in ssg.ssg.nodes(data=True)]
        expected_latent_dim = self.layer_cfg['encoder']['latent_dim']

        latent_list, center_list, extent_list, orientation_list, semantic_id_list = [], [], [], [], []

        for i, data in enumerate(nodes):
            node_layer = data.get('layer')
            if node_layer != self.selected_layer:
                continue
            latent = data.get('latent')
            if latent is None:
                print(f"Warning: Node {i} with None latent, skipping.")
                continue

            # Handle terrain nodes: all fields are lists
            if node_layer == 1:
                latent_arr = np.asarray(data['latent'])
                num_patches = latent_arr.shape[0]
                
                centers = [np.array(c).reshape(3) for c in data['center'] if np.array(c).shape == (3,)]
                extents = [np.array(e).reshape(3) for e in data['extent'] if np.array(e).shape == (3,)]
                orientations = [np.array(o).reshape(3, 3) for o in data['orientation'] if np.array(o).shape == (3, 3)]
                latent_list.append(torch.tensor(latent_arr, dtype=torch.float32, device=self.device))
                center_list.append(torch.tensor(centers, dtype=torch.float32, device=self.device))
                extent_list.append(torch.tensor(extents, dtype=torch.float32, device=self.device))
                orientation_list.append(torch.tensor(orientations, dtype=torch.float32, device=self.device))
                sem_id = data['semantic_id']
                sem_ids = np.full((num_patches,), sem_id)
                semantic_id_list.append(torch.tensor(sem_ids, dtype=torch.long, device=self.device))
            else:
                latent_arr = np.asarray(latent)
                if latent_arr.ndim == 1:
                    latent_arr = latent_arr[np.newaxis, :]
                num_patches = latent_arr.shape[0]
                # For non-terrain, broadcast node's attributes to all patches
                center = np.asarray(data['center'])
                extent = np.asarray(data['extent'])
                orientation = np.asarray(data['orientation'])
                center = np.broadcast_to(center, (num_patches, *center.shape))
                extent = np.broadcast_to(extent, (num_patches, *extent.shape))
                orientation = np.broadcast_to(orientation, (num_patches, *orientation.shape))
                latent_list.append(torch.tensor(latent_arr, dtype=torch.float32, device=self.device))
                center_list.append(torch.tensor(center, dtype=torch.float32, device=self.device))
                extent_list.append(torch.tensor(extent, dtype=torch.float32, device=self.device))
                orientation_list.append(torch.tensor(orientation, dtype=torch.float32, device=self.device))
                sem_id = data['semantic_id']
                sem_ids = np.full((num_patches,), sem_id)
                semantic_id_list.append(torch.tensor(sem_ids, dtype=torch.long, device=self.device))

        if not latent_list:
            print(f"No patches in layer {self.selected_layer} with correct-shape latents to decode!")
            return None, None, None, None, None

        latent = torch.cat(latent_list, dim=0)        # (N_total_patches, latent_dim)
        center = torch.cat(center_list, dim=0)        # (N_total_patches, 3)
        extent = torch.cat(extent_list, dim=0)        # (N_total_patches, 3)
        orientation = torch.cat(orientation_list, dim=0)  # (N_total_patches, 3, 3)
        semantic_ids = torch.cat(semantic_id_list, dim=0)  # (N_total_patches,)

        with torch.no_grad():
            coarse_pts_local, coarse_mask, fine_mask, fine_pts_local = self.autoencoder.decoder(latent, extent, orientation)
            coarse_pts = coarse_pts_local + center.unsqueeze(1)
            fine_pts = fine_pts_local + center.unsqueeze(1)

        return coarse_pts, coarse_mask, fine_mask, fine_pts, semantic_ids

    
    ## Remap all patch semantic_ids to the learning_map_inv.
    def remap_patch_ids(self, patches: list) -> list:
        """
            Remap all patch semantic_ids to the learning_map_inv.
            Args:
                patches (list): List of patches.
            Returns:
                patches (list): List of patches with remapped semantic_ids.
        """
        for patch in patches:
            patch['semantic_id'] = self.kitti_config['learning_map_inv'][patch['semantic_id']]
        return patches


    ## TODO: Compute Loss.
    def compute_loss(self, reconstruction: torch.tensor, 
                     batch_points: torch.tensor,
                     return_mask: torch.tensor) -> float:
        return

    ## Full Pipeline.
    def autoencode(self, sem_scan: SemLaserScan, ssg: SSG, scan_id: str) -> tuple:
        """
            Full pipeline to encode and decode the scan.
            Args:
                sequence (str): Sequence ID.
                scan_id (str): Scan ID.
            Returns:
                latent (torch.tensor): Latent representation.
                coarse_pts (torch.tensor): Coarse points.
                coarse_mask (torch.tensor): Coarse mask.
                fine_mask (torch.tensor): Fine mask.
                fine_pts (torch.tensor): Fine points.
        """
        with torch.no_grad():
            tensor_scan = self.tensorfy_scan(sem_scan)
            nodes = self.get_graph_nodes(ssg)
            patches_dict = self.get_patches(tensor_scan, nodes, scan_id=scan_id)

            # Early check: if none of the samples have valid patches, return None
            if not any(sample["patches"] for sample in patches_dict):
                if self.verbose:
                    print(f"Warning: No valid patches extracted for scan {scan_id}.")
                return None, None, None, None, None, None, patches_dict

            tic = time.time()
            batch = self.collate_patches(patches_dict)
            toc = time.time()
            if self.verbose:
                print(f"Time taken for collating patches: {toc - tic:.2f} seconds")
            tic = time.time()
            latent = self.encoder(batch)
            toc = time.time()
            if self.verbose:
                print(f"Time taken for encoding: {toc - tic:.2f} seconds")
            tic = time.time()
            coarse_pts, coarse_mask, fine_mask, fine_pts = self.decoder(latent, batch)
            toc = time.time()
            if self.verbose:
                print(f"Time taken for decoding: {toc - tic:.2f} seconds")
        return latent, coarse_pts, coarse_mask, fine_mask, fine_pts, batch, patches_dict

    ## Get predicted points.
    def get_predicted_points(self,  pts: torch.tensor,
                             mask: torch.tensor, confidence: float = 0.5) -> np.ndarray:
        """
            From the points and mask get only the valid points
            as a numpy array.
            Args:
                pts (torch.tensor): Predicted points.
                mask (torch.tensor): Mask for the points.
                confidence (float): Confidence threshold for filtering points.
            Returns:
                valid_pts (np.ndarray): Valid points as a numpy array.
        """
        if pts is None or mask is None:
            return []
        batch_size = pts.shape[0]
        valid_pts = []
        for i in range(batch_size):
            valid_pts_i = pts[i].detach().cpu().numpy()
            valid_mask_i = mask[i].detach().cpu().numpy().squeeze()
            valid_pts_i = valid_pts_i[valid_mask_i > confidence]
            valid_pts.append(valid_pts_i)
        return valid_pts    
    
    ## Get target points.
    def get_target_points(self, pts: torch.tensor,
                          mask: torch.tensor) -> np.ndarray:
        """
            From the points and mask get only the valid points
            as a numpy array.
            Args:
                pts (torch.tensor): Predicted points.
                mask (torch.tensor): Mask for the points.
                confidence (float): Confidence threshold for filtering points.
            Returns:
                valid_pts (np.ndarray): Valid points as a numpy array.
        """
        batch_size = pts.shape[0]
        valid_pts = []
        for i in range(batch_size):
            valid_pts_i = pts[i].detach().cpu().numpy()
            valid_mask_i = mask[i].detach().cpu().numpy().squeeze()
            valid_pts_i = valid_pts_i[valid_mask_i]
            valid_pts.append(valid_pts_i)
        return valid_pts
                
    ## Get predicted decompressed point clouds.
    def get_predicted_pcds(self, pts: torch.tensor, 
                           mask: torch.tensor, 
                           confidence: float = 0.5,
                           colors: np.array=None) -> list:
        """
            Convert the predicted points to Open3D point cloud.
            Args:
                pts (torch.tensor): Predicted points.
                mask (torch.tensor): Mask for the points.
                confidence (float): Confidence threshold for filtering points.
                colors (np.array): Colors for the points.
            Returns:
                list(pcd) (o3d.geometry.PointCloud): Open3D point cloud objects.
        """
        if colors is None:
            ## Create random color for each patch with a seed.
            colors = np.random.rand(pts.shape[0], 3)
            # print(f"Colors shape: {colors.shape}")
        batch_size = pts.shape[0]
        batch = []
        for i in range(batch_size):
            valid_pts = pts[i].detach().cpu().numpy()
            valid_mask = mask[i].detach().cpu().numpy().squeeze()
            valid_pts = valid_pts[valid_mask > confidence]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_pts)
            pcd.colors = o3d.utility.Vector3dVector(valid_pts*colors[i])
            batch.append(pcd)
        return batch
    
    ## Get target point clouds.
    def get_target_pcds(self, pts: torch.tensor, 
                        mask: torch.tensor, 
                        colors: np.array=None) -> list:
        """
            Convert the predicted points to Open3D point cloud.
            Args:
                pts (torch.tensor): Predicted points.
                mask (torch.tensor): Mask for the points.
                colors (np.array): Colors for the points.
            Returns:
                list(pcd) (o3d.geometry.PointCloud): Open3D point cloud objects.
        """
        if colors is None:
            ## Create random for each patch with a seed.
            np.random.seed(420)
            colors = np.random.rand(pts.shape[0], 3)
        
        batch_size = pts.shape[0]
        batch = []
        for i in range(batch_size):
            valid_pts = pts[i].detach().cpu().numpy()
            valid_mask = mask[i].detach().cpu().numpy().astype(bool)
            valid_pts = valid_pts[valid_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors[i][valid_mask])
            batch.append(pcd)
        return batch



