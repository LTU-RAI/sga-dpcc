import os, sys, torch, time, yaml
import numpy as np
from torch_geometric.data import Dataset  # type: ignore
from torch_geometric.data import Data     # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from recnetpp_utils.LaserScan import SemLaserScan
from modules.SSG import SSG
from modules.PatchExtractor import PatchExtractor

class PatchDataset(Dataset):
    def __init__(self, root: str, split: str, config: dict, 
                 max_range: float=50.0, desired_layers:list=[2,3,4],
                 min_points: int=16, max_points: int=2048):
        """
        PatchDataset for SemanticKITTI.
        Args:
            root (str): Root directory of the dataset.
            split (str): Split to use, either 'train' or 'valid'.
            config (dict): Configuration dictionary for the dataset.
            max_range (float): Maximum range of the LiDAR sensor.
            desired_layers (list): List of layer indices to include. Only patches whose 
                                   'layer' attribute is in this list will be returned.
        """
        super(PatchDataset, self).__init__(root)
        self.config = config
        kitti_config_path = self.config['dataset']['semantic-kitti-conf']
        self.kitti_config = yaml.safe_load(open(kitti_config_path, 'r'))
        self.split = split
        self.root = root
        self.max_range = max_range
        self.max_points = max_points
        self.min_points = min_points
        self.desired_layers = desired_layers
        self.data = []
        self.load_filenames()
        self.sem_scan = SemLaserScan(
            sem_color_dict=self.kitti_config['color_map'], 
            project=False,
            H=64, W=2048, fov_up=3.0, fov_down=-24.9
        )
        self.ssg = SSG(vocabulary=self.config['vocabulary'])
        # Note: the PatchExtractor is already configured with target_layers;
        # you could set it to all layers and then filter here, or set it to desired_layers.
        self.patch_extractor = PatchExtractor(target_layers=desired_layers, max_points=self.max_points)
        
    def load_filenames(self) -> None:
        for seq in self.kitti_config['split'][self.split]:
            sequence_path = os.path.join(self.root, 'sequences', f'{seq:02d}', 'graph')
            for file in sorted(os.listdir(sequence_path)):
                self.data.append(os.path.join(sequence_path, file))
        return 

    def load_ssg(self, idx: int) -> list:
        self.ssg.reset()
        try:
            self.ssg.load_ssg(self.data[idx])
        except Exception as e:
            print(f'Error loading {self.data[idx]}: {e}')
        # Remap semantic classes.
        for node, data in self.ssg.ssg.nodes(data=True):
            semantic_id = data.get('semantic_id', None)
            if semantic_id is not None:
                data['semantic_id'] = self.kitti_config['learning_map'][semantic_id]
        nodes = [data for node, data in self.ssg.ssg.nodes(data=True)]
        return nodes
    
    def load_sem_scan(self, idx: int) -> np.ndarray:
        self.sem_scan.reset()
        try:
            self.sem_scan.open_scan(self.data[idx].replace('graph', 'velodyne').replace('.pickle', '.bin'))
            self.sem_scan.open_label(self.data[idx].replace('graph', 'labels').replace('.pickle', '.label'))
        except Exception as e:
            print(f'Error opening scan: {self.data[idx].replace("graph", "velodyne").replace(".pickle", ".bin")}: {e}')
        self.sem_scan.map_semantic_classes(self.kitti_config['learning_map'], self.kitti_config['learning_map_inv'])
        self.sem_scan.learning_map(learning_map=self.kitti_config['learning_map'])
        self.sem_scan.sem_label = np.vectorize(lambda x: self.kitti_config['learning_map'][x])(self.sem_scan.sem_label)
        # Filter points by range.
        sem_scan = np.concatenate((self.sem_scan.points,  np.expand_dims(self.sem_scan.sem_label, axis=1)), axis=1, dtype=np.float32)
        sem_scan = sem_scan[np.linalg.norm(sem_scan[:, :3], axis=1) <= self.max_range]
        # sem_scan = sem_scan[np.linalg.norm(sem_scan[:, :3], axis=1) > 4.0]
        return sem_scan

    def load_rec_scan(self, idx: int) -> np.ndarray:
        rec_scan = np.fromfile(self.data[idx].replace('graph', 'sem_recon').replace('.pickle', '.bin'), dtype=np.float32).reshape(-1, 4)
        return rec_scan

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with:
          - 'patches': a list of patch dictionaries for patches that belong to desired_layers.
                       Each patch dict should contain at least the keys: 'points', 'semantic_id', 'layer', etc.
        """
        # Load SSG and nodes.
        nodes = self.load_ssg(idx)
        # Load semantic and reconstructed scans.
        target_sem_scan = self.load_sem_scan(idx)
        rec_scan = self.load_rec_scan(idx)
        # Convert to torch tensors.
        rec_scan = torch.tensor(rec_scan, dtype=torch.float32)
        target_sem_scan = torch.tensor(target_sem_scan, dtype=torch.float32)
        # Extract patches using the patch extractor.
        patches = self.patch_extractor.extract_patches(target_sem_scan, nodes, gamma=0.0)
        # Filter patches: only keep those whose 'layer' attribute is in desired_layers.
        filtered_patches = [patch for patch in patches if patch.get('layer') in self.desired_layers]
        # Remove all patches with less than 16 points.
        filtered_patches = [patch for patch in filtered_patches if patch.get('points').shape[0] >= self.min_points]
        # filtered_patches = [patch for patch in filtered_patches if patch.get('points').shape[0] >= 32]
        # filtered_patches = [patch for patch in filtered_patches if patch.get('points').shape[0] >= 64]
        return {'patches': filtered_patches, 'scan_id': idx}

# ---------------- Test the Dataset ----------------
# if __name__ == '__main__':
#     import yaml
#     root = '/home/nikos/python_projects/recnetpp/dataset/SemanticKitti'
#     config_path = '/home/nikos/python_projects/recnetpp/config/config.yaml'
#     config = yaml.safe_load(open(config_path, 'r'))
    
#     dataset = PatchDataset(root=root, split='train', config=config, max_range=50.0, desired_layers=[2])
#     print("Number of scans in dataset:", len(dataset))
#     sample = dataset[0]
#     print("Number of patches in first sample:", len(sample['patches']))
#     # Example: print shape of points in one patch.
#     if len(sample['patches']) > 0:
#         print("Shape of patch points:", sample['patches'][0]['points'].shape)
