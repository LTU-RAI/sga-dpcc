import os
import sys, time, yaml, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d  
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.Processor import DPCC, Data
from copy import deepcopy
import pickle
from modules.Metrics import chamfer_distance

user = os.environ.get('USER')

# ---------------- Settings ----------------
root = f"/home/{user}/Documents/datasets_2/SemanticKitti"
config_path = f"/home/{user}/python_projects/sga-dpcc/config/config-latest.yaml"
selected_layers = [1, 2, 3, 4]
latents = [128, 64, 32, 128]
max_range = 50.0
processors = []

data_handler = Data(root=root, config_path=config_path)

data_handler.get_data(sequence="06", scan_id="000400")
data_handler.ssg.to_float16()
encoded_bytes = pickle.dumps(data_handler.ssg.ssg)
print(f"Size of encoded SSG (before latents): {len(encoded_bytes)} bytes")
# size of original sem scan
pts_bytes = pickle.dumps(data_handler.sem_scan.points)
labels_bytes = pickle.dumps(data_handler.sem_scan.sem_label)
print(f"Size of original sem scan (points + labels): {len(pts_bytes) + len(labels_bytes)} bytes")

for i in selected_layers:
    checkpoint_path=f"/home/{user}/python_projects/sga-dpcc/weights/checkpoints/latest/layer_{i}/autoencoder_layer_{i}_{latents[i-1]}.torch"
    processor = DPCC(data=deepcopy(data_handler), selected_layer=i, device="cuda")
    processor.load_checkpoint(checkpoint_path=checkpoint_path)
    processors.append(processor)
    print(f"Loaded processor for layer {i} from {checkpoint_path}")

pcd_combined = o3d.geometry.PointCloud()
gt_pcd = o3d.geometry.PointCloud()

start = time.time()

for processor in processors:
    tic = time.time()
    sem_scan = deepcopy(data_handler.sem_scan)
    ssg = deepcopy(data_handler.ssg)
    latent, coarse_pts, coarse_mask, fine_mask, fine_pts, batch, patches_dict =  processor.autoencode(sem_scan, ssg, scan_id="000000")
    toc = time.time()
    print(f"Time taken for autoencoding: {toc - tic:.2f} seconds")
    valid_points = processor.get_predicted_points(pts=fine_pts, mask=fine_mask)
    # valid_points = processor.get_predicted_points(pts=coarse_pts, mask=coarse_mask)
    print(f"Shape of latent: {latent.shape}")
    patches = processor.remap_patch_ids(patches_dict[0]['patches'])
    data_handler.ssg.populate_patches(patches, latent.detach().to(torch.float16).cpu().numpy(), populate_points=False)

    for i in range(len(valid_points)):
        valid_points[i] = valid_points[i].tolist()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points[i])
        semantic_id = patches[i]['semantic_id']
        color = data_handler.sem_scan.sem_color_lut[semantic_id]
        color = [color[2], color[1], color[0]]
        pcd.colors = o3d.utility.Vector3dVector([tuple(color) for _ in range(len(valid_points[i]))])
        pcd_combined += pcd

    ## Make an o3d point cloud for the ground truth
    for i in range(len(patches)):
        gt_pts = patches[i]['points'].tolist()
        if len(gt_pts) == 0:
            continue
        gt_pts = np.array(gt_pts)
        gt_pcd_temp = o3d.geometry.PointCloud()
        gt_pcd_temp.points = o3d.utility.Vector3dVector(gt_pts[:, :3])
        gt_pcd += gt_pcd_temp


stop = time.time()
print(f"Total time taken for autoencoding: {stop - start:.2f} seconds")

## Calculate BPP
encoded_bytes = pickle.dumps(data_handler.ssg.ssg)
print(f"Size of encoded SSG: {len(encoded_bytes)} bytes")
N = len(pcd_combined.points)
print(f"Number of points in predicted point cloud: {N}")
bpp = len(encoded_bytes) * 8 / N
print(f"BPP for DPCC: {bpp:.4f}")

# Do statistical outlier removal on the predicted points
# cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd_combined = pcd_combined.select_by_index(ind)
## Do voxel downsampling
# voxel_size = 0.01
# pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

o3d.io.write_point_cloud("predicted.pcd", pcd_combined)
o3d.io.write_point_cloud("gt.pcd", gt_pcd)

