"""
    Point Cloud Compression Inference
    ================================
    It encodes point clouds into compact semantic scene graphs with latent representations,
        then decodes them back to reconstruct the original point cloud.
"""

import os, sys, time, yaml, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import pickle, argparse
from copy import deepcopy
# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.Processor import DPCC, Data
from modules.Metrics import chamfer_distance


def parse_arguments():
    """Parse command-line arguments for the compression inference."""
    user = os.environ.get('USER')
    
    parser = argparse.ArgumentParser(
        description='Deep Point Cloud Compression Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and configuration paths
    parser.add_argument(
        '--root',
        type=str,
        default=f"/home/{user}/Documents/datasets_2/SemanticKitti",
        help='Root directory of the SemanticKITTI dataset'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=f"/home/{user}/python_projects/sga-dpcc/config/config-latest.yaml",
        help='Path to the configuration YAML file'
    )
    
    # Model parameters
    parser.add_argument(
        '--selected_layers',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help='List of layer indices to use for compression (space-separated)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--max_range',
        type=float,
        default=50.0,
        help='Maximum range for point cloud processing (in meters)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    # Data selection
    parser.add_argument(
        '--sequence',
        type=str,
        default='00',
        help='Sequence number from the dataset (e.g., "00")'
    )
    parser.add_argument(
        '--scan_id',
        type=str,
        default='000000',
        help='Scan ID to process (e.g., "000000")'
    )
    
    args = parser.parse_args()
    return args


def load_latent_dims(config_path, selected_layers):
    """Load latent dimensions for selected layers from autoencoder config.
    
    Args:
        config_path: Path to the main config YAML file
        selected_layers: List of layer indices to get latent dims for
        
    Returns:
        List of latent dimensions corresponding to selected_layers
    """
    # Load the main config to get autoencoder config path
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Get autoencoder config path (it should be in the same directory)
    config_dir = os.path.dirname(config_path)
    autoencoder_config_path = os.path.join(config_dir, 'autoencoder.yaml')
    
    # Load autoencoder config
    with open(autoencoder_config_path, 'r') as f:
        autoencoder_config = yaml.safe_load(f)
    
    # Extract latent dimensions for selected layers
    latent_dims = []
    for layer_idx in selected_layers:
        if layer_idx in autoencoder_config:
            latent_dim = autoencoder_config[layer_idx]['encoder']['latent_dim']
            latent_dims.append(latent_dim)
        else:
            raise ValueError(f"Layer {layer_idx} not found in autoencoder config")
    
    return latent_dims


def main():
    """Main inference pipeline for point cloud compression."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load latent dimensions from config
    latent_dims = load_latent_dims(args.config_path, args.selected_layers)
    
    print("=" * 50)
    print("Deep Point Cloud Compression - Inference")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Dataset root: {args.root}")
    print(f"Sequence: {args.sequence}, Scan ID: {args.scan_id}")
    print(f"Selected layers: {args.selected_layers}")
    print(f"Latent dimensions (from config): {latent_dims}")
    print("=" * 50)
    print()

    # ========================================
    # Load and Prepare Data
    # ========================================
    print("[1/5] Loading data...")
    data_handler = Data(root=args.root, config_path=args.config_path, max_range=args.max_range)
    data_handler.get_data(sequence=args.sequence, scan_id=args.scan_id)
    
    # Convert to float16 for compression
    data_handler.ssg.to_float16()
    
    # Calculate original data size (before compression)
    encoded_bytes = pickle.dumps(data_handler.ssg.ssg)
    pts_bytes = pickle.dumps(data_handler.sem_scan.points)
    labels_bytes = pickle.dumps(data_handler.sem_scan.sem_label)
    original_size = len(pts_bytes) + len(labels_bytes)
    
    print(f"  Size of SSG (before latents): {len(encoded_bytes) / (1024**2):.2f} MB")
    print(f"  Size of original point cloud (points + labels): {original_size / (1024**2):.2f} MB")
    print()

    # ========================================
    # Load Pre-trained Models
    # ========================================
    print("[2/5] Loading pre-trained autoencoder models...")
    user = os.environ.get('USER')
    processors = []
    
    for i, layer_idx in enumerate(args.selected_layers):
        checkpoint_path = (
            f"/home/{user}/python_projects/sga-dpcc/weights/checkpoints/latest/"
            f"layer_{layer_idx}/autoencoder_layer_{layer_idx}_{latent_dims[i]}.torch"
        )
        
        # Initialize processor for this layer
        processor = DPCC(data=deepcopy(data_handler), selected_layer=layer_idx, device=args.device)
        processor.load_checkpoint(checkpoint_path=checkpoint_path)
        processors.append(processor)
        
        print(f"  ✓ Layer {layer_idx}: latent_dim={latent_dims[i]} ({checkpoint_path})")
    
    print()

    # ========================================
    # Encode and Decode Point Clouds
    # ========================================
    print("[3/5] Running compression and decompression...")
    pcd_combined = o3d.geometry.PointCloud()  # Reconstructed point cloud
    gt_pcd = o3d.geometry.PointCloud()  # Ground truth point cloud
    
    start_time = time.time()
    
    for idx, processor in enumerate(processors):
        layer_idx = args.selected_layers[idx]
        print(f"  Processing layer {layer_idx}...", end=' ')
        
        tic = time.time()
        
        # Create deep copies to avoid modifying original data
        sem_scan = deepcopy(data_handler.sem_scan)
        ssg = deepcopy(data_handler.ssg)
        
        # Encode and decode the point cloud patches
        latent, coarse_pts, coarse_mask, fine_mask, fine_pts, batch, patches_dict = \
            processor.autoencode(sem_scan, ssg, scan_id=args.scan_id)
        
        toc = time.time()
        
        # Extract valid reconstructed points using fine resolution
        valid_points = processor.get_predicted_points(pts=fine_pts, mask=fine_mask)
        # Alternative: use coarse resolution (uncomment if needed)
        # valid_points = processor.get_predicted_points(pts=coarse_pts, mask=coarse_mask)
        
        print(f"latent shape: {latent.shape}, time: {toc - tic:.2f}s")
        
        # Remap patch IDs and populate SSG with latent representations
        patches = processor.remap_patch_ids(patches_dict[0]['patches'])
        data_handler.ssg.populate_patches(
            patches, 
            latent.detach().to(torch.float16).cpu().numpy(), 
            populate_points=False
        )
        
        # Build reconstructed point cloud with semantic colors
        for i in range(len(valid_points)):
            valid_points[i] = valid_points[i].tolist()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points[i])
            
            # Apply semantic color from the lookup table
            semantic_id = patches[i]['semantic_id']
            color = data_handler.sem_scan.sem_color_lut[semantic_id]
            color = [color[2], color[1], color[0]]  # BGR to RGB
            pcd.colors = o3d.utility.Vector3dVector(
                [tuple(color) for _ in range(len(valid_points[i]))]
            )
            pcd_combined += pcd
        
        # Build ground truth point cloud
        for i in range(len(patches)):
            gt_pts = patches[i]['points'].tolist()
            if len(gt_pts) == 0:
                continue
            gt_pts = np.array(gt_pts)
            gt_pcd_temp = o3d.geometry.PointCloud()
            gt_pcd_temp.points = o3d.utility.Vector3dVector(gt_pts[:, :3])
            gt_pcd += gt_pcd_temp
    
    total_time = time.time() - start_time
    print(f"\n  Total compression/decompression time: {total_time:.2f} seconds")
    print()


    # ========================================
    # Calculate Compression Metrics
    # ========================================
    print("[4/5] Calculating compression metrics...")
    
    # Get final compressed size (SSG with latent representations)
    encoded_bytes = pickle.dumps(data_handler.ssg.ssg)
    compressed_size = len(encoded_bytes)
    
    # Calculate bits per point (BPP)
    num_points = len(pcd_combined.points)
    bpp = (compressed_size * 8) / len(sem_scan.points) if len(sem_scan.points) > 0 else 0
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"  Compressed size: {compressed_size / (1024**2):.2f} MB")
    print(f"  Original size: {original_size / (1024**2):.2f} MB")
    print(f"  Number of points (reconstructed): {num_points:,}")
    print(f"  Bits per point (BPP): {bpp:.4f}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Size reduction: {(1 - compressed_size/original_size)*100:.2f}%")
    print()
    
    # ========================================
    # Save Output Point Clouds
    # ========================================
    print("[5/5] Saving output files...")
    
    output_predicted = "predicted.pcd"
    output_gt = "gt.pcd"
    
    o3d.io.write_point_cloud(output_predicted, pcd_combined)
    o3d.io.write_point_cloud(output_gt, gt_pcd)
    
    print(f"  ✓ Predicted point cloud saved to: {output_predicted}")
    print(f"  ✓ Ground truth point cloud saved to: {output_gt}")
    print()
    print("=" * 50)
    print("Inference completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

