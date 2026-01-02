"""
    Point Cloud Compression - Decode Only

    Decodes compressed semantic scene graph (SSG) bytes into a reconstructed
        point cloud and saves the result as a .pcd file.
"""

import os, sys, argparse, pickle, yaml
import torch, numpy as np
import open3d as o3d
from copy import deepcopy
# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.Processor import DPCC, Data  
from modules.Metrics import chamfer_distance  


def parse_arguments():
    user = os.environ.get('USER')
    parser = argparse.ArgumentParser(
        description='Decode compressed SSG bytes into a point cloud',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root', type=str, default=f"/home/{user}/Documents/datasets_2/SemanticKitti", help='Root directory of the SemanticKITTI dataset')
    parser.add_argument('--config_path', type=str, default=f"/home/{user}/python_projects/sga-dpcc/config/config-latest.yaml", help='Path to the configuration YAML file')
    parser.add_argument('--selected_layers', type=int, nargs='+', default=[1, 2, 3, 4], help='Layer indices to decode (space-separated)')
    parser.add_argument('--max_range', type=float, default=50.0, help='Maximum range for point cloud processing (in meters)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help='Device to use for decoding')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence number from the dataset (for color LUT loading)')
    parser.add_argument('--scan_id', type=str, default='000000', help='Scan ID (for color LUT loading)')
    parser.add_argument('--input_bytes', type=str, default=None, help='Path to serialized SSG bytes (defaults to {sequence}_{scan_id}.pkl)')
    parser.add_argument('--output_pcd', type=str, default=None, help='Output path for reconstructed point cloud (defaults to {sequence}_{scan_id}.pcd)')
    parser.add_argument('--skip_gt', action='store_true', default=False, help='Skip loading ground truth scan (for decoding without access to original data)')
    parser.add_argument('--compute_chamfer', action='store_true', default=False, help='Compute Chamfer distance metric (slower, requires ground truth)')
    return parser.parse_args()


def load_latent_dims(config_path, selected_layers):
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)

    config_dir = os.path.dirname(config_path)
    autoencoder_config_path = os.path.join(config_dir, 'autoencoder.yaml')
    with open(autoencoder_config_path, 'r') as f:
        autoencoder_config = yaml.safe_load(f)

    latent_dims = []
    for layer_idx in selected_layers:
        if layer_idx in autoencoder_config:
            latent_dim = autoencoder_config[layer_idx]['encoder']['latent_dim']
            latent_dims.append(latent_dim)
        else:
            raise ValueError(f"Layer {layer_idx} not found in autoencoder config")
    return latent_dims


def main():
    args = parse_arguments()
    # Derive default file names from sequence/scan if not provided
    default_stem = f"{args.sequence}_{args.scan_id}"
    if args.input_bytes is None:
        args.input_bytes = f"{default_stem}.pkl"
    if args.output_pcd is None:
        args.output_pcd = f"{default_stem}.pcd"
    latent_dims = load_latent_dims(args.config_path, args.selected_layers)

    print('=' * 50)
    print('Deep Point Cloud Compression - Decode')
    print('=' * 50)
    print(f"Device: {args.device}")
    print(f"Dataset root: {args.root}")
    print(f"Sequence: {args.sequence}, Scan ID: {args.scan_id}")
    print(f"Selected layers: {args.selected_layers}")
    print(f"Latent dimensions (from config): {latent_dims}")
    print(f"Input bytes: {args.input_bytes}")
    print(f"Output PCD: {args.output_pcd}")
    print('=' * 50)
    print()

    # Load data (for color LUT and config)
    data_handler = Data(root=args.root, config_path=args.config_path, max_range=args.max_range)
    if args.skip_gt:
        print('[1/3] Initializing decoder (skipping ground truth)...')
        # Load color LUT from config without loading the actual scan
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_handler.sem_color_lut = np.array(config['color_map'])
    else:
        print('[1/4] Loading scan metadata (for colors)...')
        data_handler.get_sem_scan(sequence=args.sequence, scan_id=args.scan_id)

    # Load encoded SSG via helper to preserve metadata
    print('[2/3] Loading encoded SSG...' if args.skip_gt else '[2/4] Loading encoded SSG...')
    data_handler.ssg.load_ssg(args.input_bytes)
    compressed_size = os.path.getsize(args.input_bytes)
    print(f"  Encoded size: {compressed_size / (1024**2):.2f} MB")

    # Decode per layer
    print('[3/3] Decoding point clouds...' if args.skip_gt else '[3/4] Decoding point clouds...')
    user = os.environ.get('USER')
    pcd_combined = o3d.geometry.PointCloud()

    # Load metrics only if ground truth is available
    original_size = None
    if not args.skip_gt:
        pts_bytes = pickle.dumps(data_handler.sem_scan.points)
        labels_bytes = pickle.dumps(data_handler.sem_scan.sem_label)
        original_size = len(pts_bytes) + len(labels_bytes)

    for i, layer_idx in enumerate(args.selected_layers):
        checkpoint_path = (
            f"/home/{user}/python_projects/sga-dpcc/weights/checkpoints/latest/"
            f"layer_{layer_idx}/autoencoder_layer_{layer_idx}_{latent_dims[i]}.torch"
        )
        processor = DPCC(data=deepcopy(data_handler), selected_layer=layer_idx, device=args.device)
        processor.load_checkpoint(checkpoint_path=checkpoint_path)

        decode_result = processor.decode_only(data_handler.ssg)
        if decode_result is None:
            print(f"  Layer {layer_idx}: no latents found, skipping")
            continue

        coarse_pts, coarse_mask, fine_mask, fine_pts, semantic_ids = decode_result
        valid_points = processor.get_predicted_points(fine_pts, fine_mask)

        for j in range(len(valid_points)):
            pts = valid_points[j].tolist()
            if len(pts) == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            sem_id = int(semantic_ids[j].item()) if hasattr(semantic_ids[j], 'item') else int(semantic_ids[j])
            color = data_handler.sem_scan.sem_color_lut[sem_id]
            color = [color[2], color[1], color[0]]  # BGR to RGB
            pcd.colors = o3d.utility.Vector3dVector([tuple(color) for _ in range(len(pts))])
            pcd_combined += pcd

        print(f"  ✓ Layer {layer_idx} decoded (latent_dim={latent_dims[i]})")

    # Metrics
    print('[4/4] Computing metrics and saving...' if not args.skip_gt else 'Saving...')
    num_points = len(pcd_combined.points)

    print(f"  Compressed size: {compressed_size / (1024**2):.2f} MB")
    if not args.skip_gt:
        bpp = (compressed_size * 8) / len(data_handler.sem_scan.points) if len(data_handler.sem_scan.points) > 0 else 0
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        print(f"  Original size: {original_size / (1024**2):.2f} MB")
        print(f"  Bits per point (BPP): {bpp:.4f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Size reduction: {(1 - compressed_size/original_size)*100:.2f}%")
    print(f"  Number of points (decoded): {num_points:,}")

    # Chamfer distance between decoded and ground truth (optional, requires ground truth)
    if args.compute_chamfer:
        if args.skip_gt:
            print('  Chamfer distance: N/A (ground truth skipped)')
        else:
            pred_pts_np = np.asarray(pcd_combined.points)
            gt_pts_np = np.asarray(data_handler.sem_scan.points)[:, :3] if len(data_handler.sem_scan.points) > 0 else np.array([])
            if pred_pts_np.size == 0 or gt_pts_np.size == 0:
                print('  Chamfer distance: N/A (empty point cloud)')
            else:
                pred_tensor = torch.from_numpy(pred_pts_np).float().to(args.device)
                gt_tensor = torch.from_numpy(gt_pts_np).float().to(args.device)
                chamfer_val = chamfer_distance(pred_tensor, gt_tensor).item()
                print(f"  Chamfer distance: {chamfer_val:.6f}")

    o3d.io.write_point_cloud(args.output_pcd, pcd_combined)
    print(f"  ✓ Decoded point cloud saved to: {args.output_pcd}")
    print('\nDone.')


if __name__ == '__main__':
    main()
