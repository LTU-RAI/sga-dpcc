"""
    Point Cloud Compression - Encode Only

    Encodes a SemanticKITTI scan into a compressed semantic scene graph (SSG)
        with latent representations and saves the serialized bytes to disk.
"""

import os, sys, argparse, pickle, yaml
import torch, numpy as np
from copy import deepcopy
# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.Processor import DPCC, Data  


def parse_arguments():
    user = os.environ.get('USER')
    parser = argparse.ArgumentParser(
        description='Encode a scan into compressed SSG bytes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root', type=str, default=f"/home/{user}/Documents/datasets_2/SemanticKitti", help='Root directory of the SemanticKITTI dataset')
    parser.add_argument('--config_path', type=str, default=f"/home/{user}/python_projects/sga-dpcc/config/config-latest.yaml", help='Path to the configuration YAML file')
    parser.add_argument('--selected_layers', type=int, nargs='+', default=[1, 2, 3, 4], help='Layer indices to use for compression (space-separated)')
    parser.add_argument('--max_range', type=float, default=50.0, help='Maximum range for point cloud processing (in meters)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--sequence', type=str, default='00', help='Sequence number from the dataset (e.g., "00")')
    parser.add_argument('--scan_id', type=str, default='000000', help='Scan ID to process (e.g., "000000")')
    parser.add_argument('--output_bytes', type=str, default=None, help='Output path for serialized SSG bytes (defaults to {sequence}_{scan_id}.pkl)')
    return parser.parse_args()


def load_latent_dims(config_path, selected_layers):
    # Load the main config to get autoencoder config path
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
    if args.output_bytes is None:
        args.output_bytes = f"{args.sequence}_{args.scan_id}.pkl"
    latent_dims = load_latent_dims(args.config_path, args.selected_layers)

    print('=' * 50)
    print('Deep Point Cloud Compression - Encode')
    print('=' * 50)
    print(f"Device: {args.device}")
    print(f"Dataset root: {args.root}")
    print(f"Sequence: {args.sequence}, Scan ID: {args.scan_id}")
    print(f"Selected layers: {args.selected_layers}")
    print(f"Latent dimensions (from config): {latent_dims}")
    print(f"Output bytes: {args.output_bytes}")
    print('=' * 50)
    print()

    # Load data
    print('[1/3] Loading data...')
    data_handler = Data(root=args.root, config_path=args.config_path, max_range=args.max_range)
    data_handler.get_data(sequence=args.sequence, scan_id=args.scan_id)
    data_handler.ssg.to_float16()

    # Original size for reference
    pts_bytes = pickle.dumps(data_handler.sem_scan.points)
    labels_bytes = pickle.dumps(data_handler.sem_scan.sem_label)
    original_size = len(pts_bytes) + len(labels_bytes)
    print(f"  Original size (points + labels): {original_size / (1024**2):.2f} MB")

    # Load models and encode
    print('[2/3] Encoding and populating SSG latents...')
    user = os.environ.get('USER')
    for i, layer_idx in enumerate(args.selected_layers):
        checkpoint_path = (
            f"/home/{user}/python_projects/sga-dpcc/weights/checkpoints/latest/"
            f"layer_{layer_idx}/autoencoder_layer_{layer_idx}_{latent_dims[i]}.torch"
        )
        processor = DPCC(data=deepcopy(data_handler), selected_layer=layer_idx, device=args.device)
        processor.load_checkpoint(checkpoint_path=checkpoint_path)

        sem_scan = deepcopy(data_handler.sem_scan)
        ssg = deepcopy(data_handler.ssg)
        latent, coarse_pts, coarse_mask, fine_mask, fine_pts, batch, patches_dict = processor.autoencode(
            sem_scan, ssg, scan_id=args.scan_id
        )
        patches = processor.remap_patch_ids(patches_dict[0]['patches'])
        data_handler.ssg.populate_patches(
            patches,
            latent.detach().to(torch.float16).cpu().numpy(),
            populate_points=False,
        )
        print(f"  ✓ Layer {layer_idx} encoded (latent_dim={latent_dims[i]})")

    # Serialize populated SSG using built-in helper
    print('[3/3] Saving encoded SSG...')
    data_handler.ssg.save_ssg(args.output_bytes)
    encoded_size = os.path.getsize(args.output_bytes)
    print(f"  Encoded size: {encoded_size / (1024**2):.2f} MB")
    print(f"  Saved to: {args.output_bytes}")
    print('\nDone.')


if __name__ == '__main__':
    main()
