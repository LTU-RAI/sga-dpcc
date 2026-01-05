import torch, time, os, sys, yaml, argparse, numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.PatchAutoEncoder import PatchEncoderTransformerWithSem, PatchDecoderFolding, PatchAutoencoderTransformer
from modules.PatchAutoEncoderDataset import PatchDataset
from losses.PatchCDLoss import ChamferDistanceLossWithMask, AssistedChamferDistanceLoss
from losses.PatchEMDLoss import EarthMoverDistanceLossWithMask
from losses.PatchVoxelLoss import VoxelReconstructionLoss
from modules.PatchCollator import PatchCollator  
from torch.utils.data.distributed import DistributedSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_distributed():
    if 'RANK' not in os.environ:
        os.environ['RANK'] = "0"
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = "1"
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = "0"
    
    # Only initialize distributed training if we have multiple processes
    if int(os.environ['WORLD_SIZE']) > 1:
        # Set default MASTER_ADDR and MASTER_PORT if not set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl')
        dist.barrier()
    
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def is_main_process():
    return int(os.environ['LOCAL_RANK']) == 0

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='Layer configuration to use (e.g., 1,2,3,4)')
parser.add_argument('--batch_size', type=int, default=8, help='Global batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--checkpoint', type=str, default="", help='Last checkpoint to load')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
parser.add_argument('--max_range', type=float, default=50.0, help='Maximum range for point cloud processing')
parser.add_argument('--config_path', type=str, default="/home/niksta/python_projects/sga-dpcc/config/config-latest.yaml", help='Path to main config YAML')
parser.add_argument('--root', type=str, default="/home/niksta/Documents/datasets_2/SemanticKitti", help='Root directory of SemanticKITTI dataset')
parser.add_argument('--save_path', type=str, default="/home/niksta/python_projects/sga-dpcc/weights/checkpoints/autoencoder", help='Path to save checkpoints')
parser.add_argument('--device', type=str, default=DEVICE, choices=['cuda', 'cpu'], help='Device to use for training')
args = parser.parse_args()
selected_layer = args.layer
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
NUM_EPOCHS = args.num_epochs
LAST = args.checkpoint
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
MAX_RANGE = args.max_range
CONFIG_PATH = args.config_path
ROOT = args.root
SAVE_PATH = args.save_path
DEVICE = args.device

class Train:
    def __init__(self, device, batch_size, num_workers, num_epochs, 
                 learning_rate, weight_decay, config_path, max_range, root, 
                 last_checkpoint, save_path):
        setup_distributed()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_range = max_range
        self.root = root
        self.save_path = save_path

        self.load_config(config_path)
        self.print_params()
        self.collate()
        self.load_models()
        self.load_dataset()
        self.get_loss_fn()
        self.get_optimizer()

        self.losses = []
        self.val_losses = []
        self.best_loss = 1e9
        self.best_val_loss = 1e9

        self.load_checkpoint(last_checkpoint)

    def load_config(self, config_path: str) -> None:
        self.config = yaml.safe_load(open(config_path, 'r'))
        # Load the semantic-kitti config as well.
        kitti_config_path = self.config['dataset']['semantic-kitti-conf']
        self.kitti_config = yaml.safe_load(open(kitti_config_path, 'r'))
        # Extract the layer-specific configuration.
        autoencoder_cfg_path = self.config['autoencoder']['config']
        self.autoencoder_cfg = yaml.safe_load(open(autoencoder_cfg_path, 'r'))
        self.layer_cfg = self.autoencoder_cfg[selected_layer]
        return

    def print_params(self) -> None:
        if is_main_process():
            print("="*25)
            print(f"{'Parameter':<15} {'Value':<10}")
            print("="*25)
            print(f"{'Device':<14}{':'} {self.device:<10}")
            print(f"{'Batch Size':<14}{':'} {self.batch_size:<10}")
            print(f"{'Num Workers':<14}{':'} {self.num_workers:<10}")
            print(f"{'Num Epochs':<14}{':'} {self.num_epochs:<10}")
            print(f"{'Learning Rate':<14}{':'} {self.learning_rate:<10}")
            print(f"{'Weight Decay':<14}{':'} {self.weight_decay:<10}")
            print(f"{'Save Path':<14}{':'} {self.save_path:<10}")
            print("="*25)
            print("Training initialized...\n")
        return
    
    def collate(self) -> None:
        self.collator = PatchCollator(self.layer_cfg['max_points'])
        print(f"Max points per patch: {self.layer_cfg['max_points']}")
        return 
    
    def load_models(self) -> None:
        # Create encoder and decoder using the layer-specific config.
        enc_cfg = self.layer_cfg['encoder']
        dec_cfg = self.layer_cfg['decoder']
        encoder = PatchEncoderTransformerWithSem(
            input_dim=enc_cfg['input_dim'],
            encoder_blocks=enc_cfg['encoder_blocks'],
            feature_dim=enc_cfg['feature_dim'],
            latent_dim=enc_cfg['latent_dim'],
            pos_enc_dim=enc_cfg['pos_enc_dim'],
            num_classes=enc_cfg['num_classes'],
            sem_dim=enc_cfg['sem_dim'],
            dropout_prob=enc_cfg['dropout_prob']
        ).to(self.device)
        decoder = PatchDecoderFolding(
            latent_dim=dec_cfg['latent_dim'],
            coarse_points=dec_cfg['coarse_points'],
            feature_dim=dec_cfg['feature_dim'],
            upsample_factors=dec_cfg['upsample_factors'],
            decoder_hidden_dims=dec_cfg['decoder_hidden_dims'],
            output_dim=dec_cfg['output_dim'],
            norm_factor=self.max_range
        ).to(self.device)
        self.autoencoder = PatchAutoencoderTransformer(encoder, decoder).to(self.device)
        if int(os.environ.get('WORLD_SIZE', 1)) > 1:
            self.autoencoder = DDP(self.autoencoder, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False)
        self.autoencoder.train()
        
        if is_main_process():
            total_params = sum(p.numel() for p in self.autoencoder.parameters())
            trainable_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
            print("="*50)
            print(f"{'Model Statistics':^50}")
            print("="*50)
            print(f"Total parameters:     {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print("="*50, "\n")
        return

    def load_dataset(self) -> None:
        # Here we assume the PatchDataset returns patches already filtered by desired layers.
        self.train_data = PatchDataset(root=self.root,
                                       split='train', 
                                       config=self.config,
                                       max_range=self.max_range,
                                       desired_layers=[selected_layer],
                                       min_points=self.layer_cfg['min_points'],
                                       max_points=self.layer_cfg['max_points'])
        self.val_data = PatchDataset(root=self.root,
                                     split='valid', 
                                     config=self.config,
                                     max_range=self.max_range,
                                     desired_layers=[selected_layer],
                                     min_points=self.layer_cfg['min_points'],
                                     max_points=self.layer_cfg['max_points'])
        # Setup samplers and dataloaders based on distributed/single-GPU mode
        if int(os.environ.get('WORLD_SIZE', 1)) > 1:
            self.train_sampler = DistributedSampler(self.train_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            self.val_sampler = DistributedSampler(self.val_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            batch_size = self.batch_size // dist.get_world_size()
        else:
            self.train_sampler = None
            self.val_sampler = None
            batch_size = self.batch_size
            
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator.collate
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=batch_size,
            sampler=self.val_sampler,
            shuffle=(self.val_sampler is None),
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator.collate
        )
        return

    def get_loss_fn(self) -> None:
        self.loss_fn = AssistedChamferDistanceLoss(max_points=self.layer_cfg['max_points'],
                                                   coarse_weight=self.layer_cfg['coarse_weight'],
                                                   repulsion_weight=self.layer_cfg['repulsion_weight'],
                                                   repulsion_threshold=self.layer_cfg['repulsion_threshold'],
                                                   density_weight=self.layer_cfg['density_weight'],
                                                   grid_size=self.layer_cfg['grid_size'],
                                                   lambda_mask_coarse_schedule=self.layer_cfg['lambda_mask_coarse_schedule'],
                                                   lambda_mask_final_schedule=self.layer_cfg['lambda_mask_final_schedule'],
                                                   mask_threshold=self.layer_cfg['mask_threshold'])
        return
    
    def get_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return

    def save_checkpoint(self) -> None:
        if is_main_process():
            save_dir = os.path.join(self.save_path, time.strftime("%Y%m%d-%H"))
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_filename = f"autoencoder_layer_{selected_layer}.torch"
            torch.save({
                'model_state_dict': self.autoencoder.state_dict(),
                'losses': self.losses,
                'val_losses': self.val_losses,
                'best_loss': self.best_loss,
                'best_val_loss': self.best_val_loss,
            }, os.path.join(save_dir, checkpoint_filename))
            np.save(os.path.join(save_dir, "losses.npy"), np.array(self.losses), allow_pickle=True)
            np.save(os.path.join(save_dir, "val_losses.npy"), np.array(self.val_losses), allow_pickle=True)
            print(f"Checkpoint saved for layer {selected_layer}.")
        return

    def load_checkpoint(self, last_checkpoint) -> None:
        checkpoint_filename = f"autoencoder_layer_{selected_layer}.torch"
        checkpoint_path = os.path.join(self.save_path, last_checkpoint, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
            self.losses = checkpoint['losses']
            self.val_losses = checkpoint['val_losses']
            self.best_loss = checkpoint['best_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            if is_main_process():
                print(f"Checkpoint for layer {selected_layer} loaded successfully.")
        else:
            if is_main_process():
                print(f"No checkpoint found for layer {selected_layer}, starting fresh.")
        return

    def train(self) -> None:
        min_loss = self.best_loss
        epoch_loop = tqdm(range(self.num_epochs), desc="Training epochs", disable=not is_main_process())
        for epoch in epoch_loop:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if int(os.environ.get('WORLD_SIZE', 1)) > 1:
                dist.barrier()
            self.autoencoder.train()
            train_losses = []
            epoch_start = time.perf_counter()
            batch_loop = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", leave=False, disable=not is_main_process())
            for step, batch in enumerate(batch_loop):
                # Batch now includes: 'points', 'mask', 'semantic', 'center', 'extent', 'orientation'
                points = batch['points'].to(self.device)         # (B_total, max_points, 3)
                mask = batch['mask'].to(self.device)               # (B_total, max_points)
                sem_tensor = batch['semantic'].to(self.device)     # (B_total,)
                center = batch['center'].to(self.device)           # (B_total, 3)
                extent = batch['extent'].to(self.device)           # (B_total, 3)
                orientation = batch['orientation'].to(self.device) # (B_total, 3, 3)
        
                forward_start = time.perf_counter()
                # Pass the additional bbox parameters to the autoencoder.
                latent, coarse_points, coarse_mask, final_mask, reconstruction = self.autoencoder(
                    points, sem_tensor, mask, center, extent, orientation)
                torch.cuda.synchronize()
                forward_time = time.perf_counter() - forward_start
                backward_start = time.perf_counter()
                self.optimizer.zero_grad()
                loss, fine_loss, coarse_loss, mask_loss, rep_loss, density_loss = self.loss_fn(
                    reconstruction, coarse_points, coarse_mask, final_mask, points, mask, epoch)
                loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                backward_time = time.perf_counter() - backward_start

                train_losses.append(loss.item())
                if is_main_process():
                    batch_loop.set_postfix(L_t=f"{loss.item():.3f}",
                                           L_f=f"{fine_loss.item():.3f}",
                                           L_c=f"{coarse_loss.item():.3f}",
                                           L_m=f"{mask_loss.item():.3f}",
                                           L_d=f"{density_loss.item():.3f}",
                                           L_r=f"{rep_loss.item():.3f}")
            epoch_loss = np.mean(train_losses)
            self.losses.append(epoch_loss)
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                self.best_model = self.autoencoder.state_dict()
                self.save_checkpoint()
            if is_main_process():
                epoch_loop.set_postfix(train_loss=f"{epoch_loss:.3f}", best_loss=f"{min_loss:.3f}")
            # if epoch % 10 == 0:
            #     val_loss = self.validate(epoch)
            #     self.val_losses.append(val_loss)
            #     if is_main_process():
            #         print(f"Epoch {epoch} validation loss: {val_loss:.3f}")
        if is_main_process():
            print(f"Training finished. Best loss: {min_loss:.3f}")

    def validate(self, epoch: int) -> float:
        self.autoencoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                points = batch['points'].to(self.device)
                mask = batch['mask'].to(self.device)
                sem_tensor = batch['semantic'].to(self.device)
                center = batch['center'].to(self.device)
                extent = batch['extent'].to(self.device)
                orientation = batch['orientation'].to(self.device)
                latent, coarse_points, coarse_mask, final_mask, reconstruction = self.autoencoder(
                    points, sem_tensor, mask, center, extent, orientation)
                loss, _, _, _, _, _ = self.loss_fn(
                    reconstruction, coarse_points, coarse_mask, final_mask, points, mask, epoch)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        self.autoencoder.train()
        return avg_val_loss

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['LOCAL_RANK'] = str(int(os.environ.get('RANK', '0')) % torch.cuda.device_count())
    train = Train(device=DEVICE,
                  batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS,
                  num_epochs=NUM_EPOCHS,
                  learning_rate=LEARNING_RATE,
                  weight_decay=WEIGHT_DECAY,
                  config_path=CONFIG_PATH,
                  max_range=MAX_RANGE,
                  root=ROOT,
                  last_checkpoint=LAST,
                  save_path=SAVE_PATH)
    train.train()
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.destroy_process_group()
