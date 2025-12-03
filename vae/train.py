# -*- coding: utf-8 -*-
"""
VAE training script.
Train a VAE model to compress WSI embeddings, typically using only "living" patients.
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import VAE, Encoder, Decoder
from loss import vae_loss
from dataset import WSIVAEDataset


class VAETrainer:
    """
    VAE trainer.
    """
    
    def __init__(self,
                 model: VAE,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 save_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 lr_patience: int = 5,
                 lr_factor: float = 0.5,
                 lr_min: float = 1e-6):
        """
        Initialize trainer.

        Args:
            model: VAE model.
            train_loader: training DataLoader.
            val_loader: optional validation DataLoader.
            device: device string.
            learning_rate: learning rate.
            save_dir: directory to save checkpoints.
            log_dir: directory to save logs.
            lr_patience: patience for LR scheduler.
            lr_factor: LR decay factor.
            lr_min: minimum LR.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer (with reasonable defaults)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # LR scheduler (ReduceLROnPlateau based on validation loss)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=True
        )
        
        # Save and log directories
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training history
        self.train_history = {
            'loss': [],
            'recon_loss': [],
            'kld_loss': []
        }
        self.val_history = {
            'loss': [],
            'recon_loss': [],
            'kld_loss': []
        }
        
        # Track LR scheduler reductions (used for dynamic resample strategy)
        self.lr_reduce_count = 0
    
    def train_epoch(self, epoch: int, global_step: int = 0) -> Tuple[Dict[str, float], int]:
        """
        Train for one epoch.

        Args:
            epoch: current epoch index.
            global_step: global step counter (for resampling strategy).

        Returns:
            A dict of training metrics and the updated global step.
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        num_batches = 0
        current_step = global_step
        
        for batch_idx, patch_features in enumerate(self.train_loader):
            # Move patch features to device
            # patch_features shape: (batch_size, feature_dim)

            # Ensure 2D tensor (batch_size, feature_dim)
            if patch_features.dim() == 1:
                patch_features = patch_features.unsqueeze(0)
            
            embeddings = patch_features.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            x_hat, z, mean, log_var = self.model(embeddings)
            loss, recon_loss, kld_loss = vae_loss(
                x=embeddings,
                x_hat=x_hat,
                mean=mean,
                log_var=log_var
            )
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            num_batches += 1
            current_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Step {current_step}, '
                      f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, '
                      f'KLD: {kld_loss.item():.4f}')
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/ReconLoss', avg_recon_loss, epoch)
        self.writer.add_scalar('Train/KLDLoss', avg_kld_loss, epoch)
        self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Update history
        self.train_history['loss'].append(avg_loss)
        self.train_history['recon_loss'].append(avg_recon_loss)
        self.train_history['kld_loss'] = self.train_history.get('kld_loss', [])
        self.train_history['kld_loss'].append(avg_kld_loss)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kld_loss': avg_kld_loss
        }, current_step
    
    def validate(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        Run validation.

        Args:
            epoch: current epoch index.

        Returns:
            Dict of validation metrics, or None if no val_loader.
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for patch_features in self.val_loader:
                # 确保是2D张量 (batch_size, feature_dim)
                if patch_features.dim() == 1:
                    patch_features = patch_features.unsqueeze(0)
                
                embeddings = patch_features.to(self.device)
                
                # 前向传播
                x_hat, z, mean, log_var = self.model(embeddings)
                
                # 计算损失（使用KLD损失）
                loss, recon_loss, kld_loss = vae_loss(
                    x=embeddings,
                    x_hat=x_hat,
                    mean=mean,
                    log_var=log_var
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kld_loss += kld_loss.item()
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/ReconLoss', avg_recon_loss, epoch)
        self.writer.add_scalar('Val/KLDLoss', avg_kld_loss, epoch)
        
        # Step LR scheduler based on validation loss
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(avg_loss)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        # Detect LR reduction (scheduler trigger)
        if new_lr < old_lr:
            self.lr_reduce_count += 1
            print(f'📉 Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e} (times triggered: {self.lr_reduce_count})')
            self.writer.add_scalar('Train/LRReduceCount', self.lr_reduce_count, epoch)
        
        # Update history
        self.val_history['loss'].append(avg_loss)
        self.val_history['recon_loss'].append(avg_recon_loss)
        self.val_history['kld_loss'] = self.val_history.get('kld_loss', [])
        self.val_history['kld_loss'].append(avg_kld_loss)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kld_loss': avg_kld_loss
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint.

        Args:
            epoch: current epoch index.
            is_best: whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'lr_reduce_count': self.lr_reduce_count
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'✅ Saved best model: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.

        Args:
            checkpoint_path: checkpoint file path.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.val_history = checkpoint.get('val_history', self.val_history)
        self.lr_reduce_count = checkpoint.get('lr_reduce_count', 0)
        print(f'✅ Loaded checkpoint: {checkpoint_path}')
        print(f'📊 LR scheduler reduce count: {self.lr_reduce_count}')
        return checkpoint['epoch']
    
    def get_resample_strategy(self, total_steps: int) -> Dict[str, any]:
        """
        Get current resampling strategy based on LR scheduler reduce count.

        Args:
            total_steps: total number of training steps.

        Returns:
            Dict describing the resample strategy.
        """
        if self.lr_reduce_count == 0:
            # Initial: resample every 10% of total steps
            resample_freq_percent = 0.10
            resample_freq_steps = int(0.10 * total_steps)
            strategy_name = "10%总步数"
        elif self.lr_reduce_count == 1:
            # After first patience trigger: resample every 5% of total steps
            resample_freq_percent = 0.05
            resample_freq_steps = int(0.05 * total_steps)
            strategy_name = "5%总步数"
        else:
            # After second patience trigger: resample every epoch
            resample_freq_percent = None
            resample_freq_steps = None
            strategy_name = "每1个epoch"
        
        return {
            'freq_percent': resample_freq_percent,
            'freq_steps': resample_freq_steps,
            'name': strategy_name,
            'lr_reduce_count': self.lr_reduce_count
        }
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train a VAE model')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='Root directory of data')
    parser.add_argument('--label_filter', type=str, default='living',
                        help='Label to keep (default: living). If empty/None, use all data.')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=None,
                        help='Input feature dimension (if None, infer automatically)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                        help='Hidden layer dimensions (default: [512, 256])')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension (default: 128)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    # Note: beta parameter is removed; the paper uses L_VAE = L_MSE + L_KLD without extra weight.
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency (every N epochs, default: 1)')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='LR scheduler patience (default: 5)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR scheduler decay factor (default: 0.5)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (default: 10). '
                             'Triggered when LR is at minimum and val loss does not improve.')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum improvement (delta) for early stopping (default: 1e-4)')
    
    # Misc arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs (default: ./logs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    
    args = parser.parse_args()
    
    # Select device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'🖥️  Using device: {device}')
    
    # Create dataset
    print('📂 Loading dataset...')
    full_dataset = WSIVAEDataset(
        csv_path=args.csv_path,
        data_root_dir=args.data_root_dir,
        label_filter=args.label_filter,
        preload_data=True,  # preload all data into memory
        print_info=True
    )
    
    # Train/val split
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'📊 Dataset split: train={train_size}, val={val_size}')
    
    # Keep reference to original dataset for resampling
    # random_split wraps the underlying dataset under .dataset
    base_dataset_for_resample = train_dataset.dataset
    
    # Create dataloaders (with optimized config)
    num_workers = min(8, os.cpu_count() or 1)  # more workers but capped by CPU cores
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,  # keep workers alive
        prefetch_factor=2  # prefetch factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    ) if val_size > 0 else None
    
    # Infer input dimension
    if args.input_dim is None:
        sample = full_dataset[0]
        if sample.dim() == 2:
            input_dim = sample.shape[1]
        else:
            input_dim = sample.shape[0]
        print(f'🔍 Inferred input_dim: {input_dim}')
    else:
        input_dim = args.input_dim
    
    # Build model
    print('🏗️  Building model...')
    encoder = Encoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim
    )
    decoder = Decoder(
        latent_dim=args.latent_dim,
        hidden_dims=list(reversed(args.hidden_dims)),
        output_dim=input_dim
    )
    model = VAE(encoder, decoder, device=device)
    
    # Use torch.compile for speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('⚡ Enabled torch.compile acceleration')
        except Exception as e:
            print(f'⚠️  Failed to enable torch.compile: {e}, falling back to eager mode')
    
    print(f'📊 Number of model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create trainer
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, f'vae_{timestamp}')
    log_dir = os.path.join(args.log_dir, f'vae_{timestamp}')
    
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        log_dir=log_dir,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min
    )
    
    # Resume training if requested
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        # Approximate trained steps
        global_step = start_epoch * len(train_loader)
    
    # Total steps
    total_steps = args.epochs * len(train_loader)
    
    # Dynamic resample strategy controlled by LR scheduler:
    # initial: 10% total steps -> after first patience: 5% -> after second: every epoch.
    last_resample_step = 0  # step when we last resampled
    last_strategy_lr_count = -1  # LR reduce count at last strategy update
    
    print(f'📊 Total training steps: {total_steps}')
    print(f'📈 LR scheduler: patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.lr_min}')
    print('🔄 Resample strategy: 10% total steps -> 5% total steps -> every epoch')
    
    # Training loop
    print('🚀 Start training...')
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0  # early stop counter
    early_stop_triggered = False  # whether early stop has been triggered
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        # Train
        train_metrics, global_step = trainer.train_epoch(epoch, global_step)
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Recon: {train_metrics["recon_loss"]:.4f}, '
              f'KLD: {train_metrics["kld_loss"]:.4f}')
        
        # Get current resample strategy based on LR scheduler state
        resample_strategy = trainer.get_resample_strategy(total_steps)
        current_lr_count = resample_strategy['lr_reduce_count']
        
        # If strategy changed, reset last_resample_step
        if current_lr_count != last_strategy_lr_count:
            print(f'🔄 Resample strategy switched: {resample_strategy["name"]} (LR reduce count: {current_lr_count})')
            last_resample_step = global_step  # reset to current step
            last_strategy_lr_count = current_lr_count
        else:
            print(f'🔄 Current resample strategy: {resample_strategy["name"]} (LR reduce count: {current_lr_count})')
        
        # Decide whether to resample
        should_resample = False
        resample_reason = None
        
        if resample_strategy['freq_steps'] is not None:
            # Step-based strategy (10% or 5% of total steps)
            next_resample_step = last_resample_step + resample_strategy['freq_steps']
            if global_step >= next_resample_step:
                should_resample = True
                resample_reason = resample_strategy['name']
                last_resample_step = global_step
        else:
            # Epoch-based strategy (every epoch)
            if epoch > 0:  # skip first epoch
                should_resample = True
                resample_reason = resample_strategy['name']
        
        if should_resample:
            print(f"🔄 Resampling training patches (reason: {resample_reason}, step: {global_step}/{total_steps})...")
            # Resample patches (updates patch_indices)
            base_dataset_for_resample.resample_patches()
            # Recreate DataLoader with new indices
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if device == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2
            )
            trainer.train_loader = train_loader
            print(f"✅ Resample finished, current train dataset size: {len(train_dataset)}")
        
        # Validation (according to frequency)
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            val_metrics = trainer.validate(epoch)
            print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
                  f'Recon: {val_metrics["recon_loss"]:.4f}, '
                  f'KLD: {val_metrics["kld_loss"]:.4f}')
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            print(f'📉 Current LR: {current_lr:.2e}')
            
            # Track best model
            current_val_loss = val_metrics["loss"]
            lr_reduce_count = trainer.lr_reduce_count
            
            if current_val_loss < best_val_loss - args.min_delta:
                # Validation loss improved sufficiently
                best_val_loss = current_val_loss
                best_epoch = epoch
                is_best = True
                early_stop_counter = 0  # reset early stop counter
                print(f'✨ Validation improved: best {best_val_loss:.4f} @ Epoch {best_epoch+1}')
            else:
                is_best = False
                # Check if early stopping conditions are met
                # Condition 1: LR at minimum and no improvement for patience epochs.
                # Condition 2: already in "every-epoch resample" phase (lr_reduce_count >= 2)
                #              and no improvement.
                should_check_early_stop = False
                early_stop_reason = None
                
                if current_lr <= args.lr_min:
                    should_check_early_stop = True
                    early_stop_reason = "LR reached minimum value"
                elif lr_reduce_count >= 2:
                    # Final phase: resample every epoch
                    should_check_early_stop = True
                    early_stop_reason = f"已进入final阶段（LR触发{lr_reduce_count}次）"
                
                if should_check_early_stop:
                    early_stop_counter += 1
                    print(f'⏳ Early stop counter: {early_stop_counter}/{args.early_stop_patience} ({early_stop_reason})')
                    if early_stop_counter >= args.early_stop_patience:
                        early_stop_triggered = True
                        print(f'🛑 Early stop triggered! {early_stop_reason} and no val improvement in {args.early_stop_patience} epochs.')
                        break
        else:
            # No validation set: always save checkpoint, but never "best"
            is_best = False
        
        # Save checkpoint
        trainer.save_checkpoint(epoch, is_best=is_best)
        
        # Break if early stop has been triggered
        if early_stop_triggered:
            break
    
    # Close resources
    trainer.close()
    print('\n✅ Training finished!')
    if early_stop_triggered:
        print('   🛑 Stopped early due to early stopping criteria.')
    if val_loader is not None:
        print(f'   Best validation loss: {best_val_loss:.4f} @ Epoch {best_epoch+1}')
    print(f'   Total epochs run: {epoch+1}/{args.epochs}')


if __name__ == '__main__':
    main()

