# -*- coding: utf-8 -*-
"""
VAE训练脚本
训练VAE模型来压缩WSI embeddings，只使用living病人的数据
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import VAE, Encoder, Decoder
from loss import vae_loss
from dataset import WSIVAEDataset


class VAETrainer:
    """
    VAE训练器类
    """
    
    def __init__(self,
                 model: VAE,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 save_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        """
        初始化训练器
        
        Args:
            model: VAE模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 设备
            learning_rate: 学习率
            save_dir: 模型保存目录
            log_dir: 日志保存目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 保存和日志目录
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 训练历史
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        num_batches = 0
        
        for batch_idx, patch_features in enumerate(self.train_loader):
            # 将patch features移动到设备
            # patch_features形状: (batch_size, feature_dim)
            
            # 确保是2D张量 (batch_size, feature_dim)
            if patch_features.dim() == 1:
                patch_features = patch_features.unsqueeze(0)
            
            embeddings = patch_features.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            x_hat, z, mean, log_var = self.model(embeddings)
            
            # 计算损失（使用KLD损失，符合论文要求）
            loss, recon_loss, kld_loss = vae_loss(
                x=embeddings,
                x_hat=x_hat,
                mean=mean,
                log_var=log_var
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, '
                      f'KLD: {kld_loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/ReconLoss', avg_recon_loss, epoch)
        self.writer.add_scalar('Train/KLDLoss', avg_kld_loss, epoch)
        
        # 更新历史
        self.train_history['loss'].append(avg_loss)
        self.train_history['recon_loss'].append(avg_recon_loss)
        self.train_history['kld_loss'] = self.train_history.get('kld_loss', [])
        self.train_history['kld_loss'].append(avg_kld_loss)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kld_loss': avg_kld_loss
        }
    
    def validate(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        验证模型
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            验证指标字典，如果val_loader为None则返回None
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
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kld_loss = total_kld_loss / num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/ReconLoss', avg_recon_loss, epoch)
        self.writer.add_scalar('Val/KLDLoss', avg_kld_loss, epoch)
        
        # 更新历史
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
        保存模型检查点
        
        Args:
            epoch: 当前epoch编号
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'✅ 保存最佳模型: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.val_history = checkpoint.get('val_history', self.val_history)
        print(f'✅ 加载检查点: {checkpoint_path}')
        return checkpoint['epoch']
    
    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练VAE模型')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV文件路径')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--label_filter', type=str, default='living',
                        help='要保留的标签（默认: living）。如果设置为None或空字符串，则使用全部数据')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=None,
                        help='输入特征维度（如果为None则自动推断）')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                        help='隐藏层维度列表（默认: [512, 256]）')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='潜在空间维度（默认: 128）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小（默认: 32）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（默认: 100）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率（默认: 1e-4）')
    # 注意：beta参数已移除，因为论文中KLD项没有权重（L_VAE = L_MSE + L_KLD）
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例（默认: 0.2）')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience（默认: 10，即10个epoch没有改善则停止）')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Early stopping最小改善阈值（默认: 1e-4）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（默认: cuda）')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录（默认: ./checkpoints）')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录（默认: ./logs）')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'🖥️  使用设备: {device}')
    
    # 创建数据集
    print('📂 加载数据集...')
    full_dataset = WSIVAEDataset(
        csv_path=args.csv_path,
        data_root_dir=args.data_root_dir,
        label_filter=args.label_filter,
        print_info=True
    )
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'📊 数据集划分: 训练集 {train_size}, 验证集 {val_size}')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if val_size > 0 else None
    
    # 获取输入维度
    if args.input_dim is None:
        sample = full_dataset[0]
        if sample.dim() == 2:
            input_dim = sample.shape[1]
        else:
            input_dim = sample.shape[0]
        print(f'🔍 自动推断输入维度: {input_dim}')
    else:
        input_dim = args.input_dim
    
    # 创建模型
    print('🏗️  构建模型...')
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
    
    print(f'📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建训练器
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
        log_dir=log_dir
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # 训练循环
    print('🚀 开始训练...')
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        # 训练
        train_metrics = trainer.train_epoch(epoch)
        print(f'训练 - Loss: {train_metrics["loss"]:.4f}, '
              f'Recon: {train_metrics["recon_loss"]:.4f}, '
              f'KLD: {train_metrics["kld_loss"]:.4f}')
        
        # 验证
        if val_loader is not None:
            val_metrics = trainer.validate(epoch)
            print(f'验证 - Loss: {val_metrics["loss"]:.4f}, '
                  f'Recon: {val_metrics["recon_loss"]:.4f}, '
                  f'KLD: {val_metrics["kld_loss"]:.4f}')
            
            # Early stopping检查
            current_val_loss = val_metrics["loss"]
            improvement = best_val_loss - current_val_loss
            
            # 检查是否有改善
            if improvement > args.early_stop_min_delta:
                # 有改善，重置patience计数器
                best_val_loss = current_val_loss
                best_epoch = epoch
                patience_counter = 0
                is_best = True
                print(f'✨ 验证损失改善: {improvement:.6f} (最佳: {best_val_loss:.4f} @ Epoch {best_epoch+1})')
            else:
                # 没有改善，增加patience计数器
                patience_counter += 1
                is_best = False
                print(f'⏳ 验证损失未改善 (patience: {patience_counter}/{args.early_stop_patience})')
            
            # Early stopping检查
            if patience_counter >= args.early_stop_patience:
                print(f'\n🛑 Early stopping触发！')
                print(f'   最佳验证损失: {best_val_loss:.4f} @ Epoch {best_epoch+1}')
                print(f'   当前验证损失: {current_val_loss:.4f}')
                print(f'   Patience: {patience_counter}/{args.early_stop_patience}')
                break
        else:
            # 没有验证集，只保存检查点
            is_best = False
        
        # 保存检查点
        trainer.save_checkpoint(epoch, is_best=is_best)
    
    # 关闭
    trainer.close()
    print(f'\n✅ 训练完成！')
    if val_loader is not None:
        print(f'   最佳验证损失: {best_val_loss:.4f} @ Epoch {best_epoch+1}')
    print(f'   总训练轮数: {epoch+1}/{args.epochs}')


if __name__ == '__main__':
    main()

