#!/usr/bin/env python3
"""
多模态对齐训练脚本
使用UNI模型进行多模态特征对齐
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
import json
from pathlib import Path

# 导入模型和训练器
from alignment_model import MultiModalAlignmentModel
from trainer import MultiModalAlignmentTrainer
from alignment_dataset import (
    create_tma_aligned_with_neg_dataset,
    build_collate_fn,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tma_dir = "/home/zheng/zheng/mini2/hancock_data/TMA/TMA_Core_encodings"
modality_names = ["CD3", "CD8", "CD56", "CD68", "CD163", "HE", "MHC1", "PDL1"]
feature_dim = 1024

def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description="多模态对齐训练")
    parser.add_argument("--align_mode", type=str, default="intersection",
                       help="对齐模式")
    parser.add_argument("--pattern", type=str, default="tma_uni_tile_1024_{marker}.npz",
                       help="文件名匹配模式，使用 {marker} 作为占位符，例如: 'tma_uni_tile_1024_{marker}.npz'")
    parser.add_argument("--mismatch_ratio", type=float, default=1.0,
                       help="负样本池大小系数")
    parser.add_argument("--seed", type=int, default=42,
                       help="负样本池随机种子")
    parser.add_argument("--lambda1", type=float, default=1.0,
                       help="labmda1")
    parser.add_argument("--lambda2", type=float, default=0.1,
                       help="lambda2")
    parser.add_argument("--tau1", type=float, default=0.1,
                       help="tau1")
    parser.add_argument("--tau2", type=float, default=0.05,
                       help="tau2")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--max_steps", type=int, default=100000,
                       help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="批次大小")
    parser.add_argument("--save_path", type=str, default="best_multimodal_alignment_model.pth",
                       help="模型保存路径")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔（步数）")
    parser.add_argument("--val_interval", type=int, default=500, help="验证间隔（步数）")
    parser.add_argument("--num_layers", type=int, default=1, 
                       help="对齐层的层数，默认为1（单层线性变换）")
    parser.add_argument("--val_max_batches", type=int, default=None, help="验证最多批次数")
    parser.add_argument("--loss_type", type=str, default="volume", help="损失类型 volume, rank1")
    parser.add_argument("--loss2_chunk_size", type=int, default=None, help="loss2 分块大小（行块尺寸）")
    parser.add_argument("--verbose_timing", action="store_true", help="启用详细性能分析（默认关闭）")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping耐心值（验证loss不改善的步数，0表示禁用）")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Early stopping最小改善阈值")
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建多模态对齐模型
    logger.info(f"🏗️ 创建多模态对齐模型...")
    
    model = MultiModalAlignmentModel(
        modality_names=modality_names,
        feature_dim=feature_dim,
        num_layers=args.num_layers
    )
    
    trainer = MultiModalAlignmentTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        val_max_batches=args.val_max_batches,
        loss2_chunk_size=args.loss2_chunk_size,
        verbose_timing=args.verbose_timing,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    # 创建带负样本池的数据集
    logger.info("📊 加载 TMA 数据集（带全局负样本池）...")
    logger.info(f"   - 文件匹配模式: {args.pattern}")
    base_ds = create_tma_aligned_with_neg_dataset(
        base_dir=tma_dir,
        modality_names=modality_names,
        align_mode=args.align_mode,
        filename_template=args.pattern,
        mismatch_ratio=args.mismatch_ratio,
        seed=args.seed,
    )

    # 使用 tuple 进行切分（示例：这里随机划分 8:1:1，你也可以预先生成 ids 并传入）
    # 现在 normalized_keys 是 5 维的 (block, x, y, patient, patch_id)
    all_tuples = base_ds.normalized_keys
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(all_tuples))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = [all_tuples[i] for i in idx[:n_train]]
    val_ids = [all_tuples[i] for i in idx[n_train:n_train+n_val]]
    test_ids = [all_tuples[i] for i in idx[n_train+n_val:]]

    splits = base_ds.split_by_ids_with_neg(
        {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
        },
        id_type='tuple',  # 使用完整键 (block, x, y, patient, patch_id) 进行切分
        mismatch_ratio=args.mismatch_ratio,
        seed=args.seed,
    )

    train_ds = splits['train']
    val_ds = splits['val']
    test_ds = splits['test']

    # 创建 collate_fn（负样本数 = ceil(batch_size * mismatch_ratio)）
    train_collate = build_collate_fn(train_ds, ratio=args.mismatch_ratio)
    val_collate = build_collate_fn(val_ds, ratio=args.mismatch_ratio)
    test_collate = build_collate_fn(test_ds, ratio=args.mismatch_ratio)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate
    )
    
    logger.info(f"训练样本数: {len(train_loader.dataset)} | 验证: {len(val_loader.dataset)} | 测试: {len(test_loader.dataset)}")
    
    # 训练模型（Step 模式）
    logger.info("🚀 开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.max_steps,
        save_path=args.save_path,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
    )
    
    logger.info(f"✅ 训练完成！记录了 {len(history['train_losses'])} 步的训练数据")
    
    # 💾 保存训练历史
    history_save_path = Path(args.save_path).with_suffix('.history.json')
    logger.info(f"💾 保存训练历史到: {history_save_path}")
    
    # 转换 tensor 为 list 以便 JSON 序列化
    history_to_save = {
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'val_steps': history['val_steps'],
        'train_svd_values': [svd.cpu().tolist() for svd in history['train_svd_values']],
        'val_svd_values': [svd.cpu().tolist() for svd in history['val_svd_values']],
        'config': {
            'max_steps': args.max_steps,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'tau1': args.tau1,
            'tau2': args.tau2,
            'mismatch_ratio': args.mismatch_ratio,
            'pattern': args.pattern,
            'align_mode': args.align_mode,
            'seed': args.seed,
            'num_layers': args.num_layers,
            'loss_type': args.loss_type,
        }
    }
    
    with open(history_save_path, 'w') as f:
        json.dump(history_to_save, f, indent=2)
    
    logger.info(f"✅ 训练历史已保存: {history_save_path}")


if __name__ == "__main__":
    main()
