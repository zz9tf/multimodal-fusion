#!/usr/bin/env python3
"""
SVD 对齐前后特征保存工具

用法示例：
python -m downstream_survival.utils.alignment_visualization \
  --results_dir /path/to/results \
  --fold_idx 0 \
  --save_dir /path/to/save/features

说明：
- 该脚本会根据配置构建数据集与模型（类似 robust_on_missing_modality.py），
  在测试集上运行前向推理，收集各模态在 SVD 对齐前后的特征，
  保存为 numpy 文件用于后续对比分析。
"""

from __future__ import annotations

import os
import json
import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

# 项目根目录
ROOT_DIR = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from datasets.multimodal_dataset import MultimodalDataset
from trainer import Trainer
from main import parse_channels, create_k_fold_splits


def _ensure_dir(path: str) -> None:
    """如果目录不存在则创建。"""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _load_configs_from_results_dir(results_dir: str) -> Dict[str, Any]:
    """
    从结果目录加载配置JSON（匹配第一个 configs_*.json 或 configs_*.JSON）。
    
    Returns:
        dict: 配置字典，包含 experiment_config 与 model_config。
    """
    candidates = []
    for name in os.listdir(results_dir):
        if name.startswith('configs_') and name.lower().endswith('.json'):
            candidates.append(os.path.join(results_dir, name))
    if not candidates:
        raise FileNotFoundError(f'未在目录找到配置文件: {results_dir}')

    cfg_path = sorted(candidates)[0]
    with open(cfg_path, 'r') as f:
        return json.load(f)


def _collect_features_from_testset(
    trainer: Trainer,
    test_dataset: Subset,
    fold_idx: int,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    在测试集上运行模型，收集 SVD 对齐前后的特征。
    
    Args:
        trainer: 训练器实例
        test_dataset: 测试集
        fold_idx: fold索引
        checkpoint_path: checkpoint路径
        device: 设备
        
    Returns:
        (original_features, aligned_features)
        两者字典键均为模态名，值为该模态下所有样本的特征数组 [N, D]。
    """
    # 加载模型
    model = trainer._init_model()
    model.eval()
    
    # 加载checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    
    # 处理 transfer_layer（类似 evaluate_fold）
    if hasattr(model, 'transfer_layer') and hasattr(model, 'create_transfer_layer'):
        transfer_layer_channels = {}
        for key in state.keys():
            if 'transfer_layer.' in key:
                parts = key.split('.')
                if len(parts) >= 3:
                    channel_name = parts[1]
                    weight_type = parts[2]
                    if channel_name not in transfer_layer_channels:
                        transfer_layer_channels[channel_name] = {}
                    transfer_layer_channels[channel_name][weight_type] = state[key]
        
        if hasattr(model, 'output_dim'):
            output_dim = model.output_dim
            for channel_name, weights in transfer_layer_channels.items():
                if channel_name not in model.transfer_layer:
                    if 'weight' in weights:
                        weight_tensor = weights['weight']
                        if len(weight_tensor.shape) == 2:
                            input_dim = weight_tensor.shape[1]
                            transfer_layer = model.create_transfer_layer(input_dim)
                            model.transfer_layer[channel_name] = transfer_layer
    
    # 加载权重
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    
    # 创建 DataLoader（batch_size=1，因为需要逐个处理样本）
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 收集特征
    original_features: Dict[str, List[np.ndarray]] = {}
    aligned_features: Dict[str, List[np.ndarray]] = {}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_data, label) in enumerate(test_loader):
            # 移动到设备
            if isinstance(input_data, dict):
                input_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_data.items()}
            label = label.to(device)
            
            # 前向传播
            out = model(input_data, label)
            
            # 提取 original_features_dict 和 aligned_svd_features_dict
            if isinstance(out, dict):
                if 'original_features_dict' in out:
                    for key, tensor in out['original_features_dict'].items():
                        arr = tensor.detach().float().cpu().numpy()
                        # batch_size=1，所以是 [1, D]，去掉 batch 维度得到 [D]
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            arr = arr[0]
                        original_features.setdefault(key, []).append(arr)
                
                if 'aligned_svd_features_dict' in out:
                    for key, tensor in out['aligned_svd_features_dict'].items():
                        arr = tensor.detach().float().cpu().numpy()
                        # batch_size=1，所以是 [1, D]，去掉 batch 维度得到 [D]
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            arr = arr[0]
                        aligned_features.setdefault(key, []).append(arr)
    
    # 将列表转换为 numpy 数组
    original_features_stacked = {}
    aligned_features_stacked = {}
    
    for key, arrays in original_features.items():
        # 统一处理：如果每个数组是 [D]，则堆叠为 [N, D]
        if all(arr.ndim == 1 for arr in arrays):
            original_features_stacked[key] = np.stack(arrays, axis=0)
        else:
            # 如果形状不一致，尝试拼接
            original_features_stacked[key] = np.concatenate(arrays, axis=0)
    
    for key, arrays in aligned_features.items():
        if all(arr.ndim == 1 for arr in arrays):
            aligned_features_stacked[key] = np.stack(arrays, axis=0)
        else:
            aligned_features_stacked[key] = np.concatenate(arrays, axis=0)
    
    return original_features_stacked, aligned_features_stacked


def _save_features(
    original_features: Dict[str, np.ndarray],
    aligned_features: Dict[str, np.ndarray],
    save_dir: str,
    fold_idx: int,
) -> None:
    """
    保存 SVD 对齐前后的特征到文件。
    
    Args:
        original_features: SVD 对齐前的特征字典
        aligned_features: SVD 对齐后的特征字典
        save_dir: 保存目录
        fold_idx: fold索引
    """
    _ensure_dir(save_dir)
    
    # 保存每个模态的特征
    for modality in sorted(set(list(original_features.keys()) + list(aligned_features.keys()))):
        safe_name = modality.replace('/', '_').replace('=', '_')
        
        if modality in original_features:
            original_path = os.path.join(save_dir, f'fold_{fold_idx}_{safe_name}_original.npy')
            np.save(original_path, original_features[modality])
            print(f'  ✅ 保存原始特征: {original_path} (shape: {original_features[modality].shape})')
        
        if modality in aligned_features:
            aligned_path = os.path.join(save_dir, f'fold_{fold_idx}_{safe_name}_aligned.npy')
            np.save(aligned_path, aligned_features[modality])
            print(f'  ✅ 保存对齐特征: {aligned_path} (shape: {aligned_features[modality].shape})')
    
    # 保存元数据
    metadata = {
        'fold_idx': fold_idx,
        'modalities': sorted(set(list(original_features.keys()) + list(aligned_features.keys()))),
        'original_features_shapes': {k: list(v.shape) for k, v in original_features.items()},
        'aligned_features_shapes': {k: list(v.shape) for k, v in aligned_features.items()},
    }
    metadata_path = os.path.join(save_dir, f'fold_{fold_idx}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  ✅ 保存元数据: {metadata_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='SVD 对齐前后特征保存工具')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='训练结果目录（包含 s_?_checkpoint.pt 与 configs_*.json）')
    parser.add_argument('--fold_idx', type=int, default=0, 
                       help='要处理的 fold 索引（默认 0）')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='特征保存目录，默认使用 results_dir/svd_features')
    parser.add_argument('--data_root_dir', type=str, default=None,
                       help='数据根目录，优先使用此参数，否则回退到configs')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='CSV路径，优先使用此参数，否则回退到configs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载配置
    configs = _load_configs_from_results_dir(args.results_dir)
    exp_cfg = configs.get('experiment_config', {})
    model_cfg = configs.get('model_config', {})

    # 2) 构建数据集
    base_target_channels = exp_cfg.get('target_channels') or model_cfg.get('channels_used_in_model')
    if not base_target_channels:
        base_target_channels = parse_channels([])
    
    dataset = MultimodalDataset(
        csv_path=args.csv_path or exp_cfg.get('csv_path'),
        data_root_dir=args.data_root_dir or exp_cfg.get('data_root_dir'),
        channels=base_target_channels,
        align_channels=exp_cfg.get('aligned_channels', None),
        alignment_model_path=exp_cfg.get('alignment_model_path', None),
        device=device,
    )
    
    print(f"📊 数据集构建完成: {len(dataset)} 个样本")

    # 3) 生成 K 折划分
    seed = exp_cfg.get('seed', 5678)
    k = exp_cfg.get('num_splits', 10)
    splits = create_k_fold_splits(dataset, k=k, seed=seed, fixed_test_split=None)
    
    if args.fold_idx >= len(splits):
        raise ValueError(f'Fold {args.fold_idx} 超出划分范围（共 {len(splits)} 个 fold）')
    
    split = splits[args.fold_idx]
    test_ds = Subset(dataset, split['test'])
    print(f"📊 Fold {args.fold_idx} 测试集: {len(test_ds)} 个样本")

    # 4) 构建训练器并加载模型
    trainer = Trainer(configs=configs, log_dir=os.path.join(args.results_dir, 'training_logs'))
    
    # 5) 获取 checkpoint
    checkpoint_path = os.path.join(args.results_dir, f's_{args.fold_idx}_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint 不存在: {checkpoint_path}')
    
    print(f"📦 加载 checkpoint: {checkpoint_path}")

    # 6) 收集特征
    print(f"🔄 在测试集上运行模型，收集特征...")
    original_features, aligned_features = _collect_features_from_testset(
        trainer=trainer,
        test_dataset=test_ds,
        fold_idx=args.fold_idx,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 7) 保存特征
    save_dir = args.save_dir or os.path.join(args.results_dir, 'svd_features')
    _ensure_dir(save_dir)
    
    print(f"💾 保存特征到: {save_dir}")
    _save_features(original_features, aligned_features, save_dir, args.fold_idx)

    print(f"✅ 完成！特征已保存到: {save_dir}")


if __name__ == '__main__':
    main()
