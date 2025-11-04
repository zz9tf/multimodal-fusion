#!/usr/bin/env python3
"""
缺模态鲁棒性评测脚本（仅测试集评测）

功能概述：
- 给定结果目录（包含10个fold的checkpoint与配置）、数据集、缺失模态描述与分割方式；
- 在推理阶段对指定模态做掩蔽（统一置零），并在各fold的测试集上评测模型表现；
- 仅输出测试集指标（如 Test AUC/ACC），保存CSV与JSON。

依赖：
- 复用 `datasets.multimodal_dataset.MultimodalDataset` 加载数据；
- 复用 `Trainer` 完成模型构建与评测（通过多种可能的接口名自适配）。
"""

import argparse
import os
import json
import sys
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np
import torch
from torch.utils.data import Subset

ROOT_DIR = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from datasets.multimodal_dataset import MultimodalDataset  # noqa: E402
from trainer import Trainer  # noqa: E402
from main import parse_channels, create_k_fold_splits  # noqa: E402


# 使用 main.parse_channels 作为单一真源

class MaskingDataset(Subset):
    """
    子集数据集包装器：在 __getitem__ 时对指定通道进行Mask。

    Mask策略：
    - zero: 将目标通道张量置零（保持形状不变）
    - drop: 等同于 zero（按用户要求，仅置零，不删除键）
    """

    def __init__(
        self,
        dataset: Subset,
        channels_to_mask: List[str],
    ) -> None:
        super().__init__(dataset.dataset, dataset.indices)
        self.base_subset = dataset
        self.channels_to_mask = set(channels_to_mask or [])
        # 固定置零策略
        self.mask_strategy = 'zero'

    def __getitem__(self, idx: int):
        base_item = self.base_subset[idx]
        # 训练管线中 dataset.__getitem__ 返回 (dict, label)
        if isinstance(base_item, tuple) and len(base_item) == 2:
            feature_dict, label = base_item
        else:
            # 兼容仅返回dict的情况
            feature_dict, label = base_item, None

        masked = dict(feature_dict)

        # 同时考虑对齐后的键：aligned_<channel>
        aligned_keys = []
        for ch in list(masked.keys()):
            if ch.startswith('aligned_'):
                aligned_keys.append(ch)

        keys_to_process = set()
        for ch in self.channels_to_mask:
            keys_to_process.add(ch)
            keys_to_process.add(f'aligned_{ch}')

        for key in list(masked.keys()):
            if key in keys_to_process:
                tensor = masked.get(key, None)
                if not isinstance(tensor, torch.Tensor):
                    # 不可用则跳过
                    continue
                # 统一按置零处理
                masked[key] = torch.zeros_like(tensor)

        if label is None:
            return masked
        return masked, label


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
        # 兼容用户示例里的命名
        alt = os.path.join(results_dir, 'configs_all_modality_clam_detach.json')
        if os.path.exists(alt):
            candidates.append(alt)
    if not candidates:
        raise FileNotFoundError(f'未在目录找到配置文件: {results_dir}')

    cfg_path = sorted(candidates)[0]
    with open(cfg_path, 'r') as f:
        return json.load(f)


def _list_checkpoints(results_dir: str) -> List[Tuple[int, str]]:
    """
    枚举各fold的checkpoint。

    期望命名：s_0_checkpoint.pt ... s_9_checkpoint.pt
    Returns: List[(fold_idx, ckpt_path)]
    """
    items: List[Tuple[int, str]] = []
    for name in os.listdir(results_dir):
        if name.startswith('s_') and name.endswith('_checkpoint.pt'):
            try:
                fold = int(name.split('_')[1])
                items.append((fold, os.path.join(results_dir, name)))
            except Exception:
                continue
    items.sort(key=lambda x: x[0])
    return items

def _load_split_from_csv(results_dir: str, fold_idx: int, dataset: MultimodalDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 results_dir/splits_{fold_idx}.csv 加载该折的 train/val/test 划分。
    
    期望CSV列名: train,val,test；
    单元格值：保存的是case_id（如 'patient_008'），而不是索引
    通过case_id映射到当前数据集的索引
    """
    import csv
    
    path = os.path.join(results_dir, f'splits_{fold_idx}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到分割文件: {path}')
    
    # 创建case_id到索引的映射
    if not hasattr(dataset, 'case_ids'):
        raise ValueError('数据集必须有case_ids属性')
    
    case_id_to_idx = {case_id: idx for idx, case_id in enumerate(dataset.case_ids)}
    
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, case_id in (('train', row.get('train')), ('val', row.get('val')), ('test', row.get('test'))):
                if case_id is None or case_id == '' or case_id.lower() == 'nan':
                    continue
                
                # 通过case_id查找对应的索引
                if case_id in case_id_to_idx:
                    idx = case_id_to_idx[case_id]
                    if col == 'train':
                        train_indices.append(idx)
                    elif col == 'val':
                        val_indices.append(idx)
                    else:
                        test_indices.append(idx)
                else:
                    # case_id不在当前数据集中（可能是旧数据集的划分）
                    print(f"⚠️ 警告：case_id {case_id} 不在当前数据集中，跳过")
    
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int), np.array(test_indices, dtype=int)


def _build_dataset(
    csv_path: str,
    data_root_dir: str,
    channels: List[str],
    align_channels: Optional[Dict[str, str]],
    device: torch.device,
) -> MultimodalDataset:
    """
    构建多模态数据集。
    """
    return MultimodalDataset(
        csv_path=csv_path,
        data_root_dir=data_root_dir,
        channels=channels,
        align_channels=align_channels,
        alignment_model_path=None,  # 不使用预训练对齐
        device=device,
        print_info=True,
    )


def _call_trainer_eval(
    trainer: Trainer,
    datasets: Tuple[Subset, Optional[Subset], Subset],
    fold_idx: int,
    checkpoint_path: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    调用训练器的评测接口（自适配不同可能的方法名）。

    Returns:
        (test_auc, test_acc)
    """
    # 优先找显式的评测接口
    if hasattr(trainer, 'evaluate_with_checkpoint'):
        res = trainer.evaluate_with_checkpoint(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path)
    elif hasattr(trainer, 'evaluate_fold'):
        res = trainer.evaluate_fold(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path)
    elif hasattr(trainer, 'test_fold'):
        res = trainer.test_fold(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path)
    elif hasattr(trainer, 'evaluate'):
        res = trainer.evaluate(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path)
    else:
        raise RuntimeError('Trainer 未提供兼容的评测接口，请在 Trainer 中实现 test/evaluate 接口。')

    # 结果兼容：常见返回 (results_dict, test_auc, val_auc, test_acc, val_acc)
    test_auc = None
    test_acc = None
    if isinstance(res, tuple):
        # 解析 test_acc（通常位于倒数第二个）
        if len(res) >= 2:
            try:
                test_acc = float(res[-2])
            except Exception:
                test_acc = None
        # 解析 test_auc（通常位于索引1）
        if len(res) >= 2:
            try:
                test_auc = float(res[1])
            except Exception:
                test_auc = None
    elif isinstance(res, dict):
        test_auc = float(res.get('test_auc')) if res.get('test_auc') is not None else None
        test_acc = float(res.get('test_acc')) if res.get('test_acc') is not None else None
    return test_auc, test_acc


def run(args: argparse.Namespace) -> None:
    """
    主流程：加载配置与checkpoint -> 构造数据与分割 -> 掩蔽指定模态 -> 评测 -> 保存报告。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 结果目录配置与checkpoint
    configs = _load_configs_from_results_dir(args.results_dir)
    exp_cfg = configs.get('experiment_config', {})
    model_cfg = configs.get('model_config', {})

    # 基础通道按主程序规则解析（支持简写）
    base_target_channels = exp_cfg.get('target_channels') or model_cfg.get('channels_used_in_model')
    if not base_target_channels:
        # 若配置里没有，使用命令行指定
        base_target_channels = parse_channels(args.target_channels or [])

    # 解析要mask的通道（支持简写与完整路径混合）
    channels_to_mask = parse_channels(args.missing_modalities or [])

    # 2) 数据集
    dataset = _build_dataset(
        csv_path=args.csv_path or exp_cfg.get('csv_path'),
        data_root_dir=args.data_root_dir or exp_cfg.get('data_root_dir'),
        channels=base_target_channels,
        align_channels=exp_cfg.get('aligned_channels'),
        device=device,
    )
    
    print(f"📊 数据集构建完成: {len(dataset)} 个样本")
    print(f"📁 data_root_dir: {args.data_root_dir or exp_cfg.get('data_root_dir')}")
    print(f"📋 channels: {base_target_channels[:5]}..." if len(base_target_channels) > 5 else f"📋 channels: {base_target_channels}")
    
    # 验证数据集顺序（打印前5个case_id）
    if hasattr(dataset, 'case_ids') and len(dataset.case_ids) > 0:
        print(f"🔍 前5个case_id: {dataset.case_ids[:5]}")
        print(f"🔍 后5个case_id: {dataset.case_ids[-5:]}")
        print(f"🔍 数据集大小: {len(dataset.case_ids)}")

    # 3) 重新生成K折划分（与训练时一致）
    seed = exp_cfg.get('seed', 5678)
    k = exp_cfg.get('num_splits', 10)
    print(f"🔧 使用配置中的 seed={seed}, k={k} 重新生成划分（与训练时一致）")
    
    # 验证数据集是否与训练时一致
    print(f"📊 数据集信息验证:")
    print(f"   数据集大小: {len(dataset)}")
    if hasattr(dataset, 'case_ids'):
        print(f"   case_ids数量: {len(dataset.case_ids)}")
        print(f"   前10个case_id: {dataset.case_ids[:10]}")
    
    splits = create_k_fold_splits(dataset, k=k, seed=seed, fixed_test_split=None)
    print(f"✅ 生成了 {len(splits)} 个 folds")
    
    # 验证Fold 0的划分（与训练时对比）
    if len(splits) > 0:
        fold0_split = splits[0]
        print(f"\n📊 Fold 0 划分验证:")
        print(f"   train: {len(fold0_split['train'])} 个")
        print(f"   val: {len(fold0_split['val'])} 个")
        print(f"   test: {len(fold0_split['test'])} 个")
        if hasattr(dataset, 'case_ids'):
            fold0_test_case_ids = [dataset.case_ids[i] for i in fold0_split['test'][:10]]
            print(f"   Fold 0 test集前10个case_id: {fold0_test_case_ids}")
    
    # 4) 获取checkpoint列表
    checkpoints = _list_checkpoints(args.results_dir)
    if not checkpoints:
        raise FileNotFoundError('未找到任何 checkpoint（形如 s_0_checkpoint.pt）。')

    # 5) 训练器
    trainer = Trainer(configs=configs, log_dir=os.path.join(args.results_dir, 'training_logs'))

    # 6) 遍历fold并评测（使用重新生成的划分）。仅在测试集上做mask与评测。
    per_fold_metrics = []

    for fold_idx, ckpt_path in checkpoints:
        if fold_idx >= len(splits):
            print(f"⚠️ Fold {fold_idx} 超出划分范围，跳过")
            continue
        
        split = splits[fold_idx]
        train_idx = split['train']
        val_idx = split['val']
        test_idx = split['test']
        
        print(f"📊 Fold {fold_idx} 划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        # 打印测试集的前5个case_id用于验证
        if hasattr(dataset, 'case_ids') and len(test_idx) > 0:
            test_case_ids = [dataset.case_ids[i] for i in test_idx[:5]]
            print(f"🔍 Fold {fold_idx} test集前5个case_id: {test_case_ids}")

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx) if len(val_idx) > 0 else None
        test_ds = Subset(dataset, test_idx)

        # 仅对测试集做mask；训练/验证保持原样
        # 如果没有指定要mask的通道，直接使用原始测试集（避免不必要的包装）
        if channels_to_mask:
            masked_test = MaskingDataset(test_ds, channels_to_mask=channels_to_mask)
        else:
            masked_test = test_ds  # 不做mask，直接使用原始测试集

        datasets_tuple = (train_ds, val_ds, masked_test)

        print(f"🔧 评测 Fold {fold_idx} checkpoint: {ckpt_path}")
        
        # 验证checkpoint是否存在且可读
        if not os.path.exists(ckpt_path):
            print(f"❌ 错误：checkpoint文件不存在: {ckpt_path}")
            continue
        
        # 检查checkpoint文件大小和时间戳
        ckpt_stat = os.stat(ckpt_path)
        print(f"📦 checkpoint文件信息: 大小={ckpt_stat.st_size} bytes, 修改时间={ckpt_stat.st_mtime}")
        
        test_auc, test_acc = _call_trainer_eval(
            trainer=trainer,
            datasets=datasets_tuple,
            fold_idx=fold_idx,
            checkpoint_path=ckpt_path,
        )

        per_fold_metrics.append({
            'fold': fold_idx,
            'test_auc': float(test_auc) if test_auc is not None else None,
            'test_acc': float(test_acc) if test_acc is not None else None,
        })

        print(f"Fold {fold_idx}: test_auc={test_auc} test_acc={test_acc}")

    # 7) 汇总并保存
    def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in values if v is not None]
        return float(np.mean(xs)) if xs else None
    def _safe_std(values: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in values if v is not None]
        return float(np.std(xs)) if xs else None

    summary = {
        'missing_modalities': args.missing_modalities,
        'per_fold': per_fold_metrics,
        'mean_test_auc': _safe_mean([m['test_auc'] for m in per_fold_metrics]),
        'std_test_auc': _safe_std([m['test_auc'] for m in per_fold_metrics]),
        'mean_test_acc': _safe_mean([m['test_acc'] for m in per_fold_metrics]),
        'std_test_acc': _safe_std([m['test_acc'] for m in per_fold_metrics]),
    }

    out_json = os.path.join(args.results_dir, 'robust_missing_eval.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # 可选CSV（按fold）
    try:
        import pandas as pd  # 仅用于保存CSV
        pd.DataFrame(per_fold_metrics).to_csv(os.path.join(args.results_dir, 'robust_missing_eval_per_fold.csv'), index=False)
    except Exception:
        pass

    print('保存评测：', out_json)


def build_argparser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析。
    """
    p = argparse.ArgumentParser(description='缺模态鲁棒性评测')
    p.add_argument('--results_dir', type=str, required=True, help='训练结果目录（包含 s_?_checkpoint.pt 与 configs_*.json）')
    p.add_argument('--data_root_dir', type=str, default=None, help='数据根目录，优先使用此参数，否则回退到configs')
    p.add_argument('--csv_path', type=str, default=None, help='CSV路径，优先使用此参数，否则回退到configs')
    p.add_argument('--target_channels', type=str, nargs='*', default=None, help='目标通道（若configs缺失时使用，支持简写）')
    p.add_argument('--missing_modalities', type=str, nargs='*', default=None, help='需Mask的模态（支持简写：如 wsi, cd3, clinical 等）；不传则不做mask')
    return p


if __name__ == '__main__':
    parser = build_argparser()
    args_ = parser.parse_args()
    run(args_)


