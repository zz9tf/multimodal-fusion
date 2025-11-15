#!/usr/bin/env python3
"""
缺模态鲁棒性评测脚本（仅测试集评测）

功能概述：
- 给定结果目录（包含10个fold的checkpoint与配置）、数据集；
- 在推理阶段通过 drop_prob 控制模态丢弃，并在各fold的测试集上评测模型表现；
- 仅输出测试集指标（如 Test AUC/ACC），保存CSV与JSON。
"""

import argparse
import os
import json
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Subset

ROOT_DIR = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from datasets.multimodal_dataset import MultimodalDataset
from trainer import Trainer
from main import parse_channels, create_k_fold_splits


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

def _call_trainer_eval(
    trainer: Trainer,
    datasets: Tuple[Subset, Optional[Subset], Subset],
    fold_idx: int,
    checkpoint_path: str,
    drop_prob: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    调用训练器的评测接口（自适配不同可能的方法名）。

    Args:
        trainer: 训练器实例
        datasets: 数据集元组
        fold_idx: fold索引
        checkpoint_path: checkpoint路径
        drop_prob: 模态丢弃概率（用于forward时传入）

    Returns:
        (test_auc, test_acc)
    """
    # 优先找显式的评测接口
    if hasattr(trainer, 'evaluate_with_checkpoint'):
        res = trainer.evaluate_with_checkpoint(
            datasets=datasets, 
            fold_idx=fold_idx, 
            checkpoint_path=checkpoint_path,
            drop_prob=drop_prob
        )
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
        base_target_channels = parse_channels(args.target_channels or [])

    # 2) 数据集
    dataset = MultimodalDataset(
        csv_path=args.csv_path or exp_cfg.get('csv_path'),
        data_root_dir=args.data_root_dir or exp_cfg.get('data_root_dir'),
        channels=base_target_channels,
        align_channels=exp_cfg.get('aligned_channels', None),
        alignment_model_path=exp_cfg.get('alignment_model_path', None),
        device=device,
    )
    
    print(f"📊 数据集构建完成: {len(dataset)} 个样本")
    print(f"📁 data_root_dir: {args.data_root_dir or exp_cfg.get('data_root_dir')}")
    print(f"📋 channels: {base_target_channels[:5]}..." if len(base_target_channels) > 5 else f"📋 channels: {base_target_channels}")

    # 3) 重新生成K折划分（与训练时一致）
    seed = exp_cfg.get('seed', 5678)
    k = exp_cfg.get('num_splits', 10)
    splits = create_k_fold_splits(dataset, k=k, seed=seed, fixed_test_split=None)
    
    # 4) 获取checkpoint列表
    checkpoints = _list_checkpoints(args.results_dir)
    if not checkpoints:
        raise FileNotFoundError('未找到任何 checkpoint（形如 s_0_checkpoint.pt）。')

    # 5) 训练器
    trainer = Trainer(configs=configs, log_dir=os.path.join(args.results_dir, 'training_logs'))

    # 6) 遍历fold并评测
    per_fold_metrics = []
    drop_prob = args.drop_prob

    for fold_idx, ckpt_path in checkpoints:
        if fold_idx >= len(splits):
            print(f"⚠️ Fold {fold_idx} 超出划分范围，跳过")
            continue
        
        split = splits[fold_idx]
        train_ds = Subset(dataset, split['train'])
        val_ds = Subset(dataset, split['val']) if len(split['val']) > 0 else None
        test_ds = Subset(dataset, split['test'])
        
        print(f"📊 Fold {fold_idx} 划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
        
        if drop_prob is not None:
            print(f"🔍 使用 drop_prob={drop_prob} 在 forward 时控制模态丢弃")
        
        if not os.path.exists(ckpt_path):
            print(f"❌ 错误：checkpoint文件不存在: {ckpt_path}")
            continue
        
        test_auc, test_acc = _call_trainer_eval(
            trainer=trainer,
            datasets=(train_ds, val_ds, test_ds),
            fold_idx=fold_idx,
            checkpoint_path=ckpt_path,
            drop_prob=drop_prob,
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
        'drop_prob': drop_prob,
        'per_fold': per_fold_metrics,
        'mean_test_auc': _safe_mean([m['test_auc'] for m in per_fold_metrics]),
        'std_test_auc': _safe_std([m['test_auc'] for m in per_fold_metrics]),
        'mean_test_acc': _safe_mean([m['test_acc'] for m in per_fold_metrics]),
        'std_test_acc': _safe_std([m['test_acc'] for m in per_fold_metrics]),
    }

    out_json = os.path.join(args.results_dir, f'robust_missing_drop_prob_{drop_prob}.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'保存评测：{out_json}')


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析。"""
    p = argparse.ArgumentParser(description='缺模态鲁棒性评测')
    p.add_argument('--results_dir', type=str, required=True, help='训练结果目录（包含 s_?_checkpoint.pt 与 configs_*.json）')
    p.add_argument('--data_root_dir', type=str, default=None, help='数据根目录，优先使用此参数，否则回退到configs')
    p.add_argument('--csv_path', type=str, default=None, help='CSV路径，优先使用此参数，否则回退到configs')
    p.add_argument('--target_channels', type=str, nargs='*', default=None, help='目标通道（若configs缺失时使用，支持简写）')
    p.add_argument('--drop_prob', type=float, default=None, help='模态丢弃概率（0.0-1.0），在 forward 时传入模型')
    return p


if __name__ == '__main__':
    parser = build_argparser()
    args_ = parser.parse_args()
    run(args_)


