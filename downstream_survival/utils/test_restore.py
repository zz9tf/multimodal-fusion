#!/usr/bin/env python3
"""
恢复脚本：加载保存的模型和split，进行验证
"""

import argparse
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Subset

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset
from main import parse_channels, create_k_fold_splits, _parse_aligned_channels, seed_torch

def _load_configs_from_results_dir(results_dir: str) -> dict:
    """从results目录加载配置"""
    configs_path = os.path.join(results_dir, 'configs_test.json')
    if not os.path.exists(configs_path):
        raise FileNotFoundError(f'未找到配置文件: {configs_path}')
    
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    
    return configs

def _load_split_from_csv(results_dir: str, fold_idx: int, dataset: MultimodalDataset) -> tuple:
    """
    从 results_dir/splits_{fold_idx}.csv 加载该折的 train/val/test 划分。
    
    期望CSV列名: train,val,test；
    单元格值：保存的是case_id（如 'patient_008'），而不是索引
    通过case_id映射到当前数据集的索引
    """
    path = os.path.join(results_dir, f'splits_{fold_idx}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到分割文件: {path}')
    
    # 创建case_id到索引的映射
    if not hasattr(dataset, 'case_ids'):
        raise ValueError('数据集必须有case_ids属性')
    
    case_id_to_idx = {case_id: idx for idx, case_id in enumerate(dataset.case_ids)}
    
    # 读取CSV文件
    df = pd.read_csv(path)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 处理train列
    if 'train' in df.columns:
        train_case_ids = df['train'].dropna().tolist()
        for case_id in train_case_ids:
            if case_id in case_id_to_idx:
                train_indices.append(case_id_to_idx[case_id])
            else:
                print(f"⚠️ 警告：case_id {case_id} 不在当前数据集中，跳过")
    
    # 处理val列
    if 'val' in df.columns:
        val_case_ids = df['val'].dropna().tolist()
        for case_id in val_case_ids:
            if case_id in case_id_to_idx:
                val_indices.append(case_id_to_idx[case_id])
            else:
                print(f"⚠️ 警告：case_id {case_id} 不在当前数据集中，跳过")
    
    # 处理test列
    if 'test' in df.columns:
        test_case_ids = df['test'].dropna().tolist()
        for case_id in test_case_ids:
            if case_id in case_id_to_idx:
                test_indices.append(case_id_to_idx[case_id])
            else:
                print(f"⚠️ 警告：case_id {case_id} 不在当前数据集中，跳过")
    
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int), np.array(test_indices, dtype=int)

def _list_checkpoints(results_dir: str):
    """列出所有checkpoint文件"""
    checkpoints = []
    for filename in os.listdir(results_dir):
        if filename.startswith('s_') and filename.endswith('_checkpoint.pt'):
            try:
                fold_idx = int(filename.split('_')[1])
                checkpoints.append((fold_idx, os.path.join(results_dir, filename)))
            except Exception:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description='恢复并验证保存的模型和split')
    
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='训练结果目录（包含configs_test.json和checkpoints）')
    parser.add_argument('--data_root_dir', type=str, default=None,
                       help='数据根目录（如果不提供，将使用配置中的值）')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='CSV文件路径（如果不提供，将使用配置中的值）')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("恢复并验证保存的模型和split")
    print("="*60)
    
    # 1. 加载配置
    print(f"\n📦 加载配置...")
    configs = _load_configs_from_results_dir(args.results_dir)
    exp_cfg = configs.get('experiment_config', {})
    model_cfg = configs.get('model_config', {})
    
    print(f"✅ 配置加载成功")
    print(f"   seed: {exp_cfg.get('seed')}")
    print(f"   k: {exp_cfg.get('num_splits')}")
    print(f"   model_type: {model_cfg.get('model_type')}")
    
    # 2. 构建数据集（使用配置中的参数）
    print(f"\n📦 构建数据集...")
    data_root_dir = args.data_root_dir or exp_cfg.get('data_root_dir')
    csv_path = args.csv_path or exp_cfg.get('csv_path')
    target_channels = exp_cfg.get('target_channels', [])
    align_channels = exp_cfg.get('aligned_channels', {})
    
    if not data_root_dir:
        raise ValueError('data_root_dir is required')
    if not csv_path:
        raise ValueError('csv_path is required')
    
    # 处理路径：如果是相对路径，转换为基于项目根目录的绝对路径
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(root_dir, csv_path)
    
    if not os.path.isabs(data_root_dir):
        data_root_dir = os.path.abspath(data_root_dir)
    
    print(f"   data_root_dir: {data_root_dir}")
    print(f"   csv_path: {csv_path}")
    
    # 验证文件存在
    if not os.path.exists(csv_path):
        # 尝试查找可能的CSV文件
        possible_names = ['survival_dataset.csv', 'survival_status_labels.csv']
        csv_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else root_dir
        csv_dir = os.path.join(root_dir, 'dataset_csv') if 'dataset_csv' in str(csv_path) else csv_dir
        
        suggestions = []
        if os.path.exists(csv_dir):
            for name in possible_names:
                possible_path = os.path.join(csv_dir, name)
                if os.path.exists(possible_path):
                    suggestions.append(possible_path)
        
        error_msg = f"❌ CSV文件不存在: {csv_path}\n"
        if suggestions:
            error_msg += f"💡 找到可能的CSV文件:\n"
            for sug in suggestions:
                error_msg += f"   - {sug}\n"
            error_msg += f"💡 请使用: --csv_path {suggestions[0]}"
        else:
            error_msg += f"💡 请确保使用绝对路径或相对于项目根目录的相对路径\n"
            error_msg += f"💡 常见的CSV文件位置: {os.path.join(root_dir, 'dataset_csv')}"
        
        raise FileNotFoundError(error_msg)
    if not os.path.exists(data_root_dir):
        raise FileNotFoundError(f"❌ 数据根目录不存在: {data_root_dir}")
    print(f"   target_channels: {target_channels[:5]}..." if len(target_channels) > 5 else f"   target_channels: {target_channels}")
    
    dataset = MultimodalDataset(
        csv_path=csv_path,
        data_root_dir=data_root_dir,
        channels=target_channels,
        align_channels=align_channels,
        alignment_model_path=None,
        device=device,
        print_info=True
    )
    
    print(f"✅ 数据集构建完成: {len(dataset)} 个样本")
    if hasattr(dataset, 'case_ids') and len(dataset.case_ids) > 0:
        print(f"🔍 前5个case_id: {dataset.case_ids[:5]}")
        print(f"🔍 数据集大小: {len(dataset.case_ids)}")
    
    # 3. 列出所有checkpoints
    print(f"\n📋 查找checkpoints...")
    checkpoints = _list_checkpoints(args.results_dir)
    if not checkpoints:
        raise FileNotFoundError('未找到任何 checkpoint（形如 s_0_checkpoint.pt）。')
    
    print(f"✅ 找到 {len(checkpoints)} 个checkpoints")
    for fold_idx, ckpt_path in checkpoints:
        print(f"   Fold {fold_idx}: {ckpt_path}")
    
    # 4. 初始化训练器
    trainer = Trainer(
        configs=configs,
        log_dir=os.path.join(args.results_dir, 'restore_logs')
    )
    
    # 5. 恢复并验证每个fold
    print(f"\n🔍 开始恢复并验证每个fold...")
    
    restored_results = []
    
    for fold_idx, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f'恢复 Fold {fold_idx}')
        print(f"{'='*60}")
        
        # 5.1 加载split
        print(f"\n📊 加载split...")
        try:
            train_indices, val_indices, test_indices = _load_split_from_csv(
                args.results_dir, fold_idx, dataset
            )
            print(f"✅ Split加载成功")
            print(f"   train: {len(train_indices)} 个")
            print(f"   val: {len(val_indices)} 个")
            print(f"   test: {len(test_indices)} 个")
            
            if hasattr(dataset, 'case_ids') and len(test_indices) > 0:
                test_case_ids = [dataset.case_ids[i] for i in test_indices[:5]]
                print(f"   test集前5个case_id: {test_case_ids}")
        except Exception as e:
            print(f"❌ Split加载失败: {e}")
            continue
        
        # 5.2 创建子数据集
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices) if len(val_indices) > 0 else None
        test_ds = Subset(dataset, test_indices)
        datasets_tuple = (train_ds, val_ds, test_ds)
        
        # 5.3 加载模型并评估
        print(f"\n🔧 加载模型并评估...")
        try:
            results_dict, test_auc, val_auc, test_acc, val_acc = trainer.evaluate_fold(
                datasets=datasets_tuple,
                fold_idx=fold_idx,
                checkpoint_path=ckpt_path
            )
            
            print(f"✅ 评估完成")
            print(f"   Test AUC: {test_auc:.4f}")
            print(f"   Val AUC: {val_auc:.4f}" if val_auc is not None else "   Val AUC: None")
            print(f"   Test Acc: {test_acc:.4f}")
            print(f"   Val Acc: {val_acc:.4f}" if val_acc is not None else "   Val Acc: None")
            
            restored_results.append({
                'fold': fold_idx,
                'test_auc': float(test_auc) if test_auc is not None else None,
                'val_auc': float(val_auc) if val_auc is not None else None,
                'test_acc': float(test_acc) if test_acc is not None else None,
                'val_acc': float(val_acc) if val_acc is not None else None,
            })
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. 比较结果
    print(f"\n{'='*60}")
    print("恢复结果摘要")
    print(f"{'='*60}")
    
    if restored_results:
        restored_df = pd.DataFrame(restored_results)
        print("\n恢复后的结果:")
        print(restored_df.to_string(index=False))
        
        # 计算统计量
        test_aucs = [r['test_auc'] for r in restored_results if r['test_auc'] is not None]
        test_accs = [r['test_acc'] for r in restored_results if r['test_acc'] is not None]
        
        if test_aucs:
            print(f"\nMean Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        if test_accs:
            print(f"Mean Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        # 保存恢复结果
        restore_results_path = os.path.join(args.results_dir, 'restored_results.csv')
        restored_df.to_csv(restore_results_path, index=False)
        print(f"\n✅ 恢复结果已保存到: {restore_results_path}")
        
        # 加载原始结果进行比较
        original_results_path = os.path.join(args.results_dir, 'detailed_results_for_plotting.json')
        if os.path.exists(original_results_path):
            print(f"\n📊 加载原始结果进行比较...")
            with open(original_results_path, 'r') as f:
                original_results = json.load(f)
            
            original_fold_results = original_results.get('fold_results', {})
            original_test_aucs = original_fold_results.get('test_auc', [])
            original_test_accs = original_fold_results.get('test_acc', [])
            
            print(f"\n原始结果（训练时）:")
            print(f"  Test AUCs: {original_test_aucs}")
            print(f"  Test Accs: {original_test_accs}")
            
            print(f"\n恢复结果（重新加载后）:")
            print(f"  Test AUCs: {[r['test_auc'] for r in restored_results]}")
            print(f"  Test Accs: {[r['test_acc'] for r in restored_results]}")
            
            # 比较差异
            if len(original_test_aucs) == len(test_aucs):
                auc_diffs = [abs(o - r) for o, r in zip(original_test_aucs, test_aucs)]
                print(f"\n差异分析:")
                print(f"  Test AUC差异: {auc_diffs}")
                print(f"  最大差异: {max(auc_diffs):.6f}")
                print(f"  平均差异: {np.mean(auc_diffs):.6f}")
                
                if max(auc_diffs) < 1e-4:
                    print(f"✅ 完美匹配！所有结果都一致")
                elif max(auc_diffs) < 1e-3:
                    print(f"⚠️ 有微小差异（可能是浮点数精度问题）")
                else:
                    print(f"❌ 存在较大差异，需要检查")
    else:
        print("❌ 没有成功恢复的结果")
    
    print(f"\n✅ 恢复验证完成")

if __name__ == "__main__":
    main()

