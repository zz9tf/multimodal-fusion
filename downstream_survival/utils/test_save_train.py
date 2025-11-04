#!/usr/bin/env python3
"""
模拟训练脚本：只训练几步，保存模型和split用于测试恢复功能
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Subset

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset
from main import parse_channels, create_k_fold_splits, _parse_aligned_channels, seed_torch

def main():
    parser = argparse.ArgumentParser(description='模拟训练：只训练几步，保存模型和split')
    
    # 数据相关参数
    parser.add_argument('--data_root_dir', type=str, required=True, help='数据根目录')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/survival_dataset.csv', help='CSV文件路径（默认: dataset_csv/survival_dataset.csv）')
    parser.add_argument('--results_dir', type=str, default='./test_results', help='结果保存目录')
    
    # 模型相关参数
    parser.add_argument('--target_channels', type=str, nargs='+', 
                       default=['CD3', 'CD8'], help='目标通道')
    parser.add_argument('--model_type', type=str, default='clam_mlp_detach', 
                       choices=['clam_mlp_detach', 'clam_mlp', 'clam'], help='模型类型')
    parser.add_argument('--input_dim', type=int, default=1024, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=128, help='输出维度')
    parser.add_argument('--n_classes', type=int, default=2, help='类别数')
    
    # 训练相关参数
    parser.add_argument('--k', type=int, default=3, help='fold数量（用于测试，只训练3个fold）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_epochs', type=int, default=5, help='最大训练轮数（用于测试，只训练5个epoch）')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--reg', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='启用早停')
    
    # 对齐相关参数
    parser.add_argument('--aligned_channels', type=str, nargs='*', default=None, 
                       help='对齐目标，格式: channel_to_align1=align_channel_name1 ...')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建结果目录
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.results_dir = os.path.join(args.results_dir, f"test_train_{timestamp}_s{args.seed}")
    os.makedirs(args.results_dir)
    
    print("="*60)
    print("模拟训练：只训练几步，保存模型和split")
    print("="*60)
    
    # 1. 解析channels
    try:
        parsed_channels = parse_channels(args.target_channels)
        print(f"✅ 成功解析通道: {len(parsed_channels)} 个")
        print(f"📋 原始通道: {args.target_channels}")
        print(f"🔗 解析后通道: {parsed_channels[:5]}..." if len(parsed_channels) > 5 else f"🔗 解析后通道: {parsed_channels}")
    except ValueError as e:
        print(f"❌ 通道解析错误: {e}")
        return
    
    # 2. 构建align_channels映射
    align_channels = _parse_aligned_channels(args.aligned_channels)
    
    # 3. 创建数据集
    print(f"\n📦 加载数据集...")
    
    # 处理路径：如果是相对路径，转换为基于项目根目录的绝对路径
    if args.csv_path and not os.path.isabs(args.csv_path):
        csv_path = os.path.join(root_dir, args.csv_path)
    else:
        csv_path = args.csv_path
    
    if args.data_root_dir and not os.path.isabs(args.data_root_dir):
        # data_root_dir 通常是绝对路径，但如果是相对路径也处理
        data_root_dir = os.path.abspath(args.data_root_dir)
    else:
        data_root_dir = args.data_root_dir
    
    print(f"   data_root_dir: {data_root_dir}")
    print(f"   csv_path: {csv_path}")
    
    # 验证文件存在
    if not os.path.exists(csv_path):
        # 尝试查找可能的CSV文件
        possible_names = ['survival_dataset.csv', 'survival_status_labels.csv']
        csv_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else root_dir
        csv_dir = os.path.join(root_dir, 'dataset_csv') if 'dataset_csv' in csv_path else csv_dir
        
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
    
    dataset = MultimodalDataset(
        csv_path=csv_path,
        data_root_dir=data_root_dir,
        channels=parsed_channels,
        align_channels=align_channels,
        alignment_model_path=None,
        device=device,
        print_info=True
    )
    
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    if hasattr(dataset, 'case_ids') and len(dataset.case_ids) > 0:
        print(f"🔍 前5个case_id: {dataset.case_ids[:5]}")
        print(f"🔍 数据集大小: {len(dataset.case_ids)}")
    
    # 4. 创建k-fold分割
    print(f"\n📊 创建 {args.k}-fold 交叉验证分割...")
    splits = create_k_fold_splits(dataset, k=args.k, seed=args.seed, fixed_test_split=None)
    print(f"✅ 创建了 {len(splits)} 个 folds")
    
    if len(splits) > 0:
        fold0_split = splits[0]
        print(f"\n📊 Fold 0 划分:")
        print(f"   train: {len(fold0_split['train'])} 个")
        print(f"   val: {len(fold0_split['val'])} 个")
        print(f"   test: {len(fold0_split['test'])} 个")
        if hasattr(dataset, 'case_ids'):
            fold0_test_case_ids = [dataset.case_ids[i] for i in fold0_split['test'][:10]]
            print(f"   Fold 0 test集前10个case_id: {fold0_test_case_ids}")
    
    # 5. 构建配置
    configs = {
        'experiment_config': {
            'data_root_dir': data_root_dir,
            'results_dir': args.results_dir,
            'csv_path': csv_path,
            'alignment_model_path': None,
            'target_channels': parsed_channels,
            'aligned_channels': align_channels,
            'exp_code': 'test_train',
            'seed': args.seed,
            'num_splits': args.k,
            'split_mode': 'random',
            'dataset_split_path': None,
            'max_epochs': args.max_epochs,
            'lr': args.lr,
            'reg': args.reg,
            'opt': args.opt,
            'early_stopping': args.early_stopping,
            'batch_size': args.batch_size,
            'scheduler_config': {'type': None}
        },
        'model_config': {
            'model_type': args.model_type,
            'input_dim': args.input_dim,
            'dropout': 0.25,
            'n_classes': args.n_classes,
            'base_loss_fn': 'ce',
            'channels_used_in_model': parsed_channels,
            'gate': True,
            'base_weight': 0.7,
            'inst_loss_fn': None,
            'model_size': 'small',
            'subtyping': False,
            'inst_number': 8,
            'return_features': False,
            'attention_only': False,
            'output_dim': args.output_dim,
        }
    }
    
    # 保存配置
    configs_path = os.path.join(args.results_dir, 'configs_test.json')
    with open(configs_path, 'w') as f:
        json.dump(configs, f, indent=2, default=str)
    print(f"\n✅ 配置已保存到: {configs_path}")
    
    # 6. 初始化训练器
    trainer = Trainer(
        configs=configs,
        log_dir=os.path.join(args.results_dir, 'training_logs')
    )
    
    # 7. 训练每个fold（只训练几个fold，用于测试）
    print(f"\n🚀 开始训练（只训练 {args.k} 个 folds，每个fold {args.max_epochs} 个epoch）...")
    
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    
    for fold_idx in range(args.k):
        print(f"\n{'='*60}")
        print(f'训练 Fold {fold_idx+1}/{args.k}')
        print(f"{'='*60}")
        
        seed_torch(args.seed)
        
        # 获取当前fold的分割
        split = splits[fold_idx]
        train_idx = split['train']
        val_idx = split['val']
        test_idx = split['test']
        
        print(f'Train samples: {len(train_idx)}')
        print(f'Val samples: {len(val_idx)}')
        print(f'Test samples: {len(test_idx)}')
        
        # 创建子数据集
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # 使用训练器进行训练
        print(f"\n📝 开始训练 Fold {fold_idx}...")
        results, test_auc, val_auc, test_acc, val_acc = trainer.train_fold(
            datasets=datasets,
            fold_idx=fold_idx
        )
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        print(f'Fold {fold_idx+1} 完成 - Test AUC: {test_auc:.4f}, Val AUC: {val_auc:.4f}')
        print(f'                Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # 验证保存的文件
        checkpoint_path = os.path.join(args.results_dir, f's_{fold_idx}_checkpoint.pt')
        split_path = os.path.join(args.results_dir, f'splits_{fold_idx}.csv')
        
        if os.path.exists(checkpoint_path):
            print(f"✅ 模型已保存: {checkpoint_path} ({os.path.getsize(checkpoint_path)} bytes)")
        else:
            print(f"❌ 模型未保存: {checkpoint_path}")
        
        if os.path.exists(split_path):
            print(f"✅ Split已保存: {split_path} ({os.path.getsize(split_path)} bytes)")
        else:
            print(f"❌ Split未保存: {split_path}")
    
    # 8. 保存最终结果摘要
    print(f"\n{'='*60}")
    print('训练结果摘要')
    print(f"{'='*60}")
    print(f'Mean Test AUC: {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}')
    print(f'Mean Val AUC: {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}')
    print(f'Mean Test Acc: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}')
    print(f'Mean Val Acc: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}')
    
    detailed_results = {
        'configurations': configs,
        'fold_results': {
            'folds': list(range(args.k)),
            'test_auc': all_test_auc,
            'val_auc': all_val_auc,
            'test_acc': all_test_acc,
            'val_acc': all_val_acc
        },
        'summary_stats': {
            'mean_test_auc': float(np.mean(all_test_auc)),
            'std_test_auc': float(np.std(all_test_auc)),
            'mean_val_auc': float(np.mean(all_val_auc)),
            'std_val_auc': float(np.std(all_val_auc)),
            'mean_test_acc': float(np.mean(all_test_acc)),
            'std_test_acc': float(np.std(all_test_acc)),
            'mean_val_acc': float(np.mean(all_val_acc)),
            'std_val_acc': float(np.std(all_val_acc))
        }
    }
    
    results_path = os.path.join(args.results_dir, 'detailed_results_for_plotting.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"\n✅ 详细结果已保存到: {results_path}")
    print(f"\n✅ 所有文件已保存到: {args.results_dir}")
    print(f"\n💡 现在可以使用 test_restore.py 来恢复并验证这些结果")

if __name__ == "__main__":
    main()

