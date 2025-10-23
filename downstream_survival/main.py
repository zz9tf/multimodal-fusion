#!/usr/bin/env python3
"""
多模态生存状态预测主程序
专注于 WSI + TMA 多模态生存状态预测任务
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Subset

# 内置工具函数，减少外部依赖
import pickle

# 添加项目路径
import sys
# Add which folder your main.py is located as root_dir
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

# 内部导入
from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset

def save_pkl(filename, data):
    """保存pickle文件"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    """加载pickle文件"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def _get_model_specific_config(args):
    """根据模型类型获取特定配置"""
    model_type = args.model_type
    
    if model_type == 'clam':
        return {
            'gate': args.gate,
            'base_weight': args.base_weight,
            'inst_loss_fn': args.inst_loss_fn,
            'model_size': args.model_size,
            'subtyping': args.subtyping,
            'inst_number': args.inst_number,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'attention_only': args.attention_only
        }
    elif model_type == 'auc_clam':
        return {
            'gate': args.gate,
            'base_weight': args.base_weight,
            'inst_loss_fn': args.inst_loss_fn,
            'model_size': args.model_size,
            'subtyping': args.subtyping,
            'inst_number': args.inst_number,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'attention_only': args.attention_only,
            'auc_loss_weight': args.auc_loss_weight,
        }
    elif model_type == 'mil':
        return {
            'model_size': args.model_size,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
        }
    elif model_type == 'clam_svd_loss':
        return {
            'gate': args.gate,
            'base_weight': args.base_weight,
            'inst_loss_fn': args.inst_loss_fn,
            'model_size': args.model_size,
            'subtyping': args.subtyping,
            'inst_number': args.inst_number,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'attention_only': args.attention_only,
            'alignment_layer_num': args.alignment_layer_num,
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'tau1': args.tau1,
            'tau2': args.tau2,
        }
    elif model_type == 'gate_shared_mil':
        return {
            'model_size': args.model_size,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'confidence_weight': args.confidence_weight,
            'feature_weight_weight': args.feature_weight_weight,
            'channels_used_in_model': args.channels_used_in_model,
        }
    elif model_type == 'gate_mil':
        return {
            'model_size': args.model_size,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'confidence_weight': args.confidence_weight,
            'feature_weight_weight': args.feature_weight_weight,
            'channels_used_in_model': args.channels_used_in_model,
        }
    elif model_type == 'gate_auc_mil':
        return {
            'model_size': args.model_size,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'confidence_weight': args.confidence_weight,
            'feature_weight_weight': args.feature_weight_weight,
            'channels_used_in_model': args.channels_used_in_model,
            'auc_loss_weight': args.auc_loss_weight,
        }
    elif model_type == 'gate_mil_detach':
        return {
            'model_size': args.model_size,
            'channels_used_in_model': args.channels_used_in_model,
            'return_features': args.return_features,
            'confidence_weight': args.confidence_weight,
            'feature_weight_weight': args.feature_weight_weight,
            'channels_used_in_model': args.channels_used_in_model,
        }
    else:
        # 为其他模型类型返回空配置，可以根据需要扩展
        return {}

def _parse_aligned_channels(aligned_channels_list):
    """解析对齐通道参数"""
    if not aligned_channels_list:
        return {}
    
    align_channels = {}
    for item in aligned_channels_list:
        if '=' in item:
            key, value = item.split('=', 1)
            align_channels[key] = value
        else:
            # 如果没有等号，假设key=value相同
            align_channels[item] = item
    
    return align_channels

def seed_torch(seed=7):
    """设置随机种子"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_k_fold_splits(dataset, k=10, seed=42):
    """创建k-fold交叉验证分割（使用sklearn的StratifiedKFold）"""
    from sklearn.model_selection import StratifiedKFold
    
    # 获取所有样本的标签
    labels = []
    for i in range(len(dataset)):
        # 从数据集中获取标签
        if hasattr(dataset, 'get_label'):
            label = dataset.get_label(i)
        else:
            # 如果是字典格式，获取label
            sample = dataset[i]
            if isinstance(sample, dict) and 'label' in sample:
                label = sample['label']
            else:
                # 假设是元组格式 (data, label)
                _, label = sample
        labels.append(label)
    
    labels = np.array(labels)
    
    # 创建分层k-fold分割
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        # 将测试集进一步分为验证集和测试集
        test_labels = labels[test_idx]
        val_test_skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        val_idx, test_idx_final = next(val_test_skf.split(test_idx, test_labels))
        
        # 转换为实际索引
        val_idx = test_idx[val_idx]
        test_idx_final = test_idx[test_idx_final]
        
        splits.append({
            'train': train_idx,
            'val': val_idx, 
            'test': test_idx_final
        })
    
    return splits

def main(args, configs):
    """主函数"""
    # 从配置中获取参数
    experiment_config = configs['experiment_config']
    
    # 加载数据集
    print('\nLoad Dataset')
    if not experiment_config['data_root_dir']:
        raise ValueError('data_root_dir is required')
    if not os.path.exists(experiment_config['data_root_dir']):
        raise ValueError('data_root_dir does not exist')
    
    print('data_root_dir: ', os.path.abspath(experiment_config['data_root_dir']))

    # 创建多模态数据集
    print(f"Target channels: {experiment_config['target_channels']}")
    
    # 构建channels列表
    channels = args.target_channels
    
    # 构建align_channels映射
    align_channels = _parse_aligned_channels(args.aligned_channels)
    
    print(f"Channels: {channels}")
    print(f"Align channels: {align_channels}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultimodalDataset(
        csv_path=experiment_config['csv_path'],
        data_root_dir=experiment_config['data_root_dir'],
        channels=channels,
        align_channels=align_channels,
        alignment_model_path=experiment_config['alignment_model_path'],
        device=device,
        print_info=True
    )
    
    # 创建结果目录
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # 创建k-fold分割
    print(f'\nCreating {args.k}-fold cross-validation splits...')
    splits = create_k_fold_splits(dataset, k=args.k, seed=args.seed)
    print(f'✅ Created {len(splits)} folds')

    # 确定fold范围
    start = 0
    end = args.k

    # 初始化训练器
    trainer = Trainer(
        configs=configs,
        log_dir=os.path.join(args.results_dir, 'training_logs')
    )

    # 存储结果
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    
    # 训练每个fold
    for i in folds:
        print(f'\n{"="*60}')
        print(f'Training Fold {i+1}/{args.k}')
        print(f'{"="*60}')
        
        seed_torch(args.seed)
        
        # 获取当前fold的分割
        split = splits[i]
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
        results, test_auc, val_auc, test_acc, val_acc = trainer.train_fold(
            datasets=datasets,
            fold_idx=i
        )
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        # 保存结果
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        
        print(f'Fold {i+1} completed - Test AUC: {test_auc:.4f}, Val AUC: {val_auc:.4f}')

    # 保存最终结果
    final_df = pd.DataFrame({
        'folds': folds, 
        'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 
        'test_acc': all_test_acc, 
        'val_acc': all_val_acc
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    # 保存详细的训练数据用于后续绘图
    detailed_results = {
        'configurations': configs,
        'fold_results': {
            'folds': folds.tolist(),
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
    
    # 保存详细结果用于绘图
    detailed_save_name = 'detailed_results_for_plotting.json'
    with open(os.path.join(args.results_dir, detailed_save_name), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # 打印最终统计
    print(f'\n{"="*60}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"="*60}')
    print(f'Mean Test AUC: {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}')
    print(f'Mean Val AUC: {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}')
    print(f'Mean Test Acc: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}')
    print(f'Mean Val Acc: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}')
    print(f'Results saved to: {os.path.join(args.results_dir, save_name)}')
    print(f'Detailed results for plotting: {os.path.join(args.results_dir, detailed_save_name)}')

# 参数解析
parser = argparse.ArgumentParser(description='多模态生存状态预测配置')

# 数据相关参数
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='数据根目录')
parser.add_argument('--results_dir', default='./results', 
                    help='结果保存目录 (default: ./results)')
parser.add_argument('--csv_path', type=str, default='dataset_csv/survival_status_labels.csv', 
                    help='CSV文件路径')
# 对齐模型相关参数
parser.add_argument('--alignment_model_path', type=str, default=None, 
                    help='预训练对齐模型路径（提供此参数将自动启用对齐功能）')
# 多模态相关参数
parser.add_argument('--target_channels', type=str, nargs='+', 
                    default=['CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1'], 
                    help='目标通道')
parser.add_argument('--aligned_channels', type=str, nargs='*', 
                    default=None,
                    help='对齐目标，格式: channel_to_align1=align_channel_name1 channel_to_align2=align_channel_name2 ...')
# 实验相关参数
parser.add_argument('--exp_code', type=str, 
                    help='实验代码，用于保存结果')
parser.add_argument('--seed', type=int, default=1, 
                    help='随机种子 (default: 1)')
parser.add_argument('--k', type=int, default=10, 
                    help='fold数量 (default: 10)')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='最大训练轮数 (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='学习率 (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='权重衰减 (default: 1e-5)')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam',
                    help='优化器类型')
parser.add_argument('--early_stopping', action='store_true', default=False, 
                    help='启用早停')
parser.add_argument('--batch_size', type=int, default=1,
                    help='批次大小 (default: 1)')

# 模型相关参数
parser.add_argument('--model_type', type=str, choices=['clam', 'auc_clam', 'clam_svd_loss', 'mil', 'gate_shared_mil', 'gate_mil', 'gate_auc_mil', 'gate_mil_detach'], 
                    default='clam', help='模型类型 (default: clam)')
parser.add_argument('--input_dim', type=int, default=1024,
                    help='输入维度')
parser.add_argument('--dropout', type=float, default=0.25, 
                    help='dropout率')
parser.add_argument('--n_classes', type=int, default=2,
                    help='类别数 (default: 2)')
parser.add_argument('--base_loss_fn', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide级别分类损失函数 (default: ce)')

# CLAM相关参数
parser.add_argument('--gate', action='store_true', default=True, 
                    help='CLAM: 使用门控注意力机制')
parser.add_argument('--base_weight', type=float, default=0.7,
                    help='CLAM: bag级别损失权重系数 (default: 0.7)')
parser.add_argument('--inst_loss_fn', type=str, choices=['svm', 'ce', None], default=None,
                    help='CLAM: 实例级别聚类损失函数 (default: None)')
parser.add_argument('--model_size', type=str, 
                    choices=['small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1'], 
                    default='small', help='模型大小')
parser.add_argument('--subtyping', action='store_true', default=False, 
                    help='子类型问题')
parser.add_argument('--inst_number', type=int, default=8, 
                    help='CLAM: 正负样本采样数量')
parser.add_argument('--channels_used_in_model', type=str, nargs='+', default=None,
                    help='模型中需要使用的通道')
parser.add_argument('--return_features', action='store_true', default=False, 
                    help='MIL & CLAM: 返回特征')
parser.add_argument('--attention_only', action='store_true', default=False, 
                    help='CLAM: 仅返回注意力')

# CLAM_SVD_LOSS相关参数
parser.add_argument('--alignment_layer_num', type=int, default=2,
                    help='CLAM_SVD_LOSS: 对齐层数')
parser.add_argument('--lambda1', type=float, default=1.0,
                    help='CLAM_SVD_LOSS: 对齐损失权重')
parser.add_argument('--lambda2', type=float, default=0.0,
                    help='CLAM_SVD_LOSS: 对齐损失权重')
parser.add_argument('--tau1', type=float, default=0.1,
                    help='CLAM_SVD_LOSS: 对齐损失权重')
parser.add_argument('--tau2', type=float, default=0.05,
                    help='CLAM_SVD_LOSS: 对齐损失权重')

# GatedMIL相关参数
parser.add_argument('--confidence_weight', type=float, default=1.0,
                    help='GatedMIL: 置信度权重')
parser.add_argument('--feature_weight_weight', type=float, default=1.0,
                    help='GatedMIL: 特征权重权重')

# AUC_CLAM & GateAUCMIL相关参数
parser.add_argument('--auc_loss_weight', type=float, default=1.0,
                    help='AUC_CLAM & GateAUCMIL: AUC损失权重')

# 解析参数
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
seed_torch(args.seed)

# 创建结果目录
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# 创建带时间戳的结果目录
args.results_dir = os.path.join(
    args.results_dir, 
    datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + str(args.exp_code) + '_s{}'.format(args.seed)
)
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# 创建精简的分类配置字典
configs = {
    'experiment_config': {
        'data_root_dir': args.data_root_dir,
        'results_dir': args.results_dir,
        'csv_path': args.csv_path,
        'alignment_model_path': args.alignment_model_path,
        'target_channels': args.target_channels,
        'aligned_channels': args.aligned_channels,
        'exp_code': args.exp_code,
        'seed': args.seed,
        'num_splits': args.k,
        'max_epochs': args.max_epochs,
        'lr': args.lr,
        'reg': args.reg,
        'opt': args.opt,
        'early_stopping': args.early_stopping,
        'batch_size': args.batch_size
    },
    
    'model_config': {
        'model_type': args.model_type,
        'input_dim': args.input_dim,
        'dropout': args.dropout,
        'n_classes': args.n_classes,
        'base_loss_fn': args.base_loss_fn,
        **_get_model_specific_config(args)
    }
}

# 保存分类配置
with open(args.results_dir + '/configs_{}.json'.format(args.exp_code), 'w') as f:
    json.dump(configs, f, indent=2)

# 打印精简配置
print("################# Configuration ###################")
print(f"\n📋 EXPERIMENT CONFIG:")
for key, val in configs['experiment_config'].items():
    print(f"  {key}: {val}")

print(f"\n📋 MODEL CONFIG:")
for key, val in configs['model_config'].items():
    print(f"  {key}: {val}")

if __name__ == "__main__":
    results = main(args, configs)
    print("finished!")
    print("end script")
