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
    
    mil_config = {
        'model_size': args.model_size,
        'return_features': args.return_features,
    }
    clam_config = {
        'gate': args.gate,
        'base_weight': args.base_weight,
        'inst_loss_fn': args.inst_loss_fn,
        'model_size': args.model_size,
        'subtyping': args.subtyping,
        'inst_number': args.inst_number,
        'return_features': args.return_features,
        'attention_only': args.attention_only
    }
    auc_config = {
        'auc_loss_weight': args.auc_loss_weight,
    }
    transfer_layer_config = {
        'output_dim': args.output_dim,
    }
    svd_config = {
        'enable_svd': args.enable_svd,
        'alignment_layer_num': args.alignment_layer_num,
        'lambda1': args.lambda1,
        'lambda2': args.lambda2,
        'tau1': args.tau1,
        'tau2': args.tau2,
    }
    dynamic_gate_config = {
        'enable_dynamic_gate': args.enable_dynamic_gate,
        'confidence_weight': args.confidence_weight,
        'feature_weight_weight': args.feature_weight_weight,
    }
    random_loss_config = {
        'enable_random_loss': args.enable_random_loss,
        'weight_random_loss': args.weight_random_loss,
    }
    if model_type == 'mil':
        return {
            **mil_config,
        }
    elif model_type == 'clam':
        return {
            **clam_config,
        }
    elif model_type == 'auc_clam':
        return {
            **clam_config,
            **auc_config,
        }
    elif model_type == 'clam_mlp':
        return {
            **clam_config,
            **transfer_layer_config
        }
    elif model_type == 'clam_mlp_detach':
        return {
            **clam_config,
            **transfer_layer_config,
        }
    elif model_type == 'svd_gate_random_clam_detach':
        return {
            **clam_config,
            **transfer_layer_config,
            **svd_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'gate_shared_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
        }
    elif model_type == 'gate_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
        }
    elif model_type == 'gate_auc_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
            **auc_config,
        }
    elif model_type == 'gate_mil_detach':
        return {
            **mil_config,
            **dynamic_gate_config,
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

def load_dataset_split(dataset_split_path):
    """
    从JSON文件加载数据集分割信息
    
    Args:
        dataset_split_path (str): 数据集分割JSON文件路径
        
    Returns:
        dict: 包含train/test分割的字典，格式为 {'train': [patient_ids], 'test': [patient_ids]}
    """
    if not os.path.exists(dataset_split_path):
        raise FileNotFoundError(f"数据集分割文件不存在: {dataset_split_path}")
    
    with open(dataset_split_path, 'r') as f:
        split_data = json.load(f)
    
    # 将JSON数据转换为train/test分割
    train_patients = []
    test_patients = []
    
    for item in split_data:
        patient_id = item['patient_id']
        dataset_type = item['dataset']
        
        if dataset_type == 'training':
            train_patients.append(patient_id)
        elif dataset_type == 'test':
            test_patients.append(patient_id)
    
    return {
        'train': train_patients,
        'test': test_patients
    }

def create_k_fold_splits(dataset, k=10, seed=42, fixed_test_split=None):
    """
    创建k-fold交叉验证分割
    
    Args:
        dataset: 数据集对象
        k (int): fold数量
        seed (int): 随机种子
        fixed_test_split (dict, optional): 固定的测试集分割，格式为 {'train': [patient_ids], 'test': [patient_ids]}
    
    Returns:
        list: 包含每个fold的train/val/test索引的列表
    """
    from sklearn.model_selection import StratifiedKFold
    
    # 获取所有样本的标签和患者ID
    labels = []
    patient_ids = []
    
    for i in range(len(dataset)):
        # 获取样本数据
        sample = dataset[i]
        
        # 从数据集中获取标签
        if hasattr(dataset, 'get_label'):
            label = dataset.get_label(i)
        elif isinstance(sample, dict) and 'label' in sample:
            label = sample['label']
        else:
            # 假设是元组格式 (data, label)
            _, label = sample
        
        labels.append(label)
        
        # 获取患者ID（假设数据集有get_patient_id方法，或者从样本中获取）
        if hasattr(dataset, 'get_patient_id'):
            patient_id = dataset.get_patient_id(i)
        elif isinstance(sample, dict) and 'patient_id' in sample:
            patient_id = sample['patient_id']
        else:
            # 如果没有患者ID，使用索引作为ID
            patient_id = str(i)
        patient_ids.append(patient_id)
    
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    splits = []
    
    if fixed_test_split is not None:
        # 使用固定的测试集分割
        print(f"🔒 使用固定测试集分割")
        print(f"📊 固定训练集患者数: {len(fixed_test_split['train'])}")
        print(f"📊 固定测试集患者数: {len(fixed_test_split['test'])}")
        
        # 找到测试集对应的索引
        test_indices = []
        for test_patient_id in fixed_test_split['test']:
            test_idx = np.where(patient_ids == test_patient_id)[0]
            if len(test_idx) > 0:
                test_indices.extend(test_idx)
        
        test_indices = np.array(test_indices)
        
        # 找到训练集对应的索引
        train_indices = []
        for train_patient_id in fixed_test_split['train']:
            train_idx = np.where(patient_ids == train_patient_id)[0]
            if len(train_idx) > 0:
                train_indices.extend(train_idx)
        
        train_indices = np.array(train_indices)
        
        # 在训练集上进行k-fold交叉验证
        train_labels = labels[train_indices]
        
        # 创建分层k-fold分割
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_indices, train_labels)):
            # 转换为实际索引
            actual_train_idx = train_indices[fold_train_idx]
            actual_val_idx = train_indices[fold_val_idx]
            
            splits.append({
                'train': actual_train_idx,
                'val': actual_val_idx,
                'test': test_indices  # 测试集始终相同
            })
    else:
        # 原始的分割方式：将测试集进一步分为验证集和测试集
        print(f"🔄 使用传统k-fold分割")
        
        # 创建分层k-fold分割
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
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

def parse_channels(channels):
    """
    解析channels列表，将简化的通道名称映射为完整的HDF5路径
    
    支持的通道类型：
    - WSI: 'wsi' -> 'wsi=features'
    - TMA Features: 'tma', 'cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1'
    - TMA Patches: 'tma_patches', 'cd163_patches', 'cd3_patches', etc.
    - Clinical: 'clinical', 'clinical_ori', 'clinical_mask', 'clinical_ori_mask'
    - Pathological: 'pathological', 'pathological_ori', 'pathological_mask', 'pathological_ori_mask'
    - Blood: 'blood', 'blood_ori', 'blood_mask', 'blood_ori_mask'
    - ICD: 'icd', 'icd_ori', 'icd_mask', 'icd_ori_mask'
    - TMA Cell Density: 'tma_cell_density', 'tma_cell_density_ori', 'tma_cell_density_mask', 'tma_cell_density_ori_mask'
    
    Args:
        channels (List[str]): 通道名称列表
        
    Returns:
        List[str]: 解析后的完整通道路径列表
        
    Raises:
        ValueError: 当输入通道名称无效时
    """
    if not channels:
        return []
    
    # TMA通道定义
    TMA_CHANNELS = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']
    
    # 支持的通道类型映射
    CHANNEL_MAPPINGS = {
        # WSI通道
        'wsi': ['wsi=features'],
        
        # TMA Features通道
        'tma': [f'tma={channel}=features' for channel in TMA_CHANNELS],
        
        # TMA Patches通道
        'tma_patches': [f'tma={channel}=patches' for channel in TMA_CHANNELS],
        
        # Clinical通道
        'clinical': ['clinical=val'],
        'clinical_ori': ['clinical=ori_val'],
        'clinical_mask': ['clinical=val', 'clinical=mask'],
        'clinical_ori_mask': ['clinical=ori_val', 'clinical=mask'],
        
        # Pathological通道
        'pathological': ['pathological=val'],
        'pathological_ori': ['pathological=ori_val'],
        'pathological_mask': ['pathological=val', 'pathological=mask'],
        'pathological_ori_mask': ['pathological=ori_val', 'pathological=mask'],
        
        # Blood通道
        'blood': ['blood=val'],
        'blood_ori': ['blood=ori_val'],
        'blood_mask': ['blood=val', 'blood=mask'],
        'blood_ori_mask': ['blood=ori_val', 'blood=mask'],
        
        # ICD通道
        'icd': ['icd=val'],
        'icd_ori': ['icd=ori_val'],
        'icd_mask': ['icd=val', 'icd=mask'],
        'icd_ori_mask': ['icd=ori_val', 'icd=mask'],
        
        # TMA Cell Density通道
        'tma_cell_density': ['tma_cell_density=val'],
        'tma_cell_density_ori': ['tma_cell_density=ori_val'],
        'tma_cell_density_mask': ['tma_cell_density=val', 'tma_cell_density=mask'],
        'tma_cell_density_ori_mask': ['tma_cell_density=ori_val', 'tma_cell_density=mask'],
    }
    
    # 添加单个TMA通道的映射
    for channel in TMA_CHANNELS:
        CHANNEL_MAPPINGS[channel] = [f'tma={channel}=features']
        CHANNEL_MAPPINGS[f'{channel}_patches'] = [f'tma={channel}=patches']
    
    parsed_channels = []
    invalid_channels = []
    
    for channel in channels:
        if channel in CHANNEL_MAPPINGS:
            parsed_channels.extend(CHANNEL_MAPPINGS[channel])
        elif '=' in channel:  # 已经是完整路径格式
            parsed_channels.append(channel)
        else:
            invalid_channels.append(channel)
    
    # 验证无效通道
    if invalid_channels:
        available_channels = list(CHANNEL_MAPPINGS.keys())
        raise ValueError(
            f"❌ 无效的通道名称: {invalid_channels}\n"
            f"📋 支持的通道类型: {available_channels}\n"
            f"💡 提示: 通道名称不区分大小写，支持单个通道或组合通道"
        )
    
    return parsed_channels

def get_available_channels():
    """
    获取所有可用的通道类型列表
    
    Returns:
        Dict[str, List[str]]: 按类别分组的可用通道字典
    """
    TMA_CHANNELS = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']
    
    return {
        'WSI通道': ['wsi'],
        'TMA Features通道': ['tma'] + TMA_CHANNELS,
        'TMA Patches通道': ['tma_patches'] + [f'{ch}_patches' for ch in TMA_CHANNELS],
        'Clinical通道': ['clinical', 'clinical_ori', 'clinical_mask', 'clinical_ori_mask'],
        'Pathological通道': ['pathological', 'pathological_ori', 'pathological_mask', 'pathological_ori_mask'],
        'Blood通道': ['blood', 'blood_ori', 'blood_mask', 'blood_ori_mask'],
        'ICD通道': ['icd', 'icd_ori', 'icd_mask', 'icd_ori_mask'],
        'TMA Cell Density通道': ['tma_cell_density', 'tma_cell_density_ori', 'tma_cell_density_mask', 'tma_cell_density_ori_mask']
    }

def print_available_channels():
    """
    打印所有可用的通道类型，用于调试和帮助
    """
    channels = get_available_channels()
    print("🔍 可用的通道类型:")
    print("=" * 50)
    
    for category, channel_list in channels.items():
        print(f"\n📁 {category}:")
        for channel in channel_list:
            print(f"  • {channel}")
    
    print("\n💡 使用示例:")
    print("  • 单个通道: ['wsi', 'clinical']")
    print("  • 组合通道: ['tma', 'blood_mask']")
    print("  • 完整路径: ['wsi=features', 'clinical=val']")

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
    
    # 测试parse_channels函数
    try:
        parsed_channels = parse_channels(channels)
        print(f"✅ 成功解析通道: {len(parsed_channels)} 个")
        print(f"📋 原始通道: {channels}")
        print(f"🔗 解析后通道: {parsed_channels}")
    except ValueError as e:
        print(f"❌ 通道解析错误: {e}")
        print_available_channels()
        return
    
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
    print(f"🔧 分割模式: {args.split_mode}")
    
    # 检查是否使用固定测试集
    fixed_test_split = None
    if args.split_mode == 'fixed':
        if not args.dataset_split_path:
            raise ValueError("❌ 使用固定测试集模式时，必须提供 --dataset_split_path 参数")
        print(f"📁 加载固定测试集分割: {args.dataset_split_path}")
        fixed_test_split = load_dataset_split(args.dataset_split_path)
        print(f"✅ 成功加载固定测试集分割")
    elif args.split_mode == 'random':
        print(f"🎲 使用随机分割模式")
    else:
        raise ValueError(f"❌ 不支持的分割模式: {args.split_mode}")
    
    splits = create_k_fold_splits(dataset, k=args.k, seed=args.seed, fixed_test_split=fixed_test_split)
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
parser.add_argument('--split_mode', type=str, choices=['random', 'fixed'], default='random',
                    help='数据集分割模式: random=随机分割, fixed=固定测试集分割 (default: random)')
parser.add_argument('--dataset_split_path', type=str, default=None,
                    help='固定测试集分割JSON文件路径 (仅在split_mode=fixed时使用)')
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
parser.add_argument('--batch_size', type=int, default=64,
                    help='批次大小 (default: 64)')
parser.add_argument('--lr_scheduler', type=str, 
                    choices=['none', 'cosine', 'cosine_warm_restart', 'step', 'plateau', 'exponential'], 
                    default='none',
                    help='学习率调度器类型 (default: none)')
parser.add_argument('--lr_scheduler_params', type=str, default='{}',
                    help='学习率调度器参数 (JSON字符串，默认: {})')

# 模型相关参数
parser.add_argument('--model_type', type=str, choices=[
    'mil', 'clam', 'auc_clam', 'clam_mlp', 'clam_mlp_detach', 'svd_gate_random_clam', 'svd_gate_random_clam_detach', 
    'gate_shared_mil', 'gate_mil_detach', 'gate_mil', 'gate_auc_mil'
    ], 
                    default='clam', help='模型类型 (default: clam)')
parser.add_argument('--input_dim', type=int, default=1024,
                    help='输入维度')
parser.add_argument('--dropout', type=float, default=0.25, 
                    help='dropout率')
parser.add_argument('--n_classes', type=int, default=2,
                    help='类别数 (default: 2)')
parser.add_argument('--base_loss_fn', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide级别分类损失函数 (default: ce)')

# CLAM 相关参数
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
parser.add_argument('--channels_used_in_model', type=str, nargs='+', 
                    default=['features', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1'],
                    help='模型中需要使用的通道')
parser.add_argument('--return_features', action='store_true', default=False, 
                    help='MIL & CLAM: 返回特征')
parser.add_argument('--attention_only', action='store_true', default=False, 
                    help='CLAM: 仅返回注意力')

# Transfer layer
parser.add_argument('--output_dim', type=int, default=128, 
                    help='Transfer layer: 模态统一的输出维度')

# SVD相关参数
parser.add_argument('--enable_svd', action='store_true', default=False, 
                    help='SVD: 启用SVD')
parser.add_argument('--alignment_layer_num', type=int, default=2,
                    help='SVD: 对齐层数')
parser.add_argument('--lambda1', type=float, default=1.0,
                    help='SVD: 对齐损失权重')
parser.add_argument('--lambda2', type=float, default=0.0,
                    help='SVD: 对齐损失权重')
parser.add_argument('--tau1', type=float, default=0.1,
                    help='SVD: 对齐损失权重')
parser.add_argument('--tau2', type=float, default=0.05,
                    help='SVD: 对齐损失权重')

# Dynamic Gate相关参数
parser.add_argument('--enable_dynamic_gate', action='store_true', default=False, 
                    help='Dynamic Gate: 启用动态门控')
parser.add_argument('--confidence_weight', type=float, default=1.0,
                    help='Dynamic Gate: 置信度权重')
parser.add_argument('--feature_weight_weight', type=float, default=1.0,
                    help='Dynamic Gate: 特征权重权重')

# AUC相关参数
parser.add_argument('--auc_loss_weight', type=float, default=1.0,
                    help='AUC: AUC损失权重')

# Random Loss相关参数
parser.add_argument('--enable_random_loss', action='store_true', default=False, 
                    help='Random Loss: 启用随机损失')
parser.add_argument('--weight_random_loss', type=float, default=0.1, 
                    help='Random Loss: 随机损失权重')
# 解析参数
args = parser.parse_args()
args.target_channels = parse_channels(args.target_channels)
args.aligned_channels = parse_channels(args.aligned_channels)
args.channels_used_in_model = parse_channels(args.channels_used_in_model)
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
        'split_mode': args.split_mode,
        'dataset_split_path': args.dataset_split_path,
        'max_epochs': args.max_epochs,
        'lr': args.lr,
        'reg': args.reg,
        'opt': args.opt,
        'early_stopping': args.early_stopping,
        'batch_size': args.batch_size,
        'scheduler_config': {
            'type': args.lr_scheduler if args.lr_scheduler != 'none' else None,
            **(json.loads(args.lr_scheduler_params) if args.lr_scheduler_params else {})
        }
    },
    
    'model_config': {
        'model_type': args.model_type,
        'input_dim': args.input_dim,
        'dropout': args.dropout,
        'n_classes': args.n_classes,
        'base_loss_fn': args.base_loss_fn,
        'channels_used_in_model': args.channels_used_in_model,
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
