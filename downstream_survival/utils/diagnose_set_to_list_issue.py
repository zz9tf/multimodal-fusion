#!/usr/bin/env python3
"""
诊断set转list导致的不一致问题
重点检查modalities_used_in_model是set，遍历顺序不确定的问题
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Subset

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

from trainer import Trainer, get_split_loader
from main import seed_torch, create_k_fold_splits
from datasets.multimodal_dataset import MultimodalDataset


def diagnose_set_to_list_issue(results_dir: str, fold_idx: int = 0):
    """
    诊断set转list导致的不一致问题
    """
    results_dir = Path(results_dir)
    configs_file = results_dir / 'configs_svd_random_clam_detach.json'
    
    if not configs_file.exists():
        config_files = list(results_dir.glob('configs_*.json'))
        if config_files:
            configs_file = config_files[0]
        else:
            raise FileNotFoundError(f"未找到配置文件: {results_dir}")
    
    # 加载配置
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    
    print(f"📋 加载配置文件: {configs_file}")
    
    # 获取seed
    seed = configs['experiment_config'].get('seed', 5678)
    print(f"🌱 使用随机种子: {seed}")
    
    # 加载数据集
    experiment_config = configs['experiment_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultimodalDataset(
        csv_path=experiment_config['csv_path'],
        data_root_dir=experiment_config['data_root_dir'],
        channels=experiment_config['target_channels'],
        align_channels={},
        alignment_model_path=experiment_config['alignment_model_path'],
        device=device,
        print_info=False
    )
    
    # 创建k-fold分割
    splits = create_k_fold_splits(dataset, k=10, seed=seed, fixed_test_split=None)
    split = splits[fold_idx]
    train_idx = split['train']
    train_dataset = Subset(dataset, train_idx)
    
    # 创建训练器
    trainer = Trainer(
        configs=configs,
        log_dir=str(results_dir / 'training_logs')
    )
    
    # 初始化两个模型
    seed_torch(seed)
    model1 = trainer._init_model()
    
    seed_torch(seed)
    model2 = trainer._init_model()
    
    # 创建数据加载器
    seed_torch(seed)
    train_loader = get_split_loader(train_dataset, training=True, weighted=True, batch_size=1)
    train_loader_list = list(train_loader)
    
    print(f"\n{'='*60}")
    print("诊断set转list导致的不一致问题")
    print(f"{'='*60}")
    
    # 获取第一个batch
    data, label = train_loader_list[0]
    label = label.to(device)
    for channel in data:
        data[channel] = data[channel].to(device)
    
    print(f"\n📊 第一个batch的数据:")
    print(f"   channels: {list(data.keys())}")
    
    # 检查多次调用_process_input_data返回的modalities_used_in_model顺序
    print(f"\n🔍 检查多次调用 _process_input_data 返回的 modalities_used_in_model 顺序:")
    
    all_orders = []
    for i in range(10):
        seed_torch(seed + 10000 + i)
        input_data, modalities_used_in_model = model1._process_input_data(data.copy())
        order = list(modalities_used_in_model)
        all_orders.append(order)
        print(f"   调用 {i+1}: {order}")
    
    # 检查顺序是否一致
    if len(set(tuple(order) for order in all_orders)) == 1:
        print(f"   ✅ 所有调用的顺序一致")
    else:
        print(f"   ❌ 顺序不一致！")
        unique_orders = set(tuple(order) for order in all_orders)
        print(f"      发现 {len(unique_orders)} 种不同的顺序")
        for idx, order in enumerate(unique_orders):
            print(f"      顺序 {idx+1}: {list(order)}")
    
    # 检查features_dict的构建顺序
    print(f"\n🔍 检查 features_dict 的构建顺序（基于 modalities_used_in_model 的遍历顺序）:")
    
    all_features_dict_orders = []
    for i in range(10):
        seed_torch(seed + 20000 + i)
        input_data, modalities_used_in_model = model1._process_input_data(data.copy())
        
        # 模拟features_dict的构建过程
        features_dict = {}
        for channel in modalities_used_in_model:  # 这里遍历set，顺序不确定
            if channel == 'wsi=features':
                features_dict[channel] = torch.randn(1, 128)
            elif channel == 'tma=features':
                features_dict[channel] = torch.randn(1, 128)
            else:
                features_dict[channel] = torch.randn(1, 128)
        
        order = list(features_dict.keys())
        all_features_dict_orders.append(order)
        print(f"   调用 {i+1}: {order}")
    
    # 检查顺序是否一致
    if len(set(tuple(order) for order in all_features_dict_orders)) == 1:
        print(f"   ✅ 所有调用的features_dict keys顺序一致")
    else:
        print(f"   ❌ features_dict keys顺序不一致！")
        unique_orders = set(tuple(order) for order in all_features_dict_orders)
        print(f"      发现 {len(unique_orders)} 种不同的顺序")
        for idx, order in enumerate(unique_orders):
            print(f"      顺序 {idx+1}: {list(order)}")
    
    # 检查random.sample的结果
    print(f"\n🔍 检查 random.sample(list(features_dict.keys()), ...) 的结果:")
    
    all_random_sample_results = []
    for i in range(10):
        seed_torch(seed + 30000 + i)
        input_data, modalities_used_in_model = model1._process_input_data(data.copy())
        
        # 模拟features_dict的构建过程
        features_dict = {}
        for channel in modalities_used_in_model:
            if channel == 'wsi=features':
                features_dict[channel] = torch.randn(1, 128)
            elif channel == 'tma=features':
                features_dict[channel] = torch.randn(1, 128)
            else:
                features_dict[channel] = torch.randn(1, 128)
        
        # 模拟random.sample
        import random
        keys_list = list(features_dict.keys())
        n = random.randint(1, len(keys_list) - 1)
        drop_modality = random.sample(keys_list, n)
        
        result = {
            'keys_list': keys_list,
            'n': n,
            'drop_modality': sorted(drop_modality)  # 排序以便比较
        }
        all_random_sample_results.append(result)
        print(f"   调用 {i+1}: keys_list={keys_list}, n={n}, drop_modality={sorted(drop_modality)}")
    
    # 检查keys_list是否一致
    keys_lists = [result['keys_list'] for result in all_random_sample_results]
    if len(set(tuple(keys) for keys in keys_lists)) == 1:
        print(f"   ✅ 所有调用的keys_list顺序一致")
    else:
        print(f"   ❌ keys_list顺序不一致！")
        unique_keys_lists = set(tuple(keys) for keys in keys_lists)
        print(f"      发现 {len(unique_keys_lists)} 种不同的keys_list顺序")
    
    # 检查drop_modality是否一致（即使keys_list顺序不同，如果random seed相同，drop_modality应该一致）
    drop_modalities = [result['drop_modality'] for result in all_random_sample_results]
    if len(set(tuple(drop) for drop in drop_modalities)) == 1:
        print(f"   ✅ 所有调用的drop_modality一致（即使keys_list顺序不同）")
    else:
        print(f"   ❌ drop_modality不一致！")
        unique_drop_modalities = set(tuple(drop) for drop in drop_modalities)
        print(f"      发现 {len(unique_drop_modalities)} 种不同的drop_modality")
    
    # 检查两个模型在相同seed下的行为
    print(f"\n🔍 检查两个模型在相同seed下的行为:")
    
    seed_torch(seed + 40000)
    input_data1, modalities_used_in_model1 = model1._process_input_data(data.copy())
    features_dict1 = {}
    for channel in modalities_used_in_model1:
        if channel == 'wsi=features':
            features_dict1[channel] = torch.randn(1, 128)
        elif channel == 'tma=features':
            features_dict1[channel] = torch.randn(1, 128)
        else:
            features_dict1[channel] = torch.randn(1, 128)
    
    seed_torch(seed + 40000)
    input_data2, modalities_used_in_model2 = model2._process_input_data(data.copy())
    features_dict2 = {}
    for channel in modalities_used_in_model2:
        if channel == 'wsi=features':
            features_dict2[channel] = torch.randn(1, 128)
        elif channel == 'tma=features':
            features_dict2[channel] = torch.randn(1, 128)
        else:
            features_dict2[channel] = torch.randn(1, 128)
    
    keys1 = list(features_dict1.keys())
    keys2 = list(features_dict2.keys())
    
    print(f"   模型1 - modalities_used_in_model顺序: {list(modalities_used_in_model1)}")
    print(f"   模型2 - modalities_used_in_model顺序: {list(modalities_used_in_model2)}")
    print(f"   模型1 - features_dict keys顺序: {keys1}")
    print(f"   模型2 - features_dict keys顺序: {keys2}")
    
    if keys1 == keys2:
        print(f"   ✅ 两个模型的features_dict keys顺序一致")
    else:
        print(f"   ❌ 两个模型的features_dict keys顺序不一致！")
        print(f"      这是导致训练不一致的根本原因！")
    
    # 检查random.sample在两个模型上的结果
    import random
    seed_torch(seed + 40000)
    n1 = random.randint(1, len(keys1) - 1)
    drop1 = random.sample(keys1, n1)
    
    seed_torch(seed + 40000)
    n2 = random.randint(1, len(keys2) - 1)
    drop2 = random.sample(keys2, n2)
    
    print(f"\n   模型1 - random.sample({keys1}, {n1}): {sorted(drop1)}")
    print(f"   模型2 - random.sample({keys2}, {n2}): {sorted(drop2)}")
    
    if sorted(drop1) == sorted(drop2):
        print(f"   ✅ random.sample结果一致（即使keys顺序不同）")
    else:
        print(f"   ❌ random.sample结果不一致！")
        print(f"      即使random seed相同，但由于keys顺序不同，random.sample的结果也不同！")
    
    # 总结
    print(f"\n{'='*60}")
    print("诊断总结")
    print(f"{'='*60}")
    print(f"问题根源:")
    print(f"1. modalities_used_in_model 是 set()，遍历顺序不确定")
    print(f"2. for channel in modalities_used_in_model: 构建 features_dict 时，keys顺序不确定")
    print(f"3. random.sample(list(features_dict.keys()), ...) 依赖 keys 的顺序")
    print(f"4. 即使 random seed 相同，如果 keys 顺序不同，random.sample 的结果也会不同")
    print(f"5. 这导致两个模型在相同 seed 下，forward 的结果不同，进而导致训练不一致")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断set转list导致的不一致问题')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果目录路径')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold索引 (default: 0)')
    
    args = parser.parse_args()
    
    try:
        diagnose_set_to_list_issue(
            args.results_dir,
            args.fold_idx
        )
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



