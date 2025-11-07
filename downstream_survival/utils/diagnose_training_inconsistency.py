#!/usr/bin/env python3
"""
诊断训练不一致的原因
重点检查set转list和字典键顺序的问题
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


def diagnose_training_inconsistency(results_dir: str, fold_idx: int = 0):
    """
    诊断训练不一致的原因
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
    trainer.loss_fn = model1.loss_fn
    
    seed_torch(seed)
    model2 = trainer._init_model()
    
    # 创建数据加载器
    seed_torch(seed)
    train_loader = get_split_loader(train_dataset, training=True, weighted=True, batch_size=1)
    train_loader_list = list(train_loader)
    
    print(f"\n{'='*60}")
    print("诊断训练不一致的原因")
    print(f"{'='*60}")
    
    # 获取第一个batch
    data, label = train_loader_list[0]
    label = label.to(device)
    for channel in data:
        data[channel] = data[channel].to(device)
    
    print(f"\n📊 第一个batch的数据:")
    print(f"   channels: {list(data.keys())}")
    print(f"   channels顺序: {list(data.keys())}")
    
    # 检查_process_input_data返回的modalities_used_in_model
    print(f"\n🔍 检查 _process_input_data 返回的 modalities_used_in_model:")
    
    seed_torch(seed + 10000)
    input_data1, modalities_used_in_model1 = model1._process_input_data(data.copy())
    print(f"   模型1 - modalities_used_in_model类型: {type(modalities_used_in_model1)}")
    print(f"   模型1 - modalities_used_in_model: {modalities_used_in_model1}")
    print(f"   模型1 - modalities_used_in_model顺序: {list(modalities_used_in_model1)}")
    
    seed_torch(seed + 10000)
    input_data2, modalities_used_in_model2 = model2._process_input_data(data.copy())
    print(f"   模型2 - modalities_used_in_model类型: {type(modalities_used_in_model2)}")
    print(f"   模型2 - modalities_used_in_model: {modalities_used_in_model2}")
    print(f"   模型2 - modalities_used_in_model顺序: {list(modalities_used_in_model2)}")
    
    # 检查顺序是否一致
    list1 = list(modalities_used_in_model1)
    list2 = list(modalities_used_in_model2)
    if list1 == list2:
        print(f"   ✅ modalities_used_in_model顺序一致")
    else:
        print(f"   ❌ modalities_used_in_model顺序不一致！")
        print(f"      模型1: {list1}")
        print(f"      模型2: {list2}")
    
    # 检查features_dict的构建顺序
    print(f"\n🔍 检查 features_dict 的构建顺序:")
    
    seed_torch(seed + 10000)
    features_dict1 = {}
    for channel in modalities_used_in_model1:
        if channel == 'wsi=features':
            features_dict1[channel] = torch.randn(1, 128)  # 模拟
        elif channel == 'tma=features':
            features_dict1[channel] = torch.randn(1, 128)  # 模拟
        else:
            features_dict1[channel] = torch.randn(1, 128)  # 模拟
    
    seed_torch(seed + 10000)
    features_dict2 = {}
    for channel in modalities_used_in_model2:
        if channel == 'wsi=features':
            features_dict2[channel] = torch.randn(1, 128)  # 模拟
        elif channel == 'tma=features':
            features_dict2[channel] = torch.randn(1, 128)  # 模拟
        else:
            features_dict2[channel] = torch.randn(1, 128)  # 模拟
    
    print(f"   模型1 - features_dict keys: {list(features_dict1.keys())}")
    print(f"   模型2 - features_dict keys: {list(features_dict2.keys())}")
    
    if list(features_dict1.keys()) == list(features_dict2.keys()):
        print(f"   ✅ features_dict keys顺序一致")
    else:
        print(f"   ❌ features_dict keys顺序不一致！")
        print(f"      模型1: {list(features_dict1.keys())}")
        print(f"      模型2: {list(features_dict2.keys())}")
    
    # 检查random.sample和random.randint的使用
    print(f"\n🔍 检查 random.sample 和 random.randint 的使用:")
    
    import random
    
    seed_torch(seed + 10000)
    keys1 = list(features_dict1.keys())
    n1 = random.randint(1, len(keys1) - 1)
    drop1 = random.sample(keys1, n1)
    print(f"   模型1 - features_dict keys: {keys1}")
    print(f"   模型1 - random.randint(1, {len(keys1)-1}): {n1}")
    print(f"   模型1 - random.sample结果: {drop1}")
    
    seed_torch(seed + 10000)
    keys2 = list(features_dict2.keys())
    n2 = random.randint(1, len(keys2) - 1)
    drop2 = random.sample(keys2, n2)
    print(f"   模型2 - features_dict keys: {keys2}")
    print(f"   模型2 - random.randint(1, {len(keys2)-1}): {n2}")
    print(f"   模型2 - random.sample结果: {drop2}")
    
    if keys1 == keys2 and n1 == n2 and set(drop1) == set(drop2):
        print(f"   ✅ random操作结果一致")
    else:
        print(f"   ❌ random操作结果不一致！")
        if keys1 != keys2:
            print(f"      features_dict keys顺序不同: {keys1} vs {keys2}")
        if n1 != n2:
            print(f"      random.randint结果不同: {n1} vs {n2}")
        if set(drop1) != set(drop2):
            print(f"      random.sample结果不同: {drop1} vs {drop2}")
    
    # 检查for循环的顺序
    print(f"\n🔍 检查 for modality in features_dict.keys() 的顺序:")
    
    seed_torch(seed + 10000)
    order1 = []
    for modality in features_dict1.keys():
        order1.append(modality)
    print(f"   模型1 - for循环顺序: {order1}")
    
    seed_torch(seed + 10000)
    order2 = []
    for modality in features_dict2.keys():
        order2.append(modality)
    print(f"   模型2 - for循环顺序: {order2}")
    
    if order1 == order2:
        print(f"   ✅ for循环顺序一致")
    else:
        print(f"   ❌ for循环顺序不一致！")
        print(f"      模型1: {order1}")
        print(f"      模型2: {order2}")
    
    # 实际运行forward检查
    print(f"\n🔍 实际运行forward检查:")
    
    # 重置模型状态
    if hasattr(model1, 'alignment_features'):
        model1.alignment_features = []
    if hasattr(model2, 'alignment_features'):
        model2.alignment_features = []
    
    # 实际运行forward
    seed_torch(seed + 10000)
    model1.eval()
    with torch.no_grad():
        results1 = model1(data.copy(), label)
    
    seed_torch(seed + 10000)
    model2.eval()
    with torch.no_grad():
        results2 = model2(data.copy(), label)
    
    # 检查alignment_features
    if hasattr(model1, 'alignment_features') and len(model1.alignment_features) > 0:
        print(f"   模型1 - alignment_features[0] keys: {list(model1.alignment_features[0].keys())}")
    if hasattr(model2, 'alignment_features') and len(model2.alignment_features) > 0:
        print(f"   模型2 - alignment_features[0] keys: {list(model2.alignment_features[0].keys())}")
    
    if (hasattr(model1, 'alignment_features') and len(model1.alignment_features) > 0 and
        hasattr(model2, 'alignment_features') and len(model2.alignment_features) > 0):
        keys1 = list(model1.alignment_features[0].keys())
        keys2 = list(model2.alignment_features[0].keys())
        if keys1 == keys2:
            print(f"   ✅ alignment_features keys顺序一致")
        else:
            print(f"   ❌ alignment_features keys顺序不一致！")
            print(f"      模型1: {keys1}")
            print(f"      模型2: {keys2}")
    
    # 检查align_forward中features.items()的顺序
    print(f"\n🔍 检查 align_forward 中 features.items() 的顺序:")
    
    # 创建测试用的features_dict
    test_features1 = {}
    test_features2 = {}
    for channel in sorted(modalities_used_in_model1):  # 使用sorted确保顺序一致
        test_features1[channel] = torch.randn(1, 128)
        test_features2[channel] = torch.randn(1, 128)
    
    seed_torch(seed + 20000)
    aligned1 = model1.align_forward(test_features1)
    keys1_aligned = list(aligned1.keys())
    print(f"   模型1 - align_forward后 keys: {keys1_aligned}")
    
    seed_torch(seed + 20000)
    aligned2 = model2.align_forward(test_features2)
    keys2_aligned = list(aligned2.keys())
    print(f"   模型2 - align_forward后 keys: {keys2_aligned}")
    
    if keys1_aligned == keys2_aligned:
        print(f"   ✅ align_forward后 keys顺序一致")
    else:
        print(f"   ❌ align_forward后 keys顺序不一致！")
        print(f"      模型1: {keys1_aligned}")
        print(f"      模型2: {keys2_aligned}")
    
    # 检查group_loss_fn中alignment_features的顺序
    print(f"\n🔍 检查 group_loss_fn 中 alignment_features 的顺序:")
    
    if hasattr(model1, 'group_loss_fn'):
        # 创建测试用的alignment_features
        model1.alignment_features = []
        model2.alignment_features = []
        
        for i in range(3):
            features_dict_batch1 = {}
            features_dict_batch2 = {}
            for channel in sorted(modalities_used_in_model1):  # 使用sorted确保顺序一致
                features_dict_batch1[channel] = torch.randn(1, 128)
                features_dict_batch2[channel] = torch.randn(1, 128)
            
            model1.alignment_features.append(features_dict_batch1)
            model2.alignment_features.append(features_dict_batch2)
        
        # 检查group_loss_fn中的keys顺序
        if len(model1.alignment_features) > 0:
            keys1_group = sorted(model1.alignment_features[0].keys())
            print(f"   模型1 - alignment_features[0] keys (sorted): {keys1_group}")
        
        if len(model2.alignment_features) > 0:
            keys2_group = sorted(model2.alignment_features[0].keys())
            print(f"   模型2 - alignment_features[0] keys (sorted): {keys2_group}")
        
        if keys1_group == keys2_group:
            print(f"   ✅ group_loss_fn中keys顺序一致（使用sorted）")
        else:
            print(f"   ❌ group_loss_fn中keys顺序不一致！")
            print(f"      模型1: {keys1_group}")
            print(f"      模型2: {keys2_group}")
    
    # 检查实际训练时的forward（多个batch）
    print(f"\n🔍 检查实际训练时的forward（训练模式，多个batch）:")
    
    # 重置模型状态
    if hasattr(model1, 'alignment_features'):
        model1.alignment_features = []
    if hasattr(model2, 'alignment_features'):
        model2.alignment_features = []
    
    # 运行多个batch，检查alignment_features的累积顺序
    for batch_idx in range(5):
        seed_torch(seed + 30000 + batch_idx)
        model1.train()
        results1_train = model1(data.copy(), label)
        
        seed_torch(seed + 30000 + batch_idx)
        model2.train()
        results2_train = model2(data.copy(), label)
        
        # 检查alignment_features中每个字典的keys顺序
        if (hasattr(model1, 'alignment_features') and len(model1.alignment_features) > batch_idx and
            hasattr(model2, 'alignment_features') and len(model2.alignment_features) > batch_idx):
            keys1_batch = list(model1.alignment_features[batch_idx].keys())
            keys2_batch = list(model2.alignment_features[batch_idx].keys())
            
            if keys1_batch != keys2_batch:
                print(f"   ❌ Batch {batch_idx+1} - alignment_features keys顺序不一致！")
                print(f"      模型1: {keys1_batch}")
                print(f"      模型2: {keys2_batch}")
            else:
                print(f"   ✅ Batch {batch_idx+1} - alignment_features keys顺序一致: {keys1_batch}")
    
    # 检查group_loss_fn中的keys顺序（使用sorted）
    if hasattr(model1, 'group_loss_fn') and hasattr(model1, 'alignment_features') and len(model1.alignment_features) > 0:
        keys1_sorted = sorted(model1.alignment_features[0].keys())
        keys2_sorted = sorted(model2.alignment_features[0].keys())
        print(f"\n   模型1 - sorted(alignment_features[0].keys()): {keys1_sorted}")
        print(f"   模型2 - sorted(alignment_features[0].keys()): {keys2_sorted}")
        
        if keys1_sorted == keys2_sorted:
            print(f"   ✅ group_loss_fn中sorted keys顺序一致")
        else:
            print(f"   ❌ group_loss_fn中sorted keys顺序不一致！")
    
    # 检查features_dict在forward过程中的实际顺序
    print(f"\n🔍 检查实际forward过程中features_dict的顺序（通过hook）:")
    
    # 使用hook捕获forward过程中的features_dict
    captured_features_dict1 = []
    captured_features_dict2 = []
    
    def capture_features_dict1(module, input, output):
        # 这里我们需要在forward中捕获，但hook可能不够
        pass
    
    # 实际运行forward，手动检查
    print(f"   手动检查forward过程中features_dict的构建顺序:")
    
    seed_torch(seed + 40000)
    input_data1, modalities_used_in_model1_forward = model1._process_input_data(data.copy())
    print(f"   模型1 - modalities_used_in_model顺序: {list(modalities_used_in_model1_forward)}")
    
    features_dict1_forward = {}
    for channel in modalities_used_in_model1_forward:
        if channel == 'wsi=features':
            features_dict1_forward[channel] = torch.randn(1, 128)
        elif channel == 'tma=features':
            features_dict1_forward[channel] = torch.randn(1, 128)
        else:
            features_dict1_forward[channel] = torch.randn(1, 128)
    
    print(f"   模型1 - features_dict构建顺序: {list(features_dict1_forward.keys())}")
    
    seed_torch(seed + 40000)
    input_data2, modalities_used_in_model2_forward = model2._process_input_data(data.copy())
    print(f"   模型2 - modalities_used_in_model顺序: {list(modalities_used_in_model2_forward)}")
    
    features_dict2_forward = {}
    for channel in modalities_used_in_model2_forward:
        if channel == 'wsi=features':
            features_dict2_forward[channel] = torch.randn(1, 128)
        elif channel == 'tma=features':
            features_dict2_forward[channel] = torch.randn(1, 128)
        else:
            features_dict2_forward[channel] = torch.randn(1, 128)
    
    print(f"   模型2 - features_dict构建顺序: {list(features_dict2_forward.keys())}")
    
    if list(features_dict1_forward.keys()) != list(features_dict2_forward.keys()):
        print(f"   ❌ features_dict构建顺序不一致！")
        print(f"      模型1: {list(features_dict1_forward.keys())}")
        print(f"      模型2: {list(features_dict2_forward.keys())}")
    else:
        print(f"   ✅ features_dict构建顺序一致")
    
    # 检查align_forward后features_dict的顺序
    if hasattr(model1, 'align_forward'):
        seed_torch(seed + 40000)
        aligned_features1 = model1.align_forward(features_dict1_forward)
        print(f"   模型1 - align_forward后 keys: {list(aligned_features1.keys())}")
        
        seed_torch(seed + 40000)
        aligned_features2 = model2.align_forward(features_dict2_forward)
        print(f"   模型2 - align_forward后 keys: {list(aligned_features2.keys())}")
        
        if list(aligned_features1.keys()) != list(aligned_features2.keys()):
            print(f"   ❌ align_forward后 keys顺序不一致！")
        else:
            print(f"   ✅ align_forward后 keys顺序一致")
    
    # 总结
    print(f"\n{'='*60}")
    print("诊断总结")
    print(f"{'='*60}")
    print(f"可能的问题:")
    print(f"1. modalities_used_in_model是set，遍历顺序不确定")
    print(f"2. features_dict.keys()的顺序可能不同（如果构建顺序不同）")
    print(f"3. random.sample(list(features_dict.keys()), ...)的顺序依赖keys的顺序")
    print(f"4. for modality in features_dict.keys()的顺序可能不同")
    print(f"5. alignment_features中字典的keys顺序可能不同")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断训练不一致的原因')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果目录路径')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold索引 (default: 0)')
    
    args = parser.parse_args()
    
    try:
        diagnose_training_inconsistency(
            args.results_dir,
            args.fold_idx
        )
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

