#!/usr/bin/env python3
"""
测试训练过程一致性
验证在相同seed和相同数据的情况下，两个模型训练后是否会产生相同的权重
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

from trainer import Trainer, get_split_loader, Logger, get_optim
from main import seed_torch, create_k_fold_splits
from datasets.multimodal_dataset import MultimodalDataset
from utils.test_model_init_consistency import compare_model_weights


def test_training_consistency(results_dir: str, num_epochs: int = 10, fold_idx: int = 0):
    """
    测试训练过程一致性
    
    Args:
        results_dir: 结果目录路径
        num_epochs: 训练epoch数
        fold_idx: fold索引
    """
    results_dir = Path(results_dir)
    configs_file = results_dir / 'configs_svd_random_clam_detach.json'
    
    if not configs_file.exists():
        # 尝试查找其他配置文件
        config_files = list(results_dir.glob('configs_*.json'))
        if config_files:
            configs_file = config_files[0]
        else:
            raise FileNotFoundError(f"未找到配置文件: {results_dir}")
    
    # 加载配置
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    
    print(f"📋 加载配置文件: {configs_file}")
    print(f"📋 模型类型: {configs['model_config']['model_type']}")
    
    # 获取seed
    seed = configs['experiment_config'].get('seed', 5678)
    print(f"🌱 使用随机种子: {seed}")
    
    # 加载数据集
    experiment_config = configs['experiment_config']
    print(f"\n📦 加载数据集...")
    print(f"   data_root_dir: {experiment_config['data_root_dir']}")
    print(f"   csv_path: {experiment_config['csv_path']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultimodalDataset(
        csv_path=experiment_config['csv_path'],
        data_root_dir=experiment_config['data_root_dir'],
        channels=experiment_config['target_channels'],
        align_channels={},
        alignment_model_path=experiment_config['alignment_model_path'],
        device=device,
        print_info=True
    )
    
    print(f"✅ 数据集加载完成，共 {len(dataset)} 个样本")
    
    # 创建k-fold分割（使用相同的seed）
    print(f"\n📊 创建数据集分割...")
    splits = create_k_fold_splits(dataset, k=10, seed=seed, fixed_test_split=None)
    split = splits[fold_idx]
    
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    print(f"   Train samples: {len(train_idx)}")
    print(f"   Val samples: {len(val_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    
    # 创建子数据集
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # 创建训练器
    trainer = Trainer(
        configs=configs,
        log_dir=str(results_dir / 'training_logs')
    )
    
    # ========== 初始化第一个模型 ==========
    print(f"\n{'='*60}")
    print("初始化第一个模型")
    print(f"{'='*60}")
    
    seed_torch(seed)
    model1 = trainer._init_model()
    trainer.loss_fn = model1.loss_fn  # 设置损失函数
    optimizer1 = get_optim(model1, trainer.opt, trainer.lr, trainer.reg)
    
    print(f"✅ 模型1初始化完成")
    print(f"   总参数数量: {sum(p.numel() for p in model1.parameters()):,}")
    
    # ========== 初始化第二个模型 ==========
    print(f"\n{'='*60}")
    print("初始化第二个模型（相同seed）")
    print(f"{'='*60}")
    
    seed_torch(seed)
    model2 = trainer._init_model()
    # 注意：model2.loss_fn 应该和 model1.loss_fn 相同，但为了安全起见，我们也设置一下
    if trainer.loss_fn is None:
        trainer.loss_fn = model2.loss_fn
    optimizer2 = get_optim(model2, trainer.opt, trainer.lr, trainer.reg)
    
    print(f"✅ 模型2初始化完成")
    print(f"   总参数数量: {sum(p.numel() for p in model2.parameters()):,}")
    
    # 验证初始化时两个模型是否一致
    print(f"\n🔍 验证初始化时两个模型是否一致...")
    init_comparison = compare_model_weights(model1, model2, tolerance=1e-6)
    if init_comparison['different_elements'] == 0:
        print(f"✅ 初始化时两个模型完全一致")
    else:
        print(f"❌ 初始化时两个模型不一致！")
        print(f"   不同的元素: {init_comparison['different_elements']:,}")
        return
    
    # ========== 创建数据加载器（确保顺序一致） ==========
    print(f"\n{'='*60}")
    print("创建数据加载器（确保顺序一致）")
    print(f"{'='*60}")
    
    # ⚠️ 重要：使用固定seed的generator确保WeightedRandomSampler的采样顺序一致
    print(f"   创建两个数据加载器（使用固定seed的generator确保采样顺序一致）...")
    seed_torch(seed)
    
    # 创建两个固定seed的generator，确保WeightedRandomSampler的采样顺序一致
    generator1 = torch.Generator()
    generator1.manual_seed(seed)
    
    generator2 = torch.Generator()
    generator2.manual_seed(seed)
    
    # 创建两个独立的DataLoader，都使用相同的generator seed
    train_loader1 = get_split_loader(train_dataset, training=True, weighted=True, batch_size=1, generator=generator1)
    train_loader2 = get_split_loader(train_dataset, training=True, weighted=True, batch_size=1, generator=generator2)
    
    # 将DataLoader转换为列表，确保两个模型使用完全相同的数据序列
    print(f"   将DataLoader转换为列表，确保数据顺序一致...")
    train_loader1_list = list(train_loader1)
    train_loader2_list = list(train_loader2)
    
    print(f"   ✅ 数据加载器创建完成")
    print(f"   Train batches (model1): {len(train_loader1_list)}")
    print(f"   Train batches (model2): {len(train_loader2_list)}")
    print(f"   ⚠️ 两个模型使用相同seed的generator，确保采样顺序一致")
    
    # ========== 训练两个模型 ==========
    print(f"\n{'='*60}")
    print(f"训练两个模型（{num_epochs} 个epoch）")
    print(f"{'='*60}")
    
    logger1 = Logger(trainer.model_config['n_classes'])
    logger2 = Logger(trainer.model_config['n_classes'])
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # ========== 训练模型1（完整的一个epoch） ==========
        print(f"\n📊 训练模型1（完整的一个epoch）...")
        seed_torch(seed + epoch)  # 每个epoch使用不同的seed，但两个模型使用相同的seed
        model1.train()
        logger1.reset_epoch_stats()
        
        batch_size = trainer.experiment_config['batch_size']
        total_loss1 = 0
        
        # 重置模型的group相关状态，确保训练时从干净状态开始
        if hasattr(model1, 'alignment_features'):
            model1.alignment_features = []
        if hasattr(model1, 'group_logits'):
            model1.group_logits = []
        if hasattr(model1, 'group_labels'):
            model1.group_labels = []
        
        # 📊 初始化results列表，用于积累每个sample的结果（batch_size=1，所以每个batch就是一个sample）
        if not hasattr(model1, '_epoch_results'):
            model1._epoch_results = []
        model1._epoch_results = []  # 每个epoch开始时清空
        
        for batch_idx, (data, label) in enumerate(train_loader1_list):
            seed_torch(seed + epoch * 10000 + batch_idx)
            
            label = label.to(device)
            for channel in data:
                data[channel] = data[channel].to(device)
            
            results1 = model1(data, label)
            Y_prob1 = results1['probabilities']
            Y_hat1 = results1['predictions']
            
            # 📊 保存每个sample的完整results（包括所有中间步骤）
            # 注意：results中的tensor需要clone并移到CPU，避免后续训练时被修改
            sample_result = {
                'epoch': epoch + 1,
                'sample_idx': batch_idx + 1,  # 因为batch_size=1，所以sample_idx就是batch_idx+1
                'results': {}
            }
            
            # 保存results中的所有内容（转换为CPU tensor或保持原样）
            for key, value in results1.items():
                if isinstance(value, torch.Tensor):
                    sample_result['results'][key] = value.detach().clone().cpu()
                elif isinstance(value, dict):
                    sample_result['results'][key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            sample_result['results'][key][sub_key] = sub_value.detach().clone().cpu()
                        else:
                            sample_result['results'][key][sub_key] = sub_value
                elif isinstance(value, (list, tuple)):
                    sample_result['results'][key] = [
                        v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v
                        for v in value
                    ]
                else:
                    sample_result['results'][key] = value
            
            model1._epoch_results.append(sample_result)
            
            # 计算损失（完全按照trainer.py的逻辑）
            results1['labels'] = label
            loss1 = trainer.loss_fn(results1['logits'], results1['labels'], results1)
            total_loss1 += loss1
            
            # 记录指标
            logger1.log_batch(Y_hat1, label, Y_prob1, loss1)
            
            if (batch_idx + 1) % batch_size == 0:
                # 反向传播（完全按照trainer.py的逻辑）
                if hasattr(model1, 'group_loss_fn'):
                    results1['group_loss'] = model1.group_loss_fn(results1)
                    total_loss1 += results1['group_loss']
                total_loss1 = total_loss1 / batch_size
                results1['total_loss'] = total_loss1.item()
                total_loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                if hasattr(model1, 'verbose_items'):
                    items = model1.verbose_items(results1)
                    if len(items) > 0:
                        print(f'   模型1 - Batch {batch_idx + 1}/{len(train_loader1_list)}: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
                total_loss1 = 0
        
        if len(train_loader1_list) % batch_size != 0:
            # 计算剩余batch的数量（完全按照trainer.py的逻辑）
            remaining_batches = len(train_loader1_list) % batch_size
            # 反向传播
            if hasattr(model1, 'group_loss_fn'):
                results1['group_loss'] = model1.group_loss_fn(results1)
                total_loss1 += results1['group_loss']
            total_loss1 = total_loss1 / remaining_batches
            results1['total_loss'] = total_loss1.item()
            total_loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            if hasattr(model1, 'verbose_items'):
                items = model1.verbose_items(results1)
                if len(items) > 0:
                    print(f'   模型1 - Final batch: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
            total_loss1 = 0
        
        train_loss1 = logger1.batch_log['loss'] / len(train_loader1_list)
        train_acc1 = logger1.get_overall_accuracy()
        print(f"   模型1 - Epoch {epoch + 1}: Loss={train_loss1:.4f}, Acc={train_acc1:.4f}")
        
        # ========== 训练模型2（完整的一个epoch） ==========
        print(f"\n📊 训练模型2（完整的一个epoch）...")
        seed_torch(seed + epoch)  # 使用相同的seed
        model2.train()
        logger2.reset_epoch_stats()
        
        total_loss2 = 0
        
        # 重置模型的group相关状态，确保训练时从干净状态开始
        if hasattr(model2, 'alignment_features'):
            model2.alignment_features = []
        if hasattr(model2, 'group_logits'):
            model2.group_logits = []
        if hasattr(model2, 'group_labels'):
            model2.group_labels = []
        
        # 📊 初始化results列表，用于积累每个sample的结果（batch_size=1，所以每个batch就是一个sample）
        if not hasattr(model2, '_epoch_results'):
            model2._epoch_results = []
        model2._epoch_results = []  # 每个epoch开始时清空
        
        for batch_idx, (data, label) in enumerate(train_loader2_list):
            # 在每个sample开始前重置seed，确保随机操作的一致性（使用相同的seed）
            # 使用独立的seed确保每个batch的随机状态是独立的
            seed_torch(seed + epoch * 10000 + batch_idx)
            
            label = label.to(device)
            for channel in data:
                data[channel] = data[channel].to(device)
            
            results2 = model2(data, label)
            Y_prob2 = results2['probabilities']
            Y_hat2 = results2['predictions']
            
            # 📊 保存每个sample的完整results（包括所有中间步骤）
            # 注意：results中的tensor需要clone并移到CPU，避免后续训练时被修改
            sample_result = {
                'epoch': epoch + 1,
                'sample_idx': batch_idx + 1,  # 因为batch_size=1，所以sample_idx就是batch_idx+1
                'results': {}
            }
            
            # 保存results中的所有内容（转换为CPU tensor或保持原样）
            for key, value in results2.items():
                if isinstance(value, torch.Tensor):
                    sample_result['results'][key] = value.detach().clone().cpu()
                elif isinstance(value, dict):
                    sample_result['results'][key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            sample_result['results'][key][sub_key] = sub_value.detach().clone().cpu()
                        else:
                            sample_result['results'][key][sub_key] = sub_value
                elif isinstance(value, (list, tuple)):
                    sample_result['results'][key] = [
                        v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v
                        for v in value
                    ]
                else:
                    sample_result['results'][key] = value
            
            model2._epoch_results.append(sample_result)
            
            # 计算损失（完全按照trainer.py的逻辑）
            results2['labels'] = label
            loss2 = trainer.loss_fn(results2['logits'], results2['labels'], results2)
            total_loss2 += loss2
            
            # 记录指标
            logger2.log_batch(Y_hat2, label, Y_prob2, loss2)
            
            if (batch_idx + 1) % batch_size == 0:
                # 反向传播（完全按照trainer.py的逻辑）
                if hasattr(model2, 'group_loss_fn'):
                    results2['group_loss'] = model2.group_loss_fn(results2)
                    total_loss2 += results2['group_loss']
                total_loss2 = total_loss2 / batch_size
                results2['total_loss'] = total_loss2.item()
                total_loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()
                if hasattr(model2, 'verbose_items'):
                    items = model2.verbose_items(results2)
                    if len(items) > 0:
                        print(f'   模型2 - Batch {batch_idx + 1}/{len(train_loader2_list)}: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
                total_loss2 = 0
        
        if len(train_loader2_list) % batch_size != 0:
            # 计算剩余batch的数量（完全按照trainer.py的逻辑）
            remaining_batches = len(train_loader2_list) % batch_size
            # 反向传播
            if hasattr(model2, 'group_loss_fn'):
                results2['group_loss'] = model2.group_loss_fn(results2)
                total_loss2 += results2['group_loss']
            total_loss2 = total_loss2 / remaining_batches
            results2['total_loss'] = total_loss2.item()
            total_loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            if hasattr(model2, 'verbose_items'):
                items = model2.verbose_items(results2)
                if len(items) > 0:
                    print(f'   模型2 - Final batch: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
            total_loss2 = 0
        
        train_loss2 = logger2.batch_log['loss'] / len(train_loader2_list)
        train_acc2 = logger2.get_overall_accuracy()
        print(f"   模型2 - Epoch {epoch + 1}: Loss={train_loss2:.4f}, Acc={train_acc2:.4f}")
        
        # 📊 比较积累的results，找出第一个不一致的步骤
        print(f"\n🔍 比较积累的results，定位第一个不一致的步骤...")
        first_inconsistency = None
        
        if hasattr(model1, '_epoch_results') and hasattr(model2, '_epoch_results'):
            results1_list = model1._epoch_results
            results2_list = model2._epoch_results
            
            # 确保两个列表长度相同
            min_len = min(len(results1_list), len(results2_list))
            print(f"   比较 {min_len} 个sample的results...")
            
            for i in range(min_len):
                item1 = results1_list[i]
                item2 = results2_list[i]
                
                # 确保是同一个epoch和sample
                if item1['epoch'] != item2['epoch'] or item1['sample_idx'] != item2['sample_idx']:
                    print(f"   ⚠️ Sample {i+1}: epoch或sample_idx不匹配")
                    continue
                
                results1 = item1['results']
                results2 = item2['results']
                
                # 完整的results比较逻辑
                inconsistency_info = []
                
                # 遍历results1中的所有key
                for key in results1:
                    if key not in results2:
                        inconsistency_info.append(f"{key}")
                        inconsistency_info.append(f"{key} not in results2")
                        continue
                    
                    val1 = results1[key]
                    val2 = results2[key]
                    
                    # 1. 处理int类型
                    if isinstance(val1, int):
                        if val1 != val2:
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} value not equal: {val1} vs {val2}")
                    
                    # 2. 处理list/tuple类型
                    elif isinstance(val1, (list, tuple)):
                        if len(val1) != len(val2):
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} length not equal: {len(val1)} vs {len(val2)}")
                        else:
                            for idx in range(len(val1)):
                                if isinstance(val1[idx], torch.Tensor) and isinstance(val2[idx], torch.Tensor):
                                    if val1[idx].shape != val2[idx].shape:
                                        inconsistency_info.append(f"{key}[{idx}]")
                                        inconsistency_info.append(f"{key}[{idx}] shape not equal: {val1[idx].shape} vs {val2[idx].shape}")
                                    else:
                                        max_diff = torch.max(torch.abs(val1[idx] - val2[idx])).item()
                                        if max_diff > 1e-6:
                                            inconsistency_info.append(f"{key}[{idx}]")
                                            inconsistency_info.append(f"{key}[{idx}] value not equal: max_diff={max_diff:.2e}")
                                elif val1[idx] != val2[idx]:
                                    inconsistency_info.append(f"{key}[{idx}]")
                                    inconsistency_info.append(f"{key}[{idx}] value not equal: {val1[idx]} vs {val2[idx]}")
                    
                    # 3. 处理dict类型
                    elif isinstance(val1, dict):
                        keys1 = sorted(val1.keys())
                        keys2 = sorted(val2.keys())
                        if keys1 != keys2:
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} keys不一致: {keys1} vs {keys2}")
                        else:
                            for sub_key in keys1:
                                sub_val1 = val1[sub_key]
                                sub_val2 = val2[sub_key]
                                
                                if isinstance(sub_val1, torch.Tensor) and isinstance(sub_val2, torch.Tensor):
                                    if sub_val1.shape != sub_val2.shape:
                                        inconsistency_info.append(f"{key}[{sub_key}]")
                                        inconsistency_info.append(f"{key}[{sub_key}] shape not equal: {sub_val1.shape} vs {sub_val2.shape}")
                                    else:
                                        # 检查tensor的dtype，如果是整数类型，只比较是否相等
                                        if sub_val1.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                                            if not torch.equal(sub_val1, sub_val2):
                                                diff_count = (sub_val1 != sub_val2).sum().item()
                                                inconsistency_info.append(f"{key}[{sub_key}]")
                                                inconsistency_info.append(f"{key}[{sub_key}] 整数tensor不一致，{diff_count} 个元素不同")
                                        else:
                                            # 浮点数类型：计算max_diff
                                            max_diff = torch.max(torch.abs(sub_val1 - sub_val2)).item()
                                            if max_diff > 1e-6:
                                                inconsistency_info.append(f"{key}[{sub_key}]")
                                                inconsistency_info.append(f"{key}[{sub_key}] value not equal: max_diff={max_diff:.2e}")
                                elif sub_val1 != sub_val2:
                                    inconsistency_info.append(f"{key}[{sub_key}]")
                                    inconsistency_info.append(f"{key}[{sub_key}] value not equal: {sub_val1} vs {sub_val2}")
                    
                    # 4. 处理tensor类型
                    elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                        if val1.shape != val2.shape:
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} shape not equal: {val1.shape} vs {val2.shape}")
                        else:
                            # 检查tensor的dtype，如果是整数类型，只比较是否相等
                            if val1.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                                if not torch.equal(val1, val2):
                                    diff_count = (val1 != val2).sum().item()
                                    inconsistency_info.append(f"{key}")
                                    inconsistency_info.append(f"{key} 整数tensor不一致，{diff_count} 个元素不同")
                            else:
                                # 浮点数类型：计算max_diff
                                max_diff = torch.max(torch.abs(val1 - val2)).item()
                                if max_diff > 1e-6:
                                    inconsistency_info.append(f"{key}")
                                    inconsistency_info.append(f"{key} value not equal: max_diff={max_diff:.2e}")
                    
                    # 5. 处理np.ndarray类型
                    elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                        if val1.shape != val2.shape:
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} shape not equal: {val1.shape} vs {val2.shape}")
                        else:
                            max_diff = np.max(np.abs(val1 - val2))
                            if max_diff > 1e-6:
                                inconsistency_info.append(f"{key}")
                                inconsistency_info.append(f"{key} value not equal: max_diff={max_diff:.2e}")
                    
                    # 6. 处理其他类型
                    else:
                        print(f'{key} {type(val1)} {type(val2)} {val1} {val2}')
                        if val1 != val2:
                            inconsistency_info.append(f"{key}")
                            inconsistency_info.append(f"{key} value not equal: {val1} vs {val2}")
                
                # 检查results2中是否有results1中没有的key
                for key in results2:
                    if key not in results1:
                        inconsistency_info.append(f"{key}")
                        inconsistency_info.append(f"{key} not in results1")
                
                # 如果发现不一致，记录第一个
                if inconsistency_info and first_inconsistency is None:
                    first_inconsistency = {
                        'epoch': item1['epoch'],
                        'sample_idx': item1['sample_idx'],
                        'results1': results1,
                        'results2': results2,
                        'info': inconsistency_info
                    }
                    print(f"\n   ❌ 发现第一个不一致！")
                    print(f"      Epoch: {first_inconsistency['epoch']}")
                    print(f"      Sample: {first_inconsistency['sample_idx']}")
                    print(f"      不一致的步骤:")
                    for info in inconsistency_info:
                        print(f"        - {info}")
                    break
            
            # 显示第一个不一致的详细信息
            if first_inconsistency:
                print(f"\n   🔍 第一个不一致的详细信息:")
                print(f"      Epoch: {first_inconsistency['epoch']}, Sample: {first_inconsistency['sample_idx']}")
                results1 = first_inconsistency['results1']
                results2 = first_inconsistency['results2']
                
                # 详细比较每个key
                all_keys = sorted(set(results1.keys()) | set(results2.keys()))
                for key in all_keys:
                    if key not in results1:
                        print(f"      ❌ {key}: 只在results2中存在")
                        continue
                    if key not in results2:
                        print(f"      ❌ {key}: 只在results1中存在")
                        continue
                    
                    val1 = results1[key]
                    val2 = results2[key]
                    
                    if isinstance(val1, dict) and isinstance(val2, dict):
                        print(f"      {key}:")
                        sub_keys = sorted(set(val1.keys()) | set(val2.keys()))
                        for sub_key in sub_keys:
                            if sub_key in val1 and sub_key in val2:
                                if isinstance(val1[sub_key], torch.Tensor) and isinstance(val2[sub_key], torch.Tensor):
                                    if val1[sub_key].shape == val2[sub_key].shape:
                                        # 检查tensor的dtype，如果是整数类型，只比较是否相等
                                        if val1[sub_key].dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                                            if not torch.equal(val1[sub_key], val2[sub_key]):
                                                diff_count = (val1[sub_key] != val2[sub_key]).sum().item()
                                                total_count = val1[sub_key].numel()
                                                print(f"         ❌ [{sub_key}]: 整数tensor不一致，{diff_count}/{total_count} 个元素不同")
                                            else:
                                                print(f"         ✅ [{sub_key}]: 一致")
                                        else:
                                            # 浮点数类型：计算max_diff
                                            max_diff = torch.max(torch.abs(val1[sub_key] - val2[sub_key])).item()
                                            if max_diff > 1e-6:
                                                print(f"         ❌ [{sub_key}]: max_diff={max_diff:.2e}")
                                            else:
                                                print(f"         ✅ [{sub_key}]: 一致")
                                    else:
                                        print(f"         ❌ [{sub_key}]: 形状不一致 {val1[sub_key].shape} vs {val2[sub_key].shape}")
                                else:
                                    if val1[sub_key] == val2[sub_key]:
                                        print(f"         ✅ [{sub_key}]: 一致")
                                    else:
                                        print(f"         ❌ [{sub_key}]: 不一致 {val1[sub_key]} vs {val2[sub_key]}")
                            elif sub_key in val1:
                                print(f"         ❌ [{sub_key}]: 只在results1中存在")
                            else:
                                print(f"         ❌ [{sub_key}]: 只在results2中存在")
                    elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                        if val1.shape == val2.shape:
                            # 检查tensor的dtype，如果是整数类型，需要转换为float才能计算mean
                            if val1.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                                # 整数类型：只比较是否相等
                                if not torch.equal(val1, val2):
                                    # 计算不相等元素的数量
                                    diff_count = (val1 != val2).sum().item()
                                    total_count = val1.numel()
                                    print(f"      ❌ {key}: 整数tensor不一致，{diff_count}/{total_count} 个元素不同")
                                else:
                                    print(f"      ✅ {key}: 一致")
                            else:
                                # 浮点数类型：计算max_diff和mean_diff
                                max_diff = torch.max(torch.abs(val1 - val2)).item()
                                mean_diff = torch.mean(torch.abs(val1 - val2)).item()
                                if max_diff > 1e-6:
                                    print(f"      ❌ {key}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
                                else:
                                    print(f"      ✅ {key}: 一致")
                        else:
                            print(f"      ❌ {key}: 形状不一致 {val1.shape} vs {val2.shape}")
                    else:
                        if val1 == val2:
                            print(f"      ✅ {key}: 一致")
                        else:
                            print(f"      ❌ {key}: 不一致 {val1} vs {val2}")
                
                # 发现第一个不一致后立即退出
                print(f"\n{'='*60}")
                print(f"❌ 发现不一致，程序退出")
                print(f"{'='*60}")
                sys.exit(1)
            else:
                print(f"   ✅ 所有sample的results都一致")
        
        # 清空results列表，准备下一个epoch
        if hasattr(model1, '_epoch_results'):
            model1._epoch_results = []
        if hasattr(model2, '_epoch_results'):
            model2._epoch_results = []
        
        # 比较训练后的权重
        print(f"\n🔍 比较训练后的权重...")
        epoch_comparison = compare_model_weights(model1, model2, tolerance=1e-6)
        
        print(f"   📈 Epoch {epoch + 1} 比较结果:")
        print(f"      参数元素总数: {epoch_comparison['total_param_elements']:,}")
        print(f"      匹配的元素: {epoch_comparison['matching_elements']:,} ({100*epoch_comparison['matching_elements']/epoch_comparison['total_param_elements']:.2f}%)")
        print(f"      不同的元素: {epoch_comparison['different_elements']:,} ({100*epoch_comparison['different_elements']/epoch_comparison['total_param_elements']:.2f}%)")
        print(f"      全局最大绝对差: {epoch_comparison['global_max_abs_diff']:.6e}")
        print(f"      全局平均绝对差: {epoch_comparison['global_mean_abs_diff']:.6e}")
        
        if epoch_comparison['different_elements'] > 0:
            print(f"   ⚠️ Epoch {epoch + 1} 后两个模型不一致！")
            if epoch_comparison['different_keys']:
                print(f"      有差异的参数张量: {len(epoch_comparison['different_keys'])} 个")
                for diff_info in epoch_comparison['different_keys'][:5]:
                    if isinstance(diff_info, dict):
                        print(f"        - {diff_info['key']}: max_diff={diff_info['max_diff']:.6e}")
        else:
            print(f"   ✅ Epoch {epoch + 1} 后两个模型完全一致！")
    
    # ========== 最终比较 ==========
    print(f"\n{'='*60}")
    print("最终比较")
    print(f"{'='*60}")
    
    final_comparison = compare_model_weights(model1, model2, tolerance=1e-6)
    
    print(f"\n📊 最终比较结果:")
    print(f"   参数元素总数: {final_comparison['total_param_elements']:,}")
    print(f"   匹配的元素: {final_comparison['matching_elements']:,} ({100*final_comparison['matching_elements']/final_comparison['total_param_elements']:.2f}%)")
    print(f"   不同的元素: {final_comparison['different_elements']:,} ({100*final_comparison['different_elements']/final_comparison['total_param_elements']:.2f}%)")
    print(f"   完全匹配的张量: {final_comparison['matching_param_tensors']}")
    print(f"   有差异的张量: {final_comparison['different_param_tensors']}")
    print(f"   全局最大绝对差: {final_comparison['global_max_abs_diff']:.6e}")
    print(f"   全局平均绝对差: {final_comparison['global_mean_abs_diff']:.6e}")
    
    if final_comparison['different_keys']:
        print(f"\n   ⚠️ 有差异的参数张量 ({len(final_comparison['different_keys'])} 个):")
        for diff_info in final_comparison['different_keys'][:10]:
            if isinstance(diff_info, dict):
                print(f"     - {diff_info['key']}:")
                print(f"       shape: {diff_info['shape']}, numel: {diff_info['numel']:,}")
                print(f"       匹配元素: {diff_info['matching_elements']:,}, 不同元素: {diff_info['different_elements']:,}")
                print(f"       max_diff: {diff_info['max_diff']:.6e}, mean_diff: {diff_info['mean_diff']:.6e}")
    
    # 判断是否一致
    is_consistent = (
        final_comparison['different_elements'] == 0 and
        len(final_comparison['missing_keys_1']) == 0 and
        len(final_comparison['missing_keys_2']) == 0 and
        final_comparison['global_max_abs_diff'] < 1e-6
    )
    
    print(f"\n{'='*60}")
    if is_consistent:
        print("✅ 结论: 训练后两个模型权重完全一致！")
        print(f"   - 训练过程是确定性的")
        print(f"   - 所有 {final_comparison['total_param_elements']:,} 个参数元素完全匹配")
    else:
        print("❌ 结论: 训练后两个模型权重不一致！")
        print(f"   - 不同的元素: {final_comparison['different_elements']:,}")
        print(f"   - 可能的原因:")
        print(f"     1. 训练过程中使用了非确定性操作（如dropout在训练模式）")
        print(f"     2. CUDA操作的非确定性")
        print(f"     3. 浮点数运算的累积误差")
        print(f"     4. 某些随机操作未设置seed")
    print(f"{'='*60}")
    
    return {
        'init_comparison': init_comparison,
        'final_comparison': final_comparison,
        'is_consistent': is_consistent
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试训练过程一致性')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果目录路径')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练epoch数 (default: 10)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold索引 (default: 0)')
    
    args = parser.parse_args()
    
    try:
        results = test_training_consistency(
            args.results_dir,
            args.num_epochs,
            args.fold_idx
        )
        
        if not results['is_consistent']:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

