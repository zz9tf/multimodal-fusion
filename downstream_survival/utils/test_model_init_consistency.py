#!/usr/bin/env python3
"""
测试模型初始化一致性
验证在 seed_torch 的情况下，两次初始化模型是否会产生相同的权重
包括：初始化 -> 保存 -> 等待10秒 -> 加载 -> 再次初始化的完整流程
"""

import os
import sys
import torch
import numpy as np
import json
import time
from pathlib import Path

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

from trainer import Trainer
from main import seed_torch


def compare_model_weights(model1, model2, tolerance=1e-6):
    """
    比较两个模型的权重是否一致
    逐一比较所有参数元素（445k+个参数）
    
    Args:
        model1: 第一个模型
        model2: 第二个模型
        tolerance: 数值容差
        
    Returns:
        dict: 比较结果
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    results = {
        'total_param_tensors': 0,          # 参数张量个数（state_dict键数量）
        'total_param_elements': 0,         # 参数元素总数（标量个数，应该是445k+）
        'matching_elements': 0,            # 在容差内相等的元素个数
        'different_elements': 0,           # 超过容差的元素个数
        'matching_param_tensors': 0,       # 在容差内完全相等的参数张量个数
        'different_param_tensors': 0,      # 超过容差的参数张量个数
        'global_max_abs_diff': 0.0,        # 全局元素级最大绝对差
        'global_mean_abs_diff': 0.0,       # 全局元素级平均绝对差（按元素加权）
        'different_keys': [],              # 超出容差的键列表（带统计）
        'missing_keys_1': [],
        'missing_keys_2': []
    }
    
    all_keys = set(state_dict1.keys()) | set(state_dict2.keys())
    results['total_param_tensors'] = len(all_keys)
    
    # 全局元素级统计
    global_max = 0.0
    sum_abs_diff = 0.0
    total_elems = 0
    matching_elems = 0
    different_elems = 0
    
    print(f"   🔍 开始逐一比较所有参数元素...")
    
    for key in sorted(all_keys):  # 按字母顺序排序，便于查看
        if key not in state_dict1:
            results['missing_keys_1'].append(key)
            continue
        if key not in state_dict2:
            results['missing_keys_2'].append(key)
            continue
        
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        if param1.shape != param2.shape:
            results['different_keys'].append(f"{key}: shape mismatch ({param1.shape} vs {param2.shape})")
            results['different_param_tensors'] += 1
            continue
        
        # 计算差异
        diff = torch.abs(param1 - param2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # 元素级累积
        elem_count = diff.numel()
        total_elems += elem_count
        
        # 统计匹配和不匹配的元素数量
        matching_mask = diff <= tolerance
        matching_count = int(matching_mask.sum().item())
        different_count = elem_count - matching_count
        
        matching_elems += matching_count
        different_elems += different_count
        
        sum_abs_diff += float(diff.sum().item())
        if max_diff > global_max:
            global_max = max_diff
        
        # 如果该张量有任何元素超过容差，记录详细信息
        if max_diff > tolerance:
            results['different_param_tensors'] += 1
            results['different_keys'].append({
                'key': key,
                'shape': list(param1.shape),
                'numel': elem_count,
                'matching_elements': matching_count,
                'different_elements': different_count,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_location': None  # 可以添加最大差异的位置
            })
        else:
            results['matching_param_tensors'] += 1
    
    results['total_param_elements'] = int(total_elems)
    results['matching_elements'] = int(matching_elems)
    results['different_elements'] = int(different_elems)
    results['global_max_abs_diff'] = float(global_max)
    results['global_mean_abs_diff'] = float(sum_abs_diff / total_elems) if total_elems > 0 else 0.0
    
    print(f"   ✅ 比较完成: 共比较 {total_elems:,} 个参数元素")
    
    return results


def test_model_init_consistency(results_dir: str, fold_idx: int = 0, wait_seconds: int = 10):
    """
    测试模型初始化一致性（包括保存、加载、多次初始化）
    
    Args:
        results_dir: 结果目录路径
        fold_idx: fold索引
        wait_seconds: 保存后等待的秒数
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
    
    # 验证模型类型
    expected_model_type = 'svd_gate_random_clam_detach'
    actual_model_type = configs['model_config']['model_type']
    if actual_model_type != expected_model_type:
        print(f"⚠️ 警告: 模型类型不匹配！期望: {expected_model_type}, 实际: {actual_model_type}")
    else:
        print(f"✅ 模型类型验证通过: {actual_model_type}")
    
    # 打印关键配置信息
    print(f"\n📋 关键模型配置:")
    model_config = configs['model_config']
    print(f"   input_dim: {model_config.get('input_dim', 'N/A')}")
    print(f"   output_dim: {model_config.get('output_dim', 'N/A')}")
    print(f"   n_classes: {model_config.get('n_classes', 'N/A')}")
    print(f"   dropout: {model_config.get('dropout', 'N/A')}")
    print(f"   enable_svd: {model_config.get('enable_svd', 'N/A')}")
    print(f"   enable_random_loss: {model_config.get('enable_random_loss', 'N/A')}")
    print(f"   channels_used_in_model: {len(model_config.get('channels_used_in_model', []))} 个通道")
    
    # 获取seed
    seed = configs['experiment_config'].get('seed', 5678)
    print(f"\n🌱 使用随机种子: {seed}")
    
    # 创建训练器
    trainer = Trainer(
        configs=configs,
        log_dir=str(results_dir / 'training_logs')
    )
    
    # 验证训练器使用的配置
    print(f"✅ 训练器已创建，使用配置: {trainer.model_config['model_type']}")
    
    # ========== 第一次初始化模型 ==========
    print(f"\n{'='*60}")
    print("第一次初始化模型")
    print(f"{'='*60}")
    
    seed_torch(seed)
    model1 = trainer._init_model()
    state_dict1 = model1.state_dict()
    
    print(f"✅ 模型1初始化完成")
    print(f"   模型类型: {type(model1).__name__}")
    
    # 统计实际参数数量
    total_params = sum(p.numel() for p in model1.parameters())
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    state_dict_keys = len(state_dict1)
    
    print(f"   state_dict 键数量: {state_dict_keys}")
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数数量: {trainable_params:,}")
    print(f"   不可训练参数数量: {total_params - trainable_params:,}")
    
    # 验证模型是否是正确的类型
    from models.svd_gate_random_clam_detach import SVDGateRandomClamDetach
    if isinstance(model1, SVDGateRandomClamDetach):
        print(f"   ✅ 模型类型验证通过: SVDGateRandomClamDetach")
    else:
        print(f"   ⚠️ 警告: 模型类型不匹配！期望: SVDGateRandomClamDetach, 实际: {type(model1).__name__}")
    
    # 检查 transfer_layer 是否已创建
    if hasattr(model1, 'transfer_layer'):
        transfer_layer_count = len(model1.transfer_layer)
        print(f"   transfer_layer 数量: {transfer_layer_count}")
        if transfer_layer_count > 0:
            print(f"   transfer_layer 通道: {list(model1.transfer_layer.keys())}")
        else:
            print(f"   ⚠️ transfer_layer 尚未创建（将在 forward 时动态创建）")
    
    # 检查 alignment_layers 是否已创建
    if hasattr(model1, 'alignment_layers'):
        alignment_layers_count = len(model1.alignment_layers)
        print(f"   alignment_layers 数量: {alignment_layers_count}")
        if alignment_layers_count > 0:
            print(f"   alignment_layers 通道: {list(model1.alignment_layers.keys())}")
    
    # 检查 TCPClassifier 和 TCPConfidenceLayer 是否已创建
    if hasattr(model1, 'TCPClassifier'):
        tcp_classifier_count = len(model1.TCPClassifier)
        print(f"   TCPClassifier 数量: {tcp_classifier_count}")
    if hasattr(model1, 'TCPConfidenceLayer'):
        tcp_confidence_count = len(model1.TCPConfidenceLayer)
        print(f"   TCPConfidenceLayer 数量: {tcp_confidence_count}")
    
    # 列出所有参数键（按类别分组）
    print(f"\n   📋 参数键分类统计:")
    param_keys_by_type = {
        'attention_net': [],  # CLAM注意力网络
        'classifiers': [],  # CLAM分类器
        'instance_classifiers': [],  # CLAM实例分类器
        'transfer_layer': [],  # Transfer层
        'alignment_layers': [],  # SVD对齐层
        'TCPClassifier': [],  # 动态门控分类器
        'TCPConfidenceLayer': [],  # 动态门控置信度层
        'fusion_prediction': [],  # 融合预测层
        'other': []
    }
    
    for key in state_dict1.keys():
        if 'attention_net' in key:
            param_keys_by_type['attention_net'].append(key)
        elif 'classifiers' in key and 'instance' not in key:
            param_keys_by_type['classifiers'].append(key)
        elif 'instance_classifiers' in key:
            param_keys_by_type['instance_classifiers'].append(key)
        elif 'transfer_layer' in key:
            param_keys_by_type['transfer_layer'].append(key)
        elif 'alignment_layers' in key:
            param_keys_by_type['alignment_layers'].append(key)
        elif 'TCPClassifier' in key:
            param_keys_by_type['TCPClassifier'].append(key)
        elif 'TCPConfidenceLayer' in key:
            param_keys_by_type['TCPConfidenceLayer'].append(key)
        elif 'fusion_prediction' in key or 'fusion' in key.lower():
            param_keys_by_type['fusion_prediction'].append(key)
        else:
            param_keys_by_type['other'].append(key)
    
    # 计算每个类别的参数量
    for param_type, keys in param_keys_by_type.items():
        if keys:
            # 计算该类别的总参数量
            type_params = sum(state_dict1[key].numel() for key in keys)
            print(f"     {param_type}: {len(keys)} 个参数键, {type_params:,} 个参数")
            if len(keys) <= 5:
                for key in keys:
                    param_size = state_dict1[key].numel()
                    print(f"       - {key} ({param_size:,} 参数)")
            else:
                for key in keys[:3]:
                    param_size = state_dict1[key].numel()
                    print(f"       - {key} ({param_size:,} 参数)")
                print(f"       ... 还有 {len(keys) - 3} 个参数键")
    
    # 统计各层的参数量
    print(f"\n   📊 各层参数量统计:")
    layer_stats = {}
    
    # CLAM层参数
    clam_params = sum(state_dict1[key].numel() for key in param_keys_by_type['attention_net'] + 
                     param_keys_by_type['classifiers'] + param_keys_by_type['instance_classifiers'])
    if clam_params > 0:
        layer_stats['CLAM层'] = clam_params
        print(f"     CLAM层: {clam_params:,} 个参数")
        print(f"       - attention_net: {sum(state_dict1[key].numel() for key in param_keys_by_type['attention_net']):,}")
        print(f"       - classifiers: {sum(state_dict1[key].numel() for key in param_keys_by_type['classifiers']):,}")
        print(f"       - instance_classifiers: {sum(state_dict1[key].numel() for key in param_keys_by_type['instance_classifiers']):,}")
    
    # Transfer层参数
    transfer_params = sum(state_dict1[key].numel() for key in param_keys_by_type['transfer_layer'])
    if transfer_params > 0:
        layer_stats['Transfer层'] = transfer_params
        print(f"     Transfer层: {transfer_params:,} 个参数")
    
    # SVD对齐层参数
    svd_params = sum(state_dict1[key].numel() for key in param_keys_by_type['alignment_layers'])
    if svd_params > 0:
        layer_stats['SVD对齐层'] = svd_params
        print(f"     SVD对齐层: {svd_params:,} 个参数")
    
    # 动态门控层参数
    gate_params = sum(state_dict1[key].numel() for key in param_keys_by_type['TCPClassifier'] + 
                     param_keys_by_type['TCPConfidenceLayer'])
    if gate_params > 0:
        layer_stats['动态门控层'] = gate_params
        print(f"     动态门控层: {gate_params:,} 个参数")
        print(f"       - TCPClassifier: {sum(state_dict1[key].numel() for key in param_keys_by_type['TCPClassifier']):,}")
        print(f"       - TCPConfidenceLayer: {sum(state_dict1[key].numel() for key in param_keys_by_type['TCPConfidenceLayer']):,}")
    
    # 融合预测层参数
    fusion_params = sum(state_dict1[key].numel() for key in param_keys_by_type['fusion_prediction'])
    if fusion_params > 0:
        layer_stats['融合预测层'] = fusion_params
        print(f"     融合预测层: {fusion_params:,} 个参数")
    
    # 其他参数
    other_params = sum(state_dict1[key].numel() for key in param_keys_by_type['other'])
    if other_params > 0:
        layer_stats['其他'] = other_params
        print(f"     其他: {other_params:,} 个参数")
        for key in param_keys_by_type['other'][:5]:
            print(f"       - {key}")
    
    # 验证总参数量
    calculated_total = sum(layer_stats.values())
    print(f"\n     ✅ 计算的总参数量: {calculated_total:,}")
    print(f"     ✅ 实际的总参数量: {total_params:,}")
    if calculated_total == total_params:
        print(f"     ✅ 参数量统计一致")
    else:
        print(f"     ⚠️ 参数量统计不一致，差异: {abs(calculated_total - total_params):,}")
    
    # 获取第一个参数的统计信息作为示例
    first_key = list(state_dict1.keys())[0]
    first_param = state_dict1[first_key]
    print(f"\n   示例参数 ({first_key}):")
    print(f"     shape: {first_param.shape}")
    print(f"     mean: {first_param.mean().item():.6f}")
    print(f"     std: {first_param.std().item():.6f}")
    print(f"     min: {first_param.min().item():.6f}")
    print(f"     max: {first_param.max().item():.6f}")
    
    # ========== 保存模型 ==========
    print(f"\n{'='*60}")
    print("保存模型")
    print(f"{'='*60}")
    
    checkpoint_path = results_dir / f"test_init_consistency_checkpoint.pt"
    torch.save(model1.state_dict(), checkpoint_path)
    print(f"✅ 模型已保存到: {checkpoint_path}")
    
    # ========== 等待指定秒数 ==========
    print(f"\n{'='*60}")
    print(f"等待 {wait_seconds} 秒...")
    print(f"{'='*60}")
    
    for i in range(wait_seconds, 0, -1):
        print(f"   倒计时: {i} 秒", end='\r')
        time.sleep(1)
    print(f"   ✅ 等待完成")
    
    # ========== 加载模型 ==========
    print(f"\n{'='*60}")
    print("加载模型")
    print(f"{'='*60}")
    
    model2_loaded = trainer._init_model()
    state_dict_loaded = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 处理动态创建的transfer_layer（如果需要）
    if hasattr(model2_loaded, 'transfer_layer') and hasattr(model2_loaded, 'create_transfer_layer'):
        transfer_layer_channels = {}
        for key in state_dict_loaded.keys():
            if 'transfer_layer.' in key:
                parts = key.split('.')
                if len(parts) >= 3:
                    channel_name = parts[1]
                    weight_type = parts[2]
                    
                    if channel_name not in transfer_layer_channels:
                        transfer_layer_channels[channel_name] = {}
                    transfer_layer_channels[channel_name][weight_type] = state_dict_loaded[key]
        
        if hasattr(model2_loaded, 'output_dim'):
            output_dim = model2_loaded.output_dim
            for channel_name, weights in transfer_layer_channels.items():
                if channel_name not in model2_loaded.transfer_layer:
                    if 'weight' in weights:
                        weight_tensor = weights['weight']
                        if len(weight_tensor.shape) == 2:
                            input_dim = weight_tensor.shape[1]
                            transfer_layer = model2_loaded.create_transfer_layer(input_dim)
                            model2_loaded.transfer_layer[channel_name] = transfer_layer
    
    model2_loaded.load_state_dict(state_dict_loaded, strict=False)
    state_dict2_loaded = model2_loaded.state_dict()
    
    # 统计实际参数数量
    total_params_2 = sum(p.numel() for p in model2_loaded.parameters())
    trainable_params_2 = sum(p.numel() for p in model2_loaded.parameters() if p.requires_grad)
    state_dict_keys_2 = len(state_dict2_loaded)
    
    print(f"✅ 模型2（加载）初始化完成")
    print(f"   state_dict 键数量: {state_dict_keys_2}")
    print(f"   总参数数量: {total_params_2:,}")
    print(f"   可训练参数数量: {trainable_params_2:,}")
    
    # 获取第一个参数的统计信息作为示例
    first_key2_loaded = list(state_dict2_loaded.keys())[0]
    first_param2_loaded = state_dict2_loaded[first_key2_loaded]
    print(f"   示例参数 ({first_key2_loaded}):")
    print(f"     shape: {first_param2_loaded.shape}")
    print(f"     mean: {first_param2_loaded.mean().item():.6f}")
    print(f"     std: {first_param2_loaded.std().item():.6f}")
    print(f"     min: {first_param2_loaded.min().item():.6f}")
    print(f"     max: {first_param2_loaded.max().item():.6f}")
    
    # ========== 第三次初始化模型（相同seed） ==========
    print(f"\n{'='*60}")
    print("第三次初始化模型（相同seed）")
    print(f"{'='*60}")
    
    seed_torch(seed)
    model3 = trainer._init_model()
    state_dict3 = model3.state_dict()
    
    # 统计实际参数数量
    total_params_3 = sum(p.numel() for p in model3.parameters())
    trainable_params_3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    state_dict_keys_3 = len(state_dict3)
    
    print(f"✅ 模型3初始化完成")
    print(f"   state_dict 键数量: {state_dict_keys_3}")
    print(f"   总参数数量: {total_params_3:,}")
    print(f"   可训练参数数量: {trainable_params_3:,}")
    
    # 获取第一个参数的统计信息作为示例
    first_key3 = list(state_dict3.keys())[0]
    first_param3 = state_dict3[first_key3]
    print(f"   示例参数 ({first_key3}):")
    print(f"     shape: {first_param3.shape}")
    print(f"     mean: {first_param3.mean().item():.6f}")
    print(f"     std: {first_param3.std().item():.6f}")
    print(f"     min: {first_param3.min().item():.6f}")
    print(f"     max: {first_param3.max().item():.6f}")
    
    # ========== 比较三个模型的权重 ==========
    print(f"\n{'='*60}")
    print("比较三个模型的权重")
    print(f"{'='*60}")
    
    # 比较 模型1 vs 模型2（加载）
    print(f"\n📊 比较 模型1 vs 模型2（加载）:")
    comparison_1_2 = compare_model_weights(model1, model2_loaded, tolerance=1e-6)
    print(f"\n   📈 比较结果统计:")
    print(f"   参数张量个数: {comparison_1_2['total_param_tensors']}")
    print(f"   参数元素总数: {comparison_1_2['total_param_elements']:,}")
    print(f"   匹配的元素: {comparison_1_2['matching_elements']:,} ({100*comparison_1_2['matching_elements']/comparison_1_2['total_param_elements']:.2f}%)")
    print(f"   不同的元素: {comparison_1_2['different_elements']:,} ({100*comparison_1_2['different_elements']/comparison_1_2['total_param_elements']:.2f}%)")
    print(f"   完全匹配的张量: {comparison_1_2['matching_param_tensors']}")
    print(f"   有差异的张量: {comparison_1_2['different_param_tensors']}")
    print(f"   全局最大绝对差: {comparison_1_2['global_max_abs_diff']:.6e}")
    print(f"   全局平均绝对差: {comparison_1_2['global_mean_abs_diff']:.6e}")
    
    if comparison_1_2['different_keys']:
        print(f"\n   ⚠️ 有差异的参数张量 ({len(comparison_1_2['different_keys'])} 个):")
        for diff_info in comparison_1_2['different_keys'][:10]:  # 只显示前10个
            if isinstance(diff_info, dict):
                print(f"     - {diff_info['key']}:")
                print(f"       shape: {diff_info['shape']}, numel: {diff_info['numel']:,}")
                print(f"       匹配元素: {diff_info['matching_elements']:,}, 不同元素: {diff_info['different_elements']:,}")
                print(f"       max_diff: {diff_info['max_diff']:.6e}, mean_diff: {diff_info['mean_diff']:.6e}")
            else:
                print(f"     - {diff_info}")
        if len(comparison_1_2['different_keys']) > 10:
            print(f"     ... 还有 {len(comparison_1_2['different_keys']) - 10} 个有差异的张量")
    
    # 比较 模型1 vs 模型3（第三次初始化）
    print(f"\n📊 比较 模型1 vs 模型3（第三次初始化）:")
    comparison_1_3 = compare_model_weights(model1, model3, tolerance=1e-6)
    print(f"\n   📈 比较结果统计:")
    print(f"   参数张量个数: {comparison_1_3['total_param_tensors']}")
    print(f"   参数元素总数: {comparison_1_3['total_param_elements']:,}")
    print(f"   匹配的元素: {comparison_1_3['matching_elements']:,} ({100*comparison_1_3['matching_elements']/comparison_1_3['total_param_elements']:.2f}%)")
    print(f"   不同的元素: {comparison_1_3['different_elements']:,} ({100*comparison_1_3['different_elements']/comparison_1_3['total_param_elements']:.2f}%)")
    print(f"   完全匹配的张量: {comparison_1_3['matching_param_tensors']}")
    print(f"   有差异的张量: {comparison_1_3['different_param_tensors']}")
    print(f"   全局最大绝对差: {comparison_1_3['global_max_abs_diff']:.6e}")
    print(f"   全局平均绝对差: {comparison_1_3['global_mean_abs_diff']:.6e}")
    
    if comparison_1_3['different_keys']:
        print(f"\n   ⚠️ 有差异的参数张量 ({len(comparison_1_3['different_keys'])} 个):")
        for diff_info in comparison_1_3['different_keys'][:10]:
            if isinstance(diff_info, dict):
                print(f"     - {diff_info['key']}:")
                print(f"       shape: {diff_info['shape']}, numel: {diff_info['numel']:,}")
                print(f"       匹配元素: {diff_info['matching_elements']:,}, 不同元素: {diff_info['different_elements']:,}")
                print(f"       max_diff: {diff_info['max_diff']:.6e}, mean_diff: {diff_info['mean_diff']:.6e}")
            else:
                print(f"     - {diff_info}")
        if len(comparison_1_3['different_keys']) > 10:
            print(f"     ... 还有 {len(comparison_1_3['different_keys']) - 10} 个有差异的张量")
    
    # 比较 模型2（加载）vs 模型3（第三次初始化）
    print(f"\n📊 比较 模型2（加载）vs 模型3（第三次初始化）:")
    comparison_2_3 = compare_model_weights(model2_loaded, model3, tolerance=1e-6)
    print(f"\n   📈 比较结果统计:")
    print(f"   参数张量个数: {comparison_2_3['total_param_tensors']}")
    print(f"   参数元素总数: {comparison_2_3['total_param_elements']:,}")
    print(f"   匹配的元素: {comparison_2_3['matching_elements']:,} ({100*comparison_2_3['matching_elements']/comparison_2_3['total_param_elements']:.2f}%)")
    print(f"   不同的元素: {comparison_2_3['different_elements']:,} ({100*comparison_2_3['different_elements']/comparison_2_3['total_param_elements']:.2f}%)")
    print(f"   完全匹配的张量: {comparison_2_3['matching_param_tensors']}")
    print(f"   有差异的张量: {comparison_2_3['different_param_tensors']}")
    print(f"   全局最大绝对差: {comparison_2_3['global_max_abs_diff']:.6e}")
    print(f"   全局平均绝对差: {comparison_2_3['global_mean_abs_diff']:.6e}")
    
    if comparison_2_3['different_keys']:
        print(f"\n   ⚠️ 有差异的参数张量 ({len(comparison_2_3['different_keys'])} 个):")
        for diff_info in comparison_2_3['different_keys'][:10]:
            if isinstance(diff_info, dict):
                print(f"     - {diff_info['key']}:")
                print(f"       shape: {diff_info['shape']}, numel: {diff_info['numel']:,}")
                print(f"       匹配元素: {diff_info['matching_elements']:,}, 不同元素: {diff_info['different_elements']:,}")
                print(f"       max_diff: {diff_info['max_diff']:.6e}, mean_diff: {diff_info['mean_diff']:.6e}")
            else:
                print(f"     - {diff_info}")
        if len(comparison_2_3['different_keys']) > 10:
            print(f"     ... 还有 {len(comparison_2_3['different_keys']) - 10} 个有差异的张量")
    
    # 判断是否一致（基于元素级比较）
    is_consistent_1_2 = (
        comparison_1_2['different_elements'] == 0 and
        len(comparison_1_2['missing_keys_1']) == 0 and
        len(comparison_1_2['missing_keys_2']) == 0 and
        comparison_1_2['global_max_abs_diff'] < 1e-6
    )
    
    is_consistent_1_3 = (
        comparison_1_3['different_elements'] == 0 and
        len(comparison_1_3['missing_keys_1']) == 0 and
        len(comparison_1_3['missing_keys_2']) == 0 and
        comparison_1_3['global_max_abs_diff'] < 1e-6
    )
    
    is_consistent_2_3 = (
        comparison_2_3['different_elements'] == 0 and
        len(comparison_2_3['missing_keys_1']) == 0 and
        len(comparison_2_3['missing_keys_2']) == 0 and
        comparison_2_3['global_max_abs_diff'] < 1e-6
    )
    
    all_consistent = is_consistent_1_2 and is_consistent_1_3 and is_consistent_2_3
    
    print(f"\n{'='*60}")
    print("最终结论")
    print(f"{'='*60}")
    print(f"模型1 vs 模型2（加载）: {'✅ 一致' if is_consistent_1_2 else '❌ 不一致'}")
    print(f"模型1 vs 模型3（第三次初始化）: {'✅ 一致' if is_consistent_1_3 else '❌ 不一致'}")
    print(f"模型2（加载）vs 模型3（第三次初始化）: {'✅ 一致' if is_consistent_2_3 else '❌ 不一致'}")
    
    if all_consistent:
        print(f"\n✅ 结论: 三次模型权重完全一致！")
        print(f"   - 初始化 -> 保存 -> 加载: 一致")
        print(f"   - 初始化 -> 再次初始化: 一致")
        print(f"   - 加载 -> 再次初始化: 一致")
    else:
        print(f"\n❌ 结论: 三次模型权重不一致！")
        if not is_consistent_1_2:
            print(f"   ⚠️ 保存/加载过程可能引入了差异")
        if not is_consistent_1_3:
            print(f"   ⚠️ 多次初始化可能产生了不同的权重")
        if not is_consistent_2_3:
            print(f"   ⚠️ 加载的模型与重新初始化的模型不一致")
        print(f"\n   可能的原因:")
        print(f"   1. 模型初始化过程中使用了非确定性操作")
        print(f"   2. 某些层使用了随机初始化但未设置seed")
        print(f"   3. 动态创建的层（如transfer_layer）可能在不同时机创建")
        print(f"   4. 保存/加载过程中可能丢失了某些状态")
    print(f"{'='*60}")
    
    # 清理临时文件
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"\n🧹 已清理临时文件: {checkpoint_path}")
    
    return {
        'comparison_1_2': comparison_1_2,
        'comparison_1_3': comparison_1_3,
        'comparison_2_3': comparison_2_3,
        'is_consistent_1_2': is_consistent_1_2,
        'is_consistent_1_3': is_consistent_1_3,
        'is_consistent_2_3': is_consistent_2_3,
        'all_consistent': all_consistent
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试模型初始化一致性（包括保存、加载、多次初始化）')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果目录路径')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold索引 (default: 0)')
    parser.add_argument('--wait_seconds', type=int, default=10,
                        help='保存后等待的秒数 (default: 10)')
    
    args = parser.parse_args()
    
    try:
        results = test_model_init_consistency(
            args.results_dir,
            args.fold_idx,
            args.wait_seconds
        )
        
        if not results['all_consistent']:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

