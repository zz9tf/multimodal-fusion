#!/usr/bin/env python3
"""
真实场景测试：即使channels_used_in_model相同，遍历字典顺序可能不一致
重点：模拟实际代码中的逻辑
"""
import random
import os

def test_real_scenario():
    """模拟真实场景中的问题"""
    
    print("=" * 70)
    print("真实场景：即使channels_used_in_model相同，遍历字典顺序可能不一致")
    print("=" * 70)
    
    # 模拟_process_input_data的逻辑
    def process_input_data(channels_used_in_model):
        """模拟_process_input_data的逻辑"""
        modalities_used_in_model = set()
        for channel in channels_used_in_model:
            if channel.startswith('wsi='):
                modalities_used_in_model.add('wsi=features')
            if channel.startswith('tma='):
                modalities_used_in_model.add('tma=features')
            elif channel.endswith('=mask'):
                continue
            else:
                modalities_used_in_model.add(channel)
        return modalities_used_in_model
    
    # 模拟forward中的逻辑
    def forward_logic(modalities_used_in_model):
        """模拟forward中构建features_dict的逻辑"""
        features_dict = {}
        # 🔴 关键问题：遍历set，顺序可能不确定
        for channel in modalities_used_in_model:
            # 模拟添加特征
            features_dict[channel] = f"feature_{channel}"
        return features_dict
    
    # 场景1: 相同的channels_used_in_model，但可能在不同运行中顺序不同
    channels = ['CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1', 'wsi=features', 'tma=features']
    
    print(f"输入channels_used_in_model: {channels}")
    print()
    
    # 测试：多次运行，观察顺序是否稳定
    print("=" * 70)
    print("测试1: 多次运行，观察set和dict的遍历顺序")
    print("=" * 70)
    
    set_orders = []
    dict_orders = []
    
    for i in range(5):
        # 处理输入
        modalities = process_input_data(channels)
        set_order = list(modalities)
        set_orders.append(set_order)
        
        # 构建features_dict
        features_dict = forward_logic(modalities)
        dict_order = list(features_dict.keys())
        dict_orders.append(dict_order)
        
        print(f"运行{i+1}:")
        print(f"  set遍历顺序: {set_order}")
        print(f"  dict.keys()顺序: {dict_order}")
        print()
    
    # 检查是否相同
    set_all_same = all(order == set_orders[0] for order in set_orders)
    dict_all_same = all(order == dict_orders[0] for order in dict_orders)
    
    print(f"所有运行set遍历顺序相同: {set_all_same}")
    print(f"所有运行dict.keys()顺序相同: {dict_all_same}")
    print()
    
    # 测试2: 模拟不同运行之间的差异（比如hash seed不同）
    print("=" * 70)
    print("测试2: 模拟不同运行之间的差异（PYTHONHASHSEED）")
    print("=" * 70)
    
    # 保存原始hash seed
    original_hashseed = os.environ.get('PYTHONHASHSEED')
    
    hashseed_orders = []
    for hashseed in [None, '0', '1', '2']:
        if hashseed is None:
            if 'PYTHONHASHSEED' in os.environ:
                del os.environ['PYTHONHASHSEED']
        else:
            os.environ['PYTHONHASHSEED'] = hashseed
        
        # 重新创建set（模拟新进程）
        modalities = process_input_data(channels)
        set_order = list(modalities)
        hashseed_orders.append(set_order)
        
        print(f"PYTHONHASHSEED={hashseed}: {set_order}")
    
    # 恢复原始hash seed
    if original_hashseed is None:
        if 'PYTHONHASHSEED' in os.environ:
            del os.environ['PYTHONHASHSEED']
    else:
        os.environ['PYTHONHASHSEED'] = original_hashseed
    
    hashseed_all_same = all(order == hashseed_orders[0] for order in hashseed_orders)
    print(f"不同PYTHONHASHSEED下顺序相同: {hashseed_all_same}")
    print()
    
    # 测试3: 对random.sample的影响
    print("=" * 70)
    print("测试3: 对random.sample的影响（即使seed相同）")
    print("=" * 70)
    
    # 如果顺序不同，random.sample的结果也会不同
    if not dict_all_same:
        print("⚠️  警告：dict.keys()顺序不同，即使seed相同，random.sample结果也会不同！")
        random.seed(42)
        sample1 = random.sample(dict_orders[0], min(3, len(dict_orders[0])))
        random.seed(42)
        sample2 = random.sample(dict_orders[1], min(3, len(dict_orders[1])))
        print(f"运行1的random.sample: {sample1}")
        print(f"运行2的random.sample: {sample2}")
        print(f"结果相同: {sample1 == sample2}")
    else:
        print("✓ dict.keys()顺序相同，random.sample结果相同")
        random.seed(42)
        sample1 = random.sample(dict_orders[0], min(3, len(dict_orders[0])))
        random.seed(42)
        sample2 = random.sample(dict_orders[1], min(3, len(dict_orders[1])))
        print(f"运行1的random.sample: {sample1}")
        print(f"运行2的random.sample: {sample2}")
        print(f"结果相同: {sample1 == sample2}")
    print()
    
    # 测试4: 对torch.cat的影响
    print("=" * 70)
    print("测试4: 对特征拼接的影响")
    print("=" * 70)
    
    # 模拟torch.cat拼接
    def concat_features(features_dict):
        """模拟torch.cat拼接特征"""
        # 方式1: 使用values() - 顺序依赖于插入顺序
        values = list(features_dict.values())
        # 方式2: 使用keys()遍历 - 顺序依赖于插入顺序
        keys_order = [features_dict[k] for k in features_dict.keys()]
        return values, keys_order
    
    concat_results = []
    for i, dict_order in enumerate(dict_orders):
        # 重建features_dict
        features_dict = {k: f"feature_{k}" for k in dict_order}
        values, keys_order = concat_features(features_dict)
        concat_results.append((values, keys_order))
        print(f"运行{i+1} - 拼接顺序: {dict_order}")
    
    # 检查拼接顺序是否相同
    concat_all_same = all(
        r1[0] == r2[0] and r1[1] == r2[1]
        for r1, r2 in zip(concat_results, concat_results[1:])
    )
    print(f"所有运行拼接顺序相同: {concat_all_same}")
    if not concat_all_same:
        print("⚠️  警告：拼接顺序不同会导致特征顺序不同！")
    print()
    
    # 解决方案
    print("=" * 70)
    print("解决方案：使用sorted确保顺序一致")
    print("=" * 70)
    
    def forward_logic_sorted(modalities_used_in_model):
        """使用sorted确保顺序一致"""
        features_dict = {}
        # 🔧 关键修复：对set进行排序
        for channel in sorted(modalities_used_in_model):
            features_dict[channel] = f"feature_{channel}"
        return features_dict
    
    sorted_dict_orders = []
    for i in range(5):
        modalities = process_input_data(channels)
        features_dict = forward_logic_sorted(modalities)
        dict_order = list(features_dict.keys())
        sorted_dict_orders.append(dict_order)
        print(f"运行{i+1} - sorted后dict.keys()顺序: {dict_order}")
    
    sorted_all_same = all(order == sorted_dict_orders[0] for order in sorted_dict_orders)
    print(f"所有运行sorted后顺序相同: {sorted_all_same}")
    print()
    
    # 测试sorted后的random.sample
    random.seed(42)
    sorted_samples = []
    for dict_order in sorted_dict_orders:
        random.seed(42)
        sorted_samples.append(random.sample(dict_order, min(3, len(dict_order))))
    
    sorted_all_same = all(s == sorted_samples[0] for s in sorted_samples)
    print(f"所有运行sorted后random.sample结果相同: {sorted_all_same}")
    if sorted_all_same:
        print("✓ 使用sorted后，所有结果都相同！")
    print()
    
    # 总结
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("问题：")
    print("  1. 即使channels_used_in_model相同，遍历set的顺序可能不确定")
    print("  2. 遍历dict.keys()的顺序依赖于插入顺序")
    print("  3. 如果顺序不同，random.sample和torch.cat的结果也会不同")
    print()
    print("解决方案：")
    print("  1. 使用sorted()对set进行排序")
    print("  2. 使用sorted()对dict.keys()进行排序")
    print("  3. 确保所有遍历操作都使用排序后的顺序")

if __name__ == '__main__':
    test_real_scenario()

