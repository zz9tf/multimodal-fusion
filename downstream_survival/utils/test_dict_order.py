#!/usr/bin/env python3
"""
测试：即使channels_used_in_model相同，遍历字典的顺序也可能不一致
"""
import random

def test_dict_order_issue():
    """测试字典遍历顺序不一致的问题"""
    
    print("=" * 70)
    print("问题场景：即使channels_used_in_model相同，遍历字典顺序可能不同")
    print("=" * 70)
    
    # 模拟_process_input_data的逻辑
    def process_channels(channels_used_in_model):
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
    
    # 场景：相同的channels_used_in_model
    channels = ['CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1', 'wsi=features', 'tma=features']
    
    print(f"输入channels_used_in_model: {channels}")
    print()
    
    # 多次运行，观察set的遍历顺序
    print("=" * 70)
    print("测试1: 多次运行，观察set的遍历顺序")
    print("=" * 70)
    
    set_iterations = []
    for i in range(5):
        modalities = process_channels(channels)
        # 模拟遍历set
        iteration_order = list(modalities)
        set_iterations.append(iteration_order)
        print(f"运行{i+1} - set遍历顺序: {iteration_order}")
    
    # 检查是否相同
    all_same = all(iter == set_iterations[0] for iter in set_iterations)
    print(f"所有运行set遍历顺序相同: {all_same}")
    print()
    
    # 测试2: 构建features_dict的顺序
    print("=" * 70)
    print("测试2: 构建features_dict的顺序（模拟forward中的逻辑）")
    print("=" * 70)
    
    def build_features_dict(modalities_used_in_model):
        """模拟forward中构建features_dict的逻辑"""
        features_dict = {}
        # 关键：遍历set，顺序可能不确定
        for channel in modalities_used_in_model:
            # 模拟添加特征
            features_dict[channel] = f"feature_{channel}"
        return features_dict
    
    dict_iterations = []
    for i in range(5):
        modalities = process_channels(channels)
        features_dict = build_features_dict(modalities)
        # 获取keys的顺序
        keys_order = list(features_dict.keys())
        dict_iterations.append(keys_order)
        print(f"运行{i+1} - features_dict.keys()顺序: {keys_order}")
    
    # 检查是否相同
    all_same = all(iter == dict_iterations[0] for iter in dict_iterations)
    print(f"所有运行features_dict.keys()顺序相同: {all_same}")
    print()
    
    # 测试3: 对random.sample的影响
    print("=" * 70)
    print("测试3: 对random.sample的影响")
    print("=" * 70)
    
    random.seed(42)
    for i, keys_order in enumerate(dict_iterations):
        random.seed(42)
        sample = random.sample(keys_order, min(3, len(keys_order)))
        print(f"运行{i+1} - random.sample结果: {sample}")
    
    # 检查是否相同
    random.seed(42)
    samples = []
    for keys_order in dict_iterations:
        random.seed(42)
        samples.append(random.sample(keys_order, min(3, len(keys_order))))
    
    all_same = all(s == samples[0] for s in samples)
    print(f"所有运行random.sample结果相同: {all_same}")
    if not all_same:
        print("⚠️  警告：即使seed相同，如果keys顺序不同，random.sample结果也不同！")
    print()
    
    # 测试4: 对torch.cat的影响
    print("=" * 70)
    print("测试4: 对torch.cat的影响（模拟拼接特征）")
    print("=" * 70)
    
    import torch
    
    def concat_features(features_dict):
        """模拟torch.cat拼接特征"""
        # 方式1: 使用values() - 顺序依赖于插入顺序
        values_order = list(features_dict.values())
        # 方式2: 使用keys()遍历 - 顺序依赖于插入顺序
        keys_order = [features_dict[k] for k in features_dict.keys()]
        return values_order, keys_order
    
    concat_results = []
    for i, keys_order in enumerate(dict_iterations):
        # 重建features_dict
        features_dict = {k: torch.randn(10) for k in keys_order}
        values_order, keys_order_concat = concat_features(features_dict)
        concat_results.append((values_order, keys_order_concat))
        print(f"运行{i+1} - 拼接顺序（keys）: {list(features_dict.keys())}")
    
    # 检查拼接顺序是否相同
    all_same = all(
        list(r1[0]) == list(r2[0]) and list(r1[1]) == list(r2[1])
        for r1, r2 in zip(concat_results, concat_results[1:])
    )
    print(f"所有运行拼接顺序相同: {all_same}")
    if not all_same:
        print("⚠️  警告：即使元素相同，拼接顺序不同会导致特征顺序不同！")
    print()
    
    # 解决方案
    print("=" * 70)
    print("解决方案：使用sorted确保顺序一致")
    print("=" * 70)
    
    def build_features_dict_sorted(modalities_used_in_model):
        """使用sorted确保顺序一致"""
        features_dict = {}
        # 🔧 关键修复：对set进行排序
        for channel in sorted(modalities_used_in_model):
            features_dict[channel] = f"feature_{channel}"
        return features_dict
    
    sorted_dict_iterations = []
    for i in range(5):
        modalities = process_channels(channels)
        features_dict = build_features_dict_sorted(modalities)
        keys_order = list(features_dict.keys())
        sorted_dict_iterations.append(keys_order)
        print(f"运行{i+1} - sorted后features_dict.keys()顺序: {keys_order}")
    
    # 检查是否相同
    all_same = all(iter == sorted_dict_iterations[0] for iter in sorted_dict_iterations)
    print(f"所有运行sorted后顺序相同: {all_same}")
    print()
    
    # 测试sorted后的random.sample
    random.seed(42)
    sorted_samples = []
    for keys_order in sorted_dict_iterations:
        random.seed(42)
        sorted_samples.append(random.sample(keys_order, min(3, len(keys_order))))
    
    all_same = all(s == sorted_samples[0] for s in sorted_samples)
    print(f"所有运行sorted后random.sample结果相同: {all_same}")
    if all_same:
        print("✓ 使用sorted后，所有结果都相同！")

if __name__ == '__main__':
    test_dict_order_issue()

