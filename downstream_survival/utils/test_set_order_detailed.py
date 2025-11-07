#!/usr/bin/env python3
"""
详细测试set转list顺序不一致的情况
重点测试：当输入列表顺序不同时，set的插入顺序是否会导致list顺序不同
"""
import random
import sys

def test_real_world_scenario():
    """模拟真实场景：channels_used_in_model顺序不同"""
    
    print("=" * 70)
    print("真实场景测试：模拟_process_input_data中的逻辑")
    print("=" * 70)
    
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
    
    # 场景1: 不同的输入顺序
    channels1 = ['CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1', 'wsi=features', 'tma=features']
    channels2 = ['wsi=features', 'tma=features', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1']
    channels3 = ['HE', 'MHC1', 'PDL1', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'wsi=features', 'tma=features']
    
    set1 = process_channels(channels1)
    set2 = process_channels(channels2)
    set3 = process_channels(channels3)
    
    list1 = list(set1)
    list2 = list(set2)
    list3 = list(set3)
    
    print(f"输入1: {channels1}")
    print(f"输入2: {channels2}")
    print(f"输入3: {channels3}")
    print()
    print(f"set1: {set1}")
    print(f"set2: {set2}")
    print(f"set3: {set3}")
    print()
    print(f"list1: {list1}")
    print(f"list2: {list2}")
    print(f"list3: {list3}")
    print()
    print(f"list1 == list2: {list1 == list2}")
    print(f"list1 == list3: {list1 == list3}")
    print(f"list2 == list3: {list2 == list3}")
    print()
    
    # 检查顺序是否相同（元素相同但顺序不同）
    print(f"set1 == set2: {set1 == set2} (元素相同)")
    print(f"list1顺序相同: {list1 == sorted(list1)}")
    print(f"list2顺序相同: {list2 == sorted(list2)}")
    print(f"list3顺序相同: {list3 == sorted(list3)}")
    print()
    
    # 关键测试：如果顺序不同，会影响后续操作
    print("=" * 70)
    print("关键测试：顺序不同对random.sample的影响")
    print("=" * 70)
    
    random.seed(42)
    sample1 = random.sample(list1, min(3, len(list1)))
    random.seed(42)
    sample2 = random.sample(list2, min(3, len(list2)))
    random.seed(42)
    sample3 = random.sample(list3, min(3, len(list3)))
    
    print(f"random.seed(42)后，从list1采样: {sample1}")
    print(f"random.seed(42)后，从list2采样: {sample2}")
    print(f"random.seed(42)后，从list3采样: {sample3}")
    print()
    print(f"sample1 == sample2: {sample1 == sample2}")
    print(f"sample1 == sample3: {sample1 == sample3}")
    print()
    
    # 如果顺序不同，即使seed相同，random.sample的结果也可能不同
    if list1 != list2:
        print("⚠️  警告：list1和list2顺序不同，即使seed相同，random.sample结果可能不同！")
    else:
        print("✓ list1和list2顺序相同，random.sample结果相同")
    print()
    
    # 解决方案：使用sorted
    print("=" * 70)
    print("解决方案：使用sorted确保顺序一致")
    print("=" * 70)
    
    sorted_list1 = sorted(list(set1))
    sorted_list2 = sorted(list(set2))
    sorted_list3 = sorted(list(set3))
    
    random.seed(42)
    sorted_sample1 = random.sample(sorted_list1, min(3, len(sorted_list1)))
    random.seed(42)
    sorted_sample2 = random.sample(sorted_list2, min(3, len(sorted_list2)))
    random.seed(42)
    sorted_sample3 = random.sample(sorted_list3, min(3, len(sorted_list3)))
    
    print(f"sorted_list1: {sorted_list1}")
    print(f"sorted_list2: {sorted_list2}")
    print(f"sorted_list3: {sorted_list3}")
    print()
    print(f"sorted_sample1: {sorted_sample1}")
    print(f"sorted_sample2: {sorted_sample2}")
    print(f"sorted_sample3: {sorted_sample3}")
    print()
    print(f"所有sorted_sample相同: {sorted_sample1 == sorted_sample2 == sorted_sample3}")
    print()

def test_python_version_behavior():
    """测试Python版本对set顺序的影响"""
    print("=" * 70)
    print("Python版本信息")
    print("=" * 70)
    print(f"Python版本: {sys.version}")
    print()
    print("Python 3.7+ 中，set保持插入顺序（insertion order）")
    print("但插入顺序依赖于输入列表的顺序")
    print("如果输入列表顺序不同，set的迭代顺序就会不同")
    print("即使元素相同，list(set)的顺序也可能不同")
    print()

if __name__ == '__main__':
    test_python_version_behavior()
    test_real_world_scenario()

