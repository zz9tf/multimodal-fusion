#!/usr/bin/env python3
"""
测试set转list顺序不一致的情况
"""
import random

def test_set_order_issues():
    """测试set转list顺序不一致的各种情况"""
    
    print("=" * 60)
    print("测试1: channels_used_in_model顺序不同")
    print("=" * 60)
    
    # 情况1: 列表顺序不同，但元素相同
    channels1 = ['CD3', 'CD8', 'CD56', 'wsi=features', 'tma=features']
    channels2 = ['wsi=features', 'tma=features', 'CD3', 'CD8', 'CD56']
    
    set1 = set()
    for channel in channels1:
        if channel.startswith('wsi='):
            set1.add('wsi=features')
        if channel.startswith('tma='):
            set1.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set1.add(channel)
    
    set2 = set()
    for channel in channels2:
        if channel.startswith('wsi='):
            set2.add('wsi=features')
        if channel.startswith('tma='):
            set2.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set2.add(channel)
    
    list1 = list(set1)
    list2 = list(set2)
    
    print(f"channels1: {channels1}")
    print(f"channels2: {channels2}")
    print(f"set1: {set1}")
    print(f"set2: {set2}")
    print(f"list1: {list1}")
    print(f"list2: {list2}")
    print(f"list1 == list2: {list1 == list2}")
    print(f"list1顺序相同: {list1 == sorted(list1)}")
    print()
    
    print("=" * 60)
    print("测试2: 重复元素导致插入顺序不同")
    print("=" * 60)
    
    # 情况2: 列表中有重复元素，但插入顺序不同
    channels3 = ['CD3', 'CD8', 'CD3', 'CD56', 'CD8']
    channels4 = ['CD56', 'CD3', 'CD8', 'CD3', 'CD8']
    
    set3 = set()
    for channel in channels3:
        set3.add(channel)
    
    set4 = set()
    for channel in channels4:
        set4.add(channel)
    
    list3 = list(set3)
    list4 = list(set4)
    
    print(f"channels3: {channels3}")
    print(f"channels4: {channels4}")
    print(f"set3: {set3}")
    print(f"set4: {set4}")
    print(f"list3: {list3}")
    print(f"list4: {list4}")
    print(f"list3 == list4: {list3 == list4}")
    print()
    
    print("=" * 60)
    print("测试3: 条件分支导致插入顺序不同")
    print("=" * 60)
    
    # 情况3: 条件分支导致某些元素被跳过或重复添加
    channels5 = ['wsi=features', 'CD3', 'tma=features', 'CD8']
    channels6 = ['CD3', 'wsi=features', 'CD8', 'tma=features']
    
    set5 = set()
    for channel in channels5:
        if channel.startswith('wsi='):
            set5.add('wsi=features')
        if channel.startswith('tma='):  # 注意：这里用的是if，不是elif
            set5.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set5.add(channel)
    
    set6 = set()
    for channel in channels6:
        if channel.startswith('wsi='):
            set6.add('wsi=features')
        if channel.startswith('tma='):
            set6.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set6.add(channel)
    
    list5 = list(set5)
    list6 = list(set6)
    
    print(f"channels5: {channels5}")
    print(f"channels6: {channels6}")
    print(f"set5: {set5}")
    print(f"set6: {set6}")
    print(f"list5: {list5}")
    print(f"list6: {list6}")
    print(f"list5 == list6: {list5 == list6}")
    print()
    
    print("=" * 60)
    print("测试4: 多次运行，观察顺序是否稳定")
    print("=" * 60)
    
    # 情况4: 多次运行，观察顺序是否稳定
    channels = ['CD3', 'CD8', 'CD56', 'CD68', 'CD163']
    results = []
    for i in range(5):
        s = set()
        for channel in channels:
            s.add(channel)
        results.append(list(s))
    
    print(f"原始channels: {channels}")
    for i, result in enumerate(results):
        print(f"运行{i+1}: {result}")
    
    # 检查所有结果是否相同
    all_same = all(result == results[0] for result in results)
    print(f"所有运行结果相同: {all_same}")
    print()
    
    print("=" * 60)
    print("测试5: 使用sorted确保顺序一致")
    print("=" * 60)
    
    # 使用sorted确保顺序一致
    channels7 = ['CD3', 'CD8', 'CD56', 'wsi=features', 'tma=features']
    channels8 = ['wsi=features', 'tma=features', 'CD3', 'CD8', 'CD56']
    
    set7 = set()
    for channel in channels7:
        if channel.startswith('wsi='):
            set7.add('wsi=features')
        if channel.startswith('tma='):
            set7.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set7.add(channel)
    
    set8 = set()
    for channel in channels8:
        if channel.startswith('wsi='):
            set8.add('wsi=features')
        if channel.startswith('tma='):
            set8.add('tma=features')
        elif channel.endswith('=mask'):
            continue
        else:
            set8.add(channel)
    
    list7 = sorted(set7)
    list8 = sorted(set8)
    
    print(f"channels7: {channels7}")
    print(f"channels8: {channels8}")
    print(f"set7: {set7}")
    print(f"set8: {set8}")
    print(f"sorted(list7): {list7}")
    print(f"sorted(list8): {list8}")
    print(f"sorted后相同: {list7 == list8}")
    print()

if __name__ == '__main__':
    test_set_order_issues()

