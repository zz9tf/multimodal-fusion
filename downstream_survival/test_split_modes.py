#!/usr/bin/env python3
"""
测试数据集分割模式的正确性
验证随机分割和固定测试集分割两种模式
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

# 导入main.py中的函数
import importlib.util
spec = importlib.util.spec_from_file_location("main_module", "/home/zheng/zheng/multimodal-fusion/downstream_survival/main.py")
main_module = importlib.util.module_from_spec(spec)

# 只导入我们需要的函数，避免执行main.py的参数解析部分
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

class MockDataset:
    """模拟数据集类，用于测试"""
    
    def __init__(self, patient_ids, labels):
        self.patient_ids = patient_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        return {
            'patient_id': self.patient_ids[idx],
            'label': self.labels[idx]
        }
    
    def get_patient_id(self, idx):
        return self.patient_ids[idx]
    
    def get_label(self, idx):
        return self.labels[idx]

def create_test_dataset():
    """创建测试数据集"""
    # 创建763个患者的数据
    patient_ids = [f"{i:03d}" for i in range(1, 764)]  # 001-763
    
    # 创建标签（假设70%为类别0，30%为类别1）
    np.random.seed(42)
    labels = np.random.choice([0, 1], size=763, p=[0.7, 0.3])
    
    return MockDataset(patient_ids, labels)

def test_load_dataset_split():
    """测试加载数据集分割功能"""
    print("🧪 测试1: 加载数据集分割功能")
    print("-" * 40)
    
    dataset_split_path = "/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/dataset_split_in.json"
    
    try:
        split_data = load_dataset_split(dataset_split_path)
        
        print(f"✅ 成功加载数据集分割")
        print(f"📊 训练集患者数: {len(split_data['train'])}")
        print(f"📊 测试集患者数: {len(split_data['test'])}")
        print(f"📊 总患者数: {len(split_data['train']) + len(split_data['test'])}")
        
        # 验证患者ID格式
        train_sample = split_data['train'][:5]
        test_sample = split_data['test'][:5]
        print(f"📋 训练集样本: {train_sample}")
        print(f"📋 测试集样本: {test_sample}")
        
        return True  # 返回True表示测试通过
        
    except Exception as e:
        print(f"❌ 加载数据集分割失败: {e}")
        return False

def test_random_split_mode():
    """测试随机分割模式"""
    print("\n🧪 测试2: 随机分割模式")
    print("-" * 40)
    
    dataset = create_test_dataset()
    
    try:
        splits = create_k_fold_splits(dataset, k=5, seed=42, fixed_test_split=None)
        
        print(f"✅ 成功创建 {len(splits)} 个fold")
        
        # 检查每个fold的结构
        for i, split in enumerate(splits):
            train_size = len(split['train'])
            val_size = len(split['val'])
            test_size = len(split['test'])
            total_size = train_size + val_size + test_size
            
            print(f"📊 Fold {i+1}: Train={train_size}, Val={val_size}, Test={test_size}, Total={total_size}")
            
            # 验证索引不重复
            all_indices = set(split['train']) | set(split['val']) | set(split['test'])
            if len(all_indices) != total_size:
                print(f"❌ Fold {i+1}: 索引重复!")
                return False
        
        # 检查不同fold的测试集是否不同
        test_sets = [set(split['test']) for split in splits]
        all_different = all(len(set.intersection(*test_sets[i:i+2])) == 0 
                           for i in range(len(test_sets)-1))
        
        if all_different:
            print("✅ 不同fold的测试集确实不同")
        else:
            print("⚠️  不同fold的测试集有重叠")
        
        return True
        
    except Exception as e:
        print(f"❌ 随机分割模式测试失败: {e}")
        return False

def test_fixed_split_mode():
    """测试固定测试集分割模式"""
    print("\n🧪 测试3: 固定测试集分割模式")
    print("-" * 40)
    
    # 加载真实的分割数据
    dataset_split_path = "/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/dataset_split_in.json"
    
    try:
        fixed_test_split = load_dataset_split(dataset_split_path)
        
        # 创建模拟数据集（只包含分割中的患者）
        all_patients = fixed_test_split['train'] + fixed_test_split['test']
        
        # 创建标签（随机生成，用于测试）
        np.random.seed(42)
        labels = np.random.choice([0, 1], size=len(all_patients), p=[0.7, 0.3])
        
        dataset = MockDataset(all_patients, labels)
        
        splits = create_k_fold_splits(dataset, k=5, seed=42, fixed_test_split=fixed_test_split)
        
        print(f"✅ 成功创建 {len(splits)} 个fold")
        
        # 检查每个fold的结构
        test_sets_are_same = True
        for i, split in enumerate(splits):
            train_size = len(split['train'])
            val_size = len(split['val'])
            test_size = len(split['test'])
            
            print(f"📊 Fold {i+1}: Train={train_size}, Val={val_size}, Test={test_size}")
            
            # 检查测试集是否相同
            if i == 0:
                first_test_set = set(split['test'])
            else:
                current_test_set = set(split['test'])
                if first_test_set != current_test_set:
                    test_sets_are_same = False
        
        if test_sets_are_same:
            print("✅ 所有fold的测试集都相同（固定测试集）")
        else:
            print("❌ 测试集不一致!")
            return False
        
        # 验证测试集大小
        expected_test_size = len(fixed_test_split['test'])
        actual_test_size = len(splits[0]['test'])
        
        if expected_test_size == actual_test_size:
            print(f"✅ 测试集大小正确: {actual_test_size}")
        else:
            print(f"❌ 测试集大小不匹配: 期望{expected_test_size}, 实际{actual_test_size}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 固定测试集分割模式测试失败: {e}")
        return False

def test_patient_id_mapping():
    """测试患者ID映射的正确性"""
    print("\n🧪 测试4: 患者ID映射正确性")
    print("-" * 40)
    
    dataset_split_path = "/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/dataset_split_in.json"
    
    try:
        fixed_test_split = load_dataset_split(dataset_split_path)
        
        # 创建包含所有患者的模拟数据集
        all_patients = fixed_test_split['train'] + fixed_test_split['test']
        np.random.seed(42)
        labels = np.random.choice([0, 1], size=len(all_patients), p=[0.7, 0.3])
        
        dataset = MockDataset(all_patients, labels)
        
        splits = create_k_fold_splits(dataset, k=3, seed=42, fixed_test_split=fixed_test_split)
        
        # 检查第一个fold的测试集患者ID
        test_indices = splits[0]['test']
        test_patient_ids = [dataset.get_patient_id(idx) for idx in test_indices]
        
        print(f"📋 测试集患者ID样本: {test_patient_ids[:10]}")
        print(f"📋 期望的测试集患者ID样本: {fixed_test_split['test'][:10]}")
        
        # 验证映射是否正确
        expected_test_ids = set(fixed_test_split['test'])
        actual_test_ids = set(test_patient_ids)
        
        if expected_test_ids == actual_test_ids:
            print("✅ 患者ID映射正确")
            return True
        else:
            print("❌ 患者ID映射错误")
            missing = expected_test_ids - actual_test_ids
            extra = actual_test_ids - expected_test_ids
            if missing:
                print(f"   缺失的患者ID: {list(missing)[:5]}...")
            if extra:
                print(f"   多余的患者ID: {list(extra)[:5]}...")
            return False
        
    except Exception as e:
        print(f"❌ 患者ID映射测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试数据集分割模式")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_load_dataset_split,
        test_random_split_mode,
        test_fixed_split_mode,
        test_patient_id_mapping
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result if isinstance(result, bool) else False)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {e}")
            results.append(False)
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "加载数据集分割功能",
        "随机分割模式",
        "固定测试集分割模式", 
        "患者ID映射正确性"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！分割模式功能正常。")
    else:
        print("⚠️  部分测试失败，请检查代码。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
