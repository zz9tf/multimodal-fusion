"""
比较两个 DataLoader 的 samples 是否一样（通过 patient_id/case_id）
"""

import sys
import os
from typing import List
from torch.utils.data import DataLoader

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from trainer import get_split_loader


def get_loader_case_ids(loader: DataLoader) -> List[str]:
    """
    从 DataLoader 中提取所有 case_id（patient_id）
    
    Args:
        loader: 数据加载器
        
    Returns:
        case_ids: case_id 列表，按照 DataLoader 的迭代顺序
    """
    dataset_ref = loader.dataset
    case_ids = []
    
    # 遍历 DataLoader 获取实际的 case_id 顺序
    for batch_idx, (data, label) in enumerate(loader):
        # 从数据集中获取 case_id
        if hasattr(dataset_ref, 'case_ids'):
            # 直接数据集（拥有 case_ids 属性）
            case_id = dataset_ref.case_ids[batch_idx]
        elif hasattr(dataset_ref, 'dataset') and hasattr(dataset_ref.dataset, 'case_ids') and hasattr(dataset_ref, 'indices'):
            # Subset 数据集（没有 case_ids 属性，需从原数据集映射）
            # DataLoader 会按照 dataset_ref.indices 的顺序迭代，所以 batch_idx 对应 dataset_ref.indices[batch_idx]
            base = dataset_ref.dataset.case_ids
            base_list = list(base) if not isinstance(base, list) else base
            case_id = base_list[dataset_ref.indices[batch_idx]]
        else:
            # 降级：使用索引作为 case_id
            case_id = f"sample_{batch_idx}"
        
        case_ids.append(case_id)
    
    return case_ids


def compare_loader_samples(loader1: DataLoader, loader2: DataLoader, name1: str = "Loader1", name2: str = "Loader2") -> bool:
    """
    比较两个 DataLoader 的 samples 是否一样（通过 patient_id/case_id）
    
    Args:
        loader1: 第一个数据加载器
        loader2: 第二个数据加载器
        name1: 第一个加载器的名称（用于日志）
        name2: 第二个加载器的名称（用于日志）
        
    Returns:
        is_same: 如果 samples 一样返回 True，否则返回 False
    """
    case_ids1 = get_loader_case_ids(loader1)
    case_ids2 = get_loader_case_ids(loader2)
    
    # 比较长度
    if len(case_ids1) != len(case_ids2):
        print(f"⚠️ {name1} 和 {name2} 的样本数量不同: {len(case_ids1)} vs {len(case_ids2)}")
        return False
    
    # 比较每个位置的 case_id
    differences = []
    for i, (cid1, cid2) in enumerate(zip(case_ids1, case_ids2)):
        if cid1 != cid2:
            differences.append((i, cid1, cid2))
    
    if differences:
        print(f"⚠️ {name1} 和 {name2} 的样本顺序不同:")
        print(f"   总样本数: {len(case_ids1)}")
        print(f"   不同位置数: {len(differences)}")
        if len(differences) <= 10:
            print(f"   前 {len(differences)} 个不同位置:")
            for idx, cid1, cid2 in differences:
                print(f"     位置 {idx}: {name1}={cid1}, {name2}={cid2}")
        else:
            print(f"   前 10 个不同位置:")
            for idx, cid1, cid2 in differences[:10]:
                print(f"     位置 {idx}: {name1}={cid1}, {name2}={cid2}")
            print(f"   ... 还有 {len(differences) - 10} 个不同位置")
        return False
    else:
        print(f"✅ {name1} 和 {name2} 的样本顺序一致 (共 {len(case_ids1)} 个样本)")
        return True


if __name__ == "__main__":
    import argparse
    from datasets.multimodal_dataset import MultimodalDataset
    from main import create_k_fold_splits, parse_channels
    from torch.utils.data import Subset
    
    parser = argparse.ArgumentParser(description="比较 DataLoader 的 samples 是否一样")
    parser.add_argument("--data_root_dir", type=str, required=True, help="数据根目录")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 文件路径")
    parser.add_argument("--channels", type=str, required=True, help="通道列表，用空格分隔")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold 索引")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--compare_splits", action="store_true", help="比较 train、val、test 之间的 samples 是否一致")
    parser.add_argument("--compare_same_split", action="store_true", help="比较同一个 split 的两个 DataLoader 是否一致")
    parser.add_argument("--split_type", type=str, choices=["train", "val", "test"], default="val", help="比较哪个 split（仅在 --compare_same_split 时使用）")
    
    args = parser.parse_args()
    
    # 解析通道
    target_channels = args.channels.split()
    channels = parse_channels(target_channels)
    
    # 加载数据集
    print(f"📂 加载数据集: {args.data_root_dir}")
    dataset = MultimodalDataset(
        csv_path=args.csv_path,
        data_root_dir=args.data_root_dir,
        channels=channels,
        align_channels=None,
        alignment_model_path=None,
        device="cpu",
        print_info=True
    )
    
    # 创建 K 折分割
    print(f"🔄 创建 K 折分割 (fold={args.fold_idx}, seed={args.seed})")
    k_fold_splits = create_k_fold_splits(dataset, k=10, seed=args.seed)
    
    if args.fold_idx >= len(k_fold_splits):
        print(f"❌ Fold 索引 {args.fold_idx} 超出范围 (共 {len(k_fold_splits)} 个 folds)")
        sys.exit(1)
    
    split = k_fold_splits[args.fold_idx]
    
    # 创建 Subset 数据集
    train_subset = Subset(dataset, split['train'])
    val_subset = Subset(dataset, split['val'])
    test_subset = Subset(dataset, split['test'])
    
    print(f"\n📊 Split 大小:")
    print(f"   Train: {len(train_subset)}")
    print(f"   Val: {len(val_subset)}")
    print(f"   Test: {len(test_subset)}")
    
    # 比较不同 split 之间的 samples
    if args.compare_splits:
        print(f"\n🔍 比较 train、val、test 之间的 samples 是否一致...")
        
        # 创建 DataLoader
        train_loader = get_split_loader(train_subset, training=False, weighted=False, batch_size=1, generator=None)
        val_loader = get_split_loader(val_subset, training=False, weighted=False, batch_size=1, generator=None)
        test_loader = get_split_loader(test_subset, training=False, weighted=False, batch_size=1, generator=None)
        
        # 获取所有 case_ids
        train_case_ids = set(get_loader_case_ids(train_loader))
        val_case_ids = set(get_loader_case_ids(val_loader))
        test_case_ids = set(get_loader_case_ids(test_loader))
        
        # 比较是否有重叠
        train_val_overlap = train_case_ids & val_case_ids
        train_test_overlap = train_case_ids & test_case_ids
        val_test_overlap = val_case_ids & test_case_ids
        
        all_same = True
        
        if train_val_overlap:
            print(f"⚠️ Train 和 Val 有重叠的 samples: {len(train_val_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(train_val_overlap))[:10]}{'...' if len(train_val_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Train 和 Val 没有重叠的 samples")
        
        if train_test_overlap:
            print(f"⚠️ Train 和 Test 有重叠的 samples: {len(train_test_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(train_test_overlap))[:10]}{'...' if len(train_test_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Train 和 Test 没有重叠的 samples")
        
        if val_test_overlap:
            print(f"⚠️ Val 和 Test 有重叠的 samples: {len(val_test_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(val_test_overlap))[:10]}{'...' if len(val_test_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Val 和 Test 没有重叠的 samples")
        
        # 检查是否有遗漏的 samples
        all_case_ids = train_case_ids | val_case_ids | test_case_ids
        dataset_case_ids = set(dataset.case_ids)
        missing_case_ids = dataset_case_ids - all_case_ids
        
        if missing_case_ids:
            print(f"⚠️ 有 {len(missing_case_ids)} 个 samples 没有被包含在任何 split 中")
            print(f"   遗漏的 case_ids: {sorted(list(missing_case_ids))[:10]}{'...' if len(missing_case_ids) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ 所有 samples 都被包含在 split 中")
        
        if all_same:
            print(f"\n✅ 所有 split 之间没有重叠，且所有 samples 都被包含")
            sys.exit(0)
        else:
            print(f"\n❌ 发现 split 之间的重叠或遗漏")
            sys.exit(1)
    
    # 比较同一个 split 的两个 DataLoader
    elif args.compare_same_split:
        print(f"\n🔧 创建两个 DataLoader (split_type={args.split_type})")
        if args.split_type == "train":
            split_dataset = train_subset
            split_name = "Train"
        elif args.split_type == "val":
            split_dataset = val_subset
            split_name = "Val"
        else:
            split_dataset = test_subset
            split_name = "Test"
        
        # 创建两个独立的 DataLoader
        loader1 = get_split_loader(split_dataset, training=False, weighted=False, batch_size=1, generator=None)
        loader2 = get_split_loader(split_dataset, training=False, weighted=False, batch_size=1, generator=None)
        
        # 比较 samples
        print(f"\n🔍 比较两个 {split_name} Loader 的样本顺序...")
        is_same = compare_loader_samples(loader1, loader2, name1=f"{split_name} Loader 1", name2=f"{split_name} Loader 2")
        
        if is_same:
            print(f"\n✅ 两个 {split_name} Loader 的样本顺序一致")
            sys.exit(0)
        else:
            print(f"\n❌ 两个 {split_name} Loader 的样本顺序不一致")
            sys.exit(1)
    
    # 默认：同时比较 split 之间和同一个 split 的两个 DataLoader
    else:
        print(f"\n🔍 比较 train、val、test 之间的 samples 是否一致...")
        
        # 创建 DataLoader
        train_loader = get_split_loader(train_subset, training=False, weighted=False, batch_size=1, generator=None)
        val_loader = get_split_loader(val_subset, training=False, weighted=False, batch_size=1, generator=None)
        test_loader = get_split_loader(test_subset, training=False, weighted=False, batch_size=1, generator=None)
        
        # 获取所有 case_ids
        train_case_ids = set(get_loader_case_ids(train_loader))
        val_case_ids = set(get_loader_case_ids(val_loader))
        test_case_ids = set(get_loader_case_ids(test_loader))
        
        # 比较是否有重叠
        train_val_overlap = train_case_ids & val_case_ids
        train_test_overlap = train_case_ids & test_case_ids
        val_test_overlap = val_case_ids & test_case_ids
        
        all_same = True
        
        if train_val_overlap:
            print(f"⚠️ Train 和 Val 有重叠的 samples: {len(train_val_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(train_val_overlap))[:10]}{'...' if len(train_val_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Train 和 Val 没有重叠的 samples")
        
        if train_test_overlap:
            print(f"⚠️ Train 和 Test 有重叠的 samples: {len(train_test_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(train_test_overlap))[:10]}{'...' if len(train_test_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Train 和 Test 没有重叠的 samples")
        
        if val_test_overlap:
            print(f"⚠️ Val 和 Test 有重叠的 samples: {len(val_test_overlap)} 个")
            print(f"   重叠的 case_ids: {sorted(list(val_test_overlap))[:10]}{'...' if len(val_test_overlap) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ Val 和 Test 没有重叠的 samples")
        
        # 检查是否有遗漏的 samples
        all_case_ids = train_case_ids | val_case_ids | test_case_ids
        dataset_case_ids = set(dataset.case_ids)
        missing_case_ids = dataset_case_ids - all_case_ids
        
        if missing_case_ids:
            print(f"⚠️ 有 {len(missing_case_ids)} 个 samples 没有被包含在任何 split 中")
            print(f"   遗漏的 case_ids: {sorted(list(missing_case_ids))[:10]}{'...' if len(missing_case_ids) > 10 else ''}")
            all_same = False
        else:
            print(f"✅ 所有 samples 都被包含在 split 中")
        
        # 比较同一个 split 的两个 DataLoader
        print(f"\n🔍 比较同一个 split 的两个 DataLoader 是否一致...")
        
        # 说明 weighted 的作用
        print(f"\n📖 Weighted 参数说明:")
        print(f"   - weighted=True: 使用 WeightedRandomSampler，根据类别权重进行采样，平衡类别分布")
        print(f"   - weighted=False: 使用 shuffle=True，只是随机打乱顺序，不进行类别平衡")
        print(f"   - 注意：如果没有提供 generator，每次创建 DataLoader 时都会使用不同的随机状态")
        
        # 检查类别分布
        print(f"\n📊 检查 Train 数据集的类别分布...")
        from trainer import make_weights_for_balanced_classes_split
        weights = make_weights_for_balanced_classes_split(train_subset)
        print(f"   WeightedRandomSampler 的权重已计算，权重范围: {weights.min():.4f} - {weights.max():.4f}")
        
        # 获取标签分布
        train_labels = []
        for i in range(len(train_subset)):
            if hasattr(train_subset, 'dataset') and hasattr(train_subset.dataset, 'get_label'):
                original_idx = train_subset.indices[i]
                label = train_subset.dataset.get_label(original_idx)
            else:
                label = train_subset.get_label(i)
            train_labels.append(label)
        
        from collections import Counter
        label_counts = Counter(train_labels)
        print(f"   类别分布: {dict(label_counts)}")
        
        # 比较 Train Loader（使用 weighted=True 和 weighted=False，使用相同的 generator）
        print(f"\n📊 比较 Train Loader（weighted=True vs weighted=False，使用相同的 generator）...")
        import torch
        generator = torch.Generator().manual_seed(args.seed)
        train_loader_weighted = get_split_loader(train_subset, training=True, weighted=True, batch_size=1, generator=generator)
        generator2 = torch.Generator().manual_seed(args.seed)
        train_loader_unweighted = get_split_loader(train_subset, training=True, weighted=False, batch_size=1, generator=generator2)
        train_same_weighted = compare_loader_samples(train_loader_weighted, train_loader_unweighted, name1="Train Loader (weighted=True)", name2="Train Loader (weighted=False)")
        
        # 比较 Train Loader（两个 weighted=True 的 loader，使用相同的 generator）
        print(f"\n📊 比较 Train Loader（两个 weighted=True 的 loader，使用相同的 generator）...")
        generator3 = torch.Generator().manual_seed(args.seed)
        train_loader1_weighted = get_split_loader(train_subset, training=True, weighted=True, batch_size=1, generator=generator3)
        generator4 = torch.Generator().manual_seed(args.seed)
        train_loader2_weighted = get_split_loader(train_subset, training=True, weighted=True, batch_size=1, generator=generator4)
        train_same_weighted_self = compare_loader_samples(train_loader1_weighted, train_loader2_weighted, name1="Train Loader 1 (weighted=True)", name2="Train Loader 2 (weighted=True)")
        
        # 比较 Train Loader（两个 weighted=False 的 loader，使用相同的 generator）
        print(f"\n📊 比较 Train Loader（两个 weighted=False 的 loader，使用相同的 generator）...")
        generator5 = torch.Generator().manual_seed(args.seed)
        train_loader1_unweighted = get_split_loader(train_subset, training=True, weighted=False, batch_size=1, generator=generator5)
        generator6 = torch.Generator().manual_seed(args.seed)
        train_loader2_unweighted = get_split_loader(train_subset, training=True, weighted=False, batch_size=1, generator=generator6)
        train_same_unweighted_self = compare_loader_samples(train_loader1_unweighted, train_loader2_unweighted, name1="Train Loader 1 (weighted=False)", name2="Train Loader 2 (weighted=False)")
        
        # 检查 weighted 对类别分布的影响
        print(f"\n📊 检查 weighted 对类别分布的影响...")
        train_case_ids_weighted = get_loader_case_ids(train_loader1_weighted)
        train_case_ids_unweighted = get_loader_case_ids(train_loader1_unweighted)
        
        # 获取每个 case_id 的标签
        case_id_to_label = {}
        for i in range(len(train_subset)):
            if hasattr(train_subset, 'dataset') and hasattr(train_subset.dataset, 'case_ids'):
                original_idx = train_subset.indices[i]
                case_id = train_subset.dataset.case_ids[original_idx]
                label = train_subset.dataset.get_label(original_idx)
            else:
                case_id = train_subset.case_ids[i]
                label = train_subset.get_label(i)
            case_id_to_label[case_id] = label
        
        # 统计 weighted 和 unweighted 的类别分布
        weighted_labels = [case_id_to_label[cid] for cid in train_case_ids_weighted]
        unweighted_labels = [case_id_to_label[cid] for cid in train_case_ids_unweighted]
        
        weighted_label_counts = Counter(weighted_labels)
        unweighted_label_counts = Counter(unweighted_labels)
        
        print(f"   Weighted=True 的类别分布: {dict(weighted_label_counts)}")
        print(f"   Weighted=False 的类别分布: {dict(unweighted_label_counts)}")
        
        # 计算类别比例
        total_weighted = sum(weighted_label_counts.values())
        total_unweighted = sum(unweighted_label_counts.values())
        
        print(f"   Weighted=True 的类别比例:")
        for label, count in sorted(weighted_label_counts.items()):
            print(f"     {label}: {count}/{total_weighted} ({count/total_weighted*100:.2f}%)")
        
        print(f"   Weighted=False 的类别比例:")
        for label, count in sorted(unweighted_label_counts.items()):
            print(f"     {label}: {count}/{total_unweighted} ({count/total_unweighted*100:.2f}%)")
        
        # 比较 Val Loader
        print(f"\n📊 比较 Val Loader...")
        val_loader1 = get_split_loader(val_subset, training=False, weighted=False, batch_size=1, generator=None)
        val_loader2 = get_split_loader(val_subset, training=False, weighted=False, batch_size=1, generator=None)
        val_same = compare_loader_samples(val_loader1, val_loader2, name1="Val Loader 1", name2="Val Loader 2")
        
        # 比较 Test Loader
        print(f"\n📊 比较 Test Loader...")
        test_loader1 = get_split_loader(test_subset, training=False, weighted=False, batch_size=1, generator=None)
        test_loader2 = get_split_loader(test_subset, training=False, weighted=False, batch_size=1, generator=None)
        test_same = compare_loader_samples(test_loader1, test_loader2, name1="Test Loader 1", name2="Test Loader 2")
        
        # 检查 Train Loader 的内容是否与 dataset 一致
        print(f"\n📊 检查 Train Loader 的内容是否与 dataset 一致...")
        train_case_ids_loader = set(get_loader_case_ids(train_loader1_weighted))
        train_case_ids_dataset = set([dataset.case_ids[i] for i in split['train']])
        
        if train_case_ids_loader == train_case_ids_dataset:
            print(f"✅ Train Loader 的内容与 dataset 一致 (共 {len(train_case_ids_loader)} 个 samples)")
        else:
            print(f"⚠️ Train Loader 的内容与 dataset 不一致")
            missing_in_loader = train_case_ids_dataset - train_case_ids_loader
            extra_in_loader = train_case_ids_loader - train_case_ids_dataset
            if missing_in_loader:
                print(f"   Loader 中缺少的 case_ids: {sorted(list(missing_in_loader))[:10]}{'...' if len(missing_in_loader) > 10 else ''}")
            if extra_in_loader:
                print(f"   Loader 中多余的 case_ids: {sorted(list(extra_in_loader))[:10]}{'...' if len(extra_in_loader) > 10 else ''}")
            train_same_weighted_self = False
        
        if all_same and train_same_weighted_self and train_same_unweighted_self and val_same and test_same:
            print(f"\n✅ 所有检查通过")
            sys.exit(0)
        else:
            print(f"\n❌ 发现一些问题")
            sys.exit(1)

