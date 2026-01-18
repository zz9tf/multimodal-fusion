#!/usr/bin/env python3
"""
SVD 对齐前后特征热力图可视化工具

用法示例：
python -m downstream_survival.utils.plot_alignment_heatmap \
  --features_dir /path/to/svd_features \
  --fold_idx 0 \
  --output_dir /path/to/output
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def load_features(features_dir: str, fold_idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    加载对齐前后的特征。
    
    Args:
        features_dir: 特征文件目录
        fold_idx: fold索引
        
    Returns:
        (original_features, aligned_features, modalities)
    """
    # 加载元数据获取模态列表
    metadata_path = os.path.join(features_dir, f'fold_{fold_idx}_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'元数据文件不存在: {metadata_path}')
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    modalities = sorted(metadata['modalities'])
    
    # 加载每个模态的特征
    original_features = {}
    aligned_features = {}
    
    for modality in modalities:
        safe_name = modality.replace('/', '_').replace('=', '_')
        
        original_path = os.path.join(features_dir, f'fold_{fold_idx}_{safe_name}_original.npy')
        aligned_path = os.path.join(features_dir, f'fold_{fold_idx}_{safe_name}_aligned.npy')
        
        if os.path.exists(original_path):
            original_features[modality] = np.load(original_path)
        if os.path.exists(aligned_path):
            aligned_features[modality] = np.load(aligned_path)
    
    return original_features, aligned_features, modalities


def load_patient_ids(results_dir: str, fold_idx: int) -> Optional[List[str]]:
    """
    从 splits CSV 文件中加载 patient ids。
    
    Args:
        results_dir: 结果目录（包含 splits_*.csv）
        fold_idx: fold索引
        
    Returns:
        test 集的 patient id 列表，如果文件不存在则返回 None
    """
    splits_path = os.path.join(results_dir, f'splits_{fold_idx}.csv')
    if not os.path.exists(splits_path):
        return None
    
    try:
        df = pd.read_csv(splits_path)
        if 'test' in df.columns:
            # 获取 test 集的 patient ids，过滤空值
            test_patients = df['test'].dropna().tolist()
            return test_patients
    except Exception as e:
        print(f"⚠️ 加载 patient ids 失败: {e}")
        return None
    
    return None


def plot_sample_heatmap(
    original_features: Dict[str, np.ndarray],
    aligned_features: Dict[str, np.ndarray],
    modalities: List[str],
    sample_idx: int,
    output_path: str,
    patient_id: str = None,
) -> None:
    """
    为单个样本绘制热力图：左边对齐前，右边对齐后。
    
    Args:
        original_features: 对齐前的特征字典，每个模态为 [N, 128]
        aligned_features: 对齐后的特征字典，每个模态为 [N, 128]
        modalities: 模态列表
        sample_idx: 样本索引
        output_path: 输出路径
        patient_id: 患者ID（可选，如果没有则使用 sample_idx）
    """
    # 构建单个样本的特征矩阵：7个模态 × 128维
    # 左边：对齐前，右边：对齐后
    n_modalities = len(modalities)
    n_dims = 128
    
    # 创建特征矩阵：[7, 256] (左边128维 + 右边128维)
    feature_matrix = np.zeros((n_modalities, n_dims * 2))
    
    for i, modality in enumerate(modalities):
        if modality in original_features:
            # 左边：对齐前的特征
            feature_matrix[i, :n_dims] = original_features[modality][sample_idx]
        if modality in aligned_features:
            # 右边：对齐后的特征
            feature_matrix[i, n_dims:] = aligned_features[modality][sample_idx]
    
    # 设置绘图风格（适合文献）
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'figure.dpi': 300,
        'axes.linewidth': 1.2,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
    })
    
    # 创建图形（调整尺寸适合文献）
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 绘制热力图（不使用白条分隔）
    sns.heatmap(feature_matrix, ax=ax, cmap='viridis', 
                # cbar_kws={'label': 'Feature Value', 'shrink': 0.8, 'aspect': 20},
                xticklabels=False, yticklabels=True)
    
    # 设置y轴标签为模态名（更清晰的格式）
    y_tick_positions = np.arange(n_modalities) + 0.5
    ax.set_yticks(y_tick_positions)
    
    # 模态名称映射（原始名称 -> 显示名称）
    modality_name_mapping = {
        'wsi=features': 'WSI',
        'tma=features': 'TMA',
        'clinical=val': 'Clinical',
        'pathological=val': 'Pathological',
        'blood=val': 'Blood',
        'icd=val': 'ICD',
        'tma_cell_density=val': 'TMA Cell Density',
    }
    
    # 格式化模态名称，使其更清晰
    modality_labels = []
    for mod in modalities:
        # 使用映射表获取清晰的名称，如果没有映射则使用原始名称
        if mod in modality_name_mapping:
            label = modality_name_mapping[mod]
        else:
            # 如果没有映射，尝试美化原始名称
            label = mod.replace('=', ': ').replace('/', '_').replace('_', ' ').title()
        modality_labels.append(label)
    
    ax.set_yticklabels(modality_labels, fontsize=12, fontweight='normal', 
                      rotation=0, ha='right', va='center')
    
    # 添加分隔线（中间分隔对齐前后）
    ax.axvline(x=n_dims, color='red', linestyle='--', linewidth=2.5, alpha=0.8, zorder=10)
    
    # 移除x轴和y轴标签
    ax.set_xlabel('', fontsize=0)
    ax.set_ylabel('', fontsize=0)
    
    # 调整布局，为顶部和底部文本留出空间
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 底部留出5%空间，顶部留出5%空间
    
    # 获取 axes 的位置（在 figure 坐标系中）
    ax_pos = ax.get_position()
    ax_left = ax_pos.x0
    ax_right = ax_pos.x1
    ax_width = ax_pos.width
    ax_center = ax_left + ax_width / 2
    
    # 计算左右两部分的中心位置
    # 热力图被分成两部分：左边（Before SVD）和右边（After SVD）
    left_center = ax_left + ax_width / 4  # 左边部分的中心
    right_center = ax_left + 3 * ax_width / 4  # 右边部分的中心
    
    # 在图形顶部添加文本标注（Before/After SVD），与热力图对齐
    # 左边：Before SVD
    fig.text(left_center, 0.98, 'Before SVD', ha='center', va='top', 
            fontsize=15, fontweight='bold', color='black')
    # 右边：After SVD
    fig.text(right_center, 0.98, 'After SVD', ha='center', va='top',
            fontsize=15, fontweight='bold', color='black')
    
    # 在图形底部添加标题（使用 patient id，如果没有则使用 sample_idx），居中
    if patient_id is not None:
        title = f'Patient {patient_id}'
    else:
        title = f'Patient {sample_idx}'
    fig.text(ax_center, -0.02, title, ha='center', va='bottom', 
            fontsize=16, fontweight='bold', color='black')
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='SVD 对齐前后特征热力图可视化工具')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='特征文件目录（包含 fold_*_*_original.npy 和 fold_*_*_aligned.npy）')
    parser.add_argument('--fold_idx', type=int, default=0,
                       help='要处理的 fold 索引（默认 0）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，默认使用 features_dir/heatmaps')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='要绘制的样本索引列表，默认绘制所有样本')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='结果目录（包含 splits_*.csv），用于获取 patient ids')
    args = parser.parse_args()
    
    # 确定输出目录
    output_dir = args.output_dir or os.path.join(args.features_dir, 'heatmaps')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载特征
    print(f"📦 加载特征...")
    original_features, aligned_features, modalities = load_features(args.features_dir, args.fold_idx)
    
    # 确定样本数量
    n_samples = None
    for modality in modalities:
        if modality in original_features:
            n_samples = original_features[modality].shape[0]
            break
        if modality in aligned_features:
            n_samples = aligned_features[modality].shape[0]
            break
    
    if n_samples is None:
        raise ValueError('无法确定样本数量')
    
    print(f"📊 找到 {n_samples} 个样本，{len(modalities)} 个模态")
    print(f"📋 模态列表: {modalities}")
    
    # 加载 patient ids（如果提供了 results_dir）
    patient_ids = None
    if args.results_dir:
        print(f"📋 加载 patient ids...")
        patient_ids = load_patient_ids(args.results_dir, args.fold_idx)
        if patient_ids:
            print(f"✅ 找到 {len(patient_ids)} 个 patient ids")
        else:
            print(f"⚠️ 未找到 patient ids，将使用 sample_idx")
    
    # 确定要绘制的样本索引
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        sample_indices = list(range(n_samples))
    
    print(f"🎨 开始绘制 {len(sample_indices)} 个样本的热力图...")
    
    # 遍历所有样本并绘制热力图
    for sample_idx in sample_indices:
        if sample_idx >= n_samples:
            print(f"⚠️ 跳过样本 {sample_idx}（超出范围）")
            continue
        
        # 获取 patient id
        patient_id = None
        if patient_ids and sample_idx < len(patient_ids):
            patient_id = patient_ids[sample_idx]
            # 移除 'patient_' 前缀（如果存在）
            if patient_id.startswith('patient_'):
                patient_id = patient_id.replace('patient_', '')
        
        output_path = os.path.join(output_dir, f'sample_{sample_idx}_heatmap.png')
        plot_sample_heatmap(
            original_features=original_features,
            aligned_features=aligned_features,
            modalities=modalities,
            sample_idx=sample_idx,
            output_path=output_path,
            patient_id=patient_id,
        )
    
    print(f"✅ 完成！所有热力图已保存到: {output_dir}")


if __name__ == '__main__':
    main()

