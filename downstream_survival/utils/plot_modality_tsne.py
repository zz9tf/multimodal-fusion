#!/usr/bin/env python3
"""
模态点 t-SNE 可视化工具

每个模态是一个点，同样的样本索引颜色相同，显示对齐前后的变化。

用法示例：
python -m downstream_survival.utils.plot_modality_tsne \
  --features_dir /path/to/svd_features \
  --fold_idx 0 \
  --output_dir /path/to/output
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import matplotlib.cm as cm
from matplotlib.patches import Patch


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


def prepare_modality_points(
    features: Dict[str, np.ndarray],
    modalities: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    准备模态点数据。
    
    对于每个样本，每个模态是一个点。
    返回所有点的特征矩阵和对应的模态索引、样本索引。
    
    Args:
        features: 特征字典，每个模态为 [N, D]
        modalities: 模态列表
        
    Returns:
        (points_matrix, modality_indices, sample_indices)
        - points_matrix: [N * n_modalities, D] 所有点的特征矩阵
        - modality_indices: [N * n_modalities] 每个点对应的模态索引
        - sample_indices: [N * n_modalities] 每个点对应的样本索引（patient id）
    """
    n_samples = None
    n_modalities = len(modalities)
    
    # 确定样本数量
    for modality in modalities:
        if modality in features:
            n_samples = features[modality].shape[0]
            break
    
    if n_samples is None:
        raise ValueError('无法确定样本数量')
    
    # 构建所有点的特征矩阵
    points_list = []
    modality_indices_list = []
    sample_indices_list = []
    
    for sample_idx in range(n_samples):
        for modality_idx, modality in enumerate(modalities):
            if modality in features:
                # 每个模态是一个点
                point_feature = features[modality][sample_idx]  # [D]
                points_list.append(point_feature)
                modality_indices_list.append(modality_idx)
                sample_indices_list.append(sample_idx)
    
    points_matrix = np.stack(points_list, axis=0)  # [N * n_modalities, D]
    modality_indices = np.array(modality_indices_list)  # [N * n_modalities]
    sample_indices = np.array(sample_indices_list)  # [N * n_modalities]
    
    return points_matrix, modality_indices, sample_indices


def plot_modality_tsne(
    original_features: Dict[str, np.ndarray],
    aligned_features: Dict[str, np.ndarray],
    modalities: List[str],
    output_path: str,
    method: str = 'tsne',
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """
    绘制模态点的降维可视化（PCA 或 t-SNE）。
    
    Args:
        original_features: 对齐前的特征字典
        aligned_features: 对齐后的特征字典
        modalities: 模态列表
        output_path: 输出路径
        method: 降维方法，'pca' 或 'tsne'（默认 'tsne'）
        perplexity: t-SNE 的 perplexity 参数（仅当 method='tsne' 时使用）
        random_state: 随机种子
    """
    # 准备数据
    original_points, original_modality_indices, original_sample_indices = prepare_modality_points(original_features, modalities)
    aligned_points, aligned_modality_indices, aligned_sample_indices = prepare_modality_points(aligned_features, modalities)
    
    print(f"📊 原始特征点: {original_points.shape}, 对齐后特征点: {aligned_points.shape}")
    
    # 设置绘图风格
    plt.rcParams.update({
        'font.size': 10,
        'figure.dpi': 150,
    })
    
    # 创建图形：左右对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 对原始特征进行降维
    if method.lower() == 'pca':
        print(f"🔄 对原始特征进行 PCA 降维...")
        reducer_original = PCA(n_components=2, random_state=random_state)
        original_2d = reducer_original.fit_transform(original_points)
        
        print(f"🔄 对对齐后特征进行 PCA 降维...")
        reducer_aligned = PCA(n_components=2, random_state=random_state)
        aligned_2d = reducer_aligned.fit_transform(aligned_points)
        
        method_name = 'PCA'
        xlabel = 'PCA Component 1'
        ylabel = 'PCA Component 2'
    elif method.lower() == 'tsne':
        print(f"🔄 对原始特征进行 t-SNE 降维...")
        reducer_original = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
        original_2d = reducer_original.fit_transform(original_points)
        
        print(f"🔄 对对齐后特征进行 t-SNE 降维...")
        reducer_aligned = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
        aligned_2d = reducer_aligned.fit_transform(aligned_points)
        
        method_name = 't-SNE'
        xlabel = 't-SNE Component 1'
        ylabel = 't-SNE Component 2'
    else:
        raise ValueError(f"不支持的降维方法: {method}，支持 'pca' 或 'tsne'")
    
    # 为每个模态分配不同的颜色
    n_modalities = len(modalities)
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_modalities))
    modality_color_map = {i: colors[i] for i in range(n_modalities)}
    
    # 创建图例句柄（两个子图共用）
    legend_handles = [
        Patch(facecolor=modality_color_map[i], edgecolor='black', linewidth=0.5, alpha=0.6, label=modalities[i])
        for i in range(n_modalities)
    ]
    
    # 绘制原始特征
    ax1 = axes[0]
    for modality_idx in range(n_modalities):
        mask = original_modality_indices == modality_idx
        modality_points = original_2d[mask]
        sample_ids = original_sample_indices[mask]
        color = modality_color_map[modality_idx]
        
        ax1.scatter(modality_points[:, 0], modality_points[:, 1],
                   c=[color], s=100, alpha=0.6,
                   edgecolors='black', linewidth=0.5)
        
        # 标注patient id
        for i, (point, patient_id) in enumerate(zip(modality_points, sample_ids)):
            ax1.annotate(f'P{patient_id}', 
                        xy=(point[0], point[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
    
    ax1.set_title('(a) Original Features', fontsize=30, fontweight='bold', pad=10)
    ax1.set_xlabel(xlabel, fontsize=26)
    ax1.set_ylabel(ylabel, fontsize=26)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)
    
    # 绘制对齐后特征
    ax2 = axes[1]
    for modality_idx in range(n_modalities):
        mask = aligned_modality_indices == modality_idx
        modality_points = aligned_2d[mask]
        sample_ids = aligned_sample_indices[mask]
        color = modality_color_map[modality_idx]
        
        ax2.scatter(modality_points[:, 0], modality_points[:, 1],
                   c=[color], s=100, alpha=0.6,
                   edgecolors='black', linewidth=0.5)
        
        # 标注patient id
        for i, (point, patient_id) in enumerate(zip(modality_points, sample_ids)):
            ax2.annotate(f'P{patient_id}', 
                        xy=(point[0], point[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
    
    ax2.set_title('(b) Aligned Features', fontsize=30, fontweight='bold', pad=10)
    ax2.set_xlabel(xlabel, fontsize=26)
    ax2.set_ylabel(ylabel, fontsize=26)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ 保存 {method_name} 可视化: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='模态点 t-SNE 可视化工具')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='特征文件目录（包含 fold_*_*_original.npy 和 fold_*_*_aligned.npy）')
    parser.add_argument('--fold_idx', type=int, default=0,
                       help='要处理的 fold 索引（默认 0）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，默认使用 features_dir/tsne_modality')
    parser.add_argument('--method', type=str, default='tsne', choices=['pca', 'tsne'],
                       help='降维方法：pca 或 tsne（默认 tsne）')
    parser.add_argument('--perplexity', type=float, default=30.0,
                       help='t-SNE 的 perplexity 参数（默认 30.0，仅当 method=tsne 时使用）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子（默认 42）')
    args = parser.parse_args()
    
    # 确定输出目录
    method_name = args.method.lower()
    output_dir = args.output_dir or os.path.join(args.features_dir, f'{method_name}_modality')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载特征
    print(f"📦 加载特征...")
    original_features, aligned_features, modalities = load_features(args.features_dir, args.fold_idx)
    
    print(f"📊 找到 {len(modalities)} 个模态")
    print(f"📋 模态列表: {modalities}")
    
    # 绘制降维可视化
    print(f"🎨 开始绘制模态点 {method_name.upper()} 可视化...")
    output_path = os.path.join(output_dir, f'fold_{args.fold_idx}_modality_{method_name}.png')
    
    plot_modality_tsne(
        original_features=original_features,
        aligned_features=aligned_features,
        modalities=modalities,
        output_path=output_path,
        method=args.method,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )
    
    print(f"✅ 完成！{method_name.upper()} 可视化已保存到: {output_path}")


if __name__ == '__main__':
    main()

