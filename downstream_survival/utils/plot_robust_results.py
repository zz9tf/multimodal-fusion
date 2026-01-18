#!/usr/bin/env python3
"""
绘制缺模态鲁棒性评测结果的箱线图（适合论文发表）

功能：
- 读取不同 drop_prob 的评测结果 JSON 文件
- 绘制每个 drop_prob 下所有 fold 的箱线图
- 美观的学术风格图表
"""

import argparse
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple

# 设置学术论文风格的 matplotlib 参数
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,  # 坐标轴标签字体
    'axes.titlesize': 16,   # 子图标题字体
    'xtick.labelsize': 11,  # X轴刻度标签字体
    'ytick.labelsize': 11,  # Y轴刻度标签字体
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

def load_results_from_dir(results_dir: str) -> Tuple[List[float], Dict[float, List[float]], Dict[float, List[float]]]:
    """
    从结果目录加载所有 drop_prob 的评测结果
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        (drop_probs, auc_data, acc_data)
        - drop_probs: drop_prob 值列表
        - auc_data: {drop_prob: [fold1_auc, fold2_auc, ...]}
        - acc_data: {drop_prob: [fold1_acc, fold2_acc, ...]}
    """
    pattern = os.path.join(results_dir, 'robust_missing_drop_prob_*.json')
    json_files = sorted(glob.glob(pattern))
    
    drop_probs = []
    auc_data = {}
    acc_data = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                drop_prob = data.get('drop_prob')
                if drop_prob is None:
                    continue
                
                drop_probs.append(drop_prob)
                
                # 提取每个 fold 的数据
                per_fold = data.get('per_fold', [])
                auc_values = [item['test_auc'] for item in per_fold if 'test_auc' in item]
                acc_values = [item['test_acc'] for item in per_fold if 'test_acc' in item]
                
                auc_data[drop_prob] = auc_values
                acc_data[drop_prob] = acc_values
                
        except Exception as e:
            print(f"⚠️ 读取文件失败 {json_file}: {e}")
            continue
    
    # 按 drop_prob 排序
    drop_probs = sorted(set(drop_probs))
    return drop_probs, auc_data, acc_data

def plot_boxplot(drop_probs: List[float], auc_data: Dict[float, List[float]], 
                 acc_data: Dict[float, List[float]], output_path: str = None):
    """
    绘制箱线图（学术论文风格）
    
    Args:
        drop_probs: drop_prob 值列表
        auc_data: {drop_prob: [fold1_auc, fold2_auc, ...]}
        acc_data: {drop_prob: [fold1_acc, fold2_acc, ...]}
        output_path: 输出图片路径（可选）
    """
    # 准备数据
    auc_list = [auc_data[dp] for dp in drop_probs]
    acc_list = [acc_data[dp] for dp in drop_probs]
    
    # 创建图形（增大尺寸以适应更大的字体）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 设置颜色方案（学术风格）
    box_colors = ['#4A90E2', '#50C878', '#FF6B6B', '#FFD93D', '#95E1D3', 
                  '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF', '#FFD3A5', '#C7CEEA']
    
    # 绘制 AUC 箱线图（不显示异常值）
    bp1 = ax1.boxplot(auc_list, positions=range(len(drop_probs)), widths=0.6,
                      patch_artist=True, showmeans=True, meanline=True,
                      showfliers=False,  # 不显示异常值（黑点）
                      boxprops=dict(linewidth=1.5, facecolor='white'),
                      medianprops=dict(linewidth=2, color='#2C3E50'),
                      meanprops=dict(linewidth=1.5, linestyle='--', color='#E74C3C'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    
    # 美化箱线图
    for patch in bp1['boxes']:
        patch.set_facecolor('#E8F4F8')
        patch.set_edgecolor('#3498DB')
        patch.set_alpha(0.8)
    
    ax1.set_xlabel('Drop Probability', fontsize=16)
    ax1.set_ylabel('Test AUC', fontsize=16)
    ax1.set_title('(a) Test AUC', fontweight='bold', pad=10, fontsize=20)
    ax1.set_xticks(range(len(drop_probs)))
    ax1.set_xticklabels([f'{dp:.1f}' for dp in drop_probs], fontsize=11)
    ax1.set_ylim([0.2, 1.0])
    ax1.set_yticks(np.arange(0.2, 1.1, 0.1))
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random (0.5)')
    
    # 绘制 ACC 箱线图（不显示异常值）
    bp2 = ax2.boxplot(acc_list, positions=range(len(drop_probs)), widths=0.6,
                      patch_artist=True, showmeans=True, meanline=True,
                      showfliers=False,  # 不显示异常值（黑点）
                      boxprops=dict(linewidth=1.5, facecolor='white'),
                      medianprops=dict(linewidth=2, color='#2C3E50'),
                      meanprops=dict(linewidth=1.5, linestyle='--', color='#E74C3C'),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    
    # 美化箱线图
    for patch in bp2['boxes']:
        patch.set_facecolor('#FFF4E6')
        patch.set_edgecolor('#F39C12')
        patch.set_alpha(0.8)
    
    ax2.set_xlabel('Drop Probability', fontsize=16)
    ax2.set_ylabel('Test Accuracy', fontsize=16)
    ax2.set_title('(b) Test Accuracy', fontweight='bold', pad=10, fontsize=20)
    ax2.set_xticks(range(len(drop_probs)))
    ax2.set_xticklabels([f'{dp:.1f}' for dp in drop_probs], fontsize=11)
    ax2.set_ylim([0.2, 1.0])
    ax2.set_yticks(np.arange(0.2, 1.1, 0.1))
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加图例（无阴影）- 两个子图都要有图例
    median_line = mpatches.Rectangle((0, 0), 1, 1, fc='#2C3E50', linewidth=2)
    mean_line = mpatches.Rectangle((0, 0), 1, 1, fc='#E74C3C', linewidth=1.5, linestyle='--')
    ax1.legend([median_line, mean_line], ['Median', 'Mean'], 
              loc='upper right', frameon=True, fancybox=False, shadow=False, fontsize=12)
    ax2.legend([median_line, mean_line], ['Median', 'Mean'], 
              loc='upper right', frameon=True, fancybox=False, shadow=False, fontsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = 'robust_results_boxplot.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✅ 图片已保存到: {output_path}')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='绘制缺模态鲁棒性评测结果箱线图（学术论文风格）')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='结果目录路径（包含 robust_missing_drop_prob_*.json 文件）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，默认为 results_dir/robust_results_boxplot.png）')
    
    args = parser.parse_args()
    
    # 加载结果
    print(f'📂 从目录加载结果: {args.results_dir}')
    drop_probs, auc_data, acc_data = load_results_from_dir(args.results_dir)
    
    if not drop_probs:
        print('❌ 未找到任何结果文件')
        return
    
    print(f'✅ 找到 {len(drop_probs)} 个 drop_prob 值')
    for dp in drop_probs:
        print(f'   drop_prob={dp:.1f}: {len(auc_data[dp])} folds, '
              f'AUC={np.mean(auc_data[dp]):.4f}±{np.std(auc_data[dp]):.4f}, '
              f'ACC={np.mean(acc_data[dp]):.4f}±{np.std(acc_data[dp]):.4f}')
    
    # 确定输出路径
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.results_dir, 'robust_results_boxplot.png')
    
    # 绘制箱线图
    plot_boxplot(drop_probs, auc_data, acc_data, output_path)

if __name__ == '__main__':
    main()

