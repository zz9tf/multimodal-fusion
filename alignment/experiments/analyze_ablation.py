#!/usr/bin/env python3
"""
Ablation Study 结果分析脚本
快速分析和可视化所有 ablation study 的结果
"""

import json
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(ablation_dir):
    """
    加载某个 ablation study 的所有结果
    
    Args:
        ablation_dir: ablation study 目录路径
        
    Returns:
        dict: {参数值: 最佳验证loss}
    """
    results = {}
    history_files = glob.glob(f"{ablation_dir}/*.history.json")
    
    for f in history_files:
        try:
            with open(f, 'r') as file:
                history = json.load(file)
                
                if not history['val_losses']:
                    print(f"⚠️  警告: {f} 没有验证数据")
                    continue
                
                config = history['config']
                best_loss = min(history['val_losses'])
                
                # 根据目录名确定参数名
                dir_name = Path(ablation_dir).name
                param_name = dir_name.replace('ablation_', '')
                
                param_value = config.get(param_name)
                if param_value is not None:
                    results[param_value] = {
                        'best_val_loss': best_loss,
                        'final_train_loss': history['train_losses'][-1],
                        'num_steps': len(history['train_losses']),
                    }
        except Exception as e:
            print(f"❌ 读取 {f} 失败: {e}")
    
    return results


def plot_ablation(results, param_name, save_path=None):
    """
    绘制 ablation study 结果图
    
    Args:
        results: 结果字典
        param_name: 参数名称
        save_path: 保存路径（可选）
    """
    if not results:
        print(f"⚠️  {param_name} 没有结果数据")
        return
    
    # 排序
    param_values = sorted(results.keys())
    val_losses = [results[p]['best_val_loss'] for p in param_values]
    train_losses = [results[p]['final_train_loss'] for p in param_values]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 验证 loss
    ax1.plot(param_values, val_losses, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Best Validation Loss', fontsize=12)
    ax1.set_title(f'{param_name} Ablation Study - Validation Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 标注最优值
    min_idx = np.argmin(val_losses)
    best_param = param_values[min_idx]
    best_loss = val_losses[min_idx]
    ax1.plot(best_param, best_loss, 'r*', markersize=20, label=f'Best: {param_name}={best_param}')
    ax1.legend()
    
    # 训练 loss
    ax2.plot(param_values, train_losses, marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel(param_name, fontsize=12)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title(f'{param_name} Ablation Study - Training Loss', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(results, param_name):
    """打印结果摘要"""
    if not results:
        print(f"⚠️  {param_name} 没有结果数据")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 {param_name.upper()} Ablation Study Summary")
    print(f"{'='*60}")
    
    # 按验证 loss 排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])
    
    print(f"\n{'参数值':<15} {'验证Loss':<15} {'训练Loss':<15} {'训练步数':<15}")
    print("-" * 60)
    
    for param_value, metrics in sorted_results:
        print(f"{str(param_value):<15} "
              f"{metrics['best_val_loss']:<15.4f} "
              f"{metrics['final_train_loss']:<15.4f} "
              f"{metrics['num_steps']:<15}")
    
    # 最优值
    best_param, best_metrics = sorted_results[0]
    print(f"\n🏆 最优参数值: {param_name}={best_param}")
    print(f"   - 最佳验证 Loss: {best_metrics['best_val_loss']:.4f}")
    print(f"   - 最终训练 Loss: {best_metrics['final_train_loss']:.4f}")
    
    # 统计信息
    all_val_losses = [m['best_val_loss'] for m in results.values()]
    print(f"\n📈 统计信息:")
    print(f"   - 平均验证 Loss: {np.mean(all_val_losses):.4f}")
    print(f"   - 标准差: {np.std(all_val_losses):.4f}")
    print(f"   - 最小值: {np.min(all_val_losses):.4f}")
    print(f"   - 最大值: {np.max(all_val_losses):.4f}")


def analyze_all_ablations(results_dir, output_dir=None):
    """分析所有 ablation studies"""
    results_dir = Path(results_dir)
    ablation_dirs = sorted(results_dir.glob('ablation_*'))
    
    if not ablation_dirs:
        print(f"❌ 在 {results_dir} 中未找到 ablation study 结果")
        return
    
    print(f"找到 {len(ablation_dirs)} 个 ablation studies:")
    for d in ablation_dirs:
        print(f"  - {d.name}")
    print()
    
    all_results = {}
    
    for ablation_dir in ablation_dirs:
        param_name = ablation_dir.name.replace('ablation_', '')
        print(f"\n{'='*60}")
        print(f"📂 分析 {param_name}...")
        print(f"{'='*60}")
        
        results = load_results(ablation_dir)
        all_results[param_name] = results
        
        if results:
            print_summary(results, param_name)
            
            # 保存图表
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"{param_name}_ablation.png"
                plot_ablation(results, param_name, save_path)
            else:
                plot_ablation(results, param_name)
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("🎯 所有 Ablation Studies 总结")
    print(f"{'='*60}\n")
    
    summary = []
    for param_name, results in all_results.items():
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])
            best_param, best_metrics = sorted_results[0]
            summary.append({
                'param': param_name,
                'best_value': best_param,
                'best_loss': best_metrics['best_val_loss']
            })
    
    summary.sort(key=lambda x: x['best_loss'])
    
    print(f"{'参数':<20} {'最优值':<15} {'最佳Loss':<15}")
    print("-" * 50)
    for s in summary:
        print(f"{s['param']:<20} {str(s['best_value']):<15} {s['best_loss']:<15.4f}")


def main():
    parser = argparse.ArgumentParser(description="分析 Ablation Study 结果")
    parser.add_argument("--results_dir", type=str, 
                       default="/home/zheng/zheng/multimodal-fusion/results",
                       help="结果目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="图表保存目录（不提供则显示图表）")
    parser.add_argument("--param", type=str, default=None,
                       help="只分析指定参数（不提供则分析所有）")
    
    args = parser.parse_args()
    
    if args.param:
        # 分析单个参数
        ablation_dir = Path(args.results_dir) / f"ablation_{args.param}"
        if not ablation_dir.exists():
            print(f"❌ 目录不存在: {ablation_dir}")
            return
        
        results = load_results(ablation_dir)
        print_summary(results, args.param)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{args.param}_ablation.png"
            plot_ablation(results, args.param, save_path)
        else:
            plot_ablation(results, args.param)
    else:
        # 分析所有参数
        analyze_all_ablations(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

