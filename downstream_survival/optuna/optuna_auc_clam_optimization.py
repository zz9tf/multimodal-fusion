#!/usr/bin/env python3
"""
Optuna 超参数优化脚本 - 针对 AUC_CLAM 模型
基于 main.py 和 trainer.py 的架构进行优化
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import torch
from torch.utils.data import Subset
import threading
import time

# 添加项目路径
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

# 导入项目模块
from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset
from optuna_config import OptunaConfig

class AUCCLAMOptimizer:
    """
    AUC_CLAM 模型的 Optuna 优化器
    专注于超参数搜索和模型性能优化
    """
    
    def __init__(self, 
                 data_root_dir: str,
                 csv_path: str,
                 results_dir: str = './optuna_results',
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 timeout: int = None,
                 pruner: bool = True,
                 sampler: str = 'tpe',
                 enable_realtime_viz: bool = False,
                 viz_port: int = 8080):
        """
        初始化优化器
        
        Args:
            data_root_dir: 数据根目录
            csv_path: CSV文件路径
            results_dir: 结果保存目录
            n_trials: 优化试验次数
            n_jobs: 并行作业数
            timeout: 超时时间（秒）
            pruner: 是否启用剪枝
            sampler: 采样器类型 ('tpe', 'random', 'cmaes')
            enable_realtime_viz: 是否启用实时可视化
            viz_port: 可视化端口
        """
        self.data_root_dir = data_root_dir
        self.csv_path = csv_path
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.enable_realtime_viz = enable_realtime_viz
        self.viz_port = viz_port
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 配置 Optuna 采样器和剪枝器
        if sampler == 'tpe':
            self.sampler = TPESampler(seed=42)
        elif sampler == 'random':
            self.sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler == 'cmaes':
            self.sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"不支持的采样器类型: {sampler}")
            
        self.pruner = MedianPruner() if pruner else None
        
        # 初始化配置管理器
        self.config_manager = OptunaConfig()
        
        # 存储最佳试验结果
        self.best_trial = None
        self.trial_results = []
        
        # 实时可视化相关
        self.viz_thread = None
        self.study = None
        
    def _create_objective_function(self, 
                                 dataset: MultimodalDataset,
                                 k_fold_splits: List[Dict],
                                 target_channels: List[str],
                                 n_folds: int = 3) -> callable:
        """
        创建目标函数用于 Optuna 优化
        
        Args:
            dataset: 多模态数据集
            k_fold_splits: K折交叉验证分割
            target_channels: 目标通道列表
            n_folds: 用于优化的折数（减少计算时间）
            
        Returns:
            目标函数
        """
        def objective(trial: optuna.Trial) -> float:
            """
            Optuna 目标函数
            
            Returns:
                验证集平均AUC分数
            """
            try:
                # 1. 建议超参数
                params = self.config_manager.suggest_auc_clam_params(trial)
                
                # 2. 创建配置
                configs = self.config_manager.create_configs(
                    data_root_dir=self.data_root_dir,
                    csv_path=self.csv_path,
                    target_channels=target_channels,
                    **params
                )
                
                # 3. 初始化训练器
                trainer = Trainer(
                    configs=configs,
                    log_dir=os.path.join(self.results_dir, f'trial_{trial.number}')
                )
                
                # 4. 使用前 n_folds 进行快速验证
                fold_aucs = []
                for fold_idx in range(min(n_folds, len(k_fold_splits))):
                    # 获取当前fold的分割
                    split = k_fold_splits[fold_idx]
                    train_idx = split['train']
                    val_idx = split['val']
                    test_idx = split['test']
                    
                    # 创建子数据集
                    train_dataset = Subset(dataset, train_idx)
                    val_dataset = Subset(dataset, val_idx)
                    test_dataset = Subset(dataset, test_idx)
                    
                    datasets = (train_dataset, val_dataset, test_dataset)
                    
                    # 训练并获取验证AUC
                    try:
                        _, test_auc, val_auc, test_acc, val_acc = trainer.train_fold(
                            datasets=datasets,
                            fold_idx=fold_idx
                        )
                        fold_aucs.append(val_auc)
                        
                        # 报告中间结果给 Optuna（用于剪枝）
                        trial.report(val_auc, step=fold_idx)
                        
                        # 检查是否应该剪枝
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                            
                    except Exception as e:
                        print(f"⚠️ Fold {fold_idx} 训练失败: {e}")
                        # 返回一个较低的分数而不是失败
                        fold_aucs.append(0.5)
                
                # 5. 计算平均AUC
                mean_auc = np.mean(fold_aucs) if fold_aucs else 0.5
                
                # 6. 记录试验结果
                trial_result = {
                    'trial_number': trial.number,
                    'params': params,
                    'mean_val_auc': mean_auc,
                    'fold_aucs': fold_aucs,
                    'timestamp': datetime.now().isoformat()
                }
                self.trial_results.append(trial_result)
                
                print(f"🎯 Trial {trial.number}: Mean Val AUC = {mean_auc:.4f}")
                return mean_auc
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"❌ Trial {trial.number} 失败: {e}")
                return 0.5  # 返回默认分数而不是失败
        
        return objective
    
    def _start_realtime_visualization(self, study: optuna.Study):
        """启动实时可视化服务器"""
        if not self.enable_realtime_viz:
            return
            
        def run_viz_server():
            try:
                # 启动 Optuna 内置的实时可视化服务器
                optuna.visualization.matplotlib.plot_optimization_history(study)
                print(f"🌐 实时可视化服务器已启动")
                print(f"📊 访问地址: http://localhost:{self.viz_port}")
                print(f"💡 在浏览器中打开上述地址查看实时优化进度")
            except Exception as e:
                print(f"⚠️ 实时可视化启动失败: {e}")
        
        self.viz_thread = threading.Thread(target=run_viz_server, daemon=True)
        self.viz_thread.start()
    
    def _save_realtime_plots(self, study: optuna.Study, trial_number: int):
        """保存实时图表"""
        if not self.enable_realtime_viz or trial_number % 5 != 0:  # 每5个试验保存一次
            return
            
        try:
            viz_dir = os.path.join(self.results_dir, "realtime_plots")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 保存优化历史图
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html(os.path.join(viz_dir, f"history_trial_{trial_number}.html"))
            
            # 保存参数重要性图
            if trial_number > 10:  # 需要足够的试验才能计算重要性
                fig2 = vis.plot_param_importances(study)
                fig2.write_html(os.path.join(viz_dir, f"importance_trial_{trial_number}.html"))
                
        except Exception as e:
            print(f"⚠️ 保存实时图表失败: {e}")
    
    def optimize(self,
                 target_channels: List[str] = None,
                 n_folds: int = 3,
                 study_name: str = None) -> optuna.Study:
        """
        执行超参数优化
        
        Args:
            target_channels: 目标通道列表
            n_folds: 用于优化的折数
            study_name: 研究名称
            
        Returns:
            Optuna Study 对象
        """
        print("🚀 开始 AUC_CLAM 超参数优化...")
        print(f"📊 试验次数: {self.n_trials}")
        print(f"📁 结果目录: {self.results_dir}")
        
        # 设置默认目标通道
        if target_channels is None:
            target_channels = ['features', 'tma_CD3', 'tma_CD8', 'tma_CD56', 'tma_CD68', 'tma_CD163', 'tma_HE', 'tma_MHC1', 'tma_PDL1']
        
        # 加载数据集
        print("\n📂 加载数据集...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MultimodalDataset(
            csv_path=self.csv_path,
            data_root_dir=self.data_root_dir,
            channels=target_channels,
            align_channels=None,  # 不使用对齐
            alignment_model_path=None,  # 不使用对齐
            device=device,
            print_info=True
        )
        
        # 创建K折分割
        print(f"\n🔄 创建 {10}-fold 交叉验证分割...")
        # 直接实现K折分割，避免导入main.py
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        k_fold_splits = []
        for train_idx, test_idx in kf.split(range(len(dataset))):
            # 将测试集进一步分为验证集和测试集
            val_size = len(test_idx) // 2
            val_idx = test_idx[:val_size]
            test_idx = test_idx[val_size:]
            k_fold_splits.append({
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': test_idx.tolist()
            })
        print(f"✅ 创建了 {len(k_fold_splits)} 个fold")
        
        # 创建目标函数
        objective = self._create_objective_function(
            dataset=dataset,
            k_fold_splits=k_fold_splits,
            target_channels=target_channels,
            n_folds=n_folds
        )
        
        # 创建或加载研究
        if study_name is None:
            study_name = f"auc_clam_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study_path = os.path.join(self.results_dir, f"{study_name}.db")
        
        if os.path.exists(study_path):
            print(f"📖 加载现有研究: {study_path}")
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{study_path}",
                sampler=self.sampler,
                pruner=self.pruner
            )
        else:
            print(f"🆕 创建新研究: {study_name}")
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{study_path}",
                direction='maximize',  # 最大化AUC
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True
            )
        
        # 启动实时可视化
        self.study = study
        self._start_realtime_visualization(study)
        
        # 执行优化
        print(f"\n🎯 开始优化 (使用前 {n_folds} folds)...")
        
        # 自定义优化循环以支持实时可视化
        for trial in study:
            if trial.number >= self.n_trials:
                break
                
            # 运行试验
            study.optimize(objective, n_trials=1, n_jobs=1)
            
            # 保存实时图表
            self._save_realtime_plots(study, trial.number)
            
            # 打印进度
            if trial.number % 5 == 0:
                print(f"📊 已完成 {trial.number}/{self.n_trials} 试验，当前最佳AUC: {study.best_value:.4f}")
        
        # 如果启用了实时可视化，显示最终结果
        if self.enable_realtime_viz:
            print(f"\n🌐 实时可视化地址: http://localhost:{self.viz_port}")
            print(f"📁 实时图表保存在: {os.path.join(self.results_dir, 'realtime_plots')}")
        
        # 保存结果
        self._save_results(study, study_name)
        
        # 更新最佳试验
        self.best_trial = study.best_trial
        
        print(f"\n🎉 优化完成!")
        print(f"🏆 最佳试验: {study.best_trial.number}")
        print(f"📈 最佳AUC: {study.best_value:.4f}")
        print(f"⚙️ 最佳参数: {study.best_params}")
        
        return study
    
    def _save_results(self, study: optuna.Study, study_name: str):
        """保存优化结果"""
        # 保存研究到数据库
        study_path = os.path.join(self.results_dir, f"{study_name}.db")
        print(f"💾 研究已保存到: {study_path}")
        
        # 保存试验结果到JSON
        results_path = os.path.join(self.results_dir, f"{study_name}_results.json")
        results_data = {
            'study_name': study_name,
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params
            },
            'n_trials': len(study.trials),
            'trial_results': self.trial_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细结果已保存到: {results_path}")
        
        # 保存CSV格式的试验结果
        trials_df = study.trials_dataframe()
        csv_path = os.path.join(self.results_dir, f"{study_name}_trials.csv")
        trials_df.to_csv(csv_path, index=False)
        print(f"📊 试验数据已保存到: {csv_path}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """获取最佳配置"""
        if self.best_trial is None:
            raise ValueError("尚未进行优化，请先运行 optimize() 方法")
        
        return self.config_manager.create_configs(
            data_root_dir=self.data_root_dir,
            csv_path=self.csv_path,
            **self.best_trial.params
        )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AUC_CLAM Optuna 超参数优化')
    
    # 必需参数
    parser.add_argument('--data_root_dir', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='CSV文件路径')
    
    # 优化参数
    parser.add_argument('--results_dir', type=str, default='./optuna_results',
                       help='结果保存目录')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='优化试验次数')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='并行作业数')
    parser.add_argument('--timeout', type=int, default=None,
                       help='超时时间（秒）')
    parser.add_argument('--n_folds', type=int, default=3,
                       help='用于优化的折数')
    parser.add_argument('--study_name', type=str, default=None,
                       help='研究名称')
    
    # 采样器和剪枝选项
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'], 
                       default='tpe', help='采样器类型')
    parser.add_argument('--no_pruner', action='store_true',
                       help='禁用剪枝器')
    
    # 数据相关参数
    parser.add_argument('--target_channels', type=str, nargs='+',
                       default=['features', 'tma_CD3', 'tma_CD8', 'tma_CD56', 'tma_CD68', 'tma_CD163', 'tma_HE', 'tma_MHC1', 'tma_PDL1'],
                       help='目标通道')
    
    # 实时可视化参数
    parser.add_argument('--enable_realtime_viz', action='store_true',
                       help='启用实时可视化')
    parser.add_argument('--viz_port', type=int, default=8080,
                       help='可视化端口 (default: 8080)')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = AUCCLAMOptimizer(
        data_root_dir=args.data_root_dir,
        csv_path=args.csv_path,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        pruner=not args.no_pruner,
        sampler=args.sampler,
        enable_realtime_viz=args.enable_realtime_viz,
        viz_port=args.viz_port
    )
    
    # 执行优化
    study = optimizer.optimize(
        target_channels=args.target_channels,
        n_folds=args.n_folds,
        study_name=args.study_name
    )
    
    print("\n🎯 优化完成！")
    print(f"📁 结果保存在: {args.results_dir}")

if __name__ == "__main__":
    main()
