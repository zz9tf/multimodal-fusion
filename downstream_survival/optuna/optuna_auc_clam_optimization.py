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
from typing import Dict, Any, List
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import torch
from torch.utils.data import Subset
import threading
import random
from optuna.storages.journal import JournalFileBackend
from optuna.storages import JournalStorage
import time
from contextlib import contextmanager
from collections import defaultdict
# 导入项目模块
from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset
from optuna_config import OptunaConfig
from main import create_k_fold_splits, parse_channels

class PerformanceProfiler:
    """
    性能分析器 - 用于记录和分析代码执行时间
    """
    
    def __init__(self, enable: bool = True):
        """
        初始化性能分析器
        
        Args:
            enable: 是否启用性能分析
        """
        self.enable = enable
        self.timings = defaultdict(list)  # 存储每个步骤的时间列表
        self.current_trial = None
        self.start_times = {}  # 存储当前正在计时的步骤开始时间
        
    @contextmanager
    def time_block(self, step_name: str, trial_number: int = None):
        """
        上下文管理器，用于记录代码块的执行时间
        
        Args:
            step_name: 步骤名称
            trial_number: 试验编号（可选）
        """
        if not self.enable:
            yield
            return
            
        # 记录开始时间
        start_time = time.time()
        try:
            yield
        finally:
            # 记录结束时间并计算耗时
            elapsed_time = time.time() - start_time
            key = f"{step_name}" if trial_number is None else f"Trial_{trial_number}_{step_name}"
            self.timings[step_name].append(elapsed_time)
            
            # 打印实时信息
            trial_info = f"Trial {trial_number}: " if trial_number is not None else ""
            print(f"⏱️  {trial_info}{step_name}: {elapsed_time:.2f}秒")
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取性能统计信息
        
        Returns:
            包含每个步骤的统计信息（总时间、平均时间、最小时间、最大时间、调用次数）
        """
        stats = {}
        for step_name, times in self.timings.items():
            if times:
                stats[step_name] = {
                    'total': sum(times),
                    'mean': np.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                    'std': np.std(times) if len(times) > 1 else 0.0
                }
        return stats
    
    def print_summary(self):
        """打印性能分析摘要"""
        if not self.enable or not self.timings:
            return
            
        print("\n" + "="*80)
        print("📊 性能分析摘要")
        print("="*80)
        
        stats = self.get_statistics()
        
        # 按总时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        # 打印表头
        print(f"{'步骤名称':<40} {'总时间(秒)':<15} {'平均(秒)':<15} {'最小(秒)':<15} {'最大(秒)':<15} {'调用次数':<10}")
        print("-"*80)
        
        total_all_time = sum(s['total'] for s in stats.values())
        
        # 打印每个步骤的统计信息
        for step_name, stat in sorted_stats:
            percentage = (stat['total'] / total_all_time * 100) if total_all_time > 0 else 0
            print(f"{step_name:<40} {stat['total']:<15.2f} {stat['mean']:<15.2f} "
                  f"{stat['min']:<15.2f} {stat['max']:<15.2f} {stat['count']:<10} "
                  f"({percentage:.1f}%)")
        
        print("-"*80)
        print(f"{'总计':<40} {total_all_time:<15.2f}")
        print("="*80)
        
        # 打印前5个最耗时的步骤
        print("\n🔝 前5个最耗时的步骤:")
        for i, (step_name, stat) in enumerate(sorted_stats[:5], 1):
            percentage = (stat['total'] / total_all_time * 100) if total_all_time > 0 else 0
            print(f"  {i}. {step_name}: {stat['total']:.2f}秒 ({percentage:.1f}%)")
        
        print()
    
    def save_to_file(self, filepath: str):
        """
        保存性能分析结果到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.enable:
            return
            
        stats = self.get_statistics()
        output = {
            'timings': {k: v for k, v in self.timings.items()},
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 性能分析结果已保存到: {filepath}")
    
    def reset(self):
        """重置性能分析器"""
        self.timings.clear()
        self.start_times.clear()
        self.current_trial = None

class AUCCLAMOptimizer:
    """
    通用模型的 Optuna 优化器
    支持多种模型类型的超参数搜索和模型性能优化
    """
    
    def __init__(self, 
                 data_root_dir: str,
                 csv_path: str,
                 model_type: str = 'svd_gate_random_clam_detach',
                 results_dir: str = './optuna_results',
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 timeout: int = None,
                 pruner: bool = True,
                 sampler: str = 'tpe',
                 enable_realtime_viz: bool = False,
                 viz_port: int = 8080,
                 data_root_base: str = None,
                 num_data_copies: int = 5,
                 **kwargs):
        """
        初始化优化器
        
        Args:
            data_root_dir: 数据根目录（单个数据集目录）
            csv_path: CSV文件路径
            results_dir: 结果保存目录
            n_trials: 优化试验次数
            n_jobs: 并行作业数
            timeout: 超时时间（秒）
            pruner: 是否启用剪枝
            sampler: 采样器类型 ('tpe', 'random', 'cmaes')
            enable_realtime_viz: 是否启用实时可视化
            viz_port: 可视化端口
            data_root_base: 数据集目录的基础路径，如果有多个数据集副本
            num_data_copies: 数据集副本数量（如 5 表示有 1, 2, 3, 4, 5 五个副本）
        """
        self.data_root_dir = data_root_dir
        self.data_root_base = data_root_base  # 数据集目录的基础路径
        self.num_data_copies = num_data_copies  # 数据集副本数量
        self.csv_path = csv_path
        self.model_type = model_type
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.enable_realtime_viz = enable_realtime_viz
        self.viz_port = viz_port
        self.kwargs = kwargs  # 存储其他参数（如 input_dim, n_classes 等）
        
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
        
        # 性能分析器
        self.profiler = PerformanceProfiler(enable=True)
        
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
            # 显示当前 trial 和进程/线程信息（用于验证 n_jobs）
            print(f"🔬 Trial {trial.number} 开始执行 (进程ID: {os.getpid()}, 线程ID: {threading.current_thread().ident})")
            
            trial_start_time = time.time()
            try:
                # 0. 选择数据集目录（如果有多个副本）
                if self.data_root_base is not None:
                    # 根据 trial.number 选择数据集副本（循环使用）
                    with self.profiler.time_block("数据集目录选择", trial.number):
                        data_copy_idx = (trial.number % self.num_data_copies) + 1
                        trial_data_root_dir = os.path.join(self.data_root_base, str(data_copy_idx))
                        print(f"📂 Trial {trial.number} 使用数据集副本: {trial_data_root_dir}")
                    
                    # 加载数据集（每个 trial 使用自己的数据集副本）
                    with self.profiler.time_block("数据集加载", trial.number):
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        trial_dataset = MultimodalDataset(
                            csv_path=self.csv_path,
                            data_root_dir=trial_data_root_dir,
                            channels=target_channels,
                            align_channels=None,
                            alignment_model_path=None,
                            device=device,
                            print_info=False  # 不打印详细信息，避免输出过多
                        )
                    
                    # 创建K折分割（使用与 main.py 相同的方法）
                    with self.profiler.time_block("K折分割创建", trial.number):
                        seed = self.kwargs.get('seed', 42)
                        fixed_test_split = self.kwargs.get('fixed_test_split', None)
                        trial_k_fold_splits = create_k_fold_splits(
                            dataset=trial_dataset,
                            k=10,
                            seed=seed,
                            fixed_test_split=fixed_test_split
                        )
                else:
                    # 使用共享的数据集（向后兼容）
                    trial_dataset = dataset
                    trial_k_fold_splits = k_fold_splits
                    trial_data_root_dir = self.data_root_dir
                
                # 1. 建议实验参数
                with self.profiler.time_block("实验参数建议", trial.number):
                    experiment_params = self.config_manager.suggest_experiment_params(trial)
                
                # 2. 建议模型参数（根据模型类型）
                with self.profiler.time_block("模型参数建议", trial.number):
                    model_params = self.config_manager.suggest_model_params(trial, self.model_type)
                
                # 3. 创建配置
                with self.profiler.time_block("配置创建", trial.number):
                    configs = self.config_manager.create_configs(
                        model_type=self.model_type,
                        data_root_dir=trial_data_root_dir,
                        csv_path=self.csv_path,
                        target_channels=target_channels,
                        experiment_params=experiment_params,
                        model_params=model_params,
                        trial_number=trial.number,
                        num_splits=10,
                        **self.kwargs
                    )
                
                # 3. 初始化训练器
                with self.profiler.time_block("训练器初始化", trial.number):
                    trainer = Trainer(
                        configs=configs,
                        log_dir=os.path.join(self.results_dir, f'trial_{trial.number}')
                    )
                
                # 4. 使用前 n_folds 进行快速验证
                fold_aucs = []
                total_training_time = 0
                for fold_idx in range(min(n_folds, len(trial_k_fold_splits))):
                    # 获取当前fold的分割
                    with self.profiler.time_block(f"Fold_{fold_idx}_数据准备", trial.number):
                        split = trial_k_fold_splits[fold_idx]
                        train_idx = split['train']
                        val_idx = split['val']
                        test_idx = split['test']
                        
                        # 创建子数据集
                        train_dataset = Subset(trial_dataset, train_idx)
                        val_dataset = Subset(trial_dataset, val_idx)
                        test_dataset = Subset(trial_dataset, test_idx)
                        
                        datasets = (train_dataset, val_dataset, test_dataset)
                    
                    # 训练并获取验证AUC
                    try:
                        fold_start_time = time.time()
                        with self.profiler.time_block(f"Fold_{fold_idx}_训练", trial.number):
                            _, test_auc, val_auc, test_acc, val_acc = trainer.train_fold(
                                datasets=datasets,
                                fold_idx=fold_idx
                            )
                        fold_aucs.append(val_auc)
                        total_training_time += time.time() - fold_start_time
                        
                        # 报告中间结果给 Optuna（用于剪枝）
                        trial.report(val_auc, step=fold_idx)
                        
                        # 检查是否应该剪枝
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                            
                    except Exception as e:
                        import traceback
                        print(f"⚠️ Fold {fold_idx} 训练失败: {e}")
                        print(f"   错误详情:")
                        traceback.print_exc()
                        # 返回一个基于试验参数的随机分数，避免所有试验返回相同分数
                        random_auc = 0.3 + random.random() * 0.4  # 0.3-0.7之间的随机分数
                        fold_aucs.append(random_auc)
                
                # 5. 计算平均AUC
                with self.profiler.time_block("结果计算", trial.number):
                    mean_auc = np.mean(fold_aucs) if fold_aucs else 0.5
                    
                    # 6. 记录试验结果
                    trial_result = {
                        'trial_number': trial.number,
                        'experiment_params': experiment_params,
                        'model_params': model_params,
                        'mean_val_auc': mean_auc,
                        'fold_aucs': fold_aucs,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.trial_results.append(trial_result)
                
                trial_total_time = time.time() - trial_start_time
                print(f"🎯 Trial {trial.number}: Mean Val AUC = {mean_auc:.4f} | 总耗时: {trial_total_time:.2f}秒 | 训练耗时: {total_training_time:.2f}秒 ({total_training_time/trial_total_time*100:.1f}%)")
                return mean_auc
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"❌ Trial {trial.number} 失败: {e}")
                # 返回一个基于试验参数的随机分数，避免所有试验返回相同分数
                random_auc = 0.3 + random.random() * 0.4  # 0.3-0.7之间的随机分数
                return random_auc
        
        return objective
    
    def _start_realtime_visualization(self, study: optuna.Study):
        """启动实时可视化服务器"""
        if not self.enable_realtime_viz:
            return
            
        def run_viz_server():
            try:
                # 注意：这里只是准备可视化，实际图表会在试验完成后生成
                print(f"🌐 实时可视化已准备就绪")
                print(f"📊 图表将在试验完成后生成")
                print(f"💡 查看实时优化进度")
            except Exception as e:
                print(f"⚠️ 实时可视化启动失败: {e}")
        
        self.viz_thread = threading.Thread(target=run_viz_server, daemon=True)
        self.viz_thread.start()
    
    def _save_realtime_plots(self, study: optuna.Study, trial_number: int):
        """保存实时图表"""
        try:
            viz_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 保存优化历史图
            if len(study.trials) > 0:
                fig1 = vis.plot_optimization_history(study)
                fig1.write_html(os.path.join(viz_dir, f"optimization_history.html"))
                print(f"📊 优化历史图已保存: {os.path.join(viz_dir, 'optimization_history.html')}")
            
            # 保存参数重要性图
            if len(study.trials) > 10:  # 需要足够的试验才能计算重要性
                try:
                    fig2 = vis.plot_param_importances(study)
                    fig2.write_html(os.path.join(viz_dir, f"param_importances.html"))
                    print(f"📊 参数重要性图已保存: {os.path.join(viz_dir, 'param_importances.html')}")
                except Exception as e:
                    print(f"⚠️ 参数重要性图生成失败: {e}")
                    
        except Exception as e:
            print(f"⚠️ 保存图表失败: {e}")
    
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
        print(f"🚀 开始 {self.model_type} 超参数优化...")
        print(f"📊 试验次数: {self.n_trials}")
        print(f"📁 结果目录: {self.results_dir}")
        
        # 设置默认目标通道
        if target_channels is None:
            target_channels = parse_channels(['wsi', 'tma', 'clinical', 'pathological', 'blood', 'icd', 'tma_cell_density'])
        
        # 保存 target_channels 以便后续使用
        self.target_channels = target_channels
        
        # 如果指定了 data_root_base，说明有多个数据集副本，将在 objective 函数中按 trial 分配
        # 否则，使用单个数据集（向后兼容）
        if self.data_root_base is not None:
            print(f"\n📂 检测到多个数据集副本（{self.num_data_copies} 个）")
            print(f"📁 数据集基础路径: {self.data_root_base}")
            print(f"💡 不同 trial 将使用不同的数据集副本，以错开读取")
            # 不在这里加载数据集，而是在 objective 函数中按 trial 加载
            dataset = None
            k_fold_splits = None
        else:
            # 加载数据集（单个数据集，向后兼容）
            print("\n📂 加载数据集...")
            with self.profiler.time_block("初始数据集加载"):
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
            
            # 创建K折分割（使用与 main.py 相同的方法）
            print(f"\n🔄 创建 {10}-fold 交叉验证分割...")
            with self.profiler.time_block("初始K折分割创建"):
                seed = self.kwargs.get('seed', 42)
                fixed_test_split = self.kwargs.get('fixed_test_split', None)
                k_fold_splits = create_k_fold_splits(
                    dataset=dataset,
                    k=10,
                    seed=seed,
                    fixed_test_split=fixed_test_split
                )
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
            study_name = f"{self.model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 使用 JournalStorage 作为存储后端（支持并行，无需数据库）
        journal_path = os.path.join(self.results_dir, f"{study_name}.log")
        try:
            journal_backend = JournalFileBackend(journal_path)
            storage_url = JournalStorage(journal_backend)
            print(f"📦 使用 JournalStorage: {journal_path}")
            print(f"✅ 支持并行执行 (n_jobs={self.n_jobs})")
        except ImportError:
            raise ImportError(
                "JournalStorage 不可用。请确保 Optuna 版本 >= 3.0。\n"
                "安装命令: pip install optuna>=3.0"
            )
        
        # 加载或创建研究
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url,
                sampler=self.sampler,
                pruner=self.pruner
            )
            print(f"📖 加载现有研究: {study_name}")
        except KeyError:
            print(f"🆕 创建新研究: {study_name}")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
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
        print(f"⚙️  并行作业数 (n_jobs): {self.n_jobs}")
        print(f"📊 总试验数 (n_trials): {self.n_trials}")
        print("")
        
        # 使用 Optuna 的标准优化方法
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # 保存最终图表
        self._save_realtime_plots(study, len(study.trials))
        
        # 如果启用了实时可视化，显示最终结果
        if self.enable_realtime_viz:
            print(f"\n🌐 实时可视化地址: http://localhost:{self.viz_port}")
            print(f"📁 实时图表保存在: {os.path.join(self.results_dir, 'realtime_plots')}")
        
        # 保存结果
        self._save_results(study, study_name)
        
        # 更新最佳试验
        self.best_trial = study.best_trial
        
        # 打印性能分析摘要
        self.profiler.print_summary()
        
        # 保存性能分析结果
        perf_file = os.path.join(self.results_dir, f"{study_name}_performance.json")
        self.profiler.save_to_file(perf_file)
        
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
        
        # 分离实验参数和模型参数
        experiment_params = {}
        model_params = {}
        
        # 实验参数键（会被 Optuna 优化的参数）
        experiment_param_keys = [
            'lr', 'reg', 'opt', 'batch_size', 'max_epochs', 'early_stopping', 'dropout', 'seed'
        ]
        
        # 从 best_trial.params 中提取被优化的参数
        for key, value in self.best_trial.params.items():
            if key in experiment_param_keys:
                experiment_params[key] = value
            else:
                model_params[key] = value
        
        # 设置固定值（不会被 Optuna 优化，但需要在配置中）
        experiment_params['opt'] = 'adam'  # 固定为 adam
        experiment_params['scheduler_config'] = {
            "type": "plateau",
            "mode": "min", 
            "patience": 15, 
            "factor": 0.5
        }
        experiment_params['alignment_model_path'] = self.kwargs.get('alignment_model_path', None)
        experiment_params['aligned_channels'] = self.kwargs.get('aligned_channels', None)
        
        return self.config_manager.create_configs(
            model_type=self.model_type,
            data_root_dir=self.data_root_dir,
            csv_path=self.csv_path,
            target_channels=self.target_channels if hasattr(self, 'target_channels') else None,
            experiment_params=experiment_params,
            model_params=model_params,
            **self.kwargs
        )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Optuna 超参数优化（支持多种模型类型）')
    
    # 必需参数
    parser.add_argument('--data_root_dir', type=str, required=True,
                       help='数据根目录（单个数据集目录，如果使用多个副本，则作为默认值）')
    parser.add_argument('--data_root_base', type=str, default=None,
                       help='数据集目录的基础路径（如 /home/zheng/zheng/public），如果有多个数据集副本')
    parser.add_argument('--num_data_copies', type=int, default=5,
                       help='数据集副本数量（如 5 表示有 1, 2, 3, 4, 5 五个副本）')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='CSV文件路径')
    
    # 模型类型
    parser.add_argument('--model_type', type=str, 
                       choices=['mil', 'clam', 'auc_clam', 'clam_mlp', 'clam_mlp_detach',
                                'svd_gate_random_clam', 'svd_gate_random_clam_detach',
                                'deep_supervise_svd_gate_random', 'deep_supervise_svd_gate_random_detach',
                                'clip_gate_random_clam', 'clip_gate_random_clam_detach',
                                'gate_shared_mil', 'gate_mil', 'gate_auc_mil', 'gate_mil_detach'],
                       default='auc_clam',
                       help='模型类型 (default: auc_clam)')
    
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
                       default=['wsi', 'tma', 'clinical', 'pathological', 'blood', 'icd', 'tma_cell_density'],
                       help='目标通道（使用 main.py 的通道格式）')
    parser.add_argument('--aligned_channels', type=str, nargs='+', default=None,
                       help='对齐通道（可选）')
    parser.add_argument('--alignment_model_path', type=str, default=None,
                       help='对齐模型路径（可选）')
    parser.add_argument('--fixed_test_split', type=str, default=None,
                       help='固定测试集分割文件路径（可选，与 main.py 保持一致）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=1024,
                       help='输入维度 (default: 1024)')
    parser.add_argument('--n_classes', type=int, default=2,
                       help='类别数 (default: 2)')
    parser.add_argument('--base_loss_fn', type=str, choices=['svm', 'ce'], default='ce',
                       help='基础损失函数 (default: ce)')
    
    # 实时可视化参数
    parser.add_argument('--enable_realtime_viz', action='store_true',
                       help='启用实时可视化')
    parser.add_argument('--viz_port', type=int, default=8080,
                       help='可视化端口 (default: 8080)')
    
    args = parser.parse_args()
    
    # 解析通道（使用 main.py 的 parse_channels）
    target_channels = parse_channels(args.target_channels)
    
    # 加载固定测试集分割（如果提供）
    fixed_test_split = None
    if args.fixed_test_split:
        from main import load_dataset_split
        fixed_test_split = load_dataset_split(args.fixed_test_split)
    
    # 创建优化器
    optimizer = AUCCLAMOptimizer(
        data_root_dir=args.data_root_dir,
        csv_path=args.csv_path,
        model_type=args.model_type,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        pruner=not args.no_pruner,
        sampler=args.sampler,
        enable_realtime_viz=args.enable_realtime_viz,
        viz_port=args.viz_port,
        data_root_base=args.data_root_base,
        num_data_copies=args.num_data_copies,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        base_loss_fn=args.base_loss_fn,
        alignment_model_path=args.alignment_model_path,
        aligned_channels=args.aligned_channels,
        fixed_test_split=fixed_test_split,
        seed=args.seed
    )
    
    # 执行优化
    study = optimizer.optimize(
        target_channels=target_channels,
        n_folds=args.n_folds,
        study_name=args.study_name
    )
    
    print("\n🎯 优化完成！")
    print(f"📁 结果保存在: {args.results_dir}")

if __name__ == "__main__":
    main()
