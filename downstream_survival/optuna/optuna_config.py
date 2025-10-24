#!/usr/bin/env python3
"""
Optuna 配置管理工具
用于管理 AUC_CLAM 模型的超参数搜索空间和配置生成
"""

import os
import json
from typing import Dict, Any, List, Optional
import optuna

class OptunaConfig:
    """
    Optuna 配置管理器
    负责定义超参数搜索空间和生成训练配置
    """
    
    def __init__(self):
        """初始化配置管理器"""
        self.base_config = {
            'model_type': 'auc_clam',
            'n_classes': 2,
            'base_loss_fn': 'ce',
            'subtyping': False,
            'return_features': False,
            'attention_only': False,
        }
    
    def suggest_auc_clam_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        为 AUC_CLAM 模型建议超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            参数字典
        """
        params = {}
        
        # === 实验配置参数 ===
        # 学习率 - 基于默认值 1e-4 的合理范围
        params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        
        # 权重衰减 - 基于默认值 1e-5 的合理范围
        params['reg'] = trial.suggest_float('reg', 1e-6, 1e-4, log=True)
        
        # 优化器类型 - 基于 main.py 的 choices
        params['opt'] = trial.suggest_categorical('opt', ['adam', 'sgd'])
        
        # 批次大小 - 基于默认值 1，但允许更大的批次
        params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        
        # 最大训练轮数 - 基于默认值 200 的合理范围
        params['max_epochs'] = trial.suggest_int('max_epochs', 100, 300)
        
        # 早停 - 基于默认值 False
        params['early_stopping'] = trial.suggest_categorical('early_stopping', [True, False])
        
        # === 模型结构参数 ===
        # 输入维度 - 基于默认值 1024
        params['input_dim'] = 1024
        
        # Dropout率 - 基于默认值 0.25 的合理范围
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        
        # 模型大小 - 基于 main.py 的 choices
        params['model_size'] = trial.suggest_categorical('model_size', 
            ['small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2'])
        
        # === CLAM 特定参数 ===
        # 门控机制 - 基于默认值 True
        params['gate'] = trial.suggest_categorical('gate', [True, False])
        
        # Bag级别损失权重 - 基于默认值 0.7 的合理范围
        params['base_weight'] = trial.suggest_float('base_weight', 0.3, 0.5, 0.7, 0.9)
        
        # 实例级别损失函数 - 基于 main.py 的 choices
        params['inst_loss_fn'] = trial.suggest_categorical('inst_loss_fn', [None, 'svm', 'ce'])
        
        # 正负样本采样数量 - 基于默认值 8 的合理范围
        params['inst_number'] = trial.suggest_int('inst_number', 4, 16, 32, 64)
        
        # 通道使用策略 - 总是使用所有通道
        params['channels_used_in_model'] = ['features', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1']
        
        # === AUC_CLAM 特定参数 ===
        # AUC损失权重 - 基于默认值 1.0 的合理范围
        params['auc_loss_weight'] = trial.suggest_float('auc_loss_weight', 0.1, 0.5, 1.0, 1.5, 2.0)
        
        return params
    
    def create_configs(self, 
                      data_root_dir: str,
                      csv_path: str,
                      target_channels: List[str],
                      **params) -> Dict[str, Any]:
        """
        创建完整的训练配置
        
        Args:
            data_root_dir: 数据根目录
            csv_path: CSV文件路径
            target_channels: 目标通道列表
            **params: 超参数
            
        Returns:
            配置字典
        """
        # 创建结果目录
        results_dir = f"./optuna_results/trial_{params.get('trial_number', 'unknown')}"
        
        # 实验配置
        experiment_config = {
            'data_root_dir': data_root_dir,
            'results_dir': results_dir,
            'csv_path': csv_path,
            'alignment_model_path': None,  # 不使用对齐
            'target_channels': target_channels,
            'aligned_channels': None,  # 不使用对齐
            'exp_code': f"optuna_auc_clam_{params.get('trial_number', 'unknown')}",
            'seed': 42,  # 固定种子确保可重复性
            'num_splits': 10,
            'max_epochs': params.get('max_epochs', 200),
            'lr': params.get('lr', 1e-4),
            'reg': params.get('reg', 1e-5),
            'opt': params.get('opt', 'adam'),
            'early_stopping': params.get('early_stopping', False),
            'batch_size': params.get('batch_size', 1)
        }
        
        # 模型配置
        model_config = {
            'model_type': 'auc_clam',
            'input_dim': params.get('input_dim', 1024),
            'dropout': params.get('dropout', 0.25),
            'n_classes': 2,
            'base_loss_fn': 'ce',
            
            # CLAM 参数
            'gate': params.get('gate', True),
            'base_weight': params.get('base_weight', 0.7),
            'inst_loss_fn': params.get('inst_loss_fn', None),
            'model_size': params.get('model_size', 'small'),
            'subtyping': False,
            'inst_number': params.get('inst_number', 8),
            'channels_used_in_model': params.get('channels_used_in_model', None),
            'return_features': False,
            'attention_only': False,
            
            # AUC_CLAM 特定参数
            'auc_loss_weight': params.get('auc_loss_weight', 1.0),
            
            # 高级参数
            'lr_scheduler': params.get('lr_scheduler', None),
            'grad_clip': params.get('grad_clip', None),
            'label_smoothing': params.get('label_smoothing', 0.0)
        }
        
        # 添加学习率调度器参数
        if model_config['lr_scheduler'] == 'step':
            model_config['lr_step_size'] = params.get('lr_step_size', 50)
            model_config['lr_gamma'] = params.get('lr_gamma', 0.5)
        
        return {
            'experiment_config': experiment_config,
            'model_config': model_config
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        """
        获取参数范围信息（用于文档和可视化）
        
        Returns:
            参数范围字典
        """
        return {
            'lr': {'type': 'float', 'range': [1e-5, 1e-3], 'log': True, 'description': '学习率'},
            'reg': {'type': 'float', 'range': [1e-6, 1e-4], 'log': True, 'description': '权重衰减'},
            'opt': {'type': 'categorical', 'choices': ['adam', 'sgd'], 'description': '优化器类型'},
            'batch_size': {'type': 'categorical', 'choices': [64, 128, 256, 512], 'description': '批次大小'},
            'max_epochs': {'type': 'int', 'range': [100, 300], 'description': '最大训练轮数'},
            'early_stopping': {'type': 'categorical', 'choices': [True, False], 'description': '早停'},
            'input_dim': {'type': 'fixed', 'value': 1024, 'description': '输入维度'},
            'dropout': {'type': 'categorical', 'choices': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'description': 'Dropout率'},
            'model_size': {'type': 'categorical', 'choices': ['small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2'], 'description': '模型大小'},
            'gate': {'type': 'categorical', 'choices': [True, False], 'description': '门控机制'},
            'base_weight': {'type': 'categorical', 'choices': [0.3, 0.5, 0.7, 0.9], 'description': 'Bag级别损失权重'},
            'inst_loss_fn': {'type': 'categorical', 'choices': [None, 'svm', 'ce'], 'description': '实例级别损失函数'},
            'inst_number': {'type': 'categorical', 'choices': [4, 16, 32, 64], 'description': '正负样本采样数量'},
            'channels_used_in_model': {'type': 'fixed', 'value': ['features', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1'], 'description': '通道使用策略'},
            'auc_loss_weight': {'type': 'float', 'range': [0.1, 2.0], 'description': 'AUC损失权重'},
        }
    
    def save_config_template(self, filepath: str):
        """
        保存配置模板到文件
        
        Args:
            filepath: 保存路径
        """
        template = {
            'description': 'AUC_CLAM Optuna 超参数优化配置模板',
            'parameter_ranges': self.get_param_ranges(),
            'base_config': self.base_config,
            'usage': {
                'example_command': 'python optuna_auc_clam_optimization.py --data_root_dir /path/to/data --csv_path /path/to/labels.csv --n_trials 100',
                'recommended_settings': {
                    'n_trials': 100,
                    'n_folds': 3,
                    'sampler': 'tpe',
                    'pruner': True
                }
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"📄 配置模板已保存到: {filepath}")

def main():
    """测试配置管理器"""
    config_manager = OptunaConfig()
    
    # 保存配置模板
    template_path = os.path.join(os.path.dirname(__file__), 'config_template.json')
    config_manager.save_config_template(template_path)
    
    # 打印参数范围
    print("📊 AUC_CLAM 超参数搜索空间:")
    param_ranges = config_manager.get_param_ranges()
    for param, info in param_ranges.items():
        print(f"  {param}: {info['description']} - {info}")

if __name__ == "__main__":
    main()
