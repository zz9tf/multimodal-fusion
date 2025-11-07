#!/usr/bin/env python3
"""
Optuna 配置管理工具
用于管理各种模型的超参数搜索空间和配置生成
按照实验配置和模型组件配置分类组织
"""

import os
import json
from typing import Dict, Any, List, Optional
import optuna


class OptunaConfig:
    """
    Optuna 配置管理器
    负责定义超参数搜索空间和生成训练配置
    按照实验配置和模型组件配置分类组织
    """
    
    def __init__(self):
        """初始化配置管理器"""
        pass
    
    # ========== 实验配置参数范围 ==========
    
    def suggest_experiment_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议实验级别的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            实验参数字典
        """
        params = {}
        
        params['alignment_model_path'] = None
        params['aligned_channels'] = None
        
        # 随机种子
        params['seed'] = trial.suggest_int('seed', 1, 10000)
        
        # 最大训练轮数
        params['max_epochs'] = trial.suggest_int('max_epochs', 100, 300)
        
        # 学习率 - 基于默认值 1e-4 的合理范围
        params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        
        # 权重衰减 - 基于默认值 1e-5 的合理范围
        params['reg'] = trial.suggest_float('reg', 1e-6, 1e-4, log=True)
        
        # 优化器类型
        params['opt'] = 'adam'
        
        # 早停
        params['early_stopping'] = trial.suggest_categorical('early_stopping', [True, False])
        
        # 批次大小
        params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # Dropout率
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.9)
        
        params['scheduler_config'] = {
            "type": "plateau",
            "mode": "min", 
            "patience": 15, 
            "factor": 0.5
        }
        
        return params
    
    # ========== 模型组件配置参数范围 ==========
    
    def suggest_mil_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 MIL 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            MIL 参数字典
        """
        params = {}
        
        # 模型大小
        params['model_size'] = trial.suggest_categorical('model_size', 
            ['64*32', '32*16', '16*8', '8*4', '4*2', '2*1'])
        
        # 返回特征
        params['return_features'] = trial.suggest_categorical('return_features', [True, False])
        
        return params
    
    def suggest_clam_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 CLAM 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            CLAM 参数字典
        """
        params = {}
        
        # 门控机制
        params['gate'] = trial.suggest_categorical('gate', [True, False])
        
        # Bag级别损失权重
        params['base_weight'] = trial.suggest_float('base_weight', 0.3, 0.9)
        
        # 实例级别损失函数
        params['inst_loss_fn'] = trial.suggest_categorical('inst_loss_fn', [None, 'ce'])
        
        # 模型大小
        params['model_size'] = trial.suggest_categorical('model_size', 
            ['64*32', '32*16', '16*8', '8*4', '4*2', '2*1'])
        
        # 子类型问题
        params['subtyping'] = trial.suggest_categorical('subtyping', [True, False])
        
        # 正负样本采样数量
        params['inst_number'] = trial.suggest_categorical('inst_number', [4, 8])
        
        # 返回特征
        params['return_features'] = trial.suggest_categorical('return_features', [True, False])
        
        # 仅返回注意力
        params['attention_only'] = False
        
        return params
    
    def suggest_auc_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 AUC 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            AUC 参数字典
        """
        params = {}
        
        # AUC损失权重
        params['auc_loss_weight'] = trial.suggest_float('auc_loss_weight', 0.1, 2.0)
        
        return params
    
    def suggest_transfer_layer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 Transfer Layer 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            Transfer Layer 参数字典
        """
        params = {}
        
        # 输出维度
        params['output_dim'] = trial.suggest_categorical('output_dim', [64, 128, 256])
        
        return params
    
    def suggest_svd_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 SVD 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            SVD 参数字典
        """
        params = {}
        
        # 启用SVD
        params['enable_svd'] = trial.suggest_categorical('enable_svd', [True, False])
        
        # 对齐层数
        params['alignment_layer_num'] = trial.suggest_int('alignment_layer_num', 1, 4)
        
        # 对齐损失权重
        params['lambda1'] = trial.suggest_float('lambda1', 0.1, 2.0)
        params['lambda2'] = trial.suggest_float('lambda2', 0.0, 1.0)
        
        # 温度参数
        params['tau1'] = trial.suggest_float('tau1', 0.01, 0.5)
        params['tau2'] = trial.suggest_float('tau2', 0.01, 0.5)
        
        # 返回SVD特征
        params['return_svd_features'] = False
        
        return params
    
    def suggest_clip_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 CLIP 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            CLIP 参数字典
        """
        params = {}
        
        # 对齐层数
        params['alignment_layer_num'] = trial.suggest_int('alignment_layer_num', 1, 4)
        
        # 启用CLIP
        params['enable_clip'] = True
        
        # 初始tau
        params['clip_init_tau'] = trial.suggest_float('clip_init_tau', 0.01, 0.2)
        
        return params
    
    def suggest_dynamic_gate_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 Dynamic Gate 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            Dynamic Gate 参数字典
        """
        params = {}
        
        # 启用动态门控
        params['enable_dynamic_gate'] = False
        
        # 置信度权重
        params['confidence_weight'] = trial.suggest_float('confidence_weight', 0.1, 2.0)
        
        # 特征权重权重
        params['feature_weight_weight'] = trial.suggest_float('feature_weight_weight', 0.1, 2.0)
        
        return params
    
    def suggest_random_loss_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        建议 Random Loss 组件的超参数
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            Random Loss 参数字典
        """
        params = {}
        
        # 启用随机损失
        params['enable_random_loss'] = trial.suggest_categorical('enable_random_loss', [True, False])
        
        # 随机损失权重
        params['weight_random_loss'] = trial.suggest_float('weight_random_loss', 0.01, 1.0)
        
        return params
    
    # ========== 模型类型配置组合 ==========
    
    def suggest_model_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """
        根据模型类型建议超参数（组合各个组件）
        
        Args:
            trial: Optuna 试验对象
            model_type: 模型类型
            
        Returns:
            模型参数字典
        """
        params = {}
        
        # 根据模型类型组合不同的组件配置
        if model_type == 'mil':
            params.update(self.suggest_mil_params(trial))
        
        elif model_type == 'clam':
            params.update(self.suggest_clam_params(trial))
        
        elif model_type == 'auc_clam':
            params.update(self.suggest_clam_params(trial))
            params.update(self.suggest_auc_params(trial))
        
        elif model_type in ['clam_mlp', 'clam_mlp_detach']:
            params.update(self.suggest_clam_params(trial))
            params.update(self.suggest_transfer_layer_params(trial))
        
        elif model_type in ['svd_gate_random_clam', 'svd_gate_random_clam_detach',
                           'deep_supervise_svd_gate_random', 'deep_supervise_svd_gate_random_detach']:
            params.update(self.suggest_clam_params(trial))
            params.update(self.suggest_transfer_layer_params(trial))
            params.update(self.suggest_svd_params(trial))
            params.update(self.suggest_dynamic_gate_params(trial))
            params.update(self.suggest_random_loss_params(trial))
        
        elif model_type in ['clip_gate_random_clam', 'clip_gate_random_clam_detach']:
            params.update(self.suggest_clam_params(trial))
            params.update(self.suggest_transfer_layer_params(trial))
            params.update(self.suggest_clip_params(trial))
            params.update(self.suggest_dynamic_gate_params(trial))
            params.update(self.suggest_random_loss_params(trial))
        
        elif model_type in ['gate_shared_mil', 'gate_mil', 'gate_mil_detach']:
            params.update(self.suggest_mil_params(trial))
            params.update(self.suggest_dynamic_gate_params(trial))
        
        elif model_type == 'gate_auc_mil':
            params.update(self.suggest_mil_params(trial))
            params.update(self.suggest_dynamic_gate_params(trial))
            params.update(self.suggest_auc_params(trial))
        
        else:
            # 默认只使用 CLAM 配置
            params.update(self.suggest_clam_params(trial))
        
        return params
    
    # ========== 配置生成 ==========
    
    def create_configs(self, 
                      model_type: str,
                      data_root_dir: str,
                      csv_path: str,
                      target_channels: List[str],
                      experiment_params: Dict[str, Any],
                      model_params: Dict[str, Any],
                      trial_number: int = None,
                      num_splits: int = 10,
                      **kwargs) -> Dict[str, Any]:
        """
        创建完整的训练配置
        
        Args:
            model_type: 模型类型
            data_root_dir: 数据根目录
            csv_path: CSV文件路径
            target_channels: 目标通道列表
            experiment_params: 实验参数
            model_params: 模型参数
            trial_number: 试验编号
            seed: 随机种子
            num_splits: K折数量
            **kwargs: 其他参数
            
        Returns:
            配置字典
        """
        # 创建结果目录
        if trial_number is not None:
            results_dir = f"./optuna_results/trial_{trial_number}"
        else:
            results_dir = "./optuna_results/trial_unknown"
        
        # 实验配置
        experiment_config = {
            'data_root_dir': data_root_dir,
            'results_dir': results_dir,
            'csv_path': csv_path,
            'alignment_model_path': experiment_params.get('alignment_model_path', None),
            'target_channels': target_channels,
            'aligned_channels': experiment_params.get('aligned_channels', None),
            'exp_code': f"optuna_{model_type}_{trial_number if trial_number is not None else 'unknown'}",
            'seed': experiment_params.get('seed', 42),
            'num_splits': num_splits,
            'max_epochs': experiment_params.get('max_epochs', 200),
            'lr': experiment_params.get('lr', 1e-4),
            'reg': experiment_params.get('reg', 1e-5),
            'opt': experiment_params.get('opt', 'adam'),
            'early_stopping': experiment_params.get('early_stopping', False),
            'batch_size': experiment_params.get('batch_size', 64),
            'scheduler_config': experiment_params.get('scheduler_config', {'type': None}),
        }
        
        # 模型配置
        model_config = {
            'model_type': model_type,
            'input_dim': 1024,
            'dropout': experiment_params.get('dropout', 0.25),
            'n_classes': 2,
            'base_loss_fn': 'ce',
            'channels_used_in_model': target_channels,  # 使用传入的 target_channels，而不是硬编码
        }
        
        # 添加模型特定参数
        model_config.update(model_params)
        
        return {
            'experiment_config': experiment_config,
            'model_config': model_config
        }
    
    # ========== 参数范围信息（用于文档和可视化） ==========
    
    def get_experiment_param_ranges(self) -> Dict[str, Any]:
        """获取实验参数范围信息"""
        return {
            'lr': {'type': 'float', 'range': [1e-5, 1e-3], 'log': True, 'description': '学习率'},
            'reg': {'type': 'float', 'range': [1e-6, 1e-4], 'log': True, 'description': '权重衰减'},
            'opt': {'type': 'fixed', 'value': 'adam', 'description': '优化器类型（固定为adam）'},
            'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256], 'description': '批次大小'},
            'max_epochs': {'type': 'int', 'range': [100, 300], 'description': '最大训练轮数'},
            'early_stopping': {'type': 'categorical', 'choices': [True, False], 'description': '早停'},
            'dropout': {'type': 'float', 'range': [0.1, 0.9], 'description': 'Dropout率'},
        }
    
    def get_component_param_ranges(self, component_name: str) -> Dict[str, Any]:
        """获取指定组件的参数范围信息"""
        ranges = {
            'mil': {
                'model_size': {'type': 'categorical', 'choices': ['64*32', '32*16', '16*8', '8*4', '4*2', '2*1'], 'description': '模型大小'},
                'return_features': {'type': 'categorical', 'choices': [True, False], 'description': '返回特征'},
            },
            'clam': {
                'gate': {'type': 'categorical', 'choices': [True, False], 'description': '门控机制'},
                'base_weight': {'type': 'float', 'range': [0.3, 0.9], 'description': 'Bag级别损失权重'},
                'inst_loss_fn': {'type': 'categorical', 'choices': [None, 'ce'], 'description': '实例级别损失函数'},
                'model_size': {'type': 'categorical', 'choices': ['64*32', '32*16', '16*8', '8*4', '4*2', '2*1'], 'description': '模型大小'},
                'subtyping': {'type': 'categorical', 'choices': [True, False], 'description': '子类型问题'},
                'inst_number': {'type': 'categorical', 'choices': [4, 8, 16, 32, 64], 'description': '正负样本采样数量'},
                'return_features': {'type': 'categorical', 'choices': [True, False], 'description': '返回特征'},
                'attention_only': {'type': 'categorical', 'choices': [True, False], 'description': '仅返回注意力'},
            },
            'auc': {
                'auc_loss_weight': {'type': 'float', 'range': [0.1, 2.0], 'description': 'AUC损失权重'},
            },
            'transfer_layer': {
                'output_dim': {'type': 'categorical', 'choices': [64, 128, 256], 'description': '输出维度'},
            },
            'svd': {
                'enable_svd': {'type': 'categorical', 'choices': [True, False], 'description': '启用SVD'},
                'alignment_layer_num': {'type': 'int', 'range': [1, 4], 'description': '对齐层数'},
                'lambda1': {'type': 'float', 'range': [0.1, 2.0], 'description': '对齐损失权重1'},
                'lambda2': {'type': 'float', 'range': [0.0, 1.0], 'description': '对齐损失权重2'},
                'tau1': {'type': 'float', 'range': [0.01, 0.5], 'description': '温度参数1'},
                'tau2': {'type': 'float', 'range': [0.01, 0.5], 'description': '温度参数2'},
                'return_svd_features': {'type': 'fixed', 'value': False, 'description': '返回SVD特征（固定为False）'},
            },
            'clip': {
                'alignment_layer_num': {'type': 'int', 'range': [1, 4], 'description': '对齐层数'},
                'enable_clip': {'type': 'fixed', 'value': True, 'description': '启用CLIP（固定为True）'},
                'clip_init_tau': {'type': 'float', 'range': [0.01, 0.2], 'description': '初始tau'},
            },
            'dynamic_gate': {
                'enable_dynamic_gate': {'type': 'fixed', 'value': False, 'description': '启用动态门控（固定为False）'},
                'confidence_weight': {'type': 'float', 'range': [0.1, 2.0], 'description': '置信度权重'},
                'feature_weight_weight': {'type': 'float', 'range': [0.1, 2.0], 'description': '特征权重权重'},
            },
            'random_loss': {
                'enable_random_loss': {'type': 'categorical', 'choices': [True, False], 'description': '启用随机损失'},
                'weight_random_loss': {'type': 'float', 'range': [0.01, 1.0], 'description': '随机损失权重'},
            },
        }
        
        return ranges.get(component_name, {})
    
    def get_model_param_ranges(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型类型的参数范围信息"""
        ranges = {}
        
        # 实验参数
        ranges.update(self.get_experiment_param_ranges())
        
        # 根据模型类型组合组件参数
        if model_type == 'mil':
            ranges.update(self.get_component_param_ranges('mil'))
        
        elif model_type == 'clam':
            ranges.update(self.get_component_param_ranges('clam'))
        
        elif model_type == 'auc_clam':
            ranges.update(self.get_component_param_ranges('clam'))
            ranges.update(self.get_component_param_ranges('auc'))
        
        elif model_type in ['clam_mlp', 'clam_mlp_detach']:
            ranges.update(self.get_component_param_ranges('clam'))
            ranges.update(self.get_component_param_ranges('transfer_layer'))
        
        elif model_type in ['svd_gate_random_clam', 'svd_gate_random_clam_detach',
                           'deep_supervise_svd_gate_random', 'deep_supervise_svd_gate_random_detach']:
            ranges.update(self.get_component_param_ranges('clam'))
            ranges.update(self.get_component_param_ranges('transfer_layer'))
            ranges.update(self.get_component_param_ranges('svd'))
            ranges.update(self.get_component_param_ranges('dynamic_gate'))
            ranges.update(self.get_component_param_ranges('random_loss'))
        
        elif model_type in ['clip_gate_random_clam', 'clip_gate_random_clam_detach']:
            ranges.update(self.get_component_param_ranges('clam'))
            ranges.update(self.get_component_param_ranges('transfer_layer'))
            ranges.update(self.get_component_param_ranges('clip'))
            ranges.update(self.get_component_param_ranges('dynamic_gate'))
            ranges.update(self.get_component_param_ranges('random_loss'))
        
        elif model_type in ['gate_shared_mil', 'gate_mil', 'gate_mil_detach']:
            ranges.update(self.get_component_param_ranges('mil'))
            ranges.update(self.get_component_param_ranges('dynamic_gate'))
        
        elif model_type == 'gate_auc_mil':
            ranges.update(self.get_component_param_ranges('mil'))
            ranges.update(self.get_component_param_ranges('dynamic_gate'))
            ranges.update(self.get_component_param_ranges('auc'))
        
        return ranges
    
    def save_config_template(self, model_type: str, filepath: str):
        """
        保存配置模板到文件
        
        Args:
            model_type: 模型类型
            filepath: 保存路径
        """
        template = {
            'description': f'{model_type} Optuna 超参数优化配置模板',
            'model_type': model_type,
            'experiment_params': self.get_experiment_param_ranges(),
            'component_params': {
                'mil': self.get_component_param_ranges('mil'),
                'clam': self.get_component_param_ranges('clam'),
                'auc': self.get_component_param_ranges('auc'),
                'transfer_layer': self.get_component_param_ranges('transfer_layer'),
                'svd': self.get_component_param_ranges('svd'),
                'clip': self.get_component_param_ranges('clip'),
                'dynamic_gate': self.get_component_param_ranges('dynamic_gate'),
                'random_loss': self.get_component_param_ranges('random_loss'),
            },
            'model_params': self.get_model_param_ranges(model_type),
            'usage': {
                'example_command': f'python optuna_optimization.py --model_type {model_type} --data_root_dir /path/to/data --csv_path /path/to/labels.csv --n_trials 100',
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
    
    # 测试不同模型类型
    model_types = [
        # 'mil', 'clam', 'auc_clam', 'clam_mlp', 'clam_mlp_detach',
        # 'svd_gate_random_clam', 'svd_gate_random_clam_detach',
        # 'clip_gate_random_clam', 'clip_gate_random_clam_detach',
        # 'gate_shared_mil', 'gate_mil', 'gate_auc_mil'
        'svd_gate_random_clam_detach'
    ]
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"📊 {model_type} 超参数搜索空间:")
        print(f"{'='*60}")
        
        param_ranges = config_manager.get_model_param_ranges(model_type)
        for param, info in param_ranges.items():
            print(f"  {param}: {info['description']} - {info}")
        
        # 保存配置模板
        template_path = os.path.join(os.path.dirname(__file__), f'config_template_{model_type}.json')
        config_manager.save_config_template(model_type, template_path)


if __name__ == "__main__":
    main()
