#!/usr/bin/env python3
"""
Optuna Ask-and-Tell 模式脚本
允许 Optuna 建议参数，然后由用户自己的程序执行这些参数

使用方式：
1. 使用 ask() 获取 Optuna 建议的参数
2. 将参数保存到文件或传递给自己的程序
3. 自己的程序执行完成后，使用 tell() 报告结果
"""

import os
import json
import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages.journal import JournalFileBackend
from optuna.storages import JournalStorage
from optuna_config_loader import OptunaConfigLoader
from typing import Dict, Any


class OptunaAskTellManager:
    """
    Optuna Ask-and-Tell 管理器
    用于手动控制 Optuna 试验流程
    """
    
    def __init__(self,
                 study_name: str,
                 model_type: str = 'svd_gate_random_clam_detach',
                 results_dir: str = './optuna_results',
                 sampler: str = 'tpe',
                 pruner: bool = True,
                 config_file: str = None,
                 **kwargs):
        """
        初始化管理器
        
        Args:
            study_name: 研究名称
            model_type: 模型类型
            results_dir: 结果保存目录
            sampler: 采样器类型 ('tpe', 'random', 'cmaes')
            pruner: 是否启用剪枝
            config_file: 配置文件路径（YAML），包含参数范围和固定值
            **kwargs: 其他参数（如 input_dim, n_classes 等）
        """
        self.study_name = study_name
        self.model_type = model_type
        self.config_file = config_file
        self.kwargs = kwargs
        
        # 在 results_dir 下创建 study_name 文件夹
        self.results_dir = os.path.join(results_dir, study_name)
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
        
        # 初始化配置加载器（支持从文件加载）
        self.config_loader = OptunaConfigLoader(config_file=config_file) if config_file else None
        
        # 创建或加载研究
        self._init_study()
    
    def _init_study(self):
        """初始化或加载 Optuna Study"""
        # 使用 JournalStorage 作为存储后端（支持并行，无需数据库）
        journal_path = os.path.join(self.results_dir, "study.log")
        try:
            journal_backend = JournalFileBackend(journal_path)
            storage_url = JournalStorage(journal_backend)
            print(f"📦 使用 JournalStorage: {journal_path}")
        except ImportError:
            raise ImportError(
                "JournalStorage 不可用。请确保 Optuna 版本 >= 3.0。\n"
                "安装命令: pip install optuna>=3.0"
            )
        
        # 加载或创建研究
        try:
            self.study = optuna.load_study(
                study_name=self.study_name,
                storage=storage_url,
                sampler=self.sampler,
                pruner=self.pruner
            )
            print(f"📖 加载现有研究: {self.study_name}")
            print(f"📊 已有试验数: {len(self.study.trials)}")
        except KeyError:
            print(f"🆕 创建新研究: {self.study_name}")
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=storage_url,
                direction='maximize',  # 最大化AUC
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True
            )
    
    def ask(self) -> Dict[str, Any]:
        """
        向 Optuna 请求建议的参数
        
        Returns:
            包含建议参数的字典，包括：
            - trial_number: 试验编号
            - experiment_params: 实验参数
            - model_params: 模型参数
            - trial_id: 内部 trial ID（用于 tell()）
        """
        # 创建新的 trial
        trial = self.study.ask()
        
        print(f"\n🔬 请求新参数 (Trial {trial.number})...")
        
        if not self.config_loader:
            raise ValueError("必须提供配置文件 (config_file) 来定义搜索空间")
        
        # 使用配置文件创建完整配置
        full_config = self.config_loader.create_full_config(trial)
        
        # 分离实验参数和模型参数
        experiment_params = full_config.get('experiment_config', {})
        model_params = full_config.get('model_config', {})
        
        # 构建返回结果
        result = {
            'trial_number': trial.number,
            'experiment_params': experiment_params,
            'model_params': model_params,
            'trial_id': trial._trial_id,  # 内部 trial ID，用于 tell()
            'configs': full_config  # 完整配置
        }
        
        # 打印参数摘要
        print(f"\n📋 Trial {trial.number} 建议参数:")
        if experiment_params:
            lr = experiment_params.get('lr', 'N/A')
            lr_str = f"{lr:.6f}" if isinstance(lr, (int, float)) else str(lr)
            batch_size = experiment_params.get('batch_size', 'N/A')
            max_epochs = experiment_params.get('max_epochs', 'N/A')
            print(f"   实验参数: lr={lr_str}, batch_size={batch_size}, max_epochs={max_epochs}")
        if model_params:
            print(f"   模型参数: {len(model_params)} 个参数")
        
        return result
    
    def tell(self, 
             trial_id: int,
             value: float = None,
             state: optuna.trial.TrialState = None) -> None:
        """
        向 Optuna 报告试验结果
        
        Args:
            trial_id: 内部 trial ID（必需）
            value: 目标函数值（如 AUC 分数）
            state: 试验状态（可选，默认为 COMPLETE 或 FAIL）
        """
        # 验证必需参数
        if trial_id is None:
            raise ValueError("必须提供 trial_id")
        
        if state is None:
            if value is not None:
                state = optuna.trial.TrialState.COMPLETE
            else:
                state = optuna.trial.TrialState.FAIL
        
        # 报告结果
        print(f"\n📊 报告结果 (Trial ID: {trial_id})...")
        
        try:
            self.study.tell(trial_id, value, state=state)
            
            if state == optuna.trial.TrialState.COMPLETE:
                print(f"✅ 成功报告结果: value={value:.4f}")
            elif state == optuna.trial.TrialState.FAIL:
                print(f"❌ 报告试验失败")
            elif state == optuna.trial.TrialState.PRUNED:
                print(f"✂️  报告试验被剪枝")
            
            # 打印当前最佳结果
            if len(self.study.trials) > 0:
                best_trial = self.study.best_trial
                print(f"🏆 当前最佳: Trial {best_trial.number}, value={self.study.best_value:.4f}")
        
        except Exception as e:
            print(f"❌ 报告结果失败: {e}")
            raise
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        获取当前最佳参数
        
        Returns:
            包含最佳参数的字典
        """
        if len(self.study.trials) == 0:
            raise ValueError("还没有完成的试验")
        
        best_trial = self.study.best_trial
        return {
            'trial_number': best_trial.number,
            'value': self.study.best_value,
            'params': best_trial.params
        }
    
    def get_study_summary(self) -> Dict[str, Any]:
        """
        获取研究摘要
        
        Returns:
            研究摘要字典
        """
        return {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'n_complete': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_fail': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'best_value': self.study.best_value if len(self.study.trials) > 0 else None,
            'best_trial_number': self.study.best_trial.number if len(self.study.trials) > 0 else None,
        }
    
    def save_results(self, output_file: str = None):
        """
        保存试验结果到文件
        
        Args:
            output_file: 输出文件路径（可选，默认使用 study_name）
        """
        if output_file is None:
            output_file = os.path.join(self.results_dir, f"{self.study_name}_trials.csv")
        
        # 保存 CSV 格式
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(output_file, index=False)
        print(f"💾 试验结果已保存到: {output_file}")
        
        # 保存 JSON 格式
        json_file = output_file.replace('.csv', '.json')
        summary = self.get_study_summary()
        summary['trials'] = [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': t.state.name
            }
            for t in self.study.trials
        ]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 试验摘要已保存到: {json_file}")


def main():
    """主函数 - 演示如何使用 Ask-and-Tell 模式"""
    parser = argparse.ArgumentParser(
        description='Optuna Ask-and-Tell 模式 - 手动控制试验流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 请求参数:
   python optuna_ask_tell.py ask --study_name my_study --model_type auc_clam \\
       --config_file config.yaml

2. 报告结果:
   python optuna_ask_tell.py tell --study_name my_study \\
       --trial_id 0 --value 0.85

3. 查看摘要:
   python optuna_ask_tell.py summary --study_name my_study
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # Ask 命令
    ask_parser = subparsers.add_parser('ask', help='请求 Optuna 建议的参数')
    ask_parser.add_argument('--study_name', type=str, required=True,
                           help='研究名称')
    ask_parser.add_argument('--model_type', type=str, default='svd_gate_random_clam_detach',
                           help='模型类型')
    ask_parser.add_argument('--results_dir', type=str, default='./optuna_results',
                           help='结果保存目录')
    ask_parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'],
                           default='tpe', help='采样器类型')
    ask_parser.add_argument('--no_pruner', action='store_true',
                           help='禁用剪枝器')
    ask_parser.add_argument('--config_file', type=str, default=None,
                           help='配置文件路径（YAML），包含参数范围和固定值')
    
    # Tell 命令
    tell_parser = subparsers.add_parser('tell', help='报告试验结果')
    tell_parser.add_argument('--study_name', type=str, required=True,
                            help='研究名称')
    tell_parser.add_argument('--results_dir', type=str, default='./optuna_results',
                            help='结果保存目录')
    tell_parser.add_argument('--trial_id', type=int, required=True,
                            help='内部 trial ID（必需）')
    tell_parser.add_argument('--value', type=float, default=None,
                            help='目标函数值（如 AUC 分数）')
    tell_parser.add_argument('--state', type=str, choices=['COMPLETE', 'FAIL', 'PRUNED'],
                            default=None, help='试验状态')
    
    # Summary 命令
    summary_parser = subparsers.add_parser('summary', help='查看研究摘要')
    summary_parser.add_argument('--study_name', type=str, required=True,
                               help='研究名称')
    summary_parser.add_argument('--results_dir', type=str, default='./optuna_results',
                               help='结果保存目录')
    summary_parser.add_argument('--save', type=str, default=None,
                               help='保存结果到文件')
    
    args = parser.parse_args()
    
    if args.command == 'ask':
        # 创建管理器
        manager = OptunaAskTellManager(
            study_name=args.study_name,
            model_type=args.model_type,
            results_dir=args.results_dir,
            sampler=args.sampler,
            pruner=not args.no_pruner,
            config_file=getattr(args, 'config_file', None)
        )
        
        # 请求参数
        result = manager.ask()
        
        # 打印结果
        print(f"\n✅ 成功获取参数:")
        print(f"   Trial Number: {result['trial_number']}")
        print(f"   Trial ID: {result['trial_id']}")
        print(f"\n💡 下一步: 使用这些参数运行你的程序，然后使用以下命令报告结果:")
        print(f"   python optuna_ask_tell.py tell --study_name {args.study_name} \\")
        print(f"       --trial_id {result['trial_id']} --value <你的结果>")
    
    elif args.command == 'tell':
        # 创建管理器
        manager = OptunaAskTellManager(
            study_name=args.study_name,
            results_dir=args.results_dir
        )
        
        # 报告结果
        state = None
        if args.state:
            state = optuna.trial.TrialState[args.state]
        
        manager.tell(
            trial_id=args.trial_id,
            value=args.value,
            state=state
        )
        
        # 保存结果
        manager.save_results()
    
    elif args.command == 'summary':
        # 创建管理器
        manager = OptunaAskTellManager(
            study_name=args.study_name,
            results_dir=args.results_dir
        )
        
        # 获取摘要
        summary = manager.get_study_summary()
        
        # 打印摘要
        print(f"\n📊 研究摘要: {summary['study_name']}")
        print(f"   总试验数: {summary['n_trials']}")
        print(f"   完成: {summary['n_complete']}")
        print(f"   失败: {summary['n_fail']}")
        print(f"   剪枝: {summary['n_pruned']}")
        if summary['best_value'] is not None:
            print(f"   最佳值: {summary['best_value']:.4f} (Trial {summary['best_trial_number']})")
        
        # 保存结果
        if args.save:
            manager.save_results(args.save)
        else:
            manager.save_results()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

