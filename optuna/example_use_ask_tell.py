#!/usr/bin/env python3
"""
示例：如何在你的程序中使用 Optuna Ask-and-Tell 模式

这个示例展示了如何：
1. 从 Optuna 获取建议的参数
2. 使用这些参数运行你的程序
3. 将结果报告回 Optuna
"""

import os
import sys
import json
import argparse
import optuna
import subprocess
import tempfile
from pathlib import Path
from optuna_ask_tell import OptunaAskTellManager
# 注意：parse_channels 在 example_use_ask_tell.py 中不再需要，因为配置从文件读取


def run_training_with_subprocess(params: dict, configs: dict, main_script_path: str = None):
    """
    使用 subprocess 运行训练脚本
    
    Args:
        params: 包含 experiment_params 和 model_params 的字典
        configs: 完整配置
        main_script_path: main.py 脚本路径（如果为 None，则使用默认路径）
        
    Returns:
        目标函数值（如 AUC 分数）
    """
    print(f"\n🚀 开始训练 (Trial {params['trial_number']})...")
    
    # 确定 main.py 路径
    if main_script_path is None:
        # 默认路径：假设在 optuna 目录下，main.py 在 task_executor 目录
        current_dir = Path(__file__).parent
        main_script_path = current_dir / 'task_executor' / 'main.py'
        if not main_script_path.exists():
            # 尝试其他路径
            main_script_path = current_dir.parent / 'downstream_survival' / 'main.py'
    
    if not os.path.exists(main_script_path):
        raise FileNotFoundError(f"找不到 main.py 脚本: {main_script_path}")
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        config_file = f.name
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    # 创建临时结果文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        result_file = f.name
    
    try:
        # 构建命令
        cmd = [
            sys.executable,  # 使用当前 Python 解释器
            str(main_script_path),
            '--config_file', config_file,
            '--output_result_file', result_file
        ]
        
        print(f"📝 执行命令: {' '.join(cmd)}")
        print(f"📁 配置文件: {config_file}")
        print(f"📁 结果文件: {result_file}")
        
        # 运行 subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出
        for line in process.stdout:
            print(line, end='')
            # 尝试从输出中提取结果分数（如果提前完成）
            if 'RESULT_SCORE:' in line:
                try:
                    score_str = line.split('RESULT_SCORE:')[1].strip()
                    score = float(score_str)
                    # 等待进程完成
                    process.wait()
                    return score
                except:
                    pass
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code != 0:
            raise RuntimeError(f"训练进程返回非零退出码: {return_code}")
        
        # 读取结果文件
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"结果文件不存在: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        if result_data['status'] == 'error':
            raise RuntimeError(f"训练失败: {result_data.get('error', 'Unknown error')}")
        
        # 提取 AUC 分数
        result = result_data['result']
        auc_score = result['mean_val_auc']
        
        print(f"✅ 训练完成，AUC = {auc_score:.4f}")
        
        return auc_score
        
    finally:
        # 清理临时文件
        try:
            if os.path.exists(config_file):
                os.unlink(config_file)
            if os.path.exists(result_file):
                os.unlink(result_file)
        except:
            pass


def main():
    """主函数 - 演示完整流程"""
    parser = argparse.ArgumentParser(
        description='示例：使用 Optuna Ask-and-Tell 模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 单次试验（请求参数 -> 运行 -> 报告结果）:
   python example_use_ask_tell.py --study_name my_study \\
       --model_type auc_clam --data_root_dir /path/to/data \\
       --csv_path /path/to/labels.csv

2. 批量试验（运行多个试验）:
   python example_use_ask_tell.py --study_name my_study \\
       --model_type auc_clam --n_trials 10 \\
       --data_root_dir /path/to/data --csv_path /path/to/labels.csv

3. 仅请求参数（不运行训练）:
   python example_use_ask_tell.py --study_name my_study \\
       --model_type auc_clam --ask_only \\
       --output_params trial_params.json

4. 仅报告结果（从文件读取）:
   python example_use_ask_tell.py --study_name my_study \\
       --tell_only --params_file trial_params.json --value 0.85
        """
    )
    
    parser.add_argument('--study_name', type=str, required=True,
                       help='研究名称')
    parser.add_argument('--model_type', type=str, default='svd_gate_random_clam_detach',
                       help='模型类型')
    parser.add_argument('--results_dir', type=str, default='./optuna_results',
                       help='结果保存目录')
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'],
                       default='tpe', help='采样器类型')
    parser.add_argument('--no_pruner', action='store_true',
                       help='禁用剪枝器')
    
    parser.add_argument('--n_trials', type=int, default=1,
                       help='要运行的试验数量')
    
    parser.add_argument('--ask_only', action='store_true',
                       help='仅请求参数，不运行训练')
    parser.add_argument('--tell_only', action='store_true',
                       help='仅报告结果，不请求新参数')
    
    parser.add_argument('--trial_id', type=int, default=None,
                       help='Trial ID（用于 tell_only）')
    parser.add_argument('--value', type=float, default=None,
                       help='要报告的结果值（用于 tell_only）')
    
    parser.add_argument('--main_script_path', type=str, default=None,
                       help='main.py 脚本路径（默认自动查找）')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径（YAML），包含参数范围和固定值')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = OptunaAskTellManager(
        study_name=args.study_name,
        model_type=args.model_type,
        results_dir=args.results_dir,
        sampler=args.sampler,
        pruner=not args.no_pruner,
        config_file=args.config_file
    )
    
    if args.tell_only:
        # 仅报告结果模式
        if not hasattr(args, 'trial_id') or args.trial_id is None:
            print("❌ 错误: tell_only 模式需要 --trial_id 参数")
            sys.exit(1)
        
        if args.value is None:
            print("❌ 错误: tell_only 模式需要 --value 参数")
            sys.exit(1)
        
        print(f"📊 报告结果...")
        manager.tell(trial_id=args.trial_id, value=args.value)
        manager.save_results()
        print(f"✅ 结果已报告")
    
    elif args.ask_only:
        # 仅请求参数模式
        print(f"🔬 请求参数...")
        result = manager.ask()
        print(f"✅ 参数已获取")
        print(f"\n💡 下一步: 使用这些参数运行你的程序，然后使用以下命令报告结果:")
        print(f"   python example_use_ask_tell.py --study_name {args.study_name} \\")
        print(f"       --tell_only --trial_id {result['trial_id']} --value <你的结果>")
    
    else:
        # 完整流程：请求参数 -> 运行训练 -> 报告结果
        print(f"🚀 开始运行 {args.n_trials} 个试验...")
        
        for i in range(args.n_trials):
            print(f"\n{'='*80}")
            print(f"📋 试验 {i+1}/{args.n_trials}")
            print(f"{'='*80}")
            
            try:
                # 1. 请求参数
                params_result = manager.ask()
                
                # 2. 使用 Optuna 建议的完整配置（已经包含固定值和搜索空间参数）
                full_configs = params_result.get('configs', {
                    'experiment_config': params_result['experiment_params'],
                    'model_config': params_result['model_params']
                })
                
                # 3. 运行训练（使用 subprocess 调用 main.py）
                auc_value = run_training_with_subprocess(
                    params=params_result,
                    configs=full_configs,
                    main_script_path=args.main_script_path
                )
                
                # 4. 报告结果
                manager.tell(
                    trial_id=params_result['trial_id'],
                    value=auc_value
                )
                
                print(f"✅ 试验 {i+1} 完成")
            
            except Exception as e:
                print(f"❌ 试验 {i+1} 失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 报告失败
                try:
                    manager.tell(
                        trial_id=params_result['trial_id'],
                        state=optuna.trial.TrialState.FAIL
                    )
                except:
                    pass
        
        # 保存最终结果
        manager.save_results()
        
        # 打印摘要
        summary = manager.get_study_summary()
        print(f"\n{'='*80}")
        print(f"📊 最终摘要")
        print(f"{'='*80}")
        print(f"   总试验数: {summary['n_trials']}")
        print(f"   完成: {summary['n_complete']}")
        print(f"   失败: {summary['n_fail']}")
        if summary['best_value'] is not None:
            print(f"   最佳值: {summary['best_value']:.4f} (Trial {summary['best_trial_number']})")
        
        # 获取最佳参数
        try:
            best_params = manager.get_best_params()
            print(f"\n🏆 最佳参数:")
            print(f"   Trial Number: {best_params['trial_number']}")
            print(f"   Value: {best_params['value']:.4f}")
            print(f"   参数: {json.dumps(best_params['params'], indent=2, ensure_ascii=False)}")
        except:
            pass


if __name__ == "__main__":
    main()

