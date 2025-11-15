#!/usr/bin/env python3
"""
Optuna 配置加载器
从 YAML 文件读取配置，通过字符串解析自动判断固定值和搜索空间

配置格式说明：

1. 固定值（不带 * 前缀）：
   - 直接写值，如：42, 3.14, "adam", true, false, null, [1, 2, 3], {key: value}
   - 这些值会直接使用，不经过 Optuna 优化

2. 搜索空间（带 * 前缀）：
   - *int:1-10          # 整数范围 [1, 10]
   - *float:1.1-20.0    # 浮点数范围 [1.1, 20.0]
   - *log:1e-5-1e-3     # 对数范围 [1e-5, 1e-3]（浮点数，log=True）
   - *a,b,c             # 分类选择（categorical），从 a, b, c 中选择
   - *mul:a,b,c         # 多选列表（categorical），从 a, b, c 中选择（与 *a,b,c 相同）
   - *bool              # 布尔值选择 [true, false]
   - *1-10              # 自动推断类型（整数或浮点数）

配置示例（YAML）：
experiment_config:
  data_root_dir: "/path/to/data"           # 固定值
  opt: "adam"                              # 固定值
  n_classes: 2                             # 固定值
  lr: "*log:1e-5-1e-3"                    # 对数范围
  batch_size: "*32,64,128,256"            # 分类选择
  max_epochs: "*int:100-300"              # 整数范围
  dropout: "*float:0.1-0.9"               # 浮点数范围
  early_stopping: "*bool"                 # 布尔值选择

model_config:
  input_dim: 1024                         # 固定值
  model_size: "*64*32,32*16,16*8"        # 分类选择（字符串）
"""

import os
from typing import Dict, Any, List, Optional, Union
import optuna

# 导入 yaml（必需）
try:
    import yaml
except ImportError:
    raise ImportError("需要安装 PyYAML: pip install pyyaml")


class OptunaConfigLoader:
    """
    Optuna 配置加载器
    从 JSON/YAML 文件读取配置，通过字符串解析自动判断固定值和搜索空间
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化配置加载器
        
        Args:
            config_file: 配置文件路径（JSON 或 YAML）
        """
        self.config_file = config_file
        self.raw_config = {}
        self.fixed_values = {}
        self.search_space = {}
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """
        从 YAML 文件加载配置并自动解析固定值和搜索空间
        
        Args:
            config_file: YAML 配置文件路径
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        # 只支持 YAML
        if not (config_file.endswith('.yaml') or config_file.endswith('.yml')):
            raise ValueError(f"只支持 YAML 文件（.yaml 或 .yml），当前文件: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        if self.raw_config is None:
            raise ValueError(f"配置文件为空或格式错误: {config_file}")
        
        # 自动解析固定值和搜索空间
        self._parse_config()
        
        print(f"✅ 从配置文件加载: {config_file}")
        print(f"   固定值参数: {self._count_fixed_values()} 个")
        print(f"   搜索空间参数: {self._count_search_space()} 个")
    
    def _parse_config(self):
        """解析配置，自动分离固定值和搜索空间"""
        self.fixed_values = {}
        self.search_space = {}
        
        # 遍历所有配置节
        for section_name, section_data in self.raw_config.items():
            if not isinstance(section_data, dict):
                # 如果不是字典，直接作为固定值
                self.fixed_values[section_name] = section_data
                continue
            
            fixed_section = {}
            search_section = {}
            
            # 遍历节内的每个参数
            for param_name, param_value in section_data.items():
                if self._is_search_space(param_value):
                    # 是搜索空间，解析并存储
                    search_section[param_name] = self._parse_search_space(param_name, param_value)
                else:
                    # 是固定值，直接存储
                    fixed_section[param_name] = param_value
            
            if fixed_section:
                self.fixed_values[section_name] = fixed_section
            if search_section:
                self.search_space[section_name] = search_section
    
    def _is_search_space(self, value: Any) -> bool:
        """判断值是否是搜索空间定义（以 * 开头的字符串）"""
        return isinstance(value, str) and value.startswith('*')
    
    def _parse_search_space(self, param_name: str, value: str) -> Dict[str, Any]:
        """
        解析搜索空间字符串
        
        支持的格式：
        - *int:1-10          # 整数范围
        - *float:1.1-20.0    # 浮点数范围
        - *log:1e-5-1e-3     # 对数范围
        - *a,b,c             # 分类选择
        - *mul:a,b,c         # 分类选择（与 *a,b,c 相同）
        - *bool              # 布尔值选择
        """
        if not value.startswith('*'):
            raise ValueError(f"参数 '{param_name}' 的搜索空间定义必须以 * 开头: {value}")
        
        value = value[1:]  # 移除开头的 *
        
        # 处理布尔值
        if value == 'bool':
            return {
                'type': 'categorical',
                'choices': [True, False]
            }
        
        # 处理分类选择（*a,b,c 或 *mul:a,b,c）
        if value.startswith('mul:'):
            value = value[4:]  # 移除 'mul:'
        
        if ',' in value:
            # 分类选择
            choices_str = value.split(',')
            choices = []
            for choice_str in choices_str:
                choice_str = choice_str.strip()
                # 尝试转换为数字
                try:
                    if '.' in choice_str or 'e' in choice_str.lower():
                        choices.append(float(choice_str))
                    else:
                        choices.append(int(choice_str))
                except ValueError:
                    # 无法转换为数字，保持字符串
                    # 处理布尔值字符串
                    if choice_str.lower() == 'true':
                        choices.append(True)
                    elif choice_str.lower() == 'false':
                        choices.append(False)
                    elif choice_str.lower() == 'null' or choice_str.lower() == 'none':
                        choices.append(None)
                    else:
                        choices.append(choice_str)
            
            return {
                'type': 'categorical',
                'choices': choices
            }
        
        # 处理范围（*int:1-10, *float:1.1-20.0, *log:1e-5-1e-3）
        if ':' in value:
            type_part, range_part = value.split(':', 1)
            
            # 解析范围
            if '-' not in range_part:
                raise ValueError(f"参数 '{param_name}' 的范围定义必须包含 - 分隔符: {value}")
            
            min_str, max_str = range_part.split('-', 1)
            min_str = min_str.strip()
            max_str = max_str.strip()
            
            # 判断类型
            if type_part == 'log':
                # 对数范围（浮点数）
                try:
                    min_val = float(min_str)
                    max_val = float(max_str)
                except ValueError:
                    raise ValueError(f"参数 '{param_name}' 的对数范围值必须是数字: {value}")
                
                if min_val <= 0 or max_val <= 0:
                    raise ValueError(f"参数 '{param_name}' 的对数范围值必须大于 0: {value}")
                
                return {
                    'type': 'float',
                    'range': [min_val, max_val],
                    'log': True
                }
            
            elif type_part == 'int':
                # 整数范围
                try:
                    min_val = int(min_str)
                    max_val = int(max_str)
                except ValueError:
                    raise ValueError(f"参数 '{param_name}' 的整数范围值必须是整数: {value}")
                
                if min_val >= max_val:
                    raise ValueError(f"参数 '{param_name}' 的范围最小值必须小于最大值: {value}")
                
                return {
                    'type': 'int',
                    'range': [min_val, max_val]
                }
            
            elif type_part == 'float':
                # 浮点数范围
                try:
                    min_val = float(min_str)
                    max_val = float(max_str)
                except ValueError:
                    raise ValueError(f"参数 '{param_name}' 的浮点数范围值必须是数字: {value}")
                
                if min_val >= max_val:
                    raise ValueError(f"参数 '{param_name}' 的范围最小值必须小于最大值: {value}")
                
                return {
                    'type': 'float',
                    'range': [min_val, max_val],
                    'log': False
                }
            
            else:
                raise ValueError(f"参数 '{param_name}' 不支持的类型前缀: {type_part}，支持的类型: int, float, log")
        
        # 如果没有类型前缀，尝试自动推断
        if '-' in value:
            # 可能是范围，尝试解析
            min_str, max_str = value.split('-', 1)
            min_str = min_str.strip()
            max_str = max_str.strip()
            
            # 尝试判断是整数还是浮点数
            try:
                if '.' in min_str or '.' in max_str or 'e' in min_str.lower() or 'e' in max_str.lower():
                    # 浮点数
                    min_val = float(min_str)
                    max_val = float(max_str)
                    return {
                        'type': 'float',
                        'range': [min_val, max_val],
                        'log': False
                    }
                else:
                    # 整数
                    min_val = int(min_str)
                    max_val = int(max_str)
                    return {
                        'type': 'int',
                        'range': [min_val, max_val]
                    }
            except ValueError:
                raise ValueError(f"参数 '{param_name}' 无法解析范围: {value}")
        
        raise ValueError(f"参数 '{param_name}' 无法解析搜索空间定义: {value}")
    
    def _count_fixed_values(self) -> int:
        """统计固定值参数数量"""
        count = 0
        for section in self.fixed_values.values():
            if isinstance(section, dict):
                count += len(section)
        return count
    
    def _count_search_space(self) -> int:
        """统计搜索空间参数数量"""
        count = 0
        for section in self.search_space.values():
            if isinstance(section, dict):
                count += len(section)
        return count
    
    def get_fixed_values(self, section: str = None) -> Dict[str, Any]:
        """
        获取固定值
        
        Args:
            section: 配置节名称（如 'experiment_config'），如果为 None 则返回所有固定值
            
        Returns:
            固定值字典
        """
        if section:
            return self.fixed_values.get(section, {})
        return self.fixed_values
    
    def get_search_space(self, section: str = None) -> Dict[str, Any]:
        """
        获取搜索空间
        
        Args:
            section: 配置节名称（如 'experiment_config'），如果为 None 则返回所有搜索空间
            
        Returns:
            搜索空间字典
        """
        if section:
            return self.search_space.get(section, {})
        return self.search_space
    
    def suggest_params(self, trial: optuna.Trial, section: str) -> Dict[str, Any]:
        """
        根据搜索空间建议参数
        
        Args:
            trial: Optuna 试验对象
            section: 配置节名称（如 'experiment_config'）
            
        Returns:
            建议的参数字典
        """
        params = {}
        search_space = self.get_search_space(section)
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                # 分类选择
                choices = param_config['choices']
                params[param_name] = trial.suggest_categorical(param_name, choices)
            
            elif param_type == 'int':
                # 整数范围
                min_val, max_val = param_config['range']
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            
            elif param_type == 'float':
                # 浮点数范围
                min_val, max_val = param_config['range']
                log = param_config.get('log', False)
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=log)
        
        return params
    
    def create_full_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        创建完整配置（固定值 + Optuna 建议的参数）
        
        Args:
            trial: Optuna 试验对象
            
        Returns:
            完整配置字典
        """
        full_config = {}
        
        # 遍历所有配置节
        for section_name in set(list(self.fixed_values.keys()) + list(self.search_space.keys())):
            # 获取固定值
            fixed_section = self.get_fixed_values(section_name)
            
            # 获取 Optuna 建议的参数
            suggested_section = self.suggest_params(trial, section_name)
            
            # 合并配置（Optuna 建议的参数会覆盖固定值中的同名参数）
            full_config[section_name] = {
                **fixed_section,
                **suggested_section
            }
        
        return full_config


def create_config_template(output_file: str = 'optuna_config_template.yaml'):
    """
    创建配置模板文件（YAML 格式，使用字符串解析格式）
    
    Args:
        output_file: 输出文件路径（.yaml 或 .yml）
    """
    template = {
        "description": "Optuna 超参数优化配置模板（使用字符串解析格式）",
        "experiment_config": {
            "data_root_dir": "/path/to/data",
            "csv_path": "/path/to/labels.csv",
            "alignment_model_path": None,
            "aligned_channels": None,
            "opt": "adam",
            "scheduler_config": {
                "type": "plateau",
                "mode": "min",
                "patience": 15,
                "factor": 0.5
            },
            "target_channels": ["wsi", "tma", "clinical"],
            "exp_code": "my_experiment",
            "num_splits": 10,
            "split_mode": "fixed",
            "dataset_split_path": "/path/to/split.json",
            "seed": "*int:1-10000",
            "max_epochs": "*int:100-300",
            "lr": "*log:1e-5-1e-3",
            "reg": "*log:1e-6-1e-4",
            "early_stopping": "*bool",
            "batch_size": "*32,64,128,256",
            "dropout": "*float:0.1-0.9"
        },
        "model_config": {
            "input_dim": 1024,
            "n_classes": 2,
            "base_loss_fn": "ce",
            "model_type": "auc_clam",
            "model_size": "*64*32,32*16,16*8,8*4,4*2,2*1",
            "gate": "*bool",
            "base_weight": "*float:0.3-0.9",
            "inst_loss_fn": "*null,ce",
            "subtyping": "*bool",
            "inst_number": "*4,8",
            "return_features": "*bool",
            "auc_loss_weight": "*float:0.1-2.0",
            "output_dim": "*64,128,256",
            "enable_svd": "*bool",
            "alignment_layer_num": "*int:1-4",
            "lambda1": "*float:0.1-2.0",
            "lambda2": "*float:0.0-1.0",
            "tau1": "*float:0.01-0.5",
            "tau2": "*float:0.01-0.5",
            "enable_random_loss": "*bool",
            "weight_random_loss": "*float:0.01-1.0"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✅ 配置模板已保存到: {output_file}")


if __name__ == "__main__":
    # 测试和创建模板
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna 配置加载器工具')
    parser.add_argument('--create_template', type=str, default=None,
                       help='创建配置模板文件')
    parser.add_argument('--test', type=str, default=None,
                       help='测试配置文件')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_config_template(args.create_template)
    
    if args.test:
        loader = OptunaConfigLoader(args.test)
        print("\n固定值:")
        print(yaml.dump(loader.get_fixed_values(), default_flow_style=False, allow_unicode=True, sort_keys=False))
        print("\n搜索空间:")
        print(yaml.dump(loader.get_search_space(), default_flow_style=False, allow_unicode=True, sort_keys=False))
        
        # 测试解析
        print("\n测试解析搜索空间:")
        for section_name, section_space in loader.get_search_space().items():
            print(f"\n{section_name}:")
            for param_name, param_config in section_space.items():
                print(f"  {param_name}: {param_config}")

