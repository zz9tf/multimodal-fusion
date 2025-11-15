# Optuna Ask-and-Tell 模式使用指南

## 📖 简介

Optuna Ask-and-Tell 模式允许你手动控制 Optuna 的试验流程：
1. **Ask（请求）**: 从 Optuna 获取建议的超参数
2. **执行**: 使用这些参数运行你自己的程序
3. **Tell（报告）**: 将执行结果报告回 Optuna

这种模式特别适合以下场景：
- 需要在不同的机器或环境中运行试验
- 需要手动控制试验的执行时机
- 需要将 Optuna 集成到现有的工作流中
- 需要在分布式环境中运行试验

## 🚀 快速开始

### 方式一：使用命令行工具

#### 1. 请求参数

```bash
python optuna_ask_tell.py ask \
    --study_name my_study \
    --model_type auc_clam \
    --data_root_dir /path/to/data \
    --csv_path /path/to/labels.csv \
    --output_params trial_params.json
```

这会：
- 创建一个新的 study（如果不存在）
- 请求 Optuna 建议的参数
- 将参数保存到 `trial_params.json`

#### 2. 运行你的程序

使用保存的参数运行你的训练程序：

```python
# 你的训练代码
import json

# 加载参数
with open('trial_params.json', 'r') as f:
    params = json.load(f)

# 使用参数进行训练
configs = params['configs']
# ... 你的训练代码 ...
result_auc = train_model(configs)

# 将结果写回参数文件
params['result'] = result_auc
params['state'] = 'COMPLETE'
with open('trial_params.json', 'w') as f:
    json.dump(params, f, indent=2)
```

#### 3. 报告结果

```bash
python optuna_ask_tell.py tell \
    --study_name my_study \
    --params_file trial_params.json \
    --value 0.85
```

或者，如果参数文件中已经包含了结果：

```bash
python optuna_ask_tell.py tell \
    --study_name my_study \
    --params_file trial_params.json
```

### 方式二：在 Python 代码中使用

```python
from optuna_ask_tell import OptunaAskTellManager
from main import parse_channels

# 1. 创建管理器
manager = OptunaAskTellManager(
    study_name='my_study',
    model_type='auc_clam',
    results_dir='./optuna_results',
    data_root_dir='/path/to/data',
    csv_path='/path/to/labels.csv'
)

# 2. 请求参数
target_channels = parse_channels(['wsi', 'tma', 'clinical'])
params_result = manager.ask(
    target_channels=target_channels,
    save_params_file='trial_params.json'
)

# 3. 运行你的训练代码
configs = params_result['configs']
# ... 你的训练代码 ...
auc_value = train_model(configs)  # 返回 AUC 分数

# 4. 报告结果
manager.tell(
    trial_id=params_result['trial_id'],
    trial_number=params_result['trial_number'],
    value=auc_value
)

# 5. 查看摘要
summary = manager.get_study_summary()
print(f"最佳值: {summary['best_value']}")
```

## 📋 完整示例

查看 `example_use_ask_tell.py` 了解完整的使用示例。

### 运行单个试验

```bash
python example_use_ask_tell.py \
    --study_name my_study \
    --model_type auc_clam \
    --data_root_dir /path/to/data \
    --csv_path /path/to/labels.csv
```

### 运行多个试验

```bash
python example_use_ask_tell.py \
    --study_name my_study \
    --model_type auc_clam \
    --n_trials 10 \
    --data_root_dir /path/to/data \
    --csv_path /path/to/labels.csv
```

## 🔧 API 参考

### OptunaAskTellManager

#### `__init__(study_name, model_type, results_dir, sampler, pruner, **kwargs)`

创建管理器实例。

**参数：**
- `study_name`: 研究名称（字符串）
- `model_type`: 模型类型（字符串）
- `results_dir`: 结果保存目录（字符串，默认：'./optuna_results'）
- `sampler`: 采样器类型（'tpe', 'random', 'cmaes'，默认：'tpe'）
- `pruner`: 是否启用剪枝（布尔值，默认：True）
- `**kwargs`: 其他参数（如 `data_root_dir`, `csv_path` 等）

#### `ask(target_channels=None, save_params_file=None)`

请求 Optuna 建议的参数。

**参数：**
- `target_channels`: 目标通道列表（可选）
- `save_params_file`: 保存参数的文件路径（可选）

**返回：**
包含以下键的字典：
- `trial_number`: 试验编号
- `trial_id`: 内部 trial ID（用于 tell）
- `experiment_params`: 实验参数
- `model_params`: 模型参数
- `configs`: 完整配置（如果提供了 target_channels）
- `params_file`: 参数文件路径（如果保存了）

#### `tell(trial_number=None, trial_id=None, value=None, state=None, params_file=None)`

报告试验结果。

**参数：**
- `trial_number`: 试验编号（可选）
- `trial_id`: 内部 trial ID（必需，除非提供了 params_file）
- `value`: 目标函数值（如 AUC 分数）
- `state`: 试验状态（可选，默认根据 value 自动判断）
- `params_file`: 参数文件路径（可选，会从中读取 trial_id 和结果）

#### `get_best_params()`

获取当前最佳参数。

**返回：**
包含以下键的字典：
- `trial_number`: 最佳试验编号
- `value`: 最佳值
- `params`: 最佳参数

#### `get_study_summary()`

获取研究摘要。

**返回：**
包含研究统计信息的字典。

#### `save_results(output_file=None)`

保存试验结果到文件。

**参数：**
- `output_file`: 输出文件路径（可选，默认使用 study_name）

## 📁 参数文件格式

参数文件（JSON 格式）包含以下字段：

```json
{
  "trial_number": 0,
  "trial_id": 0,
  "experiment_params": {
    "lr": 0.0001,
    "batch_size": 64,
    "max_epochs": 200,
    ...
  },
  "model_params": {
    "model_size": "64*32",
    "gate": true,
    ...
  },
  "configs": {
    "experiment_config": {...},
    "model_config": {...}
  },
  "timestamp": "2024-01-01T12:00:00",
  "result": 0.85,  // 可选：训练结果
  "state": "COMPLETE"  // 可选：试验状态
}
```

## 💡 使用场景

### 场景 1: 在不同机器上运行试验

1. 在机器 A 上请求参数：
```bash
python optuna_ask_tell.py ask --study_name my_study --output_params trial_0.json
```

2. 将 `trial_0.json` 传输到机器 B

3. 在机器 B 上运行训练，并将结果写入文件：
```python
# 训练代码
params = json.load(open('trial_0.json'))
result = train(params['configs'])
params['result'] = result
params['state'] = 'COMPLETE'
json.dump(params, open('trial_0.json', 'w'), indent=2)
```

4. 将更新后的 `trial_0.json` 传回机器 A

5. 在机器 A 上报告结果：
```bash
python optuna_ask_tell.py tell --study_name my_study --params_file trial_0.json
```

### 场景 2: 批量请求参数，稍后执行

```bash
# 请求 10 个试验的参数
for i in {0..9}; do
    python optuna_ask_tell.py ask \
        --study_name my_study \
        --output_params trial_${i}.json
done

# 稍后执行（可以在不同的时间、不同的机器上）
for i in {0..9}; do
    # 运行训练
    python your_training_script.py --params_file trial_${i}.json
    
    # 报告结果
    python optuna_ask_tell.py tell \
        --study_name my_study \
        --params_file trial_${i}.json
done
```

### 场景 3: 集成到现有工作流

```python
from optuna_ask_tell import OptunaAskTellManager

# 在你的工作流中
manager = OptunaAskTellManager(study_name='my_study', ...)

# 请求参数
params = manager.ask()

# 提交到任务队列（如 SLURM、Kubernetes 等）
submit_job(params)

# 任务完成后，从结果中读取并报告
result = get_job_result(job_id)
manager.tell(trial_id=params['trial_id'], value=result['auc'])
```

## 🔍 查看结果

### 查看研究摘要

```bash
python optuna_ask_tell.py summary --study_name my_study
```

### 查看最佳参数

```python
from optuna_ask_tell import OptunaAskTellManager

manager = OptunaAskTellManager(study_name='my_study')
best_params = manager.get_best_params()
print(f"最佳值: {best_params['value']}")
print(f"最佳参数: {best_params['params']}")
```

### 保存结果到 CSV

```python
manager.save_results('results.csv')
```

## ⚠️ 注意事项

1. **Trial ID 的重要性**: `trial_id` 是 Optuna 内部使用的标识符，必须正确传递给 `tell()` 方法。建议始终保存参数文件，以便后续报告结果。

2. **状态管理**: 
   - `COMPLETE`: 试验成功完成
   - `FAIL`: 试验失败
   - `PRUNED`: 试验被剪枝（通常不需要手动设置）

3. **并发安全**: JournalStorage 支持多进程/多机器并发，但建议：
   - 每个试验使用唯一的参数文件
   - 在报告结果前确保训练已完成

4. **参数文件**: 参数文件包含 `trial_id`，这是报告结果所必需的。请妥善保存参数文件，直到成功报告结果。

## 🆚 与标准模式的对比

| 特性 | 标准模式 (`study.optimize()`) | Ask-and-Tell 模式 |
|------|------------------------------|-------------------|
| 执行控制 | Optuna 自动执行 | 手动控制 |
| 分布式支持 | 需要共享存储 | 支持（通过参数文件） |
| 灵活性 | 较低 | 高 |
| 使用复杂度 | 简单 | 中等 |
| 适用场景 | 单机/集群自动优化 | 分布式/手动控制 |

## 📚 更多信息

- [Optuna 官方文档 - Ask-and-Tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)
- [Optuna JournalStorage 文档](https://optuna.readthedocs.io/en/stable/reference/storages.html#optuna.storages.JournalStorage)

