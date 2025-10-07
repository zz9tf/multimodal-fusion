# Ablation Study 实验指南

## 📋 概述

本目录包含 7 个独立的 ablation study 脚本，用于系统性地评估不同超参数对模型性能的影响。每个脚本测试一个参数的 10 个不同值。

## 📂 脚本列表

| 脚本文件 | 测试参数 | 测试值数量 | 参数范围 |
|---------|---------|-----------|---------|
| `ablation_mismatch_ratio.sh` | 负样本比例 | 10 | 0.1 ~ 10.0 |
| `ablation_seed.sh` | 随机种子 | 10 | 42 ~ 9999 |
| `ablation_lambda1.sh` | 对比损失权重 | 10 | 0.0 ~ 5.0 |
| `ablation_lambda2.sh` | 匹配预测损失权重 | 10 | 0.0 ~ 0.5 |
| `ablation_tau1.sh` | 温度参数1 | 10 | 0.01 ~ 1.0 |
| `ablation_tau2.sh` | 温度参数2 | 10 | 0.01 ~ 0.5 |
| `ablation_num_layers.sh` | 对齐层数 | 10 | 1 ~ 10 |

## 🚀 使用方法

### 方式 1: 运行单个 Ablation Study

```bash
# 给脚本添加执行权限
chmod +x ablation_mismatch_ratio.sh

# 运行单个实验
./ablation_mismatch_ratio.sh
```

### 方式 2: 运行所有 Ablation Studies

```bash
# 给主脚本添加执行权限
chmod +x run_all_ablations.sh

# 运行所有实验（⚠️ 需要很长时间）
./run_all_ablations.sh
```

### 方式 3: 后台运行

```bash
# 后台运行并保存日志
nohup ./ablation_mismatch_ratio.sh > logs/ablation_mismatch_ratio.log 2>&1 &

# 查看进度
tail -f logs/ablation_mismatch_ratio.log
```

## 📊 实验配置

### 固定参数（所有实验共用）

```bash
MAX_STEPS=10000          # 训练步数
BATCH_SIZE=64            # 批次大小
LEARNING_RATE=1e-4       # 学习率
WEIGHT_DECAY=1e-5        # 权重衰减
LOG_INTERVAL=100         # 日志间隔
VAL_INTERVAL=500         # 验证间隔
```

### 默认参数值

```bash
MISMATCH_RATIO=1.0       # 负样本比例
SEED=42                  # 随机种子
LAMBDA1=1.0              # 对比损失权重
LAMBDA2=0.1              # 匹配预测损失权重
TAU1=0.1                 # 温度参数1
TAU2=0.05                # 温度参数2
NUM_LAYERS=2             # 对齐层数
```

## 📁 结果目录结构

```
results/
├── ablation_mismatch_ratio/
│   ├── model_ratio_0.1.pth
│   ├── model_ratio_0.1.history.json
│   ├── model_ratio_0.3.pth
│   ├── model_ratio_0.3.history.json
│   └── ...
├── ablation_seed/
│   ├── model_seed_42.pth
│   ├── model_seed_42.history.json
│   └── ...
├── ablation_lambda1/
├── ablation_lambda2/
├── ablation_tau1/
├── ablation_tau2/
└── ablation_num_layers/
```

每个实验会生成：
- `.pth` 文件：最佳模型权重
- `.history.json` 文件：训练历史（loss、SVD值等）

## 📈 参数说明

### 1. `mismatch_ratio` - 负样本比例

**作用**: 控制负样本数量与正样本的比例

**测试值**: `0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0`

**预期影响**:
- 过小：模型难以学习区分匹配/不匹配样本
- 过大：训练时间增加，可能引入噪声
- 最优值：通常在 0.5-2.0 之间

### 2. `seed` - 随机种子

**作用**: 评估模型稳定性和可重复性

**测试值**: `42, 123, 456, 789, 1024, 2048, 3141, 5926, 8888, 9999`

**预期影响**:
- 不同 seed 的结果变化越小，模型越稳定
- 用于计算平均性能和标准差

### 3. `lambda1` - 对比损失权重

**作用**: 控制 SVD 秩约束的重要性

**测试值**: `0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0`

**预期影响**:
- 0.0: 完全不使用对比损失
- 过小：对齐效果不佳
- 过大：可能导致收敛困难

### 4. `lambda2` - 匹配预测损失权重

**作用**: 控制负样本判别的重要性

**测试值**: `0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5`

**预期影响**:
- 0.0: 完全不使用匹配预测损失
- 过小：难以区分正负样本
- 过大：可能偏向判别任务，忽略对齐

### 5. `tau1` - SVD 温度参数

**作用**: 控制 SVD 损失的平滑度

**测试值**: `0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0`

**预期影响**:
- 过小：损失过于陡峭，优化困难
- 过大：损失过于平滑，收敛慢

### 6. `tau2` - 对比学习温度参数

**作用**: 控制对比损失的平滑度

**测试值**: `0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5`

**预期影响**:
- 过小：过于关注困难样本
- 过大：忽略样本间差异

### 7. `num_layers` - 对齐层数

**作用**: 控制网络深度和表达能力

**测试值**: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`

**预期影响**:
- 1层：最简单，可能欠拟合
- 2-3层：通常最优
- >5层：可能过拟合或梯度消失

## 🔍 结果分析

### 分析单个实验

```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('results/ablation_lambda1/model_lambda1_1.0.history.json') as f:
    history = json.load(f)

# 获取最佳验证 loss
best_val_loss = min(history['val_losses'])
print(f"Best validation loss: {best_val_loss:.4f}")

# 绘制训练曲线
plt.plot(history['val_losses'])
plt.show()
```

### 对比所有实验

```python
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# 收集所有 lambda1 实验的结果
results = {}
for f in glob.glob('results/ablation_lambda1/*.history.json'):
    with open(f) as file:
        history = json.load(file)
        lambda1 = history['config']['lambda1']
        best_loss = min(history['val_losses'])
        results[lambda1] = best_loss

# 绘制对比图
lambdas = sorted(results.keys())
losses = [results[l] for l in lambdas]
plt.plot(lambdas, losses, marker='o')
plt.xlabel('lambda1')
plt.ylabel('Best Validation Loss')
plt.title('Lambda1 Ablation Study')
plt.show()
```

## ⏱️ 时间估算

假设每个实验需要 X 分钟（取决于 `MAX_STEPS`）：

- 单个 ablation study: 10 × X 分钟
- 所有 ablation studies: 70 × X 分钟

**建议**:
- 先用较小的 `MAX_STEPS` (如 2000) 快速测试
- 确认代码无误后，再用完整的步数运行

## 💡 使用建议

1. **优先级排序**: 根据预期影响，优先运行重要参数的 ablation
   - 高优先级: `lambda1`, `lambda2`, `num_layers`
   - 中优先级: `tau1`, `tau2`, `mismatch_ratio`
   - 低优先级: `seed` (用于最终评估)

2. **分批运行**: 不要一次运行所有实验
   - 先运行 1-2 个 ablation study
   - 分析结果后再继续

3. **调整参数范围**: 根据初步结果调整测试范围
   - 如果最优值在边界，扩展范围
   - 如果变化不大，缩小范围增加精度

4. **GPU 资源**: 可以修改脚本使用不同的 GPU
   ```bash
   # 编辑脚本，修改 CUDA_VISIBLE_DEVICES
   CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0
   CUDA_VISIBLE_DEVICES=1  # 使用 GPU 1
   ```

## 🐛 故障排除

### 问题：脚本无法执行
```bash
chmod +x ablation_*.sh
chmod +x run_all_ablations.sh
```

### 问题：GPU 内存不足
- 减小 `BATCH_SIZE`
- 减少 `num_layers`

### 问题：磁盘空间不足
- 每个模型约 100-200 MB
- 70 个实验约需要 7-14 GB

### 问题：想要中断某个实验
```bash
# 查找进程
ps aux | grep run.py

# 终止进程
kill -9 <PID>
```

## 📞 总结

这套 ablation study 脚本提供了：
- ✅ 7 个关键超参数的系统性评估
- ✅ 每个参数 10 个测试值
- ✅ 自动化运行和结果保存
- ✅ 清晰的目录结构

运行完成后，你将获得完整的超参数分析，用于论文、报告或进一步优化！

