# Ablation Study 统一配置文档

## 📋 概述

本文档定义了所有ablation study实验的统一配置标准，确保实验的一致性和可重复性。

## 🔧 统一配置参数

### 固定参数（所有实验保持一致）
```bash
# 基础配置
MISMATCH_RATIO=1.0          # 负样本比例
SEED=42                     # 随机种子
MAX_STEPS=2000              # 最大训练步数
BATCH_SIZE=64               # 批次大小
LEARNING_RATE=1e-4          # 学习率
WEIGHT_DECAY=1e-5           # 权重衰减

# 模型架构
NUM_LAYERS=2                # 对齐层数（除num_layers实验外）
LAMBDA1=1.0                 # 对比损失权重（除lambda1实验外）
LAMBDA2=0.1                 # 匹配预测损失权重（除lambda2实验外）
TAU1=0.1                    # 温度参数1（除tau1实验外）
TAU2=0.05                   # 温度参数2（除tau2实验外）

# 日志和验证频率
LOG_INTERVAL=200            # 日志输出间隔（每200步）
VAL_INTERVAL=400            # 验证间隔（每400步）
```

### 实验特定参数

#### 1. Lambda1 实验 (`ablation_lambda1.sh`)
- **GPU**: 0
- **测试值**: `(0.0 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)`
- **目的**: 测试对比损失权重对模型性能的影响

#### 2. Lambda2 实验 (`ablation_lambda2.sh`)
- **GPU**: 1
- **测试值**: `(0.0 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5)`
- **目的**: 测试匹配预测损失权重对模型性能的影响

#### 3. Tau1 实验 (`ablation_tau1.sh`)
- **GPU**: 2
- **测试值**: `(0.01 0.05 0.1 0.2 0.3 0.5 1.0 2.0 5.0 10.0)`
- **目的**: 测试温度参数1对模型性能的影响

#### 4. Tau2 实验 (`ablation_tau2.sh`)
- **GPU**: 3
- **测试值**: `(0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5)`
- **目的**: 测试温度参数2对模型性能的影响

#### 5. Num Layers 实验 (`ablation_num_layers.sh`)
- **GPU**: 0
- **测试值**: `(1 2 3 4 5 6 7 8 9 10)`
- **目的**: 测试不同对齐层数对模型性能的影响

#### 6. Mismatch Ratio 实验 (`ablation_mismatch_ratio.sh`)
- **GPU**: 1
- **测试值**: `(0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0 10.0)`
- **目的**: 测试不同的负样本比例对模型性能的影响

#### 7. Seed 实验 (`ablation_seed.sh`)
- **GPU**: 2
- **测试值**: `(42 123 456 789 1024 2048 3141 5926 8888 9999)`
- **目的**: 测试不同随机种子对模型性能的影响（评估模型稳定性）

#### 8. Loss2 Chunk Size 实验 (`ablation_loss2_chunk_size.sh`)
- **GPU**: 3
- **测试值**: `(2 4 8 16 32 64 128 256 512 1024)`
- **目的**: 测试loss2分块大小对模型性能和训练效率的影响

## 🎯 优化说明

### 日志和验证频率优化
- **LOG_INTERVAL=200**: 每200步输出一次训练日志，平衡信息量和性能
- **VAL_INTERVAL=400**: 每400步进行一次验证，确保及时监控过拟合

### GPU分配策略
- 使用4个GPU并行运行不同实验
- 避免GPU资源冲突
- 提高实验效率

### 参数范围优化
- 所有实验统一测试10个不同的参数值
- 参数范围覆盖从极小到极大的合理区间
- 确保实验结果的统计显著性

## 🚀 使用方法

### 运行单个实验
```bash
cd /home/zheng/zheng/multimodal-fusion/ablation_study
bash ablation_lambda1.sh
```

### 运行所有实验
```bash
cd /home/zheng/zheng/multimodal-fusion/ablation_study
bash run_all_ablations.sh
```

## 📊 结果分析

所有实验结果将保存在 `/home/zheng/zheng/multimodal-fusion/results/` 目录下，按实验类型分类：

```
results/
├── ablation_lambda1/
├── ablation_lambda2/
├── ablation_tau1/
├── ablation_tau2/
├── ablation_num_layers/
├── ablation_mismatch_ratio/
├── ablation_seed/
└── ablation_loss2_chunk_size/
```

每个实验目录包含：
- 模型权重文件 (`.pth`)
- 训练历史文件 (`.history.json`)
- 可视化图表 (`.png`)

## 🔍 注意事项

1. **环境要求**: 确保conda环境 `multimodal-fusion` 已正确配置
2. **GPU内存**: 根据GPU内存调整batch_size
3. **存储空间**: 确保有足够的磁盘空间存储实验结果
4. **实验顺序**: 建议按GPU分配顺序运行，避免资源冲突

## 📈 预期结果

通过统一的配置，我们可以：
- 确保实验结果的公平比较
- 提高实验的可重复性
- 优化训练效率
- 便于结果分析和可视化
