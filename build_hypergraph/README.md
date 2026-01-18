# Hypergraph Preprocessing Pipeline

这个模块实现了WSI和TMA数据的hypergraph预处理流程，将计算密集的部分提前完成，减少训练时的计算开销。

## 流程概述

1. **WSI聚合**：将大量WSI patches聚合成super patches（基于similarity的KMeans聚类）
2. **Similarity计算**：计算WSI super patches和TMA patches之间的similarity
3. **分组**：基于similarity将WSI super patches和TMA分成n个组
4. **Hypergraph构建**：使用KNN和KMeans构建hypergraph结构
5. **存储**：将预处理结果存储回h5文件，供训练时使用

## 使用方法

### 预处理数据

```bash
python -m build_hypergraph.preprocess_hypergraph \
    --csv_path downstream_survival/dataset_csv/survival_dataset.csv \
    --data_root_dir /path/to/data/root \
    --num_wsi_super_patches 100 \
    --num_groups 10 \
    --hypergraph_k 5 \
    --num_hyperedges 10 \
    --lambda_h 1.0 \
    --lambda_g 1.0 \
    --output_stats stats.json
```

### 参数说明

- `--num_wsi_super_patches`: WSI聚合后的super patch数量（默认100）
- `--num_groups`: similarity分组数量（默认10）
- `--hypergraph_k`: KNN的k值（默认5）
- `--num_hyperedges`: KMeans生成的hyperedges数量（默认10）
- `--lambda_h`: morphological similarity的缩放参数（默认1.0）
- `--lambda_g`: spatial similarity的缩放参数（默认1.0）
- `--output_stats`: 保存统计信息的JSON文件路径

### 在训练中使用预处理数据

在配置文件中，将channels设置为：

```python
channels = [
    'hypergraph=wsi_super_features',
    'hypergraph=tma_features',
    'hypergraph=edge_index',
    'clinical=val',  # 其他模态
    # ...
]
```

模型会自动检测并使用预处理好的hypergraph数据。

## 存储格式

预处理后的数据存储在h5文件的`hypergraph`组下：

```
hypergraph/
├── wsi_super/
│   ├── features      # [N_super, D] WSI super patch features
│   └── positions     # [N_super, 2/3] WSI super patch positions
├── tma/
│   └── features      # [N_tma, D] TMA features
├── similarity/       # Similarity matrices for fast parameter tuning
│   ├── wsi_internal  # [N_wsi, N_wsi] WSI internal similarity matrix
│   └── wsi_tma       # [N_super, N_tma] WSI-TMA similarity matrix
├── edge_index        # [2, E] Hypergraph edge indices
├── edge_weights      # [E] Edge weights (optional)
└── group_labels      # [N_super] Group labels
```

**关键点**：`similarity/`目录下存储的similarity matrices允许快速调整参数而无需重新计算similarity，这是快速实验的关键。

## 优势

1. **减少训练时计算**：similarity计算和hypergraph构建在预处理阶段完成
2. **快速参数调优**：存储similarity matrices，可以快速调整聚合尺度而无需重新计算
3. **可重复使用**：预处理结果可多次使用，无需重复计算
4. **灵活配置**：支持不同的聚合参数和hypergraph构建策略

## 快速参数调优

预处理时会自动存储similarity matrices到h5文件中。之后可以快速调整参数而无需重新计算similarity：

### 快速重建单个文件

```python
from build_hypergraph import rebuild_hypergraph_from_similarity

# 快速调整参数，无需重新计算similarity
rebuild_hypergraph_from_similarity(
    h5_path='path/to/file.h5',
    num_wsi_super_patches=150,  # 调整super patch数量
    num_groups=15,               # 调整分组数量
    hypergraph_k=7,              # 调整KNN的k
    threshold_median_ratio=0.5   # 调整edge过滤阈值
)
```

### 批量快速重建

```python
from build_hypergraph import batch_rebuild_hypergraph

# 批量调整整个数据集的参数
batch_rebuild_hypergraph(
    csv_path='dataset.csv',
    data_root_dir='/path/to/data',
    num_wsi_super_patches=150,
    num_groups=15,
    output_stats_path='new_stats.json'
)
```

### 命令行快速重建

使用提供的示例脚本：

```bash
python build_hypergraph/quick_rebuild_example.py \
    --csv_path downstream_survival/dataset_csv/survival_dataset.csv \
    --data_root_dir /path/to/data \
    --num_wsi_super_patches 150 \
    --num_groups 15 \
    --hypergraph_k 7 \
    --output_stats new_stats.json
```

或者直接使用Python：

```bash
python -c "
from build_hypergraph import batch_rebuild_hypergraph
batch_rebuild_hypergraph(
    csv_path='downstream_survival/dataset_csv/survival_dataset.csv',
    data_root_dir='/path/to/data',
    num_wsi_super_patches=150,
    num_groups=15
)
"
```

**性能优势**：这样可以在几秒到几分钟内尝试不同的参数组合，而不需要重新计算similarity（通常需要几分钟到几小时，取决于数据集大小）。

## 注意事项

- 如果h5文件中没有预处理好的hypergraph数据，模型会自动回退到使用原始WSI/TMA features
- 建议先在小数据集上测试参数，找到合适的`num_wsi_super_patches`和`num_groups`
- similarity scores会保存在统计文件中，可用于分析不同参数的效果

