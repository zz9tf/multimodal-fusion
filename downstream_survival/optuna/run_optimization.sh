#!/bin/bash

cd /home/zheng/zheng/multimodal-fusion/downstream_survival/optuna

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1
export PYTHONPATH="/home/zheng/zheng/multimodal-fusion/downstream_survival:$PYTHONPATH"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion

# 基本配置
DATA_ROOT_DIR="/home/zheng/zheng/mini2/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
RESULTS_DIR="./optuna_results"
N_TRIALS=100
N_FOLDS=10
N_JOBS=10

# 目标通道
TARGET_CHANNELS="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 创建结果目录
mkdir -p $RESULTS_DIR

echo "🚀 开始 AUC_CLAM 超参数优化..."
echo "📊 试验次数: $N_TRIALS"
echo "📁 结果目录: $RESULTS_DIR"
echo "🎯 目标通道: $TARGET_CHANNELS"

# 运行优化（启用实时可视化）
python optuna_auc_clam_optimization.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --csv_path "$CSV_PATH" \
    --results_dir "$RESULTS_DIR" \
    --n_trials $N_TRIALS \
    --n_folds $N_FOLDS \
    --n_jobs $N_JOBS \
    --target_channels $TARGET_CHANNELS \
    --sampler tpe \
    --enable_realtime_viz \
    --viz_port 8080 \
    --study_name "auc_clam_optimization_$(date +%Y%m%d_%H%M%S)"

echo "✅ 优化完成！"
echo "📁 结果保存在: $RESULTS_DIR"
