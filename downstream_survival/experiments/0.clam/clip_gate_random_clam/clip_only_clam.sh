#!/bin/bash

# =============================================================================
# SVD-only 实验脚本
# 仅使用SVD对齐功能的多模态融合实验
# =============================================================================

# 环境设置
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/3"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# 实验 & 训练参数
EXP_CODE="clip_only_clam"
SEED=5678
K_FOLDS=10
SPLIT_MODE="fixed"
DATASET_SPLIT_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/dataset_split_in.json"
MAX_EPOCHS=200
LEARNING_RATE=1e-4
LR_SCHEDULER="plateau"
LR_SCHEDULER_PARAMS='{"mode": "min", "patience": 15, "factor": 0.5}'
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"  # 启用早停
BATCH_SIZE=64

# 模型参数 - 使用基础CLAM模型
MODEL_TYPE="clip_gate_random_clam"  # 使用基础CLAM模型，不包含gate和random功能
INPUT_DIM=1024
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"

# CLAM特定参数
BASE_WEIGHT=0.9
INST_LOSS_FN="ce"
MODEL_SIZE="64*32"
SUBTYPING="--subtyping"
INST_NUMBER=8
CHANNELS_USED_IN_MODEL="wsi tma clinical pathological blood icd tma_cell_density"
OUTPUT_DIM=128

# CLIP特定参数 - 启用CLIP对齐
ENABLE_CLIP="--enable_clip"
ALIGNMENT_LAYER_NUM=2
CLIP_INIT_TAU=0.07

echo "🚀 开始CLIP-only实验..."
echo "📊 实验代码: $EXP_CODE"
echo "🎯 目标通道: $TARGET_CHANNELS"
echo "🔧 CLIP参数: ALIGNMENT_LAYER_NUM=$ALIGNMENT_LAYER_NUM, CLIP_INIT_TAU=$CLIP_INIT_TAU"

# 运行训练
python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
    --exp_code "$EXP_CODE" \
    --seed $SEED \
    --k $K_FOLDS \
    --split_mode $SPLIT_MODE \
    --dataset_split_path $DATASET_SPLIT_PATH \
    --max_epochs $MAX_EPOCHS \
    --lr $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --lr_scheduler_params "$LR_SCHEDULER_PARAMS" \
    --reg $WEIGHT_DECAY \
    --opt $OPTIMIZER \
    $EARLY_STOPPING \
    --batch_size $BATCH_SIZE \
    --model_type $MODEL_TYPE \
    --input_dim $INPUT_DIM \
    --dropout $DROPOUT \
    --n_classes $N_CLASSES \
    --base_loss_fn $BASE_LOSS_FN \
    --base_weight $BASE_WEIGHT \
    --inst_loss_fn $INST_LOSS_FN \
    --model_size $MODEL_SIZE \
    --subtyping $SUBTYPING \
    --inst_number $INST_NUMBER \
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --output_dim $OUTPUT_DIM \
    $ENABLE_CLIP \
    --alignment_layer_num $ALIGNMENT_LAYER_NUM \
    --clip_init_tau $CLIP_INIT_TAU


echo "✅ CLIP-only实验完成!"
