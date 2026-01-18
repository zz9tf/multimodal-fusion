#!/bin/bash

# =============================================================================
# SVD + Random Loss 实验脚本（尽量不启用 Dynamic Gate 的影响）
# 说明：当前模型默认启用动态门控，为尽量隔离其影响，将权重设为0。
# =============================================================================

# Environment Setup
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Data-related parameters
DATA_ROOT_DIR="/home/zheng/zheng/public/4"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# Experiment & Training parameters
EXP_CODE="ds_svd_random"
SEED=5678
K_FOLDS=10
SPLIT_MODE="random"
MAX_EPOCHS=200
LEARNING_RATE=1e-4
LR_SCHEDULER="plateau"
LR_SCHEDULER_PARAMS='{"mode": "min", "patience": 15, "factor": 0.5}'
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"
BATCH_SIZE=64

# 模型与CLAM参数
MODEL_TYPE="deep_supervise_svd_gate_random"
INPUT_DIM=1024
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"
GATE="--gate"
BASE_WEIGHT=0.9
INST_LOSS_FN="ce"
MODEL_SIZE="64*32"
SUBTYPING="--subtyping"
INST_NUMBER=8
CHANNELS_USED_IN_MODEL="wsi tma clinical pathological blood icd tma_cell_density"
OUTPUT_DIM=128

# SVD特定参数 - 启用SVD对齐
ENABLE_SVD="--enable_svd"
ALIGNMENT_LAYER_NUM=2
LAMBDA1=0.1
LAMBDA2=0.1
TAU1=1.0
TAU2=1.0

# Random Loss参数
ENABLE_RANDOM_LOSS="--enable_random_loss"
WEIGHT_RANDOM_LOSS=0.1

echo "🚀 开始 Deep Supervise + SVD + Random Loss 实验..."
echo "📊 实验代码: $EXP_CODE"
echo "🎯 目标通道: $TARGET_CHANNELS"
echo "🔧 SVD参数: ENABLE_SVD=$ENABLE_SVD, ALIGNMENT_LAYER_NUM=$ALIGNMENT_LAYER_NUM, LAMBDA1=$LAMBDA1, LAMBDA2=$LAMBDA2, TAU1=$TAU1, TAU2=$TAU2"
echo "🔧 Random Loss参数: WEIGHT_RANDOM_LOSS=$WEIGHT_RANDOM_LOSS"

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
    --exp_code "$EXP_CODE" \
    --seed $SEED \
    --k $K_FOLDS \
    --split_mode $SPLIT_MODE \
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
    --gate $GATE \
    --base_weight $BASE_WEIGHT \
    --inst_loss_fn $INST_LOSS_FN \
    --model_size $MODEL_SIZE \
    --subtyping $SUBTYPING \
    --inst_number $INST_NUMBER \
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --output_dim $OUTPUT_DIM \
    $ENABLE_SVD \
    --alignment_layer_num $ALIGNMENT_LAYER_NUM \
    --lambda1 $LAMBDA1 \
    --lambda2 $LAMBDA2 \
    --tau1 $TAU1 \
    --tau2 $TAU2 \
    $ENABLE_RANDOM_LOSS \
    --weight_random_loss $WEIGHT_RANDOM_LOSS

echo "✅ Deep Supervise + SVD + Random Loss 实验完成!"
