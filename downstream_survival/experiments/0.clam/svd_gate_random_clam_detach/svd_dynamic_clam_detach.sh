#!/bin/bash

# =============================================================================
# SVD + Dynamic Gate 实验脚本（不启用 Random Loss）
# =============================================================================

# 环境设置
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/1"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# 实验 & 训练参数
EXP_CODE="svd_dynamic_clam_detach"
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
MODEL_TYPE="svd_gate_random_clam_detach"  # 复用组合模型，但不启用随机损失
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

# SVD参数
ENABLE_SVD="--enable_svd"
ALIGNMENT_LAYER_NUM=2
LAMBDA1=0.1
LAMBDA2=0.1
TAU1=1.0
TAU2=1.0

# Dynamic Gate参数
ENABLE_DYNAMIC_GATE="--enable_dynamic_gate"
CONFIDENCE_WEIGHT=0.1
FEATURE_WEIGHT_WEIGHT=0.1

echo "🚀 开始 SVD + Dynamic Gate 实验..."

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
    $ENABLE_DYNAMIC_GATE \
    --confidence_weight $CONFIDENCE_WEIGHT \
    --feature_weight_weight $FEATURE_WEIGHT_WEIGHT

echo "✅ SVD + Dynamic Gate 实验完成!"
