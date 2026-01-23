#!/bin/bash

# =============================================================================
# 输出维度（output_dim）消融实验脚本
# =============================================================================
set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zz/zheng/multimodal-fusion/downstream_survival

# Accept random seed as command line argument
SEED=${1:-5678}

# Device and public directory assignment
CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# =============================================================================
# Data-related parameters
# =============================================================================
DATA_ROOT_DIR="/home/zz/zheng/mini2/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zz/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zz/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# =============================================================================
# Experiment & Training parameters（除 output_dim 外与基线一致）
# =============================================================================
K_FOLDS=10
MAX_EPOCHS=200
LEARNING_RATE=1e-4
LR_SCHEDULER="plateau"
LR_SCHEDULER_PARAMS='{"mode": "min", "patience": 15, "factor": 0.5}'
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"  # 启用早停
BATCH_SIZE=64

# =============================================================================
# 模型参数
# =============================================================================
MODEL_TYPE="clam_detach"
INPUT_DIM=1024
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"

# =============================================================================
# CLAM 特定参数
# =============================================================================
GATE="--gate"
BASE_WEIGHT=0.9
INST_LOSS_FN="ce"
MODEL_SIZE="64*32"
SUBTYPING="--subtyping"
INST_NUMBER=8
CHANNELS_USED_IN_MODEL="wsi tma clinical pathological blood icd tma_cell_density"

# =============================================================================
# 消融列表：需要测试的输出维度
# =============================================================================
OUTPUT_DIMS=(18 36 64 256 512)

# =============================================================================
# 运行消融实验（循环不同的 output_dim）
# =============================================================================
for OUTPUT_DIM in "${OUTPUT_DIMS[@]}"; do
    EXP_CODE="ablate_output_dim_${OUTPUT_DIM}"

    echo "==============================="
    echo "[Ablation] output_dim=${OUTPUT_DIM}"
    echo "exp_code=${EXP_CODE}"
    echo "==============================="

    python main.py \
        --data_root_dir "$DATA_ROOT_DIR" \
        --results_dir "$RESULTS_DIR" \
        --csv_path "$CSV_PATH" \
        --target_channel $TARGET_CHANNELS \
        --exp_code "$EXP_CODE" \
        --seed $SEED \
        --k $K_FOLDS \
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
        $GATE \
        --base_weight $BASE_WEIGHT \
        --inst_loss_fn $INST_LOSS_FN \
        --model_size $MODEL_SIZE \
        $SUBTYPING \
        --inst_number $INST_NUMBER \
        --channels_used_in_model $CHANNELS_USED_IN_MODEL \
        --output_dim $OUTPUT_DIM
done


