#!/bin/bash

# =============================================================================
# Dynamic Gate + Random Loss 实验脚本（尽量不启用 SVD 的影响）
# 说明：当前模型默认启用SVD，为尽量隔离其影响，将SVD损失权重设为0，并增大tau。
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
EXP_CODE="dynamic_random_clam"
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
MODEL_TYPE="svd_gate_random_clam"
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

# 为尽量削弱SVD影响：对齐层数置0，损失权重置0，温度增大
ALIGNMENT_LAYER_NUM=0
LAMBDA1=0.0
LAMBDA2=0.0
TAU1=1e6
TAU2=1e6

# Dynamic Gate参数
ENABLE_DYNAMIC_GATE="--enable_dynamic_gate"
CONFIDENCE_WEIGHT=0.1
FEATURE_WEIGHT_WEIGHT=0.1

# Random Loss参数
ENABLE_RANDOM_LOSS="--enable_random_loss"
WEIGHT_RANDOM_LOSS=0.1

echo "🚀 开始 Dynamic Gate + Random Loss 实验..."

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
    --alignment_layer_num $ALIGNMENT_LAYER_NUM \
    --lambda1 $LAMBDA1 \
    --lambda2 $LAMBDA2 \
    --tau1 $TAU1 \
    --tau2 $TAU2 \
    $ENABLE_DYNAMIC_GATE \
    --confidence_weight $CONFIDENCE_WEIGHT \
    --feature_weight_weight $FEATURE_WEIGHT_WEIGHT \
    $ENABLE_RANDOM_LOSS \
    --weight_random_loss $WEIGHT_RANDOM_LOSS

echo "✅ Dynamic Gate + Random Loss 实验完成!"
