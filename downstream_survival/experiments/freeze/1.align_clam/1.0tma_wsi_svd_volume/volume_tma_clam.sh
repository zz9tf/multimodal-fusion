#!/bin/bash

# =============================================================================
# 环境设置
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
ALIGNMENT_MODEL_PATH="/home/zheng/zheng/multimodal-fusion/alignment/results/volume/test_volume_multimodal_alignment_model.pth"
TARGET_CHANNELS="tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"
ALIGNED_CHANNELS="tma_CD3=CD3 tma_CD8=CD8 tma_CD56=CD56 tma_CD68=CD68 tma_CD163=CD163 tma_HE=HE tma_MHC1=MHC1 tma_PDL1=PDL1"

# 实验 & 训练参数
EXP_CODE="volume_tma_tma_wsi_clam" # volume alignment tma & normal tma & wsi with clam model
SEED=5678
K_FOLDS=10
MAX_EPOCHS=200
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"  # 启用早停
BATCH_SIZE=1

# 模型参数
MODEL_TYPE="clam"
INPUT_DIM=1024
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"

# CLAM特定参数
GATE="--gate"
BASE_WEIGHT=0.9
INST_LOSS_FN="ce"
MODEL_SIZE="64*32"
SUBTYPING="--subtyping"
INST_NUMBER=8
CHANNELS_USED_IN_MODEL="aligned_tma_CD3 aligned_tma_CD8 aligned_tma_CD56 aligned_tma_CD68 aligned_tma_CD163 aligned_tma_HE aligned_tma_MHC1 aligned_tma_PDL1"

# 运行训练
python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --alignment_model_path "$ALIGNMENT_MODEL_PATH" \
    --target_channel $TARGET_CHANNELS \
    --aligned_channels $ALIGNED_CHANNELS \
    --exp_code "$EXP_CODE" \
    --seed $SEED \
    --k $K_FOLDS \
    --max_epochs $MAX_EPOCHS \
    --lr $LEARNING_RATE \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL

