#!/bin/bash

# =============================================================================
# 环境设置
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/2"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# 实验 & 训练参数
EXP_CODE="mfmf"
SEED=5678
K_FOLDS=10
SPLIT_MODE="random"
MAX_EPOCHS=200
LEARNING_RATE=1e-4
LR_SCHEDULER="plateau"
LR_SCHEDULER_PARAMS='{"mode": "min", "patience": 15, "factor": 0.5}'
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"  # 启用早停
BATCH_SIZE=64

# 模型参数
MODEL_TYPE="mfmf"
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
CHANNELS_USED_IN_MODEL="wsi tma clinical pathological blood icd tma_cell_density"
OUTPUT_DIM=128

# Attention相关参数
ATTENTION_NUM_HEADS=8
# other tma | wsi | reconstruct
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "tma"}, {"q": "result", "kv": "wsi"}, {"q": "reconstruct", "kv": "result"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "other"}, {"q": "result", "kv": "wsi"}, {"q": "reconstruct", "kv": "result"}]'
# # other tma | reconstruct | wsi
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "tma"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}]'
FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "other"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}]'
# # other wsi | tma | reconstruct
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "wsi"}, {"q": "result", "kv": "tma"}, {"q": "reconstruct", "kv": "result"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "wsi", "kv": "other"}, {"q": "result", "kv": "tma"}, {"q": "reconstruct", "kv": "result"}]'
# # other wsi | reconstruct | tma
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "wsi"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "wsi", "kv": "other"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}]'
# # other reconstruct | tma | wsi
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}, {"q": "result", "kv": "wsi"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "other"}, {"q": "result", "kv": "tma"}, {"q": "result", "kv": "wsi"}]'
# # other reconstruct | wsi | tma
# FUSION_BLOCKS_SEQUENCE='[{"q": "other", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "tma"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "other"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "tma"}]'

# # tma reconstruct | other | wsi
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
# # tma reconstruct | wsi | other
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "other"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "other"}]'
# # tma wsi | other | reconstruct
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "wsi"}, {"q": "result", "kv": "other"}, {"q": "reconstruct", "kv": "result"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "wsi", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "reconstruct", "kv": "result"}]'
# # tma wsi | reconstruct | other
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "wsi"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "other"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "wsi", "kv": "tma"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "other"}]'

# # reconstruct tma | other | wsi
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
# # reconstruct wsi | tma | other
# FUSION_BLOCKS_SEQUENCE='[{"q": "reconstruct", "kv": "wsi"}, {"q": "result", "kv": "tma"}, {"q": "other", "kv": "result"}]'
# FUSION_BLOCKS_SEQUENCE='[{"q": "wsi", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}, {"q": "other", "kv": "result"}]'


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
    --attention_num_heads $ATTENTION_NUM_HEADS \
    --fusion_blocks_sequence "$FUSION_BLOCKS_SEQUENCE"
