#!/bin/bash

# =============================================================================
# Environment Setup
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Data-related parameters
DATA_ROOT_DIR="/home/zheng/zheng/public/2"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="wsi tma clinical pathological blood icd tma_cell_density"

# Experiment & Training parameters
CONFIG=2
EXP_CODE="mfmf$CONFIG"
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

# 定义fusion blocks sequence配置数组
# 索引从0开始，CONFIG=3表示使用第4个配置（索引3）
declare -a FUSION_BLOCKS_SEQUENCE_LIST=(
    # Config 0: other tma | wsi | reconstruct
    '[{"q": "other", "kv": "tma"}, {"q": "result", "kv": "wsi"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 1: tma other | wsi | reconstruct
    '[{"q": "tma", "kv": "other"}, {"q": "result", "kv": "wsi"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 2: other tma | reconstruct | wsi
    '[{"q": "other", "kv": "tma"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}]'
    # Config 3: tma other | reconstruct | wsi
    '[{"q": "tma", "kv": "other"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}]'
    # Config 4: other wsi | tma | reconstruct
    '[{"q": "other", "kv": "wsi"}, {"q": "result", "kv": "tma"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 5: wsi other | tma | reconstruct
    '[{"q": "wsi", "kv": "other"}, {"q": "result", "kv": "tma"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 6: other wsi | reconstruct | tma
    '[{"q": "other", "kv": "wsi"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}]'
    # Config 7: wsi other | reconstruct | tma
    '[{"q": "wsi", "kv": "other"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}]'
    # Config 8: other reconstruct | tma | wsi
    '[{"q": "other", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}, {"q": "result", "kv": "wsi"}]'
    # Config 9: reconstruct other | tma | wsi
    '[{"q": "reconstruct", "kv": "other"}, {"q": "result", "kv": "tma"}, {"q": "result", "kv": "wsi"}]'
    # Config 10: other reconstruct | wsi | tma
    '[{"q": "other", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "tma"}]'
    # Config 11: reconstruct other | wsi | tma
    '[{"q": "reconstruct", "kv": "other"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "tma"}]'
    # Config 12: tma reconstruct | other | wsi
    '[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
    # Config 13: reconstruct tma | other | wsi
    '[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
    # Config 14: tma reconstruct | wsi | other
    '[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "other"}]'
    # Config 15: reconstruct tma | wsi | other
    '[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "wsi"}, {"q": "result", "kv": "other"}]'
    # Config 16: tma wsi | other | reconstruct
    '[{"q": "tma", "kv": "wsi"}, {"q": "result", "kv": "other"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 17: wsi tma | other | reconstruct
    '[{"q": "wsi", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "reconstruct", "kv": "result"}]'
    # Config 18: tma wsi | reconstruct | other
    '[{"q": "tma", "kv": "wsi"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "other"}]'
    # Config 19: wsi tma | reconstruct | other
    '[{"q": "wsi", "kv": "tma"}, {"q": "result", "kv": "reconstruct"}, {"q": "result", "kv": "other"}]'
    # Config 20: reconstruct tma | other | wsi
    '[{"q": "reconstruct", "kv": "tma"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
    # Config 21: tma reconstruct | other | wsi
    '[{"q": "tma", "kv": "reconstruct"}, {"q": "result", "kv": "other"}, {"q": "result", "kv": "wsi"}]'
    # Config 22: reconstruct wsi | tma | other
    '[{"q": "reconstruct", "kv": "wsi"}, {"q": "result", "kv": "tma"}, {"q": "other", "kv": "result"}]'
    # Config 23: wsi reconstruct | tma | other
    '[{"q": "wsi", "kv": "reconstruct"}, {"q": "result", "kv": "tma"}, {"q": "other", "kv": "result"}]'
)

# 根据CONFIG选择对应的fusion blocks sequence
# 检查CONFIG是否在有效范围内
if [ "$CONFIG" -ge 0 ] && [ "$CONFIG" -lt "${#FUSION_BLOCKS_SEQUENCE_LIST[@]}" ]; then
    FUSION_BLOCKS_SEQUENCE="${FUSION_BLOCKS_SEQUENCE_LIST[$CONFIG]}"
    echo "Using Config $CONFIG: $FUSION_BLOCKS_SEQUENCE"
else
    echo "Error: CONFIG=$CONFIG is out of range (0-$((${#FUSION_BLOCKS_SEQUENCE_LIST[@]}-1)))"
    exit 1
fi


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
