#!/bin/bash

# =============================================================================
# Feature Weight Ablation Study
# 测试不同feature_weight_weight对模型性能的影响
# =============================================================================

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=2
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 实验 & 训练参数
EXP_CODE="ablation_feature_weight"
SEED=5678
K_FOLDS=10
MAX_EPOCHS=200
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"
BATCH_SIZE=128

# 模型参数
MODEL_TYPE="gate_mil"
INPUT_DIM=1024
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"

# 固定参数
MODEL_SIZE="16*8"
CONFIDENCE_WEIGHT=0.5
CHANNELS_USED_IN_MODEL="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 🔬 Feature Weight Ablation Study
# 测试不同的feature_weight_weight值: 0.0, 0.01, 0.05, 0.1, 0.2, 0.5
FEATURE_WEIGHTS=(0.0 0.01 0.05 0.1 0.2 0.5)

echo "🔬 Starting Feature Weight Ablation Study..."
echo "Testing feature_weight_weight values: ${FEATURE_WEIGHTS[@]}"
echo "============================================================"

for feat_weight in "${FEATURE_WEIGHTS[@]}"; do
    echo ""
    echo "🚀 Running experiment with feature_weight_weight = $feat_weight"
    echo "------------------------------------------------------------"
    
    # 创建特定的结果目录
    SPECIFIC_RESULTS_DIR="${RESULTS_DIR}/ablation_feature_weight_${feat_weight}"
    mkdir -p "$SPECIFIC_RESULTS_DIR"
    
    # 运行训练
    python main.py \
        --data_root_dir "$DATA_ROOT_DIR" \
        --results_dir "$SPECIFIC_RESULTS_DIR" \
        --csv_path "$CSV_PATH" \
        --target_channel $TARGET_CHANNELS \
        --exp_code "${EXP_CODE}_${feat_weight}" \
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
        --model_size $MODEL_SIZE \
        --confidence_weight $CONFIDENCE_WEIGHT \
        --feature_weight_weight $feat_weight \
        --channels_used_in_model $CHANNELS_USED_IN_MODEL
    
    echo "✅ Completed experiment with feature_weight_weight = $feat_weight"
    echo "Results saved to: $SPECIFIC_RESULTS_DIR"
done

echo ""
echo "🎉 Feature Weight Ablation Study completed!"
echo "============================================================"
echo "📊 Summary of experiments:"
for feat_weight in "${FEATURE_WEIGHTS[@]}"; do
    echo "  - feature_weight_weight = $feat_weight: ${RESULTS_DIR}/ablation_feature_weight_${feat_weight}"
done
