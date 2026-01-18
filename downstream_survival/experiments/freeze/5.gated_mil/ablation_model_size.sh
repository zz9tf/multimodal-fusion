#!/bin/bash

# =============================================================================
# Model Size Ablation Study
# 测试不同model_size对模型性能的影响
# =============================================================================

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=3
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Data-related parameters
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# Experiment & Training parameters
EXP_CODE="ablation_model_size"
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
CONFIDENCE_WEIGHT=0.5
FEATURE_WEIGHT_WEIGHT=0.05
CHANNELS_USED_IN_MODEL="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 🔬 Model Size Ablation Study
# 测试不同的model_size值: 16*8, 32*16, 64*32, 128*64
MODEL_SIZES=("16*8" "32*16" "64*32" "128*64")

echo "🔬 Starting Model Size Ablation Study..."
echo "Testing model_size values: ${MODEL_SIZES[@]}"
echo "============================================================"

for model_size in "${MODEL_SIZES[@]}"; do
    echo ""
    echo "🚀 Running experiment with model_size = $model_size"
    echo "------------------------------------------------------------"
    
    # 创建特定的结果目录
    SPECIFIC_RESULTS_DIR="${RESULTS_DIR}/ablation_model_size_${model_size//\*/x}"
    mkdir -p "$SPECIFIC_RESULTS_DIR"
    
    # 运行训练
    python main.py \
        --data_root_dir "$DATA_ROOT_DIR" \
        --results_dir "$SPECIFIC_RESULTS_DIR" \
        --csv_path "$CSV_PATH" \
        --target_channel $TARGET_CHANNELS \
        --exp_code "${EXP_CODE}_${model_size//\*/x}" \
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
        --model_size "$model_size" \
        --confidence_weight $CONFIDENCE_WEIGHT \
        --feature_weight_weight $FEATURE_WEIGHT_WEIGHT \
        --channels_used_in_model $CHANNELS_USED_IN_MODEL
    
    echo "✅ Completed experiment with model_size = $model_size"
    echo "Results saved to: $SPECIFIC_RESULTS_DIR"
done

echo ""
echo "🎉 Model Size Ablation Study completed!"
echo "============================================================"
echo "📊 Summary of experiments:"
for model_size in "${MODEL_SIZES[@]}"; do
    echo "  - model_size = $model_size: ${RESULTS_DIR}/ablation_model_size_${model_size//\*/x}"
done
