#!/bin/bash

# =============================================================================
# Learning Rate Scheduler Ablation Study for CLAM
# 对比不同学习率调度器对CLAM模型性能的影响
# =============================================================================

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Data-related parameters
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# Experiment & Training parameters
SEED=5678
K_FOLDS=10
MAX_EPOCHS=200
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"
BATCH_SIZE=128

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
CHANNELS_USED_IN_MODEL="features tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 🔬 Learning Rate Scheduler Ablation Study
echo "🔬 Starting Learning Rate Scheduler Ablation Study for CLAM..."
echo "============================================================"

# 1. 固定学习率 (baseline)
echo ""
echo "🚀 Running baseline experiment: Fixed Learning Rate"
echo "------------------------------------------------------------"
EXP_CODE="clam_lr_scheduler_fixed"
SPECIFIC_RESULTS_DIR="${RESULTS_DIR}/clam_lr_scheduler_fixed"
mkdir -p $SPECIFIC_RESULTS_DIR

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$SPECIFIC_RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --lr_scheduler none

# echo "✅ Completed baseline experiment: Fixed Learning Rate"

# 2. Cosine Annealing
echo ""
echo "🚀 Running experiment: Cosine Annealing"
echo "------------------------------------------------------------"
EXP_CODE="clam_lr_scheduler_cosine"

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$SPECIFIC_RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --lr_scheduler cosine \
    --lr_scheduler_params '{"T_max": 200, "eta_min": 1e-6}'

echo "✅ Completed experiment: Cosine Annealing"

# 3. Cosine Annealing with Warm Restart
echo ""
echo "🚀 Running experiment: Cosine Annealing with Warm Restart"
echo "------------------------------------------------------------"
EXP_CODE="clam_lr_scheduler_cosine_restart"

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$SPECIFIC_RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --lr_scheduler cosine_warm_restart \
    --lr_scheduler_params '{"T_0": 20, "T_mult": 2, "eta_min": 1e-6}'

echo "✅ Completed experiment: Cosine Annealing with Warm Restart"

# 4. Step LR
echo ""
echo "🚀 Running experiment: Step Learning Rate"
echo "------------------------------------------------------------"
EXP_CODE="clam_lr_scheduler_step"

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$SPECIFIC_RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --lr_scheduler step \
    --lr_scheduler_params '{"step_size": 50, "gamma": 0.5}'

echo "✅ Completed experiment: Step Learning Rate"

# 5. ReduceLROnPlateau
echo ""
echo "🚀 Running experiment: Reduce LR on Plateau"
echo "------------------------------------------------------------"
EXP_CODE="clam_lr_scheduler_plateau"

python main.py \
    --data_root_dir "$DATA_ROOT_DIR" \
    --results_dir "$SPECIFIC_RESULTS_DIR" \
    --csv_path "$CSV_PATH" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL \
    --lr_scheduler plateau \
    --lr_scheduler_params '{"mode": "min", "patience": 15, "factor": 0.5}'

echo "✅ Completed experiment: Reduce LR on Plateau"

echo ""
echo "🎉 Learning Rate Scheduler Ablation Study for CLAM completed!"
echo "============================================================"
echo "📊 Summary of experiments:"
echo "  - Fixed LR (baseline): ${RESULTS_DIR}/clam_lr_scheduler_fixed"
echo "  - Cosine Annealing: ${RESULTS_DIR}/clam_lr_scheduler_cosine"
echo "  - Cosine + Warm Restart: ${RESULTS_DIR}/clam_lr_scheduler_cosine_restart"
echo "  - Step LR: ${RESULTS_DIR}/clam_lr_scheduler_step"
echo "  - ReduceLROnPlateau: ${RESULTS_DIR}/clam_lr_scheduler_plateau"
echo ""
echo "🔍 Key metrics to compare:"
echo "  - Final test AUC"
echo "  - Training stability (loss curves)"
echo "  - Convergence speed"
echo "  - Overfitting behavior"
echo "  - Learning rate curves"

