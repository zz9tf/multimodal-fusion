#!/bin/bash

# =============================================================================
# Environment Setup
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 🔬 Dropout 参数消融研究
# 基于标准任务进行 dropout 参数的系统性调整

echo "🚀 开始 Dropout 参数消融研究..."
echo "⏰ 开始时间: $(date)"
echo "=" * 50

# Data-related parameters
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
ALIGNMENT_MODEL_PATH="/home/zheng/zheng/multimodal-fusion/alignment/results/test_svd/test_multimodal_alignment_model.pth"
ALIGNED_CHANNELS="tma_CD3=CD3 tma_CD8=CD8 tma_CD56=CD56 tma_CD68=CD68 tma_CD163=CD163 tma_HE=HE tma_MHC1=MHC1 tma_PDL1=PDL1"
TARGET_CHANNELS="tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# Experiment & Training parameters
SEED=5678
K_FOLDS=10
MAX_EPOCHS=200
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adam"
EARLY_STOPPING="--early_stopping"
BATCH_SIZE=1

# 模型参数
MODEL_TYPE="clam"
INPUT_DIM=1024
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

# 基础命令模板
BASE_COMMAND="python main.py \
    --data_root_dir \"$DATA_ROOT_DIR\" \
    --results_dir \"$RESULTS_DIR\" \
    --csv_path \"$CSV_PATH\" \
    --target_channel $TARGET_CHANNELS \
    --alignment_model_path \"$ALIGNMENT_MODEL_PATH\" \
    --aligned_channels $ALIGNED_CHANNELS \
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
    --n_classes $N_CLASSES \
    --base_loss_fn $BASE_LOSS_FN \
    --gate $GATE \
    --base_weight $BASE_WEIGHT \
    --inst_loss_fn $INST_LOSS_FN \
    --model_size $MODEL_SIZE \
    --subtyping $SUBTYPING \
    --inst_number $INST_NUMBER \
    --channels_used_in_model $CHANNELS_USED_IN_MODEL"

# Dropout 值数组 (10个不同的值)
DROPOUT_VALUES=(0.05 0.1 0.2 0.4 0.8)

# 创建结果目录
RESULTS_DIR="./results/dropout_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# 循环执行每个 dropout 值的实验
for i in "${!DROPOUT_VALUES[@]}"; do
    dropout=${DROPOUT_VALUES[$i]}
    exp_name="dropout_${dropout}_exp_$((i+1))"
    
    echo ""
    echo "🧪 实验 $((i+1))/${#DROPOUT_VALUES[@]}: 测试 dropout = $dropout"
    echo "📝 实验名称: $exp_name"
    echo "🕐 开始时间: $(date)"
    
    # 构建完整命令
    FULL_COMMAND="$BASE_COMMAND --dropout $dropout --exp_code ${exp_name} --results_dir ${RESULTS_DIR}"
    
    echo "💻 执行命令: $FULL_COMMAND"
    
    # 执行训练
    eval $FULL_COMMAND
    
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $((i+1)) 完成 (dropout=$dropout)"
    else
        echo "❌ 实验 $((i+1)) 失败 (dropout=$dropout)"
    fi
    
    echo "🕑 结束时间: $(date)"
    echo "-" * 30
done

echo ""
echo "🎉 Dropout 参数消融研究完成!"
echo "📁 结果保存在: $RESULTS_DIR"
echo "⏰ 总结束时间: $(date)"

# 生成结果摘要
echo ""
echo "📊 实验参数摘要:"
echo "参数类型: Dropout"
echo "测试值: ${DROPOUT_VALUES[*]}"
echo "实验总数: ${#DROPOUT_VALUES[@]}"
echo "结果目录: $RESULTS_DIR"
