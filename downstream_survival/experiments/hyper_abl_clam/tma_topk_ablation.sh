#!/bin/bash

# =============================================================================
# 环境设置
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=2
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 🔬 Top-K 参数消融研究
# 基于标准任务进行 top-k 选择参数的系统性调整

echo "🚀 开始 Top-K 参数消融研究..."
echo "⏰ 开始时间: $(date)"
echo "=" * 50

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
TARGET_CHANNELS="tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 实验 & 训练参数
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
CHANNELS_USED_IN_MODEL="tma_CD3 tma_CD8 tma_CD56 tma_CD68 tma_CD163 tma_HE tma_MHC1 tma_PDL1"

# 基础命令模板
BASE_COMMAND="python main.py \
    --data_root_dir \"$DATA_ROOT_DIR\" \
    --results_dir \"$RESULTS_DIR\" \
    --csv_path \"$CSV_PATH\" \
    --target_channel $TARGET_CHANNELS \
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
    --channels_used_in_model $CHANNELS_USED_IN_MODEL"

# Top-K 值数组 (10个不同的值，从小到大)
TOP_K_VALUES=(1 3 8 25 50)

# 创建结果目录
RESULTS_DIR="./results/top_k_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# 循环执行每个 k 值的实验
for i in "${!TOP_K_VALUES[@]}"; do
    k_value=${TOP_K_VALUES[$i]}
    exp_name="top_k_${k_value}_exp_$((i+1))"
    
    echo ""
    echo "🧪 实验 $((i+1))/${#TOP_K_VALUES[@]}: 测试 k = $k_value"
    echo "📝 实验名称: $exp_name"
    echo "🕐 开始时间: $(date)"
    
    # 构建完整命令
    FULL_COMMAND="$BASE_COMMAND --inst_number $k_value --exp_code ${exp_name} --results_dir ${RESULTS_DIR}"
    
    echo "💻 执行命令: $FULL_COMMAND"
    
    # 执行训练
    eval $FULL_COMMAND
    
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $((i+1)) 完成 (k=$k_value)"
    else
        echo "❌ 实验 $((i+1)) 失败 (k=$k_value)"
    fi
    
    echo "🕑 结束时间: $(date)"
    echo "-" * 30
done

echo ""
echo "🎉 Top-K 参数消融研究完成!"
echo "📁 结果保存在: $RESULTS_DIR"
echo "⏰ 总结束时间: $(date)"

# 生成结果摘要
echo ""
echo "📊 实验参数摘要:"
echo "参数类型: Top-K Selection"
echo "测试值: ${TOP_K_VALUES[*]}"
echo "实验总数: ${#TOP_K_VALUES[@]}"
echo "结果目录: $RESULTS_DIR"
echo ""
echo "📈 Top-K 参数说明:"
echo "- 较小的 k 值: 更专注于最重要的 patches，可能过拟合"
echo "- 较大的 k 值: 包含更多信息，但可能引入噪声"
echo "- 标准值 k=10: 平衡性能和计算效率的经验值"
