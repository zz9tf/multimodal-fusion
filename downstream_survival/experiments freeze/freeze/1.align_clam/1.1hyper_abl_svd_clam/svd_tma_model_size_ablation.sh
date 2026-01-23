#!/bin/bash

# =============================================================================
# Environment Setup
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 🔬 嵌入维度 (Embedding Dimension) 消融研究
# 基于标准任务进行嵌入维度参数的系统性调整

echo "🚀 开始 Model Size 消融研究..."
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
DROPOUT=0.25
N_CLASSES=2
BASE_LOSS_FN="ce"

# CLAM特定参数
GATE="--gate"
BASE_WEIGHT=0.9
INST_LOSS_FN="ce"
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
    --dropout $DROPOUT \
    --n_classes $N_CLASSES \
    --base_loss_fn $BASE_LOSS_FN \
    --gate $GATE \
    --base_weight $BASE_WEIGHT \
    --inst_loss_fn $INST_LOSS_FN \
    --subtyping $SUBTYPING \
    --inst_number $INST_NUMBER \
    --channels_used_in_model $CHANNELS_USED_IN_MODEL"

# Model Size 值数组 (10个不同的维度值)
# 包含常见的2的幂次方维度和一些中间值
MODEL_SIZES=("128*64" "64*32" "32*16" "16*8" "8*4" "4*2" "2*1")

# 创建结果目录
RESULTS_DIR="./results/model_size_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# 创建结果汇总文件
SUMMARY_FILE="$RESULTS_DIR/model_size_ablation_summary.csv"
echo "model_size,experiment_name,status,log_file,memory_usage" > $SUMMARY_FILE

# 循环执行每个模型大小的实验
for i in "${!MODEL_SIZES[@]}"; do
    model_size=${MODEL_SIZES[$i]}
    exp_name="model_size_${model_size}_exp_$((i+1))"
    
    echo ""
    echo "🧪 实验 $((i+1))/${#MODEL_SIZES[@]}: 测试 model_size = $model_size"
    echo "📝 实验名称: $exp_name"
    echo "🕐 开始时间: $(date)"
    
    # 构建完整命令
    FULL_COMMAND="$BASE_COMMAND --model_size $model_size --exp_code ${exp_name} --results_dir ${RESULTS_DIR}"
    
    echo "💻 执行命令: $FULL_COMMAND"
    
    eval $FULL_COMMAND
    # 记录实验状态
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $((i+1)) 完成 (model_size=$model_size)"
        echo "$model_size,$exp_name,success,$LOG_FILE" >> $SUMMARY_FILE
        status="✅ 成功"
    else
        echo "❌ 实验 $((i+1)) 失败 (model_size=$model_size)"
        echo "$model_size,$exp_name,failed,$LOG_FILE" >> $SUMMARY_FILE
        status="❌ 失败"
    fi
    
    echo "🕑 结束时间: $(date)"
    echo "-" * 30
done

echo ""
echo "🎉 Model Size 消融研究完成!"
echo "📁 结果保存在: $RESULTS_DIR"
echo "⏰ 总结束时间: $(date)"

# 生成详细的结果摘要
echo ""
echo "📊 实验参数摘要:"
echo "参数类型: Model Size"
echo "模型大小测试值: ${MODEL_SIZES[*]}"
echo "实验总数: ${#MODEL_SIZES[@]}"
echo "结果目录: $RESULTS_DIR"
echo "汇总文件: $SUMMARY_FILE"
echo ""
echo "🔬 Model Size 研究意义:"
echo "- 评估不同模型大小对模型性能的影响"
echo "- 分析模型大小与计算资源消耗的权衡"
echo "- 找到最优的模型大小"
echo "- 验证模型在不同模型大小下的泛化能力"
echo ""
echo "📈 建议分析步骤:"
echo "1. 比较不同模型大小下的准确率和训练时间"
echo "2. 分析内存消耗与模型大小的关系"
echo "3. 识别性能收益递减的模型大小阈值"
echo "4. 评估模型大小对不同数据集大小的敏感性"
echo ""
echo "⚠️  注意事项:"
echo "- 更大模型需要更多GPU内存"
echo "- 建议监控训练过程中的内存使用情况"
echo "- 如遇到OOM错误，可考虑减小batch size"
