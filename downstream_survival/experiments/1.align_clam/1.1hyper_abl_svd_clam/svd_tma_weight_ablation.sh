#!/bin/bash

# =============================================================================
# 环境设置
# =============================================================================
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

CUDA_DEVICE=2
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# 🔬 Bag Loss 权重占比消融研究  
# 基于标准任务进行 bag loss 与 instance loss 权重平衡的系统性调整

echo "🚀 开始 Bag Loss 权重占比消融研究..."
echo "⏰ 开始时间: $(date)"
echo "=" * 50

# 数据相关参数
DATA_ROOT_DIR="/home/zheng/zheng/public/hancock_data/WSI_UNI_encodings/WSI_PrimaryTumor"
RESULTS_DIR="/home/zheng/zheng/multimodal-fusion/downstream_survival/results"
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
ALIGNMENT_MODEL_PATH="/home/zheng/zheng/multimodal-fusion/alignment/results/test_svd/test_multimodal_alignment_model.pth"
ALIGNED_CHANNELS="tma_CD3=CD3 tma_CD8=CD8 tma_CD56=CD56 tma_CD68=CD68 tma_CD163=CD163 tma_HE=HE tma_MHC1=MHC1 tma_PDL1=PDL1"
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
INST_LOSS_FN="ce"
MODEL_SIZE="64*32"
SUBTYPING="--subtyping"
INST_NUMBER=8
CHANNELS_USED_IN_MODEL="aligned_tma_CD3 aligned_tma_CD8 aligned_tma_CD56 aligned_tma_CD68 aligned_tma_CD163 aligned_tma_HE aligned_tma_MHC1 aligned_tma_PDL1"

# 基础命令模板 (base_weight 将在循环中设置)
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
    --inst_loss_fn $INST_LOSS_FN \
    --model_size $MODEL_SIZE \
    --subtyping $SUBTYPING \
    --inst_number $INST_NUMBER \
    --channels_used_in_model $CHANNELS_USED_IN_MODEL"

# Bag Weight 值数组 (10个不同的值，从1.0开始往下调整)
# bag_weight=1.0 表示100%的bag loss, 0%的instance loss
# bag_weight=0.7 是标准值 (70% bag loss, 30% instance loss)  
# bag_weight=0.0 表示0%的bag loss, 100%的instance loss
BAG_WEIGHT_VALUES=(1.0 0.8 0.5 0.2 0.1)

# 创建结果目录
RESULTS_DIR="./results/bag_weight_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# 循环执行每个 bag weight 值的实验
for i in "${!BAG_WEIGHT_VALUES[@]}"; do
    bag_weight=${BAG_WEIGHT_VALUES[$i]}
    exp_name="bag_weight_${bag_weight}_exp_$((i+1))"
    
    echo ""
    echo "🧪 实验 $((i+1))/${#BAG_WEIGHT_VALUES[@]}: 测试 bag_weight = $bag_weight"
    echo "📝 实验名称: $exp_name"
    echo "🕐 开始时间: $(date)"
    
    # 计算损失占比说明
    bag_percentage=$(echo "scale=0; $bag_weight * 100" | bc -l 2>/dev/null || echo "$(python3 -c "print(int($bag_weight * 100))")")
    inst_percentage=$(echo "scale=0; (1 - $bag_weight) * 100" | bc -l 2>/dev/null || echo "$(python3 -c "print(int((1 - $bag_weight) * 100))")")
    echo "📊 损失占比: Bag Loss ${bag_percentage}% + Instance Loss ${inst_percentage}%"
    
    # 构建完整命令 (添加 base_weight 参数)
    FULL_COMMAND="$BASE_COMMAND --base_weight $bag_weight --exp_code ${exp_name} --results_dir ${RESULTS_DIR}"
    
    echo "💻 执行命令: $FULL_COMMAND"
    
    # 执行训练
    eval $FULL_COMMAND
    
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $((i+1)) 完成 (bag_weight=$bag_weight)"
    else
        echo "❌ 实验 $((i+1)) 失败 (bag_weight=$bag_weight)"
    fi
    
    echo "🕑 结束时间: $(date)"
    echo "-" * 30
done

# 额外测试：极端值 - 纯 Instance Loss
echo ""
echo "🔄 额外测试: 极端值测试"

# 测试 bag_weight = 0.0 (纯 Instance Loss)
exp_name="pure_instance_loss_bag_weight_0.0"
echo "🧪 测试纯 Instance Loss (bag_weight = 0.0)"
echo "📊 损失占比: Bag Loss 0% + Instance Loss 100%"
FULL_COMMAND="$BASE_COMMAND --base_weight 0.0 --exp_code ${exp_name}"
LOG_FILE="$RESULTS_DIR/${exp_name}.log"
eval $FULL_COMMAND > $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 纯 Instance Loss 实验完成"
else
    echo "❌ 纯 Instance Loss 实验失败"
    echo "📋 查看日志: $LOG_FILE"
fi

echo ""
echo "🎉 Bag Loss 权重占比消融研究完成!"
echo "📁 结果保存在: $RESULTS_DIR"
echo "⏰ 总结束时间: $(date)"

# 生成结果摘要
echo ""
echo "📊 实验参数摘要:"
echo "参数类型: Bag Loss Weight (bag_weight)"
echo "测试值: ${BAG_WEIGHT_VALUES[*]} + 0.0"
echo "实验总数: $((${#BAG_WEIGHT_VALUES[@]} + 1))"
echo "结果目录: $RESULTS_DIR"
echo ""
echo "🔬 Bag Weight 参数说明:"
echo "- bag_weight = 1.0: 100% Bag Loss + 0% Instance Loss (纯袋级学习)"
echo "- bag_weight = 0.7: 70% Bag Loss + 30% Instance Loss (标准配置)"
echo "- bag_weight = 0.5: 50% Bag Loss + 50% Instance Loss (平衡配置)"
echo "- bag_weight = 0.0: 0% Bag Loss + 100% Instance Loss (纯实例学习)"
echo ""
echo "📈 建议分析要点:"
echo "1. 比较不同权重比例下的分类准确率"
echo "2. 分析 bag-level 和 instance-level 性能的权衡"
echo "3. 识别最优的损失函数平衡点"
echo "4. 评估模型对权重比例变化的敏感性"
