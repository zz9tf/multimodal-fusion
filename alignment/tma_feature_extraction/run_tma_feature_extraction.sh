#!/bin/bash

# TMA特征提取使用示例

echo "🚀 TMA特征提取示例"
echo "=================="

# 设置路径
INPUT_DIR="/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores_ori"
OUTPUT_DIR="/home/zheng/zheng/mini2/hancock_data/TMA/TMA_Core_encodings"

echo "📁 输入目录: $INPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    echo "请检查路径是否正确"
    exit 1
fi

echo "✅ 输入目录存在"
echo "📋 可用的标记目录:"
ls -1 "$INPUT_DIR" | grep "tma_tumorcenter_" | sed 's/tma_tumorcenter_//'
echo ""

# 运行特征提取
echo "🔧 开始特征提取..."

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate multimodal-fusion

# 设置参数
BATCH_SIZE=32
PATCH_SIZE=256
STRIDE=256
PHYSICAL_GPU=1
MARKERS=("CD3" "CD8" "CD56" "CD68" "CD163" "HE" "MHC1" "PDL1")

echo "🖥 设备: cuda (物理GPU=$PHYSICAL_GPU)"
echo "🔢 Batch size: $BATCH_SIZE"
echo "📏 Patch尺寸: $PATCH_SIZE"
echo "👣 步长: $STRIDE"
echo "📊 输出维度: 1024 (UNI固定)"
echo "🏷 标记列表: ${MARKERS[*]}"
echo ""

# 直接运行Python脚本
CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU PYTHONPATH=/home/zheng/zheng/multimodal-fusion python \
  /home/zheng/zheng/multimodal-fusion/alignment/tma_feature_extraction/extract_tma_features_uni.py \
  "$INPUT_DIR" \
  "$OUTPUT_DIR" \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --patch_size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --markers "${MARKERS[@]}" \
  --gpu_id 0

echo ""
echo "🎉 示例完成！"
