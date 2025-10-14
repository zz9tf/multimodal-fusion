#!/bin/bash
# 主脚本：运行所有 Ablation Studies

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

echo "========================================"
echo "🚀 开始运行所有 Ablation Studies"
echo "========================================"
echo ""

# 创建结果目录
echo "📁 创建结果目录..."
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_mismatch_ratio
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_seed
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_lambda1
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_lambda2
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_tau1
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_tau2
mkdir -p /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_num_layers
echo "✅ 目录创建完成"
echo ""

# # 1. Mismatch Ratio Ablation
# echo "========================================"
# echo "1️⃣  Running Mismatch Ratio Ablation"
# echo "========================================"
# task run "mismatch_ratio" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_mismatch_ratio.sh"
# echo ""

# 2. Seed Ablation
echo "========================================"
echo "2️⃣  Running Seed Ablation"
echo "========================================"
task run "seed" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_seed.sh"
echo ""

# # 3. Lambda1 Ablation
# echo "========================================"
# echo "3️⃣  Running Lambda1 Ablation"
# echo "========================================"
# task run "lambda1" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_lambda1.sh"
# echo ""

# 4. Lambda2 Ablation
echo "========================================"
echo "4️⃣  Running Lambda2 Ablation"
echo "========================================"
task run "lambda2" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_lambda2.sh"
echo ""

# # 5. Tau1 Ablation
# echo "========================================"
# echo "5️⃣  Running Tau1 Ablation"
# echo "========================================"
# task run "tau1" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_tau1.sh"
# echo ""

# 6. Tau2 Ablation
echo "========================================"
echo "6️⃣  Running Tau2 Ablation"
echo "========================================"
task run "tau2" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_tau2.sh"
echo ""

# # 7. Num Layers Ablation
# echo "========================================"
# echo "7️⃣  Running Num Layers Ablation"
# echo "========================================"
# task run "num_layers" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_num_layers.sh"
# echo ""

# 8. Loss2 Chunk Size Ablation
echo "========================================"
echo "8️⃣  Running Loss2 Chunk Size Ablation"
echo "========================================"
task run "loss2_chunk_size" "/home/zheng/zheng/multimodal-fusion/alignment/ablation_study/ablation_loss2_chunk_size.sh"
echo ""


echo "========================================"
echo "🎉 所有 Ablation Studies 完成！"
echo "========================================"
echo ""
echo "📊 结果保存在："
echo "   - /home/zheng/zheng/multimodal-fusion/alignment/results/ablation_*/"
echo ""

