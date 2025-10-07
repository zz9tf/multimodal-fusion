#!/bin/bash
# 串行运行所有消融实验
# 按顺序执行每个消融实验，避免资源竞争

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

echo "========================================"
echo "🚀 开始串行运行所有 Ablation Studies"
echo "========================================"
echo ""

# 创建结果目录
echo "📁 创建结果目录..."
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_mismatch_ratio
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_seed
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_lambda1
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_lambda2
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_tau1
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_tau2
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_num_layers
mkdir -p /home/zheng/zheng/multimodal-fusion/results/ablation_loss2_chunk_size
echo "✅ 目录创建完成"
echo ""

# 定义要运行的脚本列表
scripts=(
    # "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_mismatch_ratio.sh"
    "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_seed.sh"
    # "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_lambda1.sh"
    "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_lambda2.sh"
    # "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_tau1.sh"
    "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_tau2.sh"
    # "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_num_layers.sh"
    "/home/zheng/zheng/multimodal-fusion/ablation_study/ablation_loss2_chunk_size.sh"
)

# 串行执行每个脚本
total_scripts=${#scripts[@]}
current_script=1

for script in "${scripts[@]}"
do
    script_name=$(basename "$script" .sh)
    echo "========================================"
    echo "📋 [$current_script/$total_scripts] 运行: $script_name"
    echo "========================================"
    echo "⏰ 开始时间: $(date)"
    echo ""
    
    # 检查脚本是否存在
    if [ ! -f "$script" ]; then
        echo "❌ 脚本不存在: $script"
        echo "⏭️  跳过此脚本"
        echo ""
        ((current_script++))
        continue
    fi
    
    # 执行脚本
    echo "🚀 执行脚本: $script"
    bash "$script"
    exit_code=$?
    
    echo ""
    echo "⏰ 结束时间: $(date)"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $script_name 执行成功"
    else
        echo "❌ $script_name 执行失败 (退出码: $exit_code)"
    fi
    
    echo ""
    echo "📊 当前系统状态:"
    echo "   - 内存使用: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
    echo "   - 磁盘使用: $(df -h / | tail -1 | awk '{print $5}')"
    echo ""
    
    ((current_script++))
done

echo "========================================"
echo "🎉 所有消融实验完成！"
echo "========================================"
echo "📊 结果保存在："
echo "   - /home/zheng/zheng/multimodal-fusion/results/ablation_*/"
echo ""
echo "⏰ 总完成时间: $(date)"
