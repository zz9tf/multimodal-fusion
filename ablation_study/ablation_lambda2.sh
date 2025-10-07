#!/bin/bash
# Ablation Study: lambda2
# 测试匹配预测损失权重 lambda2 对模型性能的影响

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

# 固定其他参数（统一配置）
MISMATCH_RATIO=1.0
SEED=42
LAMBDA1=1.0
TAU1=0.1
TAU2=0.05
NUM_LAYERS=2
MAX_STEPS=2000
BATCH_SIZE=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
LOG_INTERVAL=200
VAL_INTERVAL=400

# 测试10个不同的 lambda2 值
LAMBDA2_VALUES=(0.0 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5)

for LAMBDA2 in "${LAMBDA2_VALUES[@]}"
do
    echo "============================================================"
    echo "Running experiment with lambda2=${LAMBDA2}"
    echo "============================================================"
    
    CUDA_VISIBLE_DEVICES=1 python /home/zheng/zheng/multimodal-fusion/run.py \
        --align_mode intersection \
        --pattern "tma_uni_tile_1024_{marker}.npz" \
        --mismatch_ratio ${MISMATCH_RATIO} \
        --seed ${SEED} \
        --lambda1 ${LAMBDA1} \
        --lambda2 ${LAMBDA2} \
        --tau1 ${TAU1} \
        --tau2 ${TAU2} \
        --num_layers ${NUM_LAYERS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --max_steps ${MAX_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --save_path /home/zheng/zheng/multimodal-fusion/results/ablation_lambda2/model_lambda2_${LAMBDA2}.pth \
        --num_workers 0 \
        --log_interval ${LOG_INTERVAL} \
        --val_interval ${VAL_INTERVAL}
    
    echo ""
    echo "Completed lambda2=${LAMBDA2}"
    echo ""
done

echo "✅ Ablation study for lambda2 completed!"

