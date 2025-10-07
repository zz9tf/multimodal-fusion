#!/bin/bash
# Ablation Study: seed
# 测试不同随机种子对模型性能的影响（评估模型稳定性）

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

# 固定其他参数（统一配置）
MISMATCH_RATIO=1.0
LAMBDA1=1.0
LAMBDA2=0.1
TAU1=0.1
TAU2=0.05
NUM_LAYERS=2
MAX_STEPS=2000
BATCH_SIZE=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
LOG_INTERVAL=200
VAL_INTERVAL=400

# 测试10个不同的 seed 值
SEEDS=(42 123 456 789 1024 2048 3141 5926 8888 9999)

for SEED in "${SEEDS[@]}"
do
    echo "============================================================"
    echo "Running experiment with seed=${SEED}"
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
        --save_path /home/zheng/zheng/multimodal-fusion/results/ablation_seed/model_seed_${SEED}.pth \
        --num_workers 0 \
        --log_interval ${LOG_INTERVAL} \
        --val_interval ${VAL_INTERVAL}
    
    echo ""
    echo "Completed seed=${SEED}"
    echo ""
done

echo "✅ Ablation study for seed completed!"

