#!/bin/bash
# Ablation Study: tau1
# 测试温度参数 tau1 对模型性能的影响

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

# 固定其他参数（统一配置）
MISMATCH_RATIO=1.0
SEED=42
LAMBDA1=1.0
LAMBDA2=0.1
TAU2=0.05
NUM_LAYERS=2
MAX_STEPS=400
BATCH_SIZE=512
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
LOG_INTERVAL=20
VAL_INTERVAL=50

# 测试5个关键的 tau1 值 (保留极值)
TAU1_VALUES=(0.05 0.1 0.2 1 10)

for TAU1 in "${TAU1_VALUES[@]}"
do
    echo "============================================================"
    echo "Running experiment with tau1=${TAU1}"
    echo "============================================================"
    
    CUDA_VISIBLE_DEVICES=3 python /home/zheng/zheng/multimodal-fusion/run.py \
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
        --save_path /home/zheng/zheng/multimodal-fusion/results/ablation_tau1/model_tau1_${TAU1}.pth \
        --num_workers 0 \
        --log_interval ${LOG_INTERVAL} \
        --val_interval ${VAL_INTERVAL} \
        --loss2_chunk_size 8
    
    echo ""
    echo "Completed tau1=${TAU1}"
    echo ""
done

echo "✅ Ablation study for tau1 completed!"

