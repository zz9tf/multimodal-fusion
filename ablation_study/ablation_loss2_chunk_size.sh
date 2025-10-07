#!/bin/bash
# Ablation Study: loss2_chunk_size
# 测试loss2分块大小对模型性能和训练效率的影响

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate multimodal-fusion

# 固定其他参数（统一配置）
MISMATCH_RATIO=1.0
SEED=42
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

# 测试10个不同的 loss2_chunk_size 值
# None表示不分块，其他值表示分块大小
LOSS2_CHUNK_SIZE_VALUES=(2 4 8 16 32 64 128 256 512 1024)

for CHUNK_SIZE in "${LOSS2_CHUNK_SIZE_VALUES[@]}"
do
    echo "============================================================"
    echo "Running experiment with loss2_chunk_size=${CHUNK_SIZE}"
    echo "============================================================"
    
    # 构建命令参数
    if [ "${CHUNK_SIZE}" = "None" ]; then
        CHUNK_SIZE_ARG=""
    else
        CHUNK_SIZE_ARG="--loss2_chunk_size ${CHUNK_SIZE}"
    fi
    
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
        --save_path /home/zheng/zheng/multimodal-fusion/results/ablation_loss2_chunk_size/model_loss2_chunk_size_${CHUNK_SIZE}.pth \
        --num_workers 0 \
        --log_interval ${LOG_INTERVAL} \
        --val_interval ${VAL_INTERVAL} \
        ${CHUNK_SIZE_ARG}
    
    echo ""
    echo "Completed loss2_chunk_size=${CHUNK_SIZE}"
    echo ""
done

echo "✅ Ablation study for loss2_chunk_size completed!"

