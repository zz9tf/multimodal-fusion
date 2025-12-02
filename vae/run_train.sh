#!/bin/bash
# VAE训练示例脚本

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/vae

# 设置参数
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
DATA_ROOT_DIR="/home/zheng/zheng/public/2"  # 请修改为实际的数据根目录
CUDA_VISIBLE_DEVICES=0

# 训练参数
BATCH_SIZE=2048
EPOCHS=100
LEARNING_RATE=1e-4
LATENT_DIM=128
HIDDEN_DIMS="512 256"

# 运行训练
# 注意：如果不想过滤标签，可以设置 --label_filter "" 或移除该参数
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python train.py \
    --csv_path ${CSV_PATH} \
    --data_root_dir ${DATA_ROOT_DIR} \
    --label_filter living \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dims ${HIDDEN_DIMS} \
    --val_split 0.2 \
    --device cuda \
    --save_dir ./checkpoints \
    --log_dir ./logs

