#!/bin/bash
# Example script to train the VAE model

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/vae

# Paths
CSV_PATH="/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv"
DATA_ROOT_DIR="/home/zheng/zheng/public/2"  # TODO: change to your real data root
CUDA_VISIBLE_DEVICES=0

# Training hyper-parameters
BATCH_SIZE=1024
EPOCHS=200
LEARNING_RATE=1e-4
LATENT_DIM=128
HIDDEN_DIMS="512 256"

# Optimization options (for faster training)
VAL_FREQ=1  # validate every N epochs

# Run training
# Note: if you do not want to filter by label, set --label_filter "" or remove the argument.
# Optimization notes:
# - torch.compile (PyTorch 2.0+) can speed up training by ~10–30%.
# - Data loading is optimized with more workers and persistent_workers.
# - Resample strategy is dynamically adjusted by LR scheduler triggers (10% -> 5% -> every epoch).
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
    --val_freq ${VAL_FREQ} \
    --device cuda \
    --save_dir ./checkpoints \
    --log_dir ./logs