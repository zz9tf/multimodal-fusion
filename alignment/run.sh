#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment (UNI 依赖 torch>=2.0, timm>=0.9.8)
conda activate multimodal-fusion
conda env list

CUDA_VISIBLE_DEVICES=1 python /home/zheng/zheng/multimodal-fusion/alignment/run.py \
    --align_mode intersection \
    --pattern "tma_uni_tile_1024_{marker}.npz" \
    --mismatch_ratio 1.0 \
    --seed 42 \
    --lambda1 1.0 \
    --lambda2 0.1 \
    --tau1 0.01 \
    --tau2 0.05 \
    --num_layers 2 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --max_steps 2000 \
    --batch_size 64 \
    --loss_type volume \
    --save_path /home/zheng/zheng/multimodal-fusion/alignment/results/test_volume_multimodal_alignment_model.pth \
    --num_workers 0 \
    --log_interval 10 \
    --val_interval 100
    

