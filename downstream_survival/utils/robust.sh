#!/bin/bash

# 为所有结果目录运行 robust_on_missing_modality.py
python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
  --results_dir /home/zheng/zheng/mini2/results/20251106-194148_svd_random_clam_detach_s5678 \
  --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv

# source ~/zheng/miniconda3/etc/profile.d/conda.sh
# conda activate multimodal-fusion
# cd /home/zheng/zheng/multimodal-fusion/downstream_survival
# python utils/test_train_save_load_consistency.py \
#     --results_dir "/home/zheng/zheng/mini2/results/20251029-121722_svd_dynamic_clam_detach_s5678" \
#     --fold_idx 0 \
#     --max_batches 2  # 可选：限制测试batch数量

# cd /home/zheng/zheng/multimodal-fusion/downstream_survival
# python3 utils/compare_loader_samples.py \
#     --data_root_dir "/home/zheng/zheng/public/1" \
#     --csv_path "/home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv" \
#     --channels "wsi tma clinical" \
#     --fold_idx 0 \
#     --seed 42 \
#     --split_type val