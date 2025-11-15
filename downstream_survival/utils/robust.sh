#!/bin/bash

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate multimodal-fusion
cd /home/zheng/zheng/multimodal-fusion/downstream_survival

# # 为所有结果目录运行 robust_on_missing_modality.py
#   python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.2

#   python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.3

#   python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.4

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.5

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.6

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.7

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.8

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 0.9

# python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/robust_on_missing_modality.py \
#   --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678 \
#   --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \
#   --drop_prob 1.0

python /home/zheng/zheng/multimodal-fusion/downstream_survival/utils/plot_robust_results.py \
  --results_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678