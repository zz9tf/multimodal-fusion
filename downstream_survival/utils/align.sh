NUM=0

# python utils/alignment_visualization.py \
#   --results_dir results/20251106-194148_svd_random_clam_detach_s5678 \
#   --fold_idx ${NUM} \
#   --save_dir results/20251106-194148_svd_random_clam_detach_s5678/svd_features

python utils/plot_alignment_heatmap.py \
  --features_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678/svd_features \
  --fold_idx ${NUM} \
  --output_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678/svd_features_fold${NUM} \
  --results_dir results/20251106-194148_svd_random_clam_detach_s5678

python utils/plot_modality_tsne.py \
  --features_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678/svd_features \
  --fold_idx ${NUM} \
  --output_dir /home/zheng/zheng/multimodal-fusion/downstream_survival/results/20251106-194148_svd_random_clam_detach_s5678/svd_features_tsne \
  --method tsne