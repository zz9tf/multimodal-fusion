#!/usr/bin/env python3
"""
SVD 对齐前后特征可视化工具

用法示例：
python -m downstream_survival.utils.alignment_visualization \
  --configs_json /path/to/results/configs_EXP.json \
  --checkpoint /path/to/model.ckpt \
  --save_dir /path/to/save/plots \
  --num_samples 200 \
  --method pca

说明：
- 该脚本会根据配置构建数据集与模型（优先使用 SVDGateRandomClamDetach），
  将模型设置为返回 SVD 对齐前后特征（return_svd_features=True），
  抽样若干样本运行前向推理，收集各模态在对齐前后的特征，
  再使用 PCA/TSNE 进行二维可视化，并保存图像。
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple, Any

import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 项目根目录（与 main.py 相同的方式）
import sys
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from datasets.multimodal_dataset import MultimodalDataset
from models.svd_gate_random_clam_detach import SVDGateRandomClamDetach


def _ensure_dir(path: str) -> None:
    """如果目录不存在则创建。"""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _load_configs(configs_json: str) -> Dict[str, Any]:
    """加载配置 JSON。"""
    with open(configs_json, 'r') as f:
        return json.load(f)


def _build_dataset(experiment_config: Dict[str, Any], device: torch.device) -> MultimodalDataset:
    """根据 experiment_config 构建 MultimodalDataset。"""
    dataset = MultimodalDataset(
        csv_path=experiment_config['csv_path'],
        data_root_dir=experiment_config['data_root_dir'],
        channels=experiment_config['target_channels'],
        align_channels=experiment_config.get('aligned_channels', {}),
        alignment_model_path=experiment_config.get('alignment_model_path', None),
        device=device,
        print_info=False,
    )
    return dataset


def _build_model(model_config: Dict[str, Any], device: torch.device, checkpoint: str | None) -> torch.nn.Module:
    """构建并加载 SVDGateRandomClamDetach 模型，若提供 checkpoint 则加载权重。"""
    # 确保启用 SVD 并返回对齐前后特征
    model_specific_config = dict(model_config)
    model_specific_config['enable_svd'] = True
    model_specific_config['return_svd_features'] = True

    model = SVDGateRandomClamDetach(model_specific_config).to(device)
    model.eval()

    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        # 兼容常见保存方式
        sd = state.get('state_dict', state.get('model_state_dict', state))
        model.load_state_dict(sd, strict=False)

    return model


def _collect_features(
    model: torch.nn.Module,
    dataset: MultimodalDataset,
    device: torch.device,
    num_samples: int,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    迭代数据集前 num_samples 个样本，收集对齐前后特征。

    返回：
    (features_before, features_after)
    两者字典键均为模态名，值为该模态下若干样本的特征列表（np.ndarray）。
    """
    features_before: Dict[str, List[np.ndarray]] = {}
    features_after: Dict[str, List[np.ndarray]] = {}

    total = min(num_samples, len(dataset))
    for i in range(total):
        sample = dataset[i]
        # 支持 (data, label) 或 dict 结构
        if isinstance(sample, tuple) and len(sample) >= 2:
            input_data, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            input_data = sample.get('data', sample)
            label = sample.get('label', None)
        else:
            # 不支持的格式则跳过
            continue

        # 将张量移动到设备
        if isinstance(input_data, dict):
            input_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_data.items()}
        elif torch.is_tensor(input_data):
            input_data = input_data.to(device)

        if torch.is_tensor(label):
            label = label.to(device)

        with torch.no_grad():
            out = model(input_data, label)

        # 期望返回包含 'features' 与 'aligned_features'
        if not isinstance(out, dict):
            continue
        if 'features' not in out or 'aligned_features' not in out:
            # 如果模型未返回该结构，跳过
            continue

        f_before: Dict[str, torch.Tensor] = out['features']
        f_after: Dict[str, torch.Tensor] = out['aligned_features']

        for key, tensor in f_before.items():
            arr = tensor.detach().float().cpu().numpy()
            features_before.setdefault(key, []).append(arr)

        for key, tensor in f_after.items():
            arr = tensor.detach().float().cpu().numpy()
            features_after.setdefault(key, []).append(arr)

    return features_before, features_after


def _stack_features(modality_to_arrays: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """将每个模态的 [样本, (N_i, D)] 列表按样本拼接为二维数组 [M, D]。"""
    stacked: Dict[str, np.ndarray] = {}
    for key, parts in modality_to_arrays.items():
        # 每个样本可能是 [N_i, D] 或 [D]；统一为 [*, D]
        normalized: List[np.ndarray] = []
        for p in parts:
            if p.ndim == 1:
                normalized.append(p[None, :])
            elif p.ndim == 2:
                normalized.append(p)
            else:
                # 更高维度，尝试展平到 [*, D]
                normalized.append(p.reshape(-1, p.shape[-1]))
        if not normalized:
            continue
        stacked[key] = np.concatenate(normalized, axis=0)
    return stacked


def _reduce_and_plot(
    features_before: Dict[str, np.ndarray],
    features_after: Dict[str, np.ndarray],
    save_dir: str,
    method: str = 'pca',
    max_points_per_modality: int = 5000,
) -> None:
    """
    对每个模态分别降维并绘图（对齐前 vs 对齐后并排）。
    method: 'pca' 或 'tsne'
    """
    _ensure_dir(save_dir)

    for modality in sorted(set(list(features_before.keys()) + list(features_after.keys()))):
        if modality not in features_before or modality not in features_after:
            continue

        Xb = features_before[modality]
        Xa = features_after[modality]

        # 子采样，避免点过多
        def subsample(X: np.ndarray) -> np.ndarray:
            if len(X) <= max_points_per_modality:
                return X
            idx = np.random.choice(len(X), size=max_points_per_modality, replace=False)
            return X[idx]

        Xb = subsample(Xb)
        Xa = subsample(Xa)

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            Yb = reducer.fit_transform(Xb)
            Ya = reducer.fit_transform(Xa)
        elif method == 'tsne':
            reducer_b = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
            reducer_a = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=42)
            Yb = reducer_b.fit_transform(Xb)
            Ya = reducer_a.fit_transform(Xa)
        else:
            raise ValueError(f"未知降维方法: {method}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(Yb[:, 0], Yb[:, 1], s=6, alpha=0.7, c='#1f77b4')
        axes[0].set_title(f"{modality} - Before SVD")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].scatter(Ya[:, 0], Ya[:, 1], s=6, alpha=0.7, c='#d62728')
        axes[1].set_title(f"{modality} - After SVD")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.suptitle(f"SVD Alignment Visualization - {modality} ({method.upper()})")
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"svd_{method}_{modality.replace('/', '_').replace('=', '_')}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='SVD 对齐前后特征可视化')
    parser.add_argument('--configs_json', type=str, required=True, help='训练时保存的 configs_*.json 路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径（可选）')
    parser.add_argument('--save_dir', type=str, default=None, help='图片保存目录，默认使用配置中的 results_dir/vis')
    parser.add_argument('--num_samples', type=int, default=200, help='抽样可视化的样本数上限')
    parser.add_argument('--method', type=str, choices=['pca', 'tsne'], default='pca', help='降维方法')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='设备优先级（默认自动检测）')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    configs = _load_configs(args.configs_json)
    experiment_config = configs['experiment_config']
    model_config = configs['model_config']

    save_dir = args.save_dir or os.path.join(experiment_config['results_dir'], 'vis')
    _ensure_dir(save_dir)

    dataset = _build_dataset(experiment_config, device)
    model = _build_model(model_config, device, args.checkpoint)

    features_before_raw, features_after_raw = _collect_features(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
    )

    features_before = _stack_features(features_before_raw)
    features_after = _stack_features(features_after_raw)

    # 保存一份原始统计信息
    stats_path = os.path.join(save_dir, 'svd_features_stats.json')
    stats = {
        'modalities_before': {k: int(v.shape[0]) for k, v in features_before.items()},
        'modalities_after': {k: int(v.shape[0]) for k, v in features_after.items()},
        'method': args.method,
        'num_samples': args.num_samples,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    _reduce_and_plot(features_before, features_after, save_dir=save_dir, method=args.method)

    print(f"✅ 可视化完成，结果保存在: {save_dir}")


if __name__ == '__main__':
    main()














