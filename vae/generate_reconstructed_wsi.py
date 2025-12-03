#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use a trained VAE to reconstruct all WSI embeddings stored in HDF5 files
and write the results into `wsi/reconstructed_features`.

Example:
    python generate_reconstructed_wsi.py \\
        --csv_path /home/zheng/zheng/multimodal-fusion/downstream_survival/dataset_csv/survival_dataset.csv \\
        --data_root_dir /data_root \\
        --checkpoint_path /home/zheng/zheng/multimodal-fusion/vae/checkpoints/vae_xxx/checkpoint_best.pth
"""

import argparse
import os
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from models import Encoder, Decoder, VAE


def _infer_input_dim(file_path: str) -> int:
    """
    Infer WSI feature dimension from a single HDF5 file.

    Args:
        file_path: absolute path to the HDF5 file.

    Returns:
        Input feature dimension (patch feature dim).
    """
    with h5py.File(file_path, "r") as f:
        feats = f["wsi"]["features"][:]

    if feats.ndim == 1:
        return int(feats.shape[0])
    return int(feats.shape[-1])


def _load_wsi_features(file_path: str) -> np.ndarray:
    """
    Load original WSI patch-level features from an HDF5 file.

    Args:
        file_path: absolute path to the HDF5 file.

    Returns:
        An ndarray of shape (num_patches, feature_dim).
    """
    with h5py.File(file_path, "r") as f:
        feats = f["wsi"]["features"][:]

    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    elif feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)
    return feats


def _ensure_output_dataset(
    file_path: str, shape: Tuple[int, int], overwrite: bool = False
) -> h5py.Dataset:
    """
    Ensure that `wsi/reconstructed_features` dataset exists in the HDF5 file.

    Args:
        file_path: absolute HDF5 path.
        shape: reconstructed features shape (num_patches, feature_dim).
        overwrite: if True and dataset exists, delete and recreate it.

    Returns:
        A writable h5py.Dataset object.
    """
    f = h5py.File(file_path, "a")
    wsi_group = f.require_group("wsi")

    if "reconstructed_features" in wsi_group:
        if overwrite:
            del wsi_group["reconstructed_features"]
        else:
            # 已存在且不覆盖，直接返回现有数据集
            return wsi_group["reconstructed_features"]

    ds = wsi_group.create_dataset(
        "reconstructed_features",
        shape=shape,
        dtype="float32",
        compression="gzip",
        shuffle=True,
    )
    return ds


def _build_vae_model(
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    device: str,
    checkpoint_path: str,
) -> VAE:
    """
    Build a VAE model and load weights from a checkpoint.

    Args:
        input_dim: input feature dimension.
        hidden_dims: encoder hidden layer dimensions.
        latent_dim: latent dimension.
        device: device string ('cpu' or 'cuda').
        checkpoint_path: path to trained checkpoint.

    Returns:
        VAE model with loaded weights in eval mode.
    """
    encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dims=list(reversed(hidden_dims)),
        output_dim=input_dim,
    )
    model = VAE(encoder=encoder, decoder=decoder, device=device)

    # Explicitly set weights_only=False for forward compatibility with PyTorch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Handle checkpoints saved from torch.compile:
    # such checkpoints usually prefix keys with "_orig_mod.", which must be stripped
    # before loading into a non-compiled model.
    needs_strip = any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys())
    if needs_strip:
        cleaned_state_dict = {}
        prefix = "_orig_mod."
        for key, value in state_dict.items():
            if isinstance(key, str) and key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            cleaned_state_dict[new_key] = value
        state_dict = cleaned_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _reconstruct_single_file(
    model: VAE,
    device: str,
    file_path: str,
    batch_size: int = 256,
    overwrite: bool = False,
) -> None:
    """
    Reconstruct all WSI patch features in a single HDF5 file and write back.

    Args:
        model: loaded VAE model.
        device: device string.
        file_path: absolute HDF5 path.
        batch_size: batch size for forward passes.
        overwrite: whether to overwrite existing `reconstructed_features`.
    """
    feats = _load_wsi_features(file_path)
    num_patches, feat_dim = feats.shape

    # Pre-create output dataset
    out_ds = _ensure_output_dataset(file_path, shape=(num_patches, feat_dim), overwrite=overwrite)

    # Forward in batches to avoid OOM
    with h5py.File(file_path, "a") as f, torch.no_grad():
        wsi_group = f["wsi"]
        out_ds = wsi_group["reconstructed_features"]

        for start in range(0, num_patches, batch_size):
            end = min(start + batch_size, num_patches)
            batch = feats[start:end]
            batch_tensor = torch.from_numpy(batch).to(device)

            x_hat, _, _, _ = model(batch_tensor)
            out_ds[start:end] = x_hat.cpu().numpy().astype(np.float32)


def main() -> None:
    """
    Entry point: iterate over all HDF5 files in the CSV and generate
    WSI reconstructed features.
    """
    parser = argparse.ArgumentParser(description="Generate WSI reconstructed_features")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to survival_dataset.csv")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory of HDF5 data")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Encoder hidden dimensions (must match training)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension (must match training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for reconstruction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu (default cuda, falls back to cpu if unavailable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing wsi/reconstructed_features",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, falling back to CPU.")
        device = "cpu"

    # Read CSV and collect all unique HDF5 paths
    df = pd.read_csv(args.csv_path)
    if "h5_file_path" not in df.columns:
        raise ValueError("CSV file is missing required column: h5_file_path")

    rel_paths = sorted(df["h5_file_path"].unique())
    abs_paths = [os.path.join(args.data_root_dir, p) for p in rel_paths]
    abs_paths = [p for p in abs_paths if os.path.exists(p)]

    if not abs_paths:
        raise RuntimeError("No valid HDF5 files found; please check csv_path and data_root_dir.")

    # Infer input dimension and build model
    input_dim = _infer_input_dim(abs_paths[0])
    print(f"🔍 Inferred input_dim = {input_dim}")

    model = _build_vae_model(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )

    print(f"✅ VAE model loaded. Start generating reconstructed_features for {len(abs_paths)} HDF5 files.")

    for idx, path in enumerate(abs_paths, start=1):
        print(f"[{idx}/{len(abs_paths)}] Processing: {path}")
        try:
            _reconstruct_single_file(
                model=model,
                device=device,
                file_path=path,
                batch_size=args.batch_size,
                overwrite=bool(args.overwrite),
            )
        except Exception as exc:
            print(f"❌ Failed to process file {path}: {exc}")

    print("🎉 All WSI reconstructed_features have been generated.")


if __name__ == "__main__":
    main()


