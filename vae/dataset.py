# -*- coding: utf-8 -*-
"""
VAE dataset for WSI embeddings.
Reads WSI embeddings and optionally keeps only patients with a specific label.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
import random
from torch.utils.data import Dataset
from typing import Dict, Optional

# Add project path for downstream_survival
sys.path.append('/home/zheng/zheng/multimodal-fusion/downstream_survival')
from datasets.multimodal_dataset import MultimodalDataset


class WSIVAEDataset(Dataset):
    """
    WSI VAE dataset.
    It reads WSI embeddings from MultimodalDataset and returns data per patch.
    Each sample is a single patch feature vector to avoid memory explosion.
    """
    
    def __init__(self, 
                 csv_path: str,
                 data_root_dir: str,
                 label_filter: Optional[str] = 'living',
                 use_all_data: bool = False,
                 random_seed: int = 42,
                 preload_data: bool = True,
                 print_info: bool = True):
        """
        Initialize WSI VAE dataset.

        Args:
            csv_path: CSV file path with columns: patient_id, case_id, label, h5_file_path.
            data_root_dir: data root directory.
            label_filter: label to keep, default 'living'. If None/empty, use all samples.
            use_all_data: whether to use all data (no patch sampling). If True, no sampling is applied.
            random_seed: random seed for patch sampling reproducibility.
            preload_data: whether to preload all data into memory (default True, speeds up training).
            print_info: whether to print dataset statistics and logs.
        """
        super().__init__()
        
        self.data_root_dir = data_root_dir
        self.label_filter = label_filter
        self.use_all_data = use_all_data
        self.random_seed = random_seed
        self.preload_data = preload_data
        self.print_info = print_info
        self.total_patches_before_sampling = 0  # number of patches before sampling
        self.total_patches_after_sampling = 0  # number of patches after sampling

        # Preloaded data cache
        self._preloaded_data = {}  # {patient_idx: wsi_features}

        # Use MultimodalDataset to read data
        self.base_dataset = MultimodalDataset(
            csv_path=csv_path,
            data_root_dir=data_root_dir,
            channels=['wsi=features'],
            align_channels=None,
            alignment_model_path=None,
            device='cpu',
            print_info=False
        )
        
        # If label_filter is set, filter dataset; otherwise use all data
        if self.label_filter is not None and self.label_filter.strip() != '':
            self._filter_by_label()
        else:
            # Use all data
            self._use_all_data()
        
        # Preload all data into memory (if enabled)
        if self.preload_data:
            self._preload_all_data()
        
        # Build patch-level index mapping
        # Each element is (patient_idx, patch_idx)
        # If use_all_data=False, patches for each patient may be subsampled.
        self._build_patch_indices()
        
        if self.print_info:
            self._print_summary()
    
    def _filter_by_label(self):
        """Filter dataset to keep only patients with the specified label."""
        filtered_indices = []
        self.case_ids = []
        
        for idx in range(len(self.base_dataset)):
            label = self.base_dataset.get_label(idx)
            if label == self.label_filter:
                filtered_indices.append(idx)
                case_id = self.base_dataset.case_ids[idx]
                self.case_ids.append(case_id)
        
        self.filtered_indices = filtered_indices
        
        if self.print_info:
            print(f"🔍 Filter label '{self.label_filter}': {len(self.base_dataset)} -> {len(self.filtered_indices)} samples")
    
    def _use_all_data(self):
        """Use all patients without any label filtering."""
        self.filtered_indices = list(range(len(self.base_dataset)))
        self.case_ids = self.base_dataset.case_ids.copy()
        
        if self.print_info:
            print(f"📦 Use all data: {len(self.filtered_indices)} patient samples")
    
    def _preload_all_data(self):
        """
        Preload all patient data into memory to avoid repeated disk reads.
        """
        if self.print_info:
            print("📥 Preloading all data into memory...")
        
        total_patches = 0
        for patient_idx in self.filtered_indices:
            try:
                channel_data, _ = self.base_dataset[patient_idx]
                wsi_features = channel_data['wsi=features']
                
                # Ensure 2D tensor
                if wsi_features.dim() == 1:
                    wsi_features = wsi_features.unsqueeze(0)
                
                # Store in memory
                self._preloaded_data[patient_idx] = wsi_features.float()
                total_patches += wsi_features.shape[0]
            except Exception as e:
                if self.print_info:
                    print(f"⚠️ Failed to preload data for patient {patient_idx}: {e}")
        
        if self.print_info:
            total_mb = sum(f.shape[0] * f.shape[-1] * 4 for f in self._preloaded_data.values()) / (1024 * 1024)
            print(f"✅ Preload finished: {len(self._preloaded_data)} patients, {total_patches} patches")
            print(f"   Estimated memory usage: {total_mb:.2f} MB ({total_mb/1024:.3f} GB)")
    
    def resample_patches(self, random_seed: Optional[int] = None):
        """
        Resample patches (used to increase data diversity during training).

        Args:
            random_seed: random seed; if None, current time is used.
        """
        if self.use_all_data:
            # When using all data, resampling is not needed
            return
        
        if random_seed is None:
            import time
            random_seed = int(time.time())
        
        # Update random seed
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Rebuild patch indices (will resample)
        self._build_patch_indices()
        
        if self.print_info:
            print(f"🔄 Finished resampling patches (random seed: {random_seed})")
            print(f"   Current number of patches: {len(self.patch_indices)}")
    
    
    def _build_patch_indices(self):
        """
        Build a mapping from global patch index to (patient_idx, patch_idx).
        Each dataset sample is a single patch to avoid memory explosion.
        Only shapes are inspected; full data is not loaded here.

        When use_all_data is False, patches are subsampled per patient:
        - if num_patches > 1000, sample 10%
        - if 100 < num_patches <= 1000, sample 20%
        - if num_patches <= 100, use all
        """
        self.patch_indices = []  # list of (patient_idx, patch_idx)
        self.patient_to_patch_range = {}  # per patient patch index range
        
        if self.print_info:
            print("📝 Building patch index mapping...")
        
        # Set random seed to ensure reproducibility
        if not self.use_all_data:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        total_patches_before = 0
        total_patches_after = 0
        
        for patient_idx in self.filtered_indices:
            # Get number of patches for this patient (shape only, no heavy load)
            try:
                channel_data, _ = self.base_dataset[patient_idx]
                wsi_features = channel_data['wsi=features']
                
                # Ensure 2D
                if wsi_features.dim() == 1:
                    num_patches = 1
                else:
                    num_patches = wsi_features.shape[0]
                
                total_patches_before += num_patches
                
                # Decide whether to sample patches based on use_all_data
                if self.use_all_data:
                    # Use all patches
                    selected_patch_indices = list(range(num_patches))
                else:
                    # Decide sampling ratio based on number of patches
                    if num_patches > 1000:
                        # If patches > 1000, sample 10%
                        sample_ratio = 0.1
                        num_samples = max(1, int(num_patches * sample_ratio))
                        selected_patch_indices = random.sample(range(num_patches), num_samples)
                        selected_patch_indices.sort()  # keep order
                    elif num_patches > 100:
                        # If 100 < patches <= 1000, sample 20%
                        sample_ratio = 0.2
                        num_samples = max(1, int(num_patches * sample_ratio))
                        selected_patch_indices = random.sample(range(num_patches), num_samples)
                        selected_patch_indices.sort()  # keep order
                    else:
                        # If patches <= 100, use all patches
                        selected_patch_indices = list(range(num_patches))
                
                total_patches_after += len(selected_patch_indices)
                
                # Record this patient's patch range
                start_idx = len(self.patch_indices)
                for patch_idx in selected_patch_indices:
                    self.patch_indices.append((patient_idx, patch_idx))
                end_idx = len(self.patch_indices)
                
                self.patient_to_patch_range[patient_idx] = (start_idx, end_idx)
            except Exception as e:
                if self.print_info:
                    print(f"⚠️ Failed to read patches for patient {patient_idx}: {e}")
                # Fallback: assume 1 patch
                start_idx = len(self.patch_indices)
                self.patch_indices.append((patient_idx, 0))
                end_idx = len(self.patch_indices)
                self.patient_to_patch_range[patient_idx] = (start_idx, end_idx)
                total_patches_before += 1
                total_patches_after += 1
        
        self.total_patches_before_sampling = total_patches_before
        self.total_patches_after_sampling = total_patches_after
        
        if self.print_info and not self.use_all_data:
            reduction_ratio = (1 - total_patches_after / total_patches_before) * 100 if total_patches_before > 0 else 0
            print(f"📊 Patch sampling stats: {total_patches_before} -> {total_patches_after} patches (reduced {reduction_ratio:.1f}%)")
            print("   Sampling rules: >1000 -> 10%, >100 -> 20%, <=100 -> all")
            print(f"   Random seed: {self.random_seed}")
    
    def _print_summary(self):
        """Print dataset summary."""
        print("📊 WSI VAE dataset summary:")
        print(f"  Number of patients: {len(self.filtered_indices)}")
        print(f"  Total number of patches: {len(self.patch_indices)}")
        if self.label_filter is not None and self.label_filter.strip() != '':
            print(f"  Label filter: {self.label_filter}")
        else:
            print("  Label filter: None (use all data)")
        if self.use_all_data:
            print("  Patch sampling: disabled (use all patches)")
        else:
            if self.total_patches_before_sampling > 0:
                reduction_ratio = (1 - self.total_patches_after_sampling / self.total_patches_before_sampling) * 100
                print(f"  Patch sampling: enabled ({self.total_patches_before_sampling} -> {self.total_patches_after_sampling} patches, reduced {reduction_ratio:.1f}%)")
                print("    Rules: >1000 -> 10%, >100 -> 20%, <=100 -> all")
            else:
                print("  Patch sampling: enabled")
                print("    Rules: >1000 -> 10%, >100 -> 20%, <=100 -> all")
        
        # Check feature dimension of the first patch (lazy load to avoid heavy init)
        if len(self) > 0:
            try:
                sample = self[0]
                if isinstance(sample, tuple):
                    patch_feature = sample[0]
                else:
                    patch_feature = sample
                print(f"  Patch feature dimension: {patch_feature.shape[0]}")
            except Exception as e:
                print(f"  ⚠️ Failed to obtain feature dimension: {e}")
    
    def __len__(self) -> int:
        """Return number of patches in the dataset."""
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single patch feature vector.

        Args:
            idx: patch index (not patient index)

        Returns:
            patch_feature: feature vector for one patch, shape (feature_dim,)
        """
        patient_idx, patch_idx = self.patch_indices[idx]
        
        # If preloaded, read directly from memory
        if self.preload_data and patient_idx in self._preloaded_data:
            wsi_features = self._preloaded_data[patient_idx]
        else:
            # Otherwise load all patches for this patient from base_dataset
            channel_data, label = self.base_dataset[patient_idx]
            wsi_features = channel_data['wsi=features']
            
            # Ensure 2D (num_patches, feature_dim)
            if wsi_features.dim() == 1:
                wsi_features = wsi_features.unsqueeze(0)
            wsi_features = wsi_features.float()
        
        # Extract the specified patch
        patch_feature = wsi_features[patch_idx]  # (feature_dim,)
        
        return patch_feature
    
    def get_feature_dim(self) -> int:
        """
        Get the feature dimension of a single patch.

        Returns:
            Feature dimension.
        """
        if len(self) == 0:
            raise ValueError("Dataset is empty; cannot infer feature dimension.")
        
        sample = self[0]
        # Each sample is now a single patch feature vector, shape (feature_dim,)
        return sample.shape[0]
    
    def get_patient_patches(self, patient_idx: int) -> torch.Tensor:
        """
        Get all patches for a given patient (for inference or post-processing).

        Args:
            patient_idx: index of the patient in filtered_indices.

        Returns:
            patches: tensor of shape (num_patches, feature_dim)
        """
        if patient_idx not in self.patient_to_patch_range:
            raise ValueError(f"Patient index {patient_idx} not found")
        
        start_idx, end_idx = self.patient_to_patch_range[patient_idx]
        patches = []
        for idx in range(start_idx, end_idx):
            patch = self[idx]
            patches.append(patch)
        
        return torch.stack(patches, dim=0)  # (num_patches, feature_dim)

