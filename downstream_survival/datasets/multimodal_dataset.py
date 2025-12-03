"""
Flexible multimodal dataset.
Supports on-demand channel loading, optional alignment, and dict-style outputs.
"""
import os
import torch
import numpy as np
import pandas as pd
import h5py
import sys
import time
import random
import threading
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Tuple

# Add multimodal-fusion alignment project path
sys.path.append('/home/zheng/zheng/multimodal-fusion/alignment')
try:
    from alignment_model import MultiModalAlignmentModel
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    print("⚠️ 对齐模型不可用，将使用标准模式")

# 全局文件锁字典，用于处理HDF5并发访问
_file_locks = {}
_lock_dict_lock = threading.Lock()

class MultimodalDataset(Dataset):
    """
    Flexible multimodal dataset.
    Supports on-demand channel loading, optional alignment, and dict-style outputs.
    """
    
    def __init__(self,
                 csv_path: str,
                 data_root_dir: str,
                 channels: List[str] = None,
                 align_channels: Dict[str, str] = None,
                 alignment_model_path: Optional[str] = None,
                 device: str = 'auto',
                 print_info: bool = True,
                 preload_all: bool = False):
        """
        Initialize a flexible multimodal dataset.
        
        Args:
            csv_path: CSV with patient_id, case_id, label, h5_file_path columns.
            data_root_dir: data root directory used to resolve relative h5 paths.
            channels: list of channel names to load, e.g. ['wsi=features', 'clinical=val'].
            align_channels: mapping of channel -> modality name, e.g. {"tma_CD3": "CD3"}.
            alignment_model_path: path to a pretrained alignment model.
            device: 'auto' (cuda if available else cpu), 'cpu', or 'cuda'.
            print_info: whether to print dataset information.
            preload_all: if True, preload all samples (channels + labels) into memory.
        """
        super().__init__()
        
        # Basic settings
        self.data_root_dir = data_root_dir
        self.channels = channels
        self.align_channels = align_channels or {}  # default: no alignment
        self.print_info = print_info
        # In-memory cache: case_id -> (channel_data_dict, label_tensor)
        self._preloaded_samples: Dict[str, Tuple[Dict[str, torch.Tensor], torch.Tensor]] = {}
        
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load CSV
        self.data_df = pd.read_csv(csv_path)
        
        # Validate CSV structure
        required_columns = ['patient_id', 'case_id', 'label', 'h5_file_path']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必需列: {missing_columns}")
        
        # Build mappings
        self.case_to_file = {}
        self.case_to_label = {}
        for _, row in self.data_df.iterrows():
            case_id = row['case_id']
            file_path = row['h5_file_path']
            label = row['label']
            
            # 处理路径：如果提供了 data_root_dir，则拼接路径
            file_path = os.path.join(self.data_root_dir, file_path)
            
            self.case_to_file[case_id] = file_path
            self.case_to_label[case_id] = label
        
        self.case_ids = sorted(self.case_to_file.keys())
        
        # Alignment model setup
        self.alignment_model = None
        if alignment_model_path and os.path.exists(alignment_model_path) and ALIGNMENT_AVAILABLE:
            self._load_alignment_model(alignment_model_path)
            if print_info:
                print(f"✅ 数据集加载对齐模型: {alignment_model_path}")
        
        # Validate channels and align_channels
        self._validate_channels()
        
        # Filter out samples missing required channels
        self._filter_missing_data()
        
        # Build label <-> int mapping
        self._build_label_mapping()
        
        if print_info:
            self._print_summary()

        # Optionally preload everything into memory
        if preload_all:
            self.preload_all_samples(print_progress=print_info)
    
    def _build_label_mapping(self):
        """Build mapping from label string to integer id."""
        # Get all unique labels
        unique_labels = set(self.case_to_label.values())
        
        # Deterministic mapping (sorted labels)
        self.label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        
        if self.print_info:
            print(f"🏷️ Label mapping: {self.label_to_int}")
    
    def _validate_channels(self):
        """Validate channels and align_channels configuration."""
        if not self.channels:
            raise ValueError("channels must not be empty")
        
        # All keys of align_channels must appear in channels
        if self.align_channels:
            missing_align = [ch for ch in self.align_channels.keys() if ch not in self.channels]
            if missing_align:
                raise ValueError(f"align_channels keys not contained in channels: {missing_align}")
        
        if self.print_info:
            print("📋 Channels config:")
            print(f"  channels to load: {self.channels}")
            print(f"  align_channels: {self.align_channels}")
    
    def _load_alignment_model(self, alignment_model_path: str):
        """Load pretrained alignment model."""
        # Use configured device
        checkpoint = torch.load(alignment_model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        # Infer modality names from checkpoint keys
        ckpt_modalities = []
        if isinstance(state_dict, dict):
            for k in state_dict.keys():
                if isinstance(k, str) and k.startswith('alignment_layers.'):
                    parts = k.split('.')
                    if len(parts) >= 3:
                        ckpt_modalities.append(parts[1])
            # De-duplicate while preserving order
            seen = set()
            ckpt_modalities = [m for m in ckpt_modalities if not (m in seen or seen.add(m))]

        # Prefer align_channels values; otherwise use inferred modalities from ckpt
        if self.align_channels:
            model_modalities = list(self.align_channels.values())
        else:
            model_modalities = ckpt_modalities if ckpt_modalities else []
        
        self.alignment_modalities = model_modalities
        
        if model_modalities:
            self.alignment_model = MultiModalAlignmentModel(
                modality_names=model_modalities,
                feature_dim=1024,
                num_layers=2
            )

            # Only load weights relevant to the selected modalities
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # Skip mlp_predictor which depends on specific modality count
                if 'mlp_predictor' in key:
                    continue
                
                # Check if this weight belongs to one of the used modalities
                should_include = True
                for modality in model_modalities:
                    if f'alignment_layers.{modality}.' in key:
                        should_include = True
                        break
                    elif 'alignment_layers.' in key:
                        # Skip keys belonging to modalities we don't use
                        ckpt_modality = key.split('.')[1]
                        if ckpt_modality not in model_modalities:
                            should_include = False
                            break
                
                if should_include:
                    filtered_state_dict[key] = value

            missing, unexpected = self.alignment_model.load_state_dict(filtered_state_dict, strict=False)
            self.alignment_model.eval()
            
            if self.print_info:
                print(f"🎯 Alignment model loaded | modalities={self.alignment_modalities}")
                print(f"   Loaded weight tensors: {len(filtered_state_dict)}")
                if missing:
                    print(f"   Missing weights: {len(missing)}")
                if unexpected:
                    print(f"   Unexpected weights: {len(unexpected)}")
        else:
            if self.print_info:
                print("⚠️ No align_channels specified; alignment model will not be used.")
            self.alignment_model = None

    def _filter_missing_data(self):
        """Filter out samples that lack any of the configured channels."""
        # Filtering always runs; print_info only controls logging.
        if self.print_info:
            print(f"🔍 Checking data completeness, channels: {self.channels}")
        
        valid_cases = []
        missing_count = 0
        
        for case_id in self.case_ids:
            file_path = self.case_to_file[case_id]
            
            if os.path.exists(file_path):
                try:
                    with h5py.File(file_path, 'r') as f:
                        missing_channels = []
                        if self.channels is None:
                            self.channels = list(f.keys())
                        for channel in self.channels:
                            channel_parts = channel.split('=')
                            if len(channel_parts) == 2:
                                if channel_parts[0] not in f or channel_parts[1] not in f[channel_parts[0]]:
                                    missing_channels.append(channel)
                            elif len(channel_parts) == 3:
                                if channel_parts[0] not in f or channel_parts[1] not in f[channel_parts[0]] or channel_parts[2] not in f[channel_parts[0]][channel_parts[1]]:
                                    missing_channels.append(channel)
                            else:
                                assert False, f"⚠️ Invalid channel format: {channel}"
                        
                        if missing_channels:
                            missing_count += 1
                            if self.print_info and missing_count <= 5:
                                print(f"  ⚠️  {case_id}: missing channels {missing_channels}")
                        else:
                            valid_cases.append(case_id)
                except Exception as e:
                    missing_count += 1
                    if self.print_info and missing_count <= 5:
                        print(f"  ❌ {case_id}: failed to read file - {e}")
            else:
                missing_count += 1
                if self.print_info and missing_count <= 5:
                    print(f"  ❌ {case_id}: file does not exist")
        
        # Update list of valid cases
        original_count = len(self.case_ids)
        self.case_ids = valid_cases
        
        if self.print_info:
            new_count = len(self.case_ids)
            print(f"📊 Data filter result: {original_count} -> {new_count} ({new_count/original_count*100:.1f}%)")

    def _print_summary(self):
        """Print a short dataset summary."""
        print("📊 Dataset summary:")
        print(f"  Total samples: {len(self.case_ids)}")
        print(f"  Channels: {self.channels}")
        print(f"  Align channels: {self.align_channels}")
        print(f"  Use alignment: {True if self.alignment_model is not None else False}")
        
        # Label statistics
        labels = [self.case_to_label[case_id] for case_id in self.case_ids]
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")

    def __len__(self) -> int:
        """Return dataset size (number of cases)."""
        return len(self.case_ids)

    def preload_all_samples(self, print_progress: bool = True) -> None:
        """
        Preload all samples (channels + labels) into memory.

        This is useful to benchmark whether the full dataset fits into RAM
        and can significantly speed up training by avoiding HDF5 I/O.
        """
        if self._preloaded_samples:
            # Already preloaded
            return

        total_cases = len(self.case_ids)
        total_bytes = 0

        for idx, case_id in enumerate(self.case_ids):
            if print_progress and idx % 20 == 0:
                print(f"📥 Preloading sample {idx+1}/{total_cases} (case_id={case_id})...")

            channel_data, label = self._load_single_case_by_id(case_id)
            self._preloaded_samples[case_id] = (channel_data, label)

            # Approximate memory usage (float32 tensors only)
            for tensor in channel_data.values():
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * 4

        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_mb / 1024
        if print_progress:
            print(f"✅ Finished preloading {len(self._preloaded_samples)} samples.")
            print(f"   Estimated tensor memory: {total_mb:.2f} MB ({total_gb:.3f} GB)")

    def _load_single_case_by_id(self, case_id: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Core logic to load a single case (channels + label) given case_id.

        This is used by both __getitem__ and preload_all_samples.
        """
        file_path = self.case_to_file[case_id]

        if not os.path.exists(file_path):
            print(f"⚠️ File does not exist: {file_path}")
            return self._get_fallback_data()

        file_lock = self._get_file_lock(file_path)

        with file_lock:
            with h5py.File(file_path, 'r') as hdf5_file:
                channel_data: Dict[str, torch.Tensor] = {}
                for channel in self.channels:
                    data = self._read_with_retry(hdf5_file, channel, max_retries=3)
                    if data is not None:
                        channel_data[channel] = torch.from_numpy(self._standardize_array(data))
                    else:
                        assert False, f"⚠️ Failed to read channel {channel}"

        # Optional alignment
        if self.alignment_model is not None and self.align_channels:
            try:
                align_data: Dict[str, torch.Tensor] = {}
                for channel, modality_name in self.align_channels.items():
                    if channel in channel_data:
                        align_data[modality_name] = channel_data[channel]

                if align_data:
                    with torch.no_grad():
                        aligned_features = self.alignment_model(align_data)

                    # Add aligned features with "aligned_" prefix
                    for modality_name, aligned_feat in aligned_features.items():
                        if isinstance(aligned_feat, torch.Tensor) and aligned_feat.numel() > 0:
                            original_channel = None
                            for ch, mod in self.align_channels.items():
                                if mod == modality_name:
                                    original_channel = ch
                                    break
                            if original_channel:
                                aligned_key = f"aligned_{original_channel}"
                                channel_data[aligned_key] = aligned_feat

                    if self.print_info and not self._preloaded_samples:
                        # Only log once when not in preload mode
                        print(f"🎯 Alignment finished, produced {len(aligned_features)} aligned feature tensors")

            except Exception as e:
                if self.print_info:
                    print(f"⚠️ Alignment failed: {e}")

        label_str = self.case_to_label[case_id]
        label = torch.tensor(self.label_to_int[label_str], dtype=torch.long)
        return channel_data, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample as a dict of tensors and its label.
        
        Args:
            idx: sample index.
            
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: (channel_data, label)
        """
        case_id = self.case_ids[idx]
        # If preloaded, return from memory cache
        if case_id in self._preloaded_samples:
            return self._preloaded_samples[case_id]

        # Fallback: on-demand load from disk
        try:
            return self._load_single_case_by_id(case_id)
        except Exception as e:
            file_path = self.case_to_file[case_id]
            print(f"❌ Failed to load file {file_path}: {e}")
            raise e

    def _standardize_array(self, arr) -> np.ndarray:
        """Standardize numpy array to 2D float32."""
        if arr is None:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0])
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(np.float32, copy=False)
    
    def _read_with_retry(self, hdf5_file, channel: str, max_retries: int = 3) -> np.ndarray:
        """
        Read data from HDF5 with retry logic to mitigate concurrent access issues.
        
        Args:
            hdf5_file: opened HDF5 file object.
            channel: channel string, e.g. "wsi=features".
            max_retries: maximum retry count.
            
        Returns:
            Numpy array with the loaded data.
        """
        channel = channel.split('=')
        for attempt in range(max_retries + 1):
            try:
                # Try to read data
                if len(channel) == 2:
                    # WSI layout
                    data = hdf5_file[channel[0]][channel[1]][:]
                elif len(channel) == 3:
                    data = hdf5_file[channel[0]][channel[1]][channel[2]][:]
                else:
                    assert False, f"⚠️ Channel {channel} 格式错误"
                return data
                
            except Exception as e:
                if attempt < max_retries:
                    # 计算退避时间：指数退避 + 随机抖动
                    base_delay = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                    jitter = random.uniform(0, 0.1)  # 0-0.1s随机抖动
                    delay = base_delay + jitter
                    
                    print(f"⚠️ 读取 {channel} 失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"🔄 等待 {delay:.2f}s 后重试...")
                    time.sleep(delay)
                else:
                    # Final failure
                    print(f"❌ Failed to read {channel} after {max_retries + 1} attempts: {e}")
                    raise e
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """
        Get a per-file lock to ensure no concurrent access to the same HDF5.
        
        Args:
            file_path: path to the HDF5 file.
            
        Returns:
            threading.Lock instance.
        """
        with _lock_dict_lock:
            if file_path not in _file_locks:
                _file_locks[file_path] = threading.Lock()
            return _file_locks[file_path]

    def get_label(self, idx: int) -> str:
        """Get the label string for a sample index."""
        case_id = self.case_ids[idx]
        return self.case_to_label[case_id]
