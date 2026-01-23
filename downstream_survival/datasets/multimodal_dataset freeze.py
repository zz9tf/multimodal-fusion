"""
Flexible multimodal dataset.
Supports on-demand channel loading, optional alignment, and dict-style outputs.
"""
import os
import torch
import numpy as np
import h5py
import sys
import time
import random
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

def parse_channels(channels: List[str]) -> List[str]:
    """
    Parse simplified channel names to full HDF5 paths.
    
    Supported formats:
    - 'wsi' -> 'wsi=features'
    - 'tma' -> all TMA markers with features
    - 'clinical' -> 'clinical=val'
    - 'cd3' -> 'tma=cd3=features'
    """
    if not channels:
        return []
    
    TMA_MARKERS = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']
    
    parsed = []
    for ch in channels:
        if '=' in ch:
            # Already in full format
            parsed.append(ch)
        elif ch == 'wsi':
            parsed.append('wsi=features')
        elif ch == 'tma':
            # All TMA markers
            parsed.extend([f'tma={marker}=features' for marker in TMA_MARKERS])
        elif ch in TMA_MARKERS:
            # Specific TMA marker
            parsed.append(f'tma={ch}=features')
        elif ch in ['clinical', 'pathological', 'blood', 'icd', 'tma_cell_density']:
            # Tabular data
            parsed.append(f'{ch}=val')
        else:
            raise ValueError(f"Unknown channel: {ch}")
    
    return parsed


class MultimodalDataset(Dataset):
    """
    Flexible multimodal dataset.
    Supports on-demand channel loading, optional alignment, and dict-style outputs.
    """
    
    def __init__(self,
                 data_root_dir: str,
                 channels: List[str] = None,
                 device: str = 'auto',
                 print_info: bool = True,
                 preload_all: bool = True):
        """
        Initialize a flexible multimodal dataset by scanning all H5 files directly.

        Args:
            data_root_dir: Root directory containing H5 files to scan.
            channels: list of channel names to load, e.g. ['wsi=features', 'clinical=val'].
            align_channels: mapping of channel -> modality name, e.g. {"tma_CD3": "CD3"}.
            device: 'auto' (cuda if available else cpu), 'cpu', or 'cuda'.
            print_info: whether to print dataset information.
            preload_all: if True, preload all samples (channels + labels) into memory.
        """
        super().__init__()

        # Basic settings
        self.data_root_dir = data_root_dir
        # Parse channels if they are in simplified format
        self.channels = parse_channels(channels) if channels else None
        self.print_info = print_info
        # In-memory cache: instance_id -> (channel_data_dict, label_tensor)
        self._preloaded_samples: Dict[str, Tuple[Dict[str, torch.Tensor], torch.Tensor]] = {}

        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Scan all H5 files directly
        self.slide_to_file = {}
        if print_info:
            print("🔍 扫描所有H5文件...")

        # Scan all H5 files in data_root_dir
        for root, dirs, files in os.walk(self.data_root_dir):
            for file in files:
                if file.endswith('.h5'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.data_root_dir)
                    self.slide_to_file[rel_path] = full_path

        if print_info:
            print(f"✅ 发现 {len(self.slide_to_file)} 个H5文件")

        self._build_instance_list()

        # Validate channels and align_channels
        self._validate_channels()

        if print_info:
            self._print_summary()

        # Optionally preload everything into memory
        if preload_all:
            self.preload_all_samples(print_progress=print_info)
    
    def _build_instance_list(self):
        """
        Build a list of all slides (one instance per H5 file).
        Each instance represents one patient's complete multimodal data.
        """
        self.instances = []

        if self.print_info:
            print("🔍 Building slide list...")

        for slide_path, file_path in self.slide_to_file.items():
            if not os.path.exists(file_path):
                if self.print_info:
                    print(f"⚠️ File not found: {file_path}")
                continue

            # Each H5 file is one instance
            self.instances.append({
                'slide_id': slide_path,
                'file_path': file_path
            })

        if self.print_info:
            print(f"✅ Found {len(self.instances)} slides")


    def _validate_channels(self):
        """Validate channels and align_channels configuration."""
        # Set default channels if not specified
        if self.channels is None:
            # Default: load all TMA markers and WSI
            tma_markers = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']
            self.channels = [f'tma={marker}=features' for marker in tma_markers] + ['wsi=features']
            if self.print_info:
                print(f"⚙️ No channels specified, using default: all TMA markers + WSI")
        
        if not self.channels:
            raise ValueError("channels must not be empty")

        if self.print_info:
            print("📋 Channels config:")
            print(f"  channels to load: {self.channels}")
    


    def _print_summary(self):
        """Print a short dataset summary."""
        print("📊 Dataset summary:")
        print(f"  Total samples: {len(self.instances)}")
        print(f"  Channels: {self.channels}")

        # Label statistics (read from HDF5 files)
        label_counts = {'living': 0, 'deceased': 0}
        for slide_path, file_path in self.slide_to_file.items():
            if os.path.exists(file_path):
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'survival_status' in f:
                            label_val = f['survival_status'][()]
                            if isinstance(label_val, bytes):
                                label_str = label_val.decode('utf-8')
                            else:
                                label_str = str(label_val)
                            if label_str in label_counts:
                                label_counts[label_str] += 1
                except:
                    pass

        print("  living: {}".format(label_counts['living']))
        print("  deceased: {}".format(label_counts['deceased']))

    def __len__(self) -> int:
        """Return dataset size (number of instances)."""
        return len(self.instances)

    def preload_all_samples(self, print_progress: bool = True) -> None:
        """
        Preload all samples (channels + labels) into memory.

        This is useful to benchmark whether the full dataset fits into RAM
        and can significantly speed up training by avoiding HDF5 I/O.
        """
        if self._preloaded_samples:
            # Already preloaded
            return

        total_slides = len(self.instances)
        total_bytes = 0

        for idx, instance_info in enumerate(self.instances):
            slide_id = instance_info['slide_id']
            if print_progress and idx % 20 == 0:
                print(f"📥 Preloading sample {idx+1}/{total_slides} (slide_id={slide_id})...")

            channel_data, label = self._load_single_instance(instance_info)
            self._preloaded_samples[slide_id] = (channel_data, label)

            # Approximate memory usage (float32 tensors only)
            for tensor in channel_data.values():
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * 4

        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_mb / 1024
        if print_progress:
            print(f"✅ Finished preloading {len(self._preloaded_samples)} samples.")
            print(f"   Estimated tensor memory: {total_mb:.2f} MB ({total_gb:.3f} GB)")

    def _load_single_instance(self, instance_info: Dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Core logic to load a single slide (all channels + label).

        This is used by both __getitem__ and preload_all_samples.
        """
        file_path = instance_info['file_path']

        if not os.path.exists(file_path):
            print(f"⚠️ File does not exist: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, 'r') as hdf5_file:
            channel_data: Dict[str, torch.Tensor] = {}
            
            # Read all requested channels
            for channel in self.channels:
                # Regular channel reading (like the old version)
                # No retries for missing channels - they either exist or don't
                data = self._read_with_retry(hdf5_file, channel, max_retries=0)
                if data is not None:
                    channel_data[channel] = torch.from_numpy(self._standardize_array(data))
                # Silently skip missing channels - this is expected behavior

            # Read label from HDF5 file and convert to int
            if 'survival_status' in hdf5_file:
                label_val = hdf5_file['survival_status'][()]
                if isinstance(label_val, bytes):
                    label_str = label_val.decode('utf-8')
                else:
                    label_str = str(label_val)

                # Convert string label to int (living=1, deceased=0)
                if label_str == 'living':
                    label_int = 1
                elif label_str == 'deceased':
                    label_int = 0
                else:
                    raise ValueError(f"Unknown label: {label_str}")

                label = torch.tensor(label_int, dtype=torch.long)
            else:
                raise ValueError(f"Label not found in HDF5 file: {file_path}")
                
        return channel_data, label

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single sample as a dict of tensors and its label.

        Args:
            idx: sample index.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: (channel_data, label)
        """
        instance_info = self.instances[idx]
        slide_id = instance_info['slide_id']
        
        # If preloaded, return from memory cache
        if slide_id in self._preloaded_samples:
            return self._preloaded_samples[slide_id]

        # Fallback: on-demand load from disk
        try:
            return self._load_single_instance(instance_info)
        except Exception as e:
            file_path = instance_info['file_path']
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
            channel: channel string, e.g. "wsi=features", "tma=cd3=features".
            max_retries: maximum retry count.
            
        Returns:
            Numpy array with the loaded data.
        """
        channel_parts = channel.split('=')
        for attempt in range(max_retries + 1):
            try:
                # Try to read data based on channel format
                if len(channel_parts) == 2:
                    # Format: "wsi=features" or "clinical=val"
                    data = hdf5_file[channel_parts[0]][channel_parts[1]][:]
                elif len(channel_parts) == 3:
                    # Format: "tma=cd3=features"
                    data = hdf5_file[channel_parts[0]][channel_parts[1]][channel_parts[2]][:]
                else:
                    raise ValueError(f"⚠️ Invalid channel format: {channel}")
                return data
                
            except Exception as e:
                if attempt < max_retries:
                    # Calculate backoff time: exponential backoff + random jitter
                    base_delay = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                    jitter = random.uniform(0, 0.1)  # 0-0.1s random jitter
                    delay = base_delay + jitter
                    
                    if self.print_info:
                        print(f"⚠️ Failed to read {channel} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"🔄 Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                else:
                    # Final failure - return None silently (missing channels are expected)
                    return None
    

    def get_label(self, idx: int) -> str:
        """Get the label string for a sample index."""
        instance_info = self.instances[idx]
        file_path = instance_info['file_path']

        if os.path.exists(file_path):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'survival_status' in f:
                        label_val = f['survival_status'][()]
                        if isinstance(label_val, bytes):
                            return label_val.decode('utf-8')
                        else:
                            return str(label_val)
                    else:
                        raise ValueError(f"No survival_status in HDF5 file: {file_path}")
            except Exception as e:
                raise ValueError(f"Could not read label from {file_path}: {e}")
        else:
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

if __name__ == "__main__":
    dataset = MultimodalDataset(
        data_root_dir="/home/zheng/zheng/public/1", 
        channels=['wsi', 'tma', 'clinical', 'pathological', 'blood', 'icd', 'tma_cell_density']
    )
    