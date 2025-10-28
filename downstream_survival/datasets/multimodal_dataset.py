"""
灵活的多模态数据集类
支持按需加载channels，按需对齐，返回字典格式
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

# 添加multimodal-fusion项目路径
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
    灵活的多模态数据集类
    支持按需加载channels，按需对齐，返回字典格式
    """
    
    def __init__(self,
                 csv_path: str,
                 data_root_dir: str,
                 channels: List[str] = None,
                 align_channels: Dict[str, str] = None,
                 alignment_model_path: Optional[str] = None,
                 device: str = 'auto',
                 print_info: bool = True):
        """
        初始化灵活的多模态数据集
        
        Args:
            csv_path: 包含 patient_id, case_id, label, h5_file_path 的 CSV 文件
            data_root_dir: 数据根目录，用于拼接相对路径
            channels: 需要加载的channel列表，如 ['features', 'tma_CD3', 'tma_CD8', ...]
            align_channels: 需要对齐的channel映射，如 {"tma_CD3": "CD3", "tma_CD8": "CD8"}
            alignment_model_path: 对齐模型路径
            device: 设备选择，'auto'自动选择，'cpu'强制CPU，'cuda'强制GPU
            print_info: 是否打印信息
        """
        super().__init__()
        
        # 基本设置
        self.data_root_dir = data_root_dir
        self.channels = channels
        self.align_channels = align_channels or {}  # 默认不对齐
        self.print_info = print_info
        
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载CSV数据
        self.data_df = pd.read_csv(csv_path)
        
        # 验证CSV文件结构
        required_columns = ['patient_id', 'case_id', 'label', 'h5_file_path']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必需列: {missing_columns}")
        
        # 创建映射
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
        
        # 获取所有case_id列表
        self.case_ids = list(self.case_to_file.keys())
        
        # 对齐模型设置
        self.alignment_model = None
        if alignment_model_path and os.path.exists(alignment_model_path) and ALIGNMENT_AVAILABLE:
            self._load_alignment_model(alignment_model_path)
            if print_info:
                print(f"✅ 数据集加载对齐模型: {alignment_model_path}")
        
        # 验证channels和align_channels
        self._validate_channels()
        
        # 过滤缺失数据
        self._filter_missing_data()
        
        # 建立标签映射字典
        self._build_label_mapping()
        
        if print_info:
            self._print_summary()
    
    def _build_label_mapping(self):
        """建立标签到数字的映射字典"""
        # 获取所有唯一的标签
        unique_labels = set(self.case_to_label.values())
        
        # 创建标签到数字的映射（按字母顺序排序确保一致性）
        self.label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        
        if self.print_info:
            print(f"🏷️ 标签映射: {self.label_to_int}")
    
    def _validate_channels(self):
        """验证channels和align_channels的合理性"""
        if not self.channels:
            raise ValueError("channels不能为空")
        
        # 检查align_channels的keys是否都在channels中
        if self.align_channels:
            missing_align = [ch for ch in self.align_channels.keys() if ch not in self.channels]
            if missing_align:
                raise ValueError(f"align_channels中的channels不在channels中: {missing_align}")
        
        if self.print_info:
            print(f"📋 Channels配置:")
            print(f"  加载channels: {self.channels}")
            print(f"  对齐channels: {self.align_channels}")
    
    def _load_alignment_model(self, alignment_model_path: str):
        """加载预训练的对齐模型"""
        # 使用指定的设备
        checkpoint = torch.load(alignment_model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        # 推断模态名称
        ckpt_modalities = []
        if isinstance(state_dict, dict):
            for k in state_dict.keys():
                if isinstance(k, str) and k.startswith('alignment_layers.'):
                    parts = k.split('.')
                    if len(parts) >= 3:
                        ckpt_modalities.append(parts[1])
            # 去重保持顺序
            seen = set()
            ckpt_modalities = [m for m in ckpt_modalities if not (m in seen or seen.add(m))]

        # 使用align_channels的values作为模型模态，如果为空则使用ckpt_modalities
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

            # 只加载匹配的权重
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # 跳过 mlp_predictor，因为它依赖于特定的模态数量
                if 'mlp_predictor' in key:
                    continue
                
                # 检查这个权重是否属于我们需要的模态
                should_include = True
                for modality in model_modalities:
                    if f'alignment_layers.{modality}.' in key:
                        should_include = True
                        break
                    elif 'alignment_layers.' in key:
                        # 如果这个key包含其他模态，则跳过
                        ckpt_modality = key.split('.')[1]
                        if ckpt_modality not in model_modalities:
                            should_include = False
                            break
                
                if should_include:
                    filtered_state_dict[key] = value

            missing, unexpected = self.alignment_model.load_state_dict(filtered_state_dict, strict=False)
            self.alignment_model.eval()
            
            if self.print_info:
                print(f"🎯 对齐模型加载成功 | modalities={self.alignment_modalities}")
                print(f"   加载的权重数量: {len(filtered_state_dict)}")
                if missing:
                    print(f"   缺失的权重: {len(missing)}")
                if unexpected:
                    print(f"   多余的权重: {len(unexpected)}")
        else:
            if self.print_info:
                print("⚠️ 没有指定align_channels，对齐模型将不会被使用")
            self.alignment_model = None

    def _filter_missing_data(self):
        """过滤掉没有指定channels数据的样本"""
        if not self.print_info:
            return
            
        print(f"🔍 检查数据完整性，channels: {self.channels}")
        
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
                                assert False, f"⚠️ Channel {channel} 格式错误"
                        
                        if missing_channels:
                            missing_count += 1
                            if missing_count <= 5:
                                print(f"  ⚠️  {case_id}: 缺少 {missing_channels}")
                        else:
                            valid_cases.append(case_id)
                except Exception as e:
                    missing_count += 1
                    if missing_count <= 5:
                        print(f"  ❌ {case_id}: 读取失败 - {e}")
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"  ❌ {case_id}: 文件不存在")
        
        # 更新数据
        original_count = len(self.case_ids)
        self.case_ids = valid_cases
        
        if self.print_info:
            new_count = len(self.case_ids)
            print(f"📊 数据过滤结果: {original_count} -> {new_count} ({new_count/original_count*100:.1f}%)")

    def _print_summary(self):
        """打印数据集摘要"""
        print(f"📊 数据集摘要:")
        print(f"  总样本数: {len(self.case_ids)}")
        print(f"  加载channels: {self.channels}")
        print(f"  对齐channels: {self.align_channels}")
        print(f"  使用对齐: {True if self.alignment_model is not None else False}")
        
        # 标签统计
        labels = [self.case_to_label[case_id] for case_id in self.case_ids]
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本，返回字典格式
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: {
                "channel_name": embeddings,
                "aligned_channel_name": embeddings,
                ...
            }
        """
        case_id = self.case_ids[idx]
        file_path = self.case_to_file[case_id]
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            return self._get_fallback_data()
        
        # 获取文件锁，确保同一文件不会被多个线程同时访问
        file_lock = self._get_file_lock(file_path)
        
        try:
            with file_lock:  # 使用文件锁保护HDF5文件访问
                with h5py.File(file_path, 'r') as hdf5_file:
                    # 加载所有指定的channels
                    channel_data = {}
                    for channel in self.channels:
                        data = self._read_with_retry(hdf5_file, channel, max_retries=3)
                        if data is not None:
                            channel_data[channel] = torch.from_numpy(self._standardize_array(data))
                        else:
                            assert False, f"⚠️ Channel {channel} 读取失败"
                
                # 如果需要对齐且对齐模型可用
                if self.alignment_model is not None and self.align_channels:
                    try:
                        # 准备对齐数据，使用align_channels的映射
                        align_data = {}
                        for channel, modality_name in self.align_channels.items():
                            if channel in channel_data:
                                align_data[modality_name] = channel_data[channel]
                        
                        if align_data:
                            with torch.no_grad():
                                aligned_features = self.alignment_model(align_data)
                            
                            # 将对齐后的特征添加到结果中
                            for modality_name, aligned_feat in aligned_features.items():
                                if isinstance(aligned_feat, torch.Tensor) and aligned_feat.numel() > 0:
                                    # 找到对应的原始channel
                                    original_channel = None
                                    for ch, mod in self.align_channels.items():
                                        if mod == modality_name:
                                            original_channel = ch
                                            break
                                    
                                    if original_channel:
                                        # 添加对齐后的特征，使用 "aligned_" 前缀
                                        aligned_key = f"aligned_{original_channel}"
                                        channel_data[aligned_key] = aligned_feat
                            
                            if self.print_info and idx == 0:  # 只在第一个样本时打印
                                print(f"🎯 对齐完成，生成了 {len(aligned_features)} 个对齐特征")
                    
                    except Exception as e:
                        if self.print_info:
                            print(f"⚠️ 对齐失败: {e}")
                
                # 获取标签并转换为数字
                label_str = self.case_to_label[case_id]
                label = torch.tensor(self.label_to_int[label_str], dtype=torch.long)
                return channel_data, label
                
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            raise e

    def _standardize_array(self, arr) -> np.ndarray:
        """标准化数组为二维格式"""
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
        带重试机制的数据读取，解决HDF5并发访问问题
        
        Args:
            hdf5_file: HDF5文件对象
            channel: 数据集名称
            max_retries: 最大重试次数
            
        Returns:
            读取的数据数组
        """
        channel = channel.split('=')
        for attempt in range(max_retries + 1):
            try:
                # 尝试读取数据
                if len(channel) == 2:
                    # WSI
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
                    # 最后一次尝试失败，抛出异常
                    print(f"❌ 读取 {channel} 最终失败: {e}")
                    raise e
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """
        获取文件锁，确保同一文件不会被多个线程同时访问
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件对应的锁对象
        """
        with _lock_dict_lock:
            if file_path not in _file_locks:
                _file_locks[file_path] = threading.Lock()
            return _file_locks[file_path]

    def get_label(self, idx: int) -> str:
        """获取指定索引的标签"""
        case_id = self.case_ids[idx]
        return self.case_to_label[case_id]

    def split_by_ids(self, id_groups: Dict[str, List[str]]) -> Dict[str, 'MultimodalDatasetView']:
        """
        根据case_id列表切分数据集
        
        Args:
            id_groups: {'train': [case_id1, ...], 'val': [...], 'test': [...]}
            
        Returns:
            {'train': FlexibleMultimodalDatasetView, 'val': ..., 'test': ...}
        """
        splits = {}
        for split_name, case_ids in id_groups.items():
            splits[split_name] = MultimodalDatasetView(
                parent=self,
                case_ids=case_ids
            )
        return splits


class MultimodalDatasetView(Dataset):
    """
    数据集子视图
    """
    
    def __init__(self, parent: MultimodalDataset, case_ids: List[str]):
        """
        初始化子视图
        
        Args:
            parent: 父数据集
            case_ids: 子集的case_id列表
        """
        self.parent = parent
        # 过滤出存在的case_id
        self.case_ids = [case_id for case_id in case_ids if case_id in parent.case_to_file]
        
        # 继承父数据集的属性
        self.channels = parent.channels
        self.align_channels = parent.align_channels
        self.alignment_model = parent.alignment_model
        self.alignment_modalities = getattr(parent, 'alignment_modalities', [])

    def __len__(self) -> int:
        """返回子视图大小"""
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取子视图中的样本"""
        case_id = self.case_ids[idx]
        file_path = self.parent.case_to_file[case_id]
        
        if not os.path.exists(file_path):
            return self.parent._get_fallback_data()
        
        try:
            with h5py.File(file_path, 'r') as hdf5_file:
                # 加载所有指定的channels
                channel_data = {}
                for channel in self.channels:
                    if channel in hdf5_file:
                        data = hdf5_file[channel][:]
                        channel_data[channel] = torch.from_numpy(self.parent._standardize_array(data))
                    else:
                        channel_data[channel] = torch.zeros((0, 0), dtype=torch.float32)
                
                # 如果需要对齐且对齐模型可用
                if self.alignment_model is not None and self.align_channels:
                    try:
                        # 准备对齐数据，使用align_channels的映射
                        align_data = {}
                        for channel, modality_name in self.align_channels.items():
                            if channel in channel_data:
                                align_data[modality_name] = channel_data[channel]
                        
                        if align_data:
                            with torch.no_grad():
                                aligned_features = self.alignment_model(align_data)
                            
                            # 将对齐后的特征添加到结果中
                            for modality_name, aligned_feat in aligned_features.items():
                                if isinstance(aligned_feat, torch.Tensor) and aligned_feat.numel() > 0:
                                    # 找到对应的原始channel
                                    original_channel = None
                                    for ch, mod in self.align_channels.items():
                                        if mod == modality_name:
                                            original_channel = ch
                                            break
                                    
                                    if original_channel:
                                        # 添加对齐后的特征，使用 "aligned_" 前缀
                                        aligned_key = f"aligned_{original_channel}"
                                        channel_data[aligned_key] = aligned_feat
                    
                    except Exception:
                        pass  # 对齐失败时忽略
                
                return channel_data
                
        except Exception:
            return self.parent._get_fallback_data()

    def get_label(self, idx: int) -> str:
        """获取指定索引的标签"""
        case_id = self.case_ids[idx]
        return self.parent.case_to_label[case_id]


def create_multimodal_dataset(
    csv_path: str,
    channels: List[str],
    align_channels: Dict[str, str] = None,
    alignment_model_path: Optional[str] = None,
    device: str = 'auto',
    print_info: bool = True
) -> MultimodalDataset:
    """
    创建灵活多模态数据集的便捷函数
    
    Args:
        csv_path: CSV文件路径
        channels: 需要加载的channel列表（必需）
        align_channels: 需要对齐的channel映射，如 {"tma_CD3": "CD3", "tma_CD8": "CD8"}
        alignment_model_path: 对齐模型路径
        print_info: 是否打印信息
    
    Returns:
        FlexibleMultimodalDataset实例
    """
    return MultimodalDataset(
        csv_path=csv_path,
        channels=channels,
        align_channels=align_channels,
        alignment_model_path=alignment_model_path,
        device=device,
        print_info=print_info
    )

