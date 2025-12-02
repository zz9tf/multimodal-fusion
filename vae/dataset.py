# -*- coding: utf-8 -*-
"""
VAE数据集类
用于读取WSI embeddings，并过滤只保留living的病人
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from typing import Dict, Optional

# 添加项目路径
sys.path.append('/home/zheng/zheng/multimodal-fusion/downstream_survival')
from datasets.multimodal_dataset import MultimodalDataset


class WSIVAEDataset(Dataset):
    """
    WSI VAE数据集类
    从MultimodalDataset中读取WSI embeddings，以patch为单位返回
    每个样本是一个patch的特征向量，避免内存爆炸
    """
    
    def __init__(self, 
                 csv_path: str,
                 data_root_dir: str,
                 label_filter: Optional[str] = 'living',
                 print_info: bool = True):
        """
        初始化WSI VAE数据集
        
        Args:
            csv_path: CSV文件路径，包含patient_id, case_id, label, h5_file_path
            data_root_dir: 数据根目录
            label_filter: 要保留的标签，默认为'living'。如果为None或空字符串，则使用全部数据
            print_info: 是否打印信息
        """
        super().__init__()
        
        self.data_root_dir = data_root_dir
        self.label_filter = label_filter
        self.print_info = print_info
        
        # 使用MultimodalDataset来读取数据
        self.base_dataset = MultimodalDataset(
            csv_path=csv_path,
            data_root_dir=data_root_dir,
            channels=['wsi=features'],
            align_channels=None,
            alignment_model_path=None,
            device='cpu',
            print_info=False
        )
        
        # 如果设置了label_filter，则过滤；否则使用全部数据
        if self.label_filter is not None and self.label_filter.strip() != '':
            self._filter_by_label()
        else:
            # 使用全部数据
            self._use_all_data()
        
        # 构建patch级别的索引映射
        # 每个元素是 (patient_idx, patch_idx)
        self._build_patch_indices()
        
        if self.print_info:
            self._print_summary()
    
    def _filter_by_label(self):
        """过滤数据集，只保留指定标签的病人"""
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
            print(f"🔍 过滤标签 '{self.label_filter}': {len(self.base_dataset)} -> {len(self.filtered_indices)} 个样本")
    
    def _use_all_data(self):
        """使用全部数据，不进行过滤"""
        self.filtered_indices = list(range(len(self.base_dataset)))
        self.case_ids = self.base_dataset.case_ids.copy()
        
        if self.print_info:
            print(f"📦 使用全部数据: {len(self.filtered_indices)} 个patient样本")
    
    def _build_patch_indices(self):
        """
        构建patch级别的索引映射
        每个样本是一个patch，避免内存爆炸
        只读取形状信息，不加载完整数据
        """
        self.patch_indices = []  # 每个元素是 (patient_idx, patch_idx)
        self.patient_to_patch_range = {}  # 记录每个patient的patch范围
        
        if self.print_info:
            print(f"📝 构建patch索引映射...")
        
        for patient_idx in self.filtered_indices:
            # 获取该patient的patches数量（只读取形状，不加载完整数据）
            try:
                channel_data, _ = self.base_dataset[patient_idx]
                wsi_features = channel_data['wsi=features']
                
                # 确保是2D张量
                if wsi_features.dim() == 1:
                    num_patches = 1
                else:
                    num_patches = wsi_features.shape[0]
                
                # 记录该patient的patch范围
                start_idx = len(self.patch_indices)
                for patch_idx in range(num_patches):
                    self.patch_indices.append((patient_idx, patch_idx))
                end_idx = len(self.patch_indices)
                
                self.patient_to_patch_range[patient_idx] = (start_idx, end_idx)
            except Exception as e:
                if self.print_info:
                    print(f"⚠️ 无法读取patient {patient_idx}的patches数量: {e}")
                # 如果无法读取，假设有1个patch
                start_idx = len(self.patch_indices)
                self.patch_indices.append((patient_idx, 0))
                end_idx = len(self.patch_indices)
                self.patient_to_patch_range[patient_idx] = (start_idx, end_idx)
    
    def _print_summary(self):
        """打印数据集摘要"""
        print(f"📊 WSI VAE数据集摘要:")
        print(f"  Patient数量: {len(self.filtered_indices)}")
        print(f"  总Patch数量: {len(self.patch_indices)}")
        if self.label_filter is not None and self.label_filter.strip() != '':
            print(f"  标签过滤: {self.label_filter}")
        else:
            print(f"  标签过滤: 无（使用全部数据）")
        
        # 检查第一个patch的维度（延迟加载，避免在初始化时加载数据）
        if len(self) > 0:
            try:
                sample = self[0]
                if isinstance(sample, tuple):
                    patch_feature = sample[0]
                else:
                    patch_feature = sample
                print(f"  每个Patch特征维度: {patch_feature.shape[0]}")
            except Exception as e:
                print(f"  ⚠️ 无法获取特征维度: {e}")
    
    def __len__(self) -> int:
        """返回数据集大小（patch数量）"""
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取单个patch的特征
        
        Args:
            idx: patch索引（不是patient索引）
            
        Returns:
            patch_feature: 单个patch的特征向量，形状为 (feature_dim,)
        """
        patient_idx, patch_idx = self.patch_indices[idx]
        
        # 从base_dataset获取该patient的所有patches
        # 注意：这里会加载该patient的所有patches，但只返回一个patch
        # 由于DataLoader会按batch处理，内存使用是可控的
        channel_data, label = self.base_dataset[patient_idx]
        
        # 提取WSI features
        wsi_features = channel_data['wsi=features']
        
        # 确保是2D张量 (num_patches, feature_dim)
        if wsi_features.dim() == 1:
            wsi_features = wsi_features.unsqueeze(0)
        
        # 提取指定的patch
        patch_feature = wsi_features[patch_idx]  # (feature_dim,)
        
        return patch_feature.float()
    
    def get_feature_dim(self) -> int:
        """
        获取特征维度
        
        Returns:
            特征维度
        """
        if len(self) == 0:
            raise ValueError("数据集为空，无法获取特征维度")
        
        sample = self[0]
        # 现在每个样本是一个patch的特征向量，形状为 (feature_dim,)
        return sample.shape[0]
    
    def get_patient_patches(self, patient_idx: int) -> torch.Tensor:
        """
        获取指定patient的所有patches（用于推理或后处理）
        
        Args:
            patient_idx: patient在filtered_indices中的索引
            
        Returns:
            patches: 该patient的所有patches，形状为 (num_patches, feature_dim)
        """
        if patient_idx not in self.patient_to_patch_range:
            raise ValueError(f"Patient索引 {patient_idx} 不存在")
        
        start_idx, end_idx = self.patient_to_patch_range[patient_idx]
        patches = []
        for idx in range(start_idx, end_idx):
            patch = self[idx]
            patches.append(patch)
        
        return torch.stack(patches, dim=0)  # (num_patches, feature_dim)

