"""
基础模型类
定义统一的模型接口和返回格式
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class BaseModel(nn.Module, ABC):
    """
    基础模型抽象类
    
    定义统一的模型接口，所有模型都应该继承此类并实现统一的forward方法
    统一的返回格式便于训练循环的统一处理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础模型
        
        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = config.get('input_dim', 1024)
        self.dropout = config.get('dropout', 0.25)
        self.n_classes = config.get('n_classes', 2)
        if config.get('base_loss_fn') is None or config.get('base_loss_fn') == 'ce':
            self.base_loss_fn = nn.CrossEntropyLoss()
        elif config.get('base_loss_fn') == 'svm':
            self.base_loss_fn = nn.SmoothTop1SVM(n_classes=self.n_classes)
        else:
            raise ValueError(f"不支持的base损失函数: {config.get('base_loss_fn')}")
    
    @abstractmethod
    def forward(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Any]:
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典，如 {"features": tensor, "aligned_features": tensor}
            **kwargs: 其他参数，支持：
                - label: 标签（用于实例评估）
                - instance_eval: 是否进行实例评估
                - return_features: 是否返回特征
                - attention_only: 是否只返回注意力权重
                - 其他模型特定参数
                
        Returns:
            Dict[str, Any]: 统一的结果字典，包含以下键：
                - 'logits': 模型输出logits [1, n_classes] 或 [N, n_classes]
                - 'probabilities': 预测概率 [1, n_classes] 或 [N, n_classes]  
                - 'predictions': 预测类别 [1] 或 [N]
                - 'attention_weights': 注意力权重（如果适用）[1, N] 或 [n_classes, N]
                - 'features': 特征表示（如果return_features=True）
                - 'additional_loss': 额外损失值（如果计算了额外损失）
                - 其他模型特定输出
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'dropout': self.dropout,
            'n_classes': self.n_classes,
            'base_loss_fn': self.base_loss_fn,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    @abstractmethod
    def _process_input_data(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        处理输入数据，将多模态数据转换为统一的张量格式
        
        Args:
            input_data: 输入数据，可以是张量或字典
            
        Returns:
            torch.Tensor: 处理后的特征张量 [N, D]
        """
        pass
    
    def _create_result_dict(self, 
                          logits: torch.Tensor,
                          probabilities: torch.Tensor, 
                          predictions: torch.Tensor,
                          **kwargs) -> Dict[str, Any]:
        """
        创建统一的结果字典
        
        Args:
            logits: 模型输出logits [1, n_classes] 或 [N, n_classes]
            probabilities: 预测概率 [1, n_classes] 或 [N, n_classes]
            predictions: 预测类别 [1] 或 [N]
            **kwargs: 其他输出，如：
                - attention_weights: 注意力权重 [1, N] 或 [n_classes, N]
                - features: 特征表示 [1, D] 或 [N, D]
                - additional_loss: 额外损失值 [1] 或 [N]
                - 其他模型特定输出
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        # 添加所有其他输出
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value
        
        return result
    
    @abstractmethod
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        计算损失
        
        Args:
            logits: 预测的logits [N, C]
            labels: 真实标签 [N]
            result: 结果字典，包含以下键：
        """
        pass
