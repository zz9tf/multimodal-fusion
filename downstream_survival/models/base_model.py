"""
Base model class
Define unified model interface and return format
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class BaseModel(nn.Module, ABC):
    """
    Base model abstract class

    Define unified model interface, all models should inherit from this class and implement unified forward method
    Unified return format facilitates unified processing of training loops
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model

        Args:
            config: Model configuration dictionary
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
            raise ValueError(f"Unsupported base loss function: {config.get('base_loss_fn')}")
    
    @abstractmethod
    def forward(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Any]:
        """
        Unified forward propagation interface

        Args:
            input_data: Input data, can be:
                - torch.Tensor: Single-modal features [N, D]
                - Dict[str, torch.Tensor]: Multimodal data dictionary, e.g., {"features": tensor, "aligned_features": tensor}
            **kwargs: Other parameters, support:
                - label: Labels (for instance evaluation)
                - instance_eval: Whether to perform instance evaluation
                - return_features: Whether to return features
                - attention_only: Whether to return only attention weights
                - Other model-specific parameters

        Returns:
            Dict[str, Any]: Unified result dictionary, containing the following keys:
                - 'logits': Model output logits [1, n_classes] or [N, n_classes]
                - 'probabilities': Prediction probabilities [1, n_classes] or [N, n_classes]
                - 'predictions': Predicted classes [1] or [N]
                - 'attention_weights': Attention weights (if applicable) [1, N] or [n_classes, N]
                - 'features': Feature representations (if return_features=True)
                - 'additional_loss': Additional loss values (if additional loss is calculated)
                - Other model-specific outputs
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dict[str, Any]: Model information dictionary
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
        Process input data, convert multimodal data to unified tensor format

        Args:
            input_data: Input data, can be tensor or dictionary

        Returns:
            torch.Tensor: Processed feature tensor [N, D]
        """
        pass
    
    def _create_result_dict(self,
                          logits: torch.Tensor,
                          probabilities: torch.Tensor,
                          predictions: torch.Tensor,
                          **kwargs) -> Dict[str, Any]:
        """
        Create unified result dictionary

        Args:
            logits: Model output logits [1, n_classes] or [N, n_classes]
            probabilities: Prediction probabilities [1, n_classes] or [N, n_classes]
            predictions: Predicted classes [1] or [N]
            **kwargs: Other outputs, such as:
                - attention_weights: Attention weights [1, N] or [n_classes, N]
                - features: Feature representations [1, D] or [N, D]
                - additional_loss: Additional loss values [1] or [N]
                - Other model-specific outputs

        Returns:
            Dict[str, Any]: Unified format result dictionary
        """
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        # 添加所有其他输出
        # 🔒 Ensure deterministic dictionary key order: use sorted() to sort keys
        for key, value in sorted(kwargs.items()):
            if value is not None:
                result[key] = value
        
        return result
    
    @abstractmethod
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate loss

        Args:
            logits: Predicted logits [N, C]
            labels: True labels [N]
            result: Result dictionary, containing the following keys:
        """
        pass
