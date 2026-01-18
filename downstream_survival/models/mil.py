import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_model import BaseModel

"""
MIL (Multiple Instance Learning) model
Pure MIL mode, does not use attention mechanism
"""

class MIL_fc(BaseModel):
    """
    MIL model (supports binary and multi-classification)

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: Dropout rate
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Validate configuration completeness
        self._validate_config(config)
        
        # 从配置中提取参数
        self.channels_used_in_model = config['channels_used_in_model']
        self.model_size = config['model_size']
        
        # Model size configuration
        self.size_dict = {
            "small": [self.input_dim, 512], 
            "big": [self.input_dim, 512], 
            "128*64": [self.input_dim, 128], 
            "64*32": [self.input_dim, 64], 
            "32*16": [self.input_dim, 32],
            "16*8": [self.input_dim, 16],
            "8*4": [self.input_dim, 8],
            "4*2": [self.input_dim, 4],
            "2*1": [self.input_dim, 2]
        }
        
        size = self.size_dict[self.model_size]
        
        # Build feature extraction layer
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
        self.fc = nn.Sequential(*fc)
        
        # Build classifier
        self.classifier = nn.Linear(size[1], self.n_classes)
    
    def _validate_config(self, config):
        """Validate configuration completeness"""
        required_params = ['n_classes', 'input_dim', 'model_size', 'dropout']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"MIL_fc configuration missing required parameters: {missing_params}")

        # Validate number of classes
        if config['n_classes'] < 2:
            raise ValueError(f"Number of classes must be >= 2, current: {config['n_classes']}")

        # Validate model size
        valid_sizes = ["small", "big", "128*64", "64*32", "32*16", "16*8", "8*4", "4*2", "2*1"]
        if config['model_size'] not in valid_sizes:
            raise ValueError(f"Unsupported model size: {config['model_size']}, supported sizes: {valid_sizes}")

        # Validate input dimension
        if config['input_dim'] <= 0:
            raise ValueError(f"Input dimension must be > 0, current: {config['input_dim']}")

        # Validate dropout rate
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"Dropout rate must be in [0,1] range, current: {config['dropout']}")
    
    def _process_input_data(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        MIL model's multimodal data fusion strategy

        Args:
            multimodal_data: Multimodal data dictionary, e.g. {"features": tensor, "aligned_features": tensor}

        Returns:
            torch.Tensor: Fused feature tensor
        """
        return torch.cat([multimodal_data[channel] for channel in self.channels_used_in_model], dim=1).squeeze(0)
    
    def forward(self, input_data, label):
        """
        Forward propagation - following the original simple approach
        """
        h = self._process_input_data(input_data)
        h = self.fc(h)
        logits = self.classifier(h)  # K x n_classes
        
        y_probs = F.softmax(logits, dim=1)
        if self.n_classes == 2:
            # Binary classification: select instance with highest positive class probability
            top_instance_idx = torch.topk(y_probs[:, 1], 1, dim=0)[1].view(1,)
            selected_logits = torch.index_select(logits, dim=0, index=top_instance_idx)
            Y_prob = torch.index_select(y_probs, dim=0, index=top_instance_idx)
            Y_hat = torch.topk(selected_logits, 1, dim=1)[1]
        else:
            # Multi-classification: select instance with highest global probability
            m = y_probs.view(1, -1).argmax(1)
            top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
            selected_logits = logits[top_indices[0]:top_indices[0]+1]
            Y_prob = y_probs[top_indices[0]:top_indices[0]+1]
            Y_hat = top_indices[1]
        
        return self._create_result_dict(
            logits=selected_logits,
            probabilities=Y_prob,
            predictions=Y_hat,
        )
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate loss
        """
        return self.base_loss_fn(logits, labels)