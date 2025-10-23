import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base_model import BaseModel

"""
MIL (Multiple Instance Learning) 模型
纯MIL模式，不使用注意力机制
"""

class MIL_fc(BaseModel):
    """
    MIL模型（支持二分类和多分类）
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 验证配置完整性
        self._validate_config(config)
        
        # 从配置中提取参数
        self.channels_used_in_model = config['channels_used_in_model']
        self.model_size = config['model_size']
        
        # 模型大小配置
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
        
        # 构建特征提取层
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
        self.fc = nn.Sequential(*fc)
        
        # 构建分类器
        self.classifier = nn.Linear(size[1], self.n_classes)
    
    def _validate_config(self, config):
        """验证配置完整性"""
        required_params = ['n_classes', 'input_dim', 'model_size', 'dropout']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"MIL_fc配置缺少必需参数: {missing_params}")
        
        # 验证类别数量
        if config['n_classes'] < 2:
            raise ValueError(f"类别数量必须 >= 2，当前: {config['n_classes']}")
        
        # 验证模型大小
        valid_sizes = ["small", "big", "128*64", "64*32", "32*16", "16*8", "8*4", "4*2", "2*1"]
        if config['model_size'] not in valid_sizes:
            raise ValueError(f"不支持的模型大小: {config['model_size']}，支持的大小: {valid_sizes}")
        
        # 验证输入维度
        if config['input_dim'] <= 0:
            raise ValueError(f"输入维度必须 > 0，当前: {config['input_dim']}")
        
        # 验证dropout率
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"dropout率必须在[0,1]范围内，当前: {config['dropout']}")
    
    def _process_input_data(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        MIL模型的多模态数据融合策略
        
        Args:
            multimodal_data: 多模态数据字典，如 {"features": tensor, "aligned_features": tensor}
            
        Returns:
            torch.Tensor: 融合后的特征张量
        """
        return torch.cat([multimodal_data[channel] for channel in self.channels_used_in_model], dim=1).squeeze(0)
    
    def forward(self, input_data, label):
        """
        前向传播 - 按照原来的简单写法
        """
        h = self._process_input_data(input_data)
        h = self.fc(h)
        logits = self.classifier(h)  # K x n_classes
        
        y_probs = F.softmax(logits, dim=1)
        if self.n_classes == 2:
            # 二分类：选择正类概率最高的实例
            top_instance_idx = torch.topk(y_probs[:, 1], 1, dim=0)[1].view(1,)
            selected_logits = torch.index_select(logits, dim=0, index=top_instance_idx)
            Y_prob = torch.index_select(y_probs, dim=0, index=top_instance_idx)
            Y_hat = torch.topk(selected_logits, 1, dim=1)[1]
        else:
            # 多分类：选择全局概率最高的实例
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
        计算损失
        """
        return self.base_loss_fn(logits, labels)