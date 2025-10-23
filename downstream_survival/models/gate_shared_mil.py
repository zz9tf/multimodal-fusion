import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Dict

class PositiveSwish(nn.Module):
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * torch.sigmoid(x) + self.c

class GateSharedMIL(BaseModel):
    """
    SharedGatedGTECLAM 模型
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.model_size = config.get('model_size', 'small')
        self.channels_used_in_model = config.get('channels_used_in_model', [])
        self.confidence_weight = config.get('confidence_weight', 1)
        self.feature_weight_weight = config.get('feature_weight_weight', 1)
        
        self.size_dict = {
            "small": [self.input_dim, 512, 256], 
            "big": [self.input_dim, 512, 384], 
            "128*64": [self.input_dim, 128, 64], 
            "64*32": [self.input_dim, 64, 32], 
            "32*16": [self.input_dim, 32, 16],
            "16*8": [self.input_dim, 16, 8],
            "8*4": [self.input_dim, 8, 4],
            "4*2": [self.input_dim, 4, 2],
            "2*1": [self.input_dim, 2, 1]
        }
        
        size = self.size_dict[self.model_size]
        
        self.ChannelFeatureWeightorCreator = lambda: nn.Sequential(nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid())
        self.SampleAttCreator = lambda: nn.Sequential(nn.Linear(self.input_dim, size[1]), nn.Linear(size[1], size[2]), nn.Linear(size[2], 1), nn.Dropout(self.dropout), nn.Softmax(dim=1))
        self.TCPClassifierCreator = lambda: nn.Sequential(
            nn.Linear(self.input_dim, size[1]), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(size[1], size[2]), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(size[2], self.n_classes)
        )
        self.TCPConfidenceLayerCreator = lambda: nn.Sequential(nn.Linear(self.input_dim, size[1]), nn.Linear(size[1], size[2]), nn.Linear(size[2], 1), nn.Dropout(self.dropout), PositiveSwish())
        
        self.ChannelFeatureWeightor = self.ChannelFeatureWeightorCreator()
        self.SampleAtt = self.SampleAttCreator()
        self.TCPClassifier = self.TCPClassifierCreator()
        self.TCPConfidenceLayer = self.TCPConfidenceLayerCreator()
        self.classifiers = nn.Sequential(
            nn.Linear(len(self.channels_used_in_model)*self.input_dim, self.input_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_dim, size[1]), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(size[1], size[2]), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(size[2], self.n_classes)  # 最后一层无激活函数
        )
        
        self.TCPLogitsLoss_fn = nn.CrossEntropyLoss(reduction='none')
        self.TCPConfidenceLoss_fn = nn.MSELoss(reduction='none')
        
        self.TCPLogits = dict()
        self.TCPConfidence = dict()
        self.FeatureWeight = dict()
        
    def _positive_swish(self, x, c=0.3):
        return x * torch.sigmoid(x) + c
    
    def _validate_config(self, config):
        """
        验证配置
        """
        if 'confidence_weight' not in config:
            raise ValueError("配置中缺少 'confidence_weight' 参数")
        if 'feature_weight_weight' not in config:
            raise ValueError("配置中缺少 'feature_weight_weight' 参数")
        if 'channels_used_in_model' not in config:
            raise ValueError("配置中缺少 'channels_used_in_model' 参数")
        if 'model_size' not in config:
            raise ValueError("配置中缺少 'model_size' 参数")
        
    def _process_input_data(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        处理输入数据，将多模态数据转换为统一的张量格式
        """
        input_features = {channel: input_data[channel].squeeze(0) for channel in self.channels_used_in_model} # [channel, simple_number, simple_dim]
        return input_features
    
    def forward(self, input_data, label):
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典
            **kwargs: 其他参数，支持：
                - label: 标签（用于实例评估）
                - instance_eval: 是否进行实例评估
                - return_features: 是否返回特征
                - attention_only: 是否只返回注意力权重
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        # 处理输入数据（支持多模态）
        input_features = self._process_input_data(input_data)
        
        result_kwargs = dict()
        result_kwargs['feature_weight_loss'] = 0
        result_kwargs['confidence_logits_loss'] = 0
        result_kwargs['confidence_loss'] = 0
        
        conf_h = torch.zeros(1, len(self.channels_used_in_model)*self.input_dim, device=self.device)
        for i, channel in enumerate(input_features):
            # [N, D] -> [N, D]
            self.FeatureWeight[channel] = self.ChannelFeatureWeightor(input_features[channel])
            input_features[channel] = self.FeatureWeight[channel] * input_features[channel]
            # [N, D] -> [N, 1]
            A = self.SampleAtt(input_features[channel]).T
            # [1, N]*[N, D] -> [1, D]
            # MIL: combined features
            h = torch.mm(A, input_features[channel])
            # [1, D] -> [1, n_classes]
            self.TCPLogits[channel] = self.TCPClassifier(h)
            # [1, D] -> [1, 1]
            self.TCPConfidence[channel] = self.TCPConfidenceLayer(h)
            
            # Confidence weighted combined features
            # input_features[channel]: [1, D]
            conf_h[:, i*self.input_dim:(i+1)*self.input_dim] = h * self.TCPConfidence[channel]
            result_kwargs['feature_weight_loss'] += torch.mean(self.FeatureWeight[channel])
            # pred: [1, n_classes]
            pred = F.softmax(self.TCPLogits[channel], dim = 1)
            # p_target: [1]
            p_target = torch.gather(pred, 1, label.unsqueeze(1)).view(-1)
            # confidence -> TCPLogits & TCPLogits -> labels
            logits_loss = torch.mean(self.TCPLogitsLoss_fn(self.TCPLogits[channel], label))
            confidence_loss = torch.mean(self.TCPConfidenceLoss_fn(self.TCPConfidence[channel].view(-1), p_target))
            
            result_kwargs['confidence_logits_loss'] += logits_loss
            result_kwargs['confidence_loss'] += confidence_loss
            
        result_kwargs['feature_weight_loss'] /= len(self.channels_used_in_model)
        result_kwargs['confidence_logits_loss'] /= len(self.channels_used_in_model)
        result_kwargs['confidence_loss'] /= len(self.channels_used_in_model)
        
        # [1, D*number_of_modalities] -> [1, n_classes]
        logits = self.classifiers(conf_h) # [1, n_classes]
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        
        # 构建统一的结果字典
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
        
    def verbose_items(self, result: Dict[str, float]) -> list:
        """
        返回可打印的指标
        
        Returns:
            list: 包含 (key, value) 元组的列表，用于训练日志打印
        """
        return [
            ('base', result['base_loss']),
            ('conf', result['confidence_loss']*self.confidence_weight),
            ('conf_logits', result['confidence_logits_loss']*self.confidence_weight),
            ('featW', result['feature_weight_loss'])
        ]
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        计算损失
        """
        result['base_loss'] = self.base_loss_fn(logits, labels)
        result['confidence_total_loss'] = (result['confidence_loss'] + result['confidence_logits_loss'])*self.confidence_weight
        return result['base_loss'] + result['confidence_total_loss'] + result['feature_weight_loss']
