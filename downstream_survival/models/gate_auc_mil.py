import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate_shared_mil import GateSharedMIL
from typing import Dict, List, Tuple
from libauc.losses import AUCMLoss

class GateAUCMIL(GateSharedMIL):
    """
    GatedGTECLAM 模型
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - gate: 是否使用门控注意力
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    - shared_gated: 是否共享门控注意力
    - use_auc_loss: 是否使用AUC损失
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.auc_loss = AUCMLoss(margin=1.0)
        self.auc_loss_weight = config.get('auc_loss_weight', 1.0)
        
        self.ChannelFeatureWeightor = nn.ModuleDict({channel: self.ChannelFeatureWeightorCreator() for channel in self.channels_used_in_model})
        # SampleAtt: [channel, simple_number, simple_dim] -> weight [channel, simple_number, 1]
        self.SampleAtt = nn.ModuleDict({channel: self.SampleAttCreator() for channel in self.channels_used_in_model})
        self.TCPClassifier = nn.ModuleDict({channel: self.TCPClassifierCreator() for channel in self.channels_used_in_model})
        self.TCPConfidenceLayer = nn.ModuleDict({channel: self.TCPConfidenceLayerCreator() for channel in self.channels_used_in_model})
    
    def forward(self, input_data, label=None, return_features=False, **kwargs):
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典
            label: 标签（用于实例评估，可选）
            return_features: 是否返回特征信息
            **kwargs: 其他参数，支持：
                - instance_eval: 是否进行实例评估
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        # 处理输入数据（支持多模态）
        input_features = self._process_input_data(input_data)
        
        result_kwargs = dict()
        result_kwargs['feature_weight_loss'] = 0
        result_kwargs['confidence_logits_loss'] = 0
        result_kwargs['confidence_loss'] = 0
        
        # 初始化feature存储字典（如果需要返回features）
        if return_features:
            result_kwargs['raw_features'] = {}  # 原始输入特征
            result_kwargs['weighted_features'] = {}  # 特征权重加权后的特征
            result_kwargs['attention_weights'] = {}  # 注意力权重
            result_kwargs['combined_features'] = {}  # MIL聚合后的特征
            result_kwargs['confidence_scores'] = {}  # 置信度分数
            result_kwargs['feature_weights'] = {}  # 特征权重
        
        conf_h = torch.zeros(1, len(self.channels_used_in_model)*self.input_dim, device=self.device)
        for i, channel in enumerate(input_features):
            # 保存原始特征（如果需要）
            if return_features or attention_only:
                result_kwargs['raw_features'][channel] = input_features[channel].clone()
            
            # [N, D] -> [N, D]
            self.FeatureWeight[channel] = self.ChannelFeatureWeightor[channel](input_features[channel])
            input_features[channel] = self.FeatureWeight[channel] * input_features[channel]
            
            # 保存加权特征和特征权重（如果需要）
            if return_features or attention_only:
                result_kwargs['weighted_features'][channel] = input_features[channel].clone()
                result_kwargs['feature_weights'][channel] = self.FeatureWeight[channel].clone()
            
            # [N, D] -> [N, 1]
            A = self.SampleAtt[channel](input_features[channel]).T
            
            # 保存注意力权重（如果需要）
            if return_features:
                result_kwargs['attention_weights'][channel] = A.clone()
            
            # [1, N]*[N, D] -> [1, D]
            # MIL: combined features
            h = torch.mm(A, input_features[channel])
            
            # 保存聚合特征（如果需要）
            if return_features:
                result_kwargs['combined_features'][channel] = h.clone()
            
            # [1, D] -> [1, n_classes]
            self.TCPLogits[channel] = self.TCPClassifier[channel](h)
            # [1, D] -> [1, 1]
            self.TCPConfidence[channel] = self.TCPConfidenceLayer[channel](h)
            
            # 保存置信度分数（如果需要）
            if return_features:
                result_kwargs['confidence_scores'][channel] = self.TCPConfidence[channel].clone()
            
            # Confidence weighted combined features
            # input_features[channel]: [1, D]
            input_features[channel] = h * self.TCPConfidence[channel]
            conf_h[:, i*self.input_dim:(i+1)*self.input_dim] = input_features[channel]*self.TCPConfidence[channel]
            result_kwargs['feature_weight_loss'] += torch.mean(self.FeatureWeight[channel])
            
            # 只有在有label时才计算损失
            if label is not None:
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
        if label is not None:
            result_kwargs['confidence_logits_loss'] /= len(self.channels_used_in_model)
            result_kwargs['confidence_loss'] /= len(self.channels_used_in_model)
        
        logits = self.classifiers(conf_h) # [1, n_classes]
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        # 如果只需要注意力权重，直接返回
        if attention_only:
            return {
                'attention_weights': result_kwargs['attention_weights'],
                'feature_weights': result_kwargs['feature_weights'],
                'confidence_scores': result_kwargs['confidence_scores']
            }
        
        # 构建统一的结果字典
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        损失函数
        """
        if 'group_predictions' not in result:
            result['group_predictions'] = []
        if 'group_labels' not in result:
            result['group_labels'] = []
        result['group_predictions'].append(result['predictions'])
        result['group_labels'].append(labels)
        
        return super().loss_fn(logits, labels, result)
    
    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        计算分组损失
        """
        logits = torch.cat(self.group_logits, dim=0).to(self.device)
        targets = torch.cat(self.group_labels, dim=0).to(self.device)
        self.group_logits = []
        self.group_labels = []
        loss = self.auc_loss_fn(logits, targets).squeeze()
        if not hasattr(self, 'epoch_auc_loss'):
            self.epoch_auc_loss = []
        self.epoch_auc_loss.append(loss.item())
        return loss

    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        if 'auc_loss' not in result:
            return []
        if 'is_epoch' in result:
            epoch_auc_loss = sum(self.epoch_auc_loss)/len(self.epoch_auc_loss)
            self.epoch_auc_loss = []
            return [
                ('auc_loss', epoch_auc_loss)
            ]
        return [
            ('auc_loss', result['auc_loss'])
        ]
    
    def extract_features(self, input_data, feature_type='all', channel=None):
        """
        便捷的特征提取方法
        
        Args:
            input_data: 输入数据
            feature_type: 特征类型，可选：
                - 'raw': 原始输入特征
                - 'weighted': 特征权重加权后的特征  
                - 'attention': 注意力权重
                - 'combined': MIL聚合后的特征
                - 'confidence': 置信度分数
                - 'feature_weights': 特征权重
                - 'all': 所有特征（默认）
            channel: 指定通道，如果为None则返回所有通道
            
        Returns:
            Dict[str, torch.Tensor]: 特征字典
        """
        with torch.no_grad():
            result = self.forward(input_data, return_features=True)
            
            if feature_type == 'all':
                return result
            elif feature_type in ['raw_features', 'weighted_features', 'attention_weights', 
                                'combined_features', 'confidence_scores', 'feature_weights']:
                features = result.get(feature_type, {})
                if channel is not None and channel in features:
                    return {channel: features[channel]}
                return features
            else:
                raise ValueError(f"不支持的特征类型: {feature_type}")
    
    def get_attention_weights(self, input_data, channel=None):
        """
        获取注意力权重
        
        Args:
            input_data: 输入数据
            channel: 指定通道，如果为None则返回所有通道
            
        Returns:
            Dict[str, torch.Tensor]: 注意力权重字典
        """
        return self.extract_features(input_data, feature_type='attention_weights', channel=channel)
    
    def get_confidence_scores(self, input_data, channel=None):
        """
        获取置信度分数
        
        Args:
            input_data: 输入数据
            channel: 指定通道，如果为None则返回所有通道
            
        Returns:
            Dict[str, torch.Tensor]: 置信度分数字典
        """
        return self.extract_features(input_data, feature_type='confidence_scores', channel=channel)