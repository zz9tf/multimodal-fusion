import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate_shared_mil import GateSharedMIL

class GateMIL(GateSharedMIL):
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
        
        self.ChannelFeatureWeightor = nn.ModuleDict({channel: self.ChannelFeatureWeightorCreator() for channel in self.channels_used_in_model})
        # SampleAtt: [channel, simple_number, simple_dim] -> weight [channel, simple_number, 1]
        self.SampleAtt = nn.ModuleDict({channel: self.SampleAttCreator() for channel in self.channels_used_in_model})
        self.TCPClassifier = nn.ModuleDict({channel: self.TCPClassifierCreator() for channel in self.channels_used_in_model})
        self.TCPConfidenceLayer = nn.ModuleDict({channel: self.TCPConfidenceLayerCreator() for channel in self.channels_used_in_model})
    
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
            self.FeatureWeight[channel] = self.ChannelFeatureWeightor[channel](input_features[channel])
            input_features[channel] = self.FeatureWeight[channel] * input_features[channel]
            # [N, D] -> [N, 1]
            A = self.SampleAtt[channel](input_features[channel]).T
            # [1, N]*[N, D] -> [1, D]
            # MIL: combined features
            h = torch.mm(A, input_features[channel])
            # [1, D] -> [1, n_classes]
            self.TCPLogits[channel] = self.TCPClassifier[channel](h)
            # [1, D] -> [1, 1]
            self.TCPConfidence[channel] = self.TCPConfidenceLayer[channel](h)
            
            # Confidence weighted combined features
            # input_features[channel]: [1, D]
            input_features[channel] = h * self.TCPConfidence[channel]
            conf_h[:, i*self.input_dim:(i+1)*self.input_dim] = input_features[channel]*self.TCPConfidence[channel]
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
