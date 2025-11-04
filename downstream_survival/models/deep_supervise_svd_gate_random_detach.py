import torch
import torch.nn.functional as F
from .deep_supervise_svd_gate_random import DeepSuperviseSVDGateRandomClam
import random

class DeepSuperviseSVDGateRandomClamDetach(DeepSuperviseSVDGateRandomClam):
    """
    CLAM MLP 模型
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - gate: 是否使用门控注意力
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    """
    
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_data, label):
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典
            label: 标签（用于实例评估）
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        # 初始化结果字典
        result_kwargs = {}
        
        # 初始化融合特征
        features_dict = {}
        for channel in modalities_used_in_model:
            if channel == 'wsi=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
                features_dict[channel] = clam_result_kwargs['features'].detach()
            elif channel == 'tma=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
                features_dict[channel] = clam_result_kwargs['features'].detach()
            else:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                features_dict[channel] = self.transfer_layer[channel](input_data[channel])
                deep_supervise_result_kwargs = self.deep_supervise_forward(channel, features_dict[channel], label)
                for key, value in deep_supervise_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
                features_dict[channel] = features_dict[channel].detach()
        
        if self.enable_svd:
            if not hasattr(self, 'alignment_features'):
                self.alignment_features = []
            if self.return_svd_features:
                original_features_dict = features_dict.copy()
                features_dict = self.align_forward(features_dict)
                return {
                    'features': original_features_dict,
                    'aligned_features': features_dict,
                }
            else:
                features_dict = self.align_forward(features_dict)
            self.alignment_features.append(features_dict)
            if self.enable_dynamic_gate:
                result = self.gated_forward(features_dict, label)
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
        else:
            if self.enable_dynamic_gate:
                result = self.gated_forward(features_dict, label)
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
                
        if self.enable_random_loss:
            drop_modality = random.sample(list(features_dict.keys()), random.randint(1, len(features_dict)-1))
            h_partial = []
            for modality in features_dict.keys():
                if modality not in drop_modality:
                    h_partial.append(features_dict[modality])
                else:
                    h_partial.append(torch.zeros_like(features_dict[modality]).to(self.device))
            h_partial = torch.cat(h_partial, dim=1).to(self.device)
            logits = self.fusion_prediction(h_partial.detach())
            result_kwargs['random_partial_loss'] = self.base_loss_fn(logits, label)
            
        h = torch.cat(list(features_dict.values()), dim=1).to(self.device)

        logits = self.fusion_prediction(h.detach())
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # 更新结果字典
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)
