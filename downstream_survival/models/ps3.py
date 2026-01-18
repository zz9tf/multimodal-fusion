import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple

class PS3(ClamMLP):
    """
    PS3 model, using Cross Attention mechanism for multimodal fusion

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: Dropout rate
    - gate: Whether to use gated attention
    - inst_number: Number of positive/negative samples
    - instance_loss_fn: Instance loss function
    - subtyping: Whether it's a subtyping problem
    - cross_attn_dim: Cross Attention dimension, defaults to output_dim
    - num_heads: Number of attention heads (currently implemented as single head, reserved for future expansion)
    - cross_attn_dropout: Cross Attention dropout rate, default 0.1
    """
    
    def __init__(self, config):
        """
        Initialize PS3 model, set Cross Attention related parameters

        @param {Dict} config - Model configuration dictionary, containing all parameters required by the model
        """
        super().__init__(config)
        self.modality_order = sorted(self.used_modality)
        self.token_norm = nn.LayerNorm(self.output_dim).to(self.device)  # Token normalization
        self.qkv_proj = nn.Linear(self.output_dim, 3 * self.output_dim).to(self.device)
        self.modality_mlp_layers = nn.ModuleDict(
            {
                channel: nn.Linear(self.output_dim, self.output_dim) 
                for channel in self.modality_order
            }
        ).to(self.device)
        
        self.modality_fusion_layer = nn.Sequential(
                nn.Linear(len(self.modality_order) * self.output_dim, self.size[1]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.size[1], self.n_classes)
        ).to(self.device)
        
    
    def forward(self, input_data, label):
        """
        统一的前向传播接口，使用 Cross Attention 进行多模态特征融合
        
        Process:
        1. Extract features for each modality (WSI/TMA use CLAM, other modalities use transfer layer)
        2. Generate Q, K, V projections for each modality
        3. For each modality, use its Q to query all modalities' K, calculate attention weights
        4. Use attention weights to perform weighted summation of all modalities' V
        5. Concatenate all fused features and perform classification through fusion_prediction
        
        @param {torch.Tensor|Dict[str, torch.Tensor]} input_data - 输入数据
            - torch.Tensor: Single-modal features [N, D]
            - Dict[str, torch.Tensor]: Multimodal data dictionary, key is modality name
        @param {torch.Tensor} label - Label tensor for instance evaluation [1]
                
        @returns {Dict[str, Any]} Unified format result dictionary, contains:
            - Y_prob: Prediction probabilities [1, n_classes]
            - Y_hat: 预测类别 [1, 1]
            - 各模态的 CLAM 相关结果（如果适用）
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        # 初始化结果字典
        result_kwargs = {}
        
        # 收集所有模态的特征
        modality_features = {}
        for channel in modalities_used_in_model:
            features = None
            if channel == 'wsi=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                modality_features[channel] = clam_result_kwargs['features'].detach()
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
            elif channel == 'tma=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                modality_features[channel] = clam_result_kwargs['features'].detach()
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
            else:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                modality_features[channel] = self.transfer_layer[channel](input_data[channel])
        
        # 收集所有模态特征并拼接: [num_modalities, output_dim]
        # 每个 modality_features[channel] 是 [1, output_dim]，使用 cat 在 dim=0 上拼接
        h = torch.cat([modality_features[channel] for channel in self.modality_order], dim=0)  # [num_modalities, output_dim]
        # 🔹 Step 1: Token Normalization (对每个模态的 token 进行 normalization)
        h = self.token_norm(h)  # [num_modalities, output_dim]
        
        # 🔹 Step 2: QKV Projection (并行计算所有模态的 Q, K, V)
        qkv_h = self.qkv_proj(h)  # [num_modalities, 3 * output_dim]
        # 🔹 Step 3: Split Q, K, V
        q, k, v = qkv_h.chunk(3, dim=-1)  # 每个都是 [num_modalities, output_dim]
        
        # 🔹 Step 4: Cross Attention 计算
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        # q: [num_modalities, output_dim], k: [num_modalities, output_dim]
        # 输出: [num_modalities, num_modalities] (每个模态对所有模态的注意力分数)
        attn_scores = torch.mm(q, k.transpose(0, 1)) / np.sqrt(self.output_dim)  # [num_modalities, num_modalities]
        
        # 应用 softmax 得到注意力权重
        attention_weights = F.softmax(attn_scores, dim=-1)  # [num_modalities, num_modalities]
        
        # 使用注意力权重对 V 进行加权求和
        # attention_weights: [num_modalities, num_modalities]
        # v: [num_modalities, output_dim]
        # 输出: [num_modalities, output_dim] (每个模态的融合后特征)
        h = torch.mm(attention_weights, v)  # [num_modalities, output_dim]
        
        # 🔹 Step 4.5: 对每个模态应用独立的 MLP 和 normalization
        # 优化方案：先应用所有 MLP，然后一次性应用 normalization（高效且简洁）
        # 由于每个模态的 MLP 不同，无法完全并行化，但 normalization 可以批量处理
        # 使用列表推导式 + cat 比循环 + stack 更高效（减少 squeeze 操作）
        h_mlp_list = [
            self.modality_mlp_layers[channel](h[index:index+1, :])  # [1, output_dim]
            for index, channel in enumerate(self.modality_order)
        ]
        h_mlp = torch.cat(h_mlp_list, dim=0)  # [num_modalities, output_dim]
        # 一次性对所有模态应用 normalization（更高效，从 num_modalities 次调用减少到 1 次）
        h = self.token_norm(h_mlp)  # [num_modalities, output_dim]
        
        # 🔹 Step 5: Flatten 并拼接所有模态的融合特征
        h = h.view(1, -1)  # [1, num_modalities * output_dim]
        
        # 🔹 Step 6: 通过融合预测层进行分类
        logits = self.modality_fusion_layer(h)  # [1, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # 更新结果字典
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)