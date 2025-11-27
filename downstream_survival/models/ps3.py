import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple

class PS3(ClamMLP):
    """
    PS3模型，使用 Cross Attention 机制进行多模态融合
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - gate: 是否使用门控注意力
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    - cross_attn_dim: Cross Attention 的维度，默认为 output_dim
    - num_heads: 注意力头数（当前实现为单头，保留用于未来扩展）
    - cross_attn_dropout: Cross Attention 的 dropout 率，默认 0.1
    """
    
    def __init__(self, config):
        """
        初始化 PS3 模型，设置 Cross Attention 相关参数
        
        @param {Dict} config - 模型配置字典，包含模型所需的所有参数
        """
        super().__init__(config)
        self.modality_order = sorted(self.modalities_used_in_model)
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
        
        流程：
        1. 提取各模态特征（WSI/TMA 使用 CLAM，其他模态使用 transfer layer）
        2. 为每个模态生成 Q, K, V 投影
        3. 对每个模态，使用其 Q 查询所有模态的 K，计算注意力权重
        4. 使用注意力权重对所有模态的 V 进行加权求和
        5. 拼接所有融合后的特征，通过 fusion_prediction 进行分类
        
        @param {torch.Tensor|Dict[str, torch.Tensor]} input_data - 输入数据
            - torch.Tensor: 单模态特征 [N, D]
            - Dict[str, torch.Tensor]: 多模态数据字典，key 为模态名称
        @param {torch.Tensor} label - 标签张量，用于实例评估 [1]
                
        @returns {Dict[str, Any]} 统一格式的结果字典，包含：
            - Y_prob: 预测概率 [1, n_classes]
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