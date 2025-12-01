import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple

class FBP(ClamMLP):
    """
    FBP 模型（Factorized / Bilinear 风格的多模态融合）：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小
    - dropout: dropout 率
    """

    def __init__(self, config):
        """
        初始化 FBP 模型，并构建多模态融合相关的层。

        Parameters
        ----------
        config : Dict
            模型配置字典，包含父类 `ClamMLP` 所需的所有参数。
        """
        super().__init__(config)

        # 记录当前模型实际使用到的模态，并固定一个顺序用于拼接
        self.modality_order = sorted(list(self.used_modality))

        # 对每个模态向量做一次自双线性变换：h_m' = B(h_m, h_m)
        # 输入: (num_modalities, output_dim) -> (num_modalities, output_dim)
        self.modality_bilinear_fusion_layer = nn.Bilinear(
            self.output_dim,
            self.output_dim,
            self.output_dim,
        )
        self.modality_moe_fusion_layer = nn.Linear(len(self.modality_order), 1, bias=False)
        self.moe_fusion_layer = nn.Linear(len(self.modality_order), 1, bias=False)

        # 最终使用融合后的向量做分类 / 生存预测
        self.fusion_prediction_layer = nn.Linear(self.output_dim, self.n_classes)
        
    def forward(self, input_data, label):
        """
        执行前向传播，并使用 MoE 风格的按模态注意力进行特征融合。

        融合流程（仅从 fusion 视角）：
        1. 对每个模态提取一个 bag-level 表征向量 h_m ∈ R^{output_dim}
        2. 通过双线性层对每个模态做自交互增强：h_m' = B(h_m, h_m)
        3. 使用一个线性门控网络对每个模态输出一个标量得分，并在模态维度做 softmax 得到注意力权重 α_m
        4. 按权重对模态特征加权求和，得到融合向量 h_fused = Σ_m α_m h_m'
        5. 将融合向量送入分类头 `fusion_prediction_layer` 得到 logits

        Parameters
        ----------
        input_data : torch.Tensor | Dict[str, torch.Tensor]
            - 若为 `torch.Tensor`：单模态特征，形状 [N, D]
            - 若为 `Dict[str, torch.Tensor]`：多模态数据字典，key 为模态名称
        label : torch.Tensor
            标签张量，用于实例评估，形状通常为 [1]

        Returns
        -------
        Dict[str, torch.Tensor]
            统一格式的结果字典，至少包含：
            - 'Y_prob' : [1, n_classes]，类别概率
            - 'Y_hat'  : [1, 1]，预测类别索引
            以及各模态的 CLAM 相关结果（如果适用）。
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
        
        # 收集所有模态特征并沿模态维度拼接:
        # 每个 modality_features[channel] 是 [1, output_dim]，使用 cat 在 dim=0 上拼接
        # h: [num_modalities, output_dim]
        h = torch.cat(
            [modality_features[channel] for channel in self.modality_order],
            dim=0,
        )
        
        # 1) 构造两两模态组合，并通过双线性层得到 pairwise 特征
        num_modalities = h.size(0)  # M
        h_i = h.unsqueeze(1).expand(num_modalities, num_modalities, self.output_dim)  # [M, M, D]
        h_j = h.unsqueeze(0).expand(num_modalities, num_modalities, self.output_dim)  # [M, M, D]
        pairwise_interactions = self.modality_bilinear_fusion_layer(h_i, h_j)  # [M, M, D]
        pairwise_interactions = pairwise_interactions.transpose(1, 2) # [M, D, M]
        # 2) 第一层 MoE：在“第二个模态维度”上做注意力，聚合为每个模态 i 的表示
        #    modality_interactions: [D, M]
        pairwise_interactions = self.modality_moe_fusion_layer(pairwise_interactions).squeeze(-1)
        pairwise_interactions = pairwise_interactions.transpose(0, 1)
        # 3) 第二层 MoE：在模态维度上聚合所有模态，得到全局表示
        #    modality_interactions: [D]
        pairwise_interactions = self.moe_fusion_layer(pairwise_interactions).transpose(0, 1)

        # 4) 使用融合表示进行最终预测
        logits = self.fusion_prediction_layer(pairwise_interactions)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # 更新结果字典
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)