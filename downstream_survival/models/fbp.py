import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple

class FBP(ClamMLP):
    """
    FBP model (Factorized / Bilinear style multimodal fusion):
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size
    - dropout: Dropout rate
    """

    def __init__(self, config):
        """
        Initialize FBP model and build multimodal fusion related layers.

        Parameters
        ----------
        config : Dict
            Model configuration dictionary, containing all parameters required by parent class `ClamMLP`.
        """
        super().__init__(config)

        # Record the modalities actually used by the current model, and fix an order for concatenation
        self.modality_order = sorted(list(self.used_modality))

        # Perform self-bilinear transformation on each modality vector: h_m' = B(h_m, h_m)
        # Input: (num_modalities, output_dim) -> (num_modalities, output_dim)
        self.modality_bilinear_fusion_layer = nn.Bilinear(
            self.output_dim,
            self.output_dim,
            self.output_dim,
        )
        self.modality_moe_fusion_layer = nn.Linear(len(self.modality_order), 1, bias=False)
        self.moe_fusion_layer = nn.Linear(len(self.modality_order), 1, bias=False)

        # Finally use the fused vector for classification / survival prediction
        self.fusion_prediction_layer = nn.Linear(self.output_dim, self.n_classes)
        
    def forward(self, input_data, label):
        """
        Execute forward propagation and use MoE style per-modality attention for feature fusion.

        Fusion process (from fusion perspective only):
        1. Extract a bag-level representation vector h_m ∈ R^{output_dim} for each modality
        2. Enhance self-interaction of each modality through bilinear layer: h_m' = B(h_m, h_m)
        3. Use a linear gating network to output a scalar score for each modality, and perform softmax on modality dimension to get attention weights α_m
        4. Weighted sum of modality features according to weights to get fused vector h_fused = Σ_m α_m h_m'
        5. Send fused vector to classification head `fusion_prediction_layer` to get logits

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
        
        # Collect features from all modalities
        modality_features = {}
        for channel in modalities_used_in_model:
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
        
        # Collect all modality features and concatenate along modality dimension:
        # Each modality_features[channel] is [1, output_dim], use cat on dim=0 for concatenation
        # h: [num_modalities, output_dim]
        h = torch.cat(
            [modality_features[channel] for channel in self.modality_order],
            dim=0,
        )
        
        # 1) Construct pairwise modality combinations and get pairwise features through bilinear layer
        num_modalities = h.size(0)  # M
        h_i = h.unsqueeze(1).expand(num_modalities, num_modalities, self.output_dim)  # [M, M, D]
        h_j = h.unsqueeze(0).expand(num_modalities, num_modalities, self.output_dim)  # [M, M, D]
        pairwise_interactions = self.modality_bilinear_fusion_layer(h_i, h_j)  # [M, M, D]
        pairwise_interactions = pairwise_interactions.transpose(1, 2) # [M, D, M]
        # 2) First layer MoE: perform attention on "second modality dimension", aggregate to representation for each modality i
        #    modality_interactions: [D, M]
        pairwise_interactions = self.modality_moe_fusion_layer(pairwise_interactions).squeeze(-1)
        pairwise_interactions = pairwise_interactions.transpose(0, 1)
        # 3) Second layer MoE: aggregate all modalities on modality dimension to get global representation
        #    modality_interactions: [D]
        pairwise_interactions = self.moe_fusion_layer(pairwise_interactions).transpose(0, 1)

        # 4) Use fused representation for final prediction
        logits = self.fusion_prediction_layer(pairwise_interactions)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # Update result dictionary
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)