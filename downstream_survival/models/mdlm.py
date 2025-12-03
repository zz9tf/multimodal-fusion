import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .clam_mlp import ClamMLP


class MDLM(ClamMLP):
    """
    CLAM + Late Fusion 模型，输出离散标签预测。
    """

    def __init__(self, config: Dict):
        """初始化模态顺序及 late fusion 层。"""
        super().__init__(config)
        self.modality_order = sorted(list(self.used_modality))
        self._init_prediction_heads()
        self.late_fusion_layer = None

    def _init_prediction_heads(self) -> None:
        """初始化各模态的线性分类头。"""
        self.prediction_head_dict = nn.ModuleDict(
            {
                key: nn.Linear(self.output_dim, self.n_classes)
                for key in self.modality_order
            }
        )

    def forward(self, input_data, label):
        """
        前向传播：CLAM 提取模态表征 + 线性分类 per-modality + late fusion。
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        result_kwargs = {}
        
        logits = {}
        modality_features = {}
        for channel in modalities_used_in_model:
            if channel in {"wsi=features", "tma=features"}:
                clam_result_kwargs = self._clam_forward(
                    channel, input_data[channel], label
                )
                modality_features[channel] = clam_result_kwargs["features"]
                modality_features[channel] = self.prediction_head_dict[channel](modality_features[channel])
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f"{channel}_{key}"] = value
            else:
                modality_features[channel] = input_data[channel]
    
        h = torch.cat([modality_features[channel] for channel in self.modality_order], dim=1)
        if self.late_fusion_layer is None:
            self.late_fusion_layer = nn.Linear(
                h.shape[1], self.n_classes
            ).to(self.device)
        logits = self.late_fusion_layer(h)


        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]

        result_kwargs["Y_prob"] = y_prob
        result_kwargs["Y_hat"] = y_hat
        return self._create_result_dict(logits, y_prob, y_hat, **result_kwargs)

