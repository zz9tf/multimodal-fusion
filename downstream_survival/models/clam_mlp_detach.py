import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple

class ClamMLPDetach(ClamMLP):
    """
    CLAM MLP Detach model

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: Dropout rate
    - gate: Whether to use gated attention
    - inst_number: Number of positive/negative samples
    - instance_loss_fn: Instance loss function
    - subtyping: Whether it's a subtyping problem
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_data, label):
        """
        Unified forward propagation interface

        Args:
            input_data: Input data, can be:
                - torch.Tensor: Single-modal features [N, D]
                - Dict[str, torch.Tensor]: Multimodal data dictionary
            label: Labels (for instance evaluation)

        Returns:
            Dict[str, Any]: Unified format result dictionary
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        # Initialize result dictionary
        result_kwargs = {}
        
        # Initialize fused features
        h = []
        for channel in modalities_used_in_model:
            features = None
            if channel == 'wsi=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                features = clam_result_kwargs['features'].detach()
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
            elif channel == 'tma=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                features = clam_result_kwargs['features'].detach()
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
            else:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                features = self.transfer_layer[channel](input_data[channel])
            h.append(features)

        h = torch.cat(h, dim=1).to(self.device)
        logits = self.fusion_prediction(h)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # Update result dictionary
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)