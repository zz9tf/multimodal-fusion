import torch
import torch.nn as nn
import torch.nn.functional as F
from .svd_gate_random_clam import SVDGateRandomClam
import random
from typing import Dict, List, Tuple, Optional

class SVDGateRandomClamDetach(SVDGateRandomClam):
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

    def forward(self, input_data, label, drop_prob=None):
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
        result_kwargs = {
            "raw_features_dict": {k: v.detach().clone() for k, v in input_data.items()}
        }
        
        # Initialize fused features
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
        
        # 📊 Save results after CLAM (value for each modality)
        result_kwargs['original_features_dict'] = {k: v.detach().clone() for k, v in features_dict.items()}
        
        if self.enable_svd:
            if not hasattr(self, 'alignment_features'):
                self.alignment_features = []
            features_dict = self.align_forward(features_dict)
            # 📊 Save results after SVD (value for each modality)
            result_kwargs['aligned_svd_features_dict'] = {k: v.detach().clone() for k, v in features_dict.items()}
            if self.return_svd_features:
                return {
                    'features': result_kwargs['original_features_dict'],
                    'aligned_features': features_dict,
                }
            self.alignment_features.append(features_dict)
            if self.enable_dynamic_gate:
                result = self.gated_forward(features_dict, label)
                # 📊 Save results after Dynamic Gate (value for each modality)
                result_kwargs['svd_gated_features_dict'] = {k: v.detach().clone() for k, v in result['gated_features'].items()}
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
        else:
            if self.enable_dynamic_gate:
                result = self.gated_forward(features_dict, label)
                # 📊 Save results after Dynamic Gate (value for each modality)
                result_kwargs['svd_gated_features_dict'] = {k: v.detach().clone() for k, v in result['gated_features'].items()}
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
                
        if self.enable_random_loss and self.training:
            sorted_features_dict_keys = sorted(features_dict.keys())
            drop_modality = random.sample(sorted_features_dict_keys, random.randint(1, len(features_dict)-1))
            # 📊 Save information about random operations
            result_kwargs['random_sorted_keys'] = sorted_features_dict_keys
            result_kwargs['random_drop_modality'] = drop_modality
            result_kwargs['random_n'] = len(drop_modality)
            
            h_partial = []
            for modality in sorted_features_dict_keys:
                if modality not in drop_modality:
                    h_partial.append(features_dict[modality])
                else:
                    h_partial.append(torch.zeros_like(features_dict[modality]).to(self.device))
            h_partial = torch.cat(h_partial, dim=1).to(self.device)
            # 📊 Save h after random operation
            result_kwargs['h_random'] = h_partial.detach().clone()
            logits = self.fusion_prediction(h_partial.detach())
            result_kwargs['random_partial_loss'] = self.base_loss_fn(logits, label)
            
        if self.training is False and drop_prob is not None:
            h = []
            for modality in sorted(features_dict.keys()):
                random_drop = torch.rand(1) < drop_prob
                if random_drop:
                    h.append(torch.zeros_like(features_dict[modality]).to(self.device))
                else:
                    h.append(features_dict[modality])
            h = torch.cat(h, dim=1).to(self.device)
        else:
            h = torch.cat([features_dict[mod] for mod in sorted(features_dict.keys())], dim=1).to(self.device)
        # 📊 Save final h
        result_kwargs['h'] = h.detach().clone()
        
        logits = self.fusion_prediction(h.detach())
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # Update result dictionary
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)
