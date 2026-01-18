import torch
import torch.nn.functional as F
from .deep_supervise_svd_gate_random import DeepSuperviseSVDGateRandomClam
import random

class DeepSuperviseSVDGateRandomClamDetach(DeepSuperviseSVDGateRandomClam):
    """
    CLAM MLP model

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
                
        if self.enable_random_loss and self.training:
            sorted_keys = sorted(features_dict.keys())
            drop_modality = random.sample(sorted_keys, random.randint(1, len(features_dict)-1))
            h_partial = []
            for modality in sorted_keys:
                if modality not in drop_modality:
                    h_partial.append(features_dict[modality])
                else:
                    h_partial.append(torch.zeros_like(features_dict[modality]).to(self.device))
            h_partial = torch.cat(h_partial, dim=1).to(self.device)
            logits = self.fusion_prediction(h_partial.detach())
            result_kwargs['random_partial_loss'] = self.base_loss_fn(logits, label)
            
        sorted_keys = sorted(features_dict.keys())
        h = torch.cat([features_dict[mod] for mod in sorted_keys], dim=1).to(self.device)

        logits = self.fusion_prediction(h.detach())
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # Update result dictionary
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)
