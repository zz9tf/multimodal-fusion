import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .clam_mlp import ClamMLP


class MFMF(ClamMLP):
    """
    MFMF model with configurable Perceiver IO style fusion.
    
    Supports flexible configuration of attention block order via fusion_config parameter.
    Example: fusion_config = ["other:tma", "res:wsi", "reconstruct:res"]
    - Block 1: other (Q) queries tma (K,V) -> res
    - Block 2: res (Q) queries wsi (K,V) -> res
    - Block 3: reconstruct (Q) queries res (K,V) -> output
    """

    def __init__(self, config: Dict):
        """Initialize MFMF model with configurable fusion."""
        super().__init__(config)
        
        # Fusion configuration: list of "Q_source:K_source" strings
        # Special value "res" refers to previous block output
        self.fusion_blocks_sequence = config.get('fusion_blocks_sequence', [
            {"q": "other", "kv": "tma"},
            {"q": "result", "kv": "wsi"},
            {"q": "reconstruct", "kv": "result"}
        ])
        
        attention_num_heads = config.get('attention_num_heads', 8)
        # Initialize attention blocks for fusion sequence
        self.attention_blocks = nn.ModuleDict({})
        for block in self.fusion_blocks_sequence:
            self.attention_blocks[f'{block["q"]}:{block["kv"]}'] = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=attention_num_heads,
                batch_first=False
            ).to(self.device)
        
        # Create prediction layer
        self.fusion_prediction_layer = nn.Linear(self.output_dim, self.n_classes).to(self.device)
    
    def _process_input_data(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process input data and extract modality features.
        
        Returns:
            Dictionary of modality features
        """
        feature_to_modality = {
            'wsi=features': 'wsi',
            'wsi=reconstructed_features': 'reconstruct',
            'tma': 'tma',
            'other': 'other'
        }
        tma_features = []
        other_features = []
        modality_features = {}  # For fusion module
        
        for channel in self.channels_used_in_model:
            if channel.endswith('=mask'): # process mask channel
                continue
            feat = input_data[channel].squeeze(0).to(self.device)
            # process other modality mask
            if not channel.startswith('wsi=') and not channel.startswith('tma='): 
                channel_name = channel.split('=')[0]
                if f'{channel_name}=mask' in input_data:
                    mask = input_data[f'{channel_name}=mask'].squeeze(0).to(self.device)
                    feat = feat * mask            
            # unify dimension to output_dim
            # Check if transfer_layer exists and if input dimension matches
            if channel not in self.transfer_layer:
                self.transfer_layer[channel] = self.create_transfer_layer(feat.shape[1])
            else:
                # Check if the existing transfer_layer has the correct input dimension
                existing_input_dim = self.transfer_layer[channel].in_features
                if existing_input_dim != feat.shape[1]:
                    # Recreate transfer_layer with correct input dimension
                    self.transfer_layer[channel] = self.create_transfer_layer(feat.shape[1])
            feat = self.transfer_layer[channel](feat)
            
            if channel.startswith('tma='): # process TMA channel
                tma_features.append(feat)
            elif channel.startswith('wsi='): # process WSI features & reconstructed channel
                modality_features[feature_to_modality[channel]] = feat
            else: # process other channel
                other_features.append(feat)
        
        # Concatenate TMA features
        if len(tma_features) > 0:
            tma_features = torch.cat(tma_features, dim=0).to(self.device)
            modality_features[feature_to_modality['tma']] = tma_features
        
        # Concatenate other modality features if multiple
        if len(other_features) > 0:
            other_features = torch.cat(other_features, dim=0).to(self.device)
            modality_features[feature_to_modality['other']] = other_features
        
        return modality_features

    def forward(self, input_data, label):
        """
        Forward pass: configurable attention fusion + classification.
        """
        modality_features = self._process_input_data(input_data)
        
        # Forward through attention blocks sequentially
        modality_features['result'] = None
        for block in self.fusion_blocks_sequence:
            q = modality_features[block['q']]
            kv = modality_features[block['kv']]
            q = q.unsqueeze(1)  # [seq_len, 1, embed_dim]
            k = kv.unsqueeze(1)  # [seq_len_kv, 1, embed_dim]
            v = kv.unsqueeze(1)  # [seq_len_kv, 1, embed_dim]
            
            attn_output, _ = self.attention_blocks[f'{block["q"]}:{block["kv"]}'](q, k, v)
            modality_features['result'] = attn_output.squeeze(1)
        # Add batch dimension for classification layer (expects [1, D])
        fused_output = modality_features['result'].mean(dim=0, keepdim=True)  # [1, output_dim]
        
        # Classification
        logits = self.fusion_prediction_layer(fused_output)  # [1, n_classes]
        
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        
        result_kwargs = {
            "Y_prob": y_prob,
            "Y_hat": y_hat
        }
        
        return self._create_result_dict(logits, y_prob, y_hat, **result_kwargs)

