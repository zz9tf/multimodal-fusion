import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from perceiver.model.core.modules import CrossAttentionLayer

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
        widening_factor = config.get('attention_widening_factor', 1)  # MLP widening factor
        dropout = config.get('attention_dropout', 0.0)
        residual_dropout = config.get('attention_residual_dropout', 0.0)
        
        # Initialize CrossAttentionLayer blocks for fusion sequence
        self.attention_blocks = nn.ModuleDict({})
        for block in self.fusion_blocks_sequence:
            self.attention_blocks[f'{block["q"]}:{block["kv"]}'] = CrossAttentionLayer(
                num_heads=attention_num_heads,
                num_q_input_channels=self.output_dim,
                num_kv_input_channels=self.output_dim,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                attention_residual=True,
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
            if channel.endswith('=mask'): # Process mask channel
                continue
            feat = input_data[channel].squeeze(0).to(self.device)
            # Process other modality mask
            if not channel.startswith('wsi=') and not channel.startswith('tma='): 
                channel_name = channel.split('=')[0]
                if f'{channel_name}=mask' in input_data:
                    mask = input_data[f'{channel_name}=mask'].squeeze(0).to(self.device)
                    feat = feat * mask            
            # Unify dimension to output_dim
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
            
            if channel.startswith('tma='): # Process TMA channel
                tma_features.append(feat)
            elif channel.startswith('wsi='): # Process WSI features & reconstructed channel
                modality_features[feature_to_modality[channel]] = feat
            else: # Process other channel
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
            q = modality_features[block['q']]  # [seq_len, embed_dim]
            kv = modality_features[block['kv']]  # [seq_len_kv, embed_dim]
            
            # Convert to batch-first format [1, seq_len, embed_dim] for CrossAttentionLayer
            # CrossAttentionLayer expects batch_first=True format
            q_batch = q.unsqueeze(0)  # [1, seq_len, embed_dim]
            kv_batch = kv.unsqueeze(0)  # [1, seq_len_kv, embed_dim]
            
            # CrossAttentionLayer returns ModuleOutput with last_hidden_state
            attn_output = self.attention_blocks[f'{block["q"]}:{block["kv"]}'](q_batch, kv_batch)
            # Extract last_hidden_state and remove batch dimension
            modality_features['result'] = attn_output.last_hidden_state.squeeze(0)  # [seq_len, embed_dim]
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

