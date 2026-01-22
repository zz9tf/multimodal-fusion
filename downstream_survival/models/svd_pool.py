import torch
import torch.nn as nn
import torch.nn.functional as F
from .clam_mlp import ClamMLP
import random
from typing import Dict, List, Tuple, Optional

class SVDPool(ClamMLP):
    """
    SVD Pool model

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
        
        self.alignment_channels = config.get('alignment_channels', self.used_modality)
        self.alignment_layer_num = config.get('alignment_layer_num', 2)
        self.tau1 = config.get('tau1', 0.1)
        self.tau2 = config.get('tau2', 0.1)
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.1)
        self.loss2_chunk_size = config.get('loss2_chunk_size', None)
        self.return_svd_features = config.get('return_svd_features', False)
        self._init_align_model()
        self.pooling_strategy = config.get('pooling_strategy', 'mean')
        if self.pooling_strategy == 'mean':
            self.pooling_fn = lambda x, dim: torch.mean(x, dim=dim)
        elif self.pooling_strategy == 'max':
            self.pooling_fn = lambda x, dim: torch.max(x, dim=dim).values
        elif self.pooling_strategy == 'sum':
            self.pooling_fn = lambda x, dim: torch.sum(x, dim=dim)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        self.fusion_prediction = nn.Linear(self.output_dim, self.n_classes)

    def _init_align_model(self):
        self.alignment_layers_creator = lambda: nn.Sequential(*[
            nn.Linear(self.output_dim, self.output_dim)
            for _ in range(self.alignment_layer_num)
        ])
        self.alignment_layers = nn.ModuleDict({channel: self.alignment_layers_creator() for channel in sorted(self.alignment_channels)})
    
    def align_forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate alignment forward propagation
        """
        aligned_features = {}
        for channel in sorted(features.keys()):
            feature = features[channel]
            aligned_features[channel] = self.alignment_layers[channel](feature)
        return aligned_features
        
    def _compute_rank1_loss_with_metrics(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate rank1 loss and return SVD eigenvalues (with detailed timing analysis)

        Returns:
            loss: Loss value
            svd_values: SVD eigenvalues Tensor[num_modalities]
        """
        # 1. SVD 计算和 loss1
        # L2 归一化：x <- x / (||x||_2 + ε)
        eps = 1e-8
        l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)  # [batch_size, 1, num_modalities]
        features = features / (l2_norm + eps)
        
        # U: [batch_size, feature_dim, num_modalities]
        # S(diag): [batch_size, num_modalities]
        # _: [batch_size, feature_dim, num_modalities]
        U, S, _ = torch.linalg.svd(features)
        
        # 📊 Record SVD eigenvalues: average over batch dimension (for recording representative values of single batch)
        svd_values = S.mean(dim=0)  # [num_modalities]
        
        loss1 = F.cross_entropy(S / self.tau1, torch.zeros(S.shape[0]).to(S.device).long())
        
        # 2. loss2 calculation
        U1 = U[:, :, 0] # dominate projection [batch_size, feature_dim]
        # Intra-group matrix calculation: group batches by loss2_chunk_size, only do softmax/CE within groups
        batch_count = U1.shape[0]
        if self.loss2_chunk_size is None or self.loss2_chunk_size >= batch_count:
            loss2 = F.cross_entropy((U1 @ U1.T) / self.tau2, torch.arange(batch_count, device=U1.device).long())
        else:
            c = max(1, int(self.loss2_chunk_size))
            full = (batch_count // c) * c
            loss2_sum = U1.new_tensor(0.0)
            if full > 0:
                groups = U1[:full].view(-1, c, U1.shape[1])  # [G, c, D]
                logits_gc = torch.einsum('gxd,gyd->gxy', groups, groups) / self.tau2  # [G, c, c]
                targets_gc = torch.arange(c, device=U1.device).expand(logits_gc.shape[0], c) # [G, c]
                loss2_sum = loss2_sum + F.cross_entropy(
                    logits_gc.reshape(-1, c), targets_gc.reshape(-1), reduction='sum'
                )
            if full < batch_count:
                tail = U1[full:]
                c_tail = tail.shape[0]
                logits_tail = (tail @ tail.T) / self.tau2
                targets_tail = torch.arange(c_tail, device=U1.device)
                loss2_sum = loss2_sum + F.cross_entropy(logits_tail, targets_tail, reduction='sum')
            loss2 = loss2_sum / batch_count

        return loss1 + self.lambda1 * loss2, svd_values
    
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
                features_dict[channel] = clam_result_kwargs['features']
            elif channel == 'tma=features':
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                for key, value in clam_result_kwargs.items():
                    result_kwargs[f'{channel}_{key}'] = value
                features_dict[channel] = clam_result_kwargs['features']
            else:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                features_dict[channel] = self.transfer_layer[channel](input_data[channel])
        
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

        # 将字典的值转换为张量并堆叠进行pooling
        features_tensor = torch.stack(list(features_dict.values()), dim=1)  # [batch_size, num_modalities, feature_dim]
        h = self.pooling_fn(features_tensor, dim=1).to(self.device)  # [batch_size, feature_dim]

        logits = self.fusion_prediction(h)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # Update result dictionary
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate loss
        """
        return self.base_loss_fn(logits, labels)

    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate group loss
        """
        features = [] # [batch_size, feature_dim, num_modalities]
        keys = sorted(self.alignment_features[0].keys())
        for feature_dict in self.alignment_features:
            feature = []
            for key in keys:
                feature.append(feature_dict[key])  # each: [1, feature_dim]
            # per-batch: [1, feature_dim, num_modalities] -> squeeze batch -> [feature_dim, num_modalities]
            features.append(torch.stack(feature, dim=-1).squeeze(0))
        self.alignment_features = []
        # aggregate across batches: [num_batches, feature_dim, num_modalities]
        features = torch.stack(features, dim=0)
        svd_loss, svd_values = self._compute_rank1_loss_with_metrics(features)
        result['svd_loss'] = svd_loss
        result['svd_values'] = svd_values
        return svd_loss

    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Print results
        """
        verbose_list = []
        for key, value in result.items():
            if key.endswith('_loss'):
                verbose_list.append((key, value))
            elif key.endswith('_svd_values'):
                verbose_list.append((key, value))
        return verbose_list