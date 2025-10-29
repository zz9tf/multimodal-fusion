import torch
import torch.nn as nn
import torch.nn.functional as F
from .clam_detach import ClamDetach
import random
from typing import Dict, List, Tuple, Optional

class SVDGateRandomClamDetach(ClamDetach):
    """
    CLAM 模型
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - gate: 是否使用门控注意力
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.enable_dynamic_gated = config.get('enable_dynamic_gated', True)
        self.enable_svd = config.get('enable_svd', True)
        
        if self.enable_dynamic_gated:
            self._init_dynamic_gated_model()
        if self.enable_svd:
            self.alignment_channels = config.get('alignment_channels', self.used_modality)
            self.alignment_layer_num = config.get('alignment_layer_num', 2)
            self.tau1 = config.get('tau1', 0.1)
            self.tau2 = config.get('tau2', 0.1)
            self.lambda1 = config.get('lambda1', 1.0)
            self.lambda2 = config.get('lambda2', 0.1)
            self.loss2_chunk_size = config.get('loss2_chunk_size', None)
            self._init_svd_model()
        self.enable_random_loss = config.get('enable_random_loss', True)
        self.weight_random_loss = config.get('weight_random_loss', 0.1)
    
    def _init_dynamic_gated_model(self):
        self.TCPClassifierCreator = lambda: nn.Sequential(
            nn.Linear(self.output_dim, self.size[1]), 
            nn.ReLU(), 
            nn.Dropout(self.dropout), 
            nn.Linear(self.size[1], self.n_classes)
        )
        
        self.TCPConfidenceLayerCreator = lambda: nn.Sequential(
            nn.Linear(self.output_dim, self.size[1]), 
            nn.Linear(self.size[1], self.size[2]), 
            nn.Linear(self.size[2], 1), 
            nn.Dropout(self.dropout)
        )
        
        self.TCPClassifier = nn.ModuleDict({channel: self.TCPClassifierCreator() for channel in self.used_modality})
        self.TCPConfidenceLayer = nn.ModuleDict({channel: self.TCPConfidenceLayerCreator() for channel in self.used_modality})
        self.TCPLogitsLoss_fn = nn.CrossEntropyLoss(reduction='none')
        self.TCPConfidenceLoss_fn = nn.MSELoss(reduction='none')
        
    def _init_svd_model(self):
        self.alignment_layers_creator = lambda: nn.Sequential(*[
            nn.Linear(self.output_dim, self.output_dim)
            for _ in range(self.alignment_layer_num)
        ])
        self.alignment_layers = nn.ModuleDict({channel: self.alignment_layers_creator() for channel in self.alignment_channels})

    def gated_forward(self, features: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算门控前向传播
        """
        logits_loss = 0.0
        confidence_loss = 0.0
        gated_features = {}
        
        for channel, feature in features.items():
            logits = self.TCPClassifier[channel](feature.detach())
            logits_loss = torch.mean(self.TCPLogitsLoss_fn(logits, labels))
            
            confidence = self.TCPConfidenceLayer[channel](feature.detach())
            pred = F.softmax(logits, dim = 1)
            p_target = torch.gather(pred, 1, labels.unsqueeze(1)).view(-1)
            confidence_loss = torch.mean(self.TCPConfidenceLoss_fn(confidence.view(-1), p_target))
            gated_features[channel] = feature * confidence
            logits_loss += logits_loss
            confidence_loss += confidence_loss
        
        return {
            'gated_features': gated_features,
            'gated_logits_loss': logits_loss,
            'gated_confidence_loss': confidence_loss,
        }
    
    def align_forward(self, features: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算对齐前向传播
        """
        aligned_features = {}
        for channel, feature in features.items():
            aligned_features[channel] = self.alignment_layers[channel](feature)
        loss, svd_values = self._compute_rank1_loss_with_metrics(aligned_features, labels)
        return {
            'aligned_features': aligned_features,
            'align_loss': loss,
            'align_svd_values': svd_values,
        }
        
    def _compute_rank1_loss_with_metrics(self, aligned_features: Dict[str, torch.Tensor], 
                                          aligned_negatives: Optional[Dict[str, torch.Tensor]] = None):
        """
        计算 rank1 损失并返回 SVD 特征值（带详细时间分析）
        
        Returns:
            loss: 损失值
            svd_values: SVD 特征值 Tensor[num_modalities]
        """
        # 1. SVD 计算和 loss1
        feature_list = list(aligned_features.values())
        features = torch.stack(feature_list, dim=-1).squeeze(0)  # [batch_size, feature_dim, num_modalities]
        
        # L2 归一化：x <- x / (||x||_2 + ε)
        eps = 1e-8
        l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)  # [batch_size, 1, num_modalities]
        features = features / (l2_norm + eps)
        
        # U: [batch_size, feature_dim, num_modalities]
        # S(diag): [batch_size, num_modalities]
        # _: [batch_size, feature_dim, num_modalities]
        U, S, _ = torch.linalg.svd(features)
        
        # 📊 记录 SVD 特征值：对 batch 维度求平均（用于记录单个 batch 的代表值）
        svd_values = S.mean(dim=0)  # [num_modalities]
        
        loss1 = F.cross_entropy(S / self.tau1, torch.zeros(S.shape[0]).to(S.device).long())
        
        # 2. loss2 计算
        U1 = U[:, :, 0] # dominate projection [batch_size, feature_dim]
        # 组内矩阵计算：按 loss2_chunk_size 将 batch 分组，仅组内做 softmax/CE
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

        if self.lambda2 == 0:
            return loss1 + self.lambda1 * loss2, svd_values

        # 3. loss3 (loss_IM) 计算
        batch_size = feature_list[0].shape[0]
        positive_labels = torch.ones(batch_size, device=features.device)
        
        def fuse(feat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            # 将多模态特征拼接为单向量 [N, d*K]
            return torch.cat(list(feat_dict.values()), dim=1)

        if aligned_negatives is None:
            raise RuntimeError("Negative features not provided by dataset. Ensure DataLoader yields 'features_neg' per batch.")
        neg_fused = fuse(aligned_negatives)

        pos_fused = fuse(aligned_features)
        all_features = torch.cat([pos_fused, neg_fused], dim=0)
        negative_labels = torch.zeros(neg_fused.shape[0], device=features.device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        pred_M = self.alignment_layers.mlp_predictor(all_features)
        loss_IM = F.binary_cross_entropy(pred_M.squeeze(), all_labels)
        
        total_loss = loss1 + self.lambda1 * loss2 + self.lambda2 * loss_IM
        return total_loss, svd_values
    
    def forward(self, input_data, label):
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典
            label: 标签（用于实例评估）
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        # 初始化结果字典
        result_kwargs = {}
        
        # 初始化融合特征
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
                features_dict[channel] = self.transfer_layer[channel](input_data[channel]).detach()
        
        if self.enable_svd:
            if not hasattr(self, 'alignment_features'):
                self.alignment_features = []
            result = self.align_forward(features_dict, label)
            for key, value in result.items():
                result_kwargs[f'align_{key}'] = value
            features_dict = result['aligned_features']
            self.alignment_features.append(features_dict)
            if self.enable_dynamic_gated:
                result = self.gated_forward(features_dict, label)
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
        else:
            if self.enable_dynamic_gated:
                result = self.gated_forward(features_dict, label)
                for key, value in result.items():
                    result_kwargs[f'gated_{key}'] = value
                features_dict = result['gated_features']
        
        if self.enable_random_loss:
            drop_modality = random.sample(list(features_dict.keys()), random.randint(1, len(features_dict)-1))
            h_partial = torch.cat([features_dict[modality] for modality in features_dict.keys() if modality not in drop_modality], dim=1).to(self.device)
            logits = self.fusion_prediction(h_partial.detach())
            result_kwargs['random_partial_loss'] = self.base_loss_fn(logits, label)
            
        h = torch.cat(list(features_dict.values()), dim=1).to(self.device)
        logits = self.fusion_prediction(h.detach())
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        # 更新结果字典
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        计算损失
        """
        total_loss = 0.0
        for key, value in result.items():
            if key.endswith('_loss'):
                total_loss += value
        base_loss = self.base_loss_fn(logits, labels)
        if self.enable_random_loss:
            total_loss += torch.max(0, base_loss - result['random_partial_loss'])
        return base_loss + total_loss

    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        计算组损失
        """
        features = [] # [batch_size, feature_dim, num_modalities]
        keys = sorted(self.alignment_features[0].keys())
        for feature_dict in self.alignment_features:
            feature = []
            for key in keys:
                feature.append(feature_dict[key])
            features.append(torch.stack(feature, dim=-1).squeeze(0))
        features = torch.stack(features, dim=0)
        svd_loss, svd_values = self._compute_rank1_loss_with_metrics(features)
        return svd_loss

    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        打印结果
        """
        verbose_list = []
        for key, value in result.items():
            if key.endswith('_loss'):
                verbose_list.append((key, value))
            elif key.endswith('_svd_values'):
                verbose_list.append((key, value))
        return verbose_list