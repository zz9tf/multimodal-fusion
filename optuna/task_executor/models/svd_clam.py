import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from typing import Dict, Optional
from .alignment_model import MultiModalAlignmentModel

class Attn_Net(nn.Module):
    """注意力网络（无门控）"""
    
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        初始化注意力网络
        
        Args:
            L: 输入特征维度
            D: 隐藏层维度
            dropout: 是否使用dropout
            n_classes: 输出类别数
        """
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量
            
        Returns:
            attention_weights: 注意力权重
            x: 原始输入特征
        """
        return self.module(x), x

class Attn_Net_Gated(nn.Module):
    """门控注意力网络"""
    
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        初始化门控注意力网络
        
        Args:
            L: 输入特征维度
            D: 隐藏层维度
            dropout: 是否使用dropout
            n_classes: 输出类别数
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        门控注意力前向传播
        
        Args:
            x: 输入特征张量
            
        Returns:
            attention_weights: 门控注意力权重
            x: 原始输入特征
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x

class SVD_CLAM(BaseModel):
    """
    SVD-CLAM 模型：结合SVD对齐和CLAM注意力的多模态生存预测模型
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小 ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: dropout率
    - gate: 是否使用门控注意力
    - inst_number: 正负样本采样数量
    - instance_loss_fn: 实例损失函数
    - subtyping: 是否为子类型问题
    - alignment_layer_num: 对齐层数量
    - alignment_channels: 对齐通道列表
    - tau1, tau2: 温度参数
    - lambda1, lambda2: 损失权重参数
    """
    
    def __init__(self, config):
        """
        初始化SVD-CLAM模型
        
        Args:
            config: 模型配置字典，包含所有必要的参数
        """
        super().__init__(config)
        
        # 验证配置完整性
        self._validate_config(config)
        
        # 模型大小配置
        self.size_dict = {
            "small": [self.input_dim, 512, 256], 
            "big": [self.input_dim, 512, 384], 
            "128*64": [self.input_dim, 128, 64], 
            "64*32": [self.input_dim, 64, 32], 
            "32*16": [self.input_dim, 32, 16],
            "16*8": [self.input_dim, 16, 8],
            "8*4": [self.input_dim, 8, 4],
            "4*2": [self.input_dim, 4, 2],
            "2*1": [self.input_dim, 2, 1]
        }
        
        self.gate = config.get('gate', True)
        self.base_weight = config.get('base_weight', 0.7)
        if config.get('inst_loss_fn') is None or config.get('inst_loss_fn') == 'ce':
            self.instance_loss_fn = nn.CrossEntropyLoss()
        elif config.get('inst_loss_fn') == 'svm':
            self.instance_loss_fn = nn.SmoothTop1SVM(n_classes=self.n_classes)
        else:
            raise ValueError(f"不支持的instance损失函数: {config.get('inst_loss_fn')}")

        self.model_size = config['model_size']
        self.subtyping = config.get('subtyping', False)
        self.inst_number = config.get('inst_number', 8)
        self.channels_used_in_model = config['channels_used_in_model']
        self.return_features = config.get('return_features', False)
        self.attention_only = config.get('attention_only', False)
        size = self.size_dict[self.model_size]
        self.alignment_layer_num = config.get('alignment_layer_num', 2)
        self.alignment_channels = config.get('alignment_channels', ['tma_CD3', 'tma_CD8', 'tma_CD56', 'tma_CD68', 'tma_CD163', 'tma_HE', 'tma_MHC1', 'tma_PDL1'])
        self.tau1 = config.get('tau1', 0.1)
        self.tau2 = config.get('tau2', 0.1)
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.1)
        self.loss2_chunk_size = config.get('loss2_chunk_size', None)
        
        self.alignment_layers = MultiModalAlignmentModel(
            modality_names=self.alignment_channels, 
            feature_dim=self.input_dim, 
            num_layers=self.alignment_layer_num
        )
        
        # 构建特征提取层
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
        
        # 构建注意力网络（单分支：输出1个注意力值）
        if self.gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=self.dropout, n_classes=1 if self.n_classes == 2 else self.n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=self.dropout, n_classes=1 if self.n_classes == 2 else self.n_classes
            )
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # 构建分类器
        self.classifiers = None
        if self.n_classes == 2:
            self.classifiers = nn.Linear(size[1], self.n_classes)
        else:
            self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(self.n_classes)])
        
        # 实例分类器
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(self.n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
 
    def _validate_config(self, config):
        """验证配置完整性"""
        required_params = ['n_classes', 'input_dim', 'model_size', 'dropout']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"CLAM_SB配置缺少必需参数: {missing_params}")
        
        # 验证模型大小
        valid_sizes = ["small", "big", "128*64", "64*32", "32*32", "16*8", "8*4", "4*2", "2*1"]
        if config['model_size'] not in valid_sizes:
            raise ValueError(f"不支持的模型大小: {config['model_size']}，支持的大小: {valid_sizes}")
        
        # 验证类别数量
        if config['n_classes'] < 2:
            raise ValueError(f"类别数量必须 >= 2，当前: {config['n_classes']}")
        
        # 验证输入维度
        if config['input_dim'] <= 0:
            raise ValueError(f"输入维度必须 > 0，当前: {config['input_dim']}")
        
        # 验证dropout率
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"dropout率必须在[0,1]范围内，当前: {config['dropout']}")
    
    def _process_input_data(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        处理输入数据，将多模态数据转换为统一的张量格式
        """
        aligned_features = {}
        for key in input_data:
            if key in self.channels_used_in_model and key in self.alignment_layers.modality_names:
                aligned_features[key] = input_data[key]
        aligned_features = self.alignment_layers.forward(aligned_features)
        svd_loss, svd_values = self._compute_rank1_loss_with_metrics(aligned_features)
        h = []
        keys = []
        for channel in self.channels_used_in_model:
            if channel not in aligned_features:
                keys.append(channel)
                h.append(input_data[channel])
            else:
                keys.append(channel+'_aligned')
                h.append(aligned_features[channel])
        h = torch.cat(h, dim=1).squeeze(0)
        return h, svd_loss, svd_values
        
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        """实例级评估（类内注意力分支）"""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.inst_number)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.inst_number, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.inst_number, device)
        n_targets = self.create_negative_targets(self.inst_number, device)
        
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        """实例级评估（类外注意力分支）"""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.inst_number)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.inst_number, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    def forward(self, input_data, label):
        """
        统一的前向传播接口
        
        Args:
            input_data: 输入数据，可以是：
                - torch.Tensor: 单模态特征 [N, D]
                - Dict[str, torch.Tensor]: 多模态数据字典
            **kwargs: 其他参数，支持：
                - label: 标签（用于实例评估）
                - instance_eval: 是否进行实例评估
                - return_features: 是否返回特征
                - attention_only: 是否只返回注意力权重
                
        Returns:
            Dict[str, Any]: 统一格式的结果字典
        """
        # 处理输入数据（支持多模态）
        # align the features
        h, svd_loss, svd_values = self._process_input_data(input_data)
        
        A, h = self.attention_net(h.detach())  # A: [N, 1], h: [N, D]
        A = torch.transpose(A, 1, 0)  # A: [1, N]
        
        if self.attention_only:
            return {'attention_weights': A}
        
        A_raw = A
        A = F.softmax(A, dim=1)  # A: [1, N]
        
        # 计算加权特征
        M = torch.mm(A, h)  # [1, D]
        
        # 分类
        # [1, n_classes]
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        if self.n_classes == 2:
            logits = self.classifiers(M)  # [1, 2]
        else:
            for c in range(self.n_classes):
                logits[0, c] = self.classifiers[c](M)  # [1, 1] independent linear layer for each class
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        # 构建基础结果字典
        result_kwargs = {
            'attention_weights': A_raw,
            'svd_loss': svd_loss,
            'svd_values': svd_values
        }
        # 添加特征
        if self.return_features:
            result_kwargs['features'] = M
        
        # 计算实例损失（如果需要）
        if self.base_weight < 1:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            # inst_labels: [N], like [1, 0] or [0, 0, 1]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in the class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out of the class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
            
            # add the additional loss
            result_kwargs['total_inst_loss'] = total_inst_loss
            result_kwargs['inst_labels'] = np.array(all_targets)
            result_kwargs['inst_preds'] = np.array(all_preds)
        
        # 构建统一的结果字典
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
        
    def _compute_rank1_loss_with_metrics(self, aligned_features: Dict[str, torch.Tensor], 
                                          aligned_negatives: Optional[Dict[str, torch.Tensor]] = None):
        """
        计算 rank1 损失并返回 SVD 特征值（带详细时间分析）
        
        Returns:
            loss: 损失值
            svd_values: SVD 特征值 Tensor[num_modalities]
        """
        # 1. SVD 计算和 loss1
        sorted_keys = sorted(aligned_features.keys())
        feature_list = [aligned_features[mod] for mod in sorted_keys]
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
        svd_values = S.mean(dim=0).detach()  # [num_modalities]
        
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
            sorted_keys = sorted(feat_dict.keys())
            return torch.cat([feat_dict[mod] for mod in sorted_keys], dim=1)

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
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算损失
        
        Args:
            logits: 模型输出的logits
            labels: 真实标签
            result: 包含额外损失的结果字典
            
        Returns:
            总损失值
        """
        if self.base_weight < 1:
            return (self.base_loss_fn(logits, labels) * self.base_weight + 
                   result['total_inst_loss'] * (1 - self.base_weight) + 
                   result['svd_loss'])
        else:
            return self.base_loss_fn(logits, labels)

