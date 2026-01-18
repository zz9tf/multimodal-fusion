import torch
import torch.nn as nn
import torch.nn.functional as F
from .svd_gate_random_clam import SVDGateRandomClam
from typing import Dict, List, Tuple

class ClipGateRandomClam(SVDGateRandomClam):
    """
    CLAM MLP 模型
    
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
        
        self.enable_dynamic_gate = config.get('enable_dynamic_gate', True)
        self.enable_clip = config.get('enable_clip', True)
        
        if self.enable_dynamic_gate:
            self._init_dynamic_gated_model()
        if self.enable_clip:
            self.alignment_layer_num = config.get('alignment_layer_num', 2)
            # CLIP 对齐：固定以 0 号模态为锚点，无缺失模态
            self.clip_anchor_idx = -1
            init_tau = float(config.get('clip_init_tau', 0.07))
            self.clip_logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_tau)))
            # 依赖父类提供的 alignment_layers（SVDGateRandomClam._init_svd_model）
        self.enable_random_loss = config.get('enable_random_loss', True)
        self.weight_random_loss = config.get('weight_random_loss', 0.1)

    def _compute_clip_loss_with_metrics(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算 CLIP 损失（无缺失模态，全部相对 0 号模态计算）。
        
        Args:
            features: Tensor，形状 [batch_size, feature_dim, num_modalities]
                要求按模态堆叠在最后一维。
        Returns:
            (loss, diag_sim):
                - loss: 标量损失
                - diag_sim: 平均对角相似度（用于简单监控）
        """
        # 形状检查与解包
        assert features.dim() == 3, "features 应为 [B, D, M]"
        B, D, M = features.shape
        assert M >= 2, "至少需要 2 个模态"
        
        # 温度 tau 与 L2 归一化到通道维（特征维）
        # logit_scale = 1/tau，因此 tau = exp(-logit_scale)
        tau = torch.exp(-self.clip_logit_scale)
        features = F.normalize(features, dim=1)
        
        # 锚点模态：0 号
        anchor = features[:, :, self.clip_anchor_idx]  # [B, D]
        # 逐个与其余模态做双向 CE，并对模态取平均
        loss_total = anchor.new_tensor(0.0)
        diag_sim_total = anchor.new_tensor(0.0)
        cnt = 0
        for m in range(M):
            if m == self.clip_anchor_idx:
                continue
            other = features[:, :, m]  # [B, D]
            # logits
            logits_xy = (anchor @ other.t()) / tau
            logits_yx = (other @ anchor.t()) / tau
            target = torch.arange(B, device=anchor.device)
            loss_m = F.cross_entropy(logits_xy, target) + F.cross_entropy(logits_yx, target)
            loss_total = loss_total + loss_m
            # 记录对角相似度（未除温度）
            diag_sim_total = diag_sim_total + torch.mean(torch.sum(anchor * other, dim=1))
            cnt += 1
        
        if cnt == 0:
            return {
                'clip_loss': anchor.new_tensor(0.0, requires_grad=True),
                'clip_diag_sim': anchor.new_tensor(0.0),
            }
        
        return loss_total / cnt, diag_sim_total / cnt

    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        计算组损失
        """
        if not self.enable_clip:
            return 0.0
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
        clip_loss, clip_diag_sim = self._compute_clip_loss_with_metrics(features)
        result['clip_loss'] = clip_loss
        result['clip_diag_sim'] = clip_diag_sim
        return clip_loss

    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        打印结果
        """
        verbose_list = []
        for key, value in result.items():
            if key.endswith('_loss'):
                verbose_list.append((key, value))
            elif key.endswith('_clip_diag_sim'):
                verbose_list.append((key, value))
        return verbose_list