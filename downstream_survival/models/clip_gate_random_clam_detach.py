import torch
import torch.nn as nn
import torch.nn.functional as F
from .svd_gate_random_clam_detach import SVDGateRandomClamDetach
from .clip_gate_random_clam import ClipGateRandomClam
from typing import Dict, List, Tuple

class ClipGateRandomClamDetach(SVDGateRandomClamDetach, ClipGateRandomClam):
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
        ClipGateRandomClam.__init__(self, config)
    
    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        计算组损失
        """
        return ClipGateRandomClam.group_loss_fn(self, result)
    
    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        打印结果
        """
        return ClipGateRandomClam.verbose_items(self, result)