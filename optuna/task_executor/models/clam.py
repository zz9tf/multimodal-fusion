import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from typing import Dict

class Attn_Net(nn.Module):
    """注意力网络（无门控）"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
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
        return self.module(x), x

class Attn_Net_Gated(nn.Module):
    """门控注意力网络"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
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
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x

class CLAM(BaseModel):
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
        h = torch.cat([input_data[channel] for channel in self.channels_used_in_model], dim=1).squeeze(0)
        return h
        
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
        h = self._process_input_data(input_data)
        A, h = self.attention_net(h)  # A: [N, 1], h: [N, D]
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
            logits = self.classifiers(M) # [1, 2]
        else:
            for c in range(self.n_classes):
                logits[0, c] = self.classifiers[c](M[c]) # [1, 1] independent linear layer for each class
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        # 构建基础结果字典
        result_kwargs = {
            'attention_weights': A_raw
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
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        计算损失
        """
        if self.base_weight < 1:
            return self.base_loss_fn(logits, labels)*self.base_weight + result['total_inst_loss']*(1-self.base_weight)
        else:
            return self.base_loss_fn(logits, labels)
        
    def verbose_item(self, result: Dict[str, float]) -> str:
        """
        打印详细信息
        """
        return f"total_inst_loss: {result['total_inst_loss']:.4f}, base_weight: {self.base_weight:.4f}"

