import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from typing import Dict, List, Tuple
from libauc.losses import AUCMLoss

class Attn_Net(nn.Module):
    """Attention network (without gating)"""
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
    """Gated attention network"""
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

class AUC_CLAM(BaseModel):
    """
    CLAM model

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
        
        # Validate configuration completeness
        self._validate_config(config)
        
        # Model size configuration
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
            raise ValueError(f"Unsupported instance loss function: {config.get('inst_loss_fn')}")

        self.model_size = config['model_size']
        self.subtyping = config.get('subtyping', False)
        self.inst_number = config.get('inst_number', 8)
        self.channels_used_in_model = config['channels_used_in_model']
        self.return_features = config.get('return_features', False)
        self.attention_only = config.get('attention_only', False)
        self.auc_loss_weight = config.get('auc_loss_weight', 1.0)
        self.auc_loss_fn = AUCMLoss(margin=1.0)
        
        size = self.size_dict[self.model_size]
        
        # Build feature extraction layer
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(self.dropout)]
        
        # Build attention network (single branch: output 1 attention value)
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
        
        # Build classifier
        self.classifiers = None
        if self.n_classes == 2:
            self.classifiers = nn.Linear(size[1], self.n_classes)
        else:
            self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(self.n_classes)])
        
        # Instance classifier
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(self.n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
 
    def _validate_config(self, config):
        """Validate configuration completeness"""
        required_params = ['n_classes', 'input_dim', 'model_size', 'dropout']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"CLAM_SB configuration missing required parameters: {missing_params}")
        
        # Validate model size
        valid_sizes = ["small", "big", "128*64", "64*32", "32*32", "16*8", "8*4", "4*2", "2*1"]
        if config['model_size'] not in valid_sizes:
            raise ValueError(f"Unsupported model size: {config['model_size']}, supported sizes: {valid_sizes}")
        
        # Validate number of classes
        if config['n_classes'] < 2:
            raise ValueError(f"Number of classes must be >= 2, current: {config['n_classes']}")
        
        # Validate input dimension
        if config['input_dim'] <= 0:
            raise ValueError(f"Input dimension must be > 0, current: {config['input_dim']}")
        
        # Validate dropout rate
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"Dropout rate must be in [0,1] range, current: {config['dropout']}")
    
    def _process_input_data(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process input data, convert multimodal data to unified tensor format
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
        """Instance-level evaluation (in-class attention branch)"""
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
        """Instance-level evaluation (out-of-class attention branch)"""
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
        Unified forward propagation interface

        Args:
            input_data: Input data, can be:
                - torch.Tensor: Single-modal features [N, D]
                - Dict[str, torch.Tensor]: Multimodal data dictionary
            **kwargs: Other parameters, support:
                - label: Labels (for instance evaluation)
                - instance_eval: Whether to perform instance evaluation
                - return_features: Whether to return features
                - attention_only: Whether to return only attention weights

        Returns:
            Dict[str, Any]: Unified format result dictionary
        """
        # Process input data (support multimodal)
        h = self._process_input_data(input_data)
        A, h = self.attention_net(h)  # A: [N, 1], h: [N, D]
        A = torch.transpose(A, 1, 0)  # A: [1, N]
        
        if self.attention_only:
            return {'attention_weights': A}
        
        A_raw = A
        A = F.softmax(A, dim=1)  # A: [1, N]
        
        # Calculate weighted features
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
        # Build basic result dictionary
        result_kwargs = {
            'attention_weights': A_raw
        }
        # Add features
        if self.return_features:
            result_kwargs['features'] = M
        
        # Calculate instance loss (if needed)
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
            
            # Add the additional loss
            result_kwargs['total_inst_loss'] = total_inst_loss
            result_kwargs['inst_labels'] = np.array(all_targets)
            result_kwargs['inst_preds'] = np.array(all_preds)
        
        # Build unified result dictionary
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate loss
        """
        if not hasattr(self, 'group_logits'):
            self.group_logits = []
        if not hasattr(self, 'group_labels'):
            self.group_labels = []
        self.group_logits.append(logits[:, 1] - logits[:, 0])
        self.group_labels.append(labels)
        
        if self.base_weight < 1:
            return self.base_loss_fn(logits, labels)*self.base_weight + result['total_inst_loss']*(1-self.base_weight)
        else:
            return self.base_loss_fn(logits, labels)
        
    def group_loss_fn(self, result: Dict[str, float]) -> torch.Tensor:
        """
        Calculate group loss
        """
        logits = torch.cat(self.group_logits, dim=0).to(self.device)
        targets = torch.cat(self.group_labels, dim=0).to(self.device)
        self.group_logits = []
        self.group_labels = []
        loss = self.auc_loss_fn(logits, targets).squeeze()
        if not hasattr(self, 'epoch_auc_loss'):
            self.epoch_auc_loss = []
        self.epoch_auc_loss.append(loss.item())
        return loss

    def verbose_items(self, result: Dict[str, float]) -> List[Tuple[str, float]]:
        if 'auc_loss' not in result:
            return []
        if 'is_epoch' in result:
            epoch_auc_loss = sum(self.epoch_auc_loss)/len(self.epoch_auc_loss)
            self.epoch_auc_loss = []
            return [
                ('auc_loss', epoch_auc_loss)
            ]
        return [
            ('auc_loss', result['auc_loss'])
        ]
