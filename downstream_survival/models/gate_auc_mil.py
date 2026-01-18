import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate_shared_mil import GateSharedMIL
from typing import Dict, List, Tuple
from libauc.losses import AUCMLoss

class GateAUCMIL(GateSharedMIL):
    """
    GatedGTECLAM model

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: Dropout rate
    - gate: Whether to use gated attention
    - inst_number: Number of positive/negative samples
    - instance_loss_fn: Instance loss function
    - subtyping: Whether it's a subtyping problem
    - shared_gated: Whether to share gated attention
    - use_auc_loss: Whether to use AUC loss
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.auc_loss = AUCMLoss(margin=1.0)
        self.auc_loss_weight = config.get('auc_loss_weight', 1.0)
        
        self.ChannelFeatureWeightor = nn.ModuleDict({channel: self.ChannelFeatureWeightorCreator() for channel in self.channels_used_in_model})
        # SampleAtt: [channel, simple_number, simple_dim] -> weight [channel, simple_number, 1]
        self.SampleAtt = nn.ModuleDict({channel: self.SampleAttCreator() for channel in self.channels_used_in_model})
        self.TCPClassifier = nn.ModuleDict({channel: self.TCPClassifierCreator() for channel in self.channels_used_in_model})
        self.TCPConfidenceLayer = nn.ModuleDict({channel: self.TCPConfidenceLayerCreator() for channel in self.channels_used_in_model})
    
    def forward(self, input_data, label=None, return_features=False, **kwargs):
        """
        Unified forward propagation interface

        Args:
            input_data: Input data, can be:
                - torch.Tensor: Single-modal features [N, D]
                - Dict[str, torch.Tensor]: Multimodal data dictionary
            label: Labels (for instance evaluation, optional)
            return_features: Whether to return feature information
            **kwargs: Other parameters, support:
                - instance_eval: Whether to perform instance evaluation

        Returns:
            Dict[str, Any]: Unified format result dictionary
        """
        # Process input data (support multimodal)
        input_features = self._process_input_data(input_data)
        
        result_kwargs = dict()
        result_kwargs['feature_weight_loss'] = 0
        result_kwargs['confidence_logits_loss'] = 0
        result_kwargs['confidence_loss'] = 0
        
        # Initialize feature storage dictionary (if need to return features)
        if return_features:
            result_kwargs['raw_features'] = {}  # Raw input features
            result_kwargs['weighted_features'] = {}  # Feature-weighted features
            result_kwargs['attention_weights'] = {}  # Attention weights
            result_kwargs['combined_features'] = {}  # MIL aggregated features
            result_kwargs['confidence_scores'] = {}  # Confidence scores
            result_kwargs['feature_weights'] = {}  # Feature weights
        
        conf_h = torch.zeros(1, len(self.channels_used_in_model)*self.input_dim, device=self.device)
        for i, channel in enumerate(input_features):
            # Save raw features (if needed)
            if return_features or attention_only:
                result_kwargs['raw_features'][channel] = input_features[channel].clone()
            
            # [N, D] -> [N, D]
            self.FeatureWeight[channel] = self.ChannelFeatureWeightor[channel](input_features[channel])
            input_features[channel] = self.FeatureWeight[channel] * input_features[channel]
            
            # Save weighted features and feature weights (if needed)
            if return_features or attention_only:
                result_kwargs['weighted_features'][channel] = input_features[channel].clone()
                result_kwargs['feature_weights'][channel] = self.FeatureWeight[channel].clone()
            
            # [N, D] -> [N, 1]
            A = self.SampleAtt[channel](input_features[channel]).T
            
            # Save attention weights (if needed)
            if return_features:
                result_kwargs['attention_weights'][channel] = A.clone()
            
            # [1, N]*[N, D] -> [1, D]
            # MIL: combined features
            h = torch.mm(A, input_features[channel])
            
            # Save aggregated features (if needed)
            if return_features:
                result_kwargs['combined_features'][channel] = h.clone()
            
            # [1, D] -> [1, n_classes]
            self.TCPLogits[channel] = self.TCPClassifier[channel](h)
            # [1, D] -> [1, 1]
            self.TCPConfidence[channel] = self.TCPConfidenceLayer[channel](h)
            
            # Save confidence scores (if needed)
            if return_features:
                result_kwargs['confidence_scores'][channel] = self.TCPConfidence[channel].clone()
            
            # Confidence weighted combined features
            # input_features[channel]: [1, D]
            input_features[channel] = h * self.TCPConfidence[channel]
            conf_h[:, i*self.input_dim:(i+1)*self.input_dim] = input_features[channel]*self.TCPConfidence[channel]
            result_kwargs['feature_weight_loss'] += torch.mean(self.FeatureWeight[channel])
            
            # Only calculate loss when label is available
            if label is not None:
                # pred: [1, n_classes]
                pred = F.softmax(self.TCPLogits[channel], dim = 1)
                # p_target: [1]
                p_target = torch.gather(pred, 1, label.unsqueeze(1)).view(-1)
                # confidence -> TCPLogits & TCPLogits -> labels
                logits_loss = torch.mean(self.TCPLogitsLoss_fn(self.TCPLogits[channel], label))
                confidence_loss = torch.mean(self.TCPConfidenceLoss_fn(self.TCPConfidence[channel].view(-1), p_target))
                
                result_kwargs['confidence_logits_loss'] += logits_loss
                result_kwargs['confidence_loss'] += confidence_loss
            
        result_kwargs['feature_weight_loss'] /= len(self.channels_used_in_model)
        if label is not None:
            result_kwargs['confidence_logits_loss'] /= len(self.channels_used_in_model)
            result_kwargs['confidence_loss'] /= len(self.channels_used_in_model)
        
        logits = self.classifiers(conf_h) # [1, n_classes]
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        # If only attention weights are needed, return directly
        if attention_only:
            return {
                'attention_weights': result_kwargs['attention_weights'],
                'feature_weights': result_kwargs['feature_weights'],
                'confidence_scores': result_kwargs['confidence_scores']
            }
        
        # Build unified result dictionary
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, result: Dict[str, float]) -> torch.Tensor:
        """
        Loss function
        """
        if 'group_predictions' not in result:
            result['group_predictions'] = []
        if 'group_labels' not in result:
            result['group_labels'] = []
        result['group_predictions'].append(result['predictions'])
        result['group_labels'].append(labels)
        
        return super().loss_fn(logits, labels, result)
    
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
    
    def extract_features(self, input_data):
        """
        Convenient feature extraction method

        Args:
            input_data: Input data

        Returns:
            Dict[str, torch.Tensor]: Feature dictionary
        """
        with torch.no_grad():
            result = self.forward(input_data, return_features=True)
            
            return result
    
    