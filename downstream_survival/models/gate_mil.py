import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate_shared_mil import GateSharedMIL

class GateMIL(GateSharedMIL):
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
        
        self.ChannelFeatureWeightor = nn.ModuleDict({channel: self.ChannelFeatureWeightorCreator() for channel in self.channels_used_in_model})
        # SampleAtt: [channel, simple_number, simple_dim] -> weight [channel, simple_number, 1]
        self.SampleAtt = nn.ModuleDict({channel: self.SampleAttCreator() for channel in self.channels_used_in_model})
        self.TCPClassifier = nn.ModuleDict({channel: self.TCPClassifierCreator() for channel in self.channels_used_in_model})
        self.TCPConfidenceLayer = nn.ModuleDict({channel: self.TCPConfidenceLayerCreator() for channel in self.channels_used_in_model})
    
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
        input_features = self._process_input_data(input_data)
        
        result_kwargs = dict()
        result_kwargs['feature_weight_loss'] = 0
        result_kwargs['confidence_logits_loss'] = 0
        result_kwargs['confidence_loss'] = 0
        
        
        conf_h = torch.zeros(1, len(self.channels_used_in_model)*self.input_dim, device=self.device)
        for i, channel in enumerate(input_features):
            # [N, D] -> [N, D]
            self.FeatureWeight[channel] = self.ChannelFeatureWeightor[channel](input_features[channel])
            input_features[channel] = self.FeatureWeight[channel] * input_features[channel]
            # [N, D] -> [N, 1]
            A = self.SampleAtt[channel](input_features[channel]).T
            # [1, N]*[N, D] -> [1, D]
            # MIL: combined features
            h = torch.mm(A, input_features[channel])
            # [1, D] -> [1, n_classes]
            self.TCPLogits[channel] = self.TCPClassifier[channel](h)
            # [1, D] -> [1, 1]
            self.TCPConfidence[channel] = self.TCPConfidenceLayer[channel](h)
            
            # Confidence weighted combined features
            # input_features[channel]: [1, D]
            input_features[channel] = h * self.TCPConfidence[channel]
            conf_h[:, i*self.input_dim:(i+1)*self.input_dim] = input_features[channel]*self.TCPConfidence[channel]
            result_kwargs['feature_weight_loss'] += torch.mean(self.FeatureWeight[channel])
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
        result_kwargs['confidence_logits_loss'] /= len(self.channels_used_in_model)
        result_kwargs['confidence_loss'] /= len(self.channels_used_in_model)
        
        logits = self.classifiers(conf_h) # [1, n_classes]
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        
        # Build unified result dictionary
        return self._create_result_dict(
            logits=logits,
            probabilities=Y_prob,
            predictions=Y_hat,
            **result_kwargs
        )
