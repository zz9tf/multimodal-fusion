import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .clam_mlp_detach import ClamMLP

class UniversalConnections(ClamMLP):
    """
    Universal Connections model

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size ('small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1')
    - dropout: Dropout rate
    - gate: Whether to use gated attention
    - inst_number: Number of positive/negative samples
    - instance_loss_fn: Instance loss function
    - subtyping: Whether it's a subtyping problem
    - views_num: Number of views to generate (M)
    - token_dim: Token dimension (D)
    - inference_depth: Inference depth
    """
    
    def __init__(self, config):
        super().__init__(config)
        # How many views to generate, all target conceptions I need to generate
        self.views_num = config['views_num']          # M
        # Local query, different perspective of the global view
        self.token_dim = config['token_dim']          # D
        # Inference depth
        self.inference_depth = config['inference_depth']

        # Generate view queries (from global awareness)
        self.q_gen = nn.ModuleList([
            nn.Linear(self.token_dim, self.views_num * self.token_dim)
            for _ in range(self.inference_depth)
        ])

        # Only absorb K in scoring
        # Use Wq and Wk as decomposition parameters (equivalent to learning W_score = Wq @ Wk^T)
        self.Wq = nn.ParameterList([
            nn.Parameter(torch.empty(self.token_dim, self.token_dim))
            for _ in range(self.inference_depth)
        ])
        self.Wk = nn.ParameterList([
            nn.Parameter(torch.empty(self.token_dim, self.token_dim))
            for _ in range(self.inference_depth)
        ])

        # value 路径独立
        self.Wv = nn.ParameterList([
            nn.Parameter(torch.empty(self.token_dim, self.token_dim))
            for _ in range(self.inference_depth)
        ])

        # Post-processing (optional)
        self.post = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.token_dim, self.token_dim),
                nn.GELU(),
                nn.Linear(self.token_dim, self.token_dim)
            )
            for _ in range(self.inference_depth)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameter matrices
        """
        for p in list(self.Wq) + list(self.Wk) + list(self.Wv):
            nn.init.xavier_uniform_(p)

    def _generate_global_awareness(self, X):
        """
        Generate global awareness (query / coordinate center)
        
        Args:
            X: [N, D] token feature matrix

        Returns:
            g: [1, D] Global center vector, used to generate views
        """
        # Use mean pooling to generate global center (simple and robust)
        g = X.mean(dim=0, keepdim=True)  # [1, D]
        return g

    def forward(self, input_data, label, drop_prob=None):
        """
        Unified forward propagation interface
        
        Args:
            input_data: Input data, can be:
                - torch.Tensor: Single-modal features [N, D]
                - Dict[str, torch.Tensor]: Multimodal data dictionary
            label: Labels (for instance evaluation)
            drop_prob: Dropout probability (optional)

        Returns:
            torch.Tensor: Fused token features [N', D], where N' = N + M * inference_depth
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        
        # ---- Get features from each modality and concatenate as tokens X ----
        features_dict = {}
        for channel in modalities_used_in_model:
            if channel in ['wsi=features', 'tma=features']:
                clam_result_kwargs = self._clam_forward(channel, input_data[channel], label)
                features_dict[channel] = clam_result_kwargs['features'].detach()
            else:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                features_dict[channel] = self.transfer_layer[channel](input_data[channel])

        # Concatenate tokens on dim=0 (not dim=1), keep feature dimension D
        X = torch.cat([features_dict[k] for k in modalities_used_in_model], dim=0)  # [N, D]
        
        # Ensure X is [N, D]
        assert X.dim() == 2 and X.size(1) == self.token_dim, \
            f"X should be [N, D={self.token_dim}], but got {X.shape}"

        # Generate global center
        g = self._generate_global_awareness(X)  # [1, D]

        # Multi-layer inference
        for d in range(self.inference_depth):
            # 1) Generate M view queries: Q [M, D]
            Q = self.q_gen[d](g).view(self.views_num, self.token_dim)  # [M, D]

            # 2) Absorbed score: S = (QWqWk^T) X^T, equivalent to Q' X^T
            # First absorb Wk into Q's scoring side: Q' = Q Wq Wk^T
            W_score = self.Wq[d] @ self.Wk[d].T                      # [D, D]
            Qp = Q @ W_score                                         # [M, D]

            # score: [M, N]
            S = Qp @ X.T

            # 3) attention weights
            A = F.softmax(S, dim=1)                                  # [M, N] Distribution of each view over N tokens

            # 4) values: V = X Wv
            V = X @ self.Wv[d]                                       # [N, D]

            # 5) Readout: Z = A V -> [M, D] Generate M new tokens/views
            Z = A @ V                                                # [M, D]

            # 6) Post + residual (optional)
            Z = self.post[d](Z) + Z                                  # [M, D]

            # 7) Add new tokens back to the world (grow token set)
            X = torch.cat([X, Z], dim=0)                              # [(N+M), D]

        return X
