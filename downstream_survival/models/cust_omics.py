import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clam_mlp import ClamMLP
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import HypergraphConv, GlobalAttention
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU


class HypergraphNetwork(nn.Module):
    """
    Hypergraph network for processing WSI and TMA embeddings.

    Refer to Multimodal-CustOmics implementation, using HypergraphConv for feature aggregation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_attention: bool = False
    ):
        """
        Initialize Hypergraph network.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : List[int]
            Hidden layer dimension list
        output_dim : int
            Output feature dimension
        dropout : float, optional
            Dropout rate, default 0.2
        use_attention : bool, optional
            Whether to use attention in HypergraphConv, default False
        """
        super(HypergraphNetwork, self).__init__()
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # First layer: feature transformation
        self.first_h = Sequential(
            Linear(input_dim, hidden_dims[0]),
            BatchNorm1d(hidden_dims[0]),
            ReLU()
        )
        
        # Hypergraph convolution layers
        self.convs = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.convs.append(
                HypergraphConv(
                    hidden_dims[i-1],
                    hidden_dims[i],
                    use_attention=use_attention
                )
            )
        
        # 输出层
        self.output_layer = Linear(hidden_dims[-1], output_dim)
        
        # Attention pooling (for aggregating node features to graph-level token)
        self.attention_pool = GlobalAttention(
            gate_nn=Sequential(
                Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.Tanh(),
                Linear(hidden_dims[-1] // 2, 1)
            )
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [N, input_dim]
        edge_index : torch.Tensor
            Hypergraph edge indices, shape [2, E]
        batch : torch.Tensor
            Batch indices, shape [N]

        Returns
        -------
        torch.Tensor
            Graph-level token, shape [batch_size, output_dim]
        """
        # First layer feature transformation
        x = self.first_h(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hypergraph convolution layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.output_layer(x)
        
        # Attention pooling: aggregate node features to graph-level token
        graph_token = self.attention_pool(x, batch)  # [batch_size, output_dim]
        
        return graph_token


class CustOmics(ClamMLP):
    """
    CustOmics model: Hypergraph-based multimodal fusion model.

    Design concept:
    1. Concatenate WSI and TMA embeddings, establish hypergraph, generate hypergraph token
    2. Other modalities go through transfer layer to get n tokens
    3. Use attention MoE to aggregate hypergraph token and other modality tokens

    Configuration parameters:
    - n_classes: Number of classes
    - input_dim: Input dimension
    - model_size: Model size
    - dropout: Dropout rate
    - hypergraph_hidden_dims: Hypergraph network hidden layer dimension list
    - hypergraph_dropout: Hypergraph network dropout rate
    """

    def __init__(self, config):
        """
        Initialize CustOmics model and build hypergraph network and fusion layers.

        Parameters
        ----------
        config : Dict
            Model configuration dictionary, containing all parameters required by parent class `ClamMLP`.
            Additional support:
            - hypergraph_hidden_dims (List[int], optional): hypergraph hidden layer dimensions, default [256, 256]
            - hypergraph_dropout (float, optional): hypergraph dropout rate, default 0.2
            - modality_dropout (float, optional): modality dropout rate, default 0.0
        """
        super().__init__(config)

        # Record the modalities actually used by the current model
        self.modality_order = sorted(list(self.used_modality))
        
        # Modality dropout parameter
        self.modality_dropout = config.get('modality_dropout', 0.0)
        if not 0.0 <= self.modality_dropout <= 1.0:
            raise ValueError(f"modality_dropout must be in [0.0, 1.0] range, current: {self.modality_dropout}")
        
        # Hypergraph network configuration
        hypergraph_hidden_dims = config.get('hypergraph_hidden_dims', [256, 256])
        hypergraph_dropout = config.get('hypergraph_dropout', 0.2)
        
        # Initialize hypergraph network (for processing WSI and TMA)
        # Input dimension: dimension of embeddings (usually input_dim or output_dim)
        # Here we assume embeddings are already aligned to output_dim
        self.hypergraph_net = HypergraphNetwork(
            input_dim=self.output_dim,
            hidden_dims=hypergraph_hidden_dims,
            output_dim=self.output_dim,
            dropout=hypergraph_dropout,
            use_attention=False
        )
        
        # 其他模态（非WSI/TMA）的数量
        self.other_modalities = [
            m for m in self.modality_order 
            if m not in ['wsi=features', 'tma=features']
        ]
        num_other_modalities = len(self.other_modalities)
        
        # Attention MoE融合层：聚合hypergraph token和其他模态tokens
        # 最大可能的token数量：1个hypergraph token + num_other_modalities个其他模态tokens
        # 使用动态方式，根据实际token数量创建gating network
        self.max_num_tokens = 1 + num_other_modalities
        # 使用一个通用的gating network，输出维度为最大token数
        # 实际使用时根据token数量进行截取
        self.moe_gating_network = nn.Sequential(
            nn.Linear(self.output_dim, self.max_num_tokens),
            nn.Softmax(dim=-1)
        )
        
        # 最终分类层
        self.fusion_prediction_layer = nn.Linear(self.output_dim, self.n_classes)
        
    def _build_hypergraph_edge_index(
        self,
        num_nodes: int,
        method: str = 'fully_connected'
    ) -> torch.Tensor:
        """
        构建hypergraph的edge_index。
        
        目前支持全连接方式，每个节点都连接到所有其他节点（包括自己）。

        Parameters
        ----------
        num_nodes : int
            节点数量（WSI + TMA embeddings的总数）
        method : str, optional
            构建方法，默认'fully_connected'

        Returns
        -------
        torch.Tensor
            edge_index，形状 [2, num_edges]
        """
        if method == 'fully_connected':
            # 全连接：每个节点连接到所有节点（包括自己）
            # 对于hypergraph，我们创建完全二部图结构
            # 每个节点作为一个hyperedge，连接到所有节点
            edge_list = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edge_list.append([i, j])
            
            if len(edge_list) == 0:
                # 如果没有节点，返回空的edge_index
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
            return edge_index
        else:
            raise ValueError(f"不支持的edge_index构建方法: {method}")
    
    def forward(self, input_data, label):
        """
        执行前向传播，使用hypergraph处理WSI/TMA，然后与其他模态进行MoE融合。

        融合流程：
        1. 优先使用预处理好的hypergraph数据（如果存在）
        2. 否则提取WSI和TMA的embeddings，拼接后构建hypergraph
        3. 通过hypergraph网络得到hypergraph token
        4. 其他模态经过transfer layer得到tokens
        5. 使用attention MoE聚合所有tokens
        6. 进行最终分类预测

        Parameters
        ----------
        input_data : torch.Tensor | Dict[str, torch.Tensor]
            - 若为 `torch.Tensor`：单模态特征，形状 [N, D]
            - 若为 `Dict[str, torch.Tensor]`：多模态数据字典，key 为模态名称
            支持预处理好的hypergraph数据：
            - 'hypergraph=wsi_super_features': WSI super patch features
            - 'hypergraph=tma_features': TMA features
            - 'hypergraph=edge_index': Hypergraph edge indices
            - 'hypergraph=edge_weights': Edge weights (optional)
        label : torch.Tensor
            标签张量，用于实例评估，形状通常为 [1]

        Returns
        -------
        Dict[str, torch.Tensor]
            统一格式的结果字典，至少包含：
            - 'Y_prob' : [1, n_classes]，类别概率
            - 'Y_hat'  : [1, 1]，预测类别索引
            以及各模态的 CLAM 相关结果（如果适用）。
        """
        input_data, modalities_used_in_model = self._process_input_data(input_data)
        result_kwargs = {}
        
        # ========== 步骤1: 处理WSI和TMA，构建hypergraph ==========
        # 优先检查是否有预处理好的hypergraph数据
        use_preprocessed_hypergraph = (
            'hypergraph=wsi_super_features' in input_data and
            'hypergraph=edge_index' in input_data
        )
        
        if use_preprocessed_hypergraph:
            # 使用预处理好的hypergraph数据
            wsi_super_features = input_data['hypergraph=wsi_super_features']  # [N_wsi_super, D]
            edge_index = input_data['hypergraph=edge_index']  # [2, E]
            
            # 处理维度
            if wsi_super_features.dim() == 1:
                wsi_super_features = wsi_super_features.unsqueeze(0)
            elif wsi_super_features.dim() == 3:
                wsi_super_features = wsi_super_features.squeeze(0)
            
            # 确保维度对齐
            if wsi_super_features.shape[1] != self.output_dim:
                if 'hypergraph_transfer' not in self.__dict__:
                    self.hypergraph_transfer = nn.Linear(wsi_super_features.shape[1], self.output_dim).to(self.device)
                wsi_super_features = self.hypergraph_transfer(wsi_super_features)
            
            # 如果有TMA features，也加入hypergraph nodes
            if 'hypergraph=tma_features' in input_data:
                tma_features = input_data['hypergraph=tma_features']
                if tma_features.dim() == 1:
                    tma_features = tma_features.unsqueeze(0)
                elif tma_features.dim() == 3:
                    tma_features = tma_features.squeeze(0)
                
                # 对齐维度
                if tma_features.shape[1] != self.output_dim:
                    if 'hypergraph_tma_transfer' not in self.__dict__:
                        self.hypergraph_tma_transfer = nn.Linear(tma_features.shape[1], self.output_dim).to(self.device)
                    tma_features = self.hypergraph_tma_transfer(tma_features)
                
                # 拼接WSI super patches和TMA
                hypergraph_nodes = torch.cat([wsi_super_features, tma_features], dim=0)
            else:
                hypergraph_nodes = wsi_super_features
            
            # 确保edge_index在正确的device上
            edge_index = edge_index.to(self.device)
            
            # 创建batch索引（所有节点属于同一个graph）
            num_nodes = hypergraph_nodes.shape[0]
            batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
            
            # 通过hypergraph网络得到graph-level token
            hypergraph_token = self.hypergraph_net(hypergraph_nodes, edge_index, batch)  # [1, output_dim]
        else:
            # 回退到原来的方法：从原始WSI/TMA features构建hypergraph
            image_embeddings = []  # 存储WSI和TMA的embeddings
            
            if 'wsi=features' in modalities_used_in_model and 'wsi=features' in input_data:
                wsi_emb = input_data['wsi=features']  # [N_wsi, D] 或 [1, N_wsi, D] 或 [N_wsi]
                # 处理不同维度
                if wsi_emb.dim() == 1:
                    wsi_emb = wsi_emb.unsqueeze(0)  # [1, D]
                elif wsi_emb.dim() == 3:
                    wsi_emb = wsi_emb.squeeze(0)  # [N_wsi, D]
                # wsi_emb现在是 [N_wsi, D]
                image_embeddings.append(wsi_emb)
            
            if 'tma=features' in modalities_used_in_model and 'tma=features' in input_data:
                tma_emb = input_data['tma=features']  # [N_tma, D] 或 [1, N_tma, D] 或 [N_tma]
                # 处理不同维度
                if tma_emb.dim() == 1:
                    tma_emb = tma_emb.unsqueeze(0)  # [1, D]
                elif tma_emb.dim() == 3:
                    tma_emb = tma_emb.squeeze(0)  # [N_tma, D]
                # tma_emb现在是 [N_tma, D]
                image_embeddings.append(tma_emb)
            
            # 拼接WSI和TMA embeddings
            if len(image_embeddings) > 0:
                # 确保所有embeddings维度一致
                # 如果维度不一致，使用transfer layer对齐
                aligned_embeddings = []
                for emb in image_embeddings:
                    if emb.shape[1] != self.output_dim:
                        # 需要对齐维度
                        if 'image_transfer' not in self.__dict__:
                            self.image_transfer = nn.Linear(emb.shape[1], self.output_dim).to(self.device)
                        emb = self.image_transfer(emb)
                    aligned_embeddings.append(emb)
                
                # 拼接所有image embeddings
                hypergraph_nodes = torch.cat(aligned_embeddings, dim=0)  # [N_total, output_dim]
                num_nodes = hypergraph_nodes.shape[0]
                
                # 构建hypergraph edge_index
                edge_index = self._build_hypergraph_edge_index(num_nodes, method='fully_connected')
                
                # 创建batch索引（所有节点属于同一个graph）
                batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                
                # 通过hypergraph网络得到graph-level token
                hypergraph_token = self.hypergraph_net(hypergraph_nodes, edge_index, batch)  # [1, output_dim]
            else:
                # 如果没有WSI和TMA，创建零向量
                hypergraph_token = torch.zeros(1, self.output_dim, device=self.device)
        
        # ========== 步骤2: 处理其他模态，得到tokens ==========
        other_modality_tokens = []
        for channel in modalities_used_in_model:
            if channel not in ['wsi=features', 'tma=features']:
                if channel not in self.transfer_layer:
                    self.transfer_layer[channel] = self.create_transfer_layer(input_data[channel].shape[1])
                token = self.transfer_layer[channel](input_data[channel])  # [1, output_dim]
                other_modality_tokens.append(token)
        
        # ========== 步骤3: Attention MoE融合 ==========
        # 收集所有tokens：hypergraph token + 其他模态tokens
        all_tokens = []
        
        # 添加hypergraph token（如果存在）
        if len(image_embeddings) > 0:
            all_tokens.append(hypergraph_token)
        
        # 添加其他模态tokens
        all_tokens.extend(other_modality_tokens)
        
        if len(all_tokens) == 0:
            # 如果没有tokens，使用零向量
            fused_token = torch.zeros(1, self.output_dim, device=self.device)
            moe_weights = None
        else:
            # 拼接所有tokens: [num_tokens, output_dim]
            tokens_tensor = torch.cat(all_tokens, dim=0)  # [num_tokens, output_dim]
            num_tokens = tokens_tensor.shape[0]
            
            # 计算MoE注意力权重
            # 使用平均池化得到全局表示来计算权重
            token_mean = tokens_tensor.mean(dim=0, keepdim=True)  # [1, output_dim]
            moe_weights_raw = self.moe_gating_network(token_mean)  # [1, max_num_tokens]
            
            # 根据实际token数量截取权重
            if num_tokens <= self.max_num_tokens:
                moe_weights = moe_weights_raw[:, :num_tokens]  # [1, num_tokens]
            else:
                # 如果实际token数量超过预期，动态扩展gating network
                # 这里使用简单的平均权重作为fallback
                moe_weights = torch.ones(1, num_tokens, device=self.device) / num_tokens
                moe_weights = F.softmax(moe_weights, dim=-1)
            
            # 归一化权重（确保和为1）
            moe_weights = moe_weights / (moe_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 加权聚合
            fused_token = torch.sum(moe_weights * tokens_tensor, dim=0, keepdim=True)  # [1, output_dim]
        
        # ========== 步骤4: 最终预测 ==========
        logits = self.fusion_prediction_layer(fused_token)  # [1, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # 更新结果字典
        result_kwargs['Y_prob'] = Y_prob
        result_kwargs['Y_hat'] = Y_hat
        if moe_weights is not None:
            result_kwargs['moe_weights'] = moe_weights
        
        return self._create_result_dict(logits, Y_prob, Y_hat, **result_kwargs)
