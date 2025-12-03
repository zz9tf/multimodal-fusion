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
    Hypergraph网络，用于处理WSI和TMA的embeddings。
    
    参考Multimodal-CustOmics的实现，使用HypergraphConv进行特征聚合。
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
        初始化Hypergraph网络。

        Parameters
        ----------
        input_dim : int
            输入特征维度
        hidden_dims : List[int]
            隐藏层维度列表
        output_dim : int
            输出特征维度
        dropout : float, optional
            Dropout率，默认0.2
        use_attention : bool, optional
            是否在HypergraphConv中使用attention，默认False
        """
        super(HypergraphNetwork, self).__init__()
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # 第一层：特征变换
        self.first_h = Sequential(
            Linear(input_dim, hidden_dims[0]),
            BatchNorm1d(hidden_dims[0]),
            ReLU()
        )
        
        # Hypergraph卷积层
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
        
        # 注意力池化（用于聚合节点特征为graph-level token）
        self.attention_pool = GlobalAttention(
            gate_nn=Sequential(
                Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.Tanh(),
                Linear(hidden_dims[-1] // 2, 1)
            )
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Parameters
        ----------
        x : torch.Tensor
            节点特征，形状 [N, input_dim]
        edge_index : torch.Tensor
            Hypergraph边索引，形状 [2, E]
        batch : torch.Tensor
            批次索引，形状 [N]

        Returns
        -------
        torch.Tensor
            Graph-level token，形状 [batch_size, output_dim]
        """
        # 第一层特征变换
        x = self.first_h(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hypergraph卷积层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.output_layer(x)
        
        # 注意力池化：将节点特征聚合为graph-level token
        graph_token = self.attention_pool(x, batch)  # [batch_size, output_dim]
        
        return graph_token


class CustOmics(ClamMLP):
    """
    CustOmics模型：基于Hypergraph的多模态融合模型。
    
    设计思路：
    1. 将WSI和TMA的embeddings拼接，建立hypergraph，生成hypergraph token
    2. 其他模态经过transfer layer，得到n个tokens
    3. 使用attention MoE聚合hypergraph token和其他模态tokens
    
    配置参数：
    - n_classes: 类别数量
    - input_dim: 输入维度
    - model_size: 模型大小
    - dropout: dropout率
    - hypergraph_hidden_dims: hypergraph网络的隐藏层维度列表
    - hypergraph_dropout: hypergraph网络的dropout率
    """

    def __init__(self, config):
        """
        初始化CustOmics模型，并构建hypergraph网络和融合层。

        Parameters
        ----------
        config : Dict
            模型配置字典，包含父类 `ClamMLP` 所需的所有参数。
            额外支持：
            - hypergraph_hidden_dims (List[int], optional): hypergraph隐藏层维度，默认[256, 256]
            - hypergraph_dropout (float, optional): hypergraph dropout率，默认0.2
            - modality_dropout (float, optional): 模态dropout率，默认0.0
        """
        super().__init__(config)

        # 记录当前模型实际使用到的模态
        self.modality_order = sorted(list(self.used_modality))
        
        # 模态dropout参数
        self.modality_dropout = config.get('modality_dropout', 0.0)
        if not 0.0 <= self.modality_dropout <= 1.0:
            raise ValueError(f"modality_dropout必须在[0.0, 1.0]范围内，当前: {self.modality_dropout}")
        
        # Hypergraph网络配置
        hypergraph_hidden_dims = config.get('hypergraph_hidden_dims', [256, 256])
        hypergraph_dropout = config.get('hypergraph_dropout', 0.2)
        
        # 初始化hypergraph网络（用于处理WSI和TMA）
        # 输入维度：embeddings的维度（通常是input_dim或output_dim）
        # 这里假设embeddings已经对齐到output_dim
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
        1. 提取WSI和TMA的embeddings，拼接后构建hypergraph
        2. 通过hypergraph网络得到hypergraph token
        3. 其他模态经过transfer layer得到tokens
        4. 使用attention MoE聚合所有tokens
        5. 进行最终分类预测

        Parameters
        ----------
        input_data : torch.Tensor | Dict[str, torch.Tensor]
            - 若为 `torch.Tensor`：单模态特征，形状 [N, D]
            - 若为 `Dict[str, torch.Tensor]`：多模态数据字典，key 为模态名称
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
