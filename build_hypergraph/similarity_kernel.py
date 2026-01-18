"""
Similarity kernel computation for hypergraph construction.

Based on the paper formula:
- κ_h(x_i, x_j) = e^(-λ_h ||h_i - h_j||^2)  (morphological similarity)
- κ_g(x_i, x_j) = e^(-λ_g ||g_i - g_j||^2)  (spatial similarity)
- κ(x_i, x_j) = κ_h * κ_g  (combined similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def compute_morphological_similarity(
    features: torch.Tensor,
    lambda_h: float = 1.0
) -> torch.Tensor:
    """
    Compute morphological similarity matrix K_h based on feature embeddings.
    
    Formula: κ_h(x_i, x_j) = e^(-λ_h ||h_i - h_j||^2)
    
    Parameters
    ----------
    features : torch.Tensor
        Feature embeddings, shape [N, D] where N is number of patches, D is feature dimension
    lambda_h : float, optional
        Scaling parameter for morphological similarity, default 1.0
    
    Returns
    -------
    torch.Tensor
        Morphological similarity matrix K_h, shape [N, N]
    """
    # Compute pairwise squared L2 distances: ||h_i - h_j||^2
    # Using broadcasting: (h_i - h_j)^2 = h_i^2 + h_j^2 - 2*h_i*h_j
    # More efficient: ||h_i - h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2*h_i^T*h_j
    
    # Compute squared norms for each feature vector
    features_norm_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # [N, 1]
    
    # Compute pairwise dot products
    dot_products = torch.mm(features, features.t())  # [N, N]
    
    # Compute squared distances: ||h_i - h_j||^2
    squared_distances = features_norm_sq + features_norm_sq.t() - 2 * dot_products  # [N, N]
    
    # Compute similarity: κ_h = e^(-λ_h * ||h_i - h_j||^2)
    K_h = torch.exp(-lambda_h * squared_distances)
    
    return K_h


def compute_spatial_similarity(
    positions: torch.Tensor,
    lambda_g: float = 1.0
) -> torch.Tensor:
    """
    Compute spatial similarity matrix K_g based on patch positions.
    
    Formula: κ_g(x_i, x_j) = e^(-λ_g ||g_i - g_j||^2)
    
    Parameters
    ----------
    positions : torch.Tensor
        Spatial coordinates of patches, shape [N, 2] or [N, 3] (x, y) or (x, y, z)
    lambda_g : float, optional
        Scaling parameter for spatial similarity, default 1.0
    
    Returns
    -------
    torch.Tensor
        Spatial similarity matrix K_g, shape [N, N]
    """
    # Compute pairwise squared L2 distances: ||g_i - g_j||^2
    positions_norm_sq = torch.sum(positions ** 2, dim=1, keepdim=True)  # [N, 1]
    dot_products = torch.mm(positions, positions.t())  # [N, N]
    squared_distances = positions_norm_sq + positions_norm_sq.t() - 2 * dot_products  # [N, N]
    
    # Compute similarity: κ_g = e^(-λ_g * ||g_i - g_j||^2)
    K_g = torch.exp(-lambda_g * squared_distances)
    
    return K_g

def compute_combined_similarity(
    features: torch.Tensor,
    positions: torch.Tensor,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0
) -> torch.Tensor:
    """
    Compute combined similarity matrix by multiplying morphological and spatial similarities.
    
    Formula: κ(x_i, x_j) = κ_h(h_i, h_j) * κ_g(g_i, g_j)
    
    Parameters
    ----------
    features : torch.Tensor
        Feature embeddings, shape [N, D]
    positions : torch.Tensor
        Spatial coordinates, shape [N, 2] or [N, 3]
    lambda_h : float, optional
        Scaling parameter for morphological similarity, default 1.0
    lambda_g : float, optional
        Scaling parameter for spatial similarity, default 1.0
    
    Returns
    -------
    torch.Tensor
        Combined similarity matrix K, shape [N, N]
    """
    # Compute morphological similarity
    K_h = compute_morphological_similarity(features, lambda_h)
    
    # Compute spatial similarity
    K_g = compute_spatial_similarity(positions, lambda_g)
    
    # Combine: κ = κ_h * κ_g
    K = K_h * K_g
    
    return K

def build_weighted_hypergraph(
    features: torch.Tensor,
    positions: torch.Tensor,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    threshold_median_ratio: float = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build hypergraph with edge weights based on combined similarity.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature embeddings, shape [N, D]
    positions : torch.Tensor
        Spatial coordinates, shape [N, 2] or [N, 3]
    lambda_h : float, optional
        Scaling parameter for morphological similarity, default 1.0
    lambda_g : float, optional
        Scaling parameter for spatial similarity, default 1.0
    threshold_median_ratio : float, optional
        Threshold as a ratio of the median similarity. 
        If specified, threshold = median(K) * threshold_median_ratio.
        This allows adaptive thresholding based on the similarity distribution.
        Example: threshold_median_ratio=0.5 means threshold is half of the median similarity.
        If None, all edges are kept.
    device : torch.device, optional
        Device to place tensors on
    
    Returns
    -------
    edge_index : torch.Tensor
        Edge indices, shape [2, E] where E is number of edges
    edge_weights : torch.Tensor
        Edge weights (similarity scores), shape [E]
    """
    if device is None:
        device = features.device
    
    # Ensure tensors are on the same device
    features = features.to(device)
    positions = positions.to(device)
    
    # Compute combined similarity matrix
    K = compute_combined_similarity(features, positions, lambda_h, lambda_g)
    
    N = K.shape[0]
    
    # Verify that we have more than one node
    if N <= 1:
        raise ValueError(f"Number of nodes must be greater than 1, got N={N}. "
                        f"Hypergraph construction requires at least 2 nodes.")
    
    # Compute adaptive threshold based on median
    # Exclude diagonal (self-similarity = 1.0) for more accurate median calculation
    # Diagonal elements are always 1.0 (self-similarity), so we exclude them
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    K_off_diag = K[mask]  # Get all off-diagonal elements: shape [N*(N-1)]
    # Compute median of off-diagonal similarities
    median_sim = torch.median(K_off_diag).item()
    # Set threshold as ratio of median
    threshold = median_sim * threshold_median_ratio
    # Build edge list and weights
    edge_list = []
    weight_list = []
    
    for i in range(N):
        for j in range(N):
            similarity = K[i, j].item()
            
            # Apply threshold if specified
            if threshold is not None and similarity < threshold:
                continue
            
            edge_list.append([i, j])
            weight_list.append(similarity)
    
    # Convert to tensors
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weights = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        edge_weights = torch.tensor(weight_list, dtype=torch.float32, device=device)
    
    return edge_index, edge_weights

def mean_pool_with_similarity(
    features: torch.Tensor
) -> torch.Tensor:
    """
    Perform plain mean pooling of features without similarity weights.

    For all patches j, compute the global mean feature:
    pooled = (1 / N) * Σ_j h_j

    The same pooled feature vector is assigned to all nodes.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature embeddings, shape [N, D]
    
    Returns
    -------
    torch.Tensor
        Pooled features, shape [N, D], where each row is the global mean feature
    """
    # Compute global mean feature across all nodes
    pooled_feature = torch.mean(features, dim=0, keepdim=True)  # [1, D]
    
    return pooled_feature

def build_hypergraph_data(
    features: torch.Tensor,
    positions: torch.Tensor,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    threshold_median_ratio: float = None,
    use_pooling: bool = True,
    device: Optional[torch.device] = None
) -> dict:
    """
    Build complete hypergraph data structure with similarity-based edges.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature embeddings, shape [N, D]
    positions : torch.Tensor
        Spatial coordinates, shape [N, 2] or [N, 3]
    lambda_h : float, optional
        Scaling parameter for morphological similarity, default 1.0
    lambda_g : float, optional
        Scaling parameter for spatial similarity, default 1.0
    threshold_median_ratio : float, optional
        Threshold as a ratio of the median similarity.
        If specified, threshold = median(K) * threshold_median_ratio.
        This allows adaptive thresholding based on the similarity distribution.
        Example: threshold_median_ratio=0.5 means threshold is half of the median similarity.
        If None, all edges are kept.
    use_pooling : bool, optional
        Whether to apply mean pooling, default True
    device : torch.device, optional
        Device to place tensors on
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'x': node features, shape [N, D]
        - 'edge_index': edge indices, shape [2, E]
        - 'edge_attr': edge weights (similarity scores), shape [E]
        - 'pos': node positions, shape [N, 2] or [N, 3]
        - 'pooled_feature': pooled feature (if use_pooling=True), shape [D]
    """
    if device is None:
        device = features.device
    
    features = features.to(device)
    positions = positions.to(device)
    
    # Build weighted hypergraph
    edge_index, edge_weights = build_weighted_hypergraph(
        features, positions, lambda_h, lambda_g, threshold_median_ratio, device
    )
    
    result = {
        'x': features,
        'edge_index': edge_index,
        'edge_attr': edge_weights,
        'pos': positions
    }
    
    # Apply mean pooling if requested
    if use_pooling:
        pooled_feature = mean_pool_with_similarity(features)
        result['pooled_feature'] = pooled_feature
    
    return result

