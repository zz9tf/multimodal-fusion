"""
Preprocess WSI and TMA data to build hypergraph structure.

Pipeline:
1. Load WSI patches from h5
2. Aggregate WSI patches into super patches (based on similarity)
3. Compute similarity between WSI super patches and TMA
4. Group WSI super patches + TMA based on similarity
5. Build hypergraph with KNN and KMeans
6. Store results back to h5
7. Record similarity scores for parameter tuning
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import json
from tqdm import tqdm

from .similarity_kernel import (
    compute_combined_similarity
)


def load_wsi_data(h5_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load WSI features and positions from h5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file
    
    Returns
    -------
    features : torch.Tensor
        WSI patch features, shape [N_wsi, D]
    positions : torch.Tensor
        WSI patch positions, shape [N_wsi, 2] or [N_wsi, 3]
    """
    with h5py.File(h5_path, 'r') as f:
        # Load WSI features
        if 'wsi' in f and 'features' in f['wsi']:
            wsi_features = torch.from_numpy(f['wsi']['features'][:]).float()
        else:
            raise ValueError(f"WSI features not found in {h5_path}")
        
        # Load WSI positions (if available)
        if 'wsi' in f and 'positions' in f['wsi']:
            wsi_positions = torch.from_numpy(f['wsi']['positions'][:]).float()
        else:
            # If positions not available, use dummy positions (zeros)
            wsi_positions = torch.zeros(wsi_features.shape[0], 2, dtype=torch.float32)
            print(f"⚠️  WSI positions not found, using dummy positions")
    
    return wsi_features, wsi_positions


def load_tma_data(h5_path: str) -> Optional[torch.Tensor]:
    """
    Load TMA features from h5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file
    
    Returns
    -------
    features : torch.Tensor or None
        TMA features, shape [N_tma, D] or None if not found
    """
    with h5py.File(h5_path, 'r') as f:
        if 'tma' in f and 'features' in f['tma']:
            tma_features = torch.from_numpy(f['tma']['features'][:]).float()
            return tma_features
        else:
            return None


def aggregate_wsi_super_patches(
    wsi_features: torch.Tensor,
    wsi_positions: torch.Tensor,
    num_super_patches: int,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    device: Optional[torch.device] = None,
    wsi_similarity_matrix: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
    """
    Aggregate WSI patches into super patches using similarity-based clustering.
    
    Uses KMeans clustering on similarity space to group similar patches.
    
    Parameters
    ----------
    wsi_features : torch.Tensor
        WSI patch features, shape [N_wsi, D]
    wsi_positions : torch.Tensor
        WSI patch positions, shape [N_wsi, 2] or [N_wsi, 3]
    num_super_patches : int
        Number of super patches to create
    lambda_h : float
        Scaling parameter for morphological similarity
    lambda_g : float
        Scaling parameter for spatial similarity
    device : torch.device, optional
        Device to use for computation
    
    Returns
    -------
    super_patch_features : torch.Tensor
        Aggregated super patch features, shape [num_super_patches, D]
    super_patch_positions : torch.Tensor
        Aggregated super patch positions, shape [num_super_patches, 2] or [num_super_patches, 3]
    stats : Dict
        Statistics including similarity scores and cluster info
    """
    if device is None:
        device = wsi_features.device if wsi_features.is_cuda else torch.device('cpu')
    
    wsi_features = wsi_features.to(device)
    wsi_positions = wsi_positions.to(device)
    
    N_wsi = wsi_features.shape[0]
    
    # Compute similarity matrix for WSI patches (or use provided one)
    if wsi_similarity_matrix is not None:
        K_wsi = wsi_similarity_matrix.to(device)
    else:
        K_wsi = compute_combined_similarity(
            wsi_features, wsi_positions, lambda_h, lambda_g
        )
    
    # Use KMeans on similarity space (or directly on features)
    # Option 1: Cluster on features directly (faster)
    # Option 2: Cluster on similarity matrix (more accurate but slower)
    # We use Option 1 for efficiency, but can be configured
    
    # Convert to numpy for sklearn
    features_np = wsi_features.cpu().numpy()
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=num_super_patches, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_np)
    
    # Aggregate: compute mean feature and position for each cluster
    super_patch_features = []
    super_patch_positions = []
    
    for cluster_id in range(num_super_patches):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            # Mean pooling within cluster
            cluster_features = wsi_features[mask].mean(dim=0)
            cluster_positions = wsi_positions[mask].mean(dim=0)
        else:
            raise ValueError(f"Cluster {cluster_id} is empty")
        
        super_patch_features.append(cluster_features)
        super_patch_positions.append(cluster_positions)
    
    super_patch_features = torch.stack(super_patch_features, dim=0)
    super_patch_positions = torch.stack(super_patch_positions, dim=0)
    
    # Compute statistics
    # Average similarity within clusters
    intra_cluster_sims = []
    for cluster_id in range(num_super_patches):
        mask = cluster_labels == cluster_id
        if mask.sum() > 1:
            cluster_indices = torch.where(torch.from_numpy(mask))[0]
            cluster_K = K_wsi[cluster_indices][:, cluster_indices]
            # Exclude diagonal
            mask_off_diag = ~torch.eye(cluster_K.shape[0], dtype=torch.bool, device=device)
            intra_sims = cluster_K[mask_off_diag]
            if len(intra_sims) > 0:
                intra_cluster_sims.append(intra_sims.mean().item())
    
    stats = {
        'num_original_patches': N_wsi,
        'num_super_patches': num_super_patches,
        'avg_intra_cluster_similarity': np.mean(intra_cluster_sims) if intra_cluster_sims else 0.0,
        'wsi_similarity_matrix_stats': {
            'mean': K_wsi.mean().item(),
            'std': K_wsi.std().item(),
            'min': K_wsi.min().item(),
            'max': K_wsi.max().item(),
            'median': K_wsi.median().item()
        }
    }
    
    return super_patch_features, super_patch_positions, stats, K_wsi


def compute_wsi_tma_similarity(
    wsi_features: torch.Tensor,
    wsi_positions: torch.Tensor,
    tma_features: torch.Tensor,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute similarity matrix between WSI super patches and TMA patches.
    
    Parameters
    ----------
    wsi_features : torch.Tensor
        WSI super patch features, shape [N_wsi_super, D]
    wsi_positions : torch.Tensor
        WSI super patch positions, shape [N_wsi_super, 2] or [N_wsi_super, 3]
    tma_features : torch.Tensor
        TMA patch features, shape [N_tma, D]
    lambda_h : float
        Scaling parameter for morphological similarity
    lambda_g : float
        Scaling parameter for spatial similarity
    device : torch.device, optional
        Device to use for computation
    
    Returns
    -------
    similarity_matrix : torch.Tensor
        Similarity matrix, shape [N_wsi_super, N_tma]
    stats : Dict
        Statistics about similarity scores
    """
    if device is None:
        device = wsi_features.device if wsi_features.is_cuda else torch.device('cpu')
    
    wsi_features = wsi_features.to(device)
    wsi_positions = wsi_positions.to(device)
    tma_features = tma_features.to(device)
    
    N_wsi = wsi_features.shape[0]
    N_tma = tma_features.shape[0]
    
    # For cross-modal similarity, we only use morphological similarity
    # (spatial similarity doesn't make sense between WSI and TMA)
    # Compute pairwise morphological similarity
    similarity_matrix = torch.zeros(N_wsi, N_tma, device=device)
    
    for i in range(N_wsi):
        wsi_feat = wsi_features[i:i+1]  # [1, D]
        # Compute morphological similarity with all TMA patches
        # Using broadcasting
        diff = wsi_feat - tma_features  # [N_tma, D]
        squared_dist = (diff ** 2).sum(dim=1)  # [N_tma]
        sim = torch.exp(-lambda_h * squared_dist)  # [N_tma]
        similarity_matrix[i] = sim
    
    stats = {
        'mean': similarity_matrix.mean().item(),
        'std': similarity_matrix.std().item(),
        'min': similarity_matrix.min().item(),
        'max': similarity_matrix.max().item(),
        'median': similarity_matrix.median().item()
    }
    
    return similarity_matrix, stats


def group_by_similarity(
    similarity_matrix: torch.Tensor,
    num_groups: int,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, Dict]:
    """
    Group WSI super patches and TMA patches based on similarity.
    
    Parameters
    ----------
    similarity_matrix : torch.Tensor
        Similarity matrix, shape [N_wsi_super, N_tma]
    num_groups : int
        Number of groups to create
    method : str
        Grouping method: 'kmeans' or 'knn'
    
    Returns
    -------
    group_labels : np.ndarray
        Group labels for each WSI super patch, shape [N_wsi_super]
    stats : Dict
        Statistics about grouping
    """
    similarity_np = similarity_matrix.cpu().numpy()
    N_wsi = similarity_matrix.shape[0]
    
    if method == 'kmeans':
        # Use KMeans on similarity features (each WSI super patch has N_tma similarity values)
        kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(similarity_np)
        
        stats = {
            'method': 'kmeans',
            'num_groups': num_groups,
            'group_sizes': [np.sum(group_labels == i) for i in range(num_groups)]
        }
    elif method == 'knn':
        # Use KNN: find k nearest TMA patches for each WSI super patch
        # Then group WSI super patches that share similar nearest neighbors
        k = min(num_groups, similarity_matrix.shape[1])
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(similarity_np.T)  # Fit on TMA side
        
        # For each WSI super patch, find its k nearest TMA patches
        distances, indices = knn.kneighbors(similarity_np)
        
        # Group WSI super patches by their most similar TMA patch
        most_similar_tma = indices[:, 0]  # [N_wsi]
        # Use KMeans on most_similar_tma to create num_groups
        kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(most_similar_tma.reshape(-1, 1))
        
        stats = {
            'method': 'knn',
            'num_groups': num_groups,
            'k': k,
            'group_sizes': [np.sum(group_labels == i) for i in range(num_groups)]
        }
    else:
        raise ValueError(f"Unknown grouping method: {method}")
    
    return group_labels, stats


def build_hypergraph_knn_kmeans(
    wsi_features: torch.Tensor,
    tma_features: torch.Tensor,
    group_labels: np.ndarray,
    k: int = 5,
    num_hyperedges: int = 10,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Build hypergraph using KNN and KMeans.
    
    Parameters
    ----------
    wsi_features : torch.Tensor
        WSI super patch features, shape [N_wsi_super, D]
    tma_features : torch.Tensor
        TMA patch features, shape [N_tma, D]
    group_labels : np.ndarray
        Group labels, shape [N_wsi_super]
    k : int
        Number of nearest neighbors for KNN
    num_hyperedges : int
        Number of hyperedges (from KMeans)
    device : torch.device, optional
        Device to use for computation
    
    Returns
    -------
    edge_index : torch.Tensor
        Hypergraph edge indices, shape [2, E]
    edge_weights : torch.Tensor
        Edge weights, shape [E]
    stats : Dict
        Statistics about hypergraph
    """
    if device is None:
        device = wsi_features.device if wsi_features.is_cuda else torch.device('cpu')
    
    # Combine WSI and TMA features
    all_features = torch.cat([wsi_features, tma_features], dim=0).to(device)
    N_total = all_features.shape[0]
    N_wsi = wsi_features.shape[0]
    
    # Build KNN edges
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 to exclude self
    features_np = all_features.cpu().numpy()
    knn.fit(features_np)
    distances, indices = knn.kneighbors(features_np)
    
    # Build edge list from KNN
    edge_list = []
    for i in range(N_total):
        for j_idx, neighbor_idx in enumerate(indices[i, 1:]):  # Skip self (first neighbor)
            edge_list.append([i, neighbor_idx])
    
    # Build hyperedges using KMeans
    kmeans = KMeans(n_clusters=num_hyperedges, random_state=42, n_init=10)
    hyperedge_labels = kmeans.fit_predict(features_np)
    
    # Add hyperedge connections: all nodes in the same hyperedge are connected
    for hyperedge_id in range(num_hyperedges):
        nodes_in_hyperedge = np.where(hyperedge_labels == hyperedge_id)[0]
        for i in nodes_in_hyperedge:
            for j in nodes_in_hyperedge:
                if i != j:
                    edge_list.append([i, j])
    
    # Remove duplicates and convert to tensor
    edge_list = list(set(tuple(sorted(edge)) for edge in edge_list))
    edge_list = [[e[0], e[1]] for e in edge_list]
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weights = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        
        # Compute edge weights based on feature similarity
        edge_weights = []
        for edge in edge_list:
            i, j = edge
            feat_i = all_features[i]
            feat_j = all_features[j]
            # Cosine similarity as weight
            weight = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0)).item()
            edge_weights.append(max(0.0, weight))  # Ensure non-negative
        
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    
    stats = {
        'num_nodes': N_total,
        'num_wsi_super_patches': N_wsi,
        'num_tma_patches': tma_features.shape[0],
        'num_edges': len(edge_list),
        'num_hyperedges': num_hyperedges,
        'k': k
    }
    
    return edge_index, edge_weights, stats


def save_hypergraph_to_h5(
    h5_path: str,
    wsi_super_features: torch.Tensor,
    wsi_super_positions: torch.Tensor,
    tma_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    group_labels: np.ndarray,
    stats: Dict,
    wsi_similarity_matrix: Optional[torch.Tensor] = None,
    wsi_tma_similarity_matrix: Optional[torch.Tensor] = None
):
    """
    Save hypergraph data to h5 file, including similarity matrices for fast parameter tuning.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file
    wsi_super_features : torch.Tensor
        WSI super patch features
    wsi_super_positions : torch.Tensor
        WSI super patch positions
    tma_features : torch.Tensor
        TMA features
    edge_index : torch.Tensor
        Hypergraph edge indices
    edge_weights : torch.Tensor
        Edge weights
    group_labels : np.ndarray
        Group labels
    stats : Dict
        Statistics dictionary
    wsi_similarity_matrix : torch.Tensor, optional
        WSI internal similarity matrix [N_wsi, N_wsi], for fast aggregation tuning
    wsi_tma_similarity_matrix : torch.Tensor, optional
        WSI-TMA similarity matrix [N_wsi_super, N_tma], for fast grouping tuning
    """
    with h5py.File(h5_path, 'a') as f:  # 'a' mode to append/update
        # Create or update hypergraph group
        if 'hypergraph' not in f:
            f.create_group('hypergraph')
        
        hg = f['hypergraph']
        
        # Save WSI super patches
        if 'wsi_super' not in hg:
            hg.create_group('wsi_super')
        hg['wsi_super']['features'] = wsi_super_features.cpu().numpy()
        hg['wsi_super']['positions'] = wsi_super_positions.cpu().numpy()
        
        # Save TMA features (reference)
        if 'tma' not in hg:
            hg.create_group('tma')
        hg['tma']['features'] = tma_features.cpu().numpy()
        
        # Save hypergraph structure
        hg['edge_index'] = edge_index.cpu().numpy()
        hg['edge_weights'] = edge_weights.cpu().numpy()
        hg['group_labels'] = group_labels
        
        # Save similarity matrices for fast parameter tuning
        if wsi_similarity_matrix is not None:
            if 'similarity' not in hg:
                hg.create_group('similarity')
            hg['similarity']['wsi_internal'] = wsi_similarity_matrix.cpu().numpy()
            hg['similarity'].attrs['wsi_shape'] = list(wsi_similarity_matrix.shape)
        
        if wsi_tma_similarity_matrix is not None:
            if 'similarity' not in hg:
                hg.create_group('similarity')
            hg['similarity']['wsi_tma'] = wsi_tma_similarity_matrix.cpu().numpy()
            hg['similarity'].attrs['wsi_tma_shape'] = list(wsi_tma_similarity_matrix.shape)
        
        # Save statistics as JSON string
        hg.attrs['stats'] = json.dumps(stats)


def process_single_file(
    h5_path: str,
    num_wsi_super_patches: int = 100,
    num_groups: int = 10,
    hypergraph_k: int = 5,
    num_hyperedges: int = 10,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Process a single h5 file: aggregate WSI, build hypergraph, and save results.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file
    num_wsi_super_patches : int
        Number of WSI super patches to create
    num_groups : int
        Number of similarity groups
    hypergraph_k : int
        K for KNN in hypergraph construction
    num_hyperedges : int
        Number of hyperedges from KMeans
    lambda_h : float
        Scaling parameter for morphological similarity
    lambda_g : float
        Scaling parameter for spatial similarity
    device : torch.device, optional
        Device to use for computation
    
    Returns
    -------
    stats : Dict
        Processing statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📂 Processing: {h5_path}")
    
    # Load data
    wsi_features, wsi_positions = load_wsi_data(h5_path)
    tma_features = load_tma_data(h5_path)
    
    if tma_features is None:
        print(f"⚠️  TMA features not found, skipping hypergraph construction")
        return {'status': 'skipped', 'reason': 'no_tma'}
    
    # Step 1: Aggregate WSI into super patches
    print(f"  🔄 Aggregating WSI patches into {num_wsi_super_patches} super patches...")
    wsi_super_features, wsi_super_positions, wsi_stats, wsi_sim_matrix = aggregate_wsi_super_patches(
        wsi_features, wsi_positions, num_wsi_super_patches, lambda_h, lambda_g, device
    )
    
    # Step 2: Compute WSI-TMA similarity
    print(f"  🔄 Computing WSI-TMA similarity...")
    similarity_matrix, sim_stats = compute_wsi_tma_similarity(
        wsi_super_features, wsi_super_positions, tma_features, lambda_h, lambda_g, device
    )
    
    # Step 3: Group by similarity
    print(f"  🔄 Grouping into {num_groups} groups...")
    group_labels, group_stats = group_by_similarity(similarity_matrix, num_groups, method='kmeans')
    
    # Step 4: Build hypergraph
    print(f"  🔄 Building hypergraph (KNN k={hypergraph_k}, KMeans {num_hyperedges} hyperedges)...")
    edge_index, edge_weights, hg_stats = build_hypergraph_knn_kmeans(
        wsi_super_features, tma_features, group_labels, hypergraph_k, num_hyperedges, device
    )
    
    # Step 5: Save to h5 (including similarity matrices)
    print(f"  💾 Saving to h5 (including similarity matrices for fast tuning)...")
    all_stats = {
        'wsi_aggregation': wsi_stats,
        'similarity': sim_stats,
        'grouping': group_stats,
        'hypergraph': hg_stats
    }
    save_hypergraph_to_h5(
        h5_path, wsi_super_features, wsi_super_positions, tma_features,
        edge_index, edge_weights, group_labels, all_stats,
        wsi_similarity_matrix=wsi_sim_matrix,
        wsi_tma_similarity_matrix=similarity_matrix
    )
    
    print(f"  ✅ Done!")
    
    return all_stats


def process_dataset(
    csv_path: str,
    data_root_dir: str,
    num_wsi_super_patches: int = 100,
    num_groups: int = 10,
    hypergraph_k: int = 5,
    num_hyperedges: int = 10,
    lambda_h: float = 1.0,
    lambda_g: float = 1.0,
    output_stats_path: Optional[str] = None,
    device: Optional[torch.device] = None
):
    """
    Process all files in the dataset.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with h5_file_path column
    data_root_dir : str
        Root directory for h5 files
    num_wsi_super_patches : int
        Number of WSI super patches to create
    num_groups : int
        Number of similarity groups
    hypergraph_k : int
        K for KNN in hypergraph construction
    num_hyperedges : int
        Number of hyperedges from KMeans
    lambda_h : float
        Scaling parameter for morphological similarity
    lambda_g : float
        Scaling parameter for spatial similarity
    output_stats_path : str, optional
        Path to save statistics JSON file
    device : torch.device, optional
        Device to use for computation
    """
    df = pd.read_csv(csv_path)
    
    if 'h5_file_path' not in df.columns:
        raise ValueError("CSV must contain 'h5_file_path' column")
    
    all_stats = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        h5_rel_path = row['h5_file_path']
        h5_path = os.path.join(data_root_dir, h5_rel_path)
        
        if not os.path.exists(h5_path):
            print(f"⚠️  File not found: {h5_path}")
            continue
        
        try:
            stats = process_single_file(
                h5_path, num_wsi_super_patches, num_groups,
                hypergraph_k, num_hyperedges, lambda_h, lambda_g, device
            )
            stats['case_id'] = row.get('case_id', f'case_{idx}')
            stats['h5_path'] = h5_rel_path
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error processing {h5_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save statistics
    if output_stats_path:
        with open(output_stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"📊 Statistics saved to: {output_stats_path}")
    
    return all_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess hypergraph data')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with h5_file_path column')
    parser.add_argument('--data_root_dir', type=str, required=True,
                       help='Root directory for h5 files')
    parser.add_argument('--num_wsi_super_patches', type=int, default=100,
                       help='Number of WSI super patches')
    parser.add_argument('--num_groups', type=int, default=10,
                       help='Number of similarity groups')
    parser.add_argument('--hypergraph_k', type=int, default=5,
                       help='K for KNN in hypergraph')
    parser.add_argument('--num_hyperedges', type=int, default=10,
                       help='Number of hyperedges')
    parser.add_argument('--lambda_h', type=float, default=1.0,
                       help='Scaling parameter for morphological similarity')
    parser.add_argument('--lambda_g', type=float, default=1.0,
                       help='Scaling parameter for spatial similarity')
    parser.add_argument('--output_stats', type=str, default=None,
                       help='Path to save statistics JSON')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, or cuda')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting preprocessing with device: {device}")
    print(f"   Parameters: num_wsi_super_patches={args.num_wsi_super_patches}, "
          f"num_groups={args.num_groups}, lambda_h={args.lambda_h}, lambda_g={args.lambda_g}")
    
    process_dataset(
        args.csv_path, args.data_root_dir,
        args.num_wsi_super_patches, args.num_groups,
        args.hypergraph_k, args.num_hyperedges,
        args.lambda_h, args.lambda_g,
        args.output_stats, device
    )


def load_similarity_matrices(h5_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load stored similarity matrices from h5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file
    
    Returns
    -------
    wsi_sim_matrix : torch.Tensor or None
        WSI internal similarity matrix [N_wsi, N_wsi]
    wsi_tma_sim_matrix : torch.Tensor or None
        WSI-TMA similarity matrix [N_wsi_super, N_tma]
    """
    with h5py.File(h5_path, 'r') as f:
        wsi_sim_matrix = None
        wsi_tma_sim_matrix = None
        
        if 'hypergraph' in f and 'similarity' in f['hypergraph']:
            sim_group = f['hypergraph']['similarity']
            
            if 'wsi_internal' in sim_group:
                wsi_sim_matrix = torch.from_numpy(sim_group['wsi_internal'][:]).float()
            
            if 'wsi_tma' in sim_group:
                wsi_tma_sim_matrix = torch.from_numpy(sim_group['wsi_tma'][:]).float()
        
        return wsi_sim_matrix, wsi_tma_sim_matrix


def rebuild_hypergraph_from_similarity(
    h5_path: str,
    num_wsi_super_patches: int = None,
    num_groups: int = None,
    hypergraph_k: int = 5,
    num_hyperedges: int = 10,
    threshold_median_ratio: float = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Quickly rebuild hypergraph from stored similarity matrices with different parameters.
    
    This function allows fast experimentation with different aggregation scales without
    recomputing similarity matrices.
    
    Parameters
    ----------
    h5_path : str
        Path to h5 file with stored similarity matrices
    num_wsi_super_patches : int, optional
        New number of WSI super patches (if None, use existing)
    num_groups : int, optional
        New number of groups (if None, use existing)
    hypergraph_k : int
        K for KNN in hypergraph construction
    num_hyperedges : int
        Number of hyperedges from KMeans
    threshold_median_ratio : float, optional
        Threshold ratio for filtering edges based on similarity
    device : torch.device, optional
        Device to use for computation
    
    Returns
    -------
    Dict
        Updated hypergraph data and statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔄 Rebuilding hypergraph from stored similarity matrices: {h5_path}")
    
    # Load original data
    wsi_features, wsi_positions = load_wsi_data(h5_path)
    tma_features = load_tma_data(h5_path)
    
    if tma_features is None:
        raise ValueError("TMA features not found")
    
    # Load stored similarity matrices
    wsi_sim_matrix, wsi_tma_sim_matrix = load_similarity_matrices(h5_path)
    
    if wsi_sim_matrix is None:
        print("⚠️  WSI similarity matrix not found, recomputing...")
        wsi_sim_matrix = compute_combined_similarity(
            wsi_features, wsi_positions, lambda_h=1.0, lambda_g=1.0
        )
    
    # Step 1: Re-aggregate WSI with new num_super_patches (if specified)
    wsi_stats = {}
    if num_wsi_super_patches is not None:
        print(f"  🔄 Re-aggregating WSI into {num_wsi_super_patches} super patches...")
        wsi_super_features, wsi_super_positions, wsi_stats, _ = aggregate_wsi_super_patches(
            wsi_features, wsi_positions, num_wsi_super_patches,
            lambda_h=1.0, lambda_g=1.0, device=device,
            wsi_similarity_matrix=wsi_sim_matrix
        )
        
        # Recompute WSI-TMA similarity with new super patches
        print(f"  🔄 Recomputing WSI-TMA similarity...")
        similarity_matrix, sim_stats = compute_wsi_tma_similarity(
            wsi_super_features, wsi_super_positions, tma_features,
            lambda_h=1.0, lambda_g=1.0, device=device
        )
    else:
        # Use existing super patches
        with h5py.File(h5_path, 'r') as f:
            if 'hypergraph' in f and 'wsi_super' in f['hypergraph']:
                wsi_super_features = torch.from_numpy(
                    f['hypergraph']['wsi_super']['features'][:]
                ).float().to(device)
                wsi_super_positions = torch.from_numpy(
                    f['hypergraph']['wsi_super']['positions'][:]
                ).float().to(device)
            else:
                raise ValueError("WSI super patches not found and num_wsi_super_patches not specified")
        
        # Use stored WSI-TMA similarity or recompute
        if wsi_tma_sim_matrix is not None and wsi_tma_sim_matrix.shape[0] == wsi_super_features.shape[0]:
            similarity_matrix = wsi_tma_sim_matrix.to(device)
            sim_stats = {
                'mean': similarity_matrix.mean().item(),
                'std': similarity_matrix.std().item(),
                'min': similarity_matrix.min().item(),
                'max': similarity_matrix.max().item(),
                'median': similarity_matrix.median().item()
            }
        else:
            print(f"  🔄 Recomputing WSI-TMA similarity...")
            similarity_matrix, sim_stats = compute_wsi_tma_similarity(
                wsi_super_features, wsi_super_positions, tma_features,
                lambda_h=1.0, lambda_g=1.0, device=device
            )
    
    # Step 2: Re-group with new num_groups (if specified)
    if num_groups is not None:
        print(f"  🔄 Re-grouping into {num_groups} groups...")
        group_labels, group_stats = group_by_similarity(
            similarity_matrix, num_groups, method='kmeans'
        )
    else:
        # Use existing group labels
        with h5py.File(h5_path, 'r') as f:
            if 'hypergraph' in f and 'group_labels' in f['hypergraph']:
                group_labels = f['hypergraph']['group_labels'][:]
                group_stats = {'method': 'existing', 'num_groups': len(np.unique(group_labels))}
            else:
                raise ValueError("Group labels not found and num_groups not specified")
    
    # Step 3: Rebuild hypergraph
    print(f"  🔄 Rebuilding hypergraph (KNN k={hypergraph_k}, KMeans {num_hyperedges} hyperedges)...")
    edge_index, edge_weights, hg_stats = build_hypergraph_knn_kmeans(
        wsi_super_features, tma_features, group_labels,
        hypergraph_k, num_hyperedges, device
    )
    
    # Step 4: Apply threshold filtering if specified
    if threshold_median_ratio is not None:
        print(f"  🔄 Applying threshold filtering (ratio={threshold_median_ratio})...")
        # Filter edges based on similarity threshold
        # This would require storing node-to-node similarity in hypergraph
        # For now, we filter based on edge weights
        median_weight = edge_weights.median().item()
        threshold = median_weight * threshold_median_ratio
        mask = edge_weights >= threshold
        edge_index = edge_index[:, mask]
        edge_weights = edge_weights[mask]
        hg_stats['num_edges_after_threshold'] = edge_weights.shape[0]
        hg_stats['threshold'] = threshold
        hg_stats['threshold_ratio'] = threshold_median_ratio
    
    # Step 5: Save updated hypergraph
    print(f"  💾 Saving updated hypergraph...")
    all_stats = {
        'wsi_aggregation': wsi_stats if num_wsi_super_patches is not None else {},
        'similarity': sim_stats,
        'grouping': group_stats,
        'hypergraph': hg_stats
    }
    save_hypergraph_to_h5(
        h5_path, wsi_super_features, wsi_super_positions, tma_features,
        edge_index, edge_weights, group_labels, all_stats,
        wsi_similarity_matrix=wsi_sim_matrix,
        wsi_tma_similarity_matrix=similarity_matrix
    )
    
    print(f"  ✅ Done!")
    
    return all_stats


def batch_rebuild_hypergraph(
    csv_path: str,
    data_root_dir: str,
    num_wsi_super_patches: int = None,
    num_groups: int = None,
    hypergraph_k: int = 5,
    num_hyperedges: int = 10,
    threshold_median_ratio: float = None,
    output_stats_path: Optional[str] = None,
    device: Optional[torch.device] = None
):
    """
    Batch rebuild hypergraphs from stored similarity matrices.
    
    This allows fast experimentation with different parameters across the entire dataset.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with h5_file_path column
    data_root_dir : str
        Root directory for h5 files
    num_wsi_super_patches : int, optional
        New number of WSI super patches
    num_groups : int, optional
        New number of groups
    hypergraph_k : int
        K for KNN in hypergraph construction
    num_hyperedges : int
        Number of hyperedges
    threshold_median_ratio : float, optional
        Threshold ratio for filtering edges
    output_stats_path : str, optional
        Path to save statistics JSON
    device : torch.device, optional
        Device to use for computation
    """
    df = pd.read_csv(csv_path)
    
    if 'h5_file_path' not in df.columns:
        raise ValueError("CSV must contain 'h5_file_path' column")
    
    all_stats = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rebuilding hypergraphs"):
        h5_rel_path = row['h5_file_path']
        h5_path = os.path.join(data_root_dir, h5_rel_path)
        
        if not os.path.exists(h5_path):
            print(f"⚠️  File not found: {h5_path}")
            continue
        
        try:
            stats = rebuild_hypergraph_from_similarity(
                h5_path, num_wsi_super_patches, num_groups,
                hypergraph_k, num_hyperedges, threshold_median_ratio, device
            )
            stats['case_id'] = row.get('case_id', f'case_{idx}')
            stats['h5_path'] = h5_rel_path
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error processing {h5_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save statistics
    if output_stats_path:
        with open(output_stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"📊 Statistics saved to: {output_stats_path}")
    
    return all_stats

