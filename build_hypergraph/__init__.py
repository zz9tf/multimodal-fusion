"""
Build hypergraph from WSI features and positions using similarity kernels.
"""

from .similarity_kernel import (
    compute_morphological_similarity,
    compute_spatial_similarity,
    compute_combined_similarity,
    build_weighted_hypergraph,
    mean_pool_with_similarity
)

from .preprocess_hypergraph import (
    process_single_file,
    process_dataset,
    load_wsi_data,
    load_tma_data,
    aggregate_wsi_super_patches,
    compute_wsi_tma_similarity,
    group_by_similarity,
    build_hypergraph_knn_kmeans,
    save_hypergraph_to_h5,
    load_similarity_matrices,
    rebuild_hypergraph_from_similarity,
    batch_rebuild_hypergraph
)

__all__ = [
    'compute_morphological_similarity',
    'compute_spatial_similarity',
    'compute_combined_similarity',
    'build_weighted_hypergraph',
    'mean_pool_with_similarity',
    'process_single_file',
    'process_dataset',
    'load_wsi_data',
    'load_tma_data',
    'aggregate_wsi_super_patches',
    'compute_wsi_tma_similarity',
    'group_by_similarity',
    'build_hypergraph_knn_kmeans',
    'save_hypergraph_to_h5',
    'load_similarity_matrices',
    'rebuild_hypergraph_from_similarity',
    'batch_rebuild_hypergraph'
]





