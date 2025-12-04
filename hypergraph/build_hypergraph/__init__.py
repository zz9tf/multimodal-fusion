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

__all__ = [
    'compute_morphological_similarity',
    'compute_spatial_similarity',
    'compute_combined_similarity',
    'build_weighted_hypergraph',
    'mean_pool_with_similarity'
]




