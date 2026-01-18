"""
Example script for quickly rebuilding hypergraphs with different parameters.

This demonstrates how to use stored similarity matrices to rapidly experiment
with different aggregation scales without recomputing similarity.
"""

import argparse
from build_hypergraph import batch_rebuild_hypergraph


def main():
    parser = argparse.ArgumentParser(
        description='Quickly rebuild hypergraphs with different parameters'
    )
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with h5_file_path column')
    parser.add_argument('--data_root_dir', type=str, required=True,
                       help='Root directory for h5 files')
    parser.add_argument('--num_wsi_super_patches', type=int, default=None,
                       help='New number of WSI super patches (None to keep existing)')
    parser.add_argument('--num_groups', type=int, default=None,
                       help='New number of groups (None to keep existing)')
    parser.add_argument('--hypergraph_k', type=int, default=5,
                       help='K for KNN in hypergraph')
    parser.add_argument('--num_hyperedges', type=int, default=10,
                       help='Number of hyperedges')
    parser.add_argument('--threshold_median_ratio', type=float, default=None,
                       help='Threshold ratio for filtering edges (None to disable)')
    parser.add_argument('--output_stats', type=str, default=None,
                       help='Path to save statistics JSON')
    
    args = parser.parse_args()
    
    print("🚀 Quick hypergraph rebuild from stored similarity matrices")
    print(f"   Parameters:")
    print(f"   - num_wsi_super_patches: {args.num_wsi_super_patches}")
    print(f"   - num_groups: {args.num_groups}")
    print(f"   - hypergraph_k: {args.hypergraph_k}")
    print(f"   - num_hyperedges: {args.num_hyperedges}")
    print(f"   - threshold_median_ratio: {args.threshold_median_ratio}")
    
    batch_rebuild_hypergraph(
        args.csv_path,
        args.data_root_dir,
        num_wsi_super_patches=args.num_wsi_super_patches,
        num_groups=args.num_groups,
        hypergraph_k=args.hypergraph_k,
        num_hyperedges=args.num_hyperedges,
        threshold_median_ratio=args.threshold_median_ratio,
        output_stats_path=args.output_stats
    )
    
    print("✅ Done!")


if __name__ == '__main__':
    main()






