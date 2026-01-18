#!/usr/bin/env python3
"""
SVD + Drop Modality Framework Demo

This demo shows how to use the SVD-based multimodal fusion framework
for survival prediction tasks.

The core innovation: SVD-based alignment + random modality dropping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from models.model_factory import ModelFactory

def create_mock_multimodal_data(batch_size: int = 4, num_modalities: int = 3,
                               feature_dim: int = 1024) -> Dict[str, torch.Tensor]:
    """
    Create mock multimodal data for demonstration

    Args:
        batch_size: Number of samples
        num_modalities: Number of modalities (e.g., WSI, TMA_CD3, TMA_CD8)
        feature_dim: Feature dimension for each modality

    Returns:
        Dictionary with modality names as keys and feature tensors as values
    """
    modalities = [f'modality_{i}' for i in range(num_modalities)]

    data = {}
    for modality in modalities:
        # Create random features
        features = torch.randn(batch_size, feature_dim)
        data[modality] = features

    return data

def demonstrate_svd_model():
    """
    Demonstrate the core SVD + Drop Modality framework
    """
    print("🚀 SVD + Drop Modality Framework Demo")
    print("=" * 50)

    # Model configuration - core parameters for SVD alignment
    config = {
        'model_type': 'svd_gate_random_clam',  # Core SVD model
        'n_classes': 2,                        # Survival classes (alive/dead)
        'input_dim': 1024,                     # Input feature dimension
        'dropout': 0.1,                        # Dropout rate
        'model_size': 'small',                 # Model size
        'base_loss_fn': 'ce',                  # Base loss function (ce/svm)
        'channels_used_in_model': ['modality_0', 'modality_1', 'modality_2'], # Channels to use

        # SVD-specific parameters (your core innovation)
        'enable_svd': True,                    # Enable SVD alignment
        'alignment_channels': ['modality_0', 'modality_1', 'modality_2'],
        'alignment_layer_num': 2,              # Number of alignment layers
        'tau1': 0.1,                          # Temperature for alignment loss 1
        'tau2': 0.1,                          # Temperature for alignment loss 2
        'lambda1': 1.0,                        # Weight for alignment loss 1
        'lambda2': 0.1,                        # Weight for alignment loss 2

        # Dynamic gating parameters
        'enable_dynamic_gate': True,          # Enable dynamic modality gating
        'enable_random_loss': True,           # Enable random modality dropping
        'weight_random_loss': 0.1,            # Weight for random loss
    }

    print("📋 Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create model
    print("🔧 Creating SVD Gate Random CLAM model...")
    model = ModelFactory.create_model(config)
    print(f"✅ Model created: {type(model).__name__}")
    print()

    # Create mock data
    print("📊 Creating mock multimodal data...")
    batch_size = 4
    mock_data = create_mock_multimodal_data(batch_size=batch_size,
                                          num_modalities=3,
                                          feature_dim=1024)

    print(f"📈 Mock data shape: {batch_size} samples")
    for modality, features in mock_data.items():
        print(f"  {modality}: {features.shape}")
    print()

    # Create mock labels
    labels = torch.randint(0, 2, (batch_size,))  # Binary survival labels
    print(f"🏷️  Mock labels: {labels.tolist()}")
    print()

    # Forward pass
    print("🔬 Running forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            results = model(mock_data, labels)

            print("📊 Forward pass results:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            else:
                print(f"  Output: {results}")

        except Exception as e:
            print(f"⚠️  Forward pass encountered an issue: {e}")
            print("   This is expected as some dependencies may be missing in demo")
    print()

    # Show model architecture
    print("🏗️  Model Architecture Summary:")
    print(f"  Model type: {config['model_type']}")
    print(f"  Input modalities: {config['alignment_channels']}")
    print(f"  SVD enabled: {config['enable_svd']}")
    print(f"  Dynamic gating: {config['enable_dynamic_gate']}")
    print(f"  Random loss: {config['enable_random_loss']}")
    print()

    print("✨ Demo completed! This shows the core SVD + drop modality framework.")
    print("   The key innovations:")
    print("   1. SVD-based multimodal alignment")
    print("   2. Dynamic modality gating based on confidence")
    print("   3. Random modality dropping for robustness")

def demonstrate_deep_supervise_model():
    """
    Demonstrate the deep supervise version of SVD model
    """
    print("\n🔬 Deep Supervise SVD Model Demo")
    print("=" * 40)

    config = {
        'model_type': 'deep_supervise_svd_gate_random',
        'n_classes': 2,
        'input_dim': 1024,
        'dropout': 0.1,
        'model_size': 'small',
        'base_loss_fn': 'ce',
        'channels_used_in_model': ['modality_0', 'modality_1'],

        # SVD parameters
        'enable_svd': True,
        'alignment_channels': ['modality_0', 'modality_1'],
        'alignment_layer_num': 2,
        'tau1': 0.1,
        'tau2': 0.1,
        'lambda1': 1.0,
        'lambda2': 0.1,

        # Dynamic gating
        'enable_dynamic_gate': True,
        'enable_random_loss': True,
        'weight_random_loss': 0.1,
    }

    print("📋 Deep Supervise Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    try:
        model = ModelFactory.create_model(config)
        print(f"✅ Deep supervise model created: {type(model).__name__}")
    except Exception as e:
        print(f"⚠️  Model creation failed: {e}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demonstrations
    demonstrate_svd_model()
    demonstrate_deep_supervise_model()