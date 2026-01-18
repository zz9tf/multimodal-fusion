# SVD + Drop Modality Framework Demo

This demo showcases the core innovations of our multimodal survival prediction framework, focusing on **SVD-based alignment** and **random modality dropping** mechanisms.

## 🚀 Core Innovations

### 1. SVD-based Multimodal Alignment
- Uses Singular Value Decomposition (SVD) for cross-modal feature alignment
- Learns shared representations across different modalities (WSI, TMA, clinical data)
- Temperature-controlled alignment losses for robust feature fusion

### 2. Dynamic Modality Gating
- Confidence-based gating mechanism for each modality
- Automatically learns which modalities are reliable for each sample
- Improves robustness against noisy or missing modalities

### 3. Random Modality Dropping
- During training, randomly drops modalities to simulate missing data scenarios
- Enhances model robustness and generalization
- Prevents over-reliance on any single modality

## 📁 File Structure

```
demo/
├── models/                           # Core model implementations
│   ├── base_model.py                # Base model class with unified interface
│   ├── clam_mlp.py                  # CLAM MLP base architecture
│   ├── svd_gate_random_clam.py      # Core SVD + gating model
│   ├── deep_supervise_svd_gate_random.py  # Deep supervise version
│   └── model_factory.py             # Model factory for easy instantiation
├── example_usage.py                 # Demonstration script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🔧 Key Components

### SVD Gate Random CLAM Model
The main model implementing SVD alignment and random modality dropping:

```python
config = {
    'model_type': 'svd_gate_random_clam',
    'enable_svd': True,                    # Enable SVD alignment
    'alignment_channels': ['wsi', 'tma_cd3', 'tma_cd8'],
    'tau1': 0.1, 'tau2': 0.1,            # Alignment temperatures
    'lambda1': 1.0, 'lambda2': 0.1,      # Alignment loss weights
    'enable_dynamic_gate': True,          # Dynamic gating
    'enable_random_loss': True,           # Random modality dropping
}
```

### Model Factory
Unified interface for creating different model variants:

```python
from models.model_factory import ModelFactory
model = ModelFactory.create_model(config)
```

## 🏃‍♂️ Running the Demo

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demonstration:
```bash
python example_usage.py
```

The demo will:
- ✅ Create SVD-based multimodal models
- ✅ Generate mock multimodal data
- ⚠️  Show forward pass (may fail due to device issues in demo environment)
- ✅ Display model architecture summary
- ✅ Demonstrate both regular and deep supervise variants

## 📊 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `tau1`, `tau2` | Temperature for SVD alignment losses | 0.1 |
| `lambda1`, `lambda2` | Weights for alignment losses | 1.0, 0.1 |
| `alignment_layer_num` | Number of alignment layers | 2 |
| `weight_random_loss` | Weight for random dropping loss | 0.1 |
| `enable_dynamic_gate` | Enable confidence-based gating | True |
| `enable_svd` | Enable SVD alignment | True |

## 🎯 Expected Output

The demo shows:
- Model creation with SVD alignment
- Multimodal feature processing
- Dynamic gating based on confidence scores
- Random modality dropping mechanism
- Unified training interface

## ⚠️ Note

This is a demonstration version with mock data. The full implementation requires:
- Real multimodal datasets (WSI, TMA features)
- Complete training pipeline
- Cross-validation framework
- Evaluation metrics

## 📝 Citation

If you use this framework, please cite our paper:

```
[Paper citation information]
```

## 🤝 Contact

For questions about the implementation, please refer to the paper or contact the authors.