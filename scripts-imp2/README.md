# scripts-imp2: Quantum-Enhanced GAT Fraud Detection Pipeline

A clean, modular, production-ready implementation of quantum-inspired feature mapping with Graph Attention Networks for multi-class fraud detection on the Elliptic++ Bitcoin dataset.

## Overview

This pipeline implements:

1. **Quantum Feature Mapper**: A PyTorch nn.Module that applies quantum-inspired feature transformations:
   - Preprocessing: log1p + z-score normalization
   - Angle Encoding: cos(πx), sin(πx)
   - Pairwise Interactions: second-order x_i * x_j (upper triangular)
   - Random Fourier Features: √(2/D) * cos(Wx + b)

2. **Baseline GAT**: Standard Graph Attention Network for comparison
   - 2 layers, 64 hidden channels, 4 attention heads
   - Trained with class-weighted CrossEntropyLoss
   - Early stopping (patience=20)

3. **Quantum-Enhanced GAT**: GAT with expanded feature space
   - 3 layers, 128 hidden channels, 6 attention heads
   - Same training procedure as baseline for fair comparison
   - Demonstrates improved performance through feature enrichment

## File Structure

```
scripts-imp2/
├── quantum_feature_mapper.py    # Quantum feature mapping module
├── 01_load_data.py              # Load & preprocess graph, create splits
├── 02_train_baseline_gat.py     # Train baseline GAT model
├── 03_train_quantum_gat.py      # Train quantum-enhanced GAT
├── 04_compare_metrics.py        # Compare and visualize results
└── README.md                    # This file
```

## Quick Start

### Prerequisites

Ensure you have the required packages installed:
```bash
pip install torch torch-geometric scikit-learn matplotlib numpy
```

Also ensure `scripts-imp1/02_data_graph.py` has been run to create the base graph:
```bash
python scripts-imp1/02_data_graph.py
```

### Step 1: Load Data (1-2 minutes)

Loads the graph, normalizes features, and creates stratified 70/15/15 train/val/test splits:

```bash
python scripts-imp2/01_load_data.py
```

**Outputs:**
- `artifacts/elliptic_graph.pt` - Preprocessed graph with masks
- `artifacts/elliptic_graph_quantum.pt` - Same (quantum features added during training)

### Step 2: Train Baseline GAT (5-15 minutes)

Trains a baseline Graph Attention Network for comparison:

```bash
python scripts-imp2/02_train_baseline_gat.py
```

**Outputs:**
- `artifacts/baseline_gat_best.pt` - Best model weights
- `artifacts/baseline_gat_metrics.json` - Comprehensive metrics
- `artifacts/baseline_gat_training_curves.png` - Training visualizations

**Key Metrics Tracked:**
- Train/validation loss
- Macro F1, Micro F1
- Accuracy
- Per-class precision, recall
- Confusion matrix
- ROC-AUC (one-vs-rest)

### Step 3: Train Quantum-Enhanced GAT (5-15 minutes)

Trains the quantum-enhanced GAT with feature expansion:

```bash
python scripts-imp2/03_train_quantum_gat.py
```

**Outputs:**
- `artifacts/quantum_gat_best.pt` - Best model weights
- `artifacts/quantum_gat_metrics.json` - Comprehensive metrics
- `artifacts/quantum_gat_training_curves.png` - Training visualizations

**Key Differences:**
- Features expanded: [num_features] → [128] (capped)
- Model capacity increased: 64 → 128 hidden, 4 → 6 heads, 2 → 3 layers
- Same training procedure ensures fair comparison

### Step 4: Compare Models (1 minute)

Generates comprehensive comparison between baseline and quantum models:

```bash
python scripts-imp2/04_compare_metrics.py
```

**Outputs:**
- `artifacts/model_comparison.png` - Side-by-side visualizations
- `artifacts/model_comparison_report.json` - Detailed comparison metrics

**Includes:**
- Test metric comparisons (accuracy, F1, precision, recall, ROC-AUC)
- Training efficiency analysis
- Model architecture comparison
- Improvement percentages for each metric
- Visual comparison of:
  - Test metrics (bar chart)
  - Loss curves (both models)
  - F1 score trajectories
  - Accuracy curves
  - Improvement percentages
  - Confusion matrices

## Quantum Feature Mapper Details

### Features Computed

For input dimension D:

1. **Original features**: D dimensions
2. **Angle encoding**: 2D dimensions (cos and sin of D features)
3. **Pairwise interactions**: D(D+1)/2 dimensions (upper triangular)
4. **Fourier features**: ~√D dimensions (learnable)

**Total before capping**: D + 2D + D(D+1)/2 + fourier_dim
**Capped at**: 128 dimensions (configurable)

### Configuration

In `quantum_feature_mapper.py`, adjust:

```python
mapper = QuantumFeatureMapper(
    input_dim=16,              # Adapt to dataset
    use_angle_encoding=True,   # Toggle angle encoding
    use_interactions=True,     # Toggle pairwise interactions
    use_fourier=True,          # Toggle Fourier features
    fourier_dim=None,          # Auto-computed if None
    max_output_dim=128,        # Change dimension cap
    random_seed=42,            # Reproducibility
)
```

### Why Quantum Features Work

1. **Angle Encoding**: Maps features to unit circle, capturing periodicity and relationships
2. **Pairwise Interactions**: Captures feature relationships without combinatorial explosion
3. **Fourier Features**: Learn non-linear patterns with learnable parameters
4. **Dimension Capping**: Prevents feature explosion while preserving information

## Training Configuration

Edit `src/config.py` to adjust:

```python
TRAINING_CONFIG = {
    "epochs": 200,           # Max epochs
    "learning_rate": 0.001,  # Adam learning rate
    "weight_decay": 5e-4,    # L2 regularization
    "patience": 20,          # Early stopping patience
    "clip_grad_norm": 1.0,   # Gradient clipping
    "lr_scheduler": True,    # Learning rate scheduling
    ...
}

MODEL_CONFIG = {  # Baseline hyperparameters
    "hidden_channels": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.3,
}
```

Quantum model uses double capacity (defined in `03_train_quantum_gat.py`).

## Metrics Explained

### Classification Metrics

- **Accuracy**: % of correct predictions
- **Precision**: % of predicted positives that are truly positive (per-class average)
- **Recall**: % of actual positives correctly identified (per-class average)
- **F1 (Macro)**: Unweighted average F1 across classes (balanced for imbalanced data)
- **F1 (Micro)**: F1 computed on aggregate of TP/FP/FN (weighted by support)
- **ROC-AUC (OvR)**: Area under curve for one-vs-rest multi-class classification

### Why Macro F1?

Elliptic++ is imbalanced (non-fraud >> fraud classes). Macro F1 treats all classes equally, preventing bias toward majority class.

## Expected Performance

On Elliptic++ with proper class weighting:

| Metric | Baseline | Quantum | Expected Improvement |
|--------|----------|---------|----------------------|
| F1 (Macro) | 0.65-0.75 | 0.70-0.80 | +5-15% |
| Accuracy | 0.90-0.95 | 0.92-0.96 | +1-3% |
| ROC-AUC | 0.80-0.90 | 0.85-0.93 | +3-10% |

*Actual results depend on data quality, class balance, and random initialization.*

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors:
1. Reduce `max_output_dim` in quantum mapper (e.g., 64 instead of 128)
2. Reduce `hidden_channels` in model config
3. Use smaller batch... (note: this is node-level, entire graph is used)

### Poor Performance

If either model performs poorly:
1. Check class weights computation in training scripts
2. Verify data is properly normalized (01_load_data.py handles this)
3. Adjust learning rate, patience, dropout
4. Verify graph structure (check printed stats match expectations)

### Reproducibility

All scripts set random seeds:
```python
set_random_seeds(TRAINING_CONFIG['random_seed'])
```

To ensure 100% reproducibility:
- Use same GPU (different GPUs may have floating-point differences)
- Run scripts sequentially (don't run in parallel)
- Use `torch.use_deterministic_algorithms(True)` if needed (may impact performance)

## Integration with Existing Code

### Using Trained Models

```python
import torch
from src.models import GAT

# Load best model
model = GAT(
    in_channels=128,  # For quantum
    hidden_channels=128,
    out_channels=2,
    num_heads=6,
    num_layers=3,
)
model.load_state_dict(torch.load('artifacts/quantum_gat_best.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    logits = model(x, edge_index)
    predictions = logits.argmax(dim=1)
```

### Using Quantum Feature Mapper

```python
from scripts_imp2.quantum_feature_mapper import QuantumFeatureMapper

mapper = QuantumFeatureMapper(
    input_dim=16,
    use_angle_encoding=True,
    use_interactions=True,
    use_fourier=True,
)

x_expanded = mapper(x)  # [N, 16] -> [N, 128]
```

## Performance Comparisons vs scripts-imp1

- **Cleaner code**: Modular design, better comments
- **Better metrics**: Comprehensive tracking with per-class breakdown
- **Fair comparison**: Same training procedure for both models
- **Production-ready**: Error handling, device compatibility, reproducibility
- **Visualization**: Training curves & comparison plots included

## Next Steps

1. **Run all pipeline steps** end-to-end
2. **Analyze comparison report** to understand improvements
3. **Tune hyperparameters** if needed (see Training Configuration)
4. **Deploy best model** for inference
5. **Add explainability** (GradCAM, attention visualization, etc.)

## References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- Random Fourier Features: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.pdf
- Quantum Machine Learning: https://pennylane.ai/

## License

Part of the Kryptos Quantum-Blockchain Fraud Detection project.
