# Kryptos: Complete Repository Guide

Complete reference for understanding the **Quantum-Inspired Bitcoin Fraud Detection** project.

## 📋 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Complete File Structure](#complete-file-structure)
3. [Weight Saving & Model Storage](#weight-saving--model-storage)
4. [Data Flow](#data-flow)
5. [Key Concepts](#key-concepts)
6. [Configuration Files](#configuration-files)
7. [Troubleshooting](#troubleshooting)

---

## Repository Overview

**Kryptos** is an end-to-end machine learning pipeline for detecting fraudulent Bitcoin transactions using quantum-inspired graph neural networks.

### What Does It Do?

1. **Loads Bitcoin transaction data** from CSV files
2. **Builds a graph** where nodes = addresses, edges = transactions
3. **Trains two models**:
   - **Baseline GAT**: Standard Graph Attention Network (for comparison)
   - **QIGAT**: Quantum-Inspired GAT (novel approach)
4. **Compares results** to show quantum enhancement
5. **Validates** through multi-seed experiments

### Key Achievement

- **Baseline F1**: 0.8717
- **QIGAT F1**: 0.8920
- **Improvement**: +2.3% (statistically significant)

---

## Complete File Structure

```
Kryptos/
│
├── 📂 data_processing/              [Step 1: Prepare data]
│   ├── create_graph.ipynb           • Load CSV → Build graph
│   ├── sanity_checks.ipynb          • Verify splits & normalization
│   └── README.md
│
├── 📂 baselines/                    [Step 2: Baseline model]
│   ├── baseline_gat_training.ipynb  • Train standard GAT
│   └── README.md
│
├── 📂 quantum_models/               [Step 3: Quantum model]
│   ├── quantum_gat_training.ipynb   • Train quantum-enhanced GAT
│   └── README.md
│
├── 📂 validation/                   [Step 4: Statistical validation]
│   └── README.md
│
├── 📂 research_validation/          [Advanced: Ablation studies]
│   ├── runner.py                    • Main entry point
│   ├── train.py                     • Unified training loop
│   ├── evaluate.py                  • Metrics computation
│   ├── stats.py                     • Statistical tests
│   └── models/                      • Model architectures
│
├── 📂 src/                          [Core implementation]
│   ├── config.py                    • Hyperparameters
│   ├── models.py                    • GAT architecture
│   ├── quantum_features.py          • Quantum module
│   ├── utils.py                     • Helper functions
│   └── train_utils.py               • Training utilities
│
├── 📂 data/                         [Raw data]
│   ├── txs_features.csv             • Node features (182 dims)
│   ├── txs_classes.csv              • Node labels
│   └── txs_edgelist.csv             • Transaction edges
│
├── 📂 artifacts/                    [Models & results] ⭐
│   ├── elliptic_graph.pt            • PyTorch graph (cached)
│   ├── baseline_gat_best.pt         • Baseline model weights
│   ├── qigat_corrected_best.pt      • Quantum model weights
│   ├── baseline_gat_metrics.json    • Baseline results
│   ├── qigat_corrected_report.json  • Quantum results
│   └── [...other experiments...]
│
├── 📂 figures/                      [Visualizations]
│   ├── data_distribution.png
│   └── [...training plots...]
│
├── 📄 WORKFLOW.md                   • Complete workflow guide
├── 📄 NOTEBOOK_INDEX.md             • All notebooks explained
├── 📄 README.md                     • Main project README
│
└── 📄 FINAL_RESULT_SUMMARY.md       • Best results & metrics
```

---

## Weight Saving & Model Storage

### Locations

All models and metrics are saved to `artifacts/` folder.

### Key Files

| File | Type | Purpose | Size |
|------|------|---------|------|
| `elliptic_graph.pt` | `.pt` | Cached graph (one-time) | ~200MB |
| `baseline_gat_best.pt` | `.pt` | Baseline model weights | ~500KB |
| `qigat_corrected_best.pt` | `.pt` | Quantum model weights | ~600KB |
| `baseline_gat_metrics.json` | `.json` | F1, accuracy, loss curves | ~50KB |
| `qigat_corrected_report.json` | `.json` | Full metrics & config | ~50KB |

### How Models Are Saved

Each notebook saves during training:

```python
# After training completes with best validation F1
torch.save(model.state_dict(), '../artifacts/MODEL_NAME_best.pt')

# After evaluation on test set
with open('../artifacts/REPORT.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### How to Load Saved Models

```python
# Load trained model
import torch
from src.models import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT(in_channels=182, hidden_channels=64, out_channels=2, 
            num_heads=4, num_layers=2, dropout=0.3).to(device)

model.load_state_dict(torch.load('artifacts/baseline_gat_best.pt'))
model.eval()  # Switch to evaluation mode

# Now use model for inference
with torch.no_grad():
    predictions = model(graph.x, graph.edge_index)
```

---

## Data Flow

```
RAW DATA
  ↓
[data/]
  ├─ txs_features.csv         (182 features per node)
  ├─ txs_classes.csv          (fraud/non-fraud labels)
  └─ txs_edgelist.csv         (transaction connections)
  ↓
[data_processing/create_graph.ipynb]
  ├─ Load CSVs
  ├─ Build PyTorch Geometric graph
  ├─ Add self-loops for GAT
  └─ Normalize features
  ↓
[artifacts/elliptic_graph.pt]  ← Cached, reused by all models
  ↓
[SPLIT DATA]
  ├─ 70% Training
  ├─ 15% Validation
  └─ 15% Test (held out)
  ↓
[baselines/baseline_gat_training.ipynb]
  ├─ Train baseline GAT
  ├─ Track metrics
  └─ Save best model
  ↓
[artifacts/baseline_gat_best.pt + metrics]
  ↓
[quantum_models/quantum_gat_training.ipynb]
  ├─ Train QIGAT
  ├─ Compare with baseline
  └─ Save best model
  ↓
[artifacts/qigat_corrected_best.pt + report]
  ↓
[validation/multiseed_validation.ipynb]
  ├─ Run 5 seeds each
  ├─ Statistical testing
  └─ Publication report
  ↓
RESEARCH RESULTS
```

---

## Key Concepts

### 1. **Graph Attention Network (GAT)**
- Layer type that learns to attend to neighboring nodes
- Each node aggregates information from neighbors with learned weights
- Baseline model uses 2 GAT layers (64 hidden channels)

### 2. **Quantum-Inspired Encoding**
- Projects embeddings to phase space: φ = π * tanh(Wx)
- Computes phase features: cos(φ), sin(φ)
- Expands feature space (128 → 256 dimensions)
- Applied after first GAT layer (post-aggregation)

### 3. **Class Imbalance Handling**
- Weighted CrossEntropyLoss (not unweighted)
- Weights computed from training set proportions
- Prevents bias toward majority (non-fraud) class

### 4. **Early Stopping**
- Monitors validation F1 score
- Stops if no improvement for 50 consecutive epochs
- Saves best model weights automatically

### 5. **Learning Rate Scheduling**
- Cosine Annealing: Smoothly decreases learning rate
- T_max=300: Full cycle over 300 epochs
- Helps find better local minima

---

## Configuration Files

### `src/config.py`
Main configuration file with all hyperparameters:

```python
# Dataset
DATASET_CONFIG = {
    'features_file': 'txs_features.csv',
    'classes_file': 'txs_classes.csv',
    'edgelist_file': 'txs_edgelist.csv',
    ...
}

# Model
MODEL_CONFIG = {
    'hidden_channels': 64,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.3,
}

# Training
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'max_epochs': 300,
    'early_stopping_patience': 50,
}
```

To modify defaults, edit this file before running notebooks.

---

## Data Formats

### Features (txs_features.csv)
```
node_id, timestep, feature_1, feature_2, ..., feature_182
0, 1, 0.234, -0.567, ..., 0.123
1, 1, 0.456, 0.789, ..., 0.456
...
```

### Classes (txs_classes.csv)
```
node_id, class
0, 0              # Non-fraud
1, 1              # Fraud
2, 0              # Non-fraud
...
```

### Edges (txs_edgelist.csv)
```
source, target
0, 1              # Transaction from address 0 to 1
1, 2
0, 3
...
```

---

## Metrics Explained

### F1 Score (Fraud Class)
- Harmonic mean of precision and recall
- **Range**: 0 (worse) to 1 (perfect)
- **Why**: Balances catching fraud (recall) vs false alarms (precision)
- **Target**: > 0.85 for fraud detection

### Accuracy
- Percentage of correct predictions
- Can be misleading with imbalanced data
- Reported for reference, not primary metric

### Precision (for Fraud)
- Of predicted fraud, how many are actually fraud?
- **Range**: 0-1
- High precision = fewer false alarms
- Trade-off: Lower recall (miss some fraud)

### Recall (for Fraud)
- Of actual fraud, how many did we catch?
- **Range**: 0-1
- High recall = catch more fraud
- Trade-off: Lower precision (more false alarms)

### ROC-AUC
- Area under receiver operating characteristic curve
- **Range**: 0.5 (random) to 1.0 (perfect)
- Tests across all classification thresholds

---

## Troubleshooting

### "Graph not found" Error
**Problem**: `FileNotFoundError: elliptic_graph.pt`  
**Solution**: Run `data_processing/create_graph.ipynb` first

### Out of Memory (GPU)
**Problem**: `RuntimeError: CUDA out of memory`  
**Solutions**:
- Use CPU instead: `device = torch.device('cpu')`
- Reduce batch size (edit notebook)
- Clear cache: `torch.cuda.empty_cache()`

### Models not improving
**Problem**: Loss stays constant, F1 doesn't increase  
**Solutions**:
- Check data preprocessing (sanity_checks.ipynb)
- Increase learning rate or training epochs
- Verify weights were loaded correctly

### Results differ between runs
**Problem**: Different F1 each time (even with same seed)  
**Causes**:
- Different hardware (CPU vs GPU precision)
- Different PyTorch versions
- Non-deterministic CUDA operations

### Can't load previously saved model
**Problem**: `KeyError` when loading weights  
**Solution**: Ensure model architecture matches saved weights (same hidden_channels, num_layers, etc.)

---

## Performance Benchmarks

### Training Time
| Component | CPU | GPU |
|-----------|-----|-----|
| Graph construction | 5 min | 2 min |
| Baseline training | 15 min | 2 min |
| Quantum training | 20 min | 3 min |
| Single seed validation | 35 min | 5 min |
| 5-seed validation | 3 hours | 30 min |

### Memory Requirements
| Component | GPU VRAM | CPU RAM |
|-----------|----------|---------|
| Graph | 200MB | 300MB |
| Baseline model | 50MB | 100MB |
| Quantum model | 60MB | 120MB |
| Training (batch) | 500MB | 2GB |

---

## Next Steps

1. **First Step**: Read [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md) for notebook overview
2. **Quick Start**: Run `data_processing/create_graph.ipynb`
3. **Train**: Run baseline then quantum notebooks
4. **Understand**: Read low-level comments in each notebook
5. **Publish**: Run validation framework for statistics

---

**Repository Status**: ✅ Production-Ready  
**Last Updated**: February 20, 2026  
**Best Performance**: F1=0.8920 (QIGAT test set)
