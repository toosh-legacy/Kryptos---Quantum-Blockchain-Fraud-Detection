# Baseline Models

Train and evaluate standard Graph Attention Networks (GAT) for Bitcoin fraud detection.

## Files

- **baseline_gat_training.ipynb** / **baseline_gat_training.py**: Full training pipeline for baseline GAT

## Model Architecture

```
Input (182 features)
    ↓
GAT Layer 1 (64 hidden, 4 heads)
    ↓
LayerNorm + ELU
    ↓
GAT Layer 2 (64 hidden, 4 heads)
    ↓
Classifier (2 classes)
```

## Training Details

- **Loss**: Weighted CrossEntropyLoss (handles class imbalance)
- **Optimizer**: Adam (lr=0.001, weight_decay=5e-4)
- **Scheduler**: Cosine Annealing (T_max=300 epochs)
- **Early Stopping**: patience=50, monitored on F1
- **Dropout**: 0.3

## Quick Start

```bash
python baseline_gat_training.py
```

## Output

Saves to `artifacts/`:
- `baseline_gat_best.pt` - Best model weights
- `baseline_gat_metrics.json` - Test metrics and hyperparameters

## Metrics Computed

- **F1 Score** (fraud class)
- **Accuracy**
- **Precision** & **Recall**
- **ROC-AUC**
- **Confusion Matrix**
- **Generalization Gaps** (train→val, val→test)

## Next Steps

After baseline training, train quantum variant:
- `../quantum_models/quantum_gat_training.ipynb`
