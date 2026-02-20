# Quantum Models

Train and evaluate the Quantum-Inspired Graph Attention Network (QIGAT).

## Files

- **quantum_gat_training.ipynb** / **quantum_gat_training.py**: Full training pipeline for QIGAT

## Model Architecture

```
Input (182 features) [FULL dimension - NO compression]
    ↓
GAT Layer 1 (128 hidden, 4 heads)
    ↓
Quantum Phase Block
  ├─ Learned phase: φ = π * tanh(Wx)
  ├─ Phase encoding: cos(φ), sin(φ)  [128 → 256]
  └─ LayerNorm + Dropout
    ↓
Residual Connection [learnable scaling α]
    ↓
GAT Layer 2 (128 hidden, 4 heads)
    ↓
Classifier (2 classes)
```

## Key Innovation

**Quantum Phase Encoding** is applied **after** graph aggregation (post-GAT), not on raw features.

- Learned phase projection prevents arbitrary scaling
- Residual connection protects against representation collapse
- Proper capacity (128→256→128) for nonlinear expansion

## Training Details

- **Loss**: Weighted CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001, weight_decay=5e-4)
- **Scheduler**: Cosine Annealing (T_max=300 epochs)
- **Early Stopping**: patience=50, monitored on F1
- **Dropout**: 0.5 (quantum block) + 0.3 (GAT layers)

## Quick Start

```bash
python quantum_gat_training.py
```

## Output

Saves to `artifacts/`:
- `qigat_corrected_best.pt` - Best model weights
- `qigat_corrected_report.json` - Test metrics and architecture details

## Metrics Computed

Same as baseline, plus:
- **Quantum effect analysis**: QIGAT test F1 vs Baseline F1
- **Generalization gaps** for overfitting detection

## Expected Results

Based on validation framework:
- **Test F1**: ~0.89 (on Elliptic++)
- **Accuracy**: ~98%
- **Fraud Recall**: ~75-80% (detect most fraud)
- **Generalization Gap** (train→test): < 0.05 (good)

## Comparison with Baseline

The quantum module should:
1. ✅ Improve F1 on test set
2. ✅ Maintain or reduce generalization gaps
3. ✅ Have comparable or slightly higher parameter count

## Next Steps

After quantum training, run validation:
- `../validation/multiseed_validation.ipynb`
