# Research Validation Framework

## Overview

This directory contains a publication-grade experimental validation framework for the Quantum-Inspired Graph Attention Network (QIGAT).

## Structure

```
research_validation/
├── models/
│   └── definitions.py         # 4 model architectures (A, B, C, D)
├── train.py                   # Unified training loop
├── evaluate.py                # Metrics computation
├── stats.py                   # Statistical analysis
├── robustness.py              # Noise robustness testing
├── runner.py                  # Main orchestration script
├── results/                   # Output directory
│   ├── multiseed_validation_results.json
│   └── VALIDATION_REPORT.md
└── README.md                  # This file
```

## Models Implemented

### Model A: Baseline-128 (Fair Control)
- Standard GAT with hidden=128
- ~95K parameters
- Tests baseline capacity

### Model B: Residual-Expansion-GAT (No Quantum)
- Same as QIGAT but replaces quantum block with linear nonlinear expansion
- ~1.15M parameters (matches QIGAT)
- **Tests:** Does quantum block matter, or is it just capacity?

### Model C: MLP-Control (No Graphs)
- Pure deep MLP, no graph structure
- ~1.15M parameters (matches QIGAT)
- **Tests:** Are graphs necessary, or can nonlinearity alone work?

### Model D: QIGAT (Quantum-Inspired)
- Full QIGAT with learned phase encoding
- ~1.15M parameters
- **Test model**: Does quantum contribute significantly?

## Running the Full Validation

```bash
cd research_validation
python runner.py 2>&1 | tee results/validation.log
```

**Expected time**: ~5-7 hours (CPU, 4 models × 5 seeds = 20 training runs)

## Output Files

After completion:

1. **multiseed_validation_results.json**
   - Raw results: test F1, per-class F1, precision, recall, ROC-AUC for each seed/model
   - Statistical test results
   - Parameter counts

2. **VALIDATION_REPORT.md**
   - Formatted results table
   - Statistical significance tests
   - Per-class breakdown
   - Model efficiency analysis
   - Ready for publication review

## Results Interpretation

### If Results Show:

#### QIGAT >> Baseline-128 (p < 0.05) ✅
→ QIGAT is legitimately better, not just lucky

#### QIGAT >> Residual-Expansion (p < 0.05) ✅
→ Quantum phase encoding matters (not just capacity)

#### QIGAT >> MLP-Control (p < 0.05) ✅
→ Graph structure is necessary

**If all three hold:** QIGAT is publication-ready

### Red Flags ⚠️

- **High variance** (std > 0.05): Model is unstable
- **QIGAT ≈ Residual-Expansion**: Quantum doesn't matter
- **QIGAT ≈ MLP**: Graphs don't matter
- **p-value > 0.05 on all tests**: No significance

## Key Parameters

All models trained with:
- **Optimizer**: Adam (lr=0.001, weight_decay=5e-4)
- **Loss**: Weighted CrossEntropy
- **Scheduler**: Cosine Annealing (T_max=300)
- **Early Stopping**: patience=50
- **Dropout**: 0.5
- **Gradient Clipping**: max_norm=1.0
- **Batch**: Full-batch (graph mode)

## Data Protocol

- **5 Seeds**: [0, 1, 2, 3, 4]
- **Train/Val/Test**: 49% / 21% / 30%
- **Stratified Split**: Maintains class distribution
- **Normalization**: Per-seed training mean/std

## Statistical Testing

Uses **paired t-tests** (n=5):
- Significance threshold: p < 0.05
- Tests:
  1. QIGAT vs Baseline-128
  2. QIGAT vs Residual-Expansion
  3. QIGAT vs MLP-Control

## Extension: Robustness Testing

After main validation, you can add robustness testing:

```python
from robustness import RobustnessTest

# Add to runner.py
test = RobustnessTest(model, graph, test_mask, device)
results = test.test_noise_robustness(evaluate, noise_levels=[0.01, 0.05])
```

This tests:
- F1 under σ=0.01 Gaussian noise
- F1 under σ=0.05 Gaussian noise
- Degradation compared to clean

**Claim**: "QIGAT degrades less under noise, indicating robust learned representations"

## Citation

If you use this framework:

```bibtex
@article{kryptos2026,
  title={Controlled Experimental Validation of Quantum-Inspired Graph Neural Networks},
  author={UT Dallas ACM Research},
  year={2026},
  note={Research Validation Framework}
}
```

---

**Status**: Publication-Grade Validation  
**Last Updated**: February 17, 2026
