# Validation Framework

Multi-seed statistical validation to ensure robust results.

## Files

- **multiseed_validation.ipynb** / **multiseed_validation.py**: Run models across multiple random seeds

## Purpose

Validates that improvements are:
- **Statistically significant** (paired t-test across seeds)
- **Reproducible** (results don't depend on one lucky run)
- **Generalizable** (not overfitting artifacts)

## Validation Protocol

Run both Baseline-GAT and QIGAT with 5 different random seeds each:

```
Seed 42:   Baseline ✓    QIGAT ✓
Seed 43:   Baseline ✓    QIGAT ✓
Seed 44:   Baseline ✓    QIGAT ✓
Seed 45:   Baseline ✓    QIGAT ✓
Seed 46:   Baseline ✓    QIGAT ✓
           ─────────────────────
           Paired t-test on F1 scores
```

## Output

Saves to `artifacts/`:
- `multiseed_results.json` - F1 scores for all (model × seed) combinations
- `validation_report.md` - Formatted statistical analysis
- `validation.log` - Full training logs

## Statistical Metrics

For each model (5 seeds):
- **Mean F1**: Average performance
- **Std F1**: Consistency/robustness
- **Min/Max F1**: Best/worst case scenarios

**Paired t-test**: Is QIGAT mean F1 statistically significantly > Baseline?

## Interpretation

**Success Criteria**:

| Metric | Success | Marginal | Failure |
|--------|---------|----------|---------|
| QIGAT Mean F1 - Baseline Mean F1 | > +0.02 | +0.01 to 0.02 | < +0.01 |
| QIGAT Std F1 | < 0.03 | 0.03-0.05 | > 0.05 |
| p-value (paired t-test) | < 0.05 | 0.05-0.10 | > 0.10 |
| Generalization gap | < 0.05 | 0.05-0.08 | > 0.08 |

## Quick Start

```bash
# Run full validation (5 seeds × 2 models ~7-10 hours on CPU)
python multiseed_validation.py
```

## Publication Readiness

✅ **Ready for publication if**:
- Mean QIGAT F1 > Mean Baseline F1 + 0.02
- p-value < 0.05 (statistically significant)
- Std < 0.03 (stable across seeds)
- Generalization gap < 0.05

## Next Steps

Use results to:
1. Submit to venue (conference/journal)
2. Create comparison plots for presentations
3. Document ablation studies
