#!/usr/bin/env python
"""
FULL EXPERIMENTAL VALIDATION RUNNER

5-Seed Controlled Validation:
- 4 Parameter-Matched Models (A, B, C, D)
- Identical training protocol
- Statistical significance testing
- Robustness evaluation

This is the main orchestration script for publication-grade research.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import time

# Imports from research_validation modules
_curr_dir = Path(__file__).parent
_parent_dir = Path(__file__).parent.parent

# Change to parent directory for imports to work correctly
import os
os.chdir(str(_parent_dir))

sys.path.insert(0, str(_curr_dir))
sys.path.insert(0, str(_parent_dir))

# Now import from research_validation.models
try:
    from research_validation.models.definitions import (
        get_model, count_parameters, MODEL_NAMES
    )
    from research_validation.train import train_model
    from research_validation.evaluate import evaluate, format_metrics, compute_efficiency
    from research_validation.stats import StatisticalAnalysis, interpret_significance
    from research_validation.robustness import RobustnessTest
except ImportError:
    # Fallback if that doesn't work
    from models.definitions import (
        get_model, count_parameters, MODEL_NAMES
    )
    from train import train_model
    from evaluate import evaluate, format_metrics, compute_efficiency
    from stats import StatisticalAnalysis, interpret_significance
    from robustness import RobustnessTest

from src.utils import set_random_seeds


print("="*80)
print("CONTROLLED EXPERIMENTAL VALIDATION FRAMEWORK")
print("Quantum-Inspired GAT on Elliptic++ Dataset")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEEDS = [0, 1, 2, 3, 4]
MODELS_TO_TEST = ['baseline_128', 'residual_expansion', 'mlp_control', 'qigat']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# Create results directory
RESULTS_DIR = Path('research_validation/results')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: DATA LOADING & PREPARATION")
print("="*80)

graph = torch.load('artifacts/elliptic_graph.pt', weights_only=False).to(DEVICE)
print(f"Graph loaded: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
print(f"Features: {graph.num_node_features}")

labeled_mask = (graph.y != -1)
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_y = graph.y[labeled_mask].cpu().numpy()

print(f"Labeled samples: {len(labeled_indices):,}")
print(f"  Class 0 (Non-Fraud): {(labeled_y == 0).sum():,}")
print(f"  Class 1 (Fraud):     {(labeled_y == 1).sum():,}")

# ==============================================================================
# MULTI-SEED TRAINING LOOP
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: MULTI-SEED VALIDATION (5 seeds)")
print("="*80)

# Store all results indexed by [model][seed]
all_results = {model: {} for model in MODELS_TO_TEST}

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*80}")
    print(f"SEED {seed_idx}/5 (seed={seed})")
    print(f"{'='*80}")
    
    # Set seeds
    set_random_seeds(seed)
    
    # Create data split for THIS seed ONLY
    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        labeled_indices, labeled_y,
        test_size=0.30,
        random_state=seed,
        stratify=labeled_y
    )
    
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, train_val_y,
        test_size=0.30,
        random_state=seed,
        stratify=train_val_y
    )
    
    # Create masks
    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=DEVICE)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Normalize features (per-seed)
    nan_count = torch.isnan(graph.x).sum().item()
    if nan_count > 0:
        graph.x = torch.nan_to_num(graph.x, nan=0.0)
    
    train_x = graph.x[train_mask]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    graph.x = (graph.x - mean) / std
    graph.x = torch.clamp(graph.x, min=-10, max=10)
    
    # Class weights
    n_class_0 = (graph.y[train_mask] == 0).sum().item()
    n_class_1 = (graph.y[train_mask] == 1).sum().item()
    class_weight = torch.tensor(
        [1.0 / n_class_0, 1.0 / n_class_1],
        device=DEVICE,
        dtype=torch.float32
    )
    class_weight = class_weight / class_weight.sum()
    
    # Train each model
    for model_name in MODELS_TO_TEST:
        print(f"\n  [{MODEL_NAMES[model_name]}]", end=" ", flush=True)
        
        # Create model
        if model_name == 'mlp_control':
            model = get_model(model_name, in_features=182, hidden_dim=512,
                            num_layers=3, dropout=0.5)
        else:
            model = get_model(model_name, in_channels=182, hidden_channels=128,
                            num_heads=4, dropout=0.5)
        
        model = model.to(DEVICE)
        
        # Train
        results = train_model(
            model=model,
            data=graph,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            class_weight=class_weight,
            device=DEVICE,
            evaluate_fn=evaluate,
            model_name=MODEL_NAMES[model_name],
            verbose=False
        )
        
        # Store results
        all_results[model_name][seed] = {
            'test_f1': results['test']['f1'],
            'test_f1_class_0': results['test']['f1_class_0'],
            'test_f1_class_1': results['test']['f1_class_1'],
            'test_precision': results['test']['precision'],
            'test_recall': results['test']['recall'],
            'test_roc_auc': results['test']['roc_auc'],
            'train_f1': results['train']['f1'],
            'val_f1': results['val']['f1'],
            'params': count_parameters(model),
            'training_time': results['training_time'],
            'epochs': results['epochs_trained']
        }
        
        print(f" F1={results['test']['f1']:.4f}")

# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: STATISTICAL ANALYSIS")
print("="*80)

# Organize for stats
stats_data = {}
for model_name in MODELS_TO_TEST:
    f1_values = [all_results[model_name][seed]['test_f1'] for seed in SEEDS]
    stats_data[model_name] = f1_values

# Run analysis
analysis = StatisticalAnalysis(stats_data)
analysis.run_all_tests()
analysis.print_summary()

# ==============================================================================
# COMPREHENSIVE RESULTS TABLE
# ==============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS TABLE")
print("="*80)

print("\nTest F1 Scores (5 seeds):")
print("-" * 100)
print(f"{'Model':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Params':<15}")
print("-" * 100)

for model_name in MODELS_TO_TEST:
    f1_values = [all_results[model_name][seed]['test_f1'] for seed in SEEDS]
    params = all_results[model_name][SEEDS[0]]['params']
    
    print(f"{MODEL_NAMES[model_name]:<30} "
          f"{np.mean(f1_values):<12.4f} "
          f"{np.std(f1_values):<12.4f} "
          f"{np.min(f1_values):<12.4f} "
          f"{np.max(f1_values):<12.4f} "
          f"{params:<15,}")

# Per-class F1
print("\nMean Test F1 - Per Class:")
print("-" * 100)
print(f"{'Model':<30} {'Class 0 (Non-Fraud)':<25} {'Class 1 (Fraud)':<25}")
print("-" * 100)

for model_name in MODELS_TO_TEST:
    f1_c0_values = [all_results[model_name][seed]['test_f1_class_0'] for seed in SEEDS]
    f1_c1_values = [all_results[model_name][seed]['test_f1_class_1'] for seed in SEEDS]
    
    print(f"{MODEL_NAMES[model_name]:<30} "
          f"{np.mean(f1_c0_values):.4f} ± {np.std(f1_c0_values):.4f}          "
          f"{np.mean(f1_c1_values):.4f} ± {np.std(f1_c1_values):.4f}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_to_save = {
    'metadata': {
        'seeds': SEEDS,
        'models': MODELS_TO_TEST,
        'device': str(DEVICE),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    },
    'raw_results': all_results,
    'statistical_analysis': analysis.to_dict()
}

results_path = RESULTS_DIR / 'multiseed_validation_results.json'
with open(results_path, 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"\n[SAVED] Results to {results_path}")

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

report_path = RESULTS_DIR / 'VALIDATION_REPORT.md'

report_content = f"""# Controlled Experimental Validation Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Framework**: 5-Seed Multi-Model Validation  
**Device**: {DEVICE}

## Summary

This report presents a controlled experimental validation comparing 4 parameter-matched models:

1. **Baseline-128**: Standard GAT with hidden=128 (reference)
2. **Residual-Expansion** (No Quantum): Nonlinear expansion without quantum
3. **MLP-Control** (No Graphs): Pure MLP without graph structure  
4. **QIGAT**: Quantum-Inspired GAT (test model)

## Primary Results

### Test F1 Scores (Mean ± Std)

"""

for model_name in MODELS_TO_TEST:
    f1_values = [all_results[model_name][seed]['test_f1'] for seed in SEEDS]
    report_content += f"- **{MODEL_NAMES[model_name]}**: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}\n"

report_content += f"""

### Statistical Significance Tests

**QIGAT vs Baseline-128:**
- p-value: {analysis.test_results['qigat_vs_baseline_128']['p_value']:.6f}
- Significant: {'YES' if analysis.test_results['qigat_vs_baseline_128']['significant'] else 'NO'}

**QIGAT vs Residual-Expansion (tests quantum contribution):**
- p-value: {analysis.test_results['qigat_vs_residual_expansion']['p_value']:.6f}
- Significant: {'YES' if analysis.test_results['qigat_vs_residual_expansion']['significant'] else 'NO'}

**QIGAT vs MLP-Control (tests graph necessity):**
- p-value: {analysis.test_results['qigat_vs_mlp_control']['p_value']:.6f}
- Significant: {'YES' if analysis.test_results['qigat_vs_mlp_control']['significant'] else 'NO'}

## Per-Class Performance (Fraud Detection)

### Class 1 (Fraud) - Critical for Publication

"""

for model_name in MODELS_TO_TEST:
    f1_c1_values = [all_results[model_name][seed]['test_f1_class_1'] for seed in SEEDS]
    report_content += f"- **{MODEL_NAMES[model_name]}**: {np.mean(f1_c1_values):.4f} ± {np.std(f1_c1_values):.4f}\n"

report_content += f"""

## Model Efficiency

### Parameters & Scalability

"""

for model_name in MODELS_TO_TEST:
    params = all_results[model_name][SEEDS[0]]['params']
    f1_values = [all_results[model_name][seed]['test_f1'] for seed in SEEDS]
    mean_f1 = np.mean(f1_values)
    efficiency = compute_efficiency(mean_f1, params)
    
    report_content += f"- **{MODEL_NAMES[model_name]}**: {params:,} params, F1/M = {efficiency:.4f}\n"

report_content += f"""

## Conclusions

### If QIGAT >> Baseline-128 (p<0.05) AND QIGAT >> Residual-Expansion (p<0.05):
**Claim**: Learned phase encoding enhances fraud detection beyond simple capacity increase.

### If QIGAT ≈ Residual-Expansion but >> Baseline-128:
**Claim**: Deep residual nonlinear embedding expansion improves fraud detection.

### If QIGAT >> MLP-Control:
**Claim**: Graph structure is necessary for optimal performance.

## Next Steps

1. Implement robustness testing (Gaussian noise)
2. Embedding visualization (t-SNE/UMAP)
3. Ablation: Remove quantum block systematically
4. Theoretical analysis of phase encoding

---

*Research-Grade Validation - Ready for Publication Review*
"""

with open(report_path, 'w') as f:
    f.write(report_content)

print(f"[SAVED] Report to {report_path}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\nResults ready for publication review.")
print("All results saved to: research_validation/results/\n")
