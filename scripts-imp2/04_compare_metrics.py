"""
04_compare_metrics.py: Compare baseline vs quantum-enhanced GAT models.

This script:
- Loads metrics from both baseline and quantum models
- Compares performance across all metrics
- Generates side-by-side visualizations
- Computes improvement percentages
- Creates comprehensive comparison report
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ARTIFACTS_DIR

print("=" * 70)
print("MODEL COMPARISON: BASELINE vs QUANTUM-ENHANCED GAT")
print("=" * 70)

# ===========================================================================
# LOAD METRICS
# ===========================================================================
print("\nLoading metrics...")

baseline_metrics_path = ARTIFACTS_DIR / "baseline_gat_metrics.json"
quantum_metrics_path = ARTIFACTS_DIR / "quantum_gat_metrics.json"

if not baseline_metrics_path.exists():
    print(f"✗ Baseline metrics not found: {baseline_metrics_path}")
    print("  Run: python 02_train_baseline_gat.py")
    sys.exit(1)

if not quantum_metrics_path.exists():
    print(f"✗ Quantum metrics not found: {quantum_metrics_path}")
    print("  Run: python 03_train_quantum_gat.py")
    sys.exit(1)

with open(baseline_metrics_path, 'r') as f:
    baseline = json.load(f)

with open(quantum_metrics_path, 'r') as f:
    quantum = json.load(f)

print(f"✓ Loaded baseline metrics from {baseline['timestamp']}")
print(f"✓ Loaded quantum metrics from {quantum['timestamp']}")

# ===========================================================================
# EXTRACT KEY METRICS
# ===========================================================================
baseline_test = baseline['test_metrics']
quantum_test = quantum['test_metrics']

baseline_hist = baseline['training_history']
quantum_hist = quantum['training_history']

# ===========================================================================
# COMPARISON TABLE
# ===========================================================================
print("\n" + "=" * 70)
print("TEST SET METRICS COMPARISON")
print("=" * 70)

metrics_to_compare = [
    ('Accuracy', 'accuracy'),
    ('F1 (Macro)', 'f1_macro'),
    ('F1 (Micro)', 'f1_micro'),
    ('Precision (Macro)', 'precision_macro'),
    ('Recall (Macro)', 'recall_macro'),
    ('ROC-AUC (OvR)', 'roc_auc_ovr'),
]

print(f"\n{'Metric':<20} {'Baseline':>12} {'Quantum':>12} {'Improvement':>15}")
print("-" * 70)

comparison_data = {}

for metric_name, metric_key in metrics_to_compare:
    baseline_val = baseline_test[metric_key]
    quantum_val = quantum_test[metric_key]
    
    if baseline_val > 0:
        improvement = ((quantum_val - baseline_val) / baseline_val) * 100
    else:
        improvement = 0
    
    comparison_data[metric_name] = {
        'baseline': baseline_val,
        'quantum': quantum_val,
        'improvement': improvement,
    }
    
    improvement_str = f"{improvement:+.2f}%" if improvement != 0 else "N/A"
    arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
    
    print(f"{metric_name:<20} {baseline_val:>12.4f} {quantum_val:>12.4f} {arrow} {improvement_str:>12}")

# ===========================================================================
# TRAINING EFFICIENCY
# ===========================================================================
print("\n" + "=" * 70)
print("TRAINING EFFICIENCY")
print("=" * 70)

baseline_epochs = len(baseline_hist['train_loss'])
quantum_epochs = len(quantum_hist['train_loss'])
baseline_time = baseline['training_time_seconds']
quantum_time = quantum['training_time_seconds']

print(f"\n{'Metric':<30} {'Baseline':>15} {'Quantum':>15}")
print("-" * 65)
print(f"{'Epochs trained':<30} {baseline_epochs:>15} {quantum_epochs:>15}")
print(f"{'Training time (seconds)':<30} {baseline_time:>15.1f} {quantum_time:>15.1f}")
print(f"{'Time per epoch (seconds)':<30} {baseline_time/baseline_epochs:>15.2f} {quantum_time/quantum_epochs:>15.2f}")

# ===========================================================================
# MODEL CAPACITY
# ===========================================================================
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

baseline_hp = baseline['hyperparameters']
quantum_hp = quantum['hyperparameters']

print(f"\n{'Parameter':<30} {'Baseline':>20} {'Quantum':>20}")
print("-" * 72)
print(f"{'Input features':<30} {baseline_hp.get('input_channels', 'N/A'):>20} {quantum_hp.get('input_channels', 'N/A'):>20}")
print(f"{'Hidden channels':<30} {baseline_hp['hidden_channels']:>20} {quantum_hp['hidden_channels']:>20}")
print(f"{'Num heads':<30} {baseline_hp['num_heads']:>20} {quantum_hp['num_heads']:>20}")
print(f"{'Num layers':<30} {baseline_hp['num_layers']:>20} {quantum_hp['num_layers']:>20}")
print(f"{'Dropout':<30} {baseline_hp['dropout']:>20} {quantum_hp['dropout']:>20}")

if 'quantum_mapper' in quantum_hp:
    print(f"\n{'Quantum Mapper Settings':<30} {'Disabled':>20} {'Enabled':>20}")
    print("-" * 72)
    qm = quantum_hp['quantum_mapper']
    print(f"{'  - Angle Encoding':<30} {'':>20} {'✓' if qm['use_angle_encoding'] else '✗':>20}")
    print(f"{'  - Pairwise Interactions':<30} {'':>20} {'✓' if qm['use_interactions'] else '✗':>20}")
    print(f"{'  - Fourier Features':<30} {'':>20} {'✓' if qm['use_fourier'] else '✗':>20}")
    print(f"{'  - Max output dimension':<30} {quantum_hp.get('input_features', 'N/A'):>20} {qm['max_output_dim']:>20}")
    if 'expanded_features' in quantum_hp:
        print(f"{'  - Expanded features':<30} {'':>20} {quantum_hp['expanded_features']:>20}")

# ===========================================================================
# VISUALIZATION
# ===========================================================================
print("\nGenerating comparison visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Baseline vs Quantum-Enhanced GAT', fontsize=16, fontweight='bold')

# Test metrics comparison
ax = axes[0, 0]
metrics_names = ['Accuracy', 'F1 Macro', 'F1 Micro', 'Precision', 'Recall', 'ROC-AUC']
baseline_vals = [baseline_test['accuracy'], baseline_test['f1_macro'], baseline_test['f1_micro'],
                  baseline_test['precision_macro'], baseline_test['recall_macro'], baseline_test['roc_auc_ovr']]
quantum_vals = [quantum_test['accuracy'], quantum_test['f1_macro'], quantum_test['f1_micro'],
                quantum_test['precision_macro'], quantum_test['recall_macro'], quantum_test['roc_auc_ovr']]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width/2, quantum_vals, width, label='Quantum', alpha=0.8, color='coral')

ax.set_ylabel('Score')
ax.set_title('Test Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Loss curves
ax = axes[0, 1]
ax.plot(baseline_hist['train_loss'], label='Baseline Train', linewidth=2, alpha=0.7)
ax.plot(baseline_hist['val_loss'], label='Baseline Val', linewidth=2, alpha=0.7)
ax.plot(quantum_hist['train_loss'], label='Quantum Train', linewidth=2, alpha=0.7, linestyle='--')
ax.plot(quantum_hist['val_loss'], label='Quantum Val', linewidth=2, alpha=0.7, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# F1 Macro curves
ax = axes[0, 2]
ax.plot(baseline_hist['val_f1_macro'], label='Baseline Val', linewidth=2, marker='o', markersize=4, alpha=0.7)
ax.plot(quantum_hist['val_f1_macro'], label='Quantum Val', linewidth=2, marker='s', markersize=4, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Macro F1')
ax.set_title('Validation Macro F1 Score')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy curves
ax = axes[1, 0]
ax.plot(baseline_hist['train_acc'], label='Baseline Train', linewidth=2, alpha=0.7)
ax.plot(baseline_hist['val_acc'], label='Baseline Val', linewidth=2, alpha=0.7)
ax.plot(quantum_hist['train_acc'], label='Quantum Train', linewidth=2, alpha=0.7, linestyle='--')
ax.plot(quantum_hist['val_acc'], label='Quantum Val', linewidth=2, alpha=0.7, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training & Validation Accuracy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Improvement percentages
ax = axes[1, 1]
improvements = [
    comparison_data['Accuracy']['improvement'],
    comparison_data['F1 (Macro)']['improvement'],
    comparison_data['F1 (Micro)']['improvement'],
    comparison_data['Precision (Macro)']['improvement'],
    comparison_data['Recall (Macro)']['improvement'],
]
improvement_metrics = ['Accuracy', 'F1 Macro', 'F1 Micro', 'Precision', 'Recall']
colors = ['green' if x > 0 else 'red' for x in improvements]

bars = ax.barh(improvement_metrics, improvements, color=colors, alpha=0.7)
ax.set_xlabel('Improvement (%)')
ax.set_title('Quantum Improvement over Baseline')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax.text(val, i, f' {val:+.2f}%', va='center', fontsize=9)

# Confusion matrices
ax = axes[1, 2]
baseline_cm = np.array(baseline_test['confusion_matrix'])
quantum_cm = np.array(quantum_test['confusion_matrix'])

# Normalize for visualization
baseline_cm_norm = baseline_cm / baseline_cm.sum(axis=1, keepdims=True)
im = ax.imshow(baseline_cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.set_title('Baseline Confusion Matrix (Normalized)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Add values to heatmap
for i in range(baseline_cm.shape[0]):
    for j in range(baseline_cm.shape[1]):
        text = ax.text(j, i, f'{baseline_cm_norm[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax)

plt.tight_layout()
comparison_plot_path = ARTIFACTS_DIR / "model_comparison.png"
plt.savefig(comparison_plot_path, dpi=100, bbox_inches='tight')
print(f"✓ Saved comparison plot: {comparison_plot_path}")
plt.close()

# ===========================================================================
# SAVE COMPARISON REPORT
# ===========================================================================
print("\nSaving comparison report...")

comparison_report = {
    'timestamp': datetime.now().isoformat(),
    'baseline_metrics_path': str(baseline_metrics_path),
    'quantum_metrics_path': str(quantum_metrics_path),
    'test_metrics_comparison': {
        metric_name: {
            'baseline': data['baseline'],
            'quantum': data['quantum'],
            'improvement_percent': data['improvement'],
            'quantum_better': data['quantum'] > data['baseline'],
        }
        for metric_name, data in comparison_data.items()
    },
    'summary': {
        'baseline_best_epoch': baseline['best_epoch'],
        'quantum_best_epoch': quantum['best_epoch'],
        'baseline_training_time': baseline['training_time_seconds'],
        'quantum_training_time': quantum['training_time_seconds'],
        'baseline_epochs_trained': len(baseline_hist['train_loss']),
        'quantum_epochs_trained': len(quantum_hist['train_loss']),
    },
}

# Calculate aggregate improvement
all_improvements = [v['improvement_percent'] for v in comparison_report['test_metrics_comparison'].values()]
avg_improvement = np.mean(all_improvements)

comparison_report['summary']['average_improvement_percent'] = float(avg_improvement)
comparison_report['summary']['quantum_wins'] = sum(1 for v in comparison_report['test_metrics_comparison'].values() if v['quantum_better'])
comparison_report['summary']['total_metrics'] = len(comparison_report['test_metrics_comparison'])

comparison_report_path = ARTIFACTS_DIR / "model_comparison_report.json"
with open(comparison_report_path, 'w') as f:
    json.dump(comparison_report, f, indent=2)
print(f"✓ Saved comparison report: {comparison_report_path}")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\n📊 OVERALL RESULTS:")
print(f"  Average Improvement: {avg_improvement:+.2f}%")
print(f"  Quantum Wins: {comparison_report['summary']['quantum_wins']}/{comparison_report['summary']['total_metrics']} metrics")

if avg_improvement > 0:
    print(f"\n✓ QUANTUM MODEL SHOWS IMPROVEMENT!")
    print(f"  The quantum-inspired feature expansion successfully enhances")
    print(f"  fraud detection performance on the Elliptic++ dataset.")
else:
    print(f"\n⚠ Baseline model performs better")
    print(f"  Consider tuning quantum mapper parameters or model architecture.")

print(f"\n📈 BEST PERFORMING METRICS:")
sorted_metrics = sorted(comparison_data.items(), key=lambda x: x[1]['improvement'], reverse=True)
for metric_name, data in sorted_metrics[:3]:
    if data['improvement'] >= 0:
        print(f"  • {metric_name}: {data['improvement']:+.2f}%")

print(f"\n⏱️  TRAINING EFFICIENCY:")
print(f"  Baseline: {baseline_epochs} epochs in {baseline_time:.1f}s")
print(f"  Quantum:  {quantum_epochs} epochs in {quantum_time:.1f}s")

print(f"\n💾 ARTIFACTS SAVED:")
print(f"  ✓ {comparison_plot_path.name}")
print(f"  ✓ {comparison_report_path.name}")

print("\n" + "=" * 70)
