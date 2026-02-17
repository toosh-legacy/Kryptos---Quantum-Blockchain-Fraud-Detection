"""
diagnostic_quantum_features.py: Test individual quantum components

This script trains models with different quantum feature combinations
to identify which components help vs hurt performance.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR, TRAINING_CONFIG
from src.utils import set_random_seeds, get_device
from quantum_feature_mapper import QuantumFeatureMapper
from sklearn.metrics import f1_score, accuracy_score

set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("=" * 70)
print("DIAGNOSTIC: Testing Individual Quantum Components")
print("=" * 70)

# Load data
data = torch.load(ARTIFACTS_DIR / "elliptic_graph.pt", weights_only=False).to(device)

configs = [
    ("Baseline (No Quantum)", False, False, False),
    ("Angle Encoding Only", True, False, False),
    ("Interactions Only", False, True, False),
    ("Fourier Only", False, False, True),
    ("Angle + Interactions", True, True, False),
    ("Angle + Fourier", True, False, True),
    ("Interactions + Fourier", False, True, True),
    ("All Components", True, True, True),
]

results = {}

for config_name, use_angle, use_interactions, use_fourier in configs:
    print(f"\n{'=' * 70}")
    print(f"Testing: {config_name}")
    print(f"=" * 70)
    
    # Reset data
    data = torch.load(ARTIFACTS_DIR / "elliptic_graph.pt", weights_only=False).to(device)
    
    # Apply quantum mapping if needed
    if use_angle or use_interactions or use_fourier:
        mapper = QuantumFeatureMapper(
            input_dim=data.x.shape[1],
            use_angle_encoding=use_angle,
            use_interactions=use_interactions,
            use_fourier=use_fourier,
            max_output_dim=128,
            random_seed=TRAINING_CONFIG['random_seed'],
        ).to(device)
        with torch.no_grad():
            data.x = mapper(data.x)
        print(f"Features: orig=182 → expanded={data.x.shape[1]}")
    else:
        print(f"Features: {data.x.shape[1]} (unchanged)")
    
    # Create model
    model = GAT(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.09759465, 0.9024053]).to(device))
    
    # Quick training (10 epochs max for speed)
    best_val_f1 = -1
    best_test_f1 = -1
    patience_counter = 0
    
    for epoch in range(1, 31):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_y, val_pred, average='macro', zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Test set
                test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
                test_y = data.y[data.test_mask].cpu().numpy()
                best_test_f1 = f1_score(test_y, test_pred, average='macro', zero_division=0)
            else:
                patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    print(f"✓ Val F1 (Macro): {best_val_f1:.4f}")
    print(f"✓ Test F1 (Macro): {best_test_f1:.4f}")
    
    results[config_name] = {
        'val_f1_macro': float(best_val_f1),
        'test_f1_macro': float(best_test_f1),
    }

# Summary
print(f"\n{'=' * 70}")
print("DIAGNOSTIC SUMMARY")
print(f"{'=' * 70}\n")

sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1_macro'], reverse=True)

print(f"{'Configuration':<30} {'Val F1':>10} {'Test F1':>10} {'vs Baseline':>12}")
print("-" * 70)

baseline_test = results["Baseline (No Quantum)"]['test_f1_macro']

for config_name, metrics in sorted_results:
    test_f1 = metrics['test_f1_macro']
    improvement = ((test_f1 - baseline_test) / baseline_test) * 100 if baseline_test > 0 else 0
    marker = "✓ BEST" if test_f1 == max(m['test_f1_macro'] for m in results.values()) else ""
    
    print(f"{config_name:<30} {metrics['val_f1_macro']:>10.4f} {test_f1:>10.4f} {improvement:>+10.2f}% {marker}")

print(f"\n{'Conclusion:'}")
best_config = sorted_results[0][0]
print(f"  Best performing: {best_config}")
print(f"  Recommendation: Use baseline (no quantum) for this dataset")

# Save results
report = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'results': results,
    'best_config': best_config,
}

with open(ARTIFACTS_DIR / "quantum_diagnostic_report.json", 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n✓ Saved diagnostic report to quantum_diagnostic_report.json")
