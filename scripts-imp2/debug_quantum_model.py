#!/usr/bin/env python
"""
Debug: Investigate why optimized quantum model performs poorly on test set.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import traceback
from src.models import GAT
from quantum_feature_mapper import QuantumFeatureMapper

device = 'cpu'

print("=" * 80)
print("DEBUGGING QUANTUM MODEL PERFORMANCE")
print("=" * 80)

# Load graph
print("\n1. Loading graph...")
graph = torch.load('artifacts/elliptic_graph.pt', map_location=device, weights_only=False).to(device)
print(f"   Features shape: {graph.x.shape}")
print(f"   Features min/max: {graph.x.min():.4f} / {graph.x.max():.4f}")
print(f"   Features mean/std: {graph.x.mean():.4f} / {graph.x.std():.4f}")

# Create quantum mapper
print("\n2. Creating quantum feature mapper...")
try:
    quantum_mapper = QuantumFeatureMapper(input_dim=graph.num_node_features)
    quantum_mapper = quantum_mapper.to(device)
    quantum_mapper.eval()
    print("   ✓ Quantum mapper created successfully")
except Exception as e:
    print(f"   ✗ Error creating quantum mapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Apply mapping to first batch
x_quantum = quantum_mapper(graph.x)
print(f"   Original shape: {graph.x.shape}")
print(f"   Quantum shape: {x_quantum.shape}")
print(f"   Quantum features min/max: {x_quantum.min():.4f} / {x_quantum.max():.4f}")
print(f"   Quantum features mean/std: {x_quantum.mean():.4f} / {x_quantum.std():.4f}")

# Load model
print("\n3. Loading optimized quantum model...")
model = GAT(
    in_channels=x_quantum.shape[1],
    hidden_channels=96,
    out_channels=2,
    num_heads=6,
    num_layers=3,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load('artifacts/quantum_gat_optimized_best.pt', map_location=device, weights_only=False))
model.eval()

print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\n4. Testing forward pass...")
with torch.no_grad():
    out = model(x_quantum, graph.edge_index)
    print(f"   Output shape: {out.shape}")
    print(f"   Output values min/max: {out.min():.4f} / {out.max():.4f}")
    print(f"   Output values mean/std: {out.mean():.4f} / {out.std():.4f}")
    
    # Check logits distribution
    logits_0 = out[:, 0]
    logits_1 = out[:, 1]
    print(f"\n   Logits for class 0: min={logits_0.min():.4f}, max={logits_0.max():.4f}, mean={logits_0.mean():.4f}")
    print(f"   Logits for class 1: min={logits_1.min():.4f}, max={logits_1.max():.4f}, mean={logits_1.mean():.4f}")
    
    # Get predictions
    pred = out.argmax(dim=1)
    print(f"\n   Predictions unique values: {torch.unique(pred)}")
    print(f"   Prediction distribution:")
    for cls in torch.unique(pred):
        count = (pred == cls).sum().item()
        pct = 100 * count / pred.numel()
        print(f"     Class {cls}: {count:,} ({pct:.2f}%)")
    
    # Get probabilities
    probs = F.softmax(out, dim=1)
    print(f"\n   Softmax probs min/max: {probs.min():.6f} / {probs.max():.6f}")
    print(f"   Softmax probs class 0 range: {probs[:, 0].min():.6f} to {probs[:, 0].max():.6f}")
    print(f"   Softmax probs class 1 range: {probs[:, 1].min():.6f} to {probs[:, 1].max():.6f}")

# Check test set
print("\n5. Analyzing test set predictions...")
test_mask = graph.test_mask
y_test = graph.y[test_mask]

with torch.no_grad():
    out_test = model(x_quantum, graph.edge_index)[test_mask]
    pred_test = out_test.argmax(dim=1)

print(f"   Test set size: {test_mask.sum().item()}")
print(f"   Test true labels distribution:")
for cls in torch.unique(y_test):
    count = (y_test == cls).sum().item()
    pct = 100 * count / y_test.numel()
    print(f"     Class {cls}: {count:,} ({pct:.2f}%)")

print(f"\n   Test predictions distribution:")
for cls in torch.unique(pred_test):
    count = (pred_test == cls).sum().item()
    pct = 100 * count / pred_test.numel()
    print(f"     Class {cls}: {count:,} ({pct:.2f}%)")

# Compare with baseline
print("\n6. Loading baseline model for comparison...")
model_baseline = GAT(
    in_channels=graph.num_node_features,
    hidden_channels=64,
    out_channels=2,
    num_heads=4,
    num_layers=2,
    dropout=0.3
).to(device)

model_baseline.load_state_dict(torch.load('artifacts/baseline_gat_best.pt', map_location=device, weights_only=False))
model_baseline.eval()

with torch.no_grad():
    out_baseline = model_baseline(graph.x, graph.edge_index)[test_mask]
    pred_baseline = out_baseline.argmax(dim=1)

print(f"\n   Baseline predictions distribution:")
for cls in torch.unique(pred_baseline):
    count = (pred_baseline == cls).sum().item()
    pct = 100 * count / pred_baseline.numel()
    print(f"     Class {cls}: {count:,} ({pct:.2f}%)")

# Direct comparison
print("\n7. Direct metric comparison on test set...")
from sklearn.metrics import f1_score, accuracy_score

baseline_f1 = f1_score(y_test.cpu().numpy(), pred_baseline.cpu().numpy(), average='macro')
quantum_f1 = f1_score(y_test.cpu().numpy(), pred_test.cpu().numpy(), average='macro')

baseline_acc = accuracy_score(y_test.cpu().numpy(), pred_baseline.cpu().numpy())
quantum_acc = accuracy_score(y_test.cpu().numpy(), pred_test.cpu().numpy())

print(f"\n   Baseline F1 (Macro): {baseline_f1:.4f}")
print(f"   Quantum F1 (Macro):  {quantum_f1:.4f}")
print(f"   Difference:          {quantum_f1 - baseline_f1:+.4f}")

print(f"\n   Baseline Accuracy: {baseline_acc:.4f}")
print(f"   Quantum Accuracy:  {quantum_acc:.4f}")
print(f"   Difference:        {quantum_acc - baseline_acc:+.4f}")

print("\n" + "=" * 80)
print("END DEBUG")
print("=" * 80)
