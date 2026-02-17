#!/usr/bin/env python
"""
Training script for QIGAT and Baseline GAT
Research-grade fraud detection on blockchain graphs
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
import time
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import models
from models.qigat import QIGAT, BaselineGAT
from models.feature_engineering import compute_structural_features
from src.utils import set_random_seeds

print("="*80)
print("RESEARCH-GRADE QIGAT FOR BLOCKCHAIN FRAUD DETECTION")
print("="*80)

set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ==============================================================================
# DATA LOADING
# ==============================================================================
print("Loading data...")
graph = torch.load('artifacts/elliptic_graph.pt', weights_only=False)

labeled_mask = (graph.y != -1)
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_y = graph.y[labeled_mask].cpu().numpy()

print(f"Labeled nodes: {len(labeled_indices):,}")
print(f"  Class 0: {(labeled_y == 0).sum():,} ({(labeled_y == 0).sum()/len(labeled_y)*100:.1f}%)")
print(f"  Class 1: {(labeled_y == 1).sum():,} ({(labeled_y == 1).sum()/len(labeled_y)*100:.1f}%)")

# Stratified split
train_val_idx, test_idx, train_val_y, test_y = train_test_split(
    labeled_indices, labeled_y,
    test_size=0.30,
    random_state=42,
    stratify=labeled_y
)

train_idx, val_idx, _, _ = train_test_split(
    train_val_idx, train_val_y,
    test_size=0.30,
    random_state=42,
    stratify=train_val_y
)

train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

graph = graph.to(device)

print(f"\nTrain: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")

# Handle NaN
nan_count = torch.isnan(graph.x).sum().item()
if nan_count > 0:
    graph.x = torch.nan_to_num(graph.x, nan=0.0)

# Normalize
train_x = graph.x[train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)
graph.x = (graph.x - mean) / std
graph.x = torch.clamp(graph.x, min=-10, max=10)

print("✓ Data preprocessing complete")

# ==============================================================================
# COMPUTE STRUCTURAL FEATURES
# ==============================================================================
print("\n" + "="*80)
print("COMPUTING STRUCTURAL FEATURES")
print("="*80 + "\n")

x_enriched = compute_structural_features(graph, device=device)
original_in_channels = graph.num_node_features
enriched_in_channels = x_enriched.shape[1]
print(f"Original features: {original_in_channels}")
print(f"Enriched features: {enriched_in_channels}")

# ==============================================================================
# CLASS WEIGHTS
# ==============================================================================
n_class_0 = (graph.y[train_mask] == 0).sum().item()
n_class_1 = (graph.y[train_mask] == 1).sum().item()
class_weight = torch.tensor(
    [1.0 / n_class_0, 1.0 / n_class_1],
    device=device,
    dtype=torch.float32
)
class_weight = class_weight / class_weight.sum()
print(f"Class weights: {class_weight.cpu().numpy()}")

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================
def train_epoch(model, optimizer, criterion, scheduler=None):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(x_enriched, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    
    return loss.item()

def evaluate(model, mask):
    """Evaluate model."""
    model.eval()
    
    with torch.no_grad():
        out = model(x_enriched, graph.edge_index)
        pred = out[mask].argmax(dim=1)
        prob = F.softmax(out[mask], dim=1)[:, 1]
        
        y_true = graph.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_true': y_true
    }

# ==============================================================================
# BASELINE GAT TRAINING
# ==============================================================================
print("\n" + "="*80)
print("TRAINING BASELINE GAT")
print("="*80 + "\n")

baseline_model = BaselineGAT(
    in_channels=enriched_in_channels,
    hidden_channels=64,
    out_channels=2,
    num_heads=4,
    dropout=0.3,
    num_layers=2
).to(device)

optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weight)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print(f"Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}\n")

best_val_f1 = -1
patience = 0
max_patience = 50
baseline_history = {'train_loss': [], 'val_f1': [], 'val_acc': []}

start_time = time.time()
for epoch in range(1, 301):
    loss = train_epoch(baseline_model, optimizer, criterion, scheduler)
    val_metrics = evaluate(baseline_model, val_mask)
    
    baseline_history['train_loss'].append(loss)
    baseline_history['val_f1'].append(val_metrics['f1'])
    baseline_history['val_acc'].append(val_metrics['accuracy'])
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience = 0
        torch.save(baseline_model.state_dict(), 'artifacts/baseline_qigat_research.pt')
    else:
        patience += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | Patience: {patience}/{max_patience}")
    
    if patience >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

baseline_time = time.time() - start_time
print(f"\n✓ Training completed in {baseline_time:.2f}s")

baseline_model.load_state_dict(torch.load('artifacts/baseline_qigat_research.pt', map_location=device))
baseline_train = evaluate(baseline_model, train_mask)
baseline_val = evaluate(baseline_model, val_mask)
baseline_test = evaluate(baseline_model, test_mask)

print(f"\nBaseline Results:")
print(f"  Train - F1: {baseline_train['f1']:.4f}, Acc: {baseline_train['accuracy']:.4f}")
print(f"  Val   - F1: {baseline_val['f1']:.4f}, Acc: {baseline_val['accuracy']:.4f}")
print(f"  Test  - F1: {baseline_test['f1']:.4f}, Acc: {baseline_test['accuracy']:.4f}")

# ==============================================================================
# QIGAT TRAINING
# ==============================================================================
print("\n" + "="*80)
print("TRAINING QUANTUM-INSPIRED GAT (QIGAT)")
print("="*80 + "\n")

qigat_model = QIGAT(
    in_channels=enriched_in_channels,
    hidden_channels=128,
    out_channels=2,
    num_heads=8,
    dropout=0.5,
    num_layers=2
).to(device)

optimizer = torch.optim.AdamW(qigat_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print(f"Parameters: {sum(p.numel() for p in qigat_model.parameters()):,}\n")

best_val_f1_q = -1
patience_q = 0
qigat_history = {'train_loss': [], 'val_f1': [], 'val_acc': []}

start_time = time.time()
for epoch in range(1, 301):
    loss = train_epoch(qigat_model, optimizer, criterion, scheduler)
    val_metrics = evaluate(qigat_model, val_mask)
    
    qigat_history['train_loss'].append(loss)
    qigat_history['val_f1'].append(val_metrics['f1'])
    qigat_history['val_acc'].append(val_metrics['accuracy'])
    
    if val_metrics['f1'] > best_val_f1_q:
        best_val_f1_q = val_metrics['f1']
        patience_q = 0
        torch.save(qigat_model.state_dict(), 'artifacts/qigat_research.pt')
    else:
        patience_q += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | Patience: {patience_q}/{max_patience}")
    
    if patience_q >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

qigat_time = time.time() - start_time
print(f"\n✓ Training completed in {qigat_time:.2f}s")

qigat_model.load_state_dict(torch.load('artifacts/qigat_research.pt', map_location=device))
qigat_train = evaluate(qigat_model, train_mask)
qigat_val = evaluate(qigat_model, val_mask)
qigat_test = evaluate(qigat_model, test_mask)

print(f"\nQIGAT Results:")
print(f"  Train - F1: {qigat_train['f1']:.4f}, Acc: {qigat_train['accuracy']:.4f}")
print(f"  Val   - F1: {qigat_val['f1']:.4f}, Acc: {qigat_val['accuracy']:.4f}")
print(f"  Test  - F1: {qigat_test['f1']:.4f}, Acc: {qigat_test['accuracy']:.4f}")

# ==============================================================================
# COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON: BASELINE vs QIGAT")
print("="*80)

test_f1_diff = qigat_test['f1'] - baseline_test['f1']
test_f1_pct = (test_f1_diff / baseline_test['f1']) * 100 if baseline_test['f1'] > 0 else 0

print(f"\nTEST SET:")
print(f"  Baseline F1: {baseline_test['f1']:.4f}")
print(f"  QIGAT F1:    {qigat_test['f1']:.4f}")
print(f"  Difference:  {test_f1_diff:+.4f} ({test_f1_pct:+.2f}%)")

if qigat_test['f1'] > baseline_test['f1']:
    print(f"\n🏆 QIGAT WINS! Improvement: {test_f1_pct:.2f}%")
else:
    print(f"\n⚪ Baseline leads. Gap: {abs(test_f1_pct):.2f}%")

# ==============================================================================
# DETAILED CLASSIFICATION REPORTS
# ==============================================================================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS (TEST SET)")
print("="*80)

print("\nBASELINE GAT:")
print(classification_report(baseline_test['y_true'], baseline_test['y_pred'],
                          target_names=['Non-Fraud', 'Fraud']))

print("\nQIGAT:")
print(classification_report(qigat_test['y_true'], qigat_test['y_pred'],
                          target_names=['Non-Fraud', 'Fraud']))

# ==============================================================================
# SAVE REPORT
# ==============================================================================
report = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'approach': 'Research-grade QIGAT with enriched features',
    'baseline': {
        'model': 'Standard GAT',
        'params': sum(p.numel() for p in BaselineGAT(enriched_in_channels, 64, 2, 4, 0.3, 2).parameters()),
        'test_metrics': {
            'f1': baseline_test['f1'],
            'accuracy': baseline_test['accuracy'],
            'precision': baseline_test['precision'],
            'recall': baseline_test['recall'],
            'roc_auc': baseline_test['roc_auc']
        }
    },
    'qigat': {
        'model': 'QIGAT',
        'params': sum(p.numel() for p in qigat_model.parameters()),
        'test_metrics': {
            'f1': qigat_test['f1'],
            'accuracy': qigat_test['accuracy'],
            'precision': qigat_test['precision'],
            'recall': qigat_test['recall'],
            'roc_auc': qigat_test['roc_auc']
        }
    },
    'comparison': {
        'qigat_wins': qigat_test['f1'] > baseline_test['f1'],
        'f1_diff': test_f1_diff,
        'f1_pct_diff': test_f1_pct
    }
}

with open('artifacts/qigat_research_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to artifacts/qigat_research_report.json")

print("\n" + "="*80)
print("✅ RESEARCH-GRADE QIGAT TRAINING COMPLETE!")
print("="*80)
