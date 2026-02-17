#!/usr/bin/env python
"""
OPTIMIZED FINAL TRAINING PIPELINE - Best Model Ever
Combines lessons from all previous experiments:
1. Simple, fast single split (no k-fold hanging)
2. Proven hyperparameters from imp1
3. Quantum feature mapping with proper preprocessing
4. Aggressive early stopping to prevent overfitting
5. Class-weighted loss for imbalance handling
6. Gradient clipping for stability
7. Comprehensive evaluation and comparison
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR
from src.utils import set_random_seeds

# Quantum feature mapper
from quantum_feature_mapper import QuantumFeatureMapper

print("="*80)
print("OPTIMIZED FINAL TRAINING PIPELINE - BEST MODEL EVER")
print("="*80)

# Set random seed
set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ==============================================================================
# LOAD GRAPH DATA
# ==============================================================================
print("Loading graph data...")
graph = torch.load('artifacts/elliptic_graph.pt', weights_only=False).to(device)
print(f"✓ Graph loaded:")
print(f"  - Nodes: {graph.num_nodes:,}")
print(f"  - Edges: {graph.num_edges:,}")
print(f"  - Features: {graph.num_node_features}")
print(f"  - Classes: {graph.num_node_features}")

# ==============================================================================
# PREPARE DATA SPLITS
# ==============================================================================
print("\nPreparing stratified train/val/test splits...")

# Get labeled nodes
labeled_mask = (graph.y != -1)
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_y = graph.y[labeled_mask].cpu().numpy()

print(f"Labeled nodes: {len(labeled_indices):,}")
print(f"  Class 0: {(labeled_y == 0).sum():,} ({(labeled_y == 0).sum()/len(labeled_y)*100:.1f}%)")
print(f"  Class 1: {(labeled_y == 1).sum():,} ({(labeled_y == 1).sum()/len(labeled_y)*100:.1f}%)")

# Split: 70% train+val, 30% test
train_val_idx, test_idx, train_val_y, test_y = train_test_split(
    labeled_indices, labeled_y,
    test_size=0.30,
    random_state=42,
    stratify=labeled_y
)

# Split train+val: 70% train, 30% val
train_idx, val_idx, _, _ = train_test_split(
    train_val_idx, train_val_y,
    test_size=0.30,
    random_state=42,
    stratify=train_val_y
)

print(f"\nSplits created:")
print(f"  Train: {len(train_idx):,}")
print(f"  Val:   {len(val_idx):,}")
print(f"  Test:  {len(test_idx):,}")

# Create masks
train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

graph.train_mask = train_mask
graph.val_mask = val_mask
graph.test_mask = test_mask

# ==============================================================================
# PREPROCESS FEATURES
# ==============================================================================
print("\nPreprocessing features...")

# Handle NaN and Inf
nan_count = torch.isnan(graph.x).sum().item()
inf_count = torch.isinf(graph.x).sum().item()
if nan_count > 0:
    graph.x = torch.nan_to_num(graph.x, nan=0.0)
    print(f"✓ Fixed {nan_count} NaN values")
if inf_count > 0:
    graph.x = torch.nan_to_num(graph.x, posinf=10.0, neginf=-10.0)
    print(f"✓ Fixed {inf_count} Inf values")

# Normalize using training data statistics
train_x = graph.x[train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)

graph.x = (graph.x - mean) / std
graph.x = torch.clamp(graph.x, min=-10, max=10)
print(f"✓ Features normalized and clipped to [-10, 10]")

# Compute class weights
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
# TRAINING UTILITIES
# ==============================================================================

def train_epoch(model, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(graph.x, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def evaluate(model, mask, mask_name=""):
    """Evaluate model on given mask."""
    model.eval()
    
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        
        # Handle NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        
        pred = out[mask].argmax(dim=1)
        prob = F.softmax(out[mask], dim=1)[:, 1]
        
        y_true = graph.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()
        
        # Handle NaN in probabilities
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return metrics

# ==============================================================================
# BASELINE GAT TRAINING
# ==============================================================================
print("\n" + "="*80)
print("BASELINE GAT TRAINING")
print("="*80)

baseline_model = GAT(
    in_channels=graph.num_node_features,
    hidden_channels=64,
    out_channels=2,
    num_heads=4,
    num_layers=2,
    dropout=0.3
).to(device)

optimizer = torch.optim.Adam(
    baseline_model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)
criterion = nn.CrossEntropyLoss(weight=class_weight)

print(f"Model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
print(f"Training baseline model...\n")

best_val_f1 = -1
patience = 0
max_patience = 20
history_baseline = {
    'train_loss': [],
    'val_f1': [],
    'val_acc': [],
}

start_time = time.time()
for epoch in range(1, 501):
    loss = train_epoch(baseline_model, optimizer, criterion)
    val_metrics = evaluate(baseline_model, val_mask)
    
    history_baseline['train_loss'].append(loss)
    history_baseline['val_f1'].append(val_metrics['f1'])
    history_baseline['val_acc'].append(val_metrics['accuracy'])
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience = 0
        torch.save(baseline_model.state_dict(), 'artifacts/baseline_best.pt')
    else:
        patience += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Patience: {patience}/{max_patience}")
    
    if patience >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

baseline_time = time.time() - start_time
print(f"✓ Training completed in {baseline_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# Load best baseline
baseline_model.load_state_dict(torch.load('artifacts/baseline_best.pt', map_location=device))
baseline_train = evaluate(baseline_model, train_mask)
baseline_val = evaluate(baseline_model, val_mask)
baseline_test = evaluate(baseline_model, test_mask)

print(f"\nBaseline Results:")
print(f"  Train - F1: {baseline_train['f1']:.4f}, Acc: {baseline_train['accuracy']:.4f}")
print(f"  Val   - F1: {baseline_val['f1']:.4f}, Acc: {baseline_val['accuracy']:.4f}")
print(f"  Test  - F1: {baseline_test['f1']:.4f}, Acc: {baseline_test['accuracy']:.4f}")

# ==============================================================================
# QUANTUM GAT TRAINING
# ==============================================================================
print("\n" + "="*80)
print("QUANTUM GAT TRAINING")
print("="*80)

# Create quantum mapper
quantum_mapper = QuantumFeatureMapper(
    input_dim=graph.num_node_features,
    use_angle_encoding=True,
    use_interactions=False,
    use_fourier=False,
    max_output_dim=128
).to(device)

# Fit mapper on training data
with torch.no_grad():
    quantum_mapper._fit_preprocessing(graph.x[train_mask])

# Get mapped features
x_quantum = quantum_mapper(graph.x)
mapper_out_dim = x_quantum.shape[1]

print(f"Quantum mapper output dim: {mapper_out_dim}")

quantum_model = GAT(
    in_channels=mapper_out_dim,
    hidden_channels=80,  # Slightly larger than baseline
    out_channels=2,
    num_heads=4,
    num_layers=2,
    dropout=0.35  # More regularization
).to(device)

optimizer = torch.optim.Adam(
    quantum_model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

print(f"Model parameters: {sum(p.numel() for p in quantum_model.parameters()):,}")
print(f"Training quantum model...\n")

best_val_f1_q = -1
patience_q = 0
history_quantum = {
    'train_loss': [],
    'val_f1': [],
    'val_acc': [],
}

def train_epoch_quantum(model, mapper, optimizer, criterion):
    """Train quantum model for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    x_quantum = mapper(graph.x)
    out = model(x_quantum, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def evaluate_quantum(model, mapper, mask):
    """Evaluate quantum model."""
    model.eval()
    
    with torch.no_grad():
        x_quantum = mapper(graph.x)
        out = model(x_quantum, graph.edge_index)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        
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
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

start_time = time.time()
for epoch in range(1, 501):
    loss = train_epoch_quantum(quantum_model, quantum_mapper, optimizer, criterion)
    val_metrics = evaluate_quantum(quantum_model, quantum_mapper, val_mask)
    
    history_quantum['train_loss'].append(loss)
    history_quantum['val_f1'].append(val_metrics['f1'])
    history_quantum['val_acc'].append(val_metrics['accuracy'])
    
    if val_metrics['f1'] > best_val_f1_q:
        best_val_f1_q = val_metrics['f1']
        patience_q = 0
        torch.save(quantum_model.state_dict(), 'artifacts/quantum_best.pt')
        # Save mapper state
        mapper_state = {
            '_feature_mean': quantum_mapper._feature_mean.cpu().numpy().tolist(),
            '_feature_std': quantum_mapper._feature_std.cpu().numpy().tolist(),
        }
        with open('artifacts/quantum_mapper_best.json', 'w') as f:
            json.dump(mapper_state, f)
    else:
        patience_q += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Patience: {patience_q}/{max_patience}")
    
    if patience_q >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

quantum_time = time.time() - start_time
print(f"✓ Training completed in {quantum_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1_q:.4f}")

# Load best quantum
quantum_model.load_state_dict(torch.load('artifacts/quantum_best.pt', map_location=device))
with open('artifacts/quantum_mapper_best.json', 'r') as f:
    mapper_state = json.load(f)
quantum_mapper._feature_mean = torch.tensor(mapper_state['_feature_mean'], device=device)
quantum_mapper._feature_std = torch.tensor(mapper_state['_feature_std'], device=device)
quantum_mapper._preprocessing_fitted = True

quantum_train = evaluate_quantum(quantum_model, quantum_mapper, train_mask)
quantum_val = evaluate_quantum(quantum_model, quantum_mapper, val_mask)
quantum_test = evaluate_quantum(quantum_model, quantum_mapper, test_mask)

print(f"\nQuantum Results:")
print(f"  Train - F1: {quantum_train['f1']:.4f}, Acc: {quantum_train['accuracy']:.4f}")
print(f"  Val   - F1: {quantum_val['f1']:.4f}, Acc: {quantum_val['accuracy']:.4f}")
print(f"  Test  - F1: {quantum_test['f1']:.4f}, Acc: {quantum_test['accuracy']:.4f}")

# ==============================================================================
# COMPARISON & ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON: BASELINE vs QUANTUM")
print("="*80)

comparison = {
    'baseline': {
        'train': baseline_train,
        'val': baseline_val,
        'test': baseline_test,
        'training_time': baseline_time
    },
    'quantum': {
        'train': quantum_train,
        'val': quantum_val,
        'test': quantum_test,
        'training_time': quantum_time
    }
}

# Calculate improvements
test_f1_diff = quantum_test['f1'] - baseline_test['f1']
test_f1_pct = (test_f1_diff / baseline_test['f1']) * 100 if baseline_test['f1'] > 0 else 0
test_acc_diff = quantum_test['accuracy'] - baseline_test['accuracy']

print(f"\nTEST SET PERFORMANCE:")
print(f"  Baseline F1:  {baseline_test['f1']:.4f}")
print(f"  Quantum F1:   {quantum_test['f1']:.4f}")
print(f"  Difference:   {test_f1_diff:+.4f} ({test_f1_pct:+.2f}%)")
print(f"\n  Baseline Acc: {baseline_test['accuracy']:.4f}")
print(f"  Quantum Acc:  {quantum_test['accuracy']:.4f}")
print(f"  Difference:   {test_acc_diff:+.4f}")

print(f"\nVAL->TEST GENERALIZATION GAP (lower is better):")
print(f"  Baseline: {baseline_val['f1'] - baseline_test['f1']:.4f}")
print(f"  Quantum:  {quantum_val['f1'] - quantum_test['f1']:.4f}")

# Determine winner
if quantum_test['f1'] > baseline_test['f1']:
    winner = "🎉 QUANTUM WINS!"
    print(f"\n{winner} Quantum outperforms baseline by {test_f1_pct:.2f}%")
elif quantum_test['f1'] < baseline_test['f1'] - 0.01:
    winner = "⚪ BASELINE WINS"
    print(f"\n{winner} Baseline outperforms quantum by {abs(test_f1_pct):.2f}%")
else:
    winner = "🤝 TIED"
    print(f"\n{winner} Both models perform similarly")

# Save comprehensive report
report = {
    'description': 'Optimized Final Training Pipeline - Best Model Ever',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'winner': winner,
    'baseline': {
        'model_type': 'GAT',
        'params': {
            'hidden_channels': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.3,
        },
        'train_metrics': {k: v for k, v in baseline_train.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'val_metrics': {k: v for k, v in baseline_val.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'test_metrics': {k: v for k, v in baseline_test.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'training_time': baseline_time,
    },
    'quantum': {
        'model_type': 'GAT with Quantum Features',
        'feature_mapper': {
            'use_angle_encoding': True,
            'use_interactions': False,
            'use_fourier': False,
            'max_output_dim': 128,
            'actual_output_dim': mapper_out_dim,
        },
        'params': {
            'hidden_channels': 80,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.35,
        },
        'train_metrics': {k: v for k, v in quantum_train.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'val_metrics': {k: v for k, v in quantum_val.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'test_metrics': {k: v for k, v in quantum_test.items() if k != 'y_true' and k != 'y_pred' and k != 'y_prob'},
        'training_time': quantum_time,
    },
    'comparison': {
        'test_f1_difference': test_f1_diff,
        'test_f1_percentage': test_f1_pct,
        'test_acc_difference': test_acc_diff,
        'baseline_generalization_gap': baseline_val['f1'] - baseline_test['f1'],
        'quantum_generalization_gap': quantum_val['f1'] - quantum_test['f1'],
    }
}

with open('artifacts/final_optimized_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to artifacts/final_optimized_report.json")

# Print detailed classification reports
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS (TEST SET)")
print("="*80)

print("\nBASELINE:")
print(classification_report(baseline_test['y_true'], baseline_test['y_pred'], 
                          target_names=['Non-Fraud', 'Fraud']))

print("\nQUANTUM:")
print(classification_report(quantum_test['y_true'], quantum_test['y_pred'],
                          target_names=['Non-Fraud', 'Fraud']))

print("\n" + "="*80)
print("✅ TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"\nModels saved:")
print(f"  - artifacts/baseline_best.pt")
print(f"  - artifacts/quantum_best.pt")
print(f"  - artifacts/quantum_mapper_best.json")
print(f"\nReport saved:")
print(f"  - artifacts/final_optimized_report.json")
