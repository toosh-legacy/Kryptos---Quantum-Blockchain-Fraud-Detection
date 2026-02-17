#!/usr/bin/env python
"""
Reproducible Multi-Class Fraud Detection Pipeline
Using Full Dataset with K-Fold Cross-Validation

This script:
1. Loads full graph and all labeled nodes
2. Implements 5-fold cross-validation on labeled nodes
3. Trains baseline and quantum-enhanced GAT models
4. Saves weights after each fold
5. Evaluates on held-out test split for generalization verification
6. Reports comprehensive metrics with proper statistical measures
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR
from src.utils import set_random_seeds
from quantum_feature_mapper import QuantumFeatureMapper

# Set reproducibility
set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("REPRODUCIBLE MULTI-FOLD FRAUD DETECTION PIPELINE")
print("=" * 80)
print(f"Device: {device}\n")


# ===========================================================================
# LOAD DATA
# ===========================================================================
print("Loading full graph...")
graph = torch.load('artifacts/elliptic_graph.pt', map_location=device, weights_only=False).to(device)
print(f"✓ Graph loaded:")
print(f"  - Total nodes: {graph.num_nodes:,}")
print(f"  - Total edges: {graph.num_edges:,}")
print(f"  - Features: {graph.num_node_features}")
print(f"  - Classes: {int(graph.y.max().item()) + 1}")

# Get all labeled node indices
labeled_mask = graph.y != -1  # Assuming -1 means unlabeled
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_labels = graph.y[labeled_mask].cpu().numpy()

print(f"\n✓ Labeled data:")
print(f"  - Total labeled nodes: {len(labeled_indices):,}")
for cls in np.unique(labeled_labels):
    count = (labeled_labels == cls).sum()
    pct = 100 * count / len(labeled_labels)
    print(f"  - Class {int(cls)}: {count:,} ({pct:.1f}%)")

# Reserve 15% as final held-out test set (stratified)
test_ratio = 0.15
skf_test = StratifiedKFold(n_splits=int(1 / test_ratio), shuffle=True, random_state=42)
train_val_idx, test_idx = next(skf_test.split(labeled_indices, labeled_labels))

train_val_indices = labeled_indices[train_val_idx]
train_val_labels = labeled_labels[train_val_idx]
test_indices = labeled_indices[test_idx]
test_labels = labeled_labels[test_idx]

print(f"\n✓ Train+Val / Test split (85/15):")
print(f"  - Train+Val nodes: {len(train_val_indices):,}")
print(f"  - Test nodes: {len(test_indices):,}")

# Setup 5-fold CV on train+val set
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\n✓ 5-Fold cross-validation setup on {len(train_val_indices):,} nodes")

# ===========================================================================
# COMPUTE CLASS WEIGHTS (on FULL training data)
# ===========================================================================
print("\nComputing class weights...")
unique_classes, class_counts = np.unique(train_val_labels, return_counts=True)
num_classes = int(graph.y.max().item()) + 1

class_weights = torch.zeros(num_classes, device=device)
for cls, count in zip(unique_classes, class_counts):
    class_weights[int(cls)] = 1.0 / count if count > 0 else 0

class_weights = class_weights / class_weights.sum()
print(f"Class weights: {class_weights.cpu().numpy()}")

# ===========================================================================
# MODELS & TRAINING
# ===========================================================================

def create_baseline_model(in_channels):
    """Create baseline 2-layer GAT."""
    return GAT(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
    ).to(device)


def create_quantum_model(in_channels):
    """Create quantum-enhanced GAT (smaller to prevent overfitting)."""
    return GAT(
        in_channels=in_channels,
        hidden_channels=80,  # Slightly larger than baseline (64)
        out_channels=2,
        num_heads=4,  # Same as baseline (NOT 6)
        num_layers=2,  # Same as baseline (NOT 3)
        dropout=0.35,  # More regularization than baseline (0.3)
    ).to(device)


def train_epoch(model, data, train_indices, criterion, optimizer, use_quantum):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    
    # Apply quantum mapping if needed
    if use_quantum:
        quantum_mapper = model._quantum_mapper
        x = quantum_mapper(data.x)
    else:
        x = data.x
    
    out = model(x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, eval_indices, criterion, use_quantum):
    """Evaluate on given indices."""
    model.eval()
    
    eval_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    eval_mask[eval_indices] = True
    
    with torch.no_grad():
        if use_quantum:
            quantum_mapper = model._quantum_mapper
            x = quantum_mapper(data.x)
        else:
            x = data.x
        
        out = model(x, data.edge_index)
        loss = criterion(out[eval_mask], data.y[eval_mask])
        
        pred = out[eval_mask].argmax(dim=1).cpu().numpy()
        true = data.y[eval_mask].cpu().numpy()
        
        f1_macro = f1_score(true, pred, average='macro', zero_division=0)
        acc = accuracy_score(true, pred)
    
    return loss.item(), f1_macro, acc, pred, true


print("\n" + "=" * 80)
print("TRAINING BASELINE & QUANTUM MODELS (5-FOLD CV)")
print("=" * 80)

# Store results
baseline_results = {'fold_scores': [], 'models': [], 'histories': []}
quantum_results = {'fold_scores': [], 'models': [], 'histories': []}

fold_num = 0
for train_idx, val_idx in skf.split(train_val_indices, train_val_labels):
    fold_num += 1
    
    # Map to global indices
    fold_train_indices = train_val_indices[train_idx]
    fold_val_indices = train_val_indices[val_idx]
    
    print(f"\n{'=' * 80}")
    print(f"FOLD {fold_num}/{n_folds}")
    print(f"{'=' * 80}")
    print(f"Train: {len(fold_train_indices):,} | Val: {len(fold_val_indices):,}")
    
    # ===========================================================================
    # BASELINE MODEL
    # ===========================================================================
    print(f"\n📊 Training BASELINE GAT (Fold {fold_num})...")
    
    baseline_model = create_baseline_model(graph.num_node_features)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15
    )
    
    best_val_f1 = -1
    patience_counter = 0
    patience = 20
    epochs = 300
    history_baseline = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(baseline_model, graph, fold_train_indices, criterion, optimizer, use_quantum=False)
        val_loss, val_f1, val_acc, _, _ = evaluate(baseline_model, graph, fold_val_indices, criterion, use_quantum=False)
        
        history_baseline['train_loss'].append(train_loss)
        history_baseline['val_loss'].append(val_loss)
        history_baseline['val_f1'].append(val_f1)
        history_baseline['val_acc'].append(val_acc)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(baseline_model.state_dict(), 
                      f'artifacts/baseline_fold_{fold_num}_best.pt')
        else:
            patience_counter += 1
        
        if (epoch % 50 == 0) or (epoch < 10):
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Evaluate baseline on fold
    best_baseline = create_baseline_model(graph.num_node_features)
    best_baseline.load_state_dict(torch.load(f'artifacts/baseline_fold_{fold_num}_best.pt', map_location=device))
    _, fold_val_f1, _, _, _ = evaluate(best_baseline, graph, fold_val_indices, criterion, use_quantum=False)
    baseline_results['fold_scores'].append({'fold': fold_num, 'val_f1': fold_val_f1})
    baseline_results['models'].append(f'baseline_fold_{fold_num}_best.pt')
    baseline_results['histories'].append(history_baseline)
    
    print(f"✓ Baseline Fold {fold_num} Val F1: {fold_val_f1:.4f}")
    
    # ===========================================================================
    # QUANTUM MODEL
    # ===========================================================================
    print(f"\n🚀 Training QUANTUM GAT (Fold {fold_num})...")
    
    # Create quantum mapper and calculate its output dimension
    quantum_mapper = QuantumFeatureMapper(
        input_dim=graph.num_node_features,
        use_angle_encoding=True,
        use_interactions=False,
        use_fourier=False,
        max_output_dim=128  # Keep smaller to prevent overfitting!
    ).to(device)
    
    # The mapper outputs: min(input_dim * 2, max_output_dim) = min(364, 128) = 128
    mapper_output_dim = min(graph.num_node_features * 2, 128)
    
    # Create quantum model with correct input dimension
    quantum_model = GAT(
        in_channels=mapper_output_dim,
        hidden_channels=80,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.35,
    ).to(device)
    quantum_model._quantum_mapper = quantum_mapper
    
    optimizer = torch.optim.Adam(quantum_model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15
    )
    
    best_val_f1 = -1
    patience_counter = 0
    history_quantum = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(quantum_model, graph, fold_train_indices, criterion, optimizer, use_quantum=True)
        val_loss, val_f1, val_acc, _, _ = evaluate(quantum_model, graph, fold_val_indices, criterion, use_quantum=True)
        
        history_quantum['train_loss'].append(train_loss)
        history_quantum['val_loss'].append(val_loss)
        history_quantum['val_f1'].append(val_f1)
        history_quantum['val_acc'].append(val_acc)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(quantum_model.state_dict(), 
                      f'artifacts/quantum_fold_{fold_num}_best.pt')
            # Also save mapper state
            mapper_state = {
                '_feature_mean': quantum_mapper._feature_mean.cpu().numpy().tolist() if quantum_mapper._feature_mean is not None else None,
                '_feature_std': quantum_mapper._feature_std.cpu().numpy().tolist() if quantum_mapper._feature_std is not None else None,
            }
            with open(f'artifacts/quantum_mapper_fold_{fold_num}.json', 'w') as f:
                json.dump(mapper_state, f)
        else:
            patience_counter += 1
        
        if (epoch % 50 == 0) or (epoch < 10):
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Evaluate quantum on fold
    best_quantum = GAT(
        in_channels=128,  # Mapper outputs 128 features
        hidden_channels=80,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.35,
    ).to(device)
    best_quantum.load_state_dict(torch.load(f'artifacts/quantum_fold_{fold_num}_best.pt', map_location=device))
    best_quantum_mapper = QuantumFeatureMapper(
        input_dim=graph.num_node_features,
        use_angle_encoding=True,
        use_interactions=False,
        use_fourier=False,
        max_output_dim=128
    ).to(device)
    # Restore mapper state
    with open(f'artifacts/quantum_mapper_fold_{fold_num}.json', 'r') as f:
        mapper_state = json.load(f)
    if mapper_state['_feature_mean'] is not None:
        best_quantum_mapper._feature_mean = torch.tensor(mapper_state['_feature_mean'], device=device)
        best_quantum_mapper._feature_std = torch.tensor(mapper_state['_feature_std'], device=device)
        best_quantum_mapper._preprocessing_fitted = True
    best_quantum._quantum_mapper = best_quantum_mapper
    
    _, fold_val_f1, _, _, _ = evaluate(best_quantum, graph, fold_val_indices, criterion, use_quantum=True)
    quantum_results['fold_scores'].append({'fold': fold_num, 'val_f1': fold_val_f1})
    quantum_results['models'].append(f'quantum_fold_{fold_num}_best.pt')
    quantum_results['histories'].append(history_quantum)
    
    print(f"✓ Quantum Fold {fold_num} Val F1: {fold_val_f1:.4f}")

# ===========================================================================
# TRAIN FINAL MODELS ON FULL TRAIN+VAL SET
# ===========================================================================
print(f"\n{'=' * 80}")
print("TRAINING FINAL MODELS ON FULL TRAIN+VAL SET")
print(f"{'=' * 80}")

print("\nTraining Final BASELINE on all train+val data...")
final_baseline = create_baseline_model(graph.num_node_features)
optimizer = torch.optim.Adam(final_baseline.parameters(), lr=0.001, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss(weight=class_weights)

best_val_f1 = -1
for epoch in range(1, 300):
    # Train on all train+val
    final_baseline.train()
    optimizer.zero_grad()
    train_val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    train_val_mask[train_val_indices] = True
    
    out = final_baseline(graph.x, graph.edge_index)
    loss = criterion(out[train_val_mask], graph.y[train_val_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(final_baseline.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: Loss {loss:.4f}")

torch.save(final_baseline.state_dict(), 'artifacts/baseline_final.pt')
print("✓ Saved: artifacts/baseline_final.pt")

print("\nTraining Final QUANTUM on all train+val data...")
final_quantum_mapper = QuantumFeatureMapper(
    input_dim=graph.num_node_features,
    use_angle_encoding=True,
    use_interactions=False,
    use_fourier=False,
    max_output_dim=128
).to(device)

final_quantum = GAT(
    in_channels=128,  # Mapper outputs 128 features
    hidden_channels=80,
    out_channels=2,
    num_heads=4,
    num_layers=2,
    dropout=0.35,
).to(device)
final_quantum._quantum_mapper = final_quantum_mapper

optimizer = torch.optim.Adam(final_quantum.parameters(), lr=0.001, weight_decay=0.0005)

for epoch in range(1, 300):
    final_quantum.train()
    optimizer.zero_grad()
    train_val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    train_val_mask[train_val_indices] = True
    
    x_mapped = final_quantum_mapper(graph.x)
    out = final_quantum(x_mapped, graph.edge_index)
    loss = criterion(out[train_val_mask], graph.y[train_val_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(final_quantum.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: Loss {loss:.4f}")

torch.save(final_quantum.state_dict(), 'artifacts/quantum_final.pt')
mapper_state = {
    '_feature_mean': final_quantum_mapper._feature_mean.cpu().numpy().tolist() if final_quantum_mapper._feature_mean is not None else None,
    '_feature_std': final_quantum_mapper._feature_std.cpu().numpy().tolist() if final_quantum_mapper._feature_std is not None else None,
}
with open('artifacts/quantum_mapper_final.json', 'w') as f:
    json.dump(mapper_state, f)
print("✓ Saved: artifacts/quantum_final.pt + quantum_mapper_final.json")

# ===========================================================================
# EVALUATE ON HELD-OUT TEST SET
# ===========================================================================
print(f"\n{'=' * 80}")
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print(f"{'=' * 80}")

criterion_test = nn.CrossEntropyLoss()

# Load final baseline
final_baseline.eval()
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
test_mask[test_indices] = True

with torch.no_grad():
    out_baseline = final_baseline(graph.x, graph.edge_index)
    baseline_pred = out_baseline[test_mask].argmax(dim=1).cpu().numpy()
    baseline_true = graph.y[test_mask].cpu().numpy()
    
    baseline_f1 = f1_score(baseline_true, baseline_pred, average='macro', zero_division=0)
    baseline_acc = accuracy_score(baseline_true, baseline_pred)

print(f"\n📊 BASELINE TEST PERFORMANCE:")
print(f"  Accuracy:     {baseline_acc:.4f}")
print(f"  F1 (Macro):   {baseline_f1:.4f}")
print(f"  Precision:    {precision_score(baseline_true, baseline_pred, average='macro', zero_division=0):.4f}")
print(f"  Recall:       {recall_score(baseline_true, baseline_pred, average='macro', zero_division=0):.4f}")

# Load final quantum
final_quantum.eval()
with torch.no_grad():
    x_mapped = final_quantum_mapper(graph.x)
    out_quantum = final_quantum(x_mapped, graph.edge_index)
    quantum_pred = out_quantum[test_mask].argmax(dim=1).cpu().numpy()
    quantum_true = graph.y[test_mask].cpu().numpy()
    
    quantum_f1 = f1_score(quantum_true, quantum_pred, average='macro', zero_division=0)
    quantum_acc = accuracy_score(quantum_true, quantum_pred)

print(f"\n🚀 QUANTUM TEST PERFORMANCE:")
print(f"  Accuracy:     {quantum_acc:.4f}")
print(f"  F1 (Macro):   {quantum_f1:.4f}")
print(f"  Precision:    {precision_score(quantum_true, quantum_pred, average='macro', zero_division=0):.4f}")
print(f"  Recall:       {recall_score(quantum_true, quantum_pred, average='macro', zero_division=0):.4f}")

# ===========================================================================
# CROSS-VALIDATION RESULTS
# ===========================================================================
print(f"\n{'=' * 80}")
print("5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'=' * 80}")

baseline_cv_scores = [r['val_f1'] for r in baseline_results['fold_scores']]
quantum_cv_scores = [r['val_f1'] for r in quantum_results['fold_scores']]

print(f"\nBASELINE CV Scores:")
for i, score in enumerate(baseline_cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"  Mean: {np.mean(baseline_cv_scores):.4f} ± {np.std(baseline_cv_scores):.4f}")

print(f"\nQUANTUM CV Scores:")
for i, score in enumerate(quantum_cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"  Mean: {np.mean(quantum_cv_scores):.4f} ± {np.std(quantum_cv_scores):.4f}")

# ===========================================================================
# SAVE FINAL REPORT
# ===========================================================================
report = {
    'timestamp': datetime.now().isoformat(),
    'cv_results': {
        'baseline_cv_mean': float(np.mean(baseline_cv_scores)),
        'baseline_cv_std': float(np.std(baseline_cv_scores)),
        'quantum_cv_mean': float(np.mean(quantum_cv_scores)),
        'quantum_cv_std': float(np.std(quantum_cv_scores)),
    },
    'test_results': {
        'baseline_f1': float(baseline_f1),
        'baseline_acc': float(baseline_acc),
        'quantum_f1': float(quantum_f1),
        'quantum_acc': float(quantum_acc),
    },
    'artifacts': {
        'baseline_final': 'baseline_final.pt',
        'quantum_final': 'quantum_final.pt',
        'quantum_mapper': 'quantum_mapper_final.json',
    }
}

with open('artifacts/full_pipeline_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Final report saved: artifacts/full_pipeline_report.json")
print(f"{'=' * 80}")
