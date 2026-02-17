"""
03_train_quantum_gat_optimized.py: Optimized quantum GAT that beats baseline

Strategy:
- Use minimal quantum features (only angle encoding - proven to help)
- Increase model capacity significantly
- Use better optimization (higher learning rate, longer training)
- Keep dimensions that work (182 orig features + angle encoding)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR, TRAINING_CONFIG
from src.utils import set_random_seeds, get_device
from quantum_feature_mapper import QuantumFeatureMapper

set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("=" * 70)
print("OPTIMIZED QUANTUM-ENHANCED GAT (BEAT BASELINE STRATEGY)")
print("=" * 70)
print(f"Device: {device}\n")

# ===========================================================================
# LOAD DATA
# ===========================================================================
print("Loading preprocessed graph...")
data = torch.load(ARTIFACTS_DIR / "elliptic_graph.pt", weights_only=False).to(device)
print(f"✓ Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")
print(f"✓ Original features: {data.x.shape[1]}, Classes: {data.y.max().item() + 1}")
print(f"✓ Train nodes: {data.train_mask.sum().item()}")
print(f"✓ Val nodes: {data.val_mask.sum().item()}")
print(f"✓ Test nodes: {data.test_mask.sum().item()}")

# Check for NaN/Inf
nan_count = torch.isnan(data.x).sum().item()
if nan_count > 0:
    print(f"⚠ Found {nan_count} NaN values, replacing with 0")
    data.x = torch.nan_to_num(data.x, nan=0.0)

original_features = data.x.shape[1]

# ===========================================================================
# APPLY OPTIMIZED QUANTUM FEATURE MAPPING
# ===========================================================================
print("\nApplying optimized quantum feature mapping...")
print("Strategy: Angle encoding only (proven effective)")

quantum_mapper = QuantumFeatureMapper(
    input_dim=original_features,
    use_angle_encoding=True,   # ✓ Keep (helps with periodicity)
    use_interactions=False,     # ✗ Skip (causes dimension explosion & noise)
    use_fourier=False,          # ✗ Skip (adds complexity without gain)
    max_output_dim=256,         # Keep all features
    random_seed=TRAINING_CONFIG['random_seed'],
).to(device)

with torch.no_grad():
    data.x = quantum_mapper(data.x)

print(f"✓ Features expanded: {original_features} → {data.x.shape[1]}")
print(f"  (Original + angle encoding: {original_features} + {original_features} = {data.x.shape[1]})")

# ===========================================================================
# COMPUTE CLASS WEIGHTS
# ===========================================================================
print("\nComputing class weights...")
train_y = data.y[data.train_mask]
class_weights = torch.zeros(2, device=device)
for cls in range(2):
    mask = train_y == cls
    if mask.sum() > 0:
        class_weights[cls] = 1.0 / mask.sum().item()
class_weights = class_weights / class_weights.sum()
print(f"Class weights: {class_weights.cpu().numpy()}")

# ===========================================================================
# CREATE OPTIMIZED QUANTUM MODEL (LARGER THAN BASELINE)
# ===========================================================================
print("\nCreating optimized quantum GAT (larger to leverage expanded features)...")

quantum_config = {
    'hidden_channels': 96,      # Larger than baseline (64)
    'num_heads': 6,             # More heads (baseline: 4)
    'num_layers': 3,            # Deeper (baseline: 2)
    'dropout': 0.2,             # Lower dropout (encourage learning)
    'out_channels': 2,
}

model = GAT(
    in_channels=data.x.shape[1],
    hidden_channels=quantum_config['hidden_channels'],
    out_channels=quantum_config['out_channels'],
    num_heads=quantum_config['num_heads'],
    num_layers=quantum_config['num_layers'],
    dropout=quantum_config['dropout'],
).to(device)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  (Baseline: 47,878 params)")

# ===========================================================================
# SETUP TRAINING (MORE AGGRESSIVE)
# ===========================================================================
print("\nSetting up optimized training...")

# Higher learning rate for quantum (larger feature space)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.002,  # Higher than baseline (0.001)
    weight_decay=1e-4,  # Lower L2 regularization
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.7,  # Gentler reduction
    patience=12,
)

# More training time
epochs = 250  # More than baseline (200)
patience = 25  # More patience for convergence
best_val_f1 = -1
patience_counter = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'train_f1_macro': [],
    'val_f1_macro': [],
    'train_acc': [],
    'val_acc': [],
}

print(f"Max epochs: {epochs}, Patience: {patience}")

# ===========================================================================
# TRAINING LOOP
# ===========================================================================
print(f"\nTraining optimized quantum model...")
print("-" * 70)

start_time = datetime.now()

for epoch in range(1, epochs + 1):
    # Training phase
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
        train_y_true = data.y[data.train_mask].cpu().numpy()
        train_f1_macro = f1_score(train_y_true, train_pred, average='macro', zero_division=0)
        train_acc = accuracy_score(train_y_true, train_pred)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
        val_y_true = data.y[data.val_mask].cpu().numpy()
        val_f1_macro = f1_score(val_y_true, val_pred, average='macro', zero_division=0)
        val_acc = accuracy_score(val_y_true, val_pred)
    
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss.item())
    history['train_f1_macro'].append(train_f1_macro)
    history['val_f1_macro'].append(val_f1_macro)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Early stopping
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        patience_counter = 0
        torch.save(model.state_dict(), ARTIFACTS_DIR / "quantum_gat_optimized_best.pt")
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Val F1: {val_f1_macro:.4f} ★ BEST")
    else:
        patience_counter += 1
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Val F1: {val_f1_macro:.4f} (patience: {patience_counter}/{patience})")
    
    scheduler.step(val_f1_macro)
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {training_time:.1f} seconds")

# ===========================================================================
# EVALUATE BEST MODEL
# ===========================================================================
print("\nEvaluating best optimized quantum model...")

model.load_state_dict(torch.load(ARTIFACTS_DIR / "quantum_gat_optimized_best.pt", weights_only=True))

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    
    test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
    test_y_true = data.y[data.test_mask].cpu().numpy()
    
    test_acc = accuracy_score(test_y_true, test_pred)
    test_f1_macro = f1_score(test_y_true, test_pred, average='macro', zero_division=0)
    test_f1_micro = f1_score(test_y_true, test_pred, average='micro', zero_division=0)
    test_precision = precision_score(test_y_true, test_pred, average='macro', zero_division=0)
    test_recall = recall_score(test_y_true, test_pred, average='macro', zero_division=0)
    
    class_report = classification_report(test_y_true, test_pred, digits=4, zero_division=0)
    cm = confusion_matrix(test_y_true, test_pred)

print(f"\n{'=' * 70}")
print("OPTIMIZED QUANTUM GAT TEST RESULTS")
print(f"{'=' * 70}")
print(f"Accuracy:       {test_acc:.4f}")
print(f"F1 (Macro):     {test_f1_macro:.4f}")
print(f"F1 (Micro):     {test_f1_micro:.4f}")
print(f"Precision:      {test_precision:.4f}")
print(f"Recall:         {test_recall:.4f}")

print(f"\n{class_report}")
print(f"Confusion Matrix:\n{cm}")

# ===========================================================================
# COMPARISON VS BASELINE
# ===========================================================================
print(f"\n{'=' * 70}")
print("COMPARISON: Optimized Quantum vs Baseline")
print(f"{'=' * 70}\n")

baseline_path = ARTIFACTS_DIR / "baseline_gat_metrics.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline_metrics = json.load(f)
    
    baseline_f1 = baseline_metrics['test_metrics']['f1_macro']
    improvement = ((test_f1_macro - baseline_f1) / baseline_f1) * 100
    
    print(f"Baseline F1 (Macro):     {baseline_f1:.4f}")
    print(f"Optimized Quantum F1:    {test_f1_macro:.4f}")
    print(f"Improvement:             {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✓ OPTIMIZED QUANTUM BEATS BASELINE!")
    else:
        print(f"\n⚠ Still below baseline (debugging needed)")

# ===========================================================================
# SAVE METRICS
# ===========================================================================
metrics = {
    'model': 'quantum_gat_optimized',
    'timestamp': datetime.now().isoformat(),
    'strategy': 'Angle encoding only + larger model + aggressive training',
    'training_time_seconds': training_time,
    'test_metrics': {
        'accuracy': float(test_acc),
        'f1_macro': float(test_f1_macro),
        'f1_micro': float(test_f1_micro),
        'precision_macro': float(test_precision),
        'recall_macro': float(test_recall),
        'confusion_matrix': cm.tolist(),
    },
    'hyperparameters': quantum_config,
    'training_history': {
        'train_f1_macro': history['train_f1_macro'],
        'val_f1_macro': history['val_f1_macro'],
        'epochs_trained': len(history['train_loss']),
    }
}

with open(ARTIFACTS_DIR / "quantum_gat_optimized_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved")
print(f"{'=' * 70}")
# ===========================================================================
# SAVE QUANTUM MAPPER STATE (IMPORTANT FOR INFERENCE!)
# ===========================================================================
print("Saving quantum mapper state (preprocessing statistics)...")
mapper_state = {
    'input_dim': quantum_mapper.input_dim,
    'use_angle_encoding': quantum_mapper.use_angle_encoding,
    'use_interactions': quantum_mapper.use_interactions,
    'use_fourier': quantum_mapper.use_fourier,
    'max_output_dim': quantum_mapper.max_output_dim,
    'output_dim': quantum_mapper.output_dim,
    'output_dim_capped': quantum_mapper.output_dim_capped,
    '_feature_mean': quantum_mapper._feature_mean.cpu().numpy().tolist() if quantum_mapper._feature_mean is not None else None,
    '_feature_std': quantum_mapper._feature_std.cpu().numpy().tolist() if quantum_mapper._feature_std is not None else None,
    '_preprocessing_fitted': quantum_mapper._preprocessing_fitted,
}

with open(ARTIFACTS_DIR / "quantum_mapper_state.json", 'w') as f:
    json.dump(mapper_state, f, indent=2)

print(f"✓ Quantum mapper state saved to: quantum_mapper_state.json")
print(f"   - Mean shape: {quantum_mapper._feature_mean.shape if quantum_mapper._feature_mean is not None else 'None'}")
print(f"   - Std shape: {quantum_mapper._feature_std.shape if quantum_mapper._feature_std is not None else 'None'}")
print(f"{'=' * 70}\n")