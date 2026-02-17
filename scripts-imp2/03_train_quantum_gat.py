"""
03_train_quantum_gat.py: Train quantum-enhanced GAT with feature expansion.

This script:
- Loads preprocessed graph data
- Applies quantum feature mapping (angle encoding + interactions + Fourier features)
- Creates an enhanced GAT with more capacity (128 hidden, 6 heads, 3 layers)
- Trains with same setup as baseline for fair comparison
- Tracks identical metrics for comparison
- Saves best model and comprehensive metrics JSON
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR, TRAINING_CONFIG
from src.utils import set_random_seeds, get_device
from quantum_feature_mapper import QuantumFeatureMapper

# Set reproducibility
set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("=" * 70)
print("QUANTUM-ENHANCED GAT MODEL TRAINING")
print("=" * 70)
print(f"Device: {device}\n")

# ===========================================================================
# LOAD DATA
# ===========================================================================
print("Loading preprocessed graph...")
data = torch.load(ARTIFACTS_DIR / "elliptic_graph_quantum.pt", weights_only=False).to(device)
print(f"✓ Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")
print(f"✓ Original features: {data.x.shape[1]}, Classes: {data.y.max().item() + 1}")

# Verify splits exist
if not hasattr(data, 'train_mask'):
    print("✗ Graph does not have train/val/test masks!")
    print("  Run: python 01_load_data.py")
    sys.exit(1)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

print(f"✓ Train nodes: {train_mask.sum().item()}")
print(f"✓ Val nodes: {val_mask.sum().item()}")
print(f"✓ Test nodes: {test_mask.sum().item()}")

# Check for NaN/Inf in features before quantum mapping
print("\nChecking for NaN/Inf in features...")
nan_count = torch.isnan(data.x).sum().item()
inf_count = torch.isinf(data.x).sum().item()
if nan_count > 0:
    print(f"⚠ Warning: Found {nan_count} NaN values in features")
    data.x = torch.nan_to_num(data.x, nan=0.0)
    print("✓ Replaced NaN with 0")
if inf_count > 0:
    print(f"⚠ Warning: Found {inf_count} Inf values in features")
    data.x = torch.nan_to_num(data.x, posinf=0.0, neginf=0.0)
    print("✓ Replaced Inf with 0")

# ===========================================================================
# APPLY QUANTUM FEATURE MAPPING
# ===========================================================================
print("\nApplying quantum feature mapping...")

original_features = data.x.shape[1]

quantum_mapper = QuantumFeatureMapper(
    input_dim=original_features,
    use_angle_encoding=True,
    use_interactions=True,
    use_fourier=True,
    max_output_dim=128,  # As per requirements
    random_seed=TRAINING_CONFIG['random_seed'],
).to(device)

print(f"✓ Quantum Mapper created")
print(f"  - Angle Encoding: True")
print(f"  - Pairwise Interactions: True")
print(f"  - Random Fourier Features: True")
print(f"  - Max output dimension: 128")

# Apply quantum feature expansion
with torch.no_grad():
    data.x = quantum_mapper(data.x)

print(f"✓ Features expanded: {original_features} → {data.x.shape[1]}")

# ===========================================================================
# COMPUTE CLASS WEIGHTS
# ===========================================================================
print("\nComputing class weights for imbalanced data...")
train_y = data.y[train_mask]
unique_classes, class_counts = torch.unique(train_y, return_counts=True)
num_classes = data.y.max().item() + 1

print(f"Class distribution (training set):")
for cls, count in zip(unique_classes.cpu().numpy(), class_counts.cpu().numpy()):
    pct = 100 * count / train_mask.sum().item()
    print(f"  Class {cls}: {count:6d} ({pct:5.1f}%)")

# Compute class weights (inverse frequency)
class_weights = torch.zeros(num_classes, device=device)
for cls in range(num_classes):
    mask = train_y == cls
    if mask.sum() > 0:
        class_weights[cls] = 1.0 / mask.sum().item()

class_weights = class_weights / class_weights.sum()  # Normalize
print(f"\nClass weights: {class_weights.cpu().numpy()}")

# ===========================================================================
# CREATE MODEL (WITH INCREASED CAPACITY FOR EXPANDED FEATURES)
# ===========================================================================
print("\nCreating quantum-enhanced GAT model...")

quantum_config = {
    'hidden_channels': 64,   # Reduced from 128 for CPU
    'num_heads': 4,          # Reduced from 6 for CPU
    'num_layers': 2,         # Reduced from 3 for CPU
    'dropout': 0.3,          # Reduced from 0.4
    'out_channels': 2,
}

print(f"Architecture: {quantum_config}")

model = GAT(
    in_channels=data.x.shape[1],
    hidden_channels=quantum_config['hidden_channels'],
    out_channels=quantum_config['out_channels'],
    num_heads=quantum_config['num_heads'],
    num_layers=quantum_config['num_layers'],
    dropout=quantum_config['dropout'],
).to(device)

print(f"✓ Model created:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===========================================================================
# SETUP TRAINING
# ===========================================================================
print("\nSetting up training...")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=TRAINING_CONFIG['learning_rate'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize F1
    factor=TRAINING_CONFIG['lr_factor'],
    patience=TRAINING_CONFIG['lr_patience'],
)

epochs = TRAINING_CONFIG['epochs']
patience = TRAINING_CONFIG['patience']
best_val_f1 = -1
patience_counter = 0

# Metrics tracking
history = {
    'train_loss': [],
    'val_loss': [],
    'train_f1_macro': [],
    'val_f1_macro': [],
    'train_f1_micro': [],
    'val_f1_micro': [],
    'train_acc': [],
    'val_acc': [],
}

# ===========================================================================
# TRAINING LOOP
# ===========================================================================
print(f"\nTraining for up to {epochs} epochs with early stopping (patience={patience})...")
print("-" * 70)

start_time = datetime.now()

for epoch in range(1, epochs + 1):
    # =====================================================================
    # TRAINING PHASE
    # =====================================================================
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    train_loss = criterion(out[train_mask], data.y[train_mask])
    
    train_loss.backward()
    if TRAINING_CONFIG.get('clip_grad_norm'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['clip_grad_norm'])
    optimizer.step()
    
    # Compute training metrics
    with torch.no_grad():
        train_pred = out[train_mask].argmax(dim=1).cpu().numpy()
        train_y_true = data.y[train_mask].cpu().numpy()
        
        train_acc = accuracy_score(train_y_true, train_pred)
        train_f1_macro = f1_score(train_y_true, train_pred, average='macro', zero_division=0)
        train_f1_micro = f1_score(train_y_true, train_pred, average='micro', zero_division=0)
    
    # =====================================================================
    # VALIDATION PHASE
    # =====================================================================
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_loss = criterion(out[val_mask], data.y[val_mask])
        
        val_pred = out[val_mask].argmax(dim=1).cpu().numpy()
        val_y_true = data.y[val_mask].cpu().numpy()
        
        val_acc = accuracy_score(val_y_true, val_pred)
        val_f1_macro = f1_score(val_y_true, val_pred, average='macro', zero_division=0)
        val_f1_micro = f1_score(val_y_true, val_pred, average='micro', zero_division=0)
    
    # Update history
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss.item())
    history['train_f1_macro'].append(train_f1_macro)
    history['val_f1_macro'].append(val_f1_macro)
    history['train_f1_micro'].append(train_f1_micro)
    history['val_f1_micro'].append(val_f1_micro)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Early stopping based on validation F1 (macro)
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        patience_counter = 0
        
        # Save best model
        best_model_path = ARTIFACTS_DIR / "quantum_gat_best.pt"
        torch.save(model.state_dict(), best_model_path)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"Val F1 (macro): {val_f1_macro:.4f} ★ BEST")
    else:
        patience_counter += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | "
                  f"Val F1 (macro): {val_f1_macro:.4f} (patience: {patience_counter}/{patience})")
    
    # Learning rate scheduling
    scheduler.step(val_f1_macro)
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time = (datetime.now() - start_time).total_seconds()
print(f"\nTraining completed in {training_time:.1f} seconds")

# ===========================================================================
# LOAD BEST MODEL AND EVALUATE ON TEST SET
# ===========================================================================
print("\nEvaluating best model on test set...")

best_model_path = ARTIFACTS_DIR / "quantum_gat_best.pt"
model.load_state_dict(torch.load(best_model_path, weights_only=True))

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    
    test_pred = out[test_mask].argmax(dim=1).cpu().numpy()
    test_y_true = data.y[test_mask].cpu().numpy()
    
    test_acc = accuracy_score(test_y_true, test_pred)
    test_f1_macro = f1_score(test_y_true, test_pred, average='macro', zero_division=0)
    test_f1_micro = f1_score(test_y_true, test_pred, average='micro', zero_division=0)
    test_precision = precision_score(test_y_true, test_pred, average='macro', zero_division=0)
    test_recall = recall_score(test_y_true, test_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    class_report = classification_report(test_y_true, test_pred, digits=4, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_y_true, test_pred)
    
    # ROC-AUC (one-vs-rest for multi-class)
    try:
        test_roc_auc = roc_auc_score(test_y_true, out[test_mask].cpu().numpy(), 
                                     multi_class='ovr', zero_division=0)
    except:
        test_roc_auc = -1

print(f"\n{'=' * 70}")
print("TEST SET EVALUATION (QUANTUM-ENHANCED)")
print(f"{'=' * 70}")
print(f"Accuracy:       {test_acc:.4f}")
print(f"F1 (Macro):     {test_f1_macro:.4f}")
print(f"F1 (Micro):     {test_f1_micro:.4f}")
print(f"Precision:      {test_precision:.4f}")
print(f"Recall:         {test_recall:.4f}")
print(f"ROC-AUC (OvR):  {test_roc_auc:.4f}")

print(f"\n{'Per-class metrics:'}")
print(class_report)

print(f"\nConfusion Matrix:")
print(cm)

# ===========================================================================
# SAVE COMPREHENSIVE METRICS
# ===========================================================================
print("\nSaving metrics...")

metrics = {
    'model': 'quantum_gat',
    'timestamp': datetime.now().isoformat(),
    'training_time_seconds': training_time,
    'best_epoch': len(history['val_f1_macro']) - patience_counter,
    'hyperparameters': {
        'hidden_channels': quantum_config['hidden_channels'],
        'num_heads': quantum_config['num_heads'],
        'num_layers': quantum_config['num_layers'],
        'dropout': quantum_config['dropout'],
        'learning_rate': TRAINING_CONFIG['learning_rate'],
        'weight_decay': TRAINING_CONFIG['weight_decay'],
        'epochs': epoch,
        'patience': patience,
        'quantum_mapper': {
            'use_angle_encoding': True,
            'use_interactions': True,
            'use_fourier': True,
            'max_output_dim': 128,
        },
        'input_features': original_features,
        'expanded_features': data.x.shape[1],
    },
    'training_history': {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_f1_macro': history['train_f1_macro'],
        'val_f1_macro': history['val_f1_macro'],
        'train_f1_micro': history['train_f1_micro'],
        'val_f1_micro': history['val_f1_micro'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
    },
    'test_metrics': {
        'accuracy': float(test_acc),
        'f1_macro': float(test_f1_macro),
        'f1_micro': float(test_f1_micro),
        'precision_macro': float(test_precision),
        'recall_macro': float(test_recall),
        'roc_auc_ovr': float(test_roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
    },
}

# Save JSON
metrics_path = ARTIFACTS_DIR / "quantum_gat_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Saved metrics: {metrics_path}")

# ===========================================================================
# PLOT TRAINING CURVES
# ===========================================================================
print("\nGenerating training curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss (Quantum-Enhanced)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Macro F1
axes[0, 1].plot(history['train_f1_macro'], label='Train', linewidth=2)
axes[0, 1].plot(history['val_f1_macro'], label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('Macro F1 Score (Quantum-Enhanced)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy
axes[1, 0].plot(history['train_acc'], label='Train', linewidth=2)
axes[1, 0].plot(history['val_acc'], label='Val', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Accuracy (Quantum-Enhanced)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Micro F1
axes[1, 1].plot(history['train_f1_micro'], label='Train', linewidth=2)
axes[1, 1].plot(history['val_f1_micro'], label='Val', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Micro F1 Score (Quantum-Enhanced)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = ARTIFACTS_DIR / "quantum_gat_training_curves.png"
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
print(f"✓ Saved plot: {plot_path}")
plt.close()

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("QUANTUM-ENHANCED GAT TRAINING COMPLETE")
print("=" * 70)
print(f"✓ Model saved: {best_model_path}")
print(f"✓ Metrics saved: {metrics_path}")
print(f"✓ Training curves saved: {plot_path}")
print(f"\nBest Validation F1 (Macro): {best_val_f1:.4f}")
print(f"Test F1 (Macro):            {test_f1_macro:.4f}")
print(f"Test F1 (Micro):            {test_f1_micro:.4f}")
print("=" * 70)
