"""
Quick training script for improved quantum model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import json
import time

from src.models import GAT
from src.config import QUANTUM_MODEL_CONFIG, TRAINING_CONFIG, ARTIFACTS_DIR, ARTIFACT_FILES
from src.utils import set_random_seeds, get_device
from src.train_utils import FocalLoss, compute_class_weights

# Set random seeds
set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print(f"🚀 Starting improved quantum model training...")
print(f"Device: {device}")

# Load quantum graph
graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_graph']
data = torch.load(graph_path, weights_only=False).to(device)
print(f"\n✓ Quantum graph loaded: {data}")
print(f"  Features: {data.num_node_features} dimensions")
print(f"  Nodes: {data.num_nodes}")
print(f"  Edges: {data.edge_index.shape[1]}")

# Create train/val/test splits if needed
if not hasattr(data, 'train_mask'):
    print("\n📊 Creating train/val/test splits...")
    labeled_indices = torch.where(data.labeled_mask)[0].cpu().numpy()
    labeled_y = data.y[data.labeled_mask].cpu().numpy()
    
    train_val_idx, test_idx = train_test_split(
        labeled_indices, 
        test_size=TRAINING_CONFIG['train_test_split'],
        random_state=TRAINING_CONFIG['random_seed'],
        stratify=labeled_y
    )
    
    train_val_y = data.y[train_val_idx].cpu().numpy()
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=TRAINING_CONFIG['train_val_split'],
        random_state=TRAINING_CONFIG['random_seed'],
        stratify=train_val_y
    )
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

print(f"  Train: {data.train_mask.sum()}")
print(f"  Val:   {data.val_mask.sum()}")
print(f"  Test:  {data.test_mask.sum()}")

# Check class distribution
train_labels = data.y[data.train_mask]
class_0_count = (train_labels == 0).sum().item()
class_1_count = (train_labels == 1).sum().item()
print(f"\n📊 Class distribution:")
print(f"  Class 0 (Licit):   {class_0_count:,}")
print(f"  Class 1 (Illicit): {class_1_count:,}")
print(f"  Imbalance ratio:   {class_0_count/class_1_count:.2f}:1")

# Initialize model with QUANTUM_MODEL_CONFIG
print(f"\n🔧 Initializing GAT model...")
print(f"  Config: {QUANTUM_MODEL_CONFIG}")
model = GAT(
    in_channels=data.num_node_features,
    hidden_channels=QUANTUM_MODEL_CONFIG['hidden_channels'],
    out_channels=QUANTUM_MODEL_CONFIG['out_channels'],
    num_heads=QUANTUM_MODEL_CONFIG['num_heads'],
    num_layers=QUANTUM_MODEL_CONFIG['num_layers'],
    dropout=QUANTUM_MODEL_CONFIG['dropout']
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {param_count:,}")

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=TRAINING_CONFIG['learning_rate'], 
    weight_decay=TRAINING_CONFIG['weight_decay']
)

# CRITICAL: Weighted loss for class imbalance
if TRAINING_CONFIG['use_class_weights']:
    class_weights = compute_class_weights(train_labels, num_classes=2).to(device)
    print(f"\n⚖️  Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("✓ Using weighted CrossEntropyLoss")
elif TRAINING_CONFIG['use_focal_loss']:
    criterion = FocalLoss(
        alpha=TRAINING_CONFIG['focal_alpha'], 
        gamma=TRAINING_CONFIG['focal_gamma']
    )
    print(f"✓ Using Focal Loss (alpha={TRAINING_CONFIG['focal_alpha']}, gamma={TRAINING_CONFIG['focal_gamma']})")
else:
    criterion = nn.CrossEntropyLoss()
    print("⚠️  Using standard CrossEntropyLoss")

# Learning rate scheduler
scheduler = None
if TRAINING_CONFIG['lr_scheduler']:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=TRAINING_CONFIG['lr_factor'],
        patience=TRAINING_CONFIG['lr_patience']
    )
    print(f"✓ Using ReduceLROnPlateau scheduler (patience={TRAINING_CONFIG['lr_patience']})")

# Training functions
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    
    if TRAINING_CONFIG.get('clip_grad_norm'):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=TRAINING_CONFIG['clip_grad_norm']
        )
    
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    prob = F.softmax(out[mask], dim=1)[:, 1]

    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()

    if np.isnan(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.5)
    
    if np.isinf(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except (ValueError, RuntimeError) as e:
        roc_auc = 0.5

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }

# Training loop
print(f"\n🚀 Starting training...")
print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
print(f"  Patience: {TRAINING_CONFIG['patience']}")
print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
print(f"  Gradient clipping: {TRAINING_CONFIG.get('clip_grad_norm', 'None')}")
print()

history = {
    'train_loss': [], 
    'val_acc': [], 
    'val_precision': [],
    'val_recall': [],
    'val_f1': [], 
    'val_auc': []
}
best_val_f1 = 0
patience_counter = 0
EPOCHS = TRAINING_CONFIG['epochs']
PATIENCE = TRAINING_CONFIG['patience']

model_save_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model']
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    val_metrics = evaluate(data.val_mask)
    
    history['train_loss'].append(loss)
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_auc'].append(val_metrics['roc_auc'])
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | "
              f"AUC: {val_metrics['roc_auc']:.4f}")
    
    # Update learning rate scheduler
    if scheduler is not None:
        scheduler.step(val_metrics['f1'])
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_f1,
            'val_metrics': val_metrics,
            'config': QUANTUM_MODEL_CONFIG
        }, model_save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break

training_time = time.time() - start_time

# Save results
results = {
    'training_time': training_time,
    'best_val_f1': best_val_f1,
    'total_epochs': epoch,
    'history': history
}

results_path = ARTIFACTS_DIR / 'quantum_training_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Training complete!")
print(f"  Time: {training_time:.2f}s")
print(f"  Best Val F1: {best_val_f1:.4f}")

# Final evaluation on test set
checkpoint = torch.load(model_save_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
test_metrics = evaluate(data.test_mask)

print(f"\n📊 Test Set Performance:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1 Score:  {test_metrics['f1']:.4f}")
print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")

metrics_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_metrics']
with open(metrics_path, 'w') as f:
    json.dump(test_metrics, f, indent=2)

print(f"\n✅ All done! Metrics saved to {metrics_path}")
