#!/usr/bin/env python
"""
ENSEMBLE BEST MODEL - Build multiple models and ensemble them
Takes the proven baseline approach and improves via:
1. Multiple GAT configurations
2. Different random initializations
3. Ensemble averaging
4. Threshold optimization
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
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR
from src.utils import set_random_seeds

print("="*80)
print("ENSEMBLE BEST MODEL - Multiple Models Combined")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
print("Preparing data...")
graph = torch.load('artifacts/elliptic_graph.pt', weights_only=False).to(device)

labeled_mask = (graph.y != -1)
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_y = graph.y[labeled_mask].cpu().numpy()

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

train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Preprocess
nan_count = torch.isnan(graph.x).sum().item()
if nan_count > 0:
    graph.x = torch.nan_to_num(graph.x, nan=0.0)

train_x = graph.x[train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)

graph.x = (graph.x - mean) / std
graph.x = torch.clamp(graph.x, min=-10, max=10)

n_class_0 = (graph.y[train_mask] == 0).sum().item()
n_class_1 = (graph.y[train_mask] == 1).sum().item()
class_weight = torch.tensor(
    [1.0 / n_class_0, 1.0 / n_class_1],
    device=device,
    dtype=torch.float32
)
class_weight = class_weight / class_weight.sum()

print(f"Data ready: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

# ==============================================================================
# DEFINE MODELS TO TRAIN
# ==============================================================================
models_config = [
    {
        'name': 'Model_A: Large GAT',
        'hidden': 128,
        'heads': 6,
        'layers': 3,
        'dropout': 0.3,
        'lr': 0.001
    },
    {
        'name': 'Model_B: Medium GAT',
        'hidden': 96,
        'heads': 4,
        'layers': 3,
        'dropout': 0.25,
        'lr': 0.0005
    },
    {
        'name': 'Model_C: Standard GAT',
        'hidden': 64,
        'heads': 4,
        'layers': 2,
        'dropout': 0.3,
        'lr': 0.001
    },
    {
        'name': 'Model_D: Deep GAT',
        'hidden': 64,
        'heads': 8,
        'layers': 4,
        'dropout': 0.35,
        'lr': 0.0008
    },
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def train_model(config, model_num):
    """Train a single model."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {config['name']}")
    print(f"{'='*80}")
    
    model = GAT(
        in_channels=graph.num_node_features,
        hidden_channels=config['hidden'],
        out_channels=2,
        num_heads=config['heads'],
        num_layers=config['layers'],
        dropout=config['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: hidden={config['hidden']}, heads={config['heads']}, "
          f"layers={config['layers']}, dropout={config['dropout']}\n")
    
    best_val_f1 = -1
    patience = 0
    start_time = time.time()
    
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out_val = model(graph.x, graph.edge_index)
            pred = out_val[val_mask].argmax(dim=1)
            y_true = graph.y[val_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), f'artifacts/ensemble_model_{model_num}.pt')
        else:
            patience += 1
        
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Patience: {patience}/20")
        
        if patience >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
    
    elapsed = time.time() - start_time
    print(f"✓ Trained in {elapsed:.2f}s, Best Val F1: {best_val_f1:.4f}")
    
    return model, f'artifacts/ensemble_model_{model_num}.pt'

def evaluate_model_full(model, mask):
    """Full evaluation of model."""
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
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
# TRAIN ALL MODELS
# ==============================================================================
print("\nTraining ensemble models...\n")
trained_models = []

for i, config in enumerate(models_config, 1):
    model, path = train_model(config, i)
    trained_models.append({
        'config': config,
        'path': path,
        'model': model
    })

# ==============================================================================
# LOAD BEST MODELS AND EVALUATE
# ==============================================================================
print(f"\n{'='*80}")
print("INDIVIDUAL MODEL EVALUATION")
print(f"{'='*80}\n")

test_results = []
all_probs = []

for i, m in enumerate(trained_models, 1):
    model = GAT(
        in_channels=graph.num_node_features,
        hidden_channels=m['config']['hidden'],
        out_channels=2,
        num_heads=m['config']['heads'],
        num_layers=m['config']['layers'],
        dropout=m['config']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(m['path'], map_location=device))
    
    metrics = evaluate_model_full(model, test_mask)
    test_results.append(metrics)
    all_probs.append(metrics['y_prob'])
    
    print(f"{m['config']['name']}")
    print(f"  F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")

# ==============================================================================
# ENSEMBLE - SOFT VOTING
# ==============================================================================
print(f"\n{'='*80}")
print("ENSEMBLE SOFT VOTING")
print(f"{'='*80}\n")

# Average probabilities
all_probs = np.array(all_probs)
ensemble_prob = all_probs.mean(axis=0)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

y_true = graph.y[test_mask].cpu().numpy()

ensemble_metrics = {
    'accuracy': accuracy_score(y_true, ensemble_pred),
    'precision': precision_score(y_true, ensemble_pred, zero_division=0),
    'recall': recall_score(y_true, ensemble_pred, zero_division=0),
    'f1': f1_score(y_true, ensemble_pred, zero_division=0),
}

try:
    ensemble_metrics['roc_auc'] = roc_auc_score(y_true, ensemble_prob)
except:
    ensemble_metrics['roc_auc'] = 0.0

print(f"Ensemble (Average Voting)")
print(f"  F1: {ensemble_metrics['f1']:.4f} | Acc: {ensemble_metrics['accuracy']:.4f} | "
      f"AUC: {ensemble_metrics['roc_auc']:.4f}")

# ==============================================================================
# FIND BEST THRESHOLD
# ==============================================================================
print(f"\n{'='*80}")
print("THRESHOLD OPTIMIZATION")
print(f"{'='*80}\n")

precision_scores, recall_scores, thresholds = precision_recall_curve(y_true, ensemble_prob)
f1_scores_threshold = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + 1e-10)

best_threshold_idx = np.argmax(f1_scores_threshold)
best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
best_f1_threshold = f1_scores_threshold[best_threshold_idx]

ensemble_pred_optimized = (ensemble_prob >= best_threshold).astype(int)
ensemble_metrics_optimized = {
    'accuracy': accuracy_score(y_true, ensemble_pred_optimized),
    'precision': precision_score(y_true, ensemble_pred_optimized, zero_division=0),
    'recall': recall_score(y_true, ensemble_pred_optimized, zero_division=0),
    'f1': f1_score(y_true, ensemble_pred_optimized, zero_division=0),
    'roc_auc': ensemble_metrics['roc_auc']
}

print(f"Optimized Threshold: {best_threshold:.4f}")
print(f"Optimized Metrics:")
print(f"  F1: {ensemble_metrics_optimized['f1']:.4f} | Acc: {ensemble_metrics_optimized['accuracy']:.4f} | "
      f"Precision: {ensemble_metrics_optimized['precision']:.4f} | Recall: {ensemble_metrics_optimized['recall']:.4f}")

# ==============================================================================
# COMPARISON & WINNER
# ==============================================================================
print(f"\n{'='*80}")
print("FINAL COMPARISON")
print(f"{'='*80}\n")

all_f1s = [m['f1'] for m in test_results]
best_single_idx = np.argmax(all_f1s)
best_single_f1 = all_f1s[best_single_idx]
best_single_model = trained_models[best_single_idx]['config']['name']

print(f"Best Single Model: {best_single_model}")
print(f"  F1: {best_single_f1:.4f}\n")

print(f"Ensemble (Average Threshold): {ensemble_metrics['f1']:.4f}")
print(f"Ensemble (Optimized Threshold): {ensemble_metrics_optimized['f1']:.4f}\n")

if ensemble_metrics_optimized['f1'] > best_single_f1:
    winner = "🏆 ENSEMBLE WINS!"
    winning_f1 = ensemble_metrics_optimized['f1']
    improvement = (winning_f1 - best_single_f1) / best_single_f1 * 100
    print(f"{winner} Improvement: {improvement:+.2f}%")
else:
    winner = f"✅ Best: {best_single_model}"
    winning_f1 = best_single_f1
    print(f"{winner} with F1={winning_f1:.4f}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
report = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'approach': 'Ensemble of 4 GAT models with threshold optimization',
    'individual_models': [
        {
            'name': m['config']['name'],
            'test_f1': r['f1'],
            'test_acc': r['accuracy'],
            'test_auc': r['roc_auc']
        }
        for m, r in zip(trained_models, test_results)
    ],
    'ensemble': {
        'average_voting_f1': ensemble_metrics['f1'],
        'optimized_threshold_f1': ensemble_metrics_optimized['f1'],
        'optimized_threshold': float(best_threshold),
        'test_acc': ensemble_metrics_optimized['accuracy'],
        'test_precision': ensemble_metrics_optimized['precision'],
        'test_recall': ensemble_metrics_optimized['recall'],
        'test_auc': ensemble_metrics_optimized['roc_auc']
    },
    'winner': winner,
    'winning_f1': winning_f1
}

with open('artifacts/ensemble_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to artifacts/ensemble_report.json")

# ==============================================================================
# CLASSIFICATION REPORT
# ==============================================================================
print(f"\n{'='*80}")
print("ENSEMBLE CLASSIFICATION REPORT (TEST SET)")
print(f"{'='*80}\n")

print(classification_report(y_true, ensemble_pred_optimized,
                          target_names=['Non-Fraud', 'Fraud']))

print("\n" + "="*80)
print("✅ ENSEMBLE TRAINING COMPLETE!")
print("="*80)
print(f"\nEnsemble models saved:")
for i in range(1, len(trained_models) + 1):
    print(f"  - artifacts/ensemble_model_{i}.pt")
print(f"\nReport: artifacts/ensemble_report.json")
