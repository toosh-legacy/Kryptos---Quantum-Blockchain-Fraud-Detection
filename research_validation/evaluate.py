"""
EVALUATION MODULE

Compute metrics: F1, Precision, Recall, ROC-AUC, per-class breakdown
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, precision_recall_fscore_support
)


def evaluate(model, data, mask):
    """
    Evaluate model on a given mask (train/val/test)
    
    Returns:
        dict with comprehensive metrics
    """
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        prob = F.softmax(out[mask], dim=1)[:, 1]
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        
        # Per-class metrics
        p0, r0, f0, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0], zero_division=0
        )
        p1, r1, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], zero_division=0
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc,
        'f1_class_0': f0[0],
        'f1_class_1': f1[0],
        'precision_class_0': p0[0],
        'precision_class_1': p1[0],
        'recall_class_0': r0[0],
        'recall_class_1': r1[0],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def format_metrics(metrics, prefix=''):
    """
    Format metrics for pretty printing
    
    Args:
        metrics: dict from evaluate()
        prefix: string prefix (e.g., 'Train', 'Val')
    
    Returns:
        formatted string
    """
    
    lines = []
    if prefix:
        lines.append(f"\n{prefix} Metrics:")
    
    lines.append(f"  Accuracy:         {metrics['accuracy']:.4f}")
    lines.append(f"  F1 (macro):       {metrics['f1']:.4f}")
    lines.append(f"  Precision:        {metrics['precision']:.4f}")
    lines.append(f"  Recall:           {metrics['recall']:.4f}")
    lines.append(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    
    lines.append(f"\n  Per-Class F1:")
    lines.append(f"    Class 0 (Non-Fraud): {metrics['f1_class_0']:.4f}")
    lines.append(f"    Class 1 (Fraud):     {metrics['f1_class_1']:.4f}")
    
    return '\n'.join(lines)


def compute_efficiency(test_f1, num_parameters):
    """Compute F1 per million parameters"""
    return test_f1 / (num_parameters / 1e6)
