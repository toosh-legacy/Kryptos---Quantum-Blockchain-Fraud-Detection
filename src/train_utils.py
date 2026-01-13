"""
Training utilities for model training and evaluation.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np
from typing import Dict, Tuple, Union


def train_one_epoch(
    model: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_mask: torch.Tensor
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        data: PyG Data object
        optimizer: Optimizer
        criterion: Loss function
        train_mask: Boolean mask for training nodes
        
    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_model(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    return_predictions: bool = False
) -> Union[Dict, Tuple[Dict, np.ndarray, np.ndarray]]:
    """
    Evaluate model on a specific data split.
    
    Args:
        model: Neural network model
        data: PyG Data object
        mask: Boolean mask for evaluation nodes
        return_predictions: Whether to return predictions and probabilities
        
    Returns:
        Dictionary of metrics (and optionally predictions/probabilities)
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        logits = out[mask]
        y_true = data.y[mask].cpu().numpy()
        
        # Get predictions
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    if return_predictions:
        return metrics, y_pred, y_prob
    return metrics


def get_confusion_matrix(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor
) -> np.ndarray:
    """
    Compute confusion matrix for model predictions.
    
    Args:
        model: Neural network model
        data: PyG Data object
        mask: Boolean mask for evaluation nodes
        
    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        logits = out[mask]
        y_true = data.y[mask].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
    
    return cm


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics to maximize (accuracy, f1), 'min' for loss
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -float('inf')
        else:
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current epoch score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to add to output (e.g., "Train", "Val", "Test")
    """
    prefix_str = f"{prefix} " if prefix else ""
    print(f"{prefix_str}Metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")
