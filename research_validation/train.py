"""
TRAINING MODULE FOR CONTROLLED EXPERIMENTS

Standardized training procedure for all 4 models with:
- Identical hyperparameters
- Early stopping (patience=50)
- Learning rate scheduling (Cosine Annealing)
- Gradient clipping
- Weighted loss for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time


class Trainer:
    """
    Unified trainer for all models with identical training protocol
    """
    
    def __init__(self, model, device, class_weight, learning_rate=0.001,
                 weight_decay=5e-4, patience=50, max_epochs=300):
        
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs
        
        # Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs
        )
        
        # Tracking
        self.best_val_f1 = -1
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_f1': [],
            'epoch': []
        }
        
        self.training_time = 0
    
    def train_epoch(self, data, train_mask):
        """Single training epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(self, data, train_mask, val_mask, test_mask, evaluate_fn, 
              verbose=True):
        """
        Full training loop with early stopping
        
        Args:
            data: PyG Data object
            train_mask, val_mask, test_mask: Boolean tensors
            evaluate_fn: Function to compute metrics
            verbose: Print progress
        """
        
        start_time = time.time()
        
        for epoch in range(1, self.max_epochs + 1):
            # Train step
            loss = self.train_epoch(data, train_mask)
            
            # Validation step
            val_metrics = evaluate_fn(self.model, data, val_mask)
            val_f1 = val_metrics['f1']
            
            # Track history
            self.training_history['train_loss'].append(loss)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['epoch'].append(epoch)
            
            # Early stopping check
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                # Save best model
                best_state = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'val_f1': val_f1,
                    'loss': loss
                }
            else:
                self.patience_counter += 1
            
            # Logging
            if verbose and (epoch % 10 == 0 or epoch < 5):
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                      f"Val F1: {val_f1:.4f} | Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if 'best_state' in locals():
            self.model.load_state_dict(best_state['model_state'])
        
        self.training_time = time.time() - start_time
        
        # Final evaluation
        train_metrics = evaluate_fn(self.model, data, train_mask)
        val_metrics = evaluate_fn(self.model, data, val_mask)
        test_metrics = evaluate_fn(self.model, data, test_mask)
        
        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'training_time': self.training_time,
            'epochs_trained': epoch,
            'history': self.training_history
        }


def train_model(model, data, train_mask, val_mask, test_mask, 
                class_weight, device, evaluate_fn, model_name='',
                verbose=True):
    """
    Convenience function to train a single model
    
    Returns:
        dict with train/val/test metrics
    """
    
    trainer = Trainer(
        model=model,
        device=device,
        class_weight=class_weight,
        learning_rate=0.001,
        weight_decay=5e-4,
        patience=50,
        max_epochs=300
    )
    
    if verbose:
        print(f"\nTraining {model_name}...")
    
    results = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        evaluate_fn=evaluate_fn,
        verbose=verbose
    )
    
    return results
