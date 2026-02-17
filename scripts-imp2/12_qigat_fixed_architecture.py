#!/usr/bin/env python
"""
QIGAT CORRECTED ARCHITECTURE
Fixes identified problems:
1. No early compression (182 → full dim)
2. Quantum applied AFTER first GAT (post-aggregation)
3. Learned phase projection with tanh (not raw πx)
4. Residual connection with learnable scaling
5. Weighted CE loss (not Focal)
6. Proper capacity (hidden=128)
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
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR
from src.utils import set_random_seeds

print("="*80)
print("QIGAT - CORRECTED ARCHITECTURE")
print("Quantum applied post-GAT with residual protection")
print("="*80)

set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
print("Loading data...")
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

print(f"Data: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")
print(f"Input features: {graph.num_node_features}")

# ==============================================================================
# QUANTUM PHASE BLOCK (POST-GAT)
# ==============================================================================
print("\nBuilding quantum phase block...")

class QuantumPhaseBlock(nn.Module):
    """
    Quantum phase mapping applied to GAT embeddings.
    
    Architecture:
    - Learned linear projection: φ = π * tanh(Wx)
    - Phase encoding: cos(φ), sin(φ)
    - Expands from h_dim to specified output_dim
    - Includes residual connection with learnable scaling
    
    Key: Works on refined embeddings AFTER graph aggregation, not raw features
    """
    
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Step 1: Learned linear projection
        self.W_phase = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.W_phase.weight)
        
        # Step 2: Project to output dimension if needed
        if input_dim * 2 != output_dim:
            self.compress = nn.Linear(input_dim * 2, output_dim)
        else:
            self.compress = None
        
        # Residual scaling
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # Post-quantum processing
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, h):
        """
        Forward pass through quantum phase block.
        
        Args:
            h: [N, input_dim] - GAT embeddings
            
        Returns:
            h_expanded: [N, output_dim] - quantum-enhanced embeddings
        """
        # Step 1: Project to phase
        z = self.W_phase(h)  # [N, input_dim]
        
        # Step 2: Compute phase values (bounded in [-π, π])
        phi = np.pi * torch.tanh(z)  # [N, input_dim]
        
        # Step 3: Quantum phase features
        q_cos = torch.cos(phi)  # [N, input_dim]
        q_sin = torch.sin(phi)  # [N, input_dim]
        
        # Step 4: Concatenate quantum features
        h_quantum = torch.cat([q_cos, q_sin], dim=1)  # [N, 2*input_dim]
        
        # Step 5: Compress to output dimension if needed
        if self.compress is not None:
            h_quantum = self.compress(h_quantum)  # [N, output_dim]
        
        # Step 6: LayerNorm and dropout
        h_quantum = self.norm(h_quantum)
        h_quantum = self.dropout(h_quantum)
        
        return h_quantum


# ==============================================================================
# CORRECTED QIGAT ARCHITECTURE
# ==============================================================================
print("Building corrected QIGAT model...")

class QIGAT_Corrected(nn.Module):
    """
    Corrected Quantum-Inspired Graph Attention Network.
    
    Architecture:
    1. Input: 182 features (NO compression)
    2. First GAT layer: 182 → 128
    3. LayerNorm + ELU
    4. Quantum Phase Block: 128 → 256
    5. Residual add with learnable scaling
    6. Second GAT layer: 256 → 256
    7. Output layer: 256 → 2
    
    Key improvements:
    - No early compression (182 kept intact for first GAT)
    - Quantum applied post-aggregation (when features are refined)
    - Residual connection prevents representation collapse
    - Learned phase projection (not naive πx)
    - Proper capacity for nonlinear expansion
    """
    
    def __init__(self, in_features=182, hidden_dim=128, num_heads=4, dropout=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # ===== PART A: First GAT layer (raw features → refined embeddings) =====
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat1_out_dim = hidden_dim * num_heads  # GATConv outputs hidden_dim * heads
        self.norm1 = nn.LayerNorm(self.gat1_out_dim)
        self.activation1 = nn.ELU()
        
        # ===== PART B: Quantum Phase Block (expands refined embeddings) =====
        # Quantum block: input_dim = gat1_out_dim, output_dim = 256
        self.quantum_block = QuantumPhaseBlock(self.gat1_out_dim, output_dim=256)
        
        # CRITICAL: Learnable residual scaling
        self.quantum_residual_scale = nn.Parameter(torch.tensor(0.5))
        
        # Project gat1 output for residual connection to 256-dim
        self.residual_projection = nn.Linear(self.gat1_out_dim, 256)
        
        # ===== PART C: Second GAT layer (refined + quantum → final) =====
        self.gat2 = GATConv(256, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2_out_dim = hidden_dim * num_heads
        self.norm2 = nn.LayerNorm(self.gat2_out_dim)
        self.activation2 = nn.ELU()
        
        # ===== PART D: Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(self.gat2_out_dim, self.gat2_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gat2_out_dim, 2)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass through corrected QIGAT.
        
        Args:
            x: [N, 182] - node features (FULL dimension, no compression)
            edge_index: [2, E] - edge indices
            
        Returns:
            out: [N, 2] - class logits
        """
        # Part A: First GAT - learns aggregated graph information
        h1 = self.gat1(x, edge_index)  # [N, hidden_dim * num_heads]
        h1 = self.norm1(h1)
        h1 = self.activation1(h1)
        
        # Part B: Quantum enhancement with residual
        h_quantum = self.quantum_block(h1)  # [N, 256]
        
        # Residual connection: h_residual + α * h_quantum
        h1_projected = self.residual_projection(h1)  # [N, 256]
        h_combined = h1_projected + self.quantum_residual_scale * h_quantum  # [N, 256]
        
        # Part C: Second GAT on combined representation
        h2 = self.gat2(h_combined, edge_index)  # [N, hidden_dim * num_heads]
        h2 = self.norm2(h2)
        h2 = self.activation2(h2)
        
        # Part D: Classification
        out = self.classifier(h2)  # [N, 2]
        
        return out


model = QIGAT_Corrected(
    in_features=graph.num_node_features,
    hidden_dim=128,
    num_heads=4,
    dropout=0.5
).to(device)

try:
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except:
    print("Model created successfully")

# ==============================================================================
# TRAINING SETUP
# ==============================================================================
print("\nTraining setup...")

# CRITICAL: Switch back to weighted CrossEntropy (NOT Focal Loss)
# Baseline worked perfectly with this
criterion = nn.CrossEntropyLoss(weight=class_weight)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print(f"Loss: Weighted CrossEntropy")
print(f"Optimizer: Adam (lr=0.001, weight_decay=5e-4)")
print(f"Scheduler: Cosine Annealing (T_max=300)")
print(f"Early stopping: patience=50")

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
print("\n" + "="*80)
print("TRAINING CORRECTED QIGAT")
print("="*80 + "\n")

def evaluate(model, mask):
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
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

best_val_f1 = -1
patience = 0
max_patience = 50
history = {
    'train_loss': [],
    'val_f1': [],
    'val_acc': [],
    'val_gap': []
}

start_time = time.time()

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    
    out = model(graph.x, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    # Validation
    val_metrics = evaluate(model, val_mask)
    train_metrics = evaluate(model, train_mask)
    
    val_gap = train_metrics['f1'] - val_metrics['f1']
    
    history['train_loss'].append(loss.item())
    history['val_f1'].append(val_metrics['f1'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_gap'].append(val_gap)
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience = 0
        torch.save(model.state_dict(), 'artifacts/qigat_corrected_best.pt')
    else:
        patience += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Gap: {val_gap:.4f} | Patience: {patience}/{max_patience}")
    
    if patience >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80 + "\n")

model.load_state_dict(torch.load('artifacts/qigat_corrected_best.pt', map_location=device))

qigat_train = evaluate(model, train_mask)
qigat_val = evaluate(model, val_mask)
qigat_test = evaluate(model, test_mask)

print("QIGAT (Corrected) Results:")
print(f"  Train - F1: {qigat_train['f1']:.4f}, Acc: {qigat_train['accuracy']:.4f}")
print(f"  Val   - F1: {qigat_val['f1']:.4f}, Acc: {qigat_val['accuracy']:.4f}")
print(f"  Test  - F1: {qigat_test['f1']:.4f}, Acc: {qigat_test['accuracy']:.4f}")

print(f"\nGeneralization Gaps:")
print(f"  Train→Val: {qigat_train['f1'] - qigat_val['f1']:.4f}")
print(f"  Val→Test:  {qigat_val['f1'] - qigat_test['f1']:.4f}")

print(f"\nDetailed Test Report:")
print(classification_report(qigat_test['y_true'], qigat_test['y_pred'],
                          target_names=['Non-Fraud', 'Fraud']))

# ==============================================================================
# COMPARISON WITH BASELINE
# ==============================================================================
print("\n" + "="*80)
print("COMPARISON WITH BASELINE (Reference)")
print("="*80 + "\n")

if Path('artifacts/baseline_best.pt').exists():
    baseline = GAT(
        in_channels=graph.num_node_features,
        hidden_channels=64,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    ).to(device)
    baseline.load_state_dict(torch.load('artifacts/baseline_best.pt', map_location=device))
    
    baseline.eval()
    with torch.no_grad():
        out_base = baseline(graph.x, graph.edge_index)
        baseline_test = evaluate(baseline, test_mask)
    
    print(f"Baseline GAT:        F1 = {baseline_test['f1']:.4f}")
    print(f"QIGAT (Corrected):   F1 = {qigat_test['f1']:.4f}")
    
    diff = qigat_test['f1'] - baseline_test['f1']
    pct = (diff / baseline_test['f1']) * 100
    
    print(f"Difference:          {diff:+.4f} ({pct:+.2f}%)")
    
    if qigat_test['f1'] > baseline_test['f1']:
        print(f"\n✅ SUCCESS! QIGAT OUTPERFORMS BASELINE!")
    else:
        print(f"\n⚠️  Gap remaining: {abs(diff):.4f}")

# ==============================================================================
# SAVE REPORT
# ==============================================================================
report = {
    'model': 'QIGAT (Corrected Architecture)',
    'description': 'Quantum phase mapping applied post-GAT with residual protection',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'training_time': training_time,
    'architecture': {
        'input_features': graph.num_node_features,
        'first_gat_hidden': 128,
        'quantum_expand_dim': 256,
        'second_gat_hidden': 128,
        'num_heads_gat': 4,
        'dropout': 0.5,
        'optimizer': 'AdamW (lr=0.001)',
        'loss': 'Weighted CrossEntropy (NOT Focal)',
        'key_fix': 'Quantum applied post-GAT with residual α scaling'
    },
    'test_metrics': {
        'f1': qigat_test['f1'],
        'accuracy': qigat_test['accuracy'],
        'precision': qigat_test['precision'],
        'recall': qigat_test['recall'],
        'roc_auc': qigat_test['roc_auc']
    },
    'generalization': {
        'train_f1': qigat_train['f1'],
        'val_f1': qigat_val['f1'],
        'train_to_val_gap': qigat_train['f1'] - qigat_val['f1'],
        'val_to_test_gap': qigat_val['f1'] - qigat_test['f1']
    }
}

with open('artifacts/qigat_corrected_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to artifacts/qigat_corrected_report.json")
print(f"✓ Model saved to artifacts/qigat_corrected_best.pt")

print("\n" + "="*80)
print("✅ CORRECTED QIGAT TRAINING COMPLETE!")
print("="*80)
