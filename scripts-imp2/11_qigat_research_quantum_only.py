#!/usr/bin/env python
"""
QIGAT - Quantum-Inspired Graph Attention Network
Focus ONLY on quantum model training and optimization.
Uses proven baseline as reference (no training).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
import time
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GAT
from src.config import ARTIFACTS_DIR
from src.utils import set_random_seeds

print("="*80)
print("QIGAT - QUANTUM-INSPIRED GRAPH ATTENTION NETWORK")
print("Focus: Quantum Model Only")
print("="*80)

set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading graph data...")
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

print(f"Data loaded: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")

# ==============================================================================
# PART 1: FEATURE ENRICHMENT
# ==============================================================================
print("\nPart 1: Feature Enrichment with Structural Signals...")

def compute_node_degrees(edge_index, num_nodes):
    """Compute node degrees."""
    degrees = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    for edge in edge_index.t():
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    return degrees

def compute_pagerank(edge_index, num_nodes, damping=0.85, iterations=5):
    """Approximate PageRank using sparse operations."""
    import torch_scatter
    
    pr = torch.ones(num_nodes, device=device) / num_nodes
    
    # Use sparse matrix approach
    edge_src, edge_dst = edge_index
    
    # Power iteration with sparse operations
    for _ in range(iterations):
        # Outgoing edges from each node
        out_degree = torch.zeros(num_nodes, device=device)
        out_degree.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float32))
        out_degree = torch.clamp(out_degree, min=1.0)
        
        # Distribute PageRank to neighbors
        pr_distributed = pr[edge_src] / out_degree[edge_src]
        pr_new = torch.zeros(num_nodes, device=device)
        pr_new.scatter_add_(0, edge_dst, pr_distributed)
        
        # Apply damping
        pr = (1 - damping) / num_nodes + damping * pr_new
    
    return pr

def compute_clustering_coefficient(edge_index, num_nodes):
    """Approximate local clustering coefficient using neighbor sets."""
    from collections import defaultdict
    
    cc = torch.zeros(num_nodes, device=device)
    
    # Build adjacency list
    adj_list = defaultdict(set)
    edge_src, edge_dst = edge_index
    
    for src, dst in zip(edge_src.cpu().numpy(), edge_dst.cpu().numpy()):
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    
    # Compute clustering coefficient
    for node in range(min(num_nodes, 10000)):  # Limit to first 10k nodes for efficiency
        neighbors = list(adj_list[node])
        if len(neighbors) < 2:
            continue
        
        # Count edges between neighbors
        edges = 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if neighbors[j] in adj_list[neighbors[i]]:
                    edges += 1
        
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        if possible_edges > 0:
            cc[node] = edges / possible_edges
    
    # Propagate values to remaining nodes (approximate)
    if num_nodes > 10000:
        mean_cc = cc[:10000].mean()
        cc[10000:] = mean_cc
    
    return cc

# Compute structural features (sample-based for efficiency)
print("  Computing degrees...")
degrees = compute_node_degrees(graph.edge_index, graph.num_nodes)
log_degrees = torch.log1p(degrees)

print("  Computing PageRank...")
pagerank = compute_pagerank(graph.edge_index, graph.num_nodes, iterations=5)

print("  Computing clustering coefficients...")
clustering = compute_clustering_coefficient(graph.edge_index, graph.num_nodes)

# Normalize structural features
degrees_norm = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
log_degrees_norm = (log_degrees - log_degrees.mean()) / (log_degrees.std() + 1e-8)
pagerank_norm = (pagerank - pagerank.mean()) / (pagerank.std() + 1e-8)
clustering_norm = (clustering - clustering.mean()) / (clustering.std() + 1e-8)

# Simple feature aggregation (instead of 2-hop matrix multiplication)
# Use mean of neighbor features
print("  Aggregating neighbor features...")
edge_src, edge_dst = graph.edge_index
neighbor_features = torch.zeros_like(graph.x)
neighbor_counts = torch.zeros(graph.num_nodes, 1, device=device)

for src, dst in zip(edge_src, edge_dst):
    neighbor_features[dst] += graph.x[src]
    neighbor_counts[dst] += 1

neighbor_counts = torch.clamp(neighbor_counts, min=1.0)
neighbor_features = neighbor_features / neighbor_counts

# Concatenate all enriched features
x_enriched = torch.cat([
    graph.x,
    degrees_norm.unsqueeze(1),
    log_degrees_norm.unsqueeze(1),
    pagerank_norm.unsqueeze(1),
    clustering_norm.unsqueeze(1),
    neighbor_features
], dim=1)

print(f"  Original features: {graph.x.shape[1]}")
print(f"  Enriched features: {x_enriched.shape[1]}")

# Normalize enriched features
train_x = x_enriched[train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)

x_enriched = (x_enriched - mean) / std
x_enriched = torch.clamp(x_enriched, min=-10, max=10)

# L2 normalize per node
x_enriched = F.normalize(x_enriched, p=2, dim=1)

print(f"  Features normalized and L2-normalized")

# ==============================================================================
# PART 2: QUANTUM FEATURE MAPPER
# ==============================================================================
print("\nPart 2: Quantum-Inspired Feature Mapping...")

class QuantumFeatureMapperResearch(nn.Module):
    """
    Research-grade quantum feature mapper with:
    - Learned linear projection
    - Phase encoding via tanh
    - Cos/sin quantum features
    - Second-order pairwise terms
    - Layer normalization
    - Learnable scaling
    """
    
    def __init__(self, input_dim, output_dim=256, k_pairwise=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_pairwise = k_pairwise
        
        # Step 1: Learned linear projection
        self.W_proj = nn.Linear(input_dim, input_dim)
        nn.init.orthogonal_(self.W_proj.weight)
        
        # Calculate output dimensions
        self.cos_sin_dim = input_dim * 2
        self.pairwise_dim = min(k_pairwise, input_dim * (input_dim - 1) // 2)
        total_dim = self.cos_sin_dim + self.pairwise_dim
        
        # Compression if needed
        if total_dim > output_dim:
            self.compression = nn.Linear(total_dim, output_dim)
        else:
            self.compression = None
            self.output_dim = total_dim
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(self.output_dim if self.compression is None else output_dim)
        self.dropout = nn.Dropout(0.3)
        
        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Step 1: Linear projection
        z = self.W_proj(x)  # [N, input_dim]
        
        # Step 2: Phase encoding via tanh
        phi = torch.tanh(z)  # [N, input_dim]
        
        # Step 3: Quantum features - cos and sin
        q_cos = torch.cos(np.pi * phi)  # [N, input_dim]
        q_sin = torch.sin(np.pi * phi)  # [N, input_dim]
        
        # Step 4: Second-order pairwise terms (top-k by variance)
        if self.pairwise_dim > 0:
            # Compute variance per feature
            z_var = z.var(dim=0)
            top_k_indices = torch.topk(z_var, self.k_pairwise, largest=True)[1]
            
            # Compute pairwise interactions for top-k features
            pairwise_features = []
            for i in range(len(top_k_indices)):
                for j in range(i+1, len(top_k_indices)):
                    idx_i, idx_j = top_k_indices[i], top_k_indices[j]
                    pairwise = torch.cos(phi[:, idx_i] - phi[:, idx_j])
                    pairwise_features.append(pairwise.unsqueeze(1))
                    
                    if len(pairwise_features) >= self.pairwise_dim:
                        break
                if len(pairwise_features) >= self.pairwise_dim:
                    break
            
            if pairwise_features:
                q_pairwise = torch.cat(pairwise_features, dim=1)  # [N, pairwise_dim]
            else:
                q_pairwise = torch.zeros(x.shape[0], 1, device=x.device)
        else:
            q_pairwise = torch.zeros(x.shape[0], 1, device=x.device)
        
        # Concatenate all quantum features
        Q = torch.cat([q_cos, q_sin, q_pairwise], dim=1)  # [N, total_dim]
        
        # Apply compression if needed
        if self.compression is not None:
            Q = self.compression(Q)
        
        # Apply layer normalization
        Q = self.layer_norm(Q)
        
        # Apply learnable scaling
        Q = Q * self.alpha
        
        # Apply dropout
        Q = self.dropout(Q)
        
        return Q

quantum_mapper = QuantumFeatureMapperResearch(
    input_dim=x_enriched.shape[1],
    output_dim=256,
    k_pairwise=32
).to(device)

# Get quantum features
x_quantum = quantum_mapper(x_enriched)
print(f"  Quantum mapper output dim: {x_quantum.shape[1]}")

# ==============================================================================
# PART 3: STRUCTURAL-AWARE ATTENTION GAT
# ==============================================================================
print("\nPart 3: Building Structural-Aware Attention GAT...")

class GAT_StructuralAttention(nn.Module):
    """
    GAT with structural-aware attention:
    Instead of a^T [Wh_i || Wh_j]
    Uses: a^T σ(W1 h_i + W2 h_j + W3 (h_i ⊙ h_j))
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layer stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GAT_AttentionLayer(in_channels, hidden_channels, num_heads, dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * num_heads))
        self.activations.append(nn.ELU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GAT_AttentionLayer(hidden_channels * num_heads, hidden_channels, num_heads, dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * num_heads))
            self.activations.append(nn.ELU())
        
        # Output layer
        self.output_layer = nn.Linear(hidden_channels * num_heads, out_channels)
    
    def forward(self, x, edge_index):
        for i, (conv, norm, activation) in enumerate(zip(self.convs, self.norms, self.activations)):
            x_residual = x if i == 0 else x
            x = conv(x, edge_index)
            x = norm(x)
            x = activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if x_residual.shape == x.shape:
                x = x + x_residual
        
        x = self.output_layer(x)
        return x

class GAT_AttentionLayer(nn.Module):
    """Single GAT attention layer with structural awareness."""
    
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Separate weights for i and j
        self.W1 = nn.Linear(in_channels, num_heads * hidden_channels)
        self.W2 = nn.Linear(in_channels, num_heads * hidden_channels)
        self.W3 = nn.Linear(in_channels, num_heads * hidden_channels)
        
        # Attention coefficients
        self.a = nn.Parameter(torch.zeros(num_heads, hidden_channels))
        nn.init.xavier_uniform_(self.a)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
    
    def forward(self, x, edge_index):
        N = x.shape[0]
        
        # Compute features for each node
        h1 = self.W1(x).reshape(N, self.num_heads, self.hidden_channels)
        h2 = self.W2(x).reshape(N, self.num_heads, self.hidden_channels)
        h3 = self.W3(x).reshape(N, self.num_heads, self.hidden_channels)
        
        # Element-wise product
        h_prod = h1 * h2
        
        # Compute attention: a^T σ(h1 + h2 + h_prod)
        # For each edge (i, j)
        edge_src, edge_dst = edge_index
        
        h1_src = h1[edge_src]  # [E, H, D]
        h2_dst = h2[edge_dst]  # [E, H, D]
        h_prod = h_prod[edge_src] * h_prod[edge_dst]  # [E, H, D]
        
        # Compute attention logits
        attn_input = h1_src + h2_dst + h_prod  # [E, H, D]
        attn_input = F.elu(attn_input)
        
        # Apply attention weights: [H, D] @ [E, H, D]^T -> [E, H]
        attn_logits = torch.einsum('hd,ehd->eh', self.a, attn_input)  # [E, H]
        attn_logits = attn_logits / np.sqrt(self.hidden_channels)
        
        # Aggregate attention per head
        attn_logits = attn_logits.softmax(dim=1)  # Normalize per destination
        attn_logits = F.dropout(attn_logits, p=self.dropout, training=self.training)
        
        # Message passing: aggregate using attention
        out = torch.zeros(N, self.num_heads, self.hidden_channels, device=x.device)
        for h in range(self.num_heads):
            for e_idx, (src, dst) in enumerate(edge_index.t()):
                out[dst, h] += attn_logits[e_idx, h] * h1[src, h]
        
        # Reshape output
        out = out.reshape(N, -1)
        
        return out

# ==============================================================================
# PART 4: FULL QIGAT MODEL
# ==============================================================================
print("\nPart 4: Building Full QIGAT Model...")

class QIGAT(nn.Module):
    """Quantum-Inspired Graph Attention Network (QIGAT)."""
    
    def __init__(self, input_dim, quantum_out_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.5):
        super().__init__()
        
        # Quantum feature mapper
        self.quantum_mapper = QuantumFeatureMapperResearch(input_dim, quantum_out_dim, k_pairwise=32)
        
        # Compression layer
        self.linear_compression = nn.Linear(quantum_out_dim, hidden_dim)
        self.compression_norm = nn.LayerNorm(hidden_dim)
        
        # GAT with structural attention
        self.gat = GAT_StructuralAttention(
            hidden_dim, hidden_dim, output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x, edge_index):
        # Quantum feature mapping
        x_quantum = self.quantum_mapper(x)
        
        # Compression
        x_compressed = self.linear_compression(x_quantum)
        x_compressed = self.compression_norm(x_compressed)
        x_compressed = F.elu(x_compressed)
        x_compressed = F.dropout(x_compressed, p=0.3, training=self.training)
        
        # GAT layers
        x_out = self.gat(x_compressed, edge_index)
        
        return x_out

qigat_model = QIGAT(
    input_dim=x_enriched.shape[1],
    quantum_out_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_layers=2,
    output_dim=2,
    dropout=0.5
).to(device)

print(f"QIGAT model created with {sum(p.numel() for p in qigat_model.parameters()):,} parameters")

# ==============================================================================
# PART 5: REGULARIZATION & LOSS
# ==============================================================================
print("\nPart 5: Setting up Regularization and Loss...")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

# Class weights
n_class_0 = (graph.y[train_mask] == 0).sum().item()
n_class_1 = (graph.y[train_mask] == 1).sum().item()
class_weight = torch.tensor(
    [1.0 / n_class_0, 1.0 / n_class_1],
    device=device,
    dtype=torch.float32
)
class_weight = class_weight / class_weight.sum()

criterion = FocalLoss(alpha=class_weight, gamma=2.5)

optimizer = torch.optim.AdamW(
    qigat_model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print(f"Class weights: {class_weight.cpu().numpy()}")
print(f"Optimizer: AdamW (lr=0.001, weight_decay=5e-4)")
print(f"Scheduler: Cosine Annealing (T_max=300)")

# ==============================================================================
# PART 6: TRAINING LOOP
# ==============================================================================
print("\n" + "="*80)
print("TRAINING QIGAT")
print("="*80 + "\n")

def evaluate(model, mask):
    model.eval()
    with torch.no_grad():
        out = model(x_enriched, graph.edge_index)
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
history = {'train_loss': [], 'val_f1': [], 'val_acc': [], 'lr': []}

start_time = time.time()

for epoch in range(1, 301):
    qigat_model.train()
    optimizer.zero_grad()
    
    out = qigat_model(x_enriched, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qigat_model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    # Validation
    val_metrics = evaluate(qigat_model, val_mask)
    
    history['train_loss'].append(loss.item())
    history['val_f1'].append(val_metrics['f1'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience = 0
        torch.save(qigat_model.state_dict(), 'artifacts/qigat_best.pt')
    else:
        patience += 1
    
    if epoch % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Patience: {patience}/{max_patience}")
    
    if patience >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# ==============================================================================
# PART 7: EVALUATION
# ==============================================================================
print("\n" + "="*80)
print("EVALUATION")
print("="*80 + "\n")

qigat_model.load_state_dict(torch.load('artifacts/qigat_best.pt', map_location=device))

qigat_train = evaluate(qigat_model, train_mask)
qigat_val = evaluate(qigat_model, val_mask)
qigat_test = evaluate(qigat_model, test_mask)

print("QIGAT Results:")
print(f"  Train - F1: {qigat_train['f1']:.4f}, Acc: {qigat_train['accuracy']:.4f}, AUC: {qigat_train['roc_auc']:.4f}")
print(f"  Val   - F1: {qigat_val['f1']:.4f}, Acc: {qigat_val['accuracy']:.4f}, AUC: {qigat_val['roc_auc']:.4f}")
print(f"  Test  - F1: {qigat_test['f1']:.4f}, Acc: {qigat_test['accuracy']:.4f}, AUC: {qigat_test['roc_auc']:.4f}")

print("\nDetailed Test Classification Report:")
print(classification_report(qigat_test['y_true'], qigat_test['y_pred'],
                          target_names=['Non-Fraud', 'Fraud']))

# ==============================================================================
# COMPARISON WITH REFERENCE BASELINE
# ==============================================================================
print("\n" + "="*80)
print("COMPARISON WITH BASELINE")
print("="*80 + "\n")

# Load baseline results if available
if Path('artifacts/baseline_best.pt').exists():
    print("Loading baseline model for comparison...")
    baseline = GAT(
        in_channels=graph.num_node_features,
        hidden_channels=64,
        out_channels=2,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    ).to(device)
    baseline.load_state_dict(torch.load('artifacts/baseline_best.pt', map_location=device))
    
    # Evaluate baseline on original features
    baseline.eval()
    with torch.no_grad():
        out_base = baseline(graph.x, graph.edge_index)
        baseline_test_f1 = f1_score(graph.y[test_mask].cpu().numpy(),
                                   out_base[test_mask].argmax(dim=1).cpu().numpy(),
                                   average='macro', zero_division=0)
    
    improvement = ((qigat_test['f1'] - baseline_test_f1) / baseline_test_f1) * 100
    
    print(f"Baseline F1:  {baseline_test_f1:.4f}")
    print(f"QIGAT F1:     {qigat_test['f1']:.4f}")
    print(f"Improvement:  {improvement:+.2f}%")
    
    if qigat_test['f1'] > baseline_test_f1:
        print(f"\n✅ QIGAT OUTPERFORMS BASELINE!")
    else:
        print(f"\n⚠️  Baseline still stronger")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
report = {
    'model': 'QIGAT',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'training_time': training_time,
    'test_metrics': {
        'f1': qigat_test['f1'],
        'accuracy': qigat_test['accuracy'],
        'precision': qigat_test['precision'],
        'recall': qigat_test['recall'],
        'roc_auc': qigat_test['roc_auc']
    },
    'architecture': {
        'enriched_features': x_enriched.shape[1],
        'quantum_output_dim': 256,
        'hidden_channels': 128,
        'num_heads': 8,
        'num_layers': 2,
        'dropout': 0.5
    },
    'history': {
        'final_epoch': epoch,
        'best_val_f1': best_val_f1,
        'best_epoch': epoch - patience
    }
}

with open('artifacts/qigat_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to artifacts/qigat_report.json")
print(f"✓ Model saved to artifacts/qigat_best.pt")

print("\n" + "="*80)
print("✅ QIGAT TRAINING COMPLETE!")
print("="*80)
