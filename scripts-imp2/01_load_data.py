"""
01_load_data.py: Load graph data and create train/val/test splits.

This script:
- Loads the graph from artifacts (or creates it if not available)
- Normalizes features using StandardScaler
- Creates stratified train/val/test splits (70/15/15)
- Saves the graph with masks for training scripts
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ARTIFACTS_DIR, DATA_DIR, DATASET_CONFIG, TRAINING_CONFIG
from src.utils import set_random_seeds, get_device

# Set reproducibility
set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("=" * 70)
print("STEP 1: LOAD GRAPH DATA WITH PREPROCESSING")
print("=" * 70)
print(f"Device: {device}\n")

# Define paths
baseline_graph_path = ARTIFACTS_DIR / "elliptic_graph.pt"
quantum_graph_path = ARTIFACTS_DIR / "elliptic_graph_quantum.pt"

# ===========================================================================
# LOAD OR CREATE GRAPH
# ===========================================================================
print("Loading graph data...")

if not baseline_graph_path.exists():
    print(f"⚠ Graph file not found at {baseline_graph_path}")
    print("Please run scripts-imp1/02_data_graph.py first to create the graph.")
    sys.exit(1)

# Load baseline graph
data = torch.load(baseline_graph_path, weights_only=False).to(device)
print(f"✓ Loaded graph: {data}")
print(f"  - Nodes: {data.num_nodes}")
print(f"  - Edges: {data.num_edges}")
print(f"  - Features: {data.x.shape[1]}")
print(f"  - Classes: {data.y.max().item() + 1}")

# ===========================================================================
# FEATURE NORMALIZATION
# ===========================================================================
print("\nNormalizing features...")

# Get node features
x = data.x.cpu().numpy()
print(f"Original feature range: [{np.nanmin(x):.4f}, {np.nanmax(x):.4f}]")

# Check for zero-variance features and NaN values
zero_var_features = np.where(np.std(x, axis=0) == 0)[0]
print(f"Zero-variance features: {len(zero_var_features)}")

# Standardize features (z-score normalization)
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

# Replace NaN values (from zero-variance features) with 0
x_normalized = np.nan_to_num(x_normalized, nan=0.0)

data.x = torch.FloatTensor(x_normalized).to(device)

print(f"Normalized feature range: [{np.nanmin(x_normalized):.4f}, {np.nanmax(x_normalized):.4f}]")
print(f"Feature mean: {x_normalized.mean():.6f}, Std: {x_normalized.std():.6f}")
print(f"NaN values after fix: {np.isnan(x_normalized).sum()} / {x_normalized.size}")

# ===========================================================================
# CREATE STRATIFIED TRAIN/VAL/TEST SPLITS
# ===========================================================================
print("\nCreating stratified train/val/test splits (70/15/15)...")

# Get labeled nodes
labeled_mask = data.y >= 0  # Assumes unlabeled nodes have label -1
labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
labeled_y = data.y[labeled_mask].cpu().numpy()

print(f"Total labeled nodes: {len(labeled_indices)}")
print(f"Class distribution:")
for class_idx in np.unique(labeled_y):
    count = np.sum(labeled_y == class_idx)
    pct = 100 * count / len(labeled_y)
    print(f"  Class {class_idx}: {count:6d} ({pct:5.1f}%)")

# Split: 70% train, 30% temp (val+test)
train_indices, temp_indices = train_test_split(
    labeled_indices,
    test_size=0.3,
    random_state=TRAINING_CONFIG['random_seed'],
    stratify=labeled_y,
)

# Split temp into val and test: 50/50 of the 30% = 15% each
temp_y = data.y[temp_indices].cpu().numpy()
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    random_state=TRAINING_CONFIG['random_seed'],
    stratify=temp_y,
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_indices):6d} ({100*len(train_indices)/len(labeled_indices):5.1f}%)")
print(f"  Val:   {len(val_indices):6d} ({100*len(val_indices)/len(labeled_indices):5.1f}%)")
print(f"  Test:  {len(test_indices):6d} ({100*len(test_indices)/len(labeled_indices):5.1f}%)")

# Create masks
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

# Attach masks to data
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# ===========================================================================
# SAVE PROCESSED GRAPHS
# ===========================================================================
print("\nSaving processed graphs...")

# Save baseline graph with splits
torch.save(data, baseline_graph_path)
print(f"✓ Saved: {baseline_graph_path}")

# Quantum graph is identical except for features (will be expanded during training)
# Save the same for now (quantum features added during training)
torch.save(data, quantum_graph_path)
print(f"✓ Saved: {quantum_graph_path}")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("DATA LOADING COMPLETE")
print("=" * 70)
print(f"✓ Features normalized and standardized")
print(f"✓ Stratified splits created (70/15/15)")
print(f"✓ Data saved to artifacts/")
print(f"\nReady for training with:")
print(f"  - Baseline GAT: 02_train_baseline_gat.py")
print(f"  - Quantum GAT:  03_train_quantum_gat.py")
print("=" * 70)
