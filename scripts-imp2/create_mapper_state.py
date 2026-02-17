#!/usr/bin/env python
"""
Create quantum mapper state file by fitting on training data.
This ensures the test set uses the same preprocessing statistics as training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import json
import numpy as np
from quantum_feature_mapper import QuantumFeatureMapper

device = 'cpu'

print("=" * 80)
print("CREATING QUANTUM MAPPER STATE FROM TRAINING DATA")
print("=" * 80)

# Load the original graph (before quantum mapping)
print("\nLoading original graph...")
graph = torch.load('artifacts/elliptic_graph.pt', map_location=device, weights_only=False)
print(f"✓ Graph loaded: {graph.num_nodes} nodes, {graph.num_node_features} features")

# Create a fresh mapper
print("\nCreating quantum feature mapper...")
mapper = QuantumFeatureMapper(
    input_dim=graph.num_node_features,
    use_angle_encoding=True,
    use_interactions=False,
    use_fourier=False,
    max_output_dim=256
)

# Fit the mapper on TRAINING DATA ONLY (same as training script)
print("\nFitting mapper on training data...")
train_mask = graph.train_mask
x_train = graph.x[train_mask]

print(f"  Training data shape: {x_train.shape}")
print(f"  Training data range: [{x_train.min():.4f}, {x_train.max():.4f}]")

# Apply mapper forward pass to fit preprocessing statistics
x_transformed = mapper(x_train)
print(f"✓ Mapper fitted on training data")
print(f"  Transformed shape: {x_transformed.shape}")
print(f"  Feature mean (learned): {mapper._feature_mean[:5].cpu().numpy()}")
print(f"  Feature std (learned): {mapper._feature_std[:5].cpu().numpy()}")

# Save the mapper state
print("\nSaving mapper state...")
mapper_state = {
    'input_dim': mapper.input_dim,
    'use_angle_encoding': mapper.use_angle_encoding,
    'use_interactions': mapper.use_interactions,
    'use_fourier': mapper.use_fourier,
    'max_output_dim': mapper.max_output_dim,
    'output_dim': mapper.output_dim,
    'output_dim_capped': mapper.output_dim_capped,
    '_feature_mean': mapper._feature_mean.cpu().numpy().tolist(),
    '_feature_std': mapper._feature_std.cpu().numpy().tolist(),
    '_preprocessing_fitted': mapper._preprocessing_fitted,
}

import json
with open('artifacts/quantum_mapper_state.json', 'w') as f:
    json.dump(mapper_state, f, indent=2)

print(f"✓ Mapper state saved to: artifacts/quantum_mapper_state.json")
print(f"   - Input dim: {mapper_state['input_dim']}")
print(f"   - Output dim capped: {mapper_state['output_dim_capped']}")
print(f"   - Preprocessing fitted: {mapper_state['_preprocessing_fitted']}")

print("\n" + "=" * 80)
