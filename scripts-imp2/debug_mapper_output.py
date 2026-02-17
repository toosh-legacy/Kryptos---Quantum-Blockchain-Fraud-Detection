#!/usr/bin/env python
"""Debug the quantum mapper output dimensions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import json
from quantum_feature_mapper import QuantumFeatureMapper

device = 'cpu'

# Load graph
print("Loading graph...")
graph = torch.load('artifacts/elliptic_graph.pt', map_location=device, weights_only=False)
print(f"Original features: {graph.x.shape}")

# Load mapper state
print("\nLoading mapper state...")
with open('artifacts/quantum_mapper_state.json', 'r') as f:
    mapper_state = json.load(f)

print(f"Mapper state output_dim_capped: {mapper_state['output_dim_capped']}")

# Create mapper and restore state
print("\nCreating mapper...")
mapper = QuantumFeatureMapper(input_dim=graph.num_node_features)
mapper._feature_mean = torch.tensor(mapper_state['_feature_mean'], dtype=torch.float32)
mapper._feature_std = torch.tensor(mapper_state['_feature_std'], dtype=torch.float32)
mapper._preprocessing_fitted = True

print(f"Mapper output_dim_capped: {mapper.output_dim_capped}")
print(f"Mapper output_dim: {mapper.output_dim}")

# Test on small batch
print("\nTesting mapper on small batch...")
x_small = graph.x[:100]
x_mapped = mapper(x_small)
print(f"Input shape: {x_small.shape}")
print(f"Output shape: {x_mapped.shape}")

# Test on full graph
print("\nTesting mapper on full graph...")
x_full_mapped = mapper(graph.x)
print(f"Full input shape: {graph.x.shape}")
print(f"Full output shape: {x_full_mapped.shape}")

# Check if output matches expected
if x_full_mapped.shape[1] == 256:
    print("\n✓ Mapper output matches expected 256 dimensions")
else:
    print(f"\n✗ Mapper output {x_full_mapped.shape[1]} != 256 expected")
