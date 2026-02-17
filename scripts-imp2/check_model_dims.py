#!/usr/bin/env python
"""Check the actual input dimension expected by the saved model."""

import torch

print("Loading quantum model state dict...")
state_dict = torch.load('artifacts/quantum_gat_optimized_best.pt', weights_only=False)

# Find the first layer's input weight matrix
for key, tensor in state_dict.items():
    if 'lin' in key or 'weight' in key:
        print(f"{key}: {tensor.shape}")
        if len(tensor.shape) == 2 and tensor.shape[1] > 100:  # Likely first layer input
            print(f"  → Input dimension: {tensor.shape[1]}")
            break

# Also check convolution layers
for key, tensor in state_dict.items():
    if 'convs' in key and 'lin' in key:
        print(f"{key}: {tensor.shape}")
