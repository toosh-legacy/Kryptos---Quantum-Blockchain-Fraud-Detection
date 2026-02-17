#!/usr/bin/env python
"""
Test: Compare quantum mapper statistics between training and testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from quantum_feature_mapper import QuantumFeatureMapper

device = 'cpu'

print("=" * 80)
print("QUANTUM MAPPER STATISTICS TEST")
print("=" * 80)

# Load the saved trained model's configuration
print("\nLoading saved quantum model...")
quantum_model = torch.load('artifacts/quantum_gat_optimized_best.pt', map_location=device, weights_only=False)
print(f"✓ Loaded model state dict with {len(quantum_model)} keys")

# Create a fresh mapper
fresh_mapper = QuantumFeatureMapper(input_dim=182)
print("\n✓ Created fresh mapper (no preprocessing statistics)")
print(f"  _preprocessing_fitted: {fresh_mapper._preprocessing_fitted}")
print(f"  _feature_mean: {fresh_mapper._feature_mean}")
print(f"  _feature_std: {fresh_mapper._feature_std}")

# The mapper statistics are NOT saved in the model state! They're internal to the mapper.
# This is the core problem! The preprocesser statistics learned during training are lost.

print("\n⚠️  PROBLEM IDENTIFIED:")
print("  The QuantumFeatureMapper learns preprocessing statistics during training")
print("  But these statistics are NOT saved with the model!")
print("  When loading for testing, a fresh mapper has no knowledge of training stats")
print("  Result: Different preprocessing = different feature transformations = poor performance")

print("\n" + "=" * 80)
print("SOLUTION: Save preprocessing statistics separately and reload them")
print("=" * 80)
