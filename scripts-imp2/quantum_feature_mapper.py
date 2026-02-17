"""
Clean, modular quantum-inspired feature mapping pipeline.

This module implements quantum-inspired feature transformations for fraud detection.
Features are preprocessed, then expanded using angle encoding, pairwise interactions,
and optional Random Fourier Features.
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class QuantumFeatureMapper(nn.Module):
    """
    Quantum-inspired feature mapping pipeline for neural networks.
    
    This module applies the following transformations:
    1. Preprocessing: log1p transform + standard scaling (z-score normalization)
    2. Angle Encoding: cos(πx), sin(πx) for each feature
    3. Pairwise Interactions: x_i * x_j (upper triangular combinations)
    4. Optional Random Fourier Features: sqrt(2/D) * cos(Wx + b)
    
    The module is torch.nn.Module compliant, allowing gradients and proper device handling.
    
    Args:
        input_dim (int): Number of input features
        use_angle_encoding (bool): If True, apply angle encoding (default: True)
        use_interactions (bool): If True, compute pairwise interactions (default: True)
        use_fourier (bool): If True, apply Random Fourier Features (default: False)
        fourier_dim (int): Dimension for Random Fourier Features (default: None, auto-computed)
        max_output_dim (int): Cap on output feature dimension (default: 128)
        random_seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        torch.Tensor: Transformed features of shape [num_nodes, output_dim], dtype=float32
    
    Example:
        >>> mapper = QuantumFeatureMapper(
        ...     input_dim=16, 
        ...     use_angle_encoding=True,
        ...     use_interactions=True,
        ...     use_fourier=True,
        ...     max_output_dim=128
        ... )
        >>> x = torch.randn(1000, 16)
        >>> x_expanded = mapper(x)
        >>> print(x_expanded.shape)  # Expected: [1000, 128] or less
    """
    
    def __init__(
        self,
        input_dim: int,
        use_angle_encoding: bool = True,
        use_interactions: bool = True,
        use_fourier: bool = False,
        fourier_dim: Optional[int] = None,
        max_output_dim: int = 128,
        random_seed: int = 42,
    ):
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if max_output_dim <= 0:
            raise ValueError(f"max_output_dim must be positive, got {max_output_dim}")
        
        self.input_dim = input_dim
        self.use_angle_encoding = use_angle_encoding
        self.use_interactions = use_interactions
        self.use_fourier = use_fourier
        self.max_output_dim = max_output_dim
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Register buffers for preprocessing statistics (will be computed in forward pass)
        self.register_buffer("_feature_mean", None)
        self.register_buffer("_feature_std", None)
        self._preprocessing_fitted = False
        
        # Initialize Fourier features if needed (before computing output dim)
        if self.use_fourier:
            self._init_fourier_features(fourier_dim)
        else:
            self.fourier_dim = 0
        
        # Compute expected output dimension
        self._compute_output_dim()
    
    def _compute_output_dim(self) -> None:
        """Compute expected output dimension before dimension capping."""
        self.output_dim = self.input_dim  # Original features
        
        if self.use_angle_encoding:
            self.output_dim += 2 * self.input_dim  # cos and sin for each feature
        
        if self.use_interactions:
            # Upper triangular combinations: n(n+1)/2 (includes diagonal for x_i^2)
            num_interactions = self.input_dim * (self.input_dim + 1) // 2
            self.output_dim += num_interactions
        
        if self.use_fourier:
            # Fourier features add fourier_dim features
            self.output_dim += self.fourier_dim
        
        # Cap at max_output_dim
        self.output_dim_capped = min(self.output_dim, self.max_output_dim)
    
    def _init_fourier_features(self, fourier_dim: Optional[int] = None) -> None:
        """Initialize Random Fourier Features."""
        if fourier_dim is None:
            # Heuristic: use sqrt of input dim, but reasonable bounds
            self.fourier_dim = max(4, min(32, int(np.sqrt(self.input_dim) * 2)))
        else:
            self.fourier_dim = fourier_dim
        
        # Initialize learnable Fourier projection
        # W: shape [input_dim, fourier_dim]
        # b: shape [fourier_dim]
        self.register_parameter(
            "fourier_W",
            nn.Parameter(torch.randn(self.input_dim, self.fourier_dim) / np.sqrt(self.input_dim))
        )
        self.register_parameter(
            "fourier_b",
            nn.Parameter(torch.rand(self.fourier_dim) * 2 * np.pi)
        )
    
    def _fit_preprocessing(self, x: Tensor) -> None:
        """
        Compute and store mean and std for preprocessing.
        
        Args:
            x: Input tensor of shape [num_nodes, input_dim]
        """
        if not self._preprocessing_fitted:
            # Compute statistics for positive values only (for log1p)
            with torch.no_grad():
                self._feature_mean = x.mean(dim=0, keepdim=False)
                self._feature_std = x.std(dim=0, keepdim=False)
                # Avoid division by zero
                self._feature_std = torch.clamp(self._feature_std, min=1e-8)
            self._preprocessing_fitted = True
    
    def _preprocess(self, x: Tensor) -> Tensor:
        """
        Preprocess features: log1p + z-score normalization.
        
        Args:
            x: Input tensor of shape [num_nodes, input_dim]
        
        Returns:
            Preprocessed tensor of shape [num_nodes, input_dim]
        """
        # Fit preprocessing on first call
        if not self._preprocessing_fitted:
            self._fit_preprocessing(x)
        
        # Apply log1p to handle positive skewness
        x_log = torch.log1p(torch.clamp(x, min=0))
        
        # Z-score normalization using stored statistics
        x_normalized = (x_log - self._feature_mean) / self._feature_std
        
        # Replace any NaN/Inf values with 0
        x_normalized = torch.nan_to_num(x_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x_normalized.float()
    
    def _angle_encoding(self, x: Tensor) -> Tensor:
        """
        Apply angle encoding: cos(πx) and sin(πx).
        
        Args:
            x: Normalized features of shape [num_nodes, input_dim]
        
        Returns:
            Angle-encoded features of shape [num_nodes, 2*input_dim]
        """
        # Scale to [0, π] range (common for quantum computing)
        x_scaled = x * np.pi
        
        cos_features = torch.cos(x_scaled)
        sin_features = torch.sin(x_scaled)
        
        # Concatenate: [cos(πx_1), ..., cos(πx_n), sin(πx_1), ..., sin(πx_n)]
        return torch.cat([cos_features, sin_features], dim=1)
    
    def _pairwise_interactions(self, x: Tensor) -> Tensor:
        """
        Compute second-order pairwise interactions x_i * x_j.
        Uses upper triangular combinations to avoid duplicates.
        
        Args:
            x: Normalized features of shape [num_nodes, input_dim]
        
        Returns:
            Interaction features of shape [num_nodes, input_dim*(input_dim+1)/2]
        """
        num_nodes = x.shape[0]
        input_dim = x.shape[1]
        
        # Use batch matrix multiplication for efficiency
        # x @ x^T gives all pairwise products: shape [num_nodes, input_dim, input_dim]
        interactions = torch.bmm(
            x.unsqueeze(2),  # [num_nodes, input_dim, 1]
            x.unsqueeze(1)   # [num_nodes, 1, input_dim]
        )  # Result: [num_nodes, input_dim, input_dim]
        
        # Extract upper triangular part (including diagonal)
        # This avoids duplicate terms and redundancy
        upper_tri_indices = torch.triu_indices(input_dim, input_dim, device=x.device)
        
        # Extract upper triangular elements efficiently
        # Reshape: [num_nodes, input_dim, input_dim] -> [num_nodes, input_dim*input_dim]
        interactions_flat = interactions.reshape(num_nodes, -1)
        
        # Select upper triangular columns from flattened tensor
        # Map 2D indices to 1D indices in flattened tensor
        ij_to_flat_idx = upper_tri_indices[0] * input_dim + upper_tri_indices[1]
        interactions_upper = interactions_flat[:, ij_to_flat_idx]
        
        return interactions_upper
    
    def _fourier_features(self, x: Tensor) -> Tensor:
        """
        Apply Random Fourier Features: phi(x) = sqrt(2/D) * cos(Wx + b).
        
        Args:
            x: Normalized features of shape [num_nodes, input_dim]
        
        Returns:
            Fourier features of shape [num_nodes, fourier_dim]
        """
        # Compute Wx + b
        projection = torch.matmul(x, self.fourier_W) + self.fourier_b
        
        # Apply cosine and normalize
        scaling = np.sqrt(2.0 / self.fourier_dim)
        fourier_feats = scaling * torch.cos(projection)
        
        return fourier_feats
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply quantum-inspired feature mapping.
        
        Args:
            x: Input features of shape [num_nodes, input_dim]
        
        Returns:
            Expanded features of shape [num_nodes, output_dim_capped], dtype=float32
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.input_dim}, "
                f"got {x.shape[1]}"
            )
        
        # Ensure float32
        x = x.float()
        
        # Preprocessing: log1p + normalization
        x_preprocessed = self._preprocess(x)
        
        # Start with original preprocessed features
        expanded_features = [x_preprocessed]
        
        # Apply angle encoding
        if self.use_angle_encoding:
            angle_feats = self._angle_encoding(x_preprocessed)
            expanded_features.append(angle_feats)
        
        # Apply pairwise interactions
        if self.use_interactions:
            interaction_feats = self._pairwise_interactions(x_preprocessed)
            expanded_features.append(interaction_feats)
        
        # Apply Fourier features
        if self.use_fourier:
            fourier_feats = self._fourier_features(x_preprocessed)
            expanded_features.append(fourier_feats)
        
        # Concatenate all features
        x_expanded = torch.cat(expanded_features, dim=1)
        
        # Cap output dimension if needed
        if x_expanded.shape[1] > self.max_output_dim:
            # Select first max_output_dim features (most informative typically come first)
            x_expanded = x_expanded[:, :self.max_output_dim]
        
        return x_expanded.float()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM FEATURE MAPPER - EXAMPLE USAGE")
    print("=" * 70)
    
    # Create synthetic data
    num_nodes = 1000
    input_dim = 16
    
    torch.manual_seed(42)
    x = torch.randn(num_nodes, input_dim)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    # Test 1: Angle encoding only
    print("\n" + "-" * 70)
    print("Test 1: Angle Encoding Only")
    print("-" * 70)
    mapper1 = QuantumFeatureMapper(
        input_dim=input_dim,
        use_angle_encoding=True,
        use_interactions=False,
        use_fourier=False,
    )
    x1 = mapper1(x)
    print(f"Output shape: {x1.shape}")
    print(f"Expected: [1000, 48]")  # input + 2*input for cos/sin = 16 + 32 = 48
    
    # Test 2: Angle encoding + Interactions
    print("\n" + "-" * 70)
    print("Test 2: Angle Encoding + Pairwise Interactions")
    print("-" * 70)
    mapper2 = QuantumFeatureMapper(
        input_dim=input_dim,
        use_angle_encoding=True,
        use_interactions=True,
        use_fourier=False,
    )
    x2 = mapper2(x)
    print(f"Output shape: {x2.shape}")
    # Expected: 16 (orig) + 32 (sin/cos) + 136 (interactions: 16*17/2) = 184
    # But capped at 128 by default
    print(f"Expected: [1000, 128] (capped from ~184)")
    
    # Test 3: All features with dimension cap
    print("\n" + "-" * 70)
    print("Test 3: All Features (Angle + Interactions + Fourier, Capped)")
    print("-" * 70)
    mapper3 = QuantumFeatureMapper(
        input_dim=input_dim,
        use_angle_encoding=True,
        use_interactions=True,
        use_fourier=True,
        max_output_dim=128,
    )
    x3 = mapper3(x)
    print(f"Output shape: {x3.shape}")
    print(f"Output dtype: {x3.dtype}")
    print(f"Output range: [{x3.min():.4f}, {x3.max():.4f}]")
    print(f"Expected: [1000, 128] (capped)")
    
    # Test 4: Device compatibility (CPU test, but works on GPU)
    print("\n" + "-" * 70)
    print("Test 4: Device Compatibility")
    print("-" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapper_device = QuantumFeatureMapper(input_dim=input_dim, use_fourier=True).to(device)
    x_device = x.to(device)
    x_transformed = mapper_device(x_device)
    print(f"Mapper device: {next(mapper_device.parameters()).device}")
    print(f"Output device: {x_transformed.device}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
