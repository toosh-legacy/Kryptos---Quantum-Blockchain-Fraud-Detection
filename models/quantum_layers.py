"""
Quantum-Inspired Feature Mapping Module
Implements structured quantum phase encoding with controlled expressivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumFeatureMappingLayer(nn.Module):
    """
    Structured quantum-inspired feature mapping layer.
    
    Process:
    1. Linear projection: z = W_proj @ x
    2. Phase encoding: phi = π * tanh(z)
    3. Quantum features: [cos(phi), sin(phi), cos(phi_i - phi_j)]
    4. Layer normalization
    5. Dropout
    6. Learnable scaling
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension (after quantum mapping)
        dropout: Dropout rate
    """
    
    def __init__(self, in_channels, out_channels=256, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear projection for phase encoding
        self.proj = nn.Linear(in_channels, in_channels)
        
        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.ones(1))
        
        # Layer normalization
        self.ln = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (compress back if needed)
        quantum_dim_uncompressed = in_channels * 2 + (in_channels * (in_channels - 1) // 8)  # Limited interactions
        self.output_proj = nn.Linear(quantum_dim_uncompressed, out_channels)
        
        self.out_ln = nn.LayerNorm(out_channels)
        
        print(f"QuantumFeatureMappingLayer: {in_channels} -> {quantum_dim_uncompressed} -> {out_channels}")
    
    def forward(self, x):
        """
        Args:
            x: [num_nodes, in_channels]
            
        Returns:
            q: [num_nodes, out_channels]
        """
        num_nodes, _ = x.shape
        
        # Step 1: Linear projection and layer norm
        z = self.proj(x)  # [num_nodes, in_channels]
        z = self.ln(z)
        
        # Step 2: Phase encoding (π * tanh)
        phi = torch.pi * torch.tanh(z)  # [num_nodes, in_channels]
        
        # Step 3a: First-order quantum features (cos and sin)
        q_cos = torch.cos(phi)  # [num_nodes, in_channels]
        q_sin = torch.sin(phi)  # [num_nodes, in_channels]
        
        # Step 3b: Second-order interaction terms (limited to top-k by variance)
        # Select top features by variance
        z_var = z.var(dim=0)
        top_k = min(int(np.sqrt(self.in_channels)), 8)  # Limit interactions
        top_k_indices = torch.topk(z_var, top_k)[1]
        
        q_interactions = []
        for i in range(top_k):
            for j in range(i+1, top_k):
                idx_i = top_k_indices[i]
                idx_j = top_k_indices[j]
                # cos(phi_i - phi_j)
                q_ij = torch.cos(phi[:, idx_i] - phi[:, idx_j])
                q_interactions.append(q_ij)
        
        if q_interactions:
            q_interactions = torch.stack(q_interactions, dim=1)  # [num_nodes, num_interactions]
        else:
            q_interactions = torch.zeros(num_nodes, 1, device=x.device)
        
        # Step 4: Concatenate all quantum features
        q = torch.cat([q_cos, q_sin, q_interactions], dim=1)  # [num_nodes, total_quantum_dim]
        
        # Apply dropout and scaling
        q = self.dropout(q)
        q = q * self.alpha
        
        # Step 5: Project to output dimension
        q = self.output_proj(q)  # [num_nodes, out_channels]
        q = self.out_ln(q)
        
        return q


class QuantumGATLayer(nn.Module):
    """
    Quantum-aware GAT layer using PyTorch Geometric.
    Uses pre-built GATConv with quantum-inspired preprocessing.
    """
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.5, concat=True):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        
        # Use GATConv for reliable attention mechanism
        self.gat = GATConv(in_channels, out_channels, heads=heads, 
                           dropout=dropout, concat=concat)
        
        # Layer norm
        output_dim = out_channels * heads if concat else out_channels
        self.ln = nn.LayerNorm(output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        print(f"QuantumGATLayer: {in_channels} -> {heads}*{out_channels} (using GATConv)")
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            
        Returns:
            out: [num_nodes, out_channels*heads]
        """
        out = self.gat(x, edge_index)
        out = self.ln(out)
        out = F.elu(out)
        out = self.dropout_layer(out)
        return out
