"""
Quantum-Inspired Graph Attention Network (QIGAT)
Research-grade model for fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .quantum_layers import QuantumFeatureMappingLayer, QuantumGATLayer


class QIGAT(nn.Module):
    """
    Quantum-Inspired Graph Attention Network
    
    Architecture:
    Input Features (enriched)
    → Quantum Feature Mapping
    → Linear Compression
    → Quantum-GAT Layer 1 (with residual)
    → Quantum-GAT Layer 2 (with residual)
    → MLP Classifier
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension for quantum mapping and GAT
        out_channels: Output dimension (number of classes)
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_layers: Number of GAT layers
    """
    
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, 
                 num_heads=8, dropout=0.5, num_layers=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Step 1: Quantum Feature Mapping
        self.quantum_map = QuantumFeatureMappingLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout=dropout
        )
        
        # Step 2: Linear compression to hidden dim
        quantum_out_dim = self._estimate_quantum_dim(in_channels)
        self.compress = nn.Linear(quantum_out_dim, hidden_channels)
        self.ln_compress = nn.LayerNorm(hidden_channels)
        
        # Step 3: Quantum-GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in = hidden_channels
            layer_out = hidden_channels // num_heads if num_heads > 0 else hidden_channels
            
            self.gat_layers.append(
                QuantumGATLayer(
                    in_channels=layer_in,
                    out_channels=layer_out,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.ln_layers.append(nn.LayerNorm(hidden_channels))
        
        # Step 4: MLP Classifier
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _estimate_quantum_dim(self, in_channels):
        """Estimate quantum mapping output dimension."""
        # cos, sin of each dimension + limited pairwise interactions
        k = min(int(np.sqrt(in_channels)), 8)
        interactions = k * (k - 1) // 2
        return in_channels * 2 + interactions
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            
        Returns:
            out: [num_nodes, out_channels]
        """
        # Step 1: Quantum mapping
        x_quantum = self.quantum_map(x)  # [N, hidden]
        
        # Step 2: Compression
        h = self.compress(x_quantum)
        h = self.ln_compress(h)
        h = F.elu(h)
        h = self.dropout_layer(h)
        
        # Step 3: Quantum-GAT layers with residual
        for i, (gat_layer, ln_layer) in enumerate(zip(self.gat_layers, self.ln_layers)):
            h_input = h
            h = gat_layer(h, edge_index)
            h = ln_layer(h)
            h = F.elu(h)
            h = self.dropout_layer(h)
            
            # Residual connection
            h = h + h_input
        
        # Step 4: MLP classifier
        out = self.mlp(h)
        
        return out


class BaselineGAT(nn.Module):
    """
    Standard GAT baseline for comparison
    """
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 num_heads=4, dropout=0.3, num_layers=2):
        super().__init__()
        
        from torch_geometric.nn import GATConv
        
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, 
                                  dropout=dropout, concat=True))
        self.lns.append(nn.LayerNorm(hidden_channels * num_heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels,
                                      heads=num_heads, dropout=dropout, concat=True))
            self.lns.append(nn.LayerNorm(hidden_channels * num_heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels,
                                  heads=1, dropout=dropout, concat=False))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            
        Returns:
            out: [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.elu(x)
            x = self.dropout_layer(x)
        
        x = self.convs[-1](x, edge_index)
        return x
