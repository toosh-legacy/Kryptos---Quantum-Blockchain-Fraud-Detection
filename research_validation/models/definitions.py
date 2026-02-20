"""
MODEL DEFINITIONS FOR CONTROLLED EXPERIMENTAL VALIDATION

4 Parameter-Matched Models for Isolation Testing:

Model A: Baseline-128 (Fair Control)
Model B: Residual-Expansion-GAT (No Quantum)
Model C: MLP-Control (No Graphs)
Model D: QIGAT (Quantum-Inspired, Reference)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv


# ============================================================================
# MODEL A: BASELINE-128 (Fair Control)
# ============================================================================

class BaselineGAT128(nn.Module):
    """
    Standard Graph Attention Network with hidden=128
    
    This is the "fair control" baseline that matches QIGAT's capacity
    through 128 hidden dimensions instead of 64.
    
    Architecture:
    - Input: 182 features
    - GAT1:  182 → 128 hidden × 4 heads = 512-dim
    - GAT2:  512 → 128 hidden × 4 heads = 512-dim
    - Classifier: 512 → 2
    
    Parameters: ~95K (control for QIGAT's 1.15M)
    """
    
    def __init__(self, in_channels=182, hidden_channels=128, num_heads=4, 
                 dropout=0.5):
        super().__init__()
        
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                            dropout=dropout)
        self.gat1_out_dim = hidden_channels * num_heads
        
        self.norm1 = nn.LayerNorm(self.gat1_out_dim)
        self.act1 = nn.ELU()
        
        self.gat2 = GATConv(self.gat1_out_dim, hidden_channels, heads=num_heads, 
                            dropout=dropout)
        self.gat2_out_dim = hidden_channels * num_heads
        
        self.norm2 = nn.LayerNorm(self.gat2_out_dim)
        self.act2 = nn.ELU()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.gat2_out_dim, self.gat2_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gat2_out_dim, 2)
        )
    
    def forward(self, x, edge_index):
        h1 = self.gat1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = self.act1(h1)
        
        h2 = self.gat2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = self.act2(h2)
        
        out = self.classifier(h2)
        return out


# ============================================================================
# MODEL B: RESIDUAL-EXPANSION-GAT (No Quantum)
# ============================================================================

class ResidualExpansionGAT(nn.Module):
    """
    GAT with Residual Nonlinear Expansion (NO Quantum Block)
    
    Tests if improvement comes from nonlinear expansion + residual connection
    rather than quantum phase encoding specifically.
    
    Architecture:
    - Input: 182 features
    - GAT1: 182 → 512-dim
    - Expansion & Residual: 512 → 256, with residual add
    - GAT2: 256 → 512-dim
    - Classifier: 512 → 2
    
    Key: Uses same residual structure as QIGAT but with simple linear/ReLU
    instead of quantum phase block.
    
    Parameters: ~1.15M (matches QIGAT)
    """
    
    def __init__(self, in_channels=182, hidden_channels=128, num_heads=4, 
                 dropout=0.5):
        super().__init__()
        
        # Part A: First GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                            dropout=dropout)
        self.gat1_out_dim = hidden_channels * num_heads  # 512
        self.norm1 = nn.LayerNorm(self.gat1_out_dim)
        self.act1 = nn.ELU()
        
        # Part B: Residual Nonlinear Expansion (NO QUANTUM)
        # Simple nonlinear block instead of quantum
        self.expansion = nn.Sequential(
            nn.Linear(self.gat1_out_dim, self.gat1_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gat1_out_dim, 256)
        )
        
        # Residual projection
        self.residual_projection = nn.Linear(self.gat1_out_dim, 256)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        
        # Part C: Second GAT
        self.gat2 = GATConv(256, hidden_channels, heads=num_heads, 
                            dropout=dropout)
        self.gat2_out_dim = hidden_channels * num_heads  # 512
        self.norm2 = nn.LayerNorm(self.gat2_out_dim)
        self.act2 = nn.ELU()
        
        # Part D: Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.gat2_out_dim, self.gat2_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gat2_out_dim, 2)
        )
    
    def forward(self, x, edge_index):
        # Part A: First GAT
        h1 = self.gat1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = self.act1(h1)
        
        # Part B: Residual expansion (NO quantum)
        h_expanded = self.expansion(h1)  # 512 → 256
        h1_projected = self.residual_projection(h1)
        h_combined = h1_projected + self.residual_scale * h_expanded
        
        # Part C: Second GAT
        h2 = self.gat2(h_combined, edge_index)
        h2 = self.norm2(h2)
        h2 = self.act2(h2)
        
        # Part D: Classifier
        out = self.classifier(h2)
        return out


# ============================================================================
# MODEL C: MLP-CONTROL (No Graph Structure)
# ============================================================================

class MLPControl(nn.Module):
    """
    Multi-Layer Perceptron Control (NO Graph Information)
    
    Tests if improvement requires graph structure or is achievable
    through deep nonlinear embeddings alone.
    
    Architecture:
    - Input: 182 features
    - MLP: 3-4 hidden layers
    - Total parameters: ~1.15M (match QIGAT)
    
    Design: Equivalent capacity to QIGAT but processes features
    independently without leveraging transaction network structure.
    
    Parameters: ~1.15M (matches QIGAT)
    """
    
    def __init__(self, in_features=182, hidden_dim=512, num_layers=3, 
                 dropout=0.5):
        super().__init__()
        
        layers = []
        in_dim = in_features
        
        # Hidden layers with increasing then decreasing dimension
        hidden_dims = [512, 1024, 512, 256][:num_layers]
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        # Final classifier
        layers.append(nn.Linear(in_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(256, 2))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, edge_index=None):
        # Note: edge_index ignored - MLP doesn't use graph structure
        return self.mlp(x)


# ============================================================================
# MODEL D: QIGAT (Quantum-Inspired GAT)
# ============================================================================

class QuantumPhaseBlock(nn.Module):
    """
    Quantum Phase Encoding Block
    
    Learned phase projection: φ = π * tanh(Wx)
    Phase features: cos(φ), sin(φ)
    """
    
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.W_phase = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.W_phase.weight)
        
        if input_dim * 2 != output_dim:
            self.compress = nn.Linear(input_dim * 2, output_dim)
        else:
            self.compress = None
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, h):
        z = self.W_phase(h)
        phi = np.pi * torch.tanh(z)
        q_cos = torch.cos(phi)
        q_sin = torch.sin(phi)
        h_quantum = torch.cat([q_cos, q_sin], dim=1)
        
        if self.compress is not None:
            h_quantum = self.compress(h_quantum)
        
        h_quantum = self.norm(h_quantum)
        h_quantum = self.dropout(h_quantum)
        
        return h_quantum


class QIGAT(nn.Module):
    """
    Quantum-Inspired Graph Attention Network (QIGAT)
    
    Architecture:
    - Input: 182 features
    - GAT1: 182 → 512-dim
    - Quantum: 512 → 256-dim (learned phase encoding)
    - Residual: α scaling for quantum contribution
    - GAT2: 256 → 512-dim
    - Classifier: 512 → 2
    
    Parameters: ~1.15M
    """
    
    def __init__(self, in_features=182, hidden_dim=128, num_heads=4, 
                 dropout=0.5):
        super().__init__()
        
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads, 
                            dropout=dropout)
        self.gat1_out_dim = hidden_dim * num_heads
        self.norm1 = nn.LayerNorm(self.gat1_out_dim)
        self.activation1 = nn.ELU()
        
        self.quantum_block = QuantumPhaseBlock(self.gat1_out_dim, output_dim=256)
        self.quantum_residual_scale = nn.Parameter(torch.tensor(0.5))
        self.residual_projection = nn.Linear(self.gat1_out_dim, 256)
        
        self.gat2 = GATConv(256, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2_out_dim = hidden_dim * num_heads
        self.norm2 = nn.LayerNorm(self.gat2_out_dim)
        self.activation2 = nn.ELU()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.gat2_out_dim, self.gat2_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gat2_out_dim, 2)
        )
    
    def forward(self, x, edge_index):
        h1 = self.gat1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = self.activation1(h1)
        
        h_quantum = self.quantum_block(h1)
        h1_projected = self.residual_projection(h1)
        h_combined = h1_projected + self.quantum_residual_scale * h_quantum
        
        h2 = self.gat2(h_combined, edge_index)
        h2 = self.norm2(h2)
        h2 = self.activation2(h2)
        
        out = self.classifier(h2)
        
        return out


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    'baseline_128': BaselineGAT128,
    'residual_expansion': ResidualExpansionGAT,
    'mlp_control': MLPControl,
    'qigat': QIGAT
}

MODEL_NAMES = {
    'baseline_128': 'Baseline GAT-128',
    'residual_expansion': 'Residual-Expansion (No Quantum)',
    'mlp_control': 'MLP Control (No Graph)',
    'qigat': 'QIGAT (Quantum-Inspired)'
}

def get_model(model_type, **kwargs):
    """Factory function to instantiate models"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_type} not found. Options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
