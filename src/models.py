"""
Neural network models for fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, List, Optional


class GAT(nn.Module):
    """
    Graph Attention Network for fraud detection.
    
    This model uses Graph Attention Networks (GAT) to learn node representations
    in a graph structure, applying attention mechanisms to weigh neighbor importance.
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Number of hidden features per attention head
        out_channels: Number of output classes (2 for binary classification)
        num_heads: Number of attention heads in GAT layers
        num_layers: Number of GAT layers
        dropout: Dropout probability
        
    Attributes:
        convs: ModuleList of GAT convolutional layers
        num_layers: Number of layers in the network
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        dropout: float = 0.3
    ):
        super(GAT, self).__init__()
        
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels * num_heads
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        )
        
        # Middle layers: hidden_channels * num_heads -> hidden_channels * num_heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads, 
                    hidden_channels, 
                    heads=num_heads, 
                    dropout=dropout
                )
            )
        
        # Last layer: hidden_channels * num_heads -> out_channels
        self.convs.append(
            GATConv(
                hidden_channels * num_heads, 
                out_channels, 
                heads=1, 
                concat=False, 
                dropout=dropout
            )
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through the GAT model.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            If return_attention_weights is False:
                Node embeddings of shape [num_nodes, out_channels]
            If return_attention_weights is True:
                Tuple of (node embeddings, list of (edge_index, attention_weights))
        """
        attention_weights = []
        
        # Process all layers except the last one
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)
            
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        if return_attention_weights:
            x, (edge_idx, attn) = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, attn))
            return x, attention_weights
        else:
            x = self.convs[-1](x, edge_index)
            return x
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
