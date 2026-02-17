"""
Feature Engineering Module
Computes structural features for fraud detection graphs
"""

import torch
import numpy as np
from torch_geometric.utils import degree, to_undirected
from sklearn.preprocessing import StandardScaler
import networkx as nx


def compute_structural_features(data, device='cpu'):
    """
    Compute structural features for each node.
    
    Features computed:
    - Degree (in + out)
    - Log-degree
    - PageRank
    - Local clustering coefficient
    - Approximate betweenness centrality
    - 2-hop neighborhood mean aggregation
    
    Args:
        data: PyTorch Geometric Data object
        device: Device to use
        
    Returns:
        x_enriched: Enriched feature matrix [num_nodes, original_dim + structural_dim]
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    x_original = data.x.cpu().numpy()
    
    print("Computing structural features...")
    
    # 1. DEGREE
    deg = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()
    deg_log = np.log1p(deg)
    
    # 2. PAGERANK (using NetworkX)
    print("  Computing PageRank...")
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.T.cpu().numpy()
    G.add_edges_from(edges)
    
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        pagerank_vec = np.array([pagerank.get(i, 0) for i in range(num_nodes)])
    except:
        pagerank_vec = np.zeros(num_nodes)
    
    # 3. LOCAL CLUSTERING COEFFICIENT
    print("  Computing clustering coefficients...")
    G_undirected = G.to_undirected()
    clustering = {}
    for node in range(num_nodes):
        try:
            clustering[node] = nx.clustering(G_undirected, node)
        except:
            clustering[node] = 0
    clustering_vec = np.array([clustering.get(i, 0) for i in range(num_nodes)])
    
    # 4. APPROXIMATE BETWEENNESS CENTRALITY (sample-based)
    print("  Computing betweenness centrality...")
    try:
        # Sample 100 random source nodes for approximation
        sample_nodes = np.random.choice(num_nodes, min(100, num_nodes), replace=False)
        betweenness = {}
        for node in range(num_nodes):
            betweenness[node] = 0
        
        for source in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, length in lengths.items():
                if source != target and length == 2:
                    for intermediate in G.neighbors(source):
                        if intermediate in G.neighbors(target):
                            if intermediate not in [source, target]:
                                betweenness[intermediate] += 1
        
        betweenness_vec = np.array([betweenness.get(i, 0) for i in range(num_nodes)])
        betweenness_vec = betweenness_vec / betweenness_vec.max() if betweenness_vec.max() > 0 else betweenness_vec
    except:
        betweenness_vec = np.zeros(num_nodes)
    
    # 5. 2-HOP NEIGHBORHOOD AGGREGATION
    print("  Computing 2-hop aggregations...")
    hop2_mean = np.zeros((num_nodes, x_original.shape[1]))
    hop2_std = np.zeros((num_nodes, x_original.shape[1]))
    
    for node in range(min(num_nodes, 10000)):  # Sample nodes for efficiency
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        hop2_neighbors = set()
        for n in neighbors:
            hop2_neighbors.update(G.successors(n))
            hop2_neighbors.update(G.predecessors(n))
        hop2_neighbors.discard(node)
        
        if len(hop2_neighbors) > 0:
            hop2_features = x_original[list(hop2_neighbors), :]
            hop2_mean[node] = hop2_features.mean(axis=0)
            hop2_std[node] = hop2_features.std(axis=0)
    
    # Normalize using first 1000 nodes
    hop2_mean = np.nan_to_num(hop2_mean, nan=0)
    hop2_std = np.nan_to_num(hop2_std, nan=0)
    
    # 6. CONCATENATE ALL STRUCTURAL FEATURES
    structural_features = np.column_stack([
        deg,
        deg_log,
        pagerank_vec,
        clustering_vec,
        betweenness_vec,
        hop2_mean,
        hop2_std
    ])
    
    print(f"Structural features shape: {structural_features.shape}")
    
    # 7. NORMALIZE
    print("Normalizing features...")
    
    # StandardScaler
    scaler = StandardScaler()
    x_enriched = np.concatenate([x_original, structural_features], axis=1)
    
    # Normalize per feature
    x_enriched = scaler.fit_transform(x_enriched)
    
    # L2 normalization per node
    x_norm = np.linalg.norm(x_enriched, axis=1, keepdims=True)
    x_norm = np.where(x_norm == 0, 1, x_norm)  # Avoid division by zero
    x_enriched = x_enriched / x_norm
    
    print(f"Enriched features shape: {x_enriched.shape}")
    print(f"Feature stats - Mean: {x_enriched.mean():.4f}, Std: {x_enriched.std():.4f}")
    
    # Convert to tensor
    x_enriched_tensor = torch.from_numpy(x_enriched).float().to(device)
    
    return x_enriched_tensor
