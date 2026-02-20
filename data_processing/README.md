# Data Processing

Prepare Bitcoin transaction graph data for model training.

## Files

- **create_graph.ipynb** / **create_graph.py**: Construct the PyTorch Geometric graph from raw CSV files
- **sanity_checks.ipynb** / **sanity_checks.py**: Verify data splits, preprocessing, and class balance

## Data Requirements

Place these files in `data/`:
- `txs_features.csv` - Node features (181 dimensions)
- `txs_classes.csv` - Node labels (fraud/non-fraud)
- `txs_edgelist.csv` - Transaction edges

## Quick Start

```bash
# Create graph (one-time)
python create_graph.py

# Verify preprocessing
python sanity_checks.py
```

## Output

Saves to `artifacts/`:
- `elliptic_graph.pt` - PyTorch Geometric Data object
- `parameter_counts.json` - Model capacity analysis

## Key Details

- **Nodes**: ~203k Bitcoin addresses
- **Edges**: ~230k transactions (undirected, with self-loops)
- **Features**: 182 per node (temporal + structural)
- **Labels**: Binary (fraud/non-fraud) + unlabeled
- **Split**: 70% train / 15% val / 15% test (stratified)

## Next Steps

After data preparation, run training:
1. `baselines/baseline_gat_training.ipynb` - Baseline GAT
2. `quantum_models/quantum_gat_training.ipynb` - Quantum GAT
3. `validation/multiseed_validation.ipynb` - Multi-seed validation
