"""
Configuration module for Aegis project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "features_file": "txs_features.csv",
    "edgelist_file": "txs_edgelist.csv",
    "classes_file": "txs_classes.csv",
    "id_column": "txId",
    "timestep_column": "time_step",
    "label_column": "class",
    "edge_source_column": "txId1",
    "edge_target_column": "txId2",
    "illicit_label": 1,
    "licit_label": 2,
    "unknown_label": 3,
}

# Model hyperparameters
MODEL_CONFIG = {
    "hidden_channels": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.3,
    "out_channels": 2,
}

# Quantum feature mapping
QUANTUM_CONFIG = {
    "expansion_factor": 2,
    "fourier_features": True,
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "patience": 20,
    "val_split": 0.2,
    "test_split": 0.2,
    "random_seed": 42,
    "train_test_split": 0.2,
    "train_val_split": 0.25,
}

# Evaluation configuration
EVAL_CONFIG = {
    "temporal_split_ratio": 0.7,
}

# Explainability configuration
EXPLAIN_CONFIG = {
    "num_explanations": 10,
    "num_top_fraud": 10,
    "k_neighbors": 5,
    "use_llm": True,
    "llm_model": "gpt-4o-mini",
}

# Device (requires torch to be imported in notebooks)
# Use: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_TYPE = "cuda"  # Preferred device type

# File names for artifacts
ARTIFACT_FILES = {
    "baseline_graph": "elliptic_graph.pt",
    "quantum_graph": "elliptic_graph_quantum.pt",
    "baseline_model": "gat_baseline.pt",
    "quantum_model": "gat_quantum.pt",
    "baseline_metrics": "gat_baseline_metrics.json",
    "quantum_metrics": "gat_quantum_metrics.json",
    "top_fraud_indices": "top_fraud_indices.npy",
    "fraud_explanations": "fraud_explanations.json",
}

# Figure names
FIGURE_FILES = {
    "data_distribution": "data_distribution.png",
    "baseline_confusion": "baseline_confusion_matrix.png",
    "baseline_roc": "baseline_roc_curve.png",
    "quantum_features": "quantum_feature_distribution.png",
    "quantum_confusion": "quantum_confusion_matrix.png",
    "comparison": "baseline_vs_quantum_comparison.png",
    "roc_comparison": "roc_comparison.png",
}