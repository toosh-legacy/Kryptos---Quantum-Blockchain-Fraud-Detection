# Kryptos Project: Quantum-Inspired GNNs for Blockchain Fraud Detection

## Overview
kryptos is a research project that combines Graph Attention Networks (GAT) with quantum-inspired feature mapping to detect fraud in blockchain transaction graphs. It also includes explainability using LLMs (e.g., GPT-4) for forensic narratives.

---

## Key Concepts
- **Graph Neural Networks (GAT):** Learns node representations using attention over neighbors.
- **Quantum-Inspired Features:** Uses Random Fourier Features to expand input features, mimicking quantum feature maps.
- **Explainability:** Extracts attention weights and uses LLMs to generate human-readable explanations for fraud predictions.

---

## Project Structure
- `notebooks/`: Step-by-step Jupyter notebooks for data prep, training, evaluation, quantum features, and explanations.
- `src/`: Core Python modules for models, quantum features, training utilities, config, and helpers.
- `data/`: Input CSVs (features, edges, classes) in Elliptic Bitcoin Dataset format.
- `artifacts/`: Saved models, metrics, and explanations.
- `figures/`: Generated plots and visualizations.

---

## Workflow (Recommended Order)
1. **01_setup.ipynb**: Environment check, directory setup, dataset verification
2. **02_data_graph.ipynb**: Load and process data, build graph, visualize
3. **03_train_gat_baseline.ipynb**: Train baseline GAT model
4. **04_eval_baseline.ipynb**: Evaluate baseline, analyze results
5. **05_quantum_feature_map.ipynb**: Apply quantum-inspired feature mapping
6. **06_train_gat_quantum.ipynb**: Train GAT on quantum features
7. **07_eval_quantum.ipynb**: Compare models, analyze improvements
8. **08_explain_llm.ipynb**: Generate LLM-based explanations (requires OpenAI API)

---

## Main Components
### 1. Models (`src/models.py`)
- `GAT`: 2-layer Graph Attention Network with configurable heads, hidden size, dropout
- Used for both baseline and quantum feature experiments

### 2. Quantum Features (`src/quantum_features.py`)
- `QuantumFeatureMap`: Expands features using Random Fourier Features
- Doubles feature dimension, captures non-linearities

### 3. Training Utilities (`src/train_utils.py`)
- Functions for training, validation, metrics, and evaluation
- Early stopping, F1/AUC/recall, confusion matrix, ROC

### 4. Config (`src/config.py`)
- Centralized paths, dataset, and model hyperparameters

### 5. Utils (`src/utils.py`)
- Device selection, random seed setting, file helpers

---

## Data Format
- **Features:** `txs_features.csv` (transaction features)
- **Edges:** `txs_edgelist.csv` (graph structure)
- **Classes:** `txs_classes.csv` (labels: 1=illicit, 2=licit, 3=unknown)

---

## Requirements
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
- GPU recommended for training (CUDA), but CPU supported

---

## Troubleshooting
- **CUDA OOM:** Lower batch size or use CPU
- **Import errors:** Upgrade torch/torch-geometric
- **OpenAI API:** Add `.env` with your key, or skip LLM cells

---

## Results (Typical)
- **Baseline GAT:** F1 ~0.55-0.65, AUC ~0.75-0.85
- **Quantum GAT:** F1 ~0.60-0.70, AUC ~0.80-0.90
- **Key finding:** Quantum features improve fraud detection

---

## Citation
```
@misc{aegis2024,
  title={Aegis: Quantum-Inspired and Explainable Graph Neural Networks for Blockchain Fraud Detection},
  author={Your Name},
  year={2024}
}
```

---

## Contact
For questions, open a GitHub issue or contact the project maintainer.
