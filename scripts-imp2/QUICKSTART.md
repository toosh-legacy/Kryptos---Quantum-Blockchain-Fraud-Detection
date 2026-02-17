# QUICK START GUIDE: scripts-imp2

## Get Started in 3 Steps

### Step 0: Prerequisites (One-time setup)

Make sure you have the base graph from scripts-imp1:

```bash
cd Kryptos---Quantum-Blockchain-Fraud-Detection
python scripts-imp1/02_data_graph.py  # Creates artifacts/elliptic_graph.pt
```

### Step 1: Run the Complete Pipeline

Execute all steps automatically end-to-end:

```bash
python scripts-imp2/run_pipeline.py
```

**What happens:**
- ✓ Loads graph and creates train/val/test splits
- ✓ Trains baseline GAT (saves metrics)
- ✓ Trains quantum-enhanced GAT (saves metrics)
- ✓ Compares both models (generates report)

**Time**: ~15-35 minutes depending on GPU/CPU

**Output files in `artifacts/`:**
```
baseline_gat_best.pt               # Best baseline model weights
baseline_gat_metrics.json          # Baseline test metrics
baseline_gat_training_curves.png   # Training visualization

quantum_gat_best.pt                # Best quantum model weights
quantum_gat_metrics.json           # Quantum test metrics
quantum_gat_training_curves.png    # Training visualization

model_comparison.png               # Side-by-side comparison
model_comparison_report.json       # Detailed analysis
```

---

### Step 2 (Optional): Run Steps Individually

If you want to run each step separately:

```bash
# Step 1: Data loading (1-2 min)
python scripts-imp2/01_load_data.py

# Step 2: Train baseline (5-15 min)
python scripts-imp2/02_train_baseline_gat.py

# Step 3: Train quantum (5-15 min)
python scripts-imp2/03_train_quantum_gat.py

# Step 4: Compare results (1 min)
python scripts-imp2/04_compare_metrics.py
```

---

## What You Get

### 📊 Performance Metrics

Both models track:
- Accuracy, Precision, Recall
- **Macro F1** (best for imbalanced data)
- Micro F1
- ROC-AUC (one-vs-rest)
- Per-class breakdown
- Confusion matrix

### 📈 Training Curves

Visualizations showing:
- Loss curves (train + validation)
- F1 score trajectories
- Accuracy curves
- Improvement over epochs

### 🔍 Comparison Report

Side-by-side analysis:
- Test metrics comparison
- Training efficiency
- Model capacity comparison
- Improvement percentages
- Which model wins on each metric

---

## Key Results to Check

After pipeline completes, check:

1. **Quantum improvement**: Open `artifacts/model_comparison_report.json`
   - Look for `average_improvement_percent`
   - More positive = quantum is better

2. **Best performing metric**: In same report
   - Check `test_metrics_comparison` section
   - See which metrics improved the most

3. **Visual comparison**: Open `artifacts/model_comparison.png`
   - Bar chart comparing test metrics
   - Training curves (loss, F1, accuracy)
   - Quantum improvement percentages

---

## Understanding the Quantum Feature Mapper

The mapper expands each feature into multiple quantum-inspired dimensions:

```
Input:   16 features
         ↓
Preprocessing: log1p + z-score normalization
         ↓
Angle Encoding:        cos(πx), sin(πx)       (+32 dims)
Pairwise Interactions: x_i * x_j              (+136 dims)
Fourier Features:      √(2/D)*cos(Wx+b)      (+16 dims)
         ↓
Total Before Cap:      ~200 dimensions
         ↓
Capped At:             128 dimensions (final output)
```

**Why it helps:**
- Captures nonlinear patterns in features
- Enriches representation for fraud detection
- Learnable Fourier features adapt during training

---

## Configuration

### To Adjust Model Hyperparameters

Edit `src/config.py`:

```python
MODEL_CONFIG = {
    "hidden_channels": 64,      # Baseline: 64, Quantum: 128
    "num_heads": 4,             # Baseline: 4, Quantum: 6
    "num_layers": 2,            # Baseline: 2, Quantum: 3
    "dropout": 0.3,             # Baseline: 0.3, Quantum: 0.4
    "out_channels": 2,
}

TRAINING_CONFIG = {
    "epochs": 200,              # Max training epochs
    "learning_rate": 0.001,     # Adam LR
    "patience": 20,             # Early stopping patience
    "weight_decay": 5e-4,       # L2 regularization
    ...
}
```

### To Adjust Quantum Feature Mapper

Edit `scripts-imp2/03_train_quantum_gat.py` (around line 50):

```python
quantum_mapper = QuantumFeatureMapper(
    input_dim=original_features,
    use_angle_encoding=True,        # Toggle on/off
    use_interactions=True,          # Toggle on/off
    use_fourier=True,               # Toggle on/off
    max_output_dim=128,             # Change cap (lower = faster)
    random_seed=42,
)
```

---

## Common Issues & Solutions

### Issue: "Graph does not have train/val/test masks"
**Solution**: Run `01_load_data.py` first

### Issue: CUDA out of memory
**Solution**: 
- Reduce `max_output_dim` to 64 in quantum mapper
- Reduce `hidden_channels` in config
- Use CPU (slower but works)

### Issue: Poor model performance
**Solution**:
- Verify data preprocessing (check output of 01_load_data.py)
- Check class weights computation (printed during training)
- Try different learning rate (e.g., 0.0005 to 0.005)
- Increase patience for early stopping

### Issue: Large/small improvements
**Solution**:
- Quantum features work best with complex patterns
- Try adjusting which components to use (angle, interactions, fourier)
- Tune model capacity (hidden_channels, num_heads, num_layers)

---

## Next Steps

1. ✅ Run the complete pipeline
2. 📊 Analyze the comparison report
3. 🔧 Fine-tune hyperparameters if needed
4. 📈 Deploy best model for inference
5. 🎨 Add explanability features (attention visualization, etc.)

---

## File Reference

| File | Purpose |
|------|---------|
| `quantum_feature_mapper.py` | Quantum feature expansion module |
| `01_load_data.py` | Load & preprocess graph + create splits |
| `02_train_baseline_gat.py` | Train baseline GAT model |
| `03_train_quantum_gat.py` | Train quantum-enhanced GAT |
| `04_compare_metrics.py` | Compare baseline vs quantum |
| `run_pipeline.py` | Execute all steps automatically |
| `README.md` | Detailed documentation |

---

## Questions?

Check:
- `README.md` - Detailed documentation
- Training output - Printed metrics during execution
- Artifact JSON files - Detailed metrics breakdowns
- Generated PNG plots - Visual comparisons

---

**Ready? Run this command:**

```bash
python scripts-imp2/run_pipeline.py
```

**Let's see those quantum improvements!** 🚀
