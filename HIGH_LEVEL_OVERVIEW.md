# How Each Component Works (High-Level Overview)

Quick explanation of what each notebook and module does without getting into code details.

## 📊 Pipeline Overview

Think of the project as a **4-stage pipeline**:

```
Stage 1         Stage 2          Stage 3           Stage 4
┌──────────┐   ┌──────────┐    ┌─────────────┐   ┌──────────────┐
│ Build    │ → │ Compare  │ →  │ Train       │ → │ Validate     │
│ Graph    │   │ Baseline │    │ Quantum     │   │ Results      │
└──────────┘   └──────────┘    └─────────────┘   └──────────────┘
```

---

## Stage 1: Data Preparation

### What: `data_processing/`

Transform raw Bitcoin data into a format models can learn from.

### Key Notebooks

#### `create_graph.ipynb` - Build the Graph
**What it does:**
- Reads 3 CSV files: features, labels, transactions
- Creates one big graph where:
  - **Nodes** = Bitcoin addresses (~203k)
  - **Edges** = Transactions between addresses (~230k)
  - **Node features** = 182 numbers describing each address
- Normalizes (standardizes) the features
- Saves the graph to `artifacts/elliptic_graph.pt` for reuse

**Why it matters:**
- All subsequent models use this cached graph
- Ensures everyone trains on exactly the same data
- Must run once before everything else

**Time**: 2-5 minutes

#### `sanity_checks.ipynb` - Verify Everything is Correct
**What it does:**
- Checks that data splits are consistent (70/15/15)
- Counts how many parameters each model has
- Verifies feature normalization worked
- Analyzes class imbalance (fraud vs non-fraud ratio)

**Why it matters:**
- Catches data problems early
- Ensures fair comparison (same data for all models)
- Documents class weights needed for training

**Time**: 1-2 minutes

---

## Stage 2: Baseline Model

### What: `baselines/`

Train a standard Graph Attention Network (no quantum) as a baseline for comparison.

#### `baseline_gat_training.ipynb` - Train Baseline GAT
**What it does:**
1. Loads the graph from `artifacts/elliptic_graph.pt`
2. Creates train/val/test splits
3. Defines a GAT model (2 layers, 64 hidden channels)
4. Trains for up to 300 epochs
5. Stops early if validation F1 stops improving
6. Reports final test results

**Model Architecture:**
```
182 node features
        ↓
    [GAT Layer 1]  (64 hidden channels, 4 attention heads)
        ↓
    [LayerNorm + ELU activation]
        ↓
    [GAT Layer 2]  (64 hidden channels, 4 attention heads)
        ↓
    [Classifier]  (outputs 2 classes: fraud/non-fraud)
```

**What "64 hidden channels" means:**
- Each node is represented by 64 numbers after first GAT layer
- These 64 numbers are learned during training
- Larger number = more model capacity, but slower training

**Why it matters:**
- Establishes a performance baseline
- Standard approach (not quantum-inspired)
- Used for fair comparison with quantum model

**Outputs:**
- `artifacts/baseline_gat_best.pt` - Model weights
- `artifacts/baseline_gat_metrics.json` - F1, accuracy, etc.

**Time**: 10-15 minutes (CPU), 2-3 minutes (GPU)  
**Expected F1**: ~0.87

---

## Stage 3: Quantum Model

### What: `quantum_models/`

Train an enhanced model that uses quantum-inspired phase encoding.

#### `quantum_gat_training.ipynb` - Train Quantum GAT (QIGAT)
**What it does:**
1. Loads same graph as baseline
2. Trains a model with an extra "Quantum Phase Block"
3. Compares final results with baseline
4. Reports improvement

**Model Architecture:**
```
182 node features [FULL - no compression]
        ↓
    [GAT Layer 1]  (128 hidden channels, 4 heads)
        ↓
    [Quantum Phase Block]  🌟 THE NEW PART
      ├─ Takes 128 numbers from GAT
      ├─ Projects to "phase" space: φ = π * tanh(Wx)
      ├─ Computes cos(φ) and sin(φ)  (trig functions)
      ├─ Expands to 256 numbers (128 × 2)
      └─ Returns enhanced representation
        ↓
    [Residual Connection]  (learnable α blending)
        ↓
    [GAT Layer 2]  (128 hidden channels, 4 heads)
        ↓
    [Classifier]  (outputs fraud/non-fraud)
```

**What the Quantum Phase Block does:**

Think of it like transforming how numbers are represented:
- **Normal representation**: Just use the numbers directly (-0.5, 0.3, etc.)
- **Phase representation**: Convert to angles (0.2π, -0.1π, etc.), then to cos/sin
- **Why**: Trig functions can capture patterns differently than raw numbers
- **Inspired by**: Quantum computing's phase encoding (but classical implementation)

**Residual Connection:**
- Blends the quantum output with the original using learnable weight α
- Prevents the model from throwing away the original information
- Helps training stability

**Why it matters:**
- Tests the quantum-inspired approach
- Shows if phase encoding actually helps
- Modest parameter increase (~10%) but potential for better fraud detection

**Outputs:**
- `artifacts/qigat_corrected_best.pt` - Model weights
- `artifacts/qigat_corrected_report.json` - Full metrics

**Time**: 15-20 minutes (CPU), 3-4 minutes (GPU)  
**Expected F1**: ~0.89 (+2-3% vs baseline)

---

## Stage 4: Validation (Optional but Important)

### What: `validation/`

Run the models multiple times with different random seeds to prove results are reliable.

#### Why Multiple Seeds?

Imagine flipping a coin 10 times:
- Sometimes you get 6 heads, 4 tails (60%)
- Sometimes you get 5 heads, 5 tails (50%)
- Same coin, but different results!

Models are similar:
- Different random initialization → different final results
- Want to show improvement is consistent across seeds, not luck

#### What Validation Does

**Protocol** (run 5 times each):
```
Seed 42  → Train Baseline F1=0.871, Train QIGAT F1=0.892
Seed 43  → Train Baseline F1=0.869, Train QIGAT F1=0.890
Seed 44  → Train Baseline F1=0.873, Train QIGAT F1=0.894
Seed 45  → Train Baseline F1=0.870, Train QIGAT F1=0.891
Seed 46  → Train Baseline F1=0.872, Train QIGAT F1=0.889
         ─────────────────────────────────────────
         Stats:
           Baseline: mean=0.871, std=0.002
           QIGAT:    mean=0.891, std=0.002
           Difference: 0.020 (statistically significant!)
```

**Statistical Test** (paired t-test):
- Compares all Baseline runs vs all QIGAT runs
- Computes p-value: probability this difference happened by chance
- p < 0.05 = statistically significant ✅

**Time**: 6-8 hours (CPU), 1-2 hours (GPU)  
**When to run**: Before submitting to conference/journal

---

## 🔧 Supporting Components

### `src/` — Core Implementation
Reusable code imported by notebooks.

#### `config.py`
- Central configuration file
- All hyperparameters in one place (learning rate, dropout, etc.)
- Notebooks import from here

#### `models.py`
- GAT architecture definition
- Only defines **what the model looks like**
- Notebooks define **how to train it**

#### `quantum_features.py`
- Quantum Phase Block implementation
- Learned phase projection
- Trig transformations

#### `utils.py`
- Helper functions (random seed setting, device selection)
- Imported by all notebooks

### `research_validation/` — Advanced Experiments
Framework for ablation studies and detailed analysis.

#### `runner.py`
- Main entry point for multi-seed validation
- Orchestrates training multiple models multiple times

#### `train.py`
- Unified training loop
- Used by all experiment variants

#### `evaluate.py`
- Metric computation (F1, accuracy, precision, recall, ROC-AUC)
- Consistent across all experiments

#### `stats.py`
- Statistical tests
- Paired t-tests, confidence intervals

---

## 📁 File Roles Summary

| File/Folder | What It Does | When to Use |
|-------------|-------------|------------|
| `create_graph.ipynb` | Build graph from CSVs | First, once per dataset |
| `sanity_checks.ipynb` | Verify data integrity | Before training |
| `baseline_gat_training.ipynb` | Train standard model | Establish baseline |
| `quantum_gat_training.ipynb` | Train quantum model | Main experiment |
| `artifacts/` | Store models & results | Results repository |
| `src/` | Core functions | Imported by notebooks |
| `research_validation/` | Multi-seed validation | Before publishing |

---

## 🎯 Workflow Decisions

### Which notebooks to run?

**Just learning?**
→ Run stages 1-2-3 (skip validation for quick understanding)

**Comparing models?**
→ Run stages 1-2-3 (shows F1 improvement)

**Publishing paper?**
→ Run stages 1-2-3-4 (includes statistics)

---

## 🚀 Key Takeaways

1. **Stage 1** prepares data (one-time setup)
2. **Stage 2** trains simple model (baseline)
3. **Stage 3** trains quantum model (new approach)
4. **Stage 4** validates results statistically (publication-grade)

Each stage depends on previous ones.  
Each saves outputs for next stage.  
Can view results without re-running (cached in `artifacts/`).

---

**For detailed code-level explanations, see the markdown cells in each notebook! ⬇️**
