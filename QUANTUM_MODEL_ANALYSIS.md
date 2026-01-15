# Quantum Model Performance Analysis & Improvement Plan

**Date:** January 14, 2026  
**Status:** Critical Performance Issues Identified

---

## Executive Summary

The quantum-enhanced GAT model is **severely underperforming** compared to the baseline GAT model:

| Metric | Baseline | Quantum | Delta |
|--------|----------|---------|-------|
| **Test F1** | 79.38% | **18.99%** | **-60.39%** ❌ |
| **Test Precision** | 86.61% | **14.18%** | **-72.43%** ❌ |
| **Test Recall** | 73.27% | **28.71%** | **-44.56%** ❌ |
| **Test Accuracy** | 96.28% | 76.09% | -20.19% ❌ |
| **Test ROC-AUC** | 96.98% | 55.42% | -41.56% ❌ |
| **Training Time** | 309.6s | 42.3s | -267.3s ✓ |

**The quantum model is barely better than random guessing (50% AUC), achieving only 55.42% ROC-AUC.**

---

## Root Cause Analysis

### 1. **Severe Class Imbalance (NOT ADDRESSED)** 🔴 CRITICAL

**Problem:** The Elliptic dataset has extreme class imbalance (~10:1 ratio, licit vs illicit transactions).

**Evidence:**
- Validation accuracy plateaus at ~90.3% (exactly the majority class ratio)
- Val F1 score **degrades from 19.16% → 1.96%** over 21 epochs
- Model learns to predict majority class almost exclusively
- Early stopping triggered at epoch 21 with best Val F1 = 19.16%

**Impact:**
```
Training Loss: 0.693 → 0.457 (decreasing ✓)
Val Accuracy:  75.5% → 90.3% (increasing ✓)
Val F1:        19.1% → 1.96% (DEGRADING ❌)
Val AUC:       55.9% → 71.2% (marginal improvement)
```

The model is **learning to ignore the minority class** (fraudulent transactions).

**Root Cause:** 
- Using `nn.CrossEntropyLoss()` with **no class weighting**
- No oversampling/undersampling strategies
- No focal loss to emphasize hard examples

---

### 2. **Quantum Feature Expansion Destroys Signal** 🔴 CRITICAL

**Problem:** The quantum feature mapping doubles feature dimensionality but dilutes discriminative information.

**Current Implementation:**
```python
# Random Fourier Features expansion: 166 → 332 dimensions
projection = torch.matmul(x, W) + b
cos_features = torch.cos(projection)
sin_features = torch.sin(projection)
transformed = torch.cat([cos_features, sin_features], dim=1) / sqrt(output_dim)
```

**Issues:**

#### a) Random Projection Without Learning
- Weight matrix `W` is **fixed random initialization** (never trained)
- Bias term `b` is random uniform [0, 2π]
- No gradient flow through quantum feature mapper
- The projection is blind to the actual fraud detection task

#### b) Uniform Scaling Destroys Magnitude Information
```python
transformed = transformed / np.sqrt(self.output_dim)
```
- Division by √332 ≈ 18.22 severely dampens all features
- Original feature magnitudes (which encode transaction patterns) are lost
- All features squeezed into tiny range around 0

#### c) Fourier Features Mismatch
- Random Fourier Features work well for smooth, continuous functions
- Fraud patterns are **discrete, sparse, and non-smooth**
- cos/sin periodicity inappropriate for categorical-style fraud indicators

**Evidence:**
```
Original Features:  Mean: varies by feature, contains fraud signals
Quantum Features:   Mean: ~0.0, Std: normalized, signal diluted
```

---

### 3. **Input Dimension Mismatch Training** ⚠️ HIGH

**Problem:** The baseline model is trained on normalized 166 features, but quantum model receives transformed 332 features with **completely different statistical properties**.

**Training Discrepancy:**

| Aspect | Baseline | Quantum |
|--------|----------|---------|
| Input Features | 166 (original) | 332 (quantum-expanded) |
| Normalization | StandardScaler on training set | Random Fourier normalization |
| Feature Range | [-10, +10] clipped | [-0.055, +0.055] typical |
| Feature Meaning | Domain interpretable | Abstract cos/sin combinations |

**Impact:**
- GAT's first attention layer must learn from scratch with quantum features
- Attention weights cannot leverage patterns learned in baseline
- Effective loss of 300+ epochs of baseline pretraining knowledge

---

### 4. **No Feature Normalization Before Quantum Transform** ⚠️ HIGH

**Problem:** Quantum features are applied to **raw, unnormalized input features**.

**Current Pipeline:**
```
Raw Features → Quantum Transform → GAT Model
     ↓                 ↓
  No normalization    Fixed random projection
```

**Baseline Pipeline:**
```
Raw Features → Normalize (mean=0, std=1) → Clip [-10,10] → GAT Model
     ↓                          ↓
  Training set stats      Prevents gradient explosion
```

**Consequence:**
- Features with large raw values dominate the quantum projection
- cos/sin transform amplifies scale differences
- Model sees inconsistent feature distributions between train/val/test

---

### 5. **Insufficient Model Capacity** ⚠️ MEDIUM

**Problem:** Same model architecture (64 hidden × 4 heads × 2 layers) used for both 166 and 332 input features.

**Baseline:**
```
166 features → [64×4=256 hidden] → [64×4=256 hidden] → 2 classes
Parameters: ~202K
```

**Quantum:**
```
332 features → [64×4=256 hidden] → [64×4=256 hidden] → 2 classes
Parameters: ~311K (54% more parameters in first layer)
```

**Issue:**
- Doubling input dimension requires significantly more capacity to learn effectively
- Hidden dimension (64) likely too small for 332 quantum features
- Attention heads (4) may need to increase for richer feature combinations

---

### 6. **Training Configuration Mismatches** ⚠️ MEDIUM

**Observation:** Quantum model stops at epoch 21 vs baseline at epoch 170.

**Training Dynamics:**

| Config | Baseline | Quantum | Analysis |
|--------|----------|---------|----------|
| Best Epoch | 170 | 1 (F1=19.16%) | Quantum overfits immediately |
| Patience Used | ~20/20 epochs | 20/20 epochs | Early stop works correctly |
| Learning Rate | 0.001 | 0.001 | May be too high for quantum |
| Weight Decay | 5e-4 | 5e-4 | Insufficient regularization |

**Problems:**
- Quantum features might need lower learning rate (more sensitive)
- Weight decay may need adjustment for expanded feature space
- No learning rate scheduling

---

### 7. **Lack of Hybrid Architecture** 💡 OPPORTUNITY

**Missed Opportunity:** Current approach uses quantum OR classical features.

**Better Approach:** Use quantum AND classical features (ensemble/hybrid).

**Why:**
- Quantum features capture non-linear patterns
- Classical features preserve domain knowledge
- Concatenation allows model to choose best representation

---

## Recommended Solutions (Prioritized)

### **Priority 1: Address Class Imbalance** 🔴 MUST FIX

#### Solution 1A: Weighted Loss Function
```python
# Compute class weights inversely proportional to frequency
train_labels = data.y[data.train_mask]
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # Normalize

# Apply to loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Expected Impact:** +30-40% F1 improvement

#### Solution 1B: Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**Expected Impact:** +35-45% F1 improvement (focuses on hard fraudulent examples)

#### Solution 1C: SMOTE or Oversampling
```python
from imblearn.over_sampling import SMOTE

# Oversample minority class in training set
train_x = data.x[data.train_mask].cpu().numpy()
train_y = data.y[data.train_mask].cpu().numpy()
smote = SMOTE(random_state=42)
train_x_balanced, train_y_balanced = smote.fit_resample(train_x, train_y)
```

**Expected Impact:** +25-35% F1 improvement

---

### **Priority 2: Fix Quantum Feature Mapping** 🔴 MUST FIX

#### Solution 2A: Learnable Quantum Features
```python
class LearnableQuantumFeatureMap(nn.Module):
    def __init__(self, input_dim, expansion_factor=2):
        super().__init__()
        self.output_dim = input_dim * expansion_factor
        
        # Learnable projection (trainable via backprop)
        self.W = nn.Parameter(torch.randn(input_dim, self.output_dim // 2) / np.sqrt(input_dim))
        self.b = nn.Parameter(torch.rand(self.output_dim // 2) * 2 * np.pi)
    
    def forward(self, x):
        projection = torch.matmul(x, self.W) + self.b
        cos_features = torch.cos(projection)
        sin_features = torch.sin(projection)
        return torch.cat([cos_features, sin_features], dim=1)
```

**Expected Impact:** +15-25% F1 improvement

#### Solution 2B: Remove Aggressive Normalization
```python
# Remove: transformed = transformed / np.sqrt(self.output_dim)
# Instead use: LayerNorm after projection
self.layer_norm = nn.LayerNorm(self.output_dim)
transformed = self.layer_norm(torch.cat([cos_features, sin_features], dim=1))
```

**Expected Impact:** +10-15% F1 improvement

#### Solution 2C: Task-Aware Quantum Kernel
```python
# Use RBF kernel instead of random Fourier (better for fraud patterns)
class RBFQuantumFeatureMap:
    def __init__(self, input_dim, num_centers=100, gamma=1.0):
        self.centers = torch.randn(num_centers, input_dim)
        self.gamma = gamma
    
    def transform(self, x):
        # Compute RBF kernel: exp(-gamma * ||x - center||^2)
        dists = torch.cdist(x, self.centers)  # [N, num_centers]
        return torch.exp(-self.gamma * dists**2)
```

**Expected Impact:** +20-30% F1 improvement

---

### **Priority 3: Pre-normalize Features** ⚠️ SHOULD FIX

#### Solution 3A: Normalize Before Quantum Transform
```python
# In 05_quantum_feature_map.ipynb
# Normalize features BEFORE quantum transformation
train_x = data.x[data.train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)

data.x = (data.x - mean) / std
data.x = torch.clamp(data.x, min=-10, max=10)

# THEN apply quantum transform
x_quantum = feature_mapper.transform(data.x)
```

**Expected Impact:** +10-20% F1 improvement

---

### **Priority 4: Hybrid Classical-Quantum Architecture** 💡 RECOMMENDED

#### Solution 4A: Feature Concatenation
```python
# Concatenate original + quantum features
x_classical = data.x  # [N, 166]
x_quantum = feature_mapper.transform(data.x)  # [N, 332]
x_hybrid = torch.cat([x_classical, x_quantum], dim=1)  # [N, 498]

# Train GAT on hybrid features
model = GAT(in_channels=498, ...)
```

**Expected Impact:** +20-35% F1 improvement

#### Solution 4B: Ensemble Model
```python
class HybridGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_classical = GAT(in_channels=166, ...)
        self.gat_quantum = GAT(in_channels=332, ...)
        self.fusion = nn.Linear(4, 2)  # Combine logits
    
    def forward(self, x_classical, x_quantum, edge_index):
        out_classical = self.gat_classical(x_classical, edge_index)
        out_quantum = self.gat_quantum(x_quantum, edge_index)
        combined = torch.cat([out_classical, out_quantum], dim=1)
        return self.fusion(combined)
```

**Expected Impact:** +25-40% F1 improvement

---

### **Priority 5: Increase Model Capacity** ⚠️ SHOULD FIX

#### Solution 5A: Scale Hidden Dimensions
```python
# For 332 quantum features, use proportionally larger hidden dim
MODEL_CONFIG_QUANTUM = {
    "hidden_channels": 128,  # Was 64 (2× increase)
    "num_heads": 6,          # Was 4 (50% increase)
    "num_layers": 3,         # Was 2 (add depth)
    "dropout": 0.4,          # Increased regularization
}
```

**Expected Impact:** +10-20% F1 improvement

---

### **Priority 6: Optimize Training Hyperparameters** ⚠️ SHOULD FIX

#### Solution 6A: Learning Rate Schedule
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, verbose=True
)

# In training loop
val_metrics = evaluate(data.val_mask)
scheduler.step(val_metrics['f1'])  # Reduce LR when F1 plateaus
```

**Expected Impact:** +5-10% F1 improvement

#### Solution 6B: Gradient Clipping
```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Expected Impact:** +5-10% F1 improvement (prevents gradient explosion)

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Expected: +50-70% F1 gain)
1. ✅ **Implement weighted CrossEntropyLoss** (Priority 1A)
2. ✅ **Normalize features before quantum transform** (Priority 3A)
3. ✅ **Make quantum projection learnable** (Priority 2A)
4. ✅ **Remove aggressive sqrt normalization** (Priority 2B)

### Phase 2: Architecture Improvements (Expected: +15-25% F1 gain)
5. ✅ **Increase model capacity** (Priority 5A)
6. ✅ **Add learning rate scheduler** (Priority 6A)
7. ✅ **Implement gradient clipping** (Priority 6B)

### Phase 3: Advanced Techniques (Expected: +10-30% F1 gain)
8. ⚪ **Try Focal Loss** (Priority 1B)
9. ⚪ **Implement hybrid classical-quantum model** (Priority 4A)
10. ⚪ **Experiment with RBF quantum kernel** (Priority 2C)

### Phase 4: Experimental (Expected: +5-15% F1 gain)
11. ⚪ **SMOTE oversampling** (Priority 1C)
12. ⚪ **Ensemble classical + quantum GATs** (Priority 4B)
13. ⚪ **Graph-level augmentation techniques**

---

## Expected Outcomes

### Conservative Estimate (Phase 1 + 2)
- **Test F1:** 18.99% → **70-75%** (+51-56%)
- **Test ROC-AUC:** 55.42% → **85-90%** (+29.5-34.5%)
- **Test Precision:** 14.18% → **70-75%**
- **Test Recall:** 28.71% → **70-75%**

### Optimistic Estimate (All Phases)
- **Test F1:** 18.99% → **80-85%** (+61-66%)
- **Test ROC-AUC:** 55.42% → **92-95%** (+36.5-39.5%)
- **Match or exceed baseline performance**

---

## Key Insights

1. **Quantum features are NOT inherently bad** - the implementation needs fixing
2. **Class imbalance is the primary killer** - must be addressed first
3. **Random projections need gradients** - make them learnable
4. **Hybrid approaches are most promising** - combine classical + quantum
5. **Validation metrics tell the story** - watch F1 degradation patterns

---

## Next Steps

1. **Implement Phase 1 fixes immediately** (2-3 hours work)
2. **Retrain quantum model with fixes** (30-60 minutes)
3. **Compare against baseline** (validate improvements)
4. **Iterate on Phase 2-3** (if Phase 1 successful)
5. **Document final results** (update metrics and analysis)

---

## References

- **Focal Loss:** [Lin et al., 2017 - Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Random Fourier Features:** [Rahimi & Recht, 2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
- **Graph Attention Networks:** [Veličković et al., 2018](https://arxiv.org/abs/1710.10903)
- **Class Imbalance in Graphs:** [Liu et al., 2021 - ImGAGN](https://arxiv.org/abs/2106.01404)

---

**Analysis completed:** January 14, 2026  
**Status:** Ready for implementation
