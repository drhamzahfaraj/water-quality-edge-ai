# Methodology

## 1. Overall Design Objective

The proposed framework is designed to **minimize energy consumption per inference** on resource-constrained IoT nodes while keeping water-quality prediction accuracy within 5% of a full-precision teacher model. To achieve this, we combine:

1. **Temporal Convolutional Network (TCN)** backbone for efficient temporal representation
2. **Variance-driven dynamic and mixed-precision quantization** to adapt computation to regime changes
3. **Knowledge distillation** to transfer accuracy from a larger teacher to a compact student
4. **Hardware-aware neural architecture search (HW-NAS)** to select architectures optimized for edge devices

All components are trained and evaluated on a stratified subset of the UNEP GEMSWater archive.

---

## 2. Dataset and Preprocessing

### 2.1 Source Data

We use the **UNEP GEMSWater Global Freshwater Quality Archive**:
- **Total records:** 20,446,832 measurements
- **Stations:** 13,660 across 37 countries
- **Time span:** 1906–2023
- **Parameters:** pH, dissolved oxygen (DO), turbidity, electrical conductivity, nitrate (NO₃⁻), phosphate (PO₄³⁻), total suspended solids (TSS), biochemical oxygen demand (BOD), chemical oxygen demand (COD), water temperature

### 2.2 Stratified Sampling (500K subset)

We extract a **stratified subset of 500,000 records (2.5%)** that balances geographic coverage, temporal span, and variance conditions while remaining computationally tractable for iterative architecture search.

**Four-stage sampling protocol:**

1. **Quality Control**
   - Remove records with >3 missing values among 10 parameters
   - Drop duplicate timestamps
   - Discard outliers exceeding 5σ from station-specific mean
   - Filter bad quality flags
   - **Result:** 18,234,109 quality-assured records (91% pass rate)

2. **Temporal Stratification**
   - 1906–1979: 5%
   - 1980–1999: 15%
   - 2000–2014: 40%
   - 2015–2023: 40%
   - **Purpose:** Mitigate bias toward recent observations

3. **Geographic Stratification**
   - Proportional sampling across 37 countries and 6 continents
   - Cap at 50 records per station per temporal stratum
   - **Result:** All regions represented; no single-station dominance

4. **Variance-Aware Oversampling**
   - Low (σ < 0.05): 42%
   - Moderate (0.05 ≤ σ < 0.15): 45%
   - High (σ ≥ 0.15): 13%
   - **Purpose:** Ensure model sees both stable and dynamic regimes

**Statistical Validation:**
- Parameter means differ by <2.3% from full archive
- Geographic correlation: r = 0.978 (p < 0.001)
- Temporal distribution KS test: p = 0.42 (not significantly different)

### 2.3 Data Preprocessing

The 500K subset is normalized per parameter using station-wise statistics and organized into:
- **Input windows:** 24 hourly samples (10 parameters each)
- **Prediction target:** Next-hour forecast for selected parameters
- **Storage:** Structured as a compact data warehouse for reproducible train/validation/test splits

---

## 3. TCN-Based Forecasting Model

### 3.1 Architecture Overview

Our base model is a **CNN-TCN hybrid**:
- **Front-end:** Shallow 1D convolutional layers project 10 input parameters into higher-dimensional feature space
- **Backbone:** Stack of L residual TCN blocks with dilated causal convolutions
- **Output:** Regression head for multi-parameter prediction

### 3.2 Temporal Convolutional Network (TCN)

Each TCN block contains:
- Two dilated causal convolutional layers
- Residual (skip) connections
- Nonlinear activation (ReLU)
- Spatial dropout for regularization

**Receptive Field:** For L residual blocks, kernel size k, and dilation factors {d₁, ..., dₗ}:

```
R = 1 + Σ(l=1 to L) 2(k - 1) · dₗ
```

**Example configuration:**
- L = 4 blocks
- Dilation factors: {1, 2, 4, 8}
- Kernel size: k = 3
- Receptive field: R = 255 time steps

**Why TCN over LSTM?**
- Parallel processing across time steps (vs. sequential in LSTM)
- 25–30% lower inference latency
- Better suited to convolution-optimized kernels on edge CPUs
- More amenable to mixed-precision quantization

---

## 4. Optimization and Model Reduction

### 4.1 Variance-Driven Dynamic Quantization

**Core Idea:** Adapt numerical precision to local signal variability.

**Variance Estimation:** Over sliding window of T = 24 samples:

```
σₜ = √(1/T Σ(i=t-T+1 to t) (xᵢ - μₜ)²)
```

**Bit-Width Assignment:**

```
bₜ = {
  4 bits   if σₜ < 0.05       (stable regime)
  6 bits   if 0.05 ≤ σₜ < 0.15 (moderate variability)
  8 bits   if σₜ ≥ 0.15        (high variability/pollution events)
}
```

**Rationale:**
- Low precision during stable periods → save energy
- High precision during dynamic/polluted periods → preserve accuracy

### 4.2 Mixed-Precision Assignment

Within each quantization regime, apply **layer-wise mixed precision**:

| Layer Type | Bit-Width |
|-----------|-----------|
| CNN front-end | bₜ |
| TCN blocks 1–2 | bₜ |
| TCN blocks 3–4 | bₜ - 1 |
| Output layer | 8 bits (fixed) |

**Implementation:** Post-training quantization with uniform affine quantizers; per-layer scale and zero-point calibrated on training data.

### 4.3 Knowledge Distillation

**Purpose:** Mitigate accuracy loss from aggressive quantization.

**Setup:**
- **Teacher:** Full-precision (32-bit) CNN-TCN trained for 100 epochs
- **Student:** Quantized, pruned CNN-TCN with ~35% sparsity trained for 150 epochs

**Loss Function:**

```
L_distill = α · L_task(y, y_student) + (1 - α) · L_KL(p_teacher || p_student)
```

where:
- L_task: Mean squared error (regression loss)
- L_KL: Kullback-Leibler divergence
- α = 0.7 (balances task loss and knowledge transfer)
- Temperature scaling: T = 3 applied to teacher logits

**Training Schedule:** Gradual reduction of α from 0.9 → 0.5 over epochs.

### 4.4 Hardware-Aware Neural Architecture Search (HW-NAS)

**Purpose:** Automatically discover architectures optimized for target edge device (Raspberry Pi 4).

**Search Space:**
- Number of TCN channels: {32, 64, 128}
- Kernel sizes: {3, 5, 7}
- Dilation patterns
- Network depth

**Objective Function:**

```
L_NAS = L_task + λ_E · E_inf + λ_L · L_inf
```

where:
- E_inf: Estimated energy per inference (from profiling models)
- L_inf: Estimated latency per inference
- λ_E = 0.1, λ_L = 0.05 (weighting coefficients)

**Search Method:** Differentiable Architecture Search (DARTS)
- 100 search iterations
- 50 candidate architectures per iteration
- Energy/latency estimates from FLOPs and memory-access models

**Result:** Student architecture on favorable accuracy-efficiency Pareto frontier.

---

## 5. Training and Evaluation Protocol

### 5.1 Dataset Splits

- **Training:** 400,000 records (80%)
- **Validation:** 50,000 records (10%)
- **Testing:** 50,000 records (10%)

**Split Strategy:**
- Geographic stratification (maintain continent/country ratios)
- Chronological split: train on 1906–2020, test on 2021–2023

### 5.2 Training Configuration

**Optimizer:** Adam
- Learning rate: 0.001
- β₁ = 0.9, β₂ = 0.999

**Regularization:**
- L2 weight decay: λ = 0.0001
- Spatial dropout: p = 0.2
- Gaussian noise augmentation: σ = 0.1
- Gradient clipping: max norm = 1.0

**Learning Rate Schedule:** Cosine annealing with warm restarts every 30 epochs

**Cross-Validation:** 5-fold with geographic stratification

**Early Stopping:** Based on validation loss

### 5.3 Baselines

1. **Non-AI:** Moving average + exponential smoothing
2. **Fixed 8-bit quantization:** Static uniform quantization
3. **Activation-aware quantization:** Per-activation dynamic range
4. **TinyML:** Aggressive 4-bit + pruning
5. **CNN-LSTM (FP32):** Full-precision recurrent baseline
6. **CNN-LSTM (quantized):** Fixed 8-bit recurrent model

### 5.4 Evaluation Metrics

**Accuracy Metrics:**
- Prediction accuracy (±5% tolerance)
- Root Mean Squared Error (RMSE)

**Efficiency Metrics:**
- Model size (MB)
- FLOPs (millions of operations)
- Latency (ms on Raspberry Pi 4)
- Power consumption (W)
- Energy per inference (mJ)

**Hardware:** All efficiency metrics estimated using analytical and profiling-based models rather than physical deployment measurements.

---

## 6. Ablation Studies

To quantify each component's contribution, we conduct systematic ablations:

| Configuration | Change | Expected Impact |
|--------------|--------|-----------------|
| **Full model** | CNN-TCN + all optimizations | Baseline |
| w/ CNN-LSTM | Replace TCN with LSTM | Higher FLOPs, latency |
| w/o Adaptive Quant | Fixed 8-bit instead | Higher power |
| w/o Distillation | No teacher model | Lower accuracy |
| w/o HW-NAS | Manual architecture | Suboptimal efficiency |
| w/o Mixed Precision | Uniform bit-width | Higher power |
| Fixed 4-bit only | Aggressive quantization | Lower accuracy |

**Retraining:** Each ablation retrained with same data splits and hyperparameters.

---

## 7. Simulation-Based Evaluation

**Important Note:** This study uses **simulation-based evaluation** rather than physical hardware deployment:

- **Power and energy** estimates derived from analytical models calibrated with hardware profiling (pyRAPL on CPU, nvidia-smi on GPU)
- **Latency** estimates based on FLOPs, memory access patterns, and quantization bit-widths
- **Battery life** projections calculated from energy-per-inference models combined with typical sensor and communication loads

All metrics should be interpreted as **analytical estimates under stated assumptions** rather than field-measured results from deployed IoT nodes.

---

## 8. Reproducibility

**Environment:**
- Python 3.10
- PyTorch 2.0
- NumPy 1.24 (random seed: 42)

**Data Access:**
- Full UNEP GEMSWater archive: [Zenodo](https://doi.org/10.5281/zenodo.10701676)
- 500K subset record IDs: `data/sampling_metadata.txt`

**Code:** See `src/` directory for model definitions, training scripts, and evaluation utilities.

---

## References

See main paper for complete bibliography. Key methodological references:

- **TCN architecture:** Yan et al. (2024), Liu et al. (2024), Wang et al. (2024)
- **Dynamic quantization:** Gholami et al. (2022), Rokh et al. (2024)
- **Knowledge distillation:** Hasan et al. (2024), Huang et al. (2024)
- **HW-NAS:** Zhou et al. (2024), Li et al. (2024), Jin et al. (2024)
- **UNEP GEMSWater:** Heinle et al. (2024), Virro et al. (2021)
