# Experimental Results

This directory contains CSV files with all numerical results reported in the paper.

## Files

### Main Results (Table 1)
**File:** `results_main.csv`

Comparison of all baselines and the proposed method on the full test set.

- **Corresponds to:** Table 1 in Section 4.3
- **Key findings:** 
  - 27.5% energy reduction vs Fixed 8-bit (2.00 → 1.45 mJ)
  - 5.6% relative accuracy gain (0.72 → 0.76 in 1-RMSE)

### Ablation: Compression & Distillation (Table 2)
**File:** `results_ablation_compression.csv`

Isolates the effect of structured pruning and knowledge distillation.

- **Corresponds to:** Table 2 in Section 4.4
- **Key findings:**
  - 43% parameter reduction (2.10M → 1.20M)
  - Distillation recovers ~1 percentage point in accuracy

### Ablation: Static vs Dynamic Quantization (Table 3)
**File:** `results_ablation_quantization.csv`

Compares static mixed precision with variance-aware dynamic policy.

- **Corresponds to:** Table 3 in Section 4.4
- **Key findings:**
  - Dynamic policy provides additional 14.7% energy reduction over static
  - +2 percentage points in normalized accuracy

### Robustness: Noise (Figure 3)
**File:** `results_robustness_noise.csv`

Accuracy under additive Gaussian noise (σ = 0.05, 0.10, 0.20).

- **Corresponds to:** Figure 3 and Section 4.5
- **Key findings:**
  - Proposed method retains 97% of clean performance at σ=0.20
  - Fixed 8-bit retains only 94%

### Robustness: Missing Data
**File:** `results_robustness_missing.csv`

Accuracy when 10%, 20%, or 30% of readings are randomly dropped.

- **Corresponds to:** Section 4.5
- **Key findings:**
  - At 30% missingness: proposed maintains 93% of clean performance
  - Static mixed precision retains only 88%

### Extreme Events & Generalization
**File:** `results_extreme_generalization.csv`

Performance on stations with extreme events and cross-station (cross-region) generalization.

- **Corresponds to:** Section 4.5
- **Key findings:**
  - On extreme-event subset: 0.73 vs 0.69 (proposed vs fixed 8-bit)
  - Cross-station: 0.72 vs 0.68

## Reproducibility

All values in these CSV files can be reproduced by running the corresponding experiment scripts with the provided configuration files in `experiments/config/`.

See main README.md for detailed instructions.
