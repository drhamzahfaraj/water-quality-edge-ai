# Data Directory

This directory contains experimental results, sampling metadata, and instructions for obtaining the UNEP GEMSWater dataset used in our research.

## 📂 Contents

### Experimental Results

1. **`results.csv`** - Main performance comparison across 7 methods
   - Columns: Method, Power_W, RMSE, Accuracy_Pct, FLOPs_M, Energy_mJ, Model_Size_MB, Latency_ms, Notes
   - 7 baseline and proposed methods evaluated on 50K test samples

2. **`ablation_results.csv`** - Component contribution analysis
   - Systematic removal of TCN, adaptive quantization, distillation, HW-NAS, mixed precision
   - 7 configurations showing individual and synergistic effects

3. **`sensitivity_results.csv`** - Hyperparameter robustness analysis
   - Variance threshold perturbations (±20%)
   - HW-NAS regularization weight variations
   - Quantization scheme comparisons (fixed vs. adaptive)
   - 15 configurations demonstrating stability

4. **`geographic_results.csv`** - Cross-continental generalization
   - Train on 5 continents, test on 6th
   - Average -2.1% accuracy degradation
   - 6 continents with RMSE, accuracy, test sample counts

5. **`sampling_metadata.txt`** - 500K subset stratification details
   - 4-stage sampling strategy (quality, temporal, geographic, variance-aware)
   - Statistical validation metrics (r=0.978 correlation, p<0.001)

### Documentation

6. **`METRICS_EXPLAINED.md`** - **⭐ READ THIS FIRST**
   - Comprehensive explanation of all performance metrics
   - Energy-Power-Latency relationship clarification
   - Analytical energy model with platform-specific constants
   - Battery life calculation methodology
   - Measurement protocols and interpretation guidelines

---

## 📊 Understanding the Metrics

**IMPORTANT:** All metrics are derived from analytical models calibrated with pyRAPL profiling on Raspberry Pi 4 emulation, not direct field measurements.

### Key Relationships

```
Energy (mJ) = Power (W) × Total_Cycle_Time (~50ms)
              ≠ Power × Inference_Latency (32ms)

Total Cycle includes:
- Sensor read: 10ms
- Preprocessing: 5ms  
- Inference: 32ms (latency reported)
- Post-processing: 3ms
```

**See [`METRICS_EXPLAINED.md`](METRICS_EXPLAINED.md) for complete details.**

---

## 🌍 UNEP GEMS/Water Dataset

### Dataset Description

Our experiments use a **stratified 500,000-record subset (2.5%)** of the UNEP GEMS/Water Global Freshwater Quality Archive.

**Specifications:**
- **Total samples:** 500,000 records (from 20.4M total)
- **Parameters:** 10 key water quality indicators
  1. pH (6-9 range)
  2. Dissolved Oxygen (0-15 mg/L)
  3. Turbidity (0-100 NTU)
  4. Conductivity (μS/cm)
  5. Nitrate NO₃ (mg/L)
  6. Phosphate PO₄ (mg/L)
  7. Total Suspended Solids TSS (mg/L)
  8. Biochemical Oxygen Demand BOD (mg/L)
  9. Chemical Oxygen Demand COD (mg/L)
  10. Temperature (°C)

- **Stations:** 13,660 monitoring stations
- **Countries:** 37 countries across 6 continents
- **Time period:** 1906–2023 (117 years)

### Sampling Strategy

**4-stage stratified sampling** with statistical representativeness guarantees:

1. **Quality Filtering** (91% pass rate)
   - ≤3 missing values among 10 parameters
   - Duplicate removal
   - Outlier detection (5σ threshold)

2. **Temporal Stratification**
   - 1906-1979: 25,000 samples (5%)
   - 1980-1999: 75,000 samples (15%)
   - 2000-2014: 200,000 samples (40%)
   - 2015-2023: 200,000 samples (40%)

3. **Geographic Proportionality**
   - North America: 119,170 records (24%)
   - Europe: 153,760 records (31%)
   - Asia: 106,310 records (21%)
   - Africa: 55,940 records (11%)
   - South America: 41,950 records (8%)
   - Oceania: 25,390 records (5%)

4. **Variance-Aware Oversampling**
   - Low (σ<0.05): 210,000 records (42%)
   - Moderate (0.05≤σ<0.15): 225,000 records (45%)
   - High (σ≥0.15): 65,000 records (13%)

**Statistical Validation:**
- Geographic correlation: **r = 0.978** (p < 0.001)
- Parameter means within **2.3%** of full archive
- Cohen's d < 0.05 (negligible effect size)
- KS test: **p = 0.42** (no significant difference)

See `sampling_metadata.txt` for complete details.

### Preprocessing Pipeline

The raw UNEP data was processed using `src/preprocess_unep.py`:

1. **Indicator selection:** 10 most prevalent parameters
2. **Missing value handling:** Mean imputation per station (~15% missing)
3. **Outlier removal:** IQR method (1.5 × IQR rule)
4. **Normalization:** Min-max scaling to [0, 1]
5. **Windowing:** 24-hour input windows, 1-hour prediction horizon
6. **Augmentation:** 20% of training data with Gaussian noise (σ=0.1)

### Data Split

- **Training:** 80% (400,000 samples)
- **Validation:** 10% (50,000 samples)  
- **Test:** 10% (50,000 samples)

Split is:
- **Temporally consistent:** 1906-2020 for train, 2021-2023 for test
- **Geographically stratified:** Preserves continent proportions

---

## 📥 Obtaining the Data

The original UNEP GEMS/Water dataset is publicly available:

**Source:** [UNEP GEMS/Water on Zenodo](https://doi.org/10.5281/zenodo.10701676) (CC BY 4.0)

**Citation:**
> Heinle, M., Schäfer, J., & Ludwig, R. (2024). UNEP GEMSWater Global Freshwater Quality Database, 1906-2023 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10701676

### Reproducing the 500K Subset

```bash
# 1. Download raw UNEP data from Zenodo (requires registration)
wget https://zenodo.org/record/10701676/files/unep_gemswater_full.csv

# 2. Run our preprocessing script
python src/preprocess_unep.py \
  --input unep_gemswater_full.csv \
  --output data/unep_subset.csv \
  --target-samples 500000 \
  --seed 42

# 3. Verify stratification
python src/verify_sampling.py --input data/unep_subset.csv
```

**Expected output:**
```
Total samples: 500,000
Parameters: 10
Stations: 13,660
Date range: 1906-01-15 to 2023-12-28
Missing values: 0 (after imputation)
Normalized range: [0.000, 1.000]
Geographic correlation with full archive: r=0.978 (p<0.001)
✓ Sampling validated successfully
```

### File Size Note

The processed `unep_subset.csv` file is approximately **150 MB** and is **not included** in this Git repository due to size constraints. Users must download and preprocess the raw data as described above.

---

## 🔍 Data Validation

After preprocessing, verify your subset:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/unep_subset.csv')

# Basic statistics
print(f"Total samples: {len(df):,}")
print(f"Parameters: {len([c for c in df.columns if c not in ['station_id', 'date']])}")
print(f"Stations: {df['station_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing values: {df.isna().sum().sum()}")

# Normalization check
numeric_cols = df.select_dtypes(include='number').columns
print(f"\nValue ranges (should be [0, 1] after normalization):")
for col in numeric_cols:
    print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")

# Variance distribution
variances = df[numeric_cols].var()
print(f"\nVariance distribution:")
print(f"  Low (<0.05): {(variances < 0.05).sum()} parameters")
print(f"  Moderate (0.05-0.15): {((variances >= 0.05) & (variances < 0.15)).sum()} parameters")
print(f"  High (≥0.15): {(variances >= 0.15).sum()} parameters")
```

---

## 📜 License

The UNEP GEMS/Water data is licensed under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** - Creative Commons Attribution 4.0 International.

**Usage Requirements:**
- ✅ Proper attribution to UNEP and original data providers
- ✅ Citation in publications (see above)
- ✅ Indication of any modifications made

---

## 🔗 Related Files

- **Paper:** `../paper/main.tex` - Full methodology and results
- **Methods:** `../METHODS.md` - Detailed experimental procedures
- **Preprocessing:** `../src/preprocess_unep.py` - Data preparation script
- **Analysis:** `../experiments/` - Experiment configuration files

---

**Last Updated:** March 1, 2026  
**Questions?** Contact f.hamzah@tu.edu.sa
