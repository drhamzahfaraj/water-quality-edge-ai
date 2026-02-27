# Data

## UNEP GEMS/Water Subset

**File:** `unep_subset.csv` (not included in repository due to size)

### Dataset Description

This directory should contain the curated subset of the UNEP GEMS/Water Global Freshwater Quality Archive used in the paper.

**Specifications:**
- **Total samples:** 500,000 records
- **Indicators:** 10 key parameters
  1. pH
  2. Dissolved Oxygen (DO)
  3. Turbidity
  4. Conductivity
  5. Nitrate (NO₃)
  6. Phosphate (PO₄)
  7. Total Suspended Solids (TSS)
  8. Biochemical Oxygen Demand (BOD)
  9. Chemical Oxygen Demand (COD)
  10. Temperature

- **Stations:** 13,660 monitoring stations
- **Countries:** 37 countries
- **Time period:** 1906–2023

### Preprocessing

The raw UNEP GEMS/Water data was processed using `src/preprocess_unep.py` with the following steps:

1. **Indicator selection:** Filtered to 10 most prevalent and relevant indicators
2. **Missing value handling:** Mean imputation per station and parameter (~15% missing)
3. **Outlier removal:** IQR method (1.5 × IQR rule)
4. **Normalization:** Min-max scaling to [0, 1]
5. **Windowing:** 24-hour input windows, 24-hour prediction horizon
6. **Augmentation:** 20% of training data augmented with Gaussian noise (σ=0.1)

### Data Split

- **Training:** 80% (400,000 samples)
- **Validation:** 10% (50,000 samples)  
- **Test:** 10% (50,000 samples)

Split is:
- **Temporally consistent:** Earlier periods for training, later for testing
- **Stratified by station:** Preserves geographic diversity

### Data Format

Expected CSV structure:

```csv
station_id,date,pH,dissolved_oxygen,turbidity,conductivity,nitrate,phosphate,total_suspended_solids,BOD,COD,temperature
...
```

Each row represents measurements at a single station and timestamp.

### Obtaining the Data

The original UNEP GEMS/Water dataset is publicly available under CC BY 4.0 license:

**Source:** [UNEP GEMS/Water on Zenodo](https://zenodo.org/records/10623615)

**Citation:**
> Virro, H., Amatulli, G., Kmoch, A., Shen, L., & Uuemaa, E. (2024). UNEP GEMS/Water Global Freshwater Quality Database, 1906-2023 (1.6.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10623615

**To reproduce the subset:**

```bash
# Download raw UNEP data from Zenodo
wget https://zenodo.org/record/10623615/files/unep_gemswater_full.csv

# Run preprocessing
python src/preprocess_unep.py \
  --input unep_gemswater_full.csv \
  --output data/unep_subset.csv \
  --target-samples 500000
```

### File Size Note

The processed `unep_subset.csv` file is approximately 150 MB and is not included in this Git repository. Users should download and preprocess the raw data as described above.

### Validation

After preprocessing, verify the dataset:

```python
import pandas as pd

df = pd.read_csv('data/unep_subset.csv')

print(f"Total samples: {len(df):,}")
print(f"Indicators: {len([c for c in df.columns if c not in ['station_id', 'date']])}")
print(f"Stations: {df['station_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Value range: [{df.select_dtypes(include='number').min().min():.3f}, {df.select_dtypes(include='number').max().max():.3f}]")
```

Expected output:
```
Total samples: 500,000
Indicators: 10
Stations: ~13,660
Date range: [varies]
Missing values: 0
Value range: [0.000, 1.000]  # After normalization
```

## License

The UNEP GEMS/Water data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Usage must include proper attribution to UNEP and the original data providers.
