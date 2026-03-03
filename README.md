# Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring

**Corresponding Author:** Hamzah Faraj  
**Email:** f.hamzah@tu.edu.sa  
**Institution:** Department of Science and Technology, Ranyah College, Taif University, Saudi Arabia


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

This repository contains the code, data, and experimental results for our paper on a hybrid edge AI framework combining temporal convolutional networks (TCN), variance-driven dynamic quantization (4-8 bits), knowledge distillation, and hardware-aware neural architecture search (HW-NAS) for power-efficient water quality monitoring on resource-constrained IoT devices.

**Key Results:**
-  **40-45% power savings** over fixed 8-bit quantization baselines
-  **95% prediction accuracy** (10-14% improvement over state-of-the-art)
-  **43M FLOPs** (31% reduction vs CNN-LSTM, 49% vs fixed 8-bit)
-  **20-26 month battery life** on IoT devices (2.5× improvement)
-  **Cross-continental robustness** (average 2.1% accuracy degradation)
-  **32 ms latency** on Raspberry Pi 4 (29% faster than CNN-LSTM)

## Key Contributions

1. **First TCN + HW-NAS integration** for water quality IoT monitoring
   - 31% FLOPs reduction vs CNN-LSTM through parallel temporal processing
   - 29% latency improvement (45ms → 32ms)
   - 13% power savings from eliminating recurrent bottlenecks

2. **Variance-driven adaptive quantization**
   - Dynamic 4-8 bit switching based on real-time signal variance
   - 28-42% power savings while maintaining 95% accuracy
   - Domain-specific thresholds: σ<0.05 (4-bit stable), 0.05≤σ<0.15 (6-bit moderate), σ≥0.15 (8-bit pollution events)

3. **Synergistic optimization pipeline**
   - HW-NAS + distillation + adaptive quantization: 15% non-additive gains
   - Mixed-precision layer assignment reduces power by 33%
   - Knowledge distillation via soft-target MSE regression (T=3 temperature scaling)

4. **Rigorous statistical validation**
   - 500K subset preserves 97.8% geographic correlation with full 20M dataset
   - Learning curve plateau at 500K (diminishing returns: 1.8% potential gain from 40× data)
   - Sensitivity analysis confirms robustness within ±20% threshold perturbations

## Repository Structure

```
water-quality-edge-ai/
├── README.md                 # This file
├── METHODS.md                # Detailed methodology documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
│
├── paper/                    # Publication materials
│   ├── main.tex              # LaTeX manuscript (publication-ready)
│   └── references.bib        # Bibliography
│
├── data/                     # Experimental results and sampling metadata
│   ├── results.csv           # Main performance comparison (7 methods)
│   ├── ablation_results.csv  # Component contribution analysis
│   ├── sensitivity_results.csv # Hyperparameter robustness analysis
│   ├── geographic_results.csv # Cross-continental generalization
│   └── sampling_metadata.txt # 500K subset stratification details
│
├── figures/                  # Publication-quality figures
│   └── figures.py      # figures reproducability
│
├── src/                      # Source code
│   ├── models.py             # CNN-TCN architecture
│   ├── models/               # Model components
│   ├── quantization/         # Adaptive quantization implementation
│   ├── preprocess_unep.py    # UNEP GEMSWater data preprocessing
│   ├── train_teacher.py      # Teacher model training
│   ├── train_student_distill.py # Student distillation training
│   ├── quantize_qat.py       # Quantization-aware training
│   ├── run_dynamic_policy.py # Dynamic quantization policy
│   └── energy_model.py       # Power/energy estimation models
│
└── experiments/              # Experimental configurations
```

## Dataset

Our experiments use a stratified 500,000-record subset (2.5%) of the **UNEP GEMSWater Global Freshwater Quality Archive**:

- **Source:** UNEP GEMSWater Dataset [Zenodo](https://doi.org/10.5281/zenodo.10701676)
- **Total records:** 20,446,832 measurements
- **Stations:** 13,660 across 37 countries
- **Time span:** 1906-2023 (117 years)
- **Parameters:** pH, dissolved oxygen, turbidity, conductivity, NO₃, PO₄, TSS, BOD, COD, temperature

### Sampling Strategy

Our 500K subset was selected using **4-stage stratified sampling** with statistical representativeness guarantees:

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
- Geographic correlation with full archive: **r = 0.978** (p < 0.001)
- Parameter means within **2.3%** of full archive
- Cohen's d < 0.05 (negligible effect size)
- KS test for temporal distribution: **p = 0.42** (no significant difference)

See `data/sampling_metadata.txt` for complete details.

## Experimental Results

### Main Performance Comparison

| Method | Power (W) | RMSE | Accuracy | FLOPs (M) | Model Size (MB) | Latency (ms) |
|--------|-----------|------|----------|-----------|-----------------|---------------|
| Non-AI Baseline | 0.05 | 0.92 | 78.4% | 0.1 | 0.05 | - |
| Fixed 8-bit | 0.38 | 0.76 | 88.5% | 85 | 28.5 | 42 |
| Activation-aware | 0.32 | 0.74 | 89.7% | 78 | 25.2 | 40 |
| TinyML | 0.28 | 0.82 | 82.3% | 52 | 18.4 | 35 |
| CNN-LSTM (FP32) | 0.45 | 0.68 | 92.8% | 95 | 32.8 | 45 |
| CNN-LSTM (Quant) | 0.24 | 0.65 | 91.2% | 62 | 8.2 | 38 |
| **CNN-TCN (Ours)** | **0.21** | **0.62** | **95.0%** | **43** | **6.5** | **32** |

**Improvement over Fixed 8-bit:** 45% power ↓, 18% RMSE ↓, 7% accuracy ↑, 49% FLOPs ↓, 24% latency ↓

### Ablation Study: Component Contributions

| Configuration | Power (W) | RMSE | Accuracy | Impact |
|--------------|-----------|------|----------|--------|
| Full Model (CNN-TCN) | 0.21 | 0.62 | 95.0% | Baseline |
| w/ CNN-LSTM instead | 0.24 | 0.65 | 91.2% | -4% accuracy |
| w/o Adaptive Quant | 0.25 | 0.65 | 93.7% | +19% power |
| w/o Distillation | 0.21 | 0.70 | 89.8% | +13% RMSE |
| w/o HW-NAS | 0.23 | 0.63 | 94.2% | +10% power |
| w/o Mixed Precision | 0.28 | 0.64 | 93.1% | +33% power |
| Fixed 4-bit only | 0.18 | 0.82 | 82.3% | -13% accuracy |

**Key Finding:** TCN + adaptive quantization + HW-NAS synergy yields **15% efficiency gains beyond additive effects**.

### Sensitivity Analysis: Hyperparameter Robustness

| Configuration | Power (W) | RMSE | Accuracy | Notes |
|--------------|-----------|------|----------|-------|
| **Variance Thresholds** | | | | |
| Baseline (τ=0.05, 0.15) | 0.21 | 0.62 | 95.0% | Optimal via grid search |
| Lower (τ=0.04, 0.12) | 0.22 | 0.60 | 95.8% | +7% power, -3% RMSE |
| Higher (τ=0.06, 0.18) | 0.19 | 0.66 | 92.4% | -9% power, +6% RMSE |
| **HW-NAS Regularization** | | | | |
| Baseline (λ_E=0.10) | 0.21 | 0.62 | 95.0% | Balances accuracy-power |
| High (λ_E=0.15) | 0.18 | 0.65 | 93.2% | -12% power, -2.1% accuracy |
| Low (λ_E=0.05) | 0.23 | 0.61 | 95.3% | +8% power, +0.3% accuracy |
| **Quantization Schemes** | | | | |
| Adaptive 4-8 bit (Ours) | 0.21 | 0.62 | 95.0% | Dynamic bit-width allocation |
| Fixed 8-bit | 0.32 | 0.61 | 95.1% | +52% power for +0.1% accuracy |
| Fixed 6-bit | 0.26 | 0.68 | 92.8% | -2.2% accuracy |
| Fixed 4-bit | 0.15 | 0.85 | 82.3% | -12.7% accuracy (unsuitable) |

**Robustness:** Performance remains within **±5%** across ±20% threshold perturbations, confirming practical deployability without extensive tuning.

### Cross-Continental Generalization

Trained on 5 continents, tested on 6th:

| Continent | RMSE | Accuracy | Degradation | Test Samples |
|-----------|------|----------|-------------|---------------|
| North America | 0.64 | 94.3% | -0.7% | 8,420 |
| **Europe** | **0.61** | **95.8%** | **+0.8%** | **10,250** |
| Asia | 0.67 | 93.1% | -2.0% | 7,085 |
| Africa | 0.70 | 91.8% | -3.4% | 3,730 |
| South America | 0.68 | 92.6% | -2.5% | 2,795 |
| Oceania | 0.72 | 90.4% | -4.8% | 1,695 |

**Average degradation:** -2.1% across continents, demonstrating strong geographic robustness.

## Hardware & Software

### Training Environment
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **Training time:** 72 GPU-hours for 500K records
- **Framework:** PyTorch 2.0, Python 3.10
- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Epochs:** 100 (teacher), 150 (student with gradual α reduction)

### Inference Testing
- **Device:** Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72 @ 1.5GHz)
- **Latency:** 32 ms per prediction (estimated via analytical models)
- **Power:** 0.21W average (calibrated with pyRAPL profiling)
- **Battery life:** 20-26 months (10,000mAh @ 5V, estimated)

> **Important Note on Evaluation Methodology**
>
> All **power, energy, latency, and battery life figures** reported in this work are obtained from **analytical and profiling-based models** applied to the processing pipeline, rather than from direct measurements on physically deployed IoT sensor nodes in field conditions. 
>
> **Estimation Methodology:**
> - Power estimates calibrated using **pyRAPL** (CPU) and **nvidia-smi** (GPU) profiling on Raspberry Pi 4 emulation
> - Latency and energy metrics derived from **FLOPs counts**, memory access patterns, and quantization bit-widths
> - Platform-specific constants: α_comp = 0.12 nJ/FLOP, α_mem = 2.3 nJ/byte (empirically calibrated for ARM Cortex-A72)
>
> These values should be interpreted as **analytical estimates under stated assumptions** rather than field-measured results. Physical field deployment validation across diverse climates (temperature extremes, humidity, vibration) remains future work to confirm thermal management, sensor drift, and communication reliability.
>
> See manuscript Section 4 (Methodology) and Section 7.4 (Limitations) for complete details.

### Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pytorch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.14.0
pyRAPL>=0.2.3
```

See `requirements.txt` for complete list.

## Methodology

For detailed methodology including:
- CNN-TCN architecture with dilated causal convolutions
- Variance-driven adaptive quantization policy
- Knowledge distillation via soft-target MSE regression
- Hardware-aware neural architecture search (DARTS-based)
- Analytical complexity and memory analysis

See **[METHODS.md](METHODS.md)** and manuscript **Section 4**.

## Impact & Applications

This framework enables **autonomous, solar-powered monitoring stations** in remote watersheds, directly supporting:

- **UN SDG 6 (Clean Water and Sanitation):** Real-time water quality monitoring in underserved regions
- **UN SDG 13 (Climate Action):** Data-driven environmental governance and pollution tracking

**Practical Deployment Benefits:**
- 60% maintenance cost reduction (biannual vs. quarterly servicing)
- 35-40% total cost of ownership reduction over 2 years
- Enables year-round operation in remote areas (satellite/LoRa connectivity)
- Supports 100-station networks with $35K-60K biennial savings

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Corresponding Author:** Hamzah Faraj  
**Email:** f.hamzah@tu.edu.sa  
**Institution:** Department of Science and Technology, Ranyah College, Taif University, Saudi Arabia

## Acknowledgments

The author acknowledges the Deanship of Graduate Studies and Scientific Research, Taif University, for funding.
