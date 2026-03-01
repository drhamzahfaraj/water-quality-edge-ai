# Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring

**Authors:** Hamzah Faraj, Mohamed S. Soliman, Abdullah H. Alshahri  
**Affiliation:** Taif University, Saudi Arabia  
**Paper Status:** Submitted February 2026

## Abstract

This repository contains the code, data, and experimental results for our paper on hybrid edge AI framework combining temporal convolutional networks (TCN), variance-driven adaptive quantization, knowledge distillation, and hardware-aware neural architecture search (HW-NAS) for power-efficient water quality monitoring.

**Key Results:**
- 40-45% power savings over fixed 8-bit quantization
- 95% prediction accuracy (10-14% improvement)
- 43M FLOPs (31% reduction vs CNN-LSTM)
- 20-26 month battery life on IoT devices

## Repository Structure

```
water-quality-edge-ai/
├── METHODS.md                 # Detailed methodology documentation
├── data/                      # Experimental results and sampling metadata
│   ├── results.csv            # Main performance comparison (7 methods)
│   ├── ablation_results.csv   # Component contribution analysis
│   ├── geographic_results.csv # Cross-continental generalization
│   └── sampling_metadata.txt  # 500K subset stratification details
├── figures/                  # Publication-quality figures (5 total)
│   ├── learning_curve.png     # Training set size vs performance
│   ├── main_results.png       # Baseline comparison (7 methods)
│   ├── tcn_vs_lstm.png        # Architecture comparison across variance
│   ├── ablation_study.png     # Component ablation analysis
│   └── geographic_generalization.png  # Cross-continental performance
├── src/                      # Source code (coming soon)
│   ├── models/               # CNN-TCN architecture
│   ├── quantization/         # Adaptive quantization implementation
│   ├── training/             # Training scripts
│   └── evaluation/           # Benchmarking and evaluation
├── experiments/              # Experimental configurations
└── requirements.txt          # Python dependencies
```

## Dataset

Our experiments use a stratified 500,000-record subset (2.5%) of the **UNEP GEMSWater Global Freshwater Quality Archive**:
- **Source:** [Zenodo](https://doi.org/10.5281/zenodo.10701676) (CC BY 4.0)
- **Total records:** 20,446,832 measurements
- **Stations:** 13,660 across 37 countries
- **Time span:** 1906-2023
- **Parameters:** pH, dissolved oxygen, turbidity, conductivity, NO₃, PO₄, TSS, BOD, COD, temperature

### Sampling Strategy

Our 500K subset was selected using **4-stage stratified sampling**:

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
- Geographic correlation with full archive: r = 0.978 (p < 0.001)
- Parameter means within 2.3% of full archive
- KS test for temporal distribution: p = 0.42 (no significant difference)

See `data/sampling_metadata.txt` for complete details.

## Experimental Results

### Main Performance Comparison

| Method | Power (W) | RMSE | Accuracy | FLOPs (M) | Model Size (MB) |
|--------|-----------|------|----------|-----------|------------------|
| Non-AI Baseline | 0.05 | 0.92 | 78.4% | 0.1 | 0.05 |
| Fixed 8-bit | 0.38 | 0.76 | 88.5% | 85 | 28.5 |
| Activation-aware | 0.32 | 0.74 | 89.7% | 78 | 25.2 |
| TinyML | 0.28 | 0.82 | 82.3% | 52 | 18.4 |
| CNN-LSTM (FP32) | 0.45 | 0.68 | 92.8% | 95 | 32.8 |
| CNN-LSTM (Quant) | 0.24 | 0.65 | 91.2% | 62 | 8.2 |
| **CNN-TCN (Ours)** | **0.21** | **0.62** | **95.0%** | **43** | **6.5** |

**Improvement over Fixed 8-bit:** 45% power, 18% RMSE, 7% accuracy, 49% FLOPs

### Ablation Study

| Configuration | Power (W) | RMSE | Accuracy | Impact |
|--------------|-----------|------|----------|--------|
| Full Model (CNN-TCN) | 0.21 | 0.62 | 95.0% | Baseline |
| w/ CNN-LSTM instead | 0.24 | 0.65 | 91.2% | -4% accuracy |
| w/o Adaptive Quant | 0.25 | 0.65 | 93.7% | +19% power |
| w/o Distillation | 0.21 | 0.70 | 89.8% | +13% RMSE |
| w/o HW-NAS | 0.23 | 0.63 | 94.2% | +10% power |
| w/o Mixed Precision | 0.28 | 0.64 | 93.1% | +33% power |
| Fixed 4-bit only | 0.18 | 0.82 | 82.3% | -13% accuracy |

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

**Average degradation:** -2.1% across continents

## Key Contributions

1. **First TCN + HW-NAS integration** for water quality IoT monitoring
   - 31% FLOPs reduction vs CNN-LSTM
   - 29% latency improvement (45ms → 32ms)
   - 13% power savings from parallel processing

2. **Variance-driven adaptive quantization**
   - Dynamic 4-8 bit switching based on real-time variance
   - 28-42% power savings while maintaining accuracy
   - Thresholds: σ<0.05 (4-bit), 0.05≤σ<0.15 (6-bit), σ≥0.15 (8-bit)

3. **Synergistic optimization pipeline**
   - HW-NAS + distillation: 15% combined gains (non-additive)
   - Mixed-precision layer assignment reduces power by 33%
   - Knowledge distillation: 25-35% compression with <2% accuracy loss

4. **Rigorous statistical validation**
   - 500K subset preserves 97.8% geographic correlation
   - Learning curve shows plateau at 500K (1.8% potential gain from full 20M dataset)
   - Cross-continental RMSE variation <8%

## Hardware & Software

**Training:**
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Time: 72 GPU-hours for 500K records
- Framework: PyTorch 2.0, Python 3.10

**Inference Testing:**
- Device: Raspberry Pi 4 Model B (4GB RAM, ARM Cortex-A72)
- Latency: 32 ms per prediction (estimated)
- Power: 0.21W average (estimated via analytical models)
- Battery life: 20-26 months (10,000mAh @ 5V, estimated)

> **⚠️ Simulation-Based Evaluation:** All power, energy, latency, and battery life figures are obtained from **analytical and profiling-based models** applied to the processing pipeline, rather than from direct measurements on deployed IoT sensor nodes. Power estimates are calibrated using pyRAPL (CPU) and nvidia-smi (GPU) profiling, while latency and energy metrics are derived from FLOPs, memory access patterns, and quantization bit-widths. These values should be interpreted as **analytical estimates under stated assumptions** rather than field-measured results.

**Dependencies:**
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

For detailed methodology including dataset preprocessing, TCN architecture, variance-driven quantization, knowledge distillation, and HW-NAS, see **[METHODS.md](METHODS.md)**.

## Citation

If you use this code or data, please cite:

```bibtex
@article{faraj2026optimizing,
  title={Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring},
  author={Faraj, Hamzah and Soliman, Mohamed S. and Alshahri, Abdullah H.},
  journal={Heliyon},
  year={2026},
  publisher={Elsevier},
  note={Submitted February 2026}
}
```

## License

MIT License - see LICENSE file for details

## Contact

**Corresponding Author:** Hamzah Faraj  
**Email:** f.hamzah@tu.edu.sa  
**Institution:** Department of Science and Technology, Ranyah College, Taif University, Saudi Arabia

## Acknowledgments

- UNEP GEMSWater team for providing the global freshwater quality dataset
- Taif University for computational resources
- PyTorch and Plotly communities for open-source tools

---

**Repository Status:** ✅ Methodology documented | ✅ Figures generated | ✅ Data published | 🔄 Code coming soon

**Last Updated:** March 1, 2026
