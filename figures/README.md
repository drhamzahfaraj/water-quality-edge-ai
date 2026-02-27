# Figures

This directory contains all figures referenced in the paper.

## Figure Files

### Figure 1: Framework Overview
**File:** `framework.pdf`

**Description:** Overall architecture of the proposed framework showing:
- Teacher model training (CNN-LSTM/CNN-TCN, 2.10M params)
- Student model compression and distillation (CNN-TCN, 1.20M params, 43% reduction)
- Quantization pipeline (PTQ + QAT with mixed precision)
- Dynamic precision policy with variance-threshold controller
- Edge inference workflow

**Referenced in:** Figure 1, Section 3 (Methodology)

---

### Figure 2: Energy-Accuracy Comparison
**File:** `energy_accuracy_comparison.pdf`

**Description:** Main results comparing all methods on the UNEP GEMS/Water test set:
- **(a)** Normalized accuracy (1-RMSE) for all baselines and proposed method
- **(b)** Energy consumption per inference (mJ)

**Key findings:**
- Proposed method achieves 0.76 in 1-RMSE (5.6% gain over fixed 8-bit)
- Energy reduced from 2.00 mJ to 1.45 mJ (27.5% reduction)

**Referenced in:** Figure 2, Section 4.3 (Main Results), corresponds to Table 1

---

### Figure 3: Noise Robustness
**File:** `noise_robustness.pdf`

**Description:** Robustness comparison under additive Gaussian noise:
- X-axis: Noise level σ (0.00 to 0.20)
- Y-axis: Normalized accuracy (1-RMSE)
- Two lines: Fixed 8-bit baseline vs. Proposed dynamic method

**Key findings:**
- At σ=0.20, proposed method retains 97% of clean performance
- Fixed 8-bit retains only 94%
- Performance gap increases with noise level

**Referenced in:** Figure 3, Section 4.5 (Robustness and Generalization)

---

## Reproducing Figures

All figures can be regenerated from the experimental results using:

```python
python src/generate_figures.py --results-dir experiments/results/ --output-dir figures/
```

Or individually:

```python
# Figure 2
python src/generate_figures.py --figure energy_accuracy --input experiments/results/results_main.csv

# Figure 3
python src/generate_figures.py --figure noise_robustness --input experiments/results/results_robustness_noise.csv
```

## Figure Specifications

- **Format:** PDF (vector graphics for publication quality)
- **Resolution:** 300 DPI (for raster elements)
- **Fonts:** Serif fonts for academic publication
- **Color scheme:** Colorblind-friendly palette
- **Size:** Designed for two-column IEEE/ACM format

## Citation

If you reuse these figures, please cite:

```bibtex
@article{faraj2025water,
  title={Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring},
  author={Faraj, Hamzah and Soliman, Mohamed S. and Alshahri, Abdullah H.},
  journal={[Journal Name]},
  year={2025}
}
```
