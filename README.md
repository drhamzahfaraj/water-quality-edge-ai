# Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring

This repository contains all code, data, and experimental results for the paper:

**"Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring"**  
by Hamzah Faraj, Mohamed S. Soliman, and Abdullah H. Alshahri  
Taif University, Saudi Arabia

## Repository Structure

```
├── data/                          # Curated UNEP GEMS/Water subset
│   ├── unep_subset.csv           # 500,000 records, 10 indicators
│   └── README.md                 # Data provenance and preprocessing details
├── src/                          # Source code
│   ├── preprocess_unep.py        # Data preprocessing pipeline
│   ├── models.py                 # Teacher and student architectures
│   ├── train_teacher.py          # Train full-precision teacher
│   ├── train_student_distill.py  # Train student with knowledge distillation
│   ├── quantize_qat.py           # Post-training quantization + QAT
│   ├── run_dynamic_policy.py     # Variance-aware dynamic inference
│   └── energy_model.py           # Software-based energy estimation
├── experiments/                   # Experimental configurations and results
│   ├── config/                   # YAML configuration files
│   └── results/                  # CSV files with all experimental results
├── figures/                       # Paper figures (PDF)
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

## Key Results

The proposed dynamic mixed-precision CNN-TCN framework achieves:

- **27.5% energy reduction** (2.00 mJ → 1.45 mJ per inference)
- **5.6% accuracy improvement** (0.72 → 0.76 in 1-RMSE)
- Compared to fixed 8-bit CNN-TCN baseline on Raspberry-Pi-class hardware

## Reproducing Experiments

### Setup

```bash
pip install -r requirements.txt
```

### Main Experiments (Table 1)

```bash
# Train teacher model
python src/train_teacher.py --config experiments/config/config_main.yaml

# Train student with distillation
python src/train_student_distill.py --config experiments/config/config_main.yaml

# Apply quantization (PTQ + QAT)
python src/quantize_qat.py --config experiments/config/config_main.yaml

# Run dynamic policy evaluation
python src/run_dynamic_policy.py --config experiments/config/config_main.yaml
```

Results are saved to `experiments/results/results_main.csv`

### Ablation Studies

**Compression & Distillation (Table 2):**
```bash
python src/train_student_distill.py --config experiments/config/config_ablation_compression.yaml
```

**Static vs Dynamic Quantization (Table 3):**
```bash
python src/run_dynamic_policy.py --config experiments/config/config_ablation_quantization.yaml
```

### Robustness Experiments

**Noise robustness (Figure 3):**
```bash
python src/run_dynamic_policy.py --config experiments/config/config_robustness.yaml --noise-levels 0.05,0.10,0.20
```

**Missing data:**
```bash
python src/run_dynamic_policy.py --config experiments/config/config_robustness.yaml --missing-rates 0.1,0.2,0.3
```

**Extreme events:**
```bash
python src/run_dynamic_policy.py --config experiments/config/config_robustness.yaml --extreme-subset
```

## Dataset

The curated subset is derived from the UNEP GEMS/Water Global Freshwater Quality Archive:
- 500,000 samples
- 10 indicators: pH, DO, turbidity, conductivity, NO₃, PO₄, TSS, BOD, COD, temperature
- 13,660 monitoring stations across 37 countries

Original data available at: [UNEP GEMS/Water on Zenodo](https://zenodo.org/records/10623615)

## Citation

If you use this code or data, please cite:

```bibtex
@article{faraj2025water,
  title={Optimizing Dynamic Quantization in Edge AI for Power-Efficient Water Quality Monitoring},
  author={Faraj, Hamzah and Soliman, Mohamed S. and Alshahri, Abdullah H.},
  journal={[Journal Name]},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgements

This work was supported by the Deanship of Graduate Studies and Scientific Research, Taif University.

## Contact

Hamzah Faraj - f.hamzah@tu.edu.sa  
Department of Science and Technology, Ranyah College, Taif University
