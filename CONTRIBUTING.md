# Contributing to Water Quality Edge AI Project

Thank you for your interest in contributing to our water quality monitoring research! This project aims to advance edge AI techniques for environmental monitoring, and we welcome contributions from the research community.

## 🎯 Project Overview

This repository implements a hybrid edge AI framework combining:
- Temporal Convolutional Networks (TCN)
- Variance-driven adaptive quantization (4-8 bits)
- Knowledge distillation
- Hardware-aware neural architecture search (HW-NAS)

**Goal:** Enable power-efficient water quality monitoring on resource-constrained IoT devices.

## 🔍 Ways to Contribute

### 1. Research Contributions
- **Algorithm improvements:** Enhanced quantization policies, TCN architectures, or NAS strategies
- **New experiments:** Additional baselines, ablation studies, or cross-dataset validation
- **Field deployment:** Physical hardware validation and real-world case studies
- **Extensions:** Graph neural networks for spatial dependencies, Bayesian uncertainty quantification

### 2. Code Contributions
- **Implementation:** Complete missing code modules (marked as "coming soon")
- **Optimization:** Performance improvements, memory efficiency, inference speed
- **Documentation:** Code comments, tutorials, usage examples
- **Testing:** Unit tests, integration tests, edge case handling

### 3. Data & Reproducibility
- **Dataset expansion:** Additional environmental datasets or longer time horizons
- **Reproducibility:** Verification of published results, hyperparameter sensitivity
- **Visualization:** New figures, interactive dashboards, or analysis tools

### 4. Documentation
- **Method clarification:** Detailed explanations of technical components
- **Tutorial development:** Step-by-step guides for deployment
- **Translation:** Documentation in other languages

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for training)
- Raspberry Pi 4 or similar ARM device (for inference testing)

### Installation

```bash
# Clone the repository
git clone https://github.com/drhamzahfaraj/water-quality-edge-ai.git
cd water-quality-edge-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download UNEP GEMSWater dataset (optional)
# Follow instructions in data/README.md
```

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Coverage report
pytest --cov=src tests/

# Specific test module
pytest tests/test_quantization.py -v
```

## 📝 Contribution Guidelines

### Code Style
- Follow **PEP 8** Python style guidelines
- Use **black** for code formatting: `black src/`
- Run **flake8** for linting: `flake8 src/`
- Maximum line length: 100 characters
- Use type hints for function signatures

### Commit Messages
Follow conventional commits format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example:**
```
feat(quantization): Add layer-wise sensitivity analysis

Implement per-layer bit-width optimization based on Hessian
trace estimation. Reduces power by 8% with <1% accuracy loss.

Closes #42
```

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following code style guidelines

3. **Add tests** for new functionality:
   ```python
   # tests/test_your_feature.py
   def test_your_feature():
       # Test implementation
       assert expected == actual
   ```

4. **Update documentation** if needed:
   - README.md for user-facing changes
   - METHODS.md for methodology updates
   - Code docstrings for API changes

5. **Run all tests** and ensure they pass:
   ```bash
   pytest tests/ -v
   black src/
   flake8 src/
   ```

6. **Commit your changes** with clear messages

7. **Push to your fork** and submit a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **PR Description Template:**
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why is this change needed?

   ## Changes
   - List of specific changes
   - New files or modules
   - Modified functionality

   ## Testing
   - How was this tested?
   - New test cases added?

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests pass locally
   - [ ] Documentation updated
   - [ ] Commit messages are clear
   ```

### Review Process
- Maintainers will review PRs within 5-7 business days
- Address review comments and push updates
- Once approved, maintainers will merge your PR
- Your contribution will be acknowledged in release notes

## 💡 Research Collaboration

### Co-authorship Opportunities
Significant research contributions may lead to co-authorship on future publications. Criteria:
- Substantial algorithmic improvements (>10% performance gain)
- Novel experimental validation or field deployment studies
- Major code infrastructure development

Contact f.hamzah@tu.edu.sa to discuss collaboration.

### Dataset Contributions
If you have water quality datasets to share:
- Ensure proper licensing (CC BY 4.0 or similar)
- Provide metadata (location, time range, parameters)
- Document preprocessing steps
- Submit via pull request to `data/` directory

## 🐛 Reporting Issues

### Bug Reports
Use the GitHub issue tracker with the "bug" label:

```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
1. Steps to reproduce
2. Code snippets if applicable
3. Error messages

**Expected behavior**
What should happen instead?

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python version: [e.g., 3.10.5]
 - PyTorch version: [e.g., 2.0.1]
 - GPU/Device: [e.g., NVIDIA RTX 4090]

**Additional context**
Screenshots, logs, or other relevant info
```

### Feature Requests
Use the "enhancement" label:

```markdown
**Feature description**
Clear description of proposed feature

**Motivation**
Why is this feature valuable?

**Proposed implementation**
Technical approach (if known)

**Alternatives considered**
Other approaches you've thought about
```

## ❓ Questions & Discussion

- **GitHub Discussions:** For general questions and research discussions
- **Issues:** For bug reports and feature requests
- **Email:** f.hamzah@tu.edu.sa for research collaboration

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🎉 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Acknowledged in future publications (for significant research contributions)

Thank you for helping advance edge AI for environmental monitoring! 🌍💧
