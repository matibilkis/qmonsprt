# Sequential Hypothesis Testing for Continuously-Monitored Quantum Systems

[![DOI](https://img.shields.io/badge/DOI-10.22331%2Fq--2024--03--20--1289-blue)](https://doi.org/10.22331/q-2024-03-20-1289)
[![arXiv](https://img.shields.io/badge/arXiv-2307.14954-b31b1b)](https://arxiv.org/abs/2307.14954v3)

This repository contains the numerical code supporting the research paper:

> **Sequential hypothesis testing for continuously-monitored quantum systems**  
> Giulio Gasbarri, Matias Bilkis, Elisabet Roda-Salichs, and John Calsamiglia  
> *Quantum* **8**, 1289 (2024)  
> [https://quantum-journal.org/papers/q-2024-03-20-1289/](https://quantum-journal.org/papers/q-2024-03-20-1289/)

## Overview

This codebase implements sequential hypothesis testing strategies for continuously-monitored quantum systems. The framework allows real-time analysis of measurement signals, enabling experiments to conclude as soon as the underlying hypothesis can be identified with a certified prescribed success probability. The implementation focuses on optomechanical systems under continuous homodyne detection in the Gaussian regime.

### Key Features

- **Sequential hypothesis testing**: Real-time analysis of measurement signals with adaptive stopping criteria
- **Quantum optomechanical simulations**: Simulation of continuously-monitored quantum systems
- **Multiple model support**: Supports mechanical damping, frequency estimation, and force detection scenarios
- **Efficient numerical integration**: Optimized stochastic differential equation solvers using Numba JIT compilation
- **Likelihood ratio computation**: Real-time log-likelihood ratio tracking for hypothesis discrimination

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Numba
- Matplotlib
- tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/matibilkis/qmonsprt.git
cd qmonsprt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
qmonsprt/
├── numerics/              # Core numerical routines
│   ├── integration/       # Stochastic differential equation integration
│   │   ├── integrate.py           # Main integration routine for damping estimation
│   │   ├── integrate_freq_force.py  # Integration for frequency/force estimation
│   │   └── steps.py               # Numerical integration steps (Robler, Ikpw)
│   └── utilities/         # Utility functions
│       ├── misc.py                # General utilities and model definitions
│       ├── misc_freq.py           # Frequency estimation utilities
│       └── misc_force.py           # Force detection utilities
├── analysis/              # Analysis and plotting scripts
│   ├── main_likelihood.py         # Likelihood analysis
│   ├── test_frequency.py          # Frequency estimation tests
│   ├── test_force.py              # Force detection tests
│   └── *_plots.py                 # Visualization scripts
├── HPC/                   # High-performance computing scripts
│   └── condor_*.sub              # HTCondor submission scripts
└── giulio_julia/          # Julia implementation (reference)

```

## Usage

### Basic Integration

Run a single trajectory simulation:

```bash
python numerics/integration/integrate.py --itraj 1 --mode damping --dt 1e-4 --total_time 8.0
```

### Batch Processing

Run multiple trajectories:

```bash
python run.py --seed 10 --mode damping --dt 1e-4 --total_time 50 --ppp 1000
```

### Multiprocessing

Run parallel simulations:

```bash
python mp_run.py --itraj 1
```

### Analysis

Analyze likelihood ratios and stopping times:

```bash
python analysis/main_likelihood.py
```

## Mathematical Background

The code simulates continuously-monitored quantum systems described by stochastic differential equations:

```
dx_t = (A_θ - χ(Σ_t) C) x_t dt + χ(Σ_t) dy
dΣ_t = A_θ Σ_t + Σ A_θ^T + D - χ(Σ_t)^T χ(Σ_t)
dy = C x_t dt + C^{-1} dW_t
```

where:
- `x_t` is the system state (first moments)
- `Σ_t` is the covariance matrix (second moments)
- `dy` is the measurement outcome
- `A_θ`, `C`, `D` are system matrices parameterized by `θ`
- `χ(Σ)` is the Kalman gain

The sequential hypothesis test uses log-likelihood ratios to decide between hypotheses `H₀` and `H₁` in real-time.

## Model Parameters

The system is parameterized by:
- `γ`: Damping rate
- `ω`: Frequency
- `n`: Thermal occupation number
- `η`: Measurement efficiency
- `κ`: Measurement strength

## Testing

The repository includes a comprehensive test suite. To run tests:

### Using pytest (recommended)

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=numerics --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v
```

### Using the test runner script

```bash
python3 tests/run_tests.py
```

### Test Structure

- `tests/test_integration.py`: Tests for numerical integration routines (Ikpw, Robler_step)
- `tests/test_utilities.py`: Tests for utility functions (path handling, stopping times, etc.)
- `tests/test_euler_update.py`: Tests for Euler update and likelihood computation
- `tests/conftest.py`: Shared pytest fixtures and configuration

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Gasbarri2024sequential,
  doi = {10.22331/q-2024-03-20-1289},
  url = {https://doi.org/10.22331/q-2024-03-20-1289},
  title = {Sequential hypothesis testing for continuously-monitored quantum systems},
  author = {Gasbarri, Giulio and Bilkis, Matias and Roda-Salichs, Elisabet and Calsamiglia, John},
  journal = {{Quantum}},
  volume = {8},
  pages = {1289},
  month = mar,
  year = {2024}
}
```

## Authors

- **Giulio Gasbarri** - Física Teòrica: Informació i Fenòmens Quàntics, Universitat Autònoma de Barcelona
- **Matias Bilkis** - Física Teòrica: Informació i Fenòmens Quàntics, Universitat Autònoma de Barcelona; Computer Vision Center, Universitat Autònoma de Barcelona
- **Elisabet Roda-Salichs** - Física Teòrica: Informació i Fenòmens Quàntics, Universitat Autònoma de Barcelona
- **John Calsamiglia** - Física Teòrica: Informació i Fenòmens Quàntics, Universitat Autònoma de Barcelona

## License

This project is associated with the research published in Quantum journal under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Acknowledgments

This work was supported by research at the Universitat Autònoma de Barcelona. The codebase has been developed to support the research on sequential hypothesis testing for quantum systems.

## Contributing

This is a research codebase. For questions or issues, please open an issue on GitHub or contact the authors.

