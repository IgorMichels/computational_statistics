# Computational Statistics - FGV/EMAp

Final project for the Computational Statistics class of the FGV/EMAp PhD program.

## Overview

This project implements MCMC methods for Bayesian inference on Gaussian mixture models:

1. **Gibbs Sampler** - Standard MCMC with conditional distributions
2. **Tempered Transitions** - MCMC with temperature ladders
3. **Comparison Framework** - Performance analysis between methods

## Setup

To get started with the project, you need to set up the virtual environment and install all dependencies. The project uses `uv` for fast dependency management and includes development tools for code quality.

```bash
make local
```

## Structure

The project is organized into several key directories containing the core implementation, data files, and generated outputs:

```
computational_statistics/
├── scripts/                     # Implementation
│   ├── samplers.py              # Core MCMC algorithms
│   ├── generate_data.py         # Data generation
│   ├── gibbs_sampler.py         # Gibbs runner
│   ├── tempered_transitions.py  # Tempered transitions runner
│   ├── comparison_tool.py       # Method comparison
│   ├── metrics.py               # Diagnostics
│   ├── plots.py                 # Visualization
│   └── state.py                 # State management
├── figures/                     # Generated plots
├── data/                        # Data files (local)
└── .venv/                       # Virtual environment (local)
```

## Usage

The workflow involves generating synthetic mixture data, running individual MCMC methods, and comparing their performance. All scripts should be run from the `scripts/` directory.

### Basic Commands

```bash
source .venv/bin/activate
cd scripts

# Generate data
python generate_data.py

# Run methods
python gibbs_sampler.py --data example_1 --K 4 --n_iter 5000
python tempered_transitions.py --data example_1 --K 4 --n_iter 5000

# Compare methods
python comparison_tool.py all --data example_1
```

### Available Tools

The project includes several development tools for maintaining code quality and formatting:

```bash
make check      # Code quality checks
make format     # Code formatting
make clean      # Clean temporary files
```

## Implementation

### Gibbs Sampler
- Sequential sampling of assignments, weights, and means
- Conjugate priors for efficiency
- Standard MCMC diagnostics

### Tempered Transitions
- Temperature ladder implementation
- Configurable temperature schedules
- Acceptance rate tracking

### Comparison Framework
- Side-by-side method execution
- Performance metrics (ESS, R-hat, runtime)
- Visualization suite

## Output

### Files Generated
- `data/{experiment}/data.npy` - Synthetic data
- `data/{experiment}/metrics.json` - Statistics
- `data/{experiment}/comparison_results/` - Comparison data
- `figures/{experiment}/` - Diagnostic plots

### Metrics
- Posterior means and credible intervals
- R-hat convergence diagnostics
- Effective sample sizes
- Runtime comparison
- Acceptance rates
