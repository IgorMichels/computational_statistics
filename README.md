# Computational Statistics - FGV/EMAp

Final project for the Computational Statistics class of the FGV/EMAp PhD program, taught by [Prof. Luiz Max Carvalho](https://emap.fgv.br/en/professors/luiz-max-fagundes-de-carvalho).

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
│   ├── generate_data.py         # Synthetic data generation
│   ├── gibbs_sampler.py         # Gibbs Sampler runner
│   ├── tempered_transitions.py  # Tempered Transitions runner
│   ├── comparison_tool.py       # Method comparison & visualization
│   ├── metrics.py               # Performance metrics
│   ├── plots.py                 # Visualization routines
│   ├── utils.py                 # Shared utility functions
│   ├── run_all_tests.py         # Automated test suite
│   ├── results.py               # Metric summarization helper
│   └── state.py                 # Sampling state representation
├── figures/                     # Experiment plots (example_1, example_2, ...)
├── data/                        # Local data files
└── .venv/                       # Local virtual environment
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

# Summarize metrics across experiments
python results.py

# Run full test suite, including data generation and metrics summary
python run_all_tests.py
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

### Utilities & Diagnostics
- Common helper functions and transformations (`utils.py`)
- Automated test execution (`run_all_tests.py`)
- Metric summarization across experiments (`results.py`)

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
