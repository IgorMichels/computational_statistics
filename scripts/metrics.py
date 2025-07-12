import json
from typing import List, Tuple

import numpy as np
from numba import jit


def acf_1d(x: np.ndarray, max_lag: int = 40):
    """Compute autocorrelation function for a 1D time series.

    Uses FFT to compute the autocorrelation function efficiently.

    Args:
        x: Input time series.
        max_lag: Maximum lag to compute autocorrelation for.

    Returns:
        Autocorrelation values from lag 0 to max_lag.
    """
    n = len(x)
    x = x - x.mean()
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    x_padded = np.zeros(n_fft)
    x_padded[:n] = x

    X = np.fft.fft(x_padded)
    ac = np.fft.ifft(X * np.conj(X)).real[: max_lag + 1]

    return ac / ac[0]


def ess_1d(x: np.ndarray, max_lag: int = 40):
    """Compute effective sample size for a 1D time series.

    Uses the autocorrelation function to estimate the effective
    sample size accounting for serial correlation.

    Args:
        x: Input time series.
        max_lag: Maximum lag to use in autocorrelation computation.

    Returns:
        Effective sample size.
    """
    rho = acf_1d(x, max_lag)[1:]
    pos = rho[rho > 0]
    return len(x) / (1 + 2 * pos.sum())


@jit(nopython=True)
def rhat_scalar_numba(chains_array: np.ndarray):
    """Compute R-hat convergence diagnostic using Numba optimization.

    Numba-optimized implementation of the potential scale reduction factor
    calculation for improved performance.

    Args:
        chains_array: 2D array where each row is a chain.

    Returns:
        R-hat value.
    """
    m, n = chains_array.shape
    chain_means = np.zeros(m)
    for i in range(m):
        chain_means[i] = np.mean(chains_array[i])

    W = 0.0
    for i in range(m):
        chain_var = 0.0
        for j in range(n):
            chain_var += (chains_array[i, j] - chain_means[i]) ** 2
        W += chain_var / (n - 1)
    W /= m

    overall_mean = np.mean(chain_means)
    B = 0.0
    for i in range(m):
        B += (chain_means[i] - overall_mean) ** 2
    B = n * B / (m - 1)

    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def rhat_scalar(chains: List[np.ndarray]):
    """Compute R-hat convergence diagnostic for multiple chains.

    Computes the potential scale reduction factor (R-hat) to assess
    convergence of MCMC chains. Values close to 1 indicate good convergence.

    Args:
        chains: List of chains for the same parameter.

    Returns:
        R-hat value (should be close to 1 for convergence).
    """
    n = min(len(c) for c in chains)
    chains_array = np.array([c[:n] for c in chains])

    return rhat_scalar_numba(chains_array)


def compute_credible_intervals(pooled: np.ndarray, alpha: float = 0.05):
    """Compute credible intervals from posterior samples.

    Calculates equal-tailed credible intervals from pooled posterior samples.

    Args:
        pooled: Pooled posterior samples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds of the credible interval.
    """
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    percentiles = np.percentile(pooled, [lower_percentile, upper_percentile], axis=0)

    return percentiles[0], percentiles[1]


def create_metrics(
    chains: List[np.ndarray], data_name: str, param_name: str = "mu"
) -> Tuple[np.ndarray, List[float], List[float], np.ndarray, np.ndarray]:
    """Create comprehensive convergence and posterior metrics from MCMC chains.

    Computes posterior mean, R-hat convergence diagnostic, effective sample size,
    and credible intervals for each parameter from multiple MCMC chains. The
    computed metrics are also saved to a JSON file in the data directory.

    Args:
        chains: List of MCMC chains, each containing samples for K parameters.
        data_name: Name of the dataset (used for saving metrics file).
        param_name: Name of the parameter type ("mu" or "sigma2").

    Returns:
        Tuple containing:
        - param_mean: Posterior mean for each parameter
        - rhat: R-hat convergence diagnostic for each parameter
        - ess: Effective sample size for each parameter
        - ci_lower: Lower bounds of 95% credible intervals
        - ci_upper: Upper bounds of 95% credible intervals
    """
    K = chains[0].shape[1]
    pooled = np.vstack(chains)
    param_mean = pooled.mean(axis=0)

    ci_lower, ci_upper = compute_credible_intervals(pooled)

    rhat = [rhat_scalar([c[:, k] for c in chains]) for k in range(K)]
    ess = [ess_1d(pooled[:, k]) for k in range(K)]

    # Load existing metrics if they exist, otherwise create new dict
    metrics_file = f"../data/{data_name}/metrics.json"
    try:
        with open(metrics_file, encoding="utf-8") as f:
            metrics_dict = json.load(f)
    except FileNotFoundError:
        metrics_dict = {}

    # Update metrics dict with current parameter
    metrics_dict[f"{param_name}_mean"] = param_mean.tolist()
    metrics_dict[f"{param_name}_rhat"] = rhat
    metrics_dict[f"{param_name}_ess"] = ess
    metrics_dict[f"{param_name}_ci_lower"] = ci_lower.tolist()
    metrics_dict[f"{param_name}_ci_upper"] = ci_upper.tolist()

    # Save updated metrics
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return param_mean, rhat, ess, ci_lower, ci_upper
