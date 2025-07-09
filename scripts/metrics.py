import json
from typing import List, Tuple

import numpy as np


def acf_1d(x: np.ndarray, max_lag: int = 40):
    """Compute autocorrelation function for a 1D time series.

    Args:
        x: Input time series.
        max_lag: Maximum lag to compute autocorrelation for.

    Returns:
        Autocorrelation values from lag 0 to max_lag.
    """
    n = len(x)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[n - 1 : n + max_lag] / n
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


def rhat_scalar(chains: List[np.ndarray]):
    """Compute R-hat convergence diagnostic for multiple chains.

    Computes the potential scale reduction factor (R-hat) to assess
    convergence of MCMC chains.

    Args:
        chains: List of chains for the same parameter.

    Returns:
        R-hat value (should be close to 1 for convergence).
    """
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    means = np.array([c.mean() for c in chains])
    W = np.array([c.var(ddof=1) for c in chains]).mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def credible_interval(samples: np.ndarray, alpha: float = 0.05):
    """Compute credible interval from posterior samples.

    Args:
        samples: Posterior samples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds of the credible interval.
    """
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return lower, upper


def create_metrics(
    chains: List[np.ndarray], K: int, data_name: str
) -> Tuple[np.ndarray, List[float], List[float], np.ndarray, np.ndarray]:
    """Create comprehensive metrics from MCMC chains.

    Computes posterior mean, R-hat convergence diagnostic, effective sample size,
    and credible intervals for each parameter from multiple MCMC chains. The
    computed metrics are also saved to a JSON file in the data directory.

    Args:
        chains: List of MCMC chains, each containing samples for K parameters.
        K: Number of parameters (mixture components).
        data_name: Name of the dataset (used for saving metrics file).

    Returns:
        Tuple containing:
        - mu_mean: Posterior mean for each parameter
        - rhat: R-hat convergence diagnostic for each parameter
        - ess: Effective sample size for each parameter
        - ci_lower: Lower bounds of 95% credible intervals
        - ci_upper: Upper bounds of 95% credible intervals
    """
    pooled = np.vstack(chains)
    mu_mean = pooled.mean(axis=0)
    rhat = [rhat_scalar([c[:, k] for c in chains]) for k in range(K)]
    ess = [ess_1d(pooled[:, k]) for k in range(K)]

    ci_lower = np.array([credible_interval(pooled[:, k])[0] for k in range(K)])
    ci_upper = np.array([credible_interval(pooled[:, k])[1] for k in range(K)])

    with open(f"../data/{data_name}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mu_mean": mu_mean.tolist(),
                "rhat": rhat,
                "ess": ess,
                "ci_lower": ci_lower.tolist(),
                "ci_upper": ci_upper.tolist(),
            },
            f,
        )

    return mu_mean, rhat, ess, ci_lower, ci_upper
