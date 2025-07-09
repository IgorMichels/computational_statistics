# pylint: disable=too-many-arguments
# pylint: disable=unexpected-keyword-arg

import numpy as np
from state import State


def sample_z(
    y: np.ndarray, state: State, rng: np.random.Generator, beta: float = 1.0
) -> np.ndarray:
    """Sample cluster assignments for each data point with tempering.

    Uses the current values of pi and mu to compute tempered posterior
    probabilities and sample new cluster assignments.

    Args:
        y: Observed data points.
        state: Current state containing pi and mu values.
        rng: Random number generator.
        beta: Temperature parameter (0 = prior only, 1 = full posterior).

    Returns:
        New cluster assignments for each data point.
    """
    K = len(state.pi)
    logw = np.empty((len(y), K))
    for k in range(K):
        log_prior = np.log(state.pi[k])
        log_like = -0.5 * (y - state.mu[k]) ** 2 - 0.5 * np.log(2 * np.pi)
        logw[:, k] = log_prior + beta * log_like
    logw -= logw.max(axis=1, keepdims=True)
    probs = np.exp(logw)
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random((len(y), 1))
    return (np.cumsum(probs, axis=1) > u).argmax(axis=1)


def sample_pi(
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Sample mixture weights from tempered Dirichlet distribution.

    Args:
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
        beta: Temperature parameter.
        alpha: Dirichlet concentration parameter.

    Returns:
        New mixture weights sampled from tempered posterior Dirichlet.
    """
    return rng.dirichlet(alpha + beta * np.bincount(z, minlength=K))


def sample_mu(
    y: np.ndarray,
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    m0: float = 0.0,
    s0_2: float = 25.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Sample mean parameters from tempered normal distributions.

    For each component, samples from the tempered posterior normal distribution
    given the assigned data points and prior parameters.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
        beta: Temperature parameter.
        m0: Prior mean for component means.
        s0_2: Prior variance for component means.

    Returns:
        New mean parameters for each component.
    """
    mu = np.empty(K)
    for k in range(K):
        idx = z == k
        n_k = idx.sum()
        if n_k == 0:
            mu[k] = rng.normal(m0, np.sqrt(s0_2))
        else:
            ybar = y[idx].mean()
            # Tempered precision: 1/s0_2 + beta * n_k
            post_prec = 1.0 / s0_2 + beta * n_k
            post_var = 1.0 / post_prec
            post_mean = post_var * (beta * n_k * ybar + m0 / s0_2)
            mu[k] = rng.normal(post_mean, np.sqrt(post_var))
    return mu
