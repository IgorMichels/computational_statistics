# pylint: disable=too-many-arguments
# pylint: disable=unexpected-keyword-arg
# pylint: disable=too-many-locals

import time

import numpy as np
from state import State
from tqdm import tqdm


def create_temperature_ladder(
    n_temps: int, max_temp: float, keep_last: bool = False
) -> np.ndarray:
    """Create a temperature ladder for tempered Gibbs sampling.

    Args:
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).

    Returns:
        Array of temperatures from 0 to 1.
    """
    temps = np.geomspace(1.0 / max_temp, 1.0, n_temps)
    if keep_last:
        return np.hstack([temps[::-1], temps])

    return np.hstack([temps[-2::-1], temps])


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


def gibbs_step(
    y: np.ndarray, state: State, K: int, rng: np.random.Generator, beta: float = 1.0
) -> State:
    """Perform one step of the tempered Gibbs sampler.

    Sequentially samples z, pi, and mu given the current state and temperature.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.
        beta: Temperature parameter.

    Returns:
        New state after one tempered Gibbs sampling step.
    """
    z = sample_z(y, state, rng, beta=beta)
    pi = sample_pi(z, K, rng, beta=beta)
    mu = sample_mu(y, z, K, rng, beta=beta)
    return State(z, pi, mu)


def tempered_transition_step(
    y: np.ndarray,
    state: State,
    K: int,
    rng: np.random.Generator,
    temperatures_ladder: np.ndarray,
    n_gibbs_per_temp: int = 10,
) -> State:
    """Perform one tempered transition step.

    Moves through the temperature ladder, performing Gibbs steps at each temperature,
    then attempts to swap between adjacent temperatures.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.
        temperatures_ladder: Array of temperatures from 0 to 1.
        n_gibbs_per_temp: Number of Gibbs steps per temperature.

    Returns:
        New state after tempered transition.
    """
    current_state = state

    for beta in temperatures_ladder:
        for _ in range(n_gibbs_per_temp):
            current_state = gibbs_step(y, current_state, K, rng, beta=beta)

    return current_state


def run_chain(
    y: np.ndarray,
    K: int,
    n_iter: int,
    burn: int,
    seed: int,
    n_temps: int = 10,
    max_temp: float = 5.0,
    n_gibbs_per_temp: int = 5,
    placebo: bool = False,
    keep_last: bool = False,
):
    """Run a single tempered transitions chain.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations.
        burn: Number of burn-in iterations to discard.
        seed: Random seed for reproducibility.
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).
        n_gibbs_per_temp: Number of Gibbs steps per temperature.
        placebo: Whether to use placebo on relabeling.

    Returns:
        Tuple of (samples, runtime) where samples contains the
        post-burn-in samples and runtime is the elapsed time.
    """
    rng = np.random.default_rng(seed)

    temperatures_ladder = create_temperature_ladder(n_temps, max_temp, keep_last)
    state = State(rng.integers(0, K, len(y)), np.ones(K) / K, rng.normal(0, 1, K))

    kept = []
    t0 = time.perf_counter()
    for it in tqdm(range(n_iter)):
        state = tempered_transition_step(
            y, state, K, rng, temperatures_ladder, n_gibbs_per_temp
        )
        if it >= burn:
            kept.append(state.relabel(placebo=placebo).mu.copy())

    return np.vstack(kept), time.perf_counter() - t0
