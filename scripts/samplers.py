# pylint: disable=too-many-arguments
# pylint: disable=unexpected-keyword-arg
# pylint: disable=too-many-locals

import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
from state import State
from tqdm import tqdm


def compute_log_likelihood(y: np.ndarray, state: State) -> float:
    """Compute the log-likelihood of the data given the current state.

    Calculates the log-likelihood of observed data points under a Gaussian
    mixture model with current parameters (mixing weights pi and means mu).
    Uses log-sum-exp trick for numerical stability.

    Args:
        y: Observed data points.
        state: Current state containing pi (mixing weights) and mu (means).

    Returns:
        Log-likelihood value for the entire dataset.
    """
    K = len(state.pi)
    log_like = 0.0

    for yi in y:
        log_weights = np.empty(K)
        for k in range(K):
            log_weights[k] = (
                np.log(state.pi[k])
                - 0.5 * (yi - state.mu[k]) ** 2
                - 0.5 * np.log(2 * np.pi)
            )

        max_log_weight = log_weights.max()
        log_like += max_log_weight + np.log(np.exp(log_weights - max_log_weight).sum())

    return log_like


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
) -> Tuple[State, bool]:
    """Perform one tempered transition step with acceptance/rejection.

    Moves through the temperature ladder, performing Gibbs steps at each temperature,
    then uses Metropolis criterion to accept or reject the proposal.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.
        temperatures_ladder: Array of temperatures from 0 to 1.
        n_gibbs_per_temp: Number of Gibbs steps per temperature.

    Returns:
        Tuple of (new_state, accepted) where accepted indicates if proposal was accepted.
    """
    initial_state = state
    current_state = state

    for beta in temperatures_ladder:
        for _ in range(n_gibbs_per_temp):
            current_state = gibbs_step(y, current_state, K, rng, beta=beta)

    if temperatures_ladder.shape[0] == 1:
        return current_state, True

    initial_loglik = compute_log_likelihood(y, initial_state)
    final_loglik = compute_log_likelihood(y, current_state)
    log_ratio = final_loglik - initial_loglik

    if np.log(rng.random()) < log_ratio:
        return current_state, True

    return initial_state, False


def _run_single_chain(args):
    """Helper function to run a single chain (for parallelization).

    Args:
        args: Tuple containing all arguments needed for run_chain.

    Returns:
        Tuple of (samples, runtime, acceptance_rate) for one chain.
    """
    (
        y,
        K,
        n_iter,
        burn,
        seed,
        n_temps,
        max_temp,
        n_gibbs_per_temp,
        placebo,
        keep_last,
    ) = args

    return run_chain(
        y,
        K,
        n_iter,
        burn,
        seed,
        n_temps,
        max_temp,
        n_gibbs_per_temp,
        placebo,
        keep_last,
    )


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
        keep_last: Whether to keep the last temperature in the ladder.

    Returns:
        Tuple of (samples, runtime, acceptance_rate) where samples contains the
        post-burn-in samples, runtime is the elapsed time, and acceptance_rate
        is the fraction of proposals that were accepted.
    """
    rng = np.random.default_rng(seed)

    temperatures_ladder = create_temperature_ladder(n_temps, max_temp, keep_last)
    state = State(rng.integers(0, K, len(y)), np.ones(K) / K, rng.normal(0, 1, K))

    kept = []
    n_accepted = 0
    t0 = time.perf_counter()

    for it in tqdm(range(n_iter)):
        state, accepted = tempered_transition_step(
            y, state, K, rng, temperatures_ladder, n_gibbs_per_temp
        )

        if accepted:
            n_accepted += 1

        if it >= burn:
            kept.append(state.relabel(placebo=placebo).mu.copy())

    acceptance_rate = n_accepted / n_iter
    return np.vstack(kept), time.perf_counter() - t0, acceptance_rate


def run_parallel_chains(
    y: np.ndarray,
    K: int,
    n_iter: int,
    burn: int,
    base_seed: int,
    n_chains: int = 4,
    n_temps: int = 10,
    max_temp: float = 5.0,
    n_gibbs_per_temp: int = 5,
    placebo: bool = False,
    keep_last: bool = False,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """Run multiple chains in parallel using multiprocessing.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations per chain.
        burn: Number of burn-in iterations to discard.
        base_seed: Base seed to generate unique seeds for each chain.
        n_chains: Number of chains to run.
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).
        n_gibbs_per_temp: Number of Gibbs steps per temperature.
        placebo: Whether to use placebo on relabeling.
        keep_last: Whether to keep the last temperature in the ladder.

    Returns:
        Tuple of (all_samples, all_runtimes, all_acceptance_rates) where:
        - all_samples: List of sample arrays, one per chain
        - all_runtimes: List of runtime values per chain
        - all_acceptance_rates: List of acceptance rates per chain
    """
    n_processes = min(n_chains, cpu_count())

    # Generate unique seeds for each chain
    seeds = [base_seed + i * 1000 for i in range(n_chains)]

    # Prepare arguments for each chain
    chain_args = [
        (
            y,
            K,
            n_iter,
            burn,
            seed,
            n_temps,
            max_temp,
            n_gibbs_per_temp,
            placebo,
            keep_last,
        )
        for seed in seeds
    ]

    print(f"Running {n_chains} chains in parallel using {n_processes} processes...")

    # Run chains in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(_run_single_chain, chain_args)

    # Separate results
    all_samples = [result[0] for result in results]
    all_runtimes = [result[1] for result in results]
    all_acceptance_rates = [result[2] for result in results]

    return all_samples, all_runtimes, all_acceptance_rates
