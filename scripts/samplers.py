# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=protected-access

import atexit
import multiprocessing
import time
from multiprocessing import cpu_count
from typing import List, Tuple, Union

import joblib
import numpy as np
from joblib import Parallel, delayed
from state import State
from tqdm import tqdm

# constants
LOG_2PI = 0.5 * np.log(2 * np.pi)


def _cleanup_multiprocessing():
    """Force cleanup of multiprocessing resources on exit."""

    if hasattr(multiprocessing, "_cleanup"):
        multiprocessing._cleanup()


atexit.register(_cleanup_multiprocessing)


def compute_log_likelihood(y: np.ndarray, state: State) -> float:
    """Compute the log-likelihood of the data given the current state.

    Calculates the log-likelihood of observed data points under a Gaussian
    mixture model with current parameters (mixing weights pi, means mu, and variances sigma2).
    Uses log-sum-exp trick for numerical stability.

    Args:
        y: Observed data points.
        state: Current state containing pi (mixing weights), mu (means), and sigma2 (variances).

    Returns:
        Log-likelihood value for the entire dataset.
    """
    log_pi = np.log(state.pi)
    diff = y[:, np.newaxis] - state.mu[np.newaxis, :]
    log_weights = (
        log_pi[np.newaxis, :]
        - 0.5 * diff**2 / state.sigma2[np.newaxis, :]
        - 0.5 * np.log(state.sigma2[np.newaxis, :])
        - LOG_2PI
    )
    max_logw = np.max(log_weights, axis=1)
    return np.sum(
        max_logw + np.log(np.sum(np.exp(log_weights - max_logw[:, np.newaxis]), axis=1))
    )


def create_temperature_ladder(n_temps: int, max_temp: float) -> np.ndarray:
    """Create a temperature ladder for tempered Gibbs sampling.

    Creates a geometric sequence of temperatures from 1/max_temp to 1.0,
    optionally including the reverse sequence for bidirectional tempering.

    Args:
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).

    Returns:
        Array of temperatures from 0 to 1.
    """
    temps = np.geomspace(1.0 / max_temp, 1.0, n_temps)
    return np.hstack([temps[:0:-1], temps])


def sample_z(
    y: np.ndarray, state: State, rng: np.random.Generator, beta: float = 1.0
) -> np.ndarray:
    """Sample cluster assignments for each data point with tempering.

    Uses the current values of pi, mu, and sigma2 to compute tempered posterior
    probabilities and sample new cluster assignments using categorical sampling.

    Args:
        y: Observed data points.
        state: Current state containing pi, mu, and sigma2 values.
        rng: Random number generator.
        beta: Temperature parameter (0 = prior only, 1 = full posterior).

    Returns:
        New cluster assignments for each data point.
    """
    log_prior = np.log(state.pi)
    diff = y[:, np.newaxis] - state.mu[np.newaxis, :]
    log_like = (
        -0.5 * diff**2 / state.sigma2[np.newaxis, :]
        - 0.5 * np.log(state.sigma2[np.newaxis, :])
        - LOG_2PI
    )
    logw = log_prior[np.newaxis, :] + beta * log_like
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

    Samples new mixing weights from the tempered posterior Dirichlet distribution
    based on current cluster assignments and temperature parameter.

    Args:
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
        beta: Temperature parameter for tempering the likelihood contribution.
        alpha: Dirichlet concentration parameter (prior).

    Returns:
        New mixture weights sampled from tempered posterior Dirichlet.
    """
    return rng.dirichlet(alpha + beta * np.bincount(z, minlength=K))


def sample_mu(
    y: np.ndarray,
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    sigma2: np.ndarray,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Sample mean parameters from tempered normal distributions.

    For each component, samples from the tempered posterior normal distribution
    given the assigned data points, current variances, and prior parameters.
    Uses conjugate normal-normal updating with tempering applied to the likelihood precision.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
        sigma2: Current variance parameters for each component.
        m0: Prior mean(s) for component means (scalar or array of size K).
        s0_2: Prior variance(s) for component means (scalar or array of size K).
        beta: Temperature parameter for tempering the likelihood contribution.

    Returns:
        New mean parameters for each component.
    """
    # Convert scalar priors to arrays if needed
    if np.isscalar(m0):
        m0 = np.full(K, m0)
    if np.isscalar(s0_2):
        s0_2 = np.full(K, s0_2)

    m0 = np.asarray(m0)
    s0_2 = np.asarray(s0_2)

    # Validate input sizes
    if len(m0) != K:
        raise ValueError(f"m0 must have length K={K}, got {len(m0)}")
    if len(s0_2) != K:
        raise ValueError(f"s0_2 must have length K={K}, got {len(s0_2)}")

    mu = np.empty(K)
    for k in range(K):
        idx = z == k
        n_k = idx.sum()
        prior_prec = 1.0 / s0_2[k]
        prior_term = m0[k] * prior_prec

        if n_k == 0:
            mu[k] = rng.normal(m0[k], np.sqrt(s0_2[k]))
        else:
            y_sum = y[idx].sum()
            likelihood_prec = beta * n_k / sigma2[k]
            post_prec = prior_prec + likelihood_prec
            post_var = 1.0 / post_prec
            post_mean = post_var * (beta * y_sum / sigma2[k] + prior_term)
            mu[k] = rng.normal(post_mean, np.sqrt(post_var))
    return mu


def sample_sigma2(
    y: np.ndarray,
    z: np.ndarray,
    mu: np.ndarray,
    K: int,
    rng: np.random.Generator,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Sample variance parameters from tempered inverse-gamma distributions.

    For each component, samples from the tempered posterior inverse-gamma distribution
    given the assigned data points, current means, and prior parameters.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        mu: Current mean parameters for each component.
        K: Number of mixture components.
        rng: Random number generator.
        alpha0: Prior shape parameter(s) for inverse-gamma distribution (scalar or array of size K).
        beta0: Prior scale parameter(s) for inverse-gamma distribution (scalar or array of size K).
        beta: Temperature parameter for tempering the likelihood contribution.

    Returns:
        New variance parameters for each component.
    """
    # Convert scalar priors to arrays if needed
    if np.isscalar(alpha0):
        alpha0 = np.full(K, alpha0)
    if np.isscalar(beta0):
        beta0 = np.full(K, beta0)

    alpha0 = np.asarray(alpha0)
    beta0 = np.asarray(beta0)

    # Validate input sizes
    if len(alpha0) != K:
        raise ValueError(f"alpha0 must have length K={K}, got {len(alpha0)}")
    if len(beta0) != K:
        raise ValueError(f"beta0 must have length K={K}, got {len(beta0)}")

    sigma2 = np.empty(K)
    for k in range(K):
        idx = z == k
        n_k = idx.sum()

        if n_k == 0:
            # Sample from prior
            sigma2[k] = 1.0 / rng.gamma(alpha0[k], 1.0 / beta0[k])
        else:
            # Sample from tempered posterior
            y_k = y[idx]
            sum_sq_dev = np.sum((y_k - mu[k]) ** 2)

            post_alpha = alpha0[k] + beta * n_k / 2.0
            post_beta = beta0[k] + beta * sum_sq_dev / 2.0

            sigma2[k] = 1.0 / rng.gamma(post_alpha, 1.0 / post_beta)

    return sigma2


def gibbs_step(
    y: np.ndarray,
    state: State,
    K: int,
    rng: np.random.Generator,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    beta: float = 1.0,
) -> State:
    """Perform one step of the tempered Gibbs sampler.

    Sequentially samples z, pi, mu, and sigma2 given the current state and temperature
    parameter. This implements one full cycle of the tempered Gibbs sampler.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.
        m0: Prior mean(s) for component means.
        s0_2: Prior variance(s) for component means.
        alpha0: Prior shape parameter(s) for inverse-gamma distribution.
        beta0: Prior scale parameter(s) for inverse-gamma distribution.
        beta: Temperature parameter for tempering.

    Returns:
        New state after one tempered Gibbs sampling step.
    """
    z = sample_z(y, state, rng, beta=beta)
    pi = sample_pi(z, K, rng, beta=beta)
    mu = sample_mu(y, z, K, rng, state.sigma2, m0=m0, s0_2=s0_2, beta=beta)
    sigma2 = sample_sigma2(y, z, mu, K, rng, alpha0=alpha0, beta0=beta0, beta=beta)
    return State(z, pi, mu, sigma2)


def tempered_transition_step(
    y: np.ndarray,
    state: State,
    K: int,
    rng: np.random.Generator,
    temperatures_ladder: np.ndarray,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    n_gibbs_per_temp: int = 10,
    placebo: bool = False,
) -> Tuple[State, bool]:
    """Perform one tempered transition step with acceptance/rejection.

    Moves through the temperature ladder, performing Gibbs steps at each temperature,
    then uses Metropolis criterion to accept or reject the proposal. This implements
    the core tempered transitions algorithm.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.
        temperatures_ladder: Array of temperatures from 0 to 1.
        m0: Prior mean(s) for component means.
        s0_2: Prior variance(s) for component means.
        alpha0: Prior shape parameter(s) for inverse-gamma distribution.
        beta0: Prior scale parameter(s) for inverse-gamma distribution.
        n_gibbs_per_temp: Number of Gibbs steps per temperature.
        placebo: Whether to use placebo relabeling to avoid label switching.

    Returns:
        Tuple of (new_state, accepted) where accepted indicates if proposal was accepted.
    """
    initial_state = state
    current_state = state

    for beta in temperatures_ladder:
        for _ in range(n_gibbs_per_temp):
            current_state = gibbs_step(
                y,
                current_state,
                K,
                rng,
                m0=m0,
                s0_2=s0_2,
                alpha0=alpha0,
                beta0=beta0,
                beta=beta,
            ).relabel(placebo=placebo)

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

    Unpacks arguments and calls run_chain. This function is designed to work
    with multiprocessing.Pool.map() which requires a single argument function.

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
        m0,
        s0_2,
        alpha0,
        beta0,
        n_temps,
        max_temp,
        n_gibbs_per_temp,
        placebo,
    ) = args

    return run_chain(
        y,
        K,
        n_iter,
        burn,
        seed,
        m0,
        s0_2,
        alpha0,
        beta0,
        n_temps,
        max_temp,
        n_gibbs_per_temp,
        placebo,
    )


def run_chain(
    y: np.ndarray,
    K: int,
    n_iter: int,
    burn: int,
    seed: int,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    n_temps: int = 10,
    max_temp: float = 5.0,
    n_gibbs_per_temp: int = 5,
    placebo: bool = False,
):
    """Run a single tempered transitions chain.

    Executes a complete tempered transitions MCMC chain with specified parameters.
    Includes burn-in period and tracks acceptance rates and runtime statistics.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations.
        burn: Number of burn-in iterations to discard.
        seed: Random seed for reproducibility.
        m0: Prior mean(s) for component means.
        s0_2: Prior variance(s) for component means.
        alpha0: Prior shape parameter(s) for inverse-gamma distribution.
        beta0: Prior scale parameter(s) for inverse-gamma distribution.
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).
        n_gibbs_per_temp: Number of Gibbs steps per temperature.
        placebo: Whether to use placebo relabeling to avoid label switching.

    Returns:
        Tuple of (samples_mu, samples_sigma2, runtime, acceptance_rate) where
        samples_mu and samples_sigma2 contain the post-burn-in samples for means
        and variances, runtime is the elapsed time, and acceptance_rate is the
        fraction of proposals that were accepted.
    """
    rng = np.random.default_rng(seed)

    temperatures_ladder = create_temperature_ladder(n_temps, max_temp)

    # Initialize state with reasonable values
    initial_z = rng.integers(0, K, len(y))
    initial_pi = np.ones(K) / K
    initial_mu = rng.normal(0, 1, K)
    initial_sigma2 = np.ones(K)  # Initialize variances to 1

    state = State(initial_z, initial_pi, initial_mu, initial_sigma2)

    kept_mu = []
    kept_sigma2 = []
    n_accepted = 0
    t0 = time.perf_counter()

    for it in tqdm(range(n_iter)):
        state, accepted = tempered_transition_step(
            y,
            state,
            K,
            rng,
            temperatures_ladder,
            m0,
            s0_2,
            alpha0,
            beta0,
            n_gibbs_per_temp,
            placebo,
        )

        if accepted:
            n_accepted += 1

        if it >= burn:
            relabeled_state = state.relabel(placebo=placebo)
            kept_mu.append(relabeled_state.mu.copy())
            kept_sigma2.append(relabeled_state.sigma2.copy())

    acceptance_rate = n_accepted / n_iter
    return (
        np.vstack(kept_mu),
        np.vstack(kept_sigma2),
        time.perf_counter() - t0,
        acceptance_rate,
    )


def run_parallel_chains(
    y: np.ndarray,
    K: int,
    n_iter: int,
    burn: int,
    base_seed: int,
    n_chains: int = 4,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    n_temps: int = 10,
    max_temp: float = 5.0,
    n_gibbs_per_temp: int = 5,
    placebo: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """Run multiple chains in parallel using multiprocessing.

    Executes multiple independent tempered transitions chains in parallel to
    improve convergence diagnostics and computational efficiency. Uses all
    available CPU cores up to the number of chains requested.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations per chain.
        burn: Number of burn-in iterations to discard.
        base_seed: Base seed to generate unique seeds for each chain.
        n_chains: Number of chains to run.
        m0: Prior mean(s) for component means.
        s0_2: Prior variance(s) for component means.
        alpha0: Prior shape parameter(s) for inverse-gamma distribution.
        beta0: Prior scale parameter(s) for inverse-gamma distribution.
        n_temps: Number of temperatures in the ladder.
        max_temp: Maximum temperature (1/beta_min).
        n_gibbs_per_temp: Number of Gibbs steps per temperature.
        placebo: Whether to use placebo relabeling to avoid label switching.

    Returns:
        Tuple of (all_samples_mu, all_samples_sigma2, all_runtimes, all_acceptance_rates) where:
        - all_samples_mu: List of mean sample arrays, one per chain
        - all_samples_sigma2: List of variance sample arrays, one per chain
        - all_runtimes: List of runtime values per chain
        - all_acceptance_rates: List of acceptance rates per chain
    """
    joblib.parallel.DEFAULT_N_JOBS = min(n_chains, cpu_count())
    seeds = [base_seed + i * 1000 for i in range(n_chains)]
    print(f"Running {n_chains} chains with joblib (backend: loky)...")
    results = Parallel(
        n_jobs=min(n_chains, cpu_count()),
        backend="loky",
        verbose=0,
        batch_size=1,
        pre_dispatch="2*n_jobs",
    )(
        delayed(_run_single_chain)(
            (
                y,
                K,
                n_iter,
                burn,
                seed,
                m0,
                s0_2,
                alpha0,
                beta0,
                n_temps,
                max_temp,
                n_gibbs_per_temp,
                placebo,
            )
        )
        for seed in seeds
    )

    all_samples_mu = [result[0] for result in results]
    all_samples_sigma2 = [result[1] for result in results]
    all_runtimes = [result[2] for result in results]
    all_acceptance_rates = [result[3] for result in results]

    return all_samples_mu, all_samples_sigma2, all_runtimes, all_acceptance_rates
