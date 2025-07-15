# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
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
    """
    Force cleanup of multiprocessing resources on exit.

    This function ensures that all multiprocessing resources are properly cleaned up
    when the Python interpreter exits. It checks if the multiprocessing module has
    a cleanup method and calls it if available.
    """
    if hasattr(multiprocessing, "_cleanup"):
        multiprocessing._cleanup()


atexit.register(_cleanup_multiprocessing)


def compute_log_likelihood(y: np.ndarray, state: State) -> float:
    """
    Compute the log-likelihood of the data given the current state.

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
    """
    Create a temperature ladder for tempered Gibbs sampling.

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
    """
    Sample cluster assignments for each data point with tempering.

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
    """
    Sample mixture weights from tempered Dirichlet distribution.

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
    """
    Sample mean parameters from tempered normal distributions.

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


def sample_mu_single_component(
    y: np.ndarray,
    z: np.ndarray,
    k: int,
    rng: np.random.Generator,
    sigma2_k: float,
    m0_k: float = 0.0,
    s0_2_k: float = 4.0,
    beta: float = 1.0,
) -> float:
    """
    Sample mean parameter for a single component from tempered normal distribution.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        k: Component index to sample.
        rng: Random number generator.
        sigma2_k: Current variance parameter for component k.
        m0_k: Prior mean for component k.
        s0_2_k: Prior variance for component k.
        beta: Temperature parameter for tempering the likelihood contribution.

    Returns:
        New mean parameter for component k.
    """
    idx = z == k
    n_k = idx.sum()
    prior_prec = 1.0 / s0_2_k
    prior_term = m0_k * prior_prec

    if n_k == 0:
        return rng.normal(m0_k, np.sqrt(s0_2_k))
    y_sum = y[idx].sum()
    likelihood_prec = beta * n_k / sigma2_k
    post_prec = prior_prec + likelihood_prec
    post_var = 1.0 / post_prec
    post_mean = post_var * (beta * y_sum / sigma2_k + prior_term)
    return rng.normal(post_mean, np.sqrt(post_var))


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
    """
    Sample variance parameters from tempered inverse-gamma distributions.

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
            sigma2[k] = 1.0 / rng.gamma(alpha0[k], 1.0 / beta0[k])
        else:
            y_k = y[idx]
            sum_sq_dev = np.sum((y_k - mu[k]) ** 2)

            post_alpha = alpha0[k] + beta * n_k / 2.0
            post_beta = beta0[k] + beta * sum_sq_dev / 2.0

            sigma2[k] = 1.0 / rng.gamma(post_alpha, 1.0 / post_beta)

    return sigma2


def sample_sigma2_single_component(
    y: np.ndarray,
    z: np.ndarray,
    mu_k: float,
    k: int,
    rng: np.random.Generator,
    alpha0_k: float = 2.0,
    beta0_k: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Sample variance parameter for a single component from tempered inverse-gamma distribution.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        mu_k: Current mean parameter for component k.
        k: Component index to sample.
        rng: Random number generator.
        alpha0_k: Prior shape parameter for component k.
        beta0_k: Prior scale parameter for component k.
        beta: Temperature parameter for tempering the likelihood contribution.

    Returns:
        New variance parameter for component k.
    """
    idx = z == k
    n_k = idx.sum()

    if n_k == 0:
        return 1.0 / rng.gamma(alpha0_k, 1.0 / beta0_k)
    y_k = y[idx]
    sum_sq_dev = np.sum((y_k - mu_k) ** 2)

    post_alpha = alpha0_k + beta * n_k / 2.0
    post_beta = beta0_k + beta * sum_sq_dev / 2.0

    return 1.0 / rng.gamma(post_alpha, 1.0 / post_beta)


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
    """
    Perform one step of the tempered Gibbs sampler.

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


def gibbs_step_with_componentwise_acceptance(
    y: np.ndarray,
    state: State,
    K: int,
    rng: np.random.Generator,
    m0: Union[float, np.ndarray] = 0.0,
    s0_2: Union[float, np.ndarray] = 4.0,
    alpha0: Union[float, np.ndarray] = 2.0,
    beta0: Union[float, np.ndarray] = 1.0,
    beta: float = 1.0,
) -> Tuple[State, dict]:
    """
    Perform one step of the tempered Gibbs sampler with componentwise acceptance testing.

    Sequentially samples and tests acceptance for z, pi, mu (component by component),
    and sigma2 (component by component).

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
        Tuple of (new_state, acceptance_info) where acceptance_info contains
        information about which components were accepted.
    """
    # Convert scalar priors to arrays if needed
    if np.isscalar(m0):
        m0 = np.full(K, m0)
    if np.isscalar(s0_2):
        s0_2 = np.full(K, s0_2)
    if np.isscalar(alpha0):
        alpha0 = np.full(K, alpha0)
    if np.isscalar(beta0):
        beta0 = np.full(K, beta0)

    m0 = np.asarray(m0)
    s0_2 = np.asarray(s0_2)
    alpha0 = np.asarray(alpha0)
    beta0 = np.asarray(beta0)

    current_state = state
    acceptance_info = {
        "z_accepted": False,
        "pi_accepted": False,
        "mu_accepted": np.zeros(K, dtype=bool),
        "sigma2_accepted": np.zeros(K, dtype=bool),
    }

    # 1. Sample z and test acceptance
    current_loglik = compute_log_likelihood(y, current_state)
    new_z = sample_z(y, current_state, rng, beta=beta)
    new_state_z = State(new_z, current_state.pi, current_state.mu, current_state.sigma2)
    new_loglik = compute_log_likelihood(y, new_state_z)

    log_ratio = new_loglik - current_loglik
    if np.log(rng.random()) < log_ratio:
        current_state = new_state_z
        current_loglik = new_loglik
        acceptance_info["z_accepted"] = True

    # 2. Sample pi and test acceptance
    new_pi = sample_pi(current_state.z, K, rng, beta=beta)
    new_state_pi = State(
        current_state.z, new_pi, current_state.mu, current_state.sigma2
    )
    new_loglik = compute_log_likelihood(y, new_state_pi)

    log_ratio = new_loglik - current_loglik
    if np.log(rng.random()) < log_ratio:
        current_state = new_state_pi
        current_loglik = new_loglik
        acceptance_info["pi_accepted"] = True

    # 3. Sample mu component by component and test acceptance
    for k in range(K):
        new_mu_k = sample_mu_single_component(
            y,
            current_state.z,
            k,
            rng,
            current_state.sigma2[k],
            m0[k],
            s0_2[k],
            beta=beta,
        )
        new_mu = current_state.mu.copy()
        new_mu[k] = new_mu_k
        new_state_mu = State(
            current_state.z, current_state.pi, new_mu, current_state.sigma2
        )
        new_loglik = compute_log_likelihood(y, new_state_mu)

        log_ratio = new_loglik - current_loglik
        if np.log(rng.random()) < log_ratio:
            current_state = new_state_mu
            current_loglik = new_loglik
            acceptance_info["mu_accepted"][k] = True

    # 4. Sample sigma2 component by component and test acceptance
    for k in range(K):
        new_sigma2_k = sample_sigma2_single_component(
            y,
            current_state.z,
            current_state.mu[k],
            k,
            rng,
            alpha0[k],
            beta0[k],
            beta=beta,
        )
        new_sigma2 = current_state.sigma2.copy()
        new_sigma2[k] = new_sigma2_k
        new_state_sigma2 = State(
            current_state.z, current_state.pi, current_state.mu, new_sigma2
        )
        new_loglik = compute_log_likelihood(y, new_state_sigma2)

        log_ratio = new_loglik - current_loglik
        if np.log(rng.random()) < log_ratio:
            current_state = new_state_sigma2
            current_loglik = new_loglik
            acceptance_info["sigma2_accepted"][k] = True

    return current_state, acceptance_info


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
    componentwise_acceptance: bool = False,
) -> Tuple[State, Union[bool, dict]]:
    """
    Perform one tempered transition step with acceptance/rejection.

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
        componentwise_acceptance: Whether to use componentwise acceptance testing.

    Returns:
        Tuple of (new_state, accepted) where accepted indicates if proposal was accepted.
        If componentwise_acceptance=True, accepted is a dict with acceptance info.
    """
    initial_state = state
    current_state = state

    if componentwise_acceptance:
        total_mu_accepted = np.zeros(K)
        total_sigma2_accepted = np.zeros(K)
        total_steps = 0

        for beta in temperatures_ladder:
            for _ in range(n_gibbs_per_temp):
                (
                    current_state,
                    acceptance_info,
                ) = gibbs_step_with_componentwise_acceptance(
                    y,
                    current_state,
                    K,
                    rng,
                    m0=m0,
                    s0_2=s0_2,
                    alpha0=alpha0,
                    beta0=beta0,
                    beta=beta,
                )
                current_state = current_state.relabel(placebo=placebo)

                total_mu_accepted += acceptance_info["mu_accepted"]
                total_sigma2_accepted += acceptance_info["sigma2_accepted"]
                total_steps += 1

        acceptance_summary = {
            "mu_accepted": total_mu_accepted,
            "sigma2_accepted": total_sigma2_accepted,
            "total_steps": total_steps,
        }
        return current_state, acceptance_summary
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
    """
    Helper function to run a single chain (for parallelization).

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
        verbose,
        loading_bar,
        componentwise_acceptance,
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
        verbose,
        loading_bar,
        componentwise_acceptance,
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
    verbose: bool = False,
    loading_bar: bool = False,
    componentwise_acceptance: bool = False,
):
    """
    Run a single tempered transitions chain.

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
        verbose: Whether to print verbose output.
        loading_bar: Whether to show progress bars during execution.
        componentwise_acceptance: Whether to use componentwise acceptance testing.
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

    if componentwise_acceptance:
        total_mu_accepted = np.zeros(K)
        total_sigma2_accepted = np.zeros(K)
        total_componentwise_steps = 0
    else:
        n_accepted = 0

    t0 = time.perf_counter()

    iterations = tqdm(range(n_iter)) if (verbose or loading_bar) else range(n_iter)
    for it in iterations:
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
            componentwise_acceptance,
        )

        if componentwise_acceptance:
            assert isinstance(
                accepted, dict
            ), "Expected dict when componentwise_acceptance is True"
            total_mu_accepted += accepted["mu_accepted"]
            total_sigma2_accepted += accepted["sigma2_accepted"]
            total_componentwise_steps += accepted["total_steps"]
        else:
            if accepted:
                n_accepted += 1

        if it >= burn:
            relabeled_state = state.relabel(placebo=placebo)
            kept_mu.append(relabeled_state.mu.copy())
            kept_sigma2.append(relabeled_state.sigma2.copy())

    if componentwise_acceptance:
        mu_acceptance_rates = total_mu_accepted / total_componentwise_steps
        sigma2_acceptance_rates = total_sigma2_accepted / total_componentwise_steps
        acceptance_rate = np.mean(
            np.concatenate([mu_acceptance_rates, sigma2_acceptance_rates])
        )
    else:
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
    verbose: bool = False,
    loading_bar: bool = False,
    componentwise_acceptance: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """
    Run multiple chains in parallel using multiprocessing.

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
        verbose: Whether to print verbose output.
        loading_bar: Whether to show progress bars during execution.
        componentwise_acceptance: Whether to use componentwise acceptance testing.
    Returns:
        Tuple of (all_samples_mu, all_samples_sigma2, all_runtimes, all_acceptance_rates) where:
        - all_samples_mu: List of mean sample arrays, one per chain
        - all_samples_sigma2: List of variance sample arrays, one per chain
        - all_runtimes: List of runtime values per chain
        - all_acceptance_rates: List of acceptance rates per chain
    """
    joblib.parallel.DEFAULT_N_JOBS = min(n_chains, cpu_count())
    seeds = [base_seed + i * 1000 for i in range(n_chains)]
    if verbose:
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
                verbose,
                loading_bar,
                componentwise_acceptance,
            )
        )
        for seed in seeds
    )

    all_samples_mu = [result[0] for result in results]
    all_samples_sigma2 = [result[1] for result in results]
    all_runtimes = [result[2] for result in results]
    all_acceptance_rates = [result[3] for result in results]

    return all_samples_mu, all_samples_sigma2, all_runtimes, all_acceptance_rates
