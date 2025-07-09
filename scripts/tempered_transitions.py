import argparse
import os
import time
import warnings

import numpy as np
from metrics import credible_interval, ess_1d, rhat_scalar
from plots import create_diagnostic_plots
from state import State
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)


def sample_z_tempered(
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
        # Tempered likelihood: (likelihood^beta) * (prior^(1-beta))
        logw[:, k] = (1 - beta) * log_prior + beta * log_like
    logw -= logw.max(axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    probs = np.exp(logw)
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random((len(y), 1))
    return (np.cumsum(probs, axis=1) > u).argmax(axis=1)


def sample_pi_tempered(
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    beta: float = 1.0,
    alpha: float = 1.0,
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
    # Tempered Dirichlet: Dir(alpha + beta * counts)
    counts = np.bincount(z, minlength=K)
    tempered_alpha = alpha + beta * counts
    return rng.dirichlet(tempered_alpha)


# pylint: disable=too-many-arguments
def sample_mu_tempered(
    y: np.ndarray,
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    beta: float = 1.0,
    m0: float = 0.0,
    s0_2: float = 25.0,
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
            post_mean = post_var * (m0 / s0_2 + beta * n_k * ybar)
            mu[k] = rng.normal(post_mean, np.sqrt(post_var))
    return mu


def tempered_gibbs_step(
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
    z = sample_z_tempered(y, state, rng, beta)
    pi = sample_pi_tempered(z, K, rng, beta)
    mu = sample_mu_tempered(y, z, K, rng, beta)
    return State(z, pi, mu)


def compute_log_likelihood(y: np.ndarray, state: State) -> float:
    """Compute log-likelihood of the data given current state.

    Args:
        y: Observed data points.
        state: Current state containing pi and mu values.

    Returns:
        Log-likelihood of the data.
    """
    K = len(state.pi)
    log_like = 0.0
    for _i, yi in enumerate(y):
        component_likes = []
        for k in range(K):
            log_comp = (
                np.log(state.pi[k])
                - 0.5 * (yi - state.mu[k]) ** 2
                - 0.5 * np.log(2 * np.pi)
            )
            component_likes.append(log_comp)
        log_like += np.log(np.sum(np.exp(component_likes)))
    return log_like


# pylint: disable=too-many-arguments
def tempered_transition_step(
    y: np.ndarray,
    state: State,
    K: int,
    rng: np.random.Generator,
    temperatures: np.ndarray,
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
        temperatures: Array of temperatures from 0 to 1.
        n_gibbs_per_temp: Number of Gibbs steps per temperature.

    Returns:
        New state after tempered transition.
    """
    current_state = state

    # Forward pass: cool down (high temp to low temp)
    for beta in reversed(temperatures):
        for _ in range(n_gibbs_per_temp):
            current_state = tempered_gibbs_step(y, current_state, K, rng, beta)

    # Backward pass: heat up (low temp to high temp)
    for beta in temperatures:
        for _ in range(n_gibbs_per_temp):
            current_state = tempered_gibbs_step(y, current_state, K, rng, beta)

    return current_state


def run_tempered_chain(
    y: np.ndarray,
    K: int,
    n_iter: int,
    burn: int,
    seed: int,
    n_temps: int = 10,
    max_temp: float = 5.0,
    n_gibbs_per_temp: int = 5,
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

    Returns:
        Tuple of (samples, runtime) where samples contains the
        post-burn-in samples and runtime is the elapsed time.
    """
    rng = np.random.default_rng(seed)

    # Create temperature ladder: geometric progression from 1/max_temp to 1
    temperatures = np.geomspace(1.0 / max_temp, 1.0, n_temps)

    # Initialize state
    state = State(rng.integers(0, K, len(y)), np.ones(K) / K, rng.normal(0, 1, K))

    kept = []
    t0 = time.perf_counter()

    for it in tqdm(range(n_iter)):
        state = tempered_transition_step(
            y, state, K, rng, temperatures, n_gibbs_per_temp
        )
        if it >= burn:
            kept.append(state.relabel().mu.copy())

    return np.vstack(kept), time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Tempered Transitions for Gaussian mixture"
    )
    ap.add_argument(
        "--data", type=str, default="../data/data.npy", help="Path to the data file"
    )
    ap.add_argument("--K", type=int, default=4, help="Number of mixture components")
    ap.add_argument("--n_iter", type=int, default=5000, help="Number of iterations")
    ap.add_argument("--burn", type=int, default=1000, help="Burn-in period")
    ap.add_argument("--chains", type=int, default=4, help="Number of chains")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--n_temps", type=int, default=10, help="Number of temperatures")
    ap.add_argument("--max_temp", type=float, default=5.0, help="Maximum temperature")
    ap.add_argument(
        "--n_gibbs_per_temp", type=int, default=1, help="Gibbs steps per temperature"
    )
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)
    y = np.load(args.data)

    chains, times = [], []
    print(f"Running {args.chains} Tempered Transitions chains...")
    for _ in range(args.chains):
        mu, rt = run_tempered_chain(
            y,
            args.K,
            args.n_iter,
            args.burn,
            int(rng_master.integers(2**32)),
            args.n_temps,
            args.max_temp,
            args.n_gibbs_per_temp,
        )
        chains.append(mu)
        times.append(rt)

    create_diagnostic_plots("tempered_transitions", chains, args.K, args.chains)

    pooled = np.vstack(chains)
    mu_mean = pooled.mean(axis=0)
    rhat = [rhat_scalar([c[:, k] for c in chains]) for k in range(args.K)]
    ess = [ess_1d(pooled[:, k]) for k in range(args.K)]

    ci_lower = np.array([credible_interval(pooled[:, k])[0] for k in range(args.K)])
    ci_upper = np.array([credible_interval(pooled[:, k])[1] for k in range(args.K)])

    print("\n=== TEMPERED TRANSITIONS SUMMARY ===")
    print("Posterior mean μ   :", np.round(mu_mean, 4))
    print("95% CI lower       :", np.round(ci_lower, 4))
    print("95% CI upper       :", np.round(ci_upper, 4))
    print("R-hat (μ)          :", np.round(rhat, 3))
    print("ESS  (μ)           :", np.round(ess, 1))
    print(f"Average time / chain: {np.mean(times):.2f}s")
    print(f"Parameters: {args.n_temps} temperatures, max_temp={args.max_temp}")
    print(f"            {args.n_gibbs_per_temp} Gibbs steps per temperature")

    print("\nDiagnostic PNGs saved in", os.getcwd())
