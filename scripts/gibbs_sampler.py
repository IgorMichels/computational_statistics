import argparse
import os
import time
import warnings

import numpy as np
from metrics import credible_interval, ess_1d, rhat_scalar
from plots import create_diagnostic_plots
from state import State

warnings.filterwarnings("ignore", category=DeprecationWarning)


def sample_z(y: np.ndarray, state: State, rng: np.random.Generator) -> np.ndarray:
    """Sample cluster assignments for each data point.

    Uses the current values of pi and mu to compute posterior
    probabilities and sample new cluster assignments.

    Args:
        y: Observed data points.
        state: Current state containing pi and mu values.
        rng: Random number generator.

    Returns:
        New cluster assignments for each data point.
    """
    K = len(state.pi)
    logw = np.empty((len(y), K))
    for k in range(K):
        log_prior = np.log(state.pi[k])
        log_like = -0.5 * (y - state.mu[k]) ** 2 - 0.5 * np.log(2 * np.pi)
        logw[:, k] = log_prior + log_like
    logw -= logw.max(axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    probs = np.exp(logw)
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random((len(y), 1))
    return (np.cumsum(probs, axis=1) > u).argmax(axis=1)


def sample_pi(
    z: np.ndarray, K: int, rng: np.random.Generator, alpha: float = 1.0
) -> np.ndarray:
    """Sample mixture weights from Dirichlet distribution.

    Args:
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
        alpha: Dirichlet concentration parameter.

    Returns:
        New mixture weights sampled from posterior Dirichlet.
    """
    return rng.dirichlet(alpha + np.bincount(z, minlength=K))


# pylint: disable=too-many-arguments
def sample_mu(
    y: np.ndarray,
    z: np.ndarray,
    K: int,
    rng: np.random.Generator,
    m0: float = 0.0,
    s0_2: float = 25.0,
) -> np.ndarray:
    """Sample mean parameters from normal distributions.

    For each component, samples from the posterior normal distribution
    given the assigned data points and prior parameters.

    Args:
        y: Observed data points.
        z: Current cluster assignments.
        K: Number of mixture components.
        rng: Random number generator.
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
            post_var = 1.0 / (n_k + 1 / s0_2)
            post_mean = post_var * (n_k * ybar + m0 / s0_2)
            mu[k] = rng.normal(post_mean, np.sqrt(post_var))
    return mu


def gibbs_step(y: np.ndarray, state: State, K: int, rng: np.random.Generator) -> State:
    """Perform one step of the Gibbs sampler.

    Sequentially samples z, pi, and mu given the current state.

    Args:
        y: Observed data points.
        state: Current state of the sampler.
        K: Number of mixture components.
        rng: Random number generator.

    Returns:
        New state after one Gibbs sampling step.
    """
    z = sample_z(y, state, rng)
    pi = sample_pi(z, K, rng)
    mu = sample_mu(y, z, K, rng)
    return State(z, pi, mu)


def run_chain(y: np.ndarray, K: int, n_iter: int, burn: int, seed: int):
    """Run a single Gibbs sampling chain.

    Args:
        y: Observed data points.
        K: Number of mixture components.
        n_iter: Total number of iterations.
        burn: Number of burn-in iterations to discard.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (samples, runtime) where samples contains the
        post-burn-in samples and runtime is the elapsed time.
    """
    rng = np.random.default_rng(seed)
    state = State(rng.integers(0, K, len(y)), np.ones(K) / K, rng.normal(0, 1, K))
    kept = []
    t0 = time.perf_counter()
    for it in range(n_iter):
        state = gibbs_step(y, state, K, rng)
        if it >= burn:
            kept.append(state.relabel().mu.copy())
    return np.vstack(kept), time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gibbs Sampler for Gaussian mixture")
    ap.add_argument(
        "--data", type=str, default="../data/data.npy", help="Path to the data file"
    )
    ap.add_argument("--K", type=int, default=4, help="Number of mixture components")
    ap.add_argument("--n_iter", type=int, default=10000, help="Number of iterations")
    ap.add_argument("--burn", type=int, default=2000, help="Burn-in period")
    ap.add_argument("--chains", type=int, default=4, help="Number of chains")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)
    y = np.load(args.data)

    chains, times = [], []
    print(f"Running {args.chains} Gibbs chains…")
    for _ in range(args.chains):
        mu, rt = run_chain(
            y, args.K, args.n_iter, args.burn, int(rng_master.integers(2**32))
        )
        chains.append(mu)
        times.append(rt)

    create_diagnostic_plots("gibbs", chains, args.K, args.chains)

    pooled = np.vstack(chains)
    mu_mean = pooled.mean(axis=0)
    rhat = [rhat_scalar([c[:, k] for c in chains]) for k in range(args.K)]
    ess = [ess_1d(pooled[:, k]) for k in range(args.K)]

    ci_lower = np.array([credible_interval(pooled[:, k])[0] for k in range(args.K)])
    ci_upper = np.array([credible_interval(pooled[:, k])[1] for k in range(args.K)])

    print("\n=== Gibbs SUMMARY ===")
    print("Posterior mean μ  :", np.round(mu_mean, 4))
    print("95% CI lower      :", np.round(ci_lower, 4))
    print("95% CI upper      :", np.round(ci_upper, 4))
    print("R‑hat (μ)         :", np.round(rhat, 3))
    print("ESS  (μ)          :", np.round(ess, 1))
    print(f"Mean runtime / chain: {np.mean(times):.2f}s")

    print("\nPNG diagnostics saved in", os.getcwd())
