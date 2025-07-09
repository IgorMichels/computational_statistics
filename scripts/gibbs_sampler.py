import argparse
import os
import time
import warnings

import numpy as np
from metrics import create_metrics
from plots import create_diagnostic_plots
from samplers import sample_mu, sample_pi, sample_z
from state import State
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    for it in tqdm(range(n_iter)):
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
    mu_mean, rhat, ess, ci_lower, ci_upper = create_metrics(chains, args.K)

    print("\n=== Gibbs SUMMARY ===")
    print("Posterior mean μ  :", np.round(mu_mean, 4))
    print("95% CI lower      :", np.round(ci_lower, 4))
    print("95% CI upper      :", np.round(ci_upper, 4))
    print("R‑hat (μ)         :", np.round(rhat, 3))
    print("ESS  (μ)          :", np.round(ess, 1))
    print(f"Mean runtime / chain: {np.mean(times):.2f}s")

    print("\nPNG diagnostics saved in", os.getcwd())
