# pylint: disable=duplicate-code

import argparse
import os
import warnings

import numpy as np
from metrics import create_metrics
from plots import create_diagnostic_plots
from samplers import run_chain

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    ap.add_argument("--placebo", action="store_true", help="Use placebo on relabeling")
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)
    y = np.load(args.data)

    chains, times = [], []
    print(f"Running {args.chains} Gibbs chains…")
    for _ in range(args.chains):
        mu, rt, _ = run_chain(
            y,
            args.K,
            args.n_iter,
            args.burn,
            int(rng_master.integers(2**32)),
            n_temps=1,
            max_temp=1,
            n_gibbs_per_temp=1,
            placebo=args.placebo,
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
