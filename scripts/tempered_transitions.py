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
    ap.add_argument("--max_temp", type=float, default=10.0, help="Maximum temperature")
    ap.add_argument(
        "--n_gibbs_per_temp", type=int, default=1, help="Gibbs steps per temperature"
    )
    ap.add_argument("--placebo", action="store_true", help="Use placebo on relabeling")
    args = ap.parse_args()

    rng_master = np.random.default_rng(args.seed)
    y = np.load(args.data)

    chains, times = [], []
    acceptance_rates = []
    print(f"Running {args.chains} Tempered Transitions chains...")
    for _ in range(args.chains):
        mu, rt, acc_rate = run_chain(
            y,
            args.K,
            args.n_iter,
            args.burn,
            int(rng_master.integers(2**32)),
            n_temps=args.n_temps,
            max_temp=args.max_temp,
            n_gibbs_per_temp=args.n_gibbs_per_temp,
            placebo=args.placebo,
            keep_last=True,
        )
        chains.append(mu)
        times.append(rt)
        acceptance_rates.append(acc_rate)

    create_diagnostic_plots("tempered_transitions", chains, args.K, args.chains)
    mu_mean, rhat, ess, ci_lower, ci_upper = create_metrics(chains, args.K)

    print("\n=== TEMPERED TRANSITIONS SUMMARY ===")
    print("Posterior mean μ   :", np.round(mu_mean, 4))
    print("95% CI lower       :", np.round(ci_lower, 4))
    print("95% CI upper       :", np.round(ci_upper, 4))
    print("R-hat (μ)          :", np.round(rhat, 3))
    print("ESS  (μ)           :", np.round(ess, 1))
    print(f"Average time / chain: {np.mean(times):.2f}s")
    print(f"Average acceptance rate: {np.mean(acceptance_rates):.3f}")
    print(f"Parameters: {args.n_temps} temperatures, max_temp={args.max_temp}")
    print(f"            {args.n_gibbs_per_temp} Gibbs steps per temperature")

    print("\nDiagnostic PNGs saved in", os.getcwd())
