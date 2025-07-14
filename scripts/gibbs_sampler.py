# pylint: disable=duplicate-code

import argparse
import warnings

import numpy as np
from metrics import create_metrics
from plots import create_diagnostic_plots
from samplers import run_parallel_chains
from utils import (
    add_gibbs_args,
    create_output_message,
    parse_all_prior_args,
    print_parameter_summary,
    print_runtime_summary,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Gibbs Sampler for Gaussian mixture with unknown variances"
    )

    # Add all arguments for Gibbs sampler
    add_gibbs_args(ap)

    args = ap.parse_args()

    # Parse prior parameters using utility function
    m0, s0_2, alpha0, beta0 = parse_all_prior_args(args)

    y = np.load(f"../data/{args.data}/data.npy")

    if args.verbose:
        print(f"Running {args.chains} Gibbs chains…")
    chains_mu, chains_sigma2, times, _ = run_parallel_chains(
        y,
        args.K,
        args.n_iter,
        args.burn,
        args.seed,
        n_chains=args.chains,
        m0=m0,
        s0_2=s0_2,
        alpha0=alpha0,
        beta0=beta0,
        n_temps=1,
        max_temp=1,
        n_gibbs_per_temp=1,
        placebo=args.placebo,
        verbose=args.verbose,
        loading_bar=args.loading_bar,
    )

    create_diagnostic_plots("gibbs", chains_mu, args.chains, args.data, param_name="mu")
    create_diagnostic_plots(
        "gibbs", chains_sigma2, args.chains, args.data, param_name="sigma2"
    )

    mu_metrics = create_metrics(chains_mu, args.data, param_name="mu")
    sigma2_metrics = create_metrics(chains_sigma2, args.data, param_name="sigma2")

    if args.verbose:
        print("\n=== GIBBS SUMMARY ===")
        print_parameter_summary("μ", mu_metrics)
        print()
        print_parameter_summary("σ²", sigma2_metrics)

        print_runtime_summary(times)
        print(create_output_message(args.data))
